from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# Calibrated from models/visual_transformer_v3_6x6/action_logs/
# 20260602_081604_step_0002.png. The captured tensor includes top/left debug
# overlay and gray browser margin, so the Minesweeper board is not the full image.
ACTION_LOG_BOARD_RECT_NORM = (
    23 / 640,
    21 / 640,
    622 / 640,
    620 / 640,
)


def _scaled_action_log_board_rect(img_w: int, img_h: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = ACTION_LOG_BOARD_RECT_NORM
    return (
        int(round(left * img_w)),
        int(round(top * img_h)),
        int(round(right * img_w)),
        int(round(bottom * img_h)),
    )


class VisualAgentCommonMixin:
    """Shared agent utilities for visual RL variants."""

    model_path: Path
    replay_path: Path
    replay_persistent_path: Path
    action_log_path: Path
    image_size: tuple[int, int]
    save_capacity: int
    priority_min: float
    priority_max: float
    log_prefix: str

    def _optimizer_state_path(self) -> Path:
        return self.model_path / "optimizer_state.pth"

    def _training_state_path(self) -> Path:
        return self.model_path / "training_state.pth"

    def _training_history_path(self) -> Path:
        return self.model_path / "training_history.pth"

    def _to_storage_state(self, state: torch.Tensor | None) -> torch.Tensor | None:
        """Hook: convert a raw env state into what should live in the replay buffer.

        Default: detach + move to CPU. v1/v2/v3 all store raw screenshots this way;
        CategorizedReplayBuffer._save_transition compresses (3, H>=64, W) tensors to
        uint8 on disk and _load_transition restores them to float [0,1] at sample time.
        """
        if state is None:
            return None
        return state.detach().cpu()

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor | None,
        reward: float,
        done: bool,
        *,
        source: str = "train",
    ) -> None:
        if source not in ("train", "eval"):
            raise ValueError(f"source must be 'train' or 'eval', got {source!r}")
        # Record raw rewards into training_history.step_rewards for the
        # train/real_reward_mean rolling metric. v2 / v3 share this mixin, so
        # neither agent keeps its own recent_real_rewards deque.
        self.training_history.record_step_reward(float(reward))
        transition = {
            "state": self._to_storage_state(state),
            "action": int(action),
            "next_state": self._to_storage_state(next_state),
            "reward": float(reward),
            "done": bool(done),
            "source": source,
        }
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) >= self.n_step:
            self._commit_n_step_transition(self.n_step)
        if done:
            self._flush_n_step_buffer()

    def _commit_n_step_transition(self, horizon: int) -> None:
        if not self.n_step_buffer:
            return

        horizon = min(horizon, len(self.n_step_buffer))
        discounted_reward = 0.0
        last_transition = None
        for step_idx in range(horizon):
            transition = self.n_step_buffer[step_idx]
            discounted_reward += (self.n_step_gamma ** step_idx) * transition["reward"]
            last_transition = transition
            if transition["done"]:
                horizon = step_idx + 1
                break

        first_transition = self.n_step_buffer[0]
        discount = self.n_step_gamma ** horizon
        self.replay_buffer.store_unscored(
            first_transition["state"],
            first_transition["action"],
            last_transition["next_state"],
            discounted_reward,
            last_transition["done"],
            discount=discount,
            n_steps=horizon,
            tail_reward=last_transition["reward"],
        )
        self.n_step_buffer.popleft()

    def _flush_n_step_buffer(self) -> None:
        while self.n_step_buffer:
            self._commit_n_step_transition(len(self.n_step_buffer))

    def _save_optimizer_state(self) -> None:
        # Optimizer state only stores optimizer / scaler / total_it / episode_count / epsilon.
        # Episode results live in TrainingHistory and are written to training_history.pth.
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "total_it": self.total_it,
            "episode_count": self.episode_count,
        }
        payload.update(self.epsilon_controller.state_dict())
        # V3 persists its monotone YOLO update latch so the unfreeze decision is
        # inherited across resumes: a from-scratch random-init run trains YOLO from
        # step 0, and a Stage 1 warm-start that has crossed the win-rate threshold
        # stays unfrozen even if win_rate later dips. Guarded so v1/v2 agents that
        # have no latch are unaffected.
        if hasattr(self, "_yolo_update_latched"):
            payload["yolo_update_latched"] = bool(self._yolo_update_latched)
        torch.save(payload, self._optimizer_state_path())

        # TrainingHistory uses an independent file, decoupled from optimizer state.
        try:
            torch.save(self.training_history.state_dict(), self._training_history_path())
        except Exception as exc:
            print(f"{self.log_prefix} Failed to save training_history: {exc}")

    def _load_optimizer_state(self) -> None:
        opt_path = self._optimizer_state_path()
        if not opt_path.exists():
            return
        try:
            state = torch.load(opt_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            self.total_it = state.get("total_it", 0)
            # Inherited YOLO update latch (V3). None when the key is absent (v1/v2
            # agents, or checkpoints saved before the latch was persisted), which
            # tells the agent's __init__ to fall back to its fresh-start decision.
            self._loaded_yolo_update_latched = state.get("yolo_update_latched", None)
            # episode_count is no longer set directly. It delegates to
            # training_history.total_episodes, restored from its own .pth file.
            # Ignore the legacy "episode_count" key in optimizer state.
            # AdaptiveEpsilonController now stores only epsilon.
            self.epsilon_controller.load_state_dict(state)

            # TrainingHistory uses its own independent file.
            self._load_training_history()

            success_msg = (
                f"Loaded optimizer ({opt_path}): total_it={self.total_it}, "
                f"episode={self.episode_count}, "
                f"epsilon={self.epsilon_controller.epsilon:.4f}, "
                f"total_episodes={self.training_history.total_episodes}"
            )
            # Route through CheckpointLogger so success prints green and the
            # [loaded_checkpoints] tracker captures opt_path. Falls back to a
            # plain log_prefix print for v1/v2 agents that have no logger.
            if hasattr(self, "checkpoint_logger"):
                self.checkpoint_logger.success(
                    "optimizer_state", opt_path, success_msg
                )
            else:
                print(f"{self.log_prefix} {success_msg}")
            if "scaler" in state and self.scaler.is_enabled():
                try:
                    self.scaler.load_state_dict(state["scaler"])
                except Exception as scaler_exc:
                    print(f"{self.log_prefix} Failed to load GradScaler state: {scaler_exc}")
        except Exception as exc:
            message = f"MISSING/FAILED optimizer checkpoint: {opt_path} | error={exc}"
            if hasattr(self, "checkpoint_logger"):
                self.checkpoint_logger.failure("optimizer_state", message)
            elif hasattr(self, "_log_checkpoint_message"):
                self._log_checkpoint_message(message, warning=True)
            else:
                print(f"{self.log_prefix} {message}")

    def _load_training_history(self) -> None:
        """Load TrainingHistory from its independent file."""
        history_path = self._training_history_path()
        if not history_path.exists():
            return
        try:
            hist_state = torch.load(
                history_path, map_location=self.device, weights_only=False
            )
            self.training_history.load_state_dict(
                hist_state, deque_cls=self.deque_cls
            )
            if hasattr(self, "checkpoint_logger"):
                self.checkpoint_logger.success(
                    "training_history",
                    history_path,
                    f"Loaded training_history: {history_path}",
                )
        except Exception as exc:
            message = (
                f"Failed to load training_history.pth ({history_path}): {exc}"
            )
            if hasattr(self, "checkpoint_logger"):
                self.checkpoint_logger.failure("training_history", message)
            else:
                print(f"{self.log_prefix} {message}")

    def save_persistent(self) -> None:
        buf = self.replay_buffer
        total = buf.size()
        if total == 0:
            return

        target = min(self.save_capacity, total)
        persistent_entries = buf.export_top_k(target, persistent_dir=self.replay_persistent_path)

        torch.save(
            {
                "persistent_entries": persistent_entries,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
            },
            self._training_state_path(),
        )
        print(f"{self.log_prefix} Persistent save: {len(persistent_entries)} entries")

    def _load_persistent_training_state(self) -> None:
        ts_path = self._training_state_path()
        if not ts_path.exists():
            return
        try:
            state = torch.load(ts_path, map_location=self.device, weights_only=False)
            persistent_entries = state.get("persistent_entries", [])
            if persistent_entries:
                # _load_persistent_buffer prints the per-entry summary and
                # records the source on the checkpoint_logger when present.
                self._load_persistent_buffer(persistent_entries, source_path=ts_path)
        except Exception as exc:
            message = f"MISSING/FAILED replay/training checkpoint: {ts_path} | error={exc}"
            if hasattr(self, "checkpoint_logger"):
                self.checkpoint_logger.failure("replay_buffer", message)
            elif hasattr(self, "_log_checkpoint_message"):
                self._log_checkpoint_message(message, warning=True)
            else:
                print(f"{self.log_prefix} {message}")

    def _load_persistent_buffer(self, persistent_entries, source_path=None) -> None:
        self.replay_path.mkdir(parents=True, exist_ok=True)
        for file_path in self.replay_path.glob("*.pt"):
            file_path.unlink()

        # Agents can declare a non-screenshot state shape via `replay_state_shape`.
        # Default = (3, *image_size); v1/v2/v3 all store screenshots and use the
        # default. (Kept as a hook for any future agent caching a non-screenshot state.)
        expected_shape = tuple(getattr(self, "replay_state_shape", (3, *self.image_size)))

        loaded_count = 0
        skipped_missing = 0
        skipped_shape_mismatch = 0
        skipped_load_error = 0
        runtime_entries = []
        for entry in persistent_entries[: self.replay_buffer.max_size]:
            transition_src_str = entry.get("transition")
            if transition_src_str is None:
                skipped_missing += 1
                continue
            transition_src = Path(transition_src_str)
            if not transition_src.exists():
                skipped_missing += 1
                continue

            # weights_only=True limits deserialization opcodes (RCE hardening). The
            # transition file is one dict {"state":..., "next_state":...}; validate the
            # state shape (and next_state's shape when present) against expected_shape so
            # stale-format data cannot crash train_step. A wrong shape drops the whole
            # transition.
            try:
                peek = torch.load(str(transition_src), map_location="cpu", weights_only=True)
                state_peek = peek.get("state") if isinstance(peek, dict) else None
                if not torch.is_tensor(state_peek) or tuple(state_peek.shape) != expected_shape:
                    skipped_shape_mismatch += 1
                    continue
                next_peek = peek.get("next_state")
                if next_peek is not None and (
                    not torch.is_tensor(next_peek) or tuple(next_peek.shape) != expected_shape
                ):
                    skipped_shape_mismatch += 1
                    continue
            except Exception:
                skipped_load_error += 1
                continue

            storage_id = loaded_count
            transition_dst = self.replay_path / f"transition_{storage_id}.pt"
            shutil.copy2(str(transition_src), str(transition_dst))

            runtime_entry = {
                "storage_id": storage_id,
                "transition": str(transition_dst),
                "action": int(entry["action"]),
                "reward": float(entry["reward"]),
                "tail_reward": float(entry.get("tail_reward", entry["reward"])),
                "done": bool(entry["done"]),
                "discount": float(entry.get("discount", 1.0)),
                "n_steps": int(entry.get("n_steps", 1)),
                "reward_type": self.replay_buffer.reward_type_for(
                    float(entry.get("tail_reward", entry["reward"])),
                    bool(entry["done"]),
                ),
                "priority": float(
                    np.clip(
                        entry.get("priority", abs(float(entry["reward"])) + 1.0),
                        self.priority_min,
                        self.priority_max,
                    )
                ),
                "insert_order": loaded_count + 1,
                # Match stage1. RAM load_from_entries setdefaults these fields, but
                # the disk path used to miss them. Without them, spread_decay continuity
                # breaks after resume and sample_count resets for age/sample decay.
                "sample_count": int(entry.get("sample_count", 0)),
                "quantile_spread": float(entry.get("quantile_spread", 0.0)),
            }
            if "reward_type" in entry:
                runtime_entry["reward_type"] = entry["reward_type"]
            runtime_entries.append(runtime_entry)
            loaded_count += 1

        self.replay_buffer.replace_entries(
            runtime_entries,
            next_storage_id=loaded_count,
            insert_counter=loaded_count,
        )
        # Show skip reasons so expected_shape changes do not just print "Loaded 0".
        # Example: V3 screenshot entries became cached features, making old entries mismatch.
        skipped_total = skipped_missing + skipped_shape_mismatch + skipped_load_error
        suffix = ""
        if skipped_total > 0:
            suffix = (
                f" | skipped {skipped_total} "
                f"(missing={skipped_missing}, shape!={tuple(expected_shape)}={skipped_shape_mismatch}, "
                f"load_error={skipped_load_error})"
            )
        success_msg = f"Loaded {loaded_count} replay buffer entries{suffix}"
        if hasattr(self, "checkpoint_logger") and loaded_count > 0:
            self.checkpoint_logger.success(
                "replay_buffer",
                source_path if source_path is not None else self._training_state_path(),
                success_msg,
            )
        else:
            print(f"{self.log_prefix} {success_msg}")

    def _purge_stale_replay_files(self) -> None:
        """Delete replay .pt files whose shape no longer matches replay_state_shape.

        Called in agent __init__ before try_load_model(). This removes stale data
        from older architectures, e.g. when V3 switched from (3, 640, 640)
        screenshots to (128, h, w) cached features.

        If persistent .pt files are deleted, delete training_state.pth too because
        its persistent_entries may point at missing paths.
        """
        expected_shape = tuple(
            getattr(self, "replay_state_shape", (3, *self.image_size))
        )

        def _purge_dir(dir_path: Path) -> int:
            if not dir_path.exists():
                return 0
            deleted = 0
            for pt_file in dir_path.glob("*.pt"):
                keep = False
                try:
                    # weights_only=True limits pickle deserialization opcodes and
                    # reduces RCE risk from corrupt/malicious files.
                    peek = torch.load(
                        str(pt_file), map_location="cpu", weights_only=True
                    )
                    # New format: each .pt is a dict {"state":..., "next_state":...}.
                    # Old single-tensor files (peek is a bare tensor) fail the dict
                    # check and are deleted — the intended clean break from the
                    # two-file-per-transition format.
                    state_peek = peek.get("state") if isinstance(peek, dict) else None
                    if torch.is_tensor(state_peek) and tuple(state_peek.shape) == expected_shape:
                        keep = True
                except Exception:
                    keep = False  # Read failure means corrupt, so delete it too.
                if not keep:
                    try:
                        pt_file.unlink()
                        deleted += 1
                    except Exception as exc:
                        print(
                            f"{self.log_prefix} Failed to unlink stale "
                            f"{pt_file}: {exc}"
                        )
            return deleted

        deleted_runtime = _purge_dir(self.replay_path)
        deleted_persistent = _purge_dir(self.replay_persistent_path)

        # Deleted persistent .pt files make training_state.pth metadata stale.
        if deleted_persistent > 0:
            ts_path = self._training_state_path()
            if ts_path.exists():
                try:
                    ts_path.unlink()
                    print(
                        f"{self.log_prefix} Purged stale training_state.pth "
                        f"(referenced deleted files)"
                    )
                except Exception as exc:
                    print(
                        f"{self.log_prefix} Failed to unlink stale "
                        f"{ts_path}: {exc}"
                    )

        if deleted_runtime or deleted_persistent:
            print(
                f"{self.log_prefix} Purged stale replay buffer files "
                f"(runtime={deleted_runtime}, persistent={deleted_persistent}) "
                f"— expected shape {expected_shape}"
            )

    def _module_grad_norm(self, module) -> float:
        grad_sq_sum = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())
        return grad_sq_sum ** 0.5

    def _trainable_module_groups(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.backbone,
            "head": self.q_network,
        }

    def _capture_trainable_weight_snapshot(self) -> dict[str, torch.Tensor]:
        snapshot = {}
        for group_name, module in self._trainable_module_groups().items():
            for param_name, param in module.named_parameters():
                snapshot[f"{group_name}.{param_name}"] = param.detach().float().cpu().clone()
        return snapshot

    def _snapshot_distance(
        self,
        current_snapshot: dict[str, torch.Tensor],
        reference_snapshot: dict[str, torch.Tensor],
        eps: float = 1e-12,
    ) -> float:
        diff_sq_sum = 0.0
        ref_sq_sum = 0.0
        for name, current_value in current_snapshot.items():
            reference_value = reference_snapshot.get(name)
            if reference_value is None:
                continue
            diff = current_value - reference_value
            diff_sq_sum += float(diff.pow(2).sum().item())
            ref_sq_sum += float(reference_value.pow(2).sum().item())
        return (diff_sq_sum ** 0.5) / max(ref_sq_sum ** 0.5, eps)

    def _tensor_norm(self, tensor: torch.Tensor | None) -> float | None:
        if tensor is None:
            return None
        return float(tensor.detach().float().norm().item())

    def _log_param_weight_and_grad_norm(
        self,
        tag_prefix: str,
        param: nn.Parameter | None,
        global_step: int,
    ) -> None:
        if param is None:
            return
        weight_norm = self._tensor_norm(param)
        if weight_norm is not None:
            self.tb_writer.add_scalar(f"weight_norm/{tag_prefix}", weight_norm, global_step)
        grad_norm = self._tensor_norm(param.grad)
        if grad_norm is not None:
            self.tb_writer.add_scalar(f"grad_norm/{tag_prefix}", grad_norm, global_step)

    def _should_log_action_image(self) -> bool:
        return bool(getattr(self, "_log_actions_this_episode", True))

    def _render_action_image(self, state, log_info, step_count, reward=None) -> "Image.Image":
        """Draw the clicked cell, grid lines, and Q/action text onto the state frame.

        Returns the annotated PIL image. Shared by log_action_image (every-N-episode
        action trace) and log_lose_image (every eval loss, for failure inspection).
        """
        from PIL import ImageDraw, ImageFont

        img_array = state.detach().cpu().clamp(0, 1).mul(255).byte().numpy().transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        img_w, img_h = img.size
        grid_left, grid_top, grid_right, grid_bottom = _scaled_action_log_board_rect(img_w, img_h)
        cell_w = (grid_right - grid_left) / self.grid_w
        cell_h = (grid_bottom - grid_top) / self.grid_h

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        row, col = log_info["row"], log_info["col"]
        left = int(round(grid_left + col * cell_w))
        top = int(round(grid_top + row * cell_h))
        right = int(round(grid_left + (col + 1) * cell_w))
        bottom = int(round(grid_top + (row + 1) * cell_h))
        draw.rectangle(
            [left, top, right, bottom],
            outline="red",
            width=4,
        )
        for r in range(1, self.grid_h):
            y = int(round(grid_top + r * cell_h))
            draw.line([grid_left, y, grid_right, y], fill="white", width=1)
        for c in range(1, self.grid_w):
            x = int(round(grid_left + c * cell_w))
            draw.line([x, grid_top, x, grid_bottom], fill="white", width=1)

        lines = [
            f"Step: {step_count}",
            f"Action: {log_info['action_id']} -> ({row},{col})",
            f"Source: {log_info.get('source', '?')}",
        ]
        if log_info.get("selected_q") is not None:
            lines.append(f"Q: {log_info['selected_q']:.4f}")
        top_actions = log_info.get("top_actions") or []
        if top_actions:
            lines.append("Top5 Q:")
            for rank, item in enumerate(top_actions[:5], start=1):
                action_id, top_row, top_col, top_q = item
                lines.append(
                    f"{rank}: {int(action_id)} ({int(top_row)},{int(top_col)}) {float(top_q):.4f}"
                )
        if reward is not None:
            lines.append(f"Reward: {reward:.1f}")

        text_y = 5
        for line in lines:
            bbox = draw.textbbox((5, text_y), line, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text((5, text_y), line, fill="white", font=font)
            text_y += 15

        return img

    def _save_action_image(
        self, state, log_info, step_count, reward, *, out_dir, label, microsecond
    ) -> None:
        """Render the annotated frame and save it to out_dir. Shared save skeleton.

        ``microsecond`` widens the timestamp so two frames written in the same second
        (e.g. two eval episodes losing on the same step) do not overwrite each other.
        """
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            img = self._render_action_image(state, log_info, step_count, reward)
            fmt = "%Y%m%d_%H%M%S_%f" if microsecond else "%Y%m%d_%H%M%S"
            ts = datetime.datetime.now().strftime(fmt)
            img.save(out_dir / f"{ts}_step_{step_count:04d}.png")
        except Exception as exc:
            print(f"{self.log_prefix} {label} failed: {exc}")

    def log_action_image(self, state, log_info, step_count, reward=None) -> None:
        if not getattr(self, "log_actions", True) or log_info is None:
            return
        if not self._should_log_action_image():
            return
        self._save_action_image(
            state, log_info, step_count, reward,
            out_dir=self.action_log_path, label="log_action_image", microsecond=False,
        )

    def _lose_log_path(self) -> Path:
        """Folder for eval-loss frames. Defaults to a `lose/` sibling of action_logs."""
        return getattr(self, "lose_log_path", None) or (self.action_log_path.parent / "lose")

    def log_lose_image(self, state, log_info, step_count, reward=None) -> None:
        """Save the frame the agent saw right before an eval loss.

        Unlike log_action_image, this ignores the every-N-episode action-trace gate:
        every eval loss is captured so the operator can inspect whether the board was
        genuinely undecidable or the loss came from a logic problem.
        """
        if not getattr(self, "log_actions", True) or log_info is None:
            return
        self._save_action_image(
            state, log_info, step_count, reward,
            out_dir=self._lose_log_path(), label="log_lose_image", microsecond=True,
        )
