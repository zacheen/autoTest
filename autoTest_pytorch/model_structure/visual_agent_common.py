from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


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

        Default: detach + move to CPU (used by v1/v2 which store raw screenshots).
        V3 overrides this to run the frozen YOLO backbone once and store the
        (128, h, w) feature tensor instead of the (3, H, W) screenshot, cutting
        per-entry size and skipping the YOLO forward at every gradient step.
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
    ) -> None:
        self.recent_real_rewards.append(float(reward))
        transition = {
            "state": self._to_storage_state(state),
            "action": int(action),
            "next_state": self._to_storage_state(next_state),
            "reward": float(reward),
            "done": bool(done),
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
        self.replay_buffer.store(
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
        # optimizer state 只含 optimizer / scaler / total_it / episode_count / epsilon。
        # Episode 結果累積/查詢搬到 TrainingHistory,寫到獨立檔 training_history.pth。
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "total_it": self.total_it,
            "episode_count": self.episode_count,
        }
        payload.update(self.epsilon_controller.state_dict())
        torch.save(payload, self._optimizer_state_path())

        # TrainingHistory 走獨立檔,跟 optimizer state 解耦。
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
            # episode_count 不再直接 set — 它是 @property delegate 到
            # training_history.total_episodes,後者由獨立 .pth 檔還原。
            # 舊 opt state 裡的 "episode_count" key 直接忽略。
            # AdaptiveEpsilonController 只剩 epsilon 一個 key。
            self.epsilon_controller.load_state_dict(state)

            # TrainingHistory:優先用獨立檔;舊 checkpoint 還沒拆檔時,
            # 從 optimizer state 撈 legacy 扁平 keys (result_window /
            # total_episodes) 餵進去,完成一次性 migration。
            self._load_training_history(legacy_state=state)

            print(
                f"{self.log_prefix} Loaded optimizer: total_it={self.total_it}, "
                f"episode={self.episode_count}, epsilon={self.epsilon_controller.epsilon:.4f}, "
                f"total_episodes={self.training_history.total_episodes}"
            )
            if "scaler" in state and self.scaler.is_enabled():
                try:
                    self.scaler.load_state_dict(state["scaler"])
                except Exception as scaler_exc:
                    print(f"{self.log_prefix} Failed to load GradScaler state: {scaler_exc}")
        except Exception as exc:
            message = f"MISSING/FAILED optimizer checkpoint: {opt_path} | error={exc}"
            if hasattr(self, "_log_checkpoint_message"):
                self._log_checkpoint_message(message, warning=True)
            else:
                print(f"{self.log_prefix} {message}")

    def _load_training_history(self, legacy_state: dict | None = None) -> None:
        """Load TrainingHistory:獨立檔優先,舊 flat optimizer state 是 fallback。

        legacy_state 是當前 optimizer state 的 dict (load 流程順手帶進來);
        若新獨立檔不存在但 legacy_state 含有舊扁平 keys,撈出來做 migration。
        """
        history_path = self._training_history_path()
        if history_path.exists():
            try:
                hist_state = torch.load(
                    history_path, map_location=self.device, weights_only=False
                )
                self.training_history.load_state_dict(
                    hist_state, deque_cls=self.deque_cls
                )
                return
            except Exception as exc:
                print(f"{self.log_prefix} Failed to load training_history.pth: {exc}")
                # 落到下面 legacy fallback
        if legacy_state and any(
            k in legacy_state for k in ("result_window", "total_episodes", "total_wins")
        ):
            legacy = {
                "results": legacy_state.get("result_window", []),
                "total_episodes": legacy_state.get("total_episodes", 0),
                "total_wins": legacy_state.get("total_wins", 0),
            }
            self.training_history.load_state_dict(legacy, deque_cls=self.deque_cls)
            print(
                f"{self.log_prefix} Migrated legacy training history from optimizer state"
            )

    def save_persistent(self) -> None:
        buf = self.replay_buffer
        if buf.size_count == 0:
            return

        target = min(self.save_capacity, buf.size_count)
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
            # Read new unified key first; fall back to legacy "persistent_index"
            # key so older save files keep loading.
            persistent_entries = state.get("persistent_entries") or state.get("persistent_index", [])
            if persistent_entries:
                self._load_persistent_buffer(persistent_entries)
        except Exception as exc:
            message = f"MISSING/FAILED replay/training checkpoint: {ts_path} | error={exc}"
            if hasattr(self, "_log_checkpoint_message"):
                self._log_checkpoint_message(message, warning=True)
            else:
                print(f"{self.log_prefix} {message}")

    def _load_persistent_buffer(self, persistent_entries) -> None:
        self.replay_path.mkdir(parents=True, exist_ok=True)
        for file_path in self.replay_path.glob("*.pt"):
            file_path.unlink()

        # Agents can declare a non-screenshot state shape via `replay_state_shape`
        # (V3 stores YOLO backbone features (128, h, w) instead of raw screenshots).
        # Default = (3, *image_size) so v1/v2 keep their screenshot validation.
        expected_shape = tuple(getattr(self, "replay_state_shape", (3, *self.image_size)))

        loaded_count = 0
        skipped_missing = 0
        skipped_shape_mismatch = 0
        skipped_load_error = 0
        self.replay_buffer.index = []
        for entry in persistent_entries[: self.replay_buffer.max_size]:
            state_src_str = entry.get("state", entry.get("state_path"))
            if state_src_str is None:
                skipped_missing += 1
                continue
            state_src = Path(state_src_str)
            if not state_src.exists():
                skipped_missing += 1
                continue

            # weights_only=True：peek 出來的東西必定是 tensor;限制反序列化能執行的
            # opcode,避免 replay_buffer_save/ 底下的 .pt 被惡意/損毀檔案 RCE。
            try:
                peek = torch.load(str(state_src), map_location="cpu", weights_only=True)
                if not torch.is_tensor(peek) or tuple(peek.shape) != expected_shape:
                    skipped_shape_mismatch += 1
                    continue
            except Exception:
                skipped_load_error += 1
                continue

            # next_state 也要 peek+驗 shape,避免 state.pt 是新格式但 next_state.pt 是舊
            # 格式(版本切換時的混雜狀態)造成 train_step 拿到形狀錯誤的 tensor 而 crash。
            next_state_dst_candidate = None
            next_src_str = entry.get("next_state", entry.get("next_state_path"))
            if next_src_str:
                next_src = Path(next_src_str)
                if next_src.exists():
                    try:
                        next_peek = torch.load(
                            str(next_src), map_location="cpu", weights_only=True
                        )
                        if torch.is_tensor(next_peek) and tuple(next_peek.shape) == expected_shape:
                            next_state_dst_candidate = next_src
                    except Exception:
                        next_state_dst_candidate = None

            storage_id = loaded_count
            state_dst = self.replay_path / f"state_{storage_id}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            if next_state_dst_candidate is not None:
                next_state_dst = self.replay_path / f"next_state_{storage_id}.pt"
                shutil.copy2(str(next_state_dst_candidate), str(next_state_dst))

            runtime_entry = {
                "storage_id": storage_id,
                "state": str(state_dst),
                "action": int(entry["action"]),
                "next_state": str(next_state_dst) if next_state_dst else None,
                "reward": float(entry["reward"]),
                "tail_reward": float(entry.get("tail_reward", entry["reward"])),
                "done": bool(entry["done"]),
                "discount": float(entry.get("discount", 1.0)),
                "n_steps": int(entry.get("n_steps", 1)),
                "reward_type": self.replay_buffer._reward_type(
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
            }
            if "reward_type" in entry:
                runtime_entry["reward_type"] = entry["reward_type"]
            self.replay_buffer.index.append(runtime_entry)
            loaded_count += 1

        self.replay_buffer.size_count = loaded_count
        self.replay_buffer.next_storage_id = loaded_count
        self.replay_buffer.insert_counter = loaded_count
        # 把略過原因攤開,避免 expected_shape 改了之後使用者只看到 "Loaded 0"
        # (例如 v3 從截圖切到 cached features 那一次,所有舊 entries 都會 shape mismatch)。
        skipped_total = skipped_missing + skipped_shape_mismatch + skipped_load_error
        suffix = ""
        if skipped_total > 0:
            suffix = (
                f" | skipped {skipped_total} "
                f"(missing={skipped_missing}, shape!={tuple(expected_shape)}={skipped_shape_mismatch}, "
                f"load_error={skipped_load_error})"
            )
        print(f"{self.log_prefix} Loaded {loaded_count} replay buffer entries{suffix}")

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

    def log_action_image(self, state, log_info, step_count, reward=None) -> None:
        if not getattr(self, "log_actions", True) or log_info is None:
            return
        if not self._should_log_action_image():
            return
        try:
            from PIL import ImageDraw, ImageFont

            self.action_log_path.mkdir(parents=True, exist_ok=True)
            img_array = state.detach().cpu().clamp(0, 1).mul(255).byte().numpy().transpose(1, 2, 0)
            img = Image.fromarray(img_array)
            img_w, img_h = img.size
            cell_w = img_w / self.grid_w
            cell_h = img_h / self.grid_h

            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                font = ImageFont.load_default()

            row, col = log_info["row"], log_info["col"]
            draw.rectangle(
                [int(col * cell_w), int(row * cell_h), int((col + 1) * cell_w), int((row + 1) * cell_h)],
                outline="red",
                width=4,
            )
            for r in range(1, self.grid_h):
                y = int(r * cell_h)
                draw.line([0, y, img_w, y], fill="white", width=1)
            for c in range(1, self.grid_w):
                x = int(c * cell_w)
                draw.line([x, 0, x, img_h], fill="white", width=1)

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

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img.save(self.action_log_path / f"{ts}_step_{step_count:04d}.png")
        except Exception as exc:
            print(f"{self.log_prefix} log_action_image failed: {exc}")
