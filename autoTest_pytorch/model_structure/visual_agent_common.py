from __future__ import annotations

import datetime
import random
import shutil
from collections import defaultdict
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
            "state": state.detach().cpu(),
            "action": int(action),
            "next_state": next_state.detach().cpu() if next_state is not None else None,
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
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
                "total_episodes": self._total_episodes,
                "total_wins": self._total_wins,
                "result_window": list(self._result_window),
            },
            self._optimizer_state_path(),
        )

    def _load_optimizer_state(self) -> None:
        opt_path = self._optimizer_state_path()
        if not opt_path.exists():
            return
        try:
            state = torch.load(opt_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            self.total_it = state.get("total_it", 0)
            self.episode_count = state.get("episode_count", 0)
            self.epsilon = state.get("epsilon", self.epsilon)
            self._total_episodes = state.get("total_episodes", 0)
            self._total_wins = state.get("total_wins", 0)
            self._result_window = self.deque_cls(state.get("result_window", []), maxlen=100)
            print(
                f"{self.log_prefix} Loaded optimizer: total_it={self.total_it}, "
                f"episode={self.episode_count}, epsilon={self.epsilon:.4f}, "
                f"total_episodes={self._total_episodes}, wins={self._total_wins}"
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

    def save_persistent(self) -> None:
        buf = self.replay_buffer
        if buf.size_count == 0:
            return

        reward_groups = defaultdict(list)
        for idx in range(buf.size_count):
            reward_groups[buf.index[idx].get("tail_reward", buf.index[idx]["reward"])].append(idx)

        target = min(self.save_capacity, buf.size_count)
        selected_indices = []
        remaining = target
        groups = sorted(reward_groups.items(), key=lambda item: len(item[1]))
        for group_idx, (_, indices) in enumerate(groups):
            if group_idx == len(groups) - 1:
                count = remaining
            else:
                count = round(len(indices) / buf.size_count * target)
            count = min(count, len(indices), remaining)
            selected_indices.extend(random.sample(indices, count))
            remaining -= count
            if remaining <= 0:
                break

        self.replay_persistent_path.mkdir(parents=True, exist_ok=True)
        for file_path in self.replay_persistent_path.glob("*.pt"):
            file_path.unlink()

        persistent_index = []
        save_idx = 0
        for old_idx in selected_indices:
            old_entry = buf.index[old_idx]
            state_src = Path(old_entry["state"])
            if not state_src.exists():
                continue

            state_dst = self.replay_persistent_path / f"state_{save_idx}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            if old_entry["next_state"]:
                next_src = Path(old_entry["next_state"])
                if next_src.exists():
                    next_state_dst = self.replay_persistent_path / f"next_state_{save_idx}.pt"
                    shutil.copy2(str(next_src), str(next_state_dst))

            persistent_index.append({
                "storage_id": save_idx,
                "state": str(state_dst),
                "action": old_entry["action"],
                "next_state": str(next_state_dst) if next_state_dst else None,
                "reward": old_entry["reward"],
                "tail_reward": float(old_entry.get("tail_reward", old_entry["reward"])),
                "done": old_entry["done"],
                "discount": float(old_entry.get("discount", 1.0)),
                "n_steps": int(old_entry.get("n_steps", 1)),
                "priority": float(old_entry.get("priority", self.priority_min)),
                "reward_type": old_entry.get(
                    "reward_type",
                    buf._reward_type(
                        float(old_entry.get("tail_reward", old_entry["reward"])),
                        bool(old_entry["done"]),
                    ),
                ),
                "insert_order": save_idx + 1,
            })
            save_idx += 1

        torch.save(
            {
                "persistent_index": persistent_index,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
            },
            self._training_state_path(),
        )
        print(f"{self.log_prefix} Persistent save: {len(persistent_index)} entries")

    def _load_persistent_training_state(self) -> None:
        ts_path = self._training_state_path()
        if not ts_path.exists():
            return
        try:
            state = torch.load(ts_path, map_location=self.device, weights_only=False)
            persistent_index = state.get("persistent_index", [])
            if persistent_index:
                self._load_persistent_buffer(persistent_index)
        except Exception as exc:
            message = f"MISSING/FAILED replay/training checkpoint: {ts_path} | error={exc}"
            if hasattr(self, "_log_checkpoint_message"):
                self._log_checkpoint_message(message, warning=True)
            else:
                print(f"{self.log_prefix} {message}")

    def _load_persistent_buffer(self, persistent_index) -> None:
        self.replay_path.mkdir(parents=True, exist_ok=True)
        for file_path in self.replay_path.glob("*.pt"):
            file_path.unlink()

        loaded_count = 0
        self.replay_buffer.index = []
        for entry in persistent_index[: self.replay_buffer.max_size]:
            state_src_str = entry.get("state", entry.get("state_path"))
            if state_src_str is None:
                continue
            state_src = Path(state_src_str)
            if not state_src.exists():
                continue

            try:
                peek = torch.load(str(state_src), map_location="cpu")
                if not torch.is_tensor(peek) or tuple(peek.shape) != (3, *self.image_size):
                    continue
            except Exception:
                continue

            storage_id = loaded_count
            state_dst = self.replay_path / f"state_{storage_id}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            next_src_str = entry.get("next_state", entry.get("next_state_path"))
            if next_src_str:
                next_src = Path(next_src_str)
                if next_src.exists():
                    next_state_dst = self.replay_path / f"next_state_{storage_id}.pt"
                    shutil.copy2(str(next_src), str(next_state_dst))

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
        print(f"{self.log_prefix} Loaded {loaded_count} replay buffer entries")

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
