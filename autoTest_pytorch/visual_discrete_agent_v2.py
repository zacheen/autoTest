"""visual_discrete_agent_v2.py — Stage 2 Agent with full end-to-end training.

Pipeline:
    screenshot (3, 640, 640)
        ↓ YOLOGridStatePredictor  (train mode, unfrozen)
    grid state (12, 6, 6) one-hot
        ↓ TransformerDiscreteAgent backbone + FQF Q-network  (train mode, unfrozen)
    Q-values → masked argmax → action_id

Training:
    Raw screenshots are stored in a DISK-backed replay buffer (so we don't blow
    up RAM).  On each train_step, YOLO converts the batch to grid states WITH
    gradients; the RL loss propagates back through YOLO.

Artifacts (under ./models/visual_transformer_v2_6x6/):
    trainable_yolo.pth / backbone.pth / fqf_network.pth / fqf_target.pth
    optimizer_state.pth
    training_state.pth        (index of persistent replay buffer entries)
    replay_buffer/            (live on-disk replay buffer, auto-written)
    replay_buffer_save/       (curated subset persisted across restarts)
    tensorboard/<timestamp>/  (TB event files)
    train_io_log.txt          (per-step text log)
"""

from __future__ import annotations

import atexit
import datetime
import hashlib
import math
import random
import shutil
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from yolo_grid_state_predictor import YOLOGridStatePredictor
from transformer_discrete_agent import (
    TransformerDiscreteAgent,
    FQF_ENTROPY_COEF,
    NUM_FQF_FRACTIONS,
    _quantile_huber_loss,
)
from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── paths ────────────────────────────────────────────────────────────
YOLO_PREDICTOR_PATH = Path("./models/yolo_grid_predictor/best.pth")

VISUAL_V2_MODEL_PATH            = Path("./models/visual_transformer_v2_6x6")
VISUAL_V2_REPLAY_PATH           = VISUAL_V2_MODEL_PATH / "replay_buffer"
VISUAL_V2_REPLAY_PERSISTENT_PATH = VISUAL_V2_MODEL_PATH / "replay_buffer_save"
VISUAL_V2_TENSORBOARD_DIR       = VISUAL_V2_MODEL_PATH / "tensorboard"
VISUAL_V2_ACTION_LOG_PATH       = VISUAL_V2_MODEL_PATH / "action_logs"

# ── grid / batch ─────────────────────────────────────────────────────
IMAGE_SIZE = (640, 640)
GRID_H = 6
GRID_W = 6
NUM_ACTIONS = GRID_H * GRID_W
VISUAL_BATCH_SIZE   = 20
VISUAL_WARMUP_STEPS = 1000   # 不到這個數量不開始訓練

# ── training hyper-params ────────────────────────────────────────────
VISUAL_GAMMA = 0.7
VISUAL_N_STEP = 1
VISUAL_GRAD_CLIP_NORM = 1.0
TRAIN_EVERY_N_STEPS = 1
TARGET_UPDATE_FREQ = 50
SAVE_EVERY_N_EPISODES = 50
VISUAL_HISTOGRAM_EVERY = 20
USE_AMP = False

# ── YOLO training control ─────────────────────────────────────────────
# False = 診斷模式：backward 會流過 YOLO（可量測 grad），但 grad 在
#         optimizer.step() 前被清零，YOLO 參數不會被更新。
# True  = 完整訓練：YOLO 和 backbone/head 一起更新。
YOLO_UPDATE_ENABLED = False

# ── learning rates per module group ──────────────────────────────────
LR_VISUAL_YOLO     = 2e-5
LR_VISUAL_BACKBONE = 3e-5
LR_VISUAL_HEAD     = 1.5e-5

# ── replay buffer ────────────────────────────────────────────────────
VISUAL_BUFFER_CAPACITY = 2048
VISUAL_SAVE_CAPACITY   = 256
VISUAL_BUFFER_OVERFLOW = 256
VISUAL_PER_ALPHA       = 0.6
VISUAL_PER_UNIFORM_MIX = 0.2
VISUAL_PRIORITY_MIN    = 0.05
VISUAL_PRIORITY_MAX    = 5.0
VISUAL_PRIORITY_EPS    = 1e-3
VISUAL_AGE_DECAY       = 0.002

LOG_ACTIONS = True


class VisualAgentV2:
    """Screenshot → YOLO → Stage 1 Q-network → action, with full end-to-end RL training.

    All parameters are unfrozen and trained via a single combined optimizer.
    Screenshots are persisted to disk by the replay buffer so RAM usage stays
    bounded regardless of buffer capacity.  TensorBoard + a text I/O log
    record per-step losses, gradient norms, BN stats and episode-level metrics.
    """

    def __init__(self, grid_h: int = GRID_H, grid_w: int = GRID_W):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w

        # ── YOLO predictor — unfrozen ──
        self.yolo_predictor = YOLOGridStatePredictor().to(device)
        if not YOLO_PREDICTOR_PATH.exists():
            raise FileNotFoundError(
                f"YOLOGridStatePredictor checkpoint not found: {YOLO_PREDICTOR_PATH}\n"
                "Run python yolo_grid_state_predictor.py first."
            )
        yolo_ckpt = torch.load(YOLO_PREDICTOR_PATH, map_location=device, weights_only=False)
        self.yolo_predictor.load_state_dict(yolo_ckpt["model"])
        print(f"[V2] YOLOGridStatePredictor loaded (best_val_acc={yolo_ckpt.get('best_val_acc', '?'):.4f})")

        # ── Stage 1 agent — unfrozen, but we replace its optimizer + buffer ──
        self.stage1 = TransformerDiscreteAgent(grid_h=grid_h, grid_w=grid_w)
        print("[V2] TransformerDiscreteAgent loaded — training enabled")
        self.backbone  = self.stage1.backbone
        self.q_network = self.stage1.q_network
        self.q_target  = self.stage1.q_target

        # ── combined optimizer (all three modules, separate LR groups) ──
        self.optimizer = optim.AdamW(
            [
                {"params": self.yolo_predictor.parameters(), "lr": LR_VISUAL_YOLO},
                {"params": self.backbone.parameters(),       "lr": LR_VISUAL_BACKBONE},
                {"params": self.q_network.parameters(),      "lr": LR_VISUAL_HEAD},
            ]
        )
        # Keep stage1.optimizer in sync so anything still pointing at it works
        self.stage1.optimizer = self.optimizer
        self.scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

        # ── replay buffer (disk-backed) ──
        VISUAL_V2_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        VISUAL_V2_REPLAY_PATH.mkdir(parents=True, exist_ok=True)
        self.replay_buffer = CategorizedReplayBuffer(
            max_size=VISUAL_BUFFER_CAPACITY,
            storage_mode="disk",
            save_dir=VISUAL_V2_REPLAY_PATH,
            win_threshold=3.0,
            lose_threshold=-1.0,
            invalid_threshold=0.0,
            overflow_margin=VISUAL_BUFFER_OVERFLOW,
            alpha=VISUAL_PER_ALPHA,
            uniform_mix=VISUAL_PER_UNIFORM_MIX,
            priority_min=VISUAL_PRIORITY_MIN,
            priority_max=VISUAL_PRIORITY_MAX,
            priority_eps=VISUAL_PRIORITY_EPS,
            age_decay=VISUAL_AGE_DECAY,
        )
        self.stage1.replay_buffer = self.replay_buffer

        # ── image preprocessing ──
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        # ── episode / step bookkeeping ──
        self.total_it = 0
        self.episode_count = 0
        self.train_every_n_steps = TRAIN_EVERY_N_STEPS
        self.pending_train_steps = 0
        self.n_step = VISUAL_N_STEP
        self.n_step_gamma = VISUAL_GAMMA
        self.n_step_buffer = deque()
        self.recent_real_rewards = deque(maxlen=100)

        # ── adaptive epsilon state (win-rate based) ──
        self.epsilon = 0.30
        self._result_window: deque[int] = deque(maxlen=100)
        self._total_wins = 0
        self._total_episodes = 0

        # ── blocked-action tracking ──
        self.blocked_actions: set[int] = set()
        self.current_state_key: str | None = None

        # ── text log + TensorBoard ──
        VISUAL_V2_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self._io_log = open(VISUAL_V2_MODEL_PATH / "train_io_log.txt", "a", encoding="utf-8")
        self._io_log.write(f"\n{'=' * 60}\n")
        self._io_log.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self._io_log.write(f"{'=' * 60}\n")
        self._io_log.flush()

        tb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log_dir = VISUAL_V2_TENSORBOARD_DIR / tb_timestamp
        self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))
        print(f"[V2] TensorBoard: tensorboard --logdir {VISUAL_V2_TENSORBOARD_DIR}")
        print(f"[V2] Current run: {self.tensorboard_log_dir}")

        self.try_load_model()
        self._set_runtime_modes()
        atexit.register(self.save_persistent)
        atexit.register(self._close_tb_writer)
        atexit.register(self._close_io_log)

    # ──────────────────────────── runtime modes ──────────────────────────

    def _set_runtime_modes(self) -> None:
        self.yolo_predictor.train()
        self.backbone.train()
        self.q_network.train()
        self.q_target.eval()

    @property
    def episode_count_public(self) -> int:
        return self._total_episodes

    # ──────────────────────────── utilities ────────────────────────────

    def preprocess_screen(self, screenshot_path) -> torch.Tensor:
        try:
            image = Image.open(screenshot_path).convert("RGB")
            return self.transform(image)
        except Exception as exc:
            print(f"[V2] preprocess_screen error: {exc}")
            return torch.zeros(3, *IMAGE_SIZE)

    def action_to_grid(self, action_id: int) -> tuple[int, int]:
        return int(action_id) // self.grid_w, int(action_id) % self.grid_w

    def _state_key(self, state: torch.Tensor) -> str:
        arr = state.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()

    # ──────────────────────────── blocked actions ──────────────────────

    def clear_blocked_actions(self, reason: str = "state changed") -> None:
        if self.blocked_actions:
            print(f"[V2] Clear blocked ({reason}): {sorted(self.blocked_actions)}")
        self.blocked_actions.clear()
        self.current_state_key = None

    def block_action_for_state(self, state: torch.Tensor, action_id: int) -> None:
        key = self._state_key(state)
        if self.current_state_key != key:
            self.current_state_key = key
            self.blocked_actions.clear()
        self.blocked_actions.add(int(action_id))
        row, col = self.action_to_grid(action_id)
        print(f"[V2] Block action {action_id} -> ({row},{col})")

    # ──────────────────────────── action selection ─────────────────────

    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> tuple[int, dict]:
        key = self._state_key(state)
        if self.current_state_key != key:
            if self.current_state_key is not None:
                self.clear_blocked_actions(reason="new screenshot")
            self.current_state_key = key

        blocked = set(self.blocked_actions)
        available = [i for i in range(self.num_actions) if i not in blocked]
        if not available:
            self.clear_blocked_actions(reason="all actions blocked")
            self.current_state_key = key
            blocked = set()
            available = list(range(self.num_actions))

        if add_noise and self.epsilon > 0 and random.random() < self.epsilon:
            action_id = random.choice(available)
            row, col = self.action_to_grid(action_id)
            return action_id, {
                "action_id": action_id, "row": row, "col": col,
                "selected_q": None, "top_actions": [],
                "source": "epsilon", "candidate_rank": len(blocked) + 1,
                "blocked_actions": sorted(blocked),
            }

        self._set_runtime_modes()
        self.yolo_predictor.eval()
        screenshot_batch = state.unsqueeze(0).to(device)
        with torch.no_grad():
            grid_state_batch = self.yolo_predictor.predict_grid_state(
                screenshot_batch, self.grid_h, self.grid_w
            )
            features = self.backbone.get_features(grid_state_batch)
            q_2d = self.q_network(features)["q_values"].squeeze(0)
            q_flat = q_2d.view(-1)

            masked_q = q_flat.clone()
            if blocked:
                blocked_idx = torch.tensor(sorted(blocked), dtype=torch.long, device=masked_q.device)
                masked_q[blocked_idx] = float("-inf")

            action_id = int(masked_q.argmax().item())
            topk = min(5, len(available))
            top_vals, top_idx = torch.topk(masked_q, k=topk)
        self._set_runtime_modes()

        row, col = self.action_to_grid(action_id)
        top_actions = [
            (idx.item(), idx.item() // self.grid_w, idx.item() % self.grid_w, float(v))
            for v, idx in zip(top_vals, top_idx)
        ]

        return action_id, {
            "action_id": action_id, "row": row, "col": col,
            "selected_q": float(q_flat[action_id].item()),
            "top_actions": top_actions, "source": "greedy",
            "candidate_rank": len(blocked) + 1,
            "blocked_actions": sorted(blocked),
        }

    # ──────────────────────────── transition storage ───────────────────

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor | None,
        reward: float,
        done: bool,
    ) -> None:
        """Store raw screenshots; buffer writes to disk automatically."""
        self.recent_real_rewards.append(float(reward))
        transition = {
            "state": state.detach().cpu(),
            "action": int(action),                   # single int action_id
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

    # ──────────────────────────── training step ────────────────────────

    def train_step(self):
        buf_size = self.replay_buffer.size()
        if buf_size < VISUAL_WARMUP_STEPS:
            return None   # warm-up: buffer 未達 VISUAL_WARMUP_STEPS 不訓練

        self.total_it += 1
        state, action, next_state, reward, done, sample_indices, is_weights, discounts, n_steps = (
            self.replay_buffer.sample(
                VISUAL_BATCH_SIZE,
                device=device,
                include_extra=True,
            )
        )
        batch_size = state.size(0)
        self._set_runtime_modes()

        # ── target branch (no gradients) ──
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(USE_AMP and device.type == "cuda")):
                next_grid = self.yolo_predictor.predict_grid_state(next_state, self.grid_h, self.grid_w)
                next_features = self.backbone.get_features(next_grid)
                next_online = self.q_network(next_features)
                next_online_q_flat = next_online["q_values"].view(batch_size, -1)
                next_best_flat = next_online_q_flat.argmax(dim=1)
                next_target = self.q_target(next_features)
                next_target_quantiles = next_target["quantiles"][
                    torch.arange(batch_size, device=device), next_best_flat
                ]
                target_quantiles = reward + (1 - done) * discounts * next_target_quantiles

        # ── current branch (gradients flow through YOLO → backbone → head) ──
        # 不呼叫 predict_grid_state（內部有 @torch.no_grad + self.eval()，會完全
        # 阻斷 YOLO 的梯度）。直接呼叫 forward() 取 logits，接 scaled-sigmoid
        # 作為 differentiable 的 grid-state 表示。
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(USE_AMP and device.type == "cuda")):
            # hard one-hot — 與 train_stage1_simple.py 的 input 格式完全相同
            # 以便確認 Q_loss 是否從 Stage 1 繼承（argmax 會阻斷 YOLO 梯度，
            # 但 yolo_pre 仍能量測到 0，代表梯度確實被擋住了）
            yolo_logits = self.yolo_predictor.forward(state, self.grid_h, self.grid_w)
            pred        = yolo_logits.argmax(dim=1, keepdim=True)          # (B, 1, H, W)
            grid_state  = torch.zeros_like(yolo_logits).scatter_(1, pred, 1.0)  # (B, 12, H, W) one-hot
            features = self.backbone.get_features(grid_state)
            q_output = self.q_network(features)
            q_2d           = q_output["q_values"]
            q_quantiles    = q_output["quantiles"]
            tau_hats       = q_output["tau_hats"]
            fraction_probs = q_output["fraction_probs"]

            action_flat = action.long()
            row_idx = action_flat // self.grid_w
            col_idx = action_flat %  self.grid_w
            q_taken = q_2d[torch.arange(batch_size, device=device), row_idx, col_idx].unsqueeze(1)
            chosen_quantiles = q_quantiles[torch.arange(batch_size, device=device), action_flat]

            per_sample_quantile_loss = _quantile_huber_loss(
                current_quantiles=chosen_quantiles.float(),
                target_quantiles=target_quantiles.detach().float(),
                tau_hats=tau_hats.detach().float(),
            )
            entropy = -(fraction_probs * torch.log(fraction_probs + 1e-8)).sum(dim=1, keepdim=True)
            per_sample_loss = per_sample_quantile_loss - FQF_ENTROPY_COEF * entropy.float()
            loss = (is_weights * per_sample_loss).mean()

            target_mean = target_quantiles.mean(dim=1, keepdim=True)
            td_error = (q_taken.detach().float() - target_mean.detach().float()).abs()

        self.replay_buffer.update_priorities(sample_indices, td_error)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # ── pre-clip gradient norms per module (diagnostic) ──
        yolo_pre     = self._module_grad_norm(self.yolo_predictor)
        backbone_pre = self._module_grad_norm(self.backbone)
        head_pre     = self._module_grad_norm(self.q_network)

        # 診斷模式：grad 已記錄，清空 YOLO grad → optimizer.step() 不更新 YOLO
        if not YOLO_UPDATE_ENABLED:
            for p in self.yolo_predictor.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        params_to_clip = (
            list(self.backbone.parameters())
            + list(self.q_network.parameters())
        )
        grad_norm_total = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=VISUAL_GRAD_CLIP_NORM)

        yolo_post     = self._module_grad_norm(self.yolo_predictor)  # 診斷模式は 0
        backbone_post = self._module_grad_norm(self.backbone)
        head_post     = self._module_grad_norm(self.q_network)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.total_it % TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
            self.q_target.eval()

        # ── logging ──
        with torch.no_grad():
            q_mean = q_taken.mean().item()
            real_reward_mean = (
                float(sum(self.recent_real_rewards) / len(self.recent_real_rewards))
                if self.recent_real_rewards else 0.0
            )
            q0 = q_2d[0].view(-1)
            top_vals, top_idx = torch.topk(q0, k=min(5, self.num_actions))
            top_actions = [
                (int(idx), *self.action_to_grid(int(idx)), float(val))
                for val, idx in zip(top_vals.tolist(), top_idx.tolist())
            ]

        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  reward_mean={reward.mean().item():.4f} | done_rate={done.mean().item():.4f}\n"
            f"  real_reward_mean={real_reward_mean:.4f}\n"
            f"  q_top5={top_actions}\n"
            f"  Q_loss={loss.item():.6f} | q_mean={q_mean:.6f} | epsilon={self.epsilon:.4f}\n"
            f"  grad_total={float(grad_norm_total):.6f} | "
            f"yolo_pre={yolo_pre:.6f} backbone_pre={backbone_pre:.6f} head_pre={head_pre:.6f}\n"
            f"---\n"
        )
        self._io_log.flush()

        self.tb_writer.add_scalar("train/Q_loss",           loss.item(),                  self.total_it)
        self.tb_writer.add_scalar("train/q_mean",           q_mean,                       self.total_it)
        self.tb_writer.add_scalar("train/real_reward_mean", real_reward_mean,             self.total_it)
        self.tb_writer.add_scalar("train/done_rate",        done.mean().item(),           self.total_it)
        self.tb_writer.add_scalar("train/epsilon",          self.epsilon,                 self.total_it)
        self.tb_writer.add_scalar("train/buffer_size",      self.replay_buffer.size(),    self.total_it)
        # td_error 診斷：找出 Q_loss 大的原因
        self.tb_writer.add_scalar("train/td_error_mean",   td_error.mean().item(),       self.total_it)
        self.tb_writer.add_scalar("train/td_error_max",    td_error.max().item(),        self.total_it)
        self.tb_writer.add_scalar("train/target_q_mean",   target_quantiles.float().mean().item(), self.total_it)
        self.tb_writer.add_scalar("train/reward_mean_batch", reward.mean().item(),        self.total_it)
        self.tb_writer.add_scalar("grad/total_norm",        float(grad_norm_total),       self.total_it)
        self.tb_writer.add_scalar("grad_pre/yolo",          yolo_pre,                     self.total_it)
        self.tb_writer.add_scalar("grad_pre/backbone",      backbone_pre,                 self.total_it)
        self.tb_writer.add_scalar("grad_pre/head",          head_pre,                     self.total_it)
        self.tb_writer.add_scalar("grad_post/yolo",         yolo_post,                    self.total_it)
        self.tb_writer.add_scalar("grad_post/backbone",     backbone_post,                self.total_it)
        self.tb_writer.add_scalar("grad_post/head",         head_post,                    self.total_it)

        if self.scaler.is_enabled():
            self.tb_writer.add_scalar("train/scaler_scale", self.scaler.get_scale(), self.total_it)

        if self.total_it % VISUAL_HISTOGRAM_EVERY == 0:
            self._log_tensorboard_histograms(self.total_it)
            self._log_batchnorm_stats(self.total_it)

        self.tb_writer.flush()

        return {"Q_loss": loss.item(), "q_mean": q_mean}

    def maybe_train_step(self, force: bool = False):
        self.pending_train_steps += 1
        if self.pending_train_steps < self.train_every_n_steps and not force:
            return None
        self.pending_train_steps = 0
        return self.train_step()

    # ──────────────────────────── episode hooks ─────────────────────────

    def reset_episode(self) -> None:
        self._flush_n_step_buffer()

    def on_episode_end(self) -> None:
        self._flush_n_step_buffer()
        self.episode_count += 1
        self.epsilon = self._adaptive_epsilon()
        self.tb_writer.add_scalar("episode/epsilon", self.epsilon, self.episode_count)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[V2] Periodic save at episode {self.episode_count} | epsilon={self.epsilon:.4f}")
            self.save_persistent()

    def log_episode_metrics(self, win: bool, invalid_click_rate: float, reward_mean: float) -> None:
        self._total_episodes += 1
        self._total_wins += int(win)
        self._result_window.append(int(win))

        rolling_wr = sum(self._result_window) / max(len(self._result_window), 1)
        overall_wr = self._total_wins / max(self._total_episodes, 1)
        next_eps = self._adaptive_epsilon()

        self.tb_writer.add_scalar("episode/reward_mean",        float(reward_mean),        self._total_episodes)
        self.tb_writer.add_scalar("episode/win",                float(bool(win)),          self._total_episodes)
        self.tb_writer.add_scalar("episode/invalid_click_rate", float(invalid_click_rate),self._total_episodes)
        self.tb_writer.add_scalar("episode/win_rate_100",       rolling_wr,                self._total_episodes)
        self.tb_writer.add_scalar("episode/win_rate_all",       overall_wr,                self._total_episodes)
        self.tb_writer.flush()

        status = "WIN " if win else "LOSE"
        print(
            f"[V2] Ep {self._total_episodes}: {status} | "
            f"invalid={invalid_click_rate:.1%} | reward={reward_mean:.3f} | "
            f"win_rate(last100)={rolling_wr:.1%} | win_rate(all)={overall_wr:.1%} | "
            f"eps(next)={next_eps:.4f}"
        )

    def _adaptive_epsilon(
        self,
        wr_min: float = 0.1, wr_max: float = 0.9,
        eps_min: float = 0.001, eps_max: float = 0.30,
    ) -> float:
        """Log-interpolate epsilon from win_rate(last100)."""
        if not self._result_window:
            return eps_max
        wr = sum(self._result_window) / len(self._result_window)
        wr = max(wr_min, min(wr_max, wr))
        t = (wr - wr_min) / (wr_max - wr_min)
        return math.exp(math.log(eps_max) + (math.log(eps_min) - math.log(eps_max)) * t)

    # ──────────────────────────── checkpoints ──────────────────────────

    def _save_model(self) -> None:
        VISUAL_V2_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.yolo_predictor.state_dict(), VISUAL_V2_MODEL_PATH / "trainable_yolo.pth")
        torch.save(self.backbone.state_dict(),       VISUAL_V2_MODEL_PATH / "backbone.pth")
        torch.save(self.q_network.state_dict(),      VISUAL_V2_MODEL_PATH / "fqf_network.pth")
        torch.save(self.q_target.state_dict(),       VISUAL_V2_MODEL_PATH / "fqf_target.pth")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scaler":    self.scaler.state_dict(),
                "total_it":  self.total_it,
                "episode_count": self.episode_count,
                "epsilon":   self.epsilon,
                "total_episodes": self._total_episodes,
                "total_wins":     self._total_wins,
                "result_window":  list(self._result_window),
            },
            VISUAL_V2_MODEL_PATH / "optimizer_state.pth",
        )

    def save_persistent(self) -> None:
        """Copy a curated subset of the on-disk replay buffer to a persistent path."""
        buf = self.replay_buffer
        if buf.size_count == 0:
            return

        reward_groups = defaultdict(list)
        for idx in range(buf.size_count):
            reward_groups[buf.index[idx].get("tail_reward", buf.index[idx]["reward"])].append(idx)

        target = min(VISUAL_SAVE_CAPACITY, buf.size_count)
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

        VISUAL_V2_REPLAY_PERSISTENT_PATH.mkdir(parents=True, exist_ok=True)
        for file_path in VISUAL_V2_REPLAY_PERSISTENT_PATH.glob("*.pt"):
            file_path.unlink()

        persistent_index = []
        save_idx = 0
        for old_idx in selected_indices:
            old_entry = buf.index[old_idx]

            state_src = Path(old_entry["state"])
            if not state_src.exists():
                continue

            state_dst = VISUAL_V2_REPLAY_PERSISTENT_PATH / f"state_{save_idx}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            if old_entry["next_state"]:
                next_src = Path(old_entry["next_state"])
                if next_src.exists():
                    next_state_dst = VISUAL_V2_REPLAY_PERSISTENT_PATH / f"next_state_{save_idx}.pt"
                    shutil.copy2(str(next_src), str(next_state_dst))

            persistent_index.append(
                {
                    "storage_id": save_idx,
                    "state": str(state_dst),
                    "action": old_entry["action"],
                    "next_state": str(next_state_dst) if next_state_dst else None,
                    "reward": old_entry["reward"],
                    "tail_reward": float(old_entry.get("tail_reward", old_entry["reward"])),
                    "done": old_entry["done"],
                    "discount": float(old_entry.get("discount", 1.0)),
                    "n_steps": int(old_entry.get("n_steps", 1)),
                    "priority": float(old_entry.get("priority", VISUAL_PRIORITY_MIN)),
                    "reward_type": old_entry.get(
                        "reward_type",
                        buf._reward_type(
                            float(old_entry.get("tail_reward", old_entry["reward"])),
                            bool(old_entry["done"]),
                        ),
                    ),
                    "insert_order": save_idx + 1,
                }
            )
            save_idx += 1

        torch.save(
            {
                "persistent_index": persistent_index,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
            },
            VISUAL_V2_MODEL_PATH / "training_state.pth",
        )
        print(f"[V2] Persistent save: {len(persistent_index)} entries")

    def try_load_model(self) -> None:
        # YOLO
        yolo_path = VISUAL_V2_MODEL_PATH / "trainable_yolo.pth"
        if yolo_path.exists():
            try:
                self.yolo_predictor.load_state_dict(torch.load(yolo_path, map_location=device))
                print("[V2] Loaded fine-tuned YOLOGridStatePredictor")
            except Exception as exc:
                print(f"[V2] Failed to load trainable YOLO: {exc}")

        # Backbone
        bb_path = VISUAL_V2_MODEL_PATH / "backbone.pth"
        if bb_path.exists():
            try:
                self.backbone.load_state_dict(torch.load(bb_path, map_location=device))
                print("[V2] Loaded fine-tuned backbone")
            except Exception as exc:
                print(f"[V2] Failed to load backbone: {exc}")

        # FQF network / target
        q_path = VISUAL_V2_MODEL_PATH / "fqf_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device))
                print("[V2] Loaded fine-tuned FQF-Network")
            except Exception as exc:
                print(f"[V2] Failed to load FQF-Network: {exc}")

        qt_path = VISUAL_V2_MODEL_PATH / "fqf_target.pth"
        if qt_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(qt_path, map_location=device))
                print("[V2] Loaded fine-tuned FQF-Target")
            except Exception as exc:
                print(f"[V2] Failed to load FQF-Target: {exc}")
        elif q_path.exists():
            self.q_target.load_state_dict(self.q_network.state_dict())
            print("[V2] fqf_target.pth missing, initialized target from fqf_network.pth")

        # Optimizer / scaler / counters
        opt_path = VISUAL_V2_MODEL_PATH / "optimizer_state.pth"
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.optimizer.load_state_dict(state["optimizer"])
                self.total_it       = state.get("total_it", 0)
                self.episode_count  = state.get("episode_count", 0)
                self.epsilon        = state.get("epsilon", self.epsilon)
                self._total_episodes = state.get("total_episodes", 0)
                self._total_wins     = state.get("total_wins", 0)
                self._result_window  = deque(state.get("result_window", []), maxlen=100)
                print(
                    f"[V2] Loaded optimizer: total_it={self.total_it}, "
                    f"episode={self.episode_count}, epsilon={self.epsilon:.4f}, "
                    f"total_episodes={self._total_episodes}, wins={self._total_wins}"
                )
                if "scaler" in state and self.scaler.is_enabled():
                    try:
                        self.scaler.load_state_dict(state["scaler"])
                    except Exception as scaler_exc:
                        print(f"[V2] Failed to load GradScaler state: {scaler_exc}")
            except Exception as exc:
                print(f"[V2] Failed to load optimizer state: {exc}")

        # Replay buffer (persistent)
        ts_path = VISUAL_V2_MODEL_PATH / "training_state.pth"
        if ts_path.exists():
            try:
                state = torch.load(ts_path, map_location=device, weights_only=False)
                persistent_index = state.get("persistent_index", [])
                if persistent_index:
                    self._load_persistent_buffer(persistent_index)
            except Exception as exc:
                print(f"[V2] Failed to load replay buffer: {exc}")

    def _load_persistent_buffer(self, persistent_index) -> None:
        VISUAL_V2_REPLAY_PATH.mkdir(parents=True, exist_ok=True)
        for file_path in VISUAL_V2_REPLAY_PATH.glob("*.pt"):
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

            # Peek at shape — skip entries that don't match current screenshot shape
            try:
                peek = torch.load(str(state_src), map_location="cpu")
                if not torch.is_tensor(peek) or tuple(peek.shape) != (3, *IMAGE_SIZE):
                    continue
            except Exception:
                continue

            storage_id = loaded_count
            state_dst = VISUAL_V2_REPLAY_PATH / f"state_{storage_id}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            next_src_str = entry.get("next_state", entry.get("next_state_path"))
            if next_src_str:
                next_src = Path(next_src_str)
                if next_src.exists():
                    next_state_dst = VISUAL_V2_REPLAY_PATH / f"next_state_{storage_id}.pt"
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
                        VISUAL_PRIORITY_MIN,
                        VISUAL_PRIORITY_MAX,
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
        print(f"[V2] Loaded {loaded_count} replay buffer entries")

    # ──────────────────────────── diagnostics ──────────────────────────

    def _module_grad_norm(self, module) -> float:
        grad_sq_sum = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())
        return grad_sq_sum ** 0.5

    def _log_tensorboard_histograms(self, global_step: int) -> None:
        module_groups = {
            "weights/yolo":     self.yolo_predictor,
            "weights/backbone": self.backbone,
            "weights/head":     self.q_network,
        }
        for tag, module in module_groups.items():
            values = [p.detach().float().reshape(-1).cpu() for p in module.parameters()]
            if values:
                self.tb_writer.add_histogram(tag, torch.cat(values), global_step)
            grads = [
                p.grad.detach().float().reshape(-1).cpu()
                for p in module.parameters() if p.grad is not None
            ]
            if grads:
                self.tb_writer.add_histogram(tag.replace("weights", "grads"), torch.cat(grads), global_step)

    def _log_batchnorm_stats(self, global_step: int) -> None:
        """Track YOLO BN running stats — drift indicates distribution shift."""
        all_means, all_vars = [], []
        for m in self.yolo_predictor.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.running_mean is not None:
                    all_means.append(m.running_mean.detach().float().cpu())
                if m.running_var is not None:
                    all_vars.append(m.running_var.detach().float().cpu())
        if all_means:
            means_cat = torch.cat(all_means)
            self.tb_writer.add_scalar("bn/running_mean_avg", means_cat.mean().item(), global_step)
            self.tb_writer.add_scalar("bn/running_mean_std", means_cat.std().item(),  global_step)
        if all_vars:
            vars_cat = torch.cat(all_vars)
            self.tb_writer.add_scalar("bn/running_var_avg",  vars_cat.mean().item(),  global_step)
            self.tb_writer.add_scalar("bn/running_var_std",  vars_cat.std().item(),   global_step)

    # ──────────────────────────── cleanup ──────────────────────────────

    def _close_io_log(self) -> None:
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def _close_tb_writer(self) -> None:
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.close()

    # ──────────────────────────── action image log ─────────────────────

    def log_action_image(self, state, log_info, step_count, reward=None) -> None:
        if not LOG_ACTIONS or log_info is None:
            return
        try:
            from PIL import ImageDraw, ImageFont

            VISUAL_V2_ACTION_LOG_PATH.mkdir(parents=True, exist_ok=True)
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
                [int(col * cell_w), int(row * cell_h),
                 int((col + 1) * cell_w), int((row + 1) * cell_h)],
                outline="red", width=4,
            )
            for r in range(1, self.grid_h):
                y = int(r * cell_h); draw.line([0, y, img_w, y], fill="white", width=1)
            for c in range(1, self.grid_w):
                x = int(c * cell_w); draw.line([x, 0, x, img_h], fill="white", width=1)

            lines = [
                f"Step: {step_count}",
                f"Action: {log_info['action_id']} -> ({row},{col})",
                f"Source: {log_info.get('source','?')}",
            ]
            if log_info.get("selected_q") is not None:
                lines.append(f"Q: {log_info['selected_q']:.4f}")
            if reward is not None:
                lines.append(f"Reward: {reward:.1f}")

            text_y = 5
            for line in lines:
                bbox = draw.textbbox((5, text_y), line, font=font)
                draw.rectangle(bbox, fill="black")
                draw.text((5, text_y), line, fill="white", font=font)
                text_y += 15

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img.save(VISUAL_V2_ACTION_LOG_PATH / f"{ts}_step_{step_count:04d}.png")
        except Exception as exc:
            print(f"[V2] log_action_image failed: {exc}")


# ──────────────────────────── factory ────────────────────────────────

_agent: VisualAgentV2 | None = None


def get_agent(screen_region=None) -> VisualAgentV2:
    global _agent
    if _agent is None:
        _agent = VisualAgentV2()
    return _agent
