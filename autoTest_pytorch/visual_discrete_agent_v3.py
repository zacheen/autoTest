"""visual_discrete_agent_v3.py — Stage 2 Agent（screenshot → FQF Q-network）。

Pipeline:
    screenshot (3, 640, 640)
        ↓ YOLOEncoderBase（FROZEN，從 YOLOGridStatePredictor checkpoint 載入）
        ↓   YOLO11n backbone → (128, 40, 40)
        ↓   token_adapter + 2D sinusoidal pos encoding → (1600, 128)
        ↓   HierarchicalEncoder [128→64→32] → (1600, 32)
    encoded memory (B, 1600, 32)
        ↓ TransformerDecoder × 2 layers（cross-attn，36 learned query tokens）
    decoded features (B, 36, 32)
        ↓ FQFQNetwork (d_model=32, num_fractions=8)
    Q-values (B, 6, 6) → masked argmax → action

架構說明：
    • VisualBackboneV3 繼承 YOLOEncoderBase（與 YOLOGridStatePredictor 共用）
    • YOLO + token_adapter + encoder 全部凍結（BN eval mode）
    • 可訓練部分：decoder + query tokens + FQF head
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
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from transformer_discrete_agent import FQF_ENTROPY_COEF, NUM_FQF_FRACTIONS, _quantile_huber_loss
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from model_structure.transformer_shared import FQFQNetwork
from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer
from model_structure.visual_agent_common import VisualAgentCommonMixin
from model_structure.yolo_encoder_base import YOLOEncoderBase, DEFAULT_ENCODER_DIMS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── debug logger (寫到檔案，CMD 刷掉也能看) ──────────────────────────
import logging as _logging
_dbg_log_path = Path("./models/visual_transformer_v3_6x6/cuda_debug.log")
_dbg_log_path.parent.mkdir(parents=True, exist_ok=True)
_dbg_logger = _logging.getLogger("cuda_dbg")
_dbg_logger.setLevel(_logging.DEBUG)
if not _dbg_logger.handlers:
    _fh = _logging.FileHandler(_dbg_log_path, mode="a", encoding="utf-8")
    _fh.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
    _dbg_logger.addHandler(_fh)

def _dbg(msg: str) -> None:
    """sync GPU then log — error surfaces at the exact op that caused it."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    _dbg_logger.debug(msg)
    for _h in _dbg_logger.handlers:
        try: _h.flush()
        except Exception: pass

def _dbg_mem(tag: str) -> None:
    """log GPU memory usage."""
    if device.type != "cuda":
        return
    try:
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        _dbg_logger.debug(f"[MEM {tag}] alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB")
    except Exception as e:
        _dbg_logger.debug(f"[MEM {tag}] failed: {e}")

def _dbg_tensor(name: str, t, *, expect_max=None, expect_min=None, check_finite: bool = True) -> None:
    """Check a tensor for NaN/Inf and out-of-range values; log shape, dtype, range.

    expect_max/expect_min: hard bounds; logs FATAL if violated (likely bad index).
    """
    try:
        if t is None:
            _dbg_logger.debug(f"[TENSOR {name}] is None"); return
        if not torch.is_tensor(t):
            _dbg_logger.debug(f"[TENSOR {name}] type={type(t).__name__}"); return
        # sync first so any pending error surfaces here, not later
        if t.is_cuda:
            torch.cuda.synchronize()
        info = f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"
        if t.numel() == 0:
            _dbg_logger.debug(f"[TENSOR {name}] {info} EMPTY"); return
        # only check stats on numeric tensors
        if t.dtype.is_floating_point:
            tmin = t.min().item(); tmax = t.max().item()
            tnan = bool(torch.isnan(t).any().item()) if check_finite else False
            tinf = bool(torch.isinf(t).any().item()) if check_finite else False
            tag = ""
            if tnan: tag += " !!NAN!!"
            if tinf: tag += " !!INF!!"
            _dbg_logger.debug(f"[TENSOR {name}] {info} min={tmin:.4g} max={tmax:.4g}{tag}")
        else:
            tmin = t.min().item(); tmax = t.max().item()
            tag = ""
            if expect_max is not None and tmax >= expect_max:
                tag += f" !!OOB max>={expect_max}!!"
            if expect_min is not None and tmin < expect_min:
                tag += f" !!OOB min<{expect_min}!!"
            _dbg_logger.debug(f"[TENSOR {name}] {info} min={tmin} max={tmax}{tag}")
        for _h in _dbg_logger.handlers:
            try: _h.flush()
            except Exception: pass
    except Exception as e:
        _dbg_logger.debug(f"[TENSOR {name}] CHECK FAILED: {e!r}")

# ── paths ────────────────────────────────────────────────────────────
YOLO_PREDICTOR_PATH = Path("./models/yolo_grid_predictor/best.pth")

VISUAL_V3_MODEL_PATH             = Path("./models/visual_transformer_v3_6x6")
VISUAL_V3_REPLAY_PATH            = VISUAL_V3_MODEL_PATH / "replay_buffer"
VISUAL_V3_REPLAY_PERSISTENT_PATH = VISUAL_V3_MODEL_PATH / "replay_buffer_save"
VISUAL_V3_TENSORBOARD_DIR        = VISUAL_V3_MODEL_PATH / "tensorboard"
VISUAL_V3_ACTION_LOG_PATH        = VISUAL_V3_MODEL_PATH / "action_logs"

# ── grid / batch ─────────────────────────────────────────────────────
IMAGE_SIZE = (640, 640)
GRID_H = 6
GRID_W = 6
NUM_ACTIONS = GRID_H * GRID_W
VISUAL_BATCH_SIZE   = 32
VISUAL_WARMUP_STEPS = 300

# ── Encoder dims（與 YOLOGridStatePredictor 共用 DEFAULT_ENCODER_DIMS = [128,64,32]）──
ENCODER_DIMS = DEFAULT_ENCODER_DIMS

# ── Decoder spec ─────────────────────────────────────────────────────
DECODER_D_MODEL    = ENCODER_DIMS[-1]   # 32
DECODER_NHEAD      = 4
DECODER_NUM_LAYERS = 2
DECODER_FF_DIM     = 128
DECODER_DROPOUT    = 0.1

# ── training hyper-params ────────────────────────────────────────────
VISUAL_GAMMA = 0.7
VISUAL_N_STEP = 1
VISUAL_GRAD_CLIP_NORM = 5.0   # 1600-token encoder, allow larger grad room early
TRAIN_EVERY_N_STEPS = 1
TARGET_UPDATE_FREQ = 50
SAVE_EVERY_N_EPISODES = 50
VISUAL_HISTOGRAM_EVERY = 20
WEIGHT_DISTANCE_LOG_EVERY = 100
USE_AMP = False

# ── learning rates ───────────────────────────────────────────────────
LR_VISUAL_BACKBONE = 5e-5   # adapter + encoder + decoder + queries (random init → larger LR)
LR_VISUAL_HEAD     = 5e-5
# Linear LR warmup over the first N optimizer steps (transformer 早期穩定)
# 從 base_lr * LR_WARMUP_START_FACTOR 線性增加到 base_lr
LR_WARMUP_STEPS         = 1000
LR_WARMUP_START_FACTOR  = 0.0

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


# ════════════════════════════════════════════════════════════════════════
# VisualBackboneV3 — YOLOEncoderBase（frozen）+ cross-attn decoder
# ════════════════════════════════════════════════════════════════════════
class VisualBackboneV3(YOLOEncoderBase):
    """screenshot (B,3,H,W) → 36 cell features (B, 36, 32).

    YOLO + token_adapter + HierarchicalEncoder 繼承自 YOLOEncoderBase，
    並在初始化時從 YOLOGridStatePredictor checkpoint 載入後凍結。
    Decoder（cross-attn + queries）為 random init，是唯一可訓練的部分。
    """

    def __init__(
        self,
        grid_h: int = GRID_H,
        grid_w: int = GRID_W,
        decoder_nhead: int = DECODER_NHEAD,
        decoder_num_layers: int = DECODER_NUM_LAYERS,
        decoder_ff_dim: int = DECODER_FF_DIM,
        decoder_dropout: float = DECODER_DROPOUT,
    ):
        super().__init__(
            encoder_dims=ENCODER_DIMS,
            nhead=DECODER_NHEAD,
            ff_mult=4,
            dropout=decoder_dropout,
        )
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_queries = grid_h * grid_w
        out_dim = self.out_dim   # 32

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, out_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=out_dim,
            nhead=decoder_nhead,
            dim_feedforward=decoder_ff_dim,
            dropout=decoder_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)

    def _build_queries(self, batch_size: int) -> torch.Tensor:
        return self.query_tokens.expand(batch_size, -1, -1)

    def get_features(self, screenshot: torch.Tensor) -> torch.Tensor:
        """screenshot (B, 3, H, W) → cell features (B, 36, out_dim)."""
        memory  = self.encode(screenshot)                        # (B, 1600, 32)
        queries = self._build_queries(memory.size(0))            # (B, 36,   32)
        return self.decoder(queries, memory)                     # (B, 36,   32)

    def forward(self, screenshot: torch.Tensor) -> torch.Tensor:
        return self.get_features(screenshot)


# ════════════════════════════════════════════════════════════════════════
# VisualAgentV3 — top-level RL agent
# ════════════════════════════════════════════════════════════════════════
class VisualAgentV3(VisualAgentCommonMixin):
    """Visual FQF agent: screenshot → frozen YOLO → custom transformer → 36 actions."""

    def __init__(self, screen_region=None, grid_h: int = GRID_H, grid_w: int = GRID_W):
        self.screen_region = screen_region
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w
        self.device = device
        self.deque_cls = deque
        self.model_path = VISUAL_V3_MODEL_PATH
        self.replay_path = VISUAL_V3_REPLAY_PATH
        self.replay_persistent_path = VISUAL_V3_REPLAY_PERSISTENT_PATH
        self.action_log_path = VISUAL_V3_ACTION_LOG_PATH
        self.image_size = IMAGE_SIZE
        self.save_capacity = VISUAL_SAVE_CAPACITY
        self.priority_min = VISUAL_PRIORITY_MIN
        self.priority_max = VISUAL_PRIORITY_MAX
        self.log_prefix = "[V3]"
        self.log_actions = LOG_ACTIONS

        # ── Backbone：載入 Predictor 權重後凍結 YOLO + encoder ──
        self.backbone = VisualBackboneV3(grid_h=grid_h, grid_w=grid_w).to(device)
        self.backbone.load_encoder_weights_from_checkpoint(YOLO_PREDICTOR_PATH)
        self.backbone.freeze()

        # ── FQF heads (random init) ──
        self.q_network = FQFQNetwork(
            d_model=self.backbone.out_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            num_fractions=NUM_FQF_FRACTIONS,
            hidden_dim=64,
        ).to(device)
        self.q_target = FQFQNetwork(
            d_model=self.backbone.out_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            num_fractions=NUM_FQF_FRACTIONS,
            hidden_dim=64,
        ).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()

        # ── Optimizer（只更新 backbone decoder + queries，YOLO+encoder 已凍結）──
        backbone_trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW([
            {"params": backbone_trainable,          "lr": LR_VISUAL_BACKBONE},
            {"params": self.q_network.parameters(), "lr": LR_VISUAL_HEAD},
        ])
        # 紀錄每個 param group 的 base lr，warmup 期間根據 total_it 動態縮放
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

        # ── replay buffer (disk-backed) ──
        VISUAL_V3_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        VISUAL_V3_REPLAY_PATH.mkdir(parents=True, exist_ok=True)
        self.replay_buffer = CategorizedReplayBuffer(
            max_size=VISUAL_BUFFER_CAPACITY,
            storage_mode="disk",
            save_dir=VISUAL_V3_REPLAY_PATH,
            win_threshold=MINESWEEPER_REWARD_CONFIG.replay_win_threshold,
            lose_threshold=MINESWEEPER_REWARD_CONFIG.replay_lose_threshold,
            invalid_threshold=MINESWEEPER_REWARD_CONFIG.replay_invalid_threshold,
            overflow_margin=VISUAL_BUFFER_OVERFLOW,
            alpha=VISUAL_PER_ALPHA,
            uniform_mix=VISUAL_PER_UNIFORM_MIX,
            priority_min=VISUAL_PRIORITY_MIN,
            priority_max=VISUAL_PRIORITY_MAX,
            priority_eps=VISUAL_PRIORITY_EPS,
            age_decay=VISUAL_AGE_DECAY,
        )

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

        # ── adaptive epsilon ──
        self.epsilon = 0.30
        self._result_window: deque[int] = deque(maxlen=100)
        self._total_wins = 0
        self._total_episodes = 0

        # ── blocked-action tracking (episode-scoped) ──
        self.blocked_actions: set[int] = set()

        # ── action-image logging (record full episode every N episodes) ──
        self.action_log_every_n_episodes = 10
        self._log_actions_this_episode = False

        # ── text log + TensorBoard ──
        VISUAL_V3_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self._io_log = open(VISUAL_V3_MODEL_PATH / "train_io_log.txt", "a", encoding="utf-8")
        self._io_log.write(f"\n{'=' * 60}\n")
        self._io_log.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self._io_log.write(f"Encoder dims: {ENCODER_DIMS}\n")
        self._io_log.write(f"{'=' * 60}\n")
        self._io_log.flush()

        tb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log_dir = VISUAL_V3_TENSORBOARD_DIR / tb_timestamp
        self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))
        print(f"[V3] TensorBoard: tensorboard --logdir {VISUAL_V3_TENSORBOARD_DIR}")
        print(f"[V3] Current run: {self.tensorboard_log_dir}")

        self.try_load_model()
        self._init_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference_step = self.total_it
        self._set_runtime_modes()
        atexit.register(self.save_persistent)
        atexit.register(self._close_tb_writer)
        atexit.register(self._close_io_log)

    # ──────────────────────────── runtime modes ──────────────────────────

    def _set_runtime_modes(self) -> None:
        self.backbone.train()          # decoder + queries → train mode
        self.backbone.set_bn_eval()    # YOLO BN layers 保持 eval（凍結 running stats）
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
            print(f"[V3] preprocess_screen error: {exc}")
            return torch.zeros(3, *IMAGE_SIZE)

    def action_to_grid(self, action_id: int) -> tuple[int, int]:
        return int(action_id) // self.grid_w, int(action_id) % self.grid_w

    def _state_key(self, state: torch.Tensor) -> str:
        arr = state.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()

    # ──────────────────────────── blocked actions ──────────────────────

    def clear_blocked_actions(self, reason: str = "episode reset") -> None:
        if self.blocked_actions:
            print(f"[V3] Clear blocked ({reason}): {sorted(self.blocked_actions)}")
        self.blocked_actions.clear()

    def block_action_for_state(self, state: torch.Tensor, action_id: int) -> None:
        self.blocked_actions.add(int(action_id))
        row, col = self.action_to_grid(action_id)
        print(f"[V3] Block action {action_id} -> ({row},{col})")

    # ──────────────────────────── action selection ─────────────────────

    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> tuple[int, dict]:
        blocked = set(self.blocked_actions)
        available = [i for i in range(self.num_actions) if i not in blocked]
        if not available:
            self.clear_blocked_actions(reason="all actions blocked")
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
        _dbg(f"[select_action] ENTER total_it={self.total_it} blocked_size={len(blocked)} available_size={len(available)}")
        _dbg_mem("select_action ENTER")
        _dbg_tensor("select_action.state(input)", state)
        _dbg("[select_action] before state.to(device)")
        screenshot_batch = state.unsqueeze(0).to(device)
        _dbg_tensor("select_action.screenshot_batch", screenshot_batch)
        _dbg("[select_action] after state.to(device)")
        with torch.no_grad():
            _dbg("[select_action] before backbone")
            features  = self.backbone.get_features(screenshot_batch)
            _dbg_tensor("select_action.features", features)
            _dbg("[select_action] after backbone")
            q_2d = self.q_network(features)["q_values"].squeeze(0)
            _dbg_tensor("select_action.q_2d", q_2d)
            _dbg("[select_action] after q_network")
            q_flat = q_2d.view(-1)
            _dbg_tensor("select_action.q_flat", q_flat)
            _dbg(f"[select_action] q_flat.numel()={q_flat.numel()} num_actions={self.num_actions}")

            masked_q = q_flat.clone()
            if blocked:
                blocked_sorted = sorted(blocked)
                # bounds-check BEFORE indexing — out-of-range = illegal memory access
                bad = [b for b in blocked_sorted if not (0 <= b < q_flat.numel())]
                if bad:
                    _dbg(f"[select_action] !!OOB blocked indices {bad} vs numel={q_flat.numel()}!!")
                blocked_idx = torch.tensor(blocked_sorted, dtype=torch.long, device=masked_q.device)
                _dbg_tensor("select_action.blocked_idx", blocked_idx,
                            expect_min=0, expect_max=q_flat.numel())
                masked_q[blocked_idx] = float("-inf")
                _dbg("[select_action] after blocked-mask scatter")

            action_id = int(masked_q.argmax().item())
            _dbg(f"[select_action] action_id={action_id} (num_actions={self.num_actions})")
            topk = min(5, len(available))
            top_vals, top_idx = torch.topk(masked_q, k=topk)
        _dbg("[select_action] done")
        _dbg_mem("select_action EXIT")
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

    # ──────────────────────────── training step ────────────────────────

    def train_step(self):
        buf_size = self.replay_buffer.size()
        if buf_size < VISUAL_WARMUP_STEPS:
            return None

        self.total_it += 1
        self._apply_lr_warmup()
        _dbg(f"[train_step] ENTER total_it={self.total_it} buf_size={buf_size}")
        _dbg_mem("train_step ENTER")
        _dbg("[train_step] before replay_buffer.sample")
        state, action, next_state, reward, done, sample_indices, is_weights, discounts, n_steps = (
            self.replay_buffer.sample(
                VISUAL_BATCH_SIZE,
                device=device,
                include_extra=True,
            )
        )
        _dbg("[train_step] after replay_buffer.sample")
        # ---- sanity-check sampled tensors ----
        _dbg_tensor("train_step.state",      state)
        _dbg_tensor("train_step.next_state", next_state)
        _dbg_tensor("train_step.action",     action,
                    expect_min=0, expect_max=self.num_actions)
        _dbg_tensor("train_step.reward",     reward)
        _dbg_tensor("train_step.done",       done)
        _dbg_tensor("train_step.is_weights", is_weights)
        _dbg_tensor("train_step.discounts",  discounts)
        batch_size = state.size(0)
        _dbg(f"[train_step] batch_size={batch_size} num_actions={self.num_actions} grid={self.grid_h}x{self.grid_w}")
        self._set_runtime_modes()

        # ── target branch (no gradients) ──
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(USE_AMP and device.type == "cuda")):
                _dbg("[train_step] before next backbone")
                next_features  = self.backbone.get_features(next_state)
                _dbg_tensor("train_step.next_features", next_features)
                _dbg("[train_step] after next backbone")
                next_online    = self.q_network(next_features)
                _dbg_tensor("train_step.next_online.q_values", next_online["q_values"])
                next_online_q_flat = next_online["q_values"].view(batch_size, -1)
                _dbg_tensor("train_step.next_online_q_flat", next_online_q_flat)
                next_best_flat = next_online_q_flat.argmax(dim=1)
                _dbg_tensor("train_step.next_best_flat", next_best_flat,
                            expect_min=0, expect_max=self.num_actions)
                next_target    = self.q_target(next_features)
                _dbg_tensor("train_step.next_target.quantiles", next_target["quantiles"])
                _dbg("[train_step] after q_target")
                next_target_quantiles = next_target["quantiles"][
                    torch.arange(batch_size, device=device), next_best_flat
                ]
                _dbg_tensor("train_step.next_target_quantiles", next_target_quantiles)
                target_quantiles = reward + (1 - done) * discounts * next_target_quantiles
                _dbg_tensor("train_step.target_quantiles", target_quantiles)
        _dbg("[train_step] target branch done")

        # ── current branch (gradients flow through decoder + FQF；YOLO+encoder 已凍結）──
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(USE_AMP and device.type == "cuda")):
            features  = self.backbone.get_features(state)
            _dbg_tensor("train_step.features", features)
            _dbg("[train_step] after current backbone")
            q_output  = self.q_network(features)
            q_2d           = q_output["q_values"]
            q_quantiles    = q_output["quantiles"]
            tau_hats       = q_output["tau_hats"]
            fraction_probs = q_output["fraction_probs"]
            _dbg_tensor("train_step.q_2d", q_2d)
            _dbg_tensor("train_step.q_quantiles", q_quantiles)
            _dbg_tensor("train_step.tau_hats", tau_hats)
            _dbg_tensor("train_step.fraction_probs", fraction_probs)

            action_flat = action.long()
            _dbg_tensor("train_step.action_flat", action_flat,
                        expect_min=0, expect_max=self.num_actions)
            row_idx = action_flat // self.grid_w
            col_idx = action_flat %  self.grid_w
            _dbg_tensor("train_step.row_idx", row_idx, expect_min=0, expect_max=self.grid_h)
            _dbg_tensor("train_step.col_idx", col_idx, expect_min=0, expect_max=self.grid_w)
            _dbg(f"[train_step] q_2d.shape={tuple(q_2d.shape)} q_quantiles.shape={tuple(q_quantiles.shape)}")
            q_taken = q_2d[torch.arange(batch_size, device=device), row_idx, col_idx].unsqueeze(1)
            _dbg("[train_step] after q_taken gather")
            chosen_quantiles = q_quantiles[torch.arange(batch_size, device=device), action_flat]
            _dbg("[train_step] after chosen_quantiles gather")

            per_sample_quantile_loss, frac_clipped = _quantile_huber_loss(
                current_quantiles=chosen_quantiles.float(),
                target_quantiles=target_quantiles.detach().float(),
                tau_hats=tau_hats.detach().float(),
                return_stats=True,
            )
            entropy = -(fraction_probs * torch.log(fraction_probs + 1e-8)).sum(dim=1, keepdim=True)
            per_sample_loss = per_sample_quantile_loss - FQF_ENTROPY_COEF * entropy.float()
            loss = (is_weights * per_sample_loss).mean()

            target_mean = target_quantiles.mean(dim=1, keepdim=True)
            td_error = (q_taken.detach().float() - target_mean.detach().float()).abs()
            fpn_norm_entropy = (entropy.mean() / math.log(NUM_FQF_FRACTIONS)).item()
            fpn_tau_std = tau_hats.std(dim=1).mean().item()

        _dbg_tensor("train_step.loss", loss)
        _dbg_tensor("train_step.td_error", td_error)
        self.replay_buffer.update_priorities(sample_indices, td_error)
        _dbg("[train_step] after update_priorities")

        self.optimizer.zero_grad(set_to_none=True)
        _dbg("[train_step] before backward")
        self.scaler.scale(loss).backward()
        _dbg("[train_step] after backward")
        _dbg_mem("train_step after backward")
        self.scaler.unscale_(self.optimizer)
        _dbg("[train_step] after unscale_")

        # ── pre-clip gradient norms (diagnostic) ──
        backbone_pre = self._module_grad_norm(self.backbone)
        head_pre     = self._module_grad_norm(self.q_network)
        _dbg(f"[train_step] grad backbone_pre={backbone_pre:.4g} head_pre={head_pre:.4g}")

        params_to_clip = (
            list(self.backbone.parameters())
            + list(self.q_network.parameters())
        )
        grad_norm_total = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=VISUAL_GRAD_CLIP_NORM)
        _dbg(f"[train_step] after clip grad_norm_total={float(grad_norm_total):.4g}")

        backbone_post = self._module_grad_norm(self.backbone)
        head_post     = self._module_grad_norm(self.q_network)

        _dbg("[train_step] before optimizer.step")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        _dbg("[train_step] after optimizer.step")
        _dbg_mem("train_step after optimizer.step")

        if self.total_it % TARGET_UPDATE_FREQ == 0:
            _dbg("[train_step] before target update")
            self.q_target.load_state_dict(self.q_network.state_dict())
            self.q_target.eval()
            _dbg("[train_step] after target update")

        weight_distance_log = None
        if self.total_it % WEIGHT_DISTANCE_LOG_EVERY == 0:
            current_snapshot = self._capture_trainable_weight_snapshot()
            init_distance = self._snapshot_distance(current_snapshot, self._init_weight_reference)
            rolling_distance = self._snapshot_distance(current_snapshot, self._rolling_weight_reference)
            rolling_step_gap = self.total_it - self._rolling_weight_reference_step
            weight_distance_log = {
                "from_init": init_distance,
                "from_prev_window": rolling_distance,
                "prev_window_step": self._rolling_weight_reference_step,
                "window_size": rolling_step_gap,
            }
            self._rolling_weight_reference = current_snapshot
            self._rolling_weight_reference_step = self.total_it

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

        q_ratio_str = f"{q_mean / real_reward_mean:.3f}" if abs(real_reward_mean) > 0.1 else "n/a"
        weight_delta_line = ""
        if weight_distance_log is not None:
            weight_delta_line = (
                f"  weight_delta_init={weight_distance_log['from_init']:.6e} | "
                f"weight_delta_prev_{weight_distance_log['window_size']}="
                f"{weight_distance_log['from_prev_window']:.6e} "
                f"(ref_step={weight_distance_log['prev_window_step']})\n"
            )
        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  real_reward_mean={real_reward_mean:.4f}\n"
            f"  q_top5={top_actions}\n"
            f"  Q_loss={loss.item():.6f} | q_mean={q_mean:.6f} | epsilon={self.epsilon:.4f}\n"
            f"  td_error_norm={td_error.mean().item():.6f} | q/reward_ratio={q_ratio_str} | frac_clipped={frac_clipped:.3f}\n"
            f"  fpn_norm_entropy={fpn_norm_entropy:.4f} | fpn_tau_std={fpn_tau_std:.4f}\n"
            f"  grad_total={float(grad_norm_total):.6f} | "
            f"backbone_pre={backbone_pre:.6f} head_pre={head_pre:.6f}\n"
            f"{weight_delta_line}"
            f"---\n"
        )
        self._io_log.flush()

        self.tb_writer.add_scalar("train/Q_loss",            loss.item(),                  self.total_it)
        self.tb_writer.add_scalar("train/q_mean",            q_mean,                       self.total_it)
        self.tb_writer.add_scalar("train/real_reward_mean",  real_reward_mean,             self.total_it)
        self.tb_writer.add_scalar("train/epsilon",           self.epsilon,                 self.total_it)
        self.tb_writer.add_scalar("train/frac_huber_clipped", frac_clipped,                 self.total_it)
        self.tb_writer.add_scalar("fpn/norm_entropy",         fpn_norm_entropy,             self.total_it)
        self.tb_writer.add_scalar("fpn/tau_std",              fpn_tau_std,                  self.total_it)
        self.tb_writer.add_scalar("train/td_error_norm",     td_error.mean().item(),       self.total_it)
        self.tb_writer.add_scalar("train/td_error_max",      td_error.max().item(),        self.total_it)
        self.tb_writer.add_scalar("train/target_q_mean",     target_quantiles.float().mean().item(), self.total_it)
        if abs(real_reward_mean) > 0.1:
            self.tb_writer.add_scalar("train/q_over_reward_ratio", q_mean / real_reward_mean, self.total_it)
        self.tb_writer.add_scalar("grad/total_norm",         float(grad_norm_total),       self.total_it)
        self.tb_writer.add_scalar("grad_pre/backbone",       backbone_pre,                 self.total_it)
        self.tb_writer.add_scalar("grad_pre/head",           head_pre,                     self.total_it)
        self.tb_writer.add_scalar("grad_post/backbone",      backbone_post,                self.total_it)
        self.tb_writer.add_scalar("grad_post/head",          head_post,                    self.total_it)
        if weight_distance_log is not None:
            self.tb_writer.add_scalar("weights/delta_from_init", weight_distance_log["from_init"], self.total_it)
            self.tb_writer.add_scalar("weights/delta_from_prev_window", weight_distance_log["from_prev_window"], self.total_it)

        if self.scaler.is_enabled():
            self.tb_writer.add_scalar("train/scaler_scale", self.scaler.get_scale(), self.total_it)

        if self.total_it % VISUAL_HISTOGRAM_EVERY == 0:
            self._log_backbone_weight_norms(self.total_it)

        self.tb_writer.flush()
        _dbg(f"[train_step] EXIT total_it={self.total_it}")
        _dbg_mem("train_step EXIT")
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
        self.clear_blocked_actions(reason="episode reset")
        next_episode_idx = self._total_episodes + 1
        self._log_actions_this_episode = (
            self.action_log_every_n_episodes > 0
            and next_episode_idx % self.action_log_every_n_episodes == 0
        )
        if self._log_actions_this_episode:
            print(f"[V3] Action-image logging enabled for episode {next_episode_idx}")

    def on_episode_end(self) -> None:
        self._flush_n_step_buffer()
        self.episode_count += 1
        self.epsilon = self._adaptive_epsilon()
        self.tb_writer.add_scalar("episode/epsilon", self.epsilon, self.episode_count)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[V3] Periodic save at episode {self.episode_count} | epsilon={self.epsilon:.4f}")
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
            f"[V3] Ep {self._total_episodes}: {status} | "
            f"invalid={invalid_click_rate:.1%} | reward={reward_mean:.3f} | "
            f"win_rate(last100)={rolling_wr:.1%} | win_rate(all)={overall_wr:.1%} | "
            f"eps(next)={next_eps:.4f}"
        )

    def _adaptive_epsilon(
        self,
        wr_min: float = 0.1, wr_max: float = 0.85,
        eps_min: float = 0.001, eps_max: float = 0.30,
    ) -> float:
        if not self._result_window:
            return eps_max
        wr = sum(self._result_window) / len(self._result_window)
        wr = max(wr_min, min(wr_max, wr))
        t = (wr - wr_min) / (wr_max - wr_min)
        return math.exp(math.log(eps_max) + (math.log(eps_min) - math.log(eps_max)) * t)

    # ──────────────────────────── checkpoints ──────────────────────────

    def _save_model(self) -> None:
        VISUAL_V3_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(),  VISUAL_V3_MODEL_PATH / "backbone.pth")
        torch.save(self.q_network.state_dict(), VISUAL_V3_MODEL_PATH / "fqf_network.pth")
        torch.save(self.q_target.state_dict(),  VISUAL_V3_MODEL_PATH / "fqf_target.pth")
        self._save_optimizer_state()

    def save_persistent(self) -> None:
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

        VISUAL_V3_REPLAY_PERSISTENT_PATH.mkdir(parents=True, exist_ok=True)
        for file_path in VISUAL_V3_REPLAY_PERSISTENT_PATH.glob("*.pt"):
            file_path.unlink()

        persistent_index = []
        save_idx = 0
        for old_idx in selected_indices:
            old_entry = buf.index[old_idx]

            state_src = Path(old_entry["state"])
            if not state_src.exists():
                continue

            state_dst = VISUAL_V3_REPLAY_PERSISTENT_PATH / f"state_{save_idx}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            if old_entry["next_state"]:
                next_src = Path(old_entry["next_state"])
                if next_src.exists():
                    next_state_dst = VISUAL_V3_REPLAY_PERSISTENT_PATH / f"next_state_{save_idx}.pt"
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
                "priority": float(old_entry.get("priority", VISUAL_PRIORITY_MIN)),
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
            VISUAL_V3_MODEL_PATH / "training_state.pth",
        )
        print(f"[V3] Persistent save: {len(persistent_index)} entries")

    def try_load_model(self) -> None:
        bb_path = VISUAL_V3_MODEL_PATH / "backbone.pth"
        if bb_path.exists():
            try:
                self.backbone.load_state_dict(torch.load(bb_path, map_location=device))
                print("[V3] Loaded backbone from previous run")
            except Exception as exc:
                print(f"[V3] Failed to load backbone: {exc}")

        q_path = VISUAL_V3_MODEL_PATH / "fqf_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device))
                print("[V3] Loaded FQF-Network from previous run")
            except Exception as exc:
                print(f"[V3] Failed to load FQF-Network: {exc}")

        qt_path = VISUAL_V3_MODEL_PATH / "fqf_target.pth"
        if qt_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(qt_path, map_location=device))
                print("[V3] Loaded FQF-Target from previous run")
            except Exception as exc:
                print(f"[V3] Failed to load FQF-Target: {exc}")
        elif q_path.exists():
            self.q_target.load_state_dict(self.q_network.state_dict())

        self._load_optimizer_state()
        self._load_persistent_training_state()

    def _load_persistent_buffer(self, persistent_index) -> None:
        VISUAL_V3_REPLAY_PATH.mkdir(parents=True, exist_ok=True)
        for file_path in VISUAL_V3_REPLAY_PATH.glob("*.pt"):
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
                if not torch.is_tensor(peek) or tuple(peek.shape) != (3, *IMAGE_SIZE):
                    continue
            except Exception:
                continue

            storage_id = loaded_count
            state_dst = VISUAL_V3_REPLAY_PATH / f"state_{storage_id}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            next_src_str = entry.get("next_state", entry.get("next_state_path"))
            if next_src_str:
                next_src = Path(next_src_str)
                if next_src.exists():
                    next_state_dst = VISUAL_V3_REPLAY_PATH / f"next_state_{storage_id}.pt"
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
        print(f"[V3] Loaded {loaded_count} replay buffer entries")

    # ──────────────────────────── lr warmup ────────────────────────────

    def _apply_lr_warmup(self) -> None:
        """Linear LR warmup: scale every param group's lr from
        (LR_WARMUP_START_FACTOR × base_lr) up to base_lr over LR_WARMUP_STEPS
        optimizer steps. After warmup, lr stays at base_lr."""
        if LR_WARMUP_STEPS <= 0:
            return
        progress = min(1.0, self.total_it / LR_WARMUP_STEPS)
        factor = LR_WARMUP_START_FACTOR + (1.0 - LR_WARMUP_START_FACTOR) * progress
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * factor

    # ──────────────────────────── diagnostics ──────────────────────────

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

    def _log_backbone_weight_norms(self, global_step: int) -> None:
        # Log per-layer transformer norms to pinpoint where instability starts.
        for layer_idx, layer in enumerate(self.backbone.encoder.layers):
            prefix = f"encoder/layer{layer_idx}"
            self_attn = layer.attn.self_attn
            self._log_param_weight_and_grad_norm(f"{prefix}/self_attn_in_proj", self_attn.in_proj_weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/self_attn_out_proj", self_attn.out_proj.weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/ffn_linear1", layer.attn.linear1.weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/ffn_linear2", layer.attn.linear2.weight, global_step)

            if isinstance(layer.proj, nn.Sequential) and len(layer.proj) > 1 and isinstance(layer.proj[1], nn.Linear):
                self._log_param_weight_and_grad_norm(f"{prefix}/proj", layer.proj[1].weight, global_step)

        for layer_idx, layer in enumerate(self.backbone.decoder.layers):
            prefix = f"decoder/layer{layer_idx}"
            self._log_param_weight_and_grad_norm(f"{prefix}/self_attn_in_proj", layer.self_attn.in_proj_weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/self_attn_out_proj", layer.self_attn.out_proj.weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/cross_attn_in_proj", layer.multihead_attn.in_proj_weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/cross_attn_out_proj", layer.multihead_attn.out_proj.weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/ffn_linear1", layer.linear1.weight, global_step)
            self._log_param_weight_and_grad_norm(f"{prefix}/ffn_linear2", layer.linear2.weight, global_step)

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
        if not self._log_actions_this_episode:
            return
        try:
            from PIL import ImageDraw, ImageFont

            VISUAL_V3_ACTION_LOG_PATH.mkdir(parents=True, exist_ok=True)
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
            img.save(VISUAL_V3_ACTION_LOG_PATH / f"{ts}_step_{step_count:04d}.png")
        except Exception as exc:
            print(f"[V3] log_action_image failed: {exc}")


# ──────────────────────────── factory ────────────────────────────────

VisualAgentV3.store_transition = VisualAgentCommonMixin.store_transition
VisualAgentV3._commit_n_step_transition = VisualAgentCommonMixin._commit_n_step_transition
VisualAgentV3._flush_n_step_buffer = VisualAgentCommonMixin._flush_n_step_buffer
VisualAgentV3.save_persistent = VisualAgentCommonMixin.save_persistent
VisualAgentV3._load_persistent_buffer = VisualAgentCommonMixin._load_persistent_buffer
VisualAgentV3._module_grad_norm = VisualAgentCommonMixin._module_grad_norm
VisualAgentV3._capture_trainable_weight_snapshot = VisualAgentCommonMixin._capture_trainable_weight_snapshot
VisualAgentV3._snapshot_distance = VisualAgentCommonMixin._snapshot_distance
VisualAgentV3._tensor_norm = VisualAgentCommonMixin._tensor_norm
VisualAgentV3._log_param_weight_and_grad_norm = VisualAgentCommonMixin._log_param_weight_and_grad_norm
VisualAgentV3.log_action_image = VisualAgentCommonMixin.log_action_image
VisualAgentV3._save_optimizer_state = VisualAgentCommonMixin._save_optimizer_state
VisualAgentV3._load_optimizer_state = VisualAgentCommonMixin._load_optimizer_state
VisualAgentV3._load_persistent_training_state = VisualAgentCommonMixin._load_persistent_training_state

_agent: VisualAgentV3 | None = None


def get_agent(screen_region=None) -> VisualAgentV3:
    global _agent
    if _agent is None:
        _agent = VisualAgentV3(screen_region=screen_region)
    return _agent
