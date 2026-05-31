import atexit
import datetime
import logging as _logging
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model_structure.transformer_shared import EncoderDecoderTransformer, FQFQNetwork, FixedSinusoidalPositionEmbedding
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from model_structure.optimizer_factory import build_fqf_optimizer
from model_structure.adaptive_epsilon import AdaptiveEpsilonController
from model_structure.history import TrainingHistory
from model_structure.hyperparameter_dump import dump_hyperparameters
from model_structure.rng_utils import seed_everything
from model_structure.archive_manager import (
    SessionArchiveManager,
    RolloverTextLog,
    swap_logger_file_handler,
)
from model_structure.training_logger import TrainingLogger
import model_structure.CategorizedReplayBuffer as _crb_module


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
# Create a 32-bit seed at import time and apply it to random / numpy / torch / cuda.
# The actual SEED is written to hyperparameters.txt by hyperparameter_dump because
# it is a module-level ALL_CAPS int. agent.__init__ also prints it for comparison.
# To reproduce a run, set `SEED: int = seed_everything(<number>)` and train from
# zero. _save_model stores RNG state in optimizer_state.pth, so resume restores
# RNG trajectory; SEED only controls first startup.
SEED: int = seed_everything()

BATCH_SIZE = 128
GRID_STATE_CHANNELS = 12
PER_CAPACITY = 10000
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
# Phase 1 stratified-balanced ratio inside each batch. 1.0 = all per-class
# stratified, with PER only filling shortages. 0.5 = original 50/50. 0.0 = pure global PER.
PER_BALANCED_RATIO = 1.0
SAVE_CAPACITY = 512
SAVE_EVERY_N_EPISODES = 500
TARGET_UPDATE_FREQ = 50
N_STEP = 1
NUM_FQF_FRACTIONS = 8
FQF_ENTROPY_COEF = 1e-3
FQF_HUBER_KAPPA = 1.0
MINIMUM_DATA_SIZE = min(PER_CAPACITY, SAVE_CAPACITY*4)-1  # below this amount, won't start training
GRAD_CLIP_NORM = 8.0
# Heavy diagnostics — log less frequently to avoid TensorBoard bloat / overhead.
HISTOGRAM_EVERY = 200          # per-layer weight/grad norms
WEIGHT_DISTANCE_LOG_EVERY = 100  # full-model weight snapshot distance
DIAGNOSTIC_LOG_EVERY = 10      # io_log / TB scalars / no_grad diagnostic block

# learning rate warmup
# Linear LR warmup over the first N optimizer steps for early transformer stability.
# Increase from base_lr * LR_WARMUP_START_FACTOR to base_lr.
LR_WARMUP_STEPS         = 2000   # Initial from-scratch warmup length.
LR_WARMUP_START_FACTOR  = 0.0
LR_RESUME_WARMUP_STEPS  = 2000   # Extra warmup on every restart, including first run.

TRANSFORMER_MODEL_PATH = Path("./models/stage1_transformer")
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_FF_DIM = 256
TRANSFORMER_DROPOUT = 0.1

# CUDA SDP backend
# mem-efficient SDP can raise "illegal instruction" on some CUDA / GPU combos.
# Leave flash / math at PyTorch defaults.
if device.type == "cuda":
    try:
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass

# debug logger: write to file, not CMD. Only active when DEBUG_CUDA_SAMPLE=True.
# True inserts cuda.synchronize and logs, slowing training. Use only for CUDA errors.
DEBUG_CUDA_SAMPLE = False

_DBG_LOG_PATH = TRANSFORMER_MODEL_PATH / "cuda_debug.log"
_DBG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_dbg_logger = _logging.getLogger("cuda_dbg.stage1")
_dbg_logger.setLevel(_logging.DEBUG)
_dbg_logger.propagate = False  # Do not propagate to root logger; avoid CMD output.
# Module-level fallback before agent instantiation writes to top-level.
# delay=True opens the file only on emit, avoiding empty logs when disabled.
# Agent __init__ swaps this handler to the archive dir.
if not _dbg_logger.handlers:
    _fh = _logging.FileHandler(_DBG_LOG_PATH, mode="a", encoding="utf-8", delay=True)
    _fh.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
    _dbg_logger.addHandler(_fh)

# Route ReplayBuffer [DBG sample] logs to the same file, replacing its old print.
_crb_module.DEBUG_CUDA_SAMPLE_LOG_PATH = _DBG_LOG_PATH


def _dbg(msg: str) -> None:
    """Log+flush first, then sync; last disk line is the op about to sync."""
    if not DEBUG_CUDA_SAMPLE:
        return
    _dbg_logger.debug(msg)
    for _h in _dbg_logger.handlers:
        try:
            _h.flush()
        except Exception:
            pass
    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _dbg_mem(tag: str) -> None:
    """Log GPU memory usage."""
    if not DEBUG_CUDA_SAMPLE or device.type != "cuda":
        return
    try:
        alloc = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        _dbg_logger.debug(f"[MEM {tag}] alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB")
    except Exception as e:
        _dbg_logger.debug(f"[MEM {tag}] failed: {e}")


def _dbg_tensor(name: str, t, *, expect_max=None, expect_min=None, check_finite: bool = True) -> None:
    """Check tensor NaN/Inf and bounds; log shape/dtype/range.

    expect_max / expect_min: hard bounds for integer tensors; violations are OOB,
    likely a bad index.
    """
    if not DEBUG_CUDA_SAMPLE:
        return
    _dbg_logger.debug(f"[TENSOR {name}] entering")
    for _h in _dbg_logger.handlers:
        try:
            _h.flush()
        except Exception:
            pass
    try:
        if t is None:
            _dbg_logger.debug(f"[TENSOR {name}] is None")
            return
        if not torch.is_tensor(t):
            _dbg_logger.debug(f"[TENSOR {name}] type={type(t).__name__}")
            return
        if t.is_cuda:
            torch.cuda.synchronize()
        info = f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"
        if t.numel() == 0:
            _dbg_logger.debug(f"[TENSOR {name}] {info} EMPTY")
            return
        if t.dtype.is_floating_point:
            tmin = t.min().item()
            tmax = t.max().item()
            tnan = bool(torch.isnan(t).any().item()) if check_finite else False
            tinf = bool(torch.isinf(t).any().item()) if check_finite else False
            tag = ""
            if tnan:
                tag += " !!NAN!!"
            if tinf:
                tag += " !!INF!!"
            _dbg_logger.debug(f"[TENSOR {name}] {info} min={tmin:.4g} max={tmax:.4g}{tag}")
        else:
            tmin = t.min().item()
            tmax = t.max().item()
            tag = ""
            if expect_max is not None and tmax >= expect_max:
                tag += f" !!OOB max>={expect_max}!!"
            if expect_min is not None and tmin < expect_min:
                tag += f" !!OOB min<{expect_min}!!"
            _dbg_logger.debug(f"[TENSOR {name}] {info} min={tmin} max={tmax}{tag}")
        for _h in _dbg_logger.handlers:
            try:
                _h.flush()
            except Exception:
                pass
    except Exception as e:
        _dbg_logger.debug(f"[TENSOR {name}] CHECK FAILED: {e!r}")
        for _h in _dbg_logger.handlers:
            try:
                _h.flush()
            except Exception:
                pass


def log_unhandled_exception(context: str = "") -> None:
    """Write traceback to cuda_debug.log from outermost exception handlers.

    Ignores DEBUG_CUDA_SAMPLE; exceptions should always be persisted.
    """
    try:
        _dbg_logger.error(f"[UNHANDLED]{(' ' + context) if context else ''}", exc_info=True)
        for _h in _dbg_logger.handlers:
            try:
                _h.flush()
            except Exception:
                pass
    except Exception:
        pass


class TransformerActorNetwork(nn.Module):
    """Grid state -> encoder-decoder transformer -> per-cell logits."""

    def __init__(
        self,
        grid_channels=GRID_STATE_CHANNELS,
        grid_h=10,
        grid_w=10,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_FF_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_tokens = grid_h * grid_w

        self.token_embed = nn.Linear(grid_channels, d_model)
        self.position = FixedSinusoidalPositionEmbedding(d_model=d_model)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        self.core = EncoderDecoderTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_head = nn.Linear(d_model, 1)

    def _embed(self, state):
        tokens = state.permute(0, 2, 3, 1).reshape(state.size(0), self.num_tokens, -1)
        x = self.token_embed(tokens)
        return x + self.position(self.grid_h, self.grid_w).unsqueeze(0)

    def _build_queries(self, batch_size):
        return self.query_tokens.expand(batch_size, -1, -1)

    def get_memory(self, state):
        return self.core.encode(self._embed(state))

    def get_features(self, state):
        memory = self.get_memory(state)
        return self.core.decode(self._build_queries(state.size(0)), memory)

    def forward(self, state):
        x = self.get_features(state)
        logits = self.output_head(x).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def load_backbone_state(self, state_dict, strict=False):
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("core.") or key.startswith("token_embed.") or key.startswith("output_head.") or key == "query_tokens":
                normalized[key] = value
            elif key.startswith("transformer."):
                normalized[f"core.transformer.{key[len('transformer.') :]}"] = value
            elif key.startswith("decoder."):
                normalized[f"core.decoder.{key[len('decoder.') :]}"] = value

        # Backward-compat: pre-HierarchicalEncoder Stage 1 checkpoints saved encoder
        # weights as `core.transformer.layers.{i}.X` (raw nn.TransformerEncoderLayer).
        # The new structure wraps each layer in HierarchicalEncoderLayer.attn, so the
        # expected key is `core.transformer.layers.{i}.attn.X`. Migrate transparently
        # so existing on-disk checkpoints keep loading.
        migrated = {}
        for key, value in normalized.items():
            if key.startswith("core.transformer.layers."):
                tail = key[len("core.transformer.layers."):]
                idx_str, _, rest = tail.partition(".")
                if rest and not rest.startswith(("attn.", "proj.", "proj")):
                    # Old style — insert .attn between layer index and inner name.
                    migrated[f"core.transformer.layers.{idx_str}.attn.{rest}"] = value
                    continue
            migrated[key] = value

        return self.load_state_dict(migrated, strict=strict)


def _quantile_huber_loss(current_quantiles, target_quantiles, tau_hats, return_stats=False):
    td = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
    abs_td = td.abs()
    huber = torch.where(
        abs_td <= FQF_HUBER_KAPPA,
        0.5 * abs_td.pow(2),
        FQF_HUBER_KAPPA * abs_td - 0.5 * FQF_HUBER_KAPPA ** 2,
    )
    tau = tau_hats.unsqueeze(2)
    quantile_weight = (tau - (td.detach() < 0).float()).abs()
    loss = (quantile_weight * huber).sum(dim=2).mean(dim=1, keepdim=True)
    if return_stats:
        frac_clipped = (abs_td > FQF_HUBER_KAPPA).float().mean().item()
        return loss, frac_clipped
    return loss


_DEFAULT_CSV_FIELDS = (
    "episode", "reward", "steps", "is_win", "invalid_rate",
    "Q_loss", "q_mean", "epsilon",
    "eval_avg_reward", "eval_win_rate", "eval_avg_steps", "eval_avg_invalid_rate",
    "timestamp",
)


class TransformerDiscreteAgent:
    """FQF agent with grid encoder-decoder transformer backbone."""

    def __init__(self, grid_h=10, grid_w=10, *, csv_fields: list[str] | None = None):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w
        # Default csv_fields match train_stage1_simple.py main-loop columns.
        # Training scripts can pass custom fields. Agent only builds TrainingLogger
        # and does not decide schema details.
        self._csv_fields = list(csv_fields) if csv_fields else list(_DEFAULT_CSV_FIELDS)

        self.backbone = TransformerActorNetwork(grid_h=grid_h, grid_w=grid_w).to(device)
        self.q_network = FQFQNetwork(
            d_model=TRANSFORMER_D_MODEL,
            grid_h=grid_h,
            grid_w=grid_w,
            num_fractions=NUM_FQF_FRACTIONS,
        ).to(device)
        self.q_target = FQFQNetwork(
            d_model=TRANSFORMER_D_MODEL,
            grid_h=grid_h,
            grid_w=grid_w,
            num_fractions=NUM_FQF_FRACTIONS,
        ).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()

        from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer

        self.optimizer = build_fqf_optimizer(self.backbone, self.q_network)
        # Record each param group's base LR; warmup scales by total_it / steps_since_resume.
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        self.replay_buffer = CategorizedReplayBuffer(
            max_size=PER_CAPACITY,
            storage_mode="ram",
            win_threshold=MINESWEEPER_REWARD_CONFIG.replay_win_threshold,
            lose_threshold=MINESWEEPER_REWARD_CONFIG.replay_lose_threshold,
            invalid_threshold=MINESWEEPER_REWARD_CONFIG.replay_invalid_threshold,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            balanced_ratio=PER_BALANCED_RATIO,
            quota_check_class="win",  # Minesweeper: win is the rare-event bottleneck class
            # Spread-decay calibrated from observed inference quantile spreads (median ~0.115,
            # p90 ~0.378). Starts disabled — latched ON by train_step once win_rate(100) > 0.4.
            spread_decay=2.0,
        )
        self.total_it = 0
        self.steps_since_resume = 0  # reset each startup; used for resume LR warmup, not saved
        # episode_count delegates to training_history.total_episodes as single source of truth.
        self.n_step = N_STEP
        self.n_step_gamma = MINESWEEPER_REWARD_CONFIG.gamma
        self.n_step_buffer = deque()
        # raw reward rolling mean moved to TrainingHistory._step_rewards.
        # store_transition records reward; train_step reads avg_step_reward().
        # Shared with v2 / v3 to avoid duplicate sum()/len() formulas.

        # adaptive epsilon
        # Shared controller class and wr/eps ranges with V3. Stage1 and Stage2 use
        # the same 6x6 Minesweeper reward signal, so they use consistent settings.
        # Episode results live in TrainingHistory; controller only consumes win_rate.
        self.epsilon_controller = AdaptiveEpsilonController()
        self.training_history = TrainingHistory()
        self.deque_cls = deque  # used by training_history.load_state_dict()

        # Lazy-captured at first train_step: if the loaded TrainingHistory shows last
        # win_rate(window=100) > 0.5, we additionally gate training on the replay buffer
        # having all 4 classes filled to their 12.5% soft-floor quota. Fresh runs
        # (history empty → win_rate=0) and weak resumes skip this gate.
        # Captured once and frozen for the whole session — won't flip when live win_rate
        # crosses 50% mid-training.
        self._class_quota_gate_enabled: bool | None = None

        # Episode-scoped blocked actions: clicked cells are masked within this episode.
        self.blocked_actions: set[int] = set()

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        # Session / hour archive directories. Each startup creates training_<ts>;
        # each hour rolls to hour_NN_<ts>. Canonical *.pth files still live at
        # TRANSFORMER_MODEL_PATH root for try_load_model, while snapshots and logs
        # also go to current_archive_dir. SessionArchiveManager handles the details.
        self.archive = SessionArchiveManager(
            model_path=TRANSFORMER_MODEL_PATH,
            log_prefix="[FQF]",
        )
        # SEED is a module-level constant from seed_everything; print it for later
        # comparison with hyperparameters.txt and console logs.
        print(f"[FQF] SEED = {SEED}")

        try:
            dump_hyperparameters(
                out_path=self.archive.session_dir / "hyperparameters.txt",
                modules=[sys.modules[__name__]],
                dataclass_instances={"reward_config": MINESWEEPER_REWARD_CONFIG},
                instance_attrs={
                    "epsilon_controller": (
                        self.epsilon_controller,
                        ["wr_min", "wr_max", "eps_min", "eps_max"],
                    ),
                    "optimizer (AdamW)": self.optimizer,
                },
                models={
                    # backbone: freeze status + torchinfo layer summary.
                    # input_size uses the real grid state tensor shape.
                    "model.backbone": (
                        self.backbone,
                        (1, GRID_STATE_CHANNELS, self.grid_h, self.grid_w),
                    ),
                    # Do not pass input_size for q_network. It consumes backbone
                    # cell features and returns dicts, so torchinfo adds little.
                    "model.q_network": (self.q_network, None),
                },
            )
        except Exception as exc:
            print(f"[FQF] hyperparameters dump failed: {exc}")

        # io_log: plain-text append file swapped to each new hour directory.
        # Banner comes from _build_io_log_banner.
        self._io_log = RolloverTextLog(banner_factory=self._build_io_log_banner)
        self._io_log.swap_to(self.archive.current_archive_dir / "train_io_log.txt")

        # cuda_debug.log follows archive dir too. The module-level handler defaults
        # to top-level and is swapped here to current_archive_dir. CategorizedReplayBuffer
        # shares the same path and is updated here.
        dbg_path = self.archive.current_archive_dir / "cuda_debug.log"
        swap_logger_file_handler(_dbg_logger, dbg_path)
        _crb_module.DEBUG_CUDA_SAMPLE_LOG_PATH = dbg_path

        # Register hour rollover callback: swap io_log, debug logger, and replay
        # buffer CUDA log. SessionArchiveManager prints rollover console messages.
        self.archive.register_on_rollover(self._on_archive_rollover)

        # TensorBoard log_dir is keyed by session timestamp. SummaryWriter is now
        # internal to TrainingLogger; all TB writes go through training_logger.
        tb_root = TRANSFORMER_MODEL_PATH / "tensorboard"
        tb_root.mkdir(parents=True, exist_ok=True)
        tb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log_dir = tb_root / tb_timestamp
        print(f"[FQF] TensorBoard: tensorboard --logdir {tb_root}")
        print(f"[FQF] Current run: {self.tensorboard_log_dir}")

        # TrainingLogger writes episode summaries to CSV + TB and owns train_step
        # per-step TB scalars. CSV writes to current hour dir and swaps on rollover.
        csv_path = self.archive.current_archive_dir / "training_log.csv"
        self.training_logger = TrainingLogger(
            csv_path=csv_path,
            csv_fields=self._csv_fields,
            tb_writer=SummaryWriter(log_dir=str(self.tensorboard_log_dir)),
        )
        self.archive.register_on_rollover(
            lambda new_dir: self.training_logger.swap_csv_to(new_dir / "training_log.csv")
        )
        print(f"[FQF] CSV log: {csv_path}")
        self._write_metric_docs()

        self.try_load_model()

        # Weight snapshots for diagnosing drift. Must be captured AFTER
        # try_load_model so that "init" reflects the actual starting point
        # of this session (including any loaded checkpoint).
        self._init_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference_step = self.total_it

        # atexit order is LIFO. Desired run order: _save_model (which flushes
        # tb via training_logger.flush()) → save_persistent → close io_log →
        # close training_logger (closes CSV + SummaryWriter together). So
        # register closes first (run last) and saves last (run first).
        atexit.register(self._close_training_logger)
        atexit.register(self._close_io_log)
        atexit.register(self.save_persistent)
        atexit.register(self._save_model)

    # Epsilon is owned by the controller; keep self.epsilon for legacy callers
    # such as select_action, TB logging, and save/load paths.
    @property
    def epsilon(self) -> float:
        return self.epsilon_controller.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.epsilon_controller.epsilon = float(value)

    # episode_count delegates to training_history.total_episodes. history.record()
    # increments it inside log_episode_metrics(), so on_episode_end no longer
    # increments it separately. Read-only; history is the source of truth.
    @property
    def episode_count(self) -> int:
        return self.training_history.total_episodes

    def clear_blocked_actions(self):
        self.blocked_actions.clear()

    def block_action_for_state(self, state, action_id: int):
        self.blocked_actions.add(int(action_id))

    def select_action(self, state, add_noise=True):
        blocked = set(self.blocked_actions)
        available = [i for i in range(self.num_actions) if i not in blocked]
        if not available:
            self.clear_blocked_actions()
            blocked = set()
            available = list(range(self.num_actions))

        if add_noise and random.random() < self.epsilon:
            action_id = random.choice(available)
            return (action_id // self.grid_w, action_id % self.grid_w)

        state_batch = state.unsqueeze(0).to(device)

        self.backbone.eval()
        self.q_network.eval()
        with torch.no_grad():
            features = self.backbone.get_features(state_batch)
            q_2d = self.q_network(features)["q_values"].squeeze(0)
            q_flat = q_2d.view(-1).clone()
            if blocked:
                blocked_idx = torch.tensor(sorted(blocked), dtype=torch.long, device=q_flat.device)
                q_flat[blocked_idx] = float("-inf")
            action_id = int(q_flat.argmax().item())
        self.backbone.train()
        self.q_network.train()

        return (action_id // self.grid_w, action_id % self.grid_w)

    def store_transition(self, state, action, next_state, reward, done):
        # Feed train/real_reward_mean with the raw reward rolling mean before
        # reward_squash. Uses the same record_step_reward path as v2 / v3.
        self.training_history.record_step_reward(float(reward))
        transition = {
            "state": state.detach().cpu(),
            "action": np.array(action, dtype=np.int64),
            "next_state": next_state.detach().cpu() if next_state is not None else None,
            "reward": float(reward),
            "done": bool(done),
        }
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) >= self.n_step:
            self._commit_n_step_transition(self.n_step)

        if done:
            self._flush_n_step_buffer()

    def _commit_n_step_transition(self, horizon):
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

    def _flush_n_step_buffer(self):
        while self.n_step_buffer:
            self._commit_n_step_transition(len(self.n_step_buffer))

    def train_step(self, preprocessor=None, extra_params_to_clip=None):
        """Run one gradient step.

        preprocessor : optional callable (batch_tensor → batch_tensor) applied to
                       raw stored states before the forward pass.  Used for end-to-end
                       training where screenshots are stored and YOLO is the preprocessor.
                       Applied WITH gradients on the current state, WITHOUT gradients on
                       next_state (target network branch stays frozen).
        extra_params_to_clip : optional iterable of extra parameters to include in the
                               gradient-norm clip (e.g. YOLO parameters).
        """
        buf_size = self.replay_buffer.size()
        if buf_size < MINIMUM_DATA_SIZE:
            return None

        # Capture-once class_quota gate decision based on loaded training_history.
        # Only enforce balanced-data warmup when resuming from a session that was already
        # performing well (win_rate > 50%); fresh / weak runs proceed without this gate.
        if self._class_quota_gate_enabled is None:
            self._class_quota_gate_enabled = self.training_history.win_rate(window=100) > 0.5
        if self._class_quota_gate_enabled and not self.replay_buffer.is_class_quota_filled():
            return None

        # Spread-decay latch — independent of class_quota gate. Live-checked every train_step
        # until it flips ON, then stays ON for the rest of the session (monotone). Activates
        # when win_rate crosses 40% on the assumption that by then the model has matured
        # enough that "wide quantile spread" mostly reflects environment stochasticity
        # rather than under-learning.
        if not self.replay_buffer.enable_spread_decay:
            if self.training_history.win_rate(window=100) > 0.4:
                self.replay_buffer.enable_spread_decay = True

        self.total_it += 1
        self.steps_since_resume += 1
        self._apply_lr_warmup()
        # Roll over each wall-clock hour; io_log / vclamp / save follow the new
        # archive dir through SessionArchiveManager on_rollover callbacks.
        self.archive.maybe_rollover()

        try:
            _dbg(f"[train_step] ENTER total_it={self.total_it} buf_size={buf_size}")
            _dbg_mem("train_step ENTER")
            _dbg("[train_step] before replay_buffer.sample")
            state, action, next_state, reward, done, per_indices, is_weights, discounts, n_steps = self.replay_buffer.sample(
                BATCH_SIZE,
                beta=PER_BETA_START + (PER_BETA_END - PER_BETA_START) * min(self.episode_count / 5000.0, 1.0),
                device=device,
                include_extra=True,
            )
            _dbg("[train_step] after replay_buffer.sample")
            _dbg_tensor("train_step.state",      state)
            _dbg_tensor("train_step.next_state", next_state)
            _dbg_tensor("train_step.action",     action,
                        expect_min=0, expect_max=max(self.grid_h, self.grid_w))
            _dbg_tensor("train_step.reward",     reward)
            _dbg_tensor("train_step.done",       done)
            _dbg_tensor("train_step.is_weights", is_weights)
            _dbg_tensor("train_step.discounts",  discounts)

            # Stage 0: buffer sample. A hit means replay data is corrupt
            # in the load or store path.
            self._assert_finite("stage0_sample", "state", state)
            self._assert_finite("stage0_sample", "next_state", next_state)
            self._assert_finite("stage0_sample", "reward", reward)
            self._assert_finite("stage0_sample", "is_weights", is_weights)
            self._assert_finite("stage0_sample", "discounts", discounts)

            action = action.long()
            row_idx = action[:, 0]
            col_idx = action[:, 1]
            action_flat = row_idx * self.grid_w + col_idx
            batch_size = state.size(0)
            _dbg_tensor("train_step.row_idx", row_idx, expect_min=0, expect_max=self.grid_h)
            _dbg_tensor("train_step.col_idx", col_idx, expect_min=0, expect_max=self.grid_w)
            _dbg_tensor("train_step.action_flat", action_flat,
                        expect_min=0, expect_max=self.num_actions)
            _dbg(f"[train_step] batch_size={batch_size} num_actions={self.num_actions} grid={self.grid_h}x{self.grid_w}")

            # Preprocess current state WITH gradients so end-to-end training flows back
            if preprocessor is not None:
                state = preprocessor(state)
                _dbg_tensor("train_step.state(after preprocessor)", state)

            with torch.no_grad():
                next_proc = preprocessor(next_state) if preprocessor is not None else next_state
                _dbg("[train_step] before next backbone")
                next_features = self.backbone.get_features(next_proc)
                _dbg_tensor("train_step.next_features", next_features)
                _dbg("[train_step] after next backbone")
                # Stage 1a: backbone forward on next_state. A hit means either
                # backbone weights or next_state are corrupt. This is the
                # earliest training-time check that can catch backbone damage.
                self._assert_finite("stage1a_target_backbone", "next_features", next_features)

                next_online = self.q_network(next_features)
                next_q_2d = next_online["q_values"]
                _dbg_tensor("train_step.next_online.q_values", next_q_2d)
                next_q_flat = next_q_2d.view(batch_size, -1)
                best_flat = next_q_flat.argmax(dim=1)
                _dbg_tensor("train_step.next_best_flat", best_flat,
                            expect_min=0, expect_max=self.num_actions)
                best_rows = best_flat // self.grid_w
                best_cols = best_flat % self.grid_w

                next_target = self.q_target(next_features)
                _dbg_tensor("train_step.next_target.quantiles", next_target["quantiles"])
                _dbg("[train_step] after q_target")
                # Stage 1b: q_target forward. A hit means q_target weights are corrupt.
                self._assert_finite("stage1b_q_target", "next_target.quantiles", next_target["quantiles"])

                next_target_quantiles = next_target["quantiles"][
                    torch.arange(batch_size, device=device), best_flat
                ]
                _dbg_tensor("train_step.next_target_quantiles", next_target_quantiles)
                target_quantiles = reward + (1 - done) * discounts * next_target_quantiles
                _dbg_tensor("train_step.target_quantiles", target_quantiles)
                # Stage 1c: target_quantiles finished. A hit means reward /
                # discounts / done are invalid if next_target_quantiles was finite.
                self._assert_finite("stage1c_target_combine", "target_quantiles", target_quantiles)
            _dbg("[train_step] target branch done")

            _dbg("[train_step] before current backbone")
            features = self.backbone.get_features(state)
            _dbg_tensor("train_step.features", features)
            _dbg("[train_step] after current backbone")
            # Stage 2: current backbone forward. Compare with stage1a to tell
            # whether the backbone or the state tensor is corrupt.
            self._assert_finite("stage2_current_backbone", "features", features)

            q_output = self.q_network(features)
            q_2d = q_output["q_values"]
            q_quantiles = q_output["quantiles"]
            tau_hats = q_output["tau_hats"]
            fraction_probs = q_output["fraction_probs"]
            _dbg_tensor("train_step.q_2d", q_2d)
            _dbg_tensor("train_step.q_quantiles", q_quantiles)
            _dbg_tensor("train_step.tau_hats", tau_hats)
            _dbg_tensor("train_step.fraction_probs", fraction_probs)
            # Stage 3: q_network forward, checking each FQF head output:
            #   - bad fraction_probs -> fraction_proposal / softmax input issue
            #   - bad tau_hats       -> cumsum / mean, usually follows fraction_probs
            #   - bad quantiles      -> cosine_embedding or value_head issue
            #   - bad q_values       -> any of the above
            self._assert_finite("stage3_q_network", "fraction_probs", fraction_probs)
            self._assert_finite("stage3_q_network", "tau_hats", tau_hats)
            self._assert_finite("stage3_q_network", "quantiles", q_quantiles)
            self._assert_finite("stage3_q_network", "q_values", q_2d)
            _dbg(f"[train_step] q_2d.shape={tuple(q_2d.shape)} q_quantiles.shape={tuple(q_quantiles.shape)}")
            q_taken = q_2d[
                torch.arange(batch_size, device=device), row_idx, col_idx
            ].unsqueeze(1)
            _dbg("[train_step] after q_taken gather")
            chosen_quantiles = q_quantiles[
                torch.arange(batch_size, device=device), action_flat
            ]
            _dbg("[train_step] after chosen_quantiles gather")

            with torch.no_grad():
                target_mean = target_quantiles.mean(dim=1, keepdim=True)
                td_error = (q_taken - target_mean).abs().detach()
                # Per-sample quantile spread for the chosen action — feeds the buffer's
                # spread_decay modifier in _effective_priority (only consulted when the
                # buffer-side latch enable_spread_decay is True; we always compute and
                # write the value so the latch flip doesn't have a cold-start period).
                chosen_q_spread = chosen_quantiles.std(dim=1).detach()

            per_sample_quantile_loss, frac_huber_clipped = _quantile_huber_loss(
                current_quantiles=chosen_quantiles,
                target_quantiles=target_quantiles.detach(),
                tau_hats=tau_hats.detach(),
                return_stats=True,
            )
            # Stage 4a: quantile huber loss. A hit usually means chosen_quantiles
            # or target_quantiles is extreme, such as td^2 overflow. torch.where
            # may choose a finite branch, but backward can still produce NaN grads
            # through 0 * inf, which stage5 catches.
            self._assert_finite("stage4a_quantile_loss", "per_sample_quantile_loss", per_sample_quantile_loss)

            entropy = -(fraction_probs * torch.log(fraction_probs + 1e-8)).sum(dim=1, keepdim=True)
            # Stage 4b: entropy. A hit means fraction_probs has NaN, which stage3
            # should normally catch first.
            self._assert_finite("stage4b_entropy", "entropy", entropy)

            per_sample_loss = per_sample_quantile_loss - FQF_ENTROPY_COEF * entropy
            loss = (is_weights * per_sample_loss).mean()
            _dbg_tensor("train_step.loss", loss)
            _dbg_tensor("train_step.td_error", td_error)

            # FQF distribution-health diagnostics (cheap, computed inside no_grad).
            with torch.no_grad():
                # Keep these as tensors; diagnostic batching calls .item() later.
                fpn_norm_entropy_t = entropy.mean() / math.log(NUM_FQF_FRACTIONS)
                fpn_tau_std_t = tau_hats.std(dim=1).mean()

            # Stage 4c: final loss. Keep the old NaN check with stage-tagged output.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[NaN-probe] non-finite at stage='stage4c_final_loss' tensor='loss' "
                    f"step={self.total_it} value={loss.item()} "
                    f"(quantile={per_sample_quantile_loss.mean().item()}, "
                    f"entropy={entropy.mean().item()})"
                )

            self.optimizer.zero_grad()
            _dbg("[train_step] before backward")
            try:
                loss.backward()
            except RuntimeError as exc:
                msg = str(exc)
                if any(tag in msg for tag in ("CUDA error", "illegal instruction", "device-side assert")):
                    _dbg_logger.warning(
                        f"[train_step] backward CUDA crash at total_it={self.total_it}; "
                        f"dropping batch and continuing. msg={msg!r}"
                    )
                    for _h in _dbg_logger.handlers:
                        try:
                            _h.flush()
                        except Exception:
                            pass
                    self.optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    return None
                raise  # Other RuntimeErrors are logged by the outer try/except.
            _dbg("[train_step] after backward")
            _dbg_mem("train_step after backward")
            all_params = list(self.backbone.parameters()) + list(self.q_network.parameters())
            if extra_params_to_clip is not None:
                all_params += list(extra_params_to_clip)
            _dbg("[train_step] before grad-finite check")
            # Stage 5: after backward. A hit means backward produced NaN/Inf grads.
            # Common cause: torch.where(td^2) with huge td backprops through 0 * inf.
            for name, param in list(self.backbone.named_parameters()) + list(self.q_network.named_parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    nan_n = int(torch.isnan(param.grad).sum().item())
                    inf_n = int(torch.isinf(param.grad).sum().item())
                    raise RuntimeError(
                        f"[NaN-probe] non-finite at stage='stage5_post_backward' tensor='grad.{name}' "
                        f"step={self.total_it} shape={tuple(param.grad.shape)} "
                        f"nan={nan_n} inf={inf_n}"
                    )
            _dbg("[train_step] after grad-finite check")

            # Pre-clip gradient norms — must be captured BEFORE clip_grad_norm_,
            # otherwise the per-param tensors are scaled in-place and we lose
            # the true magnitude that caused any explosion.
            backbone_pre = self._module_grad_norm(self.backbone)
            head_pre = self._module_grad_norm(self.q_network)
            # Extras (e.g. YOLO when Stage 2 plugs in) contribute to the clip
            # norm; track them separately so pre/post totals match the
            # parameter set actually passed to clip_grad_norm_.
            extras_pre_sq = 0.0
            if extra_params_to_clip is not None:
                for param in extra_params_to_clip:
                    if param.grad is None:
                        continue
                    extras_pre_sq += float(param.grad.detach().float().pow(2).sum().item())
            extras_pre = extras_pre_sq ** 0.5
            # Capture per-layer norm snapshots BEFORE clip — but DON'T write
            # to TB yet; writes belong in the post-try success path so a CUDA
            # crash on clip / step doesn't leave orphan per-layer rows in TB
            # without the matching global rows.
            backbone_norm_snapshots = None
            if self.total_it % HISTOGRAM_EVERY == 0:
                backbone_norm_snapshots = self._collect_backbone_weight_norm_snapshots()

            grad_norm_total = torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP_NORM)
            grad_norm_total_value = float(grad_norm_total)
            grad_clip_threshold = float(GRAD_CLIP_NORM)
            grad_clip_scale = min(1.0, grad_clip_threshold / (grad_norm_total_value + 1e-12))
            grad_clip_percent = 1.0 - grad_clip_scale
            grad_clip_excess_norm = max(0.0, grad_norm_total_value - grad_clip_threshold)
            grad_clip_excess_ratio = grad_clip_excess_norm / (grad_clip_threshold + 1e-12)

            # Stage 6: scan grads again after clipping to catch 0 * inf inside clip.
            # Stage5 should catch inf first, but clip also writes in-place.
            for name, param in list(self.backbone.named_parameters()) + list(self.q_network.named_parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    nan_n = int(torch.isnan(param.grad).sum().item())
                    inf_n = int(torch.isinf(param.grad).sum().item())
                    raise RuntimeError(
                        f"[NaN-probe] non-finite at stage='stage6_post_clip' tensor='grad.{name}' "
                        f"step={self.total_it} shape={tuple(param.grad.shape)} "
                        f"nan={nan_n} inf={inf_n}"
                    )

            backbone_post = self._module_grad_norm(self.backbone)
            head_post = self._module_grad_norm(self.q_network)
            extras_post_sq = 0.0
            if extra_params_to_clip is not None:
                for param in extra_params_to_clip:
                    if param.grad is None:
                        continue
                    extras_post_sq += float(param.grad.detach().float().pow(2).sum().item())
            extras_post = extras_post_sq ** 0.5
            grad_post_total = (backbone_post ** 2 + head_post ** 2 + extras_post_sq) ** 0.5

            _dbg("[train_step] after clip_grad_norm_")
            # Pre-step v-clamp: a hardware bit flip can make v negative, then
            # sqrt(neg)=NaN and the next weight update becomes NaN. Clamp before
            # optimizer.step() so Adam always sees v >= 0. Hits are logged to
            # stdout / io_log / vclamp_events.log / TB for later frequency checks.
            self._clamp_optimizer_v_and_log()
            _dbg("[train_step] before optimizer.step")
            self.optimizer.step()
            _dbg("[train_step] after optimizer.step")
            _dbg_mem("train_step after optimizer.step")

            # Stage 7: scan weights after optimizer.step. This is the most likely
            # source for the observed failure: finite grads enter AdamW but produce
            # NaN weights. Common causes:
            #   (a) v near denormal underflow -> sqrt(v)+eps is tiny -> huge update
            #   (b) rare fused/non-fused kernel numerical edge case
            #   (c) transient hardware bit flip
            # On hit, the weight is already corrupt, but grad / m / v for this step
            # remain in optimizer state for offline analysis. atexit writes the
            # optimizer state to a .crash file after the crash.
            for name, param in list(self.backbone.named_parameters()) + list(self.q_network.named_parameters()):
                if not torch.isfinite(param.data).all():
                    nan_n = int(torch.isnan(param.data).sum().item())
                    inf_n = int(torch.isinf(param.data).sum().item())
                    finite_mask = torch.isfinite(param.data)
                    absmax = (
                        float(param.data[finite_mask].abs().max().item())
                        if finite_mask.any() else float("nan")
                    )
                    raise RuntimeError(
                        f"[NaN-probe] non-finite at stage='stage7_post_step' tensor='weight.{name}' "
                        f"step={self.total_it} shape={tuple(param.data.shape)} "
                        f"nan={nan_n} inf={inf_n} finite_absmax={absmax:.4g}"
                    )
        except Exception as exc:
            # Non-backward CUDA crash: forward / loss / grad-check / clip /
            # optimizer.step / final sync. Tag it separately from backward crashes
            # for grep-based location counts. Non-CUDA exceptions are real errors
            # and get logged before being re-raised.
            msg = str(exc)
            if isinstance(exc, RuntimeError) and any(
                tag in msg for tag in ("CUDA error", "illegal instruction", "device-side assert")
            ):
                _dbg_logger.warning(
                    f"[train_step] non-backward CUDA crash at total_it={self.total_it}; "
                    f"dropping batch and continuing. msg={msg!r}"
                )
                for _h in _dbg_logger.handlers:
                    try:
                        _h.flush()
                    except Exception:
                        pass
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                return None
            log_unhandled_exception(f"train_step total_it={self.total_it}")
            raise

        self.replay_buffer.update_priorities(
            per_indices,
            td_error.squeeze(-1).cpu().numpy(),
            quantile_spreads=chosen_q_spread.cpu().numpy(),
        )

        if self.total_it % TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

        # Weight-drift snapshot: independent of DIAGNOSTIC_LOG_EVERY. This can
        # be any value and does not need to be a multiple of 10. init_distance /
        # rolling_distance are Python floats, so add_scalar needs no extra sync.
        if self.total_it % WEIGHT_DISTANCE_LOG_EVERY == 0:
            current_snapshot = self._capture_trainable_weight_snapshot()
            init_distance = self._snapshot_distance(current_snapshot, self._init_weight_reference)
            rolling_distance = self._snapshot_distance(current_snapshot, self._rolling_weight_reference)
            self._rolling_weight_reference = current_snapshot
            self._rolling_weight_reference_step = self.total_it
            self.training_logger.log("weights/delta_from_init", init_distance, step=self.total_it, csv=False)
            self.training_logger.log("weights/delta_from_prev_window", rolling_distance, step=self.total_it, csv=False)

        # Diagnostic / logging: only every DIAGNOSTIC_LOG_EVERY steps. Lower the
        # frequency for io_log / TB / no_grad stats; other steps return None and
        # avoid about 40 implicit GPU-to-CPU syncs.
        if self.total_it % DIAGNOSTIC_LOG_EVERY != 0:
            return None

        with torch.no_grad():
            # Pack all scalar reads into one tensor and move it to CPU once:
            # 16 .item() calls become one sync.
            q0_flat = q_2d[0].view(-1)
            _zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
            _stats = torch.stack([
                loss,
                q_taken.mean(),
                q_taken.std() if q_taken.numel() > 1 else _zero,
                q_2d.max(),
                q_2d.min(),
                q0_flat.min(),
                q0_flat.max(),
                target_quantiles.float().mean(),
                target_quantiles.float().std() if target_quantiles.numel() > 1 else _zero,
                td_error.mean(),
                td_error.max(),
                entropy.mean(),
                is_weights.float().mean(),
                is_weights.float().min(),
                fpn_norm_entropy_t,
                fpn_tau_std_t,
            ]).cpu().tolist()
            (
                loss_value, q_mean, q_taken_std, q_all_max, q_all_min,
                q0_min, q0_max,
                target_q_mean, target_q_std,
                td_error_mean, td_error_max,
                entropy_value, is_weight_mean, is_weight_min,
                fpn_norm_entropy, fpn_tau_std,
            ) = _stats

            # Keep these .item() / .tolist() calls; they only run on diagnostic steps.
            reward_counts = defaultdict(int)
            for value in reward.squeeze(-1).tolist():
                reward_counts[value] += 1

            s0 = state[0]
            unrevealed = s0[0].sum().int().item()
            flagged = s0[1].sum().int().item()
            total_cells = self.grid_h * self.grid_w
            revealed = total_cells - unrevealed - flagged
            num_counts = {}
            for ch in range(2, 11):
                count = s0[ch].sum().int().item()
                if count > 0:
                    num_counts[ch - 2] = count

            top5_vals, top5_flat = q0_flat.topk(5)
            top5_info = [
                (idx.item() // self.grid_w, idx.item() % self.grid_w, f"{val.item():.4f}")
                for val, idx in zip(top5_vals, top5_flat)
            ]

            action_list = list(zip(row_idx.tolist(), col_idx.tolist()))
            action_freq = defaultdict(int)
            for item in action_list:
                action_freq[item] += 1
            top3_actions = sorted(action_freq.items(), key=lambda item: -item[1])[:3]
            top3_str = ", ".join(f"({row},{col})x{count}" for (row, col), count in top3_actions)

        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  State:  unrevealed={unrevealed} | revealed={revealed} | flagged={flagged}"
            f" | numbers={num_counts}\n"
            f"  Batch:  rewards={dict(reward_counts)} | top_actions=[{top3_str}]\n"
            f"  Q-top5: {top5_info}\n"
            f"  Q-val:  taken_mean={q_mean:.4f}"
            f" | all: min={q0_min:.4f} max={q0_max:.4f}\n"
            f"  FQF:    loss={loss_value:.4f} | tau_entropy={entropy_value:.4f}"
            f" | epsilon={self.epsilon:.4f}\n"
            f"  TD:     mean={td_error_mean:.4f} max={td_error_max:.4f} "
            f"frac_huber_clipped={frac_huber_clipped:.3f}\n"
            f"  Grad:   pre_total={grad_norm_total_value:.4f} post_total={grad_post_total:.4f} "
            f"clip_percent={grad_clip_percent:.2%}\n"
            f"---\n"
        )
        self._io_log.flush()

        # Raw reward rolling mean from training_history._step_rewards. It is
        # maintained by store_transition and shared with v2 / v3.
        real_reward_mean = self.training_history.avg_step_reward()

        # ── TensorBoard scalars ──
        step = self.total_it
        self.training_logger.log("train/Q_loss", loss_value, step=step, csv=False)
        self.training_logger.log("train/q_mean", q_mean, step=step, csv=False)
        self.training_logger.log("train/real_reward_mean", real_reward_mean, step=step, csv=False)
        self.training_logger.log("train/q_taken_std", q_taken_std, step=step, csv=False)
        self.training_logger.log("train/q_max", q_all_max, step=step, csv=False)
        self.training_logger.log("train/q_min", q_all_min, step=step, csv=False)
        self.training_logger.log("train/target_q_mean", target_q_mean, step=step, csv=False)
        self.training_logger.log("train/target_q_std", target_q_std, step=step, csv=False)
        self.training_logger.log("train/td_error_mean", td_error_mean, step=step, csv=False)
        self.training_logger.log("train/td_error_max", td_error_max, step=step, csv=False)
        self.training_logger.log("train/frac_huber_clipped", frac_huber_clipped, step=step, csv=False)
        self.training_logger.log("train/epsilon", self.epsilon, step=step, csv=False)
        self.training_logger.log("fpn/norm_entropy", fpn_norm_entropy, step=step, csv=False)
        self.training_logger.log("fpn/tau_std", fpn_tau_std, step=step, csv=False)
        self.training_logger.log("grad/total_pre_clip", grad_norm_total_value, step=step, csv=False)
        self.training_logger.log("grad/total_post_clip", grad_post_total, step=step, csv=False)
        self.training_logger.log("grad/clip_percent", grad_clip_percent, step=step, csv=False)
        self.training_logger.log("grad/clip_excess_norm", grad_clip_excess_norm, step=step, csv=False)
        self.training_logger.log("grad/clip_excess_ratio", grad_clip_excess_ratio, step=step, csv=False)
        self.training_logger.log("grad_pre/backbone", backbone_pre, step=step, csv=False)
        self.training_logger.log("grad_pre/head", head_pre, step=step, csv=False)
        self.training_logger.log("grad_post/backbone", backbone_post, step=step, csv=False)
        self.training_logger.log("grad_post/head", head_post, step=step, csv=False)
        self.training_logger.log("grad_pre/extras", extras_pre, step=step, csv=False)
        self.training_logger.log("grad_post/extras", extras_post, step=step, csv=False)
        # check/ namespace: validation-only, not core training metrics.
        self.training_logger.log("check/is_weight_mean", is_weight_mean, step=step, csv=False)
        self.training_logger.log("check/is_weight_min", is_weight_min, step=step, csv=False)
        # IS weights are max-normalized, so max is always 1.0 and ratio = 1/min.
        # Healthy is < 10; > 100 suggests the IS formula may be broken again.
        self.training_logger.log("check/is_weight_ratio", 1.0 / max(is_weight_min, step=1e-12, csv=False), step)
        # Average sample count per entry. This grows monotonically with training
        # steps; high mean implies concentrated PER and low batch diversity.
        self.training_logger.log("check/mean_sample_count", self.replay_buffer.mean_sample_count(), step=step, csv=False)

        # Per-layer weight/grad norms — collected pre-clip inside the try
        # block above; written here so we never leave orphan rows on a
        # CUDA-recovery early return.
        if backbone_norm_snapshots is not None:
            self._write_backbone_weight_norm_snapshots(backbone_norm_snapshots, step)

        # Buffer composition — diagnoses replay drift over time.
        for bucket_name, count in self.replay_buffer.bucket_sizes().items():
            self.training_logger.log(f"buffer/bucket_{bucket_name}", count, step=step, csv=False)
        self.training_logger.log("buffer/total_size", self.replay_buffer.size(), step=step, csv=False)

        return {
            "Q_loss": loss_value,
            "q_mean": q_mean,
        }

    def reset_episode(self):
        self._flush_n_step_buffer()
        self.clear_blocked_actions()

    # ──────────────────────────── diagnostics helpers ──────────────────────

    def _write_metric_docs(self):
        """Write metric interpretation notes to TensorBoard TEXT once."""
        is_weight_mean_doc = (
            "**`check/is_weight_mean`** - Mean PER importance-sampling weight "
            "(max-normalized, so max is always 1.0; watch the mean).\n\n"
            "| Range | Meaning | Concern? |\n"
            "|---|---|---|\n"
            "| Near 1.0 | Priorities are flat; PER is close to uniform replay | "
            "PER may not be useful, or TD errors are similar |\n"
            "| Middle (0.3 ~ 0.8) | Healthy biased sampling with enough bias correction | Normal |\n"
            "| Near 0 | Priorities are highly concentrated; a few samples dominate gradients | "
            "Possible overfit to a few hard samples |\n\n"
            "Watch this with beta annealing: as beta rises, the mean should slowly "
            "fall. If it rises instead, the priority distribution may be collapsing."
        )
        self.training_logger.log_text("docs/is_weight_mean", is_weight_mean_doc, step=0)

        is_weight_ratio_doc = (
            "**`check/is_weight_ratio`** - IS weight max/min ratio (= 1.0 / min, "
            "because IS weights are max-normalized and max is always 1.0). "
            "**Primary IS bug detector**.\n\n"
            "| Value | Meaning | Concern? |\n"
            "|---|---|---|\n"
            "| < 10 | Healthy priorities; IS correction works and all batch items affect gradients | Normal |\n"
            "| 10 ~ 100 | Some sample weights are suppressed; gradient contribution is uneven | "
            "Watch for overly skewed priorities |\n"
            "| > 100 | Severe imbalance; most batch slots barely affect gradients | "
            "**Red flag**: check `_selection_priorities` / `_sample_from_bucket` sync "
            "and whether dead-entry priority_min floors are bypassed |\n\n"
            "History: before the IS bug fix, the buggy ratio often exceeded 10^4. Dead "
            "entries with raw _effective_priority=0 hit `(1e-10)^(-beta) ~= 10^4` "
            "inside the IS formula, became max, and pushed live-entry weights near 1e-5."
        )
        self.training_logger.log_text("docs/is_weight_ratio", is_weight_ratio_doc, step=0)

    def _assert_finite(self, stage, name, tensor):
        """NaN/Inf probe for training; raises with stage / tensor / step on hit.

        Only checks floating-point tensors; isfinite is not meaningful for int/bool.
        Called at each stage boundary. GPU sync cost is about 50 us, acceptable
        for always-on diagnostics.
        """
        if tensor is None:
            return
        if not tensor.dtype.is_floating_point:
            return
        if torch.isfinite(tensor).all():
            return
        nan_n = int(torch.isnan(tensor).sum().item())
        inf_n = int(torch.isinf(tensor).sum().item())
        finite_mask = torch.isfinite(tensor)
        absmax = (
            float(tensor[finite_mask].abs().max().item())
            if finite_mask.any() else float("nan")
        )
        raise RuntimeError(
            f"[NaN-probe] non-finite at stage='{stage}' tensor='{name}' "
            f"step={self.total_it} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"nan={nan_n} inf={inf_n} finite_absmax={absmax:.4g}"
        )

    def _scan_state_dict_finite(self, sd_label, state_dict):
        """Scan all floating-point tensors in state_dict and return [(label, msg)].

        Does not raise; callers decide whether to raise on load or write .crash
        files on save.
        """
        bad = []
        for key, tensor in state_dict.items():
            if not hasattr(tensor, "dtype"):
                continue
            if not tensor.dtype.is_floating_point:
                continue
            if torch.isfinite(tensor).all():
                continue
            nan_n = int(torch.isnan(tensor).sum().item())
            inf_n = int(torch.isinf(tensor).sum().item())
            bad.append((f"{sd_label}.{key}", f"nan={nan_n} inf={inf_n}"))
        return bad

    def _scan_optimizer_state(self, label, opt_state_dict):
        """Scan optimizer state_dict, nested as state[pid][key].

        Checks two invalid states:
        (a) any tensor containing NaN/Inf
        (b) exp_avg_sq < 0. Adam's second moment is mathematically non-negative;
            negatives imply bit-level corruption such as a sign-bit flip and can
            make sqrt(v) produce NaN.

        Returns a [(label, message)] list.
        """
        bad = []
        state = opt_state_dict.get("state", {}) if isinstance(opt_state_dict, dict) else {}
        for pid, pstate in state.items():
            if not isinstance(pstate, dict):
                continue
            for key, val in pstate.items():
                if not isinstance(val, torch.Tensor):
                    continue
                if not val.dtype.is_floating_point:
                    continue
                if not torch.isfinite(val).all():
                    nan_n = int(torch.isnan(val).sum().item())
                    inf_n = int(torch.isinf(val).sum().item())
                    bad.append((f"{label}.state[{pid}].{key}", f"nan={nan_n} inf={inf_n}"))
                if key == "exp_avg_sq" and (val < 0).any().item():
                    neg_n = int((val < 0).sum().item())
                    bad.append((f"{label}.state[{pid}].{key}", f"negative={neg_n} (math-impossible)"))
        return bad

    def _clamp_optimizer_v_and_log(self):
        """Pre-step guard: clamp negative Adam exp_avg_sq(v) values in-place.

        Motivation: v is mathematically always >= 0 (beta2 * v_old +
        (1 - beta2) * grad^2, both non-negative). If a negative value appears,
        the likely cause is a transient VRAM bit flip on consumer GPUs without
        ECC. If left uncleared, AdamW computes sqrt(negative)=NaN and writes a
        NaN weight, which is what stage7 catches.

        Self-healing, without raising: turn a hardware transient into a small
        recoverable event, but log every hit in four places so long-term grep can
        build frequency and location distributions:
          - stdout (visible in the training console immediately)
          - self._io_log (aligned with other step diagnostics)
          - models/stage1_transformer/vclamp_events.log (dedicated event log)
          - TensorBoard (vclamp/elements_this_step / elements_total / params_this_step)

        Also zero the paired exp_avg(m) at hit positions so contaminated momentum
        does not keep pushing the reset weight element in a bad direction.
        """
        detections = []
        # Use named_parameters for human-readable log names. optimizer.state uses
        # parameter objects as keys, without names.
        for source_name, module in (("backbone", self.backbone), ("q_network", self.q_network)):
            for pname, p in module.named_parameters():
                st = self.optimizer.state.get(p)
                if st is None:
                    continue
                v = st.get("exp_avg_sq")
                if not isinstance(v, torch.Tensor) or not v.dtype.is_floating_point:
                    continue
                neg_mask = v < 0
                if not bool(neg_mask.any().item()):
                    continue
                neg_n = int(neg_mask.sum().item())
                # Log at most 5 positions and values to avoid huge corruption logs.
                sample_pos = neg_mask.nonzero(as_tuple=False)[:5].tolist()
                sample_vals = v[neg_mask][:5].tolist()
                full_name = f"{source_name}.{pname}"
                detections.append((full_name, tuple(v.shape), neg_n, sample_pos, sample_vals))
                # In-place clamp v >= 0
                v.clamp_(min=0)
                # Clear m at the same positions.
                m = st.get("exp_avg")
                if isinstance(m, torch.Tensor) and m.shape == v.shape:
                    m[neg_mask] = 0.0

        if not detections:
            return

        total = sum(d[2] for d in detections)
        header = (
            f"[vclamp] step={self.total_it} caught {len(detections)} param(s),"
            f" {total} negative-v element(s) -- clamped to 0 (+ paired m zeroed)"
        )
        lines = [header]
        for full_name, shape, neg_n, sample_pos, sample_vals in detections:
            sample = ", ".join(
                f"@{tuple(pos)}={val:.4e}" for pos, val in zip(sample_pos, sample_vals)
            )
            lines.append(
                f"[vclamp]   {full_name} shape={shape} neg_count={neg_n} sample=[{sample}]"
            )

        # 1) stdout: immediately visible in the training console.
        for line in lines:
            print(line)
        # 2) io_log: align with training step diagnostics.
        try:
            self._io_log.write("\n".join(lines) + "\n")
            self._io_log.flush()
        except Exception:
            pass
        # 3) Dedicated event log in the current hour dir, aligned with io_log.
        try:
            with (self.current_archive_dir / "vclamp_events.log").open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass
        # 4) TensorBoard: visualize the hit timeline.
        if not hasattr(self, "_vclamp_total"):
            self._vclamp_total = 0
        self._vclamp_total += total
        try:
            self.training_logger.log("vclamp/elements_this_step", total, step=self.total_it, csv=False)
            self.training_logger.log("vclamp/elements_total", self._vclamp_total, step=self.total_it, csv=False)
            self.training_logger.log("vclamp/params_this_step", len(detections), step=self.total_it, csv=False)
        except Exception:
            pass

    # ──────────────────────────── archive directory / hour rollover ──────
    # Directory ownership (session_dir / hour_index / current_archive_dir and
    # rollover) lives in model_structure.archive_manager.SessionArchiveManager.
    # This class only keeps agent-specific hooks: io_log banner content and which
    # files are swapped on rollover.
    #
    # current_archive_dir is exposed as a read-only property delegate so existing
    # callers such as train_stage1_simple.py do not need changes.

    @property
    def current_archive_dir(self) -> Path:
        return self.archive.current_archive_dir

    def _build_io_log_banner(self, path: Path) -> list[str]:
        """Build the RolloverTextLog banner with session/hour starts and path."""
        return [
            f"Session started: {self.archive.session_start.isoformat()}",
            f"Hour {self.archive.hour_index:02d} started: {datetime.datetime.now().isoformat()}",
            f"Path: {path}",
        ]

    def _on_archive_rollover(self, new_dir: Path) -> None:
        """SessionArchiveManager rollover callback: swap io_log and debug logger.

        - io_log writes one inline rollover footer before swap, so the old file
          has a tail marker and the new file starts with RolloverTextLog's banner.
        - vclamp_events.log / training_log.csv are lazy-opened on write, so they
          follow self.archive.current_archive_dir without explicit handling here.
        - canonical *.pth files are unchanged. The next _save_model call writes
          the archive snapshot into the new hour directory.
        """
        now = datetime.datetime.now()
        try:
            self._io_log.write(f"\n--- hour rollover at {now.isoformat()} ---\n")
            self._io_log.flush()
        except Exception:
            pass
        self._io_log.swap_to(new_dir / "train_io_log.txt")
        dbg_path = new_dir / "cuda_debug.log"
        swap_logger_file_handler(_dbg_logger, dbg_path)
        _crb_module.DEBUG_CUDA_SAMPLE_LOG_PATH = dbg_path

    # ──────────────────────────── diagnostics helpers (cont.) ─────────────

    def _module_grad_norm(self, module):
        """L2 norm of all gradients inside a module (post-backward, pre-clip)."""
        grad_sq_sum = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())
        return grad_sq_sum ** 0.5

    def _trainable_module_groups(self):
        return {
            "backbone": self.backbone,
            "head": self.q_network,
        }

    def _capture_trainable_weight_snapshot(self):
        """Detached CPU snapshot of every trainable parameter, keyed by group.param."""
        snapshot = {}
        for group_name, module in self._trainable_module_groups().items():
            for param_name, param in module.named_parameters():
                snapshot[f"{group_name}.{param_name}"] = param.detach().float().cpu().clone()
        return snapshot

    def _snapshot_distance(self, current_snapshot, reference_snapshot, eps=1e-12):
        """Relative L2 distance ||current - ref|| / ||ref|| across full parameter vector."""
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

    def _tensor_norm(self, tensor):
        if tensor is None:
            return None
        return float(tensor.detach().float().norm().item())

    def _collect_param_weight_and_grad_norm(self, tag_prefix, param, out):
        """Append (tag, weight_norm, grad_norm) to ``out`` for later TB writes.

        Split from the TB write so callers inside CUDA-recovery try-blocks
        can collect snapshots, return None on crash, and only write once they
        reach the post-try success path. Writing inside the try would let a
        crash leave per-layer rows in TB without matching global rows.
        """
        if param is None:
            return
        weight_norm = self._tensor_norm(param)
        grad_norm = self._tensor_norm(param.grad)
        out.append((tag_prefix, weight_norm, grad_norm))

    def _collect_backbone_weight_norm_snapshots(self):
        """Capture per-layer weight & pre-clip grad norms for encoder & decoder.

        Returns ``list[(tag_prefix, weight_norm, grad_norm)]``. Must be called
        BEFORE ``clip_grad_norm_`` so the grad values are pre-clip. Pair with
        ``_write_backbone_weight_norm_snapshots`` for the actual TB writes.

        Uses standard ``nn.TransformerEncoderLayer`` / ``nn.TransformerDecoderLayer``
        attribute names (``self_attn.in_proj_weight``, ``linear1.weight`` …).
        """
        snapshots: list = []
        encoder_layers = getattr(self.backbone.core.transformer, "layers", None)
        if encoder_layers is not None:
            for layer_idx, layer in enumerate(encoder_layers):
                prefix = f"encoder/layer{layer_idx}"
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/self_attn_in_proj",
                    getattr(layer.self_attn, "in_proj_weight", None),
                    snapshots,
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/self_attn_out_proj",
                    layer.self_attn.out_proj.weight,
                    snapshots,
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/ffn_linear1", layer.linear1.weight, snapshots
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/ffn_linear2", layer.linear2.weight, snapshots
                )

        decoder_layers = getattr(self.backbone.core.decoder, "layers", None)
        if decoder_layers is not None:
            for layer_idx, layer in enumerate(decoder_layers):
                prefix = f"decoder/layer{layer_idx}"
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/self_attn_in_proj",
                    getattr(layer.self_attn, "in_proj_weight", None),
                    snapshots,
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/self_attn_out_proj",
                    layer.self_attn.out_proj.weight,
                    snapshots,
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/cross_attn_in_proj",
                    getattr(layer.multihead_attn, "in_proj_weight", None),
                    snapshots,
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/cross_attn_out_proj",
                    layer.multihead_attn.out_proj.weight,
                    snapshots,
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/ffn_linear1", layer.linear1.weight, snapshots
                )
                self._collect_param_weight_and_grad_norm(
                    f"{prefix}/ffn_linear2", layer.linear2.weight, snapshots
                )
        return snapshots

    def _write_backbone_weight_norm_snapshots(self, snapshots, global_step):
        """Write previously-collected per-layer snapshots to TensorBoard."""
        for tag_prefix, weight_norm, grad_norm in snapshots:
            if weight_norm is not None:
                self.training_logger.log(f"weight_norm/{tag_prefix}", weight_norm, step=global_step, csv=False)
            if grad_norm is not None:
                self.training_logger.log(f"grad_norm/{tag_prefix}", grad_norm, step=global_step, csv=False)

    def _close_training_logger(self):
        """Atexit hook: close both the CSV file handle and SummaryWriter.

        SummaryWriter only lives inside self.training_logger after __init__, with
        no external references. A single close() replaces the old separate
        _close_tb_writer path.
        """
        logger = getattr(self, "training_logger", None)
        if logger is not None and not logger.closed:
            logger.close()

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def on_episode_end(self):
        self._flush_n_step_buffer()
        # episode_count is derived from training_history via history.record() in
        # log_episode_metrics(), so do not increment here. epsilon is also updated
        # there through controller.update().
        self.training_logger.log("episode/epsilon", self.epsilon, step=self.episode_count, csv=False)
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(
                f"[FQF] Periodic save at episode {self.episode_count}"
                f" | epsilon={self.epsilon:.4f}"
            )
            self._save_model()
            self.save_persistent()

    def log_episode_metrics(
        self,
        win: bool,
        invalid_click_rate: float = 0.0,
        reward_mean: float = 0.0,
        *,
        total_reward: float = 0.0,
        steps: int = 0,
    ) -> None:
        """Record one episode result and update epsilon from rolling win rate.

        Matches the v3 / v2 interface: training scripts should call this once
        before `on_episode_end()`.
        Flow:
        1) Store result, total_reward, and steps in TrainingHistory.
        2) Read rolling win rate from history.
        3) Feed win rate to the controller for next epsilon.

        total_reward / steps are keyword-only. Legacy callers that omit them
        record 0 in history reward/steps stats, with no effect on win_rate or
        epsilon decay.
        """
        self.training_history.record(
            win=win,
            total_reward=total_reward,
            steps=steps,
            invalid_rate=invalid_click_rate,
        )
        ep_idx = self.training_history.total_episodes
        rolling_wr = self.training_history.win_rate(window=100)
        self.epsilon_controller.update(rolling_wr)

        self.training_logger.log("episode/reward_mean", float(reward_mean), step=ep_idx, csv=False)
        self.training_logger.log("episode/invalid_click_rate", float(invalid_click_rate), step=ep_idx, csv=False)

    # ──────────────────────────── lr warmup ────────────────────────────

    def _apply_lr_warmup(self) -> None:
        """Linear LR warmup using the stricter of two warmups.

        - init warmup uses total_it and applies when training from scratch.
        - resume warmup uses steps_since_resume and applies on every start,
          including restarts.
        Final factor = min(init_factor, resume_factor), so after a restart,
        resume warmup still ramps LR from LR_WARMUP_START_FACTOR * base_lr back
        to base_lr even when total_it is already large.
        """

        def _factor_from(step: int, total: int) -> float:
            if total <= 0:
                return 1.0
            progress = min(1.0, step / total)
            return LR_WARMUP_START_FACTOR + (1.0 - LR_WARMUP_START_FACTOR) * progress

        init_factor   = _factor_from(self.total_it,           LR_WARMUP_STEPS)
        resume_factor = _factor_from(self.steps_since_resume, LR_RESUME_WARMUP_STEPS)
        factor = min(init_factor, resume_factor)
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * factor

    def _save_model(self):
        # Flush TB before saving — pair the on-disk model checkpoint with the
        # matching TB scalars from this point in training.
        self.training_logger.flush()
        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        backbone_sd = self.backbone.state_dict()
        qnet_sd = self.q_network.state_dict()
        qtarget_sd = self.q_target.state_dict()

        # Optimizer state only contains optimizer / total_it / episode_count /
        # algorithm / epsilon. Episode result accumulation and queries live in
        # TrainingHistory and are saved to training_history.pth below.
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "total_it": self.total_it,
            "episode_count": self.episode_count,
            "algorithm": "FQF",
        }
        payload.update(self.epsilon_controller.state_dict())
        training_history_sd = self.training_history.state_dict()

        # RNG state snapshot: save all four RNG sources so restarts can resume
        # exactly. Without this, restarts use fresh seeds, so Minesweeper board
        # generation, epsilon-random actions, replay sampling, and similar paths
        # diverge from the save point. That makes new trajectories differ from
        # old transitions in the buffer and can cause bootstrap mismatch and win
        # rate drops. Seed-42 duplicate runs confirmed bit-identical results.
        payload["rng_state"] = {
            "python_random": random.getstate(),
            "numpy":         np.random.get_state(),
            "torch_cpu":     torch.get_rng_state(),
            "cuda":          torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Save-time NaN probe: scan all state_dicts and refuse to overwrite
        # canonical checkpoints if any are corrupt. atexit also runs after NaN
        # crashes, so this prevents replacing the last good disk checkpoint with
        # a bad one. Also scan optimizer state; negative Adam v can immediately
        # create NaN weights after the next load.
        bad = []
        bad.extend(self._scan_state_dict_finite("backbone", backbone_sd))
        bad.extend(self._scan_state_dict_finite("q_network", qnet_sd))
        bad.extend(self._scan_state_dict_finite("q_target", qtarget_sd))
        bad.extend(self._scan_optimizer_state("optimizer", payload.get("optimizer", {})))

        if bad:
            suffix = f".crash_step{self.total_it}.pth"
            print(f"[NaN-probe] _save_model: REFUSING to overwrite canonical checkpoints at step {self.total_it}")
            print(f"[NaN-probe] non-finite tensors detected:")
            for key, msg in bad:
                print(f"[NaN-probe]   {key}: {msg}")
            torch.save(backbone_sd,  TRANSFORMER_MODEL_PATH / f"backbone{suffix}")
            torch.save(qnet_sd,      TRANSFORMER_MODEL_PATH / f"fqf_network{suffix}")
            torch.save(qtarget_sd,   TRANSFORMER_MODEL_PATH / f"fqf_target{suffix}")
            torch.save(payload,      TRANSFORMER_MODEL_PATH / f"optimizer_state{suffix}")
            print(f"[NaN-probe] wrote *{suffix} files for offline analysis;"
                  f" canonical *.pth left untouched (last good state preserved)")
            return

        torch.save(backbone_sd,        TRANSFORMER_MODEL_PATH / "backbone.pth")
        torch.save(qnet_sd,            TRANSFORMER_MODEL_PATH / "fqf_network.pth")
        torch.save(qtarget_sd,         TRANSFORMER_MODEL_PATH / "fqf_target.pth")
        torch.save(payload,            TRANSFORMER_MODEL_PATH / "optimizer_state.pth")
        torch.save(training_history_sd, TRANSFORMER_MODEL_PATH / "training_history.pth")

        # Also write the same payload into the current hour directory as a
        # history snapshot. Multiple saves in one hour overwrite that hour's
        # snapshot, leaving the last save for that hour.
        try:
            archive = self.current_archive_dir
            archive.mkdir(parents=True, exist_ok=True)
            torch.save(backbone_sd,         archive / "backbone.pth")
            torch.save(qnet_sd,             archive / "fqf_network.pth")
            torch.save(qtarget_sd,          archive / "fqf_target.pth")
            torch.save(payload,             archive / "optimizer_state.pth")
            torch.save(training_history_sd, archive / "training_history.pth")
        except Exception as exc:
            # Archive failure must not block a successful canonical save.
            print(f"[FQF] WARN: archive snapshot write failed: {exc}")

    def save_persistent(self):
        buf = self.replay_buffer
        total = buf.size()
        if total == 0:
            return

        selected_entries = buf.export_top_k(min(SAVE_CAPACITY, total))

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "persistent_entries": selected_entries,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
            },
            TRANSFORMER_MODEL_PATH / "replay_buffer.pth",
        )

        saved_rewards = defaultdict(int)
        saved_buckets = defaultdict(int)
        for entry in selected_entries:
            saved_rewards[entry.get("tail_reward", entry["reward"])] += 1
            saved_buckets[entry["reward_type"]] += 1
        print("--- save info ---------------")
        print(f"[FQF] Persistent save: {len(selected_entries)} entries")
        print(f"[FQF] Tail reward distribution: {dict(saved_rewards)}")
        print(f"[FQF] Reward bucket distribution: {dict(saved_buckets)}")
        print("--- save end ---------------")

    def _resolve_load_path(self, canonical_path):
        """Choose load path: canonical first, otherwise latest archive.

        A corrupt canonical checkpoint is not treated as missing. _raise_if_corrupt
        catches it instead of silently falling back to archive, because corruption
        usually needs explicit handling such as sanitize or rollback.

        Archive scanning lives in SessionArchiveManager.find_latest_archive; this
        method only owns canonical/archive priority and the console message.
        """
        if canonical_path.exists():
            return canonical_path
        archive_path = self.archive.find_latest_archive(canonical_path.name)
        if archive_path is not None:
            print(
                f"[FQF] canonical {canonical_path.name} missing, "
                f"falling back to latest archive: {archive_path}"
            )
            return archive_path
        return None

    def _load_training_history(self, legacy_state: dict | None = None) -> None:
        """Load TrainingHistory, preferring training_history.pth.

        Legacy flattened optimizer state is the fallback for one-time migration.
        Missing canonical files automatically fall back to the latest archive via
        _resolve_load_path, matching other checkpoint load paths.
        """
        history_path = self._resolve_load_path(
            TRANSFORMER_MODEL_PATH / "training_history.pth"
        )
        if history_path is not None:
            try:
                hist_state = torch.load(
                    history_path, map_location=device, weights_only=False
                )
                self.training_history.load_state_dict(
                    hist_state, deque_cls=self.deque_cls
                )
                return
            except Exception as exc:
                print(f"[FQF] Failed to load training_history.pth: {exc}")
                # Fall through to the legacy fallback below.
        if legacy_state and any(
            k in legacy_state for k in ("result_window", "total_episodes", "total_wins")
        ):
            legacy = {
                "results": legacy_state.get("result_window", []),
                "total_episodes": legacy_state.get("total_episodes", 0),
                "total_wins": legacy_state.get("total_wins", 0),
            }
            self.training_history.load_state_dict(legacy, deque_cls=self.deque_cls)
            print("[FQF] Migrated legacy training history from optimizer state")

    def _raise_if_corrupt(self, label, path, state_dict):
        """Load-time NaN probe: raise if checkpoint contains NaN/Inf.

        This prevents silent resume. Keep the raise outside the caller's broad
        try/except path so it cannot be swallowed by message-only error handling.
        """
        bad = self._scan_state_dict_finite(label, state_dict)
        if not bad:
            return
        lines = "\n".join(f"  {k}: {msg}" for k, msg in bad)
        raise RuntimeError(
            f"[NaN-probe] disk checkpoint corrupt at {path}:\n{lines}\n"
            f"Refusing to load. Options: (a) run sanitize_checkpoint.py to clean, "
            f"(b) revert to an older checkpoint in git, (c) start training from scratch."
        )

    def try_load_model(self):
        backbone_path = self._resolve_load_path(TRANSFORMER_MODEL_PATH / "backbone.pth")
        if backbone_path is not None:
            backbone_state = None
            try:
                backbone_state = torch.load(backbone_path, map_location=device, weights_only=True)
            except Exception as exc:
                print(f"[FQF] Failed to read Backbone: {exc}")
            if backbone_state is not None:
                self._raise_if_corrupt("backbone(disk)", backbone_path, backbone_state)
                try:
                    incompatible = self.backbone.load_backbone_state(backbone_state, strict=False)
                    print("[FQF] Loaded Backbone")
                    if incompatible.missing_keys:
                        print(f"[FQF] Backbone missing keys: {incompatible.missing_keys}")
                    if incompatible.unexpected_keys:
                        print(f"[FQF] Backbone unexpected keys: {incompatible.unexpected_keys}")
                except Exception as exc:
                    print(f"[FQF] Failed to apply Backbone state: {exc}")

        q_path = self._resolve_load_path(TRANSFORMER_MODEL_PATH / "fqf_network.pth")
        if q_path is not None:
            q_state = None
            try:
                q_state = torch.load(q_path, map_location=device, weights_only=True)
            except Exception as exc:
                print(f"[FQF] Failed to read FQF-Network: {exc}")
            if q_state is not None:
                self._raise_if_corrupt("q_network(disk)", q_path, q_state)
                try:
                    self.q_network.load_state_dict(q_state)
                    print("[FQF] Loaded FQF-Network")
                except Exception as exc:
                    print(f"[FQF] Failed to apply FQF-Network state: {exc}")
        elif (TRANSFORMER_MODEL_PATH / "q_network.pth").exists():
            print("[FQF] Skip legacy q_network.pth because DDQN head shape is incompatible")

        q_target_path = self._resolve_load_path(TRANSFORMER_MODEL_PATH / "fqf_target.pth")
        if q_target_path is not None:
            qt_state = None
            try:
                qt_state = torch.load(q_target_path, map_location=device, weights_only=True)
            except Exception as exc:
                print(f"[FQF] Failed to read FQF-Target: {exc}")
            if qt_state is not None:
                self._raise_if_corrupt("q_target(disk)", q_target_path, qt_state)
                try:
                    self.q_target.load_state_dict(qt_state)
                    print("[FQF] Loaded FQF-Target")
                except Exception as exc:
                    print(f"[FQF] Failed to apply FQF-Target state: {exc}")
        elif (TRANSFORMER_MODEL_PATH / "q_target.pth").exists():
            print("[FQF] Skip legacy q_target.pth because DDQN head shape is incompatible")

        opt_path = self._resolve_load_path(TRANSFORMER_MODEL_PATH / "optimizer_state.pth")
        if opt_path is not None:
            opt_payload = None
            try:
                opt_payload = torch.load(opt_path, map_location=device, weights_only=False)
            except Exception as exc:
                print(f"[FQF] Failed to read optimizer state: {exc}")
            if opt_payload is not None:
                # Load-time optimizer probe: scan for NaN/Inf and exp_avg_sq < 0.
                # Adam v cannot be negative; if it is, bit-level corruption can
                # make AdamW compute sqrt(negative)=NaN and corrupt weights.
                bad = self._scan_optimizer_state("optimizer(disk)", opt_payload.get("optimizer", {}))
                if bad:
                    lines = "\n".join(f"  {k}: {msg}" for k, msg in bad)
                    raise RuntimeError(
                        f"[NaN-probe] disk optimizer state corrupt at {opt_path}:\n{lines}\n"
                        f"Refusing to load. Run sanitize_checkpoint.py to clean,"
                        f" or revert to an older checkpoint."
                    )
                try:
                    self.optimizer.load_state_dict(opt_payload["optimizer"])
                    self.total_it = opt_payload["total_it"]
                    # Do not set episode_count directly. It delegates to
                    # training_history.total_episodes, restored from its own .pth.
                    # Ignore legacy opt_payload["episode_count"] because it
                    # duplicates history.total_episodes. The controller only needs
                    # epsilon; flattened legacy history uses the fallback loader.
                    self.epsilon_controller.load_state_dict(opt_payload)
                    self._load_training_history(legacy_state=opt_payload)
                    print(
                        f"[FQF] Loaded optimizer: total_it={self.total_it},"
                        f" episode={self.episode_count}, epsilon={self.epsilon:.4f},"
                        f" total_episodes={self.training_history.total_episodes}"
                    )
                    # RNG state restore: resume random / numpy / torch / cuda RNG
                    # from the save point so trajectories continue from the saved
                    # distribution and do not conflict with old buffer transitions.
                    # Legacy checkpoints without rng_state are skipped for backward
                    # compatibility. torch.load(map_location=device) can move
                    # payload tensors to device, while torch.set_rng_state() needs a
                    # CPU ByteTensor, so move it to CPU before passing.
                    rng_state = opt_payload.get("rng_state")
                    if rng_state:
                        restored = []
                        try:
                            random.setstate(rng_state["python_random"])
                            restored.append("python_random")
                        except (KeyError, TypeError, ValueError) as exc:
                            print(f"[RNG] failed to restore python random: {exc}")
                        try:
                            np.random.set_state(rng_state["numpy"])
                            restored.append("numpy")
                        except (KeyError, TypeError, ValueError) as exc:
                            print(f"[RNG] failed to restore numpy: {exc}")
                        try:
                            torch_cpu_state = rng_state["torch_cpu"]
                            if torch.is_tensor(torch_cpu_state):
                                torch_cpu_state = torch_cpu_state.cpu()
                            torch.set_rng_state(torch_cpu_state)
                            restored.append("torch_cpu")
                        except (KeyError, TypeError, RuntimeError) as exc:
                            print(f"[RNG] failed to restore torch cpu: {exc}")
                        cuda_state = rng_state.get("cuda")
                        if cuda_state is not None and torch.cuda.is_available():
                            try:
                                # CUDA RNG state is list[Tensor], one per GPU.
                                # Force each element back to CPU.
                                cuda_state_cpu = [
                                    s.cpu() if torch.is_tensor(s) else s
                                    for s in cuda_state
                                ]
                                torch.cuda.set_rng_state_all(cuda_state_cpu)
                                restored.append("cuda")
                            except (TypeError, RuntimeError) as exc:
                                print(f"[RNG] failed to restore cuda: {exc}")
                        print(f"[RNG] Restored RNG state: {', '.join(restored)}")
                    else:
                        print(
                            f"[RNG] No rng_state in checkpoint (legacy save) —"
                            f" restart will use fresh RNG,trajectory 不會跟 save"
                            f" 那刻延續(這是 win rate drop 的根因,新 save 會修)"
                        )
                except Exception as exc:
                    print(f"[FQF] Failed to apply optimizer state: {exc}")

        # New name is replay_buffer.pth. If it is missing but legacy
        # training_state.pth exists, load the legacy file. The next save writes the
        # new name, and the old file can be deleted manually.
        replay_buffer_path = TRANSFORMER_MODEL_PATH / "replay_buffer.pth"
        legacy_path = TRANSFORMER_MODEL_PATH / "training_state.pth"
        if not replay_buffer_path.exists() and legacy_path.exists():
            print(f"[FQF] replay_buffer.pth not found, loading from legacy training_state.pth")
            training_state_path = legacy_path
        else:
            training_state_path = replay_buffer_path
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                persistent_entries = state.get("persistent_entries", [])
                
                # Check format to cleanly transition if an old save has a different shape
                if persistent_entries and isinstance(persistent_entries[0], dict) and "storage_id" in persistent_entries[0]:
                    # Exactly load using new format
                    self.replay_buffer.load_from_entries(persistent_entries)
                    loaded = len(persistent_entries)
                else:
                    # Legacy transition format
                    loaded = min(len(persistent_entries), PER_CAPACITY)
                    for idx in range(loaded):
                        entry = persistent_entries[idx]
                        self.replay_buffer.store(
                            entry["state"],
                            entry["action"],
                            entry["next_state"],
                            entry["reward"],
                            entry["done"],
                        )
                
                if loaded > 0:
                    print(f"[FQF] Loaded {loaded} replay buffer entries")
            except Exception as exc:
                print(f"[FQF] Failed to load replay buffer: {exc}")
