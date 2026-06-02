"""Stage 2 visual agent: screenshot -> FQF Q-network.

Pipeline (dims/layers come from yolo_encoder_base and DECODER_* constants):
    screenshot (3, 640, 640)
        ↓ YOLOEncoderBase split:
        ↓   [frozen] YOLO11n backbone → (128, 40, 40)
        ↓   ───────────── REPLAY BUFFER stores (128, 40, 40) fp32 features here ─────────────
        ↓   [trainable] token_adapter + 2D sinusoidal pos enc → (1600, encoder_dims[0])
        ↓   [trainable] HierarchicalEncoder → (1600, final_dim)
    encoded memory (B, 1600, final_dim)
        ↓ TransformerDecoder x DECODER_NUM_LAYERS (pre-LN, cross-attn, learned query tokens)
    decoded features (B, GRID_H*GRID_W, final_dim)
        ↓ FQFQNetwork (d_model=final_dim, num_fractions=NUM_FQF_FRACTIONS)
    Q-values (B, GRID_H, GRID_W) → masked argmax → action

Architecture notes:
    - VisualBackboneV3 inherits YOLOEncoderBase. Its encoder/decoder shape matches
      Stage 1: by default DEFAULT_ENCODER_START_DIM == FINAL_DIM, so token_adapter
      handles YOLO 128->64 compression.
    - Attempts to load encoder + decoder + query_tokens + FQF from Stage 1
      checkpoint as all-or-nothing. Missing files use warning + random init.
    - YOLO11n loads from yolo11n.pt and is frozen in BN eval mode. token_adapter,
      encoder, decoder, query_tokens, and FQF head are trainable.
    - Replay buffer stores YOLO backbone output instead of raw screenshots:
        - each transition drops from ~1.17 MB uint8 screenshot to ~0.78 MB fp32 features
        - training skips YOLO11n forward and runs token_adapter + encoder + decoder
        - because YOLO is frozen, cached features are equivalent to recomputing
    - Dropout starts off and latches on after win_rate(window=100) > 0.4, aligned
      with PER spread_decay.
"""

from __future__ import annotations

import atexit
import datetime
import hashlib
import math
import random
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from transformer_discrete_agent import (
    FQF_ENTROPY_COEF,
    FQF_HUBER_KAPPA,
    NUM_FQF_FRACTIONS,
    TRANSFORMER_MODEL_PATH,
    _quantile_huber_loss,
)
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from model_structure.transformer_shared import FQFQNetwork
from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer
from model_structure.visual_agent_common import VisualAgentCommonMixin
from model_structure.yolo_encoder_base import (
    YOLOEncoderBase,
    YOLO_FEATURE_CHANNELS,
    DEFAULT_ENCODER_DIMS,
    DEFAULT_ENCODER_FF_MULT,
)
from model_structure.optimizer_factory import build_fqf_optimizer, FQFOptimizerConfig
from model_structure.hyperparameter_dump import dump_hyperparameters
from model_structure.rng_utils import seed_everything
from model_structure.sdp_backend import set_sdp_all
from model_structure.archive_manager import (
    SessionArchiveManager,
    RolloverTextLog,
)
from model_structure.training_logger import TrainingLogger
from model_structure.adaptive_epsilon import AdaptiveEpsilonController
from model_structure.history import TrainingHistory
from model_structure.checkpoint_log import CheckpointLogger
from model_structure.eval_utils import log_eval_metrics as log_eval_metrics_common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CUDA SDP backend toggles ─────────────────────────────────────────
# Encoder self-attention at seq_len=1600 materializes a full (B, H, 1600, 1600)
# attention matrix under the math backend — huge memory hit. flash / mem_efficient
# compute attention tile-by-tile and never materialize the full matrix, cutting
# memory by an order of magnitude.
# Earlier NaN / "illegal instruction" with the fast SDP kernels was confirmed to
# be a GPU hardware issue, not the kernel itself; enable all three and let the
# PyTorch dispatcher pick.
#
# Two-layer constants:
#   _REQ_*    — what we'd like enabled. Flip these to disable. Underscore prefix
#               keeps them out of hyperparameter_dump.
#   USE_*_SDP — = _REQ_* AND device supports CUDA AND the PyTorch API exists.
#               ALL_CAPS so hyperparameter_dump picks them up — what gets logged
#               matches PyTorch's actual "permitted" state. Still does NOT mean
#               the kernel will run at forward time — the dispatcher may skip
#               flash for fp32 inputs, etc.
_REQ_FLASH:         bool = True
_REQ_MEM_EFFICIENT: bool = True
_REQ_MATH:          bool = True

USE_FLASH_SDP, USE_MEM_EFFICIENT_SDP, USE_MATH_SDP = set_sdp_all(
    flash=_REQ_FLASH, mem_efficient=_REQ_MEM_EFFICIENT, math=_REQ_MATH,
    device=device, log_prefix="[V3]",
)

# reproducibility
# Create a 32-bit seed at import time and apply it to random / numpy / torch / cuda.
# Same as stage1, SEED is a module-level ALL_CAPS int written to hyperparameters.txt
# and printed during agent.__init__ for later comparison.
# To reproduce a run, set `SEED: int = seed_everything(<number>)`.
SEED: int = seed_everything()

# NaN/Inf checks
# Single remaining helper: raise immediately on NaN/Inf to stop training.
# The old forced-sync debug path was removed; CUDA errors were treated as
# hardware or environment issues. Keep this for code-level NaN checks.
def _check_finite(name: str, t) -> None:
    """Raise RuntimeError if a tensor contains NaN or Inf.

    Non-tensors, non-floating tensors, and empty tensors are ignored. Cost is one
    ``.any().item()`` sync, so this is only used at hot-loop checkpoints.
    """
    if t is None or not torch.is_tensor(t):
        return
    if not t.dtype.is_floating_point or t.numel() == 0:
        return
    if torch.isnan(t).any().item() or torch.isinf(t).any().item():
        raise RuntimeError(
            f"[NaN/Inf] {name}: shape={tuple(t.shape)} dtype={t.dtype}"
        )

# paths
# Disk layout (v3, since 2026-05-23):
#   * MODEL_PATH  -> D drive (HDD): weights / training_state.pth / tensorboard /
#                    action_logs. These are low-frequency checkpoint/event writes.
#   * REPLAY_BASE -> C drive (SSD): disk-backed replay buffer hot path. The HDD is
#                    too slow for small-file random IO. Keep replay_buffer/ and
#                    replay_buffer_save/ names to preserve existing helper contracts.
# Note: training_state.pth remains under MODEL_PATH and contains absolute .pt paths
# under replay_buffer_save. pathlib handles cross-drive absolute paths. If the SSD
# cache is wiped manually, also remove the matching training_state.pth.
STAGE1_CKPT_DIR = TRANSFORMER_MODEL_PATH   # = Path("./models/stage1_transformer")

VISUAL_V3_MODEL_PATH             = Path("./models/visual_transformer_v3_6x6")
VISUAL_V3_REPLAY_BASE            = Path(r"C:\dont_move\temp\autotest")
VISUAL_V3_REPLAY_PATH            = VISUAL_V3_REPLAY_BASE / "replay_buffer"
VISUAL_V3_REPLAY_PERSISTENT_PATH = VISUAL_V3_REPLAY_BASE / "replay_buffer_save"
VISUAL_V3_TENSORBOARD_DIR        = VISUAL_V3_MODEL_PATH / "tensorboard"
VISUAL_V3_ACTION_LOG_PATH        = VISUAL_V3_MODEL_PATH / "action_logs"

# ── grid / batch ─────────────────────────────────────────────────────
IMAGE_SIZE = (640, 640)
GRID_H = 6
GRID_W = 6
NUM_ACTIONS = GRID_H * GRID_W
BATCH_SIZE = 32

# Encoder dims
# Shape is controlled by DEFAULT_ENCODER_FINAL_DIM / DEFAULT_ENCODER_TOTAL_LAYERS /
# DEFAULT_ENCODER_START_DIM in model_structure.yolo_encoder_base. Default [64]*5
# gives 4 uniform layers matching Stage 1 EncoderDecoderTransformer.
ENCODER_DIMS = DEFAULT_ENCODER_DIMS

# ── Decoder spec ─────────────────────────────────────────────────────
DECODER_D_MODEL    = ENCODER_DIMS[-1]       # = ENCODER_FINAL_DIM
DECODER_NHEAD      = 4
DECODER_NUM_LAYERS = 4
# Transformer convention: FFN width = 4 × d_model — keeps capacity ratio constant
# when d_model changes. Shared with encoder side (DEFAULT_ENCODER_FF_MULT).
DECODER_FF_DIM     = DECODER_D_MODEL * DEFAULT_ENCODER_FF_MULT
DECODER_DROPOUT    = 0.1
# Dropout starts at 0 and only turns on after the agent reaches the same competence threshold
# used by the PER spread_decay latch (transformer_discrete_agent.py:610-612). Rationale: early
# RL training has high variance and dropout adds more noise on top — we don't want to fight the
# initial bootstrap. Once win_rate(100) > threshold the model is mature enough that dropout
# helps regularize against overfitting to the current high-priority replay slice.
DROPOUT_LATCH_WR_THRESHOLD = 0.4

# ── training hyper-params ────────────────────────────────────────────
N_STEP = 1
GRAD_CLIP_NORM = 10.0   # Match TransformerDiscreteAgent; keep 10.0 after token_adapter + encoder became trainable.
TRAIN_EVERY_N_STEPS = 1
TARGET_UPDATE_FREQ = 200
SAVE_EVERY_N_EPISODES = 500
# Throttle names match stage1 for the same behavior.
# DIAGNOSTIC_LOG_EVERY: gate for io_log + batched .item() + ~22 TB scalars.
# HISTOGRAM_EVERY: gate for per-layer weight_norm/* + grad_norm/*.
# WEIGHT_DISTANCE_LOG_EVERY: whole-model weight snapshot distance.
# Values match stage1 (10 / 200 / 100); keep both agents in sync.
DIAGNOSTIC_LOG_EVERY = 10
HISTOGRAM_EVERY = 200
WEIGHT_DISTANCE_LOG_EVERY = 100
USE_AMP = False

# learning rates
# Most LR / weight_decay values use FQFOptimizerConfig defaults. V3 adds
# catastrophic-forgetting protection: Stage 1-loaded modules use 0.1x LR so sparse
# noisy RL gradients do not erase learned cell-level attention. token_adapter is
# always random init and uses fresh LR. If Stage 1 is missing, all backbone params
# are random init and use fresh LR.
# Tune via TB weight_norm/encoder.* and grad_norm/encoder.*:
#   - encoder weight delta stagnant too long -> raise LR_BACKBONE_PRETRAINED
#   - encoder RMS changes > 50% in first 1000 steps -> lower it
LR_BACKBONE_PRETRAINED = 5e-6  # Stage 1-loaded encoder + decoder + query_tokens: 0.1x fresh LR.
# Linear LR warmup over the first N optimizer steps for early transformer stability.
# Increase from base_lr * LR_WARMUP_START_FACTOR to base_lr.
LR_WARMUP_STEPS         = 2000  # Initial from-scratch warmup length.
LR_WARMUP_START_FACTOR  = 0.0
LR_RESUME_WARMUP_STEPS  = 2000  # Extra warmup on every restart, including first run.

# ── replay buffer ────────────────────────────────────────────────────
BUFFER_CAPACITY = 2048
SAVE_CAPACITY   = 512
MINIMUM_DATA_SIZE = min(BUFFER_CAPACITY, SAVE_CAPACITY*4)-1  # below this amount, won't start training
BUFFER_OVERFLOW = 256
PER_ALPHA       = 0.6
PER_UNIFORM_MIX = 0.2
PRIORITY_MIN    = 0.05
PRIORITY_MAX    = 5.0
PRIORITY_EPS    = 1e-3
AGE_DECAY       = 0.002
# PER beta annealing, aligned with transformer_discrete_agent.py PER_BETA_START/END.
# beta linearly anneals from START to END and saturates at episode_count=BETA_EP.
# Smaller beta gives weaker correction and more stable early training; beta=1
# fully corrects priority-sampling bias.
PER_BETA_START  = 0.4
PER_BETA_END    = 1.0
PER_BETA_EP     = 5000
# PER spread_decay latch, aligned with transformer_discrete_agent.py.
# FQF quantile spread feeds the priority modifier: wide spread means uncertain
# prediction, so effective priority is damped. Start off because early RL variance
# makes spread unreliable; once win_rate(100) crosses the threshold, latch ON
# permanently. Same threshold and monotone design as dropout latch.
SPREAD_DECAY             = 2.0
SPREAD_DECAY_LATCH_WR_THRESHOLD = DROPOUT_LATCH_WR_THRESHOLD

LOG_ACTIONS = True


# VisualBackboneV3: YOLOEncoderBase (frozen) + cross-attn decoder.
class VisualBackboneV3(YOLOEncoderBase):
    """screenshot (B,3,H,W) -> cell features (B, grid_h*grid_w, encoder_final_dim).

    YOLO + token_adapter + HierarchicalEncoder come from YOLOEncoderBase. The
    default encoder shape (start_dim == final_dim, 4 uniform layers at d=64)
    matches Stage 1 EncoderDecoderTransformer, so VisualAgentV3 can warm-start
    encoder + decoder + queries from Stage 1. YOLO11n backbone is frozen;
    token_adapter, encoder, decoder, and queries train with RL.
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
            ff_mult=DEFAULT_ENCODER_FF_MULT,
            dropout=decoder_dropout,
        )
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_queries = grid_h * grid_w
        out_dim = self.out_dim

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, out_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=out_dim,
            nhead=decoder_nhead,
            dim_feedforward=decoder_ff_dim,
            dropout=decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)

    def _build_queries(self, batch_size: int) -> torch.Tensor:
        return self.query_tokens.expand(batch_size, -1, -1)

    def get_features(self, screenshot: torch.Tensor) -> torch.Tensor:
        """screenshot (B, 3, H, W) → cell features (B, num_queries, out_dim).

        Live-inference path (select_action). Runs the full pipeline including the
        frozen YOLO backbone forward.
        """
        memory  = self.encode(screenshot)                        # (B, 1600, out_dim)
        queries = self._build_queries(memory.size(0))            # (B, num_queries, out_dim)
        return self.decoder(queries, memory)                     # (B, num_queries, out_dim)

    def get_features_from_cached(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """Cached YOLO features (B, 128, h, w) → cell features (B, num_queries, out_dim).

        Training-time path. Skips the frozen YOLO backbone forward — the features
        were precomputed once at storage time. token_adapter + encoder + decoder
        all run here with gradients (encoder is trainable in this configuration).
        """
        memory  = self.encode_from_backbone_features(backbone_features)
        queries = self._build_queries(memory.size(0))
        return self.decoder(queries, memory)

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
        self.save_capacity = SAVE_CAPACITY
        self.priority_min = PRIORITY_MIN
        self.priority_max = PRIORITY_MAX
        self.log_prefix = "[V3]"
        self.log_actions = LOG_ACTIONS

        # Checkpoint logger: green prints on successful load, red on failure,
        # and per-area source tracker that hyperparameters.txt emits as a
        # [loaded_checkpoints] section. Built before backbone construction so
        # the YOLO11n load can also record its source path.
        self.checkpoint_logger = CheckpointLogger("[V3 CHECKPOINT]")

        # Backbone: YOLO11n is frozen; the rest is trainable. Encoder / decoder /
        # queries are loaded from Stage 1 below when possible.
        # Replay stores YOLO backbone output (128, h, w), not raw screenshots, so
        # YOLO must stay frozen or cached features would mismatch updated weights.
        # token_adapter + encoder + decoder + query_tokens are RL fine-tune targets.
        # token_adapter is always random init because Stage 1 has no Linear(128->64).
        self.backbone = VisualBackboneV3(grid_h=grid_h, grid_w=grid_w).to(device)
        # Record which yolo11n.pt the ultralytics loader actually resolved — the
        # default "yolo11n.pt" string can map to cwd / cache / site-packages and
        # the operator otherwise has no way to tell from logs.
        self.checkpoint_logger.success(
            "yolo_backbone",
            self.backbone.feature_extractor.yolo_source,
            f"Loaded YOLO11n backbone: {self.backbone.feature_extractor.yolo_source}",
        )
        self.backbone.freeze_feature_extractor()
        # BN must be eval before dummy forward, or zero-input batch stats with
        # momentum=0.1 will contaminate checkpoint-loaded running_mean/running_var.
        # torch.no_grad does not stop BN running_stats updates; module.training does.
        self.backbone.set_bn_eval()

        # Infer backbone spatial output size for replay-buffer shape checks.
        # YOLO11n stride=16, so 640x640 -> 40x40; dummy forward also handles other sizes.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *IMAGE_SIZE, device=device)
            dummy_feat = self.backbone.extract_backbone_features(dummy)
            _, feat_c, feat_h, feat_w = dummy_feat.shape
            assert feat_c == YOLO_FEATURE_CHANNELS, (
                f"YOLO feature channels ({feat_c}) ≠ YOLO_FEATURE_CHANNELS "
                f"({YOLO_FEATURE_CHANNELS}); update yolo_encoder_base."
            )
        self.backbone_feature_shape: tuple[int, int, int] = (feat_c, feat_h, feat_w)
        # Shape check for mixin._load_persistent_buffer; V3 uses cached feature shape.
        self.replay_state_shape: tuple[int, ...] = self.backbone_feature_shape

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

        # ── Stage 1 warm-start (all-or-nothing) ─────────────────────────────
        # encoder/decoder/query_tokens match Stage 1, so they can load directly
        # from Stage 1 checkpoint together with FQF online/target.
        # If any of backbone/fqf_network/fqf_target is missing, skip all and keep
        # random init. _stage1_loaded controls optimizer pretrained_prefixes.
        self._stage1_loaded = self._load_stage1_weights(STAGE1_CKPT_DIR)

        # ── Optimizer ───────────────────────────────────────────────────────
        # Only YOLO11n (feature_extractor) is frozen. token_adapter + encoder +
        # decoder + query_tokens + FQF head all update.
        #
        # Dynamic pretrained_prefixes:
        #   - Stage 1 loaded: encoder/decoder/query_tokens use LR_BACKBONE_PRETRAINED
        #     to protect pretrained attention patterns from noisy RL gradients.
        #   - Stage 1 missing: all backbone params are random init, prefixes empty.
        # token_adapter is always random init and uses the fresh LR group.
        #
        # build_fqf_optimizer filters requires_grad=False, so frozen YOLO is excluded.
        pretrained_prefixes: tuple[str, ...] = (
            ("encoder.", "decoder.", "query_tokens") if self._stage1_loaded else ()
        )
        self.optimizer = build_fqf_optimizer(
            self.backbone,
            self.q_network,
            config=FQFOptimizerConfig(lr_backbone_pretrained=LR_BACKBONE_PRETRAINED),
            pretrained_prefixes=pretrained_prefixes,
        )
        # Record each param group's base LR; warmup scales them dynamically by total_it.
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

        # ── replay buffer (disk-backed) ──
        VISUAL_V3_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        VISUAL_V3_REPLAY_PATH.mkdir(parents=True, exist_ok=True)
        self.replay_buffer = CategorizedReplayBuffer(
            max_size=BUFFER_CAPACITY,
            storage_mode="disk",
            save_dir=VISUAL_V3_REPLAY_PATH,
            win_threshold=MINESWEEPER_REWARD_CONFIG.replay_win_threshold,
            lose_threshold=MINESWEEPER_REWARD_CONFIG.replay_lose_threshold,
            invalid_threshold=MINESWEEPER_REWARD_CONFIG.replay_invalid_threshold,
            overflow_margin=BUFFER_OVERFLOW,
            alpha=PER_ALPHA,
            uniform_mix=PER_UNIFORM_MIX,
            priority_min=PRIORITY_MIN,
            priority_max=PRIORITY_MAX,
            priority_eps=PRIORITY_EPS,
            age_decay=AGE_DECAY,
            beta_start=PER_BETA_START,
            spread_decay=SPREAD_DECAY,
            quota_check_class="win",  # Minesweeper: win is the rare-event bottleneck class
        )

        # ── image preprocessing ──
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        # ── episode / step bookkeeping ──
        self.total_it = 0
        self.steps_since_resume = 0  # reset each startup; used for resume LR warmup, not saved
        # episode_count delegates to training_history.total_episodes as the single source of truth.
        self.train_every_n_steps = TRAIN_EVERY_N_STEPS
        self.pending_train_steps = 0
        self.n_step = N_STEP
        self.n_step_gamma = MINESWEEPER_REWARD_CONFIG.gamma
        self.n_step_buffer = deque()
        # recent_real_rewards moved to TrainingHistory._step_rewards. The mixin's
        # store_transition records rewards, and train_step reads avg_step_reward().

        # ── adaptive epsilon ──
        # Shared controller; stage1 uses the same class. Episode results live in
        # TrainingHistory, and the controller only consumes win_rate.
        self.epsilon_controller = AdaptiveEpsilonController()
        self.training_history = TrainingHistory()

        # Lazy-captured at first train_step: if the loaded TrainingHistory shows last
        # win_rate(window=100) > 0.5, we additionally gate training on the replay buffer
        # having all 4 classes filled to their 12.5% soft-floor quota. Fresh runs
        # (history empty → win_rate=0) and weak resumes skip this gate.
        # Captured once and frozen for the whole session.
        self._class_quota_gate_enabled: bool | None = None

        # ── blocked-action tracking (episode-scoped) ──
        self.blocked_actions: set[int] = set()

        # ── action-image logging (record full episode every N episodes) ──
        self.action_log_every_n_episodes = 50
        self._log_actions_this_episode = False

        # Session / hour archive directories.
        # Each startup creates training_<ts>/; each hour rolls to hour_NN_<ts>/.
        # Canonical *.pth files still live at VISUAL_V3_MODEL_PATH root for direct
        # try_load_model reads. Current io_log goes to current_archive_dir.
        # Directory management is shared with stage1 via SessionArchiveManager.
        self.archive = SessionArchiveManager(
            model_path=VISUAL_V3_MODEL_PATH,
            log_prefix="[V3]",
        )
        # SEED is a module-level constant from seed_everything; print it for later
        # comparison with hyperparameters.txt and console logs.
        print(f"[V3] SEED = {SEED}")

        # ── text log + TensorBoard ──
        # io_log: plain-text append file swapped to each new hour directory.
        # Banner comes from _build_io_log_banner.
        VISUAL_V3_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self._fqf_metric_docs_sentinel = VISUAL_V3_TENSORBOARD_DIR / ".fqf_metric_docs_written"
        self._io_log = RolloverTextLog(banner_factory=self._build_io_log_banner)
        self._io_log.swap_to(self.archive.current_archive_dir / "train_io_log.txt")
        # Mirror checkpoint messages to the io_log file as well.
        self.checkpoint_logger.attach_io_log(self._io_log)

        # Register hour rollover callback: swap io_log. Console messages are
        # printed by SessionArchiveManager.
        self.archive.register_on_rollover(self._on_archive_rollover)

        # TensorBoard log_dir remains VISUAL_V3_TENSORBOARD_DIR/<ts>, decoupled
        # from archive dirs so one TB instance can browse historical runs.
        # SummaryWriter is internal to TrainingLogger; all TB writes go through it.
        tb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log_dir = VISUAL_V3_TENSORBOARD_DIR / tb_timestamp
        print(f"[V3] TensorBoard: tensorboard --logdir {VISUAL_V3_TENSORBOARD_DIR}")
        print(f"[V3] Current run: {self.tensorboard_log_dir}")

        # CSV + TB dispatcher: write each episode summary to both with raw values
        # for AI consumption. CSV rolls over hourly. Schema aligns with stage1,
        # including eval rows written by Demo_test_Minesweeper.
        # Q_loss / q_mean are per-step TB metrics, not episode-level CSV fields.
        # Agent no longer keeps a separate SummaryWriter reference.
        csv_fields = [
            "timestamp", "episode",
            "reward_mean", "is_win", "invalid_click_rate",
            "win_rate_recent", "epsilon",
            "eval_avg_reward", "eval_win_rate", "eval_avg_steps",
            "eval_avg_invalid_rate", "eval_seconds_since_last_eval",
            "eval_duration_seconds",
        ]
        csv_path = self.archive.current_archive_dir / "training_log.csv"
        self.training_logger = TrainingLogger(
            csv_path=csv_path,
            csv_fields=csv_fields,
            tb_writer=SummaryWriter(log_dir=str(self.tensorboard_log_dir)),
        )
        self.archive.register_on_rollover(
            lambda new_dir: self.training_logger.swap_csv_to(new_dir / "training_log.csv")
        )
        print(f"[V3] CSV log: {csv_path}")
        self._write_fqf_metric_docs_once()

        # Remove replay buffer .pt files from older architectures or corrupt files.
        # Must run before try_load_model(), or _load_persistent_training_state may
        # load missing/mismatched entries and print many warnings. replay_state_shape
        # was set after dummy forward above.
        self._purge_stale_replay_files()
        self.try_load_model()

        # Hyperparameters dump runs AFTER try_load_model so the
        # [loaded_checkpoints] section reflects the final effective weight
        # source per area (Stage 1 warm-start, V3 own checkpoint, random
        # init, or YOLO11n weights actually resolved by ultralytics).
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
                    # backbone:freeze status(YOLO frozen / token_adapter+encoder+decoder
                    # trainable) + torchinfo layer summary. input_size is the real
                    # screenshot shape (1, 3, 640, 640). One YOLO forward is acceptable.
                    "model.backbone": (self.backbone, (1, 3, *IMAGE_SIZE)),
                    # Do not pass input_size for q_network. Like stage1, freeze status
                    # and param counts are enough; dict-returning forward is not useful for torchinfo.
                    "model.q_network": (self.q_network, None),
                },
                loaded_checkpoints=self.checkpoint_logger.loaded_sources,
            )
        except Exception as exc:
            print(f"[V3] hyperparameters dump failed: {exc}")

        self._init_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference_step = self.total_it

        # ── Dropout latch ───────────────────────────────────────────────────
        # Start at p=0. train_step checks win_rate(100) against the threshold; the
        # first crossing latches ON permanently, matching PER spread_decay. The
        # latch flag is not saved, so re-evaluate after loading training_history.
        self._dropout_latched = False
        self._set_backbone_dropout_p(0.0)
        if self.training_history.win_rate(window=100) > DROPOUT_LATCH_WR_THRESHOLD:
            self._latch_dropout_on(reason="resume: training_history win_rate already above threshold")

        self._set_runtime_modes()
        # atexit order is LIFO: close registered first runs last; save registered
        # last runs first. _close_training_logger closes CSV + SummaryWriter.
        atexit.register(self._close_training_logger)
        atexit.register(self._close_io_log)
        atexit.register(self.save_persistent)
        atexit.register(self._save_model)

    # ──────────────────────────── Stage 1 warm-start ──────────────────────

    def _write_fqf_metric_docs(self):
        """Write FQF distribution and Huber clipping metric notes to TensorBoard."""
        fpn_norm_entropy_doc = (
            "**`fpn/norm_entropy`** — FQF fraction-proposal distribution entropy, "
            "normalized to 0~1. It shows whether the fraction proposal network spreads "
            "probability across quantile fractions or collapses into a few fractions.\n\n"
            "| 數值區間 | 代表 | 越大越好嗎? |\n"
            "|---|---|---|\n"
            "| 接近 1.0 | fraction_probs 接近平均分配,quantile coverage 很廣 | 不一定; early 正常,但長期貼 1.0 代表 FPN 幾乎沒學到重點 |\n"
            "| 0.5 ~ 0.9 | 有分配偏好,但沒有 collapse | 通常合理 |\n"
            "| < 0.3 | 少數 fraction 主宰,quantile coverage 變窄 | 偏危險,可能 FPN collapse |\n\n"
            "方向: 不是單純越大越好。太小代表 collapse; 太接近 1 且長期不動代表太 uniform。"
        )
        self.training_logger.log_text("docs/fpn_norm_entropy", fpn_norm_entropy_doc, step=0)

        fpn_tau_std_doc = (
            "**`fpn/tau_std`** — tau_hats 在 quantile axis 上的平均標準差。"
            f"目前 `NUM_FQF_FRACTIONS={NUM_FQF_FRACTIONS}`,接近平均切分時大約是 0.30。"
            "它表示 quantile fractions 覆蓋範圍有多寬。\n\n"
            "| 數值區間 | 代表 | 越大越好嗎? |\n"
            "|---|---|---|\n"
            "| 0.25 ~ 0.32 | tau_hats 覆蓋大部分 0~1 quantile range | 通常合理 |\n"
            "| 0.15 ~ 0.25 | 覆蓋偏窄,但還沒完全 collapse | 需要搭配 norm_entropy 觀察 |\n"
            "| < 0.15 | tau_hats 擠在局部區域 | 偏危險,FPN 可能 collapse |\n\n"
            "方向: 太小不好; 大到接近平均切分通常健康。但不是無限越大越好,要和 norm_entropy 一起看。"
        )
        self.training_logger.log_text("docs/fpn_tau_std", fpn_tau_std_doc, step=0)

        frac_huber_clipped_doc = (
            "**`train/frac_huber_clipped`** — quantile TD-error 中,絕對值超過 "
            f"`FQF_HUBER_KAPPA={FQF_HUBER_KAPPA}` 的比例。超過 kappa 的部分會走 Huber linear branch,"
            "表示 batch 裡有多少 target/current quantile 差距很大。\n\n"
            "| 數值區間 | 代表 | 越大越好嗎? |\n"
            "|---|---|---|\n"
            "| < 0.05 | 大部分 TD-error 在 quadratic 區域,更新溫和 | 通常合理,收斂後常見 |\n"
            "| 0.05 ~ 0.20 | 有一些大誤差,仍可接受 | early training 或策略改變時正常 |\n"
            "| > 0.30 | 很多 quantile error 被 clipping | 偏危險,可能 target scale/Q scale 不穩或 reward shock |\n\n"
            "方向: 通常越小越穩,但不是永遠越小越好。訓練早期或剛 resume 有 spike 可以接受; 長期偏高才需要擔心。"
        )
        self.training_logger.log_text("docs/frac_huber_clipped", frac_huber_clipped_doc, step=0)

    def _write_fqf_metric_docs_once(self):
        """Write FQF metric docs once per TensorBoard root."""
        if self._fqf_metric_docs_sentinel.exists():
            return
        self._write_fqf_metric_docs()
        self.training_logger.flush()
        try:
            self._fqf_metric_docs_sentinel.touch(exist_ok=True)
        except OSError as exc:
            print(f"[V3] WARN: failed to write TensorBoard FQF metric docs sentinel: {exc}")

    def _load_stage1_weights(self, stage1_dir: Path) -> bool:
        """Load encoder/decoder/queries/FQF from Stage 1 checkpoint, all-or-nothing.

        If any of backbone.pth / fqf_network.pth / fqf_target.pth is missing,
        skip the whole batch, keep random init, and return False. Only when all
        three exist do we call load_state_dict and return True, meaning the
        optimizer should put encoder/decoder/query_tokens in the pretrained group.

        Key remapping (Stage 1 → V3):
            query_tokens                              → query_tokens                       (1:1)
            core.transformer.layers.{i}.attn.<...>    → encoder.layers.{i}.attn.<...>      (new-style, post-HierarchicalEncoder)
            core.transformer.layers.{i}.<...>         → encoder.layers.{i}.attn.<...>      (old-style, pre-refactor — auto-insert .attn)
            core.decoder.layers.{i}.<...>             → decoder.layers.{i}.<...>           (1:1)
            token_embed / output_head / position → dropped
        FQF online/target load 1:1.
        """
        backbone_pth = stage1_dir / "backbone.pth"
        qnet_pth     = stage1_dir / "fqf_network.pth"
        qtgt_pth     = stage1_dir / "fqf_target.pth"

        missing = [p for p in (backbone_pth, qnet_pth, qtgt_pth) if not p.exists()]
        if missing:
            for p in missing:
                self.checkpoint_logger.failure(
                    # Area name matches the area that V3 own try_load_model may
                    # later overwrite. failure() uses setdefault so a later
                    # success will overwrite this INIT_FROM_SCRATCH marker.
                    self._stage1_area_for_path(p),
                    f"Stage 1 checkpoint 不存在: {p} — all-or-nothing 放棄整批,使用 random init",
                )
            return False

        # backbone.pth: remap encoder/decoder/query_tokens.
        # Handles both old (raw nn.TransformerEncoder) and new (HierarchicalEncoder)
        # Stage 1 formats — the only difference is whether the `attn.` prefix is already
        # present after the layer index.
        sd = torch.load(backbone_pth, map_location="cpu", weights_only=False)
        remapped: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if key == "query_tokens":
                remapped["query_tokens"] = value
            elif key.startswith("core.transformer.layers."):
                tail = key[len("core.transformer.layers."):]
                idx_str, _, rest = tail.partition(".")
                if not rest:
                    continue
                if rest.startswith(("attn.", "proj.", "proj")):
                    # New-style: .attn / .proj prefix already there — strip outer `core.transformer.` only
                    remapped[f"encoder.layers.{idx_str}.{rest}"] = value
                else:
                    # Old-style: insert .attn to align with HierarchicalEncoderLayer wrapping
                    remapped[f"encoder.layers.{idx_str}.attn.{rest}"] = value
            elif key.startswith("core.decoder."):
                # core.decoder.X → decoder.X (1:1)
                remapped[key[len("core."):]] = value
            # Drop token_embed / output_head / position / others: architecture differs or unneeded.
        missing_keys, unexpected_keys = self.backbone.load_state_dict(remapped, strict=False)
        # Missing should include feature_extractor.* / token_adapter.* /
        # memory_position.* because V3 has them and Stage 1 does not. unexpected should be empty.
        self.checkpoint_logger.success(
            "backbone",
            backbone_pth,
            f"Stage 1 backbone loaded ({len(remapped)} tensors from {backbone_pth})",
        )
        if unexpected_keys:
            self.checkpoint_logger.warn(
                f"Stage 1 backbone unexpected keys (應為空): {unexpected_keys[:5]}"
                f"{'...' if len(unexpected_keys) > 5 else ''}"
            )

        # FQF online + target
        self.q_network.load_state_dict(
            torch.load(qnet_pth, map_location="cpu", weights_only=False)
        )
        self.checkpoint_logger.success(
            "q_network", qnet_pth, f"Stage 1 fqf_network loaded: {qnet_pth}"
        )
        self.q_target.load_state_dict(
            torch.load(qtgt_pth, map_location="cpu", weights_only=False)
        )
        self.checkpoint_logger.success(
            "q_target", qtgt_pth, f"Stage 1 fqf_target loaded: {qtgt_pth}"
        )

        return True

    @staticmethod
    def _stage1_area_for_path(p: Path) -> str:
        """Map a Stage 1 checkpoint file name back to its tracker area name."""
        return {
            "backbone.pth":    "backbone",
            "fqf_network.pth": "q_network",
            "fqf_target.pth":  "q_target",
        }.get(p.name, p.name)

    # ──────────────────────────── dropout latch ────────────────────────────

    def _set_backbone_dropout_p(self, p: float) -> None:
        """Set p for all nn.Dropout modules inside backbone.

        This covers encoder + decoder internal dropout. FQFQNetwork has no
        Dropout, only Linear + GELU. nn.Dropout.p is a Python attribute, not a
        Parameter/buffer, so it is not in state_dict and checkpoint load/save
        does not affect the latched value.
        """
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(p)

    def _latch_dropout_on(self, *, reason: str) -> None:
        self._dropout_latched = True
        self._set_backbone_dropout_p(DECODER_DROPOUT)
        # Non-checkpoint event but we want red attention. logger.warn prints red
        # without touching the loaded_sources tracker.
        self.checkpoint_logger.warn(
            f"dropout latch ON (p={DECODER_DROPOUT},"
            f" threshold={DROPOUT_LATCH_WR_THRESHOLD}) — {reason}"
        )

    # ──────────────────────────── runtime modes ──────────────────────────

    def _set_runtime_modes(self) -> None:
        self.backbone.train()  # Backbone defaults to train mode, enabling dropout.
        self.backbone.feature_extractor.eval()  # Frozen YOLO11n always eval.
        self.backbone.set_bn_eval()  # Keep YOLO BN eval and freeze running stats.
        self.backbone.token_adapter.train()  # Train feature projection from RL gradients.
        self.backbone.encoder.train()  # Train global information exchange for Q-values.
        self.backbone.decoder.train()
        self.q_network.train()
        self.q_target.eval()

    @property
    def episode_count_public(self) -> int:
        return self.training_history.total_episodes

    # episode_count delegates to training_history.total_episodes, incremented by
    # history.record() in log_episode_metrics(). on_episode_end no longer increments it.
    @property
    def episode_count(self) -> int:
        return self.training_history.total_episodes

    # epsilon is managed by the controller, but self.epsilon remains for compatibility:
    # 1) select_action / TB logs / console logs read self.epsilon.
    # 2) legacy load fallbacks may assign self.epsilon; setter routes it to controller.
    @property
    def epsilon(self) -> float:
        return self.epsilon_controller.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.epsilon_controller.epsilon = float(value)

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

        self.backbone.eval()
        self.backbone.set_bn_eval()
        self.q_network.eval()
        self.q_target.eval()
        screenshot_batch = state.unsqueeze(0).to(device)
        with torch.no_grad():
            features  = self.backbone.get_features(screenshot_batch)
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

    def _to_storage_state(self, state: torch.Tensor | None) -> torch.Tensor | None:
        """Override: run frozen YOLO backbone once and store its (128, h, w) output.

        Replaces the default (3, H, W) screenshot storage. The YOLO11n backbone is
        frozen via freeze_feature_extractor(), so for a given screenshot the cached
        feature tensor is stable across training steps and equivalent to recomputing
        it. This trades a one-time forward at storage time for skipping the YOLO
        forward on every replayed batch.

        Called via VisualAgentCommonMixin.store_transition for both state and
        next_state (next_state may be None at episode end, which we forward verbatim).
        """
        if state is None:
            return None
        screenshot = state.detach()
        # _set_runtime_modes puts feature_extractor in eval and freezes BN running
        # stats; we still wrap in no_grad to avoid building autograd graph through
        # this storage-time forward.
        self.backbone.feature_extractor.eval()
        self.backbone.set_bn_eval()
        with torch.no_grad():
            features = self.backbone.extract_backbone_features(
                screenshot.unsqueeze(0).to(self.device)
            )
        return features.squeeze(0).detach().cpu()

    # store_transition / _commit_n_step_transition / _flush_n_step_buffer come from
    # VisualAgentCommonMixin. Its store_transition calls _to_storage_state above, so
    # V3 stores (128, h, w) features.

    # ──────────────────────────── training step ────────────────────────

    def train_step(self):
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

        self.total_it += 1
        self.steps_since_resume += 1
        self._apply_lr_warmup()

        # Dropout latch: aligned with stage1 spread_decay latch. One-way switch:
        # crossing threshold turns it on permanently. training_history.win_rate is
        # a 100-window rolling value recorded only on episode end, so repeated
        # train_step calls in one episode see stable results.
        if not self._dropout_latched:
            if self.training_history.win_rate(window=100) > DROPOUT_LATCH_WR_THRESHOLD:
                self._latch_dropout_on(
                    reason=f"train_step total_it={self.total_it}: win_rate(100) crossed threshold"
                )

        # Spread-decay latch: FQF quantile spread master switch for priority modifier.
        # Same threshold and monotone behavior as dropout: once ON, never OFF. State
        # lives on replay_buffer and is not saved; first train_step after resume re-evaluates.
        if not self.replay_buffer.enable_spread_decay:
            if self.training_history.win_rate(window=100) > SPREAD_DECAY_LATCH_WR_THRESHOLD:
                self.replay_buffer.enable_spread_decay = True

        # Roll over every wall-clock hour; io_log follows the new archive dir via
        # SessionArchiveManager callbacks.
        self.archive.maybe_rollover()
        # PER beta annealing: linearly move from START to END over PER_BETA_EP
        # episodes. Early weak correction is stable; later beta=1 fully corrects
        # priority-sampling bias.
        per_beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * min(
            self.episode_count / PER_BETA_EP, 1.0
        )
        state, action, next_state, reward, done, sample_indices, is_weights, discounts, n_steps = (
            self.replay_buffer.sample(
                BATCH_SIZE,
                beta=per_beta,
                device=device,
                include_extra=True,
            )
        )
        batch_size = state.size(0)
        self._set_runtime_modes()

        # Target branch (no gradients).
        # state/next_state are cached YOLO backbone features (B, 128, h, w), not screenshots.
        # YOLO11n forward already happened in store_transition; run only trainable parts here.
        # token_adapter + encoder + decoder only.
        #
        # Target branch must set token_adapter/encoder/decoder to eval, otherwise
        # dropout fires during target Q computation and adds TD target noise. no_grad
        # blocks backward only, not dropout. q_network is also set eval here so
        # double-Q argmax logic matches q_target; restore train mode afterward.
        self.backbone.token_adapter.eval()
        self.backbone.encoder.eval()
        self.backbone.decoder.eval()
        self.q_network.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(USE_AMP and device.type == "cuda")):
                next_features  = self.backbone.get_features_from_cached(next_state)
                next_online    = self.q_network(next_features)
                next_online_q_flat = next_online["q_values"].view(batch_size, -1)
                next_best_flat = next_online_q_flat.argmax(dim=1)
                next_target    = self.q_target(next_features)
                next_target_quantiles = next_target["quantiles"][
                    torch.arange(batch_size, device=device), next_best_flat
                ]
                target_quantiles = reward + (1 - done) * discounts * next_target_quantiles
        # Restore training mode for current branch dropout / BN behavior.
        self._set_runtime_modes()

        # Current branch: gradients flow through token_adapter + encoder + decoder + FQF.
        # YOLO11n is frozen and its forward already ran in store_transition.
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(USE_AMP and device.type == "cuda")):
            features  = self.backbone.get_features_from_cached(state)
            q_output  = self.q_network(features)
            q_2d           = q_output["q_values"]
            q_quantiles    = q_output["quantiles"]
            tau_hats       = q_output["tau_hats"]
            fraction_probs = q_output["fraction_probs"]

            action_flat = action.long()
            row_idx = action_flat // self.grid_w
            col_idx = action_flat %  self.grid_w
            q_taken = q_2d[torch.arange(batch_size, device=device), row_idx, col_idx].unsqueeze(1)
            chosen_quantiles = q_quantiles[torch.arange(batch_size, device=device), action_flat]

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
            # Per-sample quantile spread for the chosen action feeds buffer
            # spread_decay. Write it even when latch is OFF, so latch ON has no
            # cold-start; entries already carry spread from previous samples.
            chosen_q_spread = chosen_quantiles.std(dim=1).detach()
            # Keep both FPN stats as tensors; batch .item() under DIAGNOSTIC gate
            # to avoid two GPU->CPU syncs per step.
            fpn_norm_entropy_t = entropy.mean() / math.log(NUM_FQF_FRACTIONS)
            fpn_tau_std_t = tau_hats.std(dim=1).mean()

        # Per-step NaN check (Q1=b): raise immediately on NaN/Inf; do not wait for the next save.
        _check_finite("train_step.loss", loss)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Pre-clip gradient norms (diagnostic).
        backbone_pre = self._module_grad_norm(self.backbone)
        head_pre     = self._module_grad_norm(self.q_network)

        # Per-layer grad/weight norms must be logged BEFORE clip_grad_norm_.
        # Otherwise grads are clipped in-place and the exploding layer is hidden.
        if self.total_it % HISTOGRAM_EVERY == 0:
            self._log_backbone_weight_norms(self.total_it)

        params_to_clip = (
            list(self.backbone.parameters())
            + list(self.q_network.parameters())
        )
        grad_norm_total = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)
        grad_norm_total_value = float(grad_norm_total)
        grad_clip_threshold = float(GRAD_CLIP_NORM)
        grad_clip_scale = min(1.0, grad_clip_threshold / (grad_norm_total_value + 1e-12))
        grad_clip_percent = 1.0 - grad_clip_scale
        grad_clip_excess_norm = max(0.0, grad_norm_total_value - grad_clip_threshold)
        grad_clip_excess_ratio = grad_clip_excess_norm / (grad_clip_threshold + 1e-12)

        backbone_post = self._module_grad_norm(self.backbone)
        head_post     = self._module_grad_norm(self.q_network)
        grad_post_total = (backbone_post ** 2 + head_post ** 2) ** 0.5

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Match stage1: update priorities only after backward + clip + optimizer.step
        # all succeed. If backward fails, leave old priorities intact. Always write
        # quantile_spreads so spread_decay latch can use accumulated real spread.
        self.replay_buffer.update_priorities(
            sample_indices,
            td_error.squeeze(-1).cpu().numpy(),
            quantile_spreads=chosen_q_spread.cpu().numpy(),
        )

        if self.total_it % TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
            self.q_target.eval()

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

        # ── Diagnostic / logging — only every DIAGNOSTIC_LOG_EVERY steps ──
        # Match stage1 throttle: downsample io_log / TB scalar / no_grad stats and
        # return None on other steps. This batches seven scalar syncs into one
        # `.cpu().tolist()`. Other gates are independent but multiples of
        # DIAGNOSTIC_LOG_EVERY, so TB write order stays valid.
        if self.total_it % DIAGNOSTIC_LOG_EVERY != 0:
            return None

        # Batched stats: one .cpu().tolist() instead of seven syncs. Order must
        # match unpack below. frac_clipped is already a Python float from the loss helper.
        with torch.no_grad():
            _stats = torch.stack([
                loss,
                q_taken.mean(),
                td_error.mean(),
                td_error.max(),
                target_quantiles.float().mean(),
                fpn_norm_entropy_t,
                fpn_tau_std_t,
            ]).cpu().tolist()
            (
                loss_value, q_mean, td_error_mean, td_error_max,
                target_q_mean, fpn_norm_entropy, fpn_tau_std,
            ) = _stats

            # Raw reward rolling mean from training_history._step_rewards, maintained
            # by mixin store_transition. Python deque, no GPU sync.
            real_reward_mean = self.training_history.avg_step_reward()

            # Top-5 actions. torch.topk -> tolist sync is small enough to keep.
            q0 = q_2d[0].view(-1)
            top_vals, top_idx = torch.topk(q0, k=min(5, self.num_actions))
            top_actions = [
                (int(idx), *self.action_to_grid(int(idx)), float(val))
                for val, idx in zip(top_vals.tolist(), top_idx.tolist())
            ]

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
            f"  Q_loss={loss_value:.6f} | q_mean={q_mean:.6f} | epsilon={self.epsilon:.4f}\n"
            f"  td_error_mean={td_error_mean:.6f} | frac_clipped={frac_clipped:.3f}\n"
            f"  fpn_norm_entropy={fpn_norm_entropy:.4f} | fpn_tau_std={fpn_tau_std:.4f}\n"
            f"  grad_total_pre={grad_norm_total_value:.6f} | grad_total_post={grad_post_total:.6f} | "
            f"clip_scale={grad_clip_scale:.6f} clip_percent={grad_clip_percent:.2%}\n"
            f"  backbone_pre={backbone_pre:.6f} head_pre={head_pre:.6f}\n"
            f"{weight_delta_line}"
            f"---\n"
        )
        self._io_log.flush()

        # Per-step diagnostics use training_logger.log(..., csv=False): TB only,
        # no CSV row. It calls SummaryWriter.add_scalar while keeping the same
        # interface as episode summaries.
        step = self.total_it
        self.training_logger.log("train/Q_loss",            loss_value,                                    step=step, csv=False)
        self.training_logger.log("train/q_mean",            q_mean,                                        step=step, csv=False)
        self.training_logger.log("train/real_reward_mean",  real_reward_mean,                              step=step, csv=False)
        self.training_logger.log("train/epsilon",           self.epsilon,                                  step=step, csv=False)
        self.training_logger.log("train/frac_huber_clipped", frac_clipped,                                 step=step, csv=False)
        self.training_logger.log("fpn/norm_entropy",        fpn_norm_entropy,                              step=step, csv=False)
        self.training_logger.log("fpn/tau_std",             fpn_tau_std,                                   step=step, csv=False)
        self.training_logger.log("train/td_error_mean",     td_error_mean,                                 step=step, csv=False)
        self.training_logger.log("train/td_error_max",      td_error_max,                                  step=step, csv=False)
        self.training_logger.log("train/target_q_mean",     target_q_mean,                                 step=step, csv=False)
        self.training_logger.log("grad/total_norm",         grad_norm_total_value,                         step=step, csv=False)
        self.training_logger.log("grad/post_total_norm",    grad_post_total,                               step=step, csv=False)
        self.training_logger.log("grad/clip_percent",       grad_clip_percent,                             step=step, csv=False)
        self.training_logger.log("grad/clip_excess_norm",   grad_clip_excess_norm,                         step=step, csv=False)
        self.training_logger.log("grad/clip_excess_ratio",  grad_clip_excess_ratio,                        step=step, csv=False)
        self.training_logger.log("grad_pre/backbone",       backbone_pre,                                  step=step, csv=False)
        self.training_logger.log("grad_pre/head",           head_pre,                                      step=step, csv=False)
        self.training_logger.log("grad_post/backbone",      backbone_post,                                 step=step, csv=False)
        self.training_logger.log("grad_post/head",          head_post,                                     step=step, csv=False)
        if weight_distance_log is not None:
            self.training_logger.log("weights/delta_from_init",        weight_distance_log["from_init"],        step=step, csv=False)
            self.training_logger.log("weights/delta_from_prev_window", weight_distance_log["from_prev_window"], step=step, csv=False)

        if self.scaler.is_enabled():
            self.training_logger.log("train/scaler_scale", self.scaler.get_scale(), step=step, csv=False)

        # LR scalars: one per param group under lr/ in TB. build_fqf_optimizer adds
        # a "name" key to each group; fallback to index for old checkpoints. During
        # warmup group["lr"] is already the effective LR, so TB shows ramp + ratios.
        for idx, group in enumerate(self.optimizer.param_groups):
            tag = group.get("name") or f"group_{idx}"
            self.training_logger.log(f"lr/{tag}", group["lr"], step=step, csv=False)

        self.training_logger.flush()
        # Use batched scalar loss_value instead of loss.item(); it was already
        # synced above. Demo_test_Minesweeper checks `if loss_info`, so throttled
        # return None silently skips caller logging.
        return {"Q_loss": loss_value, "q_mean": q_mean}

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
        next_episode_idx = self.training_history.total_episodes + 1
        self._log_actions_this_episode = (
            self.action_log_every_n_episodes > 0
            and next_episode_idx % self.action_log_every_n_episodes == 0
        )
        if self._log_actions_this_episode:
            print(f"[V3] Action-image logging enabled for episode {next_episode_idx}")

    def on_episode_end(self) -> None:
        self._flush_n_step_buffer()
        # episode_count delegates to history and was incremented by history.record().
        # epsilon was already updated by controller.update in log_episode_metrics().
        self.training_logger.log("episode/epsilon", self.epsilon, step=self.episode_count, csv=False)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[V3] Periodic save at episode {self.episode_count} | epsilon={self.epsilon:.4f}")
            self.save_persistent()

    def log_episode_metrics(
        self,
        win: bool,
        invalid_click_rate: float,
        reward_mean: float,
    ) -> None:
        # 1) record result to history, 2) read rolling win rate, 3) feed win rate
        # to controller to update epsilon. V3 currently wires invalid_rate only;
        # avg_reward / reward_per_step stay 0 unless caller adds totals.
        # Absolute click counts are derivable from invalid_rate x steps, so TB logs
        # only `episode/invalid_click_rate`.
        self.training_history.record(win=win, invalid_rate=invalid_click_rate)
        ep_idx = self.training_history.total_episodes
        rolling_wr = self.training_history.win_rate(window=100)
        next_eps = self.epsilon_controller.update(rolling_wr)

        # Episode summary metrics go through TrainingLogger for TB + CSV. CSV row
        # is committed at the end. `episode/win` is omitted from TB because binary
        # noise is high; `is_win` remains in CSV for arbitrary rolling windows.
        self.training_logger.log("episode/reward_mean",        float(reward_mean),        step=ep_idx, csv_col="reward_mean")
        self.training_logger.log("episode/invalid_click_rate", float(invalid_click_rate), step=ep_idx, csv_col="invalid_click_rate")
        self.training_logger.log("episode/win_rate_recent",    rolling_wr,                step=ep_idx, csv_col="win_rate_recent")
        self.training_logger.log("is_win",                     int(bool(win)),            step=ep_idx, tb=False)
        self.training_logger.log("epsilon",                    next_eps,                  step=ep_idx, tb=False)  # TB side is written by on_episode_end.
        self.training_logger.log("timestamp",                  datetime.datetime.now().isoformat(), step=ep_idx, tb=False)
        self.training_logger.log("episode",                    ep_idx,                    step=ep_idx, tb=False)

        # Replay buffer composition is logged at episode end so fill curves and
        # bucket ratios are visible from episode 1. TB only via csv=False.
        for bucket_name, count in self.replay_buffer.bucket_sizes().items():
            self.training_logger.log(f"buffer/bucket_{bucket_name}", count, step=ep_idx, csv=False)
        self.training_logger.log("buffer/total_size", self.replay_buffer.size(), step=ep_idx, csv=False)

        self.training_logger.commit_csv_row()
        self.training_logger.flush()

        status = "WIN " if win else "LOSE"
        print(
            f"[V3] Ep {ep_idx}: {status} | "
            f"invalid={invalid_click_rate:.1%} | reward={reward_mean:.3f} | "
            f"win_rate(recent)={rolling_wr:.1%} | "
            f"eps(next)={next_eps:.4f}"
        )

    # ──────────────────────────── checkpoints ──────────────────────────

    def log_eval_metrics(
        self,
        *,
        avg_reward: float,
        win_rate: float,
        avg_steps: float,
        avg_invalid_rate: float,
        seconds_since_last_eval: float,
        duration_seconds: float,
    ) -> None:
        """Record one fixed-policy evaluation summary."""
        log_eval_metrics_common(
            self.training_logger,
            episode=self.episode_count,
            avg_reward=avg_reward,
            win_rate=win_rate,
            avg_steps=avg_steps,
            avg_invalid_rate=avg_invalid_rate,
            seconds_since_last_eval=seconds_since_last_eval,
            duration_seconds=duration_seconds,
            console_prefix="V3 EVAL",
        )

    def _scan_state_dict_finite(self, sd_label, state_dict):
        """Scan all floating tensors in a state_dict and return [(label, msg)].

        Does not raise; caller decides whether to raise or write a .crash file.
        Aligned with the stage1 method.
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
        """Scan optimizer state_dict for NaN/Inf and exp_avg_sq < 0.

        Adam's second moment should always be >= 0. Negative values indicate
        bit-level corruption and make sqrt(neg)=NaN after load. Returns
        [(label, msg)] without raising.
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

    def _save_model(self) -> None:
        VISUAL_V3_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        backbone_sd = self.backbone.state_dict()
        qnet_sd     = self.q_network.state_dict()
        qtarget_sd  = self.q_target.state_dict()
        opt_sd      = self.optimizer.state_dict()

        # Save-time NaN/Inf probe(Q1=b + Q2=b)
        # Prevent atexit from overwriting a good checkpoint after a NaN crash.
        # If any state_dict contains NaN/Inf, skip canonical *.pth writes, write
        # *.crash_stepN.pth evidence, then raise. Next startup can still load the
        # uncorrupted canonical checkpoint. Also scan optimizer state: a negative
        # Adam v value will trigger NaN weights immediately after load.
        bad = []
        bad.extend(self._scan_state_dict_finite("backbone", backbone_sd))
        bad.extend(self._scan_state_dict_finite("q_network", qnet_sd))
        bad.extend(self._scan_state_dict_finite("q_target", qtarget_sd))
        bad.extend(self._scan_optimizer_state("optimizer", opt_sd))

        if bad:
            suffix = f".crash_step{self.total_it}.pth"
            print(f"[NaN-probe] _save_model: REFUSING to overwrite canonical checkpoints at step {self.total_it}")
            print(f"[NaN-probe] non-finite tensors detected:")
            for key, msg in bad:
                print(f"[NaN-probe]   {key}: {msg}")
            torch.save(backbone_sd, VISUAL_V3_MODEL_PATH / f"backbone{suffix}")
            torch.save(qnet_sd,     VISUAL_V3_MODEL_PATH / f"fqf_network{suffix}")
            torch.save(qtarget_sd,  VISUAL_V3_MODEL_PATH / f"fqf_target{suffix}")
            torch.save({"optimizer": opt_sd},
                       VISUAL_V3_MODEL_PATH / f"optimizer_state{suffix}")
            print(f"[NaN-probe] wrote *{suffix} files for offline analysis;"
                  f" canonical *.pth left untouched (last good state preserved)")
            # Stop training. The atexit hook will call _save_model again, and
            # the probe raises again before the canonical checkpoint is overwritten.
            raise RuntimeError(
                f"[NaN-probe] _save_model: non-finite tensors at step {self.total_it}; "
                f"canonical checkpoints preserved, see *{suffix} for forensics"
            )

        torch.save(backbone_sd, VISUAL_V3_MODEL_PATH / "backbone.pth")
        torch.save(qnet_sd,     VISUAL_V3_MODEL_PATH / "fqf_network.pth")
        torch.save(qtarget_sd,  VISUAL_V3_MODEL_PATH / "fqf_target.pth")
        self._save_optimizer_state()

        # Also write checkpoint snapshots into the current hour archive. Canonical
        # checkpoint writes above remain the source used by try_load_model.
        try:
            archive = self.archive.current_archive_dir
            archive.mkdir(parents=True, exist_ok=True)
            torch.save(backbone_sd, archive / "backbone.pth")
            torch.save(qnet_sd,     archive / "fqf_network.pth")
            torch.save(qtarget_sd,  archive / "fqf_target.pth")
        except Exception as exc:
            print(f"[V3] WARN: archive snapshot write failed: {exc}")

    def _log_checkpoint_message(self, message: str, *, warning: bool = False) -> None:
        """Thin wrapper used by VisualAgentCommonMixin fallbacks.

        The mixin checks ``hasattr(self, "_log_checkpoint_message")`` for old
        v1/v2 agents that don't have a CheckpointLogger; this delegator keeps
        that contract working while routing colored output through the shared
        logger. Direct checkpoint sites in this file call ``self.checkpoint_logger``
        methods directly because they need ``success`` / ``failure`` semantics
        with area names, which this two-state wrapper cannot express.
        """
        if warning:
            self.checkpoint_logger.warn(message)
        else:
            self.checkpoint_logger.info(message)

    def try_load_model(self) -> None:
        # V3 own checkpoints. These run AFTER Stage 1 warm-start; a success here
        # overwrites the Stage 1 source recorded in loaded_sources, so the
        # tracker shows the actual weights in the model after this method ends.
        bb_path = VISUAL_V3_MODEL_PATH / "backbone.pth"
        if bb_path.exists():
            try:
                self.backbone.load_state_dict(torch.load(bb_path, map_location=device, weights_only=True))
                self.checkpoint_logger.success(
                    "backbone", bb_path, f"Loaded backbone: {bb_path}"
                )
            except Exception as exc:
                self.checkpoint_logger.failure(
                    "backbone",
                    f"MISSING/FAILED backbone checkpoint: {bb_path}"
                    f" | using initialized backbone | error={exc}",
                )
        else:
            self.checkpoint_logger.failure(
                "backbone",
                f"MISSING backbone checkpoint: {bb_path} | using initialized backbone",
            )

        q_path = VISUAL_V3_MODEL_PATH / "fqf_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device, weights_only=True))
                self.checkpoint_logger.success(
                    "q_network", q_path, f"Loaded FQF-Network: {q_path}"
                )
            except Exception as exc:
                self.checkpoint_logger.failure(
                    "q_network",
                    f"MISSING/FAILED FQF-Network checkpoint: {q_path}"
                    f" | using initialized q_network | error={exc}",
                )
        else:
            self.checkpoint_logger.failure(
                "q_network",
                f"MISSING FQF-Network checkpoint: {q_path} | using initialized q_network",
            )

        qt_path = VISUAL_V3_MODEL_PATH / "fqf_target.pth"
        if qt_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(qt_path, map_location=device, weights_only=True))
                self.checkpoint_logger.success(
                    "q_target", qt_path, f"Loaded FQF-Target: {qt_path}"
                )
            except Exception as exc:
                self.checkpoint_logger.failure(
                    "q_target",
                    f"MISSING/FAILED FQF-Target checkpoint: {qt_path}"
                    f" | copying q_network if available | error={exc}",
                )
        elif q_path.exists():
            # No on-disk q_target but q_network exists. Copy weights and mark
            # the tracker so the operator can see the special case.
            self.q_target.load_state_dict(self.q_network.state_dict())
            self.checkpoint_logger.mark_special(
                "q_target",
                "<copied from q_network>",
                f"MISSING FQF-Target checkpoint: {qt_path} | copied q_network weights",
            )
        else:
            self.checkpoint_logger.failure(
                "q_target",
                f"MISSING FQF-Target checkpoint: {qt_path} | using initialized q_target",
            )

        if not self._optimizer_state_path().exists():
            self.checkpoint_logger.failure(
                "optimizer_state",
                f"MISSING optimizer checkpoint: {self._optimizer_state_path()}"
                f" | total_it stays at {self.total_it}",
            )
        self._load_optimizer_state()
        if not self._training_state_path().exists():
            self.checkpoint_logger.failure(
                "replay_buffer",
                f"MISSING replay/training checkpoint: {self._training_state_path()}"
                f" | replay buffer starts empty",
            )
        self._load_persistent_training_state()

    # _load_persistent_buffer comes from VisualAgentCommonMixin via rebinding below.
    # It validates disk .pt shape using self.replay_state_shape, so old screenshot
    # entries (3, 640, 640) are skipped automatically.

    # ──────────────────────────── lr warmup ────────────────────────────

    def _apply_lr_warmup(self) -> None:
        """Linear LR warmup using the stricter of two warmups.

        - init warmup: progress by total_it; applies on first training from scratch.
        - resume warmup: progress by steps_since_resume; applies on every startup.
        Final factor is min(init_factor, resume_factor), so restarts still ramp LR
        from LR_WARMUP_START_FACTOR * base_lr to base_lr even when total_it is large.
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
            self.training_logger.log(f"weight_norm/{tag_prefix}", weight_norm, step=global_step, csv=False)
        grad_norm = self._tensor_norm(param.grad)
        if grad_norm is not None:
            self.training_logger.log(f"grad_norm/{tag_prefix}", grad_norm, step=global_step, csv=False)

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

    # ──────────────────────────── archive / hour rollover ─────────────
    # Directory management lives in SessionArchiveManager, shared with stage1.
    # current_archive_dir is a read-only property delegate for existing callers;
    # _build_io_log_banner / _on_archive_rollover are manager callbacks.

    @property
    def current_archive_dir(self) -> Path:
        return self.archive.current_archive_dir

    def _build_io_log_banner(self, path: Path) -> list[str]:
        """RolloverTextLog banner with session/hour start time, encoder dims, and path."""
        return [
            f"Session started: {self.archive.session_start.isoformat()}",
            f"Hour {self.archive.hour_index:02d} started: {datetime.datetime.now().isoformat()}",
            f"Encoder dims: {ENCODER_DIMS}",
            f"Path: {path}",
        ]

    def _on_archive_rollover(self, new_dir: Path) -> None:
        """SessionArchiveManager rollover callback: swap io_log."""
        now = datetime.datetime.now()
        try:
            self._io_log.write(f"\n--- hour rollover at {now.isoformat()} ---\n")
            self._io_log.flush()
        except Exception:
            pass
        self._io_log.swap_to(new_dir / "train_io_log.txt")

    # ──────────────────────────── cleanup ──────────────────────────────

    def _close_io_log(self) -> None:
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def _close_training_logger(self) -> None:
        """Atexit hook: close CSV file handle and SummaryWriter.

        SummaryWriter is only referenced inside self.training_logger, so one
        training_logger.close() call closes both CSV and TB writer.
        """
        logger = getattr(self, "training_logger", None)
        if logger is not None and not logger.closed:
            logger.close()

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
# Do not rebind _log_param_weight_and_grad_norm. V3 has its own version using
# self.training_logger.log(); the mixin version still uses old self.tb_writer and
# would raise AttributeError.
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
