"""visual_discrete_agent_v3.py — Stage 2 Agent（screenshot → FQF Q-network）。

Pipeline（dim/層數由 yolo_encoder_base 與下方 DECODER_* 常數決定）:
    screenshot (3, 640, 640)
        ↓ YOLOEncoderBase 的兩半:
        ↓   [frozen] YOLO11n backbone → (128, 40, 40)
        ↓   ───────────── REPLAY-BUFFER 在此切點儲存 (128, 40, 40) fp32 特徵 ─────────────
        ↓   [trainable] token_adapter + 2D sinusoidal pos enc → (1600, encoder_dims[0])
        ↓   [trainable] HierarchicalEncoder → (1600, final_dim)
    encoded memory (B, 1600, final_dim)
        ↓ TransformerDecoder × DECODER_NUM_LAYERS（pre-LN, cross-attn，GRID_H*GRID_W learned query tokens）
    decoded features (B, GRID_H*GRID_W, final_dim)
        ↓ FQFQNetwork (d_model=final_dim, num_fractions=NUM_FQF_FRACTIONS)
    Q-values (B, GRID_H, GRID_W) → masked argmax → action

架構說明：
    • VisualBackboneV3 繼承 YOLOEncoderBase（與 YOLOGridStatePredictor 共用）
    • 從 YOLOGridStatePredictor checkpoint 載入 feature_extractor + token_adapter + encoder 權重
    • YOLO11n backbone 凍結(BN eval mode);token_adapter + HierarchicalEncoder 解凍可訓練
    • Replay buffer 改存 YOLO backbone 輸出 (128, 40, 40) 而不是原始截圖(3, 640, 640):
        - 每筆 transition 從 ~1.17 MB(uint8 截圖)降到 ~0.78 MB(fp32 特徵)
        - 訓練時跳過 YOLO11n forward,只跑 token_adapter + encoder + decoder
        - 因為 YOLO 凍結,cached features 對訓練等同每次重算
    • 可訓練部分：token_adapter + encoder + decoder + query tokens + FQF head
"""

from __future__ import annotations

import atexit
import datetime
import hashlib
import math
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from transformer_discrete_agent import FQF_ENTROPY_COEF, NUM_FQF_FRACTIONS, _quantile_huber_loss
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from model_structure.transformer_shared import FQFQNetwork
from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer
import model_structure.CategorizedReplayBuffer as _crb_module
from model_structure.visual_agent_common import VisualAgentCommonMixin
from model_structure.yolo_encoder_base import (
    YOLOEncoderBase,
    YOLO_FEATURE_CHANNELS,
    DEFAULT_ENCODER_DIMS,
    DEFAULT_ENCODER_FF_MULT,
)
from model_structure.optimizer_factory import build_fqf_optimizer
from model_structure.adaptive_epsilon import AdaptiveEpsilonController
from model_structure.history import TrainingHistory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_RED = "\033[91;1m"
CHECKPOINT_RESET = "\033[0m"

if device.type == "cuda":
    # PyTorch 2.1 may route Transformer attention through fast SDP kernels.
    # On some CUDA 11.8 / GPU combinations those kernels can raise
    # "illegal instruction"; the math backend is slower but more stable.
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception as exc:
        print(f"[V3] Failed to configure CUDA SDP backends: {exc}")

# ── debug logger (寫到檔案，CMD 刷掉也能看) ──────────────────────────
import logging as _logging
_dbg_log_path = Path("./models/visual_transformer_v3_6x6/cuda_debug.log")
_dbg_log_path.parent.mkdir(parents=True, exist_ok=True)
_dbg_logger = _logging.getLogger("cuda_dbg")
_dbg_logger.setLevel(_logging.DEBUG)
_dbg_logger.propagate = False  # 不往 root logger 傳，避免 CMD 也被印
if not _dbg_logger.handlers:
    _fh = _logging.FileHandler(_dbg_log_path, mode="a", encoding="utf-8")
    _fh.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
    _dbg_logger.addHandler(_fh)

# 把 CategorizedReplayBuffer 的 [DBG sample] / !!BAD ACTIONS!! 也導到同一個檔
# （取代原本 hard-coded 的 print 與 open(...) 寫法）。
_crb_module.DEBUG_CUDA_SAMPLE_LOG_PATH = _dbg_log_path

def _dbg(msg: str) -> None:
    """log+flush first, then sync — last entry on disk = op about to be sync'd."""
    _dbg_logger.debug(msg)
    for _h in _dbg_logger.handlers:
        try: _h.flush()
        except Exception: pass
    if device.type == "cuda":
        torch.cuda.synchronize()

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
    # write an enter marker first (and flush) so we can see exactly which tensor
    # was about to be checked when sync raised.
    _dbg_logger.debug(f"[TENSOR {name}] entering")
    for _h in _dbg_logger.handlers:
        try: _h.flush()
        except Exception: pass
    try:
        if t is None:
            _dbg_logger.debug(f"[TENSOR {name}] is None"); return
        if not torch.is_tensor(t):
            _dbg_logger.debug(f"[TENSOR {name}] type={type(t).__name__}"); return
        # sync so any pending error surfaces here, not later
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
        for _h in _dbg_logger.handlers:
            try: _h.flush()
            except Exception: pass

# ── paths ────────────────────────────────────────────────────────────
# Disk layout（v3,2026-05-23 起):
#   * MODEL_PATH        → D 槽（HDD)。weights / training_state.pth / tensorboard / action_logs。
#                         這些檔案只在 checkpoint 或事件時寫,不在訓練熱迴路。
#   * REPLAY_BASE       → C 槽（SSD)。disk-backed replay buffer 每 step 都讀寫,
#                         D 槽是 HDD,小檔 random IO 太慢,所以拆到 C 槽。
#                         REPLAY_BASE 底下保留原本的 replay_buffer/ 與 replay_buffer_save/
#                         兩個子目錄結構,以避免動到 CategorizedReplayBuffer 與
#                         visual_agent_common.save_persistent / export_top_k 的呼叫慣例。
# 注意:training_state.pth 仍在 MODEL_PATH(D 槽);它內含 replay_buffer_save 下的 .pt 絕對
# 路徑(C 槽),pathlib 跨槽絕對路徑沒問題。如果手動清 C 槽快取,記得把對應的
# training_state.pth 也一併處理,免得 _load_persistent_buffer 拿到失效路徑。
YOLO_PREDICTOR_PATH = Path("./models/yolo_grid_predictor/best.pth")

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
VISUAL_BATCH_SIZE = 32
MINIMUM_DATA_SIZE = 2000 # below this amount, won't start training

# ── Encoder dims（與 YOLOGridStatePredictor 共用 DEFAULT_ENCODER_DIMS）──
# 形狀由 model_structure.yolo_encoder_base.DEFAULT_ENCODER_FINAL_DIM /
# DEFAULT_ENCODER_TOTAL_LAYERS 決定;改那邊兩個常數 v3 與 predictor 會同步更新。
ENCODER_DIMS = DEFAULT_ENCODER_DIMS

# ── Decoder spec ─────────────────────────────────────────────────────
DECODER_D_MODEL    = ENCODER_DIMS[-1]       # = ENCODER_FINAL_DIM
DECODER_NHEAD      = 4
DECODER_NUM_LAYERS = 4
# Transformer convention: FFN width = 4 × d_model — keeps capacity ratio constant
# when d_model changes. Shared with encoder side (DEFAULT_ENCODER_FF_MULT).
DECODER_FF_DIM     = DECODER_D_MODEL * DEFAULT_ENCODER_FF_MULT
DECODER_DROPOUT    = 0.1

# ── training hyper-params ────────────────────────────────────────────
VISUAL_N_STEP = 1
VISUAL_GRAD_CLIP_NORM = 10.0   # 與 TransformerDiscreteAgent 對齊；token_adapter + encoder 加入可訓練後仍維持 10.0
TRAIN_EVERY_N_STEPS = 1
TARGET_UPDATE_FREQ = 200
SAVE_EVERY_N_EPISODES = 100
VISUAL_HISTOGRAM_EVERY = 20
WEIGHT_DISTANCE_LOG_EVERY = 100
USE_AMP = False

# ── learning rates ───────────────────────────────────────────────────
# LR / weight_decay 由 model_structure.optimizer_factory.FQFOptimizerConfig 提供預設值；
# 需要單獨調整時，呼叫 build_fqf_optimizer(...) 時傳入自訂 config 即可。
# Linear LR warmup over the first N optimizer steps (transformer 早期穩定)
# 從 base_lr * LR_WARMUP_START_FACTOR 線性增加到 base_lr
LR_WARMUP_STEPS         = 2000   # 第一次從頭訓練的 warmup 長度
LR_WARMUP_START_FACTOR  = 0.0
LR_RESUME_WARMUP_STEPS  = 2000   # 每次重啟（包含第一次）的額外 warmup 長度

# ── replay buffer ────────────────────────────────────────────────────
VISUAL_BUFFER_CAPACITY = 2048
VISUAL_SAVE_CAPACITY   = 512
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
    """screenshot (B,3,H,W) → cell features (B, grid_h*grid_w, encoder_final_dim).

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
        self.save_capacity = VISUAL_SAVE_CAPACITY
        self.priority_min = VISUAL_PRIORITY_MIN
        self.priority_max = VISUAL_PRIORITY_MAX
        self.log_prefix = "[V3]"
        self.log_actions = LOG_ACTIONS

        # ── Backbone：載入 Predictor 權重；YOLO11n 凍結，token_adapter+encoder 可訓練 ──
        # Replay buffer 改存 YOLO backbone 輸出 (128, h, w) 而不是原始截圖,所以 YOLO
        # 必須保持凍結(否則 cache 的 features 會與更新後的權重不一致)。token_adapter
        # 與 HierarchicalEncoder 是 RL 階段的 fine-tune 目標 — 預訓練學的是
        # YOLOGridStatePredictor 的「每格分類」資訊交換 pattern,Q-value 任務需要不同
        # 的全域聚合 pattern,所以解凍它讓 RL gradient 把它特化。
        self.backbone = VisualBackboneV3(grid_h=grid_h, grid_w=grid_w).to(device)
        self.backbone.load_encoder_weights_from_checkpoint(YOLO_PREDICTOR_PATH)
        self.backbone.freeze_feature_extractor()
        # BN 必須切到 eval,否則接下來的 dummy forward 會以 momentum=0.1 把全零輸入的
        # batch stats 混進剛從 checkpoint 載入的 running_mean/running_var,污染 YOLO BN。
        # torch.no_grad 不會抑制 BN running_stats 更新 — 那是受 module.training 控制的。
        self.backbone.set_bn_eval()

        # 推算 backbone 輸出的空間尺寸(用於 replay buffer 的 shape check)。
        # YOLO11n stride=16,所以 640x640 → 40x40;dummy forward 確認且為其他輸入尺寸保險。
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *IMAGE_SIZE, device=device)
            dummy_feat = self.backbone.extract_backbone_features(dummy)
            _, feat_c, feat_h, feat_w = dummy_feat.shape
            assert feat_c == YOLO_FEATURE_CHANNELS, (
                f"YOLO feature channels ({feat_c}) ≠ YOLO_FEATURE_CHANNELS "
                f"({YOLO_FEATURE_CHANNELS}); update yolo_encoder_base."
            )
        self.backbone_feature_shape: tuple[int, int, int] = (feat_c, feat_h, feat_w)
        # 給 mixin._load_persistent_buffer 用的 shape check;v3 覆寫成 cached features 形狀。
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

        # ── Optimizer（只更新 backbone decoder + queries，YOLO+encoder 已凍結）──
        # build_fqf_optimizer 會自動只收 requires_grad=True 的 params，跳過凍結 encoder。
        self.optimizer = build_fqf_optimizer(self.backbone, self.q_network)
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
            quota_check_class="win",  # Minesweeper: win is the rare-event bottleneck class
        )

        # ── image preprocessing ──
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        # ── episode / step bookkeeping ──
        self.total_it = 0
        self.steps_since_resume = 0  # 每次啟動重置；用於 resume LR warmup（不存檔）
        # episode_count 改成 @property delegate 到 training_history.total_episodes,
        # 單一 source of truth — 不再維護獨立 counter。
        self.train_every_n_steps = TRAIN_EVERY_N_STEPS
        self.pending_train_steps = 0
        self.n_step = VISUAL_N_STEP
        self.n_step_gamma = MINESWEEPER_REWARD_CONFIG.gamma
        self.n_step_buffer = deque()
        self.recent_real_rewards = deque(maxlen=100)

        # ── adaptive epsilon ──
        # 共用 controller,stage1 (TransformerDiscreteAgent) 也用同一個 class。
        # Episode 結果累積/查詢交給 TrainingHistory,controller 只吃 win_rate。
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

        # 砍掉上一版架構留下的 replay buffer .pt(shape mismatch 或損毀)。
        # 一定要在 try_load_model() 之前跑,否則 _load_persistent_training_state
        # 會嘗試載入指向已不存在/不匹配檔案的 entries,印一堆 missing warning。
        # replay_state_shape 在上面 dummy forward 後已經設好。
        self._purge_stale_replay_files()
        self.try_load_model()
        self._init_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference_step = self.total_it
        self._set_runtime_modes()
        # 註冊順序 = LIFO：close 最先註冊 → 最後執行；存檔最後註冊 → 最先執行
        atexit.register(self._close_io_log)
        atexit.register(self._close_tb_writer)
        atexit.register(self.save_persistent)
        atexit.register(self._save_model)

    # ──────────────────────────── runtime modes ──────────────────────────

    def _set_runtime_modes(self) -> None:
        self.backbone.train()                       # 整個 backbone 預設 train（dropout 等啟用）
        self.backbone.feature_extractor.eval()      # YOLO11n 凍結，永遠 eval
        self.backbone.set_bn_eval()                 # YOLO BN 保持 eval（凍結 running stats）
        self.backbone.token_adapter.train()         # 解凍 — 從 RL gradient 學新的特徵投影
        self.backbone.encoder.train()               # 解凍 — 學 Q-value 任務的全域資訊交換
        self.backbone.decoder.train()
        self.q_network.train()
        self.q_target.eval()

    @property
    def episode_count_public(self) -> int:
        return self.training_history.total_episodes

    # episode_count delegate 到 training_history.total_episodes — 由
    # log_episode_metrics() 內的 history.record() 自動 += 1,on_episode_end
    # 不再 increment。Read-only,setter 沒提供(資料源應該是 history)。
    @property
    def episode_count(self) -> int:
        return self.training_history.total_episodes

    # epsilon 由 controller 統一管理，但保留 self.epsilon 介面：
    # 1) select_action / TB log / 印 log 都直接讀 self.epsilon
    # 2) VisualAgentCommonMixin._load_optimizer_state 內有舊版 fallback 會
    #    `self.epsilon = state.get("epsilon", self.epsilon)`，setter 把它導
    #    回 controller。新版 load 路徑會走 epsilon_controller.load_state_dict
    #    走整包還原，這個 setter 只是相容舊呼叫點。
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

    # store_transition / _commit_n_step_transition / _flush_n_step_buffer 都來自
    # VisualAgentCommonMixin（見檔尾 rebinding）。Mixin 的 store_transition 會呼叫
    # 上面的 _to_storage_state hook,所以 v3 走的是 (128, h, w) 特徵儲存路徑。

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
        # state/next_state 是預先存好的 YOLO backbone 特徵 (B, 128, h, w),不是截圖。
        # YOLO11n forward 已經在 store_transition 階段做完,這裡只跑可訓練的
        # token_adapter + encoder + decoder。
        #
        # 重要:target 分支必須把 token_adapter/encoder/decoder 切到 eval,否則 dropout
        # (p=0.1) 會在 target Q 計算時 fire,讓同一個 next_state 在不同 batch 給出隨機
        # 的 target Q,變成 TD target 的噪音來源。no_grad 只擋反傳,不擋 dropout。
        # q_network 在 _set_runtime_modes 是 train(),這裡也要切 eval — 它跟 q_target 共
        # 享決策邏輯(double-Q argmax),target 分支要兩邊一致。算完 restore 訓練模式。
        self.backbone.token_adapter.eval()
        self.backbone.encoder.eval()
        self.backbone.decoder.eval()
        self.q_network.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(USE_AMP and device.type == "cuda")):
                _dbg("[train_step] before next backbone")
                next_features  = self.backbone.get_features_from_cached(next_state)
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
        # Restore 訓練模式 — current branch 要 dropout/BN-stats 都進去
        self._set_runtime_modes()

        # ── current branch (gradients flow through token_adapter + encoder + decoder + FQF；
        #                    YOLO11n 已凍結，且其 forward 已在 store_transition 階段完成）──
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(USE_AMP and device.type == "cuda")):
            _dbg("[train_step] before current backbone")
            features  = self.backbone.get_features_from_cached(state)
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

        # ── per-layer grad/weight norms must be logged BEFORE clip_grad_norm_，
        # 否則 grad 會被 in-place 縮過,看不出哪一層真的爆掉。
        if self.total_it % VISUAL_HISTOGRAM_EVERY == 0:
            self._log_backbone_weight_norms(self.total_it)

        params_to_clip = (
            list(self.backbone.parameters())
            + list(self.q_network.parameters())
        )
        grad_norm_total = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=VISUAL_GRAD_CLIP_NORM)
        grad_norm_total_value = float(grad_norm_total)
        grad_clip_threshold = float(VISUAL_GRAD_CLIP_NORM)
        grad_clip_scale = min(1.0, grad_clip_threshold / (grad_norm_total_value + 1e-12))
        grad_clip_percent = 1.0 - grad_clip_scale
        grad_clip_excess_norm = max(0.0, grad_norm_total_value - grad_clip_threshold)
        grad_clip_excess_ratio = grad_clip_excess_norm / (grad_clip_threshold + 1e-12)
        _dbg(
            "[train_step] after clip "
            f"grad_norm_total={grad_norm_total_value:.4g} "
            f"clip_scale={grad_clip_scale:.4g} "
            f"clip_percent={grad_clip_percent:.2%}"
        )

        backbone_post = self._module_grad_norm(self.backbone)
        head_post     = self._module_grad_norm(self.q_network)
        grad_post_total = (backbone_post ** 2 + head_post ** 2) ** 0.5

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
            f"  td_error_mean={td_error.mean().item():.6f} | frac_clipped={frac_clipped:.3f}\n"
            f"  fpn_norm_entropy={fpn_norm_entropy:.4f} | fpn_tau_std={fpn_tau_std:.4f}\n"
            f"  grad_total_pre={grad_norm_total_value:.6f} | grad_total_post={grad_post_total:.6f} | "
            f"clip_scale={grad_clip_scale:.6f} clip_percent={grad_clip_percent:.2%}\n"
            f"  grad_clip_threshold={grad_clip_threshold:.6f} | "
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
        self.tb_writer.add_scalar("train/td_error_mean",     td_error.mean().item(),       self.total_it)
        self.tb_writer.add_scalar("train/td_error_max",      td_error.max().item(),        self.total_it)
        self.tb_writer.add_scalar("train/target_q_mean",     target_quantiles.float().mean().item(), self.total_it)
        self.tb_writer.add_scalar("grad/total_norm",         grad_norm_total_value,        self.total_it)
        self.tb_writer.add_scalar("grad/post_total_norm",    grad_post_total,             self.total_it)
        self.tb_writer.add_scalar("grad/clip_threshold",     grad_clip_threshold,         self.total_it)
        self.tb_writer.add_scalar("grad/clip_percent",       grad_clip_percent,           self.total_it)
        self.tb_writer.add_scalar("grad/clip_excess_norm",   grad_clip_excess_norm,       self.total_it)
        self.tb_writer.add_scalar("grad/clip_excess_ratio",  grad_clip_excess_ratio,      self.total_it)
        self.tb_writer.add_scalar("grad_pre/backbone",       backbone_pre,                 self.total_it)
        self.tb_writer.add_scalar("grad_pre/head",           head_pre,                     self.total_it)
        self.tb_writer.add_scalar("grad_post/backbone",      backbone_post,                self.total_it)
        self.tb_writer.add_scalar("grad_post/head",          head_post,                    self.total_it)
        if weight_distance_log is not None:
            self.tb_writer.add_scalar("weights/delta_from_init", weight_distance_log["from_init"], self.total_it)
            self.tb_writer.add_scalar("weights/delta_from_prev_window", weight_distance_log["from_prev_window"], self.total_it)

        if self.scaler.is_enabled():
            self.tb_writer.add_scalar("train/scaler_scale", self.scaler.get_scale(), self.total_it)

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
        next_episode_idx = self.training_history.total_episodes + 1
        self._log_actions_this_episode = (
            self.action_log_every_n_episodes > 0
            and next_episode_idx % self.action_log_every_n_episodes == 0
        )
        if self._log_actions_this_episode:
            print(f"[V3] Action-image logging enabled for episode {next_episode_idx}")

    def on_episode_end(self) -> None:
        self._flush_n_step_buffer()
        # episode_count 是 @property delegate,history.record() 已自動 += 1。
        # epsilon 也已在 log_episode_metrics() 透過 controller.update 更新好。
        self.tb_writer.add_scalar("episode/epsilon", self.epsilon, self.episode_count)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[V3] Periodic save at episode {self.episode_count} | epsilon={self.epsilon:.4f}")
            self.save_persistent()

    def log_episode_metrics(self, win: bool, invalid_click_rate: float, reward_mean: float) -> None:
        # 1) 先把結果記到 history,2) 從 history 取 rolling win rate,
        # 3) 用 win rate 餵 controller 更新 epsilon (log-interpolation)。
        # 注意:v3 沒有 total_reward / steps 可傳,只 wire invalid_rate;
        # avg_reward / reward_per_step 這邊會永遠為 0,需要的話 caller 再補。
        self.training_history.record(win=win, invalid_rate=invalid_click_rate)
        ep_idx = self.training_history.total_episodes
        rolling_wr = self.training_history.win_rate(window=100)
        next_eps = self.epsilon_controller.update(rolling_wr)

        self.tb_writer.add_scalar("episode/reward_mean",        float(reward_mean),        ep_idx)
        self.tb_writer.add_scalar("episode/win",                float(bool(win)),          ep_idx)
        self.tb_writer.add_scalar("episode/invalid_click_rate", float(invalid_click_rate),ep_idx)
        self.tb_writer.add_scalar("episode/win_rate_recent",    rolling_wr,                ep_idx)
        self.tb_writer.flush()

        status = "WIN " if win else "LOSE"
        print(
            f"[V3] Ep {ep_idx}: {status} | "
            f"invalid={invalid_click_rate:.1%} | reward={reward_mean:.3f} | "
            f"win_rate(recent)={rolling_wr:.1%} | "
            f"eps(next)={next_eps:.4f}"
        )

    # ──────────────────────────── checkpoints ──────────────────────────

    def _save_model(self) -> None:
        VISUAL_V3_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(),  VISUAL_V3_MODEL_PATH / "backbone.pth")
        torch.save(self.q_network.state_dict(), VISUAL_V3_MODEL_PATH / "fqf_network.pth")
        torch.save(self.q_target.state_dict(),  VISUAL_V3_MODEL_PATH / "fqf_target.pth")
        self._save_optimizer_state()

    def _log_checkpoint_message(self, message: str, *, warning: bool = False) -> None:
        prefix = "[V3 CHECKPOINT]"
        line = f"{prefix} {message}"
        if warning:
            print(f"{CHECKPOINT_RED}{line}{CHECKPOINT_RESET}")
        else:
            print(line)
        try:
            self._io_log.write(f"{datetime.datetime.now().isoformat()} {line}\n")
            self._io_log.flush()
        except Exception:
            pass

    def try_load_model(self) -> None:
        bb_path = VISUAL_V3_MODEL_PATH / "backbone.pth"
        if bb_path.exists():
            try:
                self.backbone.load_state_dict(torch.load(bb_path, map_location=device, weights_only=True))
                self._log_checkpoint_message(f"Loaded backbone: {bb_path}")
            except Exception as exc:
                self._log_checkpoint_message(
                    f"MISSING/FAILED backbone checkpoint: {bb_path} | using initialized backbone | error={exc}",
                    warning=True,
                )
        else:
            self._log_checkpoint_message(
                f"MISSING backbone checkpoint: {bb_path} | using initialized backbone",
                warning=True,
            )

        q_path = VISUAL_V3_MODEL_PATH / "fqf_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device, weights_only=True))
                self._log_checkpoint_message(f"Loaded FQF-Network: {q_path}")
            except Exception as exc:
                self._log_checkpoint_message(
                    f"MISSING/FAILED FQF-Network checkpoint: {q_path} | using initialized q_network | error={exc}",
                    warning=True,
                )
        else:
            self._log_checkpoint_message(
                f"MISSING FQF-Network checkpoint: {q_path} | using initialized q_network",
                warning=True,
            )

        qt_path = VISUAL_V3_MODEL_PATH / "fqf_target.pth"
        if qt_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(qt_path, map_location=device, weights_only=True))
                self._log_checkpoint_message(f"Loaded FQF-Target: {qt_path}")
            except Exception as exc:
                self._log_checkpoint_message(
                    f"MISSING/FAILED FQF-Target checkpoint: {qt_path} | copying q_network if available | error={exc}",
                    warning=True,
                )
        elif q_path.exists():
            self._log_checkpoint_message(
                f"MISSING FQF-Target checkpoint: {qt_path} | copying q_network weights",
                warning=True,
            )
            self.q_target.load_state_dict(self.q_network.state_dict())
        else:
            self._log_checkpoint_message(
                f"MISSING FQF-Target checkpoint: {qt_path} | using initialized q_target",
                warning=True,
            )

        if not self._optimizer_state_path().exists():
            self._log_checkpoint_message(
                f"MISSING optimizer checkpoint: {self._optimizer_state_path()} | total_it stays at {self.total_it}",
                warning=True,
            )
        self._load_optimizer_state()
        if not self._training_state_path().exists():
            self._log_checkpoint_message(
                f"MISSING replay/training checkpoint: {self._training_state_path()} | replay buffer starts empty",
                warning=True,
            )
        self._load_persistent_training_state()

    # _load_persistent_buffer 來自 VisualAgentCommonMixin（見檔尾 rebinding）。
    # Mixin 版本會用 self.replay_state_shape（v3 在 __init__ 設成 backbone_feature_shape）
    # 來驗證 disk 上的 .pt 形狀,所以舊截圖檔(3, 640, 640) 會被自動跳過。

    # ──────────────────────────── lr warmup ────────────────────────────

    def _apply_lr_warmup(self) -> None:
        """Linear LR warmup，兩個 warmup 取較嚴格者：
        - init warmup：以 total_it 為進度，第一次從頭訓練時生效
        - resume warmup：以 steps_since_resume 為進度，每次啟動（含重啟）都會生效
        最終 factor = min(init_factor, resume_factor)，所以重啟後雖然 total_it 已大，
        resume warmup 仍會把 LR 從 LR_WARMUP_START_FACTOR×base_lr 線性拉回 base_lr。"""

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
