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
    • VisualBackboneV3 繼承 YOLOEncoderBase;encoder/decoder 結構跟 Stage 1
      (TransformerActorNetwork) 一致 — 預設 DEFAULT_ENCODER_START_DIM == FINAL_DIM,
      encoder 全 uniform 4 層 @ d=64,把 YOLO 128→64 的壓縮交給 token_adapter。
    • 嘗試從 Stage 1 checkpoint (models/stage1_transformer/) 載入
      encoder + decoder + query_tokens + FQF (all-or-nothing);缺檔 → 紅字 warning + random init。
    • YOLO11n 從 yolo11n.pt 載入並凍結(BN eval mode);token_adapter / encoder /
      decoder / query_tokens / FQF head 全部可訓練。
    • Replay buffer 改存 YOLO backbone 輸出 (128, 40, 40) 而不是原始截圖(3, 640, 640):
        - 每筆 transition 從 ~1.17 MB(uint8 截圖)降到 ~0.78 MB(fp32 特徵)
        - 訓練時跳過 YOLO11n forward,只跑 token_adapter + encoder + decoder
        - 因為 YOLO 凍結,cached features 對訓練等同每次重算
    • Dropout 起步關閉(p=0),win_rate(window=100) > 0.4 後 latch ON(對齊 PER spread_decay)。
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
    NUM_FQF_FRACTIONS,
    TRANSFORMER_MODEL_PATH,
    _quantile_huber_loss,
)
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
from model_structure.optimizer_factory import build_fqf_optimizer, FQFOptimizerConfig
from model_structure.hyperparameter_dump import dump_hyperparameters
from model_structure.rng_utils import seed_everything
from model_structure.archive_manager import (
    SessionArchiveManager,
    RolloverTextLog,
    swap_logger_file_handler,
)
from model_structure.training_logger import TrainingLogger
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

# ── reproducibility ──────────────────────────────────────────────────
# 模組 import 時生一個 32-bit seed 並 apply 到 random / numpy / torch / cuda。
# 同 stage1,SEED 是 module-level ALL_CAPS int,會被 hyperparameter_dump 寫進
# hyperparameters.txt,並在 agent.__init__ 印到 console 方便對照。
# 想重現特定 run:把下面這行改成 `SEED: int = seed_everything(<數字>)`。
SEED: int = seed_everything()

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
VISUAL_BATCH_SIZE = 32
MINIMUM_DATA_SIZE = 2000 # below this amount, won't start training

# ── Encoder dims ─────────────────────────────────────────────────────
# 形狀由 model_structure.yolo_encoder_base 的 DEFAULT_ENCODER_FINAL_DIM /
# DEFAULT_ENCODER_TOTAL_LAYERS / DEFAULT_ENCODER_START_DIM 決定;預設 [64]*5 (4 層 uniform,
# 對齊 Stage 1 的 EncoderDecoderTransformer)。
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
VISUAL_N_STEP = 1
VISUAL_GRAD_CLIP_NORM = 10.0   # 與 TransformerDiscreteAgent 對齊；token_adapter + encoder 加入可訓練後仍維持 10.0
TRAIN_EVERY_N_STEPS = 1
TARGET_UPDATE_FREQ = 200
SAVE_EVERY_N_EPISODES = 500
# ── throttle 命名跟 stage1 對齊(同功能、同名)──────────────────────
# DIAGNOSTIC_LOG_EVERY:io_log + .item() batch + ~22 條 TB scalar 的 gate。
# HISTOGRAM_EVERY:per-layer weight_norm/* + grad_norm/* 的 gate。
# WEIGHT_DISTANCE_LOG_EVERY:整模型 weight snapshot 距離。
# 三個值跟 stage1 一致(10 / 200 / 100),改一邊另一邊也要同步,免得兩個
# agent 在同樣的事上行為不一致。
DIAGNOSTIC_LOG_EVERY = 10
HISTOGRAM_EVERY = 200
WEIGHT_DISTANCE_LOG_EVERY = 100
USE_AMP = False

# ── learning rates ───────────────────────────────────────────────────
# 大部分 LR / weight_decay 走 FQFOptimizerConfig 預設(lr_backbone=lr_head=5e-5)。
# 但 v3 多了一個 catastrophic forgetting 防護:Stage 1 載入成功時,把從 Stage 1 載入的
# 部分(encoder + decoder + query_tokens)用 0.1× LR,避免 RL 階段稀疏雜訊大的 gradient
# 把 Stage 1 學到的 cell-level attention pattern 洗掉。token_adapter 永遠是 random init
# (Stage 1 沒對應的 Linear(128→64)),所以走 fresh LR。Stage 1 缺檔時所有 backbone 都是
# random init,pretrained_prefixes 動態設為空 tuple → 全部 fresh LR。
# 觀察 TB 的 weight_norm/encoder.* 與 grad_norm/encoder.* 來調整:
#   - encoder weight delta 太久不動 → 調大 LR_BACKBONE_PRETRAINED
#   - encoder weight 在前 1000 step 內 RMS 變化 > 50% → 調小
LR_BACKBONE_PRETRAINED = 5e-6   # Stage 1 載入時:encoder + decoder + query_tokens (0.1× backbone fresh LR)
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
# PER β annealing — 對齊 transformer_discrete_agent.py 的 PER_BETA_START/END。
# β 從 BETA_START 線性 anneal 到 BETA_END,在 episode_count = BETA_EP 時飽和。
# β 小 → IS weight 偏平均(弱修正,訓練初期穩);β=1 → 完全修正 priority 抽樣 bias。
VISUAL_PER_BETA_START  = 0.4
VISUAL_PER_BETA_END    = 1.0
VISUAL_PER_BETA_EP     = 5000
# PER spread_decay latch — 對齊 transformer_discrete_agent.py 的 spread_decay 設計。
# FQF quantile spread 進 priority modifier:wide spread = uncertain prediction → 抑制
# 這類 sample 的有效 priority(避免被 env 隨機性主宰)。起步關閉(early RL variance
# 大,spread 不能代表 uncertainty);win_rate(100) > LATCH_WR_THRESHOLD 後一次性
# latch ON、永不關回 — 跟 dropout latch 同 threshold、同 monotone 設計。
VISUAL_SPREAD_DECAY             = 2.0
SPREAD_DECAY_LATCH_WR_THRESHOLD = DROPOUT_LATCH_WR_THRESHOLD

LOG_ACTIONS = True


# ════════════════════════════════════════════════════════════════════════
# VisualBackboneV3 — YOLOEncoderBase（frozen）+ cross-attn decoder
# ════════════════════════════════════════════════════════════════════════
class VisualBackboneV3(YOLOEncoderBase):
    """screenshot (B,3,H,W) → cell features (B, grid_h*grid_w, encoder_final_dim).

    YOLO + token_adapter + HierarchicalEncoder 繼承自 YOLOEncoderBase。預設 encoder
    結構 (start_dim == final_dim, 4 uniform 層 @ d=64) 跟 Stage 1 的 EncoderDecoderTransformer
    完全一致 — VisualAgentV3 會嘗試載入 Stage 1 權重 warm-start encoder + decoder + queries。
    YOLO11n backbone 在 VisualAgentV3 內凍結;token_adapter (random init,Stage 1 沒對應)
    跟 encoder/decoder/queries 都跟著 RL 訓練更新。
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

        # ── Backbone：YOLO11n 凍結,其餘可訓練;encoder/decoder/queries 等下從 Stage 1 載入 ──
        # Replay buffer 改存 YOLO backbone 輸出 (128, h, w) 而不是原始截圖,所以 YOLO
        # 必須保持凍結(否則 cache 的 features 會與更新後的權重不一致)。token_adapter +
        # encoder + decoder + query_tokens 都是 RL fine-tune 目標 — 其中 encoder/decoder/
        # queries 之後會嘗試從 Stage 1 checkpoint warm-start(見 _load_stage1_weights);
        # token_adapter 永遠是 random init(Stage 1 沒對應的 Linear(128→64))。
        self.backbone = VisualBackboneV3(grid_h=grid_h, grid_w=grid_w).to(device)
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

        # ── Stage 1 warm-start (all-or-nothing) ─────────────────────────────
        # encoder/decoder/query_tokens 結構跟 Stage 1 (TransformerActorNetwork) 對齊,
        # 所以可以直接從 Stage 1 checkpoint 載入這幾個模組 + FQF online/target。
        # 三個檔(backbone/fqf_network/fqf_target)缺任一就放棄整批,各模組維持 random init。
        # _stage1_loaded 決定 optimizer 的 pretrained_prefixes 怎麼切。
        self._stage1_loaded = self._load_stage1_weights(STAGE1_CKPT_DIR)

        # ── Optimizer ───────────────────────────────────────────────────────
        # 只有 YOLO11n (feature_extractor) 凍結。token_adapter + encoder + decoder +
        # query_tokens + FQF head 都會更新。
        #
        # pretrained_prefixes 動態:
        #   • Stage 1 載入成功 → encoder/decoder/query_tokens 用 LR_BACKBONE_PRETRAINED
        #     (5e-6, 0.1× backbone fresh,防 RL gradient 把預訓 attention pattern 洗掉)
        #   • Stage 1 缺檔   → 全部 backbone 都是 random init,prefixes 設空,統一 5e-5
        # token_adapter 永遠 random init(Stage 1 沒對應),走 fresh LR group。
        #
        # build_fqf_optimizer 內部已過濾 requires_grad=False,凍結的 YOLO 不會進來。
        pretrained_prefixes: tuple[str, ...] = (
            ("encoder.", "decoder.", "query_tokens") if self._stage1_loaded else ()
        )
        self.optimizer = build_fqf_optimizer(
            self.backbone,
            self.q_network,
            config=FQFOptimizerConfig(lr_backbone_pretrained=LR_BACKBONE_PRETRAINED),
            pretrained_prefixes=pretrained_prefixes,
        )
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
            beta_start=VISUAL_PER_BETA_START,
            spread_decay=VISUAL_SPREAD_DECAY,
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
        # recent_real_rewards 搬到 TrainingHistory._step_rewards;mixin 的
        # store_transition 會 call self.training_history.record_step_reward(reward),
        # train_step 結尾用 self.training_history.avg_step_reward() 拿 rolling mean。

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
        self.action_log_every_n_episodes = 50
        self._log_actions_this_episode = False

        # ── Session / hour archive 目錄 ──────────────────────────────
        # 每次啟動一個 training_<ts>/,每滿 1 hour 一個 hour_NN_<ts>/。
        # canonical *.pth 仍寫在 VISUAL_V3_MODEL_PATH 頂層(try_load_model 直接讀);
        # 當下時段的 io_log / cuda_debug.log 都導到 current_archive_dir。
        # 目錄管理本體在 model_structure.archive_manager.SessionArchiveManager,
        # 跟 stage1 共用。
        self.archive = SessionArchiveManager(
            model_path=VISUAL_V3_MODEL_PATH,
            log_prefix="[V3]",
        )
        # SEED 是 module-level constant(由 seed_everything 產生);印出來方便事後
        # 對照 hyperparameters.txt 與 console 訊息。
        print(f"[V3] SEED = {SEED}")

        # ── Hyperparameters dump(含 git commit + dirty flag) ─────────
        # 把 module 內 ALL_CAPS 常數、reward_config dataclass、controller / optimizer
        # 的關鍵欄位寫成單一 hyperparameters.txt,session 開始時 dump 一次。
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
                    # trainable) + torchinfo layer summary。input_size 是真實 screenshot
                    # shape (1, 3, 640, 640)。torchinfo 會跑一次 YOLO forward,~50ms 可接受。
                    "model.backbone": (self.backbone, (1, 3, *IMAGE_SIZE)),
                    # q_network 不傳 input_size — 跟 stage1 同理(freeze status / param 統計
                    # 已足夠,forward 回 dict 不適合 torchinfo)。
                    "model.q_network": (self.q_network, None),
                },
            )
        except Exception as exc:
            print(f"[V3] hyperparameters dump failed: {exc}")

        # ── text log + TensorBoard ──
        # io_log:plain-text append + 每小時 swap 到新 hour 資料夾。Banner 由
        # _build_io_log_banner 產生。
        VISUAL_V3_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self._io_log = RolloverTextLog(banner_factory=self._build_io_log_banner)
        self._io_log.swap_to(self.archive.current_archive_dir / "train_io_log.txt")

        # cuda_debug.log 也跟著 archive dir 走(module-level handler 預設指向頂層,
        # 這裡 swap 成 current_archive_dir 的版本)。CategorizedReplayBuffer 共用
        # 同一個檔案路徑,在這裡同步更新。
        dbg_path = self.archive.current_archive_dir / "cuda_debug.log"
        swap_logger_file_handler(_dbg_logger, dbg_path)
        _crb_module.DEBUG_CUDA_SAMPLE_LOG_PATH = dbg_path

        # 註冊 hour rollover callback:io_log + dbg logger 翻檔。Console print 由
        # SessionArchiveManager 在翻頁時負責。
        self.archive.register_on_rollover(self._on_archive_rollover)

        # TensorBoard log_dir 仍走頂層 VISUAL_V3_TENSORBOARD_DIR/<ts>,跟 archive
        # dir 解耦 — 這樣同一個 TB 實例可以瀏覽歷史 run。SummaryWriter 不再對外
        # 暴露為 self.tb_writer;它變成 TrainingLogger 的內部 detail,所有 TB 寫入
        # 都從 self.training_logger 走。
        tb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log_dir = VISUAL_V3_TENSORBOARD_DIR / tb_timestamp
        print(f"[V3] TensorBoard: tensorboard --logdir {VISUAL_V3_TENSORBOARD_DIR}")
        print(f"[V3] Current run: {self.tensorboard_log_dir}")

        # CSV + TB dispatcher — 每 episode summary 兩邊一起寫,raw value(不 format)
        # 給 AI 讀。Hour rollover 時跟著翻新 csv 檔。跟 stage1 的 csv_fields 結構
        # 對齊,只是 v3 沒有 eval(目前 Demo_test_Minesweeper 沒有 eval loop),
        # 所以省略 eval/* 欄位。Q_loss / q_mean 也省略 — 那兩條由 agent.train_step
        # 自己每 step 寫 TB,不適合塞進 episode-level CSV row。
        # SummaryWriter 直接餵進 TrainingLogger,agent 不再單獨持有引用。
        csv_fields = [
            "timestamp", "episode",
            "reward_mean", "is_win", "invalid_click_rate",
            "win_rate_recent", "epsilon",
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

        # 砍掉上一版架構留下的 replay buffer .pt(shape mismatch 或損毀)。
        # 一定要在 try_load_model() 之前跑,否則 _load_persistent_training_state
        # 會嘗試載入指向已不存在/不匹配檔案的 entries,印一堆 missing warning。
        # replay_state_shape 在上面 dummy forward 後已經設好。
        self._purge_stale_replay_files()
        self.try_load_model()
        self._init_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference_step = self.total_it

        # ── Dropout latch ───────────────────────────────────────────────────
        # 起步 p=0;train_step 每次檢查 win_rate(100) > DROPOUT_LATCH_WR_THRESHOLD,
        # 第一次跨過就 latch ON、永不關回。對齊 PER spread_decay 的 latch 設計。
        # latch flag 不存檔(就跟 spread_decay 一樣)— training_history.pth 載回來後,
        # 這裡先做一次 re-evaluate,resume 後不用等到第一個 train_step 才 latch。
        self._dropout_latched = False
        self._set_backbone_dropout_p(0.0)
        if self.training_history.win_rate(window=100) > DROPOUT_LATCH_WR_THRESHOLD:
            self._latch_dropout_on(reason="resume: training_history win_rate already above threshold")

        self._set_runtime_modes()
        # 註冊順序 = LIFO:close 最先註冊 → 最後執行;存檔最後註冊 → 最先執行
        # _close_training_logger 一次關 CSV file handle + SummaryWriter,取代
        # 原本的 _close_episode_logger + _close_tb_writer。
        atexit.register(self._close_training_logger)
        atexit.register(self._close_io_log)
        atexit.register(self.save_persistent)
        atexit.register(self._save_model)

    # ──────────────────────────── Stage 1 warm-start ──────────────────────

    def _load_stage1_weights(self, stage1_dir: Path) -> bool:
        """從 Stage 1 checkpoint 載 encoder/decoder/queries/FQF(all-or-nothing)。

        三個檔(backbone.pth / fqf_network.pth / fqf_target.pth)缺任一就放棄整批,
        各模組維持 random init,回傳 False。三個都在才實際呼叫 load_state_dict,
        回傳 True 表示 optimizer 應該把 encoder/decoder/query_tokens 切到 pretrained group。

        Key remapping (Stage 1 → V3):
            query_tokens                              → query_tokens                       (1:1)
            core.transformer.layers.{i}.attn.<...>    → encoder.layers.{i}.attn.<...>      (new-style, post-HierarchicalEncoder)
            core.transformer.layers.{i}.<...>         → encoder.layers.{i}.attn.<...>      (old-style, pre-refactor — auto-insert .attn)
            core.decoder.layers.{i}.<...>             → decoder.layers.{i}.<...>           (1:1)
            token_embed / output_head / position → 丟棄
        FQF online/target 直接 1:1 載入。
        """
        backbone_pth = stage1_dir / "backbone.pth"
        qnet_pth     = stage1_dir / "fqf_network.pth"
        qtgt_pth     = stage1_dir / "fqf_target.pth"

        missing = [p for p in (backbone_pth, qnet_pth, qtgt_pth) if not p.exists()]
        if missing:
            for p in missing:
                self._log_checkpoint_message(
                    f"Stage 1 checkpoint 不存在: {p} — all-or-nothing 放棄整批,使用 random init",
                    warning=True,
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
            # token_embed / output_head / position / 其他 — 丟棄(架構不同 or 不需要)
        missing_keys, unexpected_keys = self.backbone.load_state_dict(remapped, strict=False)
        # missing 預期包含 feature_extractor.* / token_adapter.* / memory_position.* —
        # 那些 V3 有但 Stage 1 沒有,屬於正常;unexpected 應為空。
        self._log_checkpoint_message(
            f"Stage 1 backbone loaded ({len(remapped)} tensors from {backbone_pth.name})"
        )
        if unexpected_keys:
            self._log_checkpoint_message(
                f"Stage 1 backbone unexpected keys (應為空): {unexpected_keys[:5]}"
                f"{'...' if len(unexpected_keys) > 5 else ''}",
                warning=True,
            )

        # FQF online + target
        self.q_network.load_state_dict(
            torch.load(qnet_pth, map_location="cpu", weights_only=False)
        )
        self._log_checkpoint_message(f"Stage 1 fqf_network loaded ({qnet_pth.name})")
        self.q_target.load_state_dict(
            torch.load(qtgt_pth, map_location="cpu", weights_only=False)
        )
        self._log_checkpoint_message(f"Stage 1 fqf_target loaded ({qtgt_pth.name})")

        return True

    # ──────────────────────────── dropout latch ────────────────────────────

    def _set_backbone_dropout_p(self, p: float) -> None:
        """掃 backbone 內所有 nn.Dropout (encoder + decoder 各層的內部 dropout),改 p。

        FQFQNetwork 內沒有 Dropout(只有 Linear + GELU),所以這裡只動 backbone。
        nn.Dropout.p 是 Python attribute、不是 Parameter/buffer,不會進 state_dict —
        所以 latch 後 _save_model / try_load_model 不會影響這個 p。
        """
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(p)

    def _latch_dropout_on(self, *, reason: str) -> None:
        self._dropout_latched = True
        self._set_backbone_dropout_p(DECODER_DROPOUT)
        self._log_checkpoint_message(
            f"dropout latch ON (p={DECODER_DROPOUT}, threshold={DROPOUT_LATCH_WR_THRESHOLD}) — {reason}",
            warning=True,
        )

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

        # Dropout latch — 對齊 transformer_discrete_agent.py:610 的 spread_decay latch 位置。
        # 一次性 switch:跨過 threshold 就 ON 永不關回。training_history.win_rate 是 100-window
        # rolling, on_episode_end 才 .record(),所以這個 check 對同一個 episode 內多次 train_step
        # 給出穩定的結果。
        if not self._dropout_latched:
            if self.training_history.win_rate(window=100) > DROPOUT_LATCH_WR_THRESHOLD:
                self._latch_dropout_on(
                    reason=f"train_step total_it={self.total_it}: win_rate(100) crossed threshold"
                )

        # Spread-decay latch — 對齊 transformer_discrete_agent.py:621-628。FQF quantile
        # spread 進 priority modifier 的 master switch。同 threshold、同 monotone:跨過
        # 就 ON、永不關回。state 在 buffer 物件上(self.replay_buffer.enable_spread_decay),
        # 不存檔(stage1 也是)— resume 後第一個 train_step 重新 evaluate 即可。
        if not self.replay_buffer.enable_spread_decay:
            if self.training_history.win_rate(window=100) > SPREAD_DECAY_LATCH_WR_THRESHOLD:
                self.replay_buffer.enable_spread_decay = True

        # Wall-clock 滿 1 小時翻頁;io_log / cuda_debug.log 都會自動跟著新的 archive
        # dir(註冊在 SessionArchiveManager 的 on_rollover callback 處理)。
        self.archive.maybe_rollover()
        _dbg(f"[train_step] ENTER total_it={self.total_it} buf_size={buf_size}")
        _dbg_mem("train_step ENTER")
        _dbg("[train_step] before replay_buffer.sample")
        # PER β annealing — 對齊 transformer_discrete_agent.py:641-646。β 從
        # BETA_START 線性 anneal 到 BETA_END(VISUAL_PER_BETA_EP 個 episode 飽和),
        # 早期偏平均(弱修正、訓練穩),後期完全修正 priority 抽樣 bias。
        per_beta = VISUAL_PER_BETA_START + (VISUAL_PER_BETA_END - VISUAL_PER_BETA_START) * min(
            self.episode_count / VISUAL_PER_BETA_EP, 1.0
        )
        state, action, next_state, reward, done, sample_indices, is_weights, discounts, n_steps = (
            self.replay_buffer.sample(
                VISUAL_BATCH_SIZE,
                beta=per_beta,
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
            # Per-sample quantile spread for the chosen action — 餵 buffer 的 spread_decay
            # modifier(_effective_priority 內,僅在 enable_spread_decay=True 時參與)。
            # 對齊 transformer_discrete_agent.py:760:不論 latch 是否 ON 都寫,讓 latch 翻
            # ON 時不會有 cold-start 期間(entry 內 quantile_spread 從上一輪 sample 起就有值)。
            chosen_q_spread = chosen_quantiles.std(dim=1).detach()
            # fpn 兩條維持 tensor;.item() 延後到 DIAGNOSTIC gate 內跟其他 scalar
            # 一起 batch,省掉每步 2 個 GPU→CPU sync。
            fpn_norm_entropy_t = entropy.mean() / math.log(NUM_FQF_FRACTIONS)
            fpn_tau_std_t = tau_hats.std(dim=1).mean()

        _dbg_tensor("train_step.loss", loss)
        _dbg_tensor("train_step.td_error", td_error)

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
        if self.total_it % HISTOGRAM_EVERY == 0:
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

        # 對齊 transformer_discrete_agent.py:955-959 ── update_priorities 放在 backward
        # + clip + optimizer.step 全部成功後才寫;backward 失敗會直接 raise,priority
        # 維持上一輪的值,避免「失敗的 step 仍把 priority 改掉」的不一致。
        # quantile_spreads 不論 enable_spread_decay 是 True/False 都寫,讓 latch 翻 ON
        # 時直接拿 sample 累積的真實 spread,而不是從 0 開始 cold-start。
        self.replay_buffer.update_priorities(
            sample_indices,
            td_error.squeeze(-1).cpu().numpy(),
            quantile_spreads=chosen_q_spread.cpu().numpy(),
        )
        _dbg("[train_step] after update_priorities")

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

        # ── Diagnostic / logging — only every DIAGNOSTIC_LOG_EVERY steps ──
        # 跟 stage1 的 throttle 策略一致:把 io_log / TB scalar / no_grad stats
        # 區塊降頻,其他 step 直接 return None。
        # 省下的 sync 包含:loss / q_taken.mean / td_error.mean / td_error.max /
        # target_quantiles.float().mean / fpn_norm_entropy / fpn_tau_std,7 個
        # `.item()` 合成一次 `.cpu().tolist()`。
        # 其他 gate(TARGET_UPDATE_FREQ / WEIGHT_DISTANCE_LOG_EVERY / HISTOGRAM_EVERY)
        # 本來就獨立 throttle,DIAGNOSTIC_LOG_EVERY=10 是其他 gate 的因數
        # (HISTOGRAM=200, WEIGHT_DISTANCE=100, TARGET=200),所以每次他們 fire 都
        # 一定也是 DIAGNOSTIC step,TB 寫入順序仍正確。
        if self.total_it % DIAGNOSTIC_LOG_EVERY != 0:
            return None

        # Batched stats: 一次 .cpu().tolist() 從 7 個 sync 變成 1 個。順序要跟下面
        # unpack 對齊。frac_clipped 已是 Python float(來自 _quantile_huber_loss
        # return_stats=True 內部 .item()),跟 stage1 一致不重新 batch。
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

            # raw reward rolling mean(來源:training_history._step_rewards,由
            # mixin store_transition 維護;Python-side deque,不走 GPU sync)。
            real_reward_mean = self.training_history.avg_step_reward()

            # top-5 actions（torch.topk → tolist 是 sync,但很小,留著)
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
            f"  grad_clip_threshold={grad_clip_threshold:.6f} | "
            f"backbone_pre={backbone_pre:.6f} head_pre={head_pre:.6f}\n"
            f"{weight_delta_line}"
            f"---\n"
        )
        self._io_log.flush()

        # Per-step diagnostics 全部走 training_logger.log(... csv=False) —— csv=False
        # 表示「只進 TB,不入 CSV row」。logger 收到後 call SummaryWriter.add_scalar,
        # 等同直接寫,但介面跟 episode summary 統一。
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
        self.training_logger.log("grad/clip_threshold",     grad_clip_threshold,                           step=step, csv=False)
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

        # ── LR scalars(每個 param group 一條,在 TB 上歸在 lr/ 分類底下)──
        # build_fqf_optimizer 在每個 group 內塞了 "name" key;若舊 checkpoint 載入後沒
        # 這個 key(理論上不會發生,但保險),fallback 用 index。warmup 期間 group["lr"]
        # 已經被 _apply_lr_warmup() in-place 改成 base_lr * factor,所以這裡讀的就是當下
        # 真正生效的 LR — TB 上能直接看到 warmup ramp + 三個 group 的 ratio。
        for idx, group in enumerate(self.optimizer.param_groups):
            tag = group.get("name") or f"group_{idx}"
            self.training_logger.log(f"lr/{tag}", group["lr"], step=step, csv=False)

        self.training_logger.flush()
        _dbg(f"[train_step] EXIT total_it={self.total_it}")
        _dbg_mem("train_step EXIT")
        # 用 batched scalar(loss_value)而不是 loss.item() — 上面已經 sync 過,
        # 不要再多一次。Demo_test_Minesweeper 端 `if loss_info:` 判 None,所以
        # throttle 跳過時(return None)caller 自動 silent skip。
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
        # episode_count 是 @property delegate,history.record() 已自動 += 1。
        # epsilon 也已在 log_episode_metrics() 透過 controller.update 更新好。
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
        # 1) 先把結果記到 history,2) 從 history 取 rolling win rate,
        # 3) 用 win rate 餵 controller 更新 epsilon (log-interpolation)。
        # 注意:v3 沒有 total_reward / steps 可傳,只 wire invalid_rate;
        # avg_reward / reward_per_step 這邊會永遠為 0,需要的話 caller 再補。
        # 點擊次數的「絕對量」(valid/invalid_clicks)不寫 TB — 比例
        # `episode/invalid_click_rate` 已足夠,raw count 可以從 invalid_rate ×
        # steps 反推,不需要兩條 derivable 曲線重複占版面。
        self.training_history.record(win=win, invalid_rate=invalid_click_rate)
        ep_idx = self.training_history.total_episodes
        rolling_wr = self.training_history.win_rate(window=100)
        next_eps = self.epsilon_controller.update(rolling_wr)

        # Episode summary metrics 走 TrainingLogger — 一次寫 TB + CSV。CSV row
        # 累積到結尾 commit_csv_row 才落地。NOTE: 拿掉 `episode/win` — 0/1 binary
        # 噪音大,看 `episode/win_rate_recent` 就好;但 `is_win` 仍寫 CSV(AI
        # 端可自己 rolling 任意 window)。
        self.training_logger.log("episode/reward_mean",        float(reward_mean),        step=ep_idx, csv_col="reward_mean")
        self.training_logger.log("episode/invalid_click_rate", float(invalid_click_rate), step=ep_idx, csv_col="invalid_click_rate")
        self.training_logger.log("episode/win_rate_recent",    rolling_wr,                step=ep_idx, csv_col="win_rate_recent")
        self.training_logger.log("is_win",                     int(bool(win)),            step=ep_idx, tb=False)
        self.training_logger.log("epsilon",                    next_eps,                  step=ep_idx, tb=False)  # TB 端由 on_episode_end 寫
        self.training_logger.log("timestamp",                  datetime.datetime.now().isoformat(), step=ep_idx, tb=False)
        self.training_logger.log("episode",                    ep_idx,                    step=ep_idx, tb=False)

        # Replay buffer composition — agent.train_step 內也寫,但 buffer 沒滿
        # MINIMUM_DATA_SIZE 前 train_step return None,那段時間 TB 空白。每個 episode
        # 從這寫保證 fill curve 與 4 個 bucket 比例從 ep 1 開始就有。TB only —
        # 走 logger 統一介面跟 csv=False。
        for bucket_name, count in self.replay_buffer.bucket_sizes().items():
            self.training_logger.log(f"buffer/bucket_{bucket_name}", count, step=ep_idx, csv=False)
        self.training_logger.log("buffer/total_size", self.replay_buffer.size(), step=ep_idx, csv=False)

        # Agent 內部 counter — total_it 看 train_step 跑了幾次;n_step_buffer_len 在
        # episode 邊界預期 = 0(reset_episode 與 on_episode_end 都 flush),長期非 0
        # 代表 flush 沒生效。TB only。
        self.training_logger.log("train/total_it",          self.total_it,           step=ep_idx, csv=False)
        self.training_logger.log("train/n_step_buffer_len", len(self.n_step_buffer), step=ep_idx, csv=False)

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

    def _save_model(self) -> None:
        VISUAL_V3_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        backbone_sd = self.backbone.state_dict()
        qnet_sd     = self.q_network.state_dict()
        qtarget_sd  = self.q_target.state_dict()
        torch.save(backbone_sd, VISUAL_V3_MODEL_PATH / "backbone.pth")
        torch.save(qnet_sd,     VISUAL_V3_MODEL_PATH / "fqf_network.pth")
        torch.save(qtarget_sd,  VISUAL_V3_MODEL_PATH / "fqf_target.pth")
        self._save_optimizer_state()

        # 同一份也寫到當下 hour 資料夾,做歷史快照(同一小時內多次 save 會 overwrite,
        # 留下「該小時最後一次 save」的狀態 — 滿足「保留每次的訓練結果」)。
        # Archive 失敗不擋 canonical save 的成功。跟 stage1 的 _save_model 同步驟。
        try:
            archive = self.archive.current_archive_dir
            archive.mkdir(parents=True, exist_ok=True)
            torch.save(backbone_sd, archive / "backbone.pth")
            torch.save(qnet_sd,     archive / "fqf_network.pth")
            torch.save(qtarget_sd,  archive / "fqf_target.pth")
        except Exception as exc:
            print(f"[V3] WARN: archive snapshot write failed: {exc}")

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
    # 目錄管理本體在 SessionArchiveManager(跟 stage1 共用同一個 class)。
    # current_archive_dir 對外是 read-only @property delegate,讓外部呼叫端
    # 沿用簡短名稱;_build_io_log_banner / _on_archive_rollover 是 manager 的
    # 兩個 callback hook。

    @property
    def current_archive_dir(self) -> Path:
        return self.archive.current_archive_dir

    def _build_io_log_banner(self, path: Path) -> list[str]:
        """RolloverTextLog banner:寫 session/hour 起始時間 + encoder dims + 檔案路徑。"""
        return [
            f"Session started: {self.archive.session_start.isoformat()}",
            f"Hour {self.archive.hour_index:02d} started: {datetime.datetime.now().isoformat()}",
            f"Encoder dims: {ENCODER_DIMS}",
            f"Path: {path}",
        ]

    def _on_archive_rollover(self, new_dir: Path) -> None:
        """SessionArchiveManager rollover callback:翻頁時 swap io_log + dbg logger。"""
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

    # ──────────────────────────── cleanup ──────────────────────────────

    def _close_io_log(self) -> None:
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def _close_training_logger(self) -> None:
        """Atexit hook:同時關 CSV file handle 與 SummaryWriter。

        SummaryWriter 自 `__init__` 以來只活在 self.training_logger 內部,沒有外
        部引用;這裡只需要 call training_logger.close() 一次,內部會把 csv 與
        TB writer 都關掉。
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
