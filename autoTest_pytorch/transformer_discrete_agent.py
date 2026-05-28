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
import model_structure.CategorizedReplayBuffer as _crb_module


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── reproducibility ──────────────────────────────────────────────────
# 模組 import 時生一個 32-bit seed,並 apply 到 random / numpy / torch / cuda。
# 實際使用的 SEED 會被 hyperparameter_dump 自動寫進 hyperparameters.txt
# (因為是 module-level ALL_CAPS int,符合 _dump_module 的篩選條件),
# agent.__init__ 也會印到 console,方便事後對照。
# 想重現特定 run:把下面這行改成 `SEED: int = seed_everything(<數字>)`,
# 並從零開始訓練 — _save_model 會把 RNG state 存進 optimizer_state.pth,
# resume 後 RNG trajectory 從 checkpoint 還原,SEED 只決定首次啟動的初始狀態。
SEED: int = seed_everything()

BATCH_SIZE = 128
GRID_STATE_CHANNELS = 12
PER_CAPACITY = 10000
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
# Phase 1 (stratified balanced) 占 batch 的比例。1.0 = 全部 batch 走 per-class
# stratified,PER 只在某類不足時補位。0.5 = 原始 50/50。0.0 = 純 PER 全域抽。
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

# ── learning rate warmup ─────────────────────────────────────────────
# Linear LR warmup over the first N optimizer steps (transformer 早期穩定)
# 從 base_lr * LR_WARMUP_START_FACTOR 線性增加到 base_lr
LR_WARMUP_STEPS         = 2000   # 第一次從頭訓練的 warmup 長度
LR_WARMUP_START_FACTOR  = 0.0
LR_RESUME_WARMUP_STEPS  = 2000   # 每次重啟（包含第一次）的額外 warmup 長度

TRANSFORMER_MODEL_PATH = Path("./models/stage1_transformer")
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_FF_DIM = 256
TRANSFORMER_DROPOUT = 0.1

# ── CUDA SDP backend ──────────────────────────────────────────────────
# mem-efficient SDP kernel 在某些 CUDA / GPU 組合上會丟 "illegal instruction"。
# flash / math 用 PyTorch 預設（皆 enabled）。
if device.type == "cuda":
    try:
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass

# ── debug logger (寫到檔案，不噴 CMD；DEBUG_CUDA_SAMPLE=True 才會啟用) ──
# True 時會插入 cuda.synchronize + 寫 log，會拖慢訓練；只在除錯 CUDA error 時開。
DEBUG_CUDA_SAMPLE = False

_DBG_LOG_PATH = TRANSFORMER_MODEL_PATH / "cuda_debug.log"
_DBG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_dbg_logger = _logging.getLogger("cuda_dbg.stage1")
_dbg_logger.setLevel(_logging.DEBUG)
_dbg_logger.propagate = False  # 不往 root logger 傳，避免 CMD 也印
# Module-level handler:agent 還沒 instantiate 之前的 fallback 寫到 top-level。
# delay=True 讓檔案只在真的有 emit 時才開出來(DEBUG_CUDA_SAMPLE=False + 沒 CUDA crash
# 的情況下就不會留下空檔)。Agent __init__ 會把這個 handler 換成指到 archive dir 的。
if not _dbg_logger.handlers:
    _fh = _logging.FileHandler(_DBG_LOG_PATH, mode="a", encoding="utf-8", delay=True)
    _fh.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
    _dbg_logger.addHandler(_fh)

# 把 ReplayBuffer 的 [DBG sample] log 也導到同一個檔（取代它原本的 print）。
_crb_module.DEBUG_CUDA_SAMPLE_LOG_PATH = _DBG_LOG_PATH


def _dbg(msg: str) -> None:
    """log+flush first, then sync — 最後寫到磁碟的那一行 = 即將同步的 op。"""
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
    """記錄 GPU memory 使用量。"""
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
    """檢查 tensor 的 NaN/Inf 與超界，記錄 shape/dtype/range。

    expect_max / expect_min: 整數 tensor 的硬界，超出記為 OOB（很可能是壞 index）。
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
    """供呼叫端在最外層 except 用，把 traceback 寫進 cuda_debug.log。

    無視 DEBUG_CUDA_SAMPLE 開關 — exception 一律要落地。
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

        return self.load_state_dict(normalized, strict=strict)


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


class TransformerDiscreteAgent:
    """FQF agent with grid encoder-decoder transformer backbone."""

    def __init__(self, grid_h=10, grid_w=10):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w

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
        # 紀錄每個 param group 的 base lr，warmup 期間根據 total_it / steps_since_resume 動態縮放
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
        self.steps_since_resume = 0  # 每次啟動重置；用於 resume LR warmup（不存檔）
        # episode_count 改成 @property delegate 到 training_history.total_episodes,
        # 單一 source of truth — 不再維護獨立 counter。
        self.n_step = N_STEP
        self.n_step_gamma = MINESWEEPER_REWARD_CONFIG.gamma
        self.n_step_buffer = deque()
        # raw reward rolling mean 搬到 TrainingHistory._step_rewards;store_transition
        # 內 call self.training_history.record_step_reward(reward),train_step 結尾
        # 用 self.training_history.avg_step_reward() 拿 mean。跟 v2 / v3 共用同一個
        # method,避免兩邊各自貼一份相同的 sum()/len() 公式。

        # ── adaptive epsilon ──
        # 跟 v3 共用同一個 controller class,並用相同 wr / eps 範圍。Stage1 與
        # stage2 對 minesweeper 6x6 的 reward signal 相同,所以套用一致的設定。
        # Episode 結果累積/查詢交給 TrainingHistory,controller 只吃 win_rate。
        self.epsilon_controller = AdaptiveEpsilonController()
        self.training_history = TrainingHistory()
        self.deque_cls = deque  # 給 training_history.load_state_dict() 用

        # Lazy-captured at first train_step: if the loaded TrainingHistory shows last
        # win_rate(window=100) > 0.5, we additionally gate training on the replay buffer
        # having all 4 classes filled to their 12.5% soft-floor quota. Fresh runs
        # (history empty → win_rate=0) and weak resumes skip this gate.
        # Captured once and frozen for the whole session — won't flip when live win_rate
        # crosses 50% mid-training.
        self._class_quota_gate_enabled: bool | None = None

        # episode-scoped blocked actions (v3 風格)：點過的格子在本 episode 內 mask 掉
        self.blocked_actions: set[int] = set()

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        # Session / hour archive 目錄 ── 每次啟動一個 training_<ts>,每滿 1 hour 一個 hour_NN_<ts>。
        # canonical *.pth 仍寫在 TRANSFORMER_MODEL_PATH 頂層(try_load_model 直接讀),
        # 同時把同一份快照寫到 current_archive_dir,把當下時段的 logs 也都導到那邊去。
        # 目錄管理本身搬到 model_structure.archive_manager.SessionArchiveManager;
        # 這裡只是組裝。
        self.archive = SessionArchiveManager(
            model_path=TRANSFORMER_MODEL_PATH,
            log_prefix="[FQF]",
        )
        # SEED 是 module-level constant(由 seed_everything 產生);印出來方便事後
        # 對照 hyperparameters.txt 與 console 訊息。
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
            )
        except Exception as exc:
            print(f"[FQF] hyperparameters dump failed: {exc}")

        # io_log:plain-text append + 每小時 swap 到新 hour 資料夾。Banner 由
        # _build_io_log_banner 產生(寫 session/hour 起始時間 + 檔案路徑)。
        self._io_log = RolloverTextLog(banner_factory=self._build_io_log_banner)
        self._io_log.swap_to(self.archive.current_archive_dir / "train_io_log.txt")

        # cuda_debug.log 也跟著 archive dir 走(module-level handler 預設指向 top-level,
        # 在這裡 swap 成 current_archive_dir 的版本)。CategorizedReplayBuffer 共用
        # 同一個檔案路徑,在這裡同步更新。
        dbg_path = self.archive.current_archive_dir / "cuda_debug.log"
        swap_logger_file_handler(_dbg_logger, dbg_path)
        _crb_module.DEBUG_CUDA_SAMPLE_LOG_PATH = dbg_path

        # 註冊 hour rollover callback:io_log + dbg logger + ReplayBuffer cuda log
        # 全部跟著翻檔。本身的 console print 由 SessionArchiveManager 在翻頁時印。
        self.archive.register_on_rollover(self._on_archive_rollover)

        # TensorBoard — own log_dir keyed by session timestamp, so the
        # training script (train_stage1_simple.py) can reuse it instead of
        # creating a second writer.
        tb_root = TRANSFORMER_MODEL_PATH / "tensorboard"
        tb_root.mkdir(parents=True, exist_ok=True)
        tb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log_dir = tb_root / tb_timestamp
        self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))
        print(f"[FQF] TensorBoard: tensorboard --logdir {tb_root}")
        print(f"[FQF] Current run: {self.tensorboard_log_dir}")
        self._write_metric_docs()

        self.try_load_model()

        # Weight snapshots for diagnosing drift. Must be captured AFTER
        # try_load_model so that "init" reflects the actual starting point
        # of this session (including any loaded checkpoint).
        self._init_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference = self._capture_trainable_weight_snapshot()
        self._rolling_weight_reference_step = self.total_it

        # atexit order is LIFO. Desired run order: _save_model (which flushes
        # tb) → save_persistent → close io_log → close tb_writer. So register
        # closes first (run last) and saves last (run first).
        atexit.register(self._close_tb_writer)
        atexit.register(self._close_io_log)
        atexit.register(self.save_persistent)
        atexit.register(self._save_model)

    # epsilon 統一由 controller 管理；保留 self.epsilon 介面以相容
    # select_action、TB log、save/load 等舊呼叫點。
    @property
    def epsilon(self) -> float:
        return self.epsilon_controller.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.epsilon_controller.epsilon = float(value)

    # episode_count delegate 到 training_history.total_episodes — 後者在
    # log_episode_metrics() 內 history.record() 時自動 += 1,on_episode_end
    # 不再需要獨立 increment。Read-only,setter 會 raise(資料源應該是 history)。
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
        # 餵 train/real_reward_mean — raw reward(reward_squash 之前)的 rolling mean
        # 走 TrainingHistory._step_rewards,跟 v2 / v3 共用同一個 record_step_reward。
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
        # Wall-clock 滿 1 小時翻頁;io_log / vclamp / save 都會自動跟著新的 archive dir
        # (註冊在 SessionArchiveManager 的 on_rollover callback 處理)
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

            # Stage 0: buffer sample — 命中代表 replay buffer 內容已壞(load 或 store 路徑)
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
                # Stage 1a: backbone forward on next_state — 命中代表 backbone weights 或
                #           next_state 已壞;這條也是訓練中最早能偵測到 backbone 損毀的點
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
                # Stage 1b: q_target forward — 命中代表 q_target weights 已壞
                self._assert_finite("stage1b_q_target", "next_target.quantiles", next_target["quantiles"])

                next_target_quantiles = next_target["quantiles"][
                    torch.arange(batch_size, device=device), best_flat
                ]
                _dbg_tensor("train_step.next_target_quantiles", next_target_quantiles)
                target_quantiles = reward + (1 - done) * discounts * next_target_quantiles
                _dbg_tensor("train_step.target_quantiles", target_quantiles)
                # Stage 1c: target_quantiles 算完 — 命中代表 reward / discounts / done 異常
                #           (如果 next_target_quantiles 在 stage1b 是 finite 的話)
                self._assert_finite("stage1c_target_combine", "target_quantiles", target_quantiles)
            _dbg("[train_step] target branch done")

            _dbg("[train_step] before current backbone")
            features = self.backbone.get_features(state)
            _dbg_tensor("train_step.features", features)
            _dbg("[train_step] after current backbone")
            # Stage 2: current backbone forward — 命中代表 backbone weights 或 state 已壞
            #          (跟 stage1a 互相對照,可以判斷壞的是 backbone 還是 state)
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
            # Stage 3: q_network forward (FQF head 四個輸出分別檢)
            #   - fraction_probs 壞 → fraction_proposal / softmax 入口問題
            #   - tau_hats 壞     → cumsum / mean(基本上 follow fraction_probs)
            #   - quantiles 壞    → cosine_embedding 或 value_head 問題
            #   - q_values 壞     → 上面任一條
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
            # Stage 4a: quantile huber loss — 命中通常代表 chosen_quantiles 或 target_quantiles
            #            其中一個極端(如 td² overflow),內部 torch.where 雖能選 finite 分支,
            #            但 backward 經 0×inf 仍會在 stage5 噴 NaN 到 grad
            self._assert_finite("stage4a_quantile_loss", "per_sample_quantile_loss", per_sample_quantile_loss)

            entropy = -(fraction_probs * torch.log(fraction_probs + 1e-8)).sum(dim=1, keepdim=True)
            # Stage 4b: entropy — 命中代表 fraction_probs 含 NaN(stage3 應該先抓到)
            self._assert_finite("stage4b_entropy", "entropy", entropy)

            per_sample_loss = per_sample_quantile_loss - FQF_ENTROPY_COEF * entropy
            loss = (is_weights * per_sample_loss).mean()
            _dbg_tensor("train_step.loss", loss)
            _dbg_tensor("train_step.td_error", td_error)

            # FQF distribution-health diagnostics (cheap, computed inside no_grad).
            with torch.no_grad():
                # 留作 tensor,實際 .item() 在後面 diagnostic 階段跟其他 stats 一起 batch。
                fpn_norm_entropy_t = entropy.mean() / math.log(NUM_FQF_FRACTIONS)
                fpn_tau_std_t = tau_hats.std(dim=1).mean()

            # Stage 4c: 最終 loss — 既有的 NaN check,訊息升級成 stage tag 格式
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
                raise  # 其他 RuntimeError 交給外層 try/except 寫 traceback
            _dbg("[train_step] after backward")
            _dbg_mem("train_step after backward")
            all_params = list(self.backbone.parameters()) + list(self.q_network.parameters())
            if extra_params_to_clip is not None:
                all_params += list(extra_params_to_clip)
            _dbg("[train_step] before grad-finite check")
            # Stage 5: backward 之後 — 命中代表 backward 路徑產生 NaN/Inf 梯度
            #          常見原因:torch.where(td²) 在 td 過大時 backward 經 0×inf
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

            # Stage 6: clip 之後再掃一次 grad — 抓 clip 內部 0×inf
            #          (理論上 stage5 已先攔 inf,但 clip 自己 in-place 寫的也要驗一次)
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
            # Pre-step v-clamp:HW bit flip 把 v 翻成負 → sqrt(neg)=NaN → 下一步 weight=NaN。
            # 在 optimizer.step() 之前 clamp,讓 Adam 永遠看到 v >= 0 的 invariant。
            # 命中時詳細寫到 stdout / io_log / vclamp_events.log / TB,事後 grep 統計頻率。
            self._clamp_optimizer_v_and_log()
            _dbg("[train_step] before optimizer.step")
            self.optimizer.step()
            _dbg("[train_step] after optimizer.step")
            _dbg_mem("train_step after optimizer.step")

            # Stage 7: optimizer.step 之後掃 weight — ★ 本次失敗最可能的源頭 ★
            #          finite grad 進 AdamW 卻產生 NaN weight,常見原因:
            #          (a) v 接近 denormal underflow → sqrt(v)+eps 異常小 → 巨大 update
            #          (b) fused/non-fused kernel 罕見數值 edge case
            #          (c) 硬體 transient bit flip(機率極低)
            #          命中時:weight 已壞,但這一步的 grad / m / v 還在 optimizer state 裡,
            #                  可以離線分析(crash 後 atexit 會把 optimizer state 寫到 .crash 檔)
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
            # ── 非 backward 區段的 CUDA crash（forward / loss / grad-check / clip / optimizer.step / 結尾 sync）──
            # tag 成 "non-backward CUDA crash" 方便和 backward 的 grep 區分，用來統計撞牆位置。
            # 其他 (非 CUDA error 的) 例外才當真錯誤，寫 traceback 後往上丟。
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

        # ── Weight-drift snapshot — independent gating ──
        # 跟 DIAGNOSTIC_LOG_EVERY 解耦,可以設成任何值(不需要是 10 的倍數)。
        # init_distance / rolling_distance 是 Python float(_snapshot_distance 內部已 .item()),
        # 直接 add_scalar 不需要再 sync。
        if self.total_it % WEIGHT_DISTANCE_LOG_EVERY == 0:
            current_snapshot = self._capture_trainable_weight_snapshot()
            init_distance = self._snapshot_distance(current_snapshot, self._init_weight_reference)
            rolling_distance = self._snapshot_distance(current_snapshot, self._rolling_weight_reference)
            self._rolling_weight_reference = current_snapshot
            self._rolling_weight_reference_step = self.total_it
            self.tb_writer.add_scalar("weights/delta_from_init", init_distance, self.total_it)
            self.tb_writer.add_scalar("weights/delta_from_prev_window", rolling_distance, self.total_it)

        # ── Diagnostic / logging — only every DIAGNOSTIC_LOG_EVERY steps ──
        # 把 io_log / TB / no_grad stats 區塊降頻;其他 step 直接 return None,
        # 省下 ~40 個隱式 GPU→CPU sync。
        if self.total_it % DIAGNOSTIC_LOG_EVERY != 0:
            return None

        with torch.no_grad():
            # 把所有要拉的 scalar 合成一個 tensor、一次 .cpu() 搬回。
            # 從 16 個 .item() (= 16 個 GPU→CPU sync) 變成 1 個 sync。
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

            # 以下 .item() / .tolist() 留著 — 只在 diagnostic 步驟跑,成本可接受。
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

        # raw reward rolling mean(來源:training_history._step_rewards,由
        # store_transition 維護;跟 v2 / v3 共用同一個 method)。
        real_reward_mean = self.training_history.avg_step_reward()

        # ── TensorBoard scalars ──
        step = self.total_it
        self.tb_writer.add_scalar("train/Q_loss", loss_value, step)
        self.tb_writer.add_scalar("train/q_mean", q_mean, step)
        self.tb_writer.add_scalar("train/real_reward_mean", real_reward_mean, step)
        self.tb_writer.add_scalar("train/q_taken_std", q_taken_std, step)
        self.tb_writer.add_scalar("train/q_max", q_all_max, step)
        self.tb_writer.add_scalar("train/q_min", q_all_min, step)
        self.tb_writer.add_scalar("train/target_q_mean", target_q_mean, step)
        self.tb_writer.add_scalar("train/target_q_std", target_q_std, step)
        self.tb_writer.add_scalar("train/td_error_mean", td_error_mean, step)
        self.tb_writer.add_scalar("train/td_error_max", td_error_max, step)
        self.tb_writer.add_scalar("train/frac_huber_clipped", frac_huber_clipped, step)
        self.tb_writer.add_scalar("train/epsilon", self.epsilon, step)
        self.tb_writer.add_scalar("fpn/norm_entropy", fpn_norm_entropy, step)
        self.tb_writer.add_scalar("fpn/tau_std", fpn_tau_std, step)
        self.tb_writer.add_scalar("grad/total_pre_clip", grad_norm_total_value, step)
        self.tb_writer.add_scalar("grad/total_post_clip", grad_post_total, step)
        self.tb_writer.add_scalar("grad/clip_percent", grad_clip_percent, step)
        self.tb_writer.add_scalar("grad/clip_excess_norm", grad_clip_excess_norm, step)
        self.tb_writer.add_scalar("grad/clip_excess_ratio", grad_clip_excess_ratio, step)
        self.tb_writer.add_scalar("grad_pre/backbone", backbone_pre, step)
        self.tb_writer.add_scalar("grad_pre/head", head_pre, step)
        self.tb_writer.add_scalar("grad_post/backbone", backbone_post, step)
        self.tb_writer.add_scalar("grad_post/head", head_post, step)
        self.tb_writer.add_scalar("grad_pre/extras", extras_pre, step)
        self.tb_writer.add_scalar("grad_post/extras", extras_post, step)
        # check/ namespace — 驗證用,不是核心訓練指標。
        self.tb_writer.add_scalar("check/is_weight_mean", is_weight_mean, step)
        self.tb_writer.add_scalar("check/is_weight_min", is_weight_min, step)
        # IS weight 是 max-normalized 所以 max 恆為 1.0,ratio = 1/min。
        # 健康範圍 < 10;若 > 100 代表 IS 公式可能又 broken(死條目 weight 爆炸之類)。
        self.tb_writer.add_scalar("check/is_weight_ratio", 1.0 / max(is_weight_min, 1e-12), step)
        # 每筆 entry 平均被抽到幾次。數值單調隨訓練步數成長;高 mean 代表 PER 集中度高,batch 多樣性低。
        self.tb_writer.add_scalar("check/mean_sample_count", self.replay_buffer.mean_sample_count(), step)

        # Per-layer weight/grad norms — collected pre-clip inside the try
        # block above; written here so we never leave orphan rows on a
        # CUDA-recovery early return.
        if backbone_norm_snapshots is not None:
            self._write_backbone_weight_norm_snapshots(backbone_norm_snapshots, step)

        # Buffer composition — diagnoses replay drift over time.
        for bucket_name, count in self.replay_buffer.bucket_sizes().items():
            self.tb_writer.add_scalar(f"buffer/bucket_{bucket_name}", count, step)
        self.tb_writer.add_scalar("buffer/total_size", self.replay_buffer.size(), step)

        return {
            "Q_loss": loss_value,
            "q_mean": q_mean,
        }

    def reset_episode(self):
        self._flush_n_step_buffer()
        self.clear_blocked_actions()

    # ──────────────────────────── diagnostics helpers ──────────────────────

    def _write_metric_docs(self):
        """把 metric 解讀表寫到 TensorBoard 的 TEXT 分頁,只寫一次。"""
        is_weight_mean_doc = (
            "**`check/is_weight_mean`** — PER importance-sampling weight 平均值 "
            "(已 normalize by max,所以 max 恆為 1.0,只看 mean)。\n\n"
            "| 數值區間 | 代表 | 該擔心嗎? |\n"
            "|---|---|---|\n"
            "| 接近 1.0 | priorities 很平均,PER 幾乎退化成 uniform replay | "
            "PER 沒在工作,可能 TD-error 都差不多 |\n"
            "| 中間 (0.3 ~ 0.8) | 健康,有偏抽但 bias correction 足夠 | 正常 |\n"
            "| 接近 0 | priorities 高度集中,少數樣本主宰梯度 | "
            "可能 over-fit 那幾個 hard sample |\n\n"
            "搭配 β annealing 看走勢:β 上升時 mean 應緩慢下降;若反向上升 "
            "代表 priority 分佈在塌掉。"
        )
        self.tb_writer.add_text("docs/is_weight_mean", is_weight_mean_doc, 0)

        is_weight_ratio_doc = (
            "**`check/is_weight_ratio`** — IS weight 的 max/min 比例 (= 1.0 / min,"
            "因為 IS weight 已 normalize by max → max 恆為 1.0)。**這是抓 IS bug 的"
            "頭號指標**。\n\n"
            "| 數值 | 代表 | 該擔心嗎? |\n"
            "|---|---|---|\n"
            "| < 10 | priorities 分佈健康,IS 修正有效,batch 內每筆都實質參與梯度 | 正常 |\n"
            "| 10 ~ 100 | 部分樣本 IS weight 被壓得低,梯度貢獻不均 | "
            "注意,可能 priority 分佈過度 skewed |\n"
            "| > 100 | 嚴重失衡,大部分 batch 名額幾乎沒貢獻梯度 | "
            "**紅燈**,檢查 `_selection_priorities` 與 `_sample_from_bucket` 是否同步、"
            "死條目 priority_min floor 是否被繞過 |\n\n"
            "歷史:修 IS bug 之前(buggy 版本)ratio 常常 > 10^4 — 死條目 raw "
            "_effective_priority=0 在 IS 公式裡撞到 `(1e-10)^(-β) ≈ 10^4` 變 max,"
            "把活條目的 weight 壓到 ~1e-5。"
        )
        self.tb_writer.add_text("docs/is_weight_ratio", is_weight_ratio_doc, 0)

    def _assert_finite(self, stage, name, tensor):
        """訓練流程的 NaN/Inf probe:命中就 raise,訊息含 stage / tensor / step。

        只檢 floating-point tensor — int/bool 用 isfinite 無意義。
        每個 stage boundary 呼叫一次,GPU sync 開銷 ~50μs,可常駐。
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
        """掃 state_dict 內所有 floating-point tensor,回傳 [(label, msg)] 列表。

        不 raise — caller 自己決定要 raise(load 路徑)還是改寫 .crash 檔(save 路徑)。
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
        """掃 optimizer state_dict (nested: state[pid][key])。

        檢查兩種異常:
        (a) 任何 tensor 含 NaN/Inf
        (b) exp_avg_sq < 0 — Adam 的 second moment 數學上不可能為負,出現必為
            bit-level corruption(sign-bit flip 等),會讓 sqrt(v) 噴 NaN

        回傳 [(label, message)] 列表。
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
        """Pre-step belt-and-suspenders:Adam 的 exp_avg_sq(v)若有負值就 in-place clamp。

        動機:v 數學上恆 >= 0(β2·v_old + (1-β2)·grad²,兩項都非負)。若觀察到負值,
              壓倒性是 consumer GPU 沒 ECC 的 VRAM transient bit flip(本案就是 sign-bit
              翻轉:-5.66e-7 vs |v|.max()=9.03e-7 同數量級)。若不清掉,AdamW 下一步
              算 sqrt(negative)=NaN,接著把 weight 寫成 NaN,就是 stage 7 攔到的爆點。

        Self-healing(不 raise) — 把 HW transient 變成可繼續訓練的小事件;但每次命中
        詳細寫四個地方,長期 grep 可以建出頻率/位置分布:
          - stdout (訓練 console 立刻可見)
          - self._io_log (跟其他 step 的 diagnostic 混在一起,容易對時間軸)
          - models/stage1_transformer/vclamp_events.log (專屬 event log,好 grep)
          - TensorBoard (vclamp/elements_this_step / elements_total / params_this_step)

        命中位置會同步把 paired exp_avg(m)歸零,避免被汙染的 momentum 殘留繼續
        把剛 reset 的 weight element 推向奇怪方向。
        """
        detections = []
        # 用 named_parameters 是為了拿 human-readable 名稱寫 log
        # (optimizer.state 的 key 是 param 物件本身,沒名字)
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
                # 最多取 5 個位置 + 數值寫 log,避免大規模 corruption 時 log 爆炸
                sample_pos = neg_mask.nonzero(as_tuple=False)[:5].tolist()
                sample_vals = v[neg_mask][:5].tolist()
                full_name = f"{source_name}.{pname}"
                detections.append((full_name, tuple(v.shape), neg_n, sample_pos, sample_vals))
                # In-place clamp v >= 0
                v.clamp_(min=0)
                # 同位置清 m
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

        # 1) stdout — 訓練 console 立刻看到
        for line in lines:
            print(line)
        # 2) io_log — 跟訓練的 step diagnostic 混在一起對時間軸
        try:
            self._io_log.write("\n".join(lines) + "\n")
            self._io_log.flush()
        except Exception:
            pass
        # 3) 專屬 event log — 寫到當下 hour 資料夾(跟 io_log 同位置,方便對時)
        try:
            with (self.current_archive_dir / "vclamp_events.log").open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass
        # 4) TensorBoard — 視覺化命中時間軸
        if not hasattr(self, "_vclamp_total"):
            self._vclamp_total = 0
        self._vclamp_total += total
        try:
            self.tb_writer.add_scalar("vclamp/elements_this_step", total, self.total_it)
            self.tb_writer.add_scalar("vclamp/elements_total", self._vclamp_total, self.total_it)
            self.tb_writer.add_scalar("vclamp/params_this_step", len(detections), self.total_it)
        except Exception:
            pass

    # ──────────────────────────── archive directory / hour rollover ──────
    # 目錄管理本體(session_dir / hour_index / current_archive_dir + 翻頁)在
    # model_structure.archive_manager.SessionArchiveManager。這裡只剩 agent 自己
    # 的耦合點:io_log banner 內容、翻頁時要 swap 哪些檔。
    #
    # current_archive_dir 對外是 read-only @property delegate,讓既有的呼叫端
    # (例如 train_stage1_simple.py 的 `agent.current_archive_dir`)不用改。

    @property
    def current_archive_dir(self) -> Path:
        return self.archive.current_archive_dir

    def _build_io_log_banner(self, path: Path) -> list[str]:
        """RolloverTextLog banner:寫 session/hour 起始時間 + 檔案路徑。"""
        return [
            f"Session started: {self.archive.session_start.isoformat()}",
            f"Hour {self.archive.hour_index:02d} started: {datetime.datetime.now().isoformat()}",
            f"Path: {path}",
        ]

    def _on_archive_rollover(self, new_dir: Path) -> None:
        """SessionArchiveManager rollover callback:翻頁時 swap io_log + dbg logger。

        - io_log 先 inline 寫一行 rollover footer 再 swap;這樣舊檔尾巴有 marker,
          新檔開頭有 RolloverTextLog 自己寫的 banner。
        - vclamp_events.log / training_log.csv 是 lazy-open(每次寫才開),
          會自動跟著 self.archive.current_archive_dir,不用在這裡顯式處理。
        - canonical *.pth 不動;_save_model 下次被叫到時自然把 archive snapshot
          寫到新的 hour 資料夾。
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
                self.tb_writer.add_scalar(f"weight_norm/{tag_prefix}", weight_norm, global_step)
            if grad_norm is not None:
                self.tb_writer.add_scalar(f"grad_norm/{tag_prefix}", grad_norm, global_step)

    def _close_tb_writer(self):
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.close()

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def on_episode_end(self):
        self._flush_n_step_buffer()
        # episode_count 已在 log_episode_metrics() 內 history.record() 時自動
        # 從 training_history 推導出來,這裡不再 += 1(它是 @property delegate)。
        # epsilon 也已在 log_episode_metrics() 透過 controller.update 更新好。
        self.tb_writer.add_scalar("episode/epsilon", self.epsilon, self.episode_count)
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
        """記錄一場 episode 結果,並依 rolling win rate 更新 epsilon。

        與 v3 / v2 介面一致:訓練腳本應在 `on_episode_end()` 之前呼叫一次。
        流程:1) 結果 (含 total_reward / steps) 記到 TrainingHistory,
        2) 從 history 取 rolling win rate,
        3) 把 win rate 餵給 controller 算 next epsilon。

        total_reward / steps 是 keyword-only。舊呼叫端不傳的話,history 內的
        reward/steps 統計就會是 0 (對 win_rate / ε 衰減無影響)。
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

        self.tb_writer.add_scalar("episode/reward_mean",        float(reward_mean),        ep_idx)
        self.tb_writer.add_scalar("episode/invalid_click_rate", float(invalid_click_rate), ep_idx)

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

    def _save_model(self):
        # Flush TB before saving — pair the on-disk model checkpoint with the
        # matching TB scalars from this point in training.
        self.tb_writer.flush()
        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        backbone_sd = self.backbone.state_dict()
        qnet_sd = self.q_network.state_dict()
        qtarget_sd = self.q_target.state_dict()

        # optimizer state 只含 optimizer / total_it / episode_count / algorithm / epsilon。
        # Episode 結果累積/查詢搬到 TrainingHistory,寫到獨立檔 training_history.pth
        # (見下方 torch.save)。
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "total_it": self.total_it,
            "episode_count": self.episode_count,
            "algorithm": "FQF",
        }
        payload.update(self.epsilon_controller.state_dict())
        training_history_sd = self.training_history.state_dict()

        # ── RNG state snapshot ─────────────────────────────────────────
        # Save 全部 4 個 RNG source 的 state,給 restart 後完整還原。
        # 不存的話,restart 會用全新的 seed,讓 Minesweeper 板生成、ε-random
        # action、replay buffer 取樣等等都跟 save 那刻不同 → trajectory 跟
        # buffer 內舊 transition 屬於不同分布 → 立即訓練時 bootstrap 矛盾
        # 造成 win rate drop。實驗(seed 42 跑兩次得到 bit-identical 結果)
        # 確認 RNG 控制可達成完整 determinism。
        payload["rng_state"] = {
            "python_random": random.getstate(),
            "numpy":         np.random.get_state(),
            "torch_cpu":     torch.get_rng_state(),
            "cuda":          torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Save-time NaN probe:掃所有 state_dict,任一壞就拒絕 overwrite canonical 檔。
        # 動機:atexit 在 NaN crash 後也會被 trigger,沒這層保護就會把磁碟上的好 checkpoint
        #      蓋成壞的(就是失敗 run 把 1 個 NaN 寫進 backbone.pth 的那條路徑)。
        # 也掃 optimizer state — Adam 的 v 為負會在下次 load 後立刻引爆 NaN weight。
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

        # 同一份也寫到當下 hour 資料夾,做歷史快照(同一小時內多次 save 會 overwrite,
        # 留下「該小時最後一次 save」的狀態 — 滿足「保留每次的訓練結果」)
        try:
            archive = self.current_archive_dir
            archive.mkdir(parents=True, exist_ok=True)
            torch.save(backbone_sd,         archive / "backbone.pth")
            torch.save(qnet_sd,             archive / "fqf_network.pth")
            torch.save(qtarget_sd,          archive / "fqf_target.pth")
            torch.save(payload,             archive / "optimizer_state.pth")
            torch.save(training_history_sd, archive / "training_history.pth")
        except Exception as exc:
            # archive 失敗不擋 canonical save 的成功
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
        """Load path 決策:canonical 存在就用它;不存在就 fallback 到最新 archive。

        Corrupt(NaN/Inf)的 canonical 不算「不存在」,會在 _raise_if_corrupt 那層攔下,
        不會 silent 走 archive — 因為 corrupt 通常代表你需要主動處理(sanitize / rollback)。

        Archive 掃描邏輯在 SessionArchiveManager.find_latest_archive;這裡只負責
        canonical 與 archive 之間的優先序與 console message。
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
        """Load TrainingHistory:獨立檔 training_history.pth 優先,
        舊扁平 optimizer state 是 fallback (一次性 migration)。

        Canonical 不存在會自動 fallback 到最新 archive (走 _resolve_load_path),
        跟其他 checkpoint 的 load 路徑一致。
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
            print("[FQF] Migrated legacy training history from optimizer state")

    def _raise_if_corrupt(self, label, path, state_dict):
        """Load-time NaN probe — checkpoint 含 NaN/Inf 就 raise,阻止 silent resume。

        將 raise 抽出來放到 try/except 之外,避免被原本「捕例外印 message 就吞掉」的
        錯誤處理蓋掉。
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
                # Load-time probe for optimizer state — 掃 NaN/Inf 跟 exp_avg_sq < 0
                # (本案就是後者:Adam 的 v 不能為負,出現必為 bit-level corruption,
                #  會讓 AdamW 算 sqrt(negative)=NaN 後把 weight 寫壞)
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
                    # episode_count 不再直接 set — 它是 @property delegate 到
                    # training_history.total_episodes,後者由獨立 .pth 檔還原。
                    # 舊 opt_payload 裡的 "episode_count" 直接忽略 (跟 history
                    # 的 total_episodes 重複了)。
                    # Controller 只剩 epsilon 一個 key;TrainingHistory 走獨立檔,
                    # 舊 checkpoint 把 history 攤平存在 opt_payload 的情況用 fallback。
                    self.epsilon_controller.load_state_dict(opt_payload)
                    self._load_training_history(legacy_state=opt_payload)
                    print(
                        f"[FQF] Loaded optimizer: total_it={self.total_it},"
                        f" episode={self.episode_count}, epsilon={self.epsilon:.4f},"
                        f" total_episodes={self.training_history.total_episodes}"
                    )
                    # ── RNG state restore ──────────────────────────────
                    # 還原 save 那刻的 random / numpy / torch / cuda RNG state,
                    # 讓 restart 後 trajectory 跟 save 那刻完整延續(避免 buffer
                    # 內舊 transition 跟新 trajectory 分布不一致導致 bootstrap 矛盾)。
                    # 舊 checkpoint 沒 rng_state 欄位時 silent skip,保持向下相容。
                    # 注意:torch.load(map_location=device) 會把整個 payload 的
                    # tensor 都搬到 device。torch.set_rng_state() 一定要 CPU
                    # ByteTensor,所以要 .cpu() 後再傳。
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
                                # cuda RNG state 是 list[Tensor],每張 GPU 一個。
                                # 強制每個 element 都搬到 CPU。
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

        # 新檔名 replay_buffer.pth;若不存在但有舊檔 training_state.pth(rename 前的版本),
        # 仍從舊檔載入(下次 save 會寫到新檔名,舊檔可以手動刪)
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
