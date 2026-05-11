import atexit
import datetime
import logging as _logging
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_structure.transformer_shared import EncoderDecoderTransformer, FQFQNetwork, FixedSinusoidalPositionEmbedding
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from model_structure.optimizer_factory import build_fqf_optimizer
import model_structure.CategorizedReplayBuffer as _crb_module


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 40
GRID_STATE_CHANNELS = 12
PER_CAPACITY = 2048
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
SAVE_CAPACITY = 512
SAVE_EVERY_N_EPISODES = 200
TARGET_UPDATE_FREQ = 1000
N_STEP = 1
NUM_FQF_FRACTIONS = 8
FQF_ENTROPY_COEF = 1e-3

TRANSFORMER_MODEL_PATH = Path("./models/stage1_transformer")
TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_DROPOUT = 0.1

# ── CUDA SDP backend (v3 經驗) ──────────────────────────────────────────
# PyTorch 2.x 在某些 CUDA / GPU 組合上，nn.TransformerEncoderLayer 走 flash 或
# mem-efficient SDP kernel 會丟 "illegal instruction"。math backend 慢但穩。
if device.type == "cuda":
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass  # 失敗就讓 PyTorch 用預設值，不噴 CMD（之後若有 logger 會記錄訓練錯誤）

# ── debug logger (寫到檔案，不噴 CMD；DEBUG_CUDA_SAMPLE=True 才會啟用) ──
# True 時會插入 cuda.synchronize + 寫 log，會拖慢訓練；只在除錯 CUDA error 時開。
DEBUG_CUDA_SAMPLE = False

_DBG_LOG_PATH = TRANSFORMER_MODEL_PATH / "cuda_debug.log"
_DBG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_dbg_logger = _logging.getLogger("cuda_dbg.stage1")
_dbg_logger.setLevel(_logging.DEBUG)
_dbg_logger.propagate = False  # 不往 root logger 傳，避免 CMD 也印
if not _dbg_logger.handlers:
    _fh = _logging.FileHandler(_DBG_LOG_PATH, mode="a", encoding="utf-8")
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
    huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
    tau = tau_hats.unsqueeze(2)
    quantile_weight = (tau - (td.detach() < 0).float()).abs()
    loss = (quantile_weight * huber).sum(dim=2).mean(dim=1, keepdim=True)
    if return_stats:
        frac_clipped = (abs_td > 1.0).float().mean().item()
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

        self.replay_buffer = CategorizedReplayBuffer(
            max_size=PER_CAPACITY,
            storage_mode="ram",
            win_threshold=MINESWEEPER_REWARD_CONFIG.replay_win_threshold,
            lose_threshold=MINESWEEPER_REWARD_CONFIG.replay_lose_threshold,
            invalid_threshold=MINESWEEPER_REWARD_CONFIG.replay_invalid_threshold,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START
        )
        self.total_it = 0
        self.episode_count = 0
        self.n_step = N_STEP
        self.n_step_gamma = MINESWEEPER_REWARD_CONFIG.gamma
        self.n_step_buffer = deque()

        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay_episodes = 5000

        # episode-scoped blocked actions (v3 風格)：點過的格子在本 episode 內 mask 掉
        self.blocked_actions: set[int] = set()

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self._io_log = open(TRANSFORMER_MODEL_PATH / "train_io_log.txt", "a", encoding="utf-8")
        self._io_log.write(f"\n{'=' * 60}\n")
        self._io_log.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self._io_log.write(f"{'=' * 60}\n")
        self._io_log.flush()

        self.try_load_model()
        atexit.register(self.save_persistent)
        atexit.register(self._close_io_log)

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
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_it += 1

        try:
            _dbg(f"[train_step] ENTER total_it={self.total_it} buf_size={self.replay_buffer.size()}")
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
                next_target_quantiles = next_target["quantiles"][
                    torch.arange(batch_size, device=device), best_flat
                ]
                _dbg_tensor("train_step.next_target_quantiles", next_target_quantiles)
                target_quantiles = reward + (1 - done) * discounts * next_target_quantiles
                _dbg_tensor("train_step.target_quantiles", target_quantiles)
            _dbg("[train_step] target branch done")

            _dbg("[train_step] before current backbone")
            features = self.backbone.get_features(state)
            _dbg_tensor("train_step.features", features)
            _dbg("[train_step] after current backbone")
            q_output = self.q_network(features)
            q_2d = q_output["q_values"]
            q_quantiles = q_output["quantiles"]
            tau_hats = q_output["tau_hats"]
            fraction_probs = q_output["fraction_probs"]
            _dbg_tensor("train_step.q_2d", q_2d)
            _dbg_tensor("train_step.q_quantiles", q_quantiles)
            _dbg_tensor("train_step.tau_hats", tau_hats)
            _dbg_tensor("train_step.fraction_probs", fraction_probs)
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

            per_sample_quantile_loss = _quantile_huber_loss(
                current_quantiles=chosen_quantiles,
                target_quantiles=target_quantiles.detach(),
                tau_hats=tau_hats.detach(),
            )
            entropy = -(fraction_probs * torch.log(fraction_probs + 1e-8)).sum(dim=1, keepdim=True)
            per_sample_loss = per_sample_quantile_loss - FQF_ENTROPY_COEF * entropy
            loss = (is_weights * per_sample_loss).mean()
            _dbg_tensor("train_step.loss", loss)
            _dbg_tensor("train_step.td_error", td_error)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected before backward: {loss.item()} "
                    f"(quantile={per_sample_quantile_loss.mean().item()}, entropy={entropy.mean().item()})"
                )

            self.optimizer.zero_grad()
            _dbg("[train_step] before backward")
            # ── Belt-and-suspenders：進 inner try 之前先排空 forward queue ──
            # isfinite(loss) 那行雖然會 sync 一次、理論上已經 drain 過 forward 的 async error，
            # 但 CUDA 的 sticky error state 有可能讓 forward 的 error 在 backward 內部被 check 到，
            # 然後被誤標成 "backward CUDA crash"。先 sync 一次保證 inner try 開始時 GPU queue 是空的，
            # 這樣 inner except 收到的 error 一定是 backward 自己產生的、不會被 forward 殘留 error 誤標。
            # 成本：每個 train_step 多一個 sync (~5-20 μs)，極小。
            if device.type == "cuda":
                torch.cuda.synchronize()
            # ── 精準包 backward：tag 成 "backward CUDA crash"，方便事後 grep 統計
            # 撞牆比例（是 backward kernel bug 還是 forward / optimizer 的）。
            # synchronize 是必要的：沒設 CUDA_LAUNCH_BLOCKING 時 backward 是 async，
            # kernel error 會延後到下一個 sync 點才暴露，要在這 sync 一次才能讓 try/except 真的攔到 backward。
            try:
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
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
            for name, param in list(self.backbone.named_parameters()) + list(self.q_network.named_parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"Non-finite gradient detected in parameter: {name}")
            _dbg("[train_step] after grad-finite check")
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            _dbg("[train_step] after clip_grad_norm_")
            _dbg("[train_step] before optimizer.step")
            self.optimizer.step()
            _dbg("[train_step] after optimizer.step")
            _dbg_mem("train_step after optimizer.step")
            # 在離開 try 前再 sync 一次：optimizer.step 會 queue Adam kernel，
            # 若有 async error 留到外面就接不到了。一次 sync 確保 try 範圍內就把錯誤撈出來。
            if device.type == "cuda":
                torch.cuda.synchronize()
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

        self.replay_buffer.update_priorities(per_indices, td_error.squeeze(-1).cpu().numpy())

        if self.total_it % TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

        with torch.no_grad():
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

            q0_flat = q_2d[0].view(-1)
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

            q_mean = q_taken.mean().item()

        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  State:  unrevealed={unrevealed} | revealed={revealed} | flagged={flagged}"
            f" | numbers={num_counts}\n"
            f"  Batch:  rewards={dict(reward_counts)} | top_actions=[{top3_str}]\n"
            f"  Q-top5: {top5_info}\n"
            f"  Q-val:  taken_mean={q_mean:.4f}"
            f" | all: min={q0_flat.min().item():.4f} max={q0_flat.max().item():.4f}\n"
            f"  FQF:    loss={loss.item():.4f} | tau_entropy={entropy.mean().item():.4f}"
            f" | epsilon={self.epsilon:.4f}\n"
            f"---\n"
        )
        self._io_log.flush()

        return {
            "Q_loss": loss.item(),
            "q_mean": q_mean,
        }

    def reset_episode(self):
        self._flush_n_step_buffer()
        self.clear_blocked_actions()

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def on_episode_end(self):
        self._flush_n_step_buffer()
        self.episode_count += 1
        decay_progress = min(self.episode_count / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_min + (0.3 - self.epsilon_min) * (1.0 - decay_progress)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(
                f"[FQF] Periodic save at episode {self.episode_count}"
                f" | epsilon={self.epsilon:.4f}"
            )
            self.save_persistent()

    def _save_model(self):
        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(), TRANSFORMER_MODEL_PATH / "backbone.pth")
        torch.save(self.q_network.state_dict(), TRANSFORMER_MODEL_PATH / "fqf_network.pth")
        torch.save(self.q_target.state_dict(), TRANSFORMER_MODEL_PATH / "fqf_target.pth")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
                "algorithm": "FQF",
            },
            TRANSFORMER_MODEL_PATH / "optimizer_state.pth",
        )

    def save_persistent(self):
        buf = self.replay_buffer
        total = buf.size()
        if total == 0:
            return

        all_entries = buf.get_all_entries()
        
        # Sort by priority locally to do Top-K before saving
        all_entries_sorted = sorted(
            all_entries, 
            key=lambda e: buf._effective_priority(e), 
            reverse=True
        )
        
        if len(all_entries_sorted) > SAVE_CAPACITY:
            all_entries_sorted = all_entries_sorted[:SAVE_CAPACITY]

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "persistent_entries": all_entries_sorted,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
            },
            TRANSFORMER_MODEL_PATH / "training_state.pth",
        )

        saved_rewards = defaultdict(int)
        saved_buckets = defaultdict(int)
        for entry in all_entries_sorted:
            saved_rewards[entry.get("tail_reward", entry["reward"])] += 1
            saved_buckets[entry["reward_type"]] += 1
        print("--- save info ---------------")
        print(f"[FQF] Persistent save: {len(all_entries_sorted)} entries")
        print(f"[FQF] Tail reward distribution: {dict(saved_rewards)}")
        print(f"[FQF] Reward bucket distribution: {dict(saved_buckets)}")
        print("--- save end ---------------")

    def try_load_model(self):
        backbone_path = TRANSFORMER_MODEL_PATH / "backbone.pth"
        if backbone_path.exists():
            try:
                backbone_state = torch.load(backbone_path, map_location=device)
                incompatible = self.backbone.load_backbone_state(backbone_state, strict=False)
                print("[FQF] Loaded Backbone")
                if incompatible.missing_keys:
                    print(f"[FQF] Backbone missing keys: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    print(f"[FQF] Backbone unexpected keys: {incompatible.unexpected_keys}")
            except Exception as exc:
                print(f"[FQF] Failed to load Backbone: {exc}")

        q_path = TRANSFORMER_MODEL_PATH / "fqf_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device))
                print("[FQF] Loaded FQF-Network")
            except Exception as exc:
                print(f"[FQF] Failed to load FQF-Network: {exc}")
        elif (TRANSFORMER_MODEL_PATH / "q_network.pth").exists():
            print("[FQF] Skip legacy q_network.pth because DDQN head shape is incompatible")

        q_target_path = TRANSFORMER_MODEL_PATH / "fqf_target.pth"
        if q_target_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(q_target_path, map_location=device))
                print("[FQF] Loaded FQF-Target")
            except Exception as exc:
                print(f"[FQF] Failed to load FQF-Target: {exc}")
        elif (TRANSFORMER_MODEL_PATH / "q_target.pth").exists():
            print("[FQF] Skip legacy q_target.pth because DDQN head shape is incompatible")

        opt_path = TRANSFORMER_MODEL_PATH / "optimizer_state.pth"
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.optimizer.load_state_dict(state["optimizer"])
                self.total_it = state["total_it"]
                self.episode_count = state.get("episode_count", 0)
                if "epsilon" in state:
                    self.epsilon = state["epsilon"]
                print(
                    f"[FQF] Loaded optimizer: total_it={self.total_it},"
                    f" episode={self.episode_count}, epsilon={self.epsilon:.4f}"
                )
            except Exception as exc:
                print(f"[FQF] Failed to load optimizer state: {exc}")

        training_state_path = TRANSFORMER_MODEL_PATH / "training_state.pth"
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
