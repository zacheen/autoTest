import random
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG


class CategorizedReplayBuffer:
    """A generic Categorized Replay Buffer.

    Supports:
    1. Categorized Bucketing based on reward thresholds (win / lose / invalid / progress).
    2. RAM-based storage (for small states) or Disk-backed storage (for memory-heavy images).
    3. Priority decay (accumulator pattern in ``_effective_priority``):
       - age_decay    : suppresses stale entries over time (always on)
       - sample_decay : optional Method B defense against stochastic-transition traps via
                        repeated-pick counting, toggled by ``enable_sample_decay`` (default OFF).
       - spread_decay : optional defense via FQF quantile-spread signal, toggled by
                        ``enable_spread_decay`` (default OFF, intended to be latched ON by the
                        agent once win_rate > 40%). Wide quantile spread = model believes the
                        outcome is stochastic → decay more.
    4. 50% balanced + 50% PER allocation for BOTH eviction (``top_k_balanced``) and
       sampling (``sample``):
       - Balanced half: each class gets a soft floor (12.5% of slots with 4 classes).
       - PER half: pure priority-weighted, cross-class. Slack from empty/starved
         classes flows here automatically.
    """

    REWARD_TYPES = ("win", "lose", "invalid", "progress")

    def __init__(
        self,
        max_size: int,
        storage_mode: str = "ram",
        save_dir: str = None,
        win_threshold: float = MINESWEEPER_REWARD_CONFIG.replay_win_threshold,
        lose_threshold: float = MINESWEEPER_REWARD_CONFIG.replay_lose_threshold,
        invalid_threshold: float = MINESWEEPER_REWARD_CONFIG.replay_invalid_threshold,
        overflow_margin: int = 256,
        alpha: float = 0.6,
        uniform_mix: float = 0.2,
        priority_min: float = 0.05,
        priority_max: float = 5.0,
        priority_eps: float = 1e-3,
        age_decay: float = 0.002,
        max_age: int = 2000,
        enable_sample_decay: bool = False,
        sample_decay: float = 0.05,
        enable_spread_decay: bool = False,
        spread_decay: float = 2.0,
        quota_check_class: str | None = None,
        beta_start: float = 0.4,
        balanced_ratio: float = 1.0,
    ):
        """
        Args:
            max_size: Maximum capacity for the buffer.
            storage_mode: "ram" (keeps tensors in memory) or "disk" (saves and loads to disk).
            save_dir: Must be provided if storage_mode is "disk". Target directory to persist pt files.
            win_threshold: Threshold to classify a "win" (must be >= and done=True).
            lose_threshold: Threshold to classify a "lose" (must be <= and done=True).
            invalid_threshold: Threshold to classify an "invalid" (must be < invalid_threshold).
            overflow_margin: Amount of items past max_size before a full prune is triggered.
            alpha: Priority coefficient (0 = uniform, 1 = full priority).
            uniform_mix: Ratio of completely random items selected per bucket before priority is considered.
                Applied to the BALANCED half of sample()/top_k_balanced; PER half is pure priority.
            priority_min: Lowest possible priority score an item can have.
            priority_max: Cap on initial priority values.
            priority_eps: Small epsilon added to TD Error to prevent 0 priority.
            age_decay: Decay factor removing priority based on how many inserts occurred since entry.
            max_age: Hard age limit in insert steps. Entries older than this get zero effective priority.
            enable_sample_decay: master switch for the Method-B sample-count decay (default False).
                **僅控制是否在 ``_effective_priority`` 套用 decay**;``sample_count`` 不論這個
                開關都會在 ``_sample_from_bucket`` 結尾無條件 +1(因為 ``mean_sample_count``
                這類觀測 metric 需要永遠有效的 counter)。當這個開關 False 時,buffer 行為
                等同 plain PER,sample_count 只是觀測值不影響抽樣。
            sample_decay: Method B — divide effective priority by (1 + sample_decay × sample_count).
                Only takes effect when ``enable_sample_decay=True``. ``sample_count`` 累計
                **所有抽樣路徑**(uniform / PER / safety padding 都算),不只 PER 半邊 — 任何
                被過度採樣的 entry 都該降溫,不分原因。Learned 過的 entry 自然受保護:
                low priority → low pick rate → sample_count 增長慢。
                ⚠️ 註:原始設計只計 PER 半邊(避免懲罰 uniform exploration);改為涵蓋全部
                路徑是為了統一 metric 語義。若日後啟用 Method B 發現 uniform 被誤殺,可考慮
                還原為 PER-only 計數,但需另開欄位避免破壞 metric。
            enable_spread_decay: master switch for the FQF quantile-spread decay (default False).
                Designed to be latched ON by the agent when win_rate crosses some threshold
                (e.g. > 40%), once the model has matured enough that "wide quantile spread" mostly
                reflects environment stochasticity rather than "deterministic but not yet learned".
                When False, ``quantile_spread`` is still written to entries by ``update_priorities``
                (so data is available for inspection / future latch flip), but never used in
                ``_effective_priority`` — buffer behaves like a plain PER buffer w.r.t. spread.
            spread_decay: divide effective priority by (1 + spread_decay × quantile_spread).
                Only takes effect when ``enable_spread_decay=True``. ``quantile_spread`` is the
                std of the 8 quantile values that FQF predicted for the chosen (state, action);
                it's updated whenever the entry is sampled and trained on (via the buffer's
                ``update_priorities`` extended signature). Wide spread = model thinks the outcome
                is stochastic = decay more (don't waste PER picks on inherently noisy entries).
                Narrow spread = model thinks deterministic = decay less (let PER keep learning).
                Default 2.0 calibrated from observed inference spread distribution
                (median ≈ 0.115, p90 ≈ 0.378), giving ~19% decay at typical and ~43% at p90.
            quota_check_class: name of the reward_type that ``is_class_quota_filled`` should
                gate on. If None (default), the gate requires EVERY class to reach the 12.5%
                soft-floor quota. If set to e.g. ``"win"``, only that class is checked — useful
                when one class is the known rare-event bottleneck (e.g. wins in Minesweeper)
                and you don't want pruning equilibrium to artificially delay training.
                Must be a member of ``REWARD_TYPES`` or None.
            beta_start: Initial Importance Sampling weight factor.
            balanced_ratio: 0.0~1.0,``sample()`` 裡 Phase 1 (stratified balanced)
                占 batch 的比例。剩下的給 Phase 2(cross-class pure-PER)當 fallback。
                - 1.0(default):每類各取 ``batch_size // 4`` 筆,PER 只在某類 entry
                  數不足時補位。完全避免 cross-class PER 偏向高 abs(reward) 的類。
                - 0.5:原始設計 50/50,Phase 1 取 ``batch_size // 8`` 每類,Phase 2
                  在 leftover pool 用 PER 全域競爭 50% 名額。
                - 0.0:完全沒 stratification,純 PER 全域抽。會被高 priority 類別主宰。
        """
        self.max_size = max_size
        self.storage_mode = storage_mode.lower()
        if self.storage_mode == "disk":
            if not save_dir:
                raise ValueError("save_dir must be provided if storage_mode is 'disk'")
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        self.win_threshold = win_threshold
        self.lose_threshold = lose_threshold
        self.invalid_threshold = invalid_threshold
        
        self.overflow_margin = overflow_margin
        self.alpha = alpha
        self.uniform_mix = uniform_mix
        self.priority_min = priority_min
        self.priority_max = priority_max
        self.priority_eps = priority_eps
        self.age_decay = age_decay
        self.max_age = int(max_age) if max_age is not None else 0
        # if max_age = 0, it means no "force" age remove
        self.enable_sample_decay = bool(enable_sample_decay)
        self.sample_decay = float(sample_decay)
        self.enable_spread_decay = bool(enable_spread_decay)
        self.spread_decay = float(spread_decay)
        if quota_check_class is not None and quota_check_class not in self.REWARD_TYPES:
            raise ValueError(
                f"quota_check_class must be one of {self.REWARD_TYPES} or None, "
                f"got {quota_check_class!r}"
            )
        self.quota_check_class = quota_check_class
        self.beta = beta_start
        if not 0.0 <= balanced_ratio <= 1.0:
            raise ValueError(f"balanced_ratio must be in [0, 1], got {balanced_ratio}")
        self.balanced_ratio = float(balanced_ratio)

        self.size_count = 0
        self.index = []
        self.next_storage_id = 0
        self.insert_counter = 0


    def _reward_type(self, reward, done):
        """Categorize the reward into one of: win, lose, invalid, progress.

        Boundary case ``reward == invalid_threshold`` is folded into ``progress``
        (it's a "no penalty" reward — closer to a valid step than to an invalid one).
        """
        reward = float(reward)
        if done and reward >= self.win_threshold:
            return "win"
        if done and reward <= self.lose_threshold:
            return "lose"
        if reward < self.invalid_threshold:
            return "invalid"
        return "progress"

    def _save_tensor(self, tensor, root_name, storage_id):
        """Save a tensor either to disk or keep it in memory based on storage mode."""
        if self.storage_mode == "disk":
            path = self.save_dir / f"{root_name}_{storage_id}.pt"
            # If it looks like a visual image, store as uint8 to save space
            if len(tensor.shape) == 3 and tensor.shape[0] == 3 and tensor.shape[1] >= 64:
                uint8_tensor = tensor.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8)
                torch.save(uint8_tensor, path)
            else:
                torch.save(tensor.cpu(), path)
            return str(path)
        else:
            return tensor.cpu().clone() if torch.is_tensor(tensor) else tensor

    def _load_tensor(self, reference):
        """Load and return a tensor from disk, or return the memory reference directly."""
        if self.storage_mode == "disk":
            tensor = torch.load(reference, map_location="cpu")
            if tensor.dtype == torch.uint8:
                return tensor.float() / 255.0
            return tensor.cpu()
        else:
            if torch.is_tensor(reference):
                return reference.detach().cpu().clone()
            return reference

    def _safe_unlink(self, path_str):
        """Safely delete a file from disk if its path is provided and exists."""
        if not path_str:
            return
        path = Path(path_str)
        if path.exists():
            path.unlink()

    def _delete_entry_files(self, entry):
        """Clean up and delete disk files associated with a removed buffer entry."""
        if self.storage_mode == "disk":
            self._safe_unlink(entry.get("state"))
            self._safe_unlink(entry.get("next_state"))

    def _effective_priority(self, entry):
        """Calculate the current priority of an entry after applying age (and optional)
        cooling factors.

        Accumulator pattern — start with ``base_priority / age_factor`` as the baseline
        ``score``, then stack additional divisors on it whenever a feature switch is on.
        New PER modifiers can be added below as new ``if self.<switch>:`` blocks without
        touching the rest of the function.

        Cooling factors currently wired in:
            - age_decay × age              : time since entry was stored (always on)
            - sample_decay × sample_count  : how many times PER has selected this entry
              — ONLY applied when ``self.enable_sample_decay`` is True (Method B switch).
              Defense against stochastic-trap entries; disabled by default.
            - spread_decay × quantile_spread: FQF-predicted quantile std for this entry
              — ONLY applied when ``self.enable_spread_decay`` is True. Wide spread =
              model believes outcome is stochastic → decay more. ``quantile_spread`` is
              updated by ``update_priorities`` whenever the entry is sampled.
        """
        priority = float(np.clip(entry.get("priority", 1.0), self.priority_min, self.priority_max))
        age = max(0, self.insert_counter - entry.get("insert_order", 0))
        if self.max_age > 0 and age > self.max_age:
            return 0.0
        age_factor = 1.0 + self.age_decay * age
        priority = priority / age_factor

        if self.enable_sample_decay:
            sample_count = int(entry.get("sample_count", 0))
            sample_factor = 1.0 + self.sample_decay * sample_count
            priority = priority / sample_factor

        if self.enable_spread_decay:
            quantile_spread = float(entry.get("quantile_spread", 0.0))
            spread_factor = 1.0 + self.spread_decay * quantile_spread
            priority = priority / spread_factor

        return priority

    def _selection_priorities(self, entries):
        """向量化計算多筆 entry 的 PER「選擇權重」(α-powered + priority_min floor)。

        這是「選擇分數」的**單一數學定義** — 單筆版 ``_selection_priority`` 只是
        薄包裝。要改公式只動這裡一處。

        ``P(select) ∝ _selection_priorities(entries)[i]`` — 抽樣時用這個算機率,
        IS weight ``w_i = (N · P_i)^(-β)`` 也用同一個值,確保「抽樣分佈」與
        「IS 修正」對得起來,不會出現抽到死條目卻 IS weight 爆炸的情況。

        與 ``_effective_priority`` 的差別:
        - 套 α 次方(``self.alpha``)讓分佈變平緩(標準 PER 公式)。
        - 對 ``priority_min`` 取 ``np.maximum``。死條目(``age > max_age``)的 eff
          是 0,沒這個 floor 它的 ``(N·P + ε)^(-β)`` 會炸到 ~10^4,跑 weights.max
          normalize 後把其他活條目的 IS weight 壓到接近 0。
        """
        eff = np.array([self._effective_priority(e) for e in entries], dtype=np.float64)
        return np.power(np.maximum(eff, self.priority_min), self.alpha)

    def _selection_priority(self, entry):
        """單筆 entry 的 PER 選擇權重,薄包裝呼叫 ``_selection_priorities``。

        數學定義全部在 ``_selection_priorities`` 裡(單一來源,避免兩處 drift)。
        這個 scalar helper 給單點查詢 / 外部 audit 用,生產 hot path 都直接走
        陣列版以享受 numpy 向量化。
        """
        return float(self._selection_priorities([entry])[0])

    def top_k_balanced(self, k):
        """Select up to ``k`` entries via stratified balanced + cross-class PER。
        比例由 ``self.balanced_ratio`` 控制(跟 ``sample()`` 用同一個參數)。

        Phase 1 — Balanced(占 ``k × balanced_ratio``):
            每類保留 ``per_class_quota = balanced_share // len(REWARD_TYPES)`` 筆,
            按 ``_effective_priority`` 排序由高到低取。某類 entry 數 < quota 時
            slack 留給 Phase 2。

        Phase 2 — PER fallback / 剩餘名額(占 ``k × (1 - balanced_ratio)`` + Phase-1 slack):
            Phase 1 未選到的 entries 跨類比 ``_effective_priority`` 全域排序,
            取 top 補滿剩餘名額。高 priority「model 還沒學會」的 entry 在這裡跨類競爭。

        典型設定:
            - ``balanced_ratio = 1.0``:每類保 ``k/4``(8 類 → 12.5%),PER 只在某類
              不足時補位。pruning 不會被高 abs(reward) 類別系統性壓擠其他類。
            - ``balanced_ratio = 0.5``:原始設計,每類保 ``k/8``,剩 50% 給 PER 跨類競爭。

        Pure function: does NOT mutate ``self.index`` or touch disk files.
        Returns a list of entry-dict references (aliases into ``self.index``).
        Returned length is ``min(k, size_count)``.
        """
        if k <= 0 or self.size_count == 0:
            return []
        if k >= self.size_count:
            return list(self.index)

        # balanced_share ≥ 0 必然成立,整數除法保持非負;quota=0 時 entries[:0] 自然
        # 是空 list,Phase 1 等同跳過 → 不用額外 if/else。
        balanced_share = int(k * self.balanced_ratio)
        per_class_quota = balanced_share // len(self.REWARD_TYPES)

        grouped_entries = {reward_type: [] for reward_type in self.REWARD_TYPES}
        for entry in self.index:
            rt = entry.get("reward_type")
            if rt in grouped_entries:
                grouped_entries[rt].append(entry)
            else:
                # Legacy entries with unknown / dropped "other" reward_type → treat as progress
                grouped_entries["progress"].append(entry)

        survivors = []
        leftovers = []
        for reward_type in self.REWARD_TYPES:
            entries = grouped_entries.get(reward_type, [])
            entries.sort(key=self._effective_priority, reverse=True)
            survivors.extend(entries[:per_class_quota])
            leftovers.extend(entries[per_class_quota:])

        remaining_slots = max(0, k - len(survivors))
        if remaining_slots > 0 and leftovers:
            leftovers.sort(key=self._effective_priority, reverse=True)
            survivors.extend(leftovers[:remaining_slots])

        return survivors[:k]

    def export_top_k(self, k, persistent_dir=None):
        """Pick top-k balanced entries ready for serialization.

        RAM mode (``storage_mode == "ram"``):
            ``persistent_dir`` must be ``None``. Returns the entry-dict
            references directly — their ``state`` / ``next_state`` are
            in-memory tensors.

        Disk mode (``storage_mode == "disk"``):
            ``persistent_dir`` must be a path. The directory is created if
            missing and cleared of stale ``.pt`` files, then each selected
            entry's state / next_state file is copied in with fresh
            ``storage_id`` numbering. Returns NEW entry dicts whose
            ``state`` / ``next_state`` paths point inside ``persistent_dir``.

        Raises:
            ValueError: if ``storage_mode`` and ``persistent_dir`` don't match.
        """
        if self.storage_mode == "ram":
            if persistent_dir is not None:
                raise ValueError("RAM-mode buffer must not be given persistent_dir")
            return self.top_k_balanced(k)

        if persistent_dir is None:
            raise ValueError("Disk-mode buffer requires persistent_dir for export")

        persistent_dir = Path(persistent_dir)
        persistent_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in persistent_dir.glob("*.pt"):
            stale_path.unlink()

        selected = self.top_k_balanced(k)
        exported = []
        save_idx = 0
        for old_entry in selected:
            state_src = Path(old_entry["state"])
            if not state_src.exists():
                continue

            state_dst = persistent_dir / f"state_{save_idx}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            if old_entry["next_state"]:
                next_src = Path(old_entry["next_state"])
                if next_src.exists():
                    next_state_dst = persistent_dir / f"next_state_{save_idx}.pt"
                    shutil.copy2(str(next_src), str(next_state_dst))

            tail_reward = float(old_entry.get("tail_reward", old_entry["reward"]))
            legacy_rt = old_entry.get("reward_type")
            if legacy_rt not in self.REWARD_TYPES:
                # Legacy "other" or missing → re-categorize against the 4-class scheme
                legacy_rt = self._reward_type(tail_reward, bool(old_entry["done"]))
            exported.append({
                "storage_id": save_idx,
                "state": str(state_dst),
                "action": old_entry["action"],
                "next_state": str(next_state_dst) if next_state_dst else None,
                "reward": old_entry["reward"],
                "tail_reward": tail_reward,
                "done": old_entry["done"],
                "discount": float(old_entry.get("discount", 1.0)),
                "n_steps": int(old_entry.get("n_steps", 1)),
                "priority": float(old_entry.get("priority", self.priority_min)),
                "reward_type": legacy_rt,
                "insert_order": save_idx + 1,
                "sample_count": int(old_entry.get("sample_count", 0)),
                "quantile_spread": float(old_entry.get("quantile_spread", 0.0)),
            })
            save_idx += 1

        return exported

    def _prune_if_needed(self):
        """Reduce buffer size to max capacity by removing lowest priority entries across buckets."""
        if self.size_count <= self.max_size + self.overflow_margin:
            return

        survivors = self.top_k_balanced(self.max_size)
        selected_ids = {entry["storage_id"] for entry in survivors}

        old_entries = self.index
        self.index = survivors
        self.size_count = len(self.index)

        # Cleanup deleted elements
        for entry in old_entries:
            if entry["storage_id"] not in selected_ids:
                self._delete_entry_files(entry)

    def _sample_from_bucket(self, entries, count, use_uniform=True):
        """Sample ``count`` entries from ``entries`` (a single bucket OR pooled leftover).

        Args:
            entries: list of entry dicts to sample from.
            count: number of entries to return.
            use_uniform: if True, the first ``round(count * uniform_mix)`` picks are
                pure random (without replacement) for exploration, then the rest are
                priority-weighted. If False, sampling is purely priority-weighted —
                used by the PER half of ``sample()`` / leftover pass in storage.
        """
        if count <= 0 or not entries:
            return []

        if use_uniform:
            num_uniform = min(len(entries), int(round(count * self.uniform_mix)))
        else:
            num_uniform = 0
        num_priority = max(0, count - num_uniform)

        chosen = []
        available = list(entries)

        if num_uniform > 0:
            uniform_pick = random.sample(available, k=min(num_uniform, len(available)))
            chosen.extend(uniform_pick)
            chosen_ids = {entry["storage_id"] for entry in uniform_pick}
            available = [entry for entry in available if entry["storage_id"] not in chosen_ids]

        if num_priority > 0 and available:
            priorities = self._selection_priorities(available)
            prob_sum = priorities.sum()

            if prob_sum <= 0 or not np.isfinite(prob_sum):
                priority_pick = random.choices(available, k=min(num_priority, len(available)))
            else:
                probabilities = priorities / prob_sum
                replace = len(available) < num_priority
                indices = np.random.choice(
                    len(available),
                    size=num_priority if replace else min(num_priority, len(available)),
                    replace=replace,
                    p=probabilities,
                )
                priority_pick = [available[int(idx)] for idx in np.atleast_1d(indices)]
            chosen.extend(priority_pick)

        # Safety padding — uniform-random fallback when bucket 太小不夠抽。
        while len(chosen) < count:
            chosen.append(random.choice(entries))

        result = chosen[:count]
        # sample_count 無條件 +1:每次 entry 進 batch 都算一次(不分 uniform / PER /
        # padding)。提供「資料被抽到次數」的觀測值。Method B(enable_sample_decay=True)
        # 的 decay 公式 1/(1 + sample_decay × sample_count) 用同一個 counter,所以
        # 啟用 Method B 時,decay 會對「任何被過度採樣的 entry」一律降溫(原始設計
        # 只懲罰 PER 半邊,改成不分路徑 — 過度採樣本身就是要避免的訊號)。
        # 同一筆 entry 在 batch 內出現多次(replace=True 時)會每出現一次 +1。
        for entry in result:
            entry["sample_count"] = int(entry.get("sample_count", 0)) + 1
        return result

    def store(
        self,
        state,
        action,
        next_state,
        reward,
        done,
        discount=1.0,
        n_steps=1,
        tail_reward=None,
    ):
        """Store a transition."""
        self.insert_counter += 1
        storage_id = self.next_storage_id
        self.next_storage_id += 1

        state_ref = self._save_tensor(state, "state", storage_id)
        next_state_ref = None
        if next_state is not None:
            next_state_ref = self._save_tensor(next_state, "next_state", storage_id)

        bucket_reward = float(reward if tail_reward is None else tail_reward)
        entry = {
            "storage_id": storage_id,
            "state": state_ref,
            "action": action.copy() if isinstance(action, np.ndarray) else np.array(action),
            "next_state": next_state_ref,
            "reward": float(reward),
            "tail_reward": bucket_reward,
            "done": bool(done),
            "discount": float(discount),
            "n_steps": int(max(1, n_steps)),
            "reward_type": self._reward_type(bucket_reward, done),
            "priority": float(np.clip(abs(float(reward)) + 1.0, self.priority_min, self.priority_max)),
            "insert_order": self.insert_counter,
            "sample_count": 0,  # 每次進 batch 都 +1(uniform/PER/padding 都算);Method B decay 也用這個
            "quantile_spread": 0.0,  # FQF quantile std — updated by update_priorities when sampled
        }

        self.index.append(entry)
        self.size_count = len(self.index)
        self._prune_if_needed()

    def update_priorities(self, sample_indices, td_errors, quantile_spreads=None):
        """Update priorities (and optional quantile_spread) from training-time signals.

        Args:
            sample_indices: indices returned by the matching ``sample()`` call.
            td_errors: per-sample |TD-error| (tensor or sequence), aligned with sample_indices.
            quantile_spreads: optional per-sample ``chosen_quantiles.std(dim=1)`` values
                (tensor or sequence). When provided, written to each entry's
                ``quantile_spread`` field so ``_effective_priority`` can pick it up
                when ``self.enable_spread_decay`` is True. We write the value
                regardless of the switch state — so flipping the switch later
                doesn't reset spread history.

        Note: ``sample_count`` 在 ``_sample_from_bucket`` 結尾無條件 +1,不在這裡動 —
        本 function 只刷新 ``priority`` 和 ``quantile_spread``。
        """
        if sample_indices is None or td_errors is None:
            return

        td_values = td_errors.detach().float().view(-1).cpu().tolist() if torch.is_tensor(td_errors) else td_errors

        if quantile_spreads is None:
            spread_values = [None] * len(td_values)
        elif torch.is_tensor(quantile_spreads):
            spread_values = quantile_spreads.detach().float().view(-1).cpu().tolist()
        else:
            spread_values = list(quantile_spreads)

        for idx, td_error, spread in zip(sample_indices, td_values, spread_values):
            if not (0 <= int(idx) < len(self.index)):
                continue
            priority = abs(float(td_error)) + self.priority_eps
            entry = self.index[int(idx)]
            entry["priority"] = float(np.clip(priority, self.priority_min, self.priority_max))
            if spread is not None:
                entry["quantile_spread"] = float(spread)

    def sample(self, batch_size, beta=None, device="cpu", include_extra=False):
        """Sample a batch via stratified balanced + PER fallback。比例由
        ``self.balanced_ratio``(constructor 參數)控制。

        Phase 1 — Balanced(占 ``batch_size × balanced_ratio`` 個名額):
            每類各取 ``per_class_count = balanced_total // len(REWARD_TYPES)`` 筆。
            類內仍走 ``_sample_from_bucket(use_uniform=True)``,所以類內仍有
            uniform_mix + alpha-weighted PER(focus on hard sample),只是**不跨類**。

        Phase 2 — PER fallback(占 ``batch_size × (1 - balanced_ratio)`` 個名額,
        加上 Phase 1 某類不足時的剩餘 slack):
            從所有未被 Phase 1 選到的 entries 用 pure-priority PER 全域抽。
            這裡是 cross-class 競爭,容易被高 abs(reward) → 高 TD-error 的類別
            (Minesweeper 的 win/lose)系統性主宰 → 想避免就把 balanced_ratio 拉高。

        典型設定:
            - ``balanced_ratio = 1.0``(default):每類 ``batch_size // 4`` 筆,
              PER 只在某類 entry 數不足時補位。最乾淨的 stratification。
            - ``balanced_ratio = 0.5``:原始 50/50 設計,Phase 1 每類 ``batch_size // 8``,
              Phase 2 用 PER 全域競爭 50% 名額。
            - ``balanced_ratio = 0.0``:純 PER 全域抽,沒 stratification。

        IS weight 由 caller 自行決定要不要乘進 loss;v3 / stage1 兩邊的 train_step
        都採 ``loss = (is_weights * per_sample_loss).mean()``,所以 ``beta`` 的
        annealing 是 caller 的責任(buffer 端只在沒帶 ``beta`` 參數時 fallback 到
        ``self.beta`` = ctor 的 ``beta_start``,不會自動 anneal)。
        """
        if self.size_count <= 0:
            raise RuntimeError("CategorizedReplayBuffer is empty")

        use_beta = beta if beta is not None else self.beta

        # === Phase 1: Balanced(占 self.balanced_ratio × batch_size) ===
        # per_class_count=0 時 take=0,Phase 1 自然空跑(無需 if/else)。
        balanced_total = int(batch_size * self.balanced_ratio)
        per_class_count = balanced_total // len(self.REWARD_TYPES)

        grouped_indices = {reward_type: [] for reward_type in self.REWARD_TYPES}
        for idx, entry in enumerate(self.index):
            rt = entry.get("reward_type")
            if rt in grouped_indices:
                grouped_indices[rt].append(idx)
            else:
                # Legacy "other" / unknown reward_type → fold into progress
                grouped_indices["progress"].append(idx)

        selected_indices = []
        selected_storage_ids = set()

        for reward_type in self.REWARD_TYPES:
            indices = grouped_indices[reward_type]
            if not indices:
                continue
            take = min(per_class_count, len(indices))
            bucket_entries = [self.index[i] for i in indices]
            id_to_idx = {self.index[i]["storage_id"]: i for i in indices}
            sampled_entries = self._sample_from_bucket(bucket_entries, take, use_uniform=True)
            for entry in sampled_entries:
                sid = entry["storage_id"]
                if sid in id_to_idx:
                    selected_indices.append(id_to_idx[sid])
                    selected_storage_ids.add(sid)

        # === Phase 2: PER fallback — 只在 Phase 1 沒填滿時觸發(某類 entry 數 < per_class_count) ===
        # 用剩餘 entries 跨類 pure-priority 補滿 batch。steady state 下通常是空跑。
        per_remaining = batch_size - len(selected_indices)
        if per_remaining > 0:
            leftover_indices = [
                i for i, entry in enumerate(self.index)
                if entry["storage_id"] not in selected_storage_ids
            ]
            if leftover_indices:
                leftover_entries = [self.index[i] for i in leftover_indices]
                id_to_idx = {self.index[i]["storage_id"]: i for i in leftover_indices}
                per_sampled = self._sample_from_bucket(leftover_entries, per_remaining, use_uniform=False)
                for entry in per_sampled:
                    sid = entry["storage_id"]
                    if sid in id_to_idx:
                        selected_indices.append(id_to_idx[sid])

        # Safety padding — should rarely trigger (size_count < batch_size case)
        while len(selected_indices) < batch_size:
            selected_indices.append(random.randint(0, len(self.index) - 1))

        selected_indices = selected_indices[:batch_size]

        # 一次取出所有選中的 entry,接著向量化算 selection priorities(給下面 IS 用)。
        # 其他 per-entry 欄位(state/action/reward/...) 還是必須 loop 拆出來。
        # 注意:sample_count 不在這裡 bump,已經在 `_sample_from_bucket` 結尾統一 bump
        # 過了(uniform / PER / padding 三條路徑都在那裡覆蓋到)。
        selected_entries = [self.index[idx] for idx in selected_indices]
        priorities_arr = self._selection_priorities(selected_entries)

        states, actions, next_states, rewards, dones, discounts, n_steps = [], [], [], [], [], [], []

        for entry in selected_entries:
            states.append(self._load_tensor(entry["state"]))
            actions.append(entry["action"])

            if entry["next_state"] is not None:
                next_states.append(self._load_tensor(entry["next_state"]))
            else:
                # create a dummy state of correct device and shape if done
                dummy = states[-1].clone()
                dummy.fill_(0)
                next_states.append(dummy)

            rewards.append(entry["reward"])
            dones.append(float(entry["done"]))
            discounts.append(float(entry.get("discount", 1.0)))
            n_steps.append(int(entry.get("n_steps", 1)))

        # Important Sampling — priorities_arr 在 selected_entries 上面已經用
        # _selection_priorities 一次算完,跟 _sample_from_bucket 的選擇分佈對齊
        # (都套 α 次方 + priority_min floor)。
        #
        # 簡化:標準 PER 公式 prob_sum = Σ_k p_k^α over 整個 buffer(O(N_buffer)),
        # 但下面 `weights = weights / weights.max()` 會把這個 batch 共用的常數
        # `(N / prob_sum)^(-β)` 整個吸收掉,逐元素歸一後結果只依賴 batch 內
        # priorities_arr 的比例。所以這裡只 sum batch 即可(O(batch_size))。
        # ⚠️ 此簡化依賴下方的 max-normalize;若把歸一化方式換成 mean、或乾脆
        # 拿掉,必須改回 `self._selection_priorities(self.index).sum()`。
        N = self.size_count
        prob_sum = priorities_arr.sum()
        probabilities = priorities_arr / (prob_sum + 1e-10)
        
        # weights formulation: (1/N * 1/P_i) ^ beta
        weights = (N * probabilities + 1e-10) ** (-use_beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        # Stacking tensors manually depends heavily on the model requirements:
        # Returning lists or direct tensors:
        tensor_states = torch.stack(states).to(device) if torch.is_tensor(states[0]) else states
        tensor_next_states = torch.stack(next_states).to(device) if torch.is_tensor(next_states[0]) else next_states

        tensor_actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        tensor_rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        tensor_dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        tensor_discounts = torch.tensor(discounts, dtype=torch.float32, device=device).unsqueeze(1)
        tensor_n_steps = torch.tensor(n_steps, dtype=torch.long, device=device).unsqueeze(1)

        result = (
            tensor_states,
            tensor_actions,
            tensor_next_states,
            tensor_rewards,
            tensor_dones,
            selected_indices,
            weights,
        )
        if include_extra:
            result = result + (tensor_discounts, tensor_n_steps)
        return result

    def size(self):
        """Return the current total number of entries in the buffer."""
        return self.size_count

    def bucket_sizes(self):
        """Return per-bucket entry counts as a dict keyed by REWARD_TYPES.

        Useful for monitoring whether rare-event buckets (win / lose / invalid)
        are starving or dominating relative to the bulk progress bucket.
        """
        counts = {reward_type: 0 for reward_type in self.REWARD_TYPES}
        for entry in self.index:
            rt = entry.get("reward_type")
            if rt in counts:
                counts[rt] += 1
            else:
                # Legacy / unknown reward_type — count under progress to avoid silent loss
                counts["progress"] = counts.get("progress", 0) + 1
        return counts

    def mean_sample_count(self) -> float:
        """整個 buffer 中,每筆 entry 平均被 sample 到幾次。

        定義:``sum(sample_count) / size_count``。``sample_count`` 在
        ``_sample_from_bucket`` 結尾無條件 +1,涵蓋 uniform / PER / safety padding
        三條路徑。Method B(``enable_sample_decay=True``)的 decay 公式也用同一個
        counter。

        判讀:
        - 數值單調隨訓練步數成長(只增不減,除非該筆 entry 被 prune)。
        - 高 mean = 平均每筆被反覆抽到,代表 PER 集中度高
        - 低 mean = 抽樣分佈分散,buffer 內樣本被均勻利用
        - 對齊看:同 env_step 下,mean_sample_count 越低代表 batch 多樣性越好
        """
        if not self.index:
            return 0.0
        return sum(int(e.get("sample_count", 0)) for e in self.index) / len(self.index)

    @property
    def class_quota(self) -> int:
        """Per-class soft-floor quota used by ``top_k_balanced``、``sample()`` 及
        ``is_class_quota_filled`` training gate。

        定義:``(max_size × balanced_ratio) // len(REWARD_TYPES)``,至少 1。
        - ``balanced_ratio = 1.0``:``max_size / 4``(每類 25%)
        - ``balanced_ratio = 0.5``:``max_size / 8``(每類 12.5%,原始設計)
        """
        balanced_share = int(self.max_size * self.balanced_ratio)
        return max(1, balanced_share // len(self.REWARD_TYPES))

    def is_class_quota_filled(self) -> bool:
        """True iff the gated class(es) have at least ``class_quota`` entries.

        Behaviour depends on ``self.quota_check_class``:
            - ``None`` (default): EVERY reward_type must reach quota — conservative,
              but pruning equilibrium can drag the loosest class out by a lot.
            - ``"<class_name>"``: ONLY that class is checked. Use this when you know
              one rare class is the true bottleneck (e.g. ``"win"`` in Minesweeper)
              and don't want other classes' pruning dynamics to delay training.

        Used as a training-readiness gate on top of ``MINIMUM_DATA_SIZE``.
        """
        quota = self.class_quota
        counts = self.bucket_sizes()
        if self.quota_check_class is not None:
            return counts.get(self.quota_check_class, 0) >= quota
        return all(count >= quota for count in counts.values())

    def get_all_entries(self):
        """Returns internal objects suitable for RAM persistent saving. 
        Note this won't move disk files, just the internal state index."""
        return self.index

    def load_from_entries(self, entries):
        """Load from a persistent index map structure in RAM。

        Backward compatibility:
            - Legacy entries without ``sample_count`` get default 0 (so they start fresh
              in the Method-B decay scheme).
            - Legacy entries without ``quantile_spread`` get default 0.0 (no spread decay
              until they're re-sampled and the FQF spread is recomputed).
            - Legacy entries with ``reward_type == "other"`` (or any value outside the
              current 4-class REWARD_TYPES) get re-categorized via ``_reward_type()``.
        """
        normalized_entries = []
        for entry in entries:
            normalized_entry = dict(entry)
            if "state" in normalized_entry:
                normalized_entry["state"] = self._load_tensor(normalized_entry["state"])
            if normalized_entry.get("next_state") is not None:
                normalized_entry["next_state"] = self._load_tensor(normalized_entry["next_state"])
            # Migrate legacy reward_type ("other" / missing) to the 4-class scheme
            if normalized_entry.get("reward_type") not in self.REWARD_TYPES:
                tail_reward = float(normalized_entry.get("tail_reward", normalized_entry.get("reward", 0.0)))
                normalized_entry["reward_type"] = self._reward_type(
                    tail_reward, bool(normalized_entry.get("done", False))
                )
            # Default sample_count + quantile_spread for legacy entries
            normalized_entry.setdefault("sample_count", 0)
            normalized_entry.setdefault("quantile_spread", 0.0)
            normalized_entries.append(normalized_entry)
        self.index = normalized_entries
        self.size_count = len(self.index)
        self.next_storage_id = max([e["storage_id"] for e in self.index], default=-1) + 1
        self.insert_counter = max([e["insert_order"] for e in self.index], default=0)
