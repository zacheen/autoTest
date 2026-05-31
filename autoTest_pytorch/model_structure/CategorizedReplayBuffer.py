import random
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG

# True prints [DBG sample] logs with cuda.synchronize; enable only for CUDA debugging.
DEBUG_CUDA_SAMPLE = False

# Set by agent modules after import. If set to a Path, _dbg_log() writes there
# instead of CMD. None means silent.
DEBUG_CUDA_SAMPLE_LOG_PATH = None


def _dbg_log(msg: str) -> None:
    """Write [DBG sample] messages to DEBUG_CUDA_SAMPLE_LOG_PATH, not CMD."""
    if DEBUG_CUDA_SAMPLE_LOG_PATH is None:
        return
    try:
        with open(DEBUG_CUDA_SAMPLE_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(f"{msg}\n")
    except Exception:
        pass

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
                Only controls whether ``_effective_priority`` applies decay.
                ``sample_count`` always increments at the end of ``_sample_from_bucket``
                because monitoring metrics such as ``mean_sample_count`` need a
                valid counter. When this is False, the buffer behaves like plain
                PER and sample_count is only observational.
            sample_decay: Method B — divide effective priority by (1 + sample_decay × sample_count).
                Only takes effect when ``enable_sample_decay=True``. ``sample_count``
                counts all sampling paths (uniform / PER / safety padding), not only
                PER, so over-sampled entries cool down regardless of cause. Learned
                entries are naturally protected: low priority means low pick rate and
                slower sample_count growth.
                Note: the original design counted only the PER half to avoid
                penalizing uniform exploration. Counting all paths makes metric
                meaning consistent. If Method B later hurts uniform exploration,
                consider adding a separate PER-only counter.
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
            balanced_ratio: 0.0~1.0 ratio of Phase 1 stratified-balanced samples
                inside ``sample()``. The rest is Phase 2 cross-class pure-PER fallback.
                - 1.0 (default): each class takes ``batch_size // 4`` samples; PER
                  only fills shortages. Avoids cross-class PER bias toward high
                  abs(reward) classes.
                - 0.5: original 50/50 design; Phase 1 takes ``batch_size // 8`` per
                  class and Phase 2 globally competes for the remaining 50%.
                - 0.0: no stratification, pure global PER dominated by high priority classes.
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
        """Vectorized PER selection weights for entries.

        This is the single mathematical definition of selection score. The scalar
        ``_selection_priority`` is only a thin wrapper.

        ``P(select) ∝ _selection_priorities(entries)[i]``. Sampling probabilities
        and IS weights ``w_i = (N · P_i)^(-β)`` use this same value, keeping the
        sampling distribution and IS correction aligned.

        Differences from ``_effective_priority``:
        - Apply α power (``self.alpha``) to soften the distribution.
        - Floor by ``priority_min``. Dead entries have effective priority 0; without
          a floor, their IS weights can explode and suppress live entries after
          max-normalization.
        """
        eff = np.array([self._effective_priority(e) for e in entries], dtype=np.float64)
        return np.power(np.maximum(eff, self.priority_min), self.alpha)

    def _selection_priority(self, entry):
        """
        Scalar PER selection weight wrapper around ``_selection_priorities``.
        """
        return float(self._selection_priorities([entry])[0])

    def top_k_balanced(self, k):
        """Select up to ``k`` entries via stratified balanced + cross-class PER.

        Ratio is controlled by ``self.balanced_ratio``, shared with ``sample()``.

        Phase 1 — Balanced (``k × balanced_ratio``):
            Keep ``per_class_quota = balanced_share // len(REWARD_TYPES)`` per class,
            sorted by ``_effective_priority`` descending. Shortage slack goes to Phase 2.

        Phase 2 — PER fallback / remaining slots:
            Globally rank unselected entries by ``_effective_priority`` and take top
            entries to fill the rest.

        Typical settings:
            - ``balanced_ratio = 1.0``: keep ``k/4`` per class; PER only fills shortages.
            - ``balanced_ratio = 0.5``: original design; keep ``k/8`` per class and
              leave 50% for cross-class PER.

        Pure function: does NOT mutate ``self.index`` or touch disk files.
        Returns a list of entry-dict references (aliases into ``self.index``).
        Returned length is ``min(k, size_count)``.
        """
        if k <= 0 or self.size_count == 0:
            return []
        if k >= self.size_count:
            return list(self.index)

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

        # Safety padding: uniform-random fallback when the bucket is too small.
        while len(chosen) < count:
            chosen.append(random.choice(entries))

        result = chosen[:count]
        # Always increment sample_count once per entry appearance in a batch,
        # across uniform / PER / padding. Method B decay uses the same counter,
        # so any over-sampled entry cools down. Duplicate appearances each count.
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
            "sample_count": 0,  # incremented on every batch entry; Method B decay uses it
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

        Note: ``sample_count`` increments at the end of ``_sample_from_bucket`` and
        is not touched here. This function only refreshes ``priority`` and
        ``quantile_spread``.
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
        """Sample a batch via stratified balanced + PER fallback.

        Ratio is controlled by constructor ``self.balanced_ratio``.

        Phase 1 — Balanced (``batch_size × balanced_ratio`` slots):
            Take ``per_class_count = balanced_total // len(REWARD_TYPES)`` per class.
            Within each class, ``_sample_from_bucket(use_uniform=True)`` still uses
            uniform_mix + alpha-weighted PER, but does not cross class boundaries.

        Phase 2 — PER fallback:
            Remaining slots plus Phase-1 slack are filled by global pure-priority
            PER from unselected entries. Raise balanced_ratio to reduce dominance
            by high abs(reward) / high TD-error classes.

        Typical settings:
            - ``balanced_ratio = 1.0``: cleanest stratification; PER only fills shortages.
            - ``balanced_ratio = 0.5``: original 50/50 design.
            - ``balanced_ratio = 0.0``: pure global PER, no stratification.

        Callers decide whether to multiply IS weights into loss. v3 and stage1 use
        ``loss = (is_weights * per_sample_loss).mean()``. Beta annealing is caller
        responsibility; buffer only falls back to ctor ``beta_start`` when beta is omitted.
        """
        if self.size_count <= 0:
            raise RuntimeError("CategorizedReplayBuffer is empty")

        use_beta = beta if beta is not None else self.beta

        # === Phase 1: Balanced (self.balanced_ratio x batch_size) ===
        # When per_class_count=0, take=0 and Phase 1 naturally no-ops.
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

        # === Phase 2: PER fallback, only when Phase 1 did not fill the batch ===
        # Fill from remaining entries with cross-class pure priority. Usually no-op at steady state.
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

        # Pull selected entries once, then vectorize selection priorities for IS.
        # Other per-entry fields still need loop unpacking. sample_count is already
        # bumped at the end of `_sample_from_bucket` for uniform / PER / padding.
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

        # ---- DBG: sanity-check stored actions before they get used to index Q tensor ----
        # Write only when DEBUG_CUDA_SAMPLE and LOG_PATH are set; no hard-coded V3 path or CMD spam.
        if DEBUG_CUDA_SAMPLE and DEBUG_CUDA_SAMPLE_LOG_PATH is not None:
            try:
                _bad_acts = []
                for i, action_value in enumerate(actions):
                    try:
                        action_int = int(np.asarray(action_value).reshape(-1)[0])
                    except Exception:
                        _bad_acts.append((i, action_value))
                        continue
                    # Check whether stored action can parse as int. Caller-side
                    # _dbg_tensor(expect_max=...) does precise GPU OOB checks.
                if _bad_acts:
                    _dbg_log(f"[REPLAY sample] !!BAD ACTIONS in batch!! {_bad_acts[:10]} (showing up to 10)")
            except Exception:
                pass

        # Importance Sampling: priorities_arr was computed once for selected_entries
        # with _selection_priorities, matching _sample_from_bucket distribution
        # (alpha power + priority_min floor).
        #
        # Simplification: standard PER sums p_k^alpha over the whole buffer, but
        # max-normalization below absorbs the batch-shared constant. After element
        # normalization, only ratios inside priorities_arr matter, so summing the
        # batch is enough. If normalization changes, restore full-buffer sum.
        N = self.size_count
        prob_sum = priorities_arr.sum()
        probabilities = priorities_arr / (prob_sum + 1e-10)
        
        # weights formulation: (1/N * 1/P_i) ^ beta
        weights = (N * probabilities + 1e-10) ** (-use_beta)
        weights = weights / weights.max()
        _dbg_cuda = DEBUG_CUDA_SAMPLE and ((device.type == "cuda") if hasattr(device, "type") else (str(device).startswith("cuda")))
        if _dbg_cuda: import torch as _t; _t.cuda.synchronize(); _dbg_log("[DBG sample] before weights.to(device)")
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        if _dbg_cuda: _t.cuda.synchronize(); _dbg_log("[DBG sample] after weights.to(device)")

        # Stacking tensors manually depends heavily on the model requirements:
        # Returning lists or direct tensors:
        if _dbg_cuda: _t.cuda.synchronize(); _dbg_log("[DBG sample] before stack states")
        tensor_states = torch.stack(states).to(device) if torch.is_tensor(states[0]) else states
        if _dbg_cuda: _t.cuda.synchronize(); _dbg_log("[DBG sample] after stack states")
        tensor_next_states = torch.stack(next_states).to(device) if torch.is_tensor(next_states[0]) else next_states
        if _dbg_cuda: _t.cuda.synchronize(); _dbg_log("[DBG sample] after stack next_states")

        tensor_actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        tensor_rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        tensor_dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        tensor_discounts = torch.tensor(discounts, dtype=torch.float32, device=device).unsqueeze(1)
        tensor_n_steps = torch.tensor(n_steps, dtype=torch.long, device=device).unsqueeze(1)
        if _dbg_cuda: _t.cuda.synchronize(); _dbg_log("[DBG sample] all tensors created")

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
        """Average sample_count per entry across the whole buffer.

        Defined as ``sum(sample_count) / size_count``. ``sample_count`` increments
        at the end of ``_sample_from_bucket`` for uniform / PER / safety padding.
        Method B decay uses the same counter.

        Read:
        - Monotonically grows with training steps unless entries are pruned.
        - High mean means repeated picks and concentrated PER.
        - Low mean means broader sampling and better buffer diversity.
        """
        if not self.index:
            return 0.0
        return sum(int(e.get("sample_count", 0)) for e in self.index) / len(self.index)

    @property
    def class_quota(self) -> int:
        """Per-class soft-floor quota for top_k_balanced, sample, and gate checks.

        Defined as ``(max_size × balanced_ratio) // len(REWARD_TYPES)``, at least 1.
        - ``balanced_ratio = 1.0``: ``max_size / 4`` (25% per class)
        - ``balanced_ratio = 0.5``: ``max_size / 8`` (12.5% per class, original design)
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
