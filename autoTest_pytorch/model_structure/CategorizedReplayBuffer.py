import random
import shutil
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
import numpy as np
import torch

from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG


class RewardType(StrEnum):
    WIN = "win"
    LOSE = "lose"
    INVALID = "invalid"
    PROGRESS = "progress"

    @classmethod
    def all(cls) -> tuple["RewardType", ...]:
        return tuple(cls)

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(reward_type.value for reward_type in cls)

    @classmethod
    def coerce(cls, value, default=None):
        if value is None:
            return default
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            return default


@dataclass(frozen=True)
class ReplayBalanceConfig:
    """Shared reward bucketing and class-balance settings."""

    win_threshold: float = MINESWEEPER_REWARD_CONFIG.replay_win_threshold
    lose_threshold: float = MINESWEEPER_REWARD_CONFIG.replay_lose_threshold
    invalid_threshold: float = MINESWEEPER_REWARD_CONFIG.replay_invalid_threshold
    balanced_ratio: float = 1.0
    quota_check_class: RewardType | str | None = None

    def __post_init__(self) -> None:
        quota_check_class = RewardType.coerce(self.quota_check_class)
        if self.quota_check_class is not None and quota_check_class is None:
            raise ValueError(
                f"quota_check_class must be one of {RewardType.values()} or None, "
                f"got {self.quota_check_class!r}"
            )
        if not 0.0 <= self.balanced_ratio <= 1.0:
            raise ValueError(f"balanced_ratio must be in [0, 1], got {self.balanced_ratio}")
        object.__setattr__(self, "quota_check_class", quota_check_class)
        object.__setattr__(self, "balanced_ratio", float(self.balanced_ratio))


class _ReplayStoreBase:
    """Shared reward bucketing and capacity helpers for replay stores."""

    REWARD_TYPES = RewardType.all()

    def __init__(
        self,
        *,
        config: ReplayBalanceConfig,
        storage_mode: str = "ram",
        save_dir=None,
    ) -> None:
        self.config = config
        self.win_threshold = config.win_threshold
        self.lose_threshold = config.lose_threshold
        self.invalid_threshold = config.invalid_threshold
        self.quota_check_class = config.quota_check_class
        self.balanced_ratio = config.balanced_ratio

        # Storage backend shared by all replay stores (main PER + pending). "ram"
        # keeps tensors in memory; "disk" persists uint8 .pt files under save_dir and
        # keeps only path strings in the index. The _save_transition / _load_transition /
        # _delete_entry_files helpers below are storage-mode aware so every subclass
        # gets identical persistence semantics.
        self.storage_mode = storage_mode.lower()
        if self.storage_mode == "disk":
            if not save_dir:
                raise ValueError("save_dir must be provided if storage_mode is 'disk'")
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

    def _reward_type(self, reward, done):
        """Categorize the reward into one of: win, lose, invalid, progress."""
        reward = float(reward)
        if done and reward >= self.win_threshold:
            return RewardType.WIN
        if done and reward <= self.lose_threshold:
            return RewardType.LOSE
        if reward < self.invalid_threshold:
            return RewardType.INVALID
        return RewardType.PROGRESS

    def reward_type_for(self, reward, done):
        """Public wrapper around reward bucketing for companion stores."""
        return self._reward_type(reward, done)

    @staticmethod
    def _coerce_reward_type(value, default=None):
        return RewardType.coerce(value, default=default)

    def class_capacities_for_size(self, size: int) -> dict[RewardType, int]:
        """Return hard per-class capacities using this store's balance settings."""
        size = max(0, int(size))
        capacities = {reward_type: 0 for reward_type in self.REWARD_TYPES}
        if size <= 0:
            return capacities

        balanced_share = int(size * self.balanced_ratio)
        base = balanced_share // len(self.REWARD_TYPES)
        remainder = balanced_share % len(self.REWARD_TYPES)
        for idx, reward_type in enumerate(self.REWARD_TYPES):
            capacities[reward_type] = base + (1 if idx < remainder else 0)

        slack = size - sum(capacities.values())
        if slack > 0:
            slack_base, slack_remainder = divmod(slack, len(self.REWARD_TYPES))
            for idx, reward_type in enumerate(self.REWARD_TYPES):
                capacities[reward_type] += slack_base + (1 if idx < slack_remainder else 0)
        return capacities

    @staticmethod
    def _to_float_list(values) -> list[float]:
        if torch.is_tensor(values):
            return values.detach().float().view(-1).cpu().tolist()
        return [float(value) for value in values]

    @staticmethod
    def _compress_tensor(tensor):
        """Quantize an image-like (3, H>=64, W) float tensor to uint8 to save space;
        otherwise pass the tensor through on CPU. Reversed by ``_decompress_tensor``."""
        if (
            torch.is_tensor(tensor)
            and len(tensor.shape) == 3
            and tensor.shape[0] == 3
            and tensor.shape[1] >= 64
        ):
            return tensor.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8)
        return tensor.cpu() if torch.is_tensor(tensor) else tensor

    @staticmethod
    def _decompress_tensor(tensor):
        """Inverse of ``_compress_tensor``: uint8 → float in [0,1], else CPU passthrough."""
        if torch.is_tensor(tensor) and tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        return tensor.cpu() if torch.is_tensor(tensor) else tensor

    def _save_transition(self, state, next_state, storage_id, prefix):
        """Persist a (state, next_state) pair as ONE record and return its reference.

        A transition is the unit of storage, so both tensors live in a single file /
        object: one write, one read, one delete, one existence check (no risk of a
        half-present transition).

        - disk mode → write ``{"state": ..., "next_state": ...}`` (each image-like
          tensor uint8-compressed; ``next_state`` may be None) to
          ``{prefix}_{storage_id}.pt`` and return the path string.
        - ram mode  → return that dict holding detached CPU clones the buffer owns.
        """
        if self.storage_mode == "disk":
            payload = {
                "state": self._compress_tensor(state),
                "next_state": self._compress_tensor(next_state) if next_state is not None else None,
            }
            path = self.save_dir / f"{prefix}_{storage_id}.pt"
            torch.save(payload, path)
            return str(path)
        return {
            "state": state.cpu().clone() if torch.is_tensor(state) else state,
            "next_state": next_state.cpu().clone() if torch.is_tensor(next_state) else next_state,
        }

    def _load_transition(self, reference, *, copy: bool = True):
        """Inverse of ``_save_transition``. Returns ``(state, next_state)`` (next_state
        may be None).

        ``copy`` only matters in RAM mode, where ``reference`` is a dict of tensors:
        - ``copy=True`` (default, safe): return independent ``.clone()`` tensors the
          caller fully owns (used by ``load_from_entries``, kept in ``self.index``).
        - ``copy=False`` (borrow): return the stored tensors read-only for the hot
          sampling path (``build_batch_from_entries``), whose very next step is a
          ``torch.stack`` that copies each element into a fresh batch — the borrow never
          escapes. Disk loads always return fresh tensors, so the borrow is moot there.
        """
        if isinstance(reference, dict):
            state = reference.get("state")
            next_state = reference.get("next_state")
            if copy:
                state = state.clone() if torch.is_tensor(state) else state
                next_state = next_state.clone() if torch.is_tensor(next_state) else next_state
            return state, next_state

        payload = torch.load(reference, map_location="cpu")
        state = self._decompress_tensor(payload["state"])
        stored_next = payload.get("next_state")
        next_state = self._decompress_tensor(stored_next) if stored_next is not None else None
        return state, next_state

    def _safe_unlink(self, path_str):
        """Safely delete a file from disk if its path is provided and exists."""
        if not path_str:
            return
        path = Path(path_str)
        if path.exists():
            path.unlink()

    def _delete_entry_files(self, entry):
        """Delete the single on-disk transition file associated with a removed entry."""
        if self.storage_mode == "disk":
            self._safe_unlink(entry.get("transition"))



class _PrioritizedReplayStore(_ReplayStoreBase):
    """A generic prioritized categorized replay store.

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
        quota_check_class: RewardType | str | None = None,
        beta_start: float = 0.4,
        balanced_ratio: float = 1.0,
        config: ReplayBalanceConfig | None = None,
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
        if config is None:
            config = ReplayBalanceConfig(
                win_threshold=win_threshold,
                lose_threshold=lose_threshold,
                invalid_threshold=invalid_threshold,
                balanced_ratio=balanced_ratio,
                quota_check_class=quota_check_class,
            )
        super().__init__(config=config, storage_mode=storage_mode, save_dir=save_dir)

        self.max_size = max_size

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
        self.beta = beta_start

        self.size_count = 0
        self.index = []
        self.next_storage_id = 0
        self.insert_counter = 0


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

        # balanced_share is non-negative; integer division keeps quota non-negative.
        # When quota=0, entries[:0] is empty, so Phase 1 skips without extra branching.
        balanced_share = int(k * self.balanced_ratio)
        per_class_quota = balanced_share // len(self.REWARD_TYPES)

        grouped_entries = {reward_type: [] for reward_type in self.REWARD_TYPES}
        for entry in self.index:
            rt = self._coerce_reward_type(entry.get("reward_type"))
            if rt in grouped_entries:
                grouped_entries[rt].append(entry)
            else:
                # Legacy entries with unknown / dropped "other" reward_type → treat as progress
                grouped_entries[RewardType.PROGRESS].append(entry)

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
            references directly — their ``transition`` is an in-memory dict of tensors.

        Disk mode (``storage_mode == "disk"``):
            ``persistent_dir`` must be a path. The directory is created if
            missing and cleared of stale ``.pt`` files, then each selected entry's
            single ``transition`` file is copied in with fresh ``storage_id``
            numbering. Returns NEW entry dicts whose ``transition`` path points
            inside ``persistent_dir``.

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
            transition_src = Path(old_entry["transition"])
            if not transition_src.exists():
                continue

            transition_dst = persistent_dir / f"transition_{save_idx}.pt"
            shutil.copy2(str(transition_src), str(transition_dst))

            tail_reward = float(old_entry.get("tail_reward", old_entry["reward"]))
            legacy_rt = self._coerce_reward_type(old_entry.get("reward_type"))
            if legacy_rt is None:
                # Legacy "other" or missing → re-categorize against the 4-class scheme
                legacy_rt = self._reward_type(tail_reward, bool(old_entry["done"]))
            exported.append({
                "storage_id": save_idx,
                "transition": str(transition_dst),
                "action": old_entry["action"],
                "reward": old_entry["reward"],
                "tail_reward": tail_reward,
                "done": old_entry["done"],
                "discount": float(old_entry.get("discount", 1.0)),
                "n_steps": int(old_entry.get("n_steps", 1)),
                "priority": float(old_entry.get("priority", self.priority_min)),
                "reward_type": legacy_rt.value,
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
        initial_priority=None,
        quantile_spread=0.0,
        transition_ref=None,
    ):
        """Store a transition.

        When ``transition_ref`` is supplied it is adopted into the index directly and
        ``_save_transition`` is skipped. A ref is whatever ``_save_transition`` returns
        for the current storage mode: a path string in disk mode, or a dict of tensors
        in RAM mode. This lets the pending store hand its already-written transition
        file (or already-cloned dict) to the main store on commit without re-serializing,
        moving, or re-cloning it. ``state`` / ``next_state`` are ignored when a ref is given.
        """
        self.insert_counter += 1
        storage_id = self.next_storage_id
        self.next_storage_id += 1

        if transition_ref is None:
            transition_ref = self._save_transition(state, next_state, storage_id, "transition")

        bucket_reward = float(reward if tail_reward is None else tail_reward)
        priority_value = abs(float(reward)) + 1.0 if initial_priority is None else float(initial_priority)
        entry = {
            "storage_id": storage_id,
            "transition": transition_ref,
            "action": action.copy() if isinstance(action, np.ndarray) else np.array(action),
            "reward": float(reward),
            "tail_reward": bucket_reward,
            "done": bool(done),
            "discount": float(discount),
            "n_steps": int(max(1, n_steps)),
            "reward_type": self._reward_type(bucket_reward, done).value,
            "priority": float(np.clip(priority_value, self.priority_min, self.priority_max)),
            "insert_order": self.insert_counter,
            "sample_count": 0,  # incremented on every batch entry; Method B decay uses it
            "quantile_spread": float(quantile_spread),
        }

        self.index.append(entry)
        self.size_count = len(self.index)
        self._prune_if_needed()
        return entry

    def store_prioritized(
        self,
        state,
        action,
        next_state,
        reward,
        done,
        *,
        priority,
        discount=1.0,
        n_steps=1,
        tail_reward=None,
        quantile_spread=0.0,
        transition_ref=None,
    ):
        """Store data whose PER priority is already known.

        ``transition_ref`` is forwarded to ``store`` so an already-saved transition
        reference can be adopted without re-serialization (see ``store``).
        """
        return self.store(
            state,
            action,
            next_state,
            reward,
            done,
            discount=discount,
            n_steps=n_steps,
            tail_reward=tail_reward,
            initial_priority=priority,
            quantile_spread=quantile_spread,
            transition_ref=transition_ref,
        )

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

        self._update_replay_priorities(sample_indices, td_values, spread_values)

    def _update_replay_priorities(self, sample_indices, td_values, spread_values):
        """Update priorities for entries already in the main PER replay."""

        for idx, td_error, spread in zip(sample_indices, td_values, spread_values):
            if not (0 <= int(idx) < len(self.index)):
                continue
            priority = abs(float(td_error)) + self.priority_eps
            entry = self.index[int(idx)]
            entry["priority"] = float(np.clip(priority, self.priority_min, self.priority_max))
            if spread is not None:
                entry["quantile_spread"] = float(spread)

    def build_batch_from_entries(
        self,
        selected_entries,
        sample_indices=None,
        beta=None,
        device="cpu",
        include_extra=False,
        is_weights=None,
        population_size=None,
    ):
        """Convert replay-format entries into the tensors consumed by agents."""
        if not selected_entries:
            raise RuntimeError("Cannot build a batch from zero entries")

        states, actions, next_states, rewards, dones, discounts, n_steps = [], [], [], [], [], [], []

        for entry in selected_entries:
            # copy=False: borrow buffer storage read-only. The torch.stack below copies
            # everything into a fresh batch tensor, so the borrow never escapes this loop.
            state, next_state = self._load_transition(entry["transition"], copy=False)
            states.append(state)
            actions.append(entry["action"])

            if next_state is not None:
                next_states.append(next_state)
            else:
                # clone() here is load-bearing: `state` may be a borrowed view of a
                # stored entry, so we must clone BEFORE the in-place fill_ to avoid
                # zeroing the buffer's own tensor.
                dummy = state.clone()
                dummy.fill_(0)
                next_states.append(dummy)

            rewards.append(entry["reward"])
            dones.append(float(entry["done"]))
            discounts.append(float(entry.get("discount", 1.0)))
            n_steps.append(int(entry.get("n_steps", 1)))

        if is_weights is None:
            use_beta = beta if beta is not None else self.beta
            priorities_arr = self._selection_priorities(selected_entries)
            prob_sum = priorities_arr.sum()
            probabilities = priorities_arr / (prob_sum + 1e-10)
            N = int(population_size or self.size_count or len(selected_entries))
            weights_arr = (N * probabilities + 1e-10) ** (-use_beta)
            weights_arr = weights_arr / weights_arr.max()
            weights = torch.tensor(weights_arr, dtype=torch.float32).unsqueeze(1).to(device)
        else:
            weights = torch.as_tensor(is_weights, dtype=torch.float32, device=device).view(-1, 1)

        # torch.stack allocates a NEW contiguous tensor and copies each element in, so the
        # batch is fully independent of buffer storage even though the per-entry loads above
        # were borrowed (copy=False). This is what makes the borrow safe.
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
            list(sample_indices or []),
            weights,
        )
        if include_extra:
            result = result + (tensor_discounts, tensor_n_steps)
        return result

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
            rt = self._coerce_reward_type(entry.get("reward_type"))
            if rt in grouped_indices:
                grouped_indices[rt].append(idx)
            else:
                # Legacy "other" / unknown reward_type → fold into progress
                grouped_indices[RewardType.PROGRESS].append(idx)

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

        selected_entries = [self.index[idx] for idx in selected_indices]
        return self.build_batch_from_entries(
            selected_entries,
            sample_indices=selected_indices,
            beta=use_beta,
            device=device,
            include_extra=include_extra,
            population_size=self.size_count,
        )

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
            rt = self._coerce_reward_type(entry.get("reward_type"))
            if rt in counts:
                counts[rt] += 1
            else:
                # Legacy / unknown reward_type — count under progress to avoid silent loss
                counts[RewardType.PROGRESS] = counts.get(RewardType.PROGRESS, 0) + 1
        return {reward_type.value: count for reward_type, count in counts.items()}

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
        """Load from a persistent index map structure in RAM.

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
            # Clean break: entries from the old two-tensor format (separate "state" /
            # "next_state" keys, no "transition") are dropped rather than crashing the
            # sampler later. Very old raw-tensor entries without "storage_id" are handled
            # by the caller's legacy store() path, not here.
            if "transition" not in entry:
                continue
            normalized_entry = dict(entry)
            # RAM-mode persistent entries carry ``transition`` as a dict of tensors;
            # clone (copy=True) so the buffer owns them long-term in self.index and does
            # not alias the caller's input. Disk-mode refs are path strings, kept as-is.
            ref = normalized_entry.get("transition")
            if isinstance(ref, dict):
                state, next_state = self._load_transition(ref, copy=True)
                normalized_entry["transition"] = {"state": state, "next_state": next_state}
            # Migrate legacy reward_type ("other" / missing) to the 4-class scheme
            reward_type = self._coerce_reward_type(normalized_entry.get("reward_type"))
            if reward_type is None:
                tail_reward = float(normalized_entry.get("tail_reward", normalized_entry.get("reward", 0.0)))
                reward_type = self._reward_type(
                    tail_reward, bool(normalized_entry.get("done", False))
                )
            normalized_entry["reward_type"] = reward_type.value
            # Default sample_count + quantile_spread for legacy entries
            normalized_entry.setdefault("sample_count", 0)
            normalized_entry.setdefault("quantile_spread", 0.0)
            normalized_entries.append(normalized_entry)
        self.index = normalized_entries
        self.size_count = len(self.index)
        self.next_storage_id = max([e["storage_id"] for e in self.index], default=-1) + 1
        self.insert_counter = max([e["insert_order"] for e in self.index], default=0)


@dataclass
class MixedReplayBatch:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    replay_indices: list[int]
    is_weights: torch.Tensor
    discounts: torch.Tensor
    n_steps: torch.Tensor
    replay_count: int
    pending_entries: list[dict[str, Any]]


class _PendingReplayStore(_ReplayStoreBase):
    """Temporary eval-data buffer consumed by normal optimizer steps."""

    def __init__(
        self,
        config: ReplayBalanceConfig,
        *,
        extra_capacity: int = 500,
        sample_ratio: float = 0.10,
        storage_mode: str = "ram",
        save_dir=None,
    ) -> None:
        if extra_capacity < 0:
            raise ValueError(f"extra_capacity must be >= 0, got {extra_capacity}")
        if not 0.0 <= sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be in [0, 1], got {sample_ratio}")

        super().__init__(config=config, storage_mode=storage_mode, save_dir=save_dir)
        self.extra_capacity = int(extra_capacity)
        self.sample_ratio = float(sample_ratio)
        self._reward_types = tuple(self.REWARD_TYPES)
        self._buckets = {reward_type: deque() for reward_type in self._reward_types}
        self._seen_counts = {reward_type: 0 for reward_type in self._reward_types}
        self._next_pending_id = 0

        # Pending is a transient staging area: it is never serialized or restored
        # (save_persistent / export_top_k / _load_persistent_buffer touch the main
        # store only). It shares the main store's save_dir, so on startup clear any
        # pending_*.pt left behind by a crashed run. The "pending_" prefix scopes this
        # glob to pending files only — it never matches the main store's state_*.pt.
        # The base __init__ already created save_dir in disk mode, so this is safe.
        if self.storage_mode == "disk":
            for stale_path in self.save_dir.glob("pending_*.pt"):
                stale_path.unlink()

    def size(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def total_capacity(self, *, replay_size: int, replay_max_size: int) -> int:
        replay_slack = max(0, int(replay_max_size) - int(replay_size))
        return replay_slack + self.extra_capacity

    def class_capacities(self, *, replay_size: int, replay_max_size: int) -> dict[RewardType, int]:
        return self.class_capacities_for_size(
            self.total_capacity(replay_size=replay_size, replay_max_size=replay_max_size)
        )

    def bucket_sizes(self) -> dict[str, int]:
        return {
            reward_type.value: len(self._buckets[reward_type])
            for reward_type in self._reward_types
        }

    def prune_to_capacity(self, *, replay_size: int, replay_max_size: int) -> None:
        total_capacity = self.total_capacity(replay_size=replay_size, replay_max_size=replay_max_size)
        if total_capacity <= 0:
            for bucket in self._buckets.values():
                while bucket:
                    self._delete_entry_files(bucket.popleft())
            return

        capacities = self.class_capacities(replay_size=replay_size, replay_max_size=replay_max_size)
        while self.size() > total_capacity:
            victim_type = self._oldest_over_quota_type(capacities)
            if victim_type is None:
                victim_type = self._largest_non_empty_type()
            if victim_type is None:
                return
            self._delete_entry_files(self._buckets[victim_type].popleft())

    def store(
        self,
        state,
        action,
        next_state,
        reward,
        done,
        *,
        replay_size: int,
        replay_max_size: int,
        discount: float = 1.0,
        n_steps: int = 1,
        tail_reward=None,
    ) -> bool:
        bucket_reward = float(reward if tail_reward is None else tail_reward)
        reward_type = self.reward_type_for(bucket_reward, done)
        self._seen_counts[reward_type] += 1
        self.prune_to_capacity(replay_size=replay_size, replay_max_size=replay_max_size)

        total_capacity = self.total_capacity(replay_size=replay_size, replay_max_size=replay_max_size)
        if total_capacity <= 0:
            return False

        bucket = self._buckets[reward_type]
        pending_id = self._next_pending_id
        self._next_pending_id += 1  # always advance, even on reject, so ids never repeat

        # Build metadata only. The (multi-MB) state tensors are written lazily by
        # _materialize() — and ONLY on a branch that actually keeps the entry — so a
        # rejected transition never writes a .pt file that would then leak on disk.
        entry = {
            "pending_id": pending_id,
            "action": action.copy() if isinstance(action, np.ndarray) else np.array(action),
            "reward": float(reward),
            "tail_reward": bucket_reward,
            "done": bool(done),
            "discount": float(discount),
            "n_steps": int(max(1, n_steps)),
            "reward_type": reward_type.value,
        }

        def _materialize() -> None:
            # Reuse the shared storage backend: disk mode writes ONE uint8
            # pending_transition_*.pt and stores its path; RAM mode stores a dict of
            # cloned tensors. The "pending_transition_" prefix keeps these files from
            # colliding with the main store's transition_*.pt in the shared save_dir.
            entry["transition"] = self._save_transition(
                state, next_state, pending_id, "pending_transition"
            )
            bucket.append(entry)

        if self.size() < total_capacity:
            _materialize()
            return True

        capacities = self.class_capacities(replay_size=replay_size, replay_max_size=replay_max_size)
        class_capacity = max(1, int(capacities.get(reward_type, 0)))
        if len(bucket) < class_capacity:
            victim_type = self._oldest_over_quota_type(capacities, exclude=reward_type)
            if victim_type is not None:
                self._delete_entry_files(self._buckets[victim_type].popleft())
                _materialize()
                return True

        keep_probability = class_capacity / max(1, self._seen_counts[reward_type])
        if random.random() > keep_probability:
            return False

        _materialize()
        self._delete_entry_files(bucket.popleft())
        return True

    def sample_entries(self, count: int, *, replay_size: int, replay_max_size: int) -> list[dict[str, Any]]:
        self.prune_to_capacity(replay_size=replay_size, replay_max_size=replay_max_size)
        count = min(max(0, int(count)), self.size())
        if count <= 0:
            return []

        balanced_total = int(count * self.balanced_ratio)
        per_class_count = balanced_total // len(self._reward_types)
        selected = []
        selected_ids = set()

        for reward_type in self._reward_types:
            bucket_entries = list(self._buckets[reward_type])
            if not bucket_entries:
                continue
            take = min(per_class_count, len(bucket_entries))
            if take <= 0:
                continue
            picked = random.sample(bucket_entries, k=take)
            selected.extend(picked)
            selected_ids.update(entry["pending_id"] for entry in picked)

        remaining = count - len(selected)
        if remaining > 0:
            leftovers = [
                entry
                for reward_type in self._reward_types
                for entry in self._buckets[reward_type]
                if entry["pending_id"] not in selected_ids
            ]
            if leftovers:
                selected.extend(random.sample(leftovers, k=min(remaining, len(leftovers))))

        return selected[:count]

    def remove_entries(self, committed_ids: set[int]) -> None:
        if not committed_ids:
            return
        for reward_type in self._reward_types:
            self._buckets[reward_type] = deque(
                entry
                for entry in self._buckets[reward_type]
                if entry["pending_id"] not in committed_ids
            )

    def _oldest_over_quota_type(
        self,
        capacities: dict[RewardType, int],
        exclude: RewardType | None = None,
    ) -> RewardType | None:
        candidates = []
        for reward_type in self._reward_types:
            if reward_type == exclude:
                continue
            bucket = self._buckets[reward_type]
            if len(bucket) <= max(0, int(capacities.get(reward_type, 0))):
                continue
            if bucket:
                candidates.append((bucket[0]["pending_id"], reward_type))
        if not candidates:
            return None
        return min(candidates)[1]

    def _largest_non_empty_type(self) -> RewardType | None:
        reward_type = max(self._reward_types, key=lambda item: len(self._buckets[item]))
        return reward_type if self._buckets[reward_type] else None


class CategorizedReplayBuffer:
    """Public replay-buffer facade controlling main PER and pending eval stores."""

    REWARD_TYPES = _ReplayStoreBase.REWARD_TYPES

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
        quota_check_class: RewardType | str | None = None,
        beta_start: float = 0.4,
        balanced_ratio: float = 1.0,
        pending_extra_capacity: int = 0,
        pending_sample_ratio: float = 0.0,
    ) -> None:
        balance_config = ReplayBalanceConfig(
            win_threshold=win_threshold,
            lose_threshold=lose_threshold,
            invalid_threshold=invalid_threshold,
            balanced_ratio=balanced_ratio,
            quota_check_class=quota_check_class,
        )
        main_store = _PrioritizedReplayStore(
            max_size=max_size,
            storage_mode=storage_mode,
            save_dir=save_dir,
            overflow_margin=overflow_margin,
            alpha=alpha,
            uniform_mix=uniform_mix,
            priority_min=priority_min,
            priority_max=priority_max,
            priority_eps=priority_eps,
            age_decay=age_decay,
            max_age=max_age,
            enable_sample_decay=enable_sample_decay,
            sample_decay=sample_decay,
            enable_spread_decay=enable_spread_decay,
            spread_decay=spread_decay,
            beta_start=beta_start,
            config=balance_config,
        )
        self._main = main_store
        # Pending shares the main store's save_dir — it does not need its own folder.
        # Ownership of every .pt is tracked purely by which in-memory list holds the
        # entry (_pending._buckets vs _main.index), never by location; on commit the
        # file is adopted into _main.index by reference with its path unchanged. The
        # "pending_" filename prefix is all that keeps pending_*.pt from colliding
        # with the main store's state_*.pt / next_state_*.pt in the shared dir.
        pending_save_dir = (
            save_dir if storage_mode.lower() == "disk" and save_dir else None
        )
        self._pending = (
            _PendingReplayStore(
                balance_config,
                extra_capacity=pending_extra_capacity,
                sample_ratio=pending_sample_ratio,
                storage_mode=storage_mode,
                save_dir=pending_save_dir,
            )
            if pending_extra_capacity > 0 or pending_sample_ratio > 0
            else None
        )

    @property
    def max_size(self) -> int:
        return self._main.max_size

    @property
    def enable_spread_decay(self) -> bool:
        return self._main.enable_spread_decay

    @enable_spread_decay.setter
    def enable_spread_decay(self, value: bool) -> None:
        self._main.enable_spread_decay = bool(value)

    def store(self, *args, **kwargs):
        return self._main.store(*args, **kwargs)

    def store_unscored(self, *args, **kwargs):
        """Store new data before TD-error priority is available."""
        if self._pending is None:
            raise RuntimeError("Unscored store requires pending buffer")
        return self._pending.store(
            *args,
            replay_size=self._main.size(),
            replay_max_size=self._main.max_size,
            **kwargs,
        )

    def store_prioritized(self, *args, **kwargs):
        return self._main.store_prioritized(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self._main.sample(*args, **kwargs)

    def size(self) -> int:
        return self._main.size()

    def bucket_sizes(self) -> dict:
        return self._main.bucket_sizes()

    def mean_sample_count(self) -> float:
        return self._main.mean_sample_count()

    def reward_type_for(self, reward, done):
        return self._main.reward_type_for(reward, done)

    def get_all_entries(self):
        return self._main.get_all_entries()

    def load_from_entries(self, entries):
        return self._main.load_from_entries(entries)

    def replace_entries(self, entries, *, next_storage_id=None, insert_counter=None) -> None:
        entries = list(entries)
        self._main.index = entries
        self._main.size_count = len(entries)
        self._main.next_storage_id = (
            len(entries) if next_storage_id is None else int(next_storage_id)
        )
        self._main.insert_counter = (
            len(entries) if insert_counter is None else int(insert_counter)
        )

    def export_top_k(self, k, persistent_dir=None):
        return self._main.export_top_k(k, persistent_dir=persistent_dir)

    def top_k_balanced(self, k):
        return self._main.top_k_balanced(k)

    def sample_training_batch(self, batch_size, beta=None, device="cpu") -> MixedReplayBatch:
        if self._pending is not None:
            return self._sample_mixed_batch(batch_size, beta=beta, device=device)
        return self._wrap_main_sample(
            self._main.sample(batch_size, beta=beta, device=device, include_extra=True)
        )

    def update_priorities(self, sample_indices, td_errors, quantile_spreads=None):
        if not isinstance(sample_indices, MixedReplayBatch):
            return self._main.update_priorities(sample_indices, td_errors, quantile_spreads)

        if td_errors is None:
            return
        td_values = _ReplayStoreBase._to_float_list(td_errors)
        spread_values = (
            None
            if quantile_spreads is None
            else _ReplayStoreBase._to_float_list(quantile_spreads)
        )

        replay_count = sample_indices.replay_count
        if replay_count > 0:
            self._main.update_priorities(
                sample_indices.replay_indices,
                td_values[:replay_count],
                None if spread_values is None else spread_values[:replay_count],
            )

        if sample_indices.pending_entries:
            if self._pending is None:
                raise RuntimeError("Pending eval store is not configured")
            self._commit_pending_entries(
                sample_indices.pending_entries,
                td_values[replay_count:],
                quantile_spreads=None if spread_values is None else spread_values[replay_count:],
            )

    def pending_size(self) -> int:
        return 0 if self._pending is None else self._pending.size()

    def pending_bucket_sizes(self) -> dict[str, int]:
        if self._pending is None:
            return {reward_type.value: 0 for reward_type in self.REWARD_TYPES}
        return self._pending.bucket_sizes()

    def training_size(self) -> int:
        return self._main.size() + self.pending_size()

    def combined_bucket_sizes(self) -> dict[str, int]:
        counts = self._main.bucket_sizes()
        if self._pending is None:
            return counts

        for reward_type, pending_count in self._pending.bucket_sizes().items():
            counts[reward_type] = counts.get(reward_type, 0) + pending_count
        return counts

    def is_training_class_quota_filled(self) -> bool:
        if self._pending is None:
            return self._main.is_class_quota_filled()

        quota = self._main.class_quota
        counts = self.combined_bucket_sizes()
        quota_check_class = self._main.quota_check_class
        if quota_check_class is not None:
            return counts.get(quota_check_class, 0) >= quota
        return all(counts.get(reward_type, 0) >= quota for reward_type in self.REWARD_TYPES)

    def _pending_sample_count(self, batch_size: int) -> int:
        if self._pending is None:
            return 0

        pending_size = self._pending.size()
        if pending_size <= 0:
            return 0

        batch_size = int(batch_size)
        default_count = int(round(batch_size * self._pending.sample_ratio))
        if self._pending.sample_ratio > 0 and default_count <= 0:
            default_count = 1
        target_pending = min(pending_size, default_count)

        replay_size = int(self._main.size())
        replay_count = min(replay_size, batch_size - target_pending)
        return min(pending_size, batch_size - replay_count)

    def _sample_mixed_batch(self, batch_size: int, *, beta=None, device="cpu") -> MixedReplayBatch:
        if self._pending is None:
            raise RuntimeError("Pending eval store is not configured")

        pending_count = self._pending_sample_count(batch_size)
        replay_count = min(int(self._main.size()), int(batch_size) - pending_count)
        pending_count = min(self._pending.size(), int(batch_size) - replay_count)

        if replay_count + pending_count < batch_size:
            raise RuntimeError(
                f"Not enough replay data for batch: replay={self._main.size()}, "
                f"pending={self._pending.size()}, batch={batch_size}"
            )

        replay_batch = None
        if replay_count > 0:
            replay_batch = self._main.sample(
                replay_count,
                beta=beta,
                device=device,
                include_extra=True,
            )

        pending_entries = self._pending.sample_entries(
            pending_count,
            replay_size=self._main.size(),
            replay_max_size=self._main.max_size,
        )
        pending_batch = None
        if pending_entries:
            pending_batch = self._main.build_batch_from_entries(
                pending_entries,
                beta=beta,
                device=device,
                include_extra=True,
                is_weights=torch.ones(len(pending_entries), 1),
            )

        if replay_batch is None:
            state, action, next_state, reward, done, _, is_weights, discounts, n_steps = pending_batch
            return MixedReplayBatch(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
                replay_indices=[],
                is_weights=is_weights,
                discounts=discounts,
                n_steps=n_steps,
                replay_count=0,
                pending_entries=pending_entries,
            )

        state, action, next_state, reward, done, replay_indices, is_weights, discounts, n_steps = replay_batch
        if pending_batch is None:
            return MixedReplayBatch(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
                replay_indices=replay_indices,
                is_weights=is_weights,
                discounts=discounts,
                n_steps=n_steps,
                replay_count=replay_count,
                pending_entries=[],
            )

        p_state, p_action, p_next_state, p_reward, p_done, _, p_weights, p_discounts, p_n_steps = pending_batch
        return MixedReplayBatch(
            state=torch.cat([state, p_state], dim=0),
            action=torch.cat([action, p_action], dim=0),
            next_state=torch.cat([next_state, p_next_state], dim=0),
            reward=torch.cat([reward, p_reward], dim=0),
            done=torch.cat([done, p_done], dim=0),
            replay_indices=replay_indices,
            is_weights=torch.cat([is_weights, p_weights], dim=0),
            discounts=torch.cat([discounts, p_discounts], dim=0),
            n_steps=torch.cat([n_steps, p_n_steps], dim=0),
            replay_count=replay_count,
            pending_entries=pending_entries,
        )

    def _commit_pending_entries(self, entries, td_errors, quantile_spreads=None) -> int:
        if self._pending is None:
            raise RuntimeError("Pending eval store is not configured")
        if not entries:
            return 0

        td_values = _ReplayStoreBase._to_float_list(td_errors)
        if quantile_spreads is None:
            spread_values = [0.0] * len(td_values)
        else:
            spread_values = _ReplayStoreBase._to_float_list(quantile_spreads)

        committed_ids = set()
        try:
            for entry, td_error, spread in zip(entries, td_values, spread_values):
                priority = abs(float(td_error)) + self._main.priority_eps
                # Zero-copy adoption: the pending store already wrote (disk mode) or
                # cloned (RAM mode) the transition, so hand that single reference to the
                # main store via transition_ref instead of re-serializing or moving it.
                self._main.store_prioritized(
                    None,
                    entry["action"],
                    None,
                    entry["reward"],
                    entry["done"],
                    priority=priority,
                    discount=entry.get("discount", 1.0),
                    n_steps=entry.get("n_steps", 1),
                    tail_reward=entry.get("tail_reward", entry["reward"]),
                    quantile_spread=spread,
                    transition_ref=entry["transition"],
                )
                committed_ids.add(entry["pending_id"])
        finally:
            # Ownership of each adopted pending_*.pt has transferred to main, whose
            # _prune_if_needed will unlink it later. Drop those entries from pending
            # bookkeeping WITHOUT deleting the file — in a `finally` so that even if an
            # iteration above raised, pending can never later prune (and delete) a file
            # that main now references. prune_to_capacity only touches entries STILL in
            # pending (committed ones are already gone), so it is safe here too.
            self._pending.remove_entries(committed_ids)
            self._pending.prune_to_capacity(
                replay_size=self._main.size(),
                replay_max_size=self._main.max_size,
            )
        return len(committed_ids)

    @staticmethod
    def _wrap_main_sample(batch) -> MixedReplayBatch:
        state, action, next_state, reward, done, sample_indices, is_weights, discounts, n_steps = batch
        replay_count = state.size(0) if torch.is_tensor(state) else len(state)
        return MixedReplayBatch(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            replay_indices=sample_indices,
            is_weights=is_weights,
            discounts=discounts,
            n_steps=n_steps,
            replay_count=replay_count,
            pending_entries=[],
        )
