import random
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG

# True 時印 [DBG sample] log（含 cuda.synchronize，會拖慢訓練）；除錯 CUDA error 才開
DEBUG_CUDA_SAMPLE = False

# 由 agent 模組（visual_discrete_agent_v3 / transformer_discrete_agent）在 import 後設定。
# 設成 Path 之後 _dbg_log() 會寫到該檔，不再噴 CMD；保持 None 則完全靜默。
DEBUG_CUDA_SAMPLE_LOG_PATH = None


def _dbg_log(msg: str) -> None:
    """把 [DBG sample] 訊息寫到 DEBUG_CUDA_SAMPLE_LOG_PATH（TXT 檔），不噴 CMD。"""
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
    3. Two-axis priority decay:
       - age_decay   : suppresses stale entries over time
       - sample_decay: suppresses entries that keep getting sampled with high TD-error
                       (Method B defense against stochastic-transition traps)
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
        sample_decay: float = 0.05,
        beta_start: float = 0.4,
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
            sample_decay: Method B — divide effective priority by (1 + sample_decay × sample_count).
                sample_count is bumped every time an entry is selected via the PER (priority-weighted)
                branch in ``_sample_from_bucket`` — uniform-mix picks don't count. This slow-cools
                entries that PER keeps favoring (stochastic-trap or hard-but-persistent transitions)
                so they don't dominate the buffer or the batch indefinitely. Learned entries are
                naturally protected: low priority → low PER pick rate → sample_count stays small.
            beta_start: Initial Importance Sampling weight factor.
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
        self.sample_decay = float(sample_decay)
        self.beta = beta_start

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
        """Calculate the current priority of an entry after applying age + sample-count decay.

        Two cooling factors:
            - age_decay × age            : time since entry was stored
            - sample_decay × sample_count: how many times PER has selected this entry
              (bumped in ``_sample_from_bucket`` at priority-pick time; uniform-mix picks
              and safety-padding picks don't contribute). Main defense against
              stochastic-trap entries — they keep getting PER-selected, sample_count
              grows, effective priority decays, eventually they stop dominating.
              Learned entries are self-protected: low priority → rarely PER-picked
              → sample_count stays small.
        """
        base_priority = float(np.clip(entry.get("priority", 1.0), self.priority_min, self.priority_max))
        age = max(0, self.insert_counter - entry.get("insert_order", 0))
        if self.max_age > 0 and age > self.max_age:
            return 0.0
        sample_count = int(entry.get("sample_count", 0))
        age_factor = 1.0 + self.age_decay * age
        sample_factor = 1.0 + self.sample_decay * sample_count
        return base_priority / age_factor / sample_factor

    def top_k_balanced(self, k):
        """Select up to ``k`` entries via 50% balanced (soft floor per class) + 50% PER.

        Phase 1 — Balanced (50% of ``k``):
            Each ``reward_type`` bucket keeps top ``(k // 2) // len(REWARD_TYPES)``
            entries by ``_effective_priority``. With 4 classes that's a 12.5%
            soft floor per class — buckets with fewer entries leave slack for
            Phase 2 rather than being padded.

        Phase 2 — PER (remaining ~50% of ``k`` + Phase-1 slack):
            All entries NOT picked in Phase 1 are pooled and ranked by
            ``_effective_priority`` globally; top remaining slots fill in.
            This is where high-priority "model hasn't learned" entries get
            extra representation regardless of class.

        Pure function: does NOT mutate ``self.index`` or touch disk files.
        Returns a list of entry-dict references (aliases into ``self.index``).
        Returned length is ``min(k, size_count)``.
        """
        if k <= 0 or self.size_count == 0:
            return []
        if k >= self.size_count:
            return list(self.index)

        balanced_share = k // 2
        per_class_quota = max(1, balanced_share // len(self.REWARD_TYPES))

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
            priorities = np.array(
                [self._effective_priority(entry) for entry in available],
                dtype=np.float64,
            )
            # Apply alpha exponent
            priorities = np.power(np.maximum(priorities, self.priority_min), self.alpha)
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
            # Method B — bump sample_count for every PER-selected entry.
            # Uniform picks above and safety padding below are NOT bumped (they don't represent
            # "PER kept favoring this entry"). Learned entries are naturally protected because
            # their low priority means they rarely land in priority_pick.
            for entry in priority_pick:
                entry["sample_count"] = int(entry.get("sample_count", 0)) + 1
            chosen.extend(priority_pick)

        # Safety padding — these are uniform-random fallbacks, not PER picks → no sample_count bump
        while len(chosen) < count:
            chosen.append(random.choice(entries))

        return chosen[:count]

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
            "sample_count": 0,  # Method B: bumps in _sample_from_bucket every time this entry is PER-selected
        }

        self.index.append(entry)
        self.size_count = len(self.index)
        self._prune_if_needed()

    def update_priorities(self, sample_indices, td_errors):
        """Update priorities using TD-error.

        Note: ``sample_count`` is bumped at PER pick time in ``_sample_from_bucket``,
        NOT here — so this function only refreshes ``priority`` from the latest TD-error.
        """
        if sample_indices is None or td_errors is None:
            return

        td_values = td_errors.detach().float().view(-1).cpu().tolist() if torch.is_tensor(td_errors) else td_errors
        for idx, td_error in zip(sample_indices, td_values):
            if not (0 <= int(idx) < len(self.index)):
                continue
            priority = abs(float(td_error)) + self.priority_eps
            self.index[int(idx)]["priority"] = float(np.clip(priority, self.priority_min, self.priority_max))

    def sample(self, batch_size, beta=None, device="cpu", include_extra=False):
        """Sample a batch via 50% balanced (soft floor per class, with uniform_mix)
        + 50% PER (pure-priority leftover pool, no uniform_mix).

        Phase 1 — Balanced half:
            ``batch_size // 2`` slots, split into ``per_class_count = (batch_size // 2) // 4``
            per reward_type. Each non-empty class contributes up to ``per_class_count``
            entries via ``_sample_from_bucket(use_uniform=True)`` — the uniform_mix
            piece gives exploration noise within each class.

        Phase 2 — PER half:
            All entries not yet picked are pooled and sampled by
            ``_sample_from_bucket(use_uniform=False)`` — pure priority weighting,
            cross-class. Soft-floor slack from Phase 1 (empty / starved classes)
            flows into this pool automatically.

        Note for specific models:
        Visual Agent typically doesn't use IS weights (ignores it),
        Transformer Agent heavily relies on them.
        """
        if self.size_count <= 0:
            raise RuntimeError("CategorizedReplayBuffer is empty")

        use_beta = beta if beta is not None else self.beta

        # === Phase 1: Balanced half ===
        balanced_total = batch_size // 2
        per_class_count = max(1, balanced_total // len(self.REWARD_TYPES))

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

        # === Phase 2: PER half — fill remaining slots from leftover pool by pure priority ===
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

        states, actions, next_states, rewards, dones, priorities, discounts, n_steps = [], [], [], [], [], [], [], []
        
        for idx in selected_indices:
            entry = self.index[idx]
            
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
            priorities.append(self._effective_priority(entry))
            discounts.append(float(entry.get("discount", 1.0)))
            n_steps.append(int(entry.get("n_steps", 1)))

        # ---- DBG: sanity-check stored actions before they get used to index Q tensor ----
        # 只有 DEBUG_CUDA_SAMPLE 開且 LOG_PATH 有設時才寫；不再 hard-code 到 v3 路徑、不噴 CMD。
        if DEBUG_CUDA_SAMPLE and DEBUG_CUDA_SAMPLE_LOG_PATH is not None:
            try:
                _bad_acts = []
                for i, action_value in enumerate(actions):
                    try:
                        action_int = int(np.asarray(action_value).reshape(-1)[0])
                    except Exception:
                        _bad_acts.append((i, action_value))
                        continue
                    # 用 storage 裡實際存的 action 值是否能被解析為整數來判斷；上界由呼叫端的
                    # _dbg_tensor(expect_max=...) 在 GPU 端做更精確的 OOB 檢查。
                if _bad_acts:
                    _dbg_log(f"[REPLAY sample] !!BAD ACTIONS in batch!! {_bad_acts[:10]} (showing up to 10)")
            except Exception:
                pass

        # Important Sampling
        N = self.size_count
        priorities_arr = np.array(priorities, dtype=np.float64)
        prob_sum = sum([self._effective_priority(entry) for entry in self.index])
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

    @property
    def class_quota(self) -> int:
        """Per-class soft-floor quota used by ``top_k_balanced`` and ``sample()``.

        With 4 classes and a 50% balanced share, this equals ``max_size // 8``
        (i.e. 12.5% of buffer capacity per class).
        """
        return max(1, (self.max_size // 2) // len(self.REWARD_TYPES))

    def is_class_quota_filled(self) -> bool:
        """True iff every reward_type bucket has at least ``class_quota`` entries.

        Useful as a training-readiness gate on top of ``MINIMUM_DATA_SIZE``: ensures
        the buffer can actually deliver a balanced batch (each class can fill its
        12.5% soft-floor slot in ``sample()``).
        """
        quota = self.class_quota
        return all(count >= quota for count in self.bucket_sizes().values())

    def get_all_entries(self):
        """Returns internal objects suitable for RAM persistent saving. 
        Note this won't move disk files, just the internal state index."""
        return self.index

    def load_from_entries(self, entries):
        """Load from a persistent index map structure in RAM。

        Backward compatibility:
            - Legacy entries without ``sample_count`` get default 0 (so they start fresh
              in the Method-B decay scheme).
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
            # Default Method-B sample_count for legacy entries
            normalized_entry.setdefault("sample_count", 0)
            normalized_entries.append(normalized_entry)
        self.index = normalized_entries
        self.size_count = len(self.index)
        self.next_storage_id = max([e["storage_id"] for e in self.index], default=-1) + 1
        self.insert_counter = max([e["insert_order"] for e in self.index], default=0)
