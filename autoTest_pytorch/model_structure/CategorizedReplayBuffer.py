import random
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

class CategorizedReplayBuffer:
    """A generic Categorized Replay Buffer.
    
    Supports:
    1. Categorized Bucketing based on reward thresholds (win/lose/progress/invalid/other).
    2. RAM-based storage (for small states) or Disk-backed storage (for memory-heavy images).
    3. Age-decay priority for preventing stale strong experiences from taking up priority.
    """

    REWARD_TYPES = ("win", "lose", "invalid", "progress", "other")

    def __init__(
        self,
        max_size: int,
        storage_mode: str = "ram",
        save_dir: str = None,
        win_threshold: float = 3.0,
        lose_threshold: float = -1.0,
        invalid_threshold: float = 0.0,
        overflow_margin: int = 256,
        alpha: float = 0.6,
        uniform_mix: float = 0.2,
        priority_min: float = 0.05,
        priority_max: float = 5.0,
        priority_eps: float = 1e-3,
        age_decay: float = 0.002,
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
            priority_min: Lowest possible priority score an item can have.
            priority_max: Cap on initial priority values.
            priority_eps: Small epsilon added to TD Error to prevent 0 priority.
            age_decay: Decay factor removing priority based on how many inserts occurred since entry.
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
        self.beta = beta_start

        self.size_count = 0
        self.index = []
        self.next_storage_id = 0
        self.insert_counter = 0


    def _reward_type(self, reward, done):
        reward = float(reward)
        if done and reward >= self.win_threshold:
            return "win"
        if done and reward <= self.lose_threshold:
            return "lose"
        if reward < self.invalid_threshold:
            return "invalid"
        if reward > self.invalid_threshold:
            return "progress"
        return "other"

    def _save_tensor(self, tensor, root_name, storage_id):
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
        if self.storage_mode == "disk":
            tensor = torch.load(reference)
            if tensor.dtype == torch.uint8:
                return tensor.float() / 255.0
            return tensor
        else:
            return reference

    def _safe_unlink(self, path_str):
        if not path_str:
            return
        path = Path(path_str)
        if path.exists():
            path.unlink()

    def _delete_entry_files(self, entry):
        if self.storage_mode == "disk":
            self._safe_unlink(entry.get("state"))
            self._safe_unlink(entry.get("next_state"))

    def _effective_priority(self, entry):
        base_priority = float(np.clip(entry.get("priority", 1.0), self.priority_min, self.priority_max))
        age = max(0, self.insert_counter - entry.get("insert_order", 0))
        aged_priority = base_priority / (1.0 + self.age_decay * age)
        return max(self.priority_min, aged_priority)

    def _prune_if_needed(self):
        if self.size_count <= self.max_size + self.overflow_margin:
            return

        bucket_quota = max(1, self.max_size // len(self.REWARD_TYPES))
        grouped_entries = {reward_type: [] for reward_type in self.REWARD_TYPES}
        for entry in self.index:
            grouped_entries.setdefault(entry["reward_type"], []).append(entry)

        selected_ids = set()
        survivors = []
        leftovers = []

        for reward_type in self.REWARD_TYPES:
            entries = grouped_entries.get(reward_type, [])
            entries.sort(key=self._effective_priority, reverse=True)
            keep = entries[:bucket_quota]
            spill = entries[bucket_quota:]
            survivors.extend(keep)
            selected_ids.update(entry["storage_id"] for entry in keep)
            leftovers.extend(spill)

        remaining_slots = max(0, self.max_size - len(survivors))
        if remaining_slots > 0 and leftovers:
            leftovers.sort(key=self._effective_priority, reverse=True)
            extra = leftovers[:remaining_slots]
            survivors.extend(extra)
            selected_ids.update(entry["storage_id"] for entry in extra)

        old_entries = self.index
        self.index = survivors[:self.max_size]
        self.size_count = len(self.index)

        # Cleanup deleted elements
        for entry in old_entries:
            if entry["storage_id"] not in selected_ids:
                self._delete_entry_files(entry)

    def _sample_from_bucket(self, entries, count):
        if count <= 0 or not entries:
            return []

        num_uniform = min(len(entries), int(round(count * self.uniform_mix)))
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
            chosen.extend(priority_pick)

        # Safety padding
        while len(chosen) < count:
            chosen.append(random.choice(entries))

        return chosen[:count]

    def store(self, state, action, next_state, reward, done):
        """Store a transition."""
        self.insert_counter += 1
        storage_id = self.next_storage_id
        self.next_storage_id += 1

        state_ref = self._save_tensor(state, "state", storage_id)
        next_state_ref = None
        if next_state is not None:
            next_state_ref = self._save_tensor(next_state, "next_state", storage_id)

        entry = {
            "storage_id": storage_id,
            "state": state_ref,
            "action": action.copy() if isinstance(action, np.ndarray) else np.array(action),
            "next_state": next_state_ref,
            "reward": float(reward),
            "done": bool(done),
            "reward_type": self._reward_type(reward, done),
            "priority": float(np.clip(abs(float(reward)) + 1.0, self.priority_min, self.priority_max)),
            "insert_order": self.insert_counter,
        }

        self.index.append(entry)
        self.size_count = len(self.index)
        self._prune_if_needed()

    def update_priorities(self, sample_indices, td_errors):
        """Update priorities using TD-error."""
        if sample_indices is None or td_errors is None:
            return

        td_values = td_errors.detach().float().view(-1).cpu().tolist() if torch.is_tensor(td_errors) else td_errors
        for idx, td_error in zip(sample_indices, td_values):
            if not (0 <= int(idx) < len(self.index)):
                continue
            priority = abs(float(td_error)) + self.priority_eps
            self.index[int(idx)]["priority"] = float(np.clip(priority, self.priority_min, self.priority_max))

    def sample(self, batch_size, beta=None, device="cpu"):
        """Sample a batch of transitions. Returns importance sampling weights if beta is used.
        
        Note for specific models:
        Visual Agent typically doesn't use IS weights (ignores it), 
        Transformer Agent heavily relies on them.
        """
        if self.size_count <= 0:
            raise RuntimeError("CategorizedReplayBuffer is empty")
            
        use_beta = beta if beta is not None else self.beta

        grouped_indices = {reward_type: [] for reward_type in self.REWARD_TYPES}
        for idx, entry in enumerate(self.index):
            grouped_indices.setdefault(entry["reward_type"], []).append(idx)

        non_empty_groups = [indices for indices in grouped_indices.values() if indices]
        base_count = max(1, batch_size // max(1, len(non_empty_groups)))
        selected_indices = []

        # Try to pull evenly across available groups
        for indices in non_empty_groups:
            take = min(base_count, len(indices))
            bucket_entries = [self.index[idx] for idx in indices]
            sampled_entries = self._sample_from_bucket(bucket_entries, take)
            selected_indices.extend(
                next(i for i in indices if self.index[i]["storage_id"] == entry["storage_id"])
                for entry in sampled_entries
            )

        # Pad remaining if unevenly divided
        while len(selected_indices) < batch_size:
            fallback_entry = self._sample_from_bucket(self.index, 1)[0]
            selected_indices.append(
                next(
                    i for i, entry in enumerate(self.index)
                    if entry["storage_id"] == fallback_entry["storage_id"]
                )
            )

        selected_indices = selected_indices[:batch_size]

        states, actions, next_states, rewards, dones, priorities = [], [], [], [], [], []
        
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

        # Important Sampling
        N = self.size_count
        priorities_arr = np.array(priorities, dtype=np.float64)
        prob_sum = sum([self._effective_priority(entry) for entry in self.index])
        probabilities = priorities_arr / (prob_sum + 1e-10)
        
        # weights formulation: (1/N * 1/P_i) ^ beta
        weights = (N * probabilities + 1e-10) ** (-use_beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        # Stacking tensors manually depends heavily on the model requirements:
        # Returning lists or direct tensors:
        tensor_states = torch.stack(states).to(device) if torch.is_tensor(states[0]) else states
        tensor_next_states = torch.stack(next_states).to(device) if torch.is_tensor(next_states[0]) else next_states
        
        tensor_actions = torch.tensor(np.array(actions), dtype=torch.float32 if not isinstance(actions[0], (int, np.integer)) else torch.long, device=device)
        tensor_rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        tensor_dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        # To support both signatures: (states, actions, next, rewards, dones, indices, weights) is standard.
        return (
            tensor_states,
            tensor_actions,
            tensor_next_states,
            tensor_rewards,
            tensor_dones,
            selected_indices,
            weights
        )

    def size(self):
        return self.size_count

    def get_all_entries(self):
        """Returns internal objects suitable for RAM persistent saving. 
        Note this won't move disk files, just the internal state index."""
        return self.index

    def load_from_entries(self, entries):
        """Load from a persistent index map structure in RAM"""
        self.index = entries
        self.size_count = len(self.index)
        self.next_storage_id = max([e["storage_id"] for e in self.index], default=-1) + 1
        self.insert_counter = max([e["insert_order"] for e in self.index], default=0)
