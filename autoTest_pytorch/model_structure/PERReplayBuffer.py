import numpy as np
import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PER 超參數
PER_CAPACITY = 50000  # 單一 buffer 容量
PER_ALPHA = 0.6       # Priority 指數 (0=uniform, 1=full priority)
PER_BETA_START = 0.4  # IS-weight 初始值
PER_BETA_END = 1.0    # IS-weight 最終值
PER_EPSILON = 1e-5    # 防止 priority=0

class SumTree:
    """Sum Tree 資料結構，用於 O(log n) 的優先級抽樣。

    葉節點存放 priority，內部節點存放子節點的 priority 總和。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx, change):
        """從葉節點往上更新 parent 的總和。"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """根據累積 sum 找到對應的葉節點。"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def update(self, data_idx, priority):
        """更新某個 data index 的 priority。"""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority):
        """新增一筆 priority，回傳對應的 data index。"""
        data_idx = self.write_ptr
        self.update(data_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def get(self, s):
        """根據累積 sum 值 s 找到 (data_idx, priority)。"""
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return data_idx, self.tree[tree_idx]

    def max_priority(self):
        """目前所有葉節點的最大 priority。"""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.size
        if self.size == 0:
            return 1.0
        return max(self.tree[leaf_start:leaf_end])

class PERReplayBuffer:
    """Prioritized Experience Replay buffer，用 SumTree 做 O(log n) 抽樣。

    Priority = (|TD-error| + epsilon) ^ alpha
    高 TD-error 的 transition 被更常抽到，也更不容易被覆蓋。
    """

    def __init__(self, capacity=PER_CAPACITY, alpha=PER_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.data = [None] * capacity

    def store(self, state, action, next_state, reward, done):
        """存 transition，初始 priority 設為目前最大值（保證至少被抽到一次）。"""
        entry = {
            'state': state.cpu().clone() if isinstance(state, torch.Tensor) else state,
            'action': action.copy() if isinstance(action, np.ndarray) else np.array(action),
            'next_state': next_state.cpu().clone() if isinstance(next_state, torch.Tensor) else next_state,
            'reward': float(reward),
            'done': bool(done),
        }
        priority = self.tree.max_priority()
        idx = self.tree.add(priority)
        self.data[idx] = entry

    def sample(self, batch_size, beta=PER_BETA_START):
        """Priority-proportional 抽樣 + importance sampling weights。

        Returns:
            (states, actions, next_states, rewards, dones, indices, weights)
        """
        indices = []
        priorities = []
        entries = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority = self.tree.get(s)

            # 防止抽到空 slot
            if self.data[idx] is None:
                idx = random.randint(0, self.tree.size - 1)
                priority = self.tree.tree[idx + self.tree.capacity - 1]

            indices.append(idx)
            priorities.append(priority)
            entries.append(self.data[idx])

        # Importance Sampling weights
        total = self.tree.total()
        N = self.tree.size
        priorities_arr = np.array(priorities, dtype=np.float64)
        probabilities = priorities_arr / (total + 1e-10)
        weights = (N * probabilities + 1e-10) ** (-beta)
        weights = weights / weights.max()  # normalize to [0, 1]
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        # 組裝 batch
        states = torch.stack([e['state'] for e in entries]).to(device)
        actions = torch.tensor(
            np.array([e['action'] for e in entries]),
            dtype=torch.float32,
        ).to(device)
        next_states = torch.stack([e['next_state'] for e in entries]).to(device)
        rewards = torch.tensor(
            [e['reward'] for e in entries],
            dtype=torch.float32,
        ).unsqueeze(1).to(device)
        dones = torch.tensor(
            [float(e['done']) for e in entries],
            dtype=torch.float32,
        ).unsqueeze(1).to(device)

        return states, actions, next_states, rewards, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        """用 TD-error 更新 priorities。"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + PER_EPSILON) ** self.alpha
            self.tree.update(idx, priority)

    def size(self):
        return self.tree.size

    def get_all_entries(self):
        """取得所有 entries（用於 persistent save）。"""
        entries = []
        for i in range(self.tree.size):
            if self.data[i] is not None:
                entries.append(self.data[i])
        return entries

    def get_top_entries(self, k):
        """取得 priority 最高的 k 筆 entries（用於 persistent save）。"""
        # 收集 (priority, index) pairs
        priorities = []
        for i in range(self.tree.size):
            if self.data[i] is not None:
                tree_idx = i + self.tree.capacity - 1
                priority = self.tree.tree[tree_idx]
                priorities.append((priority, i))

        # 按 priority 降序排序，取 top-k
        priorities.sort(key=lambda x: -x[0])
        top_k = priorities[:k]
        return [self.data[idx] for _, idx in top_k]

