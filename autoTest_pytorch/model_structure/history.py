"""history.py — Per-episode outcome tracker。

從 AdaptiveEpsilonController 拆出來,符合 SRP:
    - History 只負責「累積/查詢 episode 結果」
    - AdaptiveEpsilonController 只負責「依 win rate 算 epsilon」(吃 win_rate 為參數)

Class 結構:
    - History          : base class,提供 record / win_rate / state_dict 等通用 API
    - TrainingHistory  : training loop 用的 subclass(目前無額外行為,留作 namespace)
    - eval 端目前直接 init 一個 ephemeral `History(max_capacity=num_episodes)`,
      用 `window=None` 取整個 session 的 mean (見 run_fixed_policy_evaluation)。
      之後若要做「跨 eval session 的時間序」再加 EvalHistory subclass。

設計重點:
    - max_capacity 預設 100,當 caller 要求更大的 window 時自動擴張
      (rebuild deque + 把現有資料 copy 進去)
    - Sample 不足時用「現有資料」直接算 — 跟 controller 舊行為一致,呼叫端不變
    - state_dict() 自成一格,checkpoint 走獨立 .pth 檔(不再跟 optimizer state 混在一起)
"""

from __future__ import annotations

from collections import deque


class History:
    """Per-episode outcome tracker base class。

    用法：
        history = History()                      # max_capacity=100
        history.record(win=True)                 # 每場 episode 結束呼叫一次
        wr = history.win_rate(window=100)        # 取最後 100 場 win rate
    """

    def __init__(self, max_capacity: int = 100):
        self._max_capacity = int(max(1, max_capacity))
        self._results: deque[int] = deque(maxlen=self._max_capacity)
        self._rewards: deque[float] = deque(maxlen=self._max_capacity)
        self._steps: deque[int] = deque(maxlen=self._max_capacity)
        self._invalid_rates: deque[float] = deque(maxlen=self._max_capacity)
        self.total_episodes: int = 0

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    # ──────────────────────────── update ────────────────────────────

    def record(
        self,
        win: bool,
        *,
        total_reward: float = 0.0,
        steps: int = 0,
        invalid_rate: float = 0.0,
    ) -> None:
        """Append a single episode outcome 與 reward / steps / invalid_rate。

        total_reward / steps / invalid_rate 是 keyword-only 且有預設值,
        讓舊呼叫端 `record(win=...)` 仍可運作 — 只是這場不會貢獻對應統計。
        """
        self._results.append(int(bool(win)))
        self._rewards.append(float(total_reward))
        self._steps.append(int(steps))
        self._invalid_rates.append(float(invalid_rate))
        self.total_episodes += 1

    # ──────────────────────────── query ─────────────────────────────

    def _rolling_mean(self, source: deque, window: int | None) -> float:
        """共用的 rolling mean。

        - window=None 或 window >= len(source):用「全部現有資料」算 (eval
          session 通常這樣用 — 取整個 session 的 mean,不要 rolling window)
        - window > 目前 max_capacity:先擴張 capacity 再算 (sample 不足時
          仍用現有資料,跟舊 controller 行為一致)
        """
        if not source:
            return 0.0
        
        if window is None:
            samples = source
        else:
            window = int(max(1, window))
            if window > self._max_capacity:
                self._grow_capacity(window)
            if window >= len(source):
                samples = source
            else:
                samples = list(source)[-window:]
        return sum(samples) / len(samples)

    def win_rate(self, window: int | None = 100) -> float:
        """Rolling win rate over the last `window` episodes。

        window=None 表示「用全部現有資料」(eval session 慣用法)。

        若 window > 目前 max_capacity,先擴張 max_capacity (rebuild deque +
        copy 現有資料);擴張後尚未累積到 window 場時,用「現有資料」直接算 —
        跟舊 AdaptiveEpsilonController 的 rolling_win_rate() 行為一致。

        呼叫端拿到的數字會隨 sample 數逐步穩定;若對 sample 不足敏感,
        可同時 query `len(history)` 或 `total_episodes` 自行判斷。
        """
        return self._rolling_mean(self._results, window)

    def avg_reward(self, window: int | None = 100) -> float:
        """Rolling 平均整場 reward (跨 window 場,每場一個值)。

        window=None 表示「用全部現有資料」(eval session 慣用法)。

        注意:這裡記的是「整場 total reward」,不是 per-step mean。
        per-step mean 由 caller 自行從 (total_reward, steps) 算出。
        """
        return self._rolling_mean(self._rewards, window)

    def avg_steps(self, window: int | None = 100) -> float:
        """Rolling 平均每場 step 數。window=None 用全部現有資料。"""
        return self._rolling_mean(self._steps, window)

    def avg_invalid_rate(self, window: int | None = 100) -> float:
        """Rolling 平均每場 invalid click rate。window=None 用全部現有資料。

        Caller 在 `record(invalid_rate=...)` 時要傳這場的 invalid rate
        (0.0~1.0)。沒傳就視為 0.0,於是這場對 mean 的貢獻是 0。
        """
        return self._rolling_mean(self._invalid_rates, window)

    def __len__(self) -> int:
        return len(self._results)

    # ──────────────────────────── internal ──────────────────────────

    def _grow_capacity(self, new_capacity: int) -> None:
        """擴張所有 deque 的 maxlen。既有資料保留;sample 滿到新長度之前,
        win_rate / avg_reward / avg_steps / avg_invalid_rate 會用「現有資料」算
        (sample 不足的可接受 trade-off)。
        """
        new_capacity = int(new_capacity)
        if new_capacity <= self._max_capacity:
            return
        self._max_capacity = new_capacity
        self._results = deque(self._results, maxlen=self._max_capacity)
        self._rewards = deque(self._rewards, maxlen=self._max_capacity)
        self._steps = deque(self._steps, maxlen=self._max_capacity)
        self._invalid_rates = deque(self._invalid_rates, maxlen=self._max_capacity)

    # ──────────────────────────── checkpoint ────────────────────────

    def state_dict(self) -> dict:
        return {
            "max_capacity": self._max_capacity,
            "results": list(self._results),
            "rewards": list(self._rewards),
            "steps": list(self._steps),
            "invalid_rates": list(self._invalid_rates),
            "total_episodes": self.total_episodes,
        }

    def load_state_dict(self, state: dict, *, deque_cls=deque) -> None:
        """從 checkpoint dict 還原。

        deque_cls 沿用舊 AdaptiveEpsilonController 的擴充點,允許 agent
        注入自己的 deque 子類別(目前各 agent 都直接傳 collections.deque)。

        相容性:
            - 新格式 key 是 "results";舊扁平 checkpoint 用的是 "result_window",
              這裡兩個都吃,讓 mixin 在 migration 階段可以直接餵舊 dict。
            - "rewards" / "steps" / "invalid_rates" 是新加的 key;舊 checkpoint
              沒有 → 用空 deque 讓新指標從 resume 之後重新累積
              (avg_reward / avg_steps / avg_invalid_rate 初期會返回 0.0)。
        """
        if not state:
            return
        new_cap = int(state.get("max_capacity", self._max_capacity))
        self._max_capacity = max(1, new_cap)
        results = state.get("results", state.get("result_window", []))
        self._results = deque_cls(results, maxlen=self._max_capacity)
        rewards = state.get("rewards", [])
        steps = state.get("steps", [])
        invalid_rates = state.get("invalid_rates", [])
        self._rewards = deque_cls(rewards, maxlen=self._max_capacity)
        self._steps = deque_cls(steps, maxlen=self._max_capacity)
        self._invalid_rates = deque_cls(invalid_rates, maxlen=self._max_capacity)
        self.total_episodes = int(state.get("total_episodes", 0))


class TrainingHistory(History):
    """Training loop 用的 History subclass。

    base class 已包含 win / total_reward / steps 三個 rolling 統計,
    training script 不再需要自己維護 recent_* deque,直接 query 這裡即可。
    """
    pass
