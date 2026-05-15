"""history.py — Per-episode outcome tracker。

從 AdaptiveEpsilonController 拆出來,符合 SRP:
    - History 只負責「累積/查詢 episode 結果」
    - AdaptiveEpsilonController 只負責「依 win rate 算 epsilon」(吃 win_rate 為參數)

Class 結構:
    - History          : base class,提供 record / win_rate / state_dict 等通用 API
    - TrainingHistory  : training loop 用的 subclass(目前無額外行為,留作 namespace)
    - eval 端目前無 eval-specific 需求,直接用 History 即可;有需要再加 EvalHistory。

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
        self.total_episodes: int = 0

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    # ──────────────────────────── update ────────────────────────────

    def record(self, win: bool) -> None:
        """Append a single episode outcome (1=win, 0=loss)。"""
        outcome = int(bool(win))
        self._results.append(outcome)
        self.total_episodes += 1

    # ──────────────────────────── query ─────────────────────────────

    def win_rate(self, window: int = 100) -> float:
        """Rolling win rate over the last `window` episodes.

        若 window > 目前 max_capacity,先擴張 max_capacity (rebuild deque +
        copy 現有資料);擴張後尚未累積到 window 場時,用「現有資料」直接算 —
        跟舊 AdaptiveEpsilonController 的 rolling_win_rate() 行為一致。

        呼叫端拿到的數字會隨 sample 數逐步穩定;若對 sample 不足敏感,
        可同時 query `len(history)` 或 `total_episodes` 自行判斷。
        """
        window = int(max(1, window))
        if window > self._max_capacity:
            self._grow_capacity(window)
        if not self._results:
            return 0.0
        if window >= len(self._results):
            samples = self._results
        else:
            samples = list(self._results)[-window:]
        return sum(samples) / len(samples)

    def __len__(self) -> int:
        return len(self._results)

    # ──────────────────────────── internal ──────────────────────────

    def _grow_capacity(self, new_capacity: int) -> None:
        """擴張 deque 的 maxlen。既有資料保留;sample 滿到新長度之前,
        win_rate(N) 會用「現有資料」算 (sample 不足的可接受 trade-off)。
        """
        new_capacity = int(new_capacity)
        if new_capacity <= self._max_capacity:
            return
        self._max_capacity = new_capacity
        self._results = deque(self._results, maxlen=self._max_capacity)

    # ──────────────────────────── checkpoint ────────────────────────

    def state_dict(self) -> dict:
        return {
            "max_capacity": self._max_capacity,
            "results": list(self._results),
            "total_episodes": self.total_episodes,
        }

    def load_state_dict(self, state: dict, *, deque_cls=deque) -> None:
        """從 checkpoint dict 還原。

        deque_cls 沿用舊 AdaptiveEpsilonController 的擴充點,允許 agent
        注入自己的 deque 子類別(目前各 agent 都直接傳 collections.deque)。

        相容性:
            - 新格式 key 是 "results";舊扁平 checkpoint 用的是 "result_window",
              這裡兩個都吃,讓 mixin 在 migration 階段可以直接餵舊 dict。
        """
        if not state:
            return
        new_cap = int(state.get("max_capacity", self._max_capacity))
        self._max_capacity = max(1, new_cap)
        results = state.get("results", state.get("result_window", []))
        self._results = deque_cls(results, maxlen=self._max_capacity)
        self.total_episodes = int(state.get("total_episodes", 0))


class TrainingHistory(History):
    """Training loop 用的 History subclass。

    目前無 training-specific 行為,純作 namespace 用 — 之後若要加
    training-only 的統計(e.g. invalid_rate / reward 平均…)可以在這加 field。
    """
    pass
