"""adaptive_epsilon.py — 共用的 adaptive epsilon controller。

把 v3 / stage1 (TransformerDiscreteAgent) 內重複的「依 rolling win rate 調整 epsilon」
邏輯抽出來成一個獨立的 class，避免兩處實作各自漂移。

策略：
    - 維護最近 N 場 episode 結果的 sliding window
    - win rate 在 [wr_min, wr_max] 之間時，以對數內插把 epsilon 從 eps_max
      平滑壓到 eps_min；win rate < wr_min 時固定 eps_max；> wr_max 時固定 eps_min
    - state_dict 的 keys (`epsilon` / `total_episodes` / `total_wins` /
      `result_window`) 與既有存檔格式相容，舊 checkpoint 直接餵進去就能還原
"""

from __future__ import annotations

import math
from collections import deque


class AdaptiveEpsilonController:
    """Win-rate-based adaptive epsilon scheduler。

    用法：
        controller = AdaptiveEpsilonController()
        ...
        controller.record_episode(win=True)   # 每場結束時呼叫一次
        eps = controller.epsilon              # 拿來決定 explore/exploit
    """

    def __init__(
        self,
        wr_min: float = 0.2,
        wr_max: float = 0.85,
        eps_min: float = 0.02, # Don’t be smaller than 0.02. The lack of bad-action data may cause the model to forget how to avoid poor actions.
        eps_max: float = 0.30,
        window_size: int = 100,
    ):
        self.wr_min = wr_min
        self.wr_max = wr_max
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.window_size = window_size

        self.epsilon: float = eps_max
        self.result_window: deque[int] = deque(maxlen=window_size)
        self.total_episodes: int = 0
        self.total_wins: int = 0

    # ──────────────────────────── update ────────────────────────────

    def record_episode(self, win: bool) -> float:
        """Append a result, refresh epsilon, return the new epsilon."""
        self.total_episodes += 1
        self.total_wins += int(bool(win))
        self.result_window.append(int(bool(win)))
        self.epsilon = self.compute_epsilon()
        return self.epsilon

    def compute_epsilon(self) -> float:
        """Log-interpolate epsilon from rolling win rate（不修改 self.epsilon）。"""
        if not self.result_window:
            return self.eps_max
        wr = sum(self.result_window) / len(self.result_window)
        wr = max(self.wr_min, min(self.wr_max, wr))
        t = (wr - self.wr_min) / (self.wr_max - self.wr_min)
        return math.exp(
            math.log(self.eps_max)
            + (math.log(self.eps_min) - math.log(self.eps_max)) * t
        )

    # ──────────────────────────── queries ────────────────────────────

    def rolling_win_rate(self) -> float:
        if not self.result_window:
            return 0.0
        return sum(self.result_window) / len(self.result_window)

    def overall_win_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_wins / self.total_episodes

    # ──────────────────────────── checkpointing ──────────────────────

    def state_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "total_episodes": self.total_episodes,
            "total_wins": self.total_wins,
            "result_window": list(self.result_window),
        }

    def load_state_dict(self, state: dict, *, deque_cls=deque) -> None:
        """從 checkpoint dict 還原狀態。

        deque_cls 允許 agent 注入自己的 deque 子類別（v3 會在 __init__ 紀錄
        self.deque_cls，這裡保留相同的擴充點）。
        """
        if not state:
            return
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.total_episodes = int(state.get("total_episodes", 0))
        self.total_wins = int(state.get("total_wins", 0))
        self.result_window = deque_cls(
            state.get("result_window", []),
            maxlen=self.window_size,
        )
