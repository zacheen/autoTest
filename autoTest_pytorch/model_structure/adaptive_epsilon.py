"""adaptive_epsilon.py — 共用的 adaptive epsilon controller。

只負責「依 caller 傳進來的 win rate 算 epsilon」。Episode 結果累積/查詢已搬到
History / TrainingHistory (model_structure/history.py),呼叫端要先 record 進
history,再把 win_rate 餵進來。

策略：
    - win rate 在 [wr_min, wr_max] 之間時,以對數內插把 epsilon 從 eps_max
      平滑壓到 eps_min;win rate < wr_min 時固定 eps_max;> wr_max 時固定 eps_min
    - state_dict 只剩 epsilon 一個值 (與 optimizer state 一起存)。history
      由 TrainingHistory 寫到獨立 .pth 檔。
"""

from __future__ import annotations

import math


class AdaptiveEpsilonController:
    """Win-rate-based adaptive epsilon scheduler。

    用法：
        controller = AdaptiveEpsilonController()
        ...
        history.record(win=True)
        wr = history.win_rate(window=100)
        eps = controller.update(wr)     # 更新 self.epsilon + return
    """

    def __init__(
        self,
        wr_min: float = 0.2,
        wr_max: float = 0.85,
        eps_min: float = 0.02,  # Don’t be smaller than 0.02. The lack of bad-action data may cause the model to forget how to avoid poor actions.
        eps_max: float = 0.30,
    ):
        self.wr_min = wr_min
        self.wr_max = wr_max
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.epsilon: float = eps_max

    # ──────────────────────────── update ────────────────────────────

    def update(self, win_rate: float) -> float:
        """以 caller 算好的 win rate 推導 next epsilon,更新 self.epsilon 並回傳。"""
        self.epsilon = self.compute_epsilon(win_rate)
        return self.epsilon

    def compute_epsilon(self, win_rate: float) -> float:
        """Log-interpolate epsilon from a given win rate (pure, no side effect)。"""
        wr = max(self.wr_min, min(self.wr_max, float(win_rate)))
        t = (wr - self.wr_min) / (self.wr_max - self.wr_min)
        return math.exp(
            math.log(self.eps_max)
            + (math.log(self.eps_min) - math.log(self.eps_max)) * t
        )

    # ──────────────────────────── checkpoint ────────────────────────

    def state_dict(self) -> dict:
        return {"epsilon": self.epsilon}

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.epsilon = float(state.get("epsilon", self.epsilon))
