"""Shared adaptive epsilon controller.

Only computes epsilon from the caller-provided win rate. Episode result storage
and queries live in History / TrainingHistory (model_structure/history.py);
callers should record into history first, then pass in win_rate.

Policy:
    - Log-interpolate epsilon from eps_max to eps_min while win rate is in
      [wr_min, wr_max]; clamp to eps_max below wr_min and eps_min above wr_max.
    - state_dict stores only epsilon with optimizer state. TrainingHistory writes
      history to its own .pth file.
"""

from __future__ import annotations

import math


class AdaptiveEpsilonController:
    """Win-rate-based adaptive epsilon scheduler.

    Usage:
        controller = AdaptiveEpsilonController()
        ...
        history.record(win=True)
        wr = history.win_rate(window=100)
        eps = controller.update(wr)     # update and return self.epsilon
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

    # update

    def update(self, win_rate: float) -> float:
        """Compute next epsilon from caller-provided win rate, update, and return it."""
        self.epsilon = self.compute_epsilon(win_rate)
        return self.epsilon

    def compute_epsilon(self, win_rate: float) -> float:
        """Log-interpolate epsilon from a given win rate (pure, no side effect)."""
        wr = max(self.wr_min, min(self.wr_max, float(win_rate)))
        t = (wr - self.wr_min) / (self.wr_max - self.wr_min)
        return math.exp(
            math.log(self.eps_max)
            + (math.log(self.eps_min) - math.log(self.eps_max)) * t
        )

    # checkpoint

    def state_dict(self) -> dict:
        return {"epsilon": self.epsilon}

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.epsilon = float(state.get("epsilon", self.epsilon))
