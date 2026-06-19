from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    valid_click: float
    invalid_click: float
    lose: float
    win: float
    gamma: float

    @property
    def replay_win_threshold(self) -> float:
        return self.win

    @property
    def replay_lose_threshold(self) -> float:
        return self.lose

    @property
    def replay_invalid_threshold(self) -> float:
        return 0.0


MINESWEEPER_REWARD_CONFIG = RewardConfig(
    valid_click = 0.5,
    invalid_click = -0.75, # discrete invalid_click can be bigger than lose
    lose = -0.7,
    win = 1,
    gamma = 0.97,
)
