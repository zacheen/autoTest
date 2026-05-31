"""Per-episode outcome tracker.

Split out from AdaptiveEpsilonController to keep responsibilities separate:
    - History stores and queries episode results.
    - AdaptiveEpsilonController computes epsilon from a supplied win rate.

Class layout:
    - History          : base class with record / win_rate / state_dict APIs.
    - TrainingHistory  : training-loop subclass; currently namespace only.
    - Eval currently creates ephemeral `History(max_capacity=num_episodes)` and
      uses `window=None` for the whole-session mean (see run_fixed_policy_evaluation).
      Add EvalHistory later if cross-session eval time series are needed.

Key points:
    - max_capacity defaults to 100 and grows automatically when callers request a
      larger window.
    - If samples are insufficient, compute from existing samples, matching old
      controller behavior.
    - state_dict() is separate; checkpoints use an independent .pth file instead
      of mixing with optimizer state.
"""

from __future__ import annotations

from collections import deque


class History:
    """Per-episode outcome tracker base class.

    Usage:
        history = History()                      # max_capacity=100
        history.record(win=True)                 # call once per episode end
        wr = history.win_rate(window=100)        # last 100 episode win rate
    """

    def __init__(self, max_capacity: int = 100, step_reward_capacity: int = 100):
        self._max_capacity = int(max(1, max_capacity))
        self._results: deque[int] = deque(maxlen=self._max_capacity)
        self._rewards: deque[float] = deque(maxlen=self._max_capacity)
        self._steps: deque[int] = deque(maxlen=self._max_capacity)
        self._invalid_rates: deque[float] = deque(maxlen=self._max_capacity)
        self.total_episodes: int = 0
        # Cumulative counter, NOT bounded by deque maxlen. _results only stores
        # the last N episodes, so sum(_results) is rolling, not cumulative.
        self.total_wins: int = 0
        # Per-transition raw reward, appended once per store_transition.
        # Stored separately from episode-level _rewards so step maxlen is
        # decoupled from episode _max_capacity and not overwritten by episode stats.
        self._step_max_capacity = int(max(1, step_reward_capacity))
        self._step_rewards: deque[float] = deque(maxlen=self._step_max_capacity)

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    # update

    def record(
        self,
        win: bool,
        *,
        total_reward: float = 0.0,
        steps: int = 0,
        invalid_rate: float = 0.0,
    ) -> None:
        """Append one episode outcome with reward / steps / invalid_rate.

        total_reward / steps / invalid_rate are keyword-only with defaults, so
        old callers using `record(win=...)` still work; they just contribute 0 to
        those stats.
        """
        win_int = int(bool(win))
        self._results.append(win_int)
        self._rewards.append(float(total_reward))
        self._steps.append(int(steps))
        self._invalid_rates.append(float(invalid_rate))
        self.total_episodes += 1
        self.total_wins += win_int

    # query

    def _rolling_mean(self, source: deque, window: int | None) -> float:
        """Pure rolling mean with no side effects or capacity growth.

        - empty source : return 0.0
        - window=None  : use all existing samples (eval session mean)
        - window <= len: use last window samples
        - window > len : use all existing samples, matching old rolling_win_rate

        Call `_episode_rolling_mean(...)` when episode deques should grow for
        future accumulation. Step-level deques call this pure version directly.
        """
        if not source:
            return 0.0
        if window is None:
            samples = source
        else:
            window = int(max(1, window))
            if window >= len(source):
                samples = source
            else:
                samples = list(source)[-window:]
        return sum(samples) / len(samples)

    def _maybe_grow_episode_capacity(self, window: int | None) -> None:
        """Grow episode capacity when window exceeds _max_capacity.

        Step-level deque (_step_rewards) is outside this scope. Shared by
        episode-level means and reward_per_step.
        """
        if window is not None and int(window) > self._max_capacity:
            self._grow_capacity(int(window))

    def _episode_rolling_mean(self, source: deque, window: int | None) -> float:
        """Episode-level entry point: grow then mean.

        _rolling_mean is also used by step-level _step_rewards, whose maxlen is
        decoupled from episode capacity, so growth is kept in this wrapper.

        New episode-level metrics should use this entry; new step-level metrics
        should call _rolling_mean directly.
        """
        self._maybe_grow_episode_capacity(window)
        return self._rolling_mean(source, window)

    def win_rate(self, window: int | None = 100) -> float:
        """Rolling win rate over the last `window` episodes.

        window=None means use all existing samples, typical for eval sessions.

        If window exceeds max_capacity, grow first. Until enough samples are
        accumulated, compute from existing samples, matching old behavior.

        The value stabilizes as sample count grows. Callers sensitive to low
        sample count can also check `len(history)` or `total_episodes`.
        """
        return self._episode_rolling_mean(self._results, window)

    def avg_reward(self, window: int | None = 100) -> float:
        """Rolling average total episode reward over the window.

        window=None uses all existing samples.

        This stores total episode reward, not per-step mean.
        """
        return self._episode_rolling_mean(self._rewards, window)

    def avg_steps(self, window: int | None = 100) -> float:
        """Rolling average steps per episode. window=None uses all samples."""
        return self._episode_rolling_mean(self._steps, window)

    def avg_invalid_rate(self, window: int | None = 100) -> float:
        """Rolling average invalid-click rate per episode.

        Pass each episode's invalid rate (0.0 to 1.0) via record(). Missing
        values default to 0.0.
        """
        return self._episode_rolling_mean(self._invalid_rates, window)

    # per-step reward
    # Transition-level rolling reward, separate from per-episode stats, for the
    # `train/real_reward_mean` TB scalar in train_step. Agents call
    # record_step_reward(reward) from store_transition; train_step reads
    # avg_step_reward().
    #
    # Future per-step stats should add a _step_xxx deque plus record/query methods
    # and call _rolling_mean directly, not _episode_rolling_mean.

    def record_step_reward(self, reward: float) -> None:
        """Append one per-transition raw reward."""
        self._step_rewards.append(float(reward))

    def avg_step_reward(self, window: int | None = None) -> float:
        """Rolling average per-step raw reward; window=None uses all samples.

        Separate from episode-level avg_reward: this averages the last N
        transitions, while avg_reward averages total reward over episodes.
        Maxlen is controlled by step_reward_capacity and is not tied to episode
        capacity, so use pure _rolling_mean.
        """
        return self._rolling_mean(self._step_rewards, window)

    def reward_per_step(self, window: int | None = 100) -> float:
        """Rolling mean of per-episode per-step reward.

        For each episode, compute total_reward / max(steps, 1), then average
        over episodes. Each episode contributes equal weight.

        We do not use sum(rewards) / sum(steps), which weights longer episodes
        more heavily and reflects data density instead of policy average.

        window=None uses all samples. No data returns 0.0. steps=0 contributes 0.
        """
        if not self._rewards:
            return 0.0
        self._maybe_grow_episode_capacity(window)
        if window is None:
            n = len(self._rewards)
        else:
            n = min(int(max(1, window)), len(self._rewards))
        rewards = list(self._rewards)[-n:]
        steps = list(self._steps)[-n:]
        ratios = [r / s if s > 0 else 0.0 for r, s in zip(rewards, steps)]
        return sum(ratios) / len(ratios)

    def __len__(self) -> int:
        return len(self._results)

    # internal

    def _grow_capacity(self, new_capacity: int) -> None:
        """Grow all episode deque maxlen values while preserving existing data.

        Until enough samples fill the new length, rolling metrics use existing
        samples.
        """
        new_capacity = int(new_capacity)
        if new_capacity <= self._max_capacity:
            return
        self._max_capacity = new_capacity
        self._results = deque(self._results, maxlen=self._max_capacity)
        self._rewards = deque(self._rewards, maxlen=self._max_capacity)
        self._steps = deque(self._steps, maxlen=self._max_capacity)
        self._invalid_rates = deque(self._invalid_rates, maxlen=self._max_capacity)

    # checkpoint

    def state_dict(self) -> dict:
        return {
            "max_capacity": self._max_capacity,
            "results": list(self._results),
            "rewards": list(self._rewards),
            "steps": list(self._steps),
            "invalid_rates": list(self._invalid_rates),
            "total_episodes": self.total_episodes,
            "total_wins": self.total_wins,
            "step_reward_capacity": self._step_max_capacity,
            "step_rewards": list(self._step_rewards),
        }

    def load_state_dict(self, state: dict, *, deque_cls=deque) -> None:
        """Restore from checkpoint dict.

        deque_cls keeps the old AdaptiveEpsilonController extension point so
        agents can inject a deque subclass. Current agents pass collections.deque.

        Compatibility:
            - New key is "results"; old flat checkpoints used "result_window".
            - Missing rewards / steps / invalid_rates use empty deques and
              re-accumulate after resume.
            - Missing total_wins falls back to sum(_results), a closer lower
              bound than 0.
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
        if "total_wins" in state:
            self.total_wins = int(state["total_wins"])
        else:
            # Use sum(_results) as a lower-bound approximation if key is missing.
            self.total_wins = int(sum(self._results))
        # step_rewards was added after recent_real_rewards moved here from v3 /
        # stage1. Old checkpoints use an empty deque and re-accumulate after resume.
        step_cap = int(state.get("step_reward_capacity", self._step_max_capacity))
        self._step_max_capacity = max(1, step_cap)
        step_rewards = state.get("step_rewards", [])
        self._step_rewards = deque_cls(
            step_rewards, maxlen=self._step_max_capacity
        )


class TrainingHistory(History):
    """History subclass for training loops.

    The base class already tracks rolling win / total_reward / steps, so training
    scripts should query this instead of maintaining recent_* deques.
    """
    pass
