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

    def __init__(self, max_capacity: int = 100, step_reward_capacity: int = 100):
        self._max_capacity = int(max(1, max_capacity))
        self._results: deque[int] = deque(maxlen=self._max_capacity)
        self._rewards: deque[float] = deque(maxlen=self._max_capacity)
        self._steps: deque[int] = deque(maxlen=self._max_capacity)
        self._invalid_rates: deque[float] = deque(maxlen=self._max_capacity)
        self.total_episodes: int = 0
        # Cumulative counter, NOT bounded by deque maxlen — 用來看「整段訓練
        # 累積贏了幾場」。deque 內的 _results 只保留最後 N 場,sum(_results)
        # 是 rolling 不是 cumulative,所以另存一個 int 才能正確反映歷史總數。
        self.total_wins: int = 0
        # Per-transition raw reward(維度:每一筆 store_transition 都 append 一次)。
        # 跟 episode-level _rewards 分開存:per-step 的 maxlen 跟 episode-level
        # _max_capacity 解耦,避免 caller 要求大 window 時連帶把 step deque 撐大,
        # 也避免 step rewards 被 episode 統計覆蓋。
        self._step_max_capacity = int(max(1, step_reward_capacity))
        self._step_rewards: deque[float] = deque(maxlen=self._step_max_capacity)

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
        win_int = int(bool(win))
        self._results.append(win_int)
        self._rewards.append(float(total_reward))
        self._steps.append(int(steps))
        self._invalid_rates.append(float(invalid_rate))
        self.total_episodes += 1
        self.total_wins += win_int

    # ──────────────────────────── query ─────────────────────────────

    def _rolling_mean(self, source: deque, window: int | None) -> float:
        """純 rolling mean —— 無 side effect,不擴 source、不動其他 deque。

        - source 空    : 回 0.0
        - window=None  : 用全部現有資料(eval session 慣用法,取整個 session mean)
        - window <= len: 用最後 window 筆
        - window > len : 用全部現有資料(sample 不足時直接拿手上有的,跟舊
                         AdaptiveEpsilonController 的 rolling_win_rate 行為一致)

        想要「擴張 deque 等未來累積到 window 長度」的 caller 走
        `_episode_rolling_mean(...)`(grow + mean 一體);step-level deque
        (_step_rewards)直接 call 這個純版本,不會誤觸 episode-level capacity。
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
        """Episode-level helper:若 window > _max_capacity 就 grow,讓未來累積到
        window 長度。Step-level deque(_step_rewards)不在此影響範圍內。

        抽出來給 episode-level mean 方法 + reward_per_step 重用,避免每個 caller
        各貼一份 `if window > self._max_capacity: self._grow_capacity(window)`。
        """
        if window is not None and int(window) > self._max_capacity:
            self._grow_capacity(int(window))

    def _episode_rolling_mean(self, source: deque, window: int | None) -> float:
        """Episode-level 專用入口:grow + mean。

        為什麼不直接把 grow 塞進 _rolling_mean:_rolling_mean 也給 step-level
        (_step_rewards)用,而 step-level 的 maxlen 跟 episode-level 解耦,
        不能誤觸 _grow_capacity。所以分成「純 _rolling_mean」與「會 grow 的
        _episode_rolling_mean」兩個入口,各自有對應的 caller。

        新增 episode-level metric 走這個入口;新增 step-level metric 直接
        走純 _rolling_mean。
        """
        self._maybe_grow_episode_capacity(window)
        return self._rolling_mean(source, window)

    def win_rate(self, window: int | None = 100) -> float:
        """Rolling win rate over the last `window` episodes。

        window=None 表示「用全部現有資料」(eval session 慣用法)。

        若 window > 目前 max_capacity,先擴張 max_capacity (rebuild deque +
        copy 現有資料);擴張後尚未累積到 window 場時,用「現有資料」直接算 —
        跟舊 AdaptiveEpsilonController 的 rolling_win_rate() 行為一致。

        呼叫端拿到的數字會隨 sample 數逐步穩定;若對 sample 不足敏感,
        可同時 query `len(history)` 或 `total_episodes` 自行判斷。
        """
        return self._episode_rolling_mean(self._results, window)

    def avg_reward(self, window: int | None = 100) -> float:
        """Rolling 平均整場 reward (跨 window 場,每場一個值)。

        window=None 表示「用全部現有資料」(eval session 慣用法)。

        注意:這裡記的是「整場 total reward」,不是 per-step mean。
        per-step mean 由 caller 自行從 (total_reward, steps) 算出。
        """
        return self._episode_rolling_mean(self._rewards, window)

    def avg_steps(self, window: int | None = 100) -> float:
        """Rolling 平均每場 step 數。window=None 用全部現有資料。"""
        return self._episode_rolling_mean(self._steps, window)

    def avg_invalid_rate(self, window: int | None = 100) -> float:
        """Rolling 平均每場 invalid click rate。window=None 用全部現有資料。

        Caller 在 `record(invalid_rate=...)` 時要傳這場的 invalid rate
        (0.0~1.0)。沒傳就視為 0.0,於是這場對 mean 的貢獻是 0。
        """
        return self._episode_rolling_mean(self._invalid_rates, window)

    # ──────────────────────────── per-step reward ──────────────────
    # 跟 per-episode 統計分開的 transition-level rolling reward,給 train_step
    # 內的 `train/real_reward_mean` TB scalar 用。維護點:agent 的 store_transition
    # 內每筆都 record_step_reward(reward)。讀取點:train_step 結尾的
    # avg_step_reward()(window=None 用全部現有資料,等同 mean of deque)。
    #
    # 未來若有其他 per-step 統計(例如 avg_step_loss),加一條 _step_xxx deque
    # + record/query 兩個 method,query 內直接 call self._rolling_mean(self._step_xxx,
    # window) 即可 —— 不要走 _episode_rolling_mean,step-level capacity 跟
    # episode-level 解耦,不該誤觸 episode 端的 grow。

    def record_step_reward(self, reward: float) -> None:
        """Append 一筆 per-transition raw reward。"""
        self._step_rewards.append(float(reward))

    def avg_step_reward(self, window: int | None = None) -> float:
        """Rolling 平均 per-step raw reward(window=None 用全部現有資料)。

        跟 episode-level avg_reward 分開:這條是「過去 N 筆 transition 的平均
        reward」,episode-level 那條是「過去 N 場 episode 的 total reward 平均」。
        deque maxlen 由 ctor 的 step_reward_capacity 控制(預設 100),不會被
        episode-level _max_capacity 擴張連動 —— 所以走純 _rolling_mean,不走
        _episode_rolling_mean(後者會 grow episode-level deque)。
        """
        return self._rolling_mean(self._step_rewards, window)

    def reward_per_step(self, window: int | None = 100) -> float:
        """Rolling 平均「每場 per-step reward」(即 mean of (total_reward / steps))。

        語意:每場先算 reward_per_step = total_reward / max(steps, 1),再對
        window 場取 mean — 每場「等權重」貢獻,跟 caller 原本傳的
        episode-level `reward_mean` (TB scalar) 的滾動平均一致。

        替代語意 sum(rewards) / sum(steps) (steps 多的場貢獻較大) 不採用 —
        前者對「這個 policy 平均每步賺多少」比較有直覺,後者偏向「資料密度」。

        window=None 用全部現有資料;沒有資料 → 0.0;steps=0 的單場視為 0。
        Caller 沒傳 total_reward / steps 給 record(),這場貢獻 0/1=0。
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
            "total_wins": self.total_wins,
            "step_reward_capacity": self._step_max_capacity,
            "step_rewards": list(self._step_rewards),
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
            - "total_wins" 在更舊扁平 checkpoint 裡有 (AdaptiveEpsilonController
              曾經存過);中間版本拿掉了所以可能缺;這裡兩種情況都吃,缺
              的時候用 sum(_results) 當下限近似 (至少反映 deque 內的 wins,
              比 0 更接近真相)。
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
            # 缺 key 時用 sum(_results) 作下限近似 — 比預設 0 更接近真實值。
            self.total_wins = int(sum(self._results))
        # step_rewards 在 v3 / stage1 把 recent_real_rewards 搬進來之後才加進
        # state_dict;舊 checkpoint 缺 key 時用空 deque,real_reward_mean 在
        # resume 後重新累積到 maxlen 之前會偏向初期樣本(可接受)。
        step_cap = int(state.get("step_reward_capacity", self._step_max_capacity))
        self._step_max_capacity = max(1, step_cap)
        step_rewards = state.get("step_rewards", [])
        self._step_rewards = deque_cls(
            step_rewards, maxlen=self._step_max_capacity
        )


class TrainingHistory(History):
    """Training loop 用的 History subclass。

    base class 已包含 win / total_reward / steps 三個 rolling 統計,
    training script 不再需要自己維護 recent_* deque,直接 query 這裡即可。
    """
    pass
