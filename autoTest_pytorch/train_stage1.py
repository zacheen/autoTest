"""
Stage 1 預訓練腳本 — 用離散 grid state 訓練 SAC。

目的：
  1. 驗證 SpatialAttentionHead + SAC 能不能學會踩地雷
  2. 預訓練 SpatialAttention + SAC head 權重，供 Stage 2 載入

執行方式：
  python train_stage1.py

輸出：
  models/stage1/stage1_weights.pth  (供 Stage 2 載入)
  models/stage1/stage1_actor.pth    (Stage 1 完整 checkpoint)
  models/stage1/stage1_critic.pth
  ...
"""

import time
import numpy as np
from collections import deque

from Minesweeper.MinesweeperLogic import MinesweeperLogic
from RL_Agent import Stage1SACAgent

# ---------- 訓練參數 ----------
GRID_ROWS = 10
GRID_COLS = 10
GRID_MINES = 10
MAX_EPISODES = 10000       # 總訓練 episode 數
MAX_STEPS_PER_EPISODE = 200  # 每 episode 最多步數（防止無限迴圈）
LOG_INTERVAL = 50           # 每幾個 episode 印一次 summary
EXPORT_WEIGHTS_INTERVAL = 500  # 每幾個 episode 匯出一次 transfer weights


def action_to_grid(action, rows, cols):
    """將 action [-1, 1] 轉換為 grid 座標。

    Args:
        action: numpy array (2,) in [-1, 1]
        rows: grid 列數
        cols: grid 行數
    Returns:
        (row, col) 整數座標
    """
    # [-1, 1] → [0, 1]
    norm_x = (action[0] + 1) / 2.0
    norm_y = (action[1] + 1) / 2.0

    # [0, 1] → grid 座標，clamp 防止越界
    col = int(np.clip(norm_x * cols, 0, cols - 1))
    row = int(np.clip(norm_y * rows, 0, rows - 1))

    return row, col


def compute_reward(result):
    """根據 ClickResult 計算 reward。

    Args:
        result: MinesweeperLogic.ClickResult
    Returns:
        float: reward 值
    """
    if result.win:
        return 20.0
    if result.game_over:
        return -10.0
    if result.changed:
        return 1.0   # 有效點擊
    return -1.0       # 無效點擊（已翻開、已標旗、超出範圍）


def main():
    print("=" * 60)
    print("  Stage 1 Pre-training: Grid State → SAC")
    print("=" * 60)
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}, Mines: {GRID_MINES}")
    print(f"Max episodes: {MAX_EPISODES}")
    print()

    # 建立遊戲邏輯和 agent
    logic = MinesweeperLogic(rows=GRID_ROWS, cols=GRID_COLS, mines_count=GRID_MINES)
    agent = Stage1SACAgent()

    # 統計用
    recent_rewards = deque(maxlen=LOG_INTERVAL)
    recent_wins = deque(maxlen=LOG_INTERVAL)
    recent_steps = deque(maxlen=LOG_INTERVAL)
    total_wins = 0
    start_time = time.time()

    for episode in range(1, MAX_EPISODES + 1):
        logic.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False
        is_win = False

        while not done and episode_steps < MAX_STEPS_PER_EPISODE:
            # 取得 state
            state = logic.get_grid_state_tensor()  # (12, 10, 10)

            # 選擇動作
            action = agent.select_action(state, add_noise=True)  # (2,) in [-1, 1]

            # 轉換為 grid 座標
            row, col = action_to_grid(action, GRID_ROWS, GRID_COLS)

            # 執行動作
            result = logic.click(row, col)

            # 計算 reward
            reward = compute_reward(result)
            episode_reward += reward

            # 取得 next state
            done = result.game_over or result.win
            is_win = result.win
            next_state = logic.get_grid_state_tensor()

            # 存 transition
            agent.store_transition(state, action, next_state, reward, done)

            # 訓練一步
            agent.train_step()

            episode_steps += 1

        # Episode 結束
        agent.on_episode_end()
        if is_win:
            total_wins += 1

        recent_rewards.append(episode_reward)
        recent_wins.append(1 if is_win else 0)
        recent_steps.append(episode_steps)

        # 定期 log
        if episode % LOG_INTERVAL == 0:
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins) * 100
            avg_steps = np.mean(recent_steps)
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed

            print(f"[Ep {episode:>6d}] "
                  f"Avg Reward: {avg_reward:>7.2f} | "
                  f"Win Rate: {win_rate:>5.1f}% | "
                  f"Avg Steps: {avg_steps:>5.1f} | "
                  f"Total Wins: {total_wins} | "
                  f"Speed: {eps_per_sec:.1f} ep/s | "
                  f"Alpha: {agent.log_alpha.exp().item():.4f}")

        # 定期匯出 transfer weights
        if episode % EXPORT_WEIGHTS_INTERVAL == 0:
            agent.save_stage1_weights()

    # 訓練結束
    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    elapsed = time.time() - start_time
    print(f"Total episodes: {MAX_EPISODES}")
    print(f"Total wins: {total_wins} ({total_wins/MAX_EPISODES*100:.1f}%)")
    print(f"Total time: {elapsed:.1f}s ({MAX_EPISODES/elapsed:.1f} ep/s)")

    # 最終匯出
    agent.save_persistent()
    path = agent.save_stage1_weights()
    print(f"\nTransfer weights saved to: {path}")
    print("Use this file to initialize Stage 2 training.")


if __name__ == "__main__":
    main()
