"""
Stage 1 預訓練腳本 — 用離散 grid state 訓練 SAC。

目的：
  1. 驗證 SpatialAttentionHead + SAC 能不能學會踩地雷
  2. 預訓練 SpatialAttention + SAC head 權重，供 Stage 2 載入

執行方式：
  python train_stage1.py

監控方式：
  tensorboard --logdir runs/stage1/

輸出：
  models/stage1/stage1_weights.pth    (供 Stage 2 載入)
  models/stage1/training_log.csv      (訓練數據 CSV)
  runs/stage1/                        (TensorBoard logs)
"""

import sys
import time
import datetime
import numpy as np
from collections import deque
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from Minesweeper.MinesweeperLogic import MinesweeperLogic
from RL_Agent import Stage1SACAgent
from training_logger import TeeOutput, CSVLogger

# ---------- 訓練參數 ----------
GRID_ROWS = 10
GRID_COLS = 10
GRID_MINES = 10
MAX_EPISODES = 10000         # 總訓練 episode 數
MAX_STEPS_PER_EPISODE = 200  # 每 episode 最多步數（防止無限迴圈）
LOG_INTERVAL = 50            # 每幾個 episode 印一次 summary
EXPORT_WEIGHTS_INTERVAL = 500  # 每幾個 episode 匯出一次 transfer weights

# ---------- 評估參數 ----------
EVAL_INTERVAL = 100          # 每幾個 episode 跑一次評估
EVAL_EPISODES = 10           # 每次評估跑幾個 episode

# ---------- 路徑 ----------
TENSORBOARD_DIR = Path("./runs/stage1")
CSV_LOG_PATH = Path("./models/stage1/training_log.csv")
TXT_LOG_PATH = Path("./models/stage1/training_output.txt")


def action_to_grid(action, rows, cols):
    """將 ScaledSigmoid action ≈ [-0.05, 1.05] 轉換為 grid 座標。

    Args:
        action: numpy array (2,) ≈ [-0.05, 1.05] (ScaledSigmoid output)
        rows: grid 列數
        cols: grid 行數
    Returns:
        (row, col) 整數座標
    """
    # ScaledSigmoid 輸出已經接近 [0, 1]，直接 clip 後映射
    ax = np.clip(action[0], 0, 1)
    ay = np.clip(action[1], 0, 1)

    col = int(np.clip(ax * cols, 0, cols - 1))
    row = int(np.clip(ay * rows, 0, rows - 1))

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


def run_episode(logic, agent, add_noise=True):
    """跑一個 episode，回傳統計資料。

    Args:
        logic: MinesweeperLogic 實例
        agent: Stage1SACAgent 實例
        add_noise: True=探索模式, False=評估模式（確定性策略）
    Returns:
        dict: episode 統計
    """
    logic.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False
    is_win = False
    invalid_clicks = 0
    valid_clicks = 0
    train_info_list = []

    while not done and episode_steps < MAX_STEPS_PER_EPISODE:
        # 取得 state
        state = logic.get_grid_state_tensor()  # (12, 10, 10)

        # 選擇動作
        action = agent.select_action(state, add_noise=add_noise)  # (2,) ≈ [-0.05, 1.05]

        # 轉換為 grid 座標
        row, col = action_to_grid(action, GRID_ROWS, GRID_COLS)

        # 執行動作
        result = logic.click(row, col)

        # 計算 reward
        reward = compute_reward(result)
        episode_reward += reward

        # 統計有效/無效點擊
        if result.changed:
            valid_clicks += 1
        else:
            invalid_clicks += 1

        # 取得 next state
        done = result.game_over or result.win
        is_win = result.win
        next_state = logic.get_grid_state_tensor()

        # 存 transition + 訓練（只在 training 模式）
        if add_noise:
            agent.store_transition(state, action, next_state, reward, done)
            train_info = agent.train_step()
            if train_info is not None:
                train_info_list.append(train_info)

        episode_steps += 1

    total_clicks = valid_clicks + invalid_clicks
    invalid_rate = invalid_clicks / total_clicks if total_clicks > 0 else 0.0

    # 計算平均 loss（只有 training 模式且 buffer 夠大時有值）
    avg_actor_loss = None
    avg_critic_loss = None
    avg_alpha = None
    if train_info_list:
        avg_actor_loss = np.mean([t['actor_loss'] for t in train_info_list])
        avg_critic_loss = np.mean([t['critic_loss'] for t in train_info_list])
        avg_alpha = np.mean([t['alpha'] for t in train_info_list])

    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'is_win': is_win,
        'invalid_rate': invalid_rate,
        'valid_clicks': valid_clicks,
        'invalid_clicks': invalid_clicks,
        'actor_loss': avg_actor_loss,
        'critic_loss': avg_critic_loss,
        'alpha': avg_alpha,
    }


def run_evaluation(logic, agent):
    """跑 EVAL_EPISODES 個評估 episode（確定性策略，無 noise）。

    Returns:
        dict: 評估統計（平均值）
    """
    eval_rewards = []
    eval_wins = 0
    eval_steps = []
    eval_invalid_rates = []

    for _ in range(EVAL_EPISODES):
        stats = run_episode(logic, agent, add_noise=False)
        eval_rewards.append(stats['reward'])
        eval_steps.append(stats['steps'])
        eval_invalid_rates.append(stats['invalid_rate'])
        if stats['is_win']:
            eval_wins += 1

    return {
        'avg_reward': np.mean(eval_rewards),
        'win_rate': eval_wins / EVAL_EPISODES * 100,
        'avg_steps': np.mean(eval_steps),
        'avg_invalid_rate': np.mean(eval_invalid_rates),
    }


def main():
    # ---------- Tee stdout to txt ----------
    tee = TeeOutput(TXT_LOG_PATH)
    sys.stdout = tee

    print("=" * 60)
    print("  Stage 1 Pre-training: Grid State → SAC")
    print("=" * 60)
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}, Mines: {GRID_MINES}")
    print(f"Max episodes: {MAX_EPISODES}")
    print(f"Eval: every {EVAL_INTERVAL} episodes, {EVAL_EPISODES} eval episodes each")
    print()

    # 建立遊戲邏輯和 agent
    logic = MinesweeperLogic(rows=GRID_ROWS, cols=GRID_COLS, mines_count=GRID_MINES)
    agent = Stage1SACAgent()

    # ---------- TensorBoard ----------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = TENSORBOARD_DIR / timestamp
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")

    # ---------- CSV Logger ----------
    csv_logger = CSVLogger(CSV_LOG_PATH)
    csv_fields = [
        'episode', 'reward', 'steps', 'is_win', 'invalid_rate',
        'valid_clicks', 'invalid_clicks',
        'actor_loss', 'critic_loss', 'alpha',
        'eval_avg_reward', 'eval_win_rate', 'eval_avg_steps', 'eval_avg_invalid_rate',
        'timestamp',
    ]
    csv_logger.open(csv_fields)
    print(f"CSV log: {CSV_LOG_PATH}")
    print()

    # ---------- 統計用 ----------
    recent_rewards = deque(maxlen=LOG_INTERVAL)
    recent_wins = deque(maxlen=LOG_INTERVAL)
    recent_steps = deque(maxlen=LOG_INTERVAL)
    total_wins = 0
    start_time = time.time()

    try:
        for episode in range(1, MAX_EPISODES + 1):
            # ----- 訓練 episode -----
            stats = run_episode(logic, agent, add_noise=True)

            # Episode 結束
            agent.on_episode_end()

            if stats['is_win']:
                total_wins += 1

            recent_rewards.append(stats['reward'])
            recent_wins.append(1 if stats['is_win'] else 0)
            recent_steps.append(stats['steps'])

            # ----- TensorBoard: 每 episode 寫入 -----
            writer.add_scalar('train/episode_reward', stats['reward'], episode)
            writer.add_scalar('train/episode_steps', stats['steps'], episode)
            writer.add_scalar('train/invalid_rate', stats['invalid_rate'], episode)
            if stats['actor_loss'] is not None:
                writer.add_scalar('train/actor_loss', stats['actor_loss'], episode)
                writer.add_scalar('train/critic_loss', stats['critic_loss'], episode)
                writer.add_scalar('train/alpha', stats['alpha'], episode)

            # ----- CSV: 每 episode 基本資料 -----
            csv_row = {
                'episode': episode,
                'reward': f"{stats['reward']:.2f}",
                'steps': stats['steps'],
                'is_win': int(stats['is_win']),
                'invalid_rate': f"{stats['invalid_rate']:.4f}",
                'valid_clicks': stats['valid_clicks'],
                'invalid_clicks': stats['invalid_clicks'],
                'actor_loss': f"{stats['actor_loss']:.6f}" if stats['actor_loss'] is not None else '',
                'critic_loss': f"{stats['critic_loss']:.6f}" if stats['critic_loss'] is not None else '',
                'alpha': f"{stats['alpha']:.6f}" if stats['alpha'] is not None else '',
                'eval_avg_reward': '',
                'eval_win_rate': '',
                'eval_avg_steps': '',
                'eval_avg_invalid_rate': '',
                'timestamp': datetime.datetime.now().isoformat(),
            }

            # ----- 評估模式 -----
            if episode % EVAL_INTERVAL == 0:
                eval_stats = run_evaluation(logic, agent)

                # TensorBoard
                writer.add_scalar('eval/avg_reward', eval_stats['avg_reward'], episode)
                writer.add_scalar('eval/win_rate', eval_stats['win_rate'], episode)
                writer.add_scalar('eval/avg_steps', eval_stats['avg_steps'], episode)
                writer.add_scalar('eval/avg_invalid_rate', eval_stats['avg_invalid_rate'], episode)

                # CSV
                csv_row['eval_avg_reward'] = f"{eval_stats['avg_reward']:.2f}"
                csv_row['eval_win_rate'] = f"{eval_stats['win_rate']:.1f}"
                csv_row['eval_avg_steps'] = f"{eval_stats['avg_steps']:.1f}"
                csv_row['eval_avg_invalid_rate'] = f"{eval_stats['avg_invalid_rate']:.4f}"

                print(f"  [EVAL Ep {episode:>6d}] "
                      f"Avg Reward: {eval_stats['avg_reward']:>7.2f} | "
                      f"Win Rate: {eval_stats['win_rate']:>5.1f}% | "
                      f"Avg Steps: {eval_stats['avg_steps']:>5.1f} | "
                      f"Invalid Rate: {eval_stats['avg_invalid_rate']:.2%}")

            csv_logger.write(csv_row)

            # ----- 定期 console log -----
            if episode % LOG_INTERVAL == 0:
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins) * 100
                avg_steps = np.mean(recent_steps)
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed

                # TensorBoard: 滑動平均
                writer.add_scalar('train/avg_reward_50', avg_reward, episode)
                writer.add_scalar('train/win_rate_50', win_rate, episode)
                writer.add_scalar('train/avg_steps_50', avg_steps, episode)

                print(f"[Ep {episode:>6d}] "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Win Rate: {win_rate:>5.1f}% | "
                      f"Avg Steps: {avg_steps:>5.1f} | "
                      f"Total Wins: {total_wins} | "
                      f"Speed: {eps_per_sec:.1f} ep/s | "
                      f"Alpha: {agent.log_alpha.exp().item():.4f}")

            # ----- 定期匯出 transfer weights -----
            if episode % EXPORT_WEIGHTS_INTERVAL == 0:
                agent.save_stage1_weights()

        # ========== 訓練正常結束 ==========
        print()
        print("=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        elapsed = time.time() - start_time
        print(f"Total episodes: {MAX_EPISODES}")
        print(f"Total wins: {total_wins} ({total_wins/MAX_EPISODES*100:.1f}%)")
        print(f"Total time: {elapsed:.1f}s ({MAX_EPISODES/elapsed:.1f} ep/s)")

        # 最終評估
        print()
        print("--- Final Evaluation (10 episodes, no noise) ---")
        final_eval = run_evaluation(logic, agent)
        print(f"Avg Reward: {final_eval['avg_reward']:.2f}")
        print(f"Win Rate: {final_eval['win_rate']:.1f}%")
        print(f"Avg Steps: {final_eval['avg_steps']:.1f}")
        print(f"Invalid Rate: {final_eval['avg_invalid_rate']:.2%}")

        # TensorBoard: 最終評估
        writer.add_scalar('eval/avg_reward', final_eval['avg_reward'], MAX_EPISODES)
        writer.add_scalar('eval/win_rate', final_eval['win_rate'], MAX_EPISODES)

        # 最終匯出
        agent.save_persistent()
        path = agent.save_stage1_weights()
        print(f"\nTransfer weights saved to: {path}")
        print("Use this file to initialize Stage 2 training.")

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        elapsed = time.time() - start_time
        print(f"Completed {episode - 1} episodes in {elapsed:.1f}s")
        print("Saving checkpoint...")
        agent.save_persistent()
        agent.save_stage1_weights()

    finally:
        # 確保資源正確釋放（即使中途 crash 或 Ctrl+C）
        print(f"\nTensorBoard logs: {tb_dir}")
        print(f"CSV log: {CSV_LOG_PATH}")
        print(f"TXT log: {TXT_LOG_PATH}")
        csv_logger.close()
        writer.close()
        tee.close()


if __name__ == "__main__":
    main()
