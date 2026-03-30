"""
Stage 1 Simple MLP 實驗 — 驗證 RL pipeline 是否正常。

極簡架構：grid state (12,10,10) → flatten → MLP → 100 actions
沒有 GridEncoder，沒有 80×80，沒有 Attention。

如果這個學得會 → 問題在 GridEncoder + Attention 架構
如果這個也學不會 → 問題在 RL 設定（reward、hyperparameters、buffer）

執行方式：
  python train_stage1_simple.py

監控方式：
  tensorboard --logdir runs/stage1_simple/
"""

import os
import csv
import time
import datetime
import numpy as np
from collections import deque
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from Minesweeper.MinesweeperLogic import MinesweeperLogic
from RL_Agent import SimpleDiscreteAgent

# ---------- 訓練參數 ----------
GRID_ROWS = 10
GRID_COLS = 10
GRID_MINES = 10
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 200
LOG_INTERVAL = 50
SAVE_INTERVAL = 500

# ---------- 評估參數 ----------
EVAL_INTERVAL = 100
EVAL_EPISODES = 10

# ---------- 路徑 ----------
TENSORBOARD_DIR = Path("./runs/stage1_simple")
CSV_LOG_PATH = Path("./models/stage1_simple/training_log.csv")


def action_to_grid(action, rows, cols):
    """Discrete action [0, 99] → (row, col)."""
    return action // cols, action % cols


def compute_reward(result):
    """根據 ClickResult 計算 reward。"""
    if result.win:
        return 20.0
    if result.game_over:
        return -3.0
    if result.changed:
        return 3.0
    return -2.0


def run_episode(logic, agent, add_noise=True):
    logic.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False
    is_win = False
    invalid_clicks = 0
    valid_clicks = 0
    train_info_list = []

    while not done and episode_steps < MAX_STEPS_PER_EPISODE:
        state = logic.get_grid_state_tensor()
        action = agent.select_action(state, add_noise=add_noise)
        row, col = action_to_grid(action, GRID_ROWS, GRID_COLS)
        result = logic.click(row, col)
        reward = compute_reward(result)
        episode_reward += reward

        if result.changed:
            valid_clicks += 1
        else:
            invalid_clicks += 1

        done = result.game_over or result.win
        is_win = result.win
        next_state = logic.get_grid_state_tensor()

        if add_noise:
            agent.store_transition(state, action, next_state, reward, done)
            train_info = agent.train_step()
            if train_info is not None:
                train_info_list.append(train_info)

        episode_steps += 1

    total_clicks = valid_clicks + invalid_clicks
    invalid_rate = invalid_clicks / total_clicks if total_clicks > 0 else 0.0

    avg_actor_loss = None
    avg_critic_loss = None
    avg_alpha = None
    avg_entropy = None
    if train_info_list:
        avg_actor_loss = np.mean([t['actor_loss'] for t in train_info_list])
        avg_critic_loss = np.mean([t['critic_loss'] for t in train_info_list])
        avg_alpha = np.mean([t['alpha'] for t in train_info_list])
        avg_entropy = np.mean([t['entropy'] for t in train_info_list])

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
        'entropy': avg_entropy,
    }


def run_evaluation(logic, agent):
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


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.writer = None

    def open(self, fieldnames):
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self.file = open(self.path, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def main():
    print("=" * 60)
    print("  Stage 1 SIMPLE MLP: Grid State → MLP → 100 actions")
    print("=" * 60)
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}, Mines: {GRID_MINES}")
    print(f"Architecture: Flatten(1200) → FC(256) → FC(256) → FC(100)")
    print(f"Max episodes: {MAX_EPISODES}")
    print()

    logic = MinesweeperLogic(rows=GRID_ROWS, cols=GRID_COLS, mines_count=GRID_MINES)
    agent = SimpleDiscreteAgent()

    # TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = TENSORBOARD_DIR / timestamp
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")

    # CSV
    csv_logger = CSVLogger(CSV_LOG_PATH)
    csv_fields = [
        'episode', 'reward', 'steps', 'is_win', 'invalid_rate',
        'valid_clicks', 'invalid_clicks',
        'actor_loss', 'critic_loss', 'alpha', 'entropy',
        'eval_avg_reward', 'eval_win_rate', 'eval_avg_steps', 'eval_avg_invalid_rate',
        'timestamp',
    ]
    csv_logger.open(csv_fields)
    print(f"CSV log: {CSV_LOG_PATH}")
    print()

    recent_rewards = deque(maxlen=LOG_INTERVAL)
    recent_wins = deque(maxlen=LOG_INTERVAL)
    recent_steps = deque(maxlen=LOG_INTERVAL)
    total_wins = 0
    start_time = time.time()

    try:
        for episode in range(1, MAX_EPISODES + 1):
            stats = run_episode(logic, agent, add_noise=True)

            valid_rate = stats['valid_clicks'] / (stats['valid_clicks'] + stats['invalid_clicks']) \
                if (stats['valid_clicks'] + stats['invalid_clicks']) > 0 else 0.0
            agent.update_alpha(valid_rate)
            agent.on_episode_end()

            if stats['is_win']:
                total_wins += 1

            recent_rewards.append(stats['reward'])
            recent_wins.append(1 if stats['is_win'] else 0)
            recent_steps.append(stats['steps'])

            # TensorBoard
            writer.add_scalar('train/episode_reward', stats['reward'], episode)
            writer.add_scalar('train/episode_steps', stats['steps'], episode)
            writer.add_scalar('train/invalid_rate', stats['invalid_rate'], episode)
            if stats['actor_loss'] is not None:
                writer.add_scalar('train/actor_loss', stats['actor_loss'], episode)
                writer.add_scalar('train/critic_loss', stats['critic_loss'], episode)
                writer.add_scalar('train/alpha', stats['alpha'], episode)
                writer.add_scalar('train/entropy', stats['entropy'], episode)

            # CSV
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
                'entropy': f"{stats['entropy']:.4f}" if stats['entropy'] is not None else '',
                'eval_avg_reward': '',
                'eval_win_rate': '',
                'eval_avg_steps': '',
                'eval_avg_invalid_rate': '',
                'timestamp': datetime.datetime.now().isoformat(),
            }

            # 評估
            if episode % EVAL_INTERVAL == 0:
                eval_stats = run_evaluation(logic, agent)
                writer.add_scalar('eval/avg_reward', eval_stats['avg_reward'], episode)
                writer.add_scalar('eval/win_rate', eval_stats['win_rate'], episode)
                writer.add_scalar('eval/avg_steps', eval_stats['avg_steps'], episode)
                writer.add_scalar('eval/avg_invalid_rate', eval_stats['avg_invalid_rate'], episode)

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

            # Console log
            if episode % LOG_INTERVAL == 0:
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins) * 100
                avg_steps = np.mean(recent_steps)
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed

                writer.add_scalar('train/avg_reward_50', avg_reward, episode)
                writer.add_scalar('train/win_rate_50', win_rate, episode)

                print(f"[Ep {episode:>6d}] "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Win Rate: {win_rate:>5.1f}% | "
                      f"Avg Steps: {avg_steps:>5.1f} | "
                      f"Total Wins: {total_wins} | "
                      f"Speed: {eps_per_sec:.1f} ep/s | "
                      f"Alpha: {agent.alpha:.4f}")

            if episode % SAVE_INTERVAL == 0:
                agent._save_model()

        # 訓練正常結束
        print()
        print("=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        elapsed = time.time() - start_time
        print(f"Total episodes: {MAX_EPISODES}")
        print(f"Total wins: {total_wins} ({total_wins/MAX_EPISODES*100:.1f}%)")
        print(f"Total time: {elapsed:.1f}s ({MAX_EPISODES/elapsed:.1f} ep/s)")

        print()
        print("--- Final Evaluation ---")
        final_eval = run_evaluation(logic, agent)
        print(f"Avg Reward: {final_eval['avg_reward']:.2f}")
        print(f"Win Rate: {final_eval['win_rate']:.1f}%")
        print(f"Avg Steps: {final_eval['avg_steps']:.1f}")
        print(f"Invalid Rate: {final_eval['avg_invalid_rate']:.2%}")

        agent.save_persistent()

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted")
        agent.save_persistent()

    finally:
        print(f"\nTensorBoard logs: {tb_dir}")
        print(f"CSV log: {CSV_LOG_PATH}")
        csv_logger.close()
        writer.close()


if __name__ == "__main__":
    main()
