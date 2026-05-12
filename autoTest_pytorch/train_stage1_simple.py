"""
Stage 1 Transformer 實驗 — 用 self-attention 學習 Minesweeper 空間推理。

架構：grid state (12,10,10) → 100 tokens × 12-d + 2D pos encoding
     → 4-layer Transformer (d=64, h=4) → per-token logit → 100 actions

執行方式：
  python train_stage1_simple.py

監控方式：
  tensorboard --logdir ./models/stage1_transformer/tensorboard
"""

import os
import csv
import time
import datetime
import numpy as np
import torch
from collections import deque
from pathlib import Path

from Minesweeper.MinesweeperLogic import MinesweeperLogic
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from transformer_discrete_agent import TransformerDiscreteAgent, log_unhandled_exception

# ---------- 訓練參數 ----------
GRID_ROWS = 6
GRID_COLS = 6
GRID_MINES = 4
MAX_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 200
LOG_INTERVAL = 50
SAVE_DEMO_INTERVAL = 300

# ---------- 評估參數 ----------
EVAL_INTERVAL = 200
EVAL_EPISODES = 30
EVAL_OFFSET = 50  # 第一次 eval 在 ep 50,之後每 EVAL_INTERVAL 一次:50, 250, 450...

# ---------- 路徑 ----------
TENSORBOARD_DIR = Path("./models/stage1_transformer/tensorboard")
CSV_LOG_PATH = Path("./models/stage1_transformer/training_log.csv")


def action_to_grid(action, rows, cols):
    """Discrete action [0, 99] → (row, col)."""
    return action // cols, action % cols


def format_grid(logic, click_row=None, click_col=None, result=None):
    """用文字畫出遊戲 grid。

    符號說明：
        .  = 未翻開
        F  = 已標旗
        0-8 = 已翻開的數字
        *  = 地雷 (game over 後)
        括號 [X] = 本次點擊位置

    Returns:
        str: 格式化的 grid 文字
    """
    grid = logic.get_grid_state()
    lines = []

    # Header: column numbers
    lines.append("    " + "  ".join(f"{c}" for c in range(logic.cols)))
    lines.append("   " + "---" * logic.cols)

    for r in range(logic.rows):
        row_str = f"{r} |"
        for c in range(logic.cols):
            val = grid[r][c]
            if val == -1:
                ch = "."
            elif val == -2:
                ch = "F"
            else:
                ch = str(val)

            # 踩雷: 顯示地雷
            if result and result.game_over and result.hit_mine == (r, c):
                ch = "*"

            # 標記點擊位置
            if r == click_row and c == click_col:
                cell = f"[{ch}]"
            else:
                cell = f" {ch} "

            row_str += cell
        lines.append(row_str)

    return "\n".join(lines)


DEMO_TRAINING_EPISODES = 3  # save 時印幾場 training demo
DEMO_LOG_PATH = Path("./models/stage1_transformer/demo_log.txt")


def run_demo_episode(f, logic, agent, mode="validation"):
    """跑一場 demo episode，寫入 file。

    Args:
        f: 已開啟的 file object
        logic: MinesweeperLogic
        agent: agent instance
        mode: "training" (sample) 或 "validation" (argmax)
    """
    add_noise = (mode == "training")
    logic.reset()
    agent.reset_episode()
    done = False
    step = 0
    total_reward = 0.0

    label = "TRAINING" if mode == "training" else "VALIDATION"
    f.write(f"\n  --- {label} demo ---\n")

    while not done and step < MAX_STEPS_PER_EPISODE:
        state = torch.from_numpy(logic.get_grid_state_array())
        row, col = agent.select_action(state, add_noise=add_noise)

        result = logic.click(row, col)
        reward = compute_reward(result)
        total_reward += reward

        if not result.changed:
            agent.block_action_for_state(state, row * GRID_COLS + col)

        done = result.game_over or result.win
        step += 1

        status = "WIN!" if result.win else "BOOM!" if result.game_over else \
                 "valid" if result.changed else "invalid"
        f.write(f"\n  Step {step}: click ({row},{col}) -> {status} | reward={reward:+.0f}\n")
        f.write(format_grid(logic, click_row=row, click_col=col, result=result) + "\n")

    outcome = "WIN" if logic.is_win else "LOSE (mine)" if logic.game_over else "TIMEOUT"
    f.write(f"\n  Result: {outcome} | steps={step} | total_reward={total_reward:.1f}\n")


def run_demo_at_save(logic, agent):
    """Save model 時寫 demo episodes 到檔案: 幾場 training + 1 場 validation。"""
    DEMO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"  DEMO @ Episode {agent.episode_count}\n")
        f.write(f"{'='*50}\n")

        for i in range(DEMO_TRAINING_EPISODES):
            run_demo_episode(f, logic, agent, mode="training")

        run_demo_episode(f, logic, agent, mode="validation")

        f.write(f"{'='*50}\n")


def compute_reward(result):
    """Reward values after applying the old squash transform.

    4 cases:
        +win reward
        +valid click reward
        +lose reward
        +invalid click reward
    """
    if not result.changed:
        return MINESWEEPER_REWARD_CONFIG.invalid_click
    if result.win:
        return MINESWEEPER_REWARD_CONFIG.win
    if result.game_over:
        return MINESWEEPER_REWARD_CONFIG.lose
    return MINESWEEPER_REWARD_CONFIG.valid_click


def run_episode(logic, agent, add_noise=True):
    logic.reset()
    agent.reset_episode()
    episode_reward = 0.0
    episode_steps = 0
    done = False
    is_win = False
    invalid_clicks = 0
    valid_clicks = 0
    train_info_list = []

    while not done and episode_steps < MAX_STEPS_PER_EPISODE:
        state = torch.from_numpy(logic.get_grid_state_array())
        row, col = agent.select_action(state, add_noise=add_noise)

        result = logic.click(row, col)
        reward = compute_reward(result)
        episode_reward += reward

        if result.changed:
            valid_clicks += 1
        else:
            invalid_clicks += 1
            agent.block_action_for_state(state, row * GRID_COLS + col)

        done = result.game_over or result.win
        is_win = result.win
        next_state = torch.from_numpy(logic.get_grid_state_array())

        if add_noise:
            # 跳過第一步：第一次點擊一定有效，沒有學習價值，會稀釋 valid group
            if episode_steps > 0:
                agent.store_transition(state, (row, col), next_state, reward, done)
                train_info = agent.train_step()
                if train_info is not None:
                    train_info_list.append(train_info)

        episode_steps += 1

    total_clicks = valid_clicks + invalid_clicks
    invalid_rate = invalid_clicks / total_clicks if total_clicks > 0 else 0.0

    avg_q_loss = None
    avg_q_mean = None
    if train_info_list:
        avg_q_loss = np.mean([t['Q_loss'] for t in train_info_list])
        avg_q_mean = np.mean([t['q_mean'] for t in train_info_list])

    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'is_win': is_win,
        'invalid_rate': invalid_rate,
        'valid_clicks': valid_clicks,
        'invalid_clicks': invalid_clicks,
        'Q_loss': avg_q_loss,
        'q_mean': avg_q_mean,
    }


def run_evaluation(logic, agent):
    return run_fixed_policy_evaluation(logic, agent, num_episodes=EVAL_EPISODES)


def run_fixed_policy_evaluation(logic, agent, num_episodes):
    eval_rewards = []
    eval_wins = 0
    eval_steps = []
    eval_invalid_rates = []

    for _ in range(num_episodes):
        stats = run_episode(logic, agent, add_noise=False)
        eval_rewards.append(stats['reward'])
        eval_steps.append(stats['steps'])
        eval_invalid_rates.append(stats['invalid_rate'])
        if stats['is_win']:
            eval_wins += 1

    return {
        'avg_reward': np.mean(eval_rewards),
        'win_rate': eval_wins / num_episodes * 100,
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
    num_actions = GRID_ROWS * GRID_COLS
    print(f"  Stage 1 DDQN: Grid State → Transformer → Dueling Q → {num_actions} actions")
    print("=" * 60)
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}, Mines: {GRID_MINES}")
    print(f"Architecture: 100 tokens × 12-d → Transformer(d=32, h=4, L=4) → per-token logit")
    print(f"Max episodes: {MAX_EPISODES}")
    print()

    logic = MinesweeperLogic(rows=GRID_ROWS, cols=GRID_COLS, mines_count=GRID_MINES)
    agent = TransformerDiscreteAgent(grid_h=GRID_ROWS, grid_w=GRID_COLS)

    # TensorBoard — reuse the writer the agent created in __init__ so that
    # train-step diagnostics (td_error / grad / weights drift) land in the
    # same log_dir as the eval/episode scalars logged here.
    writer = agent.tb_writer
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")
    print(f"  Active run: {agent.tensorboard_log_dir}")

    # CSV
    csv_logger = CSVLogger(CSV_LOG_PATH)
    csv_fields = [
        'episode', 'reward', 'steps', 'is_win', 'invalid_rate',
        'valid_clicks', 'invalid_clicks',
        'Q_loss', 'q_mean', 'epsilon',
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

            agent.on_episode_end()

            if stats['is_win']:
                total_wins += 1

            recent_rewards.append(stats['reward'])
            recent_wins.append(1 if stats['is_win'] else 0)
            recent_steps.append(stats['steps'])

            # TensorBoard — episode-level scalars. NOTE: do NOT reuse the
            # `train/Q_loss` / `train/q_mean` tag names; the agent already
            # writes those per gradient step (different step axis) — sharing
            # the tag corrupts the curves with two interleaved step counters.
            writer.add_scalar('episode/reward', stats['reward'], agent.episode_count)
            writer.add_scalar('episode/steps', stats['steps'], agent.episode_count)
            writer.add_scalar('episode/invalid_rate', stats['invalid_rate'], agent.episode_count)
            if stats['Q_loss'] is not None:
                writer.add_scalar('episode/Q_loss_avg', stats['Q_loss'], agent.episode_count)
                writer.add_scalar('episode/q_mean_avg', stats['q_mean'], agent.episode_count)

            # CSV
            csv_row = {
                'episode': episode,
                'reward': f"{stats['reward']:.2f}",
                'steps': stats['steps'],
                'is_win': int(stats['is_win']),
                'invalid_rate': f"{stats['invalid_rate']:.4f}",
                'valid_clicks': stats['valid_clicks'],
                'invalid_clicks': stats['invalid_clicks'],
                'Q_loss': f"{stats['Q_loss']:.6f}" if stats['Q_loss'] is not None else '',
                'q_mean': f"{stats['q_mean']:.4f}" if stats['q_mean'] is not None else '',
                'epsilon': f"{agent.epsilon:.4f}",
                'eval_avg_reward': '',
                'eval_win_rate': '',
                'eval_avg_steps': '',
                'eval_avg_invalid_rate': '',
                'timestamp': datetime.datetime.now().isoformat(),
            }

            # 評估
            if episode >= EVAL_OFFSET and (episode - EVAL_OFFSET) % EVAL_INTERVAL == 0:
                eval_stats = run_evaluation(logic, agent)
                writer.add_scalar('eval/avg_reward', eval_stats['avg_reward'], agent.episode_count)
                writer.add_scalar('eval/win_rate', eval_stats['win_rate'], agent.episode_count)
                writer.add_scalar('eval/avg_steps', eval_stats['avg_steps'], agent.episode_count)
                writer.add_scalar('eval/avg_invalid_rate', eval_stats['avg_invalid_rate'], agent.episode_count)

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

                writer.add_scalar('train/avg_reward_50', avg_reward, agent.episode_count)
                writer.add_scalar('train/win_rate_50', win_rate, agent.episode_count)

                overall_wr = total_wins / episode * 100
                print(f"[Ep {episode:>6d}] "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Win Rate(50): {win_rate:>5.1f}% | "
                      f"Overall WR: {overall_wr:>5.1f}% | "
                      f"Total Wins: {total_wins} | "
                      f"Speed: {eps_per_sec:.1f} ep/s | "
                      f"Epsilon: {agent.epsilon:.4f}")

            if episode % SAVE_DEMO_INTERVAL == 0:
                run_demo_at_save(logic, agent)

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

        agent._save_model()
        agent.save_persistent()

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted")
        agent._save_model()
        agent.save_persistent()

    finally:
        print(f"\nTensorBoard logs: {agent.tensorboard_log_dir}")
        print(f"CSV log: {CSV_LOG_PATH}")
        csv_logger.close()
        # Don't close writer here — it's owned by the agent and will be
        # closed via the agent's atexit hook. Calling close() twice on the
        # same SummaryWriter is a no-op in practice, but explicit is better.


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 使用者主動中斷，不算錯誤，也不寫進 cuda_debug.log
        raise
    except Exception:
        # 任何未處理的例外（含 CUDA error）都寫進 cuda_debug.log，避免 CMD 被刷掉就遺失
        log_unhandled_exception("train_stage1_simple.main()")
        raise
