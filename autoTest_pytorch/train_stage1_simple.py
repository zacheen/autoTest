"""
Stage 1 Transformer 實驗 — 用 self-attention 學習 Minesweeper 空間推理。

架構：grid state (12,10,10) → 100 tokens × 12-d + 2D pos encoding
     → 4-layer Transformer (d=64, h=4) → per-token logit → 100 actions

執行方式：
  python train_stage1_simple.py

監控方式：
  tensorboard --logdir runs/stage1_transformer/
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
from RL_Agent import TransformerDiscreteAgent

# ---------- 訓練參數 ----------
GRID_ROWS = 6
GRID_COLS = 6
GRID_MINES = 4
MAX_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 200
LOG_INTERVAL = 50
SAVE_INTERVAL = 300

# ---------- 評估參數 ----------
EVAL_INTERVAL = 100
EVAL_EPISODES = 50

# ---------- 路徑 ----------
TENSORBOARD_DIR = Path("./runs/stage1_transformer")
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
    done = False
    step = 0
    total_reward = 0.0

    label = "TRAINING" if mode == "training" else "VALIDATION"
    f.write(f"\n  --- {label} demo ---\n")

    while not done and step < MAX_STEPS_PER_EPISODE:
        state = logic.get_grid_state_tensor()
        row, col = agent.select_action(state, add_noise=add_noise)

        result = logic.click(row, col)
        reward = compute_reward(result)
        total_reward += reward

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


def squash_reward(r):
    """Rainbow DQN 風格的 reward 壓縮，把 reward 壓到 [-1, +1] 附近。

    公式: sign(r) * (√(|r|+1) - 1) + 0.001 * r
    效果: +20 → +3.6, +6 → +1.6, +4 → +1.2, -3 → -1.0
    """
    return np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r


def compute_reward(result):
    """計算 reward（壓縮後）。

    4 個分類:
        +20.0 → +3.60   WIN
        +3.0  → +1.00   有效點擊（翻開新格子）
        -3.0  → -1.00   踩雷
        -2.95 → -0.99   無效點擊（點已翻開格）
    """
    if not result.changed:
        return squash_reward(-2.95)
    if result.win:
        return squash_reward(20.0)
    if result.game_over:
        return squash_reward(-3.0)
    return squash_reward(3.0)


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
        row, col = agent.select_action(state, add_noise=add_noise)

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
            # 跳過第一步：第一次點擊一定有效，沒有學習價值，會稀釋 valid group
            if episode_steps > 0:
                agent.store_transition(state, (row, col), next_state, reward, done)
                train_info = agent.train_step()
                if train_info is not None:
                    train_info_list.append(train_info)

        episode_steps += 1

    total_clicks = valid_clicks + invalid_clicks
    invalid_rate = invalid_clicks / total_clicks if total_clicks > 0 else 0.0

    avg_loss = None
    avg_q_mean = None
    if train_info_list:
        avg_loss = np.mean([t['loss'] for t in train_info_list])
        avg_q_mean = np.mean([t['q_mean'] for t in train_info_list])

    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'is_win': is_win,
        'invalid_rate': invalid_rate,
        'valid_clicks': valid_clicks,
        'invalid_clicks': invalid_clicks,
        'loss': avg_loss,
        'q_mean': avg_q_mean,
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
    num_actions = GRID_ROWS * GRID_COLS
    print(f"  Stage 1 DDQN: Grid State → Transformer → Dueling Q → {num_actions} actions")
    print("=" * 60)
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}, Mines: {GRID_MINES}")
    print(f"Architecture: 100 tokens × 12-d → Transformer(d=64, h=4, L=4) → per-token logit")
    print(f"Max episodes: {MAX_EPISODES}")
    print()

    logic = MinesweeperLogic(rows=GRID_ROWS, cols=GRID_COLS, mines_count=GRID_MINES)
    agent = TransformerDiscreteAgent(grid_h=GRID_ROWS, grid_w=GRID_COLS)

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
        'loss', 'q_mean', 'epsilon',
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

            # TensorBoard
            writer.add_scalar('train/episode_reward', stats['reward'], episode)
            writer.add_scalar('train/episode_steps', stats['steps'], episode)
            writer.add_scalar('train/invalid_rate', stats['invalid_rate'], episode)
            if stats['loss'] is not None:
                writer.add_scalar('train/loss', stats['loss'], episode)
                writer.add_scalar('train/q_mean', stats['q_mean'], episode)

            # CSV
            csv_row = {
                'episode': episode,
                'reward': f"{stats['reward']:.2f}",
                'steps': stats['steps'],
                'is_win': int(stats['is_win']),
                'invalid_rate': f"{stats['invalid_rate']:.4f}",
                'valid_clicks': stats['valid_clicks'],
                'invalid_clicks': stats['invalid_clicks'],
                'loss': f"{stats['loss']:.6f}" if stats['loss'] is not None else '',
                'q_mean': f"{stats['q_mean']:.4f}" if stats['q_mean'] is not None else '',
                'epsilon': f"{agent.epsilon:.4f}",
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

                overall_wr = total_wins / episode * 100
                print(f"[Ep {episode:>6d}] "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Win Rate(50): {win_rate:>5.1f}% | "
                      f"Overall WR: {overall_wr:>5.1f}% | "
                      f"Total Wins: {total_wins} | "
                      f"Speed: {eps_per_sec:.1f} ep/s | "
                      f"Epsilon: {agent.epsilon:.4f}")

            if episode % SAVE_INTERVAL == 0:
                agent._save_model()
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
