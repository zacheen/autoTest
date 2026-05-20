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
from pathlib import Path

from Minesweeper.MinesweeperLogic import MinesweeperLogic, get_board_config
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from transformer_discrete_agent import (
    TransformerDiscreteAgent,
    log_unhandled_exception,
    GRID_STATE_CHANNELS,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
)


# ---------- 棋盤難度 ----------
# preset 集中在 Minesweeper/MinesweeperLogic.py 的 DIFFICULTIES;train 在這裡只
# 挑一個 preset。改盤面難度只動下面這行字串就好。
GRID_CONFIG = get_board_config("training")

# ---------- 訓練參數 ----------
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
# training_log.csv 路徑由 agent.current_archive_dir 動態決定(每 hour 翻頁)


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
            agent.block_action_for_state(state, row * GRID_CONFIG.cols + col)

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
            agent.block_action_for_state(state, row * GRID_CONFIG.cols + col)

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
        self._fieldnames = None

    def open(self, fieldnames):
        self._fieldnames = fieldnames
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self.file = open(self.path, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def swap_to(self, new_path):
        """Hour rollover:關掉目前的檔,在新路徑開新檔(同樣 fieldnames),寫 header。"""
        new_path = Path(new_path)
        if new_path == self.path:
            return False
        try:
            if self.file:
                self.file.close()
        except Exception:
            pass
        self.path = new_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self.file = open(self.path, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self._fieldnames)
        if not file_exists:
            self.writer.writeheader()
        return True

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def main():
    print("=" * 60)
    num_actions = GRID_CONFIG.rows * GRID_CONFIG.cols
    print(f"  Stage 1 FQF: Grid State → Transformer → FQF Q-Network → {num_actions} actions")
    print("=" * 60)
    print(f"Grid: {GRID_CONFIG.rows}x{GRID_CONFIG.cols}, Mines: {GRID_CONFIG.mines}")
    print(
        f"Architecture: {num_actions} tokens × {GRID_STATE_CHANNELS}-d → "
        f"Transformer(d={TRANSFORMER_D_MODEL}, h={TRANSFORMER_NHEAD}, "
        f"L={TRANSFORMER_NUM_LAYERS}) → per-token logit"
    )
    print(f"Max episodes: {MAX_EPISODES}")
    print()

    logic = MinesweeperLogic(rows=GRID_CONFIG.rows, cols=GRID_CONFIG.cols, mines_count=GRID_CONFIG.mines)
    agent = TransformerDiscreteAgent(grid_h=GRID_CONFIG.rows, grid_w=GRID_CONFIG.cols)

    # TensorBoard — reuse the writer the agent created in __init__ so that
    # train-step diagnostics (td_error / grad / weights drift) land in the
    # same log_dir as the eval/episode scalars logged here.
    writer = agent.tb_writer
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")
    print(f"  Active run: {agent.tensorboard_log_dir}")

    # CSV — 寫到 agent 的當下 hour 資料夾。每滿 1 hour agent 會翻新資料夾,
    # 主迴圈在每次 write 之前 check 一下,有翻就 swap CSV 檔。
    csv_path = agent.current_archive_dir / "training_log.csv"
    csv_logger = CSVLogger(csv_path)
    csv_fields = [
        'episode', 'reward', 'steps', 'is_win', 'invalid_rate',
        'valid_clicks', 'invalid_clicks',
        'Q_loss', 'q_mean', 'epsilon',
        'eval_avg_reward', 'eval_win_rate', 'eval_avg_steps', 'eval_avg_invalid_rate',
        'timestamp',
    ]
    csv_logger.open(csv_fields)
    print(f"CSV log: {csv_path}")
    print()

    # NOTE: recent_rewards / recent_wins / recent_steps 已搬進 agent.training_history。
    # 它在 log_episode_metrics() 內 record(),console log 直接 query
    # training_history.win_rate(window=LOG_INTERVAL) 等方法,順便獲得 resume 持久化。
    total_wins = 0
    start_time = time.time()

    try:
        for episode in range(1, MAX_EPISODES + 1):
            stats = run_episode(logic, agent, add_noise=True)

            # 先讓 AdaptiveEpsilonController 看到這場結果（更新 rolling window
            # + 算出 next eps），on_episode_end 再把更新後的 epsilon 寫進 TB
            # 並做 periodic save。順序與 v3 / Demo_test_Minesweeper 一致。
            # NOTE: log_episode_metrics 期望的 reward_mean 是「每步平均 reward」，
            # 跟 Stage 2 (Demo_test_Minesweeper) 的 average_reward() 語意一致。
            # 不要傳 stats['reward']（那是整場總和）。
            episode_reward_mean = stats['reward'] / max(stats['steps'], 1)
            agent.log_episode_metrics(
                win=stats['is_win'],
                invalid_click_rate=stats['invalid_rate'],
                reward_mean=episode_reward_mean,
                total_reward=stats['reward'],
                steps=stats['steps'],
            )
            agent.on_episode_end()

            if stats['is_win']:
                total_wins += 1

            # TensorBoard — episode-level scalars. NOTE: do NOT reuse the
            # `train/Q_loss` / `train/q_mean` tag names; the agent already
            # writes those per gradient step (different step axis) — sharing
            # the tag corrupts the curves with two interleaved step counters.
            writer.add_scalar('episode/reward_sum', stats['reward'], agent.episode_count)
            writer.add_scalar('episode/steps', stats['steps'], agent.episode_count)
            # NOTE: `episode/invalid_rate` 已由 log_episode_metrics 以
            # `episode/invalid_click_rate` 名稱寫入（同值），不需在此重複寫。
            # NOTE: `episode/reward_mean`（每步平均 reward）由 log_episode_metrics
            # 寫入，這裡寫的 `episode/reward_sum` 是整場總和，兩者互補。
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

            # Hour rollover check:若 agent 已翻到下一個 hour 資料夾,把 CSV 也接過去
            target_csv = agent.current_archive_dir / "training_log.csv"
            if csv_logger.path != target_csv:
                csv_logger.swap_to(target_csv)
            csv_logger.write(csv_row)

            # Console log
            if episode % LOG_INTERVAL == 0:
                hist = agent.training_history
                avg_reward = hist.avg_reward(window=LOG_INTERVAL)
                win_rate = hist.win_rate(window=LOG_INTERVAL) * 100
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed

                writer.add_scalar('train/avg_reward_50', avg_reward, agent.episode_count)
                writer.add_scalar('train/win_rate_recent', win_rate, agent.episode_count)

                now_str = datetime.datetime.now().strftime("%H:%M")
                print(f"[{now_str}] "
                      f"[Ep {episode:>6d}] "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"Win Rate(50): {win_rate:>5.1f}% | "
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
        print(f"CSV log: {csv_logger.path}")
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
