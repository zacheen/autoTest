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
import time
import datetime

import numpy as np
import torch
from pathlib import Path

from Minesweeper.MinesweeperLogic import MinesweeperLogic, get_board_config
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from model_structure.history import History
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
    # 只計 invalid 一邊就夠了:total_clicks == episode_steps,valid 可由
    # episode_steps - invalid_clicks 反推。raw count 不外傳,只有 invalid_rate 出去。
    invalid_clicks = 0
    train_info_list = []

    while not done and episode_steps < MAX_STEPS_PER_EPISODE:
        state = torch.from_numpy(logic.get_grid_state_array())
        row, col = agent.select_action(state, add_noise=add_noise)

        result = logic.click(row, col)
        reward = compute_reward(result)
        episode_reward += reward

        if not result.changed:
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

    invalid_rate = invalid_clicks / episode_steps if episode_steps > 0 else 0.0

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
        'Q_loss': avg_q_loss,
        'q_mean': avg_q_mean,
    }


def run_evaluation(logic, agent):
    return run_fixed_policy_evaluation(logic, agent, num_episodes=EVAL_EPISODES)


def run_fixed_policy_evaluation(logic, agent, num_episodes):
    # 用一個 ephemeral History 累積本次 eval session 的每場結果,query 時
    # 走 window=None 取「全部現有資料」的 mean — 跟原本 np.mean(list) 同義,
    # 但 averaging 邏輯集中在 History,不再在 caller 重新寫一次。
    # max_capacity=num_episodes 確保 deque 不會截斷。
    eval_hist = History(max_capacity=num_episodes)

    for _ in range(num_episodes):
        stats = run_episode(logic, agent, add_noise=False)
        eval_hist.record(
            win=stats['is_win'],
            total_reward=stats['reward'],
            steps=stats['steps'],
            invalid_rate=stats['invalid_rate'],
        )

    return {
        'avg_reward':       eval_hist.avg_reward(window=None),
        'win_rate':         eval_hist.win_rate(window=None) * 100,  # 百分比
        'avg_steps':        eval_hist.avg_steps(window=None),
        'avg_invalid_rate': eval_hist.avg_invalid_rate(window=None),
    }


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

    # Agent 自己 own TrainingLogger(裡面包了 SummaryWriter + CSV);這個 train
    # script 預設不傳 csv_fields,用 TransformerDiscreteAgent 內建那組 default
    # (見 transformer_discrete_agent._DEFAULT_CSV_FIELDS)。若要自訂欄位,把
    # 想要的清單透過 ctor 的 csv_fields kw 傳進去。
    logic = MinesweeperLogic(rows=GRID_CONFIG.rows, cols=GRID_CONFIG.cols, mines_count=GRID_CONFIG.mines)
    agent = TransformerDiscreteAgent(grid_h=GRID_CONFIG.rows, grid_w=GRID_CONFIG.cols)
    logger = agent.training_logger  # 後面所有 TB / CSV 寫入都走這個

    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")
    print(f"  Active run: {agent.tensorboard_log_dir}")
    print(f"CSV log: {logger.csv_path}")
    print()

    # NOTE: recent_rewards / recent_wins / recent_steps 已搬進 agent.training_history。
    # 它在 log_episode_metrics() 內 record(),console log 直接 query
    # training_history.win_rate(window=LOG_INTERVAL) 等方法,順便獲得 resume 持久化。
    # 累計勝場改用 agent.training_history.total_wins (cumulative,resume 後持續累積)。
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
            # 累計勝場由 agent.training_history.total_wins 維護 — 不需要本地 counter。

            # Episode-summary metrics 走 TrainingLogger → 同時寫 TB + CSV row buffer。
            # NOTE: 高頻 train/* / grad/* / fpn/* / buffer/* / weight_norm/* … 由
            # agent.train_step 自己直接寫(走 total_it 軸),不繞 logger。
            # NOTE: `episode/reward_mean` / `episode/invalid_click_rate` / `episode/epsilon`
            # 由 agent.log_episode_metrics 已直接寫到 TB,這裡不重複寫。
            ep_idx = agent.episode_count
            logger.log("episode/reward_sum", stats['reward'], step=ep_idx, csv_col="reward")
            logger.log("episode/steps",      stats['steps'],  step=ep_idx, csv_col="steps")
            # is_win 只進 CSV(TB 上看 win_rate_recent 比較有意義)
            logger.log("is_win", int(stats['is_win']), step=ep_idx, tb=False)
            # invalid_rate 已由 agent 寫 TB → CSV 端也要存,但 TB 端 tb=False 避免重複
            logger.log("invalid_rate", stats['invalid_rate'], step=ep_idx, tb=False)
            logger.log("epsilon", agent.epsilon, step=ep_idx, tb=False)  # TB 端由 agent 寫
            if stats['Q_loss'] is not None:
                logger.log("episode/Q_loss_avg", stats['Q_loss'], step=ep_idx, csv_col="Q_loss")
                logger.log("episode/q_mean_avg", stats['q_mean'], step=ep_idx, csv_col="q_mean")
            logger.log("timestamp", datetime.datetime.now().isoformat(), step=ep_idx, tb=False)

            # Agent 內部 counter — TB only(per-step 性質,不入 CSV)。
            logger.log("train/total_it", agent.total_it, step=ep_idx, csv=False)
            logger.log("train/n_step_buffer_len", len(agent.n_step_buffer), step=ep_idx, csv=False)

            # 評估
            if episode >= EVAL_OFFSET and (episode - EVAL_OFFSET) % EVAL_INTERVAL == 0:
                eval_stats = run_evaluation(logic, agent)
                logger.log("eval/avg_reward",       eval_stats['avg_reward'],        step=ep_idx, csv_col="eval_avg_reward")
                logger.log("eval/win_rate",         eval_stats['win_rate'],          step=ep_idx, csv_col="eval_win_rate")
                logger.log("eval/avg_steps",        eval_stats['avg_steps'],         step=ep_idx, csv_col="eval_avg_steps")
                logger.log("eval/avg_invalid_rate", eval_stats['avg_invalid_rate'], step=ep_idx, csv_col="eval_avg_invalid_rate")

                print(f"  [EVAL Ep {episode:>6d}] "
                      f"Avg Reward: {eval_stats['avg_reward']:>7.2f} | "
                      f"Win Rate: {eval_stats['win_rate']:>5.1f}% | "
                      f"Avg Steps: {eval_stats['avg_steps']:>5.1f} | "
                      f"Invalid Rate: {eval_stats['avg_invalid_rate']:.2%}")

            # 為了讓 CSV 第一欄是 episode index 而非 episode/reward_sum 之類,
            # 最後再補一筆只進 CSV 的 episode 編號(commit 前)。
            logger.log("episode", ep_idx, step=ep_idx, tb=False)
            logger.commit_csv_row()

            # Console log
            if episode % LOG_INTERVAL == 0:
                hist = agent.training_history
                avg_reward = hist.avg_reward(window=LOG_INTERVAL)
                win_rate = hist.win_rate(window=LOG_INTERVAL) * 100
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed

                # LOG_INTERVAL=50,每 50 ep 才寫;TB only(CSV 已有 reward 與
                # eval/* 可以自己 rolling)。走 logger 統一介面跟其他寫法一致。
                logger.log("train/avg_reward_50",   avg_reward, step=agent.episode_count, csv=False)
                logger.log("train/win_rate_recent", win_rate,   step=agent.episode_count, csv=False)

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
        total_wins = agent.training_history.total_wins
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
        print(f"CSV log: {logger.csv_path}")
        logger.close()
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
