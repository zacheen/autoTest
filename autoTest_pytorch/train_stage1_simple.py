"""
Stage 1 Transformer experiment: learn Minesweeper spatial reasoning with self-attention.

Architecture: grid state (12,10,10) -> 100 tokens x 12-d + 2D pos encoding
     → 4-layer Transformer (d=64, h=4) → per-token logit → 100 actions

Usage:
  python train_stage1_simple.py

Monitoring:
  tensorboard --logdir ./models/stage1_transformer/tensorboard
"""

import os
import time
import datetime

import numpy as np
import torch
from pathlib import Path

from Minesweeper.MinesweeperLogic import MinesweeperLogic, get_board_config
from model_structure.eval_utils import (
    finish_eval_timing,
    log_eval_metrics,
    should_run_eval,
    start_eval_timing,
)
from model_structure.reward_settings import MINESWEEPER_REWARD_CONFIG
from model_structure.history import History
from transformer_discrete_agent import (
    TransformerDiscreteAgent,
    GRID_STATE_CHANNELS,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    MINIMUM_DATA_SIZE,
)


# ---------- Board difficulty ----------
# Presets live in Minesweeper/MinesweeperLogic.py DIFFICULTIES; training only
# chooses one preset here. Change board difficulty by editing this string.
GRID_CONFIG = get_board_config("training")

# ---------- Training parameters ----------
MAX_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 200
LOG_INTERVAL = 50
SAVE_DEMO_INTERVAL = 300

# ---------- Evaluation parameters ----------
EVAL_INTERVAL = 200
EVAL_EPISODES = 30
EVAL_OFFSET = 50  # First eval at ep 50, then every EVAL_INTERVAL: 50, 250, 450...
RESUME_PREFILL_EVAL_EPISODES = 300

# ---------- Paths ----------
TENSORBOARD_DIR = Path("./models/stage1_transformer/tensorboard")
# training_log.csv path is chosen dynamically by agent.current_archive_dir per hour.


def action_to_grid(action, rows, cols):
    """Discrete action [0, 99] → (row, col)."""
    return action // cols, action % cols


def format_grid(logic, click_row=None, click_col=None, result=None):
    """Render the game grid as text.

    Legend:
        .  = unrevealed
        F  = flagged
        0-8 = revealed number
        *  = mine, after game over
        [X] = clicked position

    Returns:
        str: formatted grid text
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

            # Mine hit: show mine.
            if result and result.game_over and result.hit_mine == (r, c):
                ch = "*"

            # Mark clicked position.
            if r == click_row and c == click_col:
                cell = f"[{ch}]"
            else:
                cell = f" {ch} "

            row_str += cell
        lines.append(row_str)

    return "\n".join(lines)


DEMO_TRAINING_EPISODES = 3  # Number of training demos printed on save.
DEMO_LOG_PATH = Path("./models/stage1_transformer/demo_log.txt")


def run_demo_episode(f, logic, agent, mode="validation"):
    """Run one demo episode and write it to file.

    Args:
        f: opened file object
        logic: MinesweeperLogic
        agent: agent instance
        mode: "training" (sample) or "validation" (argmax)
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
    """Write demo episodes on model save: some training demos and one validation."""
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
    # Count only invalid clicks: total_clicks == episode_steps, and valid clicks
    # can be inferred as episode_steps - invalid_clicks. Only invalid_rate leaves.
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
            # Skip first step: first click is always valid and dilutes the valid group.
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
    # Use an ephemeral History for this eval session. Query with window=None to
    # average all current data, equivalent to np.mean(list), while keeping
    # averaging logic inside History. max_capacity prevents deque truncation.
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
        'win_rate':         eval_hist.win_rate(window=None) * 100,  # Percentage.
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

    # Agent owns TrainingLogger, including SummaryWriter and CSV. This script
    # does not pass csv_fields by default, so TransformerDiscreteAgent uses its
    # built-in defaults. Pass csv_fields through ctor to customize columns.
    logic = MinesweeperLogic(rows=GRID_CONFIG.rows, cols=GRID_CONFIG.cols, mines_count=GRID_CONFIG.mines)
    agent = TransformerDiscreteAgent(grid_h=GRID_CONFIG.rows, grid_w=GRID_CONFIG.cols)
    logger = agent.training_logger  # All later TB / CSV writes go through this.

    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")
    print(f"  Active run: {agent.tensorboard_log_dir}")
    print(f"CSV log: {logger.csv_path}")
    print()

    # NOTE: recent_rewards / recent_wins / recent_steps moved into
    # agent.training_history. log_episode_metrics() records them, console logs
    # query training_history methods directly, and resume persistence comes for free.
    # Cumulative wins use agent.training_history.total_wins.
    start_time = time.time()
    last_eval_started_at = None
    resume_eval_checked = False

    try:
        for episode in range(1, MAX_EPISODES + 1):
            stats = run_episode(logic, agent, add_noise=True)

            # Let AdaptiveEpsilonController see this episode first, updating its
            # rolling window and next epsilon. on_episode_end then writes updated
            # epsilon to TB and handles periodic save, matching v3 / Demo flow.
            # NOTE: log_episode_metrics expects per-step average reward, matching
            # Stage 2 Demo_test_Minesweeper average_reward(). Do not pass
            # stats['reward'], which is the episode sum.
            episode_reward_mean = stats['reward'] / max(stats['steps'], 1)
            agent.log_episode_metrics(
                win=stats['is_win'],
                invalid_click_rate=stats['invalid_rate'],
                reward_mean=episode_reward_mean,
                total_reward=stats['reward'],
                steps=stats['steps'],
            )
            agent.on_episode_end()
            # Cumulative wins are maintained by agent.training_history.total_wins.

            # Episode-summary metrics go through TrainingLogger, writing both TB
            # and CSV row buffer. High-frequency train/* / grad/* / fpn/* /
            # weight_norm/* are written by agent.train_step on the update step.
            # NOTE: `episode/reward_mean` / `episode/invalid_click_rate` / `episode/epsilon`
            # are already written to TB by agent.log_episode_metrics; do not duplicate.
            ep_idx = agent.episode_count
            logger.log("episode/reward_sum", stats['reward'], step=ep_idx, csv_col="reward")
            logger.log("episode/steps",      stats['steps'],  step=ep_idx, csv_col="steps")
            # is_win only goes to CSV; win_rate_recent is more useful in TB.
            logger.log("is_win", int(stats['is_win']), step=ep_idx, tb=False)
            # invalid_rate is already in TB from agent; store CSV only to avoid duplicates.
            logger.log("invalid_rate", stats['invalid_rate'], step=ep_idx, tb=False)
            logger.log("epsilon", agent.epsilon, step=ep_idx, tb=False)  # TB is written by agent.
            if stats['Q_loss'] is not None:
                logger.log("episode/Q_loss_avg", stats['Q_loss'], step=ep_idx, csv_col="Q_loss")
                logger.log("episode/q_mean_avg", stats['q_mean'], step=ep_idx, csv_col="q_mean")
            logger.log("timestamp", datetime.datetime.now().isoformat(), step=ep_idx, tb=False)

            # Add CSV-only episode index last, before commit, so the first CSV
            # column is episode instead of episode/reward_sum.
            logger.log("episode", ep_idx, step=ep_idx, tb=False)
            logger.commit_csv_row()

            eval_episodes = None

            # Resume evaluation runs once with the ordinary fixed-policy eval
            # path. If replay is still below the training threshold, use a larger
            # 300-episode sample; otherwise use the normal eval size.
            if agent.is_resume_training and agent.total_it > 0 and not resume_eval_checked:
                resume_eval_checked = True
                eval_episodes = (
                    RESUME_PREFILL_EVAL_EPISODES
                    if agent.replay_buffer.size() < MINIMUM_DATA_SIZE
                    else EVAL_EPISODES
                )
                print(
                    f"[EVAL] Resume eval: replay "
                    f"{agent.replay_buffer.size()}/{MINIMUM_DATA_SIZE}, "
                    f"episodes={eval_episodes}"
                )

            # Evaluation.
            elif should_run_eval(
                episode,
                offset=EVAL_OFFSET,
                interval=EVAL_INTERVAL,
                training_started=agent.total_it > 0 and agent.replay_buffer.size() >= MINIMUM_DATA_SIZE,
            ):
                eval_episodes = EVAL_EPISODES
                print(
                    f"[EVAL] Start fixed-policy evaluation at episode {episode}: "
                    f"{eval_episodes} episodes"
                )

            if eval_episodes is not None:
                eval_started_at, seconds_since_last_eval = start_eval_timing(last_eval_started_at)
                last_eval_started_at = eval_started_at
                eval_stats = run_fixed_policy_evaluation(logic, agent, eval_episodes)
                duration_seconds = finish_eval_timing(eval_started_at)
                log_eval_metrics(
                    logger,
                    episode=ep_idx,
                    avg_reward=eval_stats['avg_reward'],
                    win_rate=eval_stats['win_rate'],
                    avg_steps=eval_stats['avg_steps'],
                    avg_invalid_rate=eval_stats['avg_invalid_rate'],
                    seconds_since_last_eval=seconds_since_last_eval,
                    duration_seconds=duration_seconds,
                    console_prefix="EVAL",
                )

            # Console log
            if episode % LOG_INTERVAL == 0:
                hist = agent.training_history
                avg_reward = hist.avg_reward(window=LOG_INTERVAL)
                win_rate = hist.win_rate(window=LOG_INTERVAL) * 100
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed

                # LOG_INTERVAL=50 means every 50 episodes. TB only; CSV already has
                # rewards and eval/* can be rolled separately. Use the logger API.
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

        # Normal training completion.
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
    main()
