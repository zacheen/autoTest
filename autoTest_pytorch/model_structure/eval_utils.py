"""Shared helpers for fixed-policy evaluation scheduling, timing, and logging."""

from __future__ import annotations

import datetime
import time
from typing import Callable


def should_run_eval(
    episode: int,
    *,
    offset: int,
    interval: int,
    training_started: bool = True,
) -> bool:
    """Return whether the current episode should trigger evaluation."""
    if not training_started:
        return False
    return episode >= offset and (episode - offset) % interval == 0


def start_eval_timing(
    last_eval_started_at: float | None,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[float, float]:
    """Return this eval start time and seconds since the previous eval start."""
    started_at = clock()
    if last_eval_started_at is None:
        return started_at, 0.0
    return started_at, started_at - last_eval_started_at


def finish_eval_timing(
    started_at: float,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Return seconds elapsed since start_eval_timing returned started_at."""
    return clock() - started_at


def log_eval_metrics(
    logger,
    *,
    episode: int,
    avg_reward: float,
    win_rate: float,
    avg_steps: float,
    avg_invalid_rate: float,
    seconds_since_last_eval: float | None = None,
    duration_seconds: float | None = None,
    console_prefix: str = "EVAL",
    log_win_rate_tb: bool = True,
) -> None:
    """Write eval metrics to TensorBoard, CSV, and console.

    ``win_rate`` is a 0~1 fraction. ``log_win_rate_tb=False`` suppresses the
    standalone ``eval/win_rate`` TB tag (CSV is always kept); v3 uses this so the
    eval win rate is shown only via the train/eval sub-run overlay.
    """
    logger.log("eval/avg_reward", avg_reward, step=episode, csv_col="eval_avg_reward")
    logger.log(
        "eval/win_rate", win_rate, step=episode, csv_col="eval_win_rate",
        tb=log_win_rate_tb,
    )
    logger.log("eval/avg_steps", avg_steps, step=episode, csv_col="eval_avg_steps")
    logger.log(
        "eval/avg_invalid_rate",
        avg_invalid_rate,
        step=episode,
        csv_col="eval_avg_invalid_rate",
    )

    time_parts = []
    if seconds_since_last_eval is not None:
        logger.log(
            "eval_dur/seconds_since_last_eval",
            seconds_since_last_eval,
            step=episode,
            csv_col="eval_seconds_since_last_eval",
        )
        time_parts.append(f"Since Last Eval: {seconds_since_last_eval:.1f}s")
    if duration_seconds is not None:
        logger.log(
            "eval_dur/duration_seconds",
            duration_seconds,
            step=episode,
            csv_col="eval_duration_seconds",
        )
        time_parts.append(f"Eval Time: {duration_seconds:.1f}s")

    logger.log("timestamp", datetime.datetime.now().isoformat(), step=episode, tb=False)
    logger.log("episode", episode, step=episode, tb=False)
    logger.commit_csv_row()
    logger.flush()

    suffix = " | " + " | ".join(time_parts) if time_parts else ""
    print(
        f"[{console_prefix} Ep {episode}] "
        f"Avg Reward: {avg_reward:.3f} | "
        f"Win Rate: {win_rate:.1%} | "
        f"Avg Steps: {avg_steps:.1f} | "
        f"Invalid Rate: {avg_invalid_rate:.2%}"
        f"{suffix}"
    )
