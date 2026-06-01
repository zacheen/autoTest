"""Colored checkpoint-load logger with a per-area source tracker.

Two responsibilities:
  1. Print load messages in ANSI green on success and ANSI red on failure, so
     the operator can tell at a glance whether a checkpoint was actually loaded.
  2. Maintain a dict[area -> source path] of the *final effective* weight source
     for each area (e.g. backbone / q_network / yolo_backbone). Later writers
     overwrite earlier ones, so after the full load chain the dict reflects
     reality even when Stage 1 warm-start is overwritten by a V3 own checkpoint.

The tracker is what dump_hyperparameters reads to emit a [loaded_checkpoints]
section, so the operator can answer "which weight did this run actually use?"
by opening hyperparameters.txt.
"""

from __future__ import annotations

import datetime
from typing import Any


CHECKPOINT_RED = "\033[91;1m"
CHECKPOINT_GREEN = "\033[92;1m"
CHECKPOINT_RESET = "\033[0m"

# Sentinel value recorded into loaded_sources when an area has no checkpoint
# at all (file missing or load failed before anything was applied).
INIT_FROM_SCRATCH = "<random init / from scratch>"


class CheckpointLogger:
    """Per-prefix logger used by Stage 1 and V3 agents.

    Usage:
        logger = CheckpointLogger("[V3 CHECKPOINT]")
        ...
        logger.attach_io_log(self._io_log)   # optional, mirrors to file
        logger.success("backbone", path)     # green print + tracker write
        logger.failure("backbone", "missing checkpoint: ...")   # red print
        logger.mark_special("q_target", "<copied from q_network>", "...")

    Then pass ``logger.loaded_sources`` to dump_hyperparameters.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.loaded_sources: dict[str, str] = {}
        self._io_log: Any = None

    def attach_io_log(self, io_log: Any) -> None:
        """Attach a RolloverTextLog-like sink. Writes are best-effort."""
        self._io_log = io_log

    # ── internal print + mirror helper ────────────────────────────────
    def _emit(self, message: str, color: str | None) -> None:
        line = f"{self.prefix} {message}"
        if color:
            print(f"{color}{line}{CHECKPOINT_RESET}")
        else:
            print(line)
        if self._io_log is not None:
            try:
                self._io_log.write(
                    f"{datetime.datetime.now().isoformat()} {line}\n"
                )
                self._io_log.flush()
            except Exception:
                # io_log is best-effort. Never let a logging failure cascade
                # into the training path.
                pass

    # ── public API ────────────────────────────────────────────────────
    def success(
        self,
        area: str,
        source: Any,
        message: str | None = None,
    ) -> None:
        """Record that `area` was loaded from `source` and print green.

        Overwrites any prior value for `area`, so the final state of
        loaded_sources reflects the last write (e.g. V3 own checkpoint
        overwrites Stage 1 warm-start). `message` defaults to
        ``f"Loaded {area}: {source}"``.
        """
        src = str(source)
        self.loaded_sources[area] = src
        text = message if message is not None else f"Loaded {area}: {src}"
        self._emit(text, CHECKPOINT_GREEN)

    def failure(self, area: str, message: str) -> None:
        """Print red. Mark area as INIT_FROM_SCRATCH only if not yet set.

        Setdefault semantics: if a prior `success` already recorded a path
        for this area (e.g. Stage 1 warm-start succeeded but V3 own
        checkpoint now fails), we keep the earlier path because that is
        the weight that actually ended up in the model.
        """
        self.loaded_sources.setdefault(area, INIT_FROM_SCRATCH)
        self._emit(message, CHECKPOINT_RED)

    def mark_special(
        self,
        area: str,
        source: str,
        message: str | None = None,
    ) -> None:
        """Record a non-file source string for `area` (e.g. copied / migrated).

        Always overwrites. `message` is printed in green when supplied.
        """
        self.loaded_sources[area] = source
        if message is not None:
            self._emit(message, CHECKPOINT_GREEN)

    def info(self, message: str) -> None:
        """Print without color; does not touch loaded_sources."""
        self._emit(message, None)

    def warn(self, message: str) -> None:
        """Print red without touching loaded_sources.

        Used for non-checkpoint attention-grabbing notices (e.g. dropout latch
        switching on) where we want operator visibility but the message is
        unrelated to weight loading.
        """
        self._emit(message, CHECKPOINT_RED)
