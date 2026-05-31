"""Archive directories and rollover helpers for training runs.

Creates one session directory under ``model_path`` and one subdirectory per
rollover window:

    <model_path>/
        training_<session_ts>/
            hour_00_<hour_ts>/
                train_io_log.txt
                cuda_debug.log
                training_log.csv
            hour_01_<hour_ts>/

Agents keep canonical checkpoints at ``model_path`` root. Hour directories hold
logs and optional checkpoint snapshots. Rollover callbacks receive the new hour
directory and reopen any per-hour resources they own.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Callable, TextIO


# Default rollover interval: one wall-clock hour.
DEFAULT_ROLLOVER_SECONDS = 3600


class SessionArchiveManager:
    """Create archive directories and notify callbacks on rollover.

    Args:
        model_path: Checkpoint root, e.g. ``./models/stage1_transformer``.
        log_prefix: Prefix for console messages.
        rollover_seconds: Seconds between rollover windows.

    Attributes:
        session_dir: ``model_path / training_<session_ts>/``.
        current_archive_dir: Active ``hour_NN_<hour_ts>/`` directory.
        hour_index: Current rollover index, starting at 0.
    """

    def __init__(
        self,
        model_path: Path,
        log_prefix: str = "[Archive]",
        rollover_seconds: int = DEFAULT_ROLLOVER_SECONDS,
    ):
        self.model_path = Path(model_path)
        self.log_prefix = log_prefix
        self.rollover_seconds = int(rollover_seconds)

        self.session_start = datetime.datetime.now()
        self._session_ts = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.model_path / f"training_{self._session_ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._hour_index = 0
        self._hour_start = self.session_start
        self.current_archive_dir = self._make_hour_dir(self._hour_index, self.session_start)
        self.current_archive_dir.mkdir(parents=True, exist_ok=True)

        self._on_rollover: list[Callable[[Path], None]] = []

        print(f"{log_prefix} Session archive: {self.session_dir}")
        print(f"{log_prefix} Current hour:    {self.current_archive_dir}")

    @property
    def hour_index(self) -> int:
        return self._hour_index

    @property
    def hour_start(self) -> datetime.datetime:
        return self._hour_start

    def _make_hour_dir(self, idx: int, start_dt: datetime.datetime) -> Path:
        ts = start_dt.strftime("%Y%m%d_%H%M%S")
        return self.session_dir / f"hour_{idx:02d}_{ts}"

    def register_on_rollover(self, callback: Callable[[Path], None]) -> None:
        """Register a callback that receives the new hour directory.

        Callbacks run in registration order. Exceptions are printed and do not
        stop later callbacks.
        """
        self._on_rollover.append(callback)

    def maybe_rollover(self) -> bool:
        """Roll to a new hour directory when the interval has elapsed.

        Returns:
            ``True`` if rollover happened; otherwise ``False``.
        """
        now = datetime.datetime.now()
        if (now - self._hour_start).total_seconds() < self.rollover_seconds:
            return False

        self._hour_index += 1
        self._hour_start = now
        self.current_archive_dir = self._make_hour_dir(self._hour_index, now)
        self.current_archive_dir.mkdir(parents=True, exist_ok=True)

        for cb in self._on_rollover:
            try:
                cb(self.current_archive_dir)
            except Exception as exc:
                # One bad callback should not block the others.
                print(f"{self.log_prefix} on_rollover callback {cb!r} failed: {exc}")

        print(f"{self.log_prefix} Hour rollover -> {self.current_archive_dir}")
        return True

    def find_latest_archive(self, filename: str) -> Path | None:
        """Return newest archived ``filename`` from ``training_*/hour_*/``.

        Uses the timestamp in the hour directory name instead of mtime.
        """
        candidates: list[tuple[str, Path]] = []
        for session_dir in self.model_path.glob("training_*"):
            if not session_dir.is_dir():
                continue
            for hour_dir in session_dir.glob("hour_*"):
                if not hour_dir.is_dir():
                    continue
                # hour_NN_YYYYMMDD_HHMMSS -> timestamp is the third split part.
                parts = hour_dir.name.split("_", 2)
                if len(parts) < 3:
                    continue
                ts_str = parts[2]  # Lexicographic order matches time order.
                path = hour_dir / filename
                if path.exists():
                    candidates.append((ts_str, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


class RolloverTextLog:
    """Append-only text log whose file path can be swapped on rollover."""

    def __init__(self, banner_factory: Callable[[Path], list[str]] | None = None):
        self._banner_factory = banner_factory
        self._file: TextIO | None = None
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def closed(self) -> bool:
        return self._file is None or self._file.closed

    def swap_to(self, path: Path) -> None:
        """Switch to ``path`` and write a banner if configured."""
        if self._file is not None and not self._file.closed:
            try:
                self._file.close()
            except Exception:
                pass
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        if self._banner_factory is not None:
            try:
                lines = self._banner_factory(self._path)
            except Exception as exc:
                lines = [f"[RolloverTextLog] banner_factory failed: {exc}"]
            if lines:
                self._file.write(f"\n{'=' * 60}\n")
                for line in lines:
                    self._file.write(f"{line}\n")
                self._file.write(f"{'=' * 60}\n")
                self._file.flush()

    def write(self, s: str) -> int:
        if self._file is None:
            return 0
        return self._file.write(s)

    def flush(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            try:
                self._file.close()
            except Exception:
                pass


def swap_logger_file_handler(
    logger: logging.Logger,
    new_path: Path,
    *,
    fmt: str = "%(asctime)s %(message)s",
    delay: bool = True,
) -> None:
    """Replace existing ``FileHandler`` objects with one writing to ``new_path``.

    Non-file handlers are left untouched. ``delay=True`` avoids creating empty
    files until the logger emits a record.
    """
    new_path = Path(new_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)
    handler = logging.FileHandler(new_path, mode="a", encoding="utf-8", delay=delay)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
