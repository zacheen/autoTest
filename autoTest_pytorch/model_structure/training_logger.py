"""TensorBoard writer plus row-buffered CSV logging.

``log()`` writes TensorBoard scalars immediately when ``tb=True``. With
``csv=True``, it buffers the value in the current row until
``commit_csv_row()`` writes that row to disk. Use ``csv=False`` for metrics
that should not enter the episode CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class CSVLogger:
    """Append CSV rows and switch files on rollover."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.writer = None
        self._fieldnames: list[str] | None = None

    def open(self, fieldnames: list[str]) -> None:
        self._fieldnames = list(fieldnames)
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self.file = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self._fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def swap_to(self, new_path: Path) -> bool:
        """Switch the open CSV file to ``new_path``.

        Returns:
            ``True`` if the path changed, otherwise ``False``.
        """
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
        self.file = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self._fieldnames)
        if not file_exists:
            self.writer.writeheader()
        return True

    def write(self, row: dict) -> None:
        if self.writer is None:
            raise RuntimeError("CSVLogger.write called before open()")
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if self.file and not self.file.closed:
            try:
                self.file.close()
            except Exception:
                pass

    @property
    def closed(self) -> bool:
        return self.file is None or self.file.closed


class TrainingLogger:
    """Facade over ``SummaryWriter`` and row-buffered CSV output.

    Args:
        csv_path: Initial CSV path.
        csv_fields: Fixed CSV columns. ``log(csv=True)`` must target one of them.
        tb_writer: TensorBoard writer for the current run.
    """

    def __init__(
        self,
        csv_path: Path,
        csv_fields: list[str],
        tb_writer: SummaryWriter,
    ):
        self._csv = CSVLogger(csv_path)
        self._csv.open(list(csv_fields))
        self._csv_fields: set[str] = set(csv_fields)
        self._tb = tb_writer
        # Buffer csv=True fields until commit_csv_row writes one CSV row.
        self._pending_row: dict[str, Any] = {}

    @property
    def csv_path(self) -> Path:
        return self._csv.path

    @property
    def csv_fields(self) -> list[str]:
        # Preserve header order for callers that display or compare the schema.
        return list(self._csv._fieldnames or [])

    def log(
        self,
        name: str,
        value: Any,
        step: int,
        *,
        tb: bool = True,
        csv: bool = True,
        csv_col: str | None = None,
    ) -> None:
        """Write one metric to TensorBoard and/or the pending CSV row.

        Args:
            name: TensorBoard tag. Also used as the CSV column when ``csv_col`` is None.
            value: TensorBoard scalar, or any CSV value when ``tb=False``.
            step: TensorBoard step. Ignored when ``tb=False``.
            tb: Write ``value`` to TensorBoard now.
            csv: Store ``value`` in the pending CSV row.
            csv_col: Optional flat CSV column name.

        Raises:
            KeyError: ``csv=True`` and the target column is not in ``csv_fields``.
        """
        if tb:
            self._tb.add_scalar(name, value, step)
        if csv:
            col = csv_col if csv_col is not None else name
            if col not in self._csv_fields:
                raise KeyError(
                    f"TrainingLogger: csv col {col!r} not in declared csv_fields "
                    f"({sorted(self._csv_fields)})"
                )
            self._pending_row[col] = value

    def commit_csv_row(self) -> None:
        """Write the pending CSV row and clear the buffer."""
        # None becomes an empty string in csv.DictWriter, then NaN in pandas.
        row = {col: self._pending_row.get(col, None) for col in self._csv.writer.fieldnames}
        self._csv.write(row)
        self._pending_row = {}

    def discard_pending(self) -> None:
        """Drop the current buffer without committing it."""
        self._pending_row = {}

    def swap_csv_to(self, new_path: Path) -> bool:
        """Switch CSV output to ``new_path``; TensorBoard stays in the same run."""
        return self._csv.swap_to(new_path)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Write TensorBoard text, usually one-shot docs at step 0."""
        self._tb.add_text(tag, text, step)

    def flush(self) -> None:
        """Flush TensorBoard events to disk."""
        self._tb.flush()

    def close(self) -> None:
        """Close CSV and TensorBoard handles."""
        self._csv.close()
        try:
            self._tb.close()
        except Exception:
            pass

    def close_csv(self) -> None:
        """Close only CSV; SummaryWriter may keep writing."""
        self._csv.close()

    @property
    def tb_writer(self) -> SummaryWriter:
        """Return the underlying writer for ops not wrapped here."""
        return self._tb

    @property
    def closed(self) -> bool:
        """CSV file state; SummaryWriter has no public closed flag."""
        return self._csv.closed
