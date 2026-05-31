"""
Pure Minesweeper game logic with no UI dependency.
Shared by the GUI (Minesweeper.py) and Stage 1 pretraining (train_stage1.py).
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


# Grid state constants for get_grid_state.
CELL_UNREVEALED = -1
CELL_FLAGGED = -2

# One-hot channel indices for get_grid_state_array.
CH_UNREVEALED = 0
CH_FLAGGED = 1
CH_NUM_0 = 2     # Revealed blank cell, number 0.
CH_NUM_1 = 3
CH_NUM_8 = 10
CH_MINE = 11
NUM_CHANNELS = 12


# ---------- Board difficulty presets ----------
@dataclass(frozen=True)
class BoardConfig:
    """Board difficulty settings: rows x cols plus mine count."""
    rows: int
    cols: int
    mines: int


DIFFICULTIES: dict[str, BoardConfig] = {
    # Small RL training board; 6x6/4 converges quickly enough to validate the pipeline.
    "training":     BoardConfig(rows=6,  cols=6,  mines=4),
    # Standard Minesweeper difficulties.
    "Beginner":     BoardConfig(rows=9,  cols=9,  mines=10),
    "Intermediate": BoardConfig(rows=16, cols=16, mines=40),
    "Expert":       BoardConfig(rows=16, cols=30, mines=99),
}


def get_board_config(name: str) -> BoardConfig:
    """Return BoardConfig by difficulty name; unknown names raise with available presets."""
    if name not in DIFFICULTIES:
        raise KeyError(
            f"Unknown difficulty {name!r}. Available: {sorted(DIFFICULTIES.keys())}"
        )
    return DIFFICULTIES[name]


@dataclass
class ClickResult:
    """Left-click result."""
    changed: bool = False                       # Whether the board changed.
    game_over: bool = False                     # Whether a mine was hit.
    win: bool = False                           # Whether the game was won.
    revealed_cells: List[Tuple[int, int, int]] = field(default_factory=list)  # Newly revealed [(r, c, number), ...].
    hit_mine: Optional[Tuple[int, int]] = None  # Mine that was hit.


@dataclass
class FlagResult:
    """Right-click flag result."""
    toggled: bool = False       # Whether flag state changed.
    is_flagged: bool = False    # Whether the cell is flagged after toggling.


class MinesweeperLogic:
    """Pure Minesweeper logic with no UI dependency."""

    def __init__(self, rows: int = 10, cols: int = 10, mines_count: int = 10):
        self.rows = rows
        self.cols = cols
        self.mines_count = mines_count

        # Game state.
        self.mines = set()
        self.revealed = set()
        self.flags = set()
        self.game_over = False
        self.is_win = False
        self.first_click = True
        self.remaining_mines = mines_count

    def reset(self):
        """Reset the game to its initial state."""
        self.mines = set()
        self.revealed = set()
        self.flags = set()
        self.game_over = False
        self.is_win = False
        self.first_click = True
        self.remaining_mines = self.mines_count

    def click(self, row: int, col: int) -> ClickResult:
        """Left-click one cell.

        Args:
            row: Row (0-indexed)
            col: Column (0-indexed)
        Returns:
            ClickResult
        """
        result = ClickResult()

        # Out of bounds, ended, flagged, or revealed means invalid click.
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return result
        if self.game_over or self.is_win:
            return result
        if (row, col) in self.flags:
            return result
        if (row, col) in self.revealed:
            return result

        # First click: place mines while keeping the first click safe.
        if self.first_click:
            self.first_click = False
            self._place_mines(row, col)

        # Reveal cells.
        newly_revealed = []
        self._reveal_cell(row, col, newly_revealed)

        if not newly_revealed:
            return result

        result.changed = True
        result.revealed_cells = newly_revealed

        # Check for mine hit.
        if (row, col) in self.mines:
            self.game_over = True
            result.game_over = True
            result.hit_mine = (row, col)
            return result

        # Check for win.
        if self._check_win():
            self.is_win = True
            result.win = True

        return result

    def flag(self, row: int, col: int) -> FlagResult:
        """Right-click to toggle a flag.

        Args:
            row: Row (0-indexed)
            col: Column (0-indexed)
        Returns:
            FlagResult
        """
        result = FlagResult()

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return result
        if self.game_over or self.is_win:
            return result
        if (row, col) in self.revealed:
            return result

        if (row, col) in self.flags:
            self.flags.remove((row, col))
            self.remaining_mines += 1
            result.toggled = True
            result.is_flagged = False
        else:
            self.flags.add((row, col))
            self.remaining_mines -= 1
            result.toggled = True
            result.is_flagged = True

        return result

    def get_grid_state(self):
        """Return current grid state as a 2D list.

        Returns:
            list[list[int]]: Per-cell value:
                -1 = unrevealed
                -2 = flagged
                0-8 = revealed number
        """
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.flags:
                    row.append(CELL_FLAGGED)
                elif (r, c) in self.revealed:
                    row.append(self._count_adjacent_mines(r, c))
                else:
                    row.append(CELL_UNREVEALED)
            grid.append(row)
        return grid

    def get_grid_state_array(self):
        """Return one-hot encoded grid state array.

        Returns:
            np.ndarray: shape (NUM_CHANNELS, rows, cols), float32
                channel 0: unrevealed
                channel 1: flagged
                channel 2-10: numbers 0-8
                channel 11: mine, visible only after game_over

        Callers that need a torch tensor should convert with ``torch.from_numpy(...)``.
        """
        arr = np.zeros((NUM_CHANNELS, self.rows, self.cols), dtype=np.float32)

        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.flags:
                    arr[CH_FLAGGED, r, c] = 1.0
                elif (r, c) in self.revealed:
                    if (r, c) in self.mines:
                        # Mines are visible only after being hit.
                        arr[CH_MINE, r, c] = 1.0
                    else:
                        num = self._count_adjacent_mines(r, c)
                        arr[CH_NUM_0 + num, r, c] = 1.0
                else:
                    arr[CH_UNREVEALED, r, c] = 1.0

        return arr

    def _get_neighbors(self, row: int, col: int):
        """Return valid 8-neighbor coordinates for (row, col)."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    neighbors.append((r, c))
        return neighbors

    def get_logically_safe_cells(self):
        """Infer logically safe cells using constraint propagation.

        Uses only visible information, revealed numbers plus inferred facts, and
        does not read self.mines directly.

        Algorithm: repeatedly scan all revealed numbered cells and infer:
          1. If unknown mines around a number = 0, all unknown neighbors are safe.
          2. If unknown neighbors = unknown mines, all unknown neighbors are mines.

        Repeat until no new inference is possible.

        Returns:
            (safe_cells: set, inferred_mines: set)
        """
        inferred_mines = set(self.flags)
        inferred_safe = set()

        changed = True
        while changed:
            changed = False
            for (nr, nc) in self.revealed:
                if (nr, nc) in self.mines:
                    continue

                k = self._count_adjacent_mines(nr, nc)
                neighbors = self._get_neighbors(nr, nc)

                # Classify neighbors.
                mine_count = 0
                unknown = []
                for (r, c) in neighbors:
                    if (r, c) in self.revealed:
                        continue
                    if (r, c) in inferred_mines:
                        mine_count += 1
                    elif (r, c) not in inferred_safe:
                        unknown.append((r, c))

                remaining_mines = k - mine_count

                if remaining_mines == 0 and unknown:
                    # All unknown neighbors are safe.
                    inferred_safe.update(unknown)
                    changed = True
                elif remaining_mines > 0 and remaining_mines == len(unknown) and unknown:
                    # All unknown neighbors are mines.
                    inferred_mines.update(unknown)
                    changed = True

        return inferred_safe, inferred_mines

    # ---------- Internal methods ----------

    def _place_mines(self, exclude_row: int, exclude_col: int):
        """Place mines, excluding the first click's surrounding 3x3 area."""
        exclude = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = exclude_row + dr, exclude_col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    exclude.add((r, c))

        available = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if (r, c) not in exclude]

        self.mines = set(random.sample(available, min(self.mines_count, len(available))))

    def _count_adjacent_mines(self, row: int, col: int) -> int:
        """Count mines around one cell."""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols and (r, c) in self.mines:
                    count += 1
        return count

    def _reveal_cell(self, row: int, col: int, newly_revealed: list):
        """Reveal one cell, recursively expanding blank regions."""
        if (row, col) in self.revealed or (row, col) in self.flags:
            return

        self.revealed.add((row, col))

        # Hit mine.
        if (row, col) in self.mines:
            newly_revealed.append((row, col, -1))  # -1 means mine.
            return

        num = self._count_adjacent_mines(row, col)
        newly_revealed.append((row, col, num))

        # Blank cell, number 0: recursively reveal neighbors.
        if num == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        self._reveal_cell(r, c, newly_revealed)

    def _check_win(self) -> bool:
        """Check whether all non-mine cells are revealed."""
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.mines and (r, c) not in self.revealed:
                    return False
        return True
