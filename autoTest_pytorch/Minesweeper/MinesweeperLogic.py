"""
踩地雷遊戲邏輯層 — 純邏輯，無 UI 依賴。
供 GUI (Minesweeper.py) 和 Stage 1 預訓練 (train_stage1.py) 共用。
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for the web UI runtime
    torch = None


# Grid state 常數 (給 get_grid_state 用)
CELL_UNREVEALED = -1
CELL_FLAGGED = -2

# One-hot channel 索引 (給 get_grid_state_tensor 用)
CH_UNREVEALED = 0
CH_FLAGGED = 1
CH_NUM_0 = 2     # 已翻開空白 (數字 0)
CH_NUM_1 = 3
CH_NUM_8 = 10
CH_MINE = 11
NUM_CHANNELS = 12


@dataclass
class ClickResult:
    """左鍵點擊的回傳結果。"""
    changed: bool = False                       # 畫面是否有變化
    game_over: bool = False                     # 是否踩雷
    win: bool = False                           # 是否贏了
    revealed_cells: List[Tuple[int, int, int]] = field(default_factory=list)  # 新翻開的 [(r, c, number), ...]
    hit_mine: Optional[Tuple[int, int]] = None  # 踩到哪顆雷


@dataclass
class FlagResult:
    """右鍵插旗的回傳結果。"""
    toggled: bool = False       # 是否有切換
    is_flagged: bool = False    # 切換後是否有旗子


class MinesweeperLogic:
    """踩地雷純遊戲邏輯，不依賴任何 UI。"""

    def __init__(self, rows: int = 10, cols: int = 10, mines_count: int = 10):
        self.rows = rows
        self.cols = cols
        self.mines_count = mines_count

        # 遊戲狀態
        self.mines = set()
        self.revealed = set()
        self.flags = set()
        self.game_over = False
        self.is_win = False
        self.first_click = True
        self.remaining_mines = mines_count

    def reset(self):
        """重置遊戲到初始狀態。"""
        self.mines = set()
        self.revealed = set()
        self.flags = set()
        self.game_over = False
        self.is_win = False
        self.first_click = True
        self.remaining_mines = self.mines_count

    def click(self, row: int, col: int) -> ClickResult:
        """左鍵點擊某格。

        Args:
            row: 列 (0-indexed)
            col: 行 (0-indexed)
        Returns:
            ClickResult
        """
        result = ClickResult()

        # 超出範圍、已結束、已標旗、已翻開 → 無效點擊
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return result
        if self.game_over or self.is_win:
            return result
        if (row, col) in self.flags:
            return result
        if (row, col) in self.revealed:
            return result

        # 第一次點擊：放地雷（保證第一下安全）
        if self.first_click:
            self.first_click = False
            self._place_mines(row, col)

        # 翻開格子
        newly_revealed = []
        self._reveal_cell(row, col, newly_revealed)

        if not newly_revealed:
            return result

        result.changed = True
        result.revealed_cells = newly_revealed

        # 檢查是否踩雷
        if (row, col) in self.mines:
            self.game_over = True
            result.game_over = True
            result.hit_mine = (row, col)
            return result

        # 檢查是否贏了
        if self._check_win():
            self.is_win = True
            result.win = True

        return result

    def flag(self, row: int, col: int) -> FlagResult:
        """右鍵切換旗子。

        Args:
            row: 列 (0-indexed)
            col: 行 (0-indexed)
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
        """取得目前 grid 狀態 (2D list)。

        Returns:
            list[list[int]]: 每格的值
                -1 = 未翻開
                -2 = 已標旗
                0-8 = 已翻開的數字
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

    def get_grid_state_tensor(self):
        """取得 one-hot 編碼的 grid state tensor。

        Returns:
            torch.Tensor: shape (NUM_CHANNELS, rows, cols), float32
                channel 0: 未翻開
                channel 1: 已標旗
                channel 2-10: 數字 0-8
                channel 11: 地雷 (只有 game_over 時才可見)
        """
        if torch is None:
            raise ImportError("torch is required to call get_grid_state_tensor().")

        tensor = torch.zeros(NUM_CHANNELS, self.rows, self.cols, dtype=torch.float32)

        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.flags:
                    tensor[CH_FLAGGED, r, c] = 1.0
                elif (r, c) in self.revealed:
                    if (r, c) in self.mines:
                        # 踩雷後才看得到地雷
                        tensor[CH_MINE, r, c] = 1.0
                    else:
                        num = self._count_adjacent_mines(r, c)
                        tensor[CH_NUM_0 + num, r, c] = 1.0
                else:
                    tensor[CH_UNREVEALED, r, c] = 1.0

        return tensor

    def _get_neighbors(self, row: int, col: int):
        """取得 (row, col) 的合法 8 鄰居座標。"""
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
        """用約束傳播推導邏輯上安全的格子。

        只用可見資訊（翻開的數字 + 已知推導）做推理，
        不直接用 self.mines 作弊。

        算法：反覆掃描所有翻開的數字格，做兩種推導：
          1. 若某數字格周圍的未知地雷數 = 0 → 所有未知鄰居都安全
          2. 若某數字格周圍的未知鄰居數 = 未知地雷數 → 所有未知鄰居都是雷

        重複直到沒有新推導。

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

                # 分類鄰居
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
                    # 所有未知鄰居都安全
                    inferred_safe.update(unknown)
                    changed = True
                elif remaining_mines > 0 and remaining_mines == len(unknown) and unknown:
                    # 所有未知鄰居都是雷
                    inferred_mines.update(unknown)
                    changed = True

        return inferred_safe, inferred_mines

    # ---------- 內部方法 ----------

    def _place_mines(self, exclude_row: int, exclude_col: int):
        """放地雷，排除第一次點擊的周圍 3x3。"""
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
        """計算某格周圍的地雷數量。"""
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
        """翻開一格（遞迴展開空白區域）。"""
        if (row, col) in self.revealed or (row, col) in self.flags:
            return

        self.revealed.add((row, col))

        # 踩雷
        if (row, col) in self.mines:
            newly_revealed.append((row, col, -1))  # -1 表示地雷
            return

        num = self._count_adjacent_mines(row, col)
        newly_revealed.append((row, col, num))

        # 如果是空白格 (數字 0)，遞迴翻開周圍
        if num == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        self._reveal_cell(r, c, newly_revealed)

    def _check_win(self) -> bool:
        """檢查是否贏了（所有非地雷格都翻開了）。"""
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.mines and (r, c) not in self.revealed:
                    return False
        return True
