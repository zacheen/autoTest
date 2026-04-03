"""
共用工具模組 — 被多個訓練腳本共用。

包含：TeeOutput, CSVLogger, compute_reward, action_to_grid
"""

import sys
import csv
import datetime
import numpy as np
from pathlib import Path


# ============================================================
# Logging
# ============================================================

class TeeOutput:
    """同時輸出到 console 和 txt 檔案。"""

    def __init__(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, 'a', encoding='utf-8')
        self.stdout = sys.stdout
        self.file.write(f"\n{'='*60}\n")
        self.file.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self.file.write(f"{'='*60}\n")
        self.file.flush()

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


class CSVLogger:
    """CSV 格式的訓練日誌。"""

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


# ============================================================
# Reward
# ============================================================

def compute_reward(result):
    """計算 reward，直接回傳 [-1, +1] 範圍的值。

    Args:
        result: MinesweeperLogic.ClickResult

    Returns:
        float: reward
            +1.0  WIN
            +0.3  有效點擊（翻開新格子）
            -1.0  踩雷
            -0.5  無效點擊（點已翻開格）
    """
    if not result.changed:
        return -0.5
    if result.win:
        return 1.0
    if result.game_over:
        return -1.0
    return 0.3


# ============================================================
# Action
# ============================================================

def action_to_grid(action, rows, cols):
    """Continuous (x, y) ∈ [0, 1]² → discrete (row, col).

    Args:
        action: numpy array (2,) with values in [0, 1]
        rows: grid 列數
        cols: grid 行數

    Returns:
        (row, col) tuple of ints
    """
    col = int(np.clip(action[0] * cols, 0, cols - 1))
    row = int(np.clip(action[1] * rows, 0, rows - 1))
    return row, col
