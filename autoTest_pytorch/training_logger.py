"""
訓練日誌工具 — TeeOutput 和 CSVLogger。

可被 train_stage1.py、Demo_test_Minesweeper.py 或任何訓練腳本共用。

使用方式：
    from training_logger import TeeOutput, CSVLogger

    # TeeOutput: 同時輸出到 console 和 txt
    tee = TeeOutput(Path("./logs/output.txt"))
    sys.stdout = tee
    print("hello")  # 會同時寫到 console 和檔案
    tee.close()      # 還原 sys.stdout

    # CSVLogger: 每 episode 寫一行到 CSV
    csv_log = CSVLogger(Path("./logs/training.csv"))
    csv_log.open(['episode', 'reward', 'win'])
    csv_log.write({'episode': 1, 'reward': 5.0, 'win': 0})
    csv_log.close()
"""

import sys
import csv
import datetime
from pathlib import Path


class TeeOutput:
    """同時輸出到 console 和 txt 檔案。

    用法：
        tee = TeeOutput(Path("output.txt"))
        sys.stdout = tee
        # ... 所有 print() 會同時寫到 console 和檔案 ...
        tee.close()  # 還原 sys.stdout
    """

    def __init__(self, filepath):
        """
        Args:
            filepath: txt 輸出檔案路徑（Path 或 str），append 模式
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, 'a', encoding='utf-8')
        self.stdout = sys.stdout
        # 寫入分隔線標記新的 session
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
        """還原 sys.stdout 並關閉檔案。"""
        sys.stdout = self.stdout
        self.file.close()


class CSVLogger:
    """CSV 格式的訓練日誌。

    用法：
        logger = CSVLogger(Path("training.csv"))
        logger.open(['episode', 'reward'])
        logger.write({'episode': 1, 'reward': 5.0})
        logger.close()
    """

    def __init__(self, path):
        """
        Args:
            path: CSV 輸出檔案路徑（Path 或 str），append 模式
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.writer = None

    def open(self, fieldnames):
        """開啟 CSV 並寫 header（如果檔案不存在或為空）。"""
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self.file = open(self.path, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def write(self, row):
        """寫一行資料。"""
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
