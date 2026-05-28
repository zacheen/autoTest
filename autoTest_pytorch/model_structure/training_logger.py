"""Episode-level metric dispatcher:同一份 metric 同時寫 CSV + TensorBoard。

設計拆分:
    CSVLogger        ── 低階 CSV append + hour rollover 的檔案 handle 管理
    TrainingLogger   ── 高階:接 SummaryWriter + CSVLogger,提供 log() / commit_csv_row()

職責邊界(跟 agent.tb_writer 分工):
    - TrainingLogger 只處理 **episode-level summary**(每 episode 1 列 CSV、
      若干個 TB scalar),由 train script / agent.log_episode_metrics 觸發。
    - 高頻 per-step diagnostic(grad/*、buffer/*、fpn/*、weight_norm/* …)仍走
      agent.tb_writer.add_scalar() 直接寫,不繞 TrainingLogger ── 那些東西每場
      episode 會產生上百筆 step-axis 的點,塞進 CSV 一列一列會炸資料量。

API:
    logger = TrainingLogger(
        csv_path=archive.current_archive_dir / "training_log.csv",
        csv_fields=["timestamp", "episode", "reward", ...],
        tb_writer=agent.tb_writer,
    )

    # 一個 metric → 兩個 flag 決定目的地
    logger.log("episode/reward_sum", stats["reward"], step=ep, csv_col="reward")
    logger.log("episode/steps",      stats["steps"],  step=ep, csv_col="steps")
    logger.log("train/total_it",     agent.total_it,  step=ep, csv=False)  # TB only
    logger.log("timestamp",          iso_str,         step=ep, tb=False)   # CSV only

    # 一場 episode 全部 metric log 完才 flush 成一列
    logger.commit_csv_row()

    # Hour rollover hook(archive_manager 翻頁時 call)
    archive.register_on_rollover(
        lambda d: logger.swap_csv_to(d / "training_log.csv")
    )

設計取捨:
    1. CSV schema 預先宣告 ── headers 開檔時就定,unexpected col 立刻 KeyError,
       比 silent drop / DictWriter ValueError 好除錯。
    2. CSV 存 raw value ── int / float / bool 直接寫,pandas / DuckDB 讀進來
       自動拿到正確 dtype。不做 f-string 格式化(format 是給人看的,跟 CSV 給
       AI 讀的目的衝突)。
    3. Tag vs col 名 ── 預設同名(`log(name=...)` 同時當 TB tag 與 CSV col)。
       TB 慣例用 slash 分群(`episode/reward_sum`)而 CSV col 通常想要 flat
       (`reward`),這時用 keyword `csv_col="reward"` override。
    4. 顯式 commit ── log() 只填 buffer,不寫檔;commit_csv_row() 才真寫一列。
       一場 episode 多個 metric 才能合成一列,auto-commit 會把每個 metric 都
       變成獨立稀疏列。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class CSVLogger:
    """CSV append + hour rollover swap。

    使用流程:
        csv_logger = CSVLogger(initial_path)
        csv_logger.open(fieldnames=[...])     # 寫 header(若是空檔)
        csv_logger.write({...})                # 寫一列
        csv_logger.swap_to(new_path)           # 翻檔到新路徑(同 fieldnames、寫 header)
        csv_logger.close()
    """

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
        """Hour rollover:關掉目前的檔,在新路徑開新檔(同 fieldnames、寫 header)。

        Returns:
            True 如果真的翻到新檔案;False 如果 new_path 等於目前 path(no-op)。
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
    """Episode-level metric dispatcher 兩個目的地:CSV + TensorBoard。

    Args:
        csv_path: 初始 CSV 路徑(會用 archive.current_archive_dir / "training_log.csv")
        csv_fields: **預先宣告** 的 CSV 欄位清單。log(csv=True) 時 csv_col 必須
            在這個清單裡,否則 KeyError。
        tb_writer: 既有的 SummaryWriter 實例(跟 agent.tb_writer 共用,讓 CSV
            紀錄跟 agent 自己的高頻 TB 寫入落在同一個 TB run)。

    使用模式:
        logger = TrainingLogger(csv_path, csv_fields, tb_writer)
        # ... 訓練迴圈 ...
        logger.log(tag, value, step=ep, tb=True, csv=True, csv_col=None)
        logger.commit_csv_row()
        # 翻頁
        archive.register_on_rollover(lambda d: logger.swap_csv_to(d / "training_log.csv"))
        # 關檔
        logger.close()  # 或丟給 atexit
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
        # 緩衝目前 episode 的 csv=True 欄位,commit_csv_row 才真寫一列。
        self._pending_row: dict[str, Any] = {}

    @property
    def csv_path(self) -> Path:
        return self._csv.path

    @property
    def csv_fields(self) -> list[str]:
        # 回 list(原順序)而不是 set,給 caller 可以印 / 比對。
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
        """寫一個 metric,兩個 bool 決定目的地。

        Args:
            name: TB tag(若 csv_col=None,也當 CSV col 名)。
            value: 數值(int / float / bool / str)。CSV 端存 raw value 不做格式化。
            step: TB scalar 的 step 軸(通常是 episode_count)。`csv=True` 但
                `tb=False` 時 step 可以填任意值(不會被用到)。
            tb: True 就 add_scalar(name, value, step) 到 TB。
            csv: True 就把 value 暫存到 row buffer(commit_csv_row 時寫)。
            csv_col: CSV col 名 override(預設 = name)。當 TB tag 含 slash 而
                CSV 想要 flat 名稱時用,例如 `name="episode/reward_sum"` 配
                `csv_col="reward"`。

        Raises:
            KeyError: csv=True 但 csv_col(或 name)不在 ctor 傳的 csv_fields 內。
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
        """把 pending buffer 寫成一列 CSV。沒填到的欄位寫空(pandas 讀進來成 NaN)。

        Buffer 完成後清空,下一場 episode 重新累積。
        """
        # 用 None 填未提供的欄位 → csv.DictWriter 寫成空字串 → pandas 讀成 NaN。
        row = {col: self._pending_row.get(col, None) for col in self._csv.writer.fieldnames}
        self._csv.write(row)
        self._pending_row = {}

    def discard_pending(self) -> None:
        """丟掉目前 buffer(不 commit)。用於 episode 中途要 reset 紀錄的場合。"""
        self._pending_row = {}

    def swap_csv_to(self, new_path: Path) -> bool:
        """Hour rollover hook:CSV 翻到新路徑。TB writer 不動(同一個 run)。"""
        return self._csv.swap_to(new_path)

    # ── TB delegations ──────────────────────────────────────────────
    # 抽出來讓 agent 完全不需要直接握 SummaryWriter handle。所有 TB 寫入 /
    # flush / close / add_text 都從 logger 出口走,單一介面。

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """寫一段 TB add_text(metric docs 之類用)。預設 step=0,只寫一次的場合用。"""
        self._tb.add_text(tag, text, step)

    def flush(self) -> None:
        """強制 flush TB events 到磁碟。Save model / checkpoint 前可以 call,確保
        on-disk 的 TB 事件跟 model state 對齊。"""
        self._tb.flush()

    def close(self) -> None:
        """同時關 CSV 與 SummaryWriter。

        Agent atexit 註冊這個一個就好,不用各自 close CSV + TB。close() 為
        idempotent(closed 後再 call 不會錯)。
        """
        self._csv.close()
        try:
            self._tb.close()
        except Exception:
            pass

    def close_csv(self) -> None:
        """只關 CSV,SummaryWriter 仍可寫(資源分階段關場景用)。"""
        self._csv.close()

    @property
    def tb_writer(self) -> SummaryWriter:
        """Underlying SummaryWriter 的 escape hatch。

        對於 scalars 請用 log() / 對於 text 用 log_text() / flush 用 flush()。
        這個 property 只給 TrainingLogger 還沒抽到 method 的進階 ops 用
        (add_histogram / add_image 等)。新 code 應該優先用上面那些 method。
        """
        return self._tb

    @property
    def closed(self) -> bool:
        """CSV file handle 狀態。SummaryWriter 的 close 不暴露 closed flag,所以
        這 property 仍以 CSV 為主 — 跟 RolloverTextLog 介面對齊。"""
        return self._csv.closed
