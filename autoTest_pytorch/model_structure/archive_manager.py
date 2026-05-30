"""Session + hour-rollover archive 資料夾管理。

被 transformer_discrete_agent.py(stage1)與 visual_discrete_agent_v3.py(stage2)
共用。把「每個訓練 session 自己一個 `training_<ts>/` 資料夾,每 wall-clock 小時
翻一個 `hour_NN_<ts>/` 子資料夾」這個 pattern 抽出來,讓兩個 agent 不用各自維護
session_start / hour_index / current_archive_dir 一堆狀態。

產生的資料夾結構:
    <model_path>/
        training_<session_ts>/          ← session_dir(hyperparameters.txt 寫這層)
            hour_00_<hour_ts>/          ← current_archive_dir
                train_io_log.txt
                training_log.csv
                <每次 save 時的 *.pth 快照>
            hour_01_<hour_ts>/
            ...

訂閱者(io_log 等)透過 ``register_on_rollover`` 註冊 callback。
每次翻頁時 manager 會把新的 ``current_archive_dir`` 傳給每個 callback,訂閱者各自
負責關掉舊 handle、開新 handle。

提供兩個 building block:
    SessionArchiveManager   ── 目錄管理本身
    RolloverTextLog         ── plain-text append 檔,支援 swap_to(new_path)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable, TextIO


# 翻頁間隔(秒)。預設 wall-clock 1 小時。
DEFAULT_ROLLOVER_SECONDS = 3600


class SessionArchiveManager:
    """每個訓練 session 一個目錄,每小時翻一個子目錄。

    Args:
        model_path: agent 的 checkpoint 根目錄(例如 ``./models/stage1_transformer``)。
        log_prefix: 印 console 訊息用的 prefix(例如 ``"[FQF]"``)。
        rollover_seconds: 翻頁間隔。預設 ``DEFAULT_ROLLOVER_SECONDS``(1 小時)。
            測試時可以調小強制觸發。

    Attributes:
        session_start: ``datetime.datetime`` —— session 起始時間。
        session_dir: ``Path`` —— ``model_path / training_<ts>/``。
        current_archive_dir: ``Path`` —— 當下小時的子資料夾。每次 ``maybe_rollover``
            翻頁後會指到新位置。
        hour_index: ``int`` —— 目前是第幾個小時(從 0 起)。
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

    # ── public properties ────────────────────────────────────────────
    @property
    def hour_index(self) -> int:
        return self._hour_index

    @property
    def hour_start(self) -> datetime.datetime:
        return self._hour_start

    # ── internal helpers ─────────────────────────────────────────────
    def _make_hour_dir(self, idx: int, start_dt: datetime.datetime) -> Path:
        ts = start_dt.strftime("%Y%m%d_%H%M%S")
        return self.session_dir / f"hour_{idx:02d}_{ts}"

    # ── public API ───────────────────────────────────────────────────
    def register_on_rollover(self, callback: Callable[[Path], None]) -> None:
        """註冊 hour rollover callback。

        每次 ``maybe_rollover()`` 翻頁成功時,所有註冊的 callback 都會被叫到一次,
        參數是新的 ``current_archive_dir``。順序依註冊順序。Callback 內部丟例外
        不會擋其他 callback;只會 console print。
        """
        self._on_rollover.append(callback)

    def maybe_rollover(self) -> bool:
        """Wall-clock check;若 ``hour_start`` 已超過 ``rollover_seconds`` 就翻頁。

        Returns:
            ``True`` 表示有翻頁(callbacks 都已執行);``False`` 表示時間未到。
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
                # 單一 callback 壞掉不擋其他 callback。
                print(f"{self.log_prefix} on_rollover callback {cb!r} failed: {exc}")

        print(f"{self.log_prefix} Hour rollover -> {self.current_archive_dir}")
        return True

    def find_latest_archive(self, filename: str) -> Path | None:
        """掃所有 ``training_*/hour_*/`` 找最新一份 ``filename``;不存在回 None。

        排序鍵是 hour 資料夾名稱裡的 timestamp(save 時嵌進去的,代表 hour 起始
        時間),不靠 mtime —— 後者會被 git checkout / cp 動到,不可靠。

        用途:checkpoint canonical 不存在時 fallback 到 archive 目錄找最新快照。
        """
        candidates: list[tuple[str, Path]] = []
        for session_dir in self.model_path.glob("training_*"):
            if not session_dir.is_dir():
                continue
            for hour_dir in session_dir.glob("hour_*"):
                if not hour_dir.is_dir():
                    continue
                # name 格式:hour_NN_YYYYMMDD_HHMMSS,timestamp 從第 3 段開始
                parts = hour_dir.name.split("_", 2)
                if len(parts) < 3:
                    continue
                ts_str = parts[2]  # "YYYYMMDD_HHMMSS",字典序 = 時間序
                path = hour_dir / filename
                if path.exists():
                    candidates.append((ts_str, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


class RolloverTextLog:
    """Plain-text append 檔,支援 hour rollover 時 swap 到新路徑。

    使用流程:
        1) ``log = RolloverTextLog(banner_factory=lambda p: [...])``
        2) 在 agent ``__init__`` 結尾 ``log.swap_to(archive.current_archive_dir / "train_io_log.txt")``
        3) 註冊 rollover callback ``archive.register_on_rollover(
               lambda d: log.swap_to(d / "train_io_log.txt"))``
        4) 訓練熱迴路內直接 ``log.write(...)``、``log.flush()``
        5) atexit 用 ``log.close()`` 關掉
        6) 翻頁前若想留 footer,自己在 callback 內先 ``log.write("--- rollover ---\\n")``
           再 ``log.swap_to(new_path)``

    Lazy-open:必須先呼叫一次 ``swap_to`` 才會建檔。在那之前 ``write`` 是 no-op。
    """

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
        """關掉舊 handle、在新路徑開新 handle、寫 banner。"""
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
