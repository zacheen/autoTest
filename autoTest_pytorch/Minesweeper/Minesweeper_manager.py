import subprocess
import sys
import time
import signal
from pathlib import Path


class Minesweeper_manager:
    def __init__(self):
        self.process = None

    def thread_start(self):
        # Clean up existing process if it exists but is dead
        if self.process and self.process.poll() is not None:
            self.process = None

        if self.process is None:
            # Launch Minesweeper.py as a completely separate subprocess
            # This avoids Windows multiprocessing spawn re-importing the caller
            minesweeper_script = Path(__file__).parent / "Minesweeper.py"
            self.process = subprocess.Popen(
                [sys.executable, str(minesweeper_script)],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            print(f"Minesweeper started (PID: {self.process.pid})")

    def thread_stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None


if __name__ == "__main__":
    mine = Minesweeper_manager()
    mine.thread_start()
    time.sleep(100)
    mine.thread_stop()
