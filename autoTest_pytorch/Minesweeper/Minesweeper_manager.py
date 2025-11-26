import tkinter as tk
from Minesweeper import Minesweeper
from multiprocessing import Process, Event
import time

def run_game(stop_event):
    """Standalone function to run the game in a separate process."""
    root = tk.Tk()
    game = Minesweeper(root, stop_event)
    root.mainloop()

class Minesweeper_manager:
    def __init__(self):
        self.process = None
        self.stop_event = None
    
    def thread_start(self):
        # Clean up existing process if it exists but is dead
        if self.process and not self.process.is_alive():
            self.process.join()
            self.process = None

        if self.process is None or not self.process.is_alive():
            self.stop_event = Event()
            self.process = Process(target=run_game, args=(self.stop_event,))
            self.process.start()
    
    def thread_stop(self):
        if self.stop_event:
            self.stop_event.set()
        
        if self.process:
            # Wait for the process to finish gracefully
            self.process.join(timeout=3)
            # If it's still alive, force terminate
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self.process = None

if __name__ == "__main__":
    mine = Minesweeper_manager()
    mine.thread_start()
    time.sleep(10)
    mine.thread_stop()
