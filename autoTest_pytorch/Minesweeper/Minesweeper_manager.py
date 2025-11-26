import tkinter as tk
from Minesweeper.Minesweeper import Minesweeper

class Minesweeper_manager:
    def start_game(self, stop_event):
        self.root = tk.Tk()
        self.game = Minesweeper(self.root, stop_event)
        self.root.mainloop()

    def thread_start(self):
        from threading import Thread, Event
        self.stop_event = Event()
        self.thread = Thread( target = lambda : self.start_game(self.stop_event) )
        self.thread.start()
    
    def thread_stop(self):
        self.stop_event.set()
        self.thread.join()

if __name__ == "__main__":
    mine = Minesweeper_manager()
    mine.thread_start()
    mine.thread_stop()
