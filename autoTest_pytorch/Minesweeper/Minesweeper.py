import tkinter as tk
from tkinter import messagebox
import random
import time

class Minesweeper:
    def __init__(self, root):
        self.root = root
        self.root.title("Minesweeper 掃雷")
        self.root.configure(bg='#f0f0f0')
        self.root.minsize(400, 450)
        
        # Configure root grid weights for expansion
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Game parameters
        self.difficulties = {
            'Beginner': {'rows': 9, 'cols': 9, 'mines': 10},
            'Intermediate': {'rows': 16, 'cols': 16, 'mines': 40},
            'Expert': {'rows': 16, 'cols': 30, 'mines': 99}
        }
        
        self.current_difficulty = 'Beginner'
        self.setup_game()
        
    def setup_game(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Get game parameters
        params = self.difficulties[self.current_difficulty]
        self.rows = params['rows']
        self.cols = params['cols']
        self.mines_count = params['mines']
        
        # Game state
        self.mines = set()
        self.revealed = set()
        self.flags = set()
        self.game_over = False
        self.first_click = True
        self.start_time = None
        self.remaining_mines = self.mines_count
        
        # Create menu bar
        self.create_menu()
        
        # Create info panel
        self.create_info_panel()
        
        # Create game board
        self.create_board()
        
        # Update timer
        # self.update_timer()
        
    def create_menu(self):
        menu_frame = tk.Frame(self.root, bg='#f0f0f0')
        menu_frame.grid(row=0, column=0, sticky='ew', pady=5)
        
        # Configure menu frame to expand horizontally
        menu_frame.grid_columnconfigure(0, weight=1)
        
        # Create inner frame for centering buttons
        button_container = tk.Frame(menu_frame, bg='#f0f0f0')
        button_container.grid(row=0, column=0)
        
        # New game button
        new_game_btn = tk.Button(
            button_container, 
            text="New Game 新遊戲", 
            command=self.setup_game,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        new_game_btn.pack(side=tk.LEFT, padx=5)
        
        # Difficulty buttons
        for difficulty in self.difficulties.keys():
            btn = tk.Button(
                button_container,
                text=difficulty,
                command=lambda d=difficulty: self.change_difficulty(d),
                bg='#2196F3' if difficulty == self.current_difficulty else '#e0e0e0',
                fg='white' if difficulty == self.current_difficulty else 'black',
                font=('Arial', 10),
                padx=10,
                pady=5
            )
            btn.pack(side=tk.LEFT, padx=2)
            
    def create_info_panel(self):
        info_frame = tk.Frame(self.root, bg='#f0f0f0')
        info_frame.grid(row=1, column=0, sticky='ew', pady=10)
        
        # Configure for centering
        info_frame.grid_columnconfigure(0, weight=1)
        
        # Create inner container for centering
        info_container = tk.Frame(info_frame, bg='#f0f0f0')
        info_container.grid(row=0, column=0)
        
        # Mine counter
        mine_frame = tk.Frame(info_container, bg='#333', padx=10, pady=5)
        mine_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(mine_frame, text="Mines 地雷:", bg='#333', fg='white', font=('Arial', 12)).pack(side=tk.LEFT)
        self.mine_label = tk.Label(mine_frame, text=str(self.remaining_mines), bg='#333', fg='#ff0000', 
                                  font=('Digital', 16, 'bold'))
        self.mine_label.pack(side=tk.LEFT, padx=5)
        
        # Timer
        timer_frame = tk.Frame(info_container, bg='#333', padx=10, pady=5)
        timer_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(timer_frame, text="Time 時間:", bg='#333', fg='white', font=('Arial', 12)).pack(side=tk.LEFT)
        self.timer_label = tk.Label(timer_frame, text="000", bg='#333', fg='#00ff00', 
                                   font=('Digital', 16, 'bold'))
        self.timer_label.pack(side=tk.LEFT, padx=5)
        
    def create_board(self):
        # Create main container frame that will expand
        self.main_board_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.main_board_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=(0, 10))
        
        # Configure the main board frame to expand
        self.main_board_frame.grid_rowconfigure(0, weight=1)
        self.main_board_frame.grid_columnconfigure(0, weight=1)
        
        # Create frame for the game board with border
        self.board_frame = tk.Frame(self.main_board_frame, bg='#d0d0d0', relief=tk.SUNKEN, bd=3)
        
        # Create buttons grid
        self.buttons = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                btn = tk.Button(
                    self.board_frame,
                    width=3,
                    height=1,
                    font=('Arial', 10, 'bold'),
                    bg='#e0e0e0',
                    relief=tk.RAISED,
                    bd=2,
                    padx=0,
                    pady=0
                )
                btn.grid(row=i, column=j, padx=1, pady=1, sticky='nsew')
                btn.bind('<Button-1>', lambda e, r=i, c=j: self.on_left_click(r, c))
                btn.bind('<Button-3>', lambda e, r=i, c=j: self.on_right_click(r, c))
                
                # Configure grid weights for each cell
                self.board_frame.grid_rowconfigure(i, weight=1)
                self.board_frame.grid_columnconfigure(j, weight=1)
                
                row.append(btn)
            self.buttons.append(row)
        
        # Place the board frame and bind resize
        self.update_board_size()
        self.main_board_frame.bind('<Configure>', lambda e: self.update_board_size())
        
    def update_board_size(self):
        """Update the board size to maintain square cells and center it"""
        # Allow GUI to update
        self.main_board_frame.update_idletasks()
        
        # Get available space
        available_width = self.main_board_frame.winfo_width()
        available_height = self.main_board_frame.winfo_height()
        
        if available_width <= 1 or available_height <= 1:
            # Schedule another update if dimensions not ready
            self.root.after(10, self.update_board_size)
            return
        
        # Calculate cell size to maintain square cells
        # Account for borders, padding between cells, and frame border
        padding_width = (self.cols + 1) * 2 + 10  # padding between cells + border
        padding_height = (self.rows + 1) * 2 + 10
        
        max_cell_width = (available_width - padding_width) / self.cols
        max_cell_height = (available_height - padding_height) / self.rows
        
        # Use the smaller dimension to maintain square cells
        cell_size = min(max_cell_width, max_cell_height)
        cell_size = max(20, min(cell_size, 45))  # Limit between 20 and 45 pixels
        
        # Calculate actual board dimensions
        board_width = int(cell_size * self.cols + padding_width)
        board_height = int(cell_size * self.rows + padding_height)
        
        # Update button sizes
        button_width = max(2, int(cell_size / 8))
        button_height = max(1, int(cell_size / 16))
        font_size = max(8, min(14, int(cell_size * 0.4)))
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.buttons[i][j].config(
                    width=button_width,
                    height=button_height,
                    font=('Arial', font_size, 'bold')
                )
        
        # Center the board frame
        # First, place it without centering to get actual size
        self.board_frame.place_forget()
        self.board_frame.place(
            x=(available_width - board_width) // 2,
            y=(available_height - board_height) // 2,
            width=board_width,
            height=board_height
        )
        
    def change_difficulty(self, difficulty):
        self.current_difficulty = difficulty
        self.setup_game()
        
    def place_mines(self, exclude_row, exclude_col):
        # Place mines randomly, excluding the first clicked cell and its neighbors
        exclude = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = exclude_row + dr, exclude_col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    exclude.add((r, c))
                    
        available = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                    if (r, c) not in exclude]
        
        self.mines = set(random.sample(available, min(self.mines_count, len(available))))
        
    def count_adjacent_mines(self, row, col):
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols and (r, c) in self.mines:
                    count += 1
        return count
        
    def on_left_click(self, row, col):
        if self.game_over or (row, col) in self.flags:
            return
            
        if self.first_click:
            self.first_click = False
            self.start_time = time.time()
            self.place_mines(row, col)
            
        self.reveal_cell(row, col)
        
        # Check win condition
        if self.check_win():
            self.game_won()
            
    def reveal_cell(self, row, col):
        if (row, col) in self.revealed or (row, col) in self.flags:
            return
            
        self.revealed.add((row, col))
        
        if (row, col) in self.mines:
            self.game_lost(row, col)
            return
            
        # Count adjacent mines
        adjacent_mines = self.count_adjacent_mines(row, col)
        
        # Update button appearance
        btn = self.buttons[row][col]
        btn.config(relief=tk.SUNKEN, bg='#ffffff')
        
        if adjacent_mines > 0:
            colors = {
                1: '#0000ff', 2: '#008000', 3: '#ff0000', 4: '#000080',
                5: '#800000', 6: '#008080', 7: '#000000', 8: '#808080'
            }
            btn.config(text=str(adjacent_mines), fg=colors.get(adjacent_mines, '#000000'))
        else:
            # Auto-reveal adjacent cells if no mines nearby
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        self.reveal_cell(r, c)
                        
    def on_right_click(self, row, col):
        if self.game_over or (row, col) in self.revealed:
            return
            
        btn = self.buttons[row][col]
        
        if (row, col) in self.flags:
            self.flags.remove((row, col))
            btn.config(text='', bg='#e0e0e0')
            self.remaining_mines += 1
        else:
            self.flags.add((row, col))
            btn.config(text='🚩', fg='red', bg='#ffff99')
            self.remaining_mines -= 1
            
        self.mine_label.config(text=str(max(0, self.remaining_mines)))
        
        # Check win condition
        if self.check_win():
            self.game_won()
            
    def check_win(self):
        # Win if all non-mine cells are revealed
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in self.mines and (i, j) not in self.revealed:
                    return False
        return True
        
    def game_lost(self, clicked_row, clicked_col):
        self.game_over = True
        
        # Show all mines
        for row, col in self.mines:
            btn = self.buttons[row][col]
            if (row, col) == (clicked_row, clicked_col):
                btn.config(text='💣', bg='#ff0000', fg='black')
            elif (row, col) in self.flags:
                btn.config(text='🚩', bg='#90EE90')  # Correctly flagged
            else:
                btn.config(text='💣', bg='#ffcccc', fg='black')
                
        # Show incorrectly flagged cells
        for row, col in self.flags:
            if (row, col) not in self.mines:
                self.buttons[row][col].config(text='❌', bg='#ffff99')
                
        messagebox.showinfo("Game Over", "You hit a mine! 你踩到地雷了！")
        
    def game_won(self):
        self.game_over = True
        
        elapsed_time = int(time.time() - self.start_time) if self.start_time else 0
        
        # Mark all mines as flagged
        for row, col in self.mines:
            self.buttons[row][col].config(text='🚩', bg='#90EE90', fg='red')
            
        messagebox.showinfo("Congratulations! 恭喜！", 
                          f"You won! 你贏了！\nTime: {elapsed_time} seconds")
        
    def update_timer(self):
        if not self.game_over and self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.timer_label.config(text=f"{elapsed:03d}")
            
        self.root.after(1000, self.update_timer)

def main():
    root = tk.Tk()
    game = Minesweeper(root)
    root.state('zoomed')
    root.mainloop()

def thread_start():
    from threading import Thread
    Thread( target=main ).start()

if __name__ == "__main__":
    thread_start()
