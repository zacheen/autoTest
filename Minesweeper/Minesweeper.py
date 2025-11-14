import tkinter as tk
from tkinter import messagebox
import random
import time

class Minesweeper:
    def __init__(self, root):
        self.root = root
        self.root.title("Minesweeper 掃雷")
        self.root.configure(bg='#f0f0f0')
        
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
        self.update_timer()
        
    def create_menu(self):
        menu_frame = tk.Frame(self.root, bg='#f0f0f0')
        menu_frame.pack(pady=5)
        
        # New game button
        new_game_btn = tk.Button(
            menu_frame, 
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
                menu_frame,
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
        info_frame.pack(pady=10)
        
        # Mine counter
        mine_frame = tk.Frame(info_frame, bg='#333', padx=10, pady=5)
        mine_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(mine_frame, text="Mines 地雷:", bg='#333', fg='white', font=('Arial', 12)).pack(side=tk.LEFT)
        self.mine_label = tk.Label(mine_frame, text=str(self.remaining_mines), bg='#333', fg='#ff0000', 
                                  font=('Digital', 16, 'bold'))
        self.mine_label.pack(side=tk.LEFT, padx=5)
        
        # Timer
        timer_frame = tk.Frame(info_frame, bg='#333', padx=10, pady=5)
        timer_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(timer_frame, text="Time 時間:", bg='#333', fg='white', font=('Arial', 12)).pack(side=tk.LEFT)
        self.timer_label = tk.Label(timer_frame, text="000", bg='#333', fg='#00ff00', 
                                   font=('Digital', 16, 'bold'))
        self.timer_label.pack(side=tk.LEFT, padx=5)
        
    def create_board(self):
        # Create frame for the game board
        board_frame = tk.Frame(self.root, bg='#d0d0d0', relief=tk.SUNKEN, bd=3)
        board_frame.pack(padx=10, pady=10)
        
        # Create buttons grid
        self.buttons = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                btn = tk.Button(
                    board_frame,
                    width=2,
                    height=1,
                    font=('Arial', 10, 'bold'),
                    bg='#e0e0e0',
                    relief=tk.RAISED,
                    bd=2
                )
                btn.grid(row=i, column=j, padx=1, pady=1)
                btn.bind('<Button-1>', lambda e, r=i, c=j: self.on_left_click(r, c))
                btn.bind('<Button-3>', lambda e, r=i, c=j: self.on_right_click(r, c))
                row.append(btn)
            self.buttons.append(row)
            
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
    root.mainloop()

if __name__ == "__main__":
    main()