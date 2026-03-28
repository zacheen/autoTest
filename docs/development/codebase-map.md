# Codebase Map

## Directory Structure

```
autoTest_clau/
├── CLAUDE.md                          # Claude Code guidance
├── .gitignore
├── README.txt                         # Original readme (legacy)
├── requirements.txt                   # Python dependencies
├── env_note.txt                       # Environment notes
├── TF安裝步驟.txt                      # TensorFlow install steps (legacy)
├── 自動化檔案結構(不包括辨識).txt        # File structure doc (without detection)
├── 自動化檔案結構(包括辨識).txt          # File structure doc (with detection)
├── 自動化電腦設定前置作業.txt            # Computer setup prerequisites
│
├── autoTest_pytorch/                  # *** Active source code ***
│   ├── Demo_test_Minesweeper.py       # Main entry point — game loop, test cases, RL orchestration
│   ├── RL_Agent.py                    # TD3 agent (Actor, Critic, ReplayBuffer, training logic)
│   ├── Tool_Main.py                   # Core utilities — screenshot, comparison, clicking, game state
│   ├── HTMLTestRun.py                 # HTML test report generator
│   ├── Gf_Except.py                   # Custom exception (Game_fail_Exception)
│   ├── Data.py                        # Label/class name mappings for detection
│   ├── identify_for_import.py         # Object detection inference utilities (legacy TF)
│   ├── Object_detection_image.py      # TensorFlow object detection demo (legacy)
│   ├── train.py                       # TensorFlow training script (legacy)
│   └── Minesweeper/
│       ├── Minesweeper.py             # Tkinter Minesweeper game implementation
│       └── Minesweeper_manager.py     # Process manager (start/stop game in subprocess)
│
├── user_change/                       # User-configurable files
│   ├── Minesweeper_input.txt          # Game settings input file
│   ├── chromedriver.exe               # ChromeDriver for Selenium (legacy web games)
│   └── game_pic/
│       └── Minesweeper_pic/           # Template images for screenshot comparison
│           ├── grid_region.txt         # Grid region coordinates
│           ├── buttons.txt            # Button region coordinates
│           ├── win.txt / lose.txt     # Win/lose detection templates
│           ├── new_game.txt           # New game button template
│           ├── confirm.txt            # Confirmation dialog template
│           └── ...
│
├── testreport/                        # Generated HTML test reports
├── docs/                              # Project documentation
├── 自動化教學/                          # Tutorial materials (Chinese)
└── 工作紀錄/                            # Work logs (Chinese)
```

## Key File Details

### `Demo_test_Minesweeper.py` (Entry Point)

- Contains `Game_test_case` (unittest.TestCase) with the full game lifecycle
- `Game_status` inner class holds per-round state (agent, screenshots, rewards, step count)
- `decide_next_step_and_play()` — core RL loop: capture → preprocess → act → click
- `update_model()` — store transition and trigger training
- `test_RL()` — outer RL loop with win/lose/timeout detection
- Main block starts Minesweeper process, initializes globals, runs infinite game loop

### `RL_Agent.py` (RL Agent)

- `TD3Agent` — full TD3 implementation with Actor, twin Critics, target networks
- `ReplayBuffer` — simple list-based circular buffer
- `select_action()` — forward pass + optional Gaussian noise
- `action_to_screen_coords()` — maps [-1,1] to pixel coordinates
- `log_action_image()` — saves annotated screenshots for debugging
- `get_agent()` — singleton pattern for agent access

### `Tool_Main.py` (Utilities)

- `Glo_var` — global state container (game name, round count, file handles, timing)
- `compare_sim()` — template matching via screenshot comparison
- `cut_pic_data()` — capture and save game region screenshots
- `click()` / `click_mid()` — mouse click execution with region validation
- `cal_time_out()` — timeout checking for game steps
- Reads config from `user_change/` directory

### `Minesweeper.py` (Game)

- Full Minesweeper implementation in tkinter
- Beginner/Intermediate/Expert difficulty levels
- Runs in a separate process via `Minesweeper_manager`

## Legacy Files (Not Active)

| File | Original Purpose |
|------|-----------------|
| `train.py` | TensorFlow object detection model training |
| `Object_detection_image.py` | TensorFlow inference demo |
| `identify_for_import.py` | Detection result classification |
| `Data.py` | Label mappings for card/game detection |
