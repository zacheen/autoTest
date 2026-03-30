# Codebase Map

## Directory Structure

```
autoTest_clau/
├── CLAUDE.md                          # Claude Code guidance
├── .gitignore
├── README.txt                         # Original readme (legacy)
├── requirements.txt                   # Python dependencies
├── env_note.txt                       # Environment notes
├── yolo11n.pt                         # YOLO11n pretrained weights (COCO)
├── TF安裝步驟.txt                      # TensorFlow install steps (legacy)
├── 自動化檔案結構(不包括辨識).txt        # File structure doc (without detection)
├── 自動化檔案結構(包括辨識).txt          # File structure doc (with detection)
├── 自動化電腦設定前置作業.txt            # Computer setup prerequisites
│
├── autoTest_pytorch/                  # *** Active source code ***
│   ├── Demo_test_Minesweeper.py       # Main entry point — game loop, test cases, RL orchestration
│   ├── RL_Agent.py                    # All RL agents and networks (SAC, Stage 1, Simple MLP)
│   ├── train_stage1.py                # Stage 1 pre-training script (GridEncoder + Attention + SAC)
│   ├── train_stage1_simple.py         # Simple MLP experiment (validates RL pipeline correctness)
│   ├── training_logger.py             # Shared logging utilities (TeeOutput, CSVLogger)
│   ├── Tool_Main.py                   # Core utilities — screenshot, comparison, clicking, game state
│   ├── HTMLTestRun.py                 # HTML test report generator
│   ├── Gf_Except.py                   # Custom exception (Game_fail_Exception)
│   ├── Data.py                        # Label/class name mappings for detection (legacy)
│   ├── identify_for_import.py         # Object detection inference utilities (legacy TF)
│   ├── Object_detection_image.py      # TensorFlow object detection demo (legacy)
│   ├── train.py                       # TensorFlow training script (legacy)
│   └── Minesweeper/
│       ├── MinesweeperLogic.py        # Pure game logic (no UI dependencies)
│       ├── Minesweeper.py             # Tkinter Minesweeper UI (delegates to MinesweeperLogic)
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

### `RL_Agent.py` (All RL Agents and Networks)

This is the largest source file (~1400 lines). Contains all neural network architectures and agent classes:

**Perception & Attention layers:**
- `YOLO11nBackbone` — YOLO11n feature extractor with mid+last layer fusion → (128, 80, 80)
- `LocalAttentionLayer` — Windowed self-attention (8×8) for neighbor-level reasoning
- `GlobalAttentionLayer` — Full self-attention (Flash Attention) for board-level strategy
- `HierarchicalAttentionHead` — 3× Local + 1× Global + Conv Downsample → 256-dim embedding
- `ScaledSigmoid` — Activation outputting ≈ [-0.05, 1.05] for direct grid coordinate mapping

**Stage 2 (visual) agents:**
- `SACActorNetwork` — YOLO11n backbone + HierarchicalAttention + Gaussian policy → (x, y)
- `SACCriticNetwork` — Twin Q-networks (embedding + action → Q-value)
- `ReplayBuffer` — Disk-backed buffer for large screenshot tensors
- `SACAgent` — Full Stage 2 SAC agent (screen capture → mouse click)

**Stage 1 (discrete) agents:**
- `GridEncoder` — ConvTranspose2d upsampling: (12, 10, 10) → (128, 80, 80)
- `Stage1ActorNetwork` — GridEncoder + HierarchicalAttention + ScaledSigmoid policy
- `Stage1CriticNetwork` — Separate GridEncoder + HierarchicalAttention + twin Q-networks
- `SumTree` — Priority-based data structure (used in replay buffer)
- `Stage1ReplayBuffer` — Per-class circular buffer with balanced sampling
- `Stage1SACAgent` — Stage 1 SAC agent with valid-rate-based alpha

**Simple MLP experiment:**
- `SimpleActorNetwork` — Flatten → MLP → 100 discrete action probabilities (with action masking)
- `SimpleCriticNetwork` — Flatten → MLP → 100 Q-values (twin)
- `SimpleDiscreteAgent` — SAC-Discrete with auto-alpha + clamp, validates RL pipeline

### `train_stage1.py` (Stage 1 Pre-training)

Standalone training script for Stage 1 continuous SAC on discrete grid state. Uses `MinesweeperLogic` API directly (no GUI). Outputs: TensorBoard logs, CSV training data, transfer weights (`stage1_weights.pth`).

### `train_stage1_simple.py` (Simple MLP Experiment)

Diagnostic experiment to isolate whether learning failures come from the RL pipeline (reward, buffer, SAC formula) or the network architecture (GridEncoder + Attention). Uses `SimpleDiscreteAgent` with 100 discrete actions.

### `training_logger.py` (Logging Utilities)

- `TeeOutput` — Dual output to console + text file (replaces sys.stdout)
- `CSVLogger` — Per-episode CSV logging with append mode

### `MinesweeperLogic.py` (Game Logic)

Pure Python Minesweeper implementation. No UI dependencies. Provides:
- `click(row, col)` → `ClickResult` (changed, game_over, win, revealed_cells, hit_mine)
- `flag(row, col)` → `FlagResult` (toggled, is_flagged)
- `get_grid_state_tensor()` → (12, 10, 10) one-hot tensor for Stage 1 input
- First-click safety guarantee (mines placed after first click)
- Recursive flood-fill for blank cells

### `Demo_test_Minesweeper.py` (Entry Point)

- Contains `Game_test_case` (unittest.TestCase) with the full game lifecycle
- `Game_status` inner class holds per-round state (agent, screenshots, rewards, step count)
- `decide_next_step_and_play()` — core RL loop: capture → preprocess → act → click
- `update_model()` — store transition and trigger training
- `test_RL()` — outer RL loop with win/lose/timeout detection
- Main block starts Minesweeper process, initializes globals, runs infinite game loop

### `Tool_Main.py` (Utilities)

- `Glo_var` — global state container (game name, round count, file handles, timing)
- `compare_sim()` — template matching via screenshot comparison
- `cut_pic_data()` — capture and save game region screenshots
- `click()` / `click_mid()` — mouse click execution with region validation
- `cal_time_out()` — timeout checking for game steps
- Reads config from `user_change/` directory

### `Minesweeper.py` (Game UI)

Full Minesweeper implementation in tkinter. Beginner/Intermediate/Expert difficulty levels. Delegates game logic to `MinesweeperLogic`. Runs in a separate process via `Minesweeper_manager`.

## Legacy Files (Not Active)

| File | Original Purpose |
|------|-----------------|
| `train.py` | TensorFlow object detection model training |
| `Object_detection_image.py` | TensorFlow inference demo |
| `identify_for_import.py` | Detection result classification |
| `Data.py` | Label mappings for card/game detection |
