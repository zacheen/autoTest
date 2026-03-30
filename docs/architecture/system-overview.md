# System Architecture Overview

## Purpose

A visual RL agent that learns to play turn-based, mouse-driven games by observing screen captures and outputting click coordinates. The system is designed to be **game-agnostic** — the same pipeline handles any game that can be played through mouse clicks. Currently targeting Minesweeper.

## High-Level Pipeline

```
Screen Capture → Feature Extraction → RL Policy → (x, y) Coordinates → Mouse Click → Reward Signal
```

### Component Breakdown

```
┌─────────────────────────────────────────────────────────┐
│                    Game Environment                      │
│  (Minesweeper / any mouse-driven game)                  │
│  - Runs as a separate process (tkinter)                 │
│  - Managed by game-specific Manager class               │
│  - Pure logic layer (MinesweeperLogic) for headless use │
└────────────────┬────────────────────────────────────────┘
                 │ Screen Capture (pyautogui) or Logic API
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Perception Layer                         │
│  Stage 1: GridEncoder (ConvTranspose2d, 10→80)          │
│  Stage 2: YOLO11n backbone (mid+last layer fusion)      │
│  Output: (B, 128, 80, 80) feature maps                 │
└────────────────┬────────────────────────────────────────┘
                 │ Feature Maps
                 ▼
┌─────────────────────────────────────────────────────────┐
│           Spatial Reasoning (HierarchicalAttention)      │
│  3× Local Attention (8×8 window) — neighbor reasoning   │
│  1× Global Self-Attention (Flash Attn) — board strategy │
│  Conv Downsample → FC → 256-dim embedding               │
└────────────────┬────────────────────────────────────────┘
                 │ Embedding Vector (256-d)
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Decision Engine (SAC Agent)               │
│  Actor: embedding → Gaussian(mean, std) → action        │
│  Critic: (embedding, action) → Q-value (twin)           │
│  Replay Buffer (per-class balanced / disk-backed)        │
│  Auto-alpha with clamp for entropy tuning               │
└────────────────┬────────────────────────────────────────┘
                 │ Action (x, y)
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Action Execution                         │
│  Stage 1: ScaledSigmoid → grid coords via logic API     │
│  Stage 2: Tanh → pixel coords → pyautogui mouse click   │
└────────────────┬────────────────────────────────────────┘
                 │ Game state changes
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Reward Evaluation                        │
│  Stage 1: MinesweeperLogic.ClickResult → reward         │
│  Stage 2: Screenshot comparison + win/lose detection    │
└─────────────────────────────────────────────────────────┘
```

## Two-Stage Training Architecture

| Stage | Input | Perception | Environment | Purpose |
|-------|-------|-----------|-------------|---------|
| Stage 1 | Grid state (12, 10, 10) | GridEncoder | MinesweeperLogic API | Validate SAC learns Minesweeper; pre-train attention + SAC heads |
| Stage 2 | Screenshot (3, 640, 640) | YOLO11n | GUI + pyautogui | Visual policy with transfer weights from Stage 1 |

Weight transfer: HierarchicalAttentionHead + SAC heads transfer from Stage 1 → Stage 2. GridEncoder is discarded (replaced by YOLO11n).

## Additional Experiment: Simple MLP

A diagnostic agent (`SimpleDiscreteAgent`) uses a flat MLP with 100 discrete actions to validate the RL pipeline independent of the attention architecture. If MLP learns but attention doesn't, the problem is in the network architecture.

## Test Harness

The system uses Python's `unittest` framework as an orchestration layer. Each game round is a test suite with ordered test cases:

1. `test_choose_room` — Navigate to game lobby
2. `test_state_prepare` — Initialize round state
3. `test_click_middle` — Wait for game board ready
4. `test_RL` — Main RL loop (observe → decide → click → reward → train)
5. `test_wait_result` — Handle game-over dialogs
6. `test_new_game` — Reset for next round

Results are logged as HTML test reports via `HTMLTestRun`.

## Key Design Decisions

- **Continuous action space** (Stage 2): Agent outputs raw (x, y) coordinates, not discrete cell IDs. This makes the system game-agnostic.
- **Discrete action space** (Simple MLP): 100 actions mapped to 10×10 grid cells with action masking. Used for pipeline validation only.
- **On-screen interaction**: Uses `pyautogui` for real mouse clicks on a visible game window, not API-level game control.
- **Screenshot-based state** (Stage 2): The agent sees exactly what a human would see — raw pixels.
- **Game logic separation**: `MinesweeperLogic` enables headless training without GUI for Stage 1.
- **Hierarchical attention**: Local attention (8×8 window) for Minesweeper neighbor reasoning + global attention for board-level strategy. Replaces earlier SpatialAttentionHead (weighted pooling) which destroyed spatial information.
