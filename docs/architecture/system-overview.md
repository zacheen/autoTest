# System Architecture Overview

## Purpose

A visual RL agent that learns to play turn-based, mouse-driven games by observing screen captures and outputting discrete grid-cell click actions. Currently targeting Minesweeper. The architecture is designed to be **game-agnostic** — any game reducible to a fixed-size discrete cell grid can reuse the same pipeline.

## High-Level Pipeline

```
Screen Capture → YOLO Feature Extraction → Encoder → Decoder → FQF Q-Network → Grid Cell → Click
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
│                 Perception Layer (frozen)                │
│  YOLO11n backbone → (B, 128, 40, 40)                   │
│  token_adapter (LayerNorm + Linear) → (B, 1600, 128)   │
│  + fixed 2D sinusoidal positional encoding              │
│  HierarchicalEncoder [128→64→32] → (B, 1600, 32)       │
└────────────────┬────────────────────────────────────────┘
                 │ Memory tokens (B, 1600, 32)
                 ▼
┌─────────────────────────────────────────────────────────┐
│           Spatial Reasoning (TransformerDecoder)         │
│  5× pre-LN cross-attention layers                       │
│  36 learned query tokens (one per 6×6 grid cell)        │
│  Query tokens attend to 1600 memory tokens              │
│  Output: per-cell embeddings (B, 36, 32)                │
└────────────────┬────────────────────────────────────────┘
                 │ Per-cell embeddings
                 ▼
┌─────────────────────────────────────────────────────────┐
│             Decision Engine (FQF Q-Network)              │
│  FQFQNetwork: quantile fraction proposal + cosine embed  │
│  Per-action distributional Q-values (B, 36)             │
│  Action masking: only unrevealed cells eligible          │
│  argmax over masked Q-values → cell index               │
└────────────────┬────────────────────────────────────────┘
                 │ Action (grid cell index, 0–35)
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Action Execution                         │
│  Stage 1: cell index → MinesweeperLogic.click(row, col) │
│  Stage 2: cell index → pixel coords → pyautogui click   │
└────────────────┬────────────────────────────────────────┘
                 │ Game state changes
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Reward Evaluation                        │
│  Stage 1: MinesweeperLogic.ClickResult → reward         │
│  Stage 2: Screenshot comparison + win/lose detection    │
│  Config: MINESWEEPER_REWARD_CONFIG in reward_settings.py│
└─────────────────────────────────────────────────────────┘
```

## Two-Stage Training Architecture

| Stage | Input | Perception | Environment | Action Space | Purpose |
|-------|-------|-----------|-------------|-------------|---------|
| Stage 1 | Grid state (12, 10, 10) | Learned TokenEmbed + EncoderDecoderTransformer | MinesweeperLogic API | Discrete 10×10 = 100 | Validate FQF learns Minesweeper; fast headless iteration |
| Stage 2 | Screenshot (3, 640, 640) | YOLO11n (frozen) + HierarchicalEncoder | GUI + pyautogui | Discrete 6×6 = 36 | Visual policy; encoder frozen, only decoder + FQF head trained |

**Note**: Both stages use **discrete** action spaces and **FQF distributional Q-learning**. The earlier SAC + continuous (x, y) approach was abandoned due to lack of convergence signal.

### Stage 2 Frozen vs. Trainable

The YOLO backbone, token adapter, and HierarchicalEncoder are loaded from the `YOLOGridStatePredictor` supervised pre-training checkpoint and kept frozen (BN in eval mode). Only the TransformerDecoder + query tokens + FQFQNetwork head are trained online via RL.

## Key Design Decisions

- **Discrete action space**: Agent selects from a fixed grid of cells (6×6 = 36 for Stage 2), not raw (x, y) coordinates. Enables action masking and clear reward attribution.
- **Action masking**: Only unrevealed cells are eligible — prevents the agent from repeatedly clicking already-known cells. Uses `-1e8` (not `-inf`) to avoid `0 × -inf = NaN` in loss.
- **Distributional RL (FQF)**: Quantile regression instead of scalar Q-values. Better handles the bimodal reward distribution (most clicks are +0.5 or -0.25; rare mines are -0.7, rare wins are +1.0).
- **Frozen encoder**: YOLO + HierarchicalEncoder pre-trained via supervised grid-state prediction. Freezing them lets RL focus entirely on decoder + decision head, dramatically reducing sample complexity.
- **Cross-attention decoder**: 1600 memory tokens compressed into 36 action-specific embeddings via cross-attention. Each query token specializes in one grid cell.
- **Pre-LN transformer layers**: `norm_first=True` throughout for training stability.
- **Fixed sinusoidal positional encoding**: Used in the encoder (YOLO feature tokens) for position information. Query tokens carry no positional encoding — position is implicit from the cell index.
- **Game logic separation**: `MinesweeperLogic` enables headless training without GUI for Stage 1. No torch dependency in game logic.
- **On-screen interaction**: Stage 2 uses `pyautogui` for real mouse clicks on a visible game window.
- **Centralized reward config**: `model_structure/reward_settings.py` (`MINESWEEPER_REWARD_CONFIG`) is the single source of truth for all reward values and gamma.

## Test Harness

The system uses Python's `unittest` framework as an orchestration layer. Each game round is a test suite with ordered test cases:

1. `test_choose_room` — Navigate to game lobby
2. `test_state_prepare` — Initialize round state
3. `test_click_middle` — Wait for game board ready
4. `test_RL` — Main RL loop (observe → decide → click → reward → train)
5. `test_wait_result` — Handle game-over dialogs
6. `test_new_game` — Reset for next round

Results are logged as HTML test reports via `HTMLTestRun`.
