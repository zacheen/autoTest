# System Architecture Overview

## Purpose

A visual RL agent that learns to play turn-based, mouse-driven games by observing screen captures and outputting click coordinates. The system is designed to be **game-agnostic** — the same pipeline handles any game that can be played through mouse clicks.

## High-Level Pipeline

```
Screen Capture → Feature Extraction → RL Policy → (x, y) Coordinates → Mouse Click → Reward Signal
```

### Component Breakdown

```
┌─────────────────────────────────────────────────────────┐
│                    Game Environment                      │
│  (Minesweeper / any mouse-driven game)                  │
│  - Runs as a separate process                           │
│  - Managed by game-specific Manager class               │
└────────────────┬────────────────────────────────────────┘
                 │ Screen Capture (pyautogui)
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Perception Layer                         │
│  - Captures game region screenshot                      │
│  - Preprocesses to fixed resolution                     │
│  - Extracts latent embedding via neural backbone        │
└────────────────┬────────────────────────────────────────┘
                 │ Latent Embedding Vector
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Decision Engine (RL Agent)               │
│  - Actor: embedding → normalized (x, y) ∈ [-1, 1]      │
│  - Critic: (embedding, action) → Q-value               │
│  - Replay Buffer for off-policy learning                │
└────────────────┬────────────────────────────────────────┘
                 │ Normalized (x, y)
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Action Execution                         │
│  - Scale (x, y) from [-1,1] to pixel coordinates        │
│  - Validate against game region boundaries              │
│  - Execute mouse click via pyautogui                    │
└────────────────┬────────────────────────────────────────┘
                 │ Game state changes
                 ▼
┌─────────────────────────────────────────────────────────┐
│                 Reward Evaluation                        │
│  - Screenshot comparison (before/after click)           │
│  - Win/lose detection via template matching             │
│  - Timeout penalties for ineffective clicks             │
└─────────────────────────────────────────────────────────┘
```

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

- **Continuous action space**: Agent outputs raw (x, y) coordinates, not discrete cell IDs. This makes the system game-agnostic.
- **On-screen interaction**: Uses `pyautogui` for real mouse clicks on a visible game window, not API-level game control.
- **Screenshot-based state**: The agent sees exactly what a human would see — raw pixels.
- **Region validation**: Click targets are validated against configurable game regions to avoid clicking outside the play area.
