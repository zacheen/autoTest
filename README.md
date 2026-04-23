# Game-Playing AI via Visual Reinforcement Learning

An AI agent that learns to play turn-based games by watching the screen — no access to game internals required.

Demonstrated on Minesweeper: the agent observes raw screenshots, infers the board state, and selects cells to reveal. It achieves a **95% win rate** on a 6×6 board with a fully visual input pipeline.

---

## Vision

Large models such as Dream-3 can play a wide range of games, but their scale makes them impractical as a developer tool — training and inference both require infrastructure far beyond a typical development machine.

The goal of this project is a **lightweight, developer-local AI agent for game balance testing**:

- **Runs entirely on a developer's own machine**, including the training phase — no cloud dependency, no GPU cluster.
- **Learns purely from the screen**, the same way a player would, requiring no access to game internals or backend code.
- **Automatically surfaces dominant strategies**: if a sure-win path exists, a sufficiently capable agent will find and exploit it — revealing balance problems that conventional play-testing and API testing miss.
- **Game-agnostic by design**: point it at any grid-based or turn-based game, collect training data by playing, and let the agent converge — without game-specific engineering.

The goal is to make high-quality AI-driven balance testing accessible to solo developers and small studios, not just teams with research infrastructure.

---

## Motivation

### V1 — Automated Game Testing
Developed a versatile automated testing library for turn-based and zero-sum games, capable of validating game logic, verifying frontend-backend data consistency, and detecting rare bugs that conventional API testing cannot catch. The library leverages computer vision, automated input simulation, and modular pipelines to accelerate testing cycles and reduce manual effort.

### V2 — Self-Learning Game Agent
The next challenge: **can an AI discover dominant strategies on its own?**

Detecting "sure-win" strategies is difficult via standard play-testing, yet their existence subtly erodes the intended challenge and long-term player engagement. Rather than borrowing backend code to simulate game states, this project trains an agent purely from screen captures — the same information a human player has. A capable enough agent will naturally exploit any such strategy if one exists.

---

## Architecture

The V2 pipeline is split into two independently trained components, then composed for inference.

```
Screenshot (640×640 RGB)
    │
    ▼
┌─────────────────────────────────┐
│  YOLOGridStatePredictor         │  supervised learning
│  YOLO11n backbone (fine-tuned)  │  screenshot → predicted grid state
│  + cross-attention decoder      │  (12-channel one-hot, size-agnostic)
└─────────────────────────────────┘
    │  (12, 6, 6) one-hot tensor
    ▼
┌─────────────────────────────────┐
│  TransformerDiscreteAgent       │  reinforcement learning (Stage 1)
│  Encoder-Decoder Transformer    │  grid state → Q-values per cell
│  + FQF distributional Q-network │  trained on symbolic simulation
└─────────────────────────────────┘
    │
    ▼
Action: click cell (row, col)
```

**Why two stages?**

Training end-to-end (pixels → action) via RL destabilizes the YOLO backbone — gradients from misaligned attention propagate back and cause gradient explosions. Decoupling into a supervised vision module and a symbolic RL policy lets each component train stably and independently.

### Stage 1 — Symbolic Policy (`train_stage1_simple.py`)
- Input: ground-truth 12-channel grid state tensor from game logic
- Architecture: token embedding → 4-layer encoder-decoder transformer (d=64, 4 heads) → FQF Q-network
- Training: ~10,000 episodes of self-play on a local 6×6 Minesweeper simulator
- Result: **95% win rate, 0% invalid click rate**

### Vision Module — `yolo_grid_state_predictor.py`
- Input: 640×640 screenshot
- Architecture: YOLO11n backbone (layers 0–6, 40×40 feature map) → token adapter → 2-layer cross-attention decoder → 12-class head
- Training: supervised on (screenshot, server-state label) pairs collected during play
- Size-agnostic: query positions are computed from normalized cell coordinates — the model generalises to any grid size without retraining
- Result: **~100% per-cell accuracy** on held-out validation frames

### Inference — `visual_discrete_agent_v2.py`
Both trained models are frozen and composed. The agent receives a screenshot, predicts the board state, and selects the highest Q-value hidden cell.

**Full visual pipeline result: 95% win rate**

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.0 |
| CUDA | 11.8 |
| ultralytics | latest |
| torchvision | compatible with torch 2.1.0 |
| Flask | (Minesweeper web server) |
| Pillow | image preprocessing |
| pyautogui, pynput | screen capture and input simulation |

```bash
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics flask pillow pyautogui pynput tensorboard
```

---

## How to Run

> Pre-requisite: trained model checkpoints must exist at  
> `autoTest_pytorch/models/stage1_transformer/` and  
> `autoTest_pytorch/models/yolo_grid_predictor/best.pth`

**1. Start the Minesweeper web server**
```bash
python autoTest_pytorch/Minesweeper_web/server.py
```
The server exposes a local API for the agent to click cells and read game state.

**2. Run the visual agent**
```bash
cd autoTest_pytorch
python Demo_test_Minesweeper.py
```
The agent will play continuously, printing win/loss and step count per episode.

> **⚠ Keyboard control:** Press **`End`** at any time to **pause** the agent. Press **`End`** again to **resume**.

**Training from scratch** (if no checkpoints exist):
```bash
# Stage 1: train symbolic policy (~1–2 hours on CPU)
cd autoTest_pytorch
python train_stage1_simple.py

# Vision module: collect data, then train
# 1. Run Demo_test_Minesweeper with COLLECT_VISION_DATASET=True in yolo_grid_state_predictor.py
# 2. After ~5000 samples:
python yolo_grid_state_predictor.py
```

---

## Repository Structure

```
autoTest_pytorch/
├── Demo_test_Minesweeper.py          # main entry point — agent plays the game
├── visual_discrete_agent_v2.py       # Stage 2 inference agent (YOLO + Stage 1)
├── transformer_discrete_agent.py     # Stage 1 FQF agent
├── yolo_grid_state_predictor.py      # YOLO vision module (model + training)
├── train_stage1_simple.py            # Stage 1 training loop
├── Minesweeper_web/
│   └── server.py                     # local Minesweeper game server (Flask)
├── model_structure/
│   └── transformer_shared.py         # shared transformer and Q-network modules
└── models/                           # saved checkpoints (not committed)
    ├── stage1_transformer/
    └── yolo_grid_predictor/
docs/
├── DEVELOPMENT_HISTORY.md            # iteration history and lessons learned
└── DESIGN_YOLO_GRID_PREDICTOR.md    # detailed design notes for the vision module
```

---

## Future Work

### End-to-End Training

The two-stage pipeline was a practical workaround for training instability, not an architectural goal. Now that the model has proven it can learn to play Minesweeper at a high level, the next direction is to collapse both stages into a single end-to-end trainable system.

The core challenge is stabilizing gradient flow through the visual backbone during RL training — the same problem that required the decoupled approach in the first place. Potential directions include:

- **Curriculum learning**: begin training on symbolic grid states (Stage 1), then progressively introduce screenshot inputs, allowing the visual encoder to warm-start from an already-capable policy rather than learning vision and strategy simultaneously from noise.
- **Gradient isolation**: stop-gradient boundaries or separate optimizers with heavily reduced learning rates for the backbone during early RL phases, preventing the attention misalignment that caused gradient explosions in earlier attempts.
- **Broader generalisation**: extending beyond 6×6 to verify that the policy and vision module generalise across board sizes and mine densities — the size-agnostic query design of `YOLOGridStatePredictor` already supports this architecturally.

---

## Legacy

`train.py`, `Object_detection_image.py`, `identify_for_import.py`, and `Data.py` are from the V1 automated testing framework. They are not part of the V2 RL agent and exist for reference only.
