# Running the System

## Prerequisites

- Python 3.10
- CUDA-capable GPU (tested on GTX 1050 Ti)
- Windows OS (uses pyautogui for screen interaction in Stage 2)
- Display must be visible for Stage 2 (not headless — the agent clicks on actual screen pixels)
- Stage 1 can run headless (no GUI required)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Key packages: `torch` (CUDA), `torchvision`, `ultralytics`, `pyautogui`, `pynput`, `opencv-python`, `pillow`, `numpy`

2. YOLO11n pretrained weights (`yolo11n.pt`) must be in the project root (used by Stage 2).

3. For Stage 2: template images must exist in `user_change/game_pic/Minesweeper_pic/` with corresponding `.txt` coordinate files. Game configuration is read from `user_change/Minesweeper_input.txt`.

## Running Stage 1 Pre-training (Headless)

From the `autoTest_pytorch/` directory:

```bash
# Main experiment: GridEncoder + HierarchicalAttention + continuous SAC
python train_stage1.py

# Diagnostic: Simple MLP + discrete SAC (validates RL pipeline)
python train_stage1_simple.py
```

Monitor with TensorBoard:
```bash
tensorboard --logdir runs/stage1/       # for train_stage1.py
tensorboard --logdir runs/stage1_simple/ # for train_stage1_simple.py
```

## Running Stage 2 Visual Agent (GUI Required)

From the `autoTest_pytorch/` directory:

```bash
python Demo_test_Minesweeper.py
```

This will:
1. Launch a Minesweeper window (separate process)
2. Start the RL game loop
3. Generate HTML test reports in `testreport/`
4. Save action log images in `models/action_logs/`
5. Save/load model weights in `models/`

## Controls

- **End key**: Toggle pause/resume during gameplay (Stage 2)
- The agent runs autonomously once started

## Outputs

| Output | Location |
|--------|----------|
| Stage 1 model weights | `models/stage1/` or `models/stage1_simple/` |
| Stage 1 transfer weights | `models/stage1/stage1_weights.pth` |
| Stage 1 training logs | `runs/stage1/`, `models/stage1/training_log.csv` |
| Stage 2 model weights | `models/actor.pth`, `models/critic.pth`, etc. |
| Action logs | `models/action_logs/` (annotated screenshots) |
| Test reports | `testreport/Report-*.html` |
| Text logs | `testreport/<timestamp>/pipe_output.txt`, `cmd_output.txt`, `error.txt` |
