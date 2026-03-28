# Running the System

## Prerequisites

- Python 3.10
- CUDA-capable GPU (tested on GTX 1050 Ti)
- Windows OS (uses pyautogui for screen interaction)
- Display must be visible (not headless — the agent clicks on actual screen pixels)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Note: `requirements.txt` currently lists legacy dependencies. For the RL agent, the actual requirements are:
   - `torch` (with CUDA support)
   - `torchvision`
   - `pyautogui`
   - `pynput`
   - `opencv-python`
   - `pillow`
   - `numpy`

2. Template images must exist in `user_change/game_pic/Minesweeper_pic/` with corresponding `.txt` coordinate files.

3. The game configuration is read from `user_change/Minesweeper_input.txt`.

## Running

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

- **End key**: Toggle pause/resume during gameplay
- The agent runs autonomously once started

## Outputs

| Output | Location |
|--------|----------|
| Model weights | `models/actor.pth`, `models/critic_1.pth`, `models/critic_2.pth` |
| Action logs | `models/action_logs/` (annotated screenshots) |
| Test reports | `testreport/Report-*.html` |
| Text logs | `testreport/<timestamp>/pipe_output.txt`, `cmd_output.txt`, `error.txt` |
