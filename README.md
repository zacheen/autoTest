# Project Documentation Index

This file serves as a reference guide to navigate all project documentation.

## Quick Links

| Topic | Location | Description |
|-------|----------|-------------|
| System Architecture | [architecture/system-overview.md](architecture/system-overview.md) | High-level pipeline, component diagram, design decisions |
| Current Implementation | [architecture/current-implementation.md](architecture/current-implementation.md) | What exists today: three agent variants, hyperparameters, iteration history |
| SAC + YOLO11n Design | [architecture/new-design-sac.md](architecture/new-design-sac.md) | Full design spec: two-stage training, attention architecture, replay buffer, reward design |
| Codebase Map | [development/codebase-map.md](development/codebase-map.md) | File-by-file breakdown of every source file and directory |
| Running the System | [development/running.md](development/running.md) | Setup, execution commands, monitoring, outputs |
| Training Pipeline | [development/training-pipeline.md](development/training-pipeline.md) | Stage 1/2 pipeline, replay buffer design, checkpoint files, reward structure |
| Lessons Learned | [LESSONS.md](LESSONS.md) | Pitfalls, debugging notes, and session history |

## Project Status

- **Previous state**: TD3 agent with ResNet18 backbone (failed — no entropy regularization, backbone too heavy)
- **Stage 1 (confirmed working)**: `train_stage1_simple.py` — grid-state tensor → encoder-decoder transformer → FQF Q-network → 36 discrete actions. Learns Minesweeper policy on 6×6 board.
- **Stage 2 (active, in progress)**: `Demo_test_Minesweeper.py` — real screen capture → YOLO11n backbone → adapter → frozen Stage 1 encoder → trained decoder + FQF head → 36 discrete actions. Input changed from grid tensor to pixel screenshot; output is discrete (not continuous).
- **Current challenge**: Making the visual pipeline learn from screen captures (backbone fine-tuning + domain adaptation from YOLO object detection features to Minesweeper cell classification).
- **Active branch**: `Rainbow_DQN`

## Key Entry Points

| What you want to do | Start here |
|---------------------|-----------|
| Understand the overall system | [architecture/system-overview.md](architecture/system-overview.md) |
| Run Stage 1 training (grid state) | `python autoTest_pytorch/train_stage1_simple.py` |
| Run Stage 2 visual agent | `python autoTest_pytorch/Demo_test_Minesweeper.py` |
| Stage 1 agent code | `autoTest_pytorch/transformer_discrete_agent.py` |
| Stage 2 visual agent code | `autoTest_pytorch/visual_discrete_agent.py` |
| Check known pitfalls | [LESSONS.md](LESSONS.md) |

## Legacy Components

The repo contains legacy TensorFlow-based object detection code (`train.py`, `Object_detection_image.py`, `identify_for_import.py`, `Data.py`) from an earlier automation testing framework. These are **not part of the active RL agent** and exist for historical reference only.
