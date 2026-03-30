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
- **Current state**: Iterating on Stage 1 pre-training with two parallel experiments:
  - `Stage1SACAgent`: GridEncoder + HierarchicalAttention + continuous SAC (main experiment)
  - `SimpleDiscreteAgent`: MLP + discrete SAC (diagnostic — validates RL pipeline)
- **Active branch**: `_test_replace_attention` — testing discrete I/O to check if RL learns
- **Next state**: Stage 2 visual training with YOLO11n backbone (pending Stage 1 success)

## Key Entry Points

| What you want to do | Start here |
|---------------------|-----------|
| Understand the overall system | [architecture/system-overview.md](architecture/system-overview.md) |
| Run Stage 1 training | [development/running.md](development/running.md) → `python train_stage1.py` |
| Run the visual agent (Stage 2) | [development/running.md](development/running.md) → `python Demo_test_Minesweeper.py` |
| Understand RL_Agent.py classes | [development/codebase-map.md](development/codebase-map.md#rl_agentpy-all-rl-agents-and-networks) |
| Understand the SAC redesign | [architecture/new-design-sac.md](architecture/new-design-sac.md) |
| Check known pitfalls | [LESSONS.md](LESSONS.md) |

## Legacy Components

The repo contains legacy TensorFlow-based object detection code (`train.py`, `Object_detection_image.py`, `identify_for_import.py`, `Data.py`) from an earlier automation testing framework. These are **not part of the active RL agent** and exist for historical reference only.
