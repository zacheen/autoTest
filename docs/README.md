# Project Documentation Index

This file serves as a reference guide to navigate all project documentation.

## Quick Links

| Topic | Location | Description |
|-------|----------|-------------|
| System Architecture | [architecture/system-overview.md](architecture/system-overview.md) | High-level system design and component relationships |
| Current Implementation | [architecture/current-implementation.md](architecture/current-implementation.md) | What exists in the codebase today (TD3 + ResNet18) |
| New Design (SAC) | [architecture/new-design-sac.md](architecture/new-design-sac.md) | Proposed SAC + YOLO11n redesign |
| Codebase Map | [development/codebase-map.md](development/codebase-map.md) | File-by-file breakdown of the source code |
| Running the System | [development/running.md](development/running.md) | How to set up and run the project |
| Training Pipeline | [development/training-pipeline.md](development/training-pipeline.md) | Data collection, training, and inference workflow |
| Lessons Learned | [LESSONS.md](LESSONS.md) | Pitfalls, debugging notes, and session history |

## Project Status

- **Current state**: TD3 agent with ResNet18 backbone, targeting Minesweeper
- **Next state**: SAC agent with YOLO11n backbone, generalized for any mouse-driven game
- **Branch**: `test_discrete` (active development)

## Legacy Components

The repo contains legacy TensorFlow-based object detection code (`train.py`, `Object_detection_image.py`, `identify_for_import.py`) from an earlier automation testing framework. These are **not part of the active RL agent** and exist for historical reference only.
