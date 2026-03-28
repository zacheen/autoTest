# Session: understand-model-structure
**Started**: 2026-03-28 13:20

## Overview
Understanding the current model structure — tracing how the SAC agent, YOLO11n backbone, replay buffer, and training pipeline connect together.

## Goals
- Understand the full model architecture (YOLO11n backbone → spatial head → SAC actor/critic)
- Trace the data flow from screenshot capture to action output
- Understand replay buffer structure and checkpoint save/load mechanism
- Clarify how Demo_test_Minesweeper.py orchestrates the training loop

## Progress
