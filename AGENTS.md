# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

**NEVER** update this file during a working session, we have other files to track project learnings and documentation references.

## Project Overview

A visual reinforcement learning agent for turn-based grid games using screen captures as input and discrete cell-click actions as output. The pipeline captures a screenshot of the game grid, extracts 40×40 feature tokens via a YOLO11n backbone, refines them through an encoder-decoder transformer, and selects a discrete grid-cell action (6×6 = 36 choices) via an FQF distributional Q-network. Training uses a two-stage approach: Stage 1 trains the transformer policy on symbolic game-state tensors (fast simulation); Stage 2 fine-tunes the full visual pipeline (YOLO backbone + adapter + decoder + FQF head) on real screen captures. Currently targeting Minesweeper, with architecture designed for game-agnostic reuse.

## Specialized Sub-Agents Available

**ALWAYS**
1. Use the appropriate specialized sub-agents available for the task being worked on.
2. Provide the specialized sub-agents with the current working session goal.
3. Run the code review agent after each code change and have the appropriate coding agents fix any issues found.

## Important Reference Files

- Starting point to understand the project is: [docs/README.md](docs/README.md)
- Important lessons learned and pitfalls to avoid: [docs/LESSONS.md](docs/LESSONS.md)
