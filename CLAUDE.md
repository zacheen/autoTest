# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**NEVER** update this file during a working session, we have other files to track project learnings and documentation references.

## Project Overview

A generalized turn-based zero-sum game agent that uses visual perception (screen captures) and reinforcement learning to autonomously play mouse-driven games. The system captures screenshots, extracts features via a neural backbone, decides click coordinates through an RL policy, and executes actions via mouse control. Currently targeting Minesweeper as the first game, with architecture designed for game-agnostic reuse.

## Specialized Sub-Agents Available

**ALWAYS**
1. Use the appropriate specialized sub-agents available for the task being worked on.
2. Provide the specialized sub-agents with the current working session goal.
3. Run the code review agent after each code change and have the appropriate coding agents fix any issues found.

## Important Reference Files

- Starting point to understand the project is: [docs/README.md](docs/README.md)
- Important lessons learned and pitfalls to avoid: [docs/LESSONS.md](docs/LESSONS.md)
