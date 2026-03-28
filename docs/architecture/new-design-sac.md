# New Design: SAC + YOLO11n

> **STATUS: DESIGN PHASE** — No code has been written for this design yet.

This document captures the proposed redesign as specified by the project owner.

## Motivation

The previous TD3 + ResNet18 implementation did not work. Key problems:
- Deterministic policy (TD3) with Gaussian noise was insufficient for exploration
- ResNet18 backbone was too heavy and not frozen, leading to slow/unstable training
- No entropy regularization meant the agent got stuck in local optima

## Proposed Architecture

### 1. Perception: YOLO11n Feature Extractor

- **Model**: YOLO11n (Nano variant) — used as a **frozen** feature extractor only
- **Input**: Screen captures resized to **640 × 640**
- **Processing**: Spatial features → Global Average Pooling (GAP) → 1D latent embedding
- **Key difference from current**: Backbone is frozen (no gradient flow), much lighter than ResNet18

### 2. Decision Engine: SAC (Soft Actor-Critic)

- **Algorithm**: SAC with automatic entropy tuning
- **State**: Latent embedding vector + optional normalized history
- **Key advantages over TD3**:
  - Stochastic policy (Gaussian) — natural exploration without additive noise
  - Entropy maximization — prevents premature convergence to suboptimal click patterns
  - More sample-efficient for continuous control

### 3. Action Space: Continuous (x, y)

- Output: Two values representing normalized coordinates
- Range: [0, 1] via Tanh activation (or sigmoid)
- Scaled to game window resolution for mouse execution
- Same concept as current implementation

### 4. Training Pipeline

| Phase | Environment | Purpose |
|-------|-------------|---------|
| Phase 1 (Local) | GTX 1050 Ti | Collect experience data (screenshots, actions, rewards) into Replay Buffer |
| Phase 2 (Cloud) | GCP | Off-policy SAC training on uploaded Replay Buffer |
| Phase 3 (Inference) | GTX 1050 Ti | Deploy optimized weights (TensorRT `.engine`) for real-time play |

### 5. Operational Flow

```
Capture Screenshot → YOLO11n Features → Latent Embedding →
SAC Actor Output (x, y) → Pixel Scaling → Mouse Click → State Verification
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Framework | PyTorch 2.1.0 |
| Backend | CUDA 11.8 / cuDNN |
| Perception | YOLO11n (Ultralytics) |
| Mouse Control | pyautogui |
| Local GPU | GTX 1050 Ti |
| Cloud Training | GCP |
| Inference Optimization | TensorRT |

## Open Design Questions

> These need to be resolved before implementation begins.

- **Embedding dimension**: What is the output size of YOLO11n after GAP? (depends on which layer we tap)
- **History mechanism**: How to incorporate action/state history — concatenation, LSTM, or frame stacking?
- **Replay buffer format**: Store raw images or pre-computed embeddings? (memory vs. compute tradeoff)
- **Reward shaping**: Keep current Minesweeper rewards or redesign?
- **TensorRT export**: Which layers get optimized — just YOLO backbone, or full actor?
