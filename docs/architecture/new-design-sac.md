# New Design: SAC + YOLO11n

> **STATUS: DESIGN PHASE** — No code has been written for this design yet.

This document captures the proposed redesign as specified by the project owner.

## Motivation

The previous TD3 + ResNet18 implementation did not work. Key problems:
- Deterministic policy (TD3) with Gaussian noise was insufficient for exploration
- ResNet18 backbone was too heavy and not frozen, leading to slow/unstable training
- No entropy regularization meant the agent got stuck in local optima

## Proposed Architecture

### 1. Perception: YOLO11n Feature Extractor (Transfer Learning)

- **Model**: YOLO11n (Nano variant) — initialized from pretrained weights (COCO), then **fine-tuned end-to-end** with SAC training
- **Input**: Screen captures resized to **640 × 640**
- **Feature Layers**: Extract from **mid-layer** (~40×40, spatial detail) and **last layer** (~20×20, semantic understanding), then fuse
- **Processing**: Dual feature maps → Upsample/align → Concatenate → Trainable Spatial Head → Spatial-aware embedding
- **Spatial Head**: Spatial attention mechanism on top of fused feature maps — learns to focus on relevant grid cells while preserving positional information (no GAP — grid position matters for click targeting)
- **Why transfer learning**: Training from scratch is too slow/unstable; pretrained weights provide good low-level features (edges, textures, colors) as a starting point. Fine-tuning adapts them to game-specific patterns (digits, cell states, grid structure).
- **Why no GAP**: Minesweeper is a grid game — the agent must know *where* a number is, not just *that* a number exists. GAP collapses all spatial info into a single vector, making it impossible to target specific cells.

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
Capture Screenshot → YOLO11n Feature Maps → Spatial Head → Spatial Embedding →
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

## Resolved Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backbone training | Transfer learning (fine-tune from pretrained) | Training from scratch is too slow; frozen backbone can't learn game-specific features |
| Spatial encoding | Spatial head (no GAP) | Grid position is essential for click targeting in Minesweeper |
| Feature layer selection | Mid-layer + last layer fusion | Mid-layer provides spatial detail for cell positioning; last layer provides semantic understanding of cell contents (digits, states) |
| Spatial head architecture | Spatial attention mechanism | Learns to focus on relevant grid cells; more expressive than plain conv+flatten for a structured grid game |
| Embedding dimension | 256 | Balances expressiveness vs. 1050 Ti inference speed. YOLO11n mid+last fusion ≈ 384 channels at 40×40 → attention compresses to 256-d vector. Large enough for SAC on 2D action space, small enough for real-time inference |
| History mechanism | None — single frame only | Agent decides purely from current screenshot. Simplifies replay buffer, training, and inference. Minesweeper board state is fully observable from a single frame |
| Replay buffer format | Raw images (640×640 PNG) + rewards on disk | Must store raw images since backbone is trainable (embeddings change as weights update). Saved to disk for upload to GCP |
| TensorRT export | Full inference path: YOLO + spatial attention + actor | Critic is not needed at inference time. Optimizing the complete forward pass (screenshot → click coordinates) gives maximum speedup on 1050 Ti |
| Fine-tuning location | GCP only | 1050 Ti is for data collection (Phase 1) and inference (Phase 3) only. All training happens on GCP (Phase 2) |

## Reward Design

| Event | Reward | Notes |
|-------|--------|-------|
| Valid click (board changes) | +1 | Flat reward, no escalation |
| Invalid click (no screen change) | -1 | Includes clicking revealed cells, flagged cells, etc. |
| Click outside game region | -1 | Treated same as invalid click |
| Hit mine (lose) | -10 | Game over |
| Win | +20 | Game over |

Design principles:
- Simple and flat — no escalating rewards that distort value estimation
- Invalid click and out-of-bounds are unified as -1 (both mean "nothing useful happened")
- Lose penalty is 10× the step penalty, win reward is 20× the step reward — clear signal without being extreme

## All Design Questions Resolved

No open questions remain. Ready for implementation when authorized.
