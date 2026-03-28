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
| Replay buffer format | Raw images (640×640 float16 tensors) + rewards on disk | Must store raw images since backbone is trainable (embeddings change as weights update). Saved to disk for upload to GCP |
| Replay buffer (runtime) | BUFFER_CAPACITY = 600 entries (~1.44 GB RAM) | Circular buffer in CPU RAM. Training samples randomly from these 600 entries |
| Replay buffer (persistent) | SAVE_CAPACITY = 150 entries (~360 MB disk) | Stratified random subset saved to disk every 50 episodes + on exit. Loaded into runtime buffer on next startup |
| TensorRT export | Full inference path: YOLO + spatial attention + actor | Critic is not needed at inference time. Optimizing the complete forward pass (screenshot → click coordinates) gives maximum speedup on 1050 Ti |
| Fine-tuning location | GCP only | 1050 Ti is for data collection (Phase 1) and inference (Phase 3) only. All training happens on GCP (Phase 2) |

## Reward Design

| Event | Reward | Notes |
|-------|--------|-------|
| Valid click (board changes) | +2, +4, +6, ... (escalating by +2 each consecutive valid click) | Rewards sustained good play; resets each episode |
| Invalid click (no screen change) | -1 | Includes clicking revealed cells, flagged cells, etc. |
| Click outside game region | -1 | Treated same as invalid click |
| Hit mine (lose) | -10 | Game over |
| Win | +20 | Game over |

Design principles:
- Valid click reward escalates starting from +2, increasing by +2 each consecutive valid click per episode — incentivizes sustained good play
- Invalid click and out-of-bounds are unified as -1 (both mean "nothing useful happened")
- Lose penalty and win reward provide strong terminal signals

## Two-Tier Replay Buffer

The replay buffer has two layers:

| Layer | Capacity | Location | Purpose |
|-------|----------|----------|---------|
| Runtime Buffer | 600 entries (~1.44 GB) | CPU RAM | Circular buffer for online training. `sample()` draws randomly from here |
| Persistent Save | 150 entries (~360 MB) | Disk (`models/replay_buffer_save/`) | Saved subset loaded on next startup |

### Save Strategy (Runtime → Persistent)

- **Trigger**: Every 50 episodes + on program exit (`atexit`)
- **Method**: Stratified random sampling by reward value
  - Group all 600 buffer entries by their reward
  - From each reward group, sample proportionally (e.g., if 30% of buffer has reward -1, then 30% of saved 150 = 45 entries with reward -1)
  - Ensures balanced reward distribution in persistent storage

### Load Strategy (Persistent → Runtime)

- On startup, `try_load_model()` loads 150 persistent entries into runtime buffer slots 0-149
- `replay_buffer.ptr = 150` → new transitions write to slots 150-599
- Runtime buffer has room for 450 more before circular overwrite begins

## Checkpoint & Resume

The agent must be able to **fully resume training** from the last saved state. A complete checkpoint includes:

| Component | File | Purpose |
|-----------|------|---------|
| Actor weights | `models/actor.pth` | Policy network (YOLO backbone + spatial head + policy head) |
| Critic weights | `models/critic.pth` | Twin Q-networks |
| Critic target weights | `models/critic_target.pth` | Soft-updated target networks |
| Entropy coefficient | `models/log_alpha.pth` | Learnable SAC temperature α |
| Training state | `models/training_state.pth` | Optimizer states, step counter, replay buffer index |

### training_state.pth contents

| Field | Type | Purpose |
|-------|------|---------|
| `actor_optimizer` | state_dict | Adam momentum/variance for actor — without this, optimizer "forgets" learning dynamics on restart |
| `critic_optimizer` | state_dict | Adam momentum/variance for critic |
| `alpha_optimizer` | state_dict | Adam momentum/variance for α |
| `total_it` | int | Total training steps completed — for logging and scheduling |
| `episode_count` | int | Total episodes completed — for periodic save scheduling |
| `persistent_index` | list[dict] | Metadata index for the 150 persistent entries (paths to `replay_buffer_save/` .pt files) |

### Resume behavior

- On `SACAgent.__init__()`, `try_load_model()` is called automatically
- Each component loads independently with try/except — partial checkpoints are tolerated
- Persistent buffer (150 entries in `models/replay_buffer_save/`) is copied into runtime buffer slots 0-149
- Runtime buffer pointer starts at 150, leaving room for 450 new transitions before circular overwrite
- Optimizer state restoration ensures Adam momentum/variance continuity (without this, training effectively "restarts" even if weights are loaded)

## All Design Questions Resolved

No open questions remain. Ready for implementation when authorized.
