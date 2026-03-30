# New Design: SAC + YOLO11n

> **STATUS: IMPLEMENTATION** — Stage 1 code implemented and iterating.

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
- **Processing**: Dual feature maps → Upsample/align → Concatenate → HierarchicalAttentionHead → 256-dim embedding
- **Attention Architecture**: Hierarchical attention replaces the old SpatialAttentionHead (which collapsed spatial info via weighted pooling). New design:
  - **3× Local Attention** (8×8 window): Each position attends to its 8×8 neighborhood. Channel reduction 128→64→32→32. Learns neighbor-level reasoning (e.g., "this cell is 1, one neighbor unrevealed").
  - **1× Global Self-Attention** (Flash Attention, 6400 positions): All positions attend to each other. Channel 32→16. Learns board-level strategy.
  - **Conv Downsample**: (16,80,80) → stride-4 convs → (64,5,5) → flatten → FC → 256-dim embedding.
  - Total parameters: ~493K. VRAM: ~1.9GB (batch=32) on 1050 Ti (fits in 4GB).
- **Why transfer learning**: Training from scratch is too slow/unstable; pretrained weights provide good low-level features (edges, textures, colors) as a starting point. Fine-tuning adapts them to game-specific patterns (digits, cell states, grid structure).
- **Why no GAP**: Minesweeper is a grid game — the agent must know *where* a number is, not just *that* a number exists. GAP collapses all spatial info into a single vector, making it impossible to target specific cells.

### 2. Decision Engine: SAC (Soft Actor-Critic)

- **Algorithm**: SAC with auto-alpha (clamped) for both Stage 1 and Stage 2
- **Alpha tuning**: Standard SAC auto-alpha with correct target entropy + clamps:
  - Discrete (Stage 1): `target_entropy = 0.5 * ln(100) ≈ 2.3` (50% of max categorical entropy)
  - Continuous (Stage 2): `target_entropy = -action_dim = -2.0`
  - Clamp: `alpha ∈ [ALPHA_MIN=0.05, ALPHA_MAX=0.3]` prevents both entropy collapse and over-exploration
  - Previous valid-rate-based alpha was removed because it didn't respond to entropy collapse (alpha stayed at 0.27 while entropy dropped to 0)
- **State**: Latent embedding vector + optional normalized history
- **Key advantages over TD3**:
  - Stochastic policy (Gaussian) — natural exploration without additive noise
  - Entropy maximization — prevents premature convergence to suboptimal click patterns
  - More sample-efficient for continuous control

### 3. Action Space: Continuous (x, y)

- Output: Two values representing normalized coordinates
- Activation: **ScaledSigmoid** (scale=1.1, shift=-0.05) — output range ≈ [-0.05, 1.05]
  - Values in [0, 1] → valid grid coordinates
  - Values < 0 or > 1 → out-of-bounds (penalized with -3 reward)
  - Advantage over Tanh: output maps directly to [0, 1] grid space without extra normalization; boundary region provides a natural "out-of-bounds" signal
- Scaled to game window resolution for mouse execution

### 4. Training Pipeline

| Phase | Environment | Purpose |
|-------|-------------|---------|
| Phase 1 (Local) | GTX 1050 Ti | Collect experience data (screenshots, actions, rewards) into Replay Buffer |
| Phase 2 (Cloud) | GCP | Off-policy SAC training on uploaded Replay Buffer |
| Phase 3 (Inference) | GTX 1050 Ti | Deploy optimized weights (TensorRT `.engine`) for real-time play |

### 5. Operational Flow

```
Capture Screenshot → YOLO11n Feature Maps → HierarchicalAttention → Embedding →
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
| Spatial encoding | HierarchicalAttentionHead (no GAP) | Grid position is essential for click targeting in Minesweeper. Previous SpatialAttentionHead (weighted pooling) failed — destroyed spatial info, causing policy collapse after 1200 episodes |
| Feature layer selection | Mid-layer + last layer fusion | Mid-layer provides spatial detail for cell positioning; last layer provides semantic understanding of cell contents (digits, states) |
| Spatial head architecture | HierarchicalAttention: 3× Local Attn (8×8 window, 128→64→32→32) + 1× Global Self-Attn (Flash Attn, 32→16) + Conv Downsample → 256 | Local attention reasons about neighbors (like Minesweeper rules); global attention captures board-level strategy; ~503K params, ~1.9GB VRAM (batch=32) |
| Alpha tuning | Auto-alpha with clamp: target_entropy=2.3 (discrete) / -2.0 (continuous), alpha ∈ [0.05, 0.3] | Previous auto-alpha had wrong target (-2.0 for discrete), valid-rate-based alpha didn't respond to entropy collapse. Clamped auto-alpha with correct discrete target prevents both issues |
| Embedding dimension | 256 | Balances expressiveness vs. 1050 Ti inference speed. YOLO11n mid+last fusion ≈ 384 channels at 40×40 → attention compresses to 256-d vector. Large enough for SAC on 2D action space, small enough for real-time inference |
| History mechanism | None — single frame only | Agent decides purely from current screenshot. Simplifies replay buffer, training, and inference. Minesweeper board state is fully observable from a single frame |
| Replay buffer format | Raw images (640×640 float16 tensors) + rewards on disk | Must store raw images since backbone is trainable (embeddings change as weights update). Saved to disk for upload to GCP |
| Replay buffer design | Per-class circular buffers | Each reward value gets its own circular buffer (max_per_class entries each). Sampling draws equally from each class → guaranteed balanced training data. Eliminates PER, SumTree, protected buffer complexity. Previous single circular buffer failed because rare experiences (mine hits, wins) were overwritten by common ones (valid/invalid clicks). |
| Replay buffer (persistent) | SAVE_CAPACITY = 150 entries (~360 MB disk) | Balanced subset saved to disk every 50 episodes + on exit. Each class gets SAVE_CAPACITY/num_classes entries. |
| Action masking (Stage 1) | Hard mask from state channel 0 (unrevealed cells) | Without masking, agent repeatedly clicks revealed cells → state unchanged → policy collapse. Masking ensures only unrevealed cells can be clicked. Uses -1e8 (not -inf) to avoid 0×(-inf)=NaN in actor loss. |
| TensorRT export | Full inference path: YOLO + HierarchicalAttention + actor | Critic is not needed at inference time. Optimizing the complete forward pass (screenshot → click coordinates) gives maximum speedup on 1050 Ti |
| Fine-tuning location | GCP only | 1050 Ti is for data collection (Phase 1) and inference (Phase 3) only. All training happens on GCP (Phase 2) |

## Reward Design

| Event | Reward | Notes |
|-------|--------|-------|
| Valid click (board changes) | +3 | Effective click that reveals new cell(s) |
| Invalid click (already revealed/flagged) | -2 | Clicking a cell that's already open or flagged |
| Click outside grid bounds | -3 | ScaledSigmoid output outside [0, 1] range |
| Hit mine (lose) | -3 | Game over — kept low to encourage exploring inside the grid |
| Win | +20 | Game over |

Design principles:
- **Graduated penalties**: out-of-bounds (-3) > revealed cell (-2) > valid (+1). Agent learns to stay in bounds first, then click unrevealed cells.
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

## Two-Stage Training Architecture

The system uses a two-stage training approach: first pre-train on discrete grid state to validate that SAC can learn Minesweeper, then transfer weights to the visual pipeline.

### Stage 1: Discrete Pre-training (Grid State → SAC)

```
Grid State (B, 12, 10, 10)
       │
       ▼
┌─ GridEncoder (discarded after Stage 1) ────────┐
│  ConvTranspose2d(12→64, k=4,s=2,p=1) + SiLU   │  10×10 → 20×20
│  ConvTranspose2d(64→128, k=4,s=2,p=1) + SiLU   │  20×20 → 40×40
│  ConvTranspose2d(128→128, k=4,s=2,p=1) + SiLU   │  40×40 → 80×80
└────────────────────────────────────────────────┘
       │
       ▼
    (B, 128, 80, 80)  ← same shape as YOLO output
       │
       ▼
┌─ HierarchicalAttentionHead (weights transferred) ┐
│  Local Attn L1 (8×8): 128→64                     │
│  Local Attn L2 (8×8): 64→32                      │
│  Local Attn L3 (8×8): 32→32 (residual)           │
│  Global Self-Attn (Flash): 32→16                  │
│  Conv Downsample → FC → embedding (B, 256)        │
└───────────────────────────────────────────────────┘
       │
       ▼
┌─ SAC Actor head (weights transferred) ─────────┐
│  mean_head → (B, 2)                             │
│  log_std_head → (B, 2)                          │
└────────────────────────────────────────────────┘
       │
       ▼
    action (x, y) ∈ [0, 1]
```

**Grid State Encoding** — 12-channel one-hot per cell:

| Channel | Meaning |
|---------|---------|
| 0 | Unrevealed (未翻開) |
| 1 | Flagged (已標旗) |
| 2 | Revealed number 0 (空白) |
| 3–9 | Revealed number 1–7 |
| 10 | Revealed number 8 |
| 11 | Mine (地雷, only visible on game over) |

**Purpose**: Validate that HierarchicalAttentionHead + SAC Actor can learn Minesweeper spatial reasoning (e.g., inferring bomb locations from adjacent numbers — local attention reasons about neighbors, global attention reasons about the full board).

**Training environment**: Runs against `MinesweeperLogic` API directly (no screenshots, no GUI). Lightweight enough for local 1050 Ti.

**Critic**: Also pre-trained in Stage 1 with its own GridEncoder + HierarchicalAttentionHead. Weights transferred to Stage 2.

### Stage 2: Visual Training (Screenshot → SAC)

```
Screenshot (B, 3, 640, 640)
       │
       ▼
┌─ YOLO11nBackbone (COCO pretrained) ───────────┐
│  mid + last → fuse → channel_reduce            │
└────────────────────────────────────────────────┘
       │
       ▼
    (B, 128, 80, 80)  ← same shape
       │
       ▼
┌─ HierarchicalAttentionHead (loaded from Stage 1) ┐
│  NOT frozen — continues fine-tuning               │
└───────────────────────────────────────────────────┘
       │
       ▼
┌─ SAC Actor head (loaded from Stage 1) ────────┐
│  NOT frozen — continues fine-tuning            │
└────────────────────────────────────────────────┘
       │
       ▼
    action (x, y) ∈ [0, 1]
```

### Weight Transfer (Stage 1 → Stage 2)

| Component | Stage 1 source | Stage 2 initialization | Frozen? |
|-----------|---------------|----------------------|---------|
| GridEncoder | Trained | **Discarded** (replaced by YOLO) | — |
| YOLO11nBackbone | — | COCO pretrained | No |
| HierarchicalAttentionHead (Actor) | Trained | Loaded from Stage 1 | No |
| HierarchicalAttentionHead (Critic) | Trained | Loaded from Stage 1 | No |
| Actor mean_head / log_std_head | Trained | Loaded from Stage 1 | No |
| Critic FC layers | Trained | Loaded from Stage 1 | No |
| Alpha | Valid-rate-based (not saved) | Stage 2 uses auto-tuning (separate mechanism) | — |

**Saved checkpoint** (end of Stage 1):

```
stage1_weights.pth = {
    "actor_attention": state_dict,      # HierarchicalAttentionHead
    "actor_mean_head": state_dict,
    "actor_log_std_head": state_dict,
    "critic_attention": state_dict,     # HierarchicalAttentionHead
    "critic_q1": state_dict,
    "critic_q2": state_dict,
}
```

GridEncoder weights are NOT saved — they are Stage 1 only.

### Why This Works

The key insight: both GridEncoder and YOLO output **(B, 128, 80, 80)** feature maps. HierarchicalAttentionHead learns game-reasoning patterns (local neighbor reasoning + global board strategy) in Stage 1, then adapts to visual features in Stage 2. This is analogous to using YOLO's COCO pretrained weights — starting from a useful initialization rather than random.

## Game Logic Separation (MinesweeperLogic)

The Minesweeper game is split into two files to support both GUI play and headless pre-training:

```
Minesweeper/
  ├── MinesweeperLogic.py    ← Pure game logic, no UI imports
  ├── Minesweeper.py         ← UI layer, delegates to MinesweeperLogic
  └── Minesweeper_manager.py ← Subprocess launcher (unchanged)
```

### MinesweeperLogic API

```
class MinesweeperLogic:

    __init__(rows, cols, mines_count)

    reset() → grid_state
        # Reset game to initial state

    click(row, col) → ClickResult
        # Left-click a cell
        # Returns: ClickResult(
        #     changed: bool,           # board changed (valid click)
        #     game_over: bool,         # hit a mine
        #     win: bool,               # all safe cells revealed
        #     revealed_cells: list,    # newly revealed [(r, c, number), ...]
        #     hit_mine: tuple|None     # (r, c) of mine hit, or None
        # )

    flag(row, col) → FlagResult
        # Right-click to toggle flag
        # Returns: FlagResult(toggled: bool, is_flagged: bool)

    get_grid_state() → 2D array
        # Per-cell values: -1=unrevealed, -2=flagged, 0-8=revealed number

    get_grid_state_tensor() → Tensor (12, rows, cols)
        # One-hot encoding for Stage 1 pre-training input

    Properties:
        rows, cols, mines_count, game_over, is_win,
        first_click, revealed, flags, mines, remaining_mines
```

### UI Layer (Minesweeper.py)

**No visual changes** — only internal refactoring:
- Replaces inline game state (`self.mines`, `self.revealed`, etc.) with `self.logic = MinesweeperLogic(...)`
- UI methods read from `ClickResult`/`FlagResult` to update button appearance
- All game rule logic (mine placement, reveal cascade, win check) moves to `MinesweeperLogic`

## Stage 1 Pre-training Script (`train_stage1.py`)

Standalone script for Stage 1 discrete pre-training. Does NOT modify `Demo_test_Minesweeper.py`.

### File Location

```
autoTest_pytorch/
  ├── Demo_test_Minesweeper.py    ← unchanged
  ├── train_stage1.py             ← NEW: Stage 1 discrete pre-training
  ├── RL_Agent.py                 ← needs new Stage1 classes
  └── Minesweeper/
        ├── MinesweeperLogic.py   ← NEW: pure game logic
        ├── Minesweeper.py        ← refactored to use MinesweeperLogic
        └── Minesweeper_manager.py
```

### train_stage1.py Responsibilities

| Function | Description |
|----------|-------------|
| Create game | `MinesweeperLogic(10, 10, 10)` |
| Create agent | `Stage1SACAgent` (GridEncoder + HierarchicalAttention + SAC) |
| Training loop | Run N episodes, each step calls `logic.click()` |
| Checkpoint | Every 50 episodes + on exit |
| Output | `models/stage1_weights.pth` (for Stage 2 to load) |
| Logging | Episode reward, win rate, loss per episode |

### Main Flow

```
train_stage1.py
│
├── Init
│   ├── MinesweeperLogic(rows=10, cols=10, mines=10)
│   ├── Stage1SACAgent(state_channels=12, grid_size=10)
│   └── Load previous checkpoint (if exists)
│
├── Training Loop (N episodes)
│   ├── logic.reset()
│   ├── while not done:
│   │   ├── state = logic.get_grid_state_tensor()      # (12, 10, 10)
│   │   ├── action = agent.select_action(state)         # (x, y) ∈ [0, 1]
│   │   ├── row, col = action_to_grid(action, size=10)
│   │   ├── result = logic.click(row, col)
│   │   ├── reward = compute_reward(result)
│   │   ├── next_state = logic.get_grid_state_tensor()
│   │   ├── agent.store(state, action, reward, next_state, done)
│   │   └── agent.train_step()
│   ├── agent.on_episode_end()
│   └── log(episode, total_reward, win/lose, steps)
│
├── Save
│   └── models/stage1_weights.pth
│
└── Print summary (win rate, avg reward)
```

### New Classes Needed in RL_Agent.py

| Class | Description |
|-------|-------------|
| `GridEncoder` | 3× ConvTranspose2d, maps (12, 10, 10) → (128, 80, 80) |
| `Stage1ActorNetwork` | GridEncoder + HierarchicalAttentionHead + mean/log_std heads |
| `Stage1CriticNetwork` | GridEncoder + HierarchicalAttentionHead + Q-value heads |
| `Stage1SACAgent` | Uses Stage1Actor/Critic, manages training loop and checkpoints |
| `save_stage1_weights()` | Saves only HierarchicalAttention + SAC heads (not GridEncoder) |

### Stage 1 Replay Buffer

Same two-tier design as Stage 2, but much smaller per-entry:

| | Stage 1 | Stage 2 |
|--|---------|---------|
| State size per entry | 12×10×10 = 1,200 floats ≈ **2.4 KB** | 3×640×640 ≈ **2.4 MB** |
| BUFFER_CAPACITY | 300 (≈ 720 KB RAM) | 300 (≈ 720 MB RAM) |
| SAVE_CAPACITY | 150 (≈ 360 KB disk) | 150 (≈ 360 MB disk) |
| Storage | All in RAM (tiny) | State images on disk as .pt files |

Save triggers: Every 50 episodes + on program exit. Same stratified random sampling by reward.

### Stage 1 Checkpoint Files

```
models/
  ├── stage1_actor.pth           # Stage1ActorNetwork full state
  ├── stage1_critic.pth          # Stage1CriticNetwork full state
  ├── stage1_critic_target.pth   # Target network
  ├── stage1_log_alpha.pth       # Entropy coefficient
  ├── stage1_training_state.pth  # Optimizers, step counter, episode count, buffer index
  ├── stage1_replay_buffer/      # Runtime buffer (tiny files)
  ├── stage1_replay_buffer_save/ # Persistent save
  └── stage1_weights.pth         # Transfer weights (HierarchicalAttention + SAC heads only)
```

`stage1_weights.pth` is the ONLY file Stage 2 needs. All other `stage1_*` files are for Stage 1 resume only.

## All Design Questions Resolved

No open questions remain. Ready for implementation when authorized.
