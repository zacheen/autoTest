# Training Pipeline

## Current Pipeline (Local Only)

The current implementation trains **online** — the agent collects experience and trains in the same process on the same machine.

```
Game Play (1050 Ti)
    │
    ├── Screenshot capture
    ├── Agent inference (select_action)
    ├── Mouse click execution
    ├── Reward computation (screenshot diff, win/lose check)
    ├── Store transition in ReplayBuffer (in-memory)
    └── Train step (every step, if buffer has enough samples)
```

### Checkpoint & Resume

### Model Weights (saved every `train_step()`)
- `actor.pth`, `critic.pth`, `critic_target.pth`, `log_alpha.pth`

### Two-Tier Replay Buffer
- **Runtime**: `replay_buffer/` — up to 600 entries (BUFFER_CAPACITY) in CPU RAM, circular overwrite
- **Persistent**: `replay_buffer_save/` — 150 entries (SAVE_CAPACITY) on disk, stratified random by reward
- **Save triggers**: Every 50 episodes + on program exit (`atexit`)
- **Training state**: `training_state.pth` (optimizer states, step counter, episode counter, persistent buffer index)

On restart, `SACAgent.__init__()` → `try_load_model()` loads persistent 150 entries into runtime buffer slots 0-149, then new data fills slots 150-599.

### Limitations

- Training competes with inference for GPU resources
- No way to batch train on collected data separately

## Two-Stage Training Pipeline

> This is part of the new SAC design — not yet implemented.

### Stage 1: Discrete Pre-training (Local, GTX 1050 Ti)

- **Script**: `train_stage1.py` (standalone, does not touch `Demo_test_Minesweeper.py`)
- Uses `MinesweeperLogic` API directly (no GUI, no screenshots)
- Input: Grid state tensor `(12, 10, 10)` — one-hot encoded cell states
- Pipeline: `GridEncoder → (128, 80, 80) → SpatialAttentionHead → (256) → SAC Actor/Critic`
- Purpose: Validate SAC can learn Minesweeper; pre-train SpatialAttentionHead + SAC heads
- Replay buffer: BUFFER_CAPACITY=300, SAVE_CAPACITY=150 (same as Stage 2, but ~2.4 KB/entry vs ~2.4 MB/entry)
- Output: `models/stage1_weights.pth` (SpatialAttention + SAC heads only, for Stage 2 to load)
- Lightweight — runs entirely on local GPU

### Stage 2: Visual Training (Local + Cloud)

#### Phase 1: Data Collection (Local, GTX 1050 Ti)

- Run game with GUI and agent in inference-only mode
- Capture screenshots as state, execute mouse clicks as actions
- Save experience tuples to replay buffer on disk

#### Phase 2: Training (Cloud, GCP)

- Upload replay buffer data to GCP
- Load Stage 1 pre-trained weights for SpatialAttentionHead + SAC heads
- Load COCO pretrained weights for YOLO11n backbone
- Fine-tune entire pipeline end-to-end (nothing frozen)
- Export trained weights

#### Phase 3: Deployment (Local, GTX 1050 Ti)

- Download trained weights
- Optionally convert to TensorRT `.engine` format for faster inference
- Run agent in inference mode with updated policy

### Weight Transfer (Stage 1 → Stage 2)

| Component | Source | Frozen? |
|-----------|--------|---------|
| GridEncoder | Discarded | — |
| YOLO11nBackbone | COCO pretrained | No |
| SpatialAttentionHead | Stage 1 pre-trained | No |
| SAC Actor/Critic heads | Stage 1 pre-trained | No |

## Reward Design (Minesweeper)

| Event | Reward | Notes |
|-------|--------|-------|
| Valid click (board changes) | +2, +4, +6... (escalating by +2) | Rewards sustained good play; resets each episode |
| Invalid click (no screen change) | -1 | Includes clicking revealed cells, flagged cells |
| Click outside game region | -1 | Treated same as invalid click |
| Hit mine (lose) | -10 | Game over |
| Win | +20 | Game over |
