# Training Pipeline

## Stage 1: Discrete Pre-training (Local, GTX 1050 Ti)

Two training scripts exist for Stage 1, targeting different architectures:

### `train_stage1.py` — Continuous SAC + Attention

- **Agent**: `Stage1SACAgent` (GridEncoder + HierarchicalAttentionHead + continuous SAC)
- **Input**: Grid state tensor `(12, 10, 10)` via `MinesweeperLogic` API (no GUI)
- **Action**: Continuous (x, y) ≈ [-0.05, 1.05] via ScaledSigmoid, mapped to grid coordinates
- **Purpose**: Validate that HierarchicalAttention + SAC can learn Minesweeper; pre-train weights for Stage 2
- **Output**: `models/stage1/stage1_weights.pth` (HierarchicalAttention + SAC heads, for Stage 2)
- **Monitoring**: `tensorboard --logdir runs/stage1/`, CSV at `models/stage1/training_log.csv`

### `train_stage1_simple.py` — Discrete SAC + MLP

- **Agent**: `SimpleDiscreteAgent` (Flatten → MLP → 100 discrete actions)
- **Input**: Same grid state tensor via `MinesweeperLogic` API
- **Action**: Discrete [0, 99] mapped to 10×10 grid cells, with action masking
- **Purpose**: Diagnose whether learning failures come from RL pipeline vs network architecture
- **Output**: `models/stage1_simple/` (checkpoint only, no transfer to Stage 2)
- **Monitoring**: `tensorboard --logdir runs/stage1_simple/`, CSV at `models/stage1_simple/training_log.csv`

### Execution

```bash
cd autoTest_pytorch

# Attention-based (main experiment)
python train_stage1.py

# Simple MLP (diagnostic)
python train_stage1_simple.py
```

### Evaluation

Both scripts run periodic evaluation (every 100 episodes, 10 eval episodes) with deterministic policy (no noise). Metrics: avg reward, win rate, avg steps, invalid click rate.

## Stage 2: Visual Training (Planned)

### Phase 1: Data Collection (Local, GTX 1050 Ti)

- Run game with GUI and agent in inference-only mode
- Capture screenshots as state, execute mouse clicks as actions
- Save experience tuples to disk-backed replay buffer (`ReplayBuffer`)

### Phase 2: Training (Cloud, GCP)

- Upload replay buffer data to GCP
- Load Stage 1 pre-trained weights for HierarchicalAttentionHead + SAC heads
- Load COCO pretrained weights for YOLO11n backbone
- Fine-tune entire pipeline end-to-end (nothing frozen)
- Export trained weights

### Phase 3: Deployment (Local, GTX 1050 Ti)

- Download trained weights
- Optionally convert to TensorRT `.engine` format for faster inference
- Run agent in inference mode with updated policy via `Demo_test_Minesweeper.py`

### Weight Transfer (Stage 1 → Stage 2)

| Component | Source | Frozen? |
|-----------|--------|---------|
| GridEncoder | Discarded | — |
| YOLO11nBackbone | COCO pretrained | No |
| HierarchicalAttentionHead | Stage 1 pre-trained | No |
| SAC Actor/Critic heads | Stage 1 pre-trained | No |

## Replay Buffer Design

### Per-Class Circular Buffer (Stage 1)

Each reward value gets its own circular buffer (`max_per_class` entries). Sampling draws equally from each class to guarantee balanced training data.

- **Capacity**: 2000 per class (in-memory, ~2.4 KB/entry for grid state)
- **Save**: 150 entries total, stratified by reward class, saved every 50 episodes + on exit
- **Load**: On restart, persistent entries are loaded back into per-class buffers

### Disk-Backed Buffer (Stage 2)

Large screenshot tensors stored as half-precision `.pt` files on disk.

- **Capacity**: 2000 entries (~2.4 MB/entry)
- **Save**: 150 entries on disk for GCP upload

## Reward Design (Minesweeper)

| Event | Reward | Notes |
|-------|--------|-------|
| Valid click (board changes) | +3 | Effective click that reveals new cell(s) |
| Invalid click (already revealed/flagged) | -2 (attention) / -2.95 (MLP) | MLP uses harsher penalty |
| Click outside grid bounds | -3 | Stage 1 attention only (ScaledSigmoid out-of-bounds) |
| Hit mine (lose) | -3 | Kept low to encourage exploring inside the grid |
| Win | +20 | Strong positive terminal signal |

## Checkpoint Files

### Stage 1 (Attention)

```
models/stage1/
├── stage1_actor.pth             # Stage1ActorNetwork full state
├── stage1_critic.pth            # Stage1CriticNetwork full state
├── stage1_critic_target.pth     # Target network
├── stage1_optimizer_state.pth   # Optimizers, step counter, episode count, valid rates
├── stage1_training_state.pth    # Persistent replay buffer entries
├── stage1_weights.pth           # Transfer weights (Attention + SAC heads only)
└── training_log.csv             # Training metrics
```

### Stage 1 (Simple MLP)

```
models/stage1_simple/
├── actor.pth                    # SimpleActorNetwork
├── critic.pth                   # SimpleCriticNetwork
├── critic_target.pth            # Target network
├── log_alpha.pth                # Learnable alpha
├── optimizer_state.pth          # Optimizers, step counter
├── training_state.pth           # Persistent replay buffer entries
└── training_log.csv             # Training metrics
```

### Stage 2

```
models/
├── actor.pth                    # SACActorNetwork (YOLO + Attention + policy)
├── critic.pth                   # SACCriticNetwork
├── critic_target.pth            # Target network
├── log_alpha.pth                # Learnable alpha
├── optimizer_state.pth          # Optimizers, step counter
├── replay_buffer/               # Runtime buffer (disk-backed .pt files)
└── replay_buffer_save/          # Persistent save for GCP upload
```
