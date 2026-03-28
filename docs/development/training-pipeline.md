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

### Limitations

- Training competes with inference for GPU resources
- Replay buffer is in-memory only (lost on restart)
- No way to batch train on collected data separately

## Proposed Pipeline (Local + Cloud)

> This is part of the new SAC design — not yet implemented.

### Phase 1: Data Collection (Local, GTX 1050 Ti)

- Run game and agent in inference-only mode
- Save experience tuples (screenshot, action, reward, next_screenshot, done) to disk
- Format: Replay Buffer serialized to files

### Phase 2: Training (Cloud, GCP)

- Upload replay buffer data to GCP
- Run off-policy SAC training on high-compute instances
- Export trained weights

### Phase 3: Deployment (Local, GTX 1050 Ti)

- Download trained weights
- Optionally convert to TensorRT `.engine` format for faster inference
- Run agent in inference mode with updated policy

## Reward Design (Minesweeper)

| Event | Current Reward | Notes |
|-------|---------------|-------|
| Valid click (board changes) | +8, +10, +12... | Escalates with consecutive valid clicks |
| Click outside game region | -12 | Penalizes invalid positions |
| No board change (timeout) | -10 | Penalizes ineffective clicks |
| Hit mine (lose) | -8 | Game over |
| Win | +15 | Game over |
