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

| Event | Reward | Notes |
|-------|--------|-------|
| Valid click (board changes) | +2, +4, +6... (escalating by +2) | Rewards sustained good play; resets each episode |
| Invalid click (no screen change) | -1 | Includes clicking revealed cells, flagged cells |
| Click outside game region | -1 | Treated same as invalid click |
| Hit mine (lose) | -10 | Game over |
| Win | +20 | Game over |
