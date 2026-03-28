# Current Implementation (TD3 + ResNet18)

## RL Algorithm: TD3 (Twin Delayed DDPG)

The current agent (`RL_Agent.py`) uses **TD3** — a deterministic policy gradient method with:

- **Twin Critics**: Two Q-networks to reduce overestimation bias
- **Delayed Policy Updates**: Actor updates every 2 critic updates (`POLICY_FREQ = 2`)
- **Target Policy Smoothing**: Noise added to target actions (`POLICY_NOISE = 0.2`)
- **Soft Target Updates**: Polyak averaging with `TAU = 0.005`

### Network Architecture

| Component | Backbone | Head |
|-----------|----------|------|
| Actor | ResNet18 (pretrained, all layers) → 512-d | FC(512→256) → FC(256→2) → Tanh |
| Critic (×2) | ResNet18 (pretrained, all layers) → 512-d | Concat(512+2) → FC(514→256) → FC(256→1) |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Actor LR | 1e-4 |
| Critic LR | 1e-3 |
| Gamma | 0.99 |
| Replay Buffer | 10,000 |
| Input Size | Variable (no resize — uses original screenshot dimensions) |
| Exploration Noise | Gaussian, σ=0.1 |

## Known Issues

1. **ResNet18 is NOT frozen** — all backbone parameters are trainable, making the model very large for the task and slow to converge.
2. **Variable input size** — the transform pipeline has no resize step (commented out), which means different screenshot sizes will cause dimension mismatches in the FC layers.
3. **Replay buffer stores full tensors on CPU** — memory-intensive for high-resolution screenshots.
4. **No entropy regularization** — TD3 is a deterministic policy; exploration relies solely on additive Gaussian noise, which decays poorly.
5. **`store_transition` defined twice** — duplicate method definition in `RL_Agent.py` (lines 193 and 330). The second definition overwrites the first.
6. **Saving model every train step** — `save_model()` is called on every `train_step()`, causing heavy I/O.

## Reward Structure (Minesweeper)

| Event | Reward |
|-------|--------|
| Valid click (screen changed) | +8, +10, +12, ... (escalating) |
| Invalid click (out of bounds) | -12 |
| No screen change (timeout) | -10 |
| Hit mine (lose) | -8 |
| Win | +15 |

## Noise Probability

Noise is applied probabilistically based on cumulative success:
- Starts at 90% noise probability
- After round 30, adjusts based on `positive_reward / (positive + negative)` ratio
