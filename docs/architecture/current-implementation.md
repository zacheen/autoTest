# Current Implementation Status

## Active Development Branch: `_test_replace_attention`

The project is iterating on Stage 1 pre-training to validate whether SAC can learn Minesweeper before investing in the full visual pipeline.

## What Exists Today

### Three Agent Variants in `RL_Agent.py`

| Agent | Architecture | Action Space | Status |
|-------|-------------|-------------|--------|
| `Stage1SACAgent` | GridEncoder + HierarchicalAttention + SAC | Continuous (x, y) via ScaledSigmoid | Iterating — testing attention architecture |
| `SimpleDiscreteAgent` | Flatten → MLP (256→256→100) + SAC-Discrete | Discrete (100 cells) with action masking | Diagnostic — validates RL pipeline |
| `SACAgent` | YOLO11n + HierarchicalAttention + SAC | Continuous (x, y) via Tanh | Stage 2 — code exists, waiting for Stage 1 success |

### Stage 1 SAC (Continuous, with Attention)

**RL Algorithm**: SAC (Soft Actor-Critic) with valid-rate-based alpha

- **Actor**: GridEncoder → (128, 80, 80) → HierarchicalAttentionHead → 256-d → Gaussian policy → ScaledSigmoid → (x, y) ≈ [-0.05, 1.05]
- **Critic**: Separate GridEncoder + HierarchicalAttentionHead → 256-d + action → twin Q-values
- **Alpha**: Not auto-tuned; set by `ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * avg_valid_rate` (sliding window of 50 episodes)
- **Replay buffer**: Per-class circular buffer (`Stage1ReplayBuffer`) with balanced sampling across reward classes

### Simple MLP (Discrete, No Attention)

**RL Algorithm**: SAC-Discrete with auto-alpha (clamped)

- **Actor**: Flatten(1200) → FC(256) → FC(256) → FC(100) → softmax (with action masking from state channel 0)
- **Critic**: Same MLP architecture → 100 Q-values (twin)
- **Alpha**: Auto-tuned with `target_entropy = 0.8 * ln(100) ≈ 3.7`, clamped to [0.05, 0.3]
- **Replay buffer**: Same `Stage1ReplayBuffer` with per-class balanced sampling

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Actor LR | 3e-4 |
| Critic LR | 3e-4 |
| Alpha LR (discrete) | 1e-5 |
| Gamma | 0.99 |
| TAU | 0.005 |
| Buffer Capacity | 2000 per class |
| Save Capacity | 150 |
| Save Every N Episodes | 50 |

### Reward Structure (Both Agents)

| Event | Reward |
|-------|--------|
| Valid click (board changes) | +3 |
| Invalid click (already revealed/flagged) | -2 to -2.95 |
| Click outside grid bounds | -3 (Stage1 only) |
| Hit mine (lose) | -3 |
| Win | +20 |

### Checkpoint & Resume

Both agents save on every episode end (`_save_model()`) and periodically save replay buffer (`save_persistent()`). Full resume on restart via `try_load_model()`.

## Recent Iteration History

1. **Initial**: SpatialAttentionHead (weighted pooling) — destroyed spatial info, policy collapsed after ~1200 episodes
2. **Changed to**: HierarchicalAttentionHead (local + global attention) — preserves spatial reasoning
3. **Added**: ScaledSigmoid activation for direct [0, 1] grid coordinate mapping
4. **Added**: Per-class replay buffer for balanced training data
5. **Added**: Simple MLP experiment to diagnose whether failure is in architecture vs RL pipeline
6. **Current branch**: `_test_replace_attention` — testing discrete I/O to check if RL can learn at all
7. **Latest fix**: Protect positive experiences in replay buffer from overwrite
