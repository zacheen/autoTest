# Session Summary: Generalized Turn-Based Game Agent (SAC Version)
**Date**: 2026-03-28 to 2026-03-30
**Project**: Visual RL agent for mouse-driven games, targeting Minesweeper (10×10, 10 mines)

---

## 1. Main Objective

Build a two-stage RL agent:
- **Stage 1**: Train SAC on discrete grid state (12-channel one-hot tensor) to validate the agent can learn Minesweeper
- **Stage 2**: Transfer learned weights to visual pipeline (YOLO11n backbone + screenshots)

---

## 2. Architecture Evolution

### Final Architecture (Stage 1 — Simple MLP experiment)
```
Grid State (12, 10, 10)
  → Flatten (1200)
  → FC(256) → SiLU → FC(256) → SiLU → FC(100) → action masking → softmax
  → Discrete action: pick 1 of 100 grid cells
```

### Full Architecture (Stage 1 — not yet validated)
```
Grid State (12, 10, 10)
  → GridEncoder (ConvTranspose ×3) → (128, 80, 80)
  → HierarchicalAttentionHead:
      Local Attn L1 (8×8, 128→64)
      Local Attn L2 (8×8, 64→32)
      Local Attn L3 (8×8, 32→32, residual)
      Global Self-Attn (Flash Attn, 32→16)
      Conv Downsample → FC → (256,)
  → SAC Actor head → action
```

### Stage 2 (design only, not implemented)
```
Screenshot (3, 640, 640)
  → YOLO11n (pretrained, fine-tuned) → (128, 80, 80)
  → HierarchicalAttentionHead (loaded from Stage 1)
  → SAC Actor head (loaded from Stage 1)
```

---

## 3. Problems Encountered and Solutions

### Problem 1: Policy Collapse — Invalid Rate → 99%
- **Root Cause**: SpatialAttentionHead collapsed (128, 80, 80) → (256,) via weighted pooling, destroying all spatial information. Agent couldn't distinguish which cell to click.
- **Solution**: Replaced with HierarchicalAttentionHead (3× Local Attention + Global Self-Attention + Conv Downsample). Preserves spatial reasoning before compressing.
- **Status**: Implemented but not yet validated (moved to Simple MLP for faster iteration).

### Problem 2: Continuous Action Space Failure
- **Root Cause**: Continuous (x, y) output mapped to 10×10 grid via `int(x*10)`. Small output changes land on same cell → no gradient signal. Invalid clicks don't change state → same output → dead loop.
- **Solution**: Switched to discrete action space (100 cells, Categorical distribution).
- **Tested**: 3 architectures (Continuous+Attention, Discrete+Attention, Simple MLP) — all failed without action masking.

### Problem 3: Entropy Collapse (alpha → 0)
- **Root Cause 1**: `TARGET_ENTROPY = -2.0` was for continuous actions. Discrete entropy is always ≥ 0, so `entropy > target` was always true → alpha decreased indefinitely.
- **Root Cause 2**: Valid-rate-based alpha didn't respond to entropy collapse — alpha stayed at 0.27 while entropy dropped to 0.
- **Root Cause 3**: `LR_ALPHA = 3e-4` too fast — alpha dropped from 0.2 to 0.05 in first train step.
- **Solution**: Auto-alpha with correct discrete target (`0.8 × ln(100) ≈ 3.7`), slower learning rate (`LR_ALPHA_DISCRETE = 1e-5`), and clamp (`alpha ∈ [0.05, 0.3]`).

### Problem 4: Replay Buffer Imbalance
- **Root Cause**: Single circular buffer (size 2000) flooded with negative reward entries. Rare positive experiences (valid clicks, wins) overwritten within 10 episodes. Even with balanced sampling, empty reward groups can't be sampled.
- **Solution**: Per-class circular buffers — each reward value gets its own independent buffer. Sampling draws equally from each class. No PER, no SumTree, no protection logic needed.

### Problem 5: Dead Loop — Same State → Same Action
- **Root Cause**: Invalid click doesn't change Minesweeper state. Same state → same policy output → same invalid action → infinite loop until MAX_STEPS (200).
- **Failed Attempts**: Attempted "click history" channel (channel 12) — added complexity without solving the core issue.
- **Solution**: Action masking — hard mask from state channel 0 (unrevealed cells). Uses `-1e8` (not `-inf`) to avoid `0 × (-inf) = NaN` in actor loss gradient.

### Problem 6: NaN in Training (0 × -inf)
- **Root Cause**: Action masking with `float('-inf')` → softmax outputs exact 0 for masked positions → `log_softmax = -inf` → `probs × log_probs = 0 × (-inf) = NaN` → gradients explode → all network weights become NaN.
- **Solution**: Use `-1e8` instead of `-inf`. Softmax outputs ~0 (not exactly 0), log_softmax outputs ~-1e8 (not -inf). Product is ~0, not NaN.

### Problem 7: Windows Multiprocessing Spawn Error
- **Root Cause**: `multiprocessing.Process` on Windows re-imports the caller's main module. `Demo_test_Minesweeper.py` imports heavy modules (YOLO, selenium) at top level → child process hangs.
- **Solution**: Changed `Minesweeper_manager` from `multiprocessing.Process` to `subprocess.Popen`. Child process runs `python Minesweeper.py` directly, doesn't re-import caller.

### Problem 8: First Click Always Valid (dilutes data)
- **Root Cause**: Minesweeper guarantees first click is safe (mines placed after first click). This valid-click experience has no learning value — any action would be valid.
- **Solution**: Skip storing first step in replay buffer (`if episode_steps > 0`).

---

## 4. Key Decisions and Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Action space | Discrete (100 cells) | Continuous (x,y) regression can't snap to grid cells; gradient signal too weak |
| Action masking | Hard mask from state channel 0 | Without masking, 6 experiments failed — agent can't learn to avoid revealed cells from reward signal alone |
| Replay buffer | Per-class circular buffers | Single buffer overwrites rare experiences; PER doesn't balance by reward type |
| Alpha tuning | Auto-alpha with clamp + correct discrete target | Valid-rate-based alpha doesn't respond to entropy; wrong target causes indefinite decrease |
| Backbone (Stage 1) | Simple MLP (for validation) | Complex architectures (GridEncoder + Attention) obscure whether RL pipeline works; validate with simplest model first |
| Game logic separation | MinesweeperLogic.py (pure logic) + Minesweeper.py (UI) | Enables headless Stage 1 training without GUI/screenshots |
| ScaledSigmoid | scale=1.1, shift=-0.05 → output ≈ [-0.05, 1.05] | Replaces tanh for continuous version; maps directly to [0,1] grid coordinates. Non-standard — user chose to keep despite it deviating from standard SAC |

---

## 5. Current Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| BATCH_SIZE | 32 | |
| LR_ACTOR | 3e-4 | |
| LR_CRITIC | 3e-4 | |
| LR_ALPHA_DISCRETE | 1e-5 | 30× slower than continuous to prevent alpha collapse |
| GAMMA | 0.99 | |
| TAU | 0.005 | Soft target update |
| ALPHA_MAX | 0.3 | |
| ALPHA_MIN | 0.05 | |
| DISCRETE_TARGET_ENTROPY | 0.8 × ln(100) ≈ 3.7 | 80% of max categorical entropy |
| BUFFER_CAPACITY | 2000 | Per-class buffer capacity |
| SAVE_CAPACITY | 150 | Persistent save to disk |
| GRID_STATE_CHANNELS | 12 | One-hot encoding channels |
| MAX_STEPS_PER_EPISODE | 200 | |
| Grid | 10×10, 10 mines | |

---

## 6. Reward Structure

| Event | Reward |
|-------|--------|
| Valid click (reveals new cell) | +3.0 |
| Invalid click (already revealed) | -2.95 |
| Hit mine (game over) | -3.0 |
| Win | +20.0 |
| First click | Skipped (not stored in buffer) |

---

## 7. File Structure

```
autoTest_pytorch/
├── RL_Agent.py                    # All model classes and agents
│   ├── SimpleActorNetwork         # MLP + action masking (active experiment)
│   ├── SimpleCriticNetwork        # MLP twin Q-networks
│   ├── SimpleDiscreteAgent        # SAC-Discrete with per-class buffer
│   ├── HierarchicalAttentionHead  # Local + Global attention (implemented, not validated)
│   ├── LocalAttentionLayer        # 8×8 window attention
│   ├── GlobalAttentionLayer       # Flash Attention on all positions
│   ├── GridEncoder                # ConvTranspose 10→80 upsampling
│   ├── Stage1ActorNetwork         # Continuous SAC actor (obsolete for now)
│   ├── Stage1CriticNetwork        # Continuous SAC critic (obsolete for now)
│   ├── Stage1SACAgent             # Continuous SAC agent (obsolete for now)
│   ├── Stage1ReplayBuffer         # Per-class circular buffer
│   ├── YOLO11nBackbone            # Stage 2 feature extractor
│   ├── SACActorNetwork            # Stage 2 actor (YOLO-based)
│   └── SACAgent                   # Stage 2 agent
├── train_stage1_simple.py         # Active training script (Simple MLP + discrete)
├── train_stage1.py                # Continuous SAC training (not active)
├── train_stage1_discrete.py       # Discrete + Attention training (not active)
├── training_logger.py             # TeeOutput for console + file logging
├── Minesweeper/
│   ├── MinesweeperLogic.py        # Pure game logic (no UI)
│   ├── Minesweeper.py             # Tkinter UI (delegates to MinesweeperLogic)
│   └── Minesweeper_manager.py     # Subprocess launcher
├── Demo_test_Minesweeper.py       # Stage 2 integration (screenshot-based)
└── models/
    └── stage1_simple/             # Active experiment checkpoints
```

---

## 8. Lessons Learned

1. **Validate RL pipeline with simplest model first** — Don't debug complex architectures (Attention, GridEncoder) until the basic RL loop works with a simple MLP.

2. **Discrete target entropy ≠ continuous target entropy** — `target = -action_dim` only works for continuous Gaussian. For discrete with N actions, use `target = fraction × ln(N)` where fraction ∈ [0.5, 0.9].

3. **Action masking is not cheating** — For grid games, preventing clicks on already-revealed cells is equivalent to what human players do. Without it, the agent wastes 99%+ of its experience on invalid actions.

4. **0 × (-inf) = NaN** — When masking logits with `-inf` before softmax, the resulting `probs × log_probs` produces NaN. Use `-1e8` instead.

5. **Per-class replay buffer > single buffer with protection** — When reward distribution is highly skewed, a single circular buffer overwrites rare experiences. Separate buffers per reward class guarantee balanced sampling with zero complexity.

6. **Auto-alpha learning rate matters** — For discrete SAC, `LR_ALPHA = 3e-4` can collapse alpha in one step. Use `1e-5` or slower.

7. **State must change between steps** — If an action doesn't change the environment state, the agent outputs the same action again. This creates dead loops that no amount of reward shaping can fix.

8. **Design before code** — User explicitly requested DESIGN MODE. All architecture decisions were discussed and approved before any code was written.

---

## 9. Next Session Starting Point

### Immediate Task
Run `train_stage1_simple.py` with current settings (action masking + per-class buffer + auto-alpha). Monitor:
- **Entropy**: Should stay near target (~3.7), NOT collapse to 0
- **Reward distribution**: Should be balanced across classes
- **Win rate**: Any wins indicate the agent is learning

### If Simple MLP Succeeds (win rate > 0%)
1. Replace Simple MLP with HierarchicalAttentionHead (already implemented in RL_Agent.py)
2. Validate attention-based architecture can also learn
3. Export Stage 1 weights for Stage 2

### If Simple MLP Fails
Investigate:
- Is entropy collapsing despite auto-alpha?
- Is the reward structure giving enough signal to distinguish safe vs dangerous cells?
- Consider curriculum learning (start with 3×3 grid, 1 mine)

### Stage 2 (not started)
- Load Stage 1 weights into YOLO-based architecture
- Train on screenshots instead of grid state tensors
- Requires YOLO11n backbone + visual action execution

### Open Questions
- ScaledSigmoid vs tanh: user chose ScaledSigmoid despite it being non-standard. May need to revisit for Stage 2.
- Hierarchical Attention VRAM: tested at 1.9GB (batch=32) on 1050 Ti (4GB). Leaves ~2GB for other components — may be tight in Stage 2 with YOLO.
