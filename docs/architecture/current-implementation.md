# Current Implementation Status

## Active Development Branch: `VisionDatasetRecorder_encoder+decoder`

Training reward_mean is slowly increasing and td_error_mean is decreasing, indicating the FQF distributional Q-network is learning.

## What Exists Today

### Two Agent Variants

| Agent | File | Architecture | Action Space | Status |
|-------|------|-------------|-------------|--------|
| `TransformerDiscreteAgent` | `transformer_discrete_agent.py` | Encoder-Decoder Transformer + FQF | Discrete 10×10 = 100 cells | Stage 1 — grid-state training |
| `VisualDiscreteAgentV3` | `visual_discrete_agent_v3.py` | YOLO (frozen) + Encoder + Decoder + FQF | Discrete 6×6 = 36 cells | Stage 2 — screenshot training, actively iterating |

---

## Stage 1: Transformer Discrete Agent (`transformer_discrete_agent.py`)

**RL Algorithm**: FQF (Fully parameterized Quantile Function) distributional Q-learning

- **Input**: Grid state `(B, 12, 10, 10)` — 12-channel one-hot per cell
- **Token embed**: `Linear(12 → 64)` + `TwoDimensionalPositionEmbedding(10, 10, 64)`
- **Query tokens**: 100 learned query vectors, 10×10 grid (no positional encoding on queries)
- **Core**: `EncoderDecoderTransformer` (pre-LN, GELU) — `d_model=64, nhead=4, num_layers=4, ff_dim=256`
- **Head**: `FQFQNetwork(d_model=64, num_fractions=8)` → Q-values `(B, 100)` over 10×10 cells
- **Action masking**: hard mask via state channel 0 (unrevealed cells only), `-1e8` for masked logits

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| LR | 5e-4 |
| Gamma | 0.9 |
| Target Update Freq | 50 steps |
| FQF Fractions | 8 |
| FQF Entropy Coef | 1e-3 |
| Buffer Capacity (PER) | 10,000 |
| Save Capacity | 2,000 |
| Save Every N Episodes | 50 |

---

## Stage 2: Visual Discrete Agent V3 (`visual_discrete_agent_v3.py`)

**RL Algorithm**: FQF distributional Q-learning with PER (Prioritized Experience Replay)

### Network Architecture

```
Screenshot (B, 3, 640, 640)
    ↓ YOLOEncoderBase  (FROZEN — loaded from YOLOGridStatePredictor checkpoint)
      YOLO11n backbone → (B, 128, 40, 40)
      token_adapter: LayerNorm(128) + Linear(128→128)
      + fixed 2D sinusoidal positional encoding
    (B, 1600, 128)
      HierarchicalEncoder [128→64→32]  (3 pre-LN self-attn blocks, FROZEN)
encoded memory (B, 1600, 32)
    ↓ TransformerDecoder × 5 layers  (pre-LN, cross-attention)
      36 learned query tokens (no positional encoding on queries)
decoded features (B, 36, 32)
    ↓ FQFQNetwork (d_model=32, num_fractions=8)
Q-values (B, 36) → masked argmax → grid cell (row, col in 6×6)
```

**Frozen**: YOLO backbone + token_adapter + HierarchicalEncoder (BN in eval mode)

**Trainable**: TransformerDecoder + 36 query tokens + FQFQNetwork head

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Grid | 6×6 = 36 actions |
| Batch Size | 32 |
| LR (decoder + head) | 5e-5 |
| LR warmup | 2,000 steps (0 → 5e-5 linearly) |
| Gamma | 0.9 |
| Target Update Freq | 1,000 steps |
| FQF Fractions | 8 |
| Grad Clip Norm | 5.0 |
| Buffer Capacity | 2,048 |
| Save Capacity | 256 |
| Minimum Data Before Training | 500 |

### Reward Structure

| Event | Reward |
|-------|--------|
| Valid click (board changes) | +0.5 |
| Invalid click (already revealed/flagged) | -0.25 |
| Hit mine (lose) | -0.7 |
| Win | +1.0 |

Reward values centralized in `model_structure/reward_settings.py` (`MINESWEEPER_REWARD_CONFIG`).

### Replay Buffer

`CategorizedReplayBuffer` with PER:

| Parameter | Value |
|-----------|-------|
| Capacity | 2,048 |
| Overflow buffer | 256 |
| PER alpha | 0.6 |
| Uniform mix ratio | 0.2 |
| Priority min / max | 0.05 / 5.0 |
| Age decay | 0.002 |

Persistent save: 256 entries to disk every 50 episodes + on exit.

### Checkpoint Files (`models/visual_transformer_v3_6x6/`)

| File | Content |
|------|---------|
| `decoder.pth` | TransformerDecoder weights + query token parameters |
| `fqf_head.pth` | FQFQNetwork weights |
| `target_decoder.pth` | Target network decoder |
| `target_fqf_head.pth` | Target network FQF head |
| `training_state.pth` | Optimizer states, step counter, episode count |
| `replay_buffer/` | Runtime PER buffer |
| `replay_buffer_save/` | Persistent saved buffer |

---

## Shared Components

### `model_structure/yolo_encoder_base.py` — `YOLOEncoderBase`

Shared between `YOLOGridStatePredictor` (supervised) and `VisualDiscreteAgentV3` (RL).  
Subclasses call `encode(screenshot)` to get memory tokens `(B, 1600, 32)` ready for cross-attention.

### `model_structure/transformer_shared.py`

| Class | Role |
|-------|------|
| `TwoDimensionalPositionEmbedding` | Learned 2D pos embedding (row + col embed concat) |
| `FixedSinusoidalPositionEmbedding` | Fixed 2D sinusoidal encoding — cached by (H, W, device, dtype) |
| `EncoderDecoderTransformer` | Pre-LN encoder + decoder used by Stage 1 |
| `FQFQNetwork` | FQF head: learned quantile fractions + cosine embedding → per-action Q-values |
| `DuelingQNetwork` | Legacy dueling head (not used in current active agents) |

### `model_structure/CategorizedReplayBuffer.py`

Prioritized replay buffer used by both agents. Supports stratified persistent save/load.

---

## Recent Iteration History

1. **SAC + continuous (x, y)** — abandoned; no convergence signal
2. **SAC-Discrete + MLP** — diagnostic baseline
3. **SAC + HierarchicalAttentionHead** — attention architecture experiments
4. **FQF + Transformer encoder-decoder** — current approach; discrete 10×10 (Stage 1) and 6×6 (Stage 2) grids
5. **pre-LN layers** — switched from post-LN for training stability
6. **Cross-attention decoder (5 layers)** — added to compress 1600 memory tokens into 36 action-specific embeddings
7. **Removed query positional encoding** — simplified; position info already in memory tokens
8. **Gamma moved to `reward_settings.py`** — single source of truth for all agents
9. **Removed Minesweeper bind with torch** — `MinesweeperLogic` no longer imports torch directly
