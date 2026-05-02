# Visual Agent V1, V2, and V3

This document explains the three implemented visual-agent versions in this repository:

- V1: `autoTest_pytorch/visual_discrete_agent.py`
- V2: `autoTest_pytorch/visual_discrete_agent_v2.py`
- V3: `autoTest_pytorch/visual_discrete_agent_v3.py`

The focus is the structure of each version, the main differences between them, and how each version conveys one data structure into the next. The current runtime entry point, `Demo_test_Minesweeper.py`, imports V3.

## Summary

| Version | Main design | Intermediate structure | Trainable decision path | Main benefit | Main risk |
|---|---|---|---|---|---|
| V1 | Direct visual transformer | Continuous YOLO visual tokens | YOLO feature extractor, adapter, decoder, FQF head | Cleanest pixels-to-action path | RL gradients can destabilize perception |
| V2 | Symbolic grid bridge | 12-channel Minesweeper grid | Stage 1 transformer and FQF head; YOLO update controlled separately | Most interpretable | Hard `argmax` grid conversion blocks smooth end-to-end learning |
| V3 | Frozen visual encoder plus trainable decoder | Frozen continuous encoder memory | Query tokens, transformer decoder, FQF head | More stable hybrid design | Frozen encoder cannot adapt if it misses policy-critical information |

All versions ultimately perform the same task:

```text
screenshot -> board-aware representation -> per-cell Q-values -> selected click action
```

For the current Minesweeper setup:

- Screenshot input: `(3, 640, 640)`.
- Board size: `6 x 6`.
- Action count: `36`.
- Internal action form: `action_id` or `(row, col)`.
- Final UI action: a pixel click coordinate produced by the surrounding game/test harness.
- Q-value head: `FQFQNetwork`, which estimates quantile values and converts them into expected Q-values.

## V1: Direct Visual Transformer

### File

`autoTest_pytorch/visual_discrete_agent.py`

### Purpose

V1 is the direct visual reinforcement-learning design. It tries to learn from screenshots to cell actions without first creating an explicit symbolic board state. The model extracts YOLO features, turns them into transformer memory tokens, decodes one token per board cell, and predicts Q-values for those cells.

### Detailed Structure

```text
Screenshot
  shape: (B, 3, 640, 640)
  meaning: RGB game-board image

YOLO11nLastFeatureExtractor
  source: YOLO11n layers 0..6
  output shape: (B, 128, 40, 40)
  meaning: dense visual feature map

Flattened YOLO tokens
  conversion: (B, 128, 40, 40) -> (B, 1600, 128)
  meaning: one token per YOLO feature-map position

Token adapter
  structure: LayerNorm(128) -> Linear(128, 64) -> GELU -> Linear(64, 64)
  output shape: (B, 1600, 64)
  meaning: visual tokens are converted into transformer dimension

Learned memory position embedding
  shape: (1600, 64)
  meaning: row/column identity for each visual token

Encoder-decoder transformer
  encoder input: (B, 1600, 64)
  decoder query input: (B, 36, 64)
  decoder output: (B, 36, 64)
  meaning: 36 learned cell queries attend to 1600 visual memory tokens

FQFQNetwork
  input shape: (B, 36, 64)
  output q_values shape: (B, 6, 6)
  meaning: estimated value of clicking each cell

Action selection
  conversion: (B, 6, 6) -> 36 flattened scores -> masked argmax
  output: action_id / (row, col)
```

### Structure Conversion

V1 conveys data through these structures:

```text
(B, 3, 640, 640)
  -> (B, 128, 40, 40)
  -> (B, 1600, 128)
  -> (B, 1600, 64)
  -> (B, 36, 64)
  -> (B, 6, 6)
  -> action
```

The most important conversion is:

```text
1600 visual tokens -> 36 board-cell decision tokens
```

That conversion is performed by the transformer decoder. Each learned cell query attends over the full visual memory and extracts a feature vector for one playable board cell.

### Training Structure

The replay buffer stores screenshot-based transitions:

```text
transition = {
  state: screenshot tensor,
  action: row/col or action id,
  next_state: next screenshot tensor,
  reward: scalar reward,
  done: terminal flag
}
```

During training, both `state` and `next_state` pass through the visual backbone. The online branch trains the visual backbone and FQF head; the target branch supplies target quantiles for Q-learning. This is powerful, but it means sparse RL loss can update the visual extractor directly.

### Practical Meaning

V1 is the purest "learn everything from pixels" design. Its weakness is stability: early bad Q-learning signals can push YOLO features, adapter layers, and decoder attention in the wrong direction.

## V2: Symbolic Grid Bridge

### File

`autoTest_pytorch/visual_discrete_agent_v2.py`

### Purpose

V2 introduces an explicit symbolic board-state structure between screenshot pixels and the policy. A supervised `YOLOGridStatePredictor` first converts the screenshot into a 12-channel grid state. That symbolic grid is then passed into the Stage 1 `TransformerDiscreteAgent`, which already maps grid states to cell Q-values.

### Detailed Structure

```text
Screenshot
  shape: (B, 3, 640, 640)
  meaning: RGB game-board image

YOLOGridStatePredictor
  input: screenshot
  output logits shape: (B, 12, H, W)
  output one-hot grid shape after argmax/scatter: (B, 12, 6, 6)
  meaning: predicted Minesweeper state for every cell

12-channel symbolic grid
  channel 0: unrevealed
  channel 1: flagged
  channels 2..10: revealed numbers 0..8
  channel 11: mine

Stage 1 transformer backbone
  token embedding: 12 channels -> d_model=64
  position embedding: learned 2D board position
  transformer: 4 encoder-decoder layers, 4 heads
  output shape: (B, 36, 64)
  meaning: one policy feature vector per cell

FQFQNetwork
  input shape: (B, 36, 64)
  output q_values shape: (B, 6, 6)
  meaning: expected click value for each board cell

Action selection
  conversion: (B, 6, 6) -> 36 flattened scores -> masked argmax
  output: action_id / (row, col)
```

### Structure Conversion

V2 conveys data through these structures:

```text
(B, 3, 640, 640)
  -> YOLOGridStatePredictor logits
(B, 12, 6, 6)
  -> class argmax per cell
(B, 1, 6, 6)
  -> one-hot symbolic grid
(B, 12, 6, 6)
  -> Stage 1 transformer features
(B, 36, 64)
  -> FQF Q-values
(B, 6, 6)
  -> action
```

The key conversion is:

```text
screenshot pixels -> per-cell class labels -> one-hot board tensor
```

This makes V2 easy to inspect. If the agent fails, the failure can be split into two questions:

- Did the visual predictor create the correct 12-channel board state?
- Did the symbolic policy choose a reasonable action from that board state?

### Training Structure

V2 also stores screenshot transitions:

```text
transition = {
  state: screenshot tensor,
  action: row/col or action id,
  next_state: next screenshot tensor,
  reward: scalar reward,
  done: terminal flag
}
```

During `train_step`:

1. `state` and `next_state` pass through `YOLOGridStatePredictor`.
2. Predictor logits become class labels with `argmax`.
3. Class labels become one-hot grid tensors.
4. The Stage 1 transformer converts the one-hot grid into 36 cell features.
5. The FQF head computes current and target quantile values.
6. Replay priorities are updated from TD error.

The file includes `YOLO_UPDATE_ENABLED`. In the current code it is `False`, so YOLO is effectively kept from optimizer updates while diagnostics can still report whether gradients would exist.

### Practical Meaning

V2 is the most interpretable version. It separates perception from strategy. Its weakness is the hard symbolic bridge: `argmax` is useful for clarity, but it creates a discrete bottleneck that does not pass smooth learning signal back into the visual predictor.

## V3: Frozen Visual Encoder and Trainable Decoder

### File

`autoTest_pytorch/visual_discrete_agent_v3.py`

### Purpose

V3 keeps the useful supervised visual representation from `YOLOGridStatePredictor` but removes the hard symbolic-grid bottleneck from the RL decision path. It loads the encoder-side weights from `./models/yolo_grid_predictor/best.pth`, freezes the encoder components, then trains a new decoder and FQF head for action selection.

### Detailed Structure

```text
Screenshot
  shape: (B, 3, 640, 640)
  meaning: RGB game-board image

YOLOEncoderBase
  loaded from: ./models/yolo_grid_predictor/best.pth
  frozen components:
    - YOLO11nLastFeatureExtractor
    - token_adapter
    - HierarchicalEncoder

YOLO11nLastFeatureExtractor
  output shape: (B, 128, 40, 40)
  meaning: dense visual feature map

Flattened YOLO tokens
  conversion: (B, 128, 40, 40) -> (B, 1600, 128)
  meaning: one token per YOLO feature-map position

Token adapter
  structure: LayerNorm(128) -> Linear(128, 128)
  output shape: (B, 1600, 128)
  meaning: visual tokens are mapped to encoder dimension

Fixed 2D sinusoidal position embedding
  shape: (1600, 128)
  meaning: non-trainable row/column location signal

HierarchicalEncoder
  dimensions: 128 -> 64 -> 32
  output shape: (B, 1600, 32)
  meaning: compact continuous visual memory

Learned cell query tokens
  shape: (B, 36, 32)
  meaning: one trainable query per board cell

Transformer decoder
  layers: 5
  d_model: 32
  heads: 4
  output shape: (B, 36, 32)
  meaning: action-specific cell features queried from frozen visual memory

FQFQNetwork
  input shape: (B, 36, 32)
  num_fractions: 8
  output q_values shape: (B, 6, 6)
  meaning: expected click value for each cell

Action selection
  conversion: (B, 6, 6) -> 36 flattened scores -> masked argmax
  output: action_id / (row, col)
```

### Structure Conversion

V3 conveys data through these structures:

```text
(B, 3, 640, 640)
  -> YOLO feature map
(B, 128, 40, 40)
  -> flattened visual tokens
(B, 1600, 128)
  -> frozen encoded memory
(B, 1600, 32)
  -> trainable cell-query features
(B, 36, 32)
  -> FQF Q-values
(B, 6, 6)
  -> action
```

The key conversion is:

```text
supervised visual memory -> RL-specific cell features
```

Unlike V2, V3 does not force the visual information through a 12-channel class grid before policy inference. Unlike V1, V3 does not let RL gradients modify the visual encoder. The trainable part starts at the cell-query decoder.

### Training Structure

V3 uses the same broad replay transition form:

```text
transition = {
  state: screenshot tensor,
  action: row/col or action id,
  next_state: next screenshot tensor,
  reward: scalar reward,
  done: terminal flag
}
```

During `train_step`:

1. `state` and `next_state` pass through the frozen `YOLOEncoderBase`.
2. The trainable decoder converts frozen memory into 36 cell features.
3. The online FQF network produces current Q-values and quantiles.
4. The target FQF network produces target quantiles.
5. Quantile Huber loss trains the decoder and online FQF network.
6. FQF entropy regularization is included.
7. The target network is periodically synchronized from the online network.

The trainable path is:

```text
query tokens + transformer decoder + online FQFQNetwork
```

### Practical Meaning

V3 is the stabilized hybrid design. It preserves supervised visual learning from the predictor, avoids V2's hard symbolic bottleneck during policy inference, and keeps RL gradients away from the YOLO encoder.

## How The Versions Differ

### Difference in Intermediate Representation

```text
V1: visual tokens only
V2: explicit 12-channel symbolic grid
V3: frozen continuous encoder memory
```

V1 and V3 both use continuous visual features. V2 uses a human-readable symbolic grid.

### Difference in Perception Training Boundary

```text
V1: RL can update the visual backbone
V2: visual predictor is supervised first; YOLO update is controlled separately
V3: visual encoder is loaded from the predictor and frozen
```

V1 has the most flexible perception path, but also the highest instability risk. V3 has the strongest stability boundary.

### Difference in Policy Input

```text
V1 policy input: 36 visual cell tokens, d_model=64
V2 policy input: 36 symbolic-transformer cell tokens, d_model=64
V3 policy input: 36 decoder cell tokens, d_model=32
```

The FQF head always receives one feature vector per action cell. The difference is how those cell features are produced.

### Difference in Debugging

```text
V1 debugging: inspect visual gradients, attention behavior, Q-value spread
V2 debugging: inspect predicted board classes and symbolic-policy action
V3 debugging: inspect frozen encoder quality, decoder learning, Q-value spread
```

V2 is easiest to explain because the intermediate board state is explicit. V3 is usually better for stable RL training because it avoids updating the visual encoder.

## Version Evolution

### V1 to V2

V1:

```text
screenshot -> trainable visual transformer -> Q-values
```

V2:

```text
screenshot -> supervised grid-state predictor -> symbolic grid -> Stage 1 policy -> Q-values
```

The system moved from direct visual RL to an explicit symbolic bridge. This improved interpretability and allowed the already-trained Stage 1 policy to be reused.

### V2 to V3

V2:

```text
screenshot -> 12-channel grid classes -> Stage 1 transformer -> Q-values
```

V3:

```text
screenshot -> frozen encoder memory -> trainable decoder -> Q-values
```

The system kept the supervised vision benefit but removed the hard `argmax` symbolic bottleneck from the RL decision path.

## Detailed Shared Structures

### Screenshot

```text
single shape: (3, 640, 640)
batch shape: (B, 3, 640, 640)
meaning: current visual observation
used by: V1, V2, V3
```

### YOLO Feature Map

```text
shape: (B, 128, 40, 40)
meaning: spatial image feature map from YOLO11n layers 0..6
used by: V1 and V3 directly; V2 inside YOLOGridStatePredictor
```

### Symbolic Grid State

```text
shape: (B, 12, 6, 6)
meaning: one-hot cell class tensor
used by: V2 and Stage 1 symbolic policy

channels:
  0  = unrevealed
  1  = flagged
  2  = revealed 0
  3  = revealed 1
  4  = revealed 2
  5  = revealed 3
  6  = revealed 4
  7  = revealed 5
  8  = revealed 6
  9  = revealed 7
  10 = revealed 8
  11 = mine
```

### Cell Feature Tokens

```text
V1: (B, 36, 64)
V2: (B, 36, 64)
V3: (B, 36, 32)
meaning: one feature vector per clickable cell
```

### FQF Output

```text
q_values: (B, 6, 6)
quantiles: (B, 36, num_fractions)
fraction_probs: (B, num_fractions)
num_fractions: 8
meaning: distributional value estimate for each cell action
```

The agent flattens `q_values` into 36 action scores, masks blocked actions, and chooses the highest remaining score.

## Final Interpretation

V1 means:

```text
learn perception and policy together from pixels
```

V2 means:

```text
convert pixels into a symbolic board, then use a symbolic policy
```

V3 means:

```text
freeze a supervised visual encoder, then train an RL decoder and Q head
```

The current project favors V3 because it keeps the useful visual abstraction from supervised learning while avoiding both V1's unstable full visual RL path and V2's hard symbolic policy bottleneck.
