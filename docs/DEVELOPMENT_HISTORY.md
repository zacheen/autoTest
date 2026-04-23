# Development History

A record of what was tried, what failed, and why — for future reference.

---

## Iteration Summary

### Attempt 1 — TD3 + ResNet18 (failed)
- **Approach**: End-to-end visual RL using TD3 with a ResNet18 image backbone.
- **Failure mode**: No entropy regularization → agent collapsed to deterministic suboptimal policy. ResNet18 too heavy for the feature resolution needed; training unstable.

### Attempt 2 — SAC + YOLO11n end-to-end (failed)
- **Approach**: Replace TD3 with SAC (entropy regularization) and ResNet18 with YOLO11n backbone (lighter, better spatial features). Train end-to-end: screenshot → YOLO → transformer → action.
- **Failure mode**: BN stats stable (confirmed via TensorBoard), but YOLO gradients spiked > 20 (other modules ~5). Root cause: encoder trained on 36 grid tokens (6×6) but receiving 1600 YOLO tokens — attention completely misaligned, amplifying gradients through YOLO.

### Attempt 3 — Decoupled two-stage pipeline (current, working)
- **Approach**: Split into two independent sub-problems.
  - **Stage 1**: Train transformer policy on symbolic grid-state tensors (fast local simulation, no screenshots). Converges to 90% win rate with 0% invalid click rate.
  - **Vision module**: Train `YOLOGridStatePredictor` with supervised learning (screenshot → grid state), using server API as ground truth label source. Converges to ~100% per-cell accuracy.
  - **Inference**: Compose both frozen models. Screenshot → YOLO → grid state → Stage 1 Q-network → action.
- **Result**: 95% win rate on 6×6 Minesweeper (4 mines).

---

## Key Debugging Episodes

### Gradient explosion in YOLO backbone
- **Symptom**: `grad_pre/yolo` norm spiking to > 20; other modules stable at ~5.
- **Diagnosis**: Added per-module gradient norm TensorBoard logging. Isolated YOLO as the source.
- **Root cause**: Frozen encoder's attention was trained on 36 tokens (grid cells), suddenly receiving 1600 YOLO spatial tokens. Misaligned attention → distorted gradients propagated back into YOLO.
- **Fix**: Decouple visual encoding from the RL policy entirely.

### High invalid click rate in V2
- **Symptom**: YOLO accuracy 100%, Stage 1 standalone 0% invalid rate, but combined V2 had high invalid rate.
- **Diagnosis**: Added per-step comparison of YOLO predicted grid vs server ground truth, plus Q-value breakdown by cell type.
- **Root cause**: `train_stage1_simple.py` used `GRID_MINES = 4`, but `Minesweeper_web/server.py` "Training 6x6" was configured with 6 mines. The grid number distribution Stage 1 trained on was completely out-of-distribution for the web game.
- **Fix**: Set server mines to 4 to match training. Long-term fix: retrain Stage 1 with 6 mines.

---

## Lessons Learned

**Decouple game logic from UI early.**
Separating game rules into a pure Python module (`MinesweeperLogic`) with no UI dependencies enabled fast headless simulation for Stage 1 training and clean label generation for the vision module.

**RL bugs are silent.**
Wrong Q-values and mismatched distributions don't crash training — they just waste GPU hours. Code review and gradient monitoring before long runs are cheaper than debugging after convergence fails.

**Distribution mismatch is easy to miss.**
Even when each component tests correctly in isolation, the training environment and deployment environment must use identical parameters (grid size, mine count, reward structure). A 2-mine difference caused complete policy failure.

**Two-stage > end-to-end for debugging.**
Each stage can be validated independently with clear metrics (classification accuracy, win rate). End-to-end RL provides no such intermediate signal.
