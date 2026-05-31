# SESSIONS

- 2026-03-28-1320: understand-model-structure (Implemented two-stage SAC training + game logic separation)

# LESSONS

## Design Patterns

**Problem**: Game logic tightly coupled to UI makes headless training (no screenshots) impossible for early validation.

**Mitigation**: Separate game logic into pure Python module with clean API (`click()`, `flag()`, `get_grid_state_tensor()`), zero UI dependencies.

**Lesson**: Decoupling game rules from rendering is critical for both AI training and UI flexibility — future game ports (web, mobile) only need to re-implement UI layer.

---

**Problem**: SAC agent getting stuck in local optima on small discrete action spaces (10×10 grid = 100 cells, only 2 actions per cell: left-click, right-click).

**Mitigation**: Use entropy regularization (automatic temperature tuning) instead of additive Gaussian noise like TD3; encourages exploration of click coordinates rather than grid cells.

**Lesson**: For low-dimensional action spaces, stochastic policies with entropy regularization outperform deterministic policies with noise injection.

---

**Problem**: Code review found 5 bugs (1 critical P0, 2 performance P1) that would have caused silent failures or 200× slowdown during training.

**Mitigation**: Always perform code review BEFORE running long training loops; focus on SAC algorithm correctness (critic using correct encoder, embedding distributions matching), disk I/O frequency, and buffer invariants.

**Lesson**: RL training bugs often manifest as slow convergence or silent failures (wrong Q-values), not crashes; code review is cheaper than 10k GPU hours of wasted training.

---

**Problem**: V3 stores every transition as a separate `.pt` file via `CategorizedReplayBuffer(storage_mode="disk")` and reads 40 of them per `sample()` call, every training step. The project root sits on a HDD, where small-file random reads are too slow to scale `BUFFER_CAPACITY` beyond a couple thousand.

**Mitigation**: Split the v3 disk layout. Keep model weights / `training_state.pth` / TensorBoard / action logs under `VISUAL_V3_MODEL_PATH` on the project drive (low-frequency writes), and route the hot replay-buffer I/O — `VISUAL_V3_REPLAY_PATH` and `VISUAL_V3_REPLAY_PERSISTENT_PATH` — to an SSD via `VISUAL_V3_REPLAY_BASE` (currently `C:\dont_move\temp\autotest\`). `training_state.pth` keeps cross-drive absolute paths to the persistent snapshot, which `pathlib` handles transparently.

**Lesson**: When a buffer is implemented as "one file per entry", the project drive is not necessarily the right home for it — separate the constants for *checkpoint* paths and *hot-loop* paths so the latter can be relocated to an SSD without touching code in `CategorizedReplayBuffer` or `visual_agent_common`. Also remember that `training_state.pth` embeds absolute paths to the persistent snapshot; if you wipe the SSD cache, wipe `training_state.pth` along with it or `_load_persistent_buffer` will reference dead paths.
