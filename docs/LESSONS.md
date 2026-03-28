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
