# TensorBoard Metrics Explanation (Visual Discrete Agent)

This document provides a detailed explanation of the metrics logged to TensorBoard by the `VisualDiscreteAgent` during training.

---

## 1. Training Metrics (`train/`)

These metrics are recorded at every **Training Step** (per batch update).

- **`train/loss`**: 
  - **Calculation**: Calculated using `F.smooth_l1_loss` between the predicted Q-values and the target Q-values.
  - **Meaning**: Represents the prediction error of the model.
  - **Expected Trend**: Should **Decrease** initially and then stabilize. If it increases indefinitely, the learning rate might be too high or the model is diverging.
- **`train/q_mean`**:
  - **Calculation**: The average Q-value of the actions selected in the current training batch.
  - **Meaning**: Indicates the agent's "confidence" in its selected actions.
  - **Expected Trend**: Should generally **Increase** over time as the agent finds higher-reward strategies.
- **`train/reward_mean`**:
  - **Calculation**: The average reward of current play. Should **Increase** 
  - **Meaning**: Helps monitor the distribution of rewards being sampled.
  - **Expected Trend**: Should **Increase** as the agent learns to win more and avoid mines.
- **`train/done_rate`**:
  - **Calculation**: The percentage of samples in the batch that reached a terminal state (`done=True`).
  - **Expected Trend**: Should be **Stable**. A extremely low done rate might mean the agent is stuck in long, unproductive loops.
- **`train/epsilon`**:
  - **Expected Trend**: Linear or exponential **Decrease** towards `epsilon_min` (0.05).

---

## 2. Gradient & Weight Metrics (`grad/`, `weights/`, `grads/`)

These metrics help monitor the health of the neural network optimization process.

### Scalar Gradient Norms
- **`grad/total_norm`**, **`grad/yolo_norm`**, etc.
- **Meaning**: Used to detect gradient issues.
- **Expected Trend**: Should be **Relatively Stable**. 
    - **Too High**: If norms explode, the training becomes unstable. 
    - **Too Low (Zero)**: If norms are zero, the model is not learning (possibly frozen or dead neurons).

### Histograms
- **`weights/*`**:
  - **Expected Trend**: Values should spread out but remain within a reasonable range (e.g., -1.0 to 1.0). A "collapsing" distribution (all values near zero) indicates a dead network.
- **`grads/*`**:
  - **Expected Trend**: Most gradients should be small but non-zero, forming a sharp peak at zero with visible "tails".

---

## 3. Episode Metrics (`episode/`)

These metrics are recorded once per **Episode** (one complete game).

- **`episode/win`**: 
  - **Expected Trend**: Should **Increase** over time. This is the ultimate goal.
- **`episode/invalid_click_rate`**:
  - **Calculation**: `(Number of Invalid Clicks) / (Total Clicks in Episode)`.
  - **Expected Trend**: Should **Decrease** sharply towards 0. If it stays high, the agent hasn't learned the basic rules.

---

## 4. Debug Metrics (`debug/`)

- **`debug/yolo_nan_grad_count`**: **Expected Trend**: Must stay **Zero**.
- **`debug/yolo_grad_param_count`**: **Expected Trend**: Should be **Constant** and match the number of trainable layers in YOLO.

---

## Reward Logic Summary

Most metrics are driven by the following reward values:
- **Win**: `+3.6`
- **Valid Click**: `+1.0`
- **Mine (Loss)**: `-1.0`
- **Invalid Click**: `-0.98` (to penalize repetitive or useless actions)
