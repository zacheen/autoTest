# TensorBoard 數值計算說明文件

本文件詳細說明了 Minesweeper RL 訓練過程中，TensorBoard 上各項指標 (Metrics) 的計算邏輯與物理意義。

---

## 1. 訓練核心指標 (Training Metrics) - `train/`

這些指標通常在每一個 **Episode** 結束或每一個 **Training Step** (Batch 更新) 時記錄。

### `train/loss`
*   **計算方式**: 
    *   **Transformer Agent**: 使用 `F.huber_loss` 計算預測 Q 值與目標 Q 值的差異，並乘以 **Importance Sampling (IS) weights**。
    *   **Visual Agent**: 使用 `F.smooth_l1_loss`。
*   **意義**: 模型預測的準確度。Loss 越低表示模型對動作價值的估計越趨於穩定。

### `train/q_mean`
*   **計算方式**: 該 Batch 中，Agent **實際選擇的動作** 所對應的 Q 值的平均值。
*   **意義**: 代表 Agent 對目前採取的策略有多大的「信心」。在訓練初期通常會劇烈變動，隨著獎勵獲取增加，Q 值應逐漸上升。

### `train/episode_reward` (Stage 1)
*   **計算方式**: 單場遊戲中所有動作獎勵的總和。
    *   **勝利**: +3.6
    *   **有效點擊**: +1.0
    *   **踩雷 (失敗)**: -1.0
    *   **無效點擊 (重複點擊)**: -0.98
*   **意義**: 衡量 Agent 在一場遊戲中的總體表現。

### `train/invalid_rate`
*   **計算方式**: `無效點擊次數 / 總點擊次數`。
*   **意義**: 評估 Agent 是否學會「不要點擊已經打開或標記的格子」。該值應隨著訓練進行逐漸降至 0。

### `train/reward_mean` (Visual Agent Batch)
*   **計算方式**: 當前 Training Batch 中所有樣本的平均獎勵。
*   **意義**: 觀察目前訓練數據池 (Replay Buffer) 中抽樣到的獎勵分佈情形。

### `train/done_rate` (Visual Agent Batch)
*   **計算方式**: 當前 Batch 中，遊戲結束 (Win or Lose) 的樣本比例。
*   **意義**: 觀察訓練過程中是否包含足夠多的結局樣本。

---

## 2. 評估指標 (Evaluation Metrics) - `eval/`

這些指標是在 **Evaluation Mode** (關閉 $\epsilon$-greedy 隨機探索，只選最優動作) 下計算的平均值。

*   **`eval/avg_reward`**: 近期評估場次的平均總獎勵。
*   **`eval/win_rate`**: 勝率百分比 (0~100%)。
*   **`eval/avg_steps`**: 平均每場遊戲走幾步。
*   **`eval/avg_invalid_rate`**: 評估模式下的無效點擊率。

---

## 3. 視覺 Agent 特有指標 - `episode/` & `grad/`

針對使用截圖訓練的 `VisualDiscreteAgent`：

### `episode/win`
*   **計算方式**: 單場結束後，若勝利則為 1，否則為 0。常用於觀察勝率曲線的平滑趨勢。

### `grad/*_norm` (Gradient Norms)
*   **計算方式**: 對不同模組 (YOLO, Backbone, Policy, Head) 的梯度進行 L2 Norm 計算。
*   **意義**: 監控是否有 **梯度消失 (Vanishing)** 或 **梯度爆炸 (Exploding)**。如果 Norm 突然變成 0 或極大值，代表網路層訓練異常。

### `debug/yolo_*` (YOLO Debug Info)
*   **意義**: 專門監控 YOLO 骨幹網路的參數更新狀態，確保卷積層有正確獲得梯度 (Requires Grad) 且沒有出現 NaN。

---

## 4. 關鍵機制背景說明

### A. 獎勵函數 (Reward Function)
模型並非直接學習「勝率」，而是極大化獎勵。目前的權重設計（如勝利 +3.6 vs 有效點擊 +1.0）是為了鼓勵模型在保證不踩雷的前提下，儘可能多探索並最終獲勝。

### B. 優先經驗回放 (Prioritized Replay Buffer)
*   Buffer 會根據 **TD-error**（預測誤差）來計算樣本的 `priority`。
*   **TD-error 越大** 的樣本，表示模型對其越不熟悉，未來被抽中訓練的機率越高。
*   在 TensorBoard 的 `train/loss` 中，IS weights 會修正因為非均勻抽樣帶來的偏差。

### C. $\epsilon$-greedy 隨機探索
*   **`train/epsilon`**: 顯示目前探索率。隨著訓練 Episode 增加，Epsilon 會從 0.3 慢慢降至 0.05，這表示模型從「亂點」轉向「專業推理」。
