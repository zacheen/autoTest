# Design: YOLO Grid-State Predictor + Stage 1 Agent

> 這份文件是 2026-04-22 session 的設計討論產出，供新 session 接續。
> 讀完這份後應該能直接開始實作。

---

## 1. 背景與動機

### 1.1 目前的問題

現有的 `visual_discrete_agent.py` 端到端訓練（截圖 → YOLO → Transformer → FQF）**學不起來**，經過這次 session 的診斷：

- BatchNorm stats 穩定（不是 BN 問題，已加 `bn/*` TensorBoard 監控確認）
- `total_norm`（clip 前）持續上升 5 → 10，但個別 `grad/*_norm`（clip 後）穩定 — 代表梯度真實大小在爆
- 加了 `grad_pre/*` 監控後發現 **YOLO 的 pre-clip 梯度最大**（5→10），其他三個（decoder/policy/head）都在 5 附近，spike 會突破 20
- 最可能的根因：**Frozen Encoder 是用 36 個 grid token 訓練的，現在收到 1600 個 YOLO token，attention 完全錯位**。梯度穿過這個「錯的 attention」傳到 YOLO 被扭曲放大

### 1.2 為什麼改走這個設計

把問題解耦成兩個獨立子問題：

| 子問題 | 新方法 | 為什麼比端到端好 |
|------|------|---------------|
| 看懂畫面：screenshot → 遊戲狀態 | 監督式學習，server API 當 ground truth | 有明確 label，比 RL 好訓練 |
| 玩遊戲：遊戲狀態 → action | 直接用 `TransformerDiscreteAgent`（Stage 1 已訓好） | 不用重訓 |

**重點洞察：** `TransformerDiscreteAgent` 的輸入是 12-channel grid state tensor。如果 YOLO 能預測出這個 tensor，直接接上去就能玩。

---

## 2. 架構設計

### 2.1 Pipeline 總覽

```
┌─ Phase 1: 監督式訓練 YOLOGridStatePredictor ──────┐
│  data: (screenshot, server_state → 12ch tensor)  │
│  loss: CrossEntropy per cell                      │
└──────────────────┬────────────────────────────────┘
                   ↓
┌─ Phase 2: 組合部署 ────────────────────────────────┐
│  Screenshot                                        │
│    ↓ YOLOGridStatePredictor（訓練好，是否凍住看表現）│
│  Predicted grid state (B, 12, H, W)                │
│    ↓ TransformerDiscreteAgent（Stage 1，凍住）     │
│  Q-values → action                                 │
└────────────────────────────────────────────────────┘

Phase 3（RL fine-tune）: 延後決定，前兩階段做完再說
```

### 2.2 YOLOGridStatePredictor 架構

```
Screenshot (B, 3, 640, 640)
    ↓ YOLO11n backbone（YOLO11nLastFeatureExtractor，會 fine-tune）
(B, 128, 40, 40)                          ← 固定 1600 tokens，跟網格大小無關
    ↓ Token adapter：LayerNorm + Linear(128→d) + GELU + Linear(d→d)
    ↓ + 2D positional encoding（40×40）
(B, 1600, d) = memory tokens

給定目標網格 (H, W)：
    每個 cell 中心的正規化座標 (y, x) ∈ [0,1]²
    ↓ coord_mlp: Linear(2, d) → GELU → Linear(d, d)
    （或改用 sinusoidal positional encoding，更穩定）
(B, H×W, d) = query tokens

Cross-Attention（query attend to memory，單層或多層）：
    (B, H×W, d)
    ↓ Linear(d, 12) = classification head
(B, H×W, 12)
    ↓ reshape
(B, 12, H, W)                             ← 交給 TransformerDiscreteAgent 當輸入
```

### 2.3 為什麼 Size-Agnostic（未來換 10×10、16×16 可重用）

- YOLO 輸出固定 40×40 = 1600 tokens（跟遊戲網格無關）
- Query 是從 `(grid_h, grid_w)` 算出來的座標 → MLP，不是學出來的參數
- Cross-attention 天然支援任意 query 數量
- 所有可學習參數（YOLO、adapter、coord_mlp、cross-attn、head）都跟 H、W 無關

---

## 3. 資料收集（整合到 Demo_test_Minesweeper.py）

### 3.1 決策

**不另寫 `collect_grid_state_dataset.py`**。直接在 `Demo_test_Minesweeper.py` 跑 RL demo 時順便蒐集 `(screenshot, server_state_12ch)` pairs，用 flag 控制要不要存。

### 3.2 Flag 設計

在 `Demo_test_Minesweeper.py` 頂層加：

```python
COLLECT_VISION_DATASET = True   # 這個 flag 控制要不要蒐集資料
VISION_DATASET_PATH = Path("./datasets/vision_supervised")
```

### 3.3 蒐集時機

在 `test_RL_server` 的主迴圈內，**每次有效的 server state 更新時**存一筆：

- 輸入：`game_status.current_pic`（已經是 preprocessed tensor）或重新讀取的 screenshot
- Label：`game_status.server_state` 轉成 12-channel tensor

需要的轉換：server 的 board state → 跟 `MinesweeperLogic.get_grid_state_tensor()` 同格式的 tensor。  
**要先確認 `MinesweeperLogic` 的 12 channels 到底是哪 12 個**（看起來是：0=未翻、1=旗、2~10=數字0~8、11=? 待確認）。

### 3.4 存檔格式建議

```
datasets/vision_supervised/
  ├── index.jsonl              # 每行一筆 {screenshot_path, label_path, grid_h, grid_w, metadata}
  ├── screenshots/
  │   └── screen_000001.pt     # uint8 tensor (3, 640, 640) 節省空間
  └── labels/
      └── label_000001.pt      # int64 tensor (H, W) with class index 0-11
```

用 class index（不是 one-hot）配合 `CrossEntropyLoss` 最省空間。

### 3.5 要注意的問題

- **資料分佈不均**：隨機 agent 大多卡在開局幾步（大多 cell 未翻）。後期局面（大多已翻開）會不夠。
  - 解法一：收集時先用 Stage 1 agent 玩（它會真的玩到後期）
  - 解法二：隨機政策 + 強制先做幾步有效點擊再開始隨機（human-like opening）
- **一局平均 15~30 步**，收集 5000~10000 筆 labels 需要跑 300~700 局
- **去重**：同一 screenshot 可能重複出現，可用 screenshot hash 跳過

---

## 4. Loss 與訓練策略

### 4.1 Loss

12 channels 是互斥的 one-hot（一個 cell 只會是其中一種狀態），所以用：

```python
# pred: (B, 12, H, W), target: (B, H, W) with integer class 0-11
loss = F.cross_entropy(pred, target)
```

**不要用 BCE**（會忽略 channel 間互斥關係）。

### 4.2 可選的輔助 Loss

- 整個板的「未翻 cell 數量」一致性
- 「已翻開的數字分佈」要合理（0 通常遠多於 8）
- 暫時先不加，如果 CE loss 學不好再考慮

### 4.3 Learning Rate 建議

| 模組 | LR | 備註 |
|------|-----|------|
| YOLO backbone | 1e-5（小） | Fine-tune（用戶要求） |
| token adapter | 5e-4 | 從頭訓 |
| coord_mlp | 5e-4 | 從頭訓 |
| cross-attention | 5e-4 | 從頭訓 |
| 12-ch head | 5e-4 | 從頭訓 |

### 4.4 訓練順序

1. 先凍 YOLO 跑 5~10 epochs，讓 adapter/coord/attn/head 先穩定
2. 再解凍 YOLO 做 fine-tune

這樣比一開始就解凍所有東西穩定。

---

## 5. 需要新增/修改的檔案

### 5.1 新增

- **`autoTest_pytorch/yolo_grid_state_predictor.py`**
  - `YOLOGridStatePredictor` class
  - `build_queries(grid_h, grid_w, device)`
  - `forward(screenshot, grid_h, grid_w) -> (B, 12, H, W)`
  - 可能內含 YOLO11nLastFeatureExtractor（從 visual_discrete_agent.py 搬過來或 import）

- **`autoTest_pytorch/train_stage2_vision.py`**
  - 讀 `datasets/vision_supervised/`
  - DataLoader + CrossEntropyLoss + 訓練迴圈
  - TensorBoard：loss、per-cell accuracy、per-class confusion matrix
  - 分階段：先凍 YOLO 訓其他，再解凍

- **`autoTest_pytorch/visual_discrete_agent_v2.py`**（或改寫現有檔）
  - 組合 `YOLOGridStatePredictor` + `TransformerDiscreteAgent`
  - 純推論，沒有 RL 訓練（Phase 3 再決定）
  - `select_action(screenshot)` 直接跑兩段網路

### 5.2 修改

- **`autoTest_pytorch/Demo_test_Minesweeper.py`**
  - 加 `COLLECT_VISION_DATASET` flag
  - 在 `test_RL_server` 迴圈內當 flag 為 True 時存 `(screenshot, 12ch label)` pairs
  - 新增 helper：`server_state_to_grid_tensor(server_state) -> (H, W) long tensor`

- **`CLAUDE.md`**、**`README.md`** — 更新 Project Overview 反映兩階段 vision supervised + frozen Stage 1 agent 的新設計

---

## 6. 已經做好的準備（這個 session 改過的）

### 6.1 `visual_discrete_agent.py`（仍保留）
- `_save_model` / `try_load_model` 新增 `encoder` key，修正 frozen encoder 沒被存的問題（q_mean 斷層）
- `_log_batchnorm_stats`：TensorBoard 記錄 `bn/running_mean_avg`、`bn/running_mean_std`、`bn/running_var_avg`、`bn/running_var_std`。**已確認掃雷截圖下 BN stats 非常穩定**（變動 < 0.002）
- `grad_pre/yolo`、`grad_pre/decoder`、`grad_pre/policy`、`grad_pre/head`：clip 前的真實梯度。**觀察到 YOLO 最大 5→10，其他 ~5，spike > 20**

這些檔案改動都可以保留 — 新設計完成後 `visual_discrete_agent.py` 可能整個被 v2 取代，或當作 Phase 3 的起點。

### 6.2 `CLAUDE.md` Project Overview 已更新成兩階段視覺訓練描述，**現在這個設計改變後需要再更新一次**（新 session 請記得改）。

### 6.3 `README.md` 也改過，**新設計也要再改一次**。

---

## 7. 尚未決定 / 需要討論

### 7.1 Phase 3：RL Fine-tune 要不要做？
- 用戶：「前面做完我再決定」
- 判斷依據：Phase 2 組合起來實測勝率
  - 如果勝率接近 Stage 1 的純符號版 → 不用做
  - 如果有明顯 gap → 考慮做（但容易把 Stage 1 訓好的 agent 搞壞）

### 7.2 Query 生成用 MLP 還是 Sinusoidal？
- MLP：參數少，但可能不夠 smooth
- Sinusoidal：更穩定、無參數，對 OOD grid size 的 generalization 較好
- **建議先 MLP 試一下，訓練不穩再換 sinusoidal**

### 7.3 Cross-Attention 幾層？
- 1 層：最簡單，參數少
- 2~3 層：容量大，但可能過擬合小資料集
- **建議從 1 層開始**

### 7.4 12 Channel 的實際定義需要確認
- 看起來是：0=未翻、1=旗、2~10=數字0~8、11=?
- 實作前必須打開 `MinesweeperLogic.get_grid_state_tensor()` 確認，尤其是 channel 11 到底代表什麼
- 要把 server API 回傳的 board 狀態正確對應到這 12 個 class index

### 7.5 要不要支援 batch 中混合不同 grid size？
- 目前假設一個 batch 內 grid size 相同
- 要支援混合的話需要 padding + mask，複雜度上升
- **建議先不支援，訓練時每個 batch 固定 grid size 即可**

### 7.6 YOLO 要不要用更深層的特徵？
- 目前用 `YOLO_LAST_LAYER_IDX = 6`（40×40，128 ch）
- 更淺層（如 idx=4，80×80，64 ch）空間解析度高，更適合小 cell
- **先用現有設定，訓練後看 confusion matrix 再決定**

---

## 8. 新 Session 起手指引

1. 讀完這份文件
2. 掃一下 `autoTest_pytorch/transformer_discrete_agent.py`（了解 Stage 1 agent 的 input 格式）
3. 掃一下 `autoTest_pytorch/Minesweeper/MinesweeperLogic.py`（確認 12 channel 定義）
4. 跟用戶確認「7. 尚未決定」的項目
5. 開始實作順序建議：
   - 先寫 `server_state_to_grid_tensor()` helper（最基礎，可單元測試）
   - 再改 `Demo_test_Minesweeper.py` 加 `COLLECT_VISION_DATASET` flag 蒐集資料
   - 跑一下蒐集幾百筆資料確認格式正確
   - 寫 `yolo_grid_state_predictor.py`
   - 寫 `train_stage2_vision.py` 跑監督訓練
   - 組合到 `visual_discrete_agent_v2.py` 部署
