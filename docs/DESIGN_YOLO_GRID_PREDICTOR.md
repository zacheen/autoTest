# Design: YOLO Grid-State Predictor + Stage 1 Agent

> 最後更新：2026-04-23 session
> 上一份版本（2026-04-22）記錄診斷過程，本份記錄**已實作完成的內容**與**下一步**。
> 新 session 讀完本文件後可直接進入「9. 新 Session 起手指引」。

---

## 1. 背景與動機（為什麼不繼續端到端訓練）

現有的 `visual_discrete_agent.py` 端到端訓練（截圖 → YOLO → Transformer → FQF）學不起來，診斷結果：

- BN stats 穩定（已用 TensorBoard `bn/*` 確認，變動 < 0.002，**不是 BN 問題**）
- `grad_pre/yolo`（clip 前）5 → 10，spike > 20；其他模組 ~5 → **YOLO 是梯度爆炸源**
- 根本原因：Frozen Encoder 用 36 個 grid token 訓練，現在收到 1600 個 YOLO token，**attention 完全錯位**，梯度穿過錯的 attention 傳到 YOLO 後被扭曲放大

**解法：解耦成兩個獨立子問題**

| 子問題 | 新方法 | 優點 |
|------|------|------|
| 看懂畫面：screenshot → grid state | 監督式學習，server API 當 ground truth | 有明確 label，訓練穩定 |
| 玩遊戲：grid state → action | 直接用 Stage 1 已訓好的 `TransformerDiscreteAgent`（凍住） | 不用重訓 |

---

## 2. 架構設計（已確定）

### 2.1 Pipeline 總覽

```
┌─ Phase 1: 監督式訓練 YOLOGridStatePredictor ──────────┐
│  data: Demo 跑 RL 時順便蒐集 (screenshot, label)      │
│  label: server_state → (H, W) class index 0~10       │
│  loss: CrossEntropyLoss                               │
└──────────────────┬────────────────────────────────────┘
                   ↓
┌─ Phase 2: 組合部署 ─────────────────────────────────── ┐
│  Screenshot                                            │
│    ↓ YOLOGridStatePredictor（訓練好）                  │
│  Predicted grid state (B, 12, H, W)                    │
│    ↓ TransformerDiscreteAgent（Stage 1，凍住）         │
│  Q-values → action                                     │
└────────────────────────────────────────────────────────┘

Phase 3（RL fine-tune）: 延後決定，組合後實測勝率再說
```

### 2.2 YOLOGridStatePredictor 架構（已實作）

```
Screenshot (B, 3, 640, 640)
    ↓ YOLO11nLastFeatureExtractor（YOLO_LAST_LAYER_IDX=6，會 fine-tune）
(B, 128, 40, 40) = 1600 tokens，固定，與網格大小無關
    ↓ token_adapter: LayerNorm(128) → Linear(128→128) → GELU → Linear(128→128)
    ↓ + TwoDimensionalPositionEmbedding(40, 40, d_model=128)
(B, 1600, 128) = memory tokens

給定目標網格 (H, W)：
    cell 中心的正規化座標 (y+0.5)/H, (x+0.5)/W ∈ [0,1]²
    ↓ coord_mlp: Linear(2,128) → GELU → Linear(128,128)
      ← 這就是 query 的 positional encoding（size-agnostic）
(B, H×W, 128) = query tokens

nn.TransformerDecoder（2 層，nhead=4，dim_ff=512）：
    self-attn（query 互相關注） + cross-attn（query attend to memory） + FFN
(B, H×W, 128)
    ↓ Linear(128, 12) = classification head
(B, H×W, 12) → reshape
(B, 12, H, W)  ← 直接餵給 TransformerDiscreteAgent
```

**超參數（`YOLOGridStatePredictor.__init__` 預設值）：**

| 參數 | 值 |
|------|-----|
| d_model | 128 |
| nhead | 4 |
| num_cross_attn_layers | 2 |
| dim_feedforward | 512 |
| dropout | 0.1 |
| num_classes | 12 |

---

## 3. 12-Channel 定義（已確認）

來源：`MinesweeperLogic.get_grid_state_tensor()` + `Minesweeper_web/server.py`

| class index | 含義 | server cell.state |
|---|---|---|
| 0 | 未翻開 (hidden) | `"hidden"` |
| 1 | 旗子 (flagged) | `"flagged"` / `"flagged_mine"` / `"wrong_flag"` |
| 2~10 | 數字 0~8 | `"revealed"` + value=n → class = 2+n |
| 11 | 地雷 | `"mine"` / `"hit_mine"` |

**重要：class 11（地雷）不蒐集訓練資料。**
- 地雷只在 `status == "lost"` 後才可見
- 推論時永遠不會遇到這種狀態
- `VisionDatasetRecorder.record()` 內部會自動過濾 `status == "lost"`

---

## 4. 資料蒐集（已實作於 Demo_test_Minesweeper.py）

### 4.1 開關

```python
# Demo_test_Minesweeper.py 頂層
COLLECT_VISION_DATASET = True
VISION_DATASET_PATH = Path("./datasets/vision_supervised")
```

### 4.2 蒐集時機

- `decide_next_step_and_play()` 每次拿到 `current_screenshot` 時，立即配上 `game_status.server_state` 存一筆
- game_over（踩雷）的最終畫面**不蒐集**（record 內部過濾）

### 4.3 存檔格式

```
autoTest_pytorch/datasets/vision_supervised/
  ├── index.jsonl                  # 每行一筆 JSON：id, screenshot, label, grid_h, grid_w, status, hash
  ├── screenshots/
  │   └── screen_000001.pt         # uint8 tensor (3, 640, 640)，節省空間
  ├── labels/
  │   └── label_000001.pt          # int64 tensor (H, W)，值域 0~11
  └── check_data/
      └── check_000001.png         # 視覺確認圖（第 1 筆 + 每 50 筆的第 3 筆自動產生）
```

### 4.4 確認圖（check_data）

每 50 筆的第 3 筆（idx == 1 or idx % 50 == 3）會產出一張 check image：
- 底圖為原始截圖（640×640）
- 白色格線劃分 grid
- 每格左上角標示 active class 符號（`?`=未翻、`F`=旗、`0`~`8`=數字、`M`=雷）
- 存到 `check_data/check_{idx:06d}.png`

**用途：目視確認 label 正確，不需要任何程式碼就能看**

### 4.5 去重

`VisionDatasetRecorder` 內部維護 screenshot 的 SHA1 hash set，同一畫面重複出現時自動跳過。

---

## 5. 訓練（已實作於 yolo_grid_state_predictor.py，`if __name__` 觸發）

### 5.1 執行方式

```bash
cd autoTest_pytorch
python yolo_grid_state_predictor.py
```

checkpoint 存在 `models/yolo_grid_predictor/`，中斷後重跑同一指令自動接續。

### 5.2 兩階段訓練

| 階段 | epochs | YOLO | 其他模組 | 目的 |
|------|--------|------|----------|------|
| Phase 1 | 0~9（PHASE1_EPOCHS=10） | **凍住** | LR 5e-4 | 讓 adapter/coord/attn/head 先穩定 |
| Phase 2 | 10~49（PHASE2_EPOCHS=40） | **解凍** LR 1e-5 | LR 5e-4 | YOLO fine-tune |

Optimizer: `AdamW`，LR Schedule: `CosineAnnealingLR`，Grad clip: `max_norm=1.0`

### 5.3 超參數（在 `YOLOGridStateTrainer` class 頂端可直接改）

```python
BATCH_SIZE    = 8
LR_YOLO       = 1e-5
LR_OTHER      = 5e-4
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 40
VAL_RATIO     = 0.15
```

### 5.4 TensorBoard 監控

```bash
tensorboard --logdir autoTest_pytorch/models/yolo_grid_predictor/tensorboard
```

觀測指標：
- `train/loss`, `val/loss`
- `train/acc`, `val/acc`（整體 pixel accuracy）
- `val/acc_hidden`, `val/acc_flag`, `val/acc_num0` ... `val/acc_num8`（各 class 準確率）

**觀察重點：**
- `val/acc_hidden` 應最先飆高（大多數 cell 是未翻開）
- `val/acc_num*` 慢一點，數字 class 樣本相對少
- 如果某個 class 的 acc 卡住不動 → 可能需要調整 YOLO layer 深度或資料量

### 5.5 Gap-tolerant Dataset 讀取

手動刪除有問題的 `.pt` 檔後，`VisionSupervisedDataset.load_and_split()` 會：
1. 讀取 `index.jsonl` 中的所有 entry
2. 逐一確認 `screenshots/screen_XXXXXX.pt` 和 `labels/label_XXXXXX.pt` 是否都存在
3. 跳過缺檔的 entry，並印出 `Skipped N entries (files missing)`
4. 對剩餘的有效 entry 做 train/val split

**檔名不連續（如有 screen_000001.pt 和 screen_001641.pt，但沒有 screen_000641.pt）不影響讀取。**

---

## 6. 檔案現況

### 6.1 `autoTest_pytorch/yolo_grid_state_predictor.py`（**主要新增檔**）

結構：
```
yolo_grid_state_predictor.py
│
├── 常數：CH_*, NUM_CHANNELS, YOLO_FEATURE_*, _CLASS_NAMES
│
├── server_state_to_grid_tensor(server_state) → (H, W) LongTensor
│     └── Web API server_state dict → class index 0~11
│
├── class VisionDatasetRecorder               ← Demo 蒐集資料用
│     ├── record(screenshot, server_state)    ← 自動過濾 lost，SHA1 去重
│     └── _save_check_image()                ← 視覺確認圖
│
├── class YOLOGridStatePredictor(nn.Module)   ← 核心模型
│     ├── forward(screenshot, grid_h, grid_w) → (B, 12, H, W) logits
│     ├── predict_grid_state()               ← 推論（one-hot 輸出）
│     ├── freeze_yolo() / unfreeze_yolo()
│     └── yolo_parameters() / non_yolo_parameters()
│
├── class VisionSupervisedDataset(Dataset)   ← 訓練資料讀取
│     ├── __getitem__() → (screenshot_float, label_long)
│     └── load_and_split()                   ← 讀 index.jsonl，過濾缺檔 entry，切 train/val
│
├── class YOLOGridStateTrainer               ← 訓練邏輯（OOP）
│     ├── _build_optimizer()
│     ├── _train_epoch()
│     ├── _validate()                        ← per-class accuracy
│     ├── _log_metrics()                     ← TensorBoard + console（只印 acc < 0.95 的 class）
│     ├── save_checkpoint() / load_checkpoint()
│     └── train()                            ← Phase 1 → Phase 2 主迴圈，支援中斷後接續
│
└── if __name__ == "__main__":               ← 直接 python yolo_grid_state_predictor.py 觸發
```

### 6.2 `autoTest_pytorch/Demo_test_Minesweeper.py`（已修改）

- 加了 `COLLECT_VISION_DATASET = True` 開關
- `Game_status.__init__` 裡建立 `self.vision_recorder`
- `_maybe_record_vision_sample()` helper
- `decide_next_step_and_play()` 每次截圖後立即蒐集一筆

### 6.3 `autoTest_pytorch/visual_discrete_agent.py`（保留，供 Phase 3 參考）

- 已加 `encoder` key 到 save/load（修正 q_mean 斷層問題）
- 已加 `bn/*` TensorBoard 監控（確認 BN 穩定）
- 已加 `grad_pre/*` 監控（確認 YOLO 是梯度源）

### 6.4 尚未寫的檔案

- **`autoTest_pytorch/visual_discrete_agent_v2.py`**：組合 `YOLOGridStatePredictor` + `TransformerDiscreteAgent`，純推論介面。**等訓練結果出來再寫。**

---

## 7. 已決定的設計問題（原 section 7 的 open questions）

| 問題 | 決定 |
|------|------|
| 7.1 Phase 3 RL fine-tune？ | 延後，等 Phase 2 實測勝率再決定 |
| 7.2 Query 用 MLP / Sinusoidal？ | **MLP**（已實作），訓練不穩再換 sinusoidal |
| 7.3 Cross-Attention 幾層？ | **2 層**，含 self-attn + cross-attn |
| 7.4 12-channel 定義 | **已確認**（見 section 3） |
| 7.5 batch 混合不同 grid size？ | **不支援**，一個 batch 固定 size |
| 7.6 YOLO 用哪層特徵？ | **維持 idx=6**（40×40, 128ch），訓完看 confusion matrix 再決定 |
| CH_MINE 樣本比例過高 | **不蒐集**，record() 過濾 status=="lost" |
| 缺檔 entry 如何處理 | **load_and_split 自動跳過**，印 skipped count |

---

## 8. 資料蒐集注意事項

### 8.1 分佈問題
隨機 agent 大多在開局幾步就踩雷，導致：
- **class 0（hidden）** 樣本非常多
- **class 3~10（數字 1~8）** 樣本相對少（需要玩到後期才會出現）

建議：蒐集夠多資料後看 per-class sample count（可從 label .pt 統計），若不均衡考慮用 weighted sampling。

### 8.2 目標資料量
- 一局 6x6 掃雷 6×6=36 cells × 步數 ≈ 幾百個 cell-labels
- 目標：**5000~10000 筆有效 pair**（即 5000~10000 個 screenshot + label）
- 對應：跑 300~700 局

### 8.3 確認方式
開 `check_data/` 裡的 PNG，目視確認格線是否對齊遊戲格子、class 符號是否正確。

---

## 9. 新 Session 起手指引

**直接跳到這裡，上面當參考。**

### 目前狀態
- [x] `yolo_grid_state_predictor.py` 寫完（含 Dataset + Trainer）
- [x] `Demo_test_Minesweeper.py` 改完（蒐集開關 + 過濾 lost 畫面）
- [ ] 資料蒐集中（需要跑 Demo 累積 ~5000 筆）
- [ ] 訓練（待資料夠後執行 `python yolo_grid_state_predictor.py`）
- [ ] 部署（`visual_discrete_agent_v2.py`，待訓練結果出來再寫）

### 下一步任務選項

#### A. 確認資料蒐集正確（現在應做）
1. 跑 `Demo_test_Minesweeper.py` 幾局
2. 確認 `datasets/vision_supervised/check_data/` 有 PNG 出現
3. 確認 PNG 裡的格線對齊、class 符號合理（大部分應該是 `?`）
4. 用以下 snippet 確認 label 分佈：
```python
import torch
from pathlib import Path
labels_dir = Path("autoTest_pytorch/datasets/vision_supervised/labels")
counts = torch.zeros(12, dtype=torch.long)
for f in sorted(labels_dir.glob("*.pt")):
    lab = torch.load(f, weights_only=True)
    for c in range(12):
        counts[c] += (lab == c).sum()
print(counts)
```

#### B. 開始訓練（資料夠後）
```bash
cd autoTest_pytorch
python yolo_grid_state_predictor.py
```
並在另一個 terminal 開 TensorBoard 監控。

#### C. 寫 visual_discrete_agent_v2.py（訓練完成後）
- 讀取 `models/yolo_grid_predictor/best.pth`
- 組合 `YOLOGridStatePredictor.predict_grid_state()` + `TransformerDiscreteAgent.select_action()`
- 替換 `Demo_test_Minesweeper.py` 的 `get_agent()` 來源
