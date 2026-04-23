"""YOLO Grid-State Predictor — 監督式訓練：screenshot → 12-channel grid state。

設計文件：docs/DESIGN_YOLO_GRID_PREDICTOR.md

Pipeline（Phase 1 監督訓練）：
    Screenshot (B, 3, 640, 640)
      ↓ YOLO11n backbone（fine-tune, LR 1e-5）
    (B, 128, 40, 40)
      ↓ token adapter + 2D positional encoding
    (B, 1600, d_model)                               = memory tokens
      ↓ coord-based queries（MLP(2 → d_model)）     = H*W queries
      ↓ Cross-Attention（2 層 TransformerDecoder，含 self-attn + cross-attn）
    (B, H*W, d_model)
      ↓ classification head Linear(d_model, 12)
    (B, 12, H, W)

該 tensor 可直接餵入凍結的 TransformerDiscreteAgent 做推論（Phase 2）。

Size-agnostic：YOLO 輸出固定 40×40，query 由 grid(h, w) 算出正規化座標 → MLP，
所有可學習參數都與 H、W 無關，之後改 10×10 或 16×16 不需重訓。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 避免 circular import：YOLO11n extractor 與 pos-embed 從 visual_discrete_agent / shared module 匯入
from model_structure.transformer_shared import TwoDimensionalPositionEmbedding


# --------------------------------------------------------------------------- #
# 12-channel 定義（須與 Minesweeper/MinesweeperLogic.py 的 get_grid_state_tensor 一致） #
# --------------------------------------------------------------------------- #
CH_UNREVEALED = 0
CH_FLAGGED = 1
CH_NUM_0 = 2           # 數字 0（全空白）
CH_NUM_8 = 10          # 數字 8
CH_MINE = 11           # 地雷（game over 才可見）
NUM_CHANNELS = 12


# --------------------------------------------------------------------------- #
# Server API state → grid tensor                                              #
# --------------------------------------------------------------------------- #
def server_state_to_grid_tensor(server_state: dict) -> torch.Tensor:
    """把 Minesweeper web API 回傳的 server_state 轉成 (H, W) 的 class-index LongTensor。

    用途：監督訓練的 label（配合 CrossEntropyLoss）。

    Mapping（server cell.state → 12-channel class index）：
        "hidden"                                           → 0  (CH_UNREVEALED)
        "flagged" / "flagged_mine" / "wrong_flag"          → 1  (CH_FLAGGED)
        "revealed" 帶 value=n (0..8)                       → 2+n (CH_NUM_0 + n)
        "mine" / "hit_mine"                                → 11 (CH_MINE)
        其他未知狀態                                        → 0  (預設視為未翻開)

    Args:
        server_state: dict，包含 "board" 為 list[list[dict]]，每格有 "state" 與 "value"。

    Returns:
        torch.LongTensor，shape (H, W)，值域 [0, 11]。
    """
    if server_state is None:
        raise ValueError("server_state is None")
    board = server_state.get("board")
    if not board:
        raise ValueError("server_state['board'] is empty or missing")

    rows = len(board)
    cols = len(board[0])
    label = torch.zeros(rows, cols, dtype=torch.long)

    for r, row_cells in enumerate(board):
        for c, cell in enumerate(row_cells):
            state = cell.get("state")
            if state == "hidden":
                label[r, c] = CH_UNREVEALED
            elif state in ("flagged", "flagged_mine", "wrong_flag"):
                label[r, c] = CH_FLAGGED
            elif state == "revealed":
                value = cell.get("value")
                n = int(value) if value is not None else 0
                n = max(0, min(8, n))
                label[r, c] = CH_NUM_0 + n
            elif state in ("mine", "hit_mine"):
                label[r, c] = CH_MINE
            else:
                label[r, c] = CH_UNREVEALED
    return label


# --------------------------------------------------------------------------- #
# Dataset 蒐集器                                                               #
# --------------------------------------------------------------------------- #
class VisionDatasetRecorder:
    """把 (screenshot, server_state_12ch_label) pairs 存到磁碟。

    目錄結構：
        dataset_dir/
          ├── index.jsonl              # 每行一筆 metadata
          ├── screenshots/
          │   └── screen_000001.pt     # uint8 tensor (3, 640, 640)
          └── labels/
              └── label_000001.pt      # int64 tensor (H, W) class index 0~11

    用 screenshot SHA1 hash 做去重（同一張畫面重複出現則跳過）。
    支援續跑：開檔時會讀取 index.jsonl 統計已有筆數與 hash。
    """

    def __init__(self, dataset_dir):
        self.dir = Path(dataset_dir)
        self.screenshots_dir = self.dir / "screenshots"
        self.labels_dir = self.dir / "labels"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"

        self._seen_hashes: set[str] = set()
        self.count = 0
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.count += 1
                    h = entry.get("hash")
                    if h:
                        self._seen_hashes.add(h)
            print(f"[VisionDataset] Resumed from {self.index_path}: {self.count} entries loaded")

    @staticmethod
    def _hash_screenshot(screen_uint8: torch.Tensor) -> str:
        return hashlib.sha1(screen_uint8.numpy().tobytes()).hexdigest()

    def record(self, screenshot: torch.Tensor, server_state: dict) -> bool:
        """存一筆 (screenshot, label) pair。

        Args:
            screenshot: (3, 640, 640) float tensor in [0, 1]（preprocess_screen 輸出）。
            server_state: dict from Web API。

        Returns:
            True 表示成功新增；False 表示資料無效或重複，略過。
        """
        if screenshot is None or server_state is None:
            return False
        board = server_state.get("board")
        if not board:
            return False

        # 踩雷後的畫面（status == "lost"）跳過：
        # 地雷只在 game over 後才可見，實際遊玩時預測器永遠不會遇到這種狀態，
        # 蒐集這些樣本只會讓 CH_MINE(class 11) 比例失真並浪費訓練資源。
        if server_state.get("status") == "lost":
            return False

        screen_cpu = screenshot.detach().cpu().clamp(0.0, 1.0)
        screen_uint8 = (screen_cpu * 255.0).to(torch.uint8)
        h = self._hash_screenshot(screen_uint8)
        if h in self._seen_hashes:
            return False

        try:
            label = server_state_to_grid_tensor(server_state)
        except ValueError:
            return False
        grid_h, grid_w = int(label.shape[0]), int(label.shape[1])

        idx = self.count + 1
        screen_path = self.screenshots_dir / f"screen_{idx:06d}.pt"
        label_path = self.labels_dir / f"label_{idx:06d}.pt"
        torch.save(screen_uint8, screen_path)
        torch.save(label, label_path)

        entry = {
            "id": idx,
            "screenshot": screen_path.name,
            "label": label_path.name,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "status": server_state.get("status"),
            "hash": h,
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._seen_hashes.add(h)
        self.count = idx

        # 第 1 筆 + 之後每 50 筆的第 3 筆 (idx % 50 == 3) 存一張可視化確認圖
        if idx == 1 or idx % 50 == 3:
            try:
                self._save_check_image(idx, screen_uint8, label)
                print("print check data pic")
            except Exception as exc:
                print(f"[VisionDataset] check image FAILED (idx={idx}): {exc}")

        return True

    # ----------- label 可視化確認圖 ----------- #
    _LABEL_SYMBOLS = {
        0:  ("?",   (120, 120, 120)),  # unrevealed — 灰
        1:  ("F",   (255, 180,   0)),  # flagged    — 黃
        2:  ("·",   (200, 200, 200)),  # 數字 0 (空格) — 淡灰
        3:  ("1",   ( 50, 130, 255)),  # 1 — 藍
        4:  ("2",   ( 50, 180,  80)),  # 2 — 綠
        5:  ("3",   (255,  60,  60)),  # 3 — 紅
        6:  ("4",   (  0,   0, 160)),  # 4 — 深藍
        7:  ("5",   (160,   0,   0)),  # 5 — 深紅
        8:  ("6",   (  0, 180, 180)),  # 6 — 青
        9:  ("7",   (  0,   0,   0)),  # 7 — 黑
        10: ("8",   ( 80,  80,  80)),  # 8 — 深灰
        11: ("M",   (255,   0, 255)),  # mine  — 紫紅
    }

    def _save_check_image(
        self,
        idx: int,
        screen_uint8: torch.Tensor,
        label: torch.Tensor,
    ) -> None:
        """把 screenshot 和 label 合成一張確認圖，每格左上角標示 active class 符號。

        存放路徑：{dataset_dir}/check_data/check_{idx:06d}.png
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return  # Pillow 未安裝時安靜略過

        check_dir = self.dir / "check_data"
        check_dir.mkdir(parents=True, exist_ok=True)

        # screen_uint8: (3, H_img, W_img) uint8 → PIL Image
        arr = screen_uint8.numpy().transpose(1, 2, 0)  # (H, W, 3)
        img = Image.fromarray(arr, mode="RGB")
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
            font_large = ImageFont.truetype("arial.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
            font_large = font

        grid_h, grid_w = int(label.shape[0]), int(label.shape[1])
        cell_w = img_w / grid_w
        cell_h = img_h / grid_h

        # 畫格線
        for row in range(1, grid_h):
            y = int(row * cell_h)
            draw.line([0, y, img_w, y], fill=(255, 255, 255), width=1)
        for col in range(1, grid_w):
            x = int(col * cell_w)
            draw.line([x, 0, x, img_h], fill=(255, 255, 255), width=1)

        # 每格左上角標 label
        for r in range(grid_h):
            for c in range(grid_w):
                cls = int(label[r, c].item())
                symbol, color = self._LABEL_SYMBOLS.get(cls, (str(cls), (255, 255, 255)))
                x0 = int(c * cell_w) + 4
                y0 = int(r * cell_h) + 3

                # 黑色陰影讓文字清晰
                draw.text((x0 + 1, y0 + 1), symbol, fill=(0, 0, 0), font=font_large)
                draw.text((x0,     y0),     symbol, fill=color,     font=font_large)

        out_path = check_dir / f"check_{idx:06d}.png"
        img.save(str(out_path))
        print(f"[VisionDataset] check image saved: {out_path} (idx={idx}, total={self.count})")


# --------------------------------------------------------------------------- #
# YOLO Grid-State Predictor 模型                                               #
# --------------------------------------------------------------------------- #
YOLO_FEATURE_SIZE = 40       # YOLO11n 第 6 層輸出 40x40（固定，與 screenshot 640x640 對應）
YOLO_FEATURE_CHANNELS = 128  # 第 6 層 channel 數


class YOLOGridStatePredictor(nn.Module):
    """Screenshot → 12-channel grid state（size-agnostic）。

    設計重點：
      * YOLO backbone 整個跟著 fine-tune（LR 設小一點：1e-5）
      * Query 由 grid(h, w) 座標即時算出，支援任意網格大小
      * Cross-attention 2 層（nn.TransformerDecoder），含 self-attn + cross-attn + FFN
      * 輸出 (B, 12, H, W) logits，配合 nn.CrossEntropyLoss（12 class 互斥）

    Args:
        d_model: 內部嵌入維度（跟 Stage 1 的 64 無關，此處只影響本模型內部容量）。
        nhead: multi-head attention 頭數。
        num_cross_attn_layers: cross-attention 層數（設計決定：2 層）。
        dim_feedforward: TransformerDecoderLayer 內 FFN 維度。
        dropout: dropout 比例。
        num_classes: 輸出 channel 數（預設 12，對應 Minesweeper 12 個 cell class）。
        yolo_model_path: YOLO11n checkpoint 路徑。
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_cross_attn_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = NUM_CHANNELS,
        yolo_model_path: str = "yolo11n.pt",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.memory_h = YOLO_FEATURE_SIZE
        self.memory_w = YOLO_FEATURE_SIZE
        self.num_memory_tokens = self.memory_h * self.memory_w

        # YOLO backbone（import 放在這裡避免 module import 時就載入 ultralytics / torch CUDA 初始化）
        from visual_discrete_agent import YOLO11nLastFeatureExtractor
        self.feature_extractor = YOLO11nLastFeatureExtractor(model_path=yolo_model_path)

        # YOLO (128, 40, 40) → token sequence (1600, d_model)
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(YOLO_FEATURE_CHANNELS),
            nn.Linear(YOLO_FEATURE_CHANNELS, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.memory_position = TwoDimensionalPositionEmbedding(
            self.memory_h, self.memory_w, d_model
        )

        # Coord-based query 生成器：(y_norm, x_norm) ∈ [0,1]² → d_model
        # 這也同時扮演 query 的 positional encoding（不同 cell 座標 → 不同 query 向量）
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Cross-Attention core：nn.TransformerDecoder（self-attn + cross-attn + FFN）x num_cross_attn_layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cross_attn_core = nn.TransformerDecoder(
            decoder_layer, num_layers=num_cross_attn_layers
        )

        # Classification head：d_model → 12 class
        self.classification_head = nn.Linear(d_model, num_classes)

    # --------------------- parameter group accessors --------------------- #
    def yolo_parameters(self):
        return list(self.feature_extractor.parameters())

    def non_yolo_parameters(self):
        """除了 YOLO backbone 以外的所有可學習參數（adapter / coord / cross-attn / head）。"""
        other = []
        other += list(self.token_adapter.parameters())
        other += list(self.memory_position.parameters())
        other += list(self.coord_mlp.parameters())
        other += list(self.cross_attn_core.parameters())
        other += list(self.classification_head.parameters())
        return other

    def freeze_yolo(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

    def unfreeze_yolo(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad_(True)

    # --------------------- forward pieces --------------------- #
    def _build_memory(self, screenshot: torch.Tensor) -> torch.Tensor:
        """screenshot (B, 3, 640, 640) → memory tokens (B, 1600, d_model)."""
        features = self.feature_extractor(screenshot)  # (B, 128, 40, 40)
        B = features.size(0)
        tokens = (
            features.permute(0, 2, 3, 1)
            .reshape(B, self.num_memory_tokens, YOLO_FEATURE_CHANNELS)
        )
        memory = self.token_adapter(tokens)
        memory = memory + self.memory_position().unsqueeze(0)
        return memory

    def _build_queries(
        self,
        grid_h: int,
        grid_w: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """根據 grid(h, w) 算 cell 中心的正規化座標，透過 coord_mlp 轉成 query tokens。

        Returns: (B, H*W, d_model)。每個 cell 的 query 依其 normalized position 而不同 →
        天然具備 positional encoding 的效果（size-agnostic）。
        """
        ys = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / float(grid_h)
        xs = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / float(grid_w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(grid_h * grid_w, 2)  # (H*W, 2)
        q = self.coord_mlp(coords)  # (H*W, d)
        return q.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        screenshot: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            screenshot: (B, 3, 640, 640) float tensor，已正規化到 [0, 1]。
            grid_h, grid_w: 目標網格大小。

        Returns:
            logits: (B, num_classes, grid_h, grid_w)，**尚未過 softmax**。
            搭配 nn.CrossEntropyLoss(logits, label_HW) 做訓練。
        """
        memory = self._build_memory(screenshot)  # (B, 1600, d)
        B = memory.size(0)
        queries = self._build_queries(
            grid_h, grid_w,
            batch_size=B,
            device=memory.device,
            dtype=memory.dtype,
        )
        decoded = self.cross_attn_core(queries, memory)  # (B, H*W, d)
        logits_flat = self.classification_head(decoded)  # (B, H*W, num_classes)
        logits = logits_flat.transpose(1, 2).reshape(B, self.num_classes, grid_h, grid_w)
        return logits

    # --------------------- inference helpers --------------------- #
    @torch.no_grad()
    def predict_grid_state(
        self,
        screenshot: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """推論介面：輸出 one-hot-like grid state tensor (B, 12, H, W)，可直接餵給 TransformerDiscreteAgent。

        注意：這裡把 argmax 結果轉成 one-hot（與 MinesweeperLogic.get_grid_state_tensor 同格式）。
        若要保留 soft distribution，改用 forward() 後手動 softmax 即可。
        """
        self.eval()
        logits = self.forward(screenshot, grid_h, grid_w)
        pred = logits.argmax(dim=1)  # (B, H, W)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, pred.unsqueeze(1), 1.0)
        return one_hot


# --------------------------------------------------------------------------- #
# Class index → 可讀名稱（TensorBoard 用）                                    #
# --------------------------------------------------------------------------- #
_CLASS_NAMES = [
    "hidden",           # 0
    "flag",             # 1
    "num0",             # 2
    "num1",             # 3
    "num2",             # 4
    "num3",             # 5
    "num4",             # 6
    "num5",             # 7
    "num6",             # 8
    "num7",             # 9
    "num8",             # 10
    "mine",             # 11（實際上訓練資料裡不會出現，但保留對應）
]


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
class VisionSupervisedDataset(Dataset):
    """讀取 VisionDatasetRecorder 存下的 (screenshot, label) pairs。

    每個 sample 回傳 (screenshot_float, label_long)：
        screenshot_float : torch.float32, shape (3, 640, 640), 值域 [0, 1]
        label_long       : torch.int64,   shape (H, W),        值域 [0, 11]
    """

    def __init__(self, dataset_dir: Path, entries: list[dict]):
        self.screenshots_dir = dataset_dir / "screenshots"
        self.labels_dir = dataset_dir / "labels"
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        screen_uint8 = torch.load(
            self.screenshots_dir / entry["screenshot"],
            map_location="cpu",
            weights_only=True,
        )
        label = torch.load(
            self.labels_dir / entry["label"],
            map_location="cpu",
            weights_only=True,
        )
        screenshot = screen_uint8.float() / 255.0
        return screenshot, label

    @staticmethod
    def load_and_split(
        dataset_dir: Path,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[dict], list[dict]]:
        """讀取 index.jsonl 並隨機切成 train / val。"""
        index_path = Path(dataset_dir) / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"找不到 index.jsonl：{index_path}")

        entries = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not entries:
            raise RuntimeError("index.jsonl 是空的，請先蒐集資料")

        rng = random.Random(seed)
        rng.shuffle(entries)
        n_val = max(1, int(len(entries) * val_ratio))
        val_entries = entries[:n_val]
        train_entries = entries[n_val:]
        return train_entries, val_entries


# --------------------------------------------------------------------------- #
# Trainer                                                                      #
# --------------------------------------------------------------------------- #
class YOLOGridStateTrainer:
    """監督式訓練：(screenshot, grid_label) → YOLOGridStatePredictor。

    兩階段訓練：
        Phase 1 (PHASE1_EPOCHS)：凍住 YOLO backbone，只訓練 adapter / coord / cross-attn / head。
                                  讓新加的模組先在不更動 YOLO 權重的情況下穩定。
        Phase 2 (PHASE2_EPOCHS)：解凍 YOLO，用極小 LR (LR_YOLO=1e-5) 做 fine-tune。
    """

    # ---- 超參數（可直接改這裡） ----
    BATCH_SIZE    = 8
    LR_YOLO       = 1e-5   # Phase 2 YOLO fine-tune LR（設小，避免破壞預訓練特徵）
    LR_OTHER      = 5e-4   # adapter / coord_mlp / cross-attn / head
    PHASE1_EPOCHS = 10     # 凍 YOLO 的暖機 epochs
    PHASE2_EPOCHS = 40     # 解凍 YOLO 後繼續訓練的 epochs
    VAL_RATIO     = 0.15
    SEED          = 42
    NUM_WORKERS   = 0      # Windows 多進程容易出問題，預設 0

    def __init__(
        self,
        dataset_dir,
        model_save_dir,
        grid_h: int = 6,
        grid_w: int = 6,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset / DataLoader
        train_entries, val_entries = VisionSupervisedDataset.load_and_split(
            self.dataset_dir, val_ratio=self.VAL_RATIO, seed=self.SEED
        )
        print(f"[Trainer] Dataset: train={len(train_entries)}, val={len(val_entries)}")

        self.train_loader = DataLoader(
            VisionSupervisedDataset(self.dataset_dir, train_entries),
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            VisionSupervisedDataset(self.dataset_dir, val_entries),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )

        # 模型 + loss
        self.model = YOLOGridStatePredictor().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        # optimizer / scheduler：在 train() 中依 phase 初始化
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

        # TensorBoard
        tb_root = self.model_save_dir / "tensorboard"
        tb_dir = tb_root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[Trainer] TensorBoard: tensorboard --logdir {tb_root}")
        print(f"[Trainer] Device: {self.device}")

        self.best_val_acc = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _build_optimizer(self, yolo_frozen: bool) -> None:
        """依照目前 phase 建立 optimizer（phase 轉換時重建）。"""
        if yolo_frozen:
            param_groups = [
                {"params": self.model.non_yolo_parameters(), "lr": self.LR_OTHER},
            ]
        else:
            param_groups = [
                {"params": self.model.yolo_parameters(),     "lr": self.LR_YOLO},
                {"params": self.model.non_yolo_parameters(), "lr": self.LR_OTHER},
            ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        n_pixels = 0

        for screen, label in self.train_loader:
            screen = screen.to(self.device)     # (B, 3, 640, 640)
            label = label.to(self.device)       # (B, H, W)

            logits = self.model(screen, self.grid_h, self.grid_w)  # (B, 12, H, W)
            loss = self.loss_fn(logits, label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            correct  += int((pred == label).sum().item())
            n_pixels += int(label.numel())

        return {
            "loss": total_loss / max(len(self.train_loader), 1),
            "acc":  correct / max(n_pixels, 1),
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        per_class_correct = torch.zeros(NUM_CHANNELS)
        per_class_total   = torch.zeros(NUM_CHANNELS)

        for screen, label in self.val_loader:
            screen = screen.to(self.device)
            label  = label.to(self.device)

            logits = self.model(screen, self.grid_h, self.grid_w)
            total_loss += float(self.loss_fn(logits, label).item())

            pred      = logits.argmax(dim=1).cpu()
            label_cpu = label.cpu()
            for c in range(NUM_CHANNELS):
                mask = label_cpu == c
                per_class_correct[c] += float((pred[mask] == c).sum().item())
                per_class_total[c]   += float(mask.sum().item())

        overall_acc = float(
            per_class_correct.sum() / (per_class_total.sum() + 1e-8)
        )
        return {
            "loss":          total_loss / max(len(self.val_loader), 1),
            "acc":           overall_acc,
            "acc_per_class": per_class_correct / (per_class_total + 1e-8),  # tensor(12,)
        }

    def _log_metrics(self, epoch: int, train_m: dict, val_m: dict) -> None:
        self.tb_writer.add_scalar("train/loss", train_m["loss"], epoch)
        self.tb_writer.add_scalar("train/acc",  train_m["acc"],  epoch)
        self.tb_writer.add_scalar("val/loss",   val_m["loss"],   epoch)
        self.tb_writer.add_scalar("val/acc",    val_m["acc"],    epoch)

        for c, name in enumerate(_CLASS_NAMES):
            self.tb_writer.add_scalar(
                f"val/acc_{name}", float(val_m["acc_per_class"][c]), epoch
            )
        self.tb_writer.flush()

        # Console：只印還沒學好（acc < 0.95）的 class，減少雜訊
        low_acc = [
            f"{_CLASS_NAMES[c]}={float(val_m['acc_per_class'][c]):.2f}"
            for c in range(NUM_CHANNELS)
            if float(val_m["acc_per_class"][c]) < 0.95
        ]
        print(
            f"  Epoch {epoch:03d} | "
            f"train loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} | "
            f"val loss={val_m['loss']:.4f} acc={val_m['acc']:.3f}"
            + (f" | low: {', '.join(low_acc)}" if low_acc else "")
        )

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        ckpt = {
            "epoch":        epoch,
            "model":        self.model.state_dict(),
            "best_val_acc": self.best_val_acc,
        }
        torch.save(ckpt, self.model_save_dir / "checkpoint.pth")
        if is_best:
            torch.save(ckpt, self.model_save_dir / "best.pth")
            print(f"  ★ best val_acc={self.best_val_acc:.4f} → best.pth saved")

    def load_checkpoint(self) -> int:
        """Checkpoint 讀取。回傳下一個要跑的 epoch（沒有 checkpoint 則回傳 0）。"""
        ckpt_path = self.model_save_dir / "checkpoint.pth"
        if not ckpt_path.exists():
            return 0
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        start_epoch = int(ckpt["epoch"]) + 1
        print(
            f"[Trainer] Resumed checkpoint: next_epoch={start_epoch}, "
            f"best_val_acc={self.best_val_acc:.4f}"
        )
        return start_epoch

    # ------------------------------------------------------------------ #
    # Main training entry point                                            #
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """兩階段訓練主迴圈。可中斷後重跑（自動讀取 checkpoint）。"""
        total_epochs = self.PHASE1_EPOCHS + self.PHASE2_EPOCHS
        start_epoch = self.load_checkpoint()

        current_phase = 0  # 用來偵測 phase 切換

        for epoch in range(start_epoch, total_epochs):
            in_phase1 = epoch < self.PHASE1_EPOCHS
            phase = 1 if in_phase1 else 2

            # Phase 轉換時重建 optimizer + scheduler
            if phase != current_phase:
                current_phase = phase
                if phase == 1:
                    print(f"\n[Trainer] === Phase 1 (epochs 0~{self.PHASE1_EPOCHS - 1}) — YOLO frozen ===")
                    self.model.freeze_yolo()
                    remaining = self.PHASE1_EPOCHS
                else:
                    print(f"\n[Trainer] === Phase 2 (epochs {self.PHASE1_EPOCHS}~{total_epochs - 1}) — YOLO unfrozen ===")
                    self.model.unfreeze_yolo()
                    remaining = self.PHASE2_EPOCHS

                self._build_optimizer(yolo_frozen=in_phase1)
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=remaining, eta_min=1e-6
                )

            train_m = self._train_epoch(epoch)
            val_m   = self._validate(epoch)
            self.scheduler.step()

            is_best = val_m["acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_m["acc"]

            self._log_metrics(epoch, train_m, val_m)
            self.save_checkpoint(epoch, is_best=is_best)

        print(f"\n[Trainer] Done. best_val_acc={self.best_val_acc:.4f}")
        self.tb_writer.close()


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ---- 改這裡就好 ----
    DATASET_DIR    = Path("./datasets/vision_supervised")
    MODEL_SAVE_DIR = Path("./models/yolo_grid_predictor")
    GRID_H = 6
    GRID_W = 6

    # ---- 選擇性覆蓋超參數 ----
    # YOLOGridStateTrainer.PHASE1_EPOCHS = 15
    # YOLOGridStateTrainer.BATCH_SIZE    = 4

    trainer = YOLOGridStateTrainer(
        dataset_dir    = DATASET_DIR,
        model_save_dir = MODEL_SAVE_DIR,
        grid_h         = GRID_H,
        grid_w         = GRID_W,
    )
    trainer.train()
