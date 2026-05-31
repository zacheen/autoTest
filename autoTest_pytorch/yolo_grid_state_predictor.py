"""YOLO Grid-State Predictor: supervised training from screenshot to 12-channel grid state.

Design doc: docs/DESIGN_YOLO_GRID_PREDICTOR.md

Pipeline, Phase 1 supervised training. Dimensions come from
yolo_encoder_base FINAL_DIM / TOTAL_LAYERS:
    Screenshot (B, 3, 640, 640)
      ↓ YOLO11n backbone (fine-tune, LR 1e-5)
    (B, 128, 40, 40)
      ↓ token adapter + 2D positional encoding
    (B, 1600, 128)                                   = memory tokens
      ↓ Hierarchical encoder (build_encoder_dims(final_dim, total_layers))
    (B, 1600, final_dim)
      ↓ learned queries + 2D positional encoding    = H*W queries
      ↓ Cross-Attention, num_cross_attn_layers TransformerDecoder layers
        with self-attn + cross-attn
    (B, H*W, final_dim)
      ↓ classification head Linear(final_dim, 12)
    (B, 12, H, W)

This tensor can be fed directly into frozen TransformerDiscreteAgent for Phase 2 inference.

Fixed-grid variant: adds a 2-layer encoder, uses learned query tokens, and binds
the model to the grid size used at initialization.
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
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_structure.yolo_encoder_base import (
    YOLOEncoderBase, HierarchicalEncoder, HierarchicalEncoderLayer,
    DEFAULT_ENCODER_DIMS, DEFAULT_ENCODER_FF_MULT, DEFAULT_ENCODER_FINAL_DIM,
)


# --------------------------------------------------------------------------- #
# 12-channel definitions, matching Minesweeper/MinesweeperLogic.py get_grid_state_array. #
# --------------------------------------------------------------------------- #
CH_UNREVEALED = 0
CH_FLAGGED = 1
CH_NUM_0 = 2           # Number 0, blank cell.
CH_NUM_8 = 10          # Number 8.
CH_MINE = 11           # Mine, visible only after game over.
NUM_CHANNELS = 12


# --------------------------------------------------------------------------- #
# Server API state → grid tensor                                              #
# --------------------------------------------------------------------------- #
def server_state_to_grid_tensor(server_state: dict) -> torch.Tensor:
    """Convert Minesweeper web API server_state to an (H, W) class-index LongTensor.

    Used as supervised training label for CrossEntropyLoss.

    Mapping, server cell.state to 12-channel class index:
        "hidden"                                           → 0  (CH_UNREVEALED)
        "flagged" / "flagged_mine" / "wrong_flag"          → 1  (CH_FLAGGED)
        "revealed" with value=n (0..8)                      → 2+n (CH_NUM_0 + n)
        "mine" / "hit_mine"                                → 11 (CH_MINE)
        any other unknown state                             → 0  (default unrevealed)

    Args:
        server_state: dict containing "board" as list[list[dict]], each with
            "state" and "value".

    Returns:
        torch.LongTensor with shape (H, W), values in [0, 11].
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
# Dataset recorder                                                             #
# --------------------------------------------------------------------------- #
class VisionDatasetRecorder:
    """Persist (screenshot, server_state_12ch_label) pairs to disk.

    Directory layout:
        dataset_dir/
          ├── index.jsonl              # One metadata record per line.
          ├── screenshots/
          │   └── screen_000001.pt     # uint8 tensor (3, 640, 640)
          └── labels/
              └── label_000001.pt      # int64 tensor (H, W) class index 0~11

    Deduplicates by screenshot SHA1 hash. Resume is supported by reading
    index.jsonl on open to recover existing count and hashes.
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
        """Store one (screenshot, label) pair.

        Args:
            screenshot: (3, 640, 640) float tensor in [0, 1], from preprocess_screen.
            server_state: dict from Web API.

        Returns:
            True if a new sample was added; False if invalid or duplicate.
        """
        if screenshot is None or server_state is None:
            return False
        board = server_state.get("board")
        if not board:
            return False

        # Skip post-mine-hit screens (status == "lost"). Mines are only visible
        # after game over, a state the predictor never sees during real play.
        # Collecting these samples skews CH_MINE (class 11) and wastes training.
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

        # Save a visual check image for the first sample and then idx % 50 == 3.
        if idx == 1 or idx % 50 == 3:
            try:
                self._save_check_image(idx, screen_uint8, label)
                print("print check data pic")
            except Exception as exc:
                print(f"[VisionDataset] check image FAILED (idx={idx}): {exc}")

        return True

    # ----------- label visualization check image ----------- #
    _LABEL_SYMBOLS = {
        0:  ("?",   (120, 120, 120)),  # unrevealed, gray.
        1:  ("F",   (255, 180,   0)),  # flagged, yellow.
        2:  ("·",   (200, 200, 200)),  # number 0, blank, light gray.
        3:  ("1",   ( 50, 130, 255)),  # 1, blue.
        4:  ("2",   ( 50, 180,  80)),  # 2, green.
        5:  ("3",   (255,  60,  60)),  # 3, red.
        6:  ("4",   (  0,   0, 160)),  # 4, dark blue.
        7:  ("5",   (160,   0,   0)),  # 5, dark red.
        8:  ("6",   (  0, 180, 180)),  # 6, cyan.
        9:  ("7",   (  0,   0,   0)),  # 7, black.
        10: ("8",   ( 80,  80,  80)),  # 8, dark gray.
        11: ("M",   (255,   0, 255)),  # mine, magenta.
    }

    def _save_check_image(
        self,
        idx: int,
        screen_uint8: torch.Tensor,
        label: torch.Tensor,
    ) -> None:
        """Combine screenshot and label into a check image with per-cell class marks.

        Output path: {dataset_dir}/check_data/check_{idx:06d}.png
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return  # Quietly skip if Pillow is not installed.

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

        # Draw grid lines.
        for row in range(1, grid_h):
            y = int(row * cell_h)
            draw.line([0, y, img_w, y], fill=(255, 255, 255), width=1)
        for col in range(1, grid_w):
            x = int(col * cell_w)
            draw.line([x, 0, x, img_h], fill=(255, 255, 255), width=1)

        # Draw label at each cell's top-left corner.
        for r in range(grid_h):
            for c in range(grid_w):
                cls = int(label[r, c].item())
                symbol, color = self._LABEL_SYMBOLS.get(cls, (str(cls), (255, 255, 255)))
                x0 = int(c * cell_w) + 4
                y0 = int(r * cell_h) + 3

                # Black shadow improves text readability.
                draw.text((x0 + 1, y0 + 1), symbol, fill=(0, 0, 0), font=font_large)
                draw.text((x0,     y0),     symbol, fill=color,     font=font_large)

        out_path = check_dir / f"check_{idx:06d}.png"
        img.save(str(out_path))
        print(f"[VisionDataset] check image saved: {out_path} (idx={idx}, total={self.count})")


# --------------------------------------------------------------------------- #
# YOLO Grid-State Predictor model                                              #
# --------------------------------------------------------------------------- #


class YOLOGridStatePredictor(YOLOEncoderBase):
    """Screenshot to 12-channel grid state, fixed-grid learned-query variant.

    Design notes:
      * Inherits YOLOEncoderBase: YOLO + token_adapter + HierarchicalEncoder.
        Dims come from yolo_encoder_base.DEFAULT_ENCODER_FINAL_DIM / TOTAL_LAYERS.
      * Fine-tunes the whole YOLO backbone with a small LR, 1e-5.
      * Learned queries + 2D positional embedding for a fixed grid
      * 2-layer cross-attention via nn.TransformerDecoder with self-attn + cross-attn + FFN
      * Outputs (B, 12, H, W) logits for nn.CrossEntropyLoss over 12 exclusive classes
    """

    def __init__(
        self,
        grid_h: int = 6,
        grid_w: int = 6,
        nhead: int = 4,
        num_cross_attn_layers: int = 2,
        # FFN width follows the same 4× d_model invariant as v3 — keeps the predictor's
        # cross-attn capacity consistent with the encoder/decoder elsewhere.
        dim_feedforward: int = DEFAULT_ENCODER_FINAL_DIM * DEFAULT_ENCODER_FF_MULT,
        dropout: float = 0.1,
        num_classes: int = NUM_CHANNELS,
        yolo_model_path: str = "yolo11n.pt",
    ):
        super().__init__(
            encoder_dims=DEFAULT_ENCODER_DIMS,
            nhead=nhead,
            ff_mult=DEFAULT_ENCODER_FF_MULT,
            dropout=dropout,
            yolo_model_path=yolo_model_path,
        )
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_queries = grid_h * grid_w
        self.num_classes = num_classes
        out_dim = self.out_dim

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, out_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cross_attn_core = nn.TransformerDecoder(
            decoder_layer, num_layers=num_cross_attn_layers
        )
        self.classification_head = nn.Linear(out_dim, num_classes)

    # --------------------- parameter group accessors --------------------- #
    def yolo_parameters(self):
        return list(self.feature_extractor.parameters())

    def non_yolo_parameters(self):
        """All trainable parameters except the YOLO backbone."""
        other = []
        other += list(self.token_adapter.parameters())
        other += list(self.encoder.parameters())
        other += [self.query_tokens]
        other += list(self.cross_attn_core.parameters())
        other += list(self.classification_head.parameters())
        return other

    # freeze_yolo / unfreeze_yolo / set_yolo_bn_eval are inherited from YOLOEncoderBase.
    def freeze_yolo(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

    def unfreeze_yolo(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad_(True)

    def set_yolo_bn_eval(self) -> None:
        self.set_bn_eval()

    # --------------------- forward --------------------- #
    def _build_queries(self, batch_size: int) -> torch.Tensor:
        return self.query_tokens.expand(batch_size, -1, -1)

    def forward(self, screenshot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            screenshot: (B, 3, 640, 640) float tensor normalized to [0, 1].

        Returns:
            logits: (B, num_classes, grid_h, grid_w), before softmax.
        """
        memory  = self.encode(screenshot)                            # (B, 1600, out_dim)
        queries = self._build_queries(batch_size=memory.size(0))     # (B, num_queries, out_dim)
        decoded = self.cross_attn_core(queries, memory)              # (B, num_queries, out_dim)
        logits_flat = self.classification_head(decoded)              # (B, num_queries, num_classes)
        B = memory.size(0)
        return logits_flat.transpose(1, 2).reshape(B, self.num_classes, self.grid_h, self.grid_w)

    # --------------------- inference helpers --------------------- #
    @torch.no_grad()
    def predict_grid_state(
        self,
        screenshot: torch.Tensor,
    ) -> torch.Tensor:
        """Inference API: return one-hot-like grid state tensor (B, 12, H, W).

        This output can be fed directly to TransformerDiscreteAgent. The argmax
        result is converted to one-hot, matching MinesweeperLogic.get_grid_state_array.
        Use forward() plus manual softmax if a soft distribution is needed.
        """
        self.eval()
        logits = self.forward(screenshot)
        pred = logits.argmax(dim=1)  # (B, H, W)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, pred.unsqueeze(1), 1.0)
        return one_hot


# --------------------------------------------------------------------------- #
# Class index to readable name, for TensorBoard.                               #
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
    "mine",             # 11, kept for mapping though training data should not contain it.
]


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
class VisionSupervisedDataset(Dataset):
    """Read (screenshot, label) pairs saved by VisionDatasetRecorder.

    Each sample returns (screenshot_float, label_long):
        screenshot_float : torch.float32, shape (3, 640, 640), values in [0, 1]
        label_long       : torch.int64,   shape (H, W),        values in [0, 11]

    Args:
        cache_in_ram: If True, load all samples into RAM in __init__ to remove
            disk I/O bottlenecks. Each screenshot is about 1.2 MB as uint8, so
            5000 samples are about 5.9 GB. Check RAM first.
    """

    SCENE_CANVAS_SIZE = (1080, 1920)   # H, W

    def __init__(
        self,
        dataset_dir: Path,
        entries: list[dict],
        cache_in_ram: bool = False,
        augment: bool = False,
        save_aug_debug: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.screenshots_dir = self.dataset_dir / "screenshots"
        self.labels_dir = self.dataset_dir / "labels"
        self.entries = entries
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.augment = augment
        self.save_aug_debug = save_aug_debug
        self.data_aug_dir = self.dataset_dir / "data_aug"
        if self.save_aug_debug:
            self.data_aug_dir.mkdir(parents=True, exist_ok=True)

        if cache_in_ram:
            n = len(entries)
            mb = n * 3 * 640 * 640 / 1024 / 1024
            print(f"[Dataset] Caching {n} samples into RAM (~{mb:.0f} MB uint8)…")
            for i, entry in enumerate(entries):
                screen_u8 = torch.load(
                    self.screenshots_dir / entry["screenshot"], map_location="cpu", weights_only=True
                )
                label = torch.load(
                    self.labels_dir / entry["label"], map_location="cpu", weights_only=True
                )
                self._cache[i] = (screen_u8, label)
                if (i + 1) % 500 == 0:
                    print(f"  cached {i + 1}/{n}…")
            print(f"[Dataset] Cache complete.")

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def output_image_size(cls) -> tuple[int, int]:
        canvas_h, canvas_w = cls.SCENE_CANVAS_SIZE
        return canvas_h // 2, canvas_w // 2

    @staticmethod
    def _resize_uint8_image(image_u8: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        resized = F.interpolate(
            image_u8.unsqueeze(0).float(),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).round().clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _make_background(height: int, width: int) -> torch.Tensor:
        mode = random.random()
        if mode < 0.25:
            value = 0 if random.random() < 0.5 else 255
            return torch.full((3, height, width), value, dtype=torch.uint8)
        if mode < 0.50:
            value = random.randint(0, 255)
            return torch.full((3, height, width), value, dtype=torch.uint8)
        if mode < 0.75:
            return torch.randint(0, 256, (3, height, width), dtype=torch.uint8)

        bg = torch.full(
            (3, height, width),
            random.randint(0, 255),
            dtype=torch.uint8,
        )
        for _ in range(random.randint(6, 16)):
            rect_h = random.randint(max(32, height // 12), max(64, height // 3))
            rect_w = random.randint(max(32, width // 12), max(64, width // 3))
            top = random.randint(0, max(height - rect_h, 0))
            left = random.randint(0, max(width - rect_w, 0))
            color = torch.randint(0, 256, (3, 1, 1), dtype=torch.uint8)
            bg[:, top:top + rect_h, left:left + rect_w] = color
        return bg

    def _compose_augmented_scene(self, screen_uint8: torch.Tensor) -> torch.Tensor:
        canvas_h, canvas_w = self.SCENE_CANVAS_SIZE
        bg = self._make_background(canvas_h, canvas_w)

        _, src_h, src_w = screen_uint8.shape
        scale = random.uniform(0.9, 1.1)
        paste_h = max(32, min(canvas_h, int(round(src_h * scale))))
        paste_w = max(32, min(canvas_w, int(round(src_w * scale))))
        board = self._resize_uint8_image(screen_uint8, paste_h, paste_w)

        top = random.randint(0, max(canvas_h - paste_h, 0))
        left = random.randint(0, max(canvas_w - paste_w, 0))
        bg[:, top:top + paste_h, left:left + paste_w] = board

        out_h, out_w = self.output_image_size()
        return self._resize_uint8_image(bg, out_h, out_w)

    @staticmethod
    def _light_image_jitter(image_float: torch.Tensor) -> torch.Tensor:
        contrast = random.uniform(0.9, 1.1)
        brightness = random.uniform(-0.05, 0.05)
        image_float = (image_float - 0.5) * contrast + 0.5 + brightness
        return image_float.clamp(0.0, 1.0)

    def _save_aug_debug_image(self, image_u8: torch.Tensor, entry: dict) -> None:
        entry_id = int(entry.get("id", 0))
        if entry_id <= 0 or entry_id % 100 != 0:
            return
        prefix = "aug" if self.augment else "orig"
        out_path = self.data_aug_dir / f"{prefix}_{entry_id:06d}.png"
        arr = image_u8.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr, mode="RGB").save(out_path)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        if self._cache:
            screen_uint8, label = self._cache[idx]
        else:
            screen_uint8 = torch.load(
                self.screenshots_dir / entry["screenshot"], map_location="cpu", weights_only=True
            )
            label = torch.load(
                self.labels_dir / entry["label"], map_location="cpu", weights_only=True
            )

        if self.augment:
            screen_uint8 = self._compose_augmented_scene(screen_uint8)
        else:
            _, src_h, src_w = screen_uint8.shape
            out_h = max(1, src_h // 2)
            out_w = max(1, src_w // 2)
            screen_uint8 = self._resize_uint8_image(screen_uint8, out_h, out_w)

        if self.save_aug_debug:
            self._save_aug_debug_image(screen_uint8, entry)

        image_float = screen_uint8.float() / 255.0
        if self.augment:
            image_float = self._light_image_jitter(image_float)
        return image_float, label

    @staticmethod
    def load_and_split(
        dataset_dir: Path,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[dict], list[dict]]:
        """Read index.jsonl, filter missing files, then split into train / val.

        This still works after manually deleting bad .pt files. Non-contiguous
        IDs, such as deleting screen_000641.pt while keeping screen_001641.pt,
        do not affect loading.
        """
        dataset_dir = Path(dataset_dir)
        index_path = dataset_dir / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"找不到 index.jsonl：{index_path}")

        screenshots_dir = dataset_dir / "screenshots"
        labels_dir      = dataset_dir / "labels"

        raw_entries: list[dict] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        # Keep only entries where both .pt files exist.
        valid_entries: list[dict] = []
        for entry in raw_entries:
            if (
                (screenshots_dir / entry["screenshot"]).exists()
                and (labels_dir / entry["label"]).exists()
            ):
                valid_entries.append(entry)

        skipped = len(raw_entries) - len(valid_entries)
        if skipped:
            print(f"[Dataset] Skipped {skipped} entries (files missing); {len(valid_entries)} valid")
        if not valid_entries:
            raise RuntimeError("有效 entry 為 0，請先蒐集資料或確認 .pt 檔路徑")

        rng = random.Random(seed)
        rng.shuffle(valid_entries)
        n_val = max(1, int(len(valid_entries) * val_ratio))
        return valid_entries[n_val:], valid_entries[:n_val]


# --------------------------------------------------------------------------- #
# Trainer                                                                      #
# --------------------------------------------------------------------------- #
class YOLOGridStateTrainer:
    """Supervised training: (screenshot, grid_label) to YOLOGridStatePredictor."""

    # ---- Hyperparameters, safe to edit here. ----
    TRAIN_BATCH_SIZE = 16
    ORIGINAL_BATCH_PROB = 0.20
    LR_YOLO       = 1e-5   # YOLO backbone LR
    LR_OTHER      = 5e-4   # adapter / encoder / learned queries / cross-attn / head
    WEIGHT_DECAY  = 1e-4
    TOTAL_EPOCHS  = 50
    VAL_RATIO     = 0.15
    SEED          = 42
    GRAD_CLIP_NORM = 1.0   # clip_grad_norm_ max_norm; too wide explodes, too tight slows convergence.
    # Number of parallel DataLoader workers.
    # 0 = main-process serial reads, lower GPU utilization. 2~4 = worker prefetch.
    # Windows needs an if __name__ guard, already present, so 2 is safe.
    NUM_WORKERS   = 2
    # True = load all data into RAM at startup to remove disk I/O.
    # Each screenshot is ~1.2 MB uint8; 5000 samples are ~5.9 GB. Check RAM first.
    CACHE_IN_RAM  = False

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

        use_pin  = self.device.type == "cuda"
        use_pw   = self.NUM_WORKERS > 0   # persistent_workers requires num_workers > 0.
        pf       = 4 if self.NUM_WORKERS > 0 else None  # prefetch_factor per worker

        self.train_loader_original = DataLoader(
            VisionSupervisedDataset(
                self.dataset_dir,
                train_entries,
                cache_in_ram=self.CACHE_IN_RAM,
                augment=False,
                save_aug_debug=False,
            ),
            batch_size=self.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=self.NUM_WORKERS,
            pin_memory=use_pin,
            persistent_workers=use_pw,
            prefetch_factor=pf,
        )
        self.train_loader_augmented = DataLoader(
            VisionSupervisedDataset(
                self.dataset_dir,
                train_entries,
                cache_in_ram=self.CACHE_IN_RAM,
                augment=True,
                save_aug_debug=True,
            ),
            batch_size=self.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=self.NUM_WORKERS,
            pin_memory=use_pin,
            persistent_workers=use_pw,
            prefetch_factor=pf,
        )
        self.val_loader_original = DataLoader(
            VisionSupervisedDataset(
                self.dataset_dir,
                val_entries,
                cache_in_ram=self.CACHE_IN_RAM,
                augment=False,
                save_aug_debug=False,
            ),
            batch_size=self.TRAIN_BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=use_pin,
            persistent_workers=use_pw,
            prefetch_factor=pf,
        )
        self.val_loader_augmented = DataLoader(
            VisionSupervisedDataset(
                self.dataset_dir,
                val_entries,
                cache_in_ram=self.CACHE_IN_RAM,
                augment=True,
                save_aug_debug=False,
            ),
            batch_size=self.TRAIN_BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=use_pin,
            persistent_workers=use_pw,
            prefetch_factor=pf,
        )

        # Model + loss.
        self.model = YOLOGridStatePredictor(
            grid_h=self.grid_h,
            grid_w=self.grid_w,
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        # optimizer / scheduler
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
    def _build_optimizer(self) -> None:
        """Build optimizer for joint YOLO + predictor training."""
        param_groups = [
            {"params": self.model.yolo_parameters(),     "lr": self.LR_YOLO},
            {"params": self.model.non_yolo_parameters(), "lr": self.LR_OTHER},
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=self.WEIGHT_DECAY)

    @staticmethod
    def _next_batch(loader, iterator):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    def _run_epoch_batches(self, original_loader, augmented_loader, *, training: bool) -> dict:
        if training:
            self.model.train()
            self.model.set_yolo_bn_eval()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        n_pixels = 0
        per_class_correct = torch.zeros(NUM_CHANNELS)
        per_class_total = torch.zeros(NUM_CHANNELS)

        original_iter = iter(original_loader)
        augmented_iter = iter(augmented_loader)
        num_steps = max(len(original_loader), len(augmented_loader))

        for _ in range(num_steps):
            use_original = random.random() < self.ORIGINAL_BATCH_PROB
            if use_original:
                (screen, label), original_iter = self._next_batch(original_loader, original_iter)
            else:
                (screen, label), augmented_iter = self._next_batch(augmented_loader, augmented_iter)

            screen = screen.to(self.device)
            label = label.to(self.device)

            if training:
                logits = self.model(screen)
                loss = self.loss_fn(logits, label)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.GRAD_CLIP_NORM)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits = self.model(screen)
                    loss = self.loss_fn(logits, label)

            total_loss += float(loss.item())
            pred = logits.argmax(dim=1).cpu()
            label_cpu = label.cpu()
            correct += int((pred == label_cpu).sum().item())
            n_pixels += int(label_cpu.numel())

            if not training:
                for c in range(NUM_CHANNELS):
                    mask = label_cpu == c
                    per_class_correct[c] += float((pred[mask] == c).sum().item())
                    per_class_total[c] += float(mask.sum().item())

        metrics = {
            "loss": total_loss / max(num_steps, 1),
            "acc": correct / max(n_pixels, 1),
        }
        if not training:
            metrics["acc_per_class"] = per_class_correct / (per_class_total + 1e-8)
        return metrics

    def _train_epoch(self, epoch: int) -> dict:
        return self._run_epoch_batches(
            self.train_loader_original,
            self.train_loader_augmented,
            training=True,
        )

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        return self._run_epoch_batches(
            self.val_loader_original,
            self.val_loader_augmented,
            training=False,
        )

    def _log_metrics(self, epoch: int, train_m: dict, val_m: dict) -> None:
        self.tb_writer.add_scalar("train/loss", train_m["loss"], epoch)
        self.tb_writer.add_scalar("train/acc",  train_m["acc"],  epoch)
        self.tb_writer.add_scalar("val/loss",   val_m["loss"],   epoch)
        self.tb_writer.add_scalar("val/acc",    val_m["acc"],    epoch)

        self.tb_writer.flush()

        # Console: only print classes that are not learned yet, acc < 0.95, to reduce noise.
        print(
            f"  Epoch {epoch:03d} | "
            f"train loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} | "
            f"val loss={val_m['loss']:.4f} acc={val_m['acc']:.3f}"
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
        """Load checkpoint and return the next epoch, or 0 if no checkpoint exists."""
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
        """Single-stage training loop; can resume after interruption via checkpoint."""
        total_epochs = self.TOTAL_EPOCHS
        start_epoch = self.load_checkpoint()
        self.model.unfreeze_yolo()
        self._build_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_epochs - start_epoch, 1), eta_min=1e-6
        )
        print(f"\n[Trainer] === Joint training (epochs {start_epoch}~{total_epochs - 1}) — YOLO trainable ===")

        for epoch in range(start_epoch, total_epochs):

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
    # ---- Edit here. ----
    DATASET_DIR    = Path("./datasets/vision_supervised")
    MODEL_SAVE_DIR = Path("./models/yolo_grid_predictor")
    GRID_H = 6
    GRID_W = 6

    # ---- Optional hyperparameter overrides. ----
    # YOLOGridStateTrainer.TOTAL_EPOCHS = 50
    # YOLOGridStateTrainer.BATCH_SIZE    = 4
    #
    # Suggestions for low GPU utilization, around 10%:
    #   1. NUM_WORKERS=2 is enabled by default for worker prefetch.
    #   2. If RAM is enough, >10 GB free, enable CACHE_IN_RAM:
    #      YOLOGridStateTrainer.CACHE_IN_RAM = True
    #   3. If GPU VRAM is enough, increase batch size:
    #      YOLOGridStateTrainer.BATCH_SIZE = 16

    trainer = YOLOGridStateTrainer(
        dataset_dir    = DATASET_DIR,
        model_save_dir = MODEL_SAVE_DIR,
        grid_h         = GRID_H,
        grid_w         = GRID_W,
    )
    trainer.train()
