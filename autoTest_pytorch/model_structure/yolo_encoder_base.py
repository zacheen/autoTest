"""Shared YOLO backbone + hierarchical encoder base.

Both YOLOGridStatePredictor and VisualAgentV3 inherit from YOLOEncoderBase.
Subclasses add their own decoder on top of encode().

Pipeline (split into two halves so V3 can cache the backbone output in replay):
    screenshot (B, 3, H, W)
      ↓ YOLO11n backbone                            ← extract_backbone_features()
    (B, 128, 40, 40)                                ← V3 stores this in replay buffer
      ↓ token_adapter: LayerNorm(128) + Linear(128→encoder_dims[0])
      ↓ + fixed 2D sinusoidal positional encoding   ← encode_from_backbone_features()
      ↓ HierarchicalEncoder (dims built by build_encoder_dims(final_dim, total_layers))
    (B, 1600, final_dim)                             ← encode() output

`encode()` is the high-level wrapper (screenshot → encoded memory). V3 calls the two
halves separately so the frozen YOLO forward only runs once at storage time, and the
trainable token_adapter + encoder run every gradient step on cached features.

Freezing API:
    freeze()                  — legacy: freezes feature_extractor + token_adapter + encoder
    freeze_feature_extractor() — V3 path: only freezes YOLO11n; lets encoder train
    unfreeze() / set_bn_eval() — unchanged

Encoder shape is controlled by DEFAULT_ENCODER_FINAL_DIM / DEFAULT_ENCODER_TOTAL_LAYERS /
DEFAULT_ENCODER_START_DIM below — geometric halving from start_dim down to final_dim, then
uniform layers at final_dim. Default config (start_dim == final_dim) gives an all-uniform
encoder identical in shape to Stage 1's EncoderDecoderTransformer, so Stage 1 weights load 1:1.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model_structure.transformer_shared import (
    FixedSinusoidalPositionEmbedding,
    HierarchicalEncoder,
    HierarchicalEncoderLayer,
)

# Re-export so existing callers (yolo_grid_state_predictor, future consumers) can keep
# importing these names from yolo_encoder_base. Authoritative definitions live in
# transformer_shared so Stage 1's EncoderDecoderTransformer can use them too without
# creating a circular dependency.
__all__ = ["HierarchicalEncoder", "HierarchicalEncoderLayer"]


YOLO_FEATURE_CHANNELS = 128
DEFAULT_ENCODER_NHEAD = 4
DEFAULT_ENCODER_FF_MULT = 4
DEFAULT_ENCODER_DROPOUT = 0.1

# Encoder shape — 從 START_DIM 開始幾何壓縮 (每層除二) 直到 FINAL_DIM,再用 uniform
# 層補滿到 TOTAL_LAYERS。START_DIM == FINAL_DIM 時 encoder 內不做壓縮,YOLO 128-ch →
# FINAL_DIM 的維度轉換由 token_adapter (LayerNorm + Linear) 負責;此時 V3 encoder 就跟
# Stage 1 的 EncoderDecoderTransformer 結構一致,可直接載 Stage 1 權重。把 START_DIM
# 改回 YOLO_FEATURE_CHANNELS 則 encoder 第一層做 self-attn at d=128 再線性壓到 FINAL_DIM。
# Both v3 and YOLOGridStatePredictor read DEFAULT_ENCODER_DIMS.
DEFAULT_ENCODER_FINAL_DIM    = 64  # 編碼最終 dim (= decoder d_model)
DEFAULT_ENCODER_TOTAL_LAYERS = 4   # encoder 總層數 (壓縮 + uniform);須 ≥ log2(start_dim/final_dim)
DEFAULT_ENCODER_START_DIM    = DEFAULT_ENCODER_FINAL_DIM   # = FINAL_DIM → encoder 全 uniform


def build_encoder_dims(
    final_dim: int,
    total_layers: int,
    nhead: int = DEFAULT_ENCODER_NHEAD,
    start_dim: int = YOLO_FEATURE_CHANNELS,
) -> list[int]:
    """建立 HierarchicalEncoder 的 dims list。

    從 start_dim 每次除二降到 final_dim(壓縮段),再把 final_dim 重複
    補滿 uniform 層直到層數 = total_layers。回傳的 list 長度 = total_layers + 1,
    第一個元素是 token_adapter 輸出 dim (= start_dim),其餘 total_layers
    個是每層的輸出 dim。

    start_dim 預設 = YOLO_FEATURE_CHANNELS,token_adapter 不做維度壓縮、encoder 內負責;
    若 start_dim == final_dim,encoder 全 uniform,token_adapter 負責把 YOLO_FEATURE_CHANNELS
    壓到 final_dim(此時結構等同 Stage 1 的 EncoderDecoderTransformer)。

    範例:
        build_encoder_dims(32, 5)               → [128, 64, 32, 32, 32, 32]   (2 壓縮 + 3 uniform)
        build_encoder_dims(64, 4)               → [128, 64, 64, 64, 64]       (1 壓縮 + 3 uniform)
        build_encoder_dims(32, 2)               → [128, 64, 32]               (純壓縮,無 uniform)
        build_encoder_dims(128, 4)              → [128, 128, 128, 128, 128]   (純 uniform,無壓縮)
        build_encoder_dims(64, 4, start_dim=64) → [64, 64, 64, 64, 64]        (壓縮交給 token_adapter,encoder 純 uniform)

    Raises:
        ValueError: final_dim/start_dim 不是 2 的次方、不在 [nhead, start_dim] 區間、
                    start_dim 不能整除 final_dim、或 total_layers 不夠壓到目標。
    """
    if not isinstance(final_dim, int) or final_dim <= 0:
        raise ValueError(f"final_dim must be a positive int, got {final_dim!r}")
    if not isinstance(total_layers, int) or total_layers < 1:
        raise ValueError(f"total_layers must be int >= 1, got {total_layers!r}")
    if not isinstance(start_dim, int) or start_dim <= 0:
        raise ValueError(f"start_dim must be a positive int, got {start_dim!r}")
    if final_dim > start_dim:
        raise ValueError(
            f"final_dim ({final_dim}) must be <= start_dim ({start_dim})"
        )
    if final_dim < nhead:
        raise ValueError(f"final_dim ({final_dim}) must be >= nhead ({nhead})")
    if final_dim % nhead != 0:
        raise ValueError(f"final_dim ({final_dim}) must be divisible by nhead ({nhead})")
    ratio = start_dim // final_dim
    if start_dim % final_dim != 0 or (ratio & (ratio - 1)) != 0:
        raise ValueError(
            f"start_dim ({start_dim}) / final_dim ({final_dim}) must be a power of 2 "
            f"(got ratio={ratio}); final_dim must be a power-of-2 divisor of start_dim."
        )

    # 壓縮段:start_dim → start_dim/2 → ... → final_dim
    compression: list[int] = []
    d = start_dim
    while d > final_dim:
        compression.append(d)
        d //= 2
    compression.append(final_dim)
    num_compression_layers = len(compression) - 1

    if total_layers < num_compression_layers:
        raise ValueError(
            f"total_layers ({total_layers}) is less than the {num_compression_layers} "
            f"compression layers needed to reach final_dim={final_dim} from "
            f"start_dim={start_dim}"
        )

    uniform_layers = total_layers - num_compression_layers
    return compression + [final_dim] * uniform_layers


DEFAULT_ENCODER_DIMS = build_encoder_dims(
    DEFAULT_ENCODER_FINAL_DIM,
    DEFAULT_ENCODER_TOTAL_LAYERS,
    start_dim=DEFAULT_ENCODER_START_DIM,
)


class YOLOEncoderBase(nn.Module):
    """YOLO backbone + token_adapter + fixed sinusoidal pos encoding + HierarchicalEncoder.

    Subclasses keep their own decoder.  Call encode(screenshot) to get
    memory tokens (B, N, out_dim) ready for cross-attention.
    """

    def __init__(
        self,
        encoder_dims: list[int] = DEFAULT_ENCODER_DIMS,
        nhead: int = DEFAULT_ENCODER_NHEAD,
        ff_mult: int = DEFAULT_ENCODER_FF_MULT,
        dropout: float = DEFAULT_ENCODER_DROPOUT,
        yolo_model_path: str = "yolo11n.pt",
    ):
        super().__init__()
        from visual_discrete_agent import YOLO11nLastFeatureExtractor
        self.feature_extractor = YOLO11nLastFeatureExtractor(model_path=yolo_model_path)

        in_dim = encoder_dims[0]
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(YOLO_FEATURE_CHANNELS),
            nn.Linear(YOLO_FEATURE_CHANNELS, in_dim),
        )
        self.memory_position = FixedSinusoidalPositionEmbedding(in_dim)
        self.encoder = HierarchicalEncoder(
            dims=encoder_dims, nhead=nhead, ff_mult=ff_mult, dropout=dropout,
        )
        self.out_dim = encoder_dims[-1]

    # ── forward ──────────────────────────────────────────────────────────

    def extract_backbone_features(self, screenshot: torch.Tensor) -> torch.Tensor:
        """screenshot (B, 3, H, W) → raw YOLO11n features (B, 128, h, w).

        This is the half that V3 caches in the replay buffer. The YOLO11n backbone
        is typically frozen (freeze_feature_extractor()) so this forward is
        deterministic across training steps for a given screenshot.
        """
        return self.feature_extractor(screenshot)

    def encode_from_backbone_features(self, features: torch.Tensor) -> torch.Tensor:
        """YOLO features (B, 128, h, w) → encoded memory tokens (B, h*w, out_dim).

        Runs the trainable half: token_adapter (LayerNorm + Linear) + fixed
        sinusoidal positional encoding + HierarchicalEncoder. V3 calls this every
        gradient step on cached features pulled from the replay buffer.
        """
        B, _, h, w = features.shape
        tokens = features.permute(0, 2, 3, 1).reshape(B, h * w, YOLO_FEATURE_CHANNELS)
        memory = self.token_adapter(tokens)
        memory = memory + self.memory_position(h, w).unsqueeze(0)
        return self.encoder(memory)

    def encode(self, screenshot: torch.Tensor) -> torch.Tensor:
        """screenshot (B, 3, H, W) → encoded memory tokens (B, N, out_dim).

        High-level wrapper used by YOLOGridStatePredictor (supervised pretrain) and
        by VisualAgentV3.select_action() (live inference, no replay cache). For
        training-time forwards on cached features, call encode_from_backbone_features
        directly.
        """
        return self.encode_from_backbone_features(self.extract_backbone_features(screenshot))

    # ── freeze / unfreeze ─────────────────────────────────────────────────

    def freeze(self) -> None:
        """Freeze feature_extractor + token_adapter + encoder (legacy full freeze).

        Used by callers that want the whole pretrained stack frozen. V3 instead
        calls freeze_feature_extractor() so the encoder can fine-tune on RL signal.
        """
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        for p in self.token_adapter.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def freeze_feature_extractor(self) -> None:
        """Freeze only the YOLO11n backbone; keep token_adapter + encoder trainable.

        V3 path: lets the encoder specialize self-attention for Q-value estimation
        while keeping the generic vision feature extractor stable.
        """
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad_(True)
        for p in self.token_adapter.parameters():
            p.requires_grad_(True)
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def set_bn_eval(self) -> None:
        """Set all BatchNorm layers in feature_extractor to eval mode (fix running stats)."""
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    # ── checkpoint ────────────────────────────────────────────────────────

    def load_encoder_weights_from_checkpoint(self, path: str | Path) -> None:
        """Load feature_extractor, token_adapter, encoder weights from a predictor checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLOGridStatePredictor checkpoint not found: {path}\n"
                "Run python yolo_grid_state_predictor.py first."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        full_state = ckpt["model"]
        PREFIXES = ("feature_extractor.", "token_adapter.", "encoder.")
        encoder_state = {
            k: v for k, v in full_state.items()
            if any(k.startswith(p) for p in PREFIXES)
        }
        if not encoder_state:
            raise RuntimeError(
                f"No encoder keys found in {path}.\n"
                f"Available keys (sample): {list(full_state.keys())[:10]}"
            )
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        print(
            f"[YOLOEncoderBase] Loaded {len(encoder_state)} tensors "
            f"(val_acc={ckpt.get('best_val_acc', '?')})"
        )
        if missing:
            print(f"  missing  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
