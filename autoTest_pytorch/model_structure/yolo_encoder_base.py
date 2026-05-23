"""Shared YOLO backbone + hierarchical encoder base.

Both YOLOGridStatePredictor and VisualAgentV3 inherit from YOLOEncoderBase.
Subclasses add their own decoder on top of encode().

Pipeline:
    screenshot (B, 3, H, W)
      ↓ YOLO11n backbone
    (B, 128, 40, 40)
      ↓ token_adapter: LayerNorm(128) + Linear(128→encoder_dims[0])
      ↓ + fixed 2D sinusoidal positional encoding
    (B, 1600, encoder_dims[0])
      ↓ HierarchicalEncoder (dims built by build_encoder_dims(final_dim, total_layers))
    (B, 1600, final_dim)                             ← encode() output

Encoder shape is controlled by DEFAULT_ENCODER_FINAL_DIM / DEFAULT_ENCODER_TOTAL_LAYERS
below — geometric halving from 128 down to final_dim, then uniform layers at final_dim.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model_structure.transformer_shared import FixedSinusoidalPositionEmbedding


YOLO_FEATURE_CHANNELS = 128
DEFAULT_ENCODER_NHEAD = 4
DEFAULT_ENCODER_FF_MULT = 4
DEFAULT_ENCODER_DROPOUT = 0.1

# Encoder shape — geometric compression (128 → 64 → ... → final_dim) followed by
# uniform-dim layers until reaching total_layers. Both v3 and YOLOGridStatePredictor
# read DEFAULT_ENCODER_DIMS; change FINAL_DIM / TOTAL_LAYERS here to update both.
DEFAULT_ENCODER_FINAL_DIM    = 64  # 編碼最終 dim (= decoder d_model)
DEFAULT_ENCODER_TOTAL_LAYERS = 1   # encoder 總層數 (壓縮 + uniform);須 ≥ log2(128/final_dim)


def build_encoder_dims(
    final_dim: int,
    total_layers: int,
    nhead: int = DEFAULT_ENCODER_NHEAD,
) -> list[int]:
    """建立 HierarchicalEncoder 的 dims list。

    從 YOLO_FEATURE_CHANNELS 每次除二降到 final_dim(壓縮段),再把 final_dim 重複
    補滿 uniform 層直到層數 = total_layers。回傳的 list 長度 = total_layers + 1,
    第一個元素是 token_adapter 輸入 dim (= YOLO_FEATURE_CHANNELS),其餘 total_layers
    個是每層的輸出 dim。

    範例:
        build_encoder_dims(32, 5) → [128, 64, 32, 32, 32, 32]   (2 壓縮 + 3 uniform)
        build_encoder_dims(64, 4) → [128, 64, 64, 64, 64]       (1 壓縮 + 3 uniform)
        build_encoder_dims(32, 2) → [128, 64, 32]               (純壓縮,無 uniform)
        build_encoder_dims(128, 4) → [128, 128, 128, 128, 128]  (純 uniform,無壓縮)

    Raises:
        ValueError: final_dim 不是 2 的次方、不在 [nhead, YOLO_FEATURE_CHANNELS] 區間、
                    YOLO_FEATURE_CHANNELS 不能整除 final_dim、或 total_layers 不夠壓到目標。
    """
    if not isinstance(final_dim, int) or final_dim <= 0:
        raise ValueError(f"final_dim must be a positive int, got {final_dim!r}")
    if not isinstance(total_layers, int) or total_layers < 1:
        raise ValueError(f"total_layers must be int >= 1, got {total_layers!r}")
    if final_dim > YOLO_FEATURE_CHANNELS:
        raise ValueError(
            f"final_dim ({final_dim}) must be <= YOLO_FEATURE_CHANNELS ({YOLO_FEATURE_CHANNELS})"
        )
    if final_dim < nhead:
        raise ValueError(f"final_dim ({final_dim}) must be >= nhead ({nhead})")
    if final_dim % nhead != 0:
        raise ValueError(f"final_dim ({final_dim}) must be divisible by nhead ({nhead})")
    ratio = YOLO_FEATURE_CHANNELS // final_dim
    if YOLO_FEATURE_CHANNELS % final_dim != 0 or (ratio & (ratio - 1)) != 0:
        raise ValueError(
            f"YOLO_FEATURE_CHANNELS ({YOLO_FEATURE_CHANNELS}) / final_dim ({final_dim}) "
            f"must be a power of 2 (got ratio={ratio}); final_dim must be a power-of-2 "
            f"divisor of YOLO_FEATURE_CHANNELS."
        )

    # 壓縮段:128 → 64 → ... → final_dim
    compression: list[int] = []
    d = YOLO_FEATURE_CHANNELS
    while d > final_dim:
        compression.append(d)
        d //= 2
    compression.append(final_dim)
    num_compression_layers = len(compression) - 1

    if total_layers < num_compression_layers:
        raise ValueError(
            f"total_layers ({total_layers}) is less than the {num_compression_layers} "
            f"compression layers needed to reach final_dim={final_dim} from "
            f"YOLO_FEATURE_CHANNELS={YOLO_FEATURE_CHANNELS}"
        )

    uniform_layers = total_layers - num_compression_layers
    return compression + [final_dim] * uniform_layers


DEFAULT_ENCODER_DIMS = build_encoder_dims(
    DEFAULT_ENCODER_FINAL_DIM, DEFAULT_ENCODER_TOTAL_LAYERS
)


class HierarchicalEncoderLayer(nn.Module):
    """Self-attention block at d_in, then optional projection to d_out."""

    def __init__(self, d_in: int, d_out: int, nhead: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=d_in, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.proj = (
            nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out))
            if d_in != d_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.attn(x))


class HierarchicalEncoder(nn.Module):
    """Encoder whose d_model shrinks layer-by-layer."""

    def __init__(self, dims: list[int], nhead: int, ff_mult: int, dropout: float):
        super().__init__()
        if len(dims) < 2:
            raise ValueError(f"HierarchicalEncoder needs at least 2 dims, got {dims}")
        for d in dims:
            if d % nhead != 0:
                raise ValueError(f"dim {d} must be divisible by nhead {nhead}")
        self.layers = nn.ModuleList([
            HierarchicalEncoderLayer(d_in, d_out, nhead, d_in * ff_mult, dropout)
            for d_in, d_out in zip(dims[:-1], dims[1:])
        ])
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


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

    def encode(self, screenshot: torch.Tensor) -> torch.Tensor:
        """screenshot (B, 3, H, W) → encoded memory tokens (B, N, out_dim)."""
        features = self.feature_extractor(screenshot)
        B, _, h, w = features.shape
        tokens = features.permute(0, 2, 3, 1).reshape(B, h * w, YOLO_FEATURE_CHANNELS)
        memory = self.token_adapter(tokens)
        memory = memory + self.memory_position(h, w).unsqueeze(0)
        return self.encoder(memory)

    # ── freeze / unfreeze ─────────────────────────────────────────────────

    def freeze(self) -> None:
        """Freeze only the base components (feature_extractor, token_adapter, encoder)."""
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        for p in self.token_adapter.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
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
