"""Shared YOLO backbone + hierarchical encoder base.

Both YOLOGridStatePredictor and VisualAgentV3 inherit from YOLOEncoderBase.
Subclasses add their own decoder on top of encode().

Pipeline:
    screenshot (B, 3, H, W)
      ↓ YOLO11n backbone
    (B, 128, 40, 40)
      ↓ token_adapter: LayerNorm(128) + Linear(128→128)
      ↓ + fixed 2D sinusoidal positional encoding
    (B, 1600, 128)
      ↓ HierarchicalEncoder [128→64→32]
    (B, 1600, 32)                                    ← encode() output
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model_structure.transformer_shared import FixedSinusoidalPositionEmbedding


YOLO_FEATURE_CHANNELS = 128
DEFAULT_ENCODER_DIMS  = [128, 64, 32]
DEFAULT_ENCODER_NHEAD = 4
DEFAULT_ENCODER_FF_MULT = 4
DEFAULT_ENCODER_DROPOUT = 0.1


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
