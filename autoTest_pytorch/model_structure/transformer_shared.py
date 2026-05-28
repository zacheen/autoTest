import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoDimensionalPositionEmbedding(nn.Module):
    """Learned 2D positional embedding flattened into a token sequence."""

    def __init__(self, height, width, d_model):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for 2D position embedding, got {d_model}")

        self.height = height
        self.width = width
        self.row_embed = nn.Embedding(height, d_model // 2)
        self.col_embed = nn.Embedding(width, d_model // 2)

        rows = torch.arange(height).unsqueeze(1).expand(height, width).reshape(-1)
        cols = torch.arange(width).unsqueeze(0).expand(height, width).reshape(-1)
        self.register_buffer("row_indices", rows)
        self.register_buffer("col_indices", cols)

    def forward(self):
        return torch.cat([
            self.row_embed(self.row_indices),
            self.col_embed(self.col_indices),
        ], dim=-1)


class FixedSinusoidalPositionEmbedding(nn.Module):
    """Fixed 2D sinusoidal positional encoding，無可學習參數。

    輸出 shape：(H*W, d_model)，可直接加到 token sequence 上。
    結果依 (height, width, device, dtype) 自動 cache，不重複計算。

    d_model 必須能被 4 整除（拆成 4 等份：sin/cos × row/col）。
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(
                f"FixedSinusoidalPositionEmbedding 需要 d_model % 4 == 0，got {d_model}"
            )
        self.d_model = d_model
        self._cache: dict = {}
        self.register_buffer("_device_tracker", torch.zeros(1), persistent=False)

    def forward(self, height: int, width: int) -> torch.Tensor:
        """回傳 (H*W, d_model) 的 positional encoding tensor。

        第一次呼叫時計算並 cache；後續相同 (height, width, device, dtype) 直接回傳。
        Device / dtype 與 module 的第一個 buffer 對齊；若無 buffer 則用 CPU float32。
        """
        # 找出目前 module 所在的 device/dtype（透過 dummy buffer 或預設值）
        try:
            ref = next(self.buffers())
            device, dtype = ref.device, ref.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32

        key = (height, width, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        quarter_dim = self.d_model // 4
        half_dim    = self.d_model // 2
        ys = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=torch.float32)
        xs = torch.linspace(0.0, 1.0, steps=width,  device=device, dtype=torch.float32)
        div_term = torch.exp(
            torch.arange(0, quarter_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(quarter_dim, 1))
        )
        y_angles = ys.unsqueeze(1) * div_term.unsqueeze(0)
        x_angles = xs.unsqueeze(1) * div_term.unsqueeze(0)
        y_embed = torch.cat([torch.sin(y_angles), torch.cos(y_angles)], dim=1)
        x_embed = torch.cat([torch.sin(x_angles), torch.cos(x_angles)], dim=1)
        pos = torch.cat([
            y_embed.unsqueeze(1).expand(height, width, half_dim),
            x_embed.unsqueeze(0).expand(height, width, half_dim),
        ], dim=2).reshape(height * width, self.d_model).to(dtype=dtype)

        self._cache[key] = pos
        return pos


class HierarchicalEncoderLayer(nn.Module):
    """Self-attention block at d_in, then optional projection to d_out.

    For uniform-dim configurations (d_in == d_out) the projection is `nn.Identity()`
    so the layer is functionally equivalent to a plain `nn.TransformerEncoderLayer`
    — just with one extra `attn.` prefix in the state_dict keys.
    """

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
    """Encoder whose d_model can shrink layer-by-layer (or stay uniform).

    Shared by Stage 1 (`EncoderDecoderTransformer`) and Stage 2 (`YOLOEncoderBase`):
    uniform-dim config gives a plain transformer stack; shrinking dims gives a
    geometric-compression stack. Both agents read `DEFAULT_ENCODER_DIMS` from
    `yolo_encoder_base` so a single constant change retunes both networks.
    """

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


class EncoderDecoderTransformer(nn.Module):
    """Shared encoder-decoder transformer core used by grid and visual agents.

    Encoder uses HierarchicalEncoder with uniform dims ([d_model] * (num_layers + 1)),
    same building block V3 uses. Lets a single arch tweak (e.g. switching START_DIM)
    propagate to Stage 1 and V3 together. The HierarchicalEncoderLayer wrapper introduces
    one extra `attn.` prefix in encoder state_dict keys — callers loading old (pre-refactor)
    checkpoints should auto-migrate (see TransformerActorNetwork.load_backbone_state).
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        if dim_feedforward % d_model != 0:
            raise ValueError(
                f"dim_feedforward ({dim_feedforward}) must be a multiple of d_model "
                f"({d_model}) so it can be expressed as ff_mult for HierarchicalEncoder; "
                f"got remainder {dim_feedforward % d_model}"
            )
        ff_mult = dim_feedforward // d_model

        # Uniform-dim encoder: dims=[d_model]*(num_layers + 1) → num_layers HierarchicalEncoderLayer
        # instances, each with proj=Identity (since d_in == d_out). Functionally identical to the
        # previous nn.TransformerEncoder stack; only state_dict key paths differ (extra .attn prefix).
        self.transformer = HierarchicalEncoder(
            dims=[d_model] * (num_layers + 1),
            nhead=nhead,
            ff_mult=ff_mult,
            dropout=dropout,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def encode(self, memory_tokens):
        return self.transformer(memory_tokens)

    def decode(self, query_tokens, memory_tokens):
        return self.decoder(query_tokens, memory_tokens)

    def forward(self, memory_tokens, query_tokens):
        memory = self.encode(memory_tokens)
        return self.decode(query_tokens, memory)


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network: token features -> V(s) + A(s,a)."""

    def __init__(self, d_model=64, grid_h=10, grid_w=10):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.value_stream = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, features):
        v = self.value_stream(features.mean(dim=1))
        a = self.advantage_stream(features).squeeze(-1)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q.view(-1, self.grid_h, self.grid_w)


class CosineQuantileEmbedding(nn.Module):
    """Embed quantile fractions with cosine features."""

    def __init__(self, d_model=64, num_cosines=64):
        super().__init__()
        self.num_cosines = num_cosines
        self.register_buffer(
            "frequencies",
            torch.arange(1, num_cosines + 1, dtype=torch.float32).view(1, 1, -1),
        )
        self.proj = nn.Sequential(
            nn.Linear(num_cosines, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, taus):
        cosines = torch.cos(torch.pi * taus.unsqueeze(-1) * self.frequencies)
        return self.proj(cosines)


class FQFQNetwork(nn.Module):
    """FQF head: learned quantile fractions + per-action quantile values."""

    def __init__(
        self,
        d_model=64,
        grid_h=10,
        grid_w=10,
        num_fractions=16,
        num_cosines=64,
        hidden_dim=64,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w
        self.num_fractions = num_fractions

        self.fraction_proposal = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_fractions),
        )
        self.quantile_embedding = CosineQuantileEmbedding(
            d_model=d_model,
            num_cosines=num_cosines,
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        batch_size = features.size(0)
        state_summary = features.mean(dim=1)
        fraction_logits = self.fraction_proposal(state_summary)
        fraction_probs = F.softmax(fraction_logits, dim=-1)

        taus = torch.cumsum(fraction_probs, dim=-1)
        taus = torch.cat(
            [torch.zeros(batch_size, 1, device=features.device, dtype=features.dtype), taus],
            dim=-1,
        )
        tau_hats = 0.5 * (taus[:, :-1] + taus[:, 1:])

        tau_embeddings = self.quantile_embedding(tau_hats)
        fused = features.unsqueeze(2) * tau_embeddings.unsqueeze(1)
        quantiles = self.value_head(fused).squeeze(-1)
        q_values = torch.sum(quantiles * fraction_probs.unsqueeze(1), dim=-1)

        return {
            "q_values": q_values.view(-1, self.grid_h, self.grid_w),
            "quantiles": quantiles,
            "taus": taus,
            "tau_hats": tau_hats,
            "fraction_probs": fraction_probs,
        }
