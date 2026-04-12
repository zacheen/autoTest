import torch
import torch.nn as nn


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


class EncoderDecoderTransformer(nn.Module):
    """Shared encoder-decoder transformer core used by grid and visual agents."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
