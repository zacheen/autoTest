import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import random
import shutil
from collections import defaultdict, deque
from PIL import Image
import atexit
import datetime
from pathlib import Path
from ultralytics import YOLO


class ScaledSigmoid(nn.Module):
    """ScaledSigmoid activation: out = scale * sigmoid(x) + shift.

    Default: scale=1.1, shift=-0.05 → output range ≈ [-0.05, 1.05]
    Values in [0, 1] map to valid grid coordinates.
    Values outside [0, 1] are out-of-bounds.
    """
    def __init__(self, scale=1.1, shift=-0.05):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.scale * torch.sigmoid(x) + self.shift


# Hyperparameters
BATCH_SIZE = 32
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005
INIT_ALPHA = 0.2
TARGET_ENTROPY = -2.0  # = -action_dim (for continuous SAC)
DISCRETE_TARGET_ENTROPY = 0.8 * np.log(100)  # ≈ 3.7 (80% of max discrete entropy ln(100)=4.6)
LR_ALPHA_DISCRETE = 1e-5  # 比 continuous 慢 30 倍，避免 alpha 降太快
ALPHA_MAX = 0.3
ALPHA_MIN = 0.15
BUFFER_CAPACITY = 2000   # Runtime circular buffer in CPU RAM (~1.44GB)
SAVE_CAPACITY = 150     # Persistent save to disk (~360MB)
SAVE_EVERY_N_EPISODES = 50
IMAGE_SIZE = (640, 640)

# YOLO11n layer indices (discovered via forward pass)
# Layer 4: C3k2, output 128ch x 80x80 (mid-layer, spatial detail)
# Layer 6: C3k2, output 128ch x 40x40 (last backbone layer, semantic)
YOLO_MID_LAYER_IDX = 4
YOLO_LAST_LAYER_IDX = 6
YOLO_MID_CHANNELS = 128
YOLO_LAST_CHANNELS = 128
FUSED_CHANNELS = YOLO_MID_CHANNELS + YOLO_LAST_CHANNELS  # 256

# Stage 1 預訓練參數
GRID_STATE_CHANNELS = 12  # one-hot channel 數量 (MinesweeperLogic.NUM_CHANNELS)

# Paths
LOG_ACTIONS = True
ACTION_LOG_PATH = Path("./models/action_logs")
REPLAY_SAVE_PATH = Path("./models/replay_buffer")
REPLAY_PERSISTENT_PATH = Path("./models/replay_buffer_save")
MODEL_SAVE_PATH = Path("./models")

# Stage 1 paths
STAGE1_MODEL_PATH = Path("./models/stage1")
STAGE1_REPLAY_PATH = Path("./models/stage1/replay_buffer")
STAGE1_REPLAY_PERSISTENT_PATH = Path("./models/stage1/replay_buffer_save")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLO11nBackbone(nn.Module):
    """YOLO11n as feature extractor with mid+last layer fusion."""

    def __init__(self, model_path='yolo11n.pt'):
        super().__init__()

        # Load pretrained YOLO11n
        yolo = YOLO(model_path)
        self.backbone_layers = nn.ModuleList()

        # Extract only backbone layers (up to and including our target layers)
        max_layer = max(YOLO_MID_LAYER_IDX, YOLO_LAST_LAYER_IDX)
        for idx in range(max_layer + 1):
            self.backbone_layers.append(yolo.model.model[idx])

        # Hook storage
        self.mid_features = None
        self.last_features = None

        # Register forward hooks
        self.backbone_layers[YOLO_MID_LAYER_IDX].register_forward_hook(self._hook_mid)
        self.backbone_layers[YOLO_LAST_LAYER_IDX].register_forward_hook(self._hook_last)

        # Channel reduction after fusion: 256ch -> 128ch
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(FUSED_CHANNELS, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        # Verify shapes with a dummy forward pass
        self._verify_shapes()

    def _hook_mid(self, module, input, output):
        self.mid_features = output

    def _hook_last(self, module, input, output):
        self.last_features = output

    def _verify_shapes(self):
        """Run a dummy forward pass to verify hook outputs match expected shapes."""
        dummy = torch.randn(1, 3, *IMAGE_SIZE)
        with torch.no_grad():
            self._forward_backbone(dummy)

        assert self.mid_features is not None, "Mid-layer hook did not fire"
        assert self.last_features is not None, "Last-layer hook did not fire"

        _, mc, mh, mw = self.mid_features.shape
        _, lc, lh, lw = self.last_features.shape

        assert mc == YOLO_MID_CHANNELS, f"Mid channels: expected {YOLO_MID_CHANNELS}, got {mc}"
        assert lc == YOLO_LAST_CHANNELS, f"Last channels: expected {YOLO_LAST_CHANNELS}, got {lc}"
        assert mh == 80 and mw == 80, f"Mid spatial: expected 80x80, got {mh}x{mw}"
        assert lh == 40 and lw == 40, f"Last spatial: expected 40x40, got {lh}x{lw}"

        print(f"YOLO11n backbone verified: mid={mc}x{mh}x{mw}, last={lc}x{lh}x{lw}")

    def _forward_backbone(self, x):
        """Forward through backbone layers sequentially."""
        for layer in self.backbone_layers:
            x = layer(x)
        return x

    def forward(self, x):
        """Forward pass returning fused feature maps.

        Args:
            x: (B, 3, 640, 640) input images
        Returns:
            (B, 128, 80, 80) fused and reduced feature maps
        """
        self._forward_backbone(x)

        # Upsample last (40x40) to match mid (80x80)
        last_up = F.interpolate(
            self.last_features,
            size=self.mid_features.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # Concatenate: 128 + 128 = 256 channels at 80x80
        fused = torch.cat([self.mid_features, last_up], dim=1)

        # Reduce channels: 256 -> 128
        fused = self.channel_reduce(fused)

        return fused


class LocalAttentionLayer(nn.Module):
    """Local attention: each position attends to its window_size×window_size neighborhood."""

    def __init__(self, in_channels, out_channels, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.q_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

        # Residual projection when channel dimensions differ
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Project Q, K, V
        q = self.q_proj(x)  # (B, C_out, H, W)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Pad to make H, W divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        _, C_out, Hp, Wp = q.shape
        nH, nW = Hp // ws, Wp // ws

        # Reshape into windows: (B, C_out, nH, ws, nW, ws) → (B*nH*nW, num_heads, ws*ws, head_dim)
        def to_windows(t):
            t = t.view(B, C_out, nH, ws, nW, ws)
            t = t.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, C_out, ws, ws)
            t = t.reshape(B * nH * nW, self.num_heads, self.head_dim, ws * ws)
            t = t.permute(0, 1, 3, 2)  # (B*nH*nW, num_heads, ws*ws, head_dim)
            return t

        q_win = to_windows(q)
        k_win = to_windows(k)
        v_win = to_windows(v)

        # Flash Attention
        attn_out = F.scaled_dot_product_attention(q_win, k_win, v_win)
        # (B*nH*nW, num_heads, ws*ws, head_dim)

        # Reshape back to spatial
        attn_out = attn_out.permute(0, 1, 3, 2)  # (B*nH*nW, num_heads, head_dim, ws*ws)
        attn_out = attn_out.reshape(B, nH, nW, C_out, ws, ws)
        attn_out = attn_out.permute(0, 3, 1, 4, 2, 5)  # (B, C_out, nH, ws, nW, ws)
        attn_out = attn_out.reshape(B, C_out, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            attn_out = attn_out[:, :, :H, :W]

        out = self.out_proj(attn_out)

        # LayerNorm (channel-last)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C_out)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, C_out, H, W)

        # Residual connection (with projection if channels differ)
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        else:
            out = out + x

        return out


class GlobalAttentionLayer(nn.Module):
    """Global self-attention: all spatial positions attend to each other via Flash Attention."""

    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        N = H * W  # 6400

        # Reshape to tokens: (B, N, C)
        tokens = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # Project Q, K, V
        q = self.q_proj(tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, N, head_dim)

        # Flash Attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        # (B, num_heads, N, head_dim)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)  # (B, N, C_out)
        out = self.out_proj(attn_out)
        out = self.norm(out)

        # Reshape back to spatial
        C_out = out.shape[-1]
        out = out.reshape(B, H, W, C_out).permute(0, 3, 1, 2)  # (B, C_out, H, W)

        return out


class HierarchicalAttentionHead(nn.Module):
    """Hierarchical attention: 3x Local Attention + Global Attention + Conv Downsample.

    Replaces SpatialAttentionHead. Preserves spatial reasoning before compressing.
    """

    def __init__(self, in_channels=128, embed_dim=256):
        super().__init__()

        # 3 layers of Local Attention (8×8 window)
        self.local1 = LocalAttentionLayer(in_channels, 64, window_size=8, num_heads=4)
        self.local2 = LocalAttentionLayer(64, 32, window_size=8, num_heads=4)
        self.local3 = LocalAttentionLayer(32, 32, window_size=8, num_heads=4)

        # Global Self-Attention
        self.global_attn = GlobalAttentionLayer(32, 16, num_heads=2)

        # Conv Downsample: (16, 80, 80) → (32, 20, 20) → (64, 5, 5)
        self.downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=4),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            nn.SiLU(inplace=True),
        )

        # FC: 64*5*5=1600 → embed_dim
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, features):
        """
        Args:
            features: (B, 128, 80, 80)
        Returns:
            (B, 256) embedding vector
        """
        # Local attention (neighbor-level reasoning)
        x = self.local1(features)   # (B, 64, 80, 80)
        x = self.local2(x)          # (B, 32, 80, 80)
        x = self.local3(x)          # (B, 32, 80, 80)

        # Global attention (board-level strategy)
        x = self.global_attn(x)     # (B, 16, 80, 80)

        # Compress to embedding
        x = self.downsample(x)      # (B, 64, 5, 5)
        x = x.flatten(1)            # (B, 1600)
        embedding = self.fc(x)      # (B, 256)

        return embedding


class SACActorNetwork(nn.Module):
    """SAC Actor: image -> Gaussian distribution over (x, y) actions."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()
        self.backbone = YOLO11nBackbone()
        self.attention = HierarchicalAttentionHead(in_channels=128, embed_dim=embed_dim)

        self.mean_head = nn.Linear(embed_dim, action_dim)
        self.log_std_head = nn.Linear(embed_dim, action_dim)

    def get_embedding(self, state):
        """Extract embedding from state image.

        Args:
            state: (B, 3, 640, 640)
        Returns:
            (B, 256) embedding
        """
        features = self.backbone(state)
        embedding = self.attention(features)
        return embedding

    def forward(self, state):
        """Get mean and log_std of Gaussian policy.

        Args:
            state: (B, 3, 640, 640)
        Returns:
            mean: (B, 2), log_std: (B, 2)
        """
        embedding = self.get_embedding(state)
        mean = self.mean_head(embedding)
        log_std = self.log_std_head(embedding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """Sample action using reparameterization trick + tanh squashing.

        Args:
            state: (B, 3, 640, 640)
        Returns:
            action: (B, 2) in [-1, 1]
            log_prob: (B, 1)
            mean: (B, 2)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log-prob with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class SACCriticNetwork(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()

        input_dim = embed_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, embedding, action):
        """
        Args:
            embedding: (B, 256)
            action: (B, 2)
        Returns:
            q1: (B, 1), q2: (B, 1)
        """
        x = torch.cat([embedding, action], dim=-1)
        return self.q1(x), self.q2(x)


# ============================================================
# Stage 1 預訓練專用 (離散 grid state → SAC)
# ============================================================

class GridEncoder(nn.Module):
    """將離散 grid state (12, 10, 10) 轉換為 (128, 80, 80) feature map。

    使用 ConvTranspose2d 逐步放大: 10→20→40→80。
    階段 2 時會被 YOLO11nBackbone 取代（權重不保留）。
    """

    def __init__(self, in_channels=GRID_STATE_CHANNELS, out_channels=128):
        super().__init__()
        self.layers = nn.Sequential(
            # 10×10 → 20×20
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            # 20×20 → 40×40
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            # 40×40 → 80×80
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 12, 10, 10) one-hot grid state
        Returns:
            (B, 128, 80, 80) feature map
        """
        return self.layers(x)


class Stage1ActorNetwork(nn.Module):
    """Stage 1 Actor: grid state → Gaussian distribution over (x, y) actions.

    GridEncoder + HierarchicalAttentionHead + mean/log_std heads。
    使用 ScaledSigmoid 取代 tanh，輸出約 [-0.05, 1.05]。
    HierarchicalAttentionHead 和 head 的權重會轉移到 Stage 2。
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()
        self.encoder = GridEncoder()
        self.attention = HierarchicalAttentionHead(in_channels=128, embed_dim=embed_dim)
        self.mean_head = nn.Linear(embed_dim, action_dim)
        self.log_std_head = nn.Linear(embed_dim, action_dim)
        self.scaled_sigmoid = ScaledSigmoid(scale=1.1, shift=-0.05)

    def get_embedding(self, state):
        """
        Args:
            state: (B, 12, 10, 10)
        Returns:
            (B, 256) embedding
        """
        features = self.encoder(state)
        embedding = self.attention(features)
        return embedding

    def forward(self, state):
        """
        Args:
            state: (B, 12, 10, 10)
        Returns:
            mean: (B, 2), log_std: (B, 2)
        """
        embedding = self.get_embedding(state)
        mean = self.mean_head(embedding)
        log_std = self.log_std_head(embedding)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """使用 ScaledSigmoid 取代 tanh 壓縮 action。

        ScaledSigmoid 輸出約 [-0.05, 1.05]：
        - [0, 1] 為有效 grid 範圍
        - <0 或 >1 為超出範圍
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = self.scaled_sigmoid(x_t)

        # Log-prob with ScaledSigmoid correction
        # ScaledSigmoid(x) = scale * sigmoid(x) + shift
        # d/dx ScaledSigmoid(x) = scale * sigmoid(x) * (1 - sigmoid(x))
        sig = torch.sigmoid(x_t)
        log_det = torch.log(self.scaled_sigmoid.scale * sig * (1 - sig) + 1e-6)
        log_prob = normal.log_prob(x_t) - log_det
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class Stage1CriticNetwork(nn.Module):
    """Stage 1 Critic: grid state embedding + action → Q-value.

    有自己的 GridEncoder 和 HierarchicalAttentionHead（不跟 Actor 共享）。
    """

    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()
        self.encoder = GridEncoder()
        self.attention = HierarchicalAttentionHead(in_channels=128, embed_dim=embed_dim)

        input_dim = embed_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1)
        )

    def get_embedding(self, state):
        """
        Args:
            state: (B, 12, 10, 10)
        Returns:
            (B, 256) embedding
        """
        features = self.encoder(state)
        embedding = self.attention(features)
        return embedding

    def forward(self, embedding, action):
        """
        Args:
            embedding: (B, 256)
            action: (B, 2)
        Returns:
            q1: (B, 1), q2: (B, 1)
        """
        x = torch.cat([embedding, action], dim=-1)
        return self.q1(x), self.q2(x)


class SumTree:
    """Sum Tree 資料結構，用於 O(log n) 的優先級抽樣。

    葉節點存放 priority，內部節點存放子節點的 priority 總和。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx, change):
        """從葉節點往上更新 parent 的總和。"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """根據累積 sum 找到對應的葉節點。"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def update(self, data_idx, priority):
        """更新某個 data index 的 priority。"""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority):
        """新增一筆 priority，回傳對應的 data index。"""
        data_idx = self.write_ptr
        self.update(data_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def get(self, s):
        """根據累積 sum 值 s 找到 (data_idx, priority)。"""
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return data_idx, self.tree[tree_idx]

    def max_priority(self):
        """目前所有葉節點的最大 priority。"""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.size
        if self.size == 0:
            return 1.0
        return max(self.tree[leaf_start:leaf_end])


class Stage1ReplayBuffer:
    """Stage 1 用的 per-class circular buffer。

    每個 reward class 有獨立的 circular buffer，互不干擾。
    Sample 時從每個 class 抽相同數量，保證訓練 data 平衡。
    """

    def __init__(self, max_per_class=BUFFER_CAPACITY):
        self.max_per_class = max_per_class
        self.class_buffers = {}   # reward → list of entries (circular)
        self.class_ptrs = {}      # reward → write pointer
        self.class_sizes = {}     # reward → current size

    def store(self, state, action, next_state, reward, done):
        """存 transition 到對應 reward class 的 buffer。"""
        entry = {
            'state': state.cpu().clone(),
            'action': action.copy() if isinstance(action, np.ndarray) else np.array(action),
            'next_state': next_state.cpu().clone(),
            'reward': float(reward),
            'done': bool(done)
        }

        r = float(reward)
        if r not in self.class_buffers:
            self.class_buffers[r] = [None] * self.max_per_class
            self.class_ptrs[r] = 0
            self.class_sizes[r] = 0

        buf = self.class_buffers[r]
        ptr = self.class_ptrs[r]
        buf[ptr] = entry
        self.class_ptrs[r] = (ptr + 1) % self.max_per_class
        self.class_sizes[r] = min(self.class_sizes[r] + 1, self.max_per_class)

    def sample(self, batch_size):
        """每個 reward class 盡量平均抽，數量不足的 class 抽完後剩餘配額給其他 class。

        回傳 (states, actions, next_states, rewards, dones)。
        """
        active_classes = {r: s for r, s in self.class_sizes.items() if s > 0}
        num_classes = len(active_classes)

        if num_classes == 0:
            raise RuntimeError("Replay buffer is empty")

        # 第一輪：平均分配，數量不夠的 class 只拿實際數量
        base_count = batch_size // num_classes
        remainder = batch_size % num_classes
        class_counts = {}
        overflow = 0
        overflow_classes = []
        for i, r in enumerate(sorted(active_classes.keys())):
            want = base_count + (1 if i < remainder else 0)
            have = active_classes[r]
            if have < want:
                class_counts[r] = have
                overflow += want - have
            else:
                class_counts[r] = want
                overflow_classes.append(r)

        # 第二輪：把剩餘配額分給還有餘裕的 class
        while overflow > 0 and overflow_classes:
            extra_per = overflow // len(overflow_classes)
            extra_rem = overflow % len(overflow_classes)
            if extra_per == 0 and extra_rem == 0:
                break
            new_overflow = 0
            new_overflow_classes = []
            for i, r in enumerate(overflow_classes):
                give = extra_per + (1 if i < extra_rem else 0)
                room = active_classes[r] - class_counts[r]
                actual = min(give, room)
                class_counts[r] += actual
                new_overflow += give - actual
                if class_counts[r] < active_classes[r]:
                    new_overflow_classes.append(r)
            overflow = new_overflow
            overflow_classes = new_overflow_classes

        # 從每個 class 隨機抽
        all_entries = []
        for r, count in class_counts.items():
            buf = self.class_buffers[r]
            size = self.class_sizes[r]
            indices = np.random.randint(0, size, size=count)
            for idx in indices:
                all_entries.append(buf[idx])

        # 組裝 batch
        states = torch.stack([e['state'] for e in all_entries]).to(device)
        actions = torch.tensor(
            np.array([e['action'] for e in all_entries]),
            dtype=torch.float32
        ).to(device)
        next_states = torch.stack([e['next_state'] for e in all_entries]).to(device)
        rewards = torch.tensor(
            [e['reward'] for e in all_entries],
            dtype=torch.float32
        ).unsqueeze(1).to(device)
        dones = torch.tensor(
            [float(e['done']) for e in all_entries],
            dtype=torch.float32
        ).unsqueeze(1).to(device)

        return states, actions, next_states, rewards, dones

    def size(self):
        return sum(self.class_sizes.values())

    def get_class_distribution(self):
        """取得每個 class 的數量（用於 logging）。"""
        return {r: s for r, s in self.class_sizes.items() if s > 0}

    def get_all_entries(self):
        """取得所有 entries（用於 persistent save）。"""
        entries = []
        for r, size in self.class_sizes.items():
            buf = self.class_buffers[r]
            for i in range(size):
                entries.append(buf[i])
        return entries


class Stage1SACAgent:
    """Stage 1 SAC Agent — 用離散 grid state 預訓練。

    訓練完成後可以匯出 HierarchicalAttentionHead + SAC head 的權重，
    供 Stage 2 的 SACAgent 載入。
    """

    def __init__(self):
        self.action_dim = 2

        # Actor
        self.actor = Stage1ActorNetwork(embed_dim=256, action_dim=self.action_dim).to(device)

        # Twin Critics + target
        self.critic = Stage1CriticNetwork(embed_dim=256, action_dim=self.action_dim).to(device)
        self.critic_target = Stage1CriticNetwork(embed_dim=256, action_dim=self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Alpha based on valid click rate (sliding window of 50 episodes)
        self.alpha = ALPHA_MAX  # 初始值：最大探索
        self.recent_valid_rates = deque(maxlen=50)

        # Replay buffer (in-memory, grid state 很小)
        self.replay_buffer = Stage1ReplayBuffer(max_per_class=BUFFER_CAPACITY)
        self.total_it = 0
        self.episode_count = 0

        # 嘗試載入之前的 checkpoint
        self.try_load_model()

        # 程式結束時存檔
        atexit.register(self.save_persistent)

    def select_action(self, state, add_noise=True):
        """根據 grid state 選擇動作。

        Args:
            state: (12, 10, 10) tensor
            add_noise: True=探索模式, False=確定性模式
        Returns:
            action: numpy array (2,) ≈ [-0.05, 1.05] (ScaledSigmoid output)
        """
        state_batch = state.unsqueeze(0).to(device)

        self.actor.eval()
        with torch.no_grad():
            if add_noise:
                action, _, _ = self.actor.sample(state_batch)
            else:
                mean, _ = self.actor(state_batch)
                action = self.actor.scaled_sigmoid(mean)
        self.actor.train()

        return action.cpu().numpy().flatten()

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state, action, next_state, reward, done)

    def train_step(self):
        """執行一步 SAC 訓練。"""
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_it += 1
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        alpha = self.alpha

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            next_embed = self.critic_target.get_embedding(next_state)
            target_q1, target_q2 = self.critic_target(next_embed, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = reward + (1 - done) * GAMMA * target_q

        current_embed = self.critic.get_embedding(state)
        current_q1, current_q2 = self.critic(current_embed, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_action, log_prob, _ = self.actor.sample(state)
        # Critic 用自己的 encoder 算 embedding (不跟 actor 共享)
        critic_embed = self.critic.get_embedding(state).detach()
        q1, q2 = self.critic(critic_embed, new_action)
        min_q = torch.min(q1, q2)
        actor_loss = (alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- No alpha update: alpha is set by update_alpha() based on valid_rate ---

        # --- Soft update target ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha,
        }

    def update_alpha(self, valid_rate):
        """根據有效點擊率更新 alpha。

        Args:
            valid_rate: float in [0, 1]，本 episode 的有效點擊率
        """
        self.recent_valid_rates.append(valid_rate)
        avg_valid_rate = np.mean(self.recent_valid_rates)
        self.alpha = ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * avg_valid_rate

    def on_episode_end(self):
        """每個 episode 結束時呼叫。"""
        self.episode_count += 1
        # 每 episode 存一次 model 權重（而不是每個 train step）
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[Stage1] Periodic save at episode {self.episode_count}")
            self.save_persistent()

    def _save_model(self):
        """存 model 權重 (每步呼叫)。"""
        STAGE1_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), STAGE1_MODEL_PATH / 'stage1_actor.pth')
        torch.save(self.critic.state_dict(), STAGE1_MODEL_PATH / 'stage1_critic.pth')
        torch.save(self.critic_target.state_dict(), STAGE1_MODEL_PATH / 'stage1_critic_target.pth')
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
            'recent_valid_rates': list(self.recent_valid_rates),
        }, STAGE1_MODEL_PATH / 'stage1_optimizer_state.pth')

    def try_load_model(self):
        """載入之前的 Stage 1 checkpoint。"""
        actor_path = STAGE1_MODEL_PATH / 'stage1_actor.pth'
        if actor_path.exists():
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=device))
                print("[Stage1] Loaded Actor")
            except Exception as e:
                print(f"[Stage1] Failed to load Actor: {e}")

        critic_path = STAGE1_MODEL_PATH / 'stage1_critic.pth'
        if critic_path.exists():
            try:
                self.critic.load_state_dict(torch.load(critic_path, map_location=device))
                print("[Stage1] Loaded Critic")
            except Exception as e:
                print(f"[Stage1] Failed to load Critic: {e}")

        critic_target_path = STAGE1_MODEL_PATH / 'stage1_critic_target.pth'
        if critic_target_path.exists():
            try:
                self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))
                print("[Stage1] Loaded Critic Target")
            except Exception as e:
                print(f"[Stage1] Failed to load Critic Target: {e}")

        opt_path = STAGE1_MODEL_PATH / 'stage1_optimizer_state.pth'
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.actor_optimizer.load_state_dict(state['actor_optimizer'])
                self.critic_optimizer.load_state_dict(state['critic_optimizer'])
                self.total_it = state['total_it']
                self.episode_count = state.get('episode_count', 0)
                # 載入 valid_rate 歷史，恢復 alpha
                saved_rates = state.get('recent_valid_rates', [])
                if saved_rates:
                    self.recent_valid_rates = deque(saved_rates, maxlen=50)
                    avg_valid_rate = np.mean(self.recent_valid_rates)
                    self.alpha = ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * avg_valid_rate
                print(f"[Stage1] Loaded optimizer state: total_it={self.total_it}, episode={self.episode_count}, alpha={self.alpha:.4f}")
            except Exception as e:
                print(f"[Stage1] Failed to load optimizer state: {e}")

        # 載入 persistent replay buffer
        training_state_path = STAGE1_MODEL_PATH / 'stage1_training_state.pth'
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                persistent_entries = state.get('persistent_entries', [])
                # 限制不超過 buffer 容量
                persistent_entries = persistent_entries[:self.replay_buffer.max_size]
                for entry in persistent_entries:
                    # 用 store 來正確更新 SumTree 的 priority
                    self.replay_buffer.store(
                        entry['state'], entry['action'],
                        entry['next_state'], entry['reward'], entry['done']
                    )
                print(f"[Stage1] Loaded {len(persistent_entries)} persistent buffer entries")
            except Exception as e:
                print(f"[Stage1] Failed to load persistent buffer: {e}")

    def save_persistent(self):
        """將 replay buffer 的 stratified random subset 存到磁碟。"""
        buf = self.replay_buffer
        if buf.size_count == 0:
            print("[Stage1] Buffer empty, skipping persistent save")
            return

        # 按 reward 分組
        reward_groups = defaultdict(list)
        for i in range(buf.size_count):
            if buf.buffer[i] is not None:
                reward_groups[buf.buffer[i]['reward']].append(i)

        # Stratified sampling
        total = buf.size_count
        target = min(SAVE_CAPACITY, total)
        selected_indices = []

        remaining = target
        groups = sorted(reward_groups.items(), key=lambda x: len(x[1]))
        for i, (reward, indices) in enumerate(groups):
            if i == len(groups) - 1:
                count = remaining
            else:
                count = round(len(indices) / total * target)
            count = min(count, len(indices), remaining)
            selected = random.sample(indices, count)
            selected_indices.extend(selected)
            remaining -= count
            if remaining <= 0:
                break

        # 收集選中的 entries
        persistent_entries = [buf.buffer[i] for i in selected_indices]

        # 存到磁碟
        STAGE1_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
            'recent_valid_rates': list(self.recent_valid_rates),
            'persistent_entries': persistent_entries,
        }, STAGE1_MODEL_PATH / 'stage1_training_state.pth')

        # Log
        saved_rewards = defaultdict(int)
        for entry in persistent_entries:
            saved_rewards[entry['reward']] += 1
        print(f"[Stage1] Persistent save: {len(persistent_entries)} entries")
        print(f"[Stage1] Reward distribution: {dict(saved_rewards)}")

    def save_stage1_weights(self):
        """匯出 Stage 1 的 transferable 權重（SpatialAttention + SAC heads）。

        GridEncoder 權重不匯出 — Stage 2 用 YOLO 取代。
        """
        STAGE1_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        weights = {
            # Actor
            'actor_attention': self.actor.attention.state_dict(),
            'actor_mean_head': self.actor.mean_head.state_dict(),
            'actor_log_std_head': self.actor.log_std_head.state_dict(),
            # Critic
            'critic_attention': self.critic.attention.state_dict(),
            'critic_q1': self.critic.q1.state_dict(),
            'critic_q2': self.critic.q2.state_dict(),
        }
        path = STAGE1_MODEL_PATH / 'stage1_weights.pth'
        torch.save(weights, path)
        print(f"[Stage1] Transfer weights saved to {path}")
        return path


# ============================================================
# Simple MLP 實驗 — 驗證 RL pipeline 是否正常
# ============================================================

SIMPLE_MODEL_PATH = Path('./models/stage1_simple')


class SimpleActorNetwork(nn.Module):
    """極簡 Actor: grid state flatten → MLP → 100 action probabilities.

    支援 action masking：已翻開的格子機率設為 0。
    """

    def __init__(self, grid_channels=GRID_STATE_CHANNELS, grid_h=10, grid_w=10, num_actions=100):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        input_dim = grid_channels * grid_h * grid_w  # 12*10*10 = 1200
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, state):
        """
        Args:
            state: (B, 12, 10, 10)
                   channel 0 = 未翻開 → 用來做 action masking
        Returns:
            probs: (B, 100), log_probs: (B, 100)
        """
        logits = self.net(state)

        # Action masking：從 state channel 0 取得未翻開 mask
        # channel 0 = 1.0 表示未翻開（可點擊），0.0 表示已翻開（不可點擊）
        # 用 -1e8 而非 -inf，避免 softmax 後 0 × log(0) = NaN 導致梯度爆炸
        mask = state[:, 0].reshape(state.size(0), -1).bool()  # (B, 100)
        logits = logits.masked_fill(~mask, -1e8)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class SimpleCriticNetwork(nn.Module):
    """極簡 Critic: grid state flatten → MLP → 100 Q-values (twin)."""

    def __init__(self, grid_channels=GRID_STATE_CHANNELS, grid_h=10, grid_w=10, num_actions=100):
        super().__init__()
        input_dim = grid_channels * grid_h * grid_w  # 1200

        self.q1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_actions),
        )
        self.q2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, state):
        """
        Args:
            state: (B, 12, 10, 10)
        Returns:
            q1: (B, 100), q2: (B, 100)
        """
        return self.q1(state), self.q2(state)


class SimpleDiscreteAgent:
    """極簡 SAC-Discrete agent — 純 MLP，無 attention，無 GridEncoder。

    用來驗證 RL pipeline（reward、buffer、SAC 公式）是否正確。
    """

    def __init__(self):
        self.num_actions = 100

        self.actor = SimpleActorNetwork().to(device)
        self.critic = SimpleCriticNetwork().to(device)
        self.critic_target = SimpleCriticNetwork().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Auto-alpha with clamp (correct target for discrete)
        self.log_alpha = torch.tensor(
            np.log(INIT_ALPHA), dtype=torch.float32,
            requires_grad=True, device=device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA_DISCRETE)
        self.target_entropy = DISCRETE_TARGET_ENTROPY

        # Replay buffer
        self.replay_buffer = Stage1ReplayBuffer(max_per_class=BUFFER_CAPACITY)
        self.total_it = 0
        self.episode_count = 0

        # Train I/O log
        SIMPLE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self._io_log = open(SIMPLE_MODEL_PATH / 'train_io_log.txt', 'a', encoding='utf-8')
        self._io_log.write(f"\n{'='*60}\n")
        self._io_log.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self._io_log.write(f"{'='*60}\n")
        self._io_log.flush()

        self.try_load_model()
        atexit.register(self.save_persistent)
        atexit.register(self._close_io_log)

    def select_action(self, state, add_noise=True):
        """選擇一個 grid cell（不做 hard mask，model 自行學會避開已翻開格子）。

        Returns:
            action: int [0, 99]
        """
        state_batch = state.unsqueeze(0).to(device)

        self.actor.eval()
        with torch.no_grad():
            probs, _ = self.actor(state_batch)
            if add_noise:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            else:
                action = probs.argmax(dim=-1).item()
        self.actor.train()

        return action

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state, action, next_state, reward, done)

    def train_step(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_it += 1
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        alpha = self.log_alpha.exp().detach()

        # action → long index
        if isinstance(action, torch.Tensor) and action.dim() > 1:
            action_idx = action.squeeze(-1).long()
        else:
            action_idx = action.long()

        # --- Critic update ---
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_state)
            next_q1, next_q2 = self.critic_target(next_state)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            target_q = reward + (1 - done) * GAMMA * next_v

        q1_all, q2_all = self.critic(state)
        q1 = q1_all.gather(1, action_idx.unsqueeze(-1))
        q2 = q2_all.gather(1, action_idx.unsqueeze(-1))

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        probs, log_probs = self.actor(state)
        with torch.no_grad():
            q1_all, q2_all = self.critic(state)
            min_q = torch.min(q1_all, q2_all)

        actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update (auto-alpha with clamp) ---
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # Clamp alpha to [ALPHA_MIN, ALPHA_MAX]
        with torch.no_grad():
            self.log_alpha.data.clamp_(np.log(ALPHA_MIN), np.log(ALPHA_MAX))

        # --- Soft update target ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        # --- I/O log ---
        with torch.no_grad():
            new_alpha = self.log_alpha.exp().item()

            # Batch reward distribution
            reward_counts = defaultdict(int)
            for r in reward.squeeze(-1).tolist():
                reward_counts[r] += 1

            # Model Input: 第一筆 state 的 grid 狀態
            s0 = state[0]  # (12, 10, 10)
            unrevealed = s0[0].sum().int().item()
            flagged = s0[1].sum().int().item()
            revealed = 100 - unrevealed - flagged
            num_counts = {}
            for ch in range(2, 11):
                cnt = s0[ch].sum().int().item()
                if cnt > 0:
                    num_counts[ch - 2] = cnt

            # Model Output: action probs top-5
            p0 = probs[0]  # (100,)
            top5_vals, top5_idx = p0.topk(5)
            top5_info = [(idx.item() // 10, idx.item() % 10, f"{val.item():.4f}")
                         for val, idx in zip(top5_vals, top5_idx)]

            # Batch actions 分布 (top-3 most frequent)
            action_list = action_idx.tolist()
            action_freq = defaultdict(int)
            for a in action_list:
                action_freq[a] += 1
            top3_actions = sorted(action_freq.items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"({a//10},{a%10})x{c}" for a, c in top3_actions)

            # Q-values 統計
            q_selected = torch.min(q1, q2).squeeze(-1)  # (B,)
            policy_mean_q = (p0 * min_q[0]).sum().item()

        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  State:  unrevealed={unrevealed} | revealed={revealed} | flagged={flagged}"
            f" | numbers={num_counts}\n"
            f"  Batch:  rewards={dict(reward_counts)} | top_actions=[{top3_str}]\n"
            f"  Probs:  top5={top5_info}\n"
            f"  Q-val:  selected: mean={q_selected.mean().item():.4f}"
            f" min={q_selected.min().item():.4f} max={q_selected.max().item():.4f}"
            f" | policy_mean={policy_mean_q:.4f}"
            f" | all: min={min_q[0].min().item():.4f} max={min_q[0].max().item():.4f}\n"
            f"  Loss:   critic={critic_loss.item():.4f} | actor={actor_loss.item():.4f}"
            f" | alpha={alpha.item():.4f}->{new_alpha:.4f} | entropy={entropy.item():.4f}\n"
            f"---\n"
        )
        self._io_log.flush()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "entropy": entropy.item(),
        }

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def on_episode_end(self):
        self.episode_count += 1
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[Simple] Periodic save at episode {self.episode_count}")
            self.save_persistent()

    def _save_model(self):
        SIMPLE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), SIMPLE_MODEL_PATH / 'actor.pth')
        torch.save(self.critic.state_dict(), SIMPLE_MODEL_PATH / 'critic.pth')
        torch.save(self.critic_target.state_dict(), SIMPLE_MODEL_PATH / 'critic_target.pth')
        torch.save(self.log_alpha, SIMPLE_MODEL_PATH / 'log_alpha.pth')
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
        }, SIMPLE_MODEL_PATH / 'optimizer_state.pth')

    def save_persistent(self):
        """儲存 replay buffer 到磁碟（每個 class 各存一些）。"""
        buf = self.replay_buffer
        total = buf.size()
        if total == 0:
            return

        all_entries = buf.get_all_entries()
        # 限制總數
        if len(all_entries) > SAVE_CAPACITY:
            # 每個 class 平均分配
            dist = buf.get_class_distribution()
            per_class = max(1, SAVE_CAPACITY // len(dist))
            selected = []
            for r, size in dist.items():
                class_entries = [e for e in all_entries if e['reward'] == r]
                if len(class_entries) > per_class:
                    class_entries = random.sample(class_entries, per_class)
                selected.extend(class_entries)
            all_entries = selected[:SAVE_CAPACITY]

        SIMPLE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save({
            'persistent_entries': all_entries,
            'total_it': self.total_it,
            'episode_count': self.episode_count,
        }, SIMPLE_MODEL_PATH / 'training_state.pth')

        saved_rewards = defaultdict(int)
        for entry in all_entries:
            saved_rewards[entry['reward']] += 1
        print(f"--- save info ---------------")
        print(f"[Simple] Persistent save: {len(all_entries)} entries")
        print(f"[Simple] Reward distribution: {dict(saved_rewards)}")
        print(f"--- save end ---------------")

    def try_load_model(self):
        actor_path = SIMPLE_MODEL_PATH / 'actor.pth'
        if actor_path.exists():
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=device))
                print("[Simple] Loaded Actor")
            except Exception as e:
                print(f"[Simple] Failed to load Actor: {e}")

        critic_path = SIMPLE_MODEL_PATH / 'critic.pth'
        if critic_path.exists():
            try:
                self.critic.load_state_dict(torch.load(critic_path, map_location=device))
                print("[Simple] Loaded Critic")
            except Exception as e:
                print(f"[Simple] Failed to load Critic: {e}")

        critic_target_path = SIMPLE_MODEL_PATH / 'critic_target.pth'
        if critic_target_path.exists():
            try:
                self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))
                print("[Simple] Loaded Critic Target")
            except Exception as e:
                print(f"[Simple] Failed to load Critic Target: {e}")

        alpha_path = SIMPLE_MODEL_PATH / 'log_alpha.pth'
        if alpha_path.exists():
            try:
                self.log_alpha = torch.load(alpha_path, map_location=device)
                self.log_alpha.requires_grad_(True)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA_DISCRETE)
                print(f"[Simple] Loaded log_alpha: alpha={self.log_alpha.exp().item():.4f}")
            except Exception as e:
                print(f"[Simple] Failed to load log_alpha: {e}")

        opt_path = SIMPLE_MODEL_PATH / 'optimizer_state.pth'
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.actor_optimizer.load_state_dict(state['actor_optimizer'])
                self.critic_optimizer.load_state_dict(state['critic_optimizer'])
                if 'alpha_optimizer' in state:
                    self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
                self.total_it = state['total_it']
                self.episode_count = state.get('episode_count', 0)
                print(f"[Simple] Loaded optimizer: total_it={self.total_it}, episode={self.episode_count}")
            except Exception as e:
                print(f"[Simple] Failed to load optimizer state: {e}")

        training_state_path = SIMPLE_MODEL_PATH / 'training_state.pth'
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                persistent_entries = state.get('persistent_entries', [])
                loaded = min(len(persistent_entries), BUFFER_CAPACITY)
                for i in range(loaded):
                    entry = persistent_entries[i]
                    self.replay_buffer.store(
                        entry['state'], entry['action'],
                        entry['next_state'], entry['reward'], entry['done']
                    )
                if loaded > 0:
                    print(f"[Simple] Loaded {loaded} replay buffer entries")
            except Exception as e:
                print(f"[Simple] Failed to load replay buffer: {e}")


# ============================================================
# Transformer 實驗 — 用 self-attention 學習 Minesweeper 空間推理
# ============================================================

TRANSFORMER_MODEL_PATH = Path('./models/stage1_transformer')


class TransformerActorNetwork(nn.Module):
    """Transformer Actor: grid state → per-token logits → action distribution.

    每個 cell 是一個 token (12-d one-hot)，經過 Transformer 後
    每個 token 輸出一個 logit 表示「該不該點這格」。
    """

    def __init__(self, grid_channels=GRID_STATE_CHANNELS, grid_h=10, grid_w=10,
                 d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_tokens = grid_h * grid_w  # 100

        # Token embedding: 12-d one-hot → d_model
        self.token_embed = nn.Linear(grid_channels, d_model)

        # Learned 2D positional encoding
        self.row_embed = nn.Embedding(grid_h, d_model // 2)
        self.col_embed = nn.Embedding(grid_w, d_model // 2)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-token output head → 1 logit per token
        self.output_head = nn.Linear(d_model, 1)

        # 預先計算 position indices
        rows = torch.arange(grid_h).unsqueeze(1).expand(grid_h, grid_w).reshape(-1)
        cols = torch.arange(grid_w).unsqueeze(0).expand(grid_h, grid_w).reshape(-1)
        self.register_buffer('row_indices', rows)
        self.register_buffer('col_indices', cols)

    def _embed(self, state):
        """State → token embeddings with positional encoding.

        Args:
            state: (B, 12, 10, 10)
        Returns:
            (B, 100, d_model)
        """
        B = state.size(0)
        # (B, 12, 10, 10) → (B, 100, 12)
        tokens = state.permute(0, 2, 3, 1).reshape(B, self.num_tokens, -1)

        # Token embedding
        x = self.token_embed(tokens)  # (B, 100, d_model)

        # Positional encoding: row_embed(32) cat col_embed(32) → (100, 64)
        pos = torch.cat([
            self.row_embed(self.row_indices),
            self.col_embed(self.col_indices),
        ], dim=-1)  # (100, d_model)

        x = x + pos.unsqueeze(0)  # broadcast over batch
        return x

    def forward(self, state):
        """
        Args:
            state: (B, 12, 10, 10)
        Returns:
            probs: (B, 100), log_probs: (B, 100)
        """
        x0 = self._embed(state)                  # (B, 100, d_model) — 保存原始 embedding
        x = x0
        for layer in self.transformer.layers:
            x = layer(x) + x0                     # 每層 + 原始 embedding (input residual)
        logits = self.output_head(x).squeeze(-1)  # (B, 100)

        # Action masking: channel 0 = 未翻開 (可點擊)
        mask = state[:, 0].reshape(state.size(0), -1).bool()  # (B, 100)
        logits = logits.masked_fill(~mask, -1e8)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class TransformerCriticNetwork(nn.Module):
    """Transformer Critic: grid state → per-token Q-values (twin).

    Q1 和 Q2 各自有獨立的 embedding + Transformer encoder + output head，
    確保 twin-Q 估計足夠獨立以有效降低 overestimation bias。
    Critic 用 2 層（比 Actor 的 4 層淺），減少梯度累積深度。
    """

    def __init__(self, grid_channels=GRID_STATE_CHANNELS, grid_h=10, grid_w=10,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_tokens = grid_h * grid_w

        def _make_branch():
            token_embed = nn.Linear(grid_channels, d_model)
            row_embed = nn.Embedding(grid_h, d_model // 2)
            col_embed = nn.Embedding(grid_w, d_model // 2)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            head = nn.Linear(d_model, 1)
            return nn.ModuleDict({
                'token_embed': token_embed,
                'row_embed': row_embed,
                'col_embed': col_embed,
                'transformer': transformer,
                'head': head,
            })

        self.q1 = _make_branch()
        self.q2 = _make_branch()

        # 預先計算 position indices
        rows = torch.arange(grid_h).unsqueeze(1).expand(grid_h, grid_w).reshape(-1)
        cols = torch.arange(grid_w).unsqueeze(0).expand(grid_h, grid_w).reshape(-1)
        self.register_buffer('row_indices', rows)
        self.register_buffer('col_indices', cols)

    def _forward_branch(self, branch, state):
        B = state.size(0)
        tokens = state.permute(0, 2, 3, 1).reshape(B, self.num_tokens, -1)
        x0 = branch['token_embed'](tokens)
        pos = torch.cat([
            branch['row_embed'](self.row_indices),
            branch['col_embed'](self.col_indices),
        ], dim=-1)
        x0 = x0 + pos.unsqueeze(0)
        x = x0
        for layer in branch['transformer'].layers:
            x = layer(x) + x0  # input residual
        return branch['head'](x).squeeze(-1)  # (B, 100)

    def forward(self, state):
        """
        Args:
            state: (B, 12, 10, 10)
        Returns:
            q1: (B, 100), q2: (B, 100)
        """
        return self._forward_branch(self.q1, state), self._forward_branch(self.q2, state)


class TransformerDiscreteAgent:
    """Transformer SAC-Discrete agent — 用 self-attention 學習空間推理。

    架構: per-cell token + 2D pos encoding → Transformer → per-token logit → action
    SAC 訓練邏輯跟 SimpleDiscreteAgent 完全相同。
    """

    def __init__(self):
        self.num_actions = 100

        self.actor = TransformerActorNetwork().to(device)
        self.critic = TransformerCriticNetwork().to(device)
        self.critic_target = TransformerCriticNetwork().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Auto-alpha with clamp
        self.log_alpha = torch.tensor(
            np.log(INIT_ALPHA), dtype=torch.float32,
            requires_grad=True, device=device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA_DISCRETE)
        self.target_entropy = DISCRETE_TARGET_ENTROPY

        # Replay buffer
        self.replay_buffer = Stage1ReplayBuffer(max_per_class=BUFFER_CAPACITY)
        self.total_it = 0
        self.episode_count = 0

        # Epsilon-greedy exploration
        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay_episodes = 5000

        # Train I/O log
        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self._io_log = open(TRANSFORMER_MODEL_PATH / 'train_io_log.txt', 'a', encoding='utf-8')
        self._io_log.write(f"\n{'='*60}\n")
        self._io_log.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self._io_log.write(f"{'='*60}\n")
        self._io_log.flush()

        self.try_load_model()
        atexit.register(self.save_persistent)
        atexit.register(self._close_io_log)

    def select_action(self, state, add_noise=True):
        """選擇一個 grid cell。

        Training 模式下有 epsilon 機率隨機選未翻開格子。

        Returns:
            action: int [0, 99]
        """
        # Epsilon-greedy: 隨機選未翻開格子
        if add_noise and random.random() < self.epsilon:
            mask = state[0].reshape(-1).numpy()  # channel 0 = 未翻開
            unrevealed = np.where(mask > 0.5)[0]
            if len(unrevealed) > 0:
                return int(np.random.choice(unrevealed))

        state_batch = state.unsqueeze(0).to(device)

        self.actor.eval()
        with torch.no_grad():
            probs, _ = self.actor(state_batch)
            if add_noise:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            else:
                action = probs.argmax(dim=-1).item()
        self.actor.train()

        return action

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state, action, next_state, reward, done)

    def train_step(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_it += 1
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        alpha = self.log_alpha.exp().detach()

        # action → long index
        if isinstance(action, torch.Tensor) and action.dim() > 1:
            action_idx = action.squeeze(-1).long()
        else:
            action_idx = action.long()

        # --- Critic update ---
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_state)
            next_q1, next_q2 = self.critic_target(next_state)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            target_q = reward + (1 - done) * GAMMA * next_v

        q1_all, q2_all = self.critic(state)
        q1 = q1_all.gather(1, action_idx.unsqueeze(-1))
        q2 = q2_all.gather(1, action_idx.unsqueeze(-1))

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        probs, log_probs = self.actor(state)
        with torch.no_grad():
            q1_all, q2_all = self.critic(state)
            min_q = torch.min(q1_all, q2_all)

        actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update (auto-alpha with clamp) ---
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.data.clamp_(np.log(ALPHA_MIN), np.log(ALPHA_MAX))

        # --- Soft update target ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        # --- I/O log ---
        with torch.no_grad():
            new_alpha = self.log_alpha.exp().item()

            # Batch reward distribution
            reward_counts = defaultdict(int)
            for r in reward.squeeze(-1).tolist():
                reward_counts[r] += 1

            # Model Input: 第一筆 state 的 grid 狀態
            s0 = state[0]  # (12, 10, 10)
            unrevealed = s0[0].sum().int().item()
            flagged = s0[1].sum().int().item()
            revealed = 100 - unrevealed - flagged
            num_counts = {}
            for ch in range(2, 11):
                cnt = s0[ch].sum().int().item()
                if cnt > 0:
                    num_counts[ch - 2] = cnt

            # Model Output: action probs top-5
            p0 = probs[0]  # (100,)
            top5_vals, top5_idx = p0.topk(5)
            top5_info = [(idx.item() // 10, idx.item() % 10, f"{val.item():.4f}")
                         for val, idx in zip(top5_vals, top5_idx)]

            # Batch actions 分布 (top-3 most frequent)
            action_list = action_idx.tolist()
            action_freq = defaultdict(int)
            for a in action_list:
                action_freq[a] += 1
            top3_actions = sorted(action_freq.items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"({a//10},{a%10})x{c}" for a, c in top3_actions)

            # Q-values 統計
            q_selected = torch.min(q1, q2).squeeze(-1)  # (B,)
            policy_mean_q = (p0 * min_q[0]).sum().item()

        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  State:  unrevealed={unrevealed} | revealed={revealed} | flagged={flagged}"
            f" | numbers={num_counts}\n"
            f"  Batch:  rewards={dict(reward_counts)} | top_actions=[{top3_str}]\n"
            f"  Probs:  top5={top5_info}\n"
            f"  Q-val:  selected: mean={q_selected.mean().item():.4f}"
            f" min={q_selected.min().item():.4f} max={q_selected.max().item():.4f}"
            f" | policy_mean={policy_mean_q:.4f}"
            f" | all: min={min_q[0].min().item():.4f} max={min_q[0].max().item():.4f}\n"
            f"  Loss:   critic={critic_loss.item():.4f} | actor={actor_loss.item():.4f}"
            f" | alpha={alpha.item():.4f}->{new_alpha:.4f} | entropy={entropy.item():.4f}\n"
            f"---\n"
        )
        self._io_log.flush()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "entropy": entropy.item(),
        }

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def on_episode_end(self):
        self.episode_count += 1
        # Epsilon decay
        decay_progress = min(self.episode_count / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_min + (0.3 - self.epsilon_min) * (1.0 - decay_progress)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[Transformer] Periodic save at episode {self.episode_count}"
                  f" | epsilon={self.epsilon:.4f}")
            self.save_persistent()

    def _save_model(self):
        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), TRANSFORMER_MODEL_PATH / 'actor.pth')
        torch.save(self.critic.state_dict(), TRANSFORMER_MODEL_PATH / 'critic.pth')
        torch.save(self.critic_target.state_dict(), TRANSFORMER_MODEL_PATH / 'critic_target.pth')
        torch.save(self.log_alpha, TRANSFORMER_MODEL_PATH / 'log_alpha.pth')
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
        }, TRANSFORMER_MODEL_PATH / 'optimizer_state.pth')

    def save_persistent(self):
        """儲存 replay buffer 到磁碟。"""
        buf = self.replay_buffer
        total = buf.size()
        if total == 0:
            return

        all_entries = buf.get_all_entries()
        if len(all_entries) > SAVE_CAPACITY:
            dist = buf.get_class_distribution()
            per_class = max(1, SAVE_CAPACITY // len(dist))
            selected = []
            for r, size in dist.items():
                class_entries = [e for e in all_entries if e['reward'] == r]
                if len(class_entries) > per_class:
                    class_entries = random.sample(class_entries, per_class)
                selected.extend(class_entries)
            all_entries = selected[:SAVE_CAPACITY]

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save({
            'persistent_entries': all_entries,
            'total_it': self.total_it,
            'episode_count': self.episode_count,
        }, TRANSFORMER_MODEL_PATH / 'training_state.pth')

        saved_rewards = defaultdict(int)
        for entry in all_entries:
            saved_rewards[entry['reward']] += 1
        print(f"--- save info ---------------")
        print(f"[Transformer] Persistent save: {len(all_entries)} entries")
        print(f"[Transformer] Reward distribution: {dict(saved_rewards)}")
        print(f"--- save end ---------------")

    def try_load_model(self):
        actor_path = TRANSFORMER_MODEL_PATH / 'actor.pth'
        if actor_path.exists():
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=device))
                print("[Transformer] Loaded Actor")
            except Exception as e:
                print(f"[Transformer] Failed to load Actor: {e}")

        critic_path = TRANSFORMER_MODEL_PATH / 'critic.pth'
        if critic_path.exists():
            try:
                self.critic.load_state_dict(torch.load(critic_path, map_location=device))
                print("[Transformer] Loaded Critic")
            except Exception as e:
                print(f"[Transformer] Failed to load Critic: {e}")

        critic_target_path = TRANSFORMER_MODEL_PATH / 'critic_target.pth'
        if critic_target_path.exists():
            try:
                self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))
                print("[Transformer] Loaded Critic Target")
            except Exception as e:
                print(f"[Transformer] Failed to load Critic Target: {e}")

        alpha_path = TRANSFORMER_MODEL_PATH / 'log_alpha.pth'
        if alpha_path.exists():
            try:
                self.log_alpha = torch.load(alpha_path, map_location=device)
                self.log_alpha.requires_grad_(True)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA_DISCRETE)
                print(f"[Transformer] Loaded log_alpha: alpha={self.log_alpha.exp().item():.4f}")
            except Exception as e:
                print(f"[Transformer] Failed to load log_alpha: {e}")

        opt_path = TRANSFORMER_MODEL_PATH / 'optimizer_state.pth'
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.actor_optimizer.load_state_dict(state['actor_optimizer'])
                self.critic_optimizer.load_state_dict(state['critic_optimizer'])
                if 'alpha_optimizer' in state:
                    self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
                self.total_it = state['total_it']
                self.episode_count = state.get('episode_count', 0)
                if 'epsilon' in state:
                    self.epsilon = state['epsilon']
                print(f"[Transformer] Loaded optimizer: total_it={self.total_it},"
                      f" episode={self.episode_count}, epsilon={self.epsilon:.4f}")
            except Exception as e:
                print(f"[Transformer] Failed to load optimizer state: {e}")

        training_state_path = TRANSFORMER_MODEL_PATH / 'training_state.pth'
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                persistent_entries = state.get('persistent_entries', [])
                loaded = min(len(persistent_entries), BUFFER_CAPACITY)
                for i in range(loaded):
                    entry = persistent_entries[i]
                    self.replay_buffer.store(
                        entry['state'], entry['action'],
                        entry['next_state'], entry['reward'], entry['done']
                    )
                if loaded > 0:
                    print(f"[Transformer] Loaded {loaded} replay buffer entries")
            except Exception as e:
                print(f"[Transformer] Failed to load replay buffer: {e}")


# ============================================================
# Stage 2 (視覺訓練) — 以下是原有的 classes
# ============================================================


class ReplayBuffer:
    """Disk-backed replay buffer for GCP upload compatibility."""

    def __init__(self, max_size=BUFFER_CAPACITY, save_dir=REPLAY_SAVE_PATH):
        self.max_size = max_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.ptr = 0
        self.size_count = 0
        self.index = []

    def store(self, state_tensor, action, next_state_tensor, reward, done):
        idx = self.ptr

        # Save tensors to disk as half-precision to save space
        state_path = self.save_dir / f"state_{idx}.pt"
        torch.save(state_tensor.cpu().half(), state_path)

        next_state_path = None
        if next_state_tensor is not None:
            next_state_path = self.save_dir / f"next_state_{idx}.pt"
            torch.save(next_state_tensor.cpu().half(), next_state_path)

        entry = {
            'state_path': str(state_path),
            'action': action.copy() if isinstance(action, np.ndarray) else np.array(action),
            'next_state_path': str(next_state_path) if next_state_path else None,
            'reward': float(reward),
            'done': bool(done)
        }

        if len(self.index) < self.max_size:
            self.index.append(entry)
        else:
            self.index[self.ptr] = entry

        self.ptr = (self.ptr + 1) % self.max_size
        self.size_count = min(self.size_count + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size_count, size=batch_size)

        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in indices:
            entry = self.index[i]
            states.append(torch.load(entry['state_path']).float())
            actions.append(entry['action'])

            if entry['next_state_path']:
                next_states.append(torch.load(entry['next_state_path']).float())
            else:
                next_states.append(torch.zeros(3, *IMAGE_SIZE))

            rewards.append(entry['reward'])
            dones.append(float(entry['done']))

        return (
            torch.stack(states).to(device),
            torch.tensor(np.array(actions), dtype=torch.float32).to(device),
            torch.stack(next_states).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def size(self):
        return self.size_count


class SACAgent:
    """SAC Agent with YOLO11n backbone and spatial attention."""

    def __init__(self, screen_region):
        self.screen_region = screen_region  # (x, y, w, h)
        self.action_dim = 2

        # Actor (owns backbone)
        self.actor = SACActorNetwork(embed_dim=256, action_dim=self.action_dim).to(device)

        # Twin Critics + target
        self.critic = SACCriticNetwork(embed_dim=256, action_dim=self.action_dim).to(device)
        self.critic_target = SACCriticNetwork(embed_dim=256, action_dim=self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Automatic entropy tuning
        self.target_entropy = TARGET_ENTROPY
        self.log_alpha = torch.tensor(
            np.log(INIT_ALPHA), dtype=torch.float32,
            requires_grad=True, device=device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        self.total_it = 0
        self.episode_count = 0

        # Image preprocessing (YOLO expects [0,1] range, 640x640)
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        self.try_load_model()

        # Register atexit handler for persistent save (once, in agent init)
        atexit.register(self.save_persistent)

    def preprocess_screen(self, screenshot_path):
        """Load and preprocess a screenshot for the agent.

        Args:
            screenshot_path: Path to screenshot image file
        Returns:
            torch.Tensor of shape (3, 640, 640)
        """
        try:
            image = Image.open(screenshot_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            print(f"Error preprocessing screen: {e}")
            return torch.zeros((3, *IMAGE_SIZE))

    def select_action(self, state, add_noise=True):
        """Select an action given the current state.

        Args:
            state: (3, 640, 640) tensor
            add_noise: If True, sample stochastically (SAC exploration).
                       If False, use deterministic mean.
        Returns:
            (action_np, log_info) where action_np is shape (2,) in [-1, 1]
        """
        state_batch = state.unsqueeze(0).to(device)

        self.actor.eval()
        with torch.no_grad():
            if add_noise:
                action, log_prob, mean = self.actor.sample(state_batch)
                raw_action = torch.tanh(mean).cpu().numpy().flatten()
                final_action = action.cpu().numpy().flatten()
            else:
                mean, _ = self.actor(state_batch)
                raw_action = torch.tanh(mean).cpu().numpy().flatten()
                final_action = raw_action.copy()
        self.actor.train()

        raw_coords = self.action_to_screen_coords(raw_action)
        final_coords = self.action_to_screen_coords(final_action)

        log_info = {
            'raw_action': raw_action,
            'final_action': final_action,
            'raw_coords': raw_coords,
            'final_coords': final_coords,
        }

        return np.clip(final_action, -1, 1), log_info

    def action_to_screen_coords(self, action):
        """Convert normalized action [-1, 1] to screen pixel coordinates.

        Args:
            action: numpy array of shape (2,) with values in [-1, 1]
        Returns:
            (screen_x, screen_y) as integers
        """
        x, y, w, h = self.screen_region

        norm_x = (action[0] + 1) / 2
        norm_y = (action[1] + 1) / 2

        screen_x = int(x + norm_x * w)
        screen_y = int(y + norm_y * h)

        return screen_x, screen_y

    def store_transition(self, state, action, next_state, reward, done):
        """Store a transition in the replay buffer.

        Args:
            state: (3, H, W) tensor
            action: numpy array of shape (2,)
            next_state: (3, H, W) tensor or None if terminal
            reward: float
            done: bool or int (0/1)
        """
        state_cpu = state.cpu()
        if next_state is not None:
            next_state_cpu = next_state.cpu()
        else:
            next_state_cpu = None

        self.replay_buffer.store(state_cpu, action, next_state_cpu, reward, done)

    def train_step(self):
        """Perform one SAC training step.

        Returns:
            dict with 'critic_loss' and 'actor_loss', or None if not enough data.
        """
        if self.replay_buffer.size() < BATCH_SIZE:
            print(f"Not enough data in replay buffer, size: {self.replay_buffer.size()}")
            return None

        print("Training step")
        self.total_it += 1

        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            next_embed = self.actor.get_embedding(next_state)
            target_q1, target_q2 = self.critic_target(next_embed, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = reward + (1 - done) * GAMMA * target_q

        # Detach embedding so critic loss doesn't update backbone
        current_embed = self.actor.get_embedding(state).detach()
        current_q1, current_q2 = self.critic(current_embed, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_action, log_prob, _ = self.actor.sample(state)
        actor_embed = self.actor.get_embedding(state)
        q1, q2 = self.critic(actor_embed.detach(), new_action)
        min_q = torch.min(q1, q2)
        actor_loss = (alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha (entropy coefficient) update ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update target critic ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        self.save_model()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def save_model(self):
        """Save model weights + optimizer state (called every train_step)."""
        print("Saving model")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), MODEL_SAVE_PATH / 'actor.pth')
        torch.save(self.critic.state_dict(), MODEL_SAVE_PATH / 'critic.pth')
        torch.save(self.critic_target.state_dict(), MODEL_SAVE_PATH / 'critic_target.pth')
        torch.save(self.log_alpha, MODEL_SAVE_PATH / 'log_alpha.pth')
        # Save optimizer state for crash resilience (without replay buffer)
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
        }, MODEL_SAVE_PATH / 'optimizer_state.pth')

    def try_load_model(self):
        actor_path = MODEL_SAVE_PATH / 'actor.pth'
        if actor_path.exists():
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=device))
                print("Loaded Actor model")
            except Exception as e:
                print(f"Failed to load Actor model: {e}")

        critic_path = MODEL_SAVE_PATH / 'critic.pth'
        if critic_path.exists():
            try:
                self.critic.load_state_dict(torch.load(critic_path, map_location=device))
                print("Loaded Critic model")
            except Exception as e:
                print(f"Failed to load Critic model: {e}")

        critic_target_path = MODEL_SAVE_PATH / 'critic_target.pth'
        if critic_target_path.exists():
            try:
                self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))
                print("Loaded Critic Target model")
            except Exception as e:
                print(f"Failed to load Critic Target model: {e}")

        alpha_path = MODEL_SAVE_PATH / 'log_alpha.pth'
        if alpha_path.exists():
            try:
                self.log_alpha = torch.load(alpha_path, map_location=device)
                self.log_alpha.requires_grad_(True)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
                print("Loaded log_alpha")
            except Exception as e:
                print(f"Failed to load log_alpha: {e}")

        # 載入 optimizer states (from most recent save — optimizer_state or training_state)
        optimizer_state_path = MODEL_SAVE_PATH / 'optimizer_state.pth'
        training_state_path = MODEL_SAVE_PATH / 'training_state.pth'

        # Prefer optimizer_state.pth (saved every train_step, more recent)
        opt_state = None
        if optimizer_state_path.exists():
            try:
                opt_state = torch.load(optimizer_state_path, map_location=device, weights_only=False)
                self.actor_optimizer.load_state_dict(opt_state['actor_optimizer'])
                self.critic_optimizer.load_state_dict(opt_state['critic_optimizer'])
                self.alpha_optimizer.load_state_dict(opt_state['alpha_optimizer'])
                self.total_it = opt_state['total_it']
                self.episode_count = opt_state.get('episode_count', 0)
                print(f"Loaded optimizer state: total_it={self.total_it}, episode={self.episode_count}")
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")

        # Load persistent replay buffer from training_state.pth
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                # If optimizer_state wasn't loaded, use training_state for optimizers too
                if opt_state is None:
                    self.actor_optimizer.load_state_dict(state['actor_optimizer'])
                    self.critic_optimizer.load_state_dict(state['critic_optimizer'])
                    self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
                    self.total_it = state['total_it']
                    self.episode_count = state.get('episode_count', 0)

                persistent_index = state.get('persistent_index', [])
                if persistent_index:
                    self._load_persistent_buffer(persistent_index)

                print(f"Loaded training state: replay_size={self.replay_buffer.size_count}")
            except Exception as e:
                print(f"Failed to load training state: {e}")

    def _load_persistent_buffer(self, persistent_index):
        """Load persistent buffer entries into runtime replay buffer."""
        REPLAY_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        # Clean old runtime buffer files
        for f in REPLAY_SAVE_PATH.glob("*.pt"):
            f.unlink()
        # Guard: don't exceed buffer capacity
        persistent_index = persistent_index[:self.replay_buffer.max_size]
        loaded_count = 0

        for entry in persistent_index:
            # Copy .pt files from persistent to runtime directory (re-index by loaded_count)
            old_state = Path(entry['state_path'])
            if not old_state.exists():
                print(f"Warning: persistent state file not found: {old_state}")
                continue

            new_state = REPLAY_SAVE_PATH / f"state_{loaded_count}.pt"
            shutil.copy2(str(old_state), str(new_state))

            new_next_state = None
            if entry['next_state_path']:
                old_next = Path(entry['next_state_path'])
                if old_next.exists():
                    new_next_state = REPLAY_SAVE_PATH / f"next_state_{loaded_count}.pt"
                    shutil.copy2(str(old_next), str(new_next_state))

            runtime_entry = {
                'state_path': str(new_state),
                'action': entry['action'].copy() if isinstance(entry['action'], np.ndarray) else np.array(entry['action']),
                'next_state_path': str(new_next_state) if new_next_state else None,
                'reward': entry['reward'],
                'done': entry['done']
            }
            self.replay_buffer.index.append(runtime_entry)
            loaded_count += 1

        self.replay_buffer.ptr = loaded_count
        self.replay_buffer.size_count = loaded_count
        print(f"Loaded {loaded_count} entries from persistent buffer into runtime buffer")

    def reset_episode(self):
        pass

    def on_episode_end(self):
        """Called when an episode ends. Handles episode counting and periodic save."""
        self.episode_count += 1
        print(f"Episode {self.episode_count} ended")
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"Periodic save triggered at episode {self.episode_count}")
            self.save_persistent()

    def save_persistent(self):
        """Save a stratified random subset of the replay buffer to disk for next startup.

        Stratified by reward: each reward value gets proportional representation.
        Saves SAVE_CAPACITY entries from the current BUFFER_CAPACITY buffer.
        """
        buf = self.replay_buffer
        if buf.size_count == 0:
            print("Replay buffer empty, skipping persistent save")
            return

        # Group entries by reward
        reward_groups = defaultdict(list)
        for i in range(buf.size_count):
            entry = buf.index[i]
            reward_groups[entry['reward']].append(i)

        # Calculate proportional counts per reward group
        total = buf.size_count
        target = min(SAVE_CAPACITY, total)
        selected_indices = []

        # Proportional sampling from each group (sorted by size, largest last for rounding remainder)
        remaining = target
        groups = sorted(reward_groups.items(), key=lambda x: len(x[1]))
        for i, (reward, indices) in enumerate(groups):
            if i == len(groups) - 1:
                # Last group gets remaining count to avoid rounding errors
                count = remaining
            else:
                count = round(len(indices) / total * target)
            count = min(count, len(indices), remaining)
            selected = random.sample(indices, count)
            selected_indices.extend(selected)
            remaining -= count
            if remaining <= 0:
                break

        # Create persistent save directory
        REPLAY_PERSISTENT_PATH.mkdir(parents=True, exist_ok=True)
        # Clean old persistent data
        for f in REPLAY_PERSISTENT_PATH.glob("*.pt"):
            f.unlink()

        # Copy selected entries with re-indexed paths
        persistent_index = []
        save_idx = 0
        for old_idx in selected_indices:
            old_entry = buf.index[old_idx]

            # Copy state tensor — skip if missing
            old_state = Path(old_entry['state_path'])
            if not old_state.exists():
                print(f"Warning: state file missing, skipping: {old_state}")
                continue

            new_state = REPLAY_PERSISTENT_PATH / f"state_{save_idx}.pt"
            shutil.copy2(str(old_state), str(new_state))

            # Copy next_state tensor
            new_next_state = None
            if old_entry['next_state_path']:
                old_next = Path(old_entry['next_state_path'])
                if old_next.exists():
                    new_next_state = REPLAY_PERSISTENT_PATH / f"next_state_{save_idx}.pt"
                    shutil.copy2(str(old_next), str(new_next_state))

            persistent_index.append({
                'state_path': str(new_state),
                'action': old_entry['action'].copy() if isinstance(old_entry['action'], np.ndarray) else np.array(old_entry['action']),
                'next_state_path': str(new_next_state) if new_next_state else None,
                'reward': old_entry['reward'],
                'done': old_entry['done']
            })
            save_idx += 1

        # Save persistent index in training_state.pth
        self._save_training_state(persistent_index)

        # Log reward distribution
        saved_rewards = defaultdict(int)
        for entry in persistent_index:
            saved_rewards[entry['reward']] += 1
        print(f"Persistent save: {len(persistent_index)} entries saved to {REPLAY_PERSISTENT_PATH}")
        print(f"Reward distribution: {dict(saved_rewards)}")

    def _save_training_state(self, persistent_index):
        """Save training state with persistent buffer index."""
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
            'persistent_index': persistent_index,
        }, MODEL_SAVE_PATH / 'training_state.pth')

    def log_action_image(self, state, log_info, step_count, reward=None):
        """Save state image with action markers for debugging."""
        if not LOG_ACTIONS or log_info is None:
            return

        from PIL import ImageDraw, ImageFont

        ACTION_LOG_PATH.mkdir(parents=True, exist_ok=True)

        # Convert tensor to PIL Image
        img_array = state.cpu().numpy()
        if img_array.ndim == 4:
            img_array = img_array.squeeze(0)
        img_array = (img_array * 255).astype(np.uint8)
        img_array = img_array.transpose(1, 2, 0)
        img = Image.fromarray(img_array)

        img_w, img_h = img.size

        # Map actions to image coordinates
        raw_norm = (log_info['raw_action'] + 1) / 2.0
        raw_img_x = int(raw_norm[0] * img_w)
        raw_img_y = int(raw_norm[1] * img_h)

        final_norm = (log_info['final_action'] + 1) / 2.0
        final_img_x = int(final_norm[0] * img_w)
        final_img_y = int(final_norm[1] * img_h)

        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Draw raw action (red dot)
        radius = 5
        draw.ellipse([raw_img_x - radius, raw_img_y - radius,
                      raw_img_x + radius, raw_img_y + radius],
                     fill='red', outline='darkred')

        # Draw final action (purple dot)
        draw.ellipse([final_img_x - radius, final_img_y - radius,
                      final_img_x + radius, final_img_y + radius],
                     fill='purple', outline='darkviolet')

        # Draw line between raw and final
        draw.line([raw_img_x, raw_img_y, final_img_x, final_img_y],
                  fill='yellow', width=1)

        # Text overlay
        raw_action = log_info['raw_action']
        final_action = log_info['final_action']
        raw_coords = log_info['raw_coords']
        final_coords = log_info['final_coords']

        text_lines = [
            f"Step: {step_count}",
            f"Raw tanh: ({raw_action[0]:.4f}, {raw_action[1]:.4f})",
            f"Raw screen: ({raw_coords[0]}, {raw_coords[1]})",
            f"Final tanh: ({final_action[0]:.4f}, {final_action[1]:.4f})",
            f"Final screen: ({final_coords[0]}, {final_coords[1]})",
        ]
        if reward is not None:
            text_lines.append(f"Reward: {reward:.1f}")

        text_y = 5
        for line in text_lines:
            bbox = draw.textbbox((5, text_y), line, font=font)
            draw.rectangle(bbox, fill='black')
            draw.text((5, text_y), line, fill='white', font=font)
            text_y += 15

        # Legend
        legend_y = img_h - 40
        draw.ellipse([10 - 4, legend_y - 4, 10 + 4, legend_y + 4], fill='red')
        draw.text((20, legend_y - 7), "Raw (mean)", fill='white', font=font)
        draw.ellipse([10 - 4, legend_y + 15 - 4, 10 + 4, legend_y + 15 + 4], fill='purple')
        draw.text((20, legend_y + 15 - 7), "Final (sampled)", fill='white', font=font)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_step_{step_count:04d}.png"
        img.save(ACTION_LOG_PATH / filename)
        print(f"Action log saved: {filename}")


_agent = None
def get_agent(screen_region=None):
    """Get or create the singleton SACAgent instance.

    Args:
        screen_region: tuple (x, y, w, h) defining the game area
    Returns:
        SACAgent instance
    """
    global _agent
    if _agent is None:
        _agent = SACAgent(screen_region)
    return _agent
