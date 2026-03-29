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
from collections import defaultdict
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
TARGET_ENTROPY = -2.0  # = -action_dim
BUFFER_CAPACITY = 600   # Runtime circular buffer in CPU RAM (~1.44GB)
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


class SpatialAttentionHead(nn.Module):
    """Spatial attention mechanism that preserves positional information."""

    def __init__(self, in_channels=128, embed_dim=256):
        super().__init__()

        # Learn attention weights per spatial location
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # Project attended features to embedding
        self.fc = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, features):
        """
        Args:
            features: (B, 128, 80, 80)
        Returns:
            (B, 256) embedding vector
        """
        B, C, H, W = features.shape

        # Compute attention map
        attn_map = self.attention_conv(features)  # (B, 1, H, W)
        attn_weights = F.softmax(attn_map.view(B, -1), dim=-1)  # (B, H*W)
        attn_weights = attn_weights.view(B, 1, H, W)  # (B, 1, H, W)

        # Weighted spatial pooling
        weighted = features * attn_weights  # (B, C, H, W)
        pooled = weighted.sum(dim=[2, 3])  # (B, C)

        # Project to embedding
        embedding = self.fc(pooled)  # (B, 256)
        return embedding


class SACActorNetwork(nn.Module):
    """SAC Actor: image -> Gaussian distribution over (x, y) actions."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()
        self.backbone = YOLO11nBackbone()
        self.attention = SpatialAttentionHead(in_channels=128, embed_dim=embed_dim)

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

    GridEncoder + SpatialAttentionHead + mean/log_std heads。
    使用 ScaledSigmoid 取代 tanh，輸出約 [-0.05, 1.05]。
    SpatialAttentionHead 和 head 的權重會轉移到 Stage 2。
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()
        self.encoder = GridEncoder()
        self.attention = SpatialAttentionHead(in_channels=128, embed_dim=embed_dim)
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

    有自己的 GridEncoder 和 SpatialAttentionHead（不跟 Actor 共享）。
    """

    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()
        self.encoder = GridEncoder()
        self.attention = SpatialAttentionHead(in_channels=128, embed_dim=embed_dim)

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


class Stage1ReplayBuffer:
    """Stage 1 用的 in-memory replay buffer (grid state 很小，不需要存磁碟)。"""

    def __init__(self, max_size=BUFFER_CAPACITY):
        self.max_size = max_size
        self.buffer = []
        self.ptr = 0
        self.size_count = 0

    def store(self, state, action, next_state, reward, done):
        """
        Args:
            state: (12, 10, 10) tensor
            action: numpy array (2,)
            next_state: (12, 10, 10) tensor
            reward: float
            done: bool
        """
        entry = {
            'state': state.cpu().clone(),
            'action': action.copy() if isinstance(action, np.ndarray) else np.array(action),
            'next_state': next_state.cpu().clone(),
            'reward': float(reward),
            'done': bool(done)
        }

        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
        else:
            self.buffer[self.ptr] = entry

        self.ptr = (self.ptr + 1) % self.max_size
        self.size_count = min(self.size_count + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size_count, size=batch_size)

        states = torch.stack([self.buffer[i]['state'] for i in indices]).to(device)
        actions = torch.tensor(
            np.array([self.buffer[i]['action'] for i in indices]),
            dtype=torch.float32
        ).to(device)
        next_states = torch.stack([self.buffer[i]['next_state'] for i in indices]).to(device)
        rewards = torch.tensor(
            [self.buffer[i]['reward'] for i in indices],
            dtype=torch.float32
        ).unsqueeze(1).to(device)
        dones = torch.tensor(
            [float(self.buffer[i]['done']) for i in indices],
            dtype=torch.float32
        ).unsqueeze(1).to(device)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.size_count

    def get_all_rewards(self):
        """取得所有 entry 的 reward（供 stratified save 用）。"""
        return [self.buffer[i]['reward'] for i in range(self.size_count)]


class Stage1SACAgent:
    """Stage 1 SAC Agent — 用離散 grid state 預訓練。

    訓練完成後可以匯出 SpatialAttentionHead + SAC head 的權重，
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

        # Entropy tuning
        self.target_entropy = TARGET_ENTROPY
        self.log_alpha = torch.tensor(
            np.log(INIT_ALPHA), dtype=torch.float32,
            requires_grad=True, device=device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        # Replay buffer (in-memory, grid state 很小)
        self.replay_buffer = Stage1ReplayBuffer(max_size=BUFFER_CAPACITY)
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
        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            # 用 critic_target 自己的 encoder (不是 actor 的)
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

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update target ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
        }

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
        torch.save(self.log_alpha, STAGE1_MODEL_PATH / 'stage1_log_alpha.pth')
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
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

        alpha_path = STAGE1_MODEL_PATH / 'stage1_log_alpha.pth'
        if alpha_path.exists():
            try:
                self.log_alpha = torch.load(alpha_path, map_location=device)
                self.log_alpha.requires_grad_(True)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
                print("[Stage1] Loaded log_alpha")
            except Exception as e:
                print(f"[Stage1] Failed to load log_alpha: {e}")

        opt_path = STAGE1_MODEL_PATH / 'stage1_optimizer_state.pth'
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.actor_optimizer.load_state_dict(state['actor_optimizer'])
                self.critic_optimizer.load_state_dict(state['critic_optimizer'])
                self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
                self.total_it = state['total_it']
                self.episode_count = state.get('episode_count', 0)
                print(f"[Stage1] Loaded optimizer state: total_it={self.total_it}, episode={self.episode_count}")
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
                    self.replay_buffer.buffer.append(entry)
                self.replay_buffer.ptr = len(persistent_entries) % self.replay_buffer.max_size
                self.replay_buffer.size_count = len(persistent_entries)
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
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_it': self.total_it,
            'episode_count': self.episode_count,
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
            # Entropy
            'log_alpha': self.log_alpha.detach().cpu(),
        }
        path = STAGE1_MODEL_PATH / 'stage1_weights.pth'
        torch.save(weights, path)
        print(f"[Stage1] Transfer weights saved to {path}")
        return path


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
