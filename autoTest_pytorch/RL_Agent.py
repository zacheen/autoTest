import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import random
from PIL import Image
import datetime
from pathlib import Path
from ultralytics import YOLO

# Hyperparameters
BATCH_SIZE = 32
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005
INIT_ALPHA = 0.2
TARGET_ENTROPY = -2.0  # = -action_dim
MEMORY_SIZE = 10000
IMAGE_SIZE = (640, 640)

# YOLO11n layer indices (discovered via forward pass)
# Layer 4: C3k2, output 128ch x 80x80 (mid-layer, spatial detail)
# Layer 6: C3k2, output 128ch x 40x40 (last backbone layer, semantic)
YOLO_MID_LAYER_IDX = 4
YOLO_LAST_LAYER_IDX = 6
YOLO_MID_CHANNELS = 128
YOLO_LAST_CHANNELS = 128
FUSED_CHANNELS = YOLO_MID_CHANNELS + YOLO_LAST_CHANNELS  # 256

# Paths
LOG_ACTIONS = True
ACTION_LOG_PATH = Path("./models/action_logs")
REPLAY_SAVE_PATH = Path("./models/replay_buffer")
MODEL_SAVE_PATH = Path("./models")

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


class ReplayBuffer:
    """Disk-backed replay buffer for GCP upload compatibility."""

    def __init__(self, max_size=MEMORY_SIZE, save_dir=REPLAY_SAVE_PATH):
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

        # Image preprocessing (YOLO expects [0,1] range, 640x640)
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        self.try_load_model()

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
        embed = self.actor.get_embedding(state)
        q1, q2 = self.critic(embed.detach(), new_action)
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
        print("Saving model")
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), MODEL_SAVE_PATH / 'actor.pth')
        torch.save(self.critic.state_dict(), MODEL_SAVE_PATH / 'critic.pth')
        torch.save(self.critic_target.state_dict(), MODEL_SAVE_PATH / 'critic_target.pth')
        torch.save(self.log_alpha, MODEL_SAVE_PATH / 'log_alpha.pth')

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

    def reset_episode(self):
        pass

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
