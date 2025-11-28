# """
# Minesweeper RL Agent - Continuous Action Space Version
# Uses TD3 (Twin Delayed DDPG) algorithm.
# The model receives a screenshot and outputs (x, y) coordinates in the range [0, 1].
# """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
from pathlib import Path
from PIL import Image
import random
import torchvision.models as models
import gc

# ==================== Config ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class Config:
    # Screen region (adjust to your game window)
    SCREEN_LEFT = 0
    SCREEN_TOP = 0
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    # Model input size – we keep the original resolution (no forced resize)
    INPUT_SIZE = None  # kept for backward compatibility

    # Training parameters
    BATCH_SIZE = 16
    GAMMA = 0.99
    TAU = 0.005  # soft‑update coefficient
    LR_ACTOR = 1e-4
    LR_CRITIC = 3e-4
    MEMORY_SIZE = 50000

    # Exploration noise
    NOISE_STD = 0.05
    NOISE_CLIP = 0.5
    POLICY_DELAY = 2  # delayed actor update

    # Model persistence
    MODEL_PATH = Path("./rl_models")
    SAVE_INTERVAL = 5

CONFIG = Config()

# ==================== Replay Buffer ====================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        # Store tensors on CPU to avoid GPU OOM
        state_cpu = state.detach().cpu()
        next_state_cpu = next_state.detach().cpu() if next_state is not None else None
        self.buffer.append(Transition(state_cpu, action, next_state_cpu, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ==================== Feature Encoder (ResNet18) ====================
class ResNetEncoder(nn.Module):
    """Pre‑trained ResNet‑18 backbone used as a feature extractor.
    The final fully‑connected layer is replaced with an identity, leaving a
    512‑dimensional feature vector regardless of input resolution (thanks to the
    adaptive average‑pool inside ResNet).
    """
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.output_size = 512

    def forward(self, x):
        # x shape: (B, 3, H, W) – any H,W >= 224 works fine
        return self.resnet(x)  # shape (B, 512)

# ==================== Actor ====================
class Actor(nn.Module):
    """Maps a screen tensor to (x, y) coordinates in [0, 1]."""
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, state):
        features = self.encoder(state)
        return self.fc(features)

# ==================== Critic ====================
class Critic(nn.Module):
    """Twin critics for TD3 – estimate Q‑values for (state, action)."""
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.q1 = nn.Sequential(
            nn.Linear(self.encoder.output_size + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.encoder.output_size + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        features = self.encoder(state)
        x = torch.cat([features, action], dim=1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        features = self.encoder(state)
        x = torch.cat([features, action], dim=1)
        return self.q1(x)

# ==================== TD3 Agent ====================
class TD3Agent:
    def __init__(self, screen_region: tuple = None):
        """Initialize the agent.
        Args:
            screen_region: (left, top, width, height) of the game window.
        """
        if screen_region:
            self.screen_left, self.screen_top, self.screen_width, self.screen_height = screen_region
        else:
            self.screen_left = CONFIG.SCREEN_LEFT
            self.screen_top = CONFIG.SCREEN_TOP
            self.screen_width = CONFIG.SCREEN_WIDTH
            self.screen_height = CONFIG.SCREEN_HEIGHT

        # Networks
        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic().to(DEVICE)
        self.critic_target = Critic().to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=CONFIG.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CONFIG.LR_CRITIC)

        # Replay buffer
        self.memory = ReplayBuffer(CONFIG.MEMORY_SIZE)

        # Statistics
        self.steps = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        self._try_load_model()

    # -------------------- Model Persistence --------------------
    def _try_load_model(self):
        """Load a saved checkpoint if it exists."""
        model_file = CONFIG.MODEL_PATH / "td3_minesweeper.pth"
        if model_file.exists():
            ckpt = torch.load(model_file, map_location=DEVICE)
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            self.actor_target.load_state_dict(ckpt['actor_target'])
            self.critic_target.load_state_dict(ckpt['critic_target'])
            self.steps = ckpt.get('steps', 0)
            print(f"Loaded model, steps: {self.steps}")

    # -------------------- Pre‑processing --------------------
    def preprocess_screen(self, screenshot_path: str) -> torch.Tensor:
        img = Image.open(screenshot_path).convert('RGB')
        w, h = img.size  # PIL is (W, H)
        new_size = (w // 3, h // 3)
        img = img.resize(new_size, Image.BILINEAR)
        arr = np.array(img).transpose((2, 0, 1)) / 255.0
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return tensor

    def preprocess_from_array(self, screen_array: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(screen_array).convert('RGB')
        w, h = img.size
        new_size = (w // 3, h // 3)
        img = img.resize(new_size, Image.BILINEAR)
        arr = np.array(img).transpose((2, 0, 1)) / 255.0
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return tensor

    # -------------------- Action Selection --------------------
    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> np.ndarray:
        """Return an (x, y) action in [0, 1]."""
        with torch.no_grad():
            action = self.actor(state.to(DEVICE)).cpu().numpy().squeeze()
        if add_noise:
            noise = np.random.normal(0, CONFIG.NOISE_STD, size=2)
            action = action + noise
            action = np.clip(action, 0.0, 1.0)
        return action

    def action_to_screen_coords(self, action: np.ndarray) -> tuple:
        """Map the normalized action to absolute screen coordinates."""
        x = int(self.screen_left + action[0] * self.screen_width)
        y = int(self.screen_top + action[1] * self.screen_height)
        return x, y

    # -------------------- Experience Storage --------------------
    def store_transition(self, state, action, next_state, reward, done):
        """Save a transition into the replay buffer and update episode reward.
        Tensors are stored on CPU to keep GPU memory free.
        """
        self.memory.push(state, action, next_state, reward, done)
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

    # -------------------- Training Step --------------------
    def log_gpu_memory(self, tag=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            # print(f"[GPU {tag}] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
    
    def train_step(self) -> dict:
        self.log_gpu_memory("Start Train")
        if len(self.memory) < CONFIG.BATCH_SIZE:
            return None

        self.steps += 1
        batch = self.memory.sample(CONFIG.BATCH_SIZE)

        # Batch tensors
        state = torch.cat(batch.state).to(DEVICE)
        action = torch.tensor(np.array(batch.action), dtype=torch.float32).to(DEVICE)
        reward = torch.tensor(batch.reward, dtype=torch.float32).to(DEVICE).unsqueeze(1)
        done = torch.tensor(batch.done, dtype=torch.float32).to(DEVICE).unsqueeze(1)

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=DEVICE)
        non_final_next = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)

        # ----- Critic update -----
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                next_action = self.actor_target(non_final_next)
                noise = torch.clamp(
                    torch.randn_like(next_action) * CONFIG.NOISE_STD,
                    -CONFIG.NOISE_CLIP, CONFIG.NOISE_CLIP)
                next_action = torch.clamp(next_action + noise, 0.0, 1.0)
                target_q1, target_q2 = self.critic_target(non_final_next, next_action)
                target_q = torch.min(target_q1, target_q2)
                target = torch.zeros(CONFIG.BATCH_SIZE, 1, device=DEVICE, dtype=target_q.dtype)
                target[non_final_mask] = target_q
                target = reward + (1 - done) * CONFIG.GAMMA * target

            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        
        critic_loss_val = critic_loss.item()
        
        del current_q1, current_q2, critic_loss, target
        del target_q1, target_q2, target_q, next_action

        # ----- Actor update (delayed) -----
        actor_loss_val = None
        if self.steps % CONFIG.POLICY_DELAY == 0:
            with torch.cuda.amp.autocast():
                actor_output = self.actor(state)
                actor_loss = -self.critic.q1_forward(state, actor_output).mean()
            
            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            
            actor_loss_val = actor_loss.item()
            
            del actor_output, actor_loss
            
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        del state, action, reward, done, non_final_mask, non_final_next

        gc.collect()
        torch.cuda.empty_cache()

        # Periodic checkpoint
        if self.steps % CONFIG.SAVE_INTERVAL == CONFIG.SAVE_INTERVAL - 1:
            self.save_model()

        return {'critic_loss': critic_loss_val, 'actor_loss': actor_loss_val}

    # -------------------- Utilities --------------------
    def _soft_update(self, source, target):
        """Soft‑update target network parameters."""
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(CONFIG.TAU * s_param.data + (1 - CONFIG.TAU) * t_param.data)

    def reset_episode(self):
        self.current_episode_reward = 0

    def save_model(self):
        CONFIG.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'scaler': self.scaler.state_dict(), # Save scaler state
        }, CONFIG.MODEL_PATH / "td3_minesweeper.pth")
        print(f"Model saved, steps: {self.steps}")

    def get_stats(self) -> dict:
        """Return training statistics for logging / debugging."""
        return {
            'steps': self.steps,
            'memory_size': len(self.memory),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'episodes': len(self.episode_rewards),
        }

# ==================== Global Agent ====================
_agent = None

def get_agent(screen_region: tuple = None) -> TD3Agent:
    """Factory that returns a singleton TD3Agent instance."""
    global _agent
    if _agent is None:
        _agent = TD3Agent(screen_region)
    return _agent