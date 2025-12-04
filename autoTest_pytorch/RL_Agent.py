# """
# Minesweeper RL Agent - Continuous Action Space Version
# Uses TD3 (Twin Delayed DDPG) algorithm.
# The model receives a screenshot and outputs (x, y) coordinates in the range [-1, 1].
# """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple, defaultdict
from pathlib import Path
from PIL import Image
import random
import torchvision.models as models
import gc
import csv
import datetime
import matplotlib.pyplot as plt

# ==================== Config ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class Config:
    # Screen region (adjust to your game window)
    SCREEN_LEFT = 0
    SCREEN_TOP = 0
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    # Training parameters
    BATCH_SIZE = 24 # my device can accept 64, but too slow
    START_TRAIN_SIZE = 100
    MEMORY_SIZE = 1000
    EPISODE_MAX_LEN = 300
    GAMMA = 0.99
    TAU = 0.002  # soft-update coefficient
    LR_ACTOR = 0.00005
    LR_CRITIC = 0.0004
    
    # LR Scheduler
    LR_DECAY_STEP = 1000
    LR_DECAY_GAMMA = 0.9

    # Exploration noise
    NOISE_STD = 0.2
    NOISE_CLIP = 0.5
    POLICY_DELAY = 2  # delayed actor update

    # Regularization
    ACTION_REG_COEF = 0.1   # if hoping Model output be close to 0
    REWARD_SCALE = 0.1      # Scale rewards to keep gradients stable

    DISCRETE = True
    if DISCRETE :
        NOISE_CLIP = 0.05
        NOISE_PROB = 0.8
        ACTION_REG_COEF = 0.02

    # Action logging
    LOG_ACTIONS = True
    ACTION_LOG_PATH = Path("./rl_models/action_logs")

    # Model persistence
    MODEL_PATH = Path("./rl_models")
    STEP_LOG_FILE = Path("./rl_models/step_log.csv")        # 每步記錄
    EPISODE_LOG_FILE = Path("./rl_models/episode_log.csv")  # 每 episode 記錄
    PLOT_FILE = Path("./rl_models/training_curves.png")
    SAVE_INTERVAL = 5

CONFIG = Config()

# ==================== Replay Buffer ====================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        state = state.detach().cpu().to(torch.float16)
        if next_state is not None:
            next_state = next_state.detach().cpu().to(torch.float16)
        print("new reward: ", reward)
        self.buffer.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size, include_latest=False):
        groups = defaultdict(list)
        for t in self.buffer:
            r = t.reward
            groups[r].append(t)
        
        sampled_transitions = []
        if include_latest and len(self.buffer) > 0:
            latest_transition = self.buffer[-1]
            sampled_transitions.append(latest_transition)
            batch_size -= 1
        
        num_groups = len(groups)
        if num_groups == 0:
            return None

        samples_per_group = batch_size // num_groups
        remainder = batch_size % num_groups
        
        # Sample from each group (positive rewards have higher priority)
        loop_order = sorted(groups.items(), key=lambda x: x[0], reverse=True)
        for r, group in loop_order:
            count = samples_per_group + (1 if remainder > 0 else 0)
            remainder -= 1
            
            if count <= 0:
                continue

            if len(group) >= count:
                sampled_transitions.extend(random.sample(group, count))
            else:
                sampled_transitions.extend(random.choices(group, k=count))
        
        random.shuffle(sampled_transitions)
        return Transition(*zip(*sampled_transitions))

    def __len__(self):
        return len(self.buffer)

# ==================== Feature Encoder (ResNet18) ====================
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.output_size = 512

    def forward(self, x):
        return self.resnet(x)

# ==================== Actor ====================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.output_layer = nn.Linear(128, 2)
        self._init_output_layer()
    
    def _init_output_layer(self):
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, state):
        features = self.encoder(state)
        x = self.fc(features)
        raw_output = self.output_layer(x)
        return torch.tanh(raw_output)

# ==================== Critic ====================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        
        self.q1_fc = nn.Sequential(
            nn.Linear(self.encoder.output_size + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.q1_out = nn.Linear(128, 1)
        
        self.q2_fc = nn.Sequential(
            nn.Linear(self.encoder.output_size + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.q2_out = nn.Linear(128, 1)
        
        self._init_output_layers()
    
    def _init_output_layers(self):
        for out_layer in [self.q1_out, self.q2_out]:
            nn.init.xavier_uniform_(out_layer.weight)
            nn.init.zeros_(out_layer.bias)

    def forward(self, state, action):
        features = self.encoder(state)
        x = torch.cat([features, action], dim=1)
        return self.q1_out(self.q1_fc(x)), self.q2_out(self.q2_fc(x))

    def q1_forward(self, state, action):
        features = self.encoder(state)
        x = torch.cat([features, action], dim=1)
        return self.q1_out(self.q1_fc(x))

# ==================== Plotting ====================
def plot_training_log(step_log_file, episode_log_file, output_file):
    """讀取 CSV 並繪製訓練曲線（每步 + 每 episode）"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ===== 讀取 Step Log =====
    step_data = {'steps': [], 'critic_loss': [], 'actor_loss': [], 'q_mean': [], 'reward': []}
    
    if Path(step_log_file).exists():
        try:
            with open(step_log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step_data['steps'].append(int(row['step']))
                    step_data['critic_loss'].append(float(row['critic_loss']))
                    actor_loss = row['actor_loss']
                    step_data['actor_loss'].append(float(actor_loss) if actor_loss else None)
                    step_data['q_mean'].append(float(row['q_mean']))
                    step_data['reward'].append(float(row['reward']))
        except Exception as e:
            print(f"Error reading step log: {e}")
    
    # ===== 讀取 Episode Log =====
    episode_data = {'episodes': [], 'total_reward': [], 'episode_steps': []}
    
    if Path(episode_log_file).exists():
        try:
            with open(episode_log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    episode_data['episodes'].append(int(row['episode']))
                    episode_data['total_reward'].append(float(row['total_reward']))
                    episode_data['episode_steps'].append(int(row['episode_steps']))
        except Exception as e:
            print(f"Error reading episode log: {e}")
    
    # ===== 繪圖 =====
    
    # 1. Critic Loss (每步)
    ax = axes[0, 0]
    if step_data['steps']:
        ax.plot(step_data['steps'], step_data['critic_loss'], 'r-', alpha=0.7, linewidth=0.5)
        # 移動平均
        window = min(50, len(step_data['critic_loss']))
        if window > 1:
            ma = np.convolve(step_data['critic_loss'], np.ones(window)/window, mode='valid')
            ax.plot(step_data['steps'][window-1:], ma, 'r-', linewidth=2, label=f'MA({window})')
            ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Critic Loss (per step)')
    ax.grid(True, alpha=0.3)
    
    # 2. Actor Loss (每步)
    ax = axes[0, 1]
    if step_data['steps']:
        actor_steps = [s for s, a in zip(step_data['steps'], step_data['actor_loss']) if a is not None]
        actor_losses = [a for a in step_data['actor_loss'] if a is not None]
        if actor_losses:
            ax.plot(actor_steps, actor_losses, 'g-', alpha=0.7, linewidth=0.5)
            window = min(50, len(actor_losses))
            if window > 1:
                ma = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
                ax.plot(actor_steps[window-1:], ma, 'g-', linewidth=2, label=f'MA({window})')
                ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Actor Loss (per step)')
    ax.grid(True, alpha=0.3)
    
    # 3. Q Mean (每步)
    ax = axes[1, 0]
    if step_data['steps']:
        ax.plot(step_data['steps'], step_data['q_mean'], 'b-', alpha=0.7, linewidth=0.5)
        window = min(50, len(step_data['q_mean']))
        if window > 1:
            ma = np.convolve(step_data['q_mean'], np.ones(window)/window, mode='valid')
            ax.plot(step_data['steps'][window-1:], ma, 'b-', linewidth=2, label=f'MA({window})')
            ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Q Value')
    ax.set_title('Q Mean (per step)')
    ax.grid(True, alpha=0.3)
    
    # 4. Episode Reward
    ax = axes[1, 1]
    if episode_data['episodes']:
        ax.plot(episode_data['episodes'], episode_data['total_reward'], 'mo-', 
                alpha=0.7, linewidth=1, markersize=4, label='Episode Reward')
        window = min(10, len(episode_data['total_reward']))
        if window > 1:
            ma = np.convolve(episode_data['total_reward'], np.ones(window)/window, mode='valid')
            ax.plot(episode_data['episodes'][window-1:], ma, 'm-', linewidth=2, label=f'MA({window})')
        ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Reward')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to {output_file}")

# ==================== TD3 Agent ====================
class TD3Agent:
    def __init__(self, screen_region: tuple = None):
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
        
        # Schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=CONFIG.LR_DECAY_STEP, gamma=CONFIG.LR_DECAY_GAMMA
        )
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=CONFIG.LR_DECAY_STEP, gamma=CONFIG.LR_DECAY_GAMMA
        )

        # Replay buffer
        self.memory = ReplayBuffer(CONFIG.MEMORY_SIZE)

        # Statistics
        self.steps = 0
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=CONFIG.EPISODE_MAX_LEN)
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
        # Logging
        self._init_log_files()
        
        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        self._try_load_model()

    def _init_log_files(self):
        """Initialize CSV log files with headers."""
        CONFIG.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        # Step log - 每步記錄
        if not CONFIG.STEP_LOG_FILE.exists():
            with open(CONFIG.STEP_LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'step', 'episode', 'critic_loss', 'actor_loss', 
                                'q_mean', 'q_std', 'reward', 'action_x', 'action_y'])
        
        # Episode log - 每 episode 記錄
        if not CONFIG.EPISODE_LOG_FILE.exists():
            with open(CONFIG.EPISODE_LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'episode', 'total_reward', 'episode_steps', 'total_steps'])

    def _try_load_model(self):
        model_file = CONFIG.MODEL_PATH / "td3_minesweeper.pth"
        if model_file.exists():
            ckpt = torch.load(model_file, map_location='cpu')
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            self.actor_target.load_state_dict(ckpt['actor_target'])
            self.critic_target.load_state_dict(ckpt['critic_target'])
            
            if 'actor_optimizer' in ckpt:
                self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            if 'critic_optimizer' in ckpt:
                self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
            if 'scaler' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler'])
            
            # Load Scheduler state
            if 'actor_scheduler' in ckpt:
                self.actor_scheduler.load_state_dict(ckpt['actor_scheduler'])
            if 'critic_scheduler' in ckpt:
                self.critic_scheduler.load_state_dict(ckpt['critic_scheduler'])

            # Load prev_train_data
            if 'prev_train_data' in ckpt:
                self.memory.buffer = deque(ckpt['prev_train_data'], maxlen=CONFIG.MEMORY_SIZE)
                print(f"Loaded prev_train_data with {len(self.memory)} transitions")

            self.steps = ckpt.get('steps', 0)
            self.episode_count = ckpt.get('episode_count', 0)
            print(f"Loaded model, steps: {self.steps}, episodes: {self.episode_count}")
            print(f"Current LR - Actor: {self.actor_scheduler.get_last_lr()[0]:.6f}, Critic: {self.critic_scheduler.get_last_lr()[0]:.6f}")

    def preprocess_screen(self, screenshot_path: str) -> torch.Tensor:
        img = Image.open(screenshot_path).convert('RGB')
        w, h = img.size
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

    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> np.ndarray:
        """Return an (x, y) action in [-1, 1]."""
        if torch.isnan(state).any():
            print("WARNING: NaN detected in state tensor. Returning random action.")
            return np.random.uniform(-1, 1, size=2)

        with torch.no_grad():
            action = self.actor(state.to(DEVICE)).cpu().numpy().squeeze()
        
        if np.isnan(action).any():
            print("WARNING: NaN detected in actor output. Returning random action.")
            return np.random.uniform(-1, 1, size=2)

        if add_noise:
            # DISCRETE should have a higher possibility of no noise, so that it would know the action is correct or not
            if CONFIG.DISCRETE and (random.random() > CONFIG.NOISE_PROB):
                print("<No noise precise click>")
            else:
                noise = np.random.normal(0, CONFIG.NOISE_STD, size=2)
                action = action + noise
                action = np.clip(action, -1.0, 1.0)
        return action

    def select_action_with_log(self, state: torch.Tensor, add_noise: bool = True) -> tuple:
        """
        Return action and log info.
        Returns: (final_action, log_info)
        """
        if torch.isnan(state).any():
            print("WARNING: NaN detected in state tensor. Returning random action.")
            random_action = np.random.uniform(-1, 1, size=2)
            return random_action, None

        with torch.no_grad():
            raw_action = self.actor(state.to(DEVICE)).cpu().numpy().squeeze()
        
        if np.isnan(raw_action).any():
            print("WARNING: NaN detected in actor output. Returning random action.")
            random_action = np.random.uniform(-1, 1, size=2)
            return random_action, None

        raw_coords = self.action_to_screen_coords(raw_action)
        
        final_action = raw_action.copy()
        noise_applied = False
        
        if add_noise:
            if CONFIG.DISCRETE and (random.random() > CONFIG.NOISE_PROB):
                print("<No noise precise click>")
            else:
                noise = np.random.normal(0, CONFIG.NOISE_STD, size=2)
                final_action = raw_action + noise
                final_action = np.clip(final_action, -1.0, 1.0)
                noise_applied = True
        
        final_coords = self.action_to_screen_coords(final_action)
        
        log_info = {
            'raw_action': raw_action,
            'final_action': final_action,
            'raw_coords': raw_coords,
            'final_coords': final_coords,
            'noise_applied': noise_applied
        }
        
        return final_action, log_info

    def log_action_image(self, state: torch.Tensor, log_info: dict, step_count: int, reward: float = None):
        """Save state image with action markers."""
        if not CONFIG.LOG_ACTIONS or log_info is None:
            return
        
        from PIL import ImageDraw, ImageFont
        
        CONFIG.ACTION_LOG_PATH.mkdir(parents=True, exist_ok=True)
        
        img_array = state.squeeze(0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        img_array = img_array.transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        
        img_w, img_h = img.size
        raw_img_x, raw_img_y = self.action_to_coords(log_info['raw_action'], img_w, img_h)
        final_img_x, final_img_y = self.action_to_coords(log_info['final_action'], img_w, img_h)
        
        radius = 5
        draw.ellipse([raw_img_x - radius, raw_img_y - radius, 
                      raw_img_x + radius, raw_img_y + radius], 
                     fill='red', outline='darkred')
        
        draw.ellipse([final_img_x - radius, final_img_y - radius,
                      final_img_x + radius, final_img_y + radius],
                     fill='purple', outline='darkviolet')
        
        if log_info['noise_applied']:
            draw.line([raw_img_x, raw_img_y, final_img_x, final_img_y], 
                      fill='yellow', width=1)
        
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
            f"Noise: {'Yes' if log_info['noise_applied'] else 'No'}",
        ]
        if reward is not None:
            text_lines.append(f"Reward: {reward:.1f}")
        
        text_y = 5
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        for line in text_lines:
            bbox = draw.textbbox((5, text_y), line, font=font)
            draw.rectangle(bbox, fill='black')
            draw.text((5, text_y), line, fill='white', font=font)
            text_y += 15
        
        legend_y = img_h - 40
        draw.ellipse([10 - 4, legend_y - 4, 10 + 4, legend_y + 4], fill='red')
        draw.text((20, legend_y - 7), "Raw (no noise)", fill='white', font=font)
        draw.ellipse([10 - 4, legend_y + 15 - 4, 10 + 4, legend_y + 15 + 4], fill='purple')
        draw.text((20, legend_y + 15 - 7), "Final (with noise)", fill='white', font=font)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_step_{step_count:04d}.png"
        img.save(CONFIG.ACTION_LOG_PATH / filename)
        print(f"Action log saved: {filename}")

    def action_to_normalized(self, action: np.ndarray) -> np.ndarray:
        """Convert action from [-1, 1] to [0, 1] normalized coordinates."""
        return (action + 1) / 2.0

    def action_to_coords(self, action: np.ndarray, width, height, left = 0, top = 0) -> tuple:
        norm_action = self.action_to_normalized(action)
        x = int(left + norm_action[0] * width)
        y = int(top + norm_action[1] * height)
        return x, y

    def action_to_screen_coords(self, action: np.ndarray) -> tuple:
        return self.action_to_coords(action, self.screen_width, self.screen_height, self.screen_left, self.screen_top)

    def store_transition(self, state, action, next_state, reward, done):
        # Scale reward for training stability
        scaled_reward = reward * CONFIG.REWARD_SCALE
        self.memory.push(state, action, next_state, scaled_reward, done)
        self.current_episode_reward += reward
        self.current_episode_steps += 1
        
        # Episode 結束時記錄
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self._log_episode()
            self.current_episode_reward = 0
            self.current_episode_steps = 0

    def _log_episode(self):
        """Log episode summary to CSV."""
        with open(CONFIG.EPISODE_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.episode_count,
                f"{self.current_episode_reward:.4f}",
                self.current_episode_steps,
                self.steps
            ])

    def _log_step(self, critic_loss, actor_loss, q_mean, q_std, reward, action):
        """Log every training step to CSV."""
        with open(CONFIG.STEP_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.steps,
                self.episode_count,
                f"{critic_loss:.6f}",
                f"{actor_loss:.6f}" if actor_loss is not None else "",
                f"{q_mean:.6f}",
                f"{q_std:.6f}",
                f"{reward:.4f}",
                f"{action[0]:.4f}" if action is not None else "",
                f"{action[1]:.4f}" if action is not None else ""
            ])

    def log_gpu_memory(self, tag=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[GPU {tag}] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
    
    def train_step(self) -> dict:
        if len(self.memory) < CONFIG.START_TRAIN_SIZE:
            return None

        self.steps += 1
        batch = self.memory.sample(CONFIG.BATCH_SIZE, include_latest=True)

        state = torch.cat(batch.state).to(DEVICE).float()
        action = torch.tensor(np.array(batch.action), dtype=torch.float32).to(DEVICE)
        reward = torch.tensor(batch.reward, dtype=torch.float32).to(DEVICE).unsqueeze(1)
        done = torch.tensor(batch.done, dtype=torch.float32).to(DEVICE).unsqueeze(1)

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=DEVICE)
        non_final_next = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE).float()

        # ----- Critic update -----
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                next_action = self.actor_target(non_final_next)
                if not CONFIG.DISCRETE:
                    noise = torch.clamp(
                        torch.randn_like(next_action) * CONFIG.NOISE_STD,
                        -CONFIG.NOISE_CLIP, CONFIG.NOISE_CLIP)
                    next_action = torch.clamp(next_action + noise, -1.0, 1.0)
                target_q1, target_q2 = self.critic_target(non_final_next, next_action)
                target_q = torch.min(target_q1, target_q2)
                target = torch.zeros(CONFIG.BATCH_SIZE, 1, device=DEVICE, dtype=target_q.dtype)
                target[non_final_mask] = target_q
                target = reward + (1 - done) * CONFIG.GAMMA * target

            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        
        # 記錄 Q 值統計
        q_mean = current_q1.mean().item()
        q_std = current_q1.std().item()
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
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
                # Actor Loss = -Q_value + Regularization
                # We want to maximize Q (minimize -Q) and minimize action magnitude
                actor_loss = -self.critic.q1_forward(state, actor_output).mean()
                actor_loss += CONFIG.ACTION_REG_COEF * (actor_output ** 2).mean()
            
            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            
            actor_loss_val = actor_loss.item()
            
            del actor_output, actor_loss
            
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        # Step schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 記錄每步數據
        latest_action = batch.action[-1] if batch.action else None
        latest_reward = batch.reward[-1] if batch.reward else 0
        self._log_step(critic_loss_val, actor_loss_val, q_mean, q_std, latest_reward, latest_action)

        del state, action, reward, done, non_final_mask, non_final_next

        gc.collect()
        torch.cuda.empty_cache()

        if self.steps % CONFIG.SAVE_INTERVAL == CONFIG.SAVE_INTERVAL - 1:
            self.save_model()

        return {'critic_loss': critic_loss_val, 'actor_loss': actor_loss_val, 'q_mean': q_mean}

    def _soft_update(self, source, target):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(CONFIG.TAU * s_param.data + (1 - CONFIG.TAU) * t_param.data)

    def reset_episode(self):
        self.current_episode_reward = 0
        self.current_episode_steps = 0

    def save_model(self):
        CONFIG.MODEL_PATH.mkdir(parents=True, exist_ok=True)

        if len(self.memory) > CONFIG.START_TRAIN_SIZE:
            batch = self.memory.sample(CONFIG.START_TRAIN_SIZE, include_latest=False)
            # 轉換回 list of Transition 格式
            prev_train_data = [
                Transition(s, a, ns, r, d) 
                for s, a, ns, r, d in zip(batch.state, batch.action, batch.next_state, batch.reward, batch.done)
            ]
        else:
            prev_train_data = list(self.memory.buffer)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'steps': self.steps,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'scaler': self.scaler.state_dict(),
            'prev_train_data': prev_train_data,
        }, CONFIG.MODEL_PATH / "td3_minesweeper.pth")
        print(f"Model saved, steps: {self.steps}, episodes: {self.episode_count}")
        
        # 更新訓練曲線圖
        self._plot_training_curves()

    def _plot_training_curves(self):
        try:
            plot_training_log(
                str(CONFIG.STEP_LOG_FILE),
                str(CONFIG.EPISODE_LOG_FILE),
                str(CONFIG.PLOT_FILE)
            )
        except Exception as e:
            print(f"Failed to plot training curves: {e}")

    def get_stats(self) -> dict:
        return {
            'steps': self.steps,
            'episodes': self.episode_count,
            'memory_size': len(self.memory),
            'avg_reward': np.mean(list(self.episode_rewards)[-CONFIG.EPISODE_MAX_LEN:]) if self.episode_rewards else 0,
        }

# ==================== Global Agent ====================
_agent = None

def get_agent(screen_region: tuple = None) -> TD3Agent:
    global _agent
    if _agent is None:
        _agent = TD3Agent(screen_region)
    return _agent