"""
Minesweeper RL Agent - 連續動作空間版本
使用 TD3 (Twin Delayed DDPG) 算法
模型直接輸出 (x, y) 座標，範圍 0~1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
from pathlib import Path
from PIL import Image
import random
import pyautogui

# ============== 設定 ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class Config:
    # 螢幕範圍（需要根據你的遊戲視窗調整）
    SCREEN_LEFT = 0
    SCREEN_TOP = 0
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    
    # 模型輸入
    INPUT_SIZE = 128
    
    # 訓練參數
    BATCH_SIZE = 16
    GAMMA = 0.99
    TAU = 0.005              # soft update 係數
    LR_ACTOR = 1e-4
    LR_CRITIC = 3e-4
    MEMORY_SIZE = 50000
    
    # 探索噪音
    NOISE_STD = 0.2          # 動作噪音標準差
    NOISE_CLIP = 0.5         # 噪音裁剪範圍
    POLICY_DELAY = 2         # 延遲更新 Actor
    
    # 儲存
    MODEL_PATH = Path("./rl_models")

CONFIG = Config()

# ============== 經驗回放 ==============
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        self.buffer.append(Transition(state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    
    def __len__(self):
        return len(self.buffer)


# ============== CNN 特徵提取器 ==============
class CNNEncoder(nn.Module):
    """共用的 CNN 特徵提取器"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        # 計算輸出大小
        with torch.no_grad():
            dummy = torch.zeros(1, 3, CONFIG.INPUT_SIZE, CONFIG.INPUT_SIZE)
            self.output_size = self.conv(dummy).view(1, -1).size(1)
    
    def forward(self, x):
        return self.conv(x).view(x.size(0), -1)


# ============== Actor 網路 ==============
class Actor(nn.Module):
    """
    輸入: 螢幕截圖
    輸出: (x, y) 座標，範圍 [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 輸出 x, y
            nn.Sigmoid()        # 限制在 [0, 1]
        )
    
    def forward(self, state):
        features = self.encoder(state)
        return self.fc(features)


# ============== Critic 網路 ==============
class Critic(nn.Module):
    """
    輸入: 螢幕截圖 + 動作 (x, y)
    輸出: Q 值
    """
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(self.encoder.output_size + 2, 256),  # +2 是動作維度
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Q2 (TD3 使用雙 Critic)
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


# ============== TD3 Agent ==============
class TD3Agent:
    def __init__(self, screen_region: tuple = None):
        """
        Args:
            screen_region: (left, top, width, height) 遊戲視窗區域
                          如果是 None，使用 Config 的預設值
        """
        if screen_region:
            self.screen_left, self.screen_top, self.screen_width, self.screen_height = screen_region
        else:
            self.screen_left = CONFIG.SCREEN_LEFT
            self.screen_top = CONFIG.SCREEN_TOP
            self.screen_width = CONFIG.SCREEN_WIDTH
            self.screen_height = CONFIG.SCREEN_HEIGHT
        
        # 網路
        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic().to(DEVICE)
        self.critic_target = Critic().to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 優化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=CONFIG.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CONFIG.LR_CRITIC)
        
        # 經驗回放
        self.memory = ReplayBuffer(CONFIG.MEMORY_SIZE)
        
        # 統計
        self.steps = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        self._try_load_model()
    
    def _try_load_model(self):
        """嘗試載入模型"""
        model_file = CONFIG.MODEL_PATH / "td3_minesweeper.pth"
        if model_file.exists():
            ckpt = torch.load(model_file, map_location=DEVICE)
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            self.actor_target.load_state_dict(ckpt['actor_target'])
            self.critic_target.load_state_dict(ckpt['critic_target'])
            self.steps = ckpt.get('steps', 0)
            print(f"Loaded model, steps: {self.steps}")
    
    def preprocess_screen(self, screenshot_path: str) -> torch.Tensor:
        """截圖轉換為模型輸入"""
        img = Image.open(screenshot_path).convert('RGB')
        img = img.resize((CONFIG.INPUT_SIZE, CONFIG.INPUT_SIZE))
        arr = np.array(img).transpose((2, 0, 1)) / 255.0
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    
    def preprocess_from_array(self, screen_array: np.ndarray) -> torch.Tensor:
        """從 numpy array 轉換"""
        img = Image.fromarray(screen_array).convert('RGB')
        img = img.resize((CONFIG.INPUT_SIZE, CONFIG.INPUT_SIZE))
        arr = np.array(img).transpose((2, 0, 1)) / 255.0
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    
    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> np.ndarray:
        """
        選擇動作
        Returns:
            action: np.ndarray of shape (2,)，值在 [0, 1] 之間
        """
        with torch.no_grad():
            action = self.actor(state.to(DEVICE)).cpu().numpy().squeeze()
        
        if add_noise:
            noise = np.random.normal(0, CONFIG.NOISE_STD, size=2)
            action = action + noise
            action = np.clip(action, 0.0, 1.0)
        
        return action
    
    def action_to_screen_coords(self, action: np.ndarray) -> tuple:
        """
        將 [0, 1] 的動作轉換為螢幕座標
        Args:
            action: (x_ratio, y_ratio) 在 [0, 1] 範圍
        Returns:
            (screen_x, screen_y) 實際像素座標
        """
        x = int(self.screen_left + action[0] * self.screen_width)
        y = int(self.screen_top + action[1] * self.screen_height)
        return x, y
    
    def store_transition(self, state, action, next_state, reward, done):
        """儲存經驗"""
        self.memory.push(state, action, next_state, reward, done)
        self.current_episode_reward += reward
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
    
    def train_step(self) -> dict:
        """執行一步訓練，返回 loss 資訊"""
        if len(self.memory) < CONFIG.BATCH_SIZE:
            return None
        
        self.steps += 1
        batch = self.memory.sample(CONFIG.BATCH_SIZE)
        
        # 準備 batch 資料
        state = torch.cat(batch.state).to(DEVICE)
        action = torch.tensor(np.array(batch.action), dtype=torch.float32).to(DEVICE)
        reward = torch.tensor(batch.reward, dtype=torch.float32).to(DEVICE).unsqueeze(1)
        done = torch.tensor(batch.done, dtype=torch.float32).to(DEVICE).unsqueeze(1)
        
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=DEVICE)
        non_final_next = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
        
        # ===== 更新 Critic =====
        with torch.no_grad():
            # Target policy smoothing
            next_action = self.actor_target(non_final_next)
            noise = torch.clamp(
                torch.randn_like(next_action) * CONFIG.NOISE_STD,
                -CONFIG.NOISE_CLIP, CONFIG.NOISE_CLIP
            )
            next_action = torch.clamp(next_action + noise, 0.0, 1.0)
            
            # Target Q
            target_q1, target_q2 = self.critic_target(non_final_next, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # 完整的 target
            target = torch.zeros(CONFIG.BATCH_SIZE, 1, device=DEVICE)
            target[non_final_mask] = target_q
            target = reward + (1 - done) * CONFIG.GAMMA * target
        
        # Current Q
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = None
        
        # ===== 延遲更新 Actor =====
        if self.steps % CONFIG.POLICY_DELAY == 0:
            # Actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            actor_loss = actor_loss.item()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss
        }
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(CONFIG.TAU * s_param.data + (1 - CONFIG.TAU) * t_param.data)
    
    def reset_episode(self):
        """重置 episode"""
        self.current_episode_reward = 0
    
    def save_model(self):
        """儲存模型"""
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
        }, CONFIG.MODEL_PATH / "td3_minesweeper.pth")
        print(f"Model saved, steps: {self.steps}")
    
    def get_stats(self) -> dict:
        """取得訓練統計"""
        return {
            'steps': self.steps,
            'memory_size': len(self.memory),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'episodes': len(self.episode_rewards),
        }


# ============== 全域 Agent ==============
_agent = None

def get_agent(screen_region: tuple = None) -> TD3Agent:
    """
    取得 Agent 實例
    Args:
        screen_region: (left, top, width, height) 遊戲視窗區域
    """
    global _agent
    if _agent is None:
        _agent = TD3Agent(screen_region)
    return _agent