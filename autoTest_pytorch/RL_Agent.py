import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import random
from PIL import Image

# Hyperparameters
BATCH_SIZE = 32
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
MEMORY_SIZE = 10000
IMAGE_SIZE = (224, 224) # Resize input to this for ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=MEMORY_SIZE):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def store(self, state, action, next_state, reward, done):
        data = (state, action, next_state, reward, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(state)
            batch_actions.append(action)
            batch_next_states.append(next_state)
            batch_rewards.append(reward)
            batch_dones.append(done)

        return (
            torch.stack(batch_states).to(device),
            torch.tensor(np.array(batch_actions), dtype=torch.float32).to(device),
            torch.stack(batch_next_states).to(device) if batch_next_states[0] is not None else None,
            torch.tensor(np.array(batch_rewards), dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(batch_dones), dtype=torch.float32).unsqueeze(1).to(device)
        )

    def size(self):
        return len(self.storage)

class Actor(nn.Module):
    def __init__(self, action_dim=2):
        super(Actor, self).__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the last FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) # Output range [-1, 1]
        return x

class Critic(nn.Module):
    def __init__(self, action_dim=2):
        super(Critic, self).__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Critic needs to process (state + action)
        # We will concatenate action to the extracted features
        self.fc1 = nn.Linear(512 + action_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, action):
        s = self.features(state)
        s = s.view(s.size(0), -1) # Flatten features
        
        x = torch.cat([s, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TD3Agent:
    def __init__(self, screen_region):
        self.screen_region = screen_region
        # screen_region format: [((x, y, w, h), True/False), ...]
        # We need the valid game area. Assuming the first one with True is the main game area.

        self.action_dim = 2 # x, y
        
        # Initialize Actor and Critics
        self.actor = Actor(self.action_dim).to(device)
        self.actor_target = Actor(self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic_1 = Critic(self.action_dim).to(device)
        self.critic_1_target = Critic(self.action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=LR_CRITIC)

        self.critic_2 = Critic(self.action_dim).to(device)
        self.critic_2_target = Critic(self.action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer()
        self.total_it = 0
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.try_load_model()

    def preprocess_screen(self, screenshot_path):
        try:
            image = Image.open(screenshot_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor # Returns (C, H, W)
        except Exception as e:
            print(f"Error preprocessing screen: {e}")
            return torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

    def select_action(self, state, add_noise=True):
        # state is (C, H, W) tensor
        state = state.unsqueeze(0).to(device) # Add batch dim -> (1, C, H, W)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = action + noise
            
        return np.clip(action, -1, 1), {} # Return action and empty log_info

    def action_to_screen_coords(self, action):
        x, y, w, h = self.screen_region
        
        # Normalize action from [-1, 1] to [0, 1]
        norm_x = (action[0] + 1) / 2
        norm_y = (action[1] + 1) / 2
        
        screen_x = int(x + norm_x * w)
        screen_y = int(y + norm_y * h)
        
        return screen_x, screen_y

    def store_transition(self, state, action, next_state, reward, done):
        # state and next_state are tensors (C, H, W)
        # If next_state is None (game over), we handle it in sample or here?
        # ReplayBuffer expects objects.
        # Note: To save memory, we might want to store them as CPU tensors or numpy arrays.
        # But for now, let's keep it simple.
        
        # Ensure they are on CPU to save GPU memory
        state_cpu = state.cpu()
        next_state_cpu = next_state.cpu() if next_state is not None else None
        
        self.replay_buffer.store(state_cpu, action, next_state_cpu, reward, done)

    def train_step(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        
        if next_state is None:
             # This case shouldn't happen with the current sample logic if we handle None correctly
             # But if we have mixed None and Tensor in a batch, stack will fail.
             # We need to handle 'done' correctly. 
             # If done=1, next_state doesn't matter for target Q.
             # Let's assume sample() handles this or we filter.
             # Actually, my sample() implementation uses stack, which will fail if some are None.
             # Fix: In sample(), if next_state is None, we should probably substitute a dummy tensor
             # OR, we ensure we don't sample 'None' next_states? No, we need them for terminal states.
             # Better: Use a mask.
             pass

        # Handle None in next_state for batching
        # In this simple implementation, let's assume we always have a next_state image 
        # even if game over (just the last frame).
        # If the user code passes None for next_state on game over, we need to fix that in store_transition or here.
        # Looking at Demo code: "game_status.current_pic if not game_status.game_over else None"
        # So next_state CAN be None.
        # We must handle this.
        # FIX: We will create a dummy zero tensor for next_state if it is None in the batch.
        # But wait, sample() does `torch.stack(batch_next_states)`. This will crash if mixed.
        # I need to modify ReplayBuffer.sample to handle this.
        # For now, let's modify store_transition to store a zero tensor if next_state is None.
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * GAMMA * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        actor_loss = None
        # Delayed policy updates
        if self.total_it % POLICY_FREQ == 0:
            # Compute actor losse
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        self.save_model()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss else None
        }

    def log_action_image(self, current_screenshot, log_info, step_count, reward=None):
        # Optional: Save image with action for debugging
        pass

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.actor.state_dict(), 'models/actor.pth')
        torch.save(self.critic_1.state_dict(), 'models/critic_1.pth')
        torch.save(self.critic_2.state_dict(), 'models/critic_2.pth')

    def try_load_model(self):
        if os.path.exists('models/actor.pth'):
            try:
                self.actor.load_state_dict(torch.load('models/actor.pth'))
                self.actor_target.load_state_dict(self.actor.state_dict())
                print("Loaded Actor model")
            except:
                print("Failed to load Actor model")
        
        if os.path.exists('models/critic_1.pth'):
            try:
                self.critic_1.load_state_dict(torch.load('models/critic_1.pth'))
                self.critic_1_target.load_state_dict(self.critic_1.state_dict())
                print("Loaded Critic 1 model")
            except:
                print("Failed to load Critic 1 model")

        if os.path.exists('models/critic_2.pth'):
            try:
                self.critic_2.load_state_dict(torch.load('models/critic_2.pth'))
                self.critic_2_target.load_state_dict(self.critic_2.state_dict())
                print("Loaded Critic 2 model")
            except:
                print("Failed to load Critic 2 model")

    def reset_episode(self):
        pass

    # Fix for store_transition to handle None next_state
    def store_transition(self, state, action, next_state, reward, done):
        state_cpu = state.cpu()
        if next_state is None:
            next_state_cpu = torch.zeros_like(state_cpu)
        else:
            next_state_cpu = next_state.cpu()
        
        self.replay_buffer.store(state_cpu, action, next_state_cpu, reward, done)


def get_agent(screen_region):
    return TD3Agent(screen_region)
