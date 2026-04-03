"""
Standard SAC with continuous (x, y) ∈ [0,1]² output + frozen Transformer backbone.

架構:
  Frozen backbone (Transformer) → condition (64-d)
  Actor: condition → Gaussian → sigmoid → (x, y) ∈ [0,1]²
  Critic: Q(condition, action) → scalar (twin Q)

執行方式:
  cd autoTest_clau
  python autoTest_pytorch/train_sac_continuous.py

監控:
  tensorboard --logdir runs/sac_continuous/
"""

import sys
import time
import datetime
import random
import atexit
import numpy as np
from collections import deque, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Minesweeper.MinesweeperLogic import MinesweeperLogic
from RL_Agent import TransformerActorNetwork, PERReplayBuffer, device, \
    PER_CAPACITY, PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_EPSILON, \
    Stage1ReplayBuffer, BUFFER_CAPACITY
from util import TeeOutput, CSVLogger, compute_reward, action_to_grid


# ============================================================
# Hyperparameters
# ============================================================

GRID_ROWS = 6
GRID_COLS = 6
GRID_MINES = 4

MAX_EPISODES = 20000
MAX_STEPS_PER_EPISODE = 200
WARMUP_STEPS = 1000       # 前 N 步用 random action 填 buffer

BATCH_SIZE = 256
GAMMA = 0.9
TAU = 0.005
LR = 3e-4
INIT_ALPHA = 0.2
TARGET_ENTROPY = -2.0     # = -action_dim

LOG_INTERVAL = 50
EVAL_INTERVAL = 200
EVAL_EPISODES = 20
SAVE_INTERVAL = 500
SAVE_CAPACITY = 5000

FROZEN_BACKBONE_PATH = Path("./models/stage1_transformer/frozen_backbone.pth")
MODEL_PATH = Path("./models/sac_continuous")
TENSORBOARD_DIR = Path("./runs/sac_continuous")
CSV_LOG_PATH = MODEL_PATH / "training_log.csv"


# ============================================================
# SAC Networks
# ============================================================

class CrossAttentionPooling(nn.Module):
    """用 learnable query token 做 cross-attention，將 N 個 token 壓成 1 個向量。

    query (1, d) attends to key/value (N, d) → output (1, d)
    """

    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))  # learnable query
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features):
        """
        Args:
            features: (B, N, d_model) — N 個 token
        Returns:
            (B, d_model) — 1 個 summary vector
        """
        B = features.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        out, _ = self.cross_attn(query, features, features)  # (B, 1, d_model)
        out = self.norm(out)
        return out.squeeze(1)  # (B, d_model)


class ContinuousSACActor(nn.Module):
    """SAC Actor: token features → cross-attention → Gaussian → sigmoid → (x, y) ∈ [0,1]².

    接收 backbone 的 per-token features (B, N, 64)，用 cross-attention 壓成 (B, 64)，
    再輸出 continuous action。
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, d_model=64, action_dim=2, hidden_dim=256, nhead=4):
        super().__init__()
        # 1 層 self-attention 讓 Actor 有自己的特徵處理
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        # Cross-attention: N tokens → 1 vector
        self.pool = CrossAttentionPooling(d_model=d_model, nhead=nhead)

        # MLP → mean + log_std
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _encode(self, features):
        """Features → summary vector (B, d_model)."""
        x = self.self_attn(features)    # (B, N, d_model)
        return self.pool(x)             # (B, d_model)

    def forward(self, features):
        """Returns mean, log_std."""
        h = self._encode(features)
        h = self.trunk(h)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, features):
        """Reparameterization trick + sigmoid squashing.

        Args:
            features: (B, N, 64) — backbone token features
        Returns:
            action: (B, 2) in [0, 1]²
            log_prob: (B, 1)
        """
        mean, log_std = self.forward(features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        z = normal.rsample()
        action = torch.sigmoid(z)

        # Log-prob with sigmoid correction
        log_prob = normal.log_prob(z) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        return action, log_prob

    def deterministic(self, features):
        """Deterministic action for evaluation."""
        mean, _ = self.forward(features)
        return torch.sigmoid(mean)


class ContinuousSACCritic(nn.Module):
    """Twin Q-Network with cross-attention: Q(features, action) → scalar.

    接收 per-token features (B, N, 64) + action (B, 2)，
    各自用 cross-attention 壓成 (B, 64)，再 concat action 算 Q-value。
    """

    def __init__(self, d_model=64, action_dim=2, hidden_dim=256, nhead=4):
        super().__init__()

        # Q1 branch
        self.q1_self_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.q1_pool = CrossAttentionPooling(d_model=d_model, nhead=nhead)
        self.q1_head = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 branch
        self.q2_self_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.q2_pool = CrossAttentionPooling(d_model=d_model, nhead=nhead)
        self.q2_head = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, action):
        """
        Args:
            features: (B, N, 64) — backbone token features
            action: (B, 2)
        Returns:
            q1: (B, 1), q2: (B, 1)
        """
        # Q1
        x1 = self.q1_self_attn(features)
        s1 = self.q1_pool(x1)  # (B, 64)
        q1 = self.q1_head(torch.cat([s1, action], dim=-1))

        # Q2
        x2 = self.q2_self_attn(features)
        s2 = self.q2_pool(x2)  # (B, 64)
        q2 = self.q2_head(torch.cat([s2, action], dim=-1))

        return q1, q2


# ============================================================
# SAC Agent
# ============================================================

class SACContinuousAgent:
    """Standard SAC with frozen Transformer backbone."""

    def __init__(self, grid_h=6, grid_w=6):
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Frozen backbone
        self.backbone = TransformerActorNetwork(grid_h=grid_h, grid_w=grid_w).to(device)
        self._load_frozen_backbone()
        self.backbone.eval()

        # Actor (with cross-attention)
        self.actor = ContinuousSACActor(d_model=64).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)

        # Critic + target (with cross-attention)
        self.critic = ContinuousSACCritic(d_model=64).to(device)
        self.critic_target = ContinuousSACCritic(d_model=64).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)

        # Auto-alpha
        self.log_alpha = torch.tensor(
            np.log(INIT_ALPHA), dtype=torch.float32,
            requires_grad=True, device=device,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = TARGET_ENTROPY

        # Replay buffer (per-class balanced sampling)
        self.replay_buffer = Stage1ReplayBuffer(max_per_class=BUFFER_CAPACITY)
        self.total_steps = 0
        self.episode_count = 0

        self.try_load_model()
        atexit.register(self.save_persistent)

    def _load_frozen_backbone(self):
        """載入並 freeze backbone。"""
        if FROZEN_BACKBONE_PATH.exists():
            state_dict = torch.load(FROZEN_BACKBONE_PATH, map_location=device)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            for param in self.backbone.parameters():
                param.requires_grad = False
            transformer_missing = [k for k in missing if 'output_head' not in k]
            if transformer_missing:
                print(f"[SAC] WARNING: Transformer weights missing: {transformer_missing}")
            else:
                print(f"[SAC] Loaded & frozen backbone OK"
                      f" (transformer: all loaded, output_head: {len(missing)} skipped)")
        else:
            print(f"[SAC] WARNING: {FROZEN_BACKBONE_PATH} not found, using random backbone")
            for param in self.backbone.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _get_features(self, state):
        """State → token features (B, N, 64) from frozen backbone."""
        return self.backbone.get_features(state)  # (B, H*W, 64)

    def select_action(self, state, deterministic=False):
        """Select continuous (x, y) action.

        Returns:
            (row, col), action_np
        """
        state_batch = state.unsqueeze(0).to(device)
        features = self._get_features(state_batch)

        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(features)
            else:
                action, _ = self.actor.sample(features)
        self.actor.train()

        action_np = action.cpu().numpy().flatten()  # (2,)
        row, col = action_to_grid(action_np, self.grid_h, self.grid_w)
        return (row, col), action_np

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state, action, next_state, reward, done)

    def train_step(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_steps += 1

        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        B = state.size(0)

        features = self._get_features(state)
        next_features = self._get_features(next_state)
        alpha = self.log_alpha.exp().detach()

        # === Critic update ===
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_features)
            tq1, tq2 = self.critic_target(next_features, next_action)
            target_q = torch.min(tq1, tq2) - alpha * next_log_prob
            target = reward + (1 - done) * GAMMA * target_q

        q1, q2 = self.critic(features, action)

        critic_loss = (
            F.huber_loss(q1, target, reduction='none') +
            F.huber_loss(q2, target, reduction='none')
        ).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # === Actor update ===
        new_action, log_prob = self.actor.sample(features)
        q1_new, q2_new = self.critic(features, new_action)
        min_q = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # === Alpha update ===
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # === Soft target update ===
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "q_mean": torch.min(q1, q2).mean().item(),
            "entropy": -log_prob.mean().item(),
        }

    def on_episode_end(self):
        self.episode_count += 1

    def _save_model(self):
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), MODEL_PATH / 'actor.pth')
        torch.save(self.critic.state_dict(), MODEL_PATH / 'critic.pth')
        torch.save(self.critic_target.state_dict(), MODEL_PATH / 'critic_target.pth')
        torch.save(self.log_alpha, MODEL_PATH / 'log_alpha.pth')
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
        }, MODEL_PATH / 'optimizer_state.pth')

    def save_persistent(self):
        buf = self.replay_buffer
        if buf.size() == 0:
            return
        entries = buf.get_all_entries()
        if len(entries) > SAVE_CAPACITY:
            entries = random.sample(entries, SAVE_CAPACITY)

        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save({
            'persistent_entries': entries,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
        }, MODEL_PATH / 'training_state.pth')

        saved_rewards = defaultdict(int)
        for e in entries:
            saved_rewards[round(e['reward'], 3)] += 1
        print(f"--- save info ---------------")
        print(f"[SAC] Persistent save: {len(entries)} entries")
        print(f"[SAC] Reward distribution: {dict(saved_rewards)}")
        print(f"--- save end ---------------")

    def try_load_model(self):
        actor_path = MODEL_PATH / 'actor.pth'
        if actor_path.exists():
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=device))
                print("[SAC] Loaded Actor")
            except Exception as e:
                print(f"[SAC] Failed to load Actor: {e}")

        critic_path = MODEL_PATH / 'critic.pth'
        if critic_path.exists():
            try:
                self.critic.load_state_dict(torch.load(critic_path, map_location=device))
                print("[SAC] Loaded Critic")
            except Exception as e:
                print(f"[SAC] Failed to load Critic: {e}")

        ct_path = MODEL_PATH / 'critic_target.pth'
        if ct_path.exists():
            try:
                self.critic_target.load_state_dict(torch.load(ct_path, map_location=device))
                print("[SAC] Loaded Critic Target")
            except Exception as e:
                print(f"[SAC] Failed to load Critic Target: {e}")

        alpha_path = MODEL_PATH / 'log_alpha.pth'
        if alpha_path.exists():
            try:
                self.log_alpha = torch.load(alpha_path, map_location=device)
                self.log_alpha.requires_grad_(True)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)
                print(f"[SAC] Loaded alpha={self.log_alpha.exp().item():.4f}")
            except Exception as e:
                print(f"[SAC] Failed to load alpha: {e}")

        opt_path = MODEL_PATH / 'optimizer_state.pth'
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.actor_optimizer.load_state_dict(state['actor_optimizer'])
                self.critic_optimizer.load_state_dict(state['critic_optimizer'])
                if 'alpha_optimizer' in state:
                    self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
                self.total_steps = state['total_steps']
                self.episode_count = state.get('episode_count', 0)
                print(f"[SAC] Loaded optimizer: steps={self.total_steps},"
                      f" episodes={self.episode_count}")
            except Exception as e:
                print(f"[SAC] Failed to load optimizer: {e}")

        ts_path = MODEL_PATH / 'training_state.pth'
        if ts_path.exists():
            try:
                state = torch.load(ts_path, map_location=device, weights_only=False)
                entries = state.get('persistent_entries', [])
                loaded = min(len(entries), BUFFER_CAPACITY)
                for i in range(loaded):
                    e = entries[i]
                    self.replay_buffer.store(
                        e['state'], e['action'], e['next_state'], e['reward'], e['done']
                    )
                if loaded > 0:
                    print(f"[SAC] Loaded {loaded} replay buffer entries")
            except Exception as e:
                print(f"[SAC] Failed to load replay buffer: {e}")


# ============================================================
# Training
# ============================================================

def run_episode(logic, agent, global_steps, add_noise=True):
    logic.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False
    is_win = False
    valid_clicks = 0
    invalid_clicks = 0
    train_info_list = []

    while not done and episode_steps < MAX_STEPS_PER_EPISODE:
        state = logic.get_grid_state_tensor()

        # Warmup: random actions
        if global_steps[0] < WARMUP_STEPS and add_noise:
            action_np = np.random.uniform(0, 1, size=(2,)).astype(np.float32)
            row, col = action_to_grid(action_np, GRID_ROWS, GRID_COLS)
        else:
            (row, col), action_np = agent.select_action(
                state, deterministic=(not add_noise)
            )

        result = logic.click(row, col)
        reward = compute_reward(result)
        episode_reward += reward

        if result.changed:
            valid_clicks += 1
        else:
            invalid_clicks += 1

        done = result.game_over or result.win
        is_win = result.win
        next_state = logic.get_grid_state_tensor()

        if add_noise:
            agent.store_transition(state, action_np, next_state, reward, done)
            global_steps[0] += 1
            if global_steps[0] >= WARMUP_STEPS:
                train_info = agent.train_step()
                if train_info is not None:
                    train_info_list.append(train_info)

        episode_steps += 1

    total_clicks = valid_clicks + invalid_clicks
    invalid_rate = invalid_clicks / total_clicks if total_clicks > 0 else 0.0

    avg_critic_loss = None
    avg_actor_loss = None
    avg_q_mean = None
    avg_alpha = None
    avg_entropy = None
    if train_info_list:
        avg_critic_loss = np.mean([t['critic_loss'] for t in train_info_list])
        avg_actor_loss = np.mean([t['actor_loss'] for t in train_info_list])
        avg_q_mean = np.mean([t['q_mean'] for t in train_info_list])
        avg_alpha = np.mean([t['alpha'] for t in train_info_list])
        avg_entropy = np.mean([t['entropy'] for t in train_info_list])

    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'is_win': is_win,
        'invalid_rate': invalid_rate,
        'valid_clicks': valid_clicks,
        'invalid_clicks': invalid_clicks,
        'critic_loss': avg_critic_loss,
        'actor_loss': avg_actor_loss,
        'q_mean': avg_q_mean,
        'alpha': avg_alpha,
        'entropy': avg_entropy,
    }


def run_evaluation(logic, agent):
    eval_rewards = []
    eval_wins = 0
    eval_steps = []
    eval_invalid_rates = []

    for _ in range(EVAL_EPISODES):
        stats = run_episode(logic, agent, [WARMUP_STEPS + 1], add_noise=False)
        eval_rewards.append(stats['reward'])
        eval_steps.append(stats['steps'])
        eval_invalid_rates.append(stats['invalid_rate'])
        if stats['is_win']:
            eval_wins += 1

    return {
        'avg_reward': np.mean(eval_rewards),
        'win_rate': eval_wins / EVAL_EPISODES * 100,
        'avg_steps': np.mean(eval_steps),
        'avg_invalid_rate': np.mean(eval_invalid_rates),
    }


def main():
    print("=" * 60)
    print("  SAC Continuous: Frozen Backbone → Actor(σ) → (x,y)")
    print("=" * 60)
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}, Mines: {GRID_MINES}")
    print(f"Max episodes: {MAX_EPISODES}, Warmup: {WARMUP_STEPS} steps")
    print()

    logic = MinesweeperLogic(rows=GRID_ROWS, cols=GRID_COLS, mines_count=GRID_MINES)
    agent = SACContinuousAgent(grid_h=GRID_ROWS, grid_w=GRID_COLS)

    # TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = TENSORBOARD_DIR / timestamp
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")

    # CSV
    csv_logger = CSVLogger(CSV_LOG_PATH)
    csv_fields = [
        'episode', 'reward', 'steps', 'is_win', 'invalid_rate',
        'valid_clicks', 'invalid_clicks',
        'critic_loss', 'actor_loss', 'q_mean', 'alpha', 'entropy',
        'eval_avg_reward', 'eval_win_rate', 'eval_avg_steps', 'eval_avg_invalid_rate',
        'timestamp',
    ]
    csv_logger.open(csv_fields)
    print(f"CSV: {CSV_LOG_PATH}")
    print()

    recent_rewards = deque(maxlen=LOG_INTERVAL)
    recent_wins = deque(maxlen=LOG_INTERVAL)
    recent_steps = deque(maxlen=LOG_INTERVAL)
    total_wins = 0
    global_steps = [0]  # mutable for pass-by-reference
    start_time = time.time()

    try:
        for episode in range(1, MAX_EPISODES + 1):
            stats = run_episode(logic, agent, global_steps, add_noise=True)
            agent.on_episode_end()

            if stats['is_win']:
                total_wins += 1

            recent_rewards.append(stats['reward'])
            recent_wins.append(1 if stats['is_win'] else 0)
            recent_steps.append(stats['steps'])

            # TensorBoard
            writer.add_scalar('train/episode_reward', stats['reward'], episode)
            writer.add_scalar('train/episode_steps', stats['steps'], episode)
            if stats['critic_loss'] is not None:
                writer.add_scalar('train/critic_loss', stats['critic_loss'], episode)
                writer.add_scalar('train/actor_loss', stats['actor_loss'], episode)
                writer.add_scalar('train/q_mean', stats['q_mean'], episode)
                writer.add_scalar('train/alpha', stats['alpha'], episode)
                writer.add_scalar('train/entropy', stats['entropy'], episode)

            # CSV
            csv_row = {
                'episode': episode,
                'reward': f"{stats['reward']:.2f}",
                'steps': stats['steps'],
                'is_win': int(stats['is_win']),
                'invalid_rate': f"{stats['invalid_rate']:.4f}",
                'valid_clicks': stats['valid_clicks'],
                'invalid_clicks': stats['invalid_clicks'],
                'critic_loss': f"{stats['critic_loss']:.6f}" if stats['critic_loss'] is not None else '',
                'actor_loss': f"{stats['actor_loss']:.6f}" if stats['actor_loss'] is not None else '',
                'q_mean': f"{stats['q_mean']:.4f}" if stats['q_mean'] is not None else '',
                'alpha': f"{stats['alpha']:.4f}" if stats['alpha'] is not None else '',
                'entropy': f"{stats['entropy']:.4f}" if stats['entropy'] is not None else '',
                'eval_avg_reward': '',
                'eval_win_rate': '',
                'eval_avg_steps': '',
                'eval_avg_invalid_rate': '',
                'timestamp': datetime.datetime.now().isoformat(),
            }

            # Eval
            if episode % EVAL_INTERVAL == 0:
                eval_stats = run_evaluation(logic, agent)
                writer.add_scalar('eval/avg_reward', eval_stats['avg_reward'], episode)
                writer.add_scalar('eval/win_rate', eval_stats['win_rate'], episode)

                csv_row['eval_avg_reward'] = f"{eval_stats['avg_reward']:.2f}"
                csv_row['eval_win_rate'] = f"{eval_stats['win_rate']:.1f}"
                csv_row['eval_avg_steps'] = f"{eval_stats['avg_steps']:.1f}"
                csv_row['eval_avg_invalid_rate'] = f"{eval_stats['avg_invalid_rate']:.4f}"

                print(f"  [EVAL Ep {episode:>6d}] "
                      f"Avg Reward: {eval_stats['avg_reward']:>7.2f} | "
                      f"Win Rate: {eval_stats['win_rate']:>5.1f}% | "
                      f"Avg Steps: {eval_stats['avg_steps']:>5.1f}")

            csv_logger.write(csv_row)

            # Console log
            if episode % LOG_INTERVAL == 0:
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins) * 100
                avg_steps = np.mean(recent_steps)
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed
                overall_wr = total_wins / episode * 100
                alpha = agent.log_alpha.exp().item()

                writer.add_scalar('train/avg_reward_50', avg_reward, episode)
                writer.add_scalar('train/win_rate_50', win_rate, episode)

                print(f"[Ep {episode:>6d}] "
                      f"Avg Reward: {avg_reward:>7.2f} | "
                      f"WR(50): {win_rate:>5.1f}% | "
                      f"Overall WR: {overall_wr:>5.1f}% | "
                      f"Wins: {total_wins} | "
                      f"α: {alpha:.4f} | "
                      f"Speed: {eps_per_sec:.1f} ep/s")

            # Save
            if episode % SAVE_INTERVAL == 0:
                agent._save_model()

        # Training complete
        print()
        print("=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        elapsed = time.time() - start_time
        print(f"Total episodes: {MAX_EPISODES}, Total wins: {total_wins}"
              f" ({total_wins/MAX_EPISODES*100:.1f}%)")
        print(f"Total time: {elapsed:.1f}s ({MAX_EPISODES/elapsed:.1f} ep/s)")

        final_eval = run_evaluation(logic, agent)
        print(f"Final eval: Win Rate={final_eval['win_rate']:.1f}%,"
              f" Avg Reward={final_eval['avg_reward']:.2f}")

        agent._save_model()
        agent.save_persistent()

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted")
        agent._save_model()
        agent.save_persistent()

    finally:
        csv_logger.close()
        writer.close()


if __name__ == "__main__":
    main()
