import atexit
import datetime
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer_shared import DuelingQNetwork, EncoderDecoderTransformer, TwoDimensionalPositionEmbedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.9
GRID_STATE_CHANNELS = 12
LR_DDQN = 5e-5
PER_CAPACITY = 10000
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
SAVE_CAPACITY = 2000
SAVE_EVERY_N_EPISODES = 50
TARGET_UPDATE_FREQ = 50

TRANSFORMER_MODEL_PATH = Path("./models/stage1_transformer")
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_FF_DIM = 256
TRANSFORMER_DROPOUT = 0.1


class TransformerActorNetwork(nn.Module):
    """Grid state -> encoder-decoder transformer -> per-cell logits."""

    def __init__(
        self,
        grid_channels=GRID_STATE_CHANNELS,
        grid_h=10,
        grid_w=10,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_FF_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_tokens = grid_h * grid_w

        self.token_embed = nn.Linear(grid_channels, d_model)
        self.position = TwoDimensionalPositionEmbedding(grid_h, grid_w, d_model)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        self.core = EncoderDecoderTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_head = nn.Linear(d_model, 1)

    def _embed(self, state):
        tokens = state.permute(0, 2, 3, 1).reshape(state.size(0), self.num_tokens, -1)
        x = self.token_embed(tokens)
        return x + self.position().unsqueeze(0)

    def _build_queries(self, batch_size):
        return self.query_tokens.expand(batch_size, -1, -1) + self.position().unsqueeze(0)

    def get_memory(self, state):
        return self.core.encode(self._embed(state))

    def get_features(self, state):
        memory = self.get_memory(state)
        return self.core.decode(self._build_queries(state.size(0)), memory)

    def forward(self, state):
        x = self.get_features(state)
        logits = self.output_head(x).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def load_backbone_state(self, state_dict, strict=False):
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("core.") or key.startswith("position.") or key.startswith("token_embed.") or key.startswith("output_head.") or key == "query_tokens":
                normalized[key] = value
            elif key.startswith("transformer."):
                normalized[f"core.transformer.{key[len('transformer.') :]}"] = value
            elif key.startswith("decoder."):
                normalized[f"core.decoder.{key[len('decoder.') :]}"] = value
            elif key.startswith("row_embed."):
                normalized[f"position.{key}"] = value
            elif key.startswith("col_embed."):
                normalized[f"position.{key}"] = value
            elif key in {"row_indices", "col_indices"}:
                normalized[f"position.{key}"] = value

        return self.load_state_dict(normalized, strict=strict)


class TransformerDiscreteAgent:
    """Dueling DDQN agent with grid encoder-decoder transformer backbone."""

    def __init__(self, grid_h=10, grid_w=10):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w

        self.backbone = TransformerActorNetwork(grid_h=grid_h, grid_w=grid_w).to(device)
        self.q_network = DuelingQNetwork(
            d_model=TRANSFORMER_D_MODEL,
            grid_h=grid_h,
            grid_w=grid_w,
        ).to(device)
        self.q_target = DuelingQNetwork(
            d_model=TRANSFORMER_D_MODEL,
            grid_h=grid_h,
            grid_w=grid_w,
        ).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()

        from RL_Agent import PERReplayBuffer

        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) + list(self.q_network.parameters()),
            lr=LR_DDQN,
        )

        self.replay_buffer = PERReplayBuffer(capacity=PER_CAPACITY, alpha=PER_ALPHA)
        self.total_it = 0
        self.episode_count = 0
        self.beta = PER_BETA_START

        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay_episodes = 5000

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self._io_log = open(TRANSFORMER_MODEL_PATH / "train_io_log.txt", "a", encoding="utf-8")
        self._io_log.write(f"\n{'=' * 60}\n")
        self._io_log.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self._io_log.write(f"{'=' * 60}\n")
        self._io_log.flush()

        self.try_load_model()
        atexit.register(self.save_persistent)
        atexit.register(self._close_io_log)

    def select_action(self, state, add_noise=True):
        if add_noise and random.random() < self.epsilon:
            row = random.randint(0, self.grid_h - 1)
            col = random.randint(0, self.grid_w - 1)
            return (row, col)

        state_batch = state.unsqueeze(0).to(device)

        self.backbone.eval()
        self.q_network.eval()
        with torch.no_grad():
            features = self.backbone.get_features(state_batch)
            q_2d = self.q_network(features).squeeze(0)
            row_q = q_2d.max(dim=1).values
            row = row_q.argmax().item()
            col = q_2d[row].argmax().item()
        self.backbone.train()
        self.q_network.train()

        return (row, col)

    def store_transition(self, state, action, next_state, reward, done):
        action_arr = np.array(action, dtype=np.int64)
        self.replay_buffer.store(state, action_arr, next_state, reward, done)

    def train_step(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_it += 1
        state, action, next_state, reward, done, per_indices, is_weights = self.replay_buffer.sample(
            BATCH_SIZE,
            beta=self.beta,
        )

        action = action.long()
        row_idx = action[:, 0]
        col_idx = action[:, 1]
        batch_size = state.size(0)

        with torch.no_grad():
            next_features = self.backbone.get_features(next_state)
            next_q_2d = self.q_network(next_features)
            next_q_flat = next_q_2d.view(batch_size, -1)
            best_flat = next_q_flat.argmax(dim=1)
            best_rows = best_flat // self.grid_w
            best_cols = best_flat % self.grid_w

            next_q_target_2d = self.q_target(next_features)
            next_q_value = next_q_target_2d[
                torch.arange(batch_size, device=device), best_rows, best_cols
            ].unsqueeze(1)
            target = reward + (1 - done) * GAMMA * next_q_value

        features = self.backbone.get_features(state)
        q_2d = self.q_network(features)
        q_taken = q_2d[
            torch.arange(batch_size, device=device), row_idx, col_idx
        ].unsqueeze(1)

        td_error = (q_taken - target).abs().detach()
        per_sample_loss = F.huber_loss(q_taken, target, reduction="none")
        loss = (is_weights * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.q_network.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        self.replay_buffer.update_priorities(per_indices, td_error.squeeze(-1).cpu().numpy())

        if self.total_it % TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

        with torch.no_grad():
            reward_counts = defaultdict(int)
            for value in reward.squeeze(-1).tolist():
                reward_counts[value] += 1

            s0 = state[0]
            unrevealed = s0[0].sum().int().item()
            flagged = s0[1].sum().int().item()
            total_cells = self.grid_h * self.grid_w
            revealed = total_cells - unrevealed - flagged
            num_counts = {}
            for ch in range(2, 11):
                count = s0[ch].sum().int().item()
                if count > 0:
                    num_counts[ch - 2] = count

            q0_flat = q_2d[0].view(-1)
            top5_vals, top5_flat = q0_flat.topk(5)
            top5_info = [
                (idx.item() // self.grid_w, idx.item() % self.grid_w, f"{val.item():.4f}")
                for val, idx in zip(top5_vals, top5_flat)
            ]

            action_list = list(zip(row_idx.tolist(), col_idx.tolist()))
            action_freq = defaultdict(int)
            for item in action_list:
                action_freq[item] += 1
            top3_actions = sorted(action_freq.items(), key=lambda item: -item[1])[:3]
            top3_str = ", ".join(f"({row},{col})x{count}" for (row, col), count in top3_actions)

            q_mean = q_taken.mean().item()

        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  State:  unrevealed={unrevealed} | revealed={revealed} | flagged={flagged}"
            f" | numbers={num_counts}\n"
            f"  Batch:  rewards={dict(reward_counts)} | top_actions=[{top3_str}]\n"
            f"  Q-top5: {top5_info}\n"
            f"  Q-val:  taken_mean={q_mean:.4f}"
            f" | all: min={q0_flat.min().item():.4f} max={q0_flat.max().item():.4f}\n"
            f"  Loss:   {loss.item():.4f} | epsilon={self.epsilon:.4f}\n"
            f"---\n"
        )
        self._io_log.flush()

        return {
            "loss": loss.item(),
            "q_mean": q_mean,
        }

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def on_episode_end(self):
        self.episode_count += 1
        decay_progress = min(self.episode_count / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_min + (0.3 - self.epsilon_min) * (1.0 - decay_progress)
        beta_progress = min(self.episode_count / 5000.0, 1.0)
        self.beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * beta_progress
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(
                f"[DDQN] Periodic save at episode {self.episode_count}"
                f" | epsilon={self.epsilon:.4f} | beta={self.beta:.4f}"
            )
            self.save_persistent()

    def _save_model(self):
        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(), TRANSFORMER_MODEL_PATH / "backbone.pth")
        torch.save(self.q_network.state_dict(), TRANSFORMER_MODEL_PATH / "q_network.pth")
        torch.save(self.q_target.state_dict(), TRANSFORMER_MODEL_PATH / "q_target.pth")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
                "beta": self.beta,
            },
            TRANSFORMER_MODEL_PATH / "optimizer_state.pth",
        )

    def save_persistent(self):
        buf = self.replay_buffer
        total = buf.size()
        if total == 0:
            return

        if total <= SAVE_CAPACITY:
            all_entries = buf.get_all_entries()
        else:
            all_entries = buf.get_top_entries(SAVE_CAPACITY)

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "persistent_entries": all_entries,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
            },
            TRANSFORMER_MODEL_PATH / "training_state.pth",
        )

        saved_rewards = defaultdict(int)
        for entry in all_entries:
            saved_rewards[entry["reward"]] += 1
        print("--- save info ---------------")
        print(f"[DDQN] Persistent save: {len(all_entries)} entries")
        print(f"[DDQN] Reward distribution: {dict(saved_rewards)}")
        print("--- save end ---------------")

    def try_load_model(self):
        backbone_path = TRANSFORMER_MODEL_PATH / "backbone.pth"
        if backbone_path.exists():
            try:
                backbone_state = torch.load(backbone_path, map_location=device)
                incompatible = self.backbone.load_backbone_state(backbone_state, strict=False)
                print("[DDQN] Loaded Backbone")
                if incompatible.missing_keys:
                    print(f"[DDQN] Backbone missing keys: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    print(f"[DDQN] Backbone unexpected keys: {incompatible.unexpected_keys}")
            except Exception as exc:
                print(f"[DDQN] Failed to load Backbone: {exc}")

        q_path = TRANSFORMER_MODEL_PATH / "q_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device))
                print("[DDQN] Loaded Q-Network")
            except Exception as exc:
                print(f"[DDQN] Failed to load Q-Network: {exc}")

        q_target_path = TRANSFORMER_MODEL_PATH / "q_target.pth"
        if q_target_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(q_target_path, map_location=device))
                print("[DDQN] Loaded Q-Target")
            except Exception as exc:
                print(f"[DDQN] Failed to load Q-Target: {exc}")

        opt_path = TRANSFORMER_MODEL_PATH / "optimizer_state.pth"
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.optimizer.load_state_dict(state["optimizer"])
                self.total_it = state["total_it"]
                self.episode_count = state.get("episode_count", 0)
                if "epsilon" in state:
                    self.epsilon = state["epsilon"]
                if "beta" in state:
                    self.beta = state["beta"]
                print(
                    f"[DDQN] Loaded optimizer: total_it={self.total_it},"
                    f" episode={self.episode_count}, epsilon={self.epsilon:.4f},"
                    f" beta={self.beta:.4f}"
                )
            except Exception as exc:
                print(f"[DDQN] Failed to load optimizer state: {exc}")

        training_state_path = TRANSFORMER_MODEL_PATH / "training_state.pth"
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                persistent_entries = state.get("persistent_entries", [])
                loaded = min(len(persistent_entries), PER_CAPACITY)
                for idx in range(loaded):
                    entry = persistent_entries[idx]
                    self.replay_buffer.store(
                        entry["state"],
                        entry["action"],
                        entry["next_state"],
                        entry["reward"],
                        entry["done"],
                    )
                if loaded > 0:
                    print(f"[DDQN] Loaded {loaded} replay buffer entries")
            except Exception as exc:
                print(f"[DDQN] Failed to load replay buffer: {exc}")
