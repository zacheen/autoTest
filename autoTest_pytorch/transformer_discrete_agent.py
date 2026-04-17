import atexit
import datetime
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_structure.transformer_shared import (
    EncoderDecoderTransformer,
    IQNQNetwork,
    TwoDimensionalPositionEmbedding,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.7
GRID_STATE_CHANNELS = 12
LR_DDQN = 5e-5
PER_CAPACITY = 10000
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
SAVE_CAPACITY = 2000
SAVE_EVERY_N_EPISODES = 50
TARGET_UPDATE_FREQ = 50
N_STEP = 1
NUM_IQN_POLICY_QUANTILES = 16
NUM_IQN_QUANTILES = 16
NUM_IQN_TARGET_QUANTILES = 16

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


def _quantile_huber_loss(current_quantiles, target_quantiles, tau_hats):
    td = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
    abs_td = td.abs()
    huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
    tau = tau_hats.unsqueeze(2)
    quantile_weight = (tau - (td.detach() < 0).float()).abs()
    return (quantile_weight * huber).sum(dim=2).mean(dim=1, keepdim=True)


class TransformerDiscreteAgent:
    """IQN agent with grid encoder-decoder transformer backbone."""

    def __init__(self, grid_h=10, grid_w=10):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w

        self.backbone = TransformerActorNetwork(grid_h=grid_h, grid_w=grid_w).to(device)
        self.q_network = IQNQNetwork(
            d_model=TRANSFORMER_D_MODEL,
            grid_h=grid_h,
            grid_w=grid_w,
            num_quantiles=NUM_IQN_QUANTILES,
        ).to(device)
        self.q_target = IQNQNetwork(
            d_model=TRANSFORMER_D_MODEL,
            grid_h=grid_h,
            grid_w=grid_w,
            num_quantiles=NUM_IQN_TARGET_QUANTILES,
        ).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()

        from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer

        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) + list(self.q_network.parameters()),
            lr=LR_DDQN,
        )

        self.replay_buffer = CategorizedReplayBuffer(
            max_size=PER_CAPACITY,
            storage_mode="ram",
            win_threshold=3.0,
            lose_threshold=-1.0, 
            invalid_threshold=0.0,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START
        )
        self.total_it = 0
        self.episode_count = 0
        self.n_step = N_STEP
        self.n_step_gamma = GAMMA
        self.n_step_buffer = deque()

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

    def _sample_iqn_taus(self, batch_size, num_quantiles):
        return self.q_network.sample_taus(
            batch_size=batch_size,
            num_quantiles=num_quantiles,
            device=device,
            dtype=torch.float32,
        )

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
            policy_taus = self._sample_iqn_taus(
                batch_size=state_batch.size(0),
                num_quantiles=NUM_IQN_POLICY_QUANTILES,
            )
            q_2d = self.q_network(features, taus=policy_taus)["q_values"].squeeze(0)
            row_q = q_2d.max(dim=1).values
            row = row_q.argmax().item()
            col = q_2d[row].argmax().item()
        self.backbone.train()
        self.q_network.train()

        return (row, col)

    def store_transition(self, state, action, next_state, reward, done):
        transition = {
            "state": state.detach().cpu(),
            "action": np.array(action, dtype=np.int64),
            "next_state": next_state.detach().cpu() if next_state is not None else None,
            "reward": float(reward),
            "done": bool(done),
        }
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) >= self.n_step:
            self._commit_n_step_transition(self.n_step)

        if done:
            self._flush_n_step_buffer()

    def _commit_n_step_transition(self, horizon):
        if not self.n_step_buffer:
            return

        horizon = min(horizon, len(self.n_step_buffer))
        discounted_reward = 0.0
        last_transition = None
        for step_idx in range(horizon):
            transition = self.n_step_buffer[step_idx]
            discounted_reward += (self.n_step_gamma ** step_idx) * transition["reward"]
            last_transition = transition
            if transition["done"]:
                horizon = step_idx + 1
                break

        first_transition = self.n_step_buffer[0]
        discount = self.n_step_gamma ** horizon
        self.replay_buffer.store(
            first_transition["state"],
            first_transition["action"],
            last_transition["next_state"],
            discounted_reward,
            last_transition["done"],
            discount=discount,
            n_steps=horizon,
            tail_reward=last_transition["reward"],
        )
        self.n_step_buffer.popleft()

    def _flush_n_step_buffer(self):
        while self.n_step_buffer:
            self._commit_n_step_transition(len(self.n_step_buffer))

    def train_step(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return None

        self.total_it += 1
        
        # CategorizedReplayBuffer stores beta internally if not provided, but we can still pass it explicitly
        state, action, next_state, reward, done, per_indices, is_weights, discounts, n_steps = self.replay_buffer.sample(
            BATCH_SIZE,
            beta=PER_BETA_START + (PER_BETA_END - PER_BETA_START) * min(self.episode_count / 5000.0, 1.0),
            device=device,
            include_extra=True,
        )

        action = action.long()
        row_idx = action[:, 0]
        col_idx = action[:, 1]
        action_flat = row_idx * self.grid_w + col_idx
        batch_size = state.size(0)

        with torch.no_grad():
            next_features = self.backbone.get_features(next_state)
            next_policy_taus = self._sample_iqn_taus(
                batch_size=batch_size,
                num_quantiles=NUM_IQN_POLICY_QUANTILES,
            )
            next_online = self.q_network(next_features, taus=next_policy_taus)
            next_q_2d = next_online["q_values"]
            next_q_flat = next_q_2d.view(batch_size, -1)
            best_flat = next_q_flat.argmax(dim=1)

            target_taus = self.q_target.sample_taus(
                batch_size=batch_size,
                num_quantiles=NUM_IQN_TARGET_QUANTILES,
                device=device,
                dtype=next_features.dtype,
            )
            next_target = self.q_target(next_features, taus=target_taus)
            next_target_quantiles = next_target["quantiles"][
                torch.arange(batch_size, device=device), best_flat
            ]
            target_quantiles = reward + (1 - done) * discounts * next_target_quantiles

        features = self.backbone.get_features(state)
        current_taus = self._sample_iqn_taus(
            batch_size=batch_size,
            num_quantiles=NUM_IQN_QUANTILES,
        )
        q_output = self.q_network(features, taus=current_taus)
        q_2d = q_output["q_values"]
        q_quantiles = q_output["quantiles"]
        sampled_taus = q_output["taus"]
        q_taken = q_2d[
            torch.arange(batch_size, device=device), row_idx, col_idx
        ].unsqueeze(1)
        chosen_quantiles = q_quantiles[
            torch.arange(batch_size, device=device), action_flat
        ]

        with torch.no_grad():
            target_mean = target_quantiles.mean(dim=1, keepdim=True)
            td_error = (q_taken - target_mean).abs().detach()

        per_sample_quantile_loss = _quantile_huber_loss(
            current_quantiles=chosen_quantiles,
            target_quantiles=target_quantiles.detach(),
            tau_hats=sampled_taus.detach(),
        )
        loss = (is_weights * per_sample_quantile_loss).mean()

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
            f"  IQN:    loss={loss.item():.4f} | quantiles={NUM_IQN_QUANTILES}"
            f" | epsilon={self.epsilon:.4f}\n"
            f"---\n"
        )
        self._io_log.flush()

        return {
            "Q_loss": loss.item(),
            "q_mean": q_mean,
        }

    def reset_episode(self):
        self._flush_n_step_buffer()

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def on_episode_end(self):
        self._flush_n_step_buffer()
        self.episode_count += 1
        decay_progress = min(self.episode_count / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_min + (0.3 - self.epsilon_min) * (1.0 - decay_progress)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(
                f"[IQN] Periodic save at episode {self.episode_count}"
                f" | epsilon={self.epsilon:.4f}"
            )
            self.save_persistent()

    def _save_model(self):
        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(), TRANSFORMER_MODEL_PATH / "backbone.pth")
        torch.save(self.q_network.state_dict(), TRANSFORMER_MODEL_PATH / "iqn_network.pth")
        torch.save(self.q_target.state_dict(), TRANSFORMER_MODEL_PATH / "iqn_target.pth")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
                "algorithm": "IQN",
            },
            TRANSFORMER_MODEL_PATH / "optimizer_state.pth",
        )

    def save_persistent(self):
        buf = self.replay_buffer
        total = buf.size()
        if total == 0:
            return

        all_entries = buf.get_all_entries()
        
        # Sort by priority locally to do Top-K before saving
        all_entries_sorted = sorted(
            all_entries, 
            key=lambda e: buf._effective_priority(e), 
            reverse=True
        )
        
        if len(all_entries_sorted) > SAVE_CAPACITY:
            all_entries_sorted = all_entries_sorted[:SAVE_CAPACITY]

        TRANSFORMER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "persistent_entries": all_entries_sorted,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
            },
            TRANSFORMER_MODEL_PATH / "training_state.pth",
        )

        saved_rewards = defaultdict(int)
        saved_buckets = defaultdict(int)
        for entry in all_entries_sorted:
            saved_rewards[entry.get("tail_reward", entry["reward"])] += 1
            saved_buckets[entry["reward_type"]] += 1
        print("--- save info ---------------")
        print(f"[IQN] Persistent save: {len(all_entries_sorted)} entries")
        print(f"[IQN] Tail reward distribution: {dict(saved_rewards)}")
        print(f"[IQN] Reward bucket distribution: {dict(saved_buckets)}")
        print("--- save end ---------------")

    def try_load_model(self):
        backbone_path = TRANSFORMER_MODEL_PATH / "backbone.pth"
        if backbone_path.exists():
            try:
                backbone_state = torch.load(backbone_path, map_location=device)
                incompatible = self.backbone.load_backbone_state(backbone_state, strict=False)
                print("[IQN] Loaded Backbone")
                if incompatible.missing_keys:
                    print(f"[IQN] Backbone missing keys: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    print(f"[IQN] Backbone unexpected keys: {incompatible.unexpected_keys}")
            except Exception as exc:
                print(f"[IQN] Failed to load Backbone: {exc}")

        q_path = TRANSFORMER_MODEL_PATH / "iqn_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device))
                print("[IQN] Loaded IQN-Network")
            except Exception as exc:
                print(f"[IQN] Failed to load IQN-Network: {exc}")
        elif (TRANSFORMER_MODEL_PATH / "fqf_network.pth").exists():
            print("[IQN] Skip legacy fqf_network.pth because FQF head shape is incompatible")
        elif (TRANSFORMER_MODEL_PATH / "q_network.pth").exists():
            print("[IQN] Skip legacy q_network.pth because DDQN head shape is incompatible")

        q_target_path = TRANSFORMER_MODEL_PATH / "iqn_target.pth"
        if q_target_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(q_target_path, map_location=device))
                print("[IQN] Loaded IQN-Target")
            except Exception as exc:
                print(f"[IQN] Failed to load IQN-Target: {exc}")
        elif (TRANSFORMER_MODEL_PATH / "fqf_target.pth").exists():
            print("[IQN] Skip legacy fqf_target.pth because FQF head shape is incompatible")
        elif (TRANSFORMER_MODEL_PATH / "q_target.pth").exists():
            print("[IQN] Skip legacy q_target.pth because DDQN head shape is incompatible")

        opt_path = TRANSFORMER_MODEL_PATH / "optimizer_state.pth"
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.optimizer.load_state_dict(state["optimizer"])
                self.total_it = state["total_it"]
                self.episode_count = state.get("episode_count", 0)
                if "epsilon" in state:
                    self.epsilon = state["epsilon"]
                print(
                    f"[IQN] Loaded optimizer: total_it={self.total_it},"
                    f" episode={self.episode_count}, epsilon={self.epsilon:.4f}"
                )
            except Exception as exc:
                print(f"[IQN] Failed to load optimizer state: {exc}")

        training_state_path = TRANSFORMER_MODEL_PATH / "training_state.pth"
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                persistent_entries = state.get("persistent_entries", [])
                
                # Check format to cleanly transition if an old save has a different shape
                if persistent_entries and isinstance(persistent_entries[0], dict) and "storage_id" in persistent_entries[0]:
                    # Exactly load using new format
                    self.replay_buffer.load_from_entries(persistent_entries)
                    loaded = len(persistent_entries)
                else:
                    # Legacy transition format
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
                    print(f"[IQN] Loaded {loaded} replay buffer entries")
            except Exception as exc:
                print(f"[IQN] Failed to load replay buffer: {exc}")
