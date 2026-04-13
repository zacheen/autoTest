import atexit
import datetime
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from transformer_discrete_agent import TRANSFORMER_MODEL_PATH
from model_structure.transformer_shared import DuelingQNetwork, EncoderDecoderTransformer, TwoDimensionalPositionEmbedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (640, 640)
LOG_ACTIONS = True
ACTION_LOG_PATH = Path("./models/action_logs")
SAVE_EVERY_N_EPISODES = 50
TARGET_UPDATE_FREQ = 50
YOLO_LAST_LAYER_IDX = 6
YOLO_LAST_CHANNELS = 128
YOLO_LAST_FEATURE_SIZE = 40

VISUAL_GRID_H = 6
VISUAL_GRID_W = 6
VISUAL_NUM_ACTIONS = VISUAL_GRID_H * VISUAL_GRID_W
VISUAL_BATCH_SIZE = 32
VISUAL_D_MODEL = 64
VISUAL_NHEAD = 4
VISUAL_BACKBONE_LAYERS = 4
VISUAL_FF_DIM = 256
VISUAL_DROPOUT = 0.1
VISUAL_GAMMA = 0.9
LR_VISUAL_YOLO = 1e-5
LR_VISUAL_POLICY = 5e-5
VISUAL_BUFFER_CAPACITY = 2048
VISUAL_SAVE_CAPACITY = 256
VISUAL_BUFFER_OVERFLOW = 256
VISUAL_PER_ALPHA = 0.6
VISUAL_PER_UNIFORM_MIX = 0.2
VISUAL_PRIORITY_MIN = 0.05
VISUAL_PRIORITY_MAX = 5.0
VISUAL_PRIORITY_EPS = 1e-3
VISUAL_AGE_DECAY = 0.002
VISUAL_HISTOGRAM_EVERY = 20
VISUAL_MODEL_PATH = Path("./models/visual_transformer_6x6")
VISUAL_REPLAY_PATH = VISUAL_MODEL_PATH / "replay_buffer"
VISUAL_REPLAY_PERSISTENT_PATH = VISUAL_MODEL_PATH / "replay_buffer_save"
VISUAL_TENSORBOARD_DIR = VISUAL_MODEL_PATH / "tensorboard"


class YOLO11nLastFeatureExtractor(nn.Module):
    """Return the last YOLO11n backbone feature map (40x40)."""

    def __init__(self, model_path="yolo11n.pt"):
        super().__init__()
        yolo = YOLO(model_path)
        self.backbone_layers = nn.ModuleList(
            yolo.model.model[idx] for idx in range(YOLO_LAST_LAYER_IDX + 1)
        )
        for param in self.backbone_layers.parameters():
            param.requires_grad_(True)
        self._verify_shapes()

    def forward(self, x):
        for layer in self.backbone_layers:
            x = layer(x)
        return x

    def _verify_shapes(self):
        dummy = torch.randn(1, 3, *IMAGE_SIZE)
        with torch.no_grad():
            features = self.forward(dummy)

        _, channels, height, width = features.shape
        assert channels == YOLO_LAST_CHANNELS, (
            f"Expected {YOLO_LAST_CHANNELS} channels, got {channels}"
        )
        assert height == YOLO_LAST_FEATURE_SIZE and width == YOLO_LAST_FEATURE_SIZE, (
            f"Expected {YOLO_LAST_FEATURE_SIZE}x{YOLO_LAST_FEATURE_SIZE}, got {height}x{width}"
        )


class VisualTransformerBackbone(nn.Module):
    """YOLO memory tokens -> frozen teacher transformer -> 6x6 query features."""

    def __init__(
        self,
        grid_h=VISUAL_GRID_H,
        grid_w=VISUAL_GRID_W,
        d_model=VISUAL_D_MODEL,
        nhead=VISUAL_NHEAD,
        num_layers=VISUAL_BACKBONE_LAYERS,
        dim_feedforward=VISUAL_FF_DIM,
        dropout=VISUAL_DROPOUT,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_queries = grid_h * grid_w
        self.memory_h = YOLO_LAST_FEATURE_SIZE
        self.memory_w = YOLO_LAST_FEATURE_SIZE
        self.num_memory_tokens = self.memory_h * self.memory_w

        self.feature_extractor = YOLO11nLastFeatureExtractor()
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(YOLO_LAST_CHANNELS),
            nn.Linear(YOLO_LAST_CHANNELS, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.memory_position = TwoDimensionalPositionEmbedding(self.memory_h, self.memory_w, d_model)
        self.query_position = TwoDimensionalPositionEmbedding(grid_h, grid_w, d_model)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, d_model) * 0.02)
        self.core = EncoderDecoderTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def _embed(self, state):
        features = self.feature_extractor(state)
        tokens = features.permute(0, 2, 3, 1).reshape(state.size(0), self.num_memory_tokens, YOLO_LAST_CHANNELS)
        x = self.token_adapter(tokens)
        return x + self.memory_position().unsqueeze(0)

    def _build_queries(self, batch_size):
        return self.query_tokens.expand(batch_size, -1, -1) + self.query_position().unsqueeze(0)

    def get_memory(self, state):
        return self.core.encode(self._embed(state))

    def get_features(self, state):
        memory = self.get_memory(state)
        return self.core.decode(self._build_queries(state.size(0)), memory)

    def forward(self, state):
        return self.get_features(state)

    def yolo_parameters(self):
        return list(self.feature_extractor.parameters())

    def adapter_parameters(self):
        return (
            list(self.token_adapter.parameters())
            + list(self.memory_position.parameters())
            + list(self.query_position.parameters())
            + [self.query_tokens]
        )

    def trainable_parameters(self):
        return self.yolo_parameters() + self.adapter_parameters()

    def load_transformer_backbone_weights(self, transformer_backbone_state):
        loaded = []
        encoder_state = {}
        decoder_state = {}
        for key, value in transformer_backbone_state.items():
            if key.startswith("core.transformer."):
                encoder_state[key[len("core.transformer."):]] = value
            elif key.startswith("transformer."):
                encoder_state[key[len("transformer."):]] = value
            elif key.startswith("core.decoder."):
                decoder_state[key[len("core.decoder."):]] = value
            elif key.startswith("decoder."):
                decoder_state[key[len("decoder."):]] = value

        if encoder_state:
            self.core.transformer.load_state_dict(encoder_state, strict=False)
            loaded.append("core.transformer")
        if decoder_state:
            self.core.decoder.load_state_dict(decoder_state, strict=False)
            loaded.append("core.decoder")

        return loaded



class VisualDiscreteAgent:
    """Visual DDQN agent: screenshot -> YOLO tokens -> frozen teacher core -> 36-class click."""

    def __init__(self, screen_region):
        self.screen_region = screen_region
        self.grid_h = VISUAL_GRID_H
        self.grid_w = VISUAL_GRID_W
        self.num_actions = VISUAL_NUM_ACTIONS

        self.backbone = VisualTransformerBackbone(grid_h=self.grid_h, grid_w=self.grid_w).to(device)
        self.q_network = DuelingQNetwork(d_model=VISUAL_D_MODEL, grid_h=self.grid_h, grid_w=self.grid_w).to(device)
        self.q_target = DuelingQNetwork(d_model=VISUAL_D_MODEL, grid_h=self.grid_h, grid_w=self.grid_w).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()
        self._load_frozen_transformer_teacher()
        self._freeze_teacher_modules()

        self.optimizer = optim.AdamW([
            {"params": self.backbone.yolo_parameters(), "lr": LR_VISUAL_YOLO},
            {"params": self.backbone.adapter_parameters(), "lr": LR_VISUAL_POLICY},
        ])
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer

        self.replay_buffer = CategorizedReplayBuffer(
            max_size=VISUAL_BUFFER_CAPACITY,
            storage_mode="disk",
            save_dir=VISUAL_REPLAY_PATH,
            win_threshold=3.0,
            lose_threshold=-1.0, 
            invalid_threshold=0.0,
            overflow_margin=VISUAL_BUFFER_OVERFLOW,
            alpha=VISUAL_PER_ALPHA,
            uniform_mix=VISUAL_PER_UNIFORM_MIX,
            priority_min=VISUAL_PRIORITY_MIN,
            priority_max=VISUAL_PRIORITY_MAX,
            priority_eps=VISUAL_PRIORITY_EPS,
            age_decay=VISUAL_AGE_DECAY
        )
        self.total_it = 0
        self.episode_count = 0
        self.epsilon = 0.30
        self.epsilon_min = 0.05
        self.epsilon_decay_episodes = 5000
        self.current_state_key = None
        self.blocked_actions_current_state = set()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        VISUAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        VISUAL_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self._io_log = open(VISUAL_MODEL_PATH / "train_io_log.txt", "a", encoding="utf-8")
        self._io_log.write(f"\n{'=' * 60}\n")
        self._io_log.write(f"Session started: {datetime.datetime.now().isoformat()}\n")
        self._io_log.write(f"{'=' * 60}\n")
        self._io_log.flush()

        tb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log_dir = VISUAL_TENSORBOARD_DIR / tb_timestamp
        self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))
        print(f"[VisualDDQN] TensorBoard: tensorboard --logdir {VISUAL_TENSORBOARD_DIR}")
        print(f"[VisualDDQN] Current run: {self.tensorboard_log_dir}")

        self.try_load_model()
        self._set_runtime_modes()
        atexit.register(self.save_persistent)
        atexit.register(self._close_tb_writer)
        atexit.register(self._close_io_log)

    def _load_frozen_transformer_teacher(self):
        backbone_path = TRANSFORMER_MODEL_PATH / "backbone.pth"
        if backbone_path.exists():
            try:
                backbone_state = torch.load(backbone_path, map_location=device)
                loaded = self.backbone.load_transformer_backbone_weights(backbone_state)
                print(f"[VisualDDQN] Loaded frozen teacher backbone parts: {loaded}")
            except Exception as exc:
                print(f"[VisualDDQN] Failed to load frozen teacher backbone: {exc}")

        q_path = TRANSFORMER_MODEL_PATH / "q_network.pth"
        if q_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(q_path, map_location=device))
                print("[VisualDDQN] Loaded frozen teacher Q-Network")
            except Exception as exc:
                print(f"[VisualDDQN] Failed to load frozen teacher Q-Network: {exc}")

        q_target_path = TRANSFORMER_MODEL_PATH / "q_target.pth"
        if q_target_path.exists():
            try:
                self.q_target.load_state_dict(torch.load(q_target_path, map_location=device))
                print("[VisualDDQN] Loaded frozen teacher Q-Target")
            except Exception as exc:
                print(f"[VisualDDQN] Failed to load frozen teacher Q-Target: {exc}")

    def _freeze_teacher_modules(self):
        for module in (self.backbone.core.transformer, self.backbone.core.decoder, self.q_network, self.q_target):
            for param in module.parameters():
                param.requires_grad_(False)

    def _set_runtime_modes(self):
        self.backbone.feature_extractor.train()
        self.backbone.token_adapter.train()
        self.backbone.memory_position.train()
        self.backbone.query_position.train()
        self.backbone.core.transformer.eval()
        self.backbone.core.decoder.eval()
        self.q_network.eval()
        self.q_target.eval()

    def preprocess_screen(self, screenshot_path):
        try:
            image = Image.open(screenshot_path).convert("RGB")
            return self.transform(image)
        except Exception as exc:
            print(f"[VisualDDQN] Error preprocessing screen: {exc}")
            return torch.zeros((3, *IMAGE_SIZE))

    def action_to_grid(self, action_id):
        row = int(action_id) // self.grid_w
        col = int(action_id) % self.grid_w
        return row, col

    def action_to_screen_coords(self, action_id):
        row, col = self.action_to_grid(action_id)
        x, y, width, height = self.screen_region
        cell_w = width / self.grid_w
        cell_h = height / self.grid_h
        return int(x + (col + 0.5) * cell_w), int(y + (row + 0.5) * cell_h)

    def _state_key(self, state):
        state_uint8 = state.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        return hashlib.sha1(state_uint8.tobytes()).hexdigest()

    def clear_blocked_actions(self, reason="state changed"):
        if self.blocked_actions_current_state:
            print(f"[VisualDDQN] Clear blocked actions ({reason}): {sorted(self.blocked_actions_current_state)}")
        self.blocked_actions_current_state.clear()
        self.current_state_key = None

    def block_action_for_state(self, state, action_id):
        state_key = self._state_key(state)
        if self.current_state_key != state_key:
            self.current_state_key = state_key
            self.blocked_actions_current_state.clear()

        self.blocked_actions_current_state.add(int(action_id))
        row, col = self.action_to_grid(action_id)
        print(f"[VisualDDQN] Block invalid action {action_id} -> ({row},{col}) for current state")

    def select_action(self, state, add_noise=True):
        state_key = self._state_key(state)
        if self.current_state_key != state_key:
            if self.current_state_key is not None:
                self.clear_blocked_actions(reason="new screenshot state")
            self.current_state_key = state_key

        blocked_actions = set(self.blocked_actions_current_state)
        available_actions = [idx for idx in range(self.num_actions) if idx not in blocked_actions]

        if not available_actions:
            self.clear_blocked_actions(reason="all actions blocked")
            self.current_state_key = state_key
            blocked_actions = set()
            available_actions = list(range(self.num_actions))

        if add_noise and random.random() < self.epsilon:
            action_id = random.choice(available_actions)
            row, col = self.action_to_grid(action_id)
            return action_id, {
                "action_id": action_id,
                "row": row,
                "col": col,
                "selected_q": None,
                "top_actions": [],
                "source": "epsilon",
                "candidate_rank": len(blocked_actions) + 1,
                "blocked_actions": sorted(blocked_actions),
            }

        state_batch = state.unsqueeze(0).to(device)
        self._set_runtime_modes()
        self.backbone.feature_extractor.eval()
        self.backbone.token_adapter.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                features = self.backbone.get_features(state_batch)
                q_2d = self.q_network(features).squeeze(0)
            q_logits = q_2d.view(-1)
            masked_logits = q_logits.clone()
            if blocked_actions:
                blocked_idx = torch.tensor(sorted(blocked_actions), dtype=torch.long, device=masked_logits.device)
                masked_logits[blocked_idx] = float("-inf")

            action_id = int(masked_logits.argmax().item())
            topk = min(5, len(available_actions))
            top_vals, top_idx = torch.topk(masked_logits, k=topk)
        self._set_runtime_modes()

        row, col = self.action_to_grid(action_id)
        top_actions = []
        for value, idx in zip(top_vals.tolist(), top_idx.tolist()):
            top_row, top_col = self.action_to_grid(idx)
            top_actions.append((idx, top_row, top_col, float(value)))

        return action_id, {
            "action_id": action_id,
            "row": row,
            "col": col,
            "selected_q": float(q_logits[action_id].item()),
            "top_actions": top_actions,
            "source": "greedy",
            "candidate_rank": len(blocked_actions) + 1,
            "blocked_actions": sorted(blocked_actions),
        }

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.store(
            state.cpu(),
            int(action),
            next_state.cpu() if next_state is not None else None,
            reward,
            done,
        )

    def train_step(self):
        if self.replay_buffer.size() < VISUAL_BATCH_SIZE:
            return None

        self.total_it += 1
        state, action, next_state, reward, done, sample_indices, _ = self.replay_buffer.sample(VISUAL_BATCH_SIZE, device=device)
        batch_size = state.size(0)
        self._set_runtime_modes()

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                next_features = self.backbone.get_features(next_state)
                next_online_q_2d = self.q_network(next_features)
                next_online_q_flat = next_online_q_2d.view(batch_size, -1)
                next_best_flat = next_online_q_flat.argmax(dim=1)
                next_best_rows = next_best_flat // self.grid_w
                next_best_cols = next_best_flat % self.grid_w
                next_target_q_2d = self.q_target(next_features)
                next_target_q = next_target_q_2d[
                    torch.arange(batch_size, device=device), next_best_rows, next_best_cols
                ].unsqueeze(1)
                target = reward + (1 - done) * VISUAL_GAMMA * next_target_q

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            features = self.backbone.get_features(state)
            q_2d = self.q_network(features)
            row_idx = action // self.grid_w
            col_idx = action % self.grid_w
            q_taken = q_2d[
                torch.arange(batch_size, device=device), row_idx, col_idx
            ].unsqueeze(1)
            loss = F.smooth_l1_loss(q_taken, target)
            td_error = (q_taken.detach() - target.detach()).abs()

        self.replay_buffer.update_priorities(sample_indices, td_error)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        yolo_params_before_step = [
            param.detach().float().cpu().clone()
            for param in self.backbone.feature_extractor.parameters()
            if param.requires_grad
        ]
        grad_norm_total = torch.nn.utils.clip_grad_norm_(self.backbone.trainable_parameters(), max_norm=1.0)
        grad_norm_yolo = self._module_grad_norm(self.backbone.feature_extractor)
        grad_norm_backbone = self._module_grad_norm(self.backbone.core)
        grad_norm_policy = (
            self._module_grad_norm(self.backbone.token_adapter)
            + self._module_grad_norm(self.backbone.memory_position)
            + self._module_grad_norm(self.backbone.query_position)
        )
        grad_norm_head = self._module_grad_norm(self.q_network)
        yolo_debug = self._module_grad_debug(self.backbone.feature_extractor)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        yolo_param_delta = self._parameter_delta_norm(
            self.backbone.feature_extractor,
            yolo_params_before_step,
        )

        if self.total_it % TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
            self.q_target.eval()

        with torch.no_grad():
            q_mean = q_taken.mean().item()
            q0 = q_2d[0].view(-1)
            top_vals, top_idx = torch.topk(q0, k=min(5, self.num_actions))
            top_actions = [
                (int(idx), *self.action_to_grid(int(idx)), float(val))
                for val, idx in zip(top_vals.tolist(), top_idx.tolist())
            ]

        self._io_log.write(
            f"[Step {self.total_it}] {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"  reward_mean={reward.mean().item():.4f} | done_rate={done.mean().item():.4f}\n"
            f"  action_batch={action.tolist()}\n"
            f"  q_top5={top_actions}\n"
            f"  loss={loss.item():.6f} | q_mean={q_mean:.6f} | epsilon={self.epsilon:.4f}\n"
            f"  grad_norm_total={float(grad_norm_total):.6f} | "
            f"yolo={grad_norm_yolo:.6f} | backbone={grad_norm_backbone:.6f} | "
            f"policy={grad_norm_policy:.6f} | head={grad_norm_head:.6f} | "
            f"yolo_param_delta={yolo_param_delta:.6f}\n"
            f"  yolo_debug="
            f"params:{yolo_debug['param_count']} | "
            f"requires_grad:{yolo_debug['requires_grad_count']} | "
            f"grad_params:{yolo_debug['grad_param_count']} | "
            f"grad_elems:{yolo_debug['grad_element_count']} | "
            f"nan_grads:{yolo_debug['nan_grad_count']}\n"
            f"---\n"
        )
        self._io_log.flush()

        self.tb_writer.add_scalar("train/loss", loss.item(), self.total_it)
        self.tb_writer.add_scalar("train/q_mean", q_mean, self.total_it)
        self.tb_writer.add_scalar("train/done_rate", done.mean().item(), self.total_it)
        self.tb_writer.add_scalar("train/epsilon", self.epsilon, self.total_it)
        self.tb_writer.add_scalar("grad/total_norm", float(grad_norm_total), self.total_it)
        self.tb_writer.add_scalar("grad/yolo_norm", grad_norm_yolo, self.total_it)
        self.tb_writer.add_scalar("grad/yolo_param_delta", yolo_param_delta, self.total_it)
        self.tb_writer.add_scalar("grad/backbone_norm", grad_norm_backbone, self.total_it)
        self.tb_writer.add_scalar("grad/policy_norm", grad_norm_policy, self.total_it)
        self.tb_writer.add_scalar("grad/head_norm", grad_norm_head, self.total_it)
        self.tb_writer.add_scalar("debug/yolo_param_count", yolo_debug["param_count"], self.total_it)
        self.tb_writer.add_scalar("debug/yolo_requires_grad_param_count", yolo_debug["requires_grad_count"], self.total_it)
        self.tb_writer.add_scalar("debug/yolo_grad_param_count", yolo_debug["grad_param_count"], self.total_it)
        self.tb_writer.add_scalar("debug/yolo_grad_element_count", yolo_debug["grad_element_count"], self.total_it)
        self.tb_writer.add_scalar("debug/yolo_nan_grad_count", yolo_debug["nan_grad_count"], self.total_it)

        if yolo_debug["grad_param_count"] == 0 and self.total_it <= 10:
            print(
                "[VisualDDQN][DEBUG] YOLO grad missing: "
                f"params={yolo_debug['param_count']}, "
                f"requires_grad={yolo_debug['requires_grad_count']}, "
                f"grad_params={yolo_debug['grad_param_count']}, "
                f"nan_grads={yolo_debug['nan_grad_count']}"
            )

        if self.total_it % VISUAL_HISTOGRAM_EVERY == 0:
            self._log_tensorboard_histograms(self.total_it)

        self.tb_writer.flush()

        return {
            "loss": loss.item(),
            "q_mean": q_mean,
        }

    def reset_episode(self):
        pass

    def log_episode_metrics(self, win, invalid_click_rate, reward_mean):
        next_episode = self.episode_count + 1
        self.tb_writer.add_scalar("episode/reward_mean", float(reward_mean), next_episode)
        self.tb_writer.add_scalar("episode/win", float(bool(win)), next_episode)
        self.tb_writer.add_scalar("episode/invalid_click_rate", float(invalid_click_rate), next_episode)
        self.tb_writer.flush()

    def on_episode_end(self):
        self.episode_count += 1
        decay_progress = min(self.episode_count / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_min + (0.30 - self.epsilon_min) * (1.0 - decay_progress)
        self.tb_writer.add_scalar("episode/epsilon", self.epsilon, self.episode_count)
        self._save_model()
        if self.episode_count % SAVE_EVERY_N_EPISODES == 0:
            print(f"[VisualDDQN] Periodic save at episode {self.episode_count} | epsilon={self.epsilon:.4f}")
            self.save_persistent()

    def _save_model(self):
        VISUAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "feature_extractor": self.backbone.feature_extractor.state_dict(),
                "token_adapter": self.backbone.token_adapter.state_dict(),
                "memory_position": self.backbone.memory_position.state_dict(),
                "query_position": self.backbone.query_position.state_dict(),
                "query_tokens": self.backbone.query_tokens.detach().cpu(),
            },
            VISUAL_MODEL_PATH / "trainable_backbone.pth",
        )
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
            },
            VISUAL_MODEL_PATH / "optimizer_state.pth",
        )

    def save_persistent(self):
        buf = self.replay_buffer
        if buf.size_count == 0:
            return

        reward_groups = defaultdict(list)
        for idx in range(buf.size_count):
            reward_groups[buf.index[idx]["reward"]].append(idx)

        target = min(VISUAL_SAVE_CAPACITY, buf.size_count)
        selected_indices = []
        remaining = target
        groups = sorted(reward_groups.items(), key=lambda item: len(item[1]))
        for group_idx, (_, indices) in enumerate(groups):
            if group_idx == len(groups) - 1:
                count = remaining
            else:
                count = round(len(indices) / buf.size_count * target)
            count = min(count, len(indices), remaining)
            selected_indices.extend(random.sample(indices, count))
            remaining -= count
            if remaining <= 0:
                break

        VISUAL_REPLAY_PERSISTENT_PATH.mkdir(parents=True, exist_ok=True)
        for file_path in VISUAL_REPLAY_PERSISTENT_PATH.glob("*.pt"):
            file_path.unlink()

        persistent_index = []
        save_idx = 0
        for old_idx in selected_indices:
            old_entry = buf.index[old_idx]

            state_src = Path(old_entry["state"])
            if not state_src.exists():
                continue

            state_dst = VISUAL_REPLAY_PERSISTENT_PATH / f"state_{save_idx}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            if old_entry["next_state"]:
                next_src = Path(old_entry["next_state"])
                if next_src.exists():
                    next_state_dst = VISUAL_REPLAY_PERSISTENT_PATH / f"next_state_{save_idx}.pt"
                    shutil.copy2(str(next_src), str(next_state_dst))

            persistent_index.append(
                {
                    "storage_id": save_idx,
                    "state": str(state_dst),
                    "action": old_entry["action"],
                    "next_state": str(next_state_dst) if next_state_dst else None,
                    "reward": old_entry["reward"],
                    "done": old_entry["done"],
                    "insert_order": save_idx + 1,
                }
            )
            save_idx += 1

        torch.save(
            {
                "persistent_index": persistent_index,
                "total_it": self.total_it,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
            },
            VISUAL_MODEL_PATH / "training_state.pth",
        )

    def try_load_model(self):
        trainable_backbone_path = VISUAL_MODEL_PATH / "trainable_backbone.pth"
        if trainable_backbone_path.exists():
            try:
                state = torch.load(trainable_backbone_path, map_location=device)
                if "feature_extractor" in state:
                    self.backbone.feature_extractor.load_state_dict(state["feature_extractor"])
                if "token_adapter" in state:
                    self.backbone.token_adapter.load_state_dict(state["token_adapter"])
                if "memory_position" in state:
                    self.backbone.memory_position.load_state_dict(state["memory_position"])
                if "query_position" in state:
                    self.backbone.query_position.load_state_dict(state["query_position"])
                if "query_tokens" in state and tuple(state["query_tokens"].shape) == tuple(self.backbone.query_tokens.shape):
                    self.backbone.query_tokens.data.copy_(state["query_tokens"].to(self.backbone.query_tokens.device))
                print("[VisualDDQN] Loaded trainable visual backbone parts")
            except Exception as exc:
                print(f"[VisualDDQN] Failed to load trainable visual backbone parts: {exc}")

        opt_path = VISUAL_MODEL_PATH / "optimizer_state.pth"
        if opt_path.exists():
            try:
                state = torch.load(opt_path, map_location=device, weights_only=False)
                self.optimizer.load_state_dict(state["optimizer"])
                self.total_it = state.get("total_it", 0)
                self.episode_count = state.get("episode_count", 0)
                self.epsilon = state.get("epsilon", self.epsilon)
                print(
                    f"[VisualDDQN] Loaded optimizer: total_it={self.total_it}, "
                    f"episode={self.episode_count}, epsilon={self.epsilon:.4f}"
                )
                if "scaler" in state:
                    print("[VisualDDQN] Skip restoring GradScaler state to avoid stale AMP optimizer stage")
            except Exception as exc:
                print(f"[VisualDDQN] Failed to load optimizer state: {exc}")

        training_state_path = VISUAL_MODEL_PATH / "training_state.pth"
        if training_state_path.exists():
            try:
                state = torch.load(training_state_path, map_location=device, weights_only=False)
                persistent_index = state.get("persistent_index", [])
                if persistent_index:
                    self._load_persistent_buffer(persistent_index)
            except Exception as exc:
                print(f"[VisualDDQN] Failed to load replay buffer: {exc}")

        self._freeze_teacher_modules()
        self._set_runtime_modes()

    def _load_persistent_buffer(self, persistent_index):
        VISUAL_REPLAY_PATH.mkdir(parents=True, exist_ok=True)
        for file_path in VISUAL_REPLAY_PATH.glob("*.pt"):
            file_path.unlink()

        loaded_count = 0
        self.replay_buffer.index = []
        for entry in persistent_index[: self.replay_buffer.max_size]:
            state_src_str = entry.get("state", entry.get("state_path"))
            if state_src_str is None:
                continue
            state_src = Path(state_src_str)
            if not state_src.exists():
                continue

            storage_id = loaded_count
            state_dst = VISUAL_REPLAY_PATH / f"state_{storage_id}.pt"
            shutil.copy2(str(state_src), str(state_dst))

            next_state_dst = None
            next_src_str = entry.get("next_state", entry.get("next_state_path"))
            if next_src_str:
                next_src = Path(next_src_str)
                if next_src.exists():
                    next_state_dst = VISUAL_REPLAY_PATH / f"next_state_{storage_id}.pt"
                    shutil.copy2(str(next_src), str(next_state_dst))

            runtime_entry = {
                "storage_id": storage_id,
                "state": str(state_dst),
                "action": int(entry["action"]),
                "next_state": str(next_state_dst) if next_state_dst else None,
                "reward": float(entry["reward"]),
                "done": bool(entry["done"]),
                "reward_type": self.replay_buffer._reward_type(float(entry["reward"]), bool(entry["done"])),
                "priority": float(np.clip(abs(float(entry["reward"])) + 1.0, VISUAL_PRIORITY_MIN, VISUAL_PRIORITY_MAX)),
                "insert_order": loaded_count + 1,
            }
            self.replay_buffer.index.append(runtime_entry)
            loaded_count += 1

        self.replay_buffer.size_count = loaded_count
        self.replay_buffer.next_storage_id = loaded_count
        self.replay_buffer.insert_counter = loaded_count
        print(f"[VisualDDQN] Loaded {loaded_count} replay buffer entries")

    def _close_io_log(self):
        if self._io_log and not self._io_log.closed:
            self._io_log.close()

    def _close_tb_writer(self):
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.close()

    def _module_grad_norm(self, module):
        grad_sq_sum = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())
        return grad_sq_sum ** 0.5

    def _module_grad_debug(self, module):
        param_count = 0
        requires_grad_count = 0
        grad_param_count = 0
        grad_element_count = 0
        nan_grad_count = 0

        for param in module.parameters():
            param_count += 1
            if param.requires_grad:
                requires_grad_count += 1
            if param.grad is None:
                continue
            grad_param_count += 1
            grad_element_count += int(param.grad.numel())
            nan_grad_count += int(torch.isnan(param.grad).sum().item())

        return {
            "param_count": param_count,
            "requires_grad_count": requires_grad_count,
            "grad_param_count": grad_param_count,
            "grad_element_count": grad_element_count,
            "nan_grad_count": nan_grad_count,
        }

    def _parameter_delta_norm(self, module, params_before_step):
        delta_sq_sum = 0.0
        before_iter = iter(params_before_step)
        for param in module.parameters():
            if not param.requires_grad:
                continue
            before = next(before_iter, None)
            if before is None:
                break
            after = param.detach().float().cpu()
            delta_sq_sum += float((after - before).pow(2).sum().item())
        return delta_sq_sum ** 0.5

    def _log_tensorboard_histograms(self, global_step):
        module_groups = {
            "weights/yolo": self.backbone.feature_extractor,
            "weights/backbone": self.backbone.core,
            "weights/policy": nn.ModuleList([
                self.backbone.token_adapter,
                self.backbone.memory_position,
                self.backbone.query_position,
            ]),
            "weights/head": self.q_network,
        }
        grad_groups = {
            "grads/yolo": self.backbone.feature_extractor,
            "grads/backbone": self.backbone.core,
            "grads/policy": nn.ModuleList([
                self.backbone.token_adapter,
                self.backbone.memory_position,
                self.backbone.query_position,
            ]),
            "grads/head": self.q_network,
        }

        for tag, module in module_groups.items():
            values = [param.detach().float().reshape(-1).cpu() for param in module.parameters()]
            if values:
                self.tb_writer.add_histogram(tag, torch.cat(values), global_step)

        for tag, module in grad_groups.items():
            values = [
                param.grad.detach().float().reshape(-1).cpu()
                for param in module.parameters()
                if param.grad is not None
            ]
            if values:
                self.tb_writer.add_histogram(tag, torch.cat(values), global_step)

    def log_action_image(self, state, log_info, step_count, reward=None):
        if not LOG_ACTIONS or log_info is None:
            return

        from PIL import ImageDraw, ImageFont

        ACTION_LOG_PATH.mkdir(parents=True, exist_ok=True)
        img_array = state.detach().cpu().clamp(0, 1).mul(255).byte().numpy().transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        img_w, img_h = img.size
        cell_w = img_w / self.grid_w
        cell_h = img_h / self.grid_h

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        row = log_info["row"]
        col = log_info["col"]
        left = int(col * cell_w)
        top = int(row * cell_h)
        right = int((col + 1) * cell_w)
        bottom = int((row + 1) * cell_h)
        draw.rectangle([left, top, right, bottom], outline="red", width=4)

        for grid_row in range(1, self.grid_h):
            y = int(grid_row * cell_h)
            draw.line([0, y, img_w, y], fill="white", width=1)
        for grid_col in range(1, self.grid_w):
            x = int(grid_col * cell_w)
            draw.line([x, 0, x, img_h], fill="white", width=1)

        text_lines = [
            f"Step: {step_count}",
            f"Action: {log_info['action_id']} -> ({row}, {col})",
            (
                f"Source: {log_info.get('source', 'unknown')} #{log_info['candidate_rank']}"
                if log_info.get("candidate_rank") is not None
                else f"Source: {log_info.get('source', 'unknown')}"
            ),
        ]
        if log_info.get("blocked_actions"):
            text_lines.append(f"Blocked: {log_info['blocked_actions']}")
        if log_info.get("selected_q") is not None:
            text_lines.append(f"Selected Q: {log_info['selected_q']:.4f}")
        if reward is not None:
            text_lines.append(f"Reward: {reward:.1f}")
        if log_info.get("top_actions"):
            top_desc = ", ".join(
                f"{idx}:({row_idx},{col_idx})={value:.3f}"
                for idx, row_idx, col_idx, value in log_info["top_actions"][:3]
            )
            text_lines.append(f"Top: {top_desc}")

        text_y = 5
        for line in text_lines:
            bbox = draw.textbbox((5, text_y), line, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text((5, text_y), line, fill="white", font=font)
            text_y += 15

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_step_{step_count:04d}.png"
        img.save(ACTION_LOG_PATH / filename)
        print(f"Action log saved: {filename}")


_agent = None


def get_agent(screen_region=None):
    global _agent
    if _agent is None:
        _agent = VisualDiscreteAgent(screen_region)
    return _agent
