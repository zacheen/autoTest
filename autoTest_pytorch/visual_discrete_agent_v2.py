"""visual_discrete_agent_v2.py — Stage 2 Agent with full end-to-end training.

Pipeline:
    screenshot (3, 640, 640)
        ↓ YOLOGridStatePredictor  (train mode, unfrozen, lr=1e-6)
    grid state (12, 6, 6) one-hot
        ↓ TransformerDiscreteAgent backbone + FQF Q-network  (train mode, unfrozen, lr=5e-5)
    Q-values → masked argmax → action_id

Training:
    Raw screenshots are stored in the replay buffer.
    On each train_step, YOLO converts the batch to grid states WITH gradients, so
    the RL loss propagates back through YOLO.  The combined optimizer covers all
    three modules (separate lr per group).
    Win-rate is logged every episode with a rolling 100-episode window.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from yolo_grid_state_predictor import YOLOGridStatePredictor
from transformer_discrete_agent import TransformerDiscreteAgent
from transformer_discrete_agent import PER_CAPACITY, PER_ALPHA, PER_BETA_START
from model_structure.CategorizedReplayBuffer import CategorizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_PREDICTOR_PATH = Path("./models/yolo_grid_predictor/best.pth")
IMAGE_SIZE = (640, 640)
GRID_H = 6
GRID_W = 6

LR_YOLO     = 1e-6   # conservative — YOLO is already well-trained
LR_STAGE1   = 5e-5


class VisualAgentV2:
    """Screenshot → YOLO → Stage 1 Q-network → action, with full end-to-end RL training.

    All parameters are unfrozen.  The combined optimizer uses a lower learning
    rate for YOLO to avoid destabilising the vision module.  Screenshots are
    stored raw in the replay buffer; YOLO runs inside the training loop so RL
    gradients reach its weights.
    """

    def __init__(self, grid_h: int = GRID_H, grid_w: int = GRID_W):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w

        # ── YOLO predictor — fully unfrozen, train() mode ──
        self.yolo_predictor = YOLOGridStatePredictor().to(device)
        if not YOLO_PREDICTOR_PATH.exists():
            raise FileNotFoundError(
                f"YOLOGridStatePredictor checkpoint not found: {YOLO_PREDICTOR_PATH}\n"
                "Run python yolo_grid_state_predictor.py first."
            )
        ckpt = torch.load(YOLO_PREDICTOR_PATH, map_location=device, weights_only=False)
        self.yolo_predictor.load_state_dict(ckpt["model"])
        self.yolo_predictor.train()
        print(f"[V2] YOLOGridStatePredictor loaded (best_val_acc={ckpt.get('best_val_acc', '?'):.4f})")

        # ── Stage 1 agent — fully unfrozen ──
        self.stage1 = TransformerDiscreteAgent(grid_h=grid_h, grid_w=grid_w)
        # backbone and q_network are in train() mode by default after __init__

        # Persisted replay buffer entries may have grid-state shape (12,6,6).
        # We now store raw screenshots (3,640,640), so replace with a fresh buffer
        # to avoid shape mismatch when stacking a batch.
        self.stage1.replay_buffer = CategorizedReplayBuffer(
            max_size=PER_CAPACITY,
            storage_mode="ram",
            win_threshold=3.0,
            lose_threshold=-1.0,
            invalid_threshold=0.0,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
        )
        print("[V2] TransformerDiscreteAgent loaded — training enabled")

        # ── Combined optimizer: YOLO (low lr) + Stage 1 ──
        # Replaces the optimizer stage1 created internally so all three modules
        # share a single update step.
        self.stage1.optimizer = optim.Adam(
            [
                {"params": self.yolo_predictor.parameters(), "lr": LR_YOLO},
                {"params": self.stage1.backbone.parameters(),  "lr": LR_STAGE1},
                {"params": self.stage1.q_network.parameters(), "lr": LR_STAGE1},
            ],
            foreach=False,
            fused=False,
        )

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        self.blocked_actions: set[int] = set()
        self.current_state_key: str | None = None

        # Win-rate tracking
        self._result_window: deque[int] = deque(maxlen=100)
        self._total_wins = 0
        self._total_episodes = 0

    @property
    def episode_count(self) -> int:
        return self._total_episodes

    # ──────────────────────────── utilities ────────────────────────────

    def preprocess_screen(self, screenshot_path) -> torch.Tensor:
        """Screenshot path → float tensor (3, 640, 640), range [0, 1]."""
        try:
            image = Image.open(screenshot_path).convert("RGB")
            return self.transform(image)
        except Exception as exc:
            print(f"[V2] preprocess_screen error: {exc}")
            return torch.zeros(3, *IMAGE_SIZE)

    def action_to_grid(self, action_id: int) -> tuple[int, int]:
        return int(action_id) // self.grid_w, int(action_id) % self.grid_w

    def _state_key(self, state: torch.Tensor) -> str:
        arr = state.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def _yolo_forward(self, batch: torch.Tensor) -> torch.Tensor:
        """YOLO forward WITH gradient tracking — used as preprocessor in train_step."""
        return self.yolo_predictor.predict_grid_state(batch, self.grid_h, self.grid_w)

    # ──────────────────────────── blocked actions ───────────────────────

    def clear_blocked_actions(self, reason: str = "state changed") -> None:
        if self.blocked_actions:
            print(f"[V2] Clear blocked ({reason}): {sorted(self.blocked_actions)}")
        self.blocked_actions.clear()
        self.current_state_key = None

    def block_action_for_state(self, state: torch.Tensor, action_id: int) -> None:
        key = self._state_key(state)
        if self.current_state_key != key:
            self.current_state_key = key
            self.blocked_actions.clear()
        self.blocked_actions.add(int(action_id))
        row, col = self.action_to_grid(action_id)
        print(f"[V2] Block action {action_id} -> ({row},{col})")

    # ──────────────────────────── action selection ──────────────────────

    def select_action(
        self, state: torch.Tensor, add_noise: bool = True
    ) -> tuple[int, dict]:
        """
        Args:
            state: preprocess_screen() output, (3, 640, 640) float tensor.
            add_noise: if True, apply epsilon-greedy using stage1.epsilon.

        Returns:
            (action_id, log_info)
        """
        key = self._state_key(state)
        if self.current_state_key != key:
            if self.current_state_key is not None:
                self.clear_blocked_actions(reason="new screenshot")
            self.current_state_key = key

        blocked = set(self.blocked_actions)
        available = [i for i in range(self.num_actions) if i not in blocked]
        if not available:
            self.clear_blocked_actions(reason="all actions blocked")
            self.current_state_key = key
            blocked = set()
            available = list(range(self.num_actions))

        # epsilon-greedy using stage1's decaying epsilon
        eps = self.stage1.epsilon
        if add_noise and eps > 0 and random.random() < eps:
            action_id = random.choice(available)
            row, col = self.action_to_grid(action_id)
            return action_id, {
                "action_id": action_id, "row": row, "col": col,
                "selected_q": None, "top_actions": [],
                "source": "epsilon", "candidate_rank": len(blocked) + 1,
                "blocked_actions": sorted(blocked),
            }

        # Inference — no gradients needed
        screenshot_batch = state.unsqueeze(0).to(device)
        with torch.no_grad():
            grid_state_batch = self.yolo_predictor.predict_grid_state(
                screenshot_batch, self.grid_h, self.grid_w
            )
            features = self.stage1.backbone.get_features(grid_state_batch)
            q_2d = self.stage1.q_network(features)["q_values"].squeeze(0)
            q_flat = q_2d.view(-1)

            masked_q = q_flat.clone()
            if blocked:
                blocked_idx = torch.tensor(sorted(blocked), dtype=torch.long, device=masked_q.device)
                masked_q[blocked_idx] = float("-inf")

            action_id = int(masked_q.argmax().item())
            topk = min(5, len(available))
            top_vals, top_idx = torch.topk(masked_q, k=topk)

        row, col = self.action_to_grid(action_id)
        top_actions = [
            (idx.item(), idx.item() // self.grid_w, idx.item() % self.grid_w, float(v))
            for v, idx in zip(top_vals, top_idx)
        ]

        return action_id, {
            "action_id": action_id, "row": row, "col": col,
            "selected_q": float(q_flat[action_id].item()),
            "top_actions": top_actions, "source": "greedy",
            "candidate_rank": len(blocked) + 1,
            "blocked_actions": sorted(blocked),
        }

    # ──────────────────────────── training interface ────────────────────

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor | None,
        reward: float,
        done: bool,
    ) -> None:
        """Store raw screenshot tensors in the replay buffer.

        TransformerDiscreteAgent expects action as (row, col).
        """
        row, col = self.action_to_grid(action)
        self.stage1.store_transition(
            state.cpu(),
            (row, col),
            next_state.cpu() if next_state is not None else None,
            reward,
            done,
        )

    def maybe_train_step(self, force: bool = False):
        """End-to-end train step: YOLO gradients flow into backbone + Q-network."""
        return self.stage1.train_step(
            preprocessor=self._yolo_forward,
            extra_params_to_clip=self.yolo_predictor.parameters(),
        )

    def reset_episode(self) -> None:
        self.stage1.reset_episode()

    def on_episode_end(self) -> None:
        self.stage1.on_episode_end()

    # ──────────────────────────── episode metrics ────────────────────────

    def log_episode_metrics(
        self, win: bool, invalid_click_rate: float, reward_mean: float
    ) -> None:
        self._total_episodes += 1
        self._total_wins += int(win)
        self._result_window.append(int(win))

        rolling_wr = sum(self._result_window) / max(len(self._result_window), 1)
        overall_wr = self._total_wins / max(self._total_episodes, 1)

        status = "WIN " if win else "LOSE"
        print(
            f"[V2] Ep {self._total_episodes}: {status} | "
            f"invalid={invalid_click_rate:.1%} | reward={reward_mean:.3f} | "
            f"win_rate(last100)={rolling_wr:.1%} | win_rate(all)={overall_wr:.1%} | "
            f"eps={self.stage1.epsilon:.4f}"
        )

    # ──────────────────────────── action image log ───────────────────────

    def log_action_image(self, state, log_info, step_count, reward=None) -> None:
        """Annotate the screenshot with the chosen cell and save to models/action_logs_v2/."""
        if log_info is None:
            return
        try:
            from PIL import ImageDraw, ImageFont

            log_dir = Path("./models/action_logs_v2")
            log_dir.mkdir(parents=True, exist_ok=True)

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

            row, col = log_info["row"], log_info["col"]
            draw.rectangle(
                [int(col * cell_w), int(row * cell_h), int((col + 1) * cell_w), int((row + 1) * cell_h)],
                outline="red", width=4,
            )
            for r in range(1, self.grid_h):
                y = int(r * cell_h)
                draw.line([0, y, img_w, y], fill="white", width=1)
            for c in range(1, self.grid_w):
                x = int(c * cell_w)
                draw.line([x, 0, x, img_h], fill="white", width=1)

            lines = [
                f"Step: {step_count}",
                f"Action: {log_info['action_id']} -> ({row},{col})",
                f"Source: {log_info.get('source','?')}",
            ]
            if log_info.get("selected_q") is not None:
                lines.append(f"Q: {log_info['selected_q']:.4f}")
            if reward is not None:
                lines.append(f"Reward: {reward:.1f}")

            text_y = 5
            for line in lines:
                bbox = draw.textbbox((5, text_y), line, font=font)
                draw.rectangle(bbox, fill="black")
                draw.text((5, text_y), line, fill="white", font=font)
                text_y += 15

            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img.save(log_dir / f"{ts}_step_{step_count:04d}.png")
        except Exception as exc:
            print(f"[V2] log_action_image failed: {exc}")


# ──────────────────────────── factory ────────────────────────────────

_agent: VisualAgentV2 | None = None


def get_agent(screen_region=None) -> VisualAgentV2:
    global _agent
    if _agent is None:
        _agent = VisualAgentV2()
    return _agent
