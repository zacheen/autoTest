"""visual_discrete_agent_v2.py — Stage 2 推論 Agent。

Pipeline:
    screenshot (3, 640, 640)
        ↓ YOLOGridStatePredictor（已訓練，凍住）
    grid state (12, 6, 6) one-hot
        ↓ TransformerDiscreteAgent backbone + FQF Q-network（Stage 1，凍住）
    Q-values → masked argmax → action_id

此模組只做推論，不做 RL 訓練（store_transition / maybe_train_step 為空操作）。
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from yolo_grid_state_predictor import YOLOGridStatePredictor
from transformer_discrete_agent import TransformerDiscreteAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_PREDICTOR_PATH = Path("./models/yolo_grid_predictor/best.pth")
IMAGE_SIZE = (640, 640)
GRID_H = 6
GRID_W = 6


class VisualAgentV2:
    """Screenshot → YOLO grid state → Stage 1 Q-network → action。

    和 VisualDiscreteAgent 實作同一組公開方法，讓 Demo_test_Minesweeper.py
    可以直接替換 get_agent() 而不用改其他程式碼。
    """

    def __init__(self, grid_h: int = GRID_H, grid_w: int = GRID_W, epsilon: float = 0.0):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_actions = grid_h * grid_w
        self.epsilon = epsilon

        # ── YOLO 預測器（凍住）──
        self.yolo_predictor = YOLOGridStatePredictor().to(device)
        if not YOLO_PREDICTOR_PATH.exists():
            raise FileNotFoundError(
                f"YOLOGridStatePredictor checkpoint not found: {YOLO_PREDICTOR_PATH}\n"
                "請先執行 python yolo_grid_state_predictor.py 完成訓練。"
            )
        ckpt = torch.load(YOLO_PREDICTOR_PATH, map_location=device, weights_only=False)
        self.yolo_predictor.load_state_dict(ckpt["model"])
        self.yolo_predictor.eval()
        for p in self.yolo_predictor.parameters():
            p.requires_grad_(False)
        print(f"[V2] YOLOGridStatePredictor loaded (best_val_acc={ckpt.get('best_val_acc', '?'):.4f})")

        # ── Stage 1 agent（只用 backbone + q_network，凍住）──
        self.stage1 = TransformerDiscreteAgent(grid_h=grid_h, grid_w=grid_w)
        self.stage1.backbone.eval()
        self.stage1.q_network.eval()
        for p in self.stage1.backbone.parameters():
            p.requires_grad_(False)
        for p in self.stage1.q_network.parameters():
            p.requires_grad_(False)
        print("[V2] TransformerDiscreteAgent (Stage 1) loaded and frozen")

        # 影像前處理（與 visual_discrete_agent.py 相同）
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        # 封鎖動作（同一畫面下不重複點已被判無效的格子）
        self.blocked_actions: set[int] = set()
        self.current_state_key: str | None = None

        self.episode_count = 0

    # ──────────────────────────── 基礎工具 ────────────────────────────

    def preprocess_screen(self, screenshot_path) -> torch.Tensor:
        """截圖路徑 → float tensor (3, 640, 640)，值域 [0, 1]。"""
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

    # ──────────────────────────── 封鎖動作 ────────────────────────────

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

    # ──────────────────────────── 動作選擇 ────────────────────────────

    def select_action(
        self, state: torch.Tensor, add_noise: bool = True
    ) -> tuple[int, dict]:
        """
        Args:
            state: preprocess_screen() 的輸出，(3, 640, 640) float tensor。
            add_noise: True 時套用 epsilon-greedy（epsilon 預設 0，即純 greedy）。

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

        # epsilon-greedy（預設 epsilon=0，純 greedy）
        if add_noise and self.epsilon > 0 and random.random() < self.epsilon:
            action_id = random.choice(available)
            row, col = self.action_to_grid(action_id)
            return action_id, {
                "action_id": action_id, "row": row, "col": col,
                "selected_q": None, "top_actions": [],
                "source": "epsilon", "candidate_rank": len(blocked) + 1,
                "blocked_actions": sorted(blocked),
            }

        # ── YOLO: screenshot → grid state ──
        screenshot_batch = state.unsqueeze(0).to(device)  # (1, 3, 640, 640)
        with torch.no_grad():
            grid_state_batch = self.yolo_predictor.predict_grid_state(
                screenshot_batch, self.grid_h, self.grid_w
            )  # (1, 12, H, W)

            # ── Stage 1: grid state → Q values ──
            features = self.stage1.backbone.get_features(grid_state_batch)  # (1, H*W, d)
            q_2d = self.stage1.q_network(features)["q_values"].squeeze(0)   # (H, W)
            q_flat = q_2d.view(-1)                                           # (H*W,)

            # 封鎖已試過的動作
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

    # ──────────────────────── 訓練介面（空操作）────────────────────────

    def store_transition(self, state, action, next_state, reward, done) -> None:
        pass

    def maybe_train_step(self, force: bool = False):
        return None

    def reset_episode(self) -> None:
        pass

    # ──────────────────────── Episode 統計 ────────────────────────────

    def log_episode_metrics(
        self, win: bool, invalid_click_rate: float, reward_mean: float
    ) -> None:
        self.episode_count += 1
        status = "WIN " if win else "LOSE"
        print(
            f"[V2] Episode {self.episode_count}: {status} | "
            f"invalid_rate={invalid_click_rate:.1%} | reward_mean={reward_mean:.3f}"
        )

    def on_episode_end(self) -> None:
        pass

    # ──────────────────────── 動作記錄圖 ──────────────────────────────

    def log_action_image(self, state, log_info, step_count, reward=None) -> None:
        """在截圖上標示選擇的格子，存到 models/action_logs_v2/。"""
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


# ──────────────────────────── Factory ────────────────────────────────

_agent: VisualAgentV2 | None = None


def get_agent(screen_region=None) -> VisualAgentV2:
    global _agent
    if _agent is None:
        _agent = VisualAgentV2()
    return _agent
