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
        # select_action 時快取最後一次 YOLO 預測的 argmax (H, W)，供 log_grid_comparison 使用
        self.last_grid_pred_argmax: torch.Tensor | None = None

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
            # 快取供 log_grid_comparison 使用
            self.last_grid_pred_argmax = grid_state_batch[0].argmax(dim=0).cpu()  # (H, W)

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

    # ──────────────────────── Grid 比較診斷 ───────────────────────────

    _CLS_SYM = ["?", "F", "0", "1", "2", "3", "4", "5", "6", "7", "8", "M"]

    def log_grid_comparison(self, server_state: dict, action_id: int) -> None:
        """將 YOLO 預測的 grid 與 server_state 實際 grid 做逐格比較，印出診斷資訊。

        呼叫時機：select_action() 之後、送出點擊之前，在 server_state 還是
        「點擊前」狀態時呼叫，才能確保 screenshot 與 server_state 同一時刻。
        """
        if self.last_grid_pred_argmax is None:
            return
        if server_state is None:
            return

        try:
            from yolo_grid_state_predictor import server_state_to_grid_tensor
            actual = server_state_to_grid_tensor(server_state)  # (H, W) LongTensor
        except Exception as exc:
            print(f"[V2 Diag] server_state_to_grid_tensor failed: {exc}")
            return

        pred = self.last_grid_pred_argmax  # (H, W)
        H, W = pred.shape
        if actual.shape != pred.shape:
            print(f"[V2 Diag] shape mismatch: pred={tuple(pred.shape)} actual={tuple(actual.shape)}")
            return

        correct = (pred == actual).sum().item()
        total = H * W
        acc = correct / total

        # 找出不一致的格子
        mismatches = []
        for r in range(H):
            for c in range(W):
                p, a = int(pred[r, c]), int(actual[r, c])
                if p != a:
                    mismatches.append((r, c, p, a))

        # 選擇的動作格子資訊
        act_row, act_col = self.action_to_grid(action_id)
        pred_cls = int(pred[act_row, act_col])
        actual_cls = int(actual[act_row, act_col])
        # class 0 = hidden → valid click；其他 → invalid
        is_valid_by_pred   = (pred_cls   == 0)
        is_valid_by_actual = (actual_cls == 0)

        sym = self._CLS_SYM

        # ── Q-value 分析 + Stage 1 原生 select_action 比對 ──
        q_flat_cpu = None
        q_hidden, q_revealed = [], []
        stage1_native_action = None
        q_v2_action = None
        q_stage1_action = None
        try:
            with torch.no_grad():
                # (A) v2 路徑：用 YOLO 預測的 one-hot 跑 Q
                one_hot = torch.zeros(1, 12, H, W, device=device)
                one_hot.scatter_(1, pred.unsqueeze(0).unsqueeze(0).to(device), 1.0)
                feats = self.stage1.backbone.get_features(one_hot)
                q_2d = self.stage1.q_network(feats)["q_values"].squeeze(0).cpu()  # (H, W)
                q_flat_cpu = q_2d.view(-1)
                q_v2_action = float(q_flat_cpu[action_id])

                # (B) Stage 1 原生路徑：直接呼叫 stage1.select_action()
                # 這和 train_stage1_simple.py 訓練/評估時用的程式碼完全一樣
                state_cpu = one_hot.squeeze(0).cpu()  # (12, H, W) on CPU，模擬 MinesweeperLogic 格式
                native_row, native_col = self.stage1.select_action(state_cpu, add_noise=False)
                # stage1.select_action 結束時會把 backbone/q_network 切回 train 模式，手動切回 eval
                self.stage1.backbone.eval()
                self.stage1.q_network.eval()
                stage1_native_action = (native_row, native_col, native_row * self.grid_w + native_col)
                q_stage1_action = float(q_2d[native_row, native_col])

            for r in range(H):
                for c in range(W):
                    q_val = float(q_2d[r, c])
                    if int(actual[r, c]) == 0:   # hidden
                        q_hidden.append((r, c, q_val))
                    else:                        # revealed / flagged
                        q_revealed.append((r, c, q_val))
        except Exception as exc:
            import traceback
            print(f"[V2 Diag] Q re-compute failed: {exc}")
            traceback.print_exc()

        # ── 印出診斷 ──
        sep = "─" * 60
        print(sep)
        print(f"[V2 Diag] YOLO acc={acc:.1%} ({correct}/{total})  action={action_id}->({act_row},{act_col})")

        # 逐格比對表（YOLO pred | Actual | Q-value）
        print(f"  {'YOLO':^{W*2}}   {'Actual':^{W*2}}   Q-values (row by row)")
        for r in range(H):
            yolo_row   = " ".join(sym[int(pred[r, c])]   for c in range(W))
            actual_row = " ".join(sym[int(actual[r, c])] for c in range(W))
            if q_flat_cpu is not None:
                q_row = " ".join(f"{float(q_2d[r, c]):+.2f}" for c in range(W))
            else:
                q_row = "n/a"
            marker = " ←" if r == act_row else ""
            print(f"  {yolo_row}   {actual_row}   {q_row}{marker}")

        # 錯誤格子
        if mismatches:
            mismatch_strs = [f"({r},{c}) YOLO={sym[p]} actual={sym[a]}" for r, c, p, a in mismatches]
            print(f"  Mismatch: {', '.join(mismatch_strs)}")
        else:
            print(f"  Mismatch: none")

        # Q-value 統計
        if q_hidden or q_revealed:
            avg_q_hidden   = sum(v for _, _, v in q_hidden)   / len(q_hidden)   if q_hidden   else float("nan")
            avg_q_revealed = sum(v for _, _, v in q_revealed) / len(q_revealed) if q_revealed else float("nan")
            print(f"  Q hidden({len(q_hidden)}格) avg={avg_q_hidden:+.4f} | "
                  f"revealed({len(q_revealed)}格) avg={avg_q_revealed:+.4f}")
            if q_hidden and q_revealed and avg_q_hidden <= avg_q_revealed:
                print(f"  !! Agent 偏好 revealed 格（Q_hidden <= Q_revealed）← Stage1 權重可能未正確載入")

        # v2 vs Stage 1 原生選擇比較
        if stage1_native_action is not None:
            native_r, native_c, native_id = stage1_native_action
            match = "✓ SAME" if native_id == action_id else "✗ DIFFER"
            print(f"  v2 action={action_id}->({act_row},{act_col}) Q={q_v2_action:+.4f} | "
                  f"stage1.select_action={native_id}->({native_r},{native_c}) Q={q_stage1_action:+.4f} {match}")
            if native_id != action_id:
                print(f"  !! v2 和 stage1.select_action() 選擇不同 → v2 的 Q 計算路徑與 Stage 1 有差異")

        # 選擇格診斷
        action_note = f"YOLO={sym[pred_cls]} actual={sym[actual_cls]}"
        if not is_valid_by_actual and is_valid_by_pred:
            verdict = "YOLO 誤判 ← YOLO 錯誤"
        elif not is_valid_by_actual and not is_valid_by_pred:
            verdict = "YOLO 正確但 Agent 仍選此格 ← Agent 錯誤"
        elif is_valid_by_actual:
            verdict = "OK (hidden cell)"
        else:
            verdict = "unknown"
        print(f"  Action cell: {action_note} → {verdict}")
        print(sep)

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
