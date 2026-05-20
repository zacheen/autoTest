"""
Probing Diagnostic — 診斷 Transformer Minesweeper agent 訓練失敗原因。

Probe 1: Attention Pattern — attention heads 有學到看鄰居嗎?
Probe 3: Q-Value Landscape — critic 能分辨安全格 vs 地雷格嗎?

執行方式:
  cd autoTest_pytorch
  python probe_diagnostic.py

輸出:
  models/stage1_transformer/probe_report.txt
"""

import sys
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from Minesweeper.MinesweeperLogic import MinesweeperLogic, BoardConfig
from RL_Agent import GRID_STATE_CHANNELS, device
from transformer_discrete_agent import TRANSFORMER_MODEL_PATH, TransformerActorNetwork
# Probe 跟 train 用同一份難度設定 — train 決定挑哪個 preset。
from train_stage1_simple import GRID_CONFIG

REPORT_PATH = TRANSFORMER_MODEL_PATH / 'probe_report.txt'
SEEDS = [42, 123, 7]
NUM_EXTRA_CLICKS = 3  # 第一次點擊後再多點幾下產生 frontier


# ============================================================
# Helpers
# ============================================================

def get_neighbors(r, c, rows, cols):
    """取得 (r, c) 的合法鄰居座標。"""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    return neighbors


def create_game_state(seed: int, config: BoardConfig):
    """建立一個有 frontier 的遊戲局面。

    Returns:
        (logic, state_tensor)
    """
    random.seed(seed)
    logic = MinesweeperLogic(rows=config.rows, cols=config.cols, mines_count=config.mines)
    # 第一次點擊中央，觸發 flood-fill
    logic.click(logic.rows // 2, logic.cols // 2)

    # 再多點幾個安全的未翻開格
    safe_unrevealed = [
        (r, c) for r in range(logic.rows) for c in range(logic.cols)
        if (r, c) not in logic.mines and (r, c) not in logic.revealed
    ]
    random.shuffle(safe_unrevealed)
    for cell in safe_unrevealed[:NUM_EXTRA_CLICKS]:
        if not logic.game_over and not logic.is_win:
            logic.click(cell[0], cell[1])

    state = torch.from_numpy(logic.get_grid_state_array())
    return logic, state


def format_grid_with_mines(logic):
    """畫出 grid，標記地雷位置 M。"""
    grid = logic.get_grid_state()
    lines = []
    lines.append("    " + "  ".join(f"{c}" for c in range(logic.cols)))
    lines.append("   " + "---" * logic.cols)

    for r in range(logic.rows):
        row_str = f"{r} |"
        for c in range(logic.cols):
            if (r, c) in logic.mines and (r, c) not in logic.revealed:
                cell = " M "
            elif grid[r][c] == -1:
                cell = " . "
            elif grid[r][c] == -2:
                cell = " F "
            else:
                cell = f" {grid[r][c]} "
            row_str += cell
        lines.append(row_str)
    return "\n".join(lines)


def get_numbered_cells(logic):
    """找出所有已翻開且數字 >= 1 的 frontier cells。"""
    grid = logic.get_grid_state()
    cells = []
    for r in range(logic.rows):
        for c in range(logic.cols):
            if grid[r][c] >= 1:  # 已翻開、數字 1-8
                cells.append((r, c, grid[r][c]))
    return cells


# ============================================================
# Probe 1: Attention Pattern
# ============================================================

def extract_attention_weights(actor, state_tensor):
    """用 hooks 擷取每層 self_attn 的 attention weights。

    Returns:
        attn_store: dict[layer_idx] → (1, nhead, T, T) where T = num_tokens
        probs: (1, T) action probabilities
    """
    attn_store = {}
    handles = []

    for layer_idx, layer in enumerate(actor.transformer.layers):
        def make_pre(li):
            def hook(module, args, kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                return args, kwargs
            return hook

        def make_post(li):
            def hook(module, input, output):
                # output = (attn_output, attn_weights)
                # attn_weights: (B, nhead, T, T)
                attn_store[li] = output[1].detach().cpu()
            return hook

        h1 = layer.self_attn.register_forward_pre_hook(
            make_pre(layer_idx), with_kwargs=True
        )
        h2 = layer.self_attn.register_forward_hook(make_post(layer_idx))
        handles.extend([h1, h2])

    # Forward pass (用 train mode 避免 PyTorch fast path 繞過 self_attn hooks)
    # no_grad 確保不會更新任何參數，dropout 的影響對診斷可忽略
    actor.train()
    with torch.no_grad():
        state_batch = state_tensor.unsqueeze(0).to(device)
        probs, _ = actor(state_batch)
    actor.eval()

    # 移除 hooks
    for h in handles:
        h.remove()

    return attn_store, probs.cpu()


def analyze_attention(attn_store, logic, f):
    """分析 attention pattern，寫入報告。

    Returns:
        metrics dict
    """
    numbered_cells = get_numbered_cells(logic)
    if not numbered_cells:
        f.write("  (No numbered frontier cells found in this board)\n")
        return None

    num_layers = len(attn_store)
    num_heads = attn_store[0].shape[1]
    num_cells = logic.rows * logic.cols  # = num_tokens (attn shape: (1, h, T, T))

    # 計算每層每 head 的鄰居 attention 比例
    layer_head_ratios = defaultdict(list)  # (layer, head) → [ratios]
    layer_head_entropies = defaultdict(list)

    for r, c, num in numbered_cells:
        tok = r * logic.cols + c
        neighbors = get_neighbors(r, c, rows=logic.rows, cols=logic.cols)
        neighbor_toks = [nr * logic.cols + nc for nr, nc in neighbors]

        for li in range(num_layers):
            for hi in range(num_heads):
                attn = attn_store[li][0, hi, tok, :]  # (num_cells,)
                neighbor_sum = attn[neighbor_toks].sum().item()
                layer_head_ratios[(li, hi)].append(neighbor_sum)

                # Attention entropy
                ent = -(attn * torch.log(attn + 1e-10)).sum().item()
                layer_head_entropies[(li, hi)].append(ent)

    # 彙總
    avg_ratios = {}
    avg_entropies = {}
    for key in layer_head_ratios:
        avg_ratios[key] = np.mean(layer_head_ratios[key])
        avg_entropies[key] = np.mean(layer_head_entropies[key])

    # 找最好的 head
    best_key = max(avg_ratios, key=avg_ratios.get)
    best_ratio = avg_ratios[best_key]

    # Overall metrics
    all_ratios = [v for v in avg_ratios.values()]
    all_entropies = [v for v in avg_entropies.values()]
    overall_ratio = np.mean(all_ratios)
    overall_entropy = np.mean(all_entropies)

    # 中央格的鄰居數作為「uniform attention」基線(corner = 3, edge = 5, center = 8)。
    center_neighbors = len(get_neighbors(logic.rows // 2, logic.cols // 2,
                                         rows=logic.rows, cols=logic.cols))
    uniform_baseline = center_neighbors / num_cells
    uniform_entropy = float(np.log(num_cells))

    # 寫入 per-layer summary
    f.write(f"\n  Per-layer neighbor attention ratio (baseline={uniform_baseline:.2f}):\n")
    for li in range(num_layers):
        head_strs = []
        for hi in range(num_heads):
            head_strs.append(f"H{hi}={avg_ratios[(li,hi)]:.3f}")
        f.write(f"    Layer {li}: {' | '.join(head_strs)}\n")

    f.write(f"\n  Per-layer attention entropy (uniform={uniform_entropy:.2f}):\n")
    for li in range(num_layers):
        head_strs = []
        for hi in range(num_heads):
            head_strs.append(f"H{hi}={avg_entropies[(li,hi)]:.2f}")
        f.write(f"    Layer {li}: {' | '.join(head_strs)}\n")

    f.write(f"\n  Best head: Layer {best_key[0]} Head {best_key[1]}"
            f" (neighbor ratio={best_ratio:.3f})\n")

    # 印出 best head 在前 3 個 numbered cell 的 attention heatmap
    show_cells = numbered_cells[:3]
    for r, c, num in show_cells:
        tok = r * logic.cols + c
        neighbors = get_neighbors(r, c, rows=logic.rows, cols=logic.cols)
        neighbor_set = set(neighbors)
        attn = attn_store[best_key[0]][0, best_key[1], tok, :]  # (num_cells,)
        neighbor_sum = attn[[nr * logic.cols + nc for nr, nc in neighbors]].sum().item()

        f.write(f"\n  Attention from cell ({r},{c}) [number={num}],"
                f" Layer {best_key[0]} Head {best_key[1]}:\n")
        f.write("       " + "     ".join(f"{cc}" for cc in range(logic.cols)) + "\n")

        for rr in range(logic.rows):
            row_str = f"  {rr} |"
            for cc in range(logic.cols):
                val = attn[rr * logic.cols + cc].item()
                if rr == r and cc == c:
                    row_str += f"  *** "
                elif (rr, cc) in neighbor_set:
                    row_str += f"[{val:.2f}]"
                else:
                    row_str += f" {val:.2f} "
            f.write(row_str + "\n")

        f.write(f"  Neighbor attn sum: {neighbor_sum:.3f}"
                f" (baseline: {len(neighbors)/num_cells:.2f})\n")

    return {
        'overall_ratio': overall_ratio,
        'overall_entropy': overall_entropy,
        'best_ratio': best_ratio,
        'best_key': best_key,
        'uniform_baseline': uniform_baseline,
        'uniform_entropy': uniform_entropy,
    }


# ============================================================
# Probe 3: Q-Value Landscape
# ============================================================

def analyze_q_values(backbone, q_network, state_tensor, logic, f):
    """分析 Q-value landscape，寫入報告。

    Returns:
        metrics dict
    """
    backbone.eval()
    q_network.eval()
    with torch.no_grad():
        state_batch = state_tensor.unsqueeze(0).to(device)
        features = backbone.get_features(state_batch)
        q_values = q_network(features)
        q_min = q_values[0].cpu().numpy()  # (num_cells,)

    q_grid = q_min.reshape(logic.rows, logic.cols)

    # 分類 cells
    safe_qs = []
    mine_qs = []
    revealed_count = 0

    for r in range(logic.rows):
        for c in range(logic.cols):
            if (r, c) in logic.revealed:
                revealed_count += 1
            elif (r, c) in logic.mines:
                mine_qs.append(q_grid[r, c])
            else:
                safe_qs.append(q_grid[r, c])

    # Q-value grid 文字
    f.write("\n  Q-Values (min of twin Q):\n")
    f.write("       " + "      ".join(f"{cc}" for cc in range(logic.cols)) + "\n")

    for r in range(logic.rows):
        row_str = f"  {r} |"
        for c in range(logic.cols):
            if (r, c) in logic.revealed:
                row_str += "  ---- "
            elif (r, c) in logic.mines:
                row_str += f"M{q_grid[r,c]:+5.2f} "
            else:
                row_str += f" {q_grid[r,c]:+5.2f} "
        f.write(row_str + "\n")

    f.write(f"  ---- = revealed, M = mine\n")

    # 統計
    safe_qs = np.array(safe_qs) if safe_qs else np.array([0.0])
    mine_qs = np.array(mine_qs) if mine_qs else np.array([0.0])

    safe_mean = safe_qs.mean()
    mine_mean = mine_qs.mean()
    q_gap = safe_mean - mine_mean
    all_unrevealed = np.concatenate([safe_qs, mine_qs])
    q_std = all_unrevealed.std()

    # Rank accuracy: safe cells 中 Q > median mine Q 的比例
    if len(mine_qs) > 0 and len(safe_qs) > 0:
        mine_median = np.median(mine_qs)
        rank_acc = (safe_qs > mine_median).mean() * 100
    else:
        rank_acc = 50.0

    f.write(f"\n  Statistics:\n")
    f.write(f"    Safe unrevealed (N={len(safe_qs)}):"
            f" mean={safe_mean:+.4f}, std={safe_qs.std():.4f},"
            f" range=[{safe_qs.min():+.4f}, {safe_qs.max():+.4f}]\n")
    f.write(f"    Mine cells      (N={len(mine_qs)}):"
            f" mean={mine_mean:+.4f}, std={mine_qs.std():.4f},"
            f" range=[{mine_qs.min():+.4f}, {mine_qs.max():+.4f}]\n")
    f.write(f"    Revealed cells: {revealed_count}\n")
    f.write(f"    Safe-Mine Q gap: {q_gap:+.4f}\n")
    f.write(f"    Rank accuracy:   {rank_acc:.1f}%\n")
    f.write(f"    Q std (unrevealed): {q_std:.4f}\n")

    return {
        'q_gap': q_gap,
        'rank_acc': rank_acc,
        'q_std': q_std,
    }


# ============================================================
# Verdict
# ============================================================

def generate_verdict(attn_metrics_list, q_metrics_list, f, config: BoardConfig):
    """根據所有 probe 結果產生診斷結論。"""
    f.write("\n" + "=" * 55 + "\n")
    f.write("  DIAGNOSTIC VERDICT\n")
    f.write("=" * 55 + "\n")

    # Attention
    valid_attn = [m for m in attn_metrics_list if m is not None]
    if valid_attn:
        avg_ratio = np.mean([m['overall_ratio'] for m in valid_attn])
        avg_entropy = np.mean([m['overall_entropy'] for m in valid_attn])
        best_ratio = max(m['best_ratio'] for m in valid_attn)
        # baseline / uniform 由 analyze_attention 從 logic.rows × logic.cols 算好寫進 metrics
        uniform_baseline = valid_attn[0]['uniform_baseline']
        uniform_entropy = valid_attn[0]['uniform_entropy']

        if avg_ratio > 0.25:
            attn_status = "STRONG — attention focuses on neighbors"
        elif avg_ratio > 0.15:
            attn_status = "LEARNING — some neighbor focus emerging"
        else:
            attn_status = "FAILING — attention is near-uniform, no spatial reasoning"

        f.write(f"\n  ATTENTION PROBE:\n")
        f.write(f"    Avg neighbor attention ratio: {avg_ratio:.3f} (baseline: {uniform_baseline:.2f})\n")
        f.write(f"    Best single head ratio:       {best_ratio:.3f}\n")
        f.write(f"    Avg attention entropy:         {avg_entropy:.2f} (uniform: {uniform_entropy:.2f})\n")
        f.write(f"    STATUS: {attn_status}\n")
    else:
        f.write(f"\n  ATTENTION PROBE: No data (no numbered cells found)\n")

    # Q-values
    valid_q = [m for m in q_metrics_list if m is not None]
    if valid_q:
        avg_gap = np.mean([m['q_gap'] for m in valid_q])
        avg_rank = np.mean([m['rank_acc'] for m in valid_q])
        avg_std = np.mean([m['q_std'] for m in valid_q])

        if avg_gap > 2.0 and avg_rank > 85:
            q_status = "STRONG — critic clearly distinguishes safe from dangerous"
        elif avg_gap > 0.5 and avg_rank > 70:
            q_status = "LEARNING — some differentiation emerging"
        else:
            q_status = "FAILING — critic cannot distinguish safe from dangerous cells"

        f.write(f"\n  Q-VALUE PROBE:\n")
        f.write(f"    Avg safe-mine Q gap: {avg_gap:+.4f}\n")
        f.write(f"    Avg rank accuracy:   {avg_rank:.1f}%\n")
        f.write(f"    Avg Q std:           {avg_std:.4f}\n")
        f.write(f"    STATUS: {q_status}\n")

    # Overall — entropy target 用 log(num_cells) 而不是寫死,跟 attention probe 的 uniform 一致
    entropy_target = (
        valid_attn[0]['uniform_entropy'] if valid_attn else float(np.log(config.rows * config.cols))
    )
    f.write(f"\n  POSSIBLE ROOT CAUSES:\n")
    f.write(f"    1. Attention entropy 遠低於 uniform({entropy_target:.2f}) — exploration / 注意力散布不足\n")
    f.write(f"    2. Epsilon decay 過快 / quota 不平衡 — replay buffer 缺少 win 樣本\n")
    f.write(f"    3. Learning rate may need warmup for Transformer architecture\n")
    f.write(f"    4. Positional encoding may not be expressive enough\n")
    f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    # 檢查 model 檔案
    actor_path = TRANSFORMER_MODEL_PATH / 'actor.pth'
    critic_path = TRANSFORMER_MODEL_PATH / 'critic.pth'

    if not actor_path.exists():
        print(f"ERROR: Actor model not found at {actor_path}")
        sys.exit(1)
    if not critic_path.exists():
        print(f"ERROR: Critic model not found at {critic_path}")
        sys.exit(1)

    # 載入 models
    print("Loading models...")
    actor = TransformerActorNetwork().to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    critic = TransformerCriticNetwork().to(device)
    critic.load_state_dict(torch.load(critic_path, map_location=device))
    critic.eval()

    print(f"Models loaded from {TRANSFORMER_MODEL_PATH}")

    # 開啟報告檔
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("=" * 55 + "\n")
        f.write("  PROBING DIAGNOSTIC REPORT\n")
        f.write("=" * 55 + "\n\n")

        attn_metrics_list = []
        q_metrics_list = []

        for seed in SEEDS:
            logic, state = create_game_state(seed, GRID_CONFIG)

            f.write(f"\n{'='*55}\n")
            f.write(f"  Seed {seed}\n")
            f.write(f"{'='*55}\n\n")
            f.write("Board state (M = mine):\n")
            f.write(format_grid_with_mines(logic) + "\n")

            # Probe 1: Attention
            f.write(f"\n--- PROBE 1: ATTENTION PATTERN ---\n")
            attn_store, probs = extract_attention_weights(actor, state)

            # 印出 action probs top-5
            p = probs[0]
            top5_vals, top5_idx = p.topk(5)
            f.write(f"\n  Action probs top-5:\n")
            for val, idx in zip(top5_vals, top5_idx):
                r, c = idx.item() // logic.cols, idx.item() % logic.cols
                is_mine = "MINE!" if (r, c) in logic.mines else ""
                f.write(f"    ({r},{c}) = {val.item():.4f} {is_mine}\n")

            attn_m = analyze_attention(attn_store, logic, f)
            attn_metrics_list.append(attn_m)

            # Probe 3: Q-values
            f.write(f"\n--- PROBE 3: Q-VALUE LANDSCAPE ---\n")
            q_m = analyze_q_values(actor, critic, state, logic, f)
            q_metrics_list.append(q_m)

        # Verdict
        generate_verdict(attn_metrics_list, q_metrics_list, f, GRID_CONFIG)

    print(f"\nReport saved to: {REPORT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
