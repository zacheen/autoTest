"""
Gradient Norm Probe — 檢查 Transformer 各層是否有 vanishing gradient。

做法：載入模型，跑一個 training step，記錄每層的 gradient norm。

執行方式:
  cd autoTest_clau (repo root)
  python autoTest_pytorch/probe_gradient.py

輸出:
  models/stage1_transformer/gradient_report.txt
"""

import sys
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from Minesweeper.MinesweeperLogic import MinesweeperLogic
from RL_Agent import (
    TransformerDiscreteAgent, TRANSFORMER_MODEL_PATH, device,
    BATCH_SIZE, GAMMA, ALPHA_MIN, ALPHA_MAX,
)

sys.path.insert(0, str(Path(__file__).parent))

REPORT_PATH = TRANSFORMER_MODEL_PATH / 'gradient_report.txt'


def collect_gradient_norms(agent):
    """跑一個 training step，收集每個 named parameter 的 gradient norm。

    Returns:
        dict: {param_name: grad_norm} or None if buffer too small
    """
    if agent.replay_buffer.size() < BATCH_SIZE:
        return None, None, None

    state, action, next_state, reward, done = agent.replay_buffer.sample(BATCH_SIZE)
    alpha = agent.log_alpha.exp().detach()

    if isinstance(action, torch.Tensor) and action.dim() > 1:
        action_idx = action.squeeze(-1).long()
    else:
        action_idx = action.long()

    # --- Critic forward + backward ---
    with torch.no_grad():
        next_probs, next_log_probs = agent.actor(next_state)
        next_q1, next_q2 = agent.critic_target(next_state)
        next_q = torch.min(next_q1, next_q2)
        next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1, keepdim=True)
        target_q = reward + (1 - done) * GAMMA * next_v

    q1_all, q2_all = agent.critic(state)
    q1 = q1_all.gather(1, action_idx.unsqueeze(-1))
    q2 = q2_all.gather(1, action_idx.unsqueeze(-1))
    critic_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(q2, target_q)

    agent.critic_optimizer.zero_grad()
    critic_loss.backward()

    critic_grads = {}
    for name, param in agent.critic.named_parameters():
        if param.grad is not None:
            critic_grads[name] = param.grad.norm().item()
        else:
            critic_grads[name] = 0.0

    # --- Actor forward + backward ---
    probs, log_probs = agent.actor(state)
    with torch.no_grad():
        q1_all, q2_all = agent.critic(state)
        min_q = torch.min(q1_all, q2_all)

    actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()

    agent.actor_optimizer.zero_grad()
    actor_loss.backward()

    actor_grads = {}
    for name, param in agent.actor.named_parameters():
        if param.grad is not None:
            actor_grads[name] = param.grad.norm().item()
        else:
            actor_grads[name] = 0.0

    return actor_grads, critic_grads, {
        'critic_loss': critic_loss.item(),
        'actor_loss': actor_loss.item(),
    }


def format_grads_by_layer(grads, f, label):
    """把 gradient norms 按 layer 分組顯示。"""
    f.write(f"\n{'='*55}\n")
    f.write(f"  {label}\n")
    f.write(f"{'='*55}\n\n")

    # 分組: embedding, transformer layers, output head
    groups = defaultdict(dict)
    for name, norm in sorted(grads.items()):
        # 找出屬於哪個 layer
        parts = name.split('.')
        if 'transformer' in name and 'layers' in name:
            # e.g. transformer.layers.0.self_attn.in_proj_weight
            # or q1.transformer.layers.0...
            layer_idx = None
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is not None:
                # 判斷是 self_attn 還是 FFN
                if 'self_attn' in name:
                    key = f"Layer {layer_idx} Self-Attn"
                elif 'linear1' in name or 'linear2' in name:
                    key = f"Layer {layer_idx} FFN"
                elif 'norm' in name:
                    key = f"Layer {layer_idx} Norm"
                else:
                    key = f"Layer {layer_idx} Other"
            else:
                key = "Other"
        elif 'token_embed' in name:
            key = "Token Embedding"
        elif 'row_embed' in name or 'col_embed' in name:
            key = "Positional Encoding"
        elif 'output_head' in name or 'head' in name:
            key = "Output Head"
        else:
            key = "Other"

        groups[key][name] = norm

    # 按順序輸出
    ordered_keys = sorted(groups.keys())
    for key in ordered_keys:
        params = groups[key]
        norms = list(params.values())
        avg_norm = np.mean(norms)
        max_norm = max(norms)
        min_norm = min(norms)

        f.write(f"  {key}:\n")
        f.write(f"    avg={avg_norm:.6f} | min={min_norm:.6f} | max={max_norm:.6f}\n")
        for pname, norm in sorted(params.items()):
            status = ""
            if norm < 1e-6:
                status = " ⚠ VANISHING"
            elif norm > 10.0:
                status = " ⚠ EXPLODING"
            f.write(f"      {pname}: {norm:.6f}{status}\n")
        f.write("\n")

    # Summary
    all_norms = list(grads.values())
    f.write(f"  SUMMARY:\n")
    f.write(f"    Total params with grad: {len(all_norms)}\n")
    f.write(f"    Overall: avg={np.mean(all_norms):.6f}"
            f" | min={min(all_norms):.6f} | max={max(all_norms):.6f}\n")
    vanishing = sum(1 for n in all_norms if n < 1e-6)
    exploding = sum(1 for n in all_norms if n > 10.0)
    if vanishing > 0:
        f.write(f"    ⚠ VANISHING gradients: {vanishing} params (norm < 1e-6)\n")
    if exploding > 0:
        f.write(f"    ⚠ EXPLODING gradients: {exploding} params (norm > 10.0)\n")
    if vanishing == 0 and exploding == 0:
        f.write(f"    ✓ No vanishing or exploding gradients detected\n")
    f.write("\n")


def main():
    # 載入 agent（會自動載入 checkpoint + replay buffer）
    print("Loading agent (with checkpoint + replay buffer)...")

    # 需要從 autoTest_pytorch 目錄 import
    agent = TransformerDiscreteAgent()

    buf_size = agent.replay_buffer.size()
    print(f"Replay buffer size: {buf_size}")

    if buf_size < BATCH_SIZE:
        print(f"ERROR: Need at least {BATCH_SIZE} entries in replay buffer, got {buf_size}")
        print("Run some training first before probing gradients.")
        sys.exit(1)

    # 跑 3 次取平均
    print("Running gradient probes (3 iterations)...")
    all_actor_grads = []
    all_critic_grads = []
    losses = None

    for i in range(3):
        ag, cg, loss = collect_gradient_norms(agent)
        if ag is None:
            print("ERROR: Failed to collect gradients")
            sys.exit(1)
        all_actor_grads.append(ag)
        all_critic_grads.append(cg)
        losses = loss

    # 平均 gradient norms
    avg_actor = {}
    for name in all_actor_grads[0]:
        avg_actor[name] = np.mean([g[name] for g in all_actor_grads])

    avg_critic = {}
    for name in all_critic_grads[0]:
        avg_critic[name] = np.mean([g[name] for g in all_critic_grads])

    # 寫報告
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("=" * 55 + "\n")
        f.write("  GRADIENT NORM REPORT (averaged over 3 iterations)\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"  Replay buffer size: {buf_size}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Critic loss: {losses['critic_loss']:.4f}\n")
        f.write(f"  Actor loss: {losses['actor_loss']:.4f}\n")
        f.write(f"  Episode count: {agent.episode_count}\n")
        f.write(f"  Alpha: {agent.log_alpha.exp().item():.4f}\n")
        f.write(f"  Epsilon: {agent.epsilon:.4f}\n")

        format_grads_by_layer(avg_actor, f, "ACTOR GRADIENTS")
        format_grads_by_layer(avg_critic, f, "CRITIC GRADIENTS (Q1 + Q2)")

        # Verdict
        f.write("=" * 55 + "\n")
        f.write("  VERDICT\n")
        f.write("=" * 55 + "\n\n")

        actor_norms = list(avg_actor.values())
        critic_norms = list(avg_critic.values())

        actor_vanish = sum(1 for n in actor_norms if n < 1e-6)
        critic_vanish = sum(1 for n in critic_norms if n < 1e-6)

        if actor_vanish > 0 or critic_vanish > 0:
            f.write("  ⚠ VANISHING GRADIENT DETECTED\n")
            f.write(f"    Actor: {actor_vanish}/{len(actor_norms)} params vanishing\n")
            f.write(f"    Critic: {critic_vanish}/{len(critic_norms)} params vanishing\n")
        else:
            f.write("  ✓ Gradients are flowing through all layers.\n")
            f.write("    The training failure is NOT caused by vanishing gradients.\n")

    print(f"\nReport saved to: {REPORT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
