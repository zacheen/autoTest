"""
Gradient Norm Probe — 檢查 DDQN 各層是否有 vanishing/exploding gradient。

執行方式:
  cd autoTest_clau (repo root)
  python autoTest_pytorch/probe_gradient.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from RL_Agent import BATCH_SIZE, GAMMA, device
from transformer_discrete_agent import TRANSFORMER_MODEL_PATH, TransformerDiscreteAgent

REPORT_PATH = TRANSFORMER_MODEL_PATH / 'gradient_report.txt'


def collect_gradient_norms(agent):
    """跑一個 training step，收集 gradient norm。"""
    if agent.replay_buffer.size() < BATCH_SIZE:
        return None, None

    state, action, next_state, reward, done = agent.replay_buffer.sample(BATCH_SIZE)

    if isinstance(action, torch.Tensor) and action.dim() > 1:
        action_idx = action.squeeze(-1).long()
    else:
        action_idx = action.long()

    # Double DQN target
    with torch.no_grad():
        next_features = agent.backbone.get_features(next_state)
        next_q_online = agent.q_network(next_features)
        next_mask = next_state[:, 0].reshape(next_state.size(0), -1).bool()
        next_q_online = next_q_online.masked_fill(~next_mask, -1e8)
        best_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = agent.q_target(next_features)
        next_q_value = next_q_target.gather(1, best_actions)
        target = reward + (1 - done) * GAMMA * next_q_value

    features = agent.backbone.get_features(state)
    q_all = agent.q_network(features)
    q_taken = q_all.gather(1, action_idx.unsqueeze(-1))
    loss = torch.nn.functional.huber_loss(q_taken, target)

    agent.optimizer.zero_grad()
    loss.backward()

    grads = {}
    for name, param in agent.backbone.named_parameters():
        grads[f"backbone.{name}"] = param.grad.norm().item() if param.grad is not None else 0.0
    for name, param in agent.q_network.named_parameters():
        grads[f"q_network.{name}"] = param.grad.norm().item() if param.grad is not None else 0.0

    return grads, {'loss': loss.item()}


def format_grads(grads, f):
    """按 layer 分組顯示 gradient norms。"""
    groups = defaultdict(dict)
    for name, norm in sorted(grads.items()):
        if 'transformer' in name and 'layers' in name:
            parts = name.split('.')
            layer_idx = None
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is not None:
                if 'self_attn' in name:
                    key = f"Layer {layer_idx} Self-Attn"
                elif 'linear' in name:
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
        elif 'value_stream' in name:
            key = "Q Value Stream"
        elif 'advantage_stream' in name:
            key = "Q Advantage Stream"
        elif 'output_head' in name:
            key = "Output Head (unused)"
        else:
            key = "Other"
        groups[key][name] = norm

    for key in sorted(groups.keys()):
        params = groups[key]
        norms = list(params.values())
        f.write(f"  {key}: avg={np.mean(norms):.6f} | min={min(norms):.6f} | max={max(norms):.6f}\n")
        for pname, norm in sorted(params.items()):
            flag = ""
            if norm < 1e-6:
                flag = " << VANISHING"
            elif norm > 10.0:
                flag = " << EXPLODING"
            f.write(f"    {pname}: {norm:.6f}{flag}\n")
        f.write("\n")

    all_norms = list(grads.values())
    vanishing = sum(1 for n in all_norms if n < 1e-6)
    exploding = sum(1 for n in all_norms if n > 10.0)
    f.write(f"  SUMMARY: {len(all_norms)} params |"
            f" avg={np.mean(all_norms):.6f} | min={min(all_norms):.6f} | max={max(all_norms):.6f}\n")
    if vanishing:
        f.write(f"  !! {vanishing} VANISHING (< 1e-6)\n")
    if exploding:
        f.write(f"  !! {exploding} EXPLODING (> 10.0)\n")
    if not vanishing and not exploding:
        f.write(f"  OK — no vanishing or exploding\n")
    f.write("\n")


def main():
    print("Loading agent...")
    agent = TransformerDiscreteAgent()

    buf_size = agent.replay_buffer.size()
    print(f"Replay buffer: {buf_size} entries")

    if buf_size < BATCH_SIZE:
        print(f"ERROR: Need {BATCH_SIZE} entries, got {buf_size}")
        sys.exit(1)

    print("Collecting gradients (3 iterations)...")
    all_grads = []
    losses = None

    for _ in range(3):
        g, loss = collect_gradient_norms(agent)
        all_grads.append(g)
        losses = loss

    avg_grads = {n: np.mean([g[n] for g in all_grads]) for n in all_grads[0]}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(f"GRADIENT REPORT (DDQN) | ep={agent.episode_count} | it={agent.total_it}\n")
        f.write(f"loss={losses['loss']:.4f} | epsilon={agent.epsilon:.4f}\n")
        f.write(f"buffer={buf_size}\n\n")
        format_grads(avg_grads, f)

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
