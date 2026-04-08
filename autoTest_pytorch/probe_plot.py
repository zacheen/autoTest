import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Fix relative imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_sac_continuous import SACContinuousAgent, BATCH_SIZE
from RL_Agent import device

def main():
    print("Loading Agent...")
    agent = SACContinuousAgent(grid_h=6, grid_w=6)
    agent.actor.train()
    agent.critic.train()
    
    # Check if replay buffer has entries
    if agent.replay_buffer.size() < BATCH_SIZE:
        print(f"Replay buffer size ({agent.replay_buffer.size()}) is less than BATCH_SIZE ({BATCH_SIZE}). Need more data.")
        return
        
    print(f"Replay buffer size: {agent.replay_buffer.size()}")

    # Hook activations to track active node ratio (SiLU/GELU > 0 or > 0.01)
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            # for SiLU, we can just say "how many nodes have output > 0.01" to be considered active
            # SiLU(x) = x * sigmoid(x). It's slightly negative for x in [-5, 0].
            # Generally, active means output > 0 (or some small positive val)
            active = (output > 0).float().mean().item()
            activations[name] = active
        return hook

    # Register hooks on SiLU/GELU layers
    for name, layer in agent.actor.named_modules():
        if isinstance(layer, (nn.SiLU, nn.GELU, nn.ReLU)):
            layer.register_forward_hook(get_activation(f"actor.{name}"))
            
    for name, layer in agent.critic.named_modules():
        if isinstance(layer, (nn.SiLU, nn.GELU, nn.ReLU)):
            layer.register_forward_hook(get_activation(f"critic.{name}"))

    print("Running train step(s) to collect gradients and activations...")
    # Call train_step() twice to guarantee we hit an exact Actor update step
    agent.train_step()
    agent.train_step()

    print("Collecting statistics...")
    # Collect Weights & Gradients
    actor_weights = []
    actor_grads = []
    
    for name, param in agent.actor.named_parameters():
        actor_weights.append(param.data.cpu().numpy().flatten())
        if param.grad is not None:
            actor_grads.append(param.grad.cpu().numpy().flatten())

    critic_weights = []
    critic_grads = []
    for name, param in agent.critic.named_parameters():
        critic_weights.append(param.data.cpu().numpy().flatten())
        if param.grad is not None:
            critic_grads.append(param.grad.cpu().numpy().flatten())
            
    actor_w_cat = np.concatenate(actor_weights) if actor_weights else np.array([])
    actor_g_cat = np.concatenate(actor_grads) if actor_grads else np.array([])
    critic_w_cat = np.concatenate(critic_weights) if critic_weights else np.array([])
    critic_g_cat = np.concatenate(critic_grads) if critic_grads else np.array([])

    print(f"Actor Weights Mean: {actor_w_cat.mean():.4f}, Std: {actor_w_cat.std():.4f}" if len(actor_w_cat) else "Actor Weights: Empty")
    print(f"Actor Gradients Mean: {actor_g_cat.mean():.6f}, Std: {actor_g_cat.std():.6f}" if len(actor_g_cat) else "Actor Gradients: Empty")
    print(f"Critic Weights Mean: {critic_w_cat.mean():.4f}, Std: {critic_w_cat.std():.4f}" if len(critic_w_cat) else "Critic Weights: Empty")
    print(f"Critic Gradients Mean: {critic_g_cat.mean():.6f}, Std: {critic_g_cat.std():.6f}" if len(critic_g_cat) else "Critic Gradients: Empty")

    print("\n--- Active Node Ratios (output > 0) ---")
    for name, ratio in activations.items():
        print(f"  {name}: {ratio:.2%}")

    # Plotting
    print("\nGenerating plots...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    if len(actor_w_cat) > 0:
        axs[0, 0].hist(actor_w_cat, bins=100, alpha=0.7, color='blue')
        axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Actor Weights Distribution')

    if len(actor_g_cat) > 0:
        axs[0, 1].hist(actor_g_cat, bins=100, alpha=0.7, color='red')
        axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Actor Gradients Distribution')

    if len(critic_w_cat) > 0:
        axs[1, 0].hist(critic_w_cat, bins=100, alpha=0.7, color='blue')
        axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Critic Weights Distribution')

    if len(critic_g_cat) > 0:
        axs[1, 1].hist(critic_g_cat, bins=100, alpha=0.7, color='red')
        axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Critic Gradients Distribution')

    plt.tight_layout()
    out_path = Path("./models/sac_continuous/dist_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()
