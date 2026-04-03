"""
畫訓練過程的 loss 和 key metrics 變化圖。

執行方式:
  cd autoTest_clau
  python autoTest_pytorch/plot_training.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("./models/sac_continuous/training_log.csv")
OUTPUT_PATH = Path("./models/sac_continuous/training_plots.png")


def smooth(data, window=50):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found")
        return

    episodes = []
    critic_loss = []
    actor_loss = []
    q_mean = []
    alpha = []
    entropy = []
    rewards = []
    eval_episodes = []
    eval_wr = []

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row['episode'])
            episodes.append(ep)
            rewards.append(float(row['reward']))

            # Training metrics (may be empty for warmup episodes)
            cl = row.get('critic_loss', '')
            if cl:
                critic_loss.append((ep, float(cl)))
            al = row.get('actor_loss', '')
            if al:
                actor_loss.append((ep, float(al)))
            qm = row.get('q_mean', '')
            if qm:
                q_mean.append((ep, float(qm)))
            a = row.get('alpha', '')
            if a:
                alpha.append((ep, float(a)))
            en = row.get('entropy', '')
            if en:
                entropy.append((ep, float(en)))

            # Eval
            ew = row.get('eval_win_rate', '')
            if ew:
                eval_episodes.append(ep)
                eval_wr.append(float(ew))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('SAC Continuous Training', fontsize=14)

    # 1. Critic Loss
    ax = axes[0, 0]
    if critic_loss:
        eps, vals = zip(*critic_loss)
        ax.plot(eps, vals, alpha=0.3, color='blue', linewidth=0.5)
        if len(vals) > 50:
            ax.plot(list(eps)[25:-24], smooth(vals), color='blue', linewidth=1.5, label='smooth')
        ax.set_title('Critic Loss')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

    # 2. Actor Loss
    ax = axes[0, 1]
    if actor_loss:
        eps, vals = zip(*actor_loss)
        ax.plot(eps, vals, alpha=0.3, color='red', linewidth=0.5)
        if len(vals) > 50:
            ax.plot(list(eps)[25:-24], smooth(vals), color='red', linewidth=1.5, label='smooth')
        ax.set_title('Actor Loss')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

    # 3. Q-mean
    ax = axes[1, 0]
    if q_mean:
        eps, vals = zip(*q_mean)
        ax.plot(eps, vals, alpha=0.3, color='green', linewidth=0.5)
        if len(vals) > 50:
            ax.plot(list(eps)[25:-24], smooth(vals), color='green', linewidth=1.5, label='smooth')
        ax.set_title('Q-value Mean')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

    # 4. Alpha + Entropy
    ax = axes[1, 1]
    if alpha:
        eps_a, vals_a = zip(*alpha)
        ax.plot(eps_a, vals_a, color='orange', linewidth=1, label='alpha')
    if entropy:
        eps_e, vals_e = zip(*entropy)
        ax.plot(eps_e, vals_e, color='purple', linewidth=1, label='entropy')
    ax.set_title('Alpha & Entropy')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Episode Reward
    ax = axes[2, 0]
    ax.plot(episodes, rewards, alpha=0.2, color='gray', linewidth=0.5)
    if len(rewards) > 50:
        ax.plot(episodes[25:-24], smooth(rewards), color='black', linewidth=1.5)
    ax.set_title('Episode Reward')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)

    # 6. Eval Win Rate
    ax = axes[2, 1]
    if eval_wr:
        ax.plot(eval_episodes, eval_wr, 'b-o', markersize=3, linewidth=1.5)
        if max(eval_wr) > 0:
            max_wr = max(eval_wr)
            max_ep = eval_episodes[eval_wr.index(max_wr)]
            ax.annotate(f'Best: {max_wr:.1f}%',
                        xy=(max_ep, max_wr),
                        xytext=(max_ep, max_wr + 3),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
    ax.set_title('Eval Win Rate (%)')
    ax.set_xlabel('Episode')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
