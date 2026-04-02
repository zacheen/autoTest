"""
畫 validation win rate 折線圖。

執行方式:
  cd autoTest_clau (repo root)
  python autoTest_pytorch/plot_winrate.py
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("./models/stage1_transformer/training_log.csv")
OUTPUT_PATH = Path("./models/stage1_transformer/eval_winrate.png")


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found")
        return

    episodes = []
    win_rates = []

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wr = row.get('eval_win_rate', '')
            if wr:  # 只取有 eval 的 row
                episodes.append(int(row['episode']))
                win_rates.append(float(wr))

    if not episodes:
        print("No eval data found in CSV")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, win_rates, 'b-o', markersize=3, linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.title('Validation Win Rate')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)

    # 標注最高點
    max_wr = max(win_rates)
    max_ep = episodes[win_rates.index(max_wr)]
    plt.annotate(f'Best: {max_wr:.1f}% @ ep{max_ep}',
                 xy=(max_ep, max_wr),
                 xytext=(max_ep + len(episodes) * 0.05, max_wr + 3),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
