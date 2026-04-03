"""
驗證 frozen backbone 權重是否正確載入且沒有被更新。

比較 frozen_backbone.pth 和 backbone.pth 中 Transformer 層的權重。

執行方式:
  cd autoTest_clau
  python autoTest_pytorch/verify_frozen.py
"""

import torch
from pathlib import Path

FROZEN_PATH = Path("./models/stage1_transformer/frozen_backbone.pth")
CURRENT_PATH = Path("./models/stage1_transformer/backbone.pth")


def main():
    if not FROZEN_PATH.exists():
        print(f"ERROR: {FROZEN_PATH} not found")
        return
    if not CURRENT_PATH.exists():
        print(f"ERROR: {CURRENT_PATH} not found")
        return

    frozen = torch.load(FROZEN_PATH, map_location='cpu')
    current = torch.load(CURRENT_PATH, map_location='cpu')

    print(f"Frozen:  {len(frozen)} params")
    print(f"Current: {len(current)} params")
    print()

    match = 0
    diff = 0
    only_frozen = 0
    only_current = 0

    # 比較共同的 key（跳過 output_head）
    for name in sorted(set(list(frozen.keys()) + list(current.keys()))):
        if 'output_head' in name:
            continue

        if name in frozen and name in current:
            if torch.equal(frozen[name], current[name]):
                match += 1
            else:
                diff += 1
                print(f"  DIFFERENT: {name}")
        elif name in frozen:
            only_frozen += 1
        else:
            only_current += 1

    print()
    print(f"Transformer weights: {match} match, {diff} different")
    if only_frozen:
        print(f"Only in frozen: {only_frozen}")
    if only_current:
        print(f"Only in current: {only_current}")

    print()
    if diff == 0:
        print("=> PASS: Frozen backbone weights are preserved (not updated by training)")
    else:
        print("=> FAIL: Backbone weights have been modified by training")


if __name__ == "__main__":
    main()
