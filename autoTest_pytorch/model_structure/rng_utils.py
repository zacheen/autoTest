"""RNG 種子工具 — 集中處理 python / numpy / torch / cuda 四個來源的 seeding。

被 transformer_discrete_agent.py(stage1)與 visual_discrete_agent_v3.py(stage2)
共用,避免兩邊各自貼一份相同的 random.seed / np.random.seed / torch.manual_seed /
torch.cuda.manual_seed_all 序列。

實際使用的 seed 會被 hyperparameter_dump 自動寫進 hyperparameters.txt
(因為 module 內 SEED 是 ALL_CAPS module-level int,符合 _dump_module 的篩選條件)。
想重現特定 run:把 module 內 `SEED = seed_everything()` 換成
`SEED = seed_everything(<hyperparameters.txt 裡的數字>)`,並從零開始訓練 —
_save_model 會把 RNG state 存進 optimizer_state.pth,resume 後 RNG trajectory
從 checkpoint 還原,SEED 只決定首次啟動的初始狀態。
"""

from __future__ import annotations

import random
import secrets

import numpy as np
import torch


def seed_everything(seed: int | None = None) -> int:
    """Seed python / numpy / torch / cuda 全部 RNG source。

    Args:
        seed: 若為 None 用 ``secrets.randbits(32)`` 產一個新的 32-bit seed。

    Returns:
        實際使用的 seed(int)。caller 可以印出來或記到 log。
    """
    if seed is None:
        seed = secrets.randbits(32)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
