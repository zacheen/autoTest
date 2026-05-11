"""Shared optimizer factory for FQF agents.

Both TransformerDiscreteAgent (stage 1) and VisualAgentV3 (stage 2) build their
optimizer through `build_fqf_optimizer`, sharing the defaults in
`FQFOptimizerConfig`. Override individual fields when an agent needs different
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FQFOptimizerConfig:
    """FQF agent 共用的 optimizer 預設值。

    需要單獨調整時，建構時覆蓋對應欄位即可。
    """
    lr_backbone: float = 5e-5
    lr_head: float = 5e-5
    weight_decay: float = 0
    foreach: bool = False
    fused: bool = False


def build_fqf_optimizer(
    backbone_module: nn.Module,
    head_module: nn.Module,
    config: Optional[FQFOptimizerConfig] = None,
) -> torch.optim.Optimizer:
    """為 FQF agent 建構 AdamW（拆 backbone / head 兩個 param group）。

    只收 requires_grad=True 的參數（自動跳過 V3 的凍結 encoder）。
    """
    if config is None:
        config = FQFOptimizerConfig()

    backbone_trainable = [p for p in backbone_module.parameters() if p.requires_grad]
    head_trainable = [p for p in head_module.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": backbone_trainable, "lr": config.lr_backbone},
            {"params": head_trainable, "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
        foreach=config.foreach,
        fused=config.fused,
    )
