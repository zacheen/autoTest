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

    lr_backbone_pretrained:
        若指定（非 None）且 build_fqf_optimizer 收到非空 pretrained_prefixes,
        backbone 內 name 以那些 prefix 開頭的 param 會被獨立成一個 group 用此 LR。
        用途:v3 的 encoder + token_adapter 從 YOLOGridStatePredictor checkpoint 載入,
        用較小 LR 防 catastrophic forgetting。None → 退回單一 backbone LR。
    """
    lr_backbone: float = 5e-5
    lr_backbone_pretrained: Optional[float] = None
    lr_head: float = 5e-5
    weight_decay: float = 1e-5
    foreach: bool = False
    fused: bool = False


def build_fqf_optimizer(
    backbone_module: nn.Module,
    head_module: nn.Module,
    config: Optional[FQFOptimizerConfig] = None,
    pretrained_prefixes: tuple[str, ...] = (),
) -> torch.optim.Optimizer:
    """為 FQF agent 建構 AdamW(2 或 3 個 param group)。

    只收 requires_grad=True 的參數(自動跳過凍結模組如 v3 的 YOLO11n)。

    param group 配置:
      • pretrained_prefixes 為空(stage 1):
          [backbone_all, head] — 維持原本 2-group 行為,backward compatible
      • pretrained_prefixes 非空(v3):
          backbone 內 named_parameters() name 以這些 prefix 開頭的 → pretrained group
          其餘 backbone 可訓練 param → fresh group
          結果為 [backbone_fresh, backbone_pretrained, head]
          其中 backbone_pretrained 使用 config.lr_backbone_pretrained
          (若 None 則退回 config.lr_backbone)。

    用途說明(v3):
        encoder + token_adapter 是 YOLOGridStatePredictor 預訓練好的,RL gradient
        稀疏雜訊大,單一大 LR 會在前幾百步把預訓練特徵洗掉(catastrophic forgetting)。
        把這兩個模組獨立成一個小 LR group(通常 0.1× base LR),decoder + queries
        繼續用 base LR 學新的 cross-attn pattern。
    """
    if config is None:
        config = FQFOptimizerConfig()

    head_trainable = [p for p in head_module.parameters() if p.requires_grad]

    if not pretrained_prefixes:
        # Stage 1 路徑:單一 backbone group,保持原行為。
        backbone_trainable = [p for p in backbone_module.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [
                {"params": backbone_trainable, "lr": config.lr_backbone},
                {"params": head_trainable, "lr": config.lr_head},
            ],
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            fused=config.fused,
        )

    # V3 路徑:依 name prefix 把 backbone 拆成 pretrained / fresh 兩組。
    pretrained_params: list[nn.Parameter] = []
    fresh_params: list[nn.Parameter] = []
    pretrained_names: list[str] = []
    fresh_names: list[str] = []
    for name, param in backbone_module.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in pretrained_prefixes):
            pretrained_params.append(param)
            pretrained_names.append(name)
        else:
            fresh_params.append(param)
            fresh_names.append(name)

    lr_pretrained = (
        config.lr_backbone_pretrained
        if config.lr_backbone_pretrained is not None
        else config.lr_backbone
    )

    param_groups: list[dict] = []
    # 順序固定 [fresh, pretrained, head]:_base_lrs / _apply_lr_warmup 是按
    # index 對應 param_group,改順序會讓 v3 內部的 LR warmup index 對不上。
    if fresh_params:
        param_groups.append({"params": fresh_params, "lr": config.lr_backbone})
    if pretrained_params:
        param_groups.append({"params": pretrained_params, "lr": lr_pretrained})
    param_groups.append({"params": head_trainable, "lr": config.lr_head})

    # 印一行 split summary,讓 caller 在 log 內肉眼確認 prefix 抓對。
    print(
        f"[build_fqf_optimizer] split by prefix {pretrained_prefixes}: "
        f"fresh={len(fresh_params)} params, "
        f"pretrained={len(pretrained_params)} params, "
        f"head={len(head_trainable)} params | "
        f"lr_fresh={config.lr_backbone:.2e} "
        f"lr_pretrained={lr_pretrained:.2e} "
        f"lr_head={config.lr_head:.2e}"
    )

    return torch.optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay,
        foreach=config.foreach,
        fused=config.fused,
    )
