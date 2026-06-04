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
    """Shared optimizer defaults for FQF agents.

    Override fields at construction when an agent needs custom values.

    lr_backbone_pretrained:
        If set and build_fqf_optimizer receives non-empty pretrained_prefixes,
        backbone params whose names start with those prefixes get a separate
        group using this LR. V3 uses this when encoder + token_adapter are loaded
        from YOLOGridStatePredictor checkpoint, reducing LR to avoid catastrophic
        forgetting. None falls back to the single backbone LR.
    lr_yolo:
        If set and build_fqf_optimizer receives non-empty yolo_prefixes, backbone
        params whose names start with those prefixes (V3 YOLO11n feature_extractor)
        get their own group at this LR. None falls back to lr_backbone.
    """
    lr_backbone: float = 5e-5
    lr_backbone_pretrained: Optional[float] = None
    lr_yolo: Optional[float] = None
    lr_head: float = 5e-5
    weight_decay: float = 1e-5
    foreach: bool = False
    fused: bool = False


def build_fqf_optimizer(
    backbone_module: nn.Module,
    head_module: nn.Module,
    config: Optional[FQFOptimizerConfig] = None,
    pretrained_prefixes: tuple[str, ...] = (),
    yolo_prefixes: tuple[str, ...] = (),
) -> torch.optim.Optimizer:
    """Build AdamW for an FQF agent with 2 to 4 param groups.

    Only includes requires_grad=True params, so any frozen module is skipped
    automatically. Pass yolo_prefixes (V3: ("feature_extractor.",)) to peel the
    trainable YOLO11n backbone into its own "backbone_yolo" group at config.lr_yolo.

    Param group layout (a "backbone_yolo" group is inserted before "head" in either
    path when yolo_prefixes match trainable params):
      - Empty pretrained_prefixes (stage 1):
          [backbone_all, (backbone_yolo), head]; identical to the original 2-group
          behavior when yolo_prefixes is empty.
      - Non-empty pretrained_prefixes (V3):
          backbone named_parameters() matching a yolo_prefix -> backbone_yolo group;
          else matching a pretrained prefix -> pretrained group; else -> fresh group.
          Result: [backbone_fresh, backbone_pretrained, (backbone_yolo), head]. The
          pretrained group uses config.lr_backbone_pretrained (or lr_backbone if None);
          the yolo group uses config.lr_yolo (or lr_backbone if None).

    V3 rationale:
        encoder + token_adapter are pretrained by YOLOGridStatePredictor. Sparse,
        noisy RL gradients can erase pretrained features early. A small LR group
        (usually 0.1x base LR) protects them while decoder + queries use base LR
        to learn new cross-attention patterns.
    """
    if config is None:
        config = FQFOptimizerConfig()

    head_trainable = [p for p in head_module.parameters() if p.requires_grad]

    # lr_yolo falls back to lr_backbone when unset. yolo_prefixes peels the YOLO11n
    # feature_extractor into its own group; empty (stage 1) leaves behavior unchanged.
    lr_yolo = config.lr_yolo if config.lr_yolo is not None else config.lr_backbone
    yolo_params: list[nn.Parameter] = []

    if not pretrained_prefixes:
        # Stage 1 path: one backbone group (plus an optional YOLO split-out).
        # "name" lets callers label lr/<name> in TB. AdamW ignores it; param_groups
        # just carry it through.
        backbone_trainable: list[nn.Parameter] = []
        for name, param in backbone_module.named_parameters():
            if not param.requires_grad:
                continue
            if yolo_prefixes and any(name.startswith(p) for p in yolo_prefixes):
                yolo_params.append(param)
            else:
                backbone_trainable.append(param)
        groups: list[dict] = [
            {"params": backbone_trainable, "lr": config.lr_backbone, "name": "backbone"},
        ]
        if yolo_params:
            groups.append({"params": yolo_params, "lr": lr_yolo, "name": "backbone_yolo"})
        groups.append({"params": head_trainable, "lr": config.lr_head, "name": "head"})
        return torch.optim.AdamW(
            groups,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            fused=config.fused,
        )

    # V3 path: split backbone into yolo / pretrained / fresh groups by name prefix.
    pretrained_params: list[nn.Parameter] = []
    fresh_params: list[nn.Parameter] = []
    pretrained_names: list[str] = []
    fresh_names: list[str] = []
    for name, param in backbone_module.named_parameters():
        if not param.requires_grad:
            continue
        if yolo_prefixes and any(name.startswith(p) for p in yolo_prefixes):
            yolo_params.append(param)
        elif any(name.startswith(p) for p in pretrained_prefixes):
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
    # Fixed order [fresh, pretrained, yolo, head]. V3 rebuilds _base_lrs from the
    # returned param_groups and _apply_lr_warmup zips by position, so any
    # deterministic order is fine as long as it is stable across save/resume.
    # "name" is for caller TB tags; AdamW itself ignores it.
    if fresh_params:
        param_groups.append({
            "params": fresh_params, "lr": config.lr_backbone, "name": "backbone_fresh",
        })
    if pretrained_params:
        param_groups.append({
            "params": pretrained_params, "lr": lr_pretrained, "name": "backbone_pretrained",
        })
    if yolo_params:
        param_groups.append({
            "params": yolo_params, "lr": lr_yolo, "name": "backbone_yolo",
        })
    param_groups.append({"params": head_trainable, "lr": config.lr_head, "name": "head"})

    # Print a split summary so callers can verify prefix matching in logs.
    print(
        f"[build_fqf_optimizer] split pretrained={pretrained_prefixes} yolo={yolo_prefixes}: "
        f"fresh={len(fresh_params)} params, "
        f"pretrained={len(pretrained_params)} params, "
        f"yolo={len(yolo_params)} params, "
        f"head={len(head_trainable)} params | "
        f"lr_fresh={config.lr_backbone:.2e} "
        f"lr_pretrained={lr_pretrained:.2e} "
        f"lr_yolo={lr_yolo:.2e} "
        f"lr_head={config.lr_head:.2e}"
    )

    return torch.optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay,
        foreach=config.foreach,
        fused=config.fused,
    )
