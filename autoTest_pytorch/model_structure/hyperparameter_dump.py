"""Write training settings to a .txt file at session start.

Three categories of settings are handled in one shot:
  - 模組層級常數 (BATCH_SIZE 之類)：從傳入的 module 撈,只取 ALL_CAPS 命名 +
    型別屬於 int/float/bool/str/Path/list/tuple 的成員。
  - dataclass 實例 (e.g. MINESWEEPER_REWARD_CONFIG)：用 dataclasses.asdict 展平。
  - 物件實例屬性 (e.g. AdaptiveEpsilonController, optimizer)：預設取 vars()
    內非底線開頭的型別匹配成員;可選擇 (obj, [fields]) tuple 傳允許欄位白名單;
    若物件有 param_groups 屬性,自動走 optimizer 特例(印每個 group lr +
    optimizer.defaults 內非 lr 的純量設定)。

Header 含 session 名 / timestamp / git commit hash + dirty flag。Git 不可用
時跳過該行。寫一次性,執行階段不更新。
"""

from __future__ import annotations

import dataclasses
import datetime
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any


_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_TYPES = (int, float, bool, str, Path, list, tuple)


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _dump_module(module: ModuleType) -> list[str]:
    lines = [f"[{module.__name__}]"]
    pairs = sorted(
        (name, value)
        for name, value in vars(module).items()
        if not name.startswith("_")
        and _NAME_RE.match(name)
        and isinstance(value, _ALLOWED_TYPES)
    )
    for name, value in pairs:
        lines.append(f"{name} = {_format_value(value)}")
    return lines


def _dump_dataclass(section_name: str, instance: Any) -> list[str]:
    lines = [f"[{section_name}]"]
    data = dataclasses.asdict(instance)
    for name in sorted(data.keys()):
        lines.append(f"{name} = {_format_value(data[name])}")
    return lines


def _dump_optimizer(section_name: str, optimizer: Any) -> list[str]:
    lines = [f"[{section_name}]"]
    defaults = getattr(optimizer, "defaults", {})
    for name in sorted(defaults.keys()):
        # 各 param_group 可能蓋掉 lr,獨立印
        if name == "lr":
            continue
        value = defaults[name]
        if isinstance(value, _ALLOWED_TYPES):
            lines.append(f"{name} = {_format_value(value)}")
    for idx, group in enumerate(optimizer.param_groups):
        lr = group.get("lr")
        if lr is not None:
            lines.append(f"group_{idx}.lr = {_format_value(lr)}")
    return lines


def _dump_model_summary(
    section_name: str,
    model: Any,
    input_size: tuple[int, ...] | None = None,
) -> list[str]:
    """Dump 一個 nn.Module 的兩段資料:

    1) Freeze status —— per top-level submodule(讀 `param.requires_grad` 算出來,
       不依賴 hardcoded flag。確實反映 build 後 freeze 設定)。
    2) torchinfo summary —— 帶 input_size 才寫,因為 torchinfo 要 dummy forward。
       輸出包含 layer-by-layer 的 output shape + param count + estimated MB。

    requirements.txt 已包含 torchinfo;這裡直接 import,不做 graceful fallback ──
    若環境有問題就讓 ImportError 噴出來,不要 silent 跳過。
    """
    lines = [f"[{section_name}]"]

    # 1. Freeze status per top-level submodule
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [total, trainable]
    for pname, param in model.named_parameters():
        top = pname.split(".", 1)[0]
        groups[top][0] += param.numel()
        if param.requires_grad:
            groups[top][1] += param.numel()

    lines.append("# Freeze status (per top-level submodule, derived from param.requires_grad)")
    for name in sorted(groups.keys()):
        total, trainable = groups[name]
        if trainable == 0:
            status = "FROZEN"
        elif trainable == total:
            status = "trainable"
        else:
            status = f"mixed (trainable={trainable:,}/{total:,})"
        lines.append(f"{name}: {status}, params={total:,}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(
        f"TOTAL: params={total_params:,} "
        f"(trainable={trainable_total:,}, frozen={total_params - trainable_total:,})"
    )

    # 2. torchinfo summary(需要 input_size)
    if input_size is not None:
        from torchinfo import summary  # 必須裝(requirements.txt + colab ipynb 已加)
        lines.append("")
        lines.append(f"# torchinfo summary (input_size={tuple(input_size)})")
        # device=str(model device) 讓 dummy input 落在跟 model 同個 device。
        device = next(model.parameters()).device
        stats = summary(
            model,
            input_size=tuple(input_size),
            depth=10,
            verbose=0,
            device=str(device),
        )
        lines.extend(str(stats).splitlines())

    return lines


def _dump_object_attrs(section_name: str, target: Any) -> list[str]:
    """target 可以是物件,或 (物件, [field 名單]) tuple 指定白名單。"""
    if (
        isinstance(target, tuple)
        and len(target) == 2
        and isinstance(target[1], (list, tuple))
    ):
        obj, fields = target
        pairs = [(name, getattr(obj, name)) for name in fields if hasattr(obj, name)]
    else:
        obj = target
        attrs = vars(obj) if hasattr(obj, "__dict__") else {}
        pairs = sorted(
            (name, value)
            for name, value in attrs.items()
            if not name.startswith("_") and isinstance(value, _ALLOWED_TYPES)
        )

    lines = [f"[{section_name}]"]
    for name, value in pairs:
        lines.append(f"{name} = {_format_value(value)}")
    return lines


def _git_info() -> tuple[str, bool] | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if commit.returncode != 0:
            return None
        diff = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=2,
        )
        return commit.stdout.strip(), diff.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def dump_hyperparameters(
    out_path: Path,
    modules: list[ModuleType] | None = None,
    dataclass_instances: dict[str, Any] | None = None,
    instance_attrs: dict[str, Any] | None = None,
    models: dict[str, tuple[Any, tuple[int, ...] | None]] | None = None,
) -> None:
    """One-shot dump of training settings to out_path。

    Args:
        out_path: output file path(寫文字檔,utf-8)。
        modules: 取裡面 module 的 ALL_CAPS module-level 常數。
        dataclass_instances: dict[section_name → dataclass instance]。
        instance_attrs: dict[section_name → object 或 (object, [field_whitelist])]。
            若 object 有 `param_groups` 屬性,走 optimizer 特例。
        models: dict[section_name → (nn.Module, input_size | None)]。
            每個 model 印兩段:
              (a) freeze status per top-level submodule(讀 requires_grad)
              (b) torchinfo summary(input_size 是 None 就跳過 (b))
            input_size 應該是 `(batch, *input_shape)`,例如 `(1, 3, 640, 640)`。
    """
    lines: list[str] = [
        "# Hyperparameters dump",
        f"# Session:   {out_path.parent.name}",
        f"# Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}",
    ]
    git = _git_info()
    if git is not None:
        commit, dirty = git
        lines.append(f"# Git:       {commit} (dirty: {dirty})")
    lines.append("")

    for module in modules or []:
        lines.extend(_dump_module(module))
        lines.append("")

    for section_name, instance in (dataclass_instances or {}).items():
        lines.extend(_dump_dataclass(section_name, instance))
        lines.append("")

    for section_name, target in (instance_attrs or {}).items():
        is_filtered_tuple = (
            isinstance(target, tuple)
            and len(target) == 2
            and isinstance(target[1], (list, tuple))
        )
        if not is_filtered_tuple and hasattr(target, "param_groups"):
            lines.extend(_dump_optimizer(section_name, target))
        else:
            lines.extend(_dump_object_attrs(section_name, target))
        lines.append("")

    for section_name, (model, input_size) in (models or {}).items():
        lines.extend(_dump_model_summary(section_name, model, input_size))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
