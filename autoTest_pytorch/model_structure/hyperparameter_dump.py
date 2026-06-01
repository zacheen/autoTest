"""Write training settings to a .txt file at session start.

Three categories of settings are handled in one shot:
  - Module-level constants such as BATCH_SIZE: read from passed modules, keeping
    ALL_CAPS names whose values are int/float/bool/str/Path/list/tuple.
  - Dataclass instances such as MINESWEEPER_REWARD_CONFIG: flattened with
    dataclasses.asdict.
  - Object instance attrs such as AdaptiveEpsilonController or optimizer: by
    default, use public attrs with allowed types from vars(). Optionally pass an
    (obj, [fields]) tuple as an allowlist. Objects with param_groups use the
    optimizer special case: print each group LR plus non-LR scalar defaults.

Header includes session name / timestamp / git commit hash + dirty flag. If Git
is unavailable, that line is skipped. Written once at startup, not updated while
running.
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
        # Each param_group may override lr, so print it separately.
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
    """Dump two sections for one nn.Module:

    1) Freeze status per top-level submodule, derived from `param.requires_grad`
       rather than hardcoded flags, so it reflects the built model.
    2) torchinfo summary, only when input_size is provided because torchinfo needs
       a dummy forward. Output includes layer output shape, param count, and
       estimated MB.

    requirements.txt includes torchinfo, so import directly. If the environment
    is wrong, let ImportError surface instead of silently skipping.
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

    # 2. torchinfo summary (requires input_size)
    if input_size is not None:
        from torchinfo import summary  # required by requirements.txt and the Colab notebook
        lines.append("")
        lines.append(f"# torchinfo summary (input_size={tuple(input_size)})")
        # device=str(model device) puts dummy input on the same device as the model.
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
    """target can be an object or an (object, [field names]) allowlist tuple."""
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
    loaded_checkpoints: dict[str, str] | None = None,
) -> None:
    """One-shot dump of training settings to out_path.

    Args:
        out_path: Output text file path, written as UTF-8.
        modules: Dump ALL_CAPS module-level constants from these modules.
        dataclass_instances: dict[section_name -> dataclass instance].
        instance_attrs: dict[section_name -> object or (object, [field_whitelist])].
            Objects with `param_groups` use the optimizer special case.
        models: dict[section_name -> (nn.Module, input_size | None)].
            Each model prints:
              (a) freeze status per top-level submodule from requires_grad
              (b) torchinfo summary, skipped when input_size is None
            input_size should be `(batch, *input_shape)`, e.g. `(1, 3, 640, 640)`.
        loaded_checkpoints: dict[area -> source path or sentinel].
            Emitted as a [loaded_checkpoints] section right after the header so
            the operator can immediately see which weights this session actually
            used (Stage 1 warm-start vs V3 own checkpoint vs random init).
            Pass ``CheckpointLogger.loaded_sources`` from model_structure.checkpoint_log.
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

    if loaded_checkpoints:
        lines.append("[loaded_checkpoints]")
        # Sort by area name for stable diffs across runs. Pad keys so values
        # line up in the text file.
        keys = sorted(loaded_checkpoints.keys())
        key_width = max(len(k) for k in keys)
        for key in keys:
            lines.append(f"{key.ljust(key_width)} = {loaded_checkpoints[key]}")
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
