"""Shared CUDA SDP backend toggles for Stage 1 and V3.

Both agents want to enable flash / mem_efficient / math attention kernels and
have the actually-permitted state logged via hyperparameter_dump. The helpers
here handle the "is this CUDA + does this API exist + did the call succeed"
guard chain and print a per-module prefix on failure.

Typical usage at module top-level — call set_sdp_all() once, unpack into the
three ALL_CAPS module constants:

    from model_structure.sdp_backend import set_sdp_all

    _REQ_FLASH:         bool = True
    _REQ_MEM_EFFICIENT: bool = True
    _REQ_MATH:          bool = True

    USE_FLASH_SDP, USE_MEM_EFFICIENT_SDP, USE_MATH_SDP = set_sdp_all(
        flash=_REQ_FLASH, mem_efficient=_REQ_MEM_EFFICIENT, math=_REQ_MATH,
        device=device, log_prefix="[V3]",
    )

The USE_* constants are picked up by hyperparameter_dump (ALL_CAPS module-level)
so each run records which kernels PyTorch was permitted to use. The _REQ_*
constants stay private (underscore prefix → skipped by hyperparameter_dump).
Tuple unpacking sacrifices the Final[bool] annotation; the ALL_CAPS naming is
what hyperparameter_dump filters on, so the dump still works.

Note: the returned "permitted" state does NOT guarantee the kernel runs at
forward time. PyTorch's SDP dispatcher may still skip flash for fp32 inputs,
for example. To know which kernel actually fires, you need a probe forward
under `torch.nn.attention.sdpa_kernel` or enable `TORCH_LOGS=+sdpa`.
"""

from __future__ import annotations

import torch


def set_sdp(name: str, requested: bool, *, device: torch.device, log_prefix: str = "") -> bool:
    """Configure a single CUDA SDP backend if supported; return the permitted state.

    Args:
        name: One of "flash", "mem_efficient", "math".
        requested: True to allow PyTorch's SDP dispatcher to use this kernel.
        device: Setup only runs when device.type == "cuda".
        log_prefix: Optional prefix on the failure print (e.g. "[V3]", "[Stage1]").

    Returns:
        The effective permitted state. False when device isn't CUDA, the API
        doesn't exist on this PyTorch version, or the underlying call raised.
    """
    if device.type != "cuda":
        return False
    attr = f"enable_{name}_sdp"
    if not hasattr(torch.backends.cuda, attr):
        return False
    try:
        getattr(torch.backends.cuda, attr)(requested)
    except Exception as exc:
        prefix = f"{log_prefix} " if log_prefix else ""
        print(f"{prefix}Failed to set {attr}: {exc}")
        return False
    return requested


def set_sdp_all(
    *,
    flash: bool,
    mem_efficient: bool,
    math: bool,
    device: torch.device,
    log_prefix: str = "",
) -> tuple[bool, bool, bool]:
    """Configure all three CUDA SDP backends in one call.

    Returns a tuple ``(flash, mem_efficient, math)`` of effective permitted
    states, suitable for direct unpacking into the three module-level
    ``USE_*_SDP`` constants. See ``set_sdp`` for per-backend semantics.
    """
    return (
        set_sdp("flash",         flash,         device=device, log_prefix=log_prefix),
        set_sdp("mem_efficient", mem_efficient, device=device, log_prefix=log_prefix),
        set_sdp("math",          math,          device=device, log_prefix=log_prefix),
    )
