"""Sanitize NaN/Inf out of stage1 checkpoint files in-place.

Scans all floating-point tensors in models/stage1_transformer/*.pth, replaces
NaN/Inf elements with 0, and backs up originals as .preSanitize_step{N}.pth.
After this, try_load_model load probes should no longer block startup.

Usage, from either repo root or autoTest_pytorch/:
    python autoTest_pytorch/sanitize_checkpoint.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch


# Resolve checkpoint path from the script location, matching
# transformer_discrete_agent.py TRANSFORMER_MODEL_PATH.
REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "models" / "stage1_transformer"

# Files to scan and whether weights_only can stay enabled. optimizer_state has
# non-tensor objects such as epsilon controller state, so weights_only must be off.
TARGETS = [
    ("backbone",        "backbone.pth",        True),
    ("fqf_network",     "fqf_network.pth",     True),
    ("fqf_target",      "fqf_target.pth",      True),
    ("optimizer_state", "optimizer_state.pth", False),
]


def sanitize_tensor(t, label):
    """Zero out non-finite elements; return (sanitized_tensor, nan_count, inf_count)."""
    if not isinstance(t, torch.Tensor):
        return t, 0, 0
    if not t.dtype.is_floating_point:
        return t, 0, 0
    if torch.isfinite(t).all():
        return t, 0, 0
    nan_mask = torch.isnan(t)
    inf_mask = torch.isinf(t)
    nan_n = int(nan_mask.sum().item())
    inf_n = int(inf_mask.sum().item())
    bad_mask = nan_mask | inf_mask
    sanitized = t.clone()
    sanitized[bad_mask] = 0.0
    print(f"  {label}: nan={nan_n} inf={inf_n} -> zeroed (shape={tuple(t.shape)})")
    return sanitized, nan_n, inf_n


def sanitize_flat_state_dict(sd, label):
    """For backbone / fqf_network / fqf_target — state_dict is flat key→tensor."""
    bad_total = 0
    for key in list(sd.keys()):
        new_t, nan_n, inf_n = sanitize_tensor(sd[key], f"{label}.{key}")
        if nan_n + inf_n > 0:
            sd[key] = new_t
            bad_total += nan_n + inf_n
    return bad_total


def sanitize_optimizer_payload(payload, label):
    """For optimizer_state.pth — payload['optimizer']['state'][pid][key] tensors.

    Also checks exp_avg_sq < 0. Adam's second moment is mathematically
    non-negative; negatives imply disk/memory bit-level corruption such as a
    sign-bit flip. AdamW would compute sqrt(negative)=NaN for that element and
    then write NaN weights. When clearing negative v, also clear paired m
    (exp_avg) at the same positions so contaminated momentum does not push the
    reset weight element in a bad direction.
    """
    bad_total = 0
    if not isinstance(payload, dict):
        return 0
    opt = payload.get("optimizer")
    if not isinstance(opt, dict):
        return 0
    state = opt.get("state", {})
    for pid, pstate in state.items():
        if not isinstance(pstate, dict):
            continue
        # Pass 1: handle NaN/Inf.
        for key in list(pstate.keys()):
            new_v, nan_n, inf_n = sanitize_tensor(
                pstate[key], f"{label}.state[{pid}].{key}"
            )
            if nan_n + inf_n > 0:
                pstate[key] = new_v
                bad_total += nan_n + inf_n
        # Pass 2: handle exp_avg_sq < 0, which is mathematically impossible.
        v = pstate.get("exp_avg_sq")
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            neg_mask = v < 0
            neg_n = int(neg_mask.sum().item())
            if neg_n > 0:
                # Clear v at those positions.
                v_new = v.clone()
                v_new[neg_mask] = 0.0
                pstate["exp_avg_sq"] = v_new
                # Clear m at the same positions so contaminated momentum cannot dominate.
                m = pstate.get("exp_avg")
                if isinstance(m, torch.Tensor) and m.shape == v.shape:
                    m_new = m.clone()
                    m_new[neg_mask] = 0.0
                    pstate["exp_avg"] = m_new
                    print(f"  {label}.state[{pid}].exp_avg_sq: negative={neg_n} -> zeroed "
                          f"(paired exp_avg same positions also zeroed) shape={tuple(v.shape)}")
                else:
                    print(f"  {label}.state[{pid}].exp_avg_sq: negative={neg_n} -> zeroed "
                          f"(WARN: exp_avg shape mismatch, not paired) shape={tuple(v.shape)}")
                bad_total += neg_n
    return bad_total


def main():
    if not CKPT_DIR.exists():
        print(f"[sanitize] checkpoint dir not found: {CKPT_DIR}")
        sys.exit(1)

    # Phase 1: load and scan without writing. Gather all info before deciding.
    loaded = {}
    for name, fname, weights_only in TARGETS:
        path = CKPT_DIR / fname
        if not path.exists():
            print(f"[sanitize] WARN: {path} not found, skipping")
            continue
        try:
            loaded[name] = (path, torch.load(path, map_location="cpu", weights_only=weights_only))
        except Exception as exc:
            print(f"[sanitize] FAIL to read {path}: {exc}")
            sys.exit(1)

    if not loaded:
        print("[sanitize] no checkpoints found, nothing to do")
        return

    # Use optimizer-state total_it as the backup step label.
    total_it = 0
    if "optimizer_state" in loaded:
        _, payload = loaded["optimizer_state"]
        total_it = int(payload.get("total_it", 0))
    suffix = f".preSanitize_step{total_it}.pth"
    print(f"[sanitize] total_it={total_it}, backup suffix={suffix}")
    print()

    # Phase 2: in-memory scan and repair. Print changed positions only.
    bad_total = 0
    for name in ("backbone", "fqf_network", "fqf_target"):
        if name not in loaded:
            continue
        _, sd = loaded[name]
        print(f"[sanitize] scanning {name}:")
        n = sanitize_flat_state_dict(sd, name)
        bad_total += n
        if n == 0:
            print(f"  (clean)")
        print()

    if "optimizer_state" in loaded:
        _, payload = loaded["optimizer_state"]
        print(f"[sanitize] scanning optimizer_state:")
        n = sanitize_optimizer_payload(payload, "optimizer_state")
        bad_total += n
        if n == 0:
            print(f"  (clean)")
        print()

    print(f"[sanitize] TOTAL sanitized elements: {bad_total}")

    if bad_total == 0:
        print("[sanitize] Nothing to write. Disk is already clean.")
        return

    # Phase 3: back up originals, then write sanitized versions.
    # Back up before writing to avoid overwriting before backup completes.
    print()
    print("[sanitize] backing up originals...")
    backups = {}
    for name, (path, _payload) in loaded.items():
        backup = path.with_suffix(suffix)
        if backup.exists():
            # Do not overwrite existing backups; rare case: sanitize ran twice.
            print(f"  WARN: backup already exists: {backup.name}, skipping move")
        else:
            shutil.copy2(str(path), str(backup))
            print(f"  {path.name} -> {backup.name}")
            backups[name] = backup

    print()
    print("[sanitize] writing sanitized files...")
    for name, (path, payload) in loaded.items():
        torch.save(payload, path)
        print(f"  wrote sanitized {path.name}")

    # Phase 4: reload once and verify files are clean.
    print()
    print("[sanitize] verifying sanitized files...")
    all_clean = True
    for name, fname, weights_only in TARGETS:
        path = CKPT_DIR / fname
        if not path.exists():
            continue
        sd = torch.load(path, map_location="cpu", weights_only=weights_only)
        bad = []
        if name == "optimizer_state":
            opt = sd.get("optimizer", {})
            for pid, pstate in opt.get("state", {}).items():
                for key, val in pstate.items():
                    if isinstance(val, torch.Tensor) and val.dtype.is_floating_point:
                        if not torch.isfinite(val).all():
                            bad.append(f"state[{pid}].{key} (non-finite)")
                        # exp_avg_sq must also have no negative values.
                        if key == "exp_avg_sq" and (val < 0).any().item():
                            neg_n = int((val < 0).sum().item())
                            bad.append(f"state[{pid}].{key} (negative={neg_n})")
        else:
            for key, val in sd.items():
                if hasattr(val, "dtype") and val.dtype.is_floating_point:
                    if not torch.isfinite(val).all():
                        bad.append(key)
        if bad:
            all_clean = False
            print(f"  [BAD] {path.name}: still non-finite at {bad}")
        else:
            print(f"  [OK]  {path.name}: clean")

    print()
    if all_clean:
        print("[sanitize] DONE. Disk checkpoints are clean. Safe to resume training.")
    else:
        print("[sanitize] WARNING: some files are still non-finite after sanitize. Check above.")
        sys.exit(2)


if __name__ == "__main__":
    main()
