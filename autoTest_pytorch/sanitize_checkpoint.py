"""Sanitize NaN/Inf out of stage1 checkpoint files in-place.

掃 models/stage1_transformer/*.pth 內所有 floating-point tensor,把 NaN/Inf
元素改成 0,並把舊檔備份成 .preSanitize_step{N}.pth。執行完之後 try_load_model
的 load probe 就不會擋下 startup,可以接著訓練。

執行方式(可從 repo root 或 autoTest_pytorch/ 任一處呼叫):
    python autoTest_pytorch/sanitize_checkpoint.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch


# 解析 checkpoint 路徑:從腳本位置往上找,跟 transformer_discrete_agent.py 的
# TRANSFORMER_MODEL_PATH = Path("./models/stage1_transformer") 對應。
REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "models" / "stage1_transformer"

# 哪些檔要掃 + load 時要不要關 weights_only(optimizer_state 有 epsilon controller
# 的 state 等非 tensor 物件,要關 weights_only)
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

    除了 NaN/Inf,也檢查 exp_avg_sq < 0 — Adam 的 second moment 數學上不可能為負,
    若出現代表磁碟/記憶體有 bit-level corruption(本案是 sign bit 翻轉),會讓 AdamW
    在那個 element 算 sqrt(negative)=NaN,然後把 weight 寫成 NaN。
    清掉 negative v 時,同位置的 m(exp_avg)也一起清零,避免帶著被汙染的 momentum
    把剛 reset 的 weight element 推到奇怪的方向。
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
        # Pass 1: 處理 NaN/Inf
        for key in list(pstate.keys()):
            new_v, nan_n, inf_n = sanitize_tensor(
                pstate[key], f"{label}.state[{pid}].{key}"
            )
            if nan_n + inf_n > 0:
                pstate[key] = new_v
                bad_total += nan_n + inf_n
        # Pass 2: 處理 exp_avg_sq < 0 (數學上不可能,必為 corruption)
        v = pstate.get("exp_avg_sq")
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            neg_mask = v < 0
            neg_n = int(neg_mask.sum().item())
            if neg_n > 0:
                # 清 v 該位置
                v_new = v.clone()
                v_new[neg_mask] = 0.0
                pstate["exp_avg_sq"] = v_new
                # 同步清 m 該位置,避免汙染的 momentum 繼續主導 update
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

    # Phase 1: 載入並掃描(不寫檔)。先把所有需要的資訊集齊再決定動作。
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

    # 從 optimizer state 取 total_it 當備份檔的步數標籤
    total_it = 0
    if "optimizer_state" in loaded:
        _, payload = loaded["optimizer_state"]
        total_it = int(payload.get("total_it", 0))
    suffix = f".preSanitize_step{total_it}.pth"
    print(f"[sanitize] total_it={total_it}, backup suffix={suffix}")
    print()

    # Phase 2: in-memory 掃描 + 修正。print 出每個被改的位置,沒命中就靜默通過。
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

    # Phase 3: 備份原檔 + 寫回 sanitized 版本。
    # 先備份再寫,避免「備份未完成但 sanitized 已蓋過去」的 race。
    print()
    print("[sanitize] backing up originals...")
    backups = {}
    for name, (path, _payload) in loaded.items():
        backup = path.with_suffix(suffix)
        if backup.exists():
            # 不覆蓋已存在的備份(罕見:已 sanitize 過一次又跑第二次)
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

    # Phase 4: 重新讀一次,驗證真的乾淨了
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
                        # exp_avg_sq 還要驗沒有負值
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
