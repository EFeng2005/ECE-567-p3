"""Summarize eval.csv + flags.json across configs on the current branch."""
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "results", "antmaze-teleport-navigate-v0")

CONFIGS = [
    ("hiql", "HIQL baseline"),
    ("chiql_phase2", "Phase-2 CHIQL (indep trunks, same tau=0.7)"),
    ("ex_chiql_phase3_design_a", "Phase-3a (indep trunks, wide tau 0.1..0.9)"),
    ("ex_chiql_phase3b", "Phase-3b (indep trunks, per-head tau 0.6..0.8)"),
    ("shared_trunk_chiql", "Shared-trunk CHIQL (1 trunk + 5 heads, same tau=0.7)"),
    ("shared_trunk_ex_chiql", "Shared-trunk EX-HIQL (1 trunk + 5 heads, per-head tau)"),
]


def read_eval(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    hdr = rows[0]
    try:
        os_idx = hdr.index("evaluation/overall_success")
        step_idx = hdr.index("step")
    except ValueError:
        return None
    out = []
    for r in rows[1:]:
        if len(r) <= max(os_idx, step_idx):
            continue
        try:
            st = int(r[step_idx])
            ov = float(r[os_idx])
            out.append((st, ov))
        except Exception:
            continue
    return out


def read_beta(flags_path):
    if not os.path.isfile(flags_path):
        return None
    with open(flags_path) as f:
        d = json.load(f)
    # try both nested and flat forms
    for k in ("agent.pessimism_beta", "pessimism_beta"):
        if k in d:
            return d[k]
    # nested under 'agent'
    if "agent" in d and isinstance(d["agent"], dict):
        return d["agent"].get("pessimism_beta")
    return None


def read_head_expectiles(flags_path):
    if not os.path.isfile(flags_path):
        return None
    with open(flags_path) as f:
        d = json.load(f)
    for k in ("agent.head_expectiles", "head_expectiles"):
        if k in d:
            return d[k]
    if "agent" in d and isinstance(d["agent"], dict):
        return d["agent"].get("head_expectiles")
    return None


for key, desc in CONFIGS:
    cfg_root = os.path.join(ROOT, key)
    print(f"=== {key}: {desc} ===")
    if not os.path.isdir(cfg_root):
        print("  (no results dir on this branch)")
        print()
        continue

    # β / head_expectiles from seed0 flags
    flags0 = os.path.join(cfg_root, "seed0", "flags.json")
    beta = read_beta(flags0)
    hx = read_head_expectiles(flags0)
    print(f"  training pessimism_beta: {beta}")
    if hx is not None:
        print(f"  head_expectiles:         {hx}")

    # Collect per-seed evals
    per_seed_last = []
    step_to_vals = {}
    for s in range(3):
        evp = os.path.join(cfg_root, f"seed{s}", "eval.csv")
        evs = read_eval(evp)
        if evs is None:
            print(f"  seed{s}: no eval.csv")
            continue
        last_step, last_ov = evs[-1]
        best = max(evs, key=lambda t: t[1])
        print(f"  seed{s}: step-1M ({last_step}) overall={last_ov:.4f}  peak={best[1]:.4f}@{best[0]}")
        per_seed_last.append(last_ov)
        for st, ov in evs:
            step_to_vals.setdefault(st, []).append(ov)

    # 3-seed mean step-1M
    if per_seed_last:
        mean_last = sum(per_seed_last) / len(per_seed_last)
        print(f"  3-seed mean at step 1M: {mean_last:.4f}")

    # 3-seed mean best-at-common-step
    best_st, best_m = None, -1
    for st, vs in step_to_vals.items():
        if len(vs) < 3:
            continue
        m = sum(vs) / len(vs)
        if m > best_m:
            best_m, best_st = m, st
    if best_st is not None:
        print(f"  3-seed mean peak (all-3-seeds step): {best_m:.4f}@step{best_st}")
    print()
