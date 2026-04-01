#!/usr/bin/env python3
"""Organize all raw OGBench results into a clean directory structure.

Input:  results/raw/ogbench_runs/OGBench/{phaseA,phaseA_run1,...,phaseD_single}/sd*/
Output: results/organized/{dataset}/{method}/seed{N}/  (symlinks to original data)
Also:   results/summary.csv  (one row per dataset x method, with mean±std over seeds)

For duplicates (same env/agent/seed), keeps the run with the latest timestamp.
"""

import json
import os
import glob
import csv
import shutil
from collections import defaultdict

BASE = "results/raw/ogbench_runs/OGBench"
OUT_DIR = "results"
SUMMARY_CSV = "results/summary.csv"

SKIP_DIRS = {"ogbench_runs_backup", "smoke", "v100_test"}


def collect_all_runs():
    """Scan all phase directories and collect run metadata."""
    runs = []
    for phase_dir in sorted(glob.glob(f"{BASE}/*")):
        dirname = os.path.basename(phase_dir)
        if any(skip in dirname for skip in SKIP_DIRS):
            continue
        for run_dir in sorted(glob.glob(f"{phase_dir}/sd*")):
            fp = os.path.join(run_dir, "flags.json")
            ep = os.path.join(run_dir, "eval.csv")
            if not os.path.exists(fp) or not os.path.exists(ep):
                continue
            f = json.load(open(fp))
            env = f["env_name"]
            agent = f["agent"]["agent_name"]
            seed = f["seed"]
            # Parse timestamp from directory name for dedup (later = better)
            ts = os.path.basename(run_dir).split(".")[-1]  # e.g. 20260328_073035
            # Get final overall_success
            with open(ep) as ef:
                lines = ef.readlines()
                last = lines[-1].strip().split(",")
                overall = float(last[-2])
                step = int(last[-1])
            runs.append({
                "env": env,
                "agent": agent,
                "seed": seed,
                "overall": overall,
                "step": step,
                "ts": ts,
                "src": os.path.abspath(run_dir),
                "phase_dir": dirname,
            })
    return runs


def dedup_runs(runs):
    """For duplicate (env, agent, seed), keep the one with latest timestamp."""
    best = {}
    for r in runs:
        key = (r["env"], r["agent"], r["seed"])
        if key not in best or r["ts"] > best[key]["ts"]:
            best[key] = r
    return list(best.values())


def organize(runs):
    """Create organized directory with copies of eval.csv, train.csv, flags.json."""
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    for r in runs:
        dest = os.path.join(OUT_DIR, r["env"], r["agent"], f"seed{r['seed']}")
        os.makedirs(dest, exist_ok=True)
        for fname in ["eval.csv", "train.csv", "flags.json"]:
            src_file = os.path.join(r["src"], fname)
            dst_file = os.path.join(dest, fname)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
        # Write a source pointer for traceability
        with open(os.path.join(dest, "source.txt"), "w") as f:
            f.write(r["src"] + "\n")


def generate_summary(runs):
    """Generate summary CSV with mean±std over seeds for each (dataset, method)."""
    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["env"], r["agent"])].append(r["overall"])

    rows = []
    for (env, agent), values in sorted(grouped.items()):
        import numpy as np
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        seeds_str = ",".join(f"{v:.3f}" for v in sorted(values))
        rows.append({
            "dataset": env,
            "method": agent,
            "mean": f"{mean:.4f}",
            "std": f"{std:.4f}",
            "n_seeds": n,
            "per_seed": seeds_str,
        })

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "mean", "std", "n_seeds", "per_seed"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    print("Collecting all runs...")
    runs = collect_all_runs()
    print(f"  Found {len(runs)} total runs")

    print("Deduplicating...")
    runs = dedup_runs(runs)
    print(f"  {len(runs)} unique runs after dedup")

    # Verify completeness
    from collections import Counter
    by_env = Counter(r["env"] for r in runs)
    for env, cnt in sorted(by_env.items()):
        expected = 15  # 5 methods x 3 seeds
        status = "OK" if cnt == expected else f"MISSING ({cnt}/{expected})"
        print(f"  {env}: {status}")

    print(f"\nOrganizing into {OUT_DIR}/...")
    organize(runs)

    print(f"Generating {SUMMARY_CSV}...")
    rows = generate_summary(runs)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Dataset':<40s} {'CRL':>8s} {'HIQL':>8s} {'QRL':>8s} {'GCIQL':>8s} {'GCIVL':>8s}")
    print("=" * 90)

    from collections import defaultdict
    table = defaultdict(dict)
    for row in rows:
        table[row["dataset"]][row["method"]] = f"{float(row['mean']):.3f}"

    for env in sorted(table):
        cols = [table[env].get(m, "  N/A") for m in ["crl", "hiql", "qrl", "gciql", "gcivl"]]
        print(f"{env:<40s} {'  '.join(cols)}")
    print("=" * 90)
    print(f"\nDone! {len(runs)} runs organized into {OUT_DIR}/")
    print(f"Summary written to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
