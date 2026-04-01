#!/usr/bin/env python3
"""Extract and summarize experiment results from processed eval.csv files.

Walks results/processed/by_dataset/{dataset}/{method}/seed{i}/eval.csv,
reads the final evaluation/overall_success from each run, computes mean and
std across 3 seeds per (dataset, method) pair, and writes:
  1. results/summary.csv  — machine-readable CSV
  2. stdout               — formatted table for quick inspection
"""

import csv
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "results"
OUTPUT_CSV = REPO_ROOT / "results" / "summary.csv"

DATASETS = [
    "antmaze-large-navigate-v0",
    "antmaze-large-stitch-v0",
    "antmaze-teleport-navigate-v0",
    "humanoidmaze-medium-navigate-v0",
    "cube-double-play-v0",
    "scene-play-v0",
    "puzzle-3x3-play-v0",
    "powderworld-medium-play-v0",
    "powderworld-hard-play-v0",
]

METHODS = ["gcivl", "gciql", "qrl", "crl", "hiql"]
SEEDS = ["seed0", "seed1", "seed2"]


def read_final_success(eval_csv: Path) -> float | None:
    """Read the last row's evaluation/overall_success from an eval.csv file."""
    if not eval_csv.exists():
        return None
    with open(eval_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    last_row = rows[-1]
    val = last_row.get("evaluation/overall_success")
    if val is None:
        return None
    return float(val)


def main():
    results = []

    for dataset in DATASETS:
        for method in METHODS:
            seed_values = []
            for seed in SEEDS:
                eval_path = PROCESSED_DIR / dataset / method / seed / "eval.csv"
                val = read_final_success(eval_path)
                if val is not None:
                    seed_values.append(val)
                else:
                    print(f"WARNING: missing {eval_path}", file=sys.stderr)

            if seed_values:
                mean = statistics.mean(seed_values)
                std = statistics.pstdev(seed_values) if len(seed_values) > 1 else 0.0
                per_seed = ",".join(f"{v:.3f}" for v in seed_values)
            else:
                mean, std, per_seed = 0.0, 0.0, ""

            results.append({
                "dataset": dataset,
                "method": method,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "n_seeds": len(seed_values),
                "per_seed": per_seed,
            })

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "mean", "std", "n_seeds", "per_seed"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {OUTPUT_CSV}")

    # Print formatted table
    print()
    header = f"{'Dataset':<36} {'GCIVL':>7} {'GCIQL':>7} {'QRL':>7} {'CRL':>7} {'HIQL':>7}"
    print(header)
    print("-" * len(header))

    for dataset in DATASETS:
        row_vals = []
        for method in METHODS:
            match = [r for r in results if r["dataset"] == dataset and r["method"] == method]
            row_vals.append(match[0]["mean"] if match else 0.0)
        best_idx = row_vals.index(max(row_vals))
        formatted = []
        for i, v in enumerate(row_vals):
            s = f"{v:.3f}"
            if i == best_idx:
                s = f"*{s}"
            formatted.append(s)
        print(f"{dataset:<36} {formatted[0]:>7} {formatted[1]:>7} {formatted[2]:>7} {formatted[3]:>7} {formatted[4]:>7}")

    print()
    print(f"Total combinations: {len(results)} (9 datasets x 5 methods)")
    missing = [r for r in results if r["n_seeds"] < 3]
    if missing:
        print(f"WARNING: {len(missing)} combinations have fewer than 3 seeds:")
        for r in missing:
            print(f"  {r['dataset']} / {r['method']}: {r['n_seeds']} seeds")


if __name__ == "__main__":
    main()
