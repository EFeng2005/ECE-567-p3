# ECE 567 Project 3: Reproducing OGBench Offline GCRL Baselines

Reproduction of offline goal-conditioned reinforcement learning (GCRL) baselines from the **OGBench** benchmark ([Park et al., 2024](https://seohong.me/projects/ogbench/)).

We evaluate **5 algorithms** across **9 datasets** (3 environment categories) with **3 random seeds** each, totaling **135 training runs**.

---

## 1. Background

OGBench is a benchmark for offline GCRL that spans 85 dataset–task combinations across locomotion mazes, robotic manipulation, and pixel-based drawing environments. The original paper reports performance for 6 methods; we reproduce the following 5:

| Method | Full Name | Core Idea |
|--------|-----------|-----------|
| **CRL** | Contrastive RL | Contrastive representation learning for goal-conditioned value functions |
| **HIQL** | Hierarchical Implicit Q-Learning | Hierarchical goal-reaching with high/low-level IQL policies |
| **QRL** | Quasimetric RL | Quasimetric distance-based value estimation |
| **GCIQL** | Goal-Conditioned Implicit Q-Learning | IQL adapted for goal-conditioned settings (Q-function) |
| **GCIVL** | Goal-Conditioned Implicit V-Learning | IQL adapted for goal-conditioned settings (V-function) |

---

## 2. Dataset Selection

We selected 9 datasets to cover the 3 environment categories in the paper and to support 4 key analysis dimensions from the original study:

### 2.1 Category Coverage

| Category | Datasets | Obs Type | Train Steps |
|----------|----------|----------|-------------|
| **Locomotion (Maze)** | `antmaze-large-navigate-v0`, `antmaze-large-stitch-v0`, `antmaze-teleport-navigate-v0`, `humanoidmaze-medium-navigate-v0` | State | 1M |
| **Manipulation** | `cube-double-play-v0`, `scene-play-v0`, `puzzle-3x3-play-v0` | State | 1M |
| **Powderworld (Drawing)** | `powderworld-medium-play-v0`, `powderworld-hard-play-v0` | Pixel | 500K |

### 2.2 Analysis Dimensions

These 9 datasets were chosen to enable the following comparisons from the paper:

1. **Cross-category comparison** — Locomotion vs. Manipulation vs. Powderworld: do methods generalize across environment types?
2. **Navigate vs. Stitch (Goal stitching)** — `antmaze-large-navigate` vs. `antmaze-large-stitch`: does the method degrade when optimal paths require stitching sub-trajectories not seen in the data?
3. **Stochastic transitions** — `antmaze-large-navigate` vs. `antmaze-teleport-navigate`: how robust are methods to stochastic environment dynamics?
4. **Difficulty gradient** — `powderworld-medium` vs. `powderworld-hard`: how do pixel-based methods scale with task complexity?
5. **Morphology** — `antmaze` (simple ant) vs. `humanoidmaze` (high-dimensional humanoid): does performance hold with more complex agents?

### 2.3 Experiment Scale

| | Count |
|---|---|
| Datasets | 9 |
| Methods | 5 |
| Seeds per experiment | 3 (seed 0, 1, 2) |
| **Total runs** | **135** |

---

## 3. Results Summary

Mean overall success rate (averaged over 3 seeds):

| Dataset | CRL | HIQL | QRL | GCIQL | GCIVL |
|---------|:---:|:----:|:---:|:-----:|:-----:|
| **Locomotion** | | | | | |
| antmaze-large-navigate | 0.865 | **0.889** | 0.807 | 0.327 | 0.149 |
| antmaze-large-stitch | 0.187 | **0.677** | 0.212 | 0.111 | 0.188 |
| antmaze-teleport-navigate | **0.539** | 0.404 | 0.327 | 0.303 | 0.391 |
| humanoidmaze-medium-navigate | 0.576 | **0.907** | 0.193 | 0.309 | 0.251 |
| **Manipulation** | | | | | |
| cube-double-play | 0.085 | 0.067 | 0.012 | **0.359** | 0.340 |
| scene-play | 0.183 | 0.348 | 0.048 | **0.512** | 0.408 |
| puzzle-3x3-play | 0.036 | 0.112 | 0.007 | **0.937** | 0.048 |
| **Powderworld** | | | | | |
| powderworld-medium-play | 0.029 | 0.131 | 0.017 | 0.173 | **0.529** |
| powderworld-hard-play | 0.000 | 0.033 | 0.000 | 0.000 | **0.048** |

Bold = best method per dataset. Full per-seed breakdown is in `results/summary.csv`.

---

## 4. Results Directory Structure

```
results/
├── summary.csv                          # Aggregated table: mean, std, per-seed values
│                                        #   Columns: dataset, method, mean, std, n_seeds, per_seed
│
├── processed/
│   └── by_dataset/                      # Clean structure: one dir per (dataset, method, seed)
│       ├── antmaze-large-navigate-v0/
│       │   ├── crl/
│       │   │   ├── seed0/
│       │   │   │   ├── eval.csv         # Evaluation metrics at each eval checkpoint
│       │   │   │   ├── train.csv        # Training loss curves
│       │   │   │   ├── flags.json       # Full hyperparameter config
│       │   │   │   └── source.txt       # Pointer to raw run directory
│       │   │   ├── seed1/
│       │   │   └── seed2/
│       │   ├── hiql/
│       │   ├── qrl/
│       │   ├── gciql/
│       │   └── gcivl/
│       ├── antmaze-large-stitch-v0/
│       │   └── ...                      # Same structure
│       └── ... (9 datasets total)
│
├── raw/
│   └── ogbench_runs/OGBench/            # Original SLURM outputs (unmodified)
│       ├── phaseA_run1/                 # Phase A, 2-GPU job GPU0 outputs
│       ├── phaseA_run2/                 # Phase A, 2-GPU job GPU1 outputs
│       ├── phaseA/                      # Phase A, 1-GPU job outputs
│       ├── phaseB_run1/                 # Phase B, seed 1
│       ├── phaseB_run2/                 # Phase B, seed 2
│       ├── phaseC_run1/                 # Phase C, GPU0 outputs
│       ├── phaseC_run2/                 # Phase C, GPU1 outputs
│       ├── phaseC/                      # Phase C, 1-GPU job outputs
│       └── phaseD_single/              # Phase D (teammate), single-GPU outputs
│
├── figures/                             # Plots for report (to be generated)
└── templates/
    └── results_template.csv
```

### Key Files

| File | Format | Description |
|------|--------|-------------|
| `results/summary.csv` | CSV | One row per (dataset, method). Columns: `dataset`, `method`, `mean`, `std`, `n_seeds`, `per_seed` |
| `*/eval.csv` | CSV | Columns: `evaluation/task1_success`, ..., `evaluation/task5_success`, `evaluation/overall_success`, `step`. Last row = final result |
| `*/train.csv` | CSV | Training metrics (losses, gradient norms, etc.) logged every 5K steps |
| `*/flags.json` | JSON | Complete hyperparameter configuration. Key fields: `env_name`, `agent.agent_name`, `seed`, `agent.alpha`, etc. |

### How to Read Results

```python
import pandas as pd

# Quick summary table
df = pd.read_csv('results/summary.csv')
pivot = df.pivot(index='dataset', columns='method', values='mean')
print(pivot)

# Read a specific run's eval curve
eval_df = pd.read_csv('results/processed/by_dataset/antmaze-large-navigate-v0/crl/seed0/eval.csv')
final_success = eval_df['evaluation/overall_success'].iloc[-1]
```

---

## 5. Hyperparameters

All hyperparameters follow the official OGBench `hyperparameters.sh` configurations. Key settings:

- **antmaze-large (navigate/stitch/teleport)**: CRL α=0.1, HIQL high/low α=3.0, QRL α=0.003, GCIQL α=0.3, GCIVL α=10.0
  - stitch datasets additionally use `actor_p_randomgoal=0.5, actor_p_trajgoal=0.5`
- **humanoidmaze-medium**: All methods use `discount=0.995`; HIQL uses `subgoal_steps=100`
- **cube-double / scene / puzzle**: CRL α=3.0, HIQL α=3.0 + `subgoal_steps=10`, QRL α=0.3, GCIQL α=1.0, GCIVL α=10.0
- **powderworld (medium/hard)**: All methods use `batch_size=256, discrete=True, encoder=impala_small, eval_temperature=0.3, train_steps=500K`

Full per-run configs are recorded in each `flags.json`.

---

## 6. Repository Layout

```
ECE-567-p3/
├── README.md                    # This file
├── cluster/greatlakes/slurm/    # SLURM sbatch templates (1-GPU and 2-GPU)
├── docs/                        # Setup guides, planning docs, report drafts
├── external/ogbench/            # Upstream OGBench code (git-ignored, cloned via bootstrap)
├── patches/                     # Local patches for OGBench (--dataset_dir, --wandb_mode)
├── results/                     # All experiment outputs (see Section 4)
├── scripts/
│   ├── setup_env.sh             # One-shot environment setup
│   ├── bootstrap_ogbench.sh     # Clone upstream OGBench + apply patches
│   ├── submit_phase{A,B,C,D}.sh # SLURM submission scripts for each phase
│   ├── organize_results.py      # Reorganize raw outputs → processed/by_dataset/
│   └── monitor_and_report.sh    # Background job monitoring
└── agents/                      # Symlink or path to OGBench agent implementations
```

---

## 7. Upstream References

- OGBench repository: https://github.com/seohongpark/ogbench
- OGBench project page: https://seohong.me/projects/ogbench/
- Paper: Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL", ICLR 2025
