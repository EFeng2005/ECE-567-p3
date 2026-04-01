# ECE 567 Project 3: Reproducing OGBench Offline GCRL Baselines

Reproduction of offline goal-conditioned reinforcement learning (GCRL) baselines from the **OGBench** benchmark ([Park et al., 2024](https://seohong.me/projects/ogbench/)).

We evaluate **5 algorithms** across **9 datasets** (3 environment categories) with **3 random seeds** each, totaling **135 training runs**.

---

## Algorithms

| Method | Full Name | Core Idea |
|--------|-----------|-----------|
| **GCIVL** | Goal-Conditioned Implicit V-Learning | IQL adapted for goal-conditioned settings (V-function) |
| **GCIQL** | Goal-Conditioned Implicit Q-Learning | IQL adapted for goal-conditioned settings (Q-function) |
| **QRL** | Quasimetric RL | Quasimetric distance-based value estimation |
| **CRL** | Contrastive RL | Contrastive representation learning for goal-conditioned value functions |
| **HIQL** | Hierarchical Implicit Q-Learning | Hierarchical goal-reaching with high/low-level IQL policies |

---

## Datasets

We selected 9 datasets covering the 3 environment categories in the paper, enabling comparisons across goal stitching, stochastic transitions, difficulty scaling, and agent morphology.

| Category | Datasets | Obs Type | Train Steps |
|----------|----------|----------|-------------|
| **Locomotion (Maze)** | `antmaze-large-navigate-v0`, `antmaze-large-stitch-v0`, `antmaze-teleport-navigate-v0`, `humanoidmaze-medium-navigate-v0` | State | 1M |
| **Manipulation** | `cube-double-play-v0`, `scene-play-v0`, `puzzle-3x3-play-v0` | State | 1M |
| **Powderworld (Drawing)** | `powderworld-medium-play-v0`, `powderworld-hard-play-v0` | Pixel | 500K |

**Experiment scale:** 9 datasets x 5 methods x 3 seeds = **135 runs**.

---

## Results

Mean overall success rate (averaged over 3 seeds). **Bold** = best method per dataset.

| Dataset | GCIVL | GCIQL | QRL | CRL | HIQL |
|---------|:-----:|:-----:|:---:|:---:|:----:|
| **Locomotion** | | | | | |
| antmaze-large-navigate | 0.149 | 0.327 | 0.807 | 0.865 | **0.889** |
| antmaze-large-stitch | 0.188 | 0.111 | 0.212 | 0.187 | **0.677** |
| antmaze-teleport-navigate | 0.391 | 0.303 | 0.327 | **0.539** | 0.404 |
| humanoidmaze-medium-navigate | 0.251 | 0.309 | 0.193 | 0.576 | **0.907** |
| **Manipulation** | | | | | |
| cube-double-play | 0.340 | **0.359** | 0.012 | 0.085 | 0.067 |
| scene-play | 0.408 | **0.512** | 0.048 | 0.183 | 0.348 |
| puzzle-3x3-play | 0.048 | **0.937** | 0.007 | 0.036 | 0.112 |
| **Powderworld** | | | | | |
| powderworld-medium-play | **0.529** | 0.173 | 0.017 | 0.029 | 0.131 |
| powderworld-hard-play | **0.048** | 0.000 | 0.000 | 0.000 | 0.033 |

To regenerate this table from raw eval data:

```bash
python3 scripts/extract_results.py
```

Full per-seed breakdown is in `results/summary.csv`.

---

## Repository Structure

```
ECE-567-p3/
├── README.md
├── project3.tex                     # LaTeX report source
├── scripts/
│   ├── extract_results.py           # Aggregate eval results -> summary.csv + table
│   ├── organize_results.py          # Reorganize raw outputs -> results/{dataset}/
│   ├── setup_env.sh                 # One-shot environment setup (venv + deps)
│   ├── bootstrap_ogbench.sh         # Clone upstream OGBench + apply patches
│   ├── train_ogbench.sbatch         # SLURM job template (single GPU)
│   └── submit_experiments.sh        # Submit all 135 runs to SLURM
├── patches/
│   └── ogbench_local.patch          # Adds --dataset_dir and --wandb_mode to OGBench
└── results/
    ├── summary.csv                  # Aggregated: dataset, method, mean, std, n_seeds, per_seed
    └── {dataset}/{method}/seed{i}/
        ├── eval.csv                 # Evaluation metrics (final row = final result)
        ├── train.csv                # Training loss curves
        └── flags.json               # Full hyperparameter config for this run
```

---

## Setup & Reproduction

OGBench source code is not included in this repo to keep the repository lightweight. To set up the environment:

```bash
# 1. Clone and patch OGBench into external/ogbench/
bash scripts/bootstrap_ogbench.sh

# 2. Create venv and install dependencies (on Great Lakes)
bash scripts/setup_env.sh /path/to/venv

# 3. Submit all 135 training runs
export VENV_PATH="/path/to/venv"
bash scripts/submit_experiments.sh
```

The patch (`patches/ogbench_local.patch`) adds two flags to the upstream code:
- `--dataset_dir`: custom dataset cache directory
- `--wandb_mode`: control Weights & Biases logging (online/offline/disabled)

---

## Hyperparameters

All hyperparameters follow the official OGBench `hyperparameters.sh` configurations. Key settings:

- **antmaze-large (navigate/stitch/teleport)**: CRL alpha=0.1, HIQL high/low alpha=3.0, QRL alpha=0.003, GCIQL alpha=0.3, GCIVL alpha=10.0; stitch datasets use `actor_p_randomgoal=0.5, actor_p_trajgoal=0.5`
- **humanoidmaze-medium**: All methods use `discount=0.995`; HIQL uses `subgoal_steps=100`
- **cube-double / scene / puzzle**: CRL alpha=3.0, HIQL alpha=3.0 + `subgoal_steps=10`, QRL alpha=0.3, GCIQL alpha=1.0, GCIVL alpha=10.0
- **powderworld (medium/hard)**: All methods use `batch_size=256, discrete=True, encoder=impala_small, eval_temperature=0.3, train_steps=500K`

Full per-run configs are recorded in each `flags.json`.

---

## References

- OGBench repository: https://github.com/seohongpark/ogbench
- OGBench project page: https://seohong.me/projects/ogbench/
- Paper: Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL", ICLR 2025
