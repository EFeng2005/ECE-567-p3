# Complete Implementation Guide: From Zero to Final Results

This is the step-by-step guide to complete the entire OGBench replication project, from environment setup to final report.

## Overview

- **5 methods**: CRL, HIQL, QRL, GCIQL, GCIVL
- **6 datasets total** (introduced in phases):
  - Phase A (3 datasets): antmaze-large-navigate-v0, cube-double-play-v0, powderworld-medium-play-v0
  - Phase C (+3 datasets): humanoidmaze-medium-navigate-v0, scene-play-v0, puzzle-3x3-play-v0
- **Phase A**: 3 datasets x 5 methods x 1 seed = 15 runs
- **Phase B**: 3 datasets x 5 methods x 3 seeds = 45 runs
- **Phase C**: 6 datasets x 5 methods x 3 seeds = 90 runs
- **Primary metric**: `evaluation/overall_success` from `eval.csv`

---

## Phase 0: Local Machine Preparation (30 min)

You are here. This phase is done on your local machine before touching the cluster.

```bash
# 0.1 Clone the upstream OGBench code
cd ~/projects/ECE-567-p3
bash scripts/bootstrap_ogbench.sh

# 0.2 Verify the patch was applied
grep "dataset_dir" external/ogbench/impls/main.py    # should show the --dataset_dir flag
grep "wandb_mode" external/ogbench/impls/main.py      # should show the --wandb_mode flag

# 0.3 Read the official hyperparameters to confirm they match our tier1_commands.sh
cat external/ogbench/impls/hyperparameters.sh | head -200
```

At this point you do NOT need to install training dependencies locally. All training runs will happen on Great Lakes.

---

## Phase 1: Great Lakes Environment Setup (1-2 hours)

SSH into Great Lakes and set up the environment once.

### 1.1 Clone the repo on Great Lakes

```bash
ssh greatlakes
cd $SCRATCH    # or wherever you keep project code
git clone git@github.com:EFang2005/ECE-567-p3.git
cd ECE-567-p3
```

### 1.2 Bootstrap OGBench

```bash
bash scripts/bootstrap_ogbench.sh
```

### 1.3 Create a Python virtual environment

```bash
module load python/3.10   # or whatever Python 3.9+ is available
module load cuda/12.1      # CUDA 12 for jax[cuda12]

python -m venv $SCRATCH/ogbench_venv
source $SCRATCH/ogbench_venv/bin/activate

# Install the OGBench benchmark package first
cd external/ogbench
pip install -e .

# Install training dependencies
cd impls
pip install -r requirements.txt

# Verify key packages
python -c "import jax; print(jax.devices())"    # should show GPU
python -c "import ogbench; print('ogbench OK')"
python -c "import mujoco; print('mujoco OK')"
```

### 1.4 Set persistent environment variables

Add to your `~/.bashrc` or a `setup_env.sh` script:

```bash
export MUJOCO_GL=egl
export SCRATCH=/scratch/your_uniqname_root/your_uniqname
export OGBENCH_VENV=$SCRATCH/ogbench_venv
export OGBENCH_DATA=$SCRATCH/ogbench/data
export WANDB_MODE=offline
```

---

## Phase 2: Smoke Test (1 run, ~2-5 hours)

Before submitting batch jobs, validate the full pipeline with ONE run.

### 2.1 Interactive smoke test on a compute node

```bash
# Request an interactive GPU session
salloc --account=ece567w26_class --partition=spgpu --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=06:00:00

# Activate environment
source $SCRATCH/ogbench_venv/bin/activate
export MUJOCO_GL=egl

# Run ONE training job
cd $SCRATCH/ECE-567-p3/external/ogbench/impls
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/hiql.py \
  --eval_episodes=50 \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --dataset_dir=$SCRATCH/ogbench/data \
  --save_dir=$SCRATCH/ECE-567-p3/results/raw/ogbench_runs \
  --video_episodes=0 \
  --wandb_mode=offline \
  --seed=0
```

### 2.2 Verify smoke test outputs

After the run completes (typically 2-5 hours for state-based tasks):

```bash
# Check that the output directory was created
ls results/raw/ogbench_runs/OGBench/Debug/

# Check that eval.csv was written
cat results/raw/ogbench_runs/OGBench/Debug/*/eval.csv | tail -5

# Look for overall_success metric
grep "overall_success" results/raw/ogbench_runs/OGBench/Debug/*/eval.csv
```

**If the smoke test fails**: debug before proceeding. Common issues:
- `MUJOCO_GL=egl` not set -> MuJoCo rendering crash
- Missing CUDA -> jax falls back to CPU (very slow)
- Dataset download fails -> check network access or pre-download datasets
- OOM -> increase `--mem` in SLURM

---

## Phase 3: Phase A Runs -- 15 runs (3 datasets x 5 methods x seed 0)

Once the smoke test succeeds, submit all Phase A runs. These cover the 3 priority datasets (antmaze, cube, powderworld) with seed 0 only, giving early signal across all environment families.

### 3.1 Submit Phase A state-based runs (10 runs)

```bash
REPO=$SCRATCH/ECE-567-p3
SBATCH=$REPO/cluster/greatlakes/slurm/train_ogbench.sbatch
COMMON="--account=ece567w26_class --partition=spgpu"
VENV=$SCRATCH/ogbench_venv
DATA=$SCRATCH/ogbench/data
SAVE=$REPO/results/raw/ogbench_runs

# ---- antmaze-large-navigate-v0 (Phase A) ----
sbatch $COMMON --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/crl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.1 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/hiql.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/qrl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.003 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/gciql.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/gcivl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0" $SBATCH

# ---- cube-double-play-v0 (Phase A) ----
sbatch $COMMON --export=ALL,ENV_NAME=cube-double-play-v0,AGENT_PATH=agents/crl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=cube-double-play-v0,AGENT_PATH=agents/hiql.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=cube-double-play-v0,AGENT_PATH=agents/qrl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=cube-double-play-v0,AGENT_PATH=agents/gciql.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=cube-double-play-v0,AGENT_PATH=agents/gcivl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0" $SBATCH
```

### 3.2 Submit Phase A Powderworld runs (5 runs)

Powderworld is pixel-based with discrete actions. It needs different flags and more time/memory.

```bash
REPO=$SCRATCH/ECE-567-p3
SBATCH=$REPO/cluster/greatlakes/slurm/train_ogbench.sbatch
COMMON="--account=ece567w26_class --partition=spgpu --mem=64G --time=18:00:00"
VENV=$SCRATCH/ogbench_venv
DATA=$SCRATCH/ogbench/data
SAVE=$REPO/results/raw/ogbench_runs

# Powderworld common flags
PWD_COMMON="--train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --video_episodes=0"

sbatch $COMMON --export=ALL,ENV_NAME=powderworld-medium-play-v0,AGENT_PATH=agents/crl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=powderworld-medium-play-v0,AGENT_PATH=agents/hiql.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="$PWD_COMMON --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=powderworld-medium-play-v0,AGENT_PATH=agents/qrl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=powderworld-medium-play-v0,AGENT_PATH=agents/gciql.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=powderworld-medium-play-v0,AGENT_PATH=agents/gcivl.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="$PWD_COMMON --agent.alpha=3.0" $SBATCH
```

### 3.3 Monitor jobs

```bash
squeue -u $USER
# Check a specific job's output
cat ogbench-train-<JOB_ID>.out
```

---

## Phase 4: Phase B Runs -- Expand to 3 Seeds (45 runs total)

After Phase A seed-0 results look reasonable, rerun the same 3 datasets with seeds 1 and 2.

Re-run the same sbatch commands from Phase 3 (sections 3.1 and 3.2) with `SEED=1` and then `SEED=2`.

This brings the total to 45 runs (3 datasets x 5 methods x 3 seeds). Only do this if:
- Phase A seed 0 results are stable
- You have enough cluster budget
- You need error bars for the report

---

## Phase 5: Phase C Runs -- Expand to 6 Datasets (90 runs total)

Add the remaining 3 datasets (humanoidmaze, scene, puzzle) with seeds 0, 1, 2.

### 5.1 Submit Phase C state-based runs

```bash
REPO=$SCRATCH/ECE-567-p3
SBATCH=$REPO/cluster/greatlakes/slurm/train_ogbench.sbatch
COMMON="--account=ece567w26_class --partition=spgpu"
VENV=$SCRATCH/ogbench_venv
DATA=$SCRATCH/ogbench/data
SAVE=$REPO/results/raw/ogbench_runs

for SEED in 0 1 2; do

# ---- humanoidmaze-medium-navigate-v0 (Phase C) ----
sbatch $COMMON --export=ALL,ENV_NAME=humanoidmaze-medium-navigate-v0,AGENT_PATH=agents/crl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.1 --agent.discount=0.995 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=humanoidmaze-medium-navigate-v0,AGENT_PATH=agents/hiql.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=humanoidmaze-medium-navigate-v0,AGENT_PATH=agents/qrl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.001 --agent.discount=0.995 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=humanoidmaze-medium-navigate-v0,AGENT_PATH=agents/gciql.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.1 --agent.discount=0.995 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=humanoidmaze-medium-navigate-v0,AGENT_PATH=agents/gcivl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=10.0 --agent.discount=0.995 --video_episodes=0" $SBATCH

# ---- scene-play-v0 (Phase C) ----
sbatch $COMMON --export=ALL,ENV_NAME=scene-play-v0,AGENT_PATH=agents/crl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=scene-play-v0,AGENT_PATH=agents/hiql.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=scene-play-v0,AGENT_PATH=agents/qrl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=scene-play-v0,AGENT_PATH=agents/gciql.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=scene-play-v0,AGENT_PATH=agents/gcivl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0" $SBATCH

# ---- puzzle-3x3-play-v0 (Phase C) ----
sbatch $COMMON --export=ALL,ENV_NAME=puzzle-3x3-play-v0,AGENT_PATH=agents/crl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=puzzle-3x3-play-v0,AGENT_PATH=agents/hiql.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=puzzle-3x3-play-v0,AGENT_PATH=agents/qrl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=puzzle-3x3-play-v0,AGENT_PATH=agents/gciql.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0" $SBATCH
sbatch $COMMON --export=ALL,ENV_NAME=puzzle-3x3-play-v0,AGENT_PATH=agents/gcivl.py,SEED=$SEED,WANDB_MODE=offline,DATASET_DIR=$DATA,SAVE_DIR=$SAVE,VENV_PATH=$VENV,EXTRA_ARGS="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0" $SBATCH

done
```

**Note**: Phase C adds 45 runs (3 datasets x 5 methods x 3 seeds). Combined with Phase B, total is 90 runs.

---

## Phase 6: Collect and Aggregate Results (1-2 hours)

### 6.1 Locate output files

Each run writes to a directory like:
```
results/raw/ogbench_runs/OGBench/<run_group>/<exp_name>/eval.csv
```

### 6.2 Extract the final overall_success

```bash
# Simple script to extract final results from all eval.csv files
cd $SCRATCH/ECE-567-p3
find results/raw -name "eval.csv" | while read f; do
  dir=$(dirname "$f")
  # Get the last line of eval.csv (final evaluation)
  last_line=$(tail -1 "$f")
  echo "$dir: $last_line"
done
```

### 6.3 Fill in the results template

Copy the `overall_success` values into `results/templates/results_template.csv`, updating:
- `status`: `planned` -> `done`
- `overall_success_mean` and `overall_success_std` (if multi-seed)
- `runtime_hours`
- `slurm_job_ids`
- `reproduced_trend`: whether the relative ranking of methods matches the paper

### 6.4 Create summary tables

Build a comparison table like:

| Dataset | CRL | HIQL | QRL | GCIQL | GCIVL |
|---------|-----|------|-----|-------|-------|
| antmaze-large-navigate-v0 | ... | ... | ... | ... | ... |
| humanoidmaze-medium-navigate-v0 | ... | ... | ... | ... | ... |
| cube-double-play-v0 | ... | ... | ... | ... | ... |
| scene-play-v0 | ... | ... | ... | ... | ... |
| puzzle-3x3-play-v0 | ... | ... | ... | ... | ... |
| powderworld-medium-play-v0 | ... | ... | ... | ... | ... |

Compare with the official OGBench paper numbers to discuss reproducibility.

---

## Phase 7: Write the Report (Phase 1: 2-page report)

Requirements from the project spec:
- 2-page report, font size no larger than 11pt
- Brief summary of environment, baseline, and reproducibility
- Public GitHub repo

### Suggested report structure

1. **Introduction** (1/4 page)
   - What is OGBench? Offline goal-conditioned RL benchmark
   - What baselines are we reproducing? CRL, HIQL, QRL, GCIQL, GCIVL

2. **Environment Description** (1/2 page)
   - Locomotion: antmaze, humanoidmaze (state-based, continuous actions)
   - Manipulation: cube, scene, puzzle (state-based, continuous actions)
   - Powderworld: pixel-based, discrete actions
   - Key differences between environment families

3. **Baseline Algorithms** (1/2 page)
   - CRL: Contrastive RL
   - HIQL: Hierarchical Implicit Q-Learning (high-level + low-level)
   - QRL: Quasimetric RL (geometry-based distances)
   - GCIQL/GCIVL: Goal-Conditioned Implicit Q/V Learning (IQL-based)

4. **Results and Reproducibility** (3/4 page)
   - Summary table of our results vs. paper
   - Discussion of which trends reproduced and which didn't
   - Runtime and compute costs
   - Any issues encountered

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Phase 0: Local prep | 30 min | None |
| Phase 1: Cluster setup | 1-2 hours | Phase 0 |
| Phase 2: Smoke test | 2-5 hours | Phase 1 |
| Phase 3: Phase A runs (15 runs, 3 datasets, seed 0) | 1 day (wall time, parallel) | Phase 2 |
| Phase 4: Phase B runs (expand to 3 seeds, 45 total) | 1-2 days | Phase 3 |
| Phase 5: Phase C runs (add 3 datasets, 90 total) | 1-2 days | Phase 4 |
| Phase 6: Result collection | 1-2 hours | Phase 3+ |
| Phase 7: Report writing | 1-2 days | Phase 6 |

**Total**: ~4-7 days end-to-end, assuming reasonable cluster queue times.

---

## Troubleshooting Checklist

| Problem | Solution |
|---------|----------|
| `MUJOCO_GL` error | `export MUJOCO_GL=egl` |
| JAX on CPU only | Load CUDA module, reinstall `jax[cuda12]` |
| Dataset download timeout | Pre-download via `ogbench.make_env_and_datasets(name, dataset_dir=...)` in a Python script |
| OOM on Powderworld | Increase `--mem=64G` or `--mem=96G` in SLURM |
| `wandb` login prompt | Use `--wandb_mode=offline` or `--wandb_mode=disabled` |
| Run finishes but no eval.csv | Check for Python errors in SLURM `.out` file |
| Patch doesn't apply | `cd external/ogbench && git checkout . && git pull && cd ../.. && bash scripts/bootstrap_ogbench.sh` |
