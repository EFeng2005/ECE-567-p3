# Environment Setup Checklist

This note summarizes the environment requirements we should prepare **before** launching OGBench runs on Great Lakes.

## Core Version Requirements

- Use **Python 3.9+** for the reference implementations in `impls`.
- The benchmark package itself supports **Python 3.8+**, but the training code explicitly documents **3.9+**.
- If using GPU training, prepare a **CUDA 12 compatible** environment because the official training dependencies use `jax[cuda12] >= 0.4.26`.

## Python Packages

### Benchmark package

The root OGBench package depends on:

- `mujoco >= 3.1.6`
- `dm_control >= 1.0.20`
- `gymnasium[mujoco]`

### Training stack

The official training implementation in `impls/requirements.txt` additionally needs:

- `ogbench`
- `jax[cuda12] >= 0.4.26`
- `flax >= 0.8.4`
- `distrax >= 0.1.5`
- `ml_collections`
- `matplotlib`
- `moviepy`
- `wandb`

## Cluster-Specific Things To Prepare

For the course allocation on Great Lakes, the Slurm account string is:

- `ece567w26_class`

That is the scheduler account you should pass to `sbatch`, even if people casually refer to it as `ece567-w26`.

### 1. Headless MuJoCo rendering

For remote or headless nodes, set:

```bash
export MUJOCO_GL=egl
```

Without this, MuJoCo rendering may fail on the cluster.

### 2. Weights & Biases access

The official training script logs to `wandb` by default, but this repo now exposes a mode flag so cluster runs can stay offline.

Supported modes:

- `--wandb_mode=online`
- `--wandb_mode=offline`
- `--wandb_mode=disabled`

For Great Lakes smoke tests, I recommend starting with:

```bash
--wandb_mode=offline
```

The training entrypoint also respects `WANDB_MODE=online|offline|disabled` so the SLURM template can set this once.

### 3. Dataset storage path

By default, OGBench downloads datasets to:

```bash
~/.ogbench/data
```

Important caveat:

- upstream OGBench still defaults to `~/.ogbench/data`
- this repo now exposes `--dataset_dir=/path/to/data` through `impls/main.py`

So on Great Lakes, you can point downloads at scratch directly:

```bash
--dataset_dir="$SCRATCH/ogbench/data"
```

The training entrypoint also respects `OGBENCH_DATASET_DIR=/path/to/data`.

The bundled SLURM template defaults to `$SCRATCH/ogbench/data` when `SCRATCH` is available, and falls back to `~/.ogbench/data` otherwise.

If you bootstrap OGBench with:

```bash
bash scripts/bootstrap_ogbench.sh
```

the repo's local patch is applied automatically so `impls/main.py` exposes both `--dataset_dir` and `--wandb_mode`.

### 4. RAM and runtime budgeting

According to the official README:

- state-based tasks usually take **2-5 hours** on a single A5000 GPU
- pixel-based tasks usually take **5-12 hours**
- some large pixel datasets may need **up to 120 GB RAM**

So for class-project runs, it is safer to start with **state-based tasks**.

## Configuration Pitfalls To Prepare Ahead Of Time

### Use the exact benchmark flags

For paper-level reproduction, do **not** invent your own defaults.

Use:

- `external/ogbench/impls/hyperparameters.sh`

Several datasets need task-specific overrides for:

- `agent.alpha`
- `agent.discount`
- `agent.actor_p_randomgoal`
- `agent.actor_p_trajgoal`
- HIQL high/low alpha values

### Pixel tasks

For pixel tasks, the README explicitly warns:

- set `--agent.encoder=impala_small`
- for HIQL, set `--agent.low_actor_rep_grad=True`

### Powderworld

For Powderworld:

- set `--agent.discrete=True`
- use `--eval_temperature=0.3`

### Early smoke tests

For first-run debugging, I recommend:

- state-based task only
- one seed
- `--video_episodes=0` to reduce video/logging friction

## Recommended First Smoke Test

Example:

```bash
export MUJOCO_GL=egl
export DATASET_DIR="${SCRATCH:-$HOME/.ogbench/data}"
export SAVE_DIR="$PWD/results/raw/ogbench_runs"
cd external/ogbench/impls
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/hiql.py \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --dataset_dir="$DATASET_DIR" \
  --save_dir="$SAVE_DIR" \
  --video_episodes=0 \
  --wandb_mode=offline \
  --seed=0
```

After that smoke test passes, the next two one-seed runs to queue are:

```bash
python main.py \
  --env_name=cube-double-play-v0 \
  --agent=agents/hiql.py \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --agent.subgoal_steps=10 \
  --dataset_dir="$DATASET_DIR" \
  --save_dir="$SAVE_DIR" \
  --video_episodes=0 \
  --wandb_mode=offline \
  --seed=0

python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/gciql.py \
  --agent.alpha=0.3 \
  --dataset_dir="$DATASET_DIR" \
  --save_dir="$SAVE_DIR" \
  --video_episodes=0 \
  --wandb_mode=offline \
  --seed=0
```

## Sources

- [OGBench README](https://github.com/seohongpark/ogbench/blob/master/README.md)
- [OGBench training requirements](https://github.com/seohongpark/ogbench/blob/master/impls/requirements.txt)
- [OGBench training entrypoint](https://github.com/seohongpark/ogbench/blob/master/impls/main.py)
