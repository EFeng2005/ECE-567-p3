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

### 1. Headless MuJoCo rendering

For remote or headless nodes, set:

```bash
export MUJOCO_GL=egl
```

Without this, MuJoCo rendering may fail on the cluster.

### 2. Weights & Biases access

The official training script logs to `wandb` by default, and the helper code initializes it in **online** mode.

That means you should decide one of these before running:

- log in to W&B on Great Lakes
- set up a team account/token for the project
- patch the logging helper to use offline or disabled mode for cluster runs

### 3. Dataset storage path

By default, OGBench downloads datasets to:

```bash
~/.ogbench/data
```

Important caveat:

- the official `impls/main.py` path does **not** expose `dataset_dir` as a flag
- it calls `ogbench.make_env_and_datasets(..., compact_dataset=True)` through `impls/utils/env_utils.py`

So if you want datasets on Great Lakes `scratch`, plan one of these:

- make `~/.ogbench/data` a symlink to scratch storage
- patch `impls/utils/env_utils.py` to pass a custom `dataset_dir`

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
cd external/ogbench/impls
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/hiql.py \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --video_episodes=0 \
  --seed=0
```

## Sources

- [OGBench README](https://github.com/seohongpark/ogbench/blob/master/README.md)
- [OGBench training requirements](https://github.com/seohongpark/ogbench/blob/master/impls/requirements.txt)
- [OGBench training entrypoint](https://github.com/seohongpark/ogbench/blob/master/impls/main.py)

