# Great Lakes Notes

This directory contains lightweight templates for launching OGBench jobs on Great Lakes.

Before launching runs, read:

- [environment_setup.md](environment_setup.md)

## Suggested Workflow

1. Clone the upstream OGBench code into `external/ogbench`.
   The bootstrap script also reapplies this repo's local `dataset_dir` and `wandb_mode` training patch.
2. Create or activate your Python environment on Great Lakes.
3. Validate one run interactively.
4. Submit batch jobs with the SLURM template in `slurm/train_ogbench.sbatch`.

For this course allocation, use the Great Lakes Slurm account:

- `ece567w26_class`

That is the real scheduler account string behind the informal course label `ece567-w26`.

## Interactive Sanity Check

Before launching many jobs, validate one run manually on a compute node.

Example outline:

```bash
cd /path/to/ECE-567-p2
bash scripts/bootstrap_ogbench.sh
export MUJOCO_GL=egl
export DATASET_DIR="${SCRATCH:-$HOME/.ogbench/data}"
export SAVE_DIR="$PWD/results/raw/ogbench_runs"
cd external/ogbench/impls
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/hiql.py \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --video_episodes=0 \
  --wandb_mode=offline \
  --dataset_dir="$DATASET_DIR" \
  --save_dir="$SAVE_DIR" \
  --seed=0
```

That smoke test should confirm all of these before you queue more work:

- the run starts successfully
- the dataset auto-downloads
- `train.csv` and `eval.csv` are written
- MuJoCo EGL rendering does not crash
- W&B offline mode does not crash

After the smoke test succeeds, the next two one-seed runs to queue are the official commands for:

- `cube-double-play-v0` + `HIQL`: `--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10`
- `antmaze-large-navigate-v0` + `GCIQL`: `--agent.alpha=0.3`

If you want to submit the first four small-matrix runs in the recommended order, use:

```bash
bash scripts/small_matrix_sbatch.sh
```

That submits:

- `antmaze-large-navigate-v0` + `HIQL`
- `cube-double-play-v0` + `HIQL`
- `antmaze-large-navigate-v0` + `GCIQL`
- `cube-double-play-v0` + `GCIQL`

## Batch Submission

Example:

```bash
sbatch --account=ece567w26_class --partition=spgpu --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/hiql.py,SEED=0,WANDB_MODE=offline,SAVE_DIR=$PWD/results/raw/ogbench_runs,EXTRA_ARGS='--agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0' cluster/greatlakes/slurm/train_ogbench.sbatch
```

Useful exported variables:

- `ENV_NAME`
- `AGENT_PATH`
- `SEED`
- `RUN_GROUP`
- `DATASET_DIR`
- `WANDB_MODE`
- `MUJOCO_GL`
- `SAVE_DIR`
- `EXTRA_ARGS`
- `VENV_PATH`

## Important Habit

Do not launch a broad sweep first.

Start with:

- one dataset
- one method
- one seed

Then expand once logging, checkpoints, and runtime all look correct.
