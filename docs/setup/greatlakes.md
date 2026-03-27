# Great Lakes Notes

This directory contains lightweight templates for launching OGBench jobs on Great Lakes.

Before launching runs, read:

- [environment_setup.md](environment_setup.md)

## Compute Account

We use the **`chaijy2`** Slurm account (Dr. Chai's dedicated A40 GPU nodes) instead of `ece567w26_class`.

| | ece567w26_class | chaijy2 |
|---|---|---|
| Max GPU/job | 1 | no hard limit |
| Max wall time | 8 hours | no hard limit |
| Max concurrent GPUs | very limited | up to 20 A40s |
| Partition | spgpu | spgpu |

To check your account access:

```bash
sacctmgr show assoc where user=$USER format=account,partition,qos -p
```

## Storage

| Path | Use for | Size |
|---|---|---|
| `/scratch/chaijy_root/chaijy2/$USER/` | Datasets, run outputs, checkpoints | 10TB shared |
| `/home/$USER/` | Code, venv | 80GB |
| `/nfs/turbo/coe-chaijy/` | Long-term dataset storage | 20TB shared |

**Important**: `/scratch` purges files after 60 days of inactivity.

## Suggested Workflow

1. Clone the upstream OGBench code into `external/ogbench` (bootstrap script).
2. Create or activate your Python environment on Great Lakes.
3. Validate one run interactively.
4. Submit all Phase A runs with `bash scripts/submit_phaseA.sh`.

## Interactive Smoke Test

Before launching many jobs, validate one run manually on a compute node.

```bash
# Request an interactive GPU session
srun --account=chaijy2 --partition=spgpu --gres=gpu:1 \
     --time=6:00:00 --mem=32G --cpus-per-task=8 --pty bash

# Setup environment
source $SCRATCH/ogbench_venv/bin/activate  # or wherever your venv is
export MUJOCO_GL=egl

# Run one smoke test
cd /path/to/ECE-567-p3/external/ogbench/impls
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/hiql.py \
  --eval_episodes=50 \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --video_episodes=0 \
  --wandb_mode=offline \
  --dataset_dir="/scratch/chaijy_root/chaijy2/$USER/ogbench/data" \
  --save_dir="/path/to/ECE-567-p3/results/raw/ogbench_runs" \
  --seed=0
```

Verify before proceeding:

- the run starts and prints training logs
- the dataset auto-downloads to the dataset_dir
- `train.csv` and `eval.csv` are written in the save_dir
- MuJoCo EGL rendering does not crash
- W&B offline mode does not crash

## Submit Phase A (15 runs)

After the smoke test passes:

```bash
bash scripts/submit_phaseA.sh
```

This submits **8 SLURM jobs** (7 x 2-GPU + 1 x 1-GPU) covering all 15 Phase A runs:
- 5 methods on antmaze-large-navigate-v0
- 5 methods on cube-double-play-v0
- 5 methods on powderworld-medium-play-v0

Each 2-GPU job runs two independent experiments in parallel on separate GPUs.

## Monitoring

```bash
# Your jobs
squeue -u $USER

# Detailed view
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"

# Check a job's output
cat ogbench-2gpu-<JOBID>.out

# Past job stats
sacct -u $USER --format=JobID,JobName,Partition,Account,AllocCPUS,ReqMem,MaxRSS,Elapsed,State -X

# Cancel a job
scancel <JOBID>
```

## Manual Single-Job Submission

```bash
sbatch --account=chaijy2 --partition=spgpu \
  --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/hiql.py,SEED=0,WANDB_MODE=offline,DATASET_DIR=/scratch/chaijy_root/chaijy2/$USER/ogbench/data,SAVE_DIR=$PWD/results/raw/ogbench_runs,VENV_PATH=$SCRATCH/ogbench_venv,EXTRA_ARGS='--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0' \
  cluster/greatlakes/slurm/train_ogbench.sbatch
```
