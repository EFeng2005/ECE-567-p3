# Great Lakes Notes

This directory contains lightweight templates for launching OGBench jobs on Great Lakes.

Before launching runs, read:

- [docs/environment_setup.md](/Users/e.y.feng/Documents/A.Umich/ece%20567/replication/docs/environment_setup.md)

## Suggested Workflow

1. Clone the upstream OGBench code into `external/ogbench`.
2. Create or activate your Python environment on Great Lakes.
3. Validate one run interactively.
4. Submit batch jobs with the SLURM template in `slurm/train_ogbench.sbatch`.

## Interactive Sanity Check

Before launching many jobs, validate one run manually on a compute node.

Example outline:

```bash
cd /path/to/ECE-567-p2
bash scripts/bootstrap_ogbench.sh
cd external/ogbench/impls
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/hiql.py --seed=0
```

## Batch Submission

Example:

```bash
sbatch --export=ALL,ENV_NAME=antmaze-large-navigate-v0,AGENT_PATH=agents/hiql.py,SEED=0 cluster/greatlakes/slurm/train_ogbench.sbatch
```

Useful exported variables:

- `ENV_NAME`
- `AGENT_PATH`
- `SEED`
- `EXTRA_ARGS`
- `VENV_PATH`

## Important Habit

Do not launch a broad sweep first.

Start with:

- one dataset
- one method
- one seed

Then expand once logging, checkpoints, and runtime all look correct.
