# ECE 567 Project 2: OGBench Replication

This repository is the working repo for our replication of **Project 3: OGBench**.

Our current focus is reproducing offline goal-conditioned RL results for the following baseline families:

- `CRL`
- `HIQL`
- `QRL`
- `Implicit Q/V Learning`

In the official OGBench codebase, the last family is usually split into:

- `GCIVL` = Goal-Conditioned Implicit V-Learning
- `GCIQL` = Goal-Conditioned Implicit Q-Learning

## Repo Goals

- Keep the replication scope explicit.
- Separate official benchmark targets from our actual completed runs.
- Make cluster execution on Great Lakes easy to repeat.
- Keep raw outputs, processed summaries, and report artifacts organized.

## Current Status

- Repository scaffolded
- Great Lakes SLURM template added
- Tracking docs added
- No benchmark runs completed yet

See:

- [docs/README.md](docs/README.md)
- [docs/plan/targets.md](docs/plan/targets.md)
- [docs/plan/status.md](docs/plan/status.md)
- [docs/plan/replication_matrix.md](docs/plan/replication_matrix.md)
- [docs/plan/replication_scope.md](docs/plan/replication_scope.md)
- [docs/setup/environment_setup.md](docs/setup/environment_setup.md)
- [docs/setup/greatlakes.md](docs/setup/greatlakes.md)
- [docs/report/phase1.md](docs/report/phase1.md)
- [results/templates/results_template.csv](results/templates/results_template.csv)

## Repository Layout

```text
cluster/greatlakes/     Great Lakes SLURM scripts
docs/                   All project documentation (setup, plan, report)
external/               Upstream code checked out locally (not yet added)
results/raw/            Raw run outputs and copied logs
results/processed/      Aggregated tables and parsed summaries
results/figures/        Plots used in the report
results/templates/      CSV templates for run tracking
scripts/                Helper scripts for bootstrap and syncing
```

## Recommended Workflow

1. Bootstrap the upstream benchmark code into `external/ogbench`.
2. Validate one end-to-end run on a small state-based task.
3. Reproduce a representative subset first.
4. Expand only after we know the training pipeline is stable.

The bootstrap helper also reapplies the repo-local OGBench patch so `--dataset_dir` and `--wandb_mode` stay
available after a fresh clone on Great Lakes.

## Upstream References

- [OGBench repository](https://github.com/seohongpark/ogbench)
- [OGBench project page](https://seohong.me/projects/ogbench/)
