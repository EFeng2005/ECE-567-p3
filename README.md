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

- [tracking/targets.md](/Users/e.y.feng/Documents/A.Umich/ece%20567/replication/tracking/targets.md)
- [tracking/status.md](/Users/e.y.feng/Documents/A.Umich/ece%20567/replication/tracking/status.md)
- [docs/replication_scope.md](/Users/e.y.feng/Documents/A.Umich/ece%20567/replication/docs/replication_scope.md)
- [docs/environment_setup.md](/Users/e.y.feng/Documents/A.Umich/ece%20567/replication/docs/environment_setup.md)
- [cluster/greatlakes/README.md](/Users/e.y.feng/Documents/A.Umich/ece%20567/replication/cluster/greatlakes/README.md)

## Repository Layout

```text
cluster/greatlakes/     Great Lakes notes and SLURM scripts
docs/                   Scope, benchmark notes, and writeup support
external/               Upstream code checked out locally (not yet added)
reports/phase1/         Material for the Phase 1 report
results/raw/            Raw run outputs and copied logs
results/processed/      Aggregated tables and parsed summaries
results/figures/        Plots used in the report
scripts/                Helper scripts for bootstrap and syncing
tracking/               What we plan to reproduce vs what we have reproduced
```

## Recommended Workflow

1. Bootstrap the upstream benchmark code into `external/ogbench`.
2. Validate one end-to-end run on a small state-based task.
3. Reproduce a representative subset first.
4. Expand only after we know the training pipeline is stable.

## Upstream References

- [OGBench repository](https://github.com/seohongpark/ogbench)
- [OGBench project page](https://seohong.me/projects/ogbench/)
