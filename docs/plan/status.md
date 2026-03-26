# Replication Status

This file tracks what we have actually completed so far.

## Setup Status

| Item | Status | Notes |
| --- | --- | --- |
| Local git repo | `done` | Scaffolded on 2026-03-26 |
| GitHub remote | `done` | `origin` set to `git@github.com:EFeng2005/ECE-567-p2.git` |
| Upstream OGBench checkout | `done` | Present under `external/ogbench` |
| Local OGBench patch automation | `done` | `scripts/bootstrap_ogbench.sh` reapplies `patches/ogbench_local.patch` after clone/pull |
| Great Lakes SLURM template | `done` | Added under `cluster/greatlakes/slurm` |
| Great Lakes dataset override | `done` | `main.py` now accepts `--dataset_dir` and the SLURM template defaults to scratch when available |
| Great Lakes wandb mode override | `done` | `main.py` now accepts `--wandb_mode=online|offline|disabled` |
| Small-matrix submission script | `done` | `scripts/small_matrix_sbatch.sh` submits the first four benchmark runs with cluster defaults |
| Experiment tracking docs | `done` | Targets, status, replication matrix, and results template are all in the repo |

## Benchmark Run Status

| Dataset | Method | Seeds | Status | Notes |
| --- | --- | --- | --- | --- |
| `antmaze-large-navigate-v0` | `HIQL` | `0` | `queued` | Smoke test. Use `--agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0` |
| `cube-double-play-v0` | `HIQL` | `0` | `queued` | Submit after smoke test. Use `--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0` |
| `antmaze-large-navigate-v0` | `GCIQL` | `0` | `queued` | Submit after smoke test. Use `--agent.alpha=0.3 --video_episodes=0` |
| `cube-double-play-v0` | `GCIQL` | `0` | `queued` | Submit after smoke test. Use `--agent.alpha=1.0 --video_episodes=0` |

## Result Summary

No reproduction metrics have been collected yet.

## Current Blockers

- The current local environment is missing the OGBench training dependencies (`jax`, `wandb`, `mujoco`, `dm_control`, `gymnasium`, `ogbench`), so no smoke test has been executed from this workstation yet.
- The next concrete milestone is still the Great Lakes smoke test on `antmaze-large-navigate-v0` with `HIQL`, `seed=0`, `MUJOCO_GL=egl`, `video_episodes=0`, and offline `wandb`.
- We still need the first successful run record with runtime, `overall_success`, and log paths before expanding to the Tier 1 matrix.

When runs start, record at least:

- commit hash
- dataset name
- method
- seed
- runtime
- final success metric
- path to raw logs
- path to processed summary
