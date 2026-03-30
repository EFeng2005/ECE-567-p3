# Replication Status

This file tracks completed work and the latest collected results.

## Setup Status

| Item | Status | Notes |
| --- | --- | --- |
| Local git repo | `done` | Working tree is on `zyq` with local replication scripts and docs |
| GitHub remote | `done` | `origin` points to `git@github.com:EFeng2005/ECE-567-p2.git` |
| Upstream OGBench checkout | `done` | Present under `external/ogbench` |
| Local OGBench patch automation | `done` | `scripts/bootstrap_ogbench.sh` reapplies `patches/ogbench_local.patch` |
| Great Lakes SLURM templates | `done` | Single-GPU and 2-GPU launchers are under `cluster/greatlakes/slurm` |
| Great Lakes dataset override | `done` | `main.py` accepts `--dataset_dir` and Phase D uses scratch datasets |
| Great Lakes wandb mode override | `done` | `main.py` accepts `--wandb_mode=online|offline|disabled` |
| Python training environment | `done` | Repo venv at `.venv/ogbench` imports `jax`, `mujoco`, `dm_control`, `wandb`, and `ogbench` |
| Scratch dataset downloads | `done` | `scene-play-v0`, `puzzle-3x3-play-v0`, and `powderworld-hard-play-v0` downloaded to scratch |
| Smoke test | `done` | `antmaze-large-navigate-v0 + HIQL + seed 0` produced `train.csv` and `eval.csv` before timeout cutoff |
| Phase D single-GPU launcher | `done` | `scripts/submit_phaseD_single_gpu.sh` submitted one run per GPU |
| Experiment tracking docs | `done` | Targets, status, replication matrix, and results template are all in the repo |

## Completed Experiment Batches

### Smoke Test

| Run | Status | Notes |
| --- | --- | --- |
| `antmaze-large-navigate-v0 + HIQL + seed 0` | `completed for pipeline validation` | Reached `step=200000`, wrote both CSVs, and confirmed dataset download, EGL rendering, and offline wandb. Slurm job `46160505` timed out at 30 minutes by design. |

### Phase D: 3 datasets x 5 methods x 3 seeds = 45 runs

- Account: `mihalcea98`
- Partition: `spgpu`
- Dataset dir: `/scratch/mihalcea_root/mihalcea98/eliotfen/ogbench/data`
- Output root: `/home/eliotfen/A.ece567/replication/ECE-567-p2/results/raw/ogbench_runs/OGBench/phaseD_single`
- Status: `45/45 runs completed`, `0` incomplete outputs, no `Traceback`/`OOM` strings in the collected `ogbench*.out` logs

| Dataset | Method | Seeds | Status | Mean overall_success | Std | Mean runtime (h) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `scene-play-v0` | `CRL` | `0,1,2` | `completed` | `0.1827` | `0.0379` | `4.051` | Per-seed success: `0.160, 0.152, 0.236` |
| `scene-play-v0` | `HIQL` | `0,1,2` | `completed` | `0.3480` | `0.0267` | `4.753` | Per-seed success: `0.320, 0.384, 0.340` |
| `scene-play-v0` | `QRL` | `0,1,2` | `completed` | `0.0480` | `0.0098` | `5.475` | Per-seed success: `0.036, 0.060, 0.048` |
| `scene-play-v0` | `GCIQL` | `0,1,2` | `completed` | `0.5120` | `0.0196` | `3.430` | Strongest Phase D scene result |
| `scene-play-v0` | `GCIVL` | `0,1,2` | `completed` | `0.4080` | `0.0344` | `3.582` | Stable second-tier scene performer |
| `puzzle-3x3-play-v0` | `CRL` | `0,1,2` | `completed` | `0.0360` | `0.0086` | `3.192` | Per-seed success: `0.028, 0.048, 0.032` |
| `puzzle-3x3-play-v0` | `HIQL` | `0,1,2` | `completed` | `0.1120` | `0.0267` | `3.628` | Per-seed success: `0.120, 0.140, 0.076` |
| `puzzle-3x3-play-v0` | `QRL` | `0,1,2` | `completed` | `0.0067` | `0.0050` | `4.141` | Near-zero across all seeds |
| `puzzle-3x3-play-v0` | `GCIQL` | `0,1,2` | `completed` | `0.9373` | `0.0381` | `2.338` | Strongest Phase D puzzle result |
| `puzzle-3x3-play-v0` | `GCIVL` | `0,1,2` | `completed` | `0.0480` | `0.0199` | `3.098` | Low but nonzero across seeds |
| `powderworld-hard-play-v0` | `CRL` | `0,1,2` | `completed` | `0.0000` | `0.0000` | `3.040` | No successful episodes observed |
| `powderworld-hard-play-v0` | `HIQL` | `0,1,2` | `completed` | `0.0333` | `0.0236` | `3.515` | Nonzero on seeds `0` and `1` only |
| `powderworld-hard-play-v0` | `QRL` | `0,1,2` | `completed` | `0.0000` | `0.0000` | `2.695` | No successful episodes observed |
| `powderworld-hard-play-v0` | `GCIQL` | `0,1,2` | `completed` | `0.0000` | `0.0000` | `2.370` | No successful episodes observed |
| `powderworld-hard-play-v0` | `GCIVL` | `0,1,2` | `completed` | `0.0480` | `0.0331` | `2.162` | Best Phase D powderworld result, but still low |

## Verification Notes

- Every Phase D run wrote `eval.csv`, `train.csv`, and `flags.json`.
- `scene-play-v0` and `puzzle-3x3-play-v0` runs all reached `1,000,000` train and eval steps.
- `powderworld-hard-play-v0` runs all reached `500,000` train and eval steps.
- A V100 spot-check on `scene-play-v0 + HIQL + seed 0` finished with final `overall_success=0.288`, which is in the same ballpark as the Phase D A40 run (`0.320`).
- Qualitatively, the Phase D outputs look internally consistent: `GCIQL` dominates `scene` and `puzzle`, while `powderworld-hard` is much harder and only `GCIVL`/`HIQL` are consistently nonzero.

## Current Blockers

- There are no remaining environment or pipeline blockers for the current OGBench setup.
- The next analysis task is to compare these Phase D values against the official benchmark figures or tables and document any trend mismatches.
- If we continue the broader course replication plan, the earlier `antmaze` / `cube` / `humanoidmaze` batches are still pending.
