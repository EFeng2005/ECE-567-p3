# Phase 3c Plan — Grad-Clipped EX-HIQL

> Parallel to [PHASE3_TRAINING_PLAN.md](PHASE3_TRAINING_PLAN.md). Targets the
> σ-creep diagnosed in [PHASE3B_REPORT.md §2](PHASE3B_REPORT.md) with a
> one-line optimizer change plus a checkpointing-cadence change. Two training
> envs × 3 seeds each. Branch: `phase3c-clip` (forked from `expectile-heads`).

---

## 1. Motivation in one paragraph

Phase-3b showed that tightening `head_expectiles` from `(0.1..0.9)` to
`(0.6..0.8)` stops the explosive divergence of Phase-3a but does not stop
drift: `v_std_across_heads` grows monotonically 4.2 → 17 over 800k steps
([PHASE3B_REPORT §2.1](PHASE3B_REPORT.md)). Once σ crosses ~10, the
inference rule `μ − β·σ` stops behaving as a "minor corrective" (β·σ / |μ|
> 15%) and degrades into σ-dominated scoring. Phase-3b's peak at step 400k
does not persist; the step-1M result is at HIQL parity. Phase-3c applies
the single numerical safeguard the Phase-3a postmortem recommended but
that was skipped in Phase-3b: **`optax.clip_by_global_norm(10.0)` on the
shared-trunk optimizer** (see [PHASE3_DESIGN_A_REPORT §7.2](PHASE3_DESIGN_A_REPORT.md)).

Hypothesis: bounded gradient norms prevent the slow magnification of
shared-trunk feature scale that feeds σ-creep, so the Phase-3b peak
should persist through late training. This is testable: if the clip holds
σ steady, step-1M evals should match or exceed the best-step evals; if it
doesn't, σ keeps drifting and we need per-head trunks ([PHASE3_DESIGN_A_REPORT §7.3](PHASE3_DESIGN_A_REPORT.md)).

---

## 2. Changes in this branch

### 2.1 Agent (`external/ogbench/impls/agents/ex_chiql.py`)

**Optimizer wrap in `create()`** — was plain `optax.adam(lr)`; now:

```python
network_tx = optax.chain(
    optax.clip_by_global_norm(config['grad_clip_norm']),
    optax.adam(learning_rate=config['lr']),
)
```

**Config** — new knob `grad_clip_norm=10.0` in `get_config()`. Everything
else unchanged from Phase-3b: `head_expectiles=(0.6, 0.65, 0.7, 0.75, 0.8)`,
`actor_expectile_index_low=actor_expectile_index_high=2` (both read τ=0.7),
`pessimism_beta=0.5`, `num_value_heads=5`, `num_subgoal_candidates=16`.

### 2.2 Runner scripts — checkpoint every 100k

`save_interval` default changed 1M → 100k in all runners so we can
β-sweep and σ-diagnose *intermediate* checkpoints, not only step-1M.
This addresses the [PHASE3B_REPORT §6 item 4](PHASE3B_REPORT.md#L171)
limitation where the step-400k peak weights were not recoverable.

Affects:
- `scripts/run_ex_chiql_local.sh` (WSL local flow)
- `scripts/train_ex_chiql_parallel.sbatch` (new — see below)

### 2.3 New SLURM infrastructure

| File | Purpose |
|---|---|
| [scripts/train_ex_chiql_parallel.sbatch](scripts/train_ex_chiql_parallel.sbatch) | Single sbatch that runs 3 seeds concurrently on 1 GPU via `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`. Mirrors the WSL pattern. For `mihalcea98` + `spgpu`/`gpu` where 1-GPU-per-seed parallelism is available. |
| [scripts/submit_ex_chiql.sh](scripts/submit_ex_chiql.sh) | Wrapper for `train_ex_chiql_parallel.sbatch`. Defaults: mihalcea98 + spgpu + `RUN_GROUP=ex_chiql_phase3b` (tweak for 3c). |
| [scripts/submit_ex_chiql_mig40.sh](scripts/submit_ex_chiql_mig40.sh) | 3 seeds × 2 envs → 6 single-seed sbatchs under `ece567w26_class` + `gpu_mig40`. Class QoS caps at 1 GPU/job, so this uses the serial-jobs pattern rather than 3-on-1 memory slicing. Accepts `ENVS="..."` and `TRAIN_STEPS=...` overrides. |
| [scripts/launch_on_gl1251.sh](scripts/launch_on_gl1251.sh) | Detached launcher (setsid + nohup) for kicking 3 parallel seeds inside an already-running interactive allocation on gl1251, via ssh. Used instead of sbatch when we have a live interactive allocation to burn. |

---

## 3. Run matrix

Two envs, three seeds each, clipped agent, every-100k checkpointing.

| # | Env | HIQL baseline (from results/summary.csv) | Hypothesis |
|---|---|---|---|
| 1 | `antmaze-teleport-navigate-v0` | 0.404 ± 0.035 | Primary target. Explicit stochasticity (teleporters). σ should localize at teleporter-adjacent states; clipped Phase-3c should hold the Phase-3b step-400k peak through step-1M. |
| 2 | `antmaze-large-stitch-v0` | 0.677 ± 0.076 | Secondary. Same antmaze dynamics (hyperparameters transfer). HIQL has ~20pt headroom. Stitching creates planning uncertainty the σ signal may or may not localize; this is an honest test of generality. |

Ruled out from consideration: `antmaze-large-navigate` (HIQL already 0.889,
no headroom), `humanoidmaze-medium` (HIQL 0.907), manipulation envs
(`scene`, `cube-double`, `puzzle-3x3`: not stochastic-dynamics failures),
`powderworld-*` (pixel, different architecture).

---

## 4. Deployment — where the experiments actually run

Constrained by Great Lakes account limits (see memory `slurm_class_walltime.md`):

| Constraint | Value |
|---|---|
| `ece567w26_class` MaxWall (all GPU partitions) | **8 h / job** |
| `ece567w26_class` GrpTRES gres/gpu per user | **1 concurrent GPU** |
| `mihalcea98` priority on GPU partitions | Essentially none (estimates 12+ days out) |
| `gpu_mig40` queue start time under class account | ~1–4 h |

Implications:
- **Can't fit 1M steps in 8h on a 3g.40gb MIG slice** (expected ~10–12h single-process). Submit with `train_steps=1000000` anyway; walltime kills processes and `save_interval=100000` preserves checkpoints up to the last completed multiple of 100k.
- **Can't parallelize across envs under class account** — only 1 GPU concurrent per user. So teleport and stitch run serially, not simultaneously.

### 4.1 Current live deployment

- **Teleport (primary)**: 3 seeds running *inside* an interactive `gpu_mig40` allocation (job 48517457 on `gl1251`), via `scripts/launch_on_gl1251.sh`. `train_steps=1000000`, 3-way XLA memory-slice (`FRACTION=0.30`), per-seed effective throughput ~1/3 of a full MIG slice.
- **Stitch (secondary)**: 3 sbatchs queued (`ex_as_s{0,1,2}`, jobs 48519436–38) on `ece567w26_class` + `gpu_mig40`. Currently pending `AssocGrpGRES` (waiting for the interactive allocation to release the 1-GPU slot). Each sbatch requests 7:55:00 walltime, 1 MIG slice, `train_steps=1000000`.

### 4.2 Expected timeline

| t (from now) | Event |
|---|---|
| 0h | Teleport seeds training (3-way GPU sharing on gl1251) |
| ~7h | Interactive 8h walltime expires → teleport killed → checkpoints persist to `scratch/ex_chiql_phase3c/`, probably ~300-500k per seed due to 3-way share |
| ~7h | Stitch seed 0 sbatch starts (full MIG slice, single process) |
| ~15h | Stitch seed 0 walltime → checkpoints ~700-800k |
| ~15h | Stitch seed 1 starts |
| ~23h | Stitch seed 1 walltime |
| ~23h | Stitch seed 2 starts |
| ~31h | Stitch seed 2 walltime → all runs complete |

Teleport runs reach fewer training steps due to compute sharing, but
every-100k checkpointing means we still get a full training trajectory
per seed — just truncated at ~step 400k-ish. That window covers
Phase-3b's peak (400k) and the σ-creep onset (500-600k), which is what
we actually need to compare.

---

## 5. Success criteria (what we'll report)

### 5.1 Per-env, per-step summary

For each env, from `scratch/ex_chiql_phase3c/<env>/sd00{0,1,2}/eval.csv`:

- **Per-seed step-by-step trajectory** (every 100k).
- **Three aggregate statistics per method** (per [PHASE3B_REPORT §5 Framing] and the recommendation from our earlier analysis):
  1. Final-step mean ± seed std
  2. Best-step mean ± seed std (per seed take max-over-steps, then average)
  3. 700k-1M (or "last 4 checkpoints") window mean ± seed std

- **Same stats for HIQL baseline** pulled from `results/antmaze-{teleport-navigate,large-stitch}-v0/hiql/seed*/eval.csv`.

### 5.2 Mechanism diagnostics (critical for phase-3c story)

After training, for each seed × env:

- **σ trajectory**: `train.csv`'s `value/v_std_across_heads` vs step. Compare to Phase-3a (exploded) and Phase-3b (sub-linear creep). Expected for Phase-3c: **flat or much slower creep**.
- **grad/norm trajectory**: should stay near the clip threshold (10.0) once training stabilizes.
- **β-sweep at step-1M** (if reachable) and at step-400k: use `scripts/eval_beta_sweep.py` on each saved checkpoint. Expected per the σ-creep theory:
  - At step ~400k (σ healthy): β=0.5 ≥ β=0 (mechanism helping)
  - At step ~1M *if σ still healthy*: β=0.5 still ≥ β=0
  - At step ~1M *if σ still drifts despite clip*: β=0 > β=0.5 (mechanism degrading)
- **σ-localization diagnostic** (`scripts/diagnose_sigma.py`) on step-1M checkpoints, re-pointed at `EXCHIQLAgent`: `per_state_mu_sig_corr_mean` should be near 0 (structurally unbiased σ). Phase-2 was −0.44, Phase-3a was +0.99, Phase-3b is unknown.

### 5.3 Decision matrix

| Clip holds σ? (mech. flat) | Perf vs HIQL | Next |
|---|---|---|
| yes | ≥ +5 pts at step 1M | Ship Phase-3c as final result. |
| yes | parity or < +5 pts | Mechanism clean; env just hard. Try LCB training target ([PHASE3_DESIGN_A_REPORT §7.4](PHASE3_DESIGN_A_REPORT.md#L300)). |
| no (σ still drifts) | any | Shared trunk is the structural bottleneck. Move to per-head trunks ([PHASE3_DESIGN_A_REPORT §7.3](PHASE3_DESIGN_A_REPORT.md#L290)). |

---

## 6. Analysis pipeline (after training)

1. Aggregate `eval.csv` across the 6 runs → write `PHASE3C_REPORT.md` with tables parallel to `PHASE3B_REPORT.md`.
2. Re-evaluate final checkpoints with `eval_episodes=500` (10× default) to cut per-seed CI from ±6 pts to ±3 pts — cheap CPU work.
3. β-sweep on step-400k, step-700k, step-1M (or closest reached) for each seed.
4. σ-diagnostic on step-1M (or closest reached) for each seed.
5. Compare across methods: HIQL (existing), C-HIQL Phase-2, EX-HIQL Phase-3a, EX-HIQL Phase-3b, EX-HIQL Phase-3c on one table per env.

---

## 7. File pointers

- **Branch**: `phase3c-clip` (commits `db49331`, `d4dede2`).
- **Agent**: [external/ogbench/impls/agents/ex_chiql.py](external/ogbench/impls/agents/ex_chiql.py) (grad_clip_norm=10.0).
- **Runner scripts**: [scripts/launch_on_gl1251.sh](scripts/launch_on_gl1251.sh), [scripts/submit_ex_chiql_mig40.sh](scripts/submit_ex_chiql_mig40.sh).
- **Live outputs** (symlinked into workspace): [scratch/phase3c_logs/](scratch/phase3c_logs/), [scratch/ex_chiql_phase3c/](scratch/ex_chiql_phase3c/).
- **Related docs**: [EXPECTILE_HIQL.md](EXPECTILE_HIQL.md), [PHASE3_TRAINING_PLAN.md](PHASE3_TRAINING_PLAN.md), [PHASE3_DESIGN_A_REPORT.md](PHASE3_DESIGN_A_REPORT.md), [PHASE3B_REPORT.md](PHASE3B_REPORT.md), [PHASE3B_INTERIM_NOTES.md](PHASE3B_INTERIM_NOTES.md).

---

## 8. Summary

Phase-3c is the minimum-change follow-up to Phase-3b: one optimizer line
(gradient clipping) and one flag (save every 100k). Runs 3 seeds on each
of two antmaze envs under the class account's 1-GPU-per-user cap, which
serializes the two envs into back-to-back batches over ~31h. The run
produces intermediate checkpoints suitable for β-sweep and σ-diagnostic
analysis, which was not possible on Phase-3b's step-1M-only saves. If
clipping holds σ steady, the expected outcome is Phase-3b's step-400k peak
(0.439 on teleport) persisting to step-1M, making the comparison vs
HIQL defensible under the standard step-1M reporting convention.
