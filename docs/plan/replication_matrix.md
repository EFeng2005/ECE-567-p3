# Detailed Replication Matrix

This file is the detailed run-level plan for the OGBench replication.

It complements:

- [targets.md](targets.md)
- [status.md](status.md)
- [scripts/tier1_commands.sh](../../scripts/tier1_commands.sh)
- [scripts/submit_phaseA.sh](../../scripts/submit_phaseA.sh)
- [results/templates/results_template.csv](../../results/templates/results_template.csv)

## What This Matrix Covers

Tier 1 includes six datasets and five core methods, executed in three phases:

- **Phase A** (priority): 3 datasets x 5 methods x 1 seed = 15 runs
- **Phase B** (expand seeds): 3 datasets x 5 methods x 3 seeds = 45 runs
- **Phase C** (expand datasets): 6 datasets x 5 methods x 3 seeds = 90 runs

Methods:

- `CRL`
- `HIQL`
- `QRL`
- `GCIQL`
- `GCIVL`

Both `GCIQL` and `GCIVL` represent the course handout's `Implicit Q/V Learning` family.
The six datasets span all three environment categories required: Locomotion, Manipulation, and Powderworld.

## Primary Metric To Record

For these goal-conditioned runs, the most important metric to record is:

- `evaluation/overall_success`

This is written by the official code into `eval.csv`.

## Tier 1 Run List

### Phase A Runs (3 datasets, seed 0 only -- submit first)

| Run ID | Phase | Category | Dataset | Method | Planned Seeds | Official Config Source | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `T1-ANT-CRL` | A | Locomotion | `antmaze-large-navigate-v0` | `CRL` | `0` | `scripts/tier1_commands.sh` | Standard locomotion reference |
| `T1-ANT-HIQL` | A | Locomotion | `antmaze-large-navigate-v0` | `HIQL` | `0` | `scripts/tier1_commands.sh` | Strong overall baseline |
| `T1-ANT-QRL` | A | Locomotion | `antmaze-large-navigate-v0` | `QRL` | `0` | `scripts/tier1_commands.sh` | Geometry-based baseline |
| `T1-ANT-GCIQL` | A | Locomotion | `antmaze-large-navigate-v0` | `GCIQL` | `0` | `scripts/tier1_commands.sh` | IQL-family representative |
| `T1-ANT-GCIVL` | A | Locomotion | `antmaze-large-navigate-v0` | `GCIVL` | `0` | `scripts/tier1_commands.sh` | IVL-family representative |
| `T1-CUBE-CRL` | A | Manipulation | `cube-double-play-v0` | `CRL` | `0` | `scripts/tier1_commands.sh` | Simple manipulation reference |
| `T1-CUBE-HIQL` | A | Manipulation | `cube-double-play-v0` | `HIQL` | `0` | `scripts/tier1_commands.sh` | Uses `subgoal_steps=10` |
| `T1-CUBE-QRL` | A | Manipulation | `cube-double-play-v0` | `QRL` | `0` | `scripts/tier1_commands.sh` | Strong manipulation comparison |
| `T1-CUBE-GCIQL` | A | Manipulation | `cube-double-play-v0` | `GCIQL` | `0` | `scripts/tier1_commands.sh` | IQL-family comparison |
| `T1-CUBE-GCIVL` | A | Manipulation | `cube-double-play-v0` | `GCIVL` | `0` | `scripts/tier1_commands.sh` | IVL-family comparison |
| `T1-PWD-CRL` | A | Powderworld | `powderworld-medium-play-v0` | `CRL` | `0` | `scripts/tier1_commands.sh` | Discrete pixel task, uses `actor_loss=awr` |
| `T1-PWD-HIQL` | A | Powderworld | `powderworld-medium-play-v0` | `HIQL` | `0` | `scripts/tier1_commands.sh` | Discrete pixel, uses `low_actor_rep_grad=True` |
| `T1-PWD-QRL` | A | Powderworld | `powderworld-medium-play-v0` | `QRL` | `0` | `scripts/tier1_commands.sh` | Discrete pixel task, uses `actor_loss=awr` |
| `T1-PWD-GCIQL` | A | Powderworld | `powderworld-medium-play-v0` | `GCIQL` | `0` | `scripts/tier1_commands.sh` | Discrete pixel task, uses `actor_loss=awr` |
| `T1-PWD-GCIVL` | A | Powderworld | `powderworld-medium-play-v0` | `GCIVL` | `0` | `scripts/tier1_commands.sh` | Discrete pixel task |

### Phase B Runs (same 3 datasets, expand to seeds 0, 1, 2)

Rerun all 15 Phase A runs with `SEED=1` and `SEED=2`. Total: 45 runs (including the seed-0 runs from Phase A).

### Phase C Runs (3 additional datasets, seeds 0, 1, 2 -- submit after Phase B)

| Run ID | Phase | Category | Dataset | Method | Planned Seeds | Official Config Source | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `T1-HUM-CRL` | C | Locomotion | `humanoidmaze-medium-navigate-v0` | `CRL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Harder long-horizon control |
| `T1-HUM-HIQL` | C | Locomotion | `humanoidmaze-medium-navigate-v0` | `HIQL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Uses discount and subgoal overrides |
| `T1-HUM-QRL` | C | Locomotion | `humanoidmaze-medium-navigate-v0` | `QRL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Uses discount override |
| `T1-HUM-GCIQL` | C | Locomotion | `humanoidmaze-medium-navigate-v0` | `GCIQL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Uses discount override |
| `T1-HUM-GCIVL` | C | Locomotion | `humanoidmaze-medium-navigate-v0` | `GCIVL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Uses discount override |
| `T1-SCENE-CRL` | C | Manipulation | `scene-play-v0` | `CRL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Sequential manipulation |
| `T1-SCENE-HIQL` | C | Manipulation | `scene-play-v0` | `HIQL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Uses `subgoal_steps=10` |
| `T1-SCENE-QRL` | C | Manipulation | `scene-play-v0` | `QRL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Strong manipulation comparison |
| `T1-SCENE-GCIQL` | C | Manipulation | `scene-play-v0` | `GCIQL` | `0, 1, 2` | `scripts/tier1_commands.sh` | IQL-family comparison |
| `T1-SCENE-GCIVL` | C | Manipulation | `scene-play-v0` | `GCIVL` | `0, 1, 2` | `scripts/tier1_commands.sh` | IVL-family comparison |
| `T1-PUZ-CRL` | C | Manipulation | `puzzle-3x3-play-v0` | `CRL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Compositional goal task |
| `T1-PUZ-HIQL` | C | Manipulation | `puzzle-3x3-play-v0` | `HIQL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Uses `subgoal_steps=10` |
| `T1-PUZ-QRL` | C | Manipulation | `puzzle-3x3-play-v0` | `QRL` | `0, 1, 2` | `scripts/tier1_commands.sh` | Combinatorial setting |
| `T1-PUZ-GCIQL` | C | Manipulation | `puzzle-3x3-play-v0` | `GCIQL` | `0, 1, 2` | `scripts/tier1_commands.sh` | IQL-family comparison |
| `T1-PUZ-GCIVL` | C | Manipulation | `puzzle-3x3-play-v0` | `GCIVL` | `0, 1, 2` | `scripts/tier1_commands.sh` | IVL-family comparison |

## How To Use This Matrix

Recommended execution order:

1. **Phase A**: Submit all 15 Phase A runs (3 datasets x 5 methods, seed 0). These cover antmaze-large-navigate-v0, cube-double-play-v0, and powderworld-medium-play-v0. This gives early signal across all three environment families.
2. **Phase B**: Once Phase A seed-0 results look reasonable, add seeds 1 and 2 for the same 3 datasets (30 more runs, 45 total).
3. **Phase C**: Add the remaining 3 datasets (humanoidmaze, scene, puzzle) with seeds 0, 1, 2 (45 more runs, 90 total).

To submit Phase A on Great Lakes, see the implementation guide:

- [implementation_guide.md](implementation_guide.md)

Fill actual outcomes in:

- [results/templates/results_template.csv](../../results/templates/results_template.csv)
