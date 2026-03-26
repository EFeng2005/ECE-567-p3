# Detailed Replication Matrix

This file is the detailed run-level plan for the OGBench replication.

It complements:

- [targets.md](targets.md)
- [status.md](status.md)
- [scripts/tier1_commands.sh](../../scripts/tier1_commands.sh)
- [scripts/small_matrix_sbatch.sh](../../scripts/small_matrix_sbatch.sh)
- [results/templates/results_template.csv](../../results/templates/results_template.csv)

## What This Matrix Covers

Tier 1 includes five datasets and four core methods:

- `CRL`
- `HIQL`
- `QRL`
- `GCIQL`

We are treating `GCIQL` as the primary representative of the course handout's
`Implicit Q/V Learning` family.
If time allows, we can extend this matrix with `GCIVL`.

## Primary Metric To Record

For these goal-conditioned runs, the most important metric to record is:

- `evaluation/overall_success`

This is written by the official code into `eval.csv`.

## Tier 1 Run List

| Run ID | Category | Dataset | Method | Planned Seeds | Official Config Source | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `T1-ANT-CRL` | Locomotion | `antmaze-large-navigate-v0` | `CRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Standard locomotion reference |
| `T1-ANT-HIQL` | Locomotion | `antmaze-large-navigate-v0` | `HIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | Strong overall baseline |
| `T1-ANT-QRL` | Locomotion | `antmaze-large-navigate-v0` | `QRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Geometry-based baseline |
| `T1-ANT-GCIQL` | Locomotion | `antmaze-large-navigate-v0` | `GCIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | IQL-family representative |
| `T1-HUM-CRL` | Locomotion | `humanoidmaze-medium-navigate-v0` | `CRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Harder long-horizon control |
| `T1-HUM-HIQL` | Locomotion | `humanoidmaze-medium-navigate-v0` | `HIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | Uses discount and subgoal overrides |
| `T1-HUM-QRL` | Locomotion | `humanoidmaze-medium-navigate-v0` | `QRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Uses discount override |
| `T1-HUM-GCIQL` | Locomotion | `humanoidmaze-medium-navigate-v0` | `GCIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | Uses discount override |
| `T1-CUBE-CRL` | Manipulation | `cube-double-play-v0` | `CRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Simple manipulation reference |
| `T1-CUBE-HIQL` | Manipulation | `cube-double-play-v0` | `HIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | Uses `subgoal_steps=10` |
| `T1-CUBE-QRL` | Manipulation | `cube-double-play-v0` | `QRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Strong manipulation comparison |
| `T1-CUBE-GCIQL` | Manipulation | `cube-double-play-v0` | `GCIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | IQL-family comparison |
| `T1-SCENE-CRL` | Manipulation | `scene-play-v0` | `CRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Sequential manipulation |
| `T1-SCENE-HIQL` | Manipulation | `scene-play-v0` | `HIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | Uses `subgoal_steps=10` |
| `T1-SCENE-QRL` | Manipulation | `scene-play-v0` | `QRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Strong manipulation comparison |
| `T1-SCENE-GCIQL` | Manipulation | `scene-play-v0` | `GCIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | IQL-family comparison |
| `T1-PUZ-CRL` | Manipulation | `puzzle-3x3-play-v0` | `CRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Compositional goal task |
| `T1-PUZ-HIQL` | Manipulation | `puzzle-3x3-play-v0` | `HIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | Uses `subgoal_steps=10` |
| `T1-PUZ-QRL` | Manipulation | `puzzle-3x3-play-v0` | `QRL` | `1 -> 3` | `scripts/tier1_commands.sh` | Combinatorial setting |
| `T1-PUZ-GCIQL` | Manipulation | `puzzle-3x3-play-v0` | `GCIQL` | `1 -> 3` | `scripts/tier1_commands.sh` | IQL-family comparison |

## How To Use This Matrix

Recommended execution order:

1. `T1-ANT-HIQL`
2. `T1-CUBE-HIQL`
3. `T1-ANT-GCIQL`
4. `T1-CUBE-GCIQL`
5. Expand to the rest of Tier 1

To submit exactly those first four runs on Great Lakes, use:

- [scripts/small_matrix_sbatch.sh](../../scripts/small_matrix_sbatch.sh)

Fill actual outcomes in:

- [results/templates/results_template.csv](../../results/templates/results_template.csv)
