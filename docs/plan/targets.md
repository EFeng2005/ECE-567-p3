# Replication Targets

This file tracks what we are aiming to reproduce, not what is already done.

## Baseline Mapping

The course handout lists four baseline groups:

- `CRL`
- `HIQL`
- `QRL`
- `Implicit Q/V Learning`

In the official OGBench repo, that last group is represented by:

- `GCIVL`
- `GCIQL`

## Recommended Scope (Phased Execution)

Execution is split into three phases to get early signal before committing full compute.

### Phase A (Priority): 3 datasets x 5 methods x 1 seed = 15 runs

This is the minimum viable slice. Run these first with seed 0 only.

| Category | Dataset | Why include it | Methods |
| --- | --- | --- | --- |
| Locomotion | `antmaze-large-navigate-v0` | Standard, widely used navigation task | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |
| Manipulation | `cube-double-play-v0` | Simple manipulation baseline | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |
| Powderworld | `powderworld-medium-play-v0` | Discrete pixel-based non-robotic task | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |

### Phase B (Expand Seeds): 3 datasets x 5 methods x 3 seeds = 45 runs

After Phase A succeeds, rerun the same 3 datasets with seeds 0, 1, 2 for error bars.

### Phase C (Expand Datasets): 6 datasets x 5 methods x 3 seeds = 90 runs

Add 3 more datasets to reach the full Tier 1 table.

| Category | Dataset | Phase | Why include it | Methods |
| --- | --- | --- | --- | --- |
| Locomotion | `antmaze-large-navigate-v0` | A | Standard, widely used navigation task | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |
| Manipulation | `cube-double-play-v0` | A | Simple manipulation baseline | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |
| Powderworld | `powderworld-medium-play-v0` | A | Discrete pixel-based non-robotic task | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |
| Locomotion | `humanoidmaze-medium-navigate-v0` | C | Harder long-horizon control | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |
| Manipulation | `scene-play-v0` | C | Sequential reasoning task | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |
| Manipulation | `puzzle-3x3-play-v0` | C | Goal composition and combinatorics | `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL` |

### Stretch Reproduction

Add these only after Phase C runs are stable and producing comparable trends.

| Category | Dataset | Why include it |
| --- | --- | --- |
| Locomotion | `antsoccer-medium-navigate-v0` | More difficult control plus object interaction |
| Powderworld | `powderworld-hard-play-v0` | Harder variant of Powderworld |
| Pixel | one visual maze or visual manipulation dataset | Tests the image-based pipeline |

## Recommended Seed Policy

- Phase A: seed `0` only (fast validation)
- Phase B: seeds `0, 1, 2` (error bars for the 3 priority datasets)
- Phase C: seeds `0, 1, 2` for the additional 3 datasets
- Stronger claim for final comparison: `5+` seeds if compute allows

We should only chase the full official seed count after we confirm the pipeline is stable.

## What We Are Not Treating As The Default Goal

We are **not** treating the full OGBench benchmark as the initial class-project target.

Reasons:

- OGBench contains `85` datasets.
- The official benchmark reports aggregate results across multiple seeds.
- Full-table reproduction is much more like a benchmarking project than a course replication milestone.

The default assumption for this repo is:

- reproduce a representative subset first
- document trends and reproducibility clearly
- expand only if compute and time allow

