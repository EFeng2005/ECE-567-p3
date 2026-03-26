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

## Recommended Scope

### Tier 1: Core Reproduction

This is the smallest benchmark slice that is still broad enough for a credible Phase 1 report.

| Category | Dataset | Why include it | Methods |
| --- | --- | --- | --- |
| Locomotion | `antmaze-large-navigate-v0` | Standard, widely used navigation task | `CRL`, `HIQL`, `QRL`, `GCIQL` |
| Locomotion | `humanoidmaze-medium-navigate-v0` | Harder long-horizon control | `CRL`, `HIQL`, `QRL`, `GCIQL` |
| Manipulation | `cube-double-play-v0` | Simple manipulation baseline | `CRL`, `HIQL`, `QRL`, `GCIQL` |
| Manipulation | `scene-play-v0` | Sequential reasoning task | `CRL`, `HIQL`, `QRL`, `GCIQL` |
| Manipulation | `puzzle-3x3-play-v0` | Goal composition and combinatorics | `CRL`, `HIQL`, `QRL`, `GCIQL` |

### Tier 2: Stretch Reproduction

Add these only after Tier 1 runs are stable and producing comparable trends.

| Category | Dataset | Why include it |
| --- | --- | --- |
| Locomotion | `antsoccer-medium-navigate-v0` | More difficult control plus object interaction |
| Drawing | `powderworld-medium-play-v0` or `powderworld-hard-play-v0` | Matches the course handout and adds non-robotic generalization |
| Pixel | one visual maze or visual manipulation dataset | Tests the image-based pipeline |
| IQL-family extension | `GCIVL` on Tier 1 datasets | Clarifies the "Implicit Q/V Learning" family |

## Recommended Seed Policy

- Smoke test: `1` seed
- Tier 1 reproduction: `2-3` seeds first
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

