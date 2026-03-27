# Replication Scope Recommendation

## Short Answer

A **full** OGBench reproduction is probably too large for a normal course project unless the team has a lot of cluster budget and is explicitly targeting the full benchmark table.

The safer plan is a **representative partial reproduction** first.

## Why Full Reproduction Is Expensive

The official benchmark includes:

- `85` datasets
- `6` reference goal-conditioned methods in the public repo
- results averaged over multiple random seeds

That means the full benchmark is on the order of **hundreds to thousands of training runs**, depending on exactly which slice and seed count you target.

Even if we only followed the four method groups named in the course handout, a rough upper-bound style estimate is:

- `85 datasets x 4 method groups x multiple seeds`

That is already far beyond a lightweight class replication.

## What Counts As A Good Partial Replication

A good partial replication should still cover:

- more than one environment family
- more than one algorithmic style
- at least one task that is not trivial
- enough seeds to show whether trends are stable

That is why the recommended default in this repo starts with **3 datasets first** (Phase A), then expands:

**Phase A** (priority -- 3 datasets, seed 0, 15 runs):

- locomotion: `antmaze-large-navigate-v0`
- manipulation: `cube-double-play-v0`
- powderworld: `powderworld-medium-play-v0`
- methods: `CRL`, `HIQL`, `QRL`, `GCIQL`, `GCIVL`

**Phase B** (expand seeds -- same 3 datasets, seeds 0, 1, 2, 45 runs total):

- Same datasets and methods as Phase A, with 3 seeds for error bars.

**Phase C** (expand datasets -- 6 datasets, seeds 0, 1, 2, 90 runs total):

- Add: `humanoidmaze-medium-navigate-v0`, `scene-play-v0`, `puzzle-3x3-play-v0`

Then, if things go well, add stretch targets:

- `antsoccer`
- `powderworld-hard-play-v0`
- one pixel-based task

## When Full Reproduction Might Be Worth It

Only aim for the full benchmark table if:

- the instructor explicitly expects it
- the team has enough cluster time to rerun failures
- you already have a stable pipeline
- you are prepared to manage many long-running jobs and result aggregation

## Practical Recommendation For Great Lakes

Use a staged plan:

1. One smoke-test run on a small state-based task (interactive session).
2. **Phase A**: All 5 methods across 3 priority datasets, seed 0 only (15 runs).
3. **Phase B**: Expand to seeds 0, 1, 2 for the same 3 datasets (45 runs).
4. **Phase C**: Add 3 more datasets with seeds 0, 1, 2 (90 runs total).
5. Add stretch tasks only if the first four steps are stable.

This is the best tradeoff between scientific credibility and actually finishing the project.

## References

- [OGBench repository](https://github.com/seohongpark/ogbench)
- [OGBench project page](https://seohong.me/projects/ogbench/)

