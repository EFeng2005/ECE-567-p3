# Phase-3b Interim Notes (2026-04-22, in progress)

> **Live-run notes** captured while training is still in flight.
> `head_expectiles=(0.6, 0.65, 0.7, 0.75, 0.8)`, β_pes=0.5, 3 seeds on `antmaze-teleport-navigate-v0`.
> Commit: `2f13adc` on branch `expectile-heads`.

## 1. Training state at t ≈ 3h30m

| Seed | Step | v_std_across_heads | grad/norm |
|---|---|---|---|
| 0 | 700,000 | 9.56 | 280 |
| 1 | 600,000 | 11.35 | 350 |
| 2 | 600,000 | 10.76 | 634 |

**σ trajectory (seed 0 as reference):** 4.2 (step 40k) → 4.3 (100k) → 4.6 (200k) → 6.2 (300k) → 7.7 (400k) → 8.7 (500k) → 9.3 (600k) → 9.6 (700k). **Monotonic drift, slow and sub-linear.** Growth ≈ +1 per 100k steps.

**Compare to Phase-3a:** at step 700k, Phase-3a was at v_std ≈ 700-1,000 and grad/norm ≈ 177,000. Phase-3b at step 700k is v_std ≈ 9.6 and grad/norm ≈ 280. About **75× smaller σ and 600× smaller gradient norms** than Phase-3a at the same training depth. The tight τ spread unambiguously fixes the numerical divergence.

## 2. Evals so far

| Seed | 100k | 200k | 300k | 400k | 500k | 600k | 700k |
|---|---|---|---|---|---|---|---|
| 0 | 0.448 | 0.420 | 0.348 | 0.376 | 0.420 | **0.388** | _eval_ |
| 1 | 0.384 | 0.412 | 0.436 | **0.472** | 0.412 | _eval_ | — |
| 2 | 0.376 | 0.356 | **0.488** | **0.468** | 0.368 | _eval_ | — |
| **Mean** | 0.403 | 0.396 | 0.424 | **0.439** | 0.400 | — | — |

**Peak**: step-400k mean = **0.439** — at that step, seeds 1 (0.472) and 2 (0.468) were both above HIQL's best-seed-final (0.440). Didn't persist: at step 500k the mean collapsed to 0.400.

## 3. Comparison to HIQL at matching steps

| Step | HIQL mean | Phase-3b mean | Δ |
|---|---|---|---|
| 100k | 0.393 | 0.403 | +1.0 |
| 300k | 0.423 | 0.424 | +0.1 |
| 400k | 0.409 | **0.439** | **+3.0** |
| 500k | 0.404 | 0.400 | −0.4 |

Tracking near-parity throughout, with one positive peak at 400k that hasn't yet repeated.

**For context, HIQL's own trajectory on this env is noisy:** seed 2 alone varied 0.504 (step 300k) → 0.472 (600k) → 0.416 (1M). ±5 pt swings per seed are just how this env evaluates.

## 4. Interpretation (honest, incomplete)

- **Mechanism verdict is positive so far**: tight τ successfully prevents the wide-τ explosion. Numerical internals look HIQL-like, not Phase-3a-like. The σ signal has a chance of being meaningful this time; diagnostic will confirm or refute after training.
- **Performance verdict is tied**: no durable improvement over HIQL through step 500k. The 400k spike was compelling for one data point but didn't carry. Need to see step-700k onward to know if something stable emerges.
- **Prediction**: given HIQL's flat-ish trajectory (0.39 → 0.40 over 100k–1M), Phase-3b at step 1M will likely land 0.39–0.43 mean. If in the 0.42–0.43 range, that's +2-3 pts over HIQL — within noise but consistently favorable across seeds.

## 5. What still needs to land

- 3 more seed-3-way evals: step 700k, 800k, 900k.
- Final step-1M eval + `params_1000000.pkl` saves.
- σ diagnostic on the saved checkpoints (expected ~10 min CPU time, auto-triggered by cron).
- β sweep on the checkpoints (separate step, ~50 min CPU).

## 6. Decision plan after training completes

| Diagnostic says | Mean @ 1M | Next action |
|---|---|---|
| `per_state_mu_sig_corr_mean` ≈ 0 | ≥ 0.42 | Ship: "tight τ fixes C-HIQL's σ." β-sweep to quantify mechanism contribution. |
| `per_state_mu_sig_corr_mean` ≈ 0 | < 0.41 | Mechanism clean, env just hard. Need Option C (LCB target) or Option D (per-head trunks). |
| Correlation still polarized | any | Tight τ reduced the magnitude but didn't change the structural problem. Proceed to per-head trunks. |

## 7. File pointers

- Live logs: `/home/y_f/ece567/logs/ex_chiql_phase3b_seed{0,1,2}.log`
- Run artifacts: `/home/y_f/ece567/runs/ex_chiql_phase3b/dummy/ex_chiql_phase3b/sd00{0,1,2}_*`
- Monitor: `scripts/monitor_ex_phase3b.sh`
- Smoke: `scripts/smoke_ex_phase3b.sh`
- Agent: `external/ogbench/impls/agents/ex_chiql.py` (commit `2f13adc`)

## 8. Full report forthcoming

This file is a live snapshot. A full `PHASE3B_REPORT.md` (parallel to `PHASE3_DESIGN_A_REPORT.md`) will be written after training + diagnostic + β sweep are complete.
