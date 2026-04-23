# Phase-3b Improvement Plan

> **Supersedes** `PHASE3B_REPORT.md §7` (Phase-3c grad-clipping recommendation).
> Written after re-examining Phase-3b training logs and the β-sweep results.
> Branch: `expectile-heads`. Env: `antmaze-teleport-navigate-v0`.

---

## 1. What Phase-3b actually shows

Three facts from the trained artifacts + architecture shape every option below:

1. **Training is numerically healthy.** `grad/norm` sits flat in the 280-650 range for the full 1M steps across all seeds — there is no growth, no blow-up. Phase-3a's explosion pathology is completely absent here.

2. **σ-creep is slow and monotonic.** `v_std_across_heads` drifts `4 → ~17` across 40k → 800k. The shape is sub-linear (roughly +1 per 100k early, slowing late), not exponential. The gradient norms are stable *while* σ grows — so σ-creep isn't a gradient problem.

3. **β is inference-only.** It appears nowhere in `value_loss`, `low_actor_loss`, `high_actor_loss`, or `total_loss` — only inside `sample_actions` at [ex_chiql.py:217-230](external/ogbench/impls/agents/ex_chiql.py#L217-L230). Actors use `vs[idx]` (the τ=0.7 head), not `μ − β·σ`. One trained checkpoint → many β values, zero retraining.

### 1.1 Correction: the ensemble is independent-trunk, not shared-trunk

Earlier Phase-3 reports (`PHASE3_DESIGN_A_REPORT.md`, `PHASE3B_REPORT.md`, and an initial draft of this plan) described EX-HIQL's value net as "shared trunk + 5 independent last-layer heads". **That was wrong.** Reading [utils/networks.py:15-25](external/ogbench/impls/utils/networks.py#L15-L25):

```python
def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        ...
    )
```

and `GCValue.setup` wraps the **entire MLP module** (trunk + output) in `ensemblize`. With `split_rngs={'params': True}`, every layer's parameters have leading axis 5 and each replica is independently initialized. So the value net is **5 fully independent MLPs**, each with its own 3×512 trunk *and* its own output Dense. There is no parameter shared across the 5 heads.

**What this changes in our narrative**:

- Phase-3a explosion wasn't "shared trunk pulled in contradictory directions by τ=0.1 vs τ=0.9". It was each independent MLP being driven to extreme expectile fixed points — a *per-head* pathology amplified by its own target-network EMA feedback loop. Shared-trunk coupling had nothing to do with it.
- σ-creep in Phase-3b isn't "shared-trunk gradient conflict widening over training". It's 5 independent value functions each converging to a different expectile-τ fixed point, with the gap between fixed points growing as V learns a wider dynamic range (longer-horizon goals, fuller reachability graph).
- The counterfactual ablation — **does shared trunk behave differently?** — is no longer hypothetical. It's a distinct architecture we have never actually tested. That's why there's a new `shared-trunk-5head` branch: to run the true comparison and see if coupled-trunk expectile heads σ-creep at a different rate.

---

## 2. Why grad clipping (the old plan) is the wrong tool

`PHASE3B_REPORT.md §7` proposed `optax.clip_by_global_norm(10.0)` as the primary Phase-3c intervention. Re-examining the data:

- The grad norms we would clip are already an order of magnitude below typical clip thresholds and are *flat*, not drifting.
- The σ-creep mechanism is structural: each head's expectile regression has a different fixed point, and as V learns a wider dynamic range over training (longer horizons, more of the reachability graph filled in), the gap between those fixed points *should* widen. This is a loss-landscape property, not an optimizer noise property.
- Clipping bounds step size, not the fixed-point separation. It would make an already-stable optimizer slightly more conservative, without touching the mechanism that makes σ creep.

**Verdict**: keep grad clipping as a precaution (cheap, one line, doesn't hurt), but stop treating it as *the* fix. It is not addressing the failure mode.

---

## 3. Ranked improvement options

Ordered by (expected gain) / (cost). Option 1 is free and should happen first.

### Option 1 — β schedule at inference [cheap, no retraining]

**Observation**: β is inference-only. The same `params_1000000.pkl` can be re-scored under any β-of-step function without a new training run. The β-sweep (flat β across all steps) already gives us a partial answer; β=0.5 is the best single value (0.423 mean), but this is averaged over steps with σ ∈ [4, 17]. A per-step β that shrinks as σ grows would keep `β·σ/|μ|` near the 15% target:

```
β(step, σ)  =  0.5  ·  min(1,  8 / σ)
```

At σ=4 → β=0.5 (full pessimism, early training), at σ=16 → β=0.25 (half-strength, late training). Or the simplest version: β=0.5 at steps where σ<10 is satisfied, β=0.25 otherwise.

**Cost**: modify only `sample_actions` to read σ from training metrics (or compute it on-the-fly from the candidate values). No training. Single evaluation sweep.

**Expected gain**: if σ-creep is the sole degradation mechanism, this should recover the step-400k peak (0.439) at step 1M. If it doesn't, other mechanisms are at play and we need Options 2-4.

**This is the experiment to run first** because it's cheap *and* it serves as a diagnostic — it distinguishes "σ-creep broke the scoring" from "the underlying V itself is worse late in training".

---

### Option 2 — L2 on ensemble last-layer weights [cheap, retraining required]

**Goal**: bound the separation between per-head solutions.

The σ between heads is driven by differences in the final-layer head weights (trunk is shared). An L2 penalty on the deviation of each head from the mean head:

```
L_reg = λ · Σ_k ‖w_k − w̄‖²    where w̄ = mean of the 5 heads
```

pulls the heads toward a common value while still allowing the expectile losses to pull them apart. The equilibrium σ is then bounded by the λ/expectile-gradient ratio.

**Cost**: ~15 lines in `value_loss`, one hyperparameter (λ), three seeds × 1M steps. Comparable cost to a normal Phase-3 run.

**Risk**: if λ is too large, the heads collapse to a single head and σ goes to 0, removing the mechanism entirely. We'd need a small grid, probably `λ ∈ {1e-4, 1e-3, 1e-2}`, which is 3× the cost.

**Expected gain**: directly targets σ-creep via a per-parameter restoring force. Works the same way regardless of whether trunks are shared or independent — it pulls each head's *weights* toward their ensemble mean rather than constraining the trunk.

---

### Option 3 — Shared-trunk variant [architectural ablation]

**What this actually is**: the *opposite* of what I originally wrote. Phase-3b's value net is already 5 independent MLPs (see §1.1). The new experiment is to test a **shared-trunk + 5 independent last-layer heads** architecture — the one the old docs mistakenly claimed we had.

**Goal**: isolate whether σ-creep is a property of the expectile-ensemble *method* (shows up under both architectures) or a property of fully-independent trunks (shows up only without trunk coupling).

With a shared trunk of feature dim 512 feeding 5 scalar heads, the trunk is trained by the sum of 5 expectile losses; the 5 heads diverge only through their final Dense projection `(512 → 1)` — ~513 params per head instead of ~800k. The disagreement σ is bounded by how far those 513-param projections can diverge while all still reading from a single shared feature.

**Cost**: smaller than the current model (~800k params for one trunk + 2.5k for 5 heads, vs ~4M for 5 independent trunks). ≈0.6× wall-clock per step. Three seeds fit in ~3h on the 5090.

**Risk**: a shared trunk trying to serve 5 different expectile objectives may develop features that are a compromise between all 5, resulting in lower per-head accuracy and a σ that's small but meaningless (heads look at near-identical features and so predict near-identical values). This is plausible and, if it happens, tells us something important: the expectile-ensemble method *needs* independent trunks to produce a σ signal that correlates with stochasticity.

**Expected outcome, two cases**:
- **If shared-trunk σ-creeps similarly to Phase-3b**: σ-creep is intrinsic to the expectile method (different fixed points → different values). Independent vs shared trunks doesn't matter; we'd need something else (Option 2, or β schedule).
- **If shared-trunk σ stays bounded (or collapses to 0)**: σ-creep was driven by each independent MLP's freedom to diverge. Shared trunk provides the bound for free; from there, whether the bounded σ is still *meaningful* is the next question.

This experiment lives on the `shared-trunk-5head` branch.

---

### Option 4 — Infrastructure: save_interval=100000 + σ-based early stopping [orthogonal]

Independent of the algorithmic choice:

- `save_interval=100000` (not 1M) so we can re-evaluate any step without re-running training. Lets us apply β-sweep *per step* and validate Option 1's schedule post-hoc.
- Early-stop-and-save when `max_seed(v_std_across_heads) > 10` is crossed, not after a fixed step budget.

**Cost**: trivial. One flag change, one callback.

**Why this matters regardless of algorithm**: any claim of the form "our method's peak is at step X" is unfalsifiable without the checkpoint at step X. The current `params_1000000.pkl`-only layout means we can't compute β-sweep at the σ<10 regime, which is exactly where our best numbers live.

---

## 4. Deprioritized / explicitly not-recommended

- **Grad clipping alone**: see §2. Include as a precaution in any Phase-3c run, but it is not the headline intervention.
- **Spread narrowing (`(0.65..0.75)` instead of `(0.6..0.8)`)**: buys a smaller σ-creep slope but also a smaller signal. Without Options 1-3, this is just trading one axis for another.
- **LR schedule**: the grad norm is flat, so the optimizer is already finding a stable step. A schedule would reduce step size without addressing the fixed-point separation. Low-yield.
- **Making β trainable** (end-to-end): β enters through an argmax (subgoal selection), which is not differentiable. Would require a relaxation (softmax over candidates) that changes the method.

---

## 5. Proposed execution order

1. **[1 day]** Implement Option 1 β-schedule at eval time. Run the eval-only sweep on the existing Phase-3b checkpoints. Determine whether σ-aware β recovers the step-400k peak at step 1M. This is the cheap diagnostic.
2. **[2 days]** If Option 1 gives mixed results → run Option 2 (L2 on head weights) with `λ ∈ {1e-4, 1e-3}`, 3 seeds each. Add `save_interval=100000` from Option 4.
3. **[3 days]** If Options 1+2 both plateau → run Option 3 (per-head trunks) as the definitive test. One configuration, 3 seeds.
4. **[always]** Run with `save_interval=100000` going forward.

Under a compute-budget constraint, stop after whichever option achieves step-1M mean ≥ 0.43 across 3 seeds with σ-healthy training. That's the "beats HIQL on this benchmark under the standard convention" bar.

---

## 6. What "success" looks like for Phase-3c

| Metric | Phase-3b (current) | Phase-3c target |
|---|---|---|
| Step-1M 3-seed mean (β=0.5) | ~0.41 (pending β-sweep) | ≥ 0.43 |
| `v_std_across_heads` at step 1M | 13-22 | ≤ 10 |
| Best-step mean | 0.439 @ step 400k | ≥ 0.44 at ≥ step 800k |
| Requires early stopping rule to beat HIQL | yes | no |

The target at step 1M (≥ 0.43) is chosen to clear HIQL's best-step mean (0.424) and exceed HIQL's step-1M mean (0.404) by ≥ 2 pts — enough to be directionally credible even if within 1σ of 250-episode eval noise.

---

## 7. File pointers

- Agent: [external/ogbench/impls/agents/ex_chiql.py](external/ogbench/impls/agents/ex_chiql.py)
- σ-diagnostic: [scripts/diagnose_sigma.py](scripts/diagnose_sigma.py), [results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/seed{0,1,2}/diagnostics/](results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/)
- β-sweep script: [scripts/launch_beta_sweep_phase3b.sh](scripts/launch_beta_sweep_phase3b.sh)
- Phase-3b report (the data this plan is built on): [PHASE3B_REPORT.md](PHASE3B_REPORT.md)
- Phase-3a postmortem (for why wide-τ exploded): [PHASE3_DESIGN_A_REPORT.md](PHASE3_DESIGN_A_REPORT.md)
- Method spec: [EXPECTILE_HIQL.md](EXPECTILE_HIQL.md)
