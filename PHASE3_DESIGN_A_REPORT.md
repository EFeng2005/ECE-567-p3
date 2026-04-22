# EX-HIQL Phase 3 Design A — Run Report (2026-04-22)

**Commit**: `8be4b68` on branch `expectile-heads`.
**Env**: `antmaze-teleport-navigate-v0`.
**Agent**: `ex_chiql` (per-head expectiles, `head_expectiles=(0.1, 0.3, 0.5, 0.7, 0.9)`, actor pinned to head 3 = τ=0.7).
**Seeds**: 3 × 1M steps, parallel on one RTX 5090 laptop inside WSL2.
**Wall time**: ~5h 40m for training (same envelope as Phase 2).

---

## 1. Executive summary

The run **completed without NaN** and the final policy is **within eval noise of plain HIQL** (0.427 mean vs HIQL's 0.404). But the value network's training internals **diverged extensively**: `grad/norm` grew from ~1,600 to millions; `v_std_across_heads` grew from ~1.2 to ~2,900; V values escaped physical bounds (up to +1,400, down to −14,700 — the env's episode length bound is ~500 steps).

The policy survived this instability because two structural choices buffer the actor from the blown-up value network:
1. The actor's advantage signal uses **only head 3 (τ=0.7)** — the head whose expectile fixed point is closest to HIQL's natural regime.
2. The low-level walker does locomotion, which is dominated by short-horizon policy imitation rather than V-accuracy.

But the designed pessimism mechanism (`μ − β · σ` scoring) did not bite: σ is huge everywhere (~1,000-3,000 range vs |μ| ~100), so the pessimism term dominates and the scoring is effectively noise. The σ signal never became localized to stochastic states as the method assumes.

**Verdict**: a parity-with-HIQL result produced by a broken value network whose actor-relevant head happened to be closest to HIQL's own optimum. Not a demonstration of the intended mechanism.

---

## 2. Final results

### 2.1 Eval trajectory (overall success rate, mean of 3 seeds)

| Step | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| 1 (untrained) | 0.000 | 0.000 | 0.000 | 0.000 |
| 100,000 | 0.352 | 0.312 | 0.428 | 0.364 |
| 200,000 | 0.400 | 0.340 | 0.404 | 0.381 |
| 300,000 | 0.412 | 0.332 | 0.428 | 0.391 |
| 400,000 | 0.408 | **0.496** | **0.480** | **0.461 (peak)** |
| 500,000 | 0.332 | 0.392 | 0.388 | 0.371 |
| 600,000 | 0.380 | 0.412 | 0.336 | 0.376 |
| 700,000 | 0.408 | 0.272 | 0.416 | 0.365 |
| 800,000 | 0.400 | 0.400 | 0.424 | 0.408 |
| 900,000 | 0.388 | 0.420 | 0.416 | 0.408 |
| **1,000,000** | **0.460** | **0.384** | **0.436** | **0.427** |

The step-400k peak (0.461) was a transient. The rest of training fluctuated in the 0.37–0.43 band. The final step-1M mean (0.427) is within standard-error of HIQL's 0.404.

### 2.2 Cross-method comparison

| Method | Final step-1M mean | Δ vs HIQL |
|---|---|---|
| Phase-1 HIQL baseline | 0.404 | — |
| Phase-2 C-HIQL β=0.5 | 0.384 | −2.0 pts |
| **Phase-3 EX-HIQL β=0.5 (this run)** | **0.427** | **+2.3 pts** |

Eval noise (1σ for a 250-episode success-rate estimate is about 3 pts). A +2.3 pt gap is within one standard error — not a statistically meaningful improvement. The story is "parity", not "gain."

### 2.3 Training-internal metrics (the explosion)

Measurements read from `train.csv` at log intervals of 5k steps. Representative values:

| Step | `v_std_across_heads` | `grad/norm` (worst seed) | `v_max` | `v_min` |
|---|---|---|---|---|
| 5,000 (smoke) | ~1.2 | — | ~0 | ~−40 |
| 30,000 | ~17 | ~1,600 | — | — |
| 250,000 | ~50 | ~2,100 | — | — |
| 400,000 | ~111 | ~6,900 | +281 | −3,102 |
| 500,000 (seed 2) | ~228 | ~16,640 | +537 | −2,855 |
| 700,000 | ~696–979 | ~177,000 | +423 | −2,047 |
| 900,000 (seed 2) | ~1,250 | ~420,000 | **+812** | **−8,767** |
| 1,000,000 | ~1,900–2,900 | ~1.5-9 M | **+1,400** | **−14,700** |

**Physical bound check**: episode length ≤ ~500, reward = −1 per step ⇒ realistic V ∈ roughly [−500, 0]. Observed `v_max = +1,400` and `v_min = −14,700` violate this by factor ~3× on the positive side and ~30× on the negative side. The value function is no longer representing "steps to goal" in any grounded sense.

---

## 3. The prediction pipeline — which heads are used where

The user's question: *"I think our pipeline only involves a single head? If not explain how are the heads used."*

**Answer: no, the pipeline involves all 5 heads.** Let me be precise about where.

### 3.1 Architecture recap (EX-HIQL = C-HIQL = HIQL value arch)

```
                  ┌─ head_1 (trained at τ=0.1) ─→ V_1(s, g)
                  │
(s, g)            ├─ head_2 (trained at τ=0.3) ─→ V_2(s, g)
  │               │
  └─► trunk ──────├─ head_3 (trained at τ=0.5) ─→ V_3(s, g)
       (shared    │
        MLP)      ├─ head_4 (trained at τ=0.7) ─→ V_4(s, g)       ← actor READS this one
                  │
                  └─ head_5 (trained at τ=0.9) ─→ V_5(s, g)
```


Five heads, **all sharing the same trunk** (one 3-layer × 512-wide MLP with LayerNorm). Only the final linear projection differs per head.

### 3.2 Where heads are used — three distinct sub-pipelines

There are three places in the code where `V(s, g)` is used, each with a different head policy:

| Sub-pipeline | File | Head usage | Receives gradient? |
|---|---|---|---|
| **Value-network training** | `ex_chiql.py value_loss` | All 5 heads, each with its own τ | **Yes** — all 5 heads AND the shared trunk |
| **Actor training (low + high)** | `ex_chiql.py low_actor_loss / high_actor_loss` | Only head 3 (τ=0.7) via `vs[3]` | No — advantages are treated as targets (stop-gradient effectively, because the value net's grad_params aren't passed) |
| **Inference (subgoal scoring)** | `ex_chiql.py sample_actions` | All 5 heads — computes μ, σ across heads | N/A (inference only) |

So the pipeline breakdown is:

- **Training the value net**: all 5 heads are fed gradients from their own per-head expectile loss. Those gradients all flow through the shared trunk. The trunk receives a *sum* of 5 gradient contributions from 5 different objectives.
- **Training the actors**: the actor's advantage reads head 3's scalar — that's a forward pass, the value net's gradient is not updated by the actor's loss. So head 3's training is driven *only* by its own value loss and the shared trunk's contribution from all other heads.
- **Inference**: all 5 heads are used. The agent computes V_i for i=1..5 on every candidate subgoal, aggregates to μ (mean) and σ (std), and scores as μ − β·σ.

### 3.3 TL;DR on the single-head question

- **Actor's scalar advantage**: 1 head (head 3). This is probably what prompted the "single head" framing.
- **Inference scoring**: 5 heads.
- **Value-net training**: 5 heads feeding gradients into 1 shared trunk.

Only the scalar advantage into the actor's gradient uses one head. Everything else is 5-head.

---

## 4. Why the heads explode — mechanism

### 4.1 The conflict

The value loss per training step is:

```
total_value_loss = sum_{i=1..5}  E_batch[  expectile_loss(τ_i, td_error_i)  ]
```

Each of the 5 expectile losses has its **own fixed point** (the value where its asymmetric squared error is minimized):

- Loss at τ=0.1 wants V(s, g) = 10%-expectile of the bootstrap-target distribution — typically the most pessimistic outcome.
- Loss at τ=0.9 wants V(s, g) = 90%-expectile — typically the most optimistic.

For a teleporter state T where the bootstrap-target distribution is wide (e.g. {−1 lucky, −51 × 99 unlucky}), these two fixed points are far apart (roughly −51 and −5 respectively).

Head_1's output and head_5's output are both computed as:

```
V_i(s, g) = W_i · h(s, g)
```

where `h(s, g) ∈ R^{512}` is the shared trunk output. The only thing that distinguishes heads is the last-layer weights `W_1, …, W_5`.

For heads 1 and 5 to emit very different outputs on the same `h`, two things can happen:

(a) **W_1 and W_5 diverge in direction**, so `W_1 · h` ≠ `W_5 · h` for generic `h`. Bounded weights; heads disagree structurally. *This is the intended mechanism.*

(b) **h becomes large in magnitude**, so small differences between W_1 and W_5 produce large differences in output. `‖h‖` inflates over training to let each head escape to its preferred fixed point. Weights can stay nearly identical as long as `h` grows. *This is what happened.*

### 4.2 Why (b) is what SGD actually does

The trunk receives gradient contributions from all 5 heads summed:

```
∇_trunk total_loss = Σ_i  ∇_trunk loss_i
```

Each head's trunk-gradient pushes the trunk in whatever direction makes *that head's* V closer to its τ-specific fixed point. If the 5 fixed points are far apart and the W_i are similar, the only way to satisfy all 5 per-head losses simultaneously is to grow ‖h‖ — which creates room for all 5 to project differently from the same feature vector.

Once ‖h‖ grows, gradients through the shared trunk scale proportionally, because V_i depends linearly on h. Now the next step's gradient is bigger. `grad/norm` balloons.

Adam masks the symptom by normalizing per-parameter updates to `~ lr = 3×10⁻⁴`, so training doesn't NaN. But the underlying feature representation has already inflated to absorb the 5 conflicting objectives. That's why we see `v_max = +1400`, `v_min = −14700` — the heads are emitting wildly different values because `h` has wandered into a weird region of feature-space.

### 4.3 Consequence for head 3 (the actor-relevant head)

Head 3 (τ=0.7) is the actor's read-out. Its OWN loss (expectile at τ=0.7) has a well-defined fixed point near HIQL's natural regime, and its OWN per-head weights are updated with a reasonable gradient. *If it had its own trunk*, it would train cleanly, like HIQL.

But it shares the trunk with heads 1 and 5, which are pulling the trunk into feature regions that are *convenient for their extreme τ* but suboptimal for τ=0.7. So head 3's output is:

```
V_3 = W_3 · h
```

where `h` has been contorted to serve 5 different τ objectives. `V_3` is still roughly-HIQL-like (because τ=0.7 matches HIQL), but *worse* than a standalone HIQL value network would produce.

This is probably why the final step-1M eval (0.427) is *modestly* above HIQL (0.404) — the actor still gets approximately-sane V_3 values — but *not* the 5+ pt improvement we claimed for E1. The improvement we might have seen from pessimistic scoring is drowned out by the quality degradation of V_3 from the shared-trunk conflict, with the net being near-zero.

### 4.4 Consequence for inference (the designed pessimism mechanism)

At inference the agent scores each candidate as `μ − β · σ`. For this run:

- `μ` ≈ mean of 5 heads → roughly-HIQL-like but degraded (as above).
- `σ` ≈ std of 5 heads → **huge everywhere** because the 5 heads have diverged to extreme values (head_1 ≈ −14,700, head_5 ≈ +1,400 at their worst).

At β = 0.5, `β · σ ≈ 500–1,400`, which is 5-15× larger than `|μ| ≈ 100`. The pessimism term dominates. The scoring is no longer "pick the best μ with a small uncertainty penalty" — it's "pick the candidate with the smallest σ, μ is a rounding error."

Since σ is NOT localized to stochastic states (it's uniformly huge — the **diagnostic we just kicked off** will confirm this), picking "smallest σ" is essentially random. The whole pessimism mechanism has degenerated to noise.

---

## 5. Prediction pipeline — what this means practically

Answering the user's question directly:

| Question | Answer |
|---|---|
| Does training use all 5 heads? | **Yes.** Value loss sums over all 5 per-head losses; gradients flow through the shared trunk to all 5 head weights. |
| Does the actor's advantage use all 5 heads? | **No, only head 3.** `vs[3]` indexing in both `low_actor_loss` and `high_actor_loss`. |
| Does the actor's gradient backprop into the value net? | **No.** Standard HIQL pattern: advantage is a scalar target, no grad through V. So the actor's training is protected from value-net instability *for what the actor reads*. |
| Does inference use all 5 heads? | **Yes.** μ and σ across all 5 for each candidate subgoal. |
| Is the gradient explosion specifically a multi-head problem? | **Yes.** It would not happen with 1 head. It arises from 5 incompatible objectives sharing one trunk. |
| Does the explosion affect head 3 specifically? | **Indirectly, yes.** Head 3's last-layer weights train fine, but the trunk it reads from is distorted by heads 1 and 5's demands. V_3 is worse than a standalone HIQL V would be. |
| Does the explosion affect the actor's behaviour? | **Only mildly.** The actor sees a slightly-degraded HIQL-like V_3. That's why the policy still ~~works~~ produces HIQL-level success, not catastrophic failure. |
| Does the explosion break the designed pessimism? | **Yes, completely.** σ is huge everywhere, so μ − β·σ is noise. The mechanism isn't engaging at all; the success-rate parity is not evidence for or against the σ-pessimism idea. |

---

## 6. Diagnosis plan

Two confirming measurements to take on the saved checkpoints:

### 6.1 σ-localization diagnostic — RESULTS

Ran `scripts/diagnose_sigma.py` on all 3 EX-HIQL checkpoints (500 states × 16 candidate subgoals each).

| Metric | Seed 0 | Seed 1 | Seed 2 | Phase-2 C-HIQL (ref) |
|---|---|---|---|---|
| **`per_state_mu_sig_corr_mean`** | **+0.997** | **+0.985** | **+0.986** | −0.44 |
| `sigma_mean` | 2,594 | 3,189 | 2,558 | 1.67 |
| `sigma_p50` | 2,950 | 3,678 | 2,895 | 0.99 |
| `sigma_p90` | 3,965 | 4,939 | 4,143 | 3.90 |
| `mu_mean` | 1,247 | 1,544 | 1,227 | −22.9 |
| `mu_std` | 618 | 807 | 682 | 17.6 |
| `overlap_beta0_vs_beta0.25` | 94.2% | 95.4% | 94.6% | 98.2% |
| `overlap_beta0_vs_beta0.5` | **0.0%** | **0.2%** | **1.0%** | 96.8% |
| `overlap_beta0_vs_beta1.0` | **0.0%** | **0.0%** | **0.0%** | 94.2% |
| `overlap_beta0_vs_beta2.0` | **0.0%** | **0.0%** | **0.0%** | 89.4% |

**Three readings from this table:**

**(a) The μ-σ correlation flipped sign from −0.44 (Phase-2 C-HIQL) to +0.99 (Phase-3 EX-HIQL).**

Both signs are bad for the method — they're just different failure modes of the same structural problem (σ is not a meaningful stochasticity signal):

- **Phase-2**: σ anti-correlated with μ. σ was large where V was uncertain in general (far from goal), not where V was inflated by lucky samples.
- **Phase-3 (this run)**: σ nearly perfectly correlated with μ. Both scale together with the feature vector's magnitude ‖h‖. The explosion of ‖h‖ drives both μ and σ to huge values; high-μ candidates also have high σ, not because stochasticity lines up with optimism but because both are functions of the same inflated trunk output.

A clean mechanism would show `per_state_mu_sig_corr_mean ≈ 0` — σ independent of μ, signalling stochasticity specifically. Neither Phase-2 nor Phase-3 is close.

**(b) σ is 1,500-3,000× larger than in Phase-2 C-HIQL** (absolute scale). Combined with `β · σ` dominating `μ`, the scoring rule `μ − β · σ` has degenerated from "slightly pessimistic ensemble argmax" to "pick the candidate with the smallest σ."

**(c) The overlap numbers are catastrophic.** At β=0.5, 99–100% of candidate picks *differ* from β=0 argmax. Phase-2 C-HIQL had 96.8% agreement at the same β; now it's 0.2%. The pessimism term is not a small correction — it's fully dictating the pick. At β=1 and β=2, 100% of picks differ — σ is ~2× the scale of μ, so any β ≥ ~0.5 swamps the mean entirely.

**The observed parity-with-HIQL success (0.427) happens *despite* the scoring being dictated by σ, not because of it.** The policy is working because the low-level walker is robust and the τ=0.7 head stays reasonable enough — not because the pessimism mechanism is routing the agent away from teleporters.

### 6.2 β=0 inference sweep (NOT YET RUN)

Run `scripts/eval_beta_sweep.py` on the 3 checkpoints with β ∈ {0, 0.25, 0.5, 1, 2}. Expected: **β=0 substantially better than β>0**, because β=0 = "use μ, ignore σ" = pure ensemble-mean scoring. If β=0 is e.g. 0.44+ and β=0.5 is 0.38, that quantifies exactly how much σ is hurting. If β=0 is at 0.43 (the current β=0.5 result) too, the mean is also broken, confirming trunk-level damage.

### 6.3 Decision rule based on diagnostics

After 6.1 and 6.2:
- If σ is huge *and* β=0 >> β=0.5 → clean story: "wide-τ ensemble explodes, σ becomes noise; fix with tight τ."
- If σ is huge *and* β=0 ≈ β=0.5 → deeper problem: trunk is damaged beyond just σ; need per-head trunks.
- If σ is reasonable (unexpected) but β=0 ≈ β=0.5 → ensemble spread isn't signal for this env even when clean.

---

## 7. Future experiments

Ranked by marginal cost and likelihood of teaching us something.

### 7.1 Tighter expectile spread (CHEAP, PRIMARY)

**Config**: `head_expectiles=(0.5, 0.6, 0.7, 0.8, 0.9)` or `(0.6, 0.65, 0.7, 0.75, 0.8)`.

Keeps the "different τ ⇒ different fixed points" mechanism but stays close to HIQL's natural regime. τ=0.5–0.9 fixed points are all finite and well-behaved; no head wants `V = +∞` or `−∞`.

**Expected outcome**: stable training, σ in a modest range (probably 0.5–5), and a meaningful test of whether σ-based pessimism helps when it's numerically clean.

**Cost**: 3 seeds × 5h = 15 GPU-hours + 1h β sweep + 15 min diagnostic ≈ 17 GPU-hours total, one day wall clock.

### 7.2 Global gradient clipping (CHEAP, ADDITIVE)

**Config change**: wrap the optimizer with `optax.clip_by_global_norm(10.0)`:

```python
network_tx = optax.chain(
    optax.clip_by_global_norm(10.0),   # NEW
    optax.adam(learning_rate=config['lr']),
)
```

This is cheap insurance against further blowups, even with the tight τ spread. Should be included in 7.1's run.

### 7.3 Per-head trunks (EXPENSIVE, PRINCIPLED)

**Config change**: 5 independent `GCValue` modules instead of one with `ensemble=True`. Each has its own 3×512 MLP trunk.

Removes the shared-trunk bottleneck entirely. Heads at τ=0.1 and τ=0.9 can develop their own feature representations without fighting. The ensemble becomes a "true" distributional estimator.

**Cost**: 5× value-network parameters (still small in absolute terms — total model still under 50 MB). 5× training compute for the value loss (the actor losses are unchanged). Expected wall-clock ~1.5× of Phase 2. 3 seeds × ~7h = 21 GPU-hours for training + 1h eval.

Run this if 7.1 is still inconclusive.

### 7.4 LCB training target (MEDIUM COST, DIFFERENT MECHANISM)

**Config change**: change the value loss to use `μ − β_train · σ` as the bootstrap target (not just at inference):

```
target = r + γ · (mean(V_heads(s', g)) − β_train · std(V_heads(s', g)))
```

Moves pessimism from *inference* into *training*. The value function itself learns a conservative estimate, not just the scoring rule. This is closer to how EDAC / SAC-N use pessimism.

Breaks the "one checkpoint fits any β" property of EX-HIQL but is a cleaner fit to the distributional-pessimism literature.

**Cost**: 3 seeds × ~5h = 15 GPU-hours + sweep over β_train ∈ {0.25, 0.5, 1}. ~45 GPU-hours total.

### 7.5 Decision matrix for next run

```
Start → Tight τ + grad clip (7.1 + 7.2)
        │
        ├── σ localizes (corr ≈ 0, per-state σ meaningful), success > HIQL + 5 pts
        │      → Ship: "tight-τ expectile ensemble fixes C-HIQL"
        │
        ├── σ localizes, success ≈ HIQL (parity)
        │      → Run LCB training target (7.4). σ is informative but not enough at inference.
        │
        └── σ still doesn't localize (corr ≈ 0 or still negative)
               → Run per-head trunks (7.3). Shared trunk is the fundamental bottleneck.
```

---

## 8. What this run is still worth

- Final checkpoints `sd00{0,1,2}/params_1000000.pkl` — save point for the σ-diagnostic and β-sweep (both in progress or next).
- Documented failure mode with precise numerical signature (§4). Useful as the "what happens with wide τ" reference in the paper.
- Confirms the Phase-2 C-HIQL root-cause hypothesis: σ from random-init shared-trunk ensemble tracks feature magnitude, not stochasticity. The wide-τ "fix" didn't change the structural problem; it just moved the feature magnitude into runaway territory instead of quietly-anti-correlated territory.

The next run (tight τ + clipping) is the one that actually tests whether per-head expectiles can make σ useful when numerically clean. This run tells us only that the original parameterization was too aggressive — which is valuable but not conclusive either way.
