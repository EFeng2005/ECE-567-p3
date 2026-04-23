# EX-HIQL Phase 3 Design A — Run Report (2026-04-22, revised 2026-04-23)

> **Architecture note (important)**: this report describes EX-HIQL's value network as **5 completely independent MLPs** — one full 3×512 trunk *and* one output head per expectile. That is what the code implements: `GCValue(ensemble=True, num_ensemble=5)` uses `ensemblize` which wraps the whole MLP in `nn.vmap` with `variable_axes={'params': 0}, split_rngs={'params': True}` (see [utils/networks.py:15-25](external/ogbench/impls/utils/networks.py#L15-L25)). Every weight in the value net has a leading axis of 5 and is independently initialized. An earlier draft of this report called the same architecture "shared trunk + 5 independent heads" — that phrasing was incorrect and has been removed. No numerical result has changed; only the mechanism narrative is corrected.

**Commit**: `8be4b68` on branch `expectile-heads` (renamed to `indep-trunk-5head`).
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
(s, g) ──┬──► MLP_1 (3×512 trunk + Dense(1), trained at τ=0.1) ─→ V_1(s, g)
         │
         ├──► MLP_2 (3×512 trunk + Dense(1), trained at τ=0.3) ─→ V_2(s, g)
         │
         ├──► MLP_3 (3×512 trunk + Dense(1), trained at τ=0.5) ─→ V_3(s, g)
         │
         ├──► MLP_4 (3×512 trunk + Dense(1), trained at τ=0.7) ─→ V_4(s, g)     ← actor READS this one
         │
         └──► MLP_5 (3×512 trunk + Dense(1), trained at τ=0.9) ─→ V_5(s, g)
```


**Five completely independent MLPs**, one per expectile. Each has its own 3-layer × 512-wide trunk with LayerNorm *and* its own scalar output projection. Nothing is shared across the 5 networks — not trunks, not layer-norm parameters, not output weights. Each was initialized with its own RNG key. A single target network mirrors this structure with 5 independent EMA copies.

### 3.2 Where heads are used — three distinct sub-pipelines

There are three places in the code where `V(s, g)` is used, each with a different head policy:

| Sub-pipeline | File | Head usage | Receives gradient? |
|---|---|---|---|
| **Value-network training** | `ex_chiql.py value_loss` | All 5 heads, each trained with its own τ against its own target-network bootstrap | **Yes** — each of the 5 independent MLPs gets its own per-head expectile gradient |
| **Actor training (low + high)** | `ex_chiql.py low_actor_loss / high_actor_loss` | Only head 3 (τ=0.7) via `vs[3]` | No — advantages are treated as targets (value-net grad_params aren't passed through) |
| **Inference (subgoal scoring)** | `ex_chiql.py sample_actions` | All 5 heads — computes μ, σ across heads | N/A (inference only) |

So the pipeline breakdown is:

- **Training the value net**: each of the 5 independent MLPs is updated by its own expectile loss at its own τ, against its own target-network's bootstrap. There is no gradient flow between different heads — nothing is shared. One epoch produces 5 parallel, isolated expectile-regression updates.
- **Training the actors**: the actor's advantage reads head 3's scalar — that's a forward pass, the value net's gradient is not updated by the actor's loss. Head 3 trains exclusively on its own expectile loss; the other 4 heads are irrelevant to head 3's trajectory.
- **Inference**: all 5 heads are used. The agent computes V_i for i=1..5 on every candidate subgoal, aggregates to μ (mean) and σ (std), and scores as μ − β·σ.

### 3.3 TL;DR on the single-head question

- **Actor's scalar advantage**: 1 head (head 3). This is probably what prompted the "single head" framing.
- **Inference scoring**: 5 heads.
- **Value-net training**: 5 fully independent heads, each trained in isolation.

Only the scalar advantage into the actor's gradient uses one head. Everything else is 5-head.

---

## 4. Why the heads explode — mechanism

### 4.1 Each MLP runs its own expectile regression in isolation

The value loss decomposes into 5 independent expectile losses, each evaluated against a different τ:

```
total_value_loss = sum_{i=1..5}  E_batch[  expectile_loss(τ_i, td_error_i)  ]
```

But — and this is the key point for this architecture — the 5 MLPs have **no shared parameters**. `∂ loss_i / ∂ θ_j = 0` for i ≠ j. Each head's gradient updates only its own 800k-parameter MLP. There is no cross-head coupling through feature space; the 5 networks are running in parallel, sharing only the batch of (s, s', r, mask, goals) they consume.

So "5 heads disagree" is not a conflict over *one* set of features — it's 5 networks arriving at 5 different learned value functions.

### 4.2 Extreme τ creates a runaway target-network feedback loop

The explosion is per-MLP, not cross-MLP. Each extreme-τ head destabilizes on its own. Consider head 5 at τ=0.9, which optimizes:

```
L_5 = E[  0.9 · I[adv ≥ 0] · (q − V_5)²  +  0.1 · I[adv < 0] · (q − V_5)²  ]
```

where `q = r + γ · V_5_target(s')` and `V_5_target` is the EMA copy of head 5 itself. Two features of this setup couple into a positive feedback loop:

1. **Asymmetric weights**. τ=0.9 puts 9× weight on samples where the current prediction is *below* the bootstrap target. This aggressively pulls V_5 upward toward the high-tail of the target distribution.

2. **Target comes from the same head**. V_5_target is V_5 from ~1000 steps ago (via `tau=0.005` EMA). So as V_5 grows, V_5_target catches up, which makes the Bellman target r + γ·V_5_target grow too, which the τ=0.9 loss then chases even higher. Feedback loop.

With `head_expectiles=(0.1, 0.3, 0.5, 0.7, 0.9)`, heads 1 and 5 are both on this runaway. Head 1 (τ=0.1) spirals down toward very negative V; head 5 (τ=0.9) spirals up toward very positive V. The middle heads (τ=0.3, 0.5, 0.7) have weaker asymmetry and stay close to a reasonable fixed point.

### 4.3 Why Adam lets it happen

Adam normalizes each parameter's update by a running estimate of its gradient magnitude:

```
Δθ ≈ lr · (mean grad) / sqrt(mean grad²) ≈ lr · sign(grad)
```

So the *step size* stays at ~lr = 3×10⁻⁴, regardless of gradient magnitude. That prevents NaN — a spike in grad/norm doesn't translate into a giant parameter update. But it means Adam doesn't stop the drift either: if the gradient has a consistent sign (which it does for an extreme-τ head locked in the feedback loop), Adam happily walks the parameters in that direction forever.

The result: head 1 and head 5 each drift unboundedly in their own direction over training, while their grad norms grow (because the loss grows quadratically in the prediction error and the error is growing). We observe `grad/norm` ballooning from ~1,600 to millions — this is the *per-MLP* gradient norm summed across all 5 heads.

### 4.4 Consequence for head 3 (the actor-relevant head)

Head 3 (τ=0.7) has its own independent MLP and its own independent target network. Its loss has a reasonable fixed point near HIQL's natural regime (HIQL uses τ=0.7 too). Because no parameter is shared with heads 1 or 5, head 3's training is completely isolated from their divergence.

**So V_3 should be ≈ HIQL-quality.** And empirically it is: the final eval (0.427) is close to HIQL (0.404). The actor's AWR advantage reads head 3, so the actor is effectively being trained against an independent-HIQL-style value function regardless of what heads 1 and 5 are doing.

This is what the "parity-with-HIQL" final number is measuring: a standalone HIQL-equivalent value net (head 3), completely unaffected by the chaos elsewhere in the ensemble. No demonstration of the pessimism mechanism — just HIQL with extra, unused-and-broken heads.

### 4.5 Consequence for inference (the designed pessimism mechanism)

At inference the agent scores each candidate as `μ − β · σ`. For this run:

- `μ` = mean of all 5 heads → dominated by the two runaway heads (head 1 ≈ −14,700, head 5 ≈ +1,400), so `μ` ≈ (−14,700 − 2,400 − 40 + 20 + 1,400)/5 ≈ −3,144 (not HIQL-like at all).
- `σ` = std of all 5 heads → huge, driven by the same runaway heads.

At β = 0.5, `β · σ ≈ 500–1,400`, comparable to or larger than `|μ|`. The pessimism term fully dictates the pick. The σ diagnostic (§6.1) confirms this and shows `per_state_mu_sig_corr ≈ +0.99` — μ and σ scale together with the candidate state's distance-from-goal, because both are dominated by the two runaway heads whose outputs are roughly linear in the trunk's activation magnitude (each independent MLP has its own trunk, but all 5 trunks see the same state input, so whatever makes one state "larger-activation" makes all of them larger-activation).

The scoring has degenerated into noise, *not* because the heads coupled through a shared trunk (they didn't) but because two of the 5 heads ran away in opposite directions and their contributions swamp the three reasonable heads.

---

## 5. Prediction pipeline — what this means practically

Answering the user's question directly:

| Question | Answer |
|---|---|
| Does training use all 5 heads? | **Yes.** Value loss sums 5 independent per-head expectile losses; each MLP gets only its own gradient (no parameter is shared). |
| Does the actor's advantage use all 5 heads? | **No, only head 3.** `vs[3]` indexing in both `low_actor_loss` and `high_actor_loss`. |
| Does the actor's gradient backprop into the value net? | **No.** Standard HIQL pattern: advantage is a scalar target, no grad through V. So the actor's training is protected from value-net instability *for what the actor reads*. |
| Does inference use all 5 heads? | **Yes.** μ and σ across all 5 for each candidate subgoal. |
| Is the divergence a multi-head coupling problem? | **No — it's a per-head stability problem.** The 5 MLPs share no parameters. The extreme-τ heads each run away on their own, via the target-network EMA feedback loop (§4.2). |
| Does the divergence affect head 3 specifically? | **No.** Head 3's MLP is entirely separate. V_3 is HIQL-quality; the degradation is confined to heads 1 and 5. |
| Does the divergence affect the actor's behaviour? | **Barely.** The actor reads head 3, which is clean. That's why the policy still lands at HIQL-level success. |
| Does the divergence break the designed pessimism? | **Yes, completely.** Two of the 5 heads are at ±thousands while the other three are reasonable, so μ − β·σ is dominated by the runaways and the pick is essentially random. The success-rate parity is actually just HIQL via head 3 — not evidence for or against σ-pessimism. |

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

Both signs are bad for the method — they're just different failure modes of the same problem (σ is not a meaningful stochasticity signal):

- **Phase-2**: σ anti-correlated with μ. With 5 MLPs all trained at the same τ=0.7, residual init-noise disagreement happened to be larger where V was smaller (far from goal), not at teleporter states. σ tracked uncertainty-in-general, not stochasticity-specifically.
- **Phase-3 (this run)**: σ nearly perfectly correlated with μ. Two of the 5 MLPs (τ=0.1 and τ=0.9) ran away in opposite directions via the per-head feedback loop (§4.2). At *any* candidate state, those two runaway heads emit large-magnitude predictions whose values scale with the state's "further from goal"-ness. The mean μ moves with them; the spread σ moves with them. Both are dictated by the same runaway outputs, so they correlate near-perfectly.

A clean mechanism would show `per_state_mu_sig_corr_mean ≈ 0` — σ independent of μ, signalling stochasticity specifically. Neither Phase-2 nor Phase-3 achieves that. The per-head-expectile idea might work with a different parameterization, but with this wide-τ spread the extreme heads destabilize before they can produce useful disagreement.

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

### 7.3 Shared-trunk variant (CHEAP, ARCHITECTURAL ABLATION)

**Config change**: replace the current 5-independent-MLPs value net with a single 3×512 trunk followed by 5 independent `Dense(1)` output heads. The output shape remains `(5, B)` so the rest of the agent is unchanged.

This is the *opposite* of the current architecture. Our current setup has 5 completely independent MLPs; a shared-trunk variant would force the 5 heads to read the same feature representation. For this run's failure (heads 1 and 5 running away via per-head target-network feedback), a shared trunk could plausibly bound the damage: the runaway extrema in heads 1 and 5 would have to be produced by Dense(1) projections on top of a feature shared with the three reasonable heads. If the trunk settles near what the reasonable heads need, the extreme heads can only express their divergence through their 513-param output layers — a much tighter budget than 800k params each.

**Cost**: smaller than current (~800k trunk + 5×513 heads ≈ 803k total vs 5×800k = 4M currently). ~0.6× wall-clock. 3 seeds × ~3h = 9 GPU-hours.

**Risk**: a shared trunk trying to serve 5 different expectile objectives may produce features that are a compromise, with the 5 heads collapsing to near-identical predictions. σ becomes small but also meaningless. This is the standard concern about shared-trunk ensembles; the experiment tells us whether it matters here.

Run this as a straightforward architectural comparison to 7.1. Lives on the `shared-trunk-5head` branch.

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
               → Run shared-trunk variant (7.3) as an architectural comparison. If shared trunk also fails to localize σ, the issue is the expectile-ensemble method itself, not the parameter-sharing choice.
```

---

## 8. What this run is still worth

- Final checkpoints `sd00{0,1,2}/params_1000000.pkl` — save point for the σ-diagnostic and β-sweep (both in progress or next).
- Documented failure mode with precise numerical signature (§4). Useful as the "what happens with wide τ" reference in the paper.
- Confirms the Phase-2 C-HIQL root-cause hypothesis: σ from a random-init 5-MLP ensemble doesn't localize to stochastic states; it tracks feature-magnitude correlates of the shared state input. Making the heads *more* different (wide-τ EX-HIQL) didn't fix this — it replaced "quiet anti-correlation driven by init noise" with "loud positive correlation driven by per-head runaway". Both failures are architecture-independent (they happen in 5-independent-MLP setups) and signal-independent (both negative and positive μ-σ correlations are bad for the method).

The next run (tight τ + clipping) is the one that actually tests whether per-head expectiles can make σ useful when numerically clean. This run tells us only that the original parameterization was too aggressive — which is valuable but not conclusive either way.
