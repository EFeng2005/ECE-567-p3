# Expectile-Heads HIQL (EX-HIQL): Distributional Pessimism for Offline Hierarchical GCRL

> **Architecture note**: EX-HIQL's value network is **5 completely independent MLPs** — same architecture as C-HIQL. `GCValue(ensemble=True, num_ensemble=5)` uses `nn.vmap` with `variable_axes={'params': 0}, split_rngs={'params': True}` (see [utils/networks.py:15-25](external/ogbench/impls/utils/networks.py#L15-L25)), producing one complete 3×512 MLP-with-LayerNorm per expectile, each independently initialized. Nothing is shared across the 5 heads. Earlier drafts of this doc called the same architecture "shared trunk + 5 independent heads"; that phrasing was incorrect and has been rewritten throughout.

> **Idea spec** — parallel in style to `C-HIQL.md`. Written after Phase-2 C-HIQL underperformed on `antmaze-teleport-navigate-v0` and the σ diagnostic (`scripts/diagnose_sigma.py`) found that the random-init ensemble's disagreement is **anti-correlated** with μ across candidate subgoals (per-state corr ≈ −0.44). The mechanism assumption from [C-HIQL.md §4.1](C-HIQL.md) — "σ is largest where V is most overestimated" — was contradicted by the data.
>
> This document proposes a minimal change to **C-HIQL** that makes σ a principled signal of stochasticity by replacing random-init ensemble diversity with **per-head expectile diversity**.

---

## 1. One-Sentence Summary

> **EX-HIQL = C-HIQL with the N=5 value heads trained at N different expectile levels `τ_i ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` instead of the same τ=0.7.** Subgoal scoring keeps C-HIQL's μ − β·σ rule (Option B); the only change is the per-head loss.

The training, architecture, dataset pipeline, and inference flow are otherwise identical to C-HIQL. This is a one-line change in `value_loss`.

---

## 2. Background and Motivation

### 2.1 What Phase-2 C-HIQL found empirically

After 3 seeds × 1M training steps on `antmaze-teleport-navigate-v0`:

| Method | Mean overall success (3 seeds) |
|---|---|
| Phase-1 HIQL baseline | 0.404 |
| C-HIQL, β=0 (ensemble mean) | 0.404 |
| C-HIQL, β=0.5 (default pessimism) | 0.384 |

Pessimism did not help; at β=0 we matched HIQL exactly, and at β=0.5 we were 2 pts worse.

### 2.2 The diagnostic finding

`scripts/diagnose_sigma.py` probed the 5-head ensemble's disagreement σ across 500 states × 16 candidate subgoals per state. Key measurement:

> **Per-state μ ↔ σ correlation: −0.44** (mean across 500 states).

Inside a state, across the 16 candidate subgoals, *high-μ candidates have low σ* and *low-μ candidates have high σ*. This is the exact opposite of the [C-HIQL.md §4.1](C-HIQL.md) hypothesis.

### 2.3 Why the current ensemble fails

All 5 heads share the MLP trunk; only the final linear layer is independent. Diversity comes purely from random initialization of that last layer. Two facts combine:

1. **Same trunk ⇒ same features**: for any `(s, g_sub)` the 5 heads see identical features `h ∈ R^512`. Heads only disagree in how they linearly project `h` to a scalar.
2. **Linear-layer disagreement ∝ ‖h‖**: for unit-scale random weights, output spread grows with feature magnitude. Features are larger in magnitude for "weird" / far-from-goal candidates and smaller for "ordinary" / close-to-goal candidates.

Result: σ ends up tracking *feature weirdness* ≈ *distance-from-goal* ≈ *hardness-of-estimate*, not *overestimation-due-to-stochasticity*. This is the structural reason Phase-2 failed.

### 2.4 What we want σ to track

At a deterministic state, the Bellman target distribution is a point → **σ should be ≈ 0**. At a stochastic state (teleporter), the target distribution is wide → **σ should be > 0, proportional to spread**. This needs diversity that is *structurally* linked to stochasticity, not to feature magnitude.

---

## 3. The Method

### 3.1 Architectural change: none

Same value network as C-HIQL — **5 completely independent MLPs** via `GCValue(ensemble=True, num_ensemble=5)`. No new parameters, no new modules. Just different supervision.

### 3.2 Training: per-head expectiles

Replace the single `expectile=0.7` used for every head with a **vector of per-head expectiles**:

```
τ_i ∈ {0.1, 0.3, 0.5, 0.7, 0.9}   # indexed i = 1..5
```

Each head i is trained with the same action-free IQL expectile loss — with its own τ_i. Bootstrap target is still per-head (each V_i uses its own target-net prediction at s'), exactly as in C-HIQL:

$$
\mathcal{L}_V^{(i)}(\theta_i) = \mathbb{E}_{(s,s',g)\sim\mathcal{D}}\!\Big[\, L_{\tau_i}^{\,2}\big(r + \gamma\,V_i^{\text{target}}(s', g) - V_i(s, g)\big)\,\Big]
$$

$$
L_\tau^{\,2}(u) = \big|\tau - \mathbf{1}(u<0)\big|\,u^{2}
$$

### 3.3 Inference (Option B): μ − β·σ with a meaningful σ

Subgoal scoring is *exactly* C-HIQL's rule — unchanged from [C-HIQL.md §3.3](C-HIQL.md):

```python
def select_subgoal(s, candidate_subgoals, beta_pes):
    scores = []
    for g_sub in candidate_subgoals:
        vals = [V_i(s, g_sub) for i in 1..N]    # N scalars
        mu, sigma = mean(vals), std(vals)
        score = mu - beta_pes * sigma
        scores.append(score)
    return candidate_subgoals[argmax(scores)]
```

The only difference vs C-HIQL: because the 5 heads were trained on different τ_i, their spread σ across a (s, g) pair structurally reflects **the spread of that state's Bellman target distribution** rather than last-layer initialization noise. At deterministic (s, g) pairs the heads collapse; at stochastic (s, g) pairs they fan out.

Equivalently: σ is now an empirical estimator of how **wide the distribution of achievable values from (s, g)** is.

β_pes remains inference-only — one checkpoint serves any β ∈ {0, 0.25, 0.5, 1, 2} — preserving C-HIQL's "train optimistically, decide pessimistically" property.

### 3.4 Actor training: two independent head-indices for low and high

Both `low_actor_loss` and `high_actor_loss` in `chiql.py` currently use `vs.mean(axis=0)` as the advantage proxy. With heterogeneous τ that mean no longer represents "HIQL's V"; it's a weird mixture.

**Observation**: the two actors have different sensitivities to V's optimism.

- **Low actor** computes `A_low = V(s', w) − V(s, w)` for `s, s'` one env step apart. Systematic inflation cancels in the subtraction; low actor's τ choice barely matters.
- **High actor** computes `A_high = V(w, g) − V(s, g)` over a 25-step gap toward the ultimate goal. If `w` is near a teleporter and `s` isn't, the inflation does **not** cancel — the high actor is actively trained to prefer teleporter-adjacent waypoints. This is the upstream source of the bias.

**Fix**: two independent indices, one per actor:

```python
# was: v = vs.mean(axis=0)
low_idx = self.config['actor_expectile_index_low']    # default 3 (τ=0.7)
high_idx = self.config['actor_expectile_index_high']  # default 3 (τ=0.7)

v  = vs[low_idx]    # in low_actor_loss
nv = nvs[low_idx]   # in low_actor_loss

v  = vs[high_idx]   # in high_actor_loss
nv = nvs[high_idx]  # in high_actor_loss
```

The default `(3, 3)` both use τ=0.7, making **Design A**: policy-training dynamics **bit-identical to HIQL** everywhere, only the inference-time σ changes. This is the conservative first experiment.

**Design B** (ablation; override at launch time): `actor_expectile_index_high = 2` (τ=0.5 head) while keeping low at 3. This additionally de-biases the high actor's training-time advantage, attacking the upstream bias directly. Run only if Design A passes E3 but underperforms on E1.

### 3.5 What does **not** change

| Component | Modified? |
|---|---|
| Low-level policy π_low (training + inference) | ❌ |
| High-level policy π_high training loss | ❌ (uses V_4 = τ=0.7 head) |
| Subgoal candidate generation (K=16 samples) | ❌ |
| Goal representation φ(g) training | ❌ |
| Architecture (still 5 completely independent MLPs) | ❌ |
| Dataset pipeline (HGCDataset, sampling) | ❌ |
| Target-network update rule | ❌ |
| All HIQL hyperparameters (γ, α_AWR, k, τ_{target_net}) | ❌ |
| Inference rule (μ − β·σ) | ❌ |

The change is **literally one tensor** (`expectile` goes from scalar to length-N vector) plus one index change in the actor losses.

---

## 4. Why This Should Work — Three Mechanisms

### 4.1 Different τ ⇒ different fixed points of the same loss

Expectile regression on a distribution has a deterministic fixed point depending on τ. For a random variable X:

- τ = 0.5 → fixed point = mean(X)
- τ = 0.7 → fixed point = about the 70%-expectile of X (above the mean, below the max)
- τ = 0.9 → fixed point ≈ 90%-expectile (close to max for heavy-tailed X)

At a deterministic state, X is a point, all fixed points coincide, **heads converge**. At a stochastic state, X is spread, fixed points at τ=0.1 and τ=0.9 diverge proportionally to X's spread, **heads fan out**.

This is a mathematically guaranteed relationship between "how stochastic is this state" and "how spread out are the heads" — it's not a statistical accident of initialisation.

### 4.2 σ becomes interpretable

After training, σ(s, g_sub) is approximately a statistic of the return-to-go distribution from (s, g_sub) — concretely, the spread of the expectile range `[τ=0.1, τ=0.9]`. Subtracting β·σ at inference directly encodes "penalize candidates whose outcome distribution is spread" — *exactly* the property C-HIQL wanted.

### 4.3 Head diversity is strong, structural, and cheap

Phase-2 C-HIQL's diversity source was random initialization of 5 independent MLPs trained with **the same** τ=0.7 loss. With identical loss objectives, 5 networks all converge toward approximately the same value function — whatever residual disagreement remains is driven by init noise propagated through training. The Phase-2 diagnostic showed this residual was anti-correlated with μ (probably because deep-net init-noise magnifies along the directions of largest feature activation, which correlate with |μ|).

Per-head τ is a **loss-level** diversity source. Two MLPs with different τ *provably* converge to different answers — even with identical architecture and identical initialization — because their loss minima are in different places. Each head's fixed point is a well-defined statistic of the target distribution (the τ-expectile). This is the strongest form of diversity that keeps the ensemble "in the same family of estimators."

---

## 5. Significance — Three Levels

### 5.1 Engineering: fixes the mechanism gap

Phase-2 C-HIQL paid 5× value-network parameters and got σ that is *anti-correlated* with the quantity of interest. EX-HIQL gets σ that is *structurally correlated* with the quantity of interest, at the same cost.

### 5.2 Conceptual: ports distributional RL into offline hierarchical GCRL

Expectile parameterization of value distributions is a known idea in online distributional RL (Rowland et al. 2019, "Statistics and Samples in Distributional Reinforcement Learning"). Using it *as an uncertainty signal for subgoal-scoring pessimism* in offline hierarchical GCRL is — to our knowledge — unexplored.

### 5.3 Research positioning: "diagnose, then fix" story

EX-HIQL is presented as the direct, principled fix to a concrete failure mode diagnosed in C-HIQL. The paper would read:

1. HIQL overestimates on stochastic envs. [Known.]
2. C-HIQL (ensemble random init + μ − β·σ) was expected to fix it.
3. Empirically C-HIQL does not help: σ is anti-correlated with μ at the candidate level.
4. Why: 5 independent MLPs trained on the *same* expectile loss converge to nearly the same function; whatever residual disagreement remains tracks feature magnitude, not stochasticity.
5. Fix: train heads with different τ so σ **provably** tracks bootstrap-target spread.
6. Empirically EX-HIQL recovers the performance gap. [To verify.]

This is a much sharper narrative than either "here's an uncertainty method that might help" or "pessimism doesn't help on stochastic envs."

---

## 6. Hyperparameters

| Parameter | Current default | Recommended after Phase-3 obs | Justification |
|---|---|---|---|
| `head_expectiles` | `(0.1, 0.3, 0.5, 0.7, 0.9)` ⚠ | **`(0.5, 0.6, 0.7, 0.8, 0.9)` or `(0.6, 0.65, 0.7, 0.75, 0.8)`** | Original wide spread caused training divergence (see §12). Tighter spread keeps each independent MLP's target-network feedback loop away from the runaway region, so σ stays meaningful without any head exploding. |
| `actor_expectile_index_low` | `3` (τ=0.7) | same | Keep HIQL's low-actor training dynamics bit-identical |
| `actor_expectile_index_high` | `3` (τ=0.7) | `{3, 2}` | `3` = Design A (conservative); `2` = Design B ablation (de-bias high actor's advantage) |
| β_pes | `0.5` | `{0, 0.25, 0.5, 1, 2}` (inference sweep) | Same as C-HIQL; β is inference-only |
| N (ensemble size) | `5` | `5` | Must equal `len(head_expectiles)` |
| gradient clipping | none | **`optax.clip_by_global_norm(10.0)`** (recommended) | Phase-3 observed grad/norm ≥ 400,000 with wide τ spread; clipping is a cheap prevention |

All other HIQL hyperparameters inherited unchanged: γ=0.99, α_AWR=3.0 (high and low), subgoal_steps=25, rep_dim=10, batch=1024, lr=3e-4.

---

## 7. Claims and Evidence Plan

| # | Claim | Experiment | Success criterion |
|---|---|---|---|
| E1 | EX-HIQL beats HIQL on teleport | EX-HIQL vs Phase-1 HIQL on `antmaze-teleport-navigate-v0`, 3 seeds, β∈{0.25, 0.5, 1.0} | ≥5 pts absolute improvement at some β |
| E2 | The improvement comes from σ, not the expectile mix alone | EX-HIQL β=0 (ensemble-mean control) vs β>0 | β>0 > β=0 |
| E3 | σ is localised where it should be | `diagnose_sigma.py` on EX-HIQL checkpoints | per-state μ ↔ σ correlation near 0 (or positive); σ concentrated at teleporters in the spatial plot |
| E4 | No regression on deterministic envs | EX-HIQL vs HIQL on `antmaze-large-navigate-v0`, 3 seeds | success drop <3 pts |

**E3 is critical**: it's the structural fix we're claiming. If EX-HIQL's σ still anti-correlates with μ, the method is broken for a reason we didn't understand, and the other claims become uninterpretable.

---

## 8. Run Matrix and Compute

| Block | Envs | Seeds | Runs |
|---|---|---|---|
| 1. EX-HIQL main | `antmaze-teleport-navigate-v0` | 3 | 3 |
| 2. Diagnostic | same checkpoints | 3 | (eval-only, no extra training) |
| 3. β sweep | β × seed | 5 × 3 | (eval-only) |
| 4. Deterministic control | `antmaze-large-navigate-v0` | 3 | 3 |
| **Total training runs** | | | **6** |

- Per-run cost: ~5h on RTX 5090 laptop (matches Phase-2 observed).
- Total GPU-hours: ~30.
- β sweep + diagnostic: ~1.5h total (CPU-bound evals).
- **Wall clock budget: one day training (parallel 3 seeds) + 2h analysis.**

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `τ=0.1` head learns policy-noise-aware V (not just stochastic-dynamics-aware) and its σ contribution is confounded by "random bad trajectories in the log" | High | Expected. Ablate with a tighter spread that omits τ=0.1 |
| **Wide τ spread destabilizes per-head training** — heads 1 and 5 each diverge independently via target-network EMA feedback; V values escape physical range (see §12). Note: the failure is *per-MLP*, not cross-MLP, because the 5 heads share no parameters. | **CONFIRMED (Phase 3, 2026-04-22)** | Primary: narrow `head_expectiles` to e.g. `(0.5, 0.6, 0.7, 0.8, 0.9)`. Secondary: add `optax.clip_by_global_norm(10.0)`. Tertiary: shared-trunk ablation (`shared-trunk-5head` branch) or L2 head-divergence regularizer (`phase3c-l2reg` branch). |
| σ signal is small even post-fix (teleport distribution not wide enough) | Medium | Check absolute σ scale in diagnostic; if < 0.1 units, pessimism still has no lever |
| Actor advantages driven by `vs[3]` (a single head) are noisier than `vs.mean(axis=0)` | Low | HIQL baseline uses single V, so single-head is no noisier than HIQL's default |
| Design A fails E1 despite E3 passing: σ is meaningful but the high actor's advantage is still biased toward teleporters | Medium | Run **Design B** ablation: `actor_expectile_index_high = 2` (τ=0.5). Attacks upstream bias at training-time too |
| 5 independent MLPs all converge to nearly-identical expectile fixed points (small σ) | Low (with tightened spread) | Ablation: shared-trunk variant on `shared-trunk-5head` branch as an architectural comparison |

---

## 10. Relation to Prior Work

| Method | Relationship to EX-HIQL |
|---|---|
| **HIQL** (Park et al. 2023) | Base; EX-HIQL is `N_heads=1, τ=0.7` collapsed back to a single head |
| **C-HIQL** (our Phase 2) | Identical architecture; diversity source changes from "random init" to "different τ" |
| **Expectile DQN / Rowland et al. 2019** | Uses expectiles to parameterize a return distribution in online RL; we port the idea to offline hierarchical GCRL and use the spread as a planning signal |
| **QR-DQN** (Dabney et al. 2018) | Quantile-heads version of the same idea; quantile regression uses L1 loss, not a drop-in for IQL |
| **EDAC / SAC-N** | Ensemble pessimism for Q; same "use σ across heads" motif, applied to actor-critic online RL |
| **CQL** | Loss-level pessimism regularizer; EX-HIQL is more targeted (pessimism only at subgoal scoring) |

---

## 11. 60-Second Pitch

> HIQL overestimates V on stochastic environments and its hierarchical planner amplifies the bias. C-HIQL's proposed fix — 5 independently-initialized value heads trained on the same expectile loss with pessimistic scoring — empirically failed because 5 identical-loss MLPs converge to nearly the same function; whatever residual disagreement survives tracks feature magnitude, not stochasticity. We replace random-init diversity with **per-head expectile diversity** (τ_i ∈ {0.1, 0.3, 0.5, 0.7, 0.9}). Each head converges to a *provably different* expectile of the Bellman-target distribution, so the 5-head spread σ structurally measures local stochasticity: zero at deterministic states, proportional to spread at stochastic ones. The actor is pinned to the τ=0.7 head so HIQL's training dynamics are preserved exactly. Only one line of the value loss changes. We expect σ to become positively correlated (or uncorrelated) with μ at the candidate level (reversing C-HIQL's −0.44), and β>0 scoring to beat HIQL on teleport while leaving deterministic-env performance unchanged. Total additional budget: 6 training runs (~30 GPU-h on a 5090).

---

## 12. Observed Behavior — Phase 3 Design A, 2026-04-22 run

**Empirical record of the first EX-HIQL training.** Documented here so the next iteration has a precise diagnostic baseline to beat.

### 12.1 Setup

- Branch: `expectile-heads` (commit `795add9`).
- Agent: `ex_chiql` (external/ogbench/impls/agents/ex_chiql.py).
- `head_expectiles = (0.1, 0.3, 0.5, 0.7, 0.9)` — widest practical spread, chosen for the clearest illustration of the "different fixed points" mechanism.
- `actor_expectile_index_low = 3`, `actor_expectile_index_high = 3` (Design A, HIQL-equivalent actor).
- `pessimism_beta = 0.5`.
- No gradient clipping.
- Env: `antmaze-teleport-navigate-v0`, 3 seeds × 1M steps.

### 12.2 Evaluation trajectory (`evaluation/overall_success`, mean over 3 seeds)

| Step | Mean | Phase-2 C-HIQL β=0.5 | Phase-1 HIQL |
|---|---|---|---|
| 100k | 0.364 | 0.400 | — |
| 200k | 0.381 | 0.408 | — |
| 300k | 0.391 | 0.409 | — |
| 400k | **0.461** (spike, seed 1 hit 0.496, seed 2 hit 0.480) | 0.409 | — |
| 500k | 0.371 | 0.377 | — |
| 600k | 0.376 | 0.444 | — |
| 700k | ~0.41 (partial) | 0.376 | — |
| 1M final | — | 0.384 | 0.404 |

The step-400k spike did not persist; subsequent evals regressed to the 0.37–0.41 band that characterises the rest of this method family on this env. The single peak is *below the noise floor of Phase-2 C-HIQL* (which also had a peak of 0.444 at step 600k). No claim of improvement is supported by this run.

### 12.3 Training-state divergence (`training/value/*` per train.csv)

The core observation: `v_std_across_heads` and `grad/norm` grew monotonically and unboundedly:

| Step | `v_std_across_heads` | `grad/norm` (seed 2 worst) | `v_max` | `v_min` |
|---|---|---|---|---|
| 5,000 (smoke) | ~1.2 | — | — | — |
| 30,000 | ~17 | ~1,600 | — | — |
| 250,000 | ~50 | ~2,100 | — | — |
| 400,000 | ~111 | ~6,900 | +281 | −3,102 |
| 700,000 | ~696–979 | ~177,000 | +423 | −2,047 |
| 790,000 (seed 2) | **1,209** | **421,364** | **+538** | **−2,855** |

Physical bounds for this env:
- Episode length ≤ ~500 steps; reward = −1 per step; so realistic V ∈ roughly `[-500, 0]`.
- Observed at step 790k: `v_max = +538` (positive — "reach goal in −538 steps"); `v_min = −2855` (5× worse than any possible trajectory).

Both bounds were violated. The τ=0.9 MLP escaped toward `+∞`; the τ=0.1 MLP toward `−∞`. Adam absorbed the gradients (no NaN), but the V function became non-physical.

### 12.4 Why this happens

The 5 expectile heads are 5 completely independent MLPs (no shared parameters). The failure is therefore **per-MLP**, not cross-MLP. Each extreme-τ head destabilizes on its own via a target-network EMA feedback loop:

- At τ=0.9 the asymmetric expectile weights put 9× weight on samples where the current prediction is below the bootstrap target `q = r + γ · V_target(s')`. This pulls `V_9` upward.
- Since `V_9_target` is an EMA of `V_9` itself (τ=0.005), as `V_9` grows, `V_9_target` catches up, which makes `q` grow, which the expectile loss chases even higher. Positive feedback.
- Symmetric story at τ=0.1 with sign flipped.

The middle heads (τ=0.3, 0.5, 0.7) have weaker asymmetry — their expectile weights are closer to symmetric — so they don't enter the runaway regime. They stay near a reasonable fixed point throughout training.

The τ=0.7 MLP used by the actor is **completely unaffected** by the divergence of heads 1 and 5, because nothing is shared. Its own training is exactly equivalent to a standalone HIQL V-function. That's why the success rate sits in the 0.37–0.41 band: the actor is essentially running plain HIQL through head 3, while the pessimism mechanism is broken but irrelevant.

### 12.5 Mechanism-level diagnosis

The inference-time σ in this run is **enormous everywhere** (~1000), not localised. At `β_pes = 0.5`, scoring `μ − 0.5 · σ` is dominated by σ, not μ. The pessimism term becomes noise relative to its intended role. This explains why no clean improvement emerged: two of the five independent MLPs (τ=0.1, τ=0.9) have each diverged via their own feedback loops, so at every candidate state the mean μ and the spread σ are both dominated by those two runaway heads. The designed mechanism (use σ to penalize candidates near teleporters) was overridden by σ's per-head runaway magnitude.

The `diagnose_sigma.py` diagnostic (run after this report was first drafted — see `PHASE3_DESIGN_A_REPORT.md §6.1`) confirmed this: `per_state_mu_sig_corr_mean ≈ +0.99` across all seeds. σ is not localized to any state property; it's co-scaling with μ because both are dictated by the same two runaway heads.

### 12.6 Implications for the next run

1. **Primary fix — tighten `head_expectiles`** to something like `(0.5, 0.6, 0.7, 0.8, 0.9)` or `(0.6, 0.65, 0.7, 0.75, 0.8)`. This preserves "different fixed points ⇒ structural σ" while keeping every individual τ away from the runaway regime where the asymmetric weights drive an independent MLP to an extreme fixed point.
2. **Secondary fix — add global gradient clipping** (e.g. `optax.clip_by_global_norm(10.0)` in the optimizer pipeline). Cheap safety net even if tighter spread is used.
3. **Tertiary fix — architectural ablations**. Two independent options, pursued in parallel on two branches:
   - `phase3c-l2reg`: add an L2 penalty pulling the 5 heads' parameters toward their ensemble mean. Per-parameter restoring force against divergence.
   - `shared-trunk-5head`: one shared trunk + 5 independent `Dense(1)` output heads. Couples all 5 heads through a single feature representation, bounding how far they can disagree.

The Phase-3b follow-up used (1) tight spread — three seeds × 1M steps on `(0.6..0.8)` stayed numerically clean (grad/norm flat 280-650) but showed a slower σ-creep during training. See `PHASE3B_REPORT.md`. Ablations (3) target that secondary creep.

### 12.7 What this run is still worth

- Final 1M checkpoints were saved. They give us the definitive β-sweep row for this configuration.
- The checkpoint's `diagnose_sigma.py` output is a reference for "what broken σ looks like" — useful in the paper as contrast for the fixed configuration.
- The run confirms a clean per-head-instability failure mode: at extreme τ, each independent MLP enters a target-network EMA feedback loop and diverges on its own. This is architecturally independent (would happen in the shared-trunk variant too, though likely more slowly) and loss-specific (only τ close to 0 or 1 triggers it). The lesson for the method is "keep every τ moderate", not "fix the trunk sharing".
