# HIQL and C-HIQL: The Full Picture

A ground-up explainer of what HIQL is doing in our codebase, why it works, where it fails, and exactly what C-HIQL changes. Intended for team members who did not design the Phase 2 proposal.

---

## 1. The problem: offline goal-conditioned RL

**Setup.** We have a pre-collected dataset of trajectories `D = {(s_t, a_t, s_{t+1})}` from some behavior policy. We are given a task at test time specified by a final goal `g`, and we must output an action policy π(a | s, g) that reaches g from any starting state.

**What makes it hard (vs. standard offline RL):**

1. **No explicit reward.** The reward signal is derived: `r = 0 if s == g else −1` (sparse), or `r = 1 if s == g else 0`. There's no shaped reward to learn from.
2. **Every test-time goal is different.** A single policy must generalize across goals never seen at training time.
3. **Long horizons.** Many OGBench tasks require 500+ steps of coordinated action (antmaze-large, humanoidmaze). Credit assignment over that horizon is brutal with sparse rewards.
4. **Offline constraint.** We cannot collect new data. The policy must extract value from only what the dataset covers, without querying the environment during training.

**What OGBench measures.** `overall_success` = fraction of test episodes where the agent reached within some radius of `g` before the time limit, averaged over multiple goals per dataset.

---

## 2. Building block 1: IQL (expectile regression)

HIQL is built on **IQL (Implicit Q-Learning)**, so you need the IQL idea first.

In standard offline Q-learning, you update Q(s, a) toward `r + γ · max_{a'} Q(s', a')`. The problem: for actions `a'` that aren't in the dataset, Q(s', a') is an extrapolation — and neural nets extrapolate optimistically. The agent learns to exploit these phantom high-Q actions, and policies fail in reality.

**IQL's fix:** don't take a max. Instead, learn a V(s) that is an **expectile** of the Q distribution over actions in the dataset:

```
V(s) ≈ arg min_V  E_{(s,a) ~ D}  expectile_loss(Q(s,a) − V(s), τ)

expectile_loss(δ, τ) = |τ − 𝟙{δ < 0}| · δ²        # τ ∈ (0.5, 1)
```

With `τ = 0.7` (HIQL's default), this asymmetrically pulls V upward when Q > V and weakly downward when Q < V. The result: V approximates "what a *slightly good* action from the dataset achieves," without ever querying unseen actions.

Then Q gets updated toward `r + γ · V(s')` — still only using in-distribution quantities.

The payoff: offline-safe value estimates that are conservative enough to avoid extrapolation exploits, but optimistic enough (τ > 0.5) to identify the better in-dataset actions.

---

## 3. Building block 2: Goal-conditioned IQL

For GCRL, everything takes `g` as an extra input: V(s, g), Q(s, a, g). OGBench has two direct variants in your Phase 1 table:

- **GCIVL**: learns only V(s, g), uses it for a policy via AWR (advantage-weighted regression).
- **GCIQL**: learns both Q(s, a, g) and V(s, g).

These are flat (non-hierarchical). They work well on manipulation (where horizons are short and state is structured) but struggle on long-horizon mazes: the advantage signal gets drowned out over 500 steps.

---

## 4. HIQL's key insight: hierarchy through learned representations

**The problem HIQL solves:** On long horizons, the advantage `A(s, a, g) = Q(s, a, g) − V(s, g)` is tiny for most actions — they all move you a little bit closer. AWR's weighting `exp(α · A)` becomes essentially uniform, and the policy collapses to behavior cloning.

**HIQL's answer:** decompose into two layers. The high level picks a **nearby** target (25 steps ahead), the low level handles the short-horizon steering. Advantages computed over 25-step horizons are large and informative; AWR works again at both levels.

But the question is: what is the nearby target?

### 4.1 The subgoal is a learned representation, not a state

HIQL introduces a small encoder `φ([s, g])` that maps `(current state, any target)` to a length-normalized 10-dim vector. This module is defined at [hiql.py:226-233](external/ogbench/impls/agents/hiql.py#L226-L233) as `goal_rep`:

```python
goal_rep_seq = [
    MLP(hidden_dims=(*value_hidden_dims, rep_dim=10), layer_norm=True),
    LengthNormalize(),
]
```

Crucially, `φ` has no loss of its own. It's trained **by backpropagation through V and the actors**. So `φ` learns to encode whatever information is useful for predicting value and choosing actions toward a goal — automatically discarding irrelevant state dimensions (joint velocities, nearby object positions that don't matter, etc.).

### 4.2 The learned 10-sphere

The length-normalize step forces φ onto the unit sphere. Two consequences:

- Magnitudes can't be gamed. The network can't hack around a loss by inflating ||φ|| — all "output energy" must go into direction.
- Distances have meaning. Cosine similarity between two φ values reflects "how similar are these goals from the perspective of what the value function cares about." This is the same trick that makes contrastive representation learning work.

### 4.3 Why the subgoal isn't just a raw state

Your first question: why 10-dim instead of a raw sub-state?

| Approach | Problem |
|---|---|
| Predict raw state `s_{t+25}` | Network must output ~30–12000 dims depending on env. Training is unstable, high-capacity, and most of the state is irrelevant (joint velocities, object colors, etc.). |
| Predict a hand-picked feature (x,y) | Would work in antmaze but not in powderworld (pixels); loses end-to-end benefit; requires env-specific design. |
| **Predict 10-dim φ([s, s_{t+25}])** | Shared across envs. Length-normalized so distances are stable. Learned end-to-end to retain only value-relevant info. HIQL's choice. |

The **subgoal representation** is literally a direction in a learned latent space. The low-level actor is conditioned on this direction and learns to steer toward it. That works because the representation was shaped by V's loss to align with "moves that increase value."

---

## 5. HIQL's four networks

HIQL has **four** learnable modules. All are defined in [hiql.py](external/ogbench/impls/agents/hiql.py):

| Module | Signature | What it does | Trained by |
|---|---|---|---|
| `goal_rep` φ | `([s; g]) → 10-dim` | Encodes (state, goal) into a length-normalized representation | Backprop through V and actors |
| `value` V | `(s, g) → scalar` | Estimates expected return from s toward g (2-head ensemble) | Expectile regression |
| `high_actor` π_high | `(s, g) → 𝓝(μ, σ) on ℝ¹⁰` | Gaussian over **subgoal representations** given (current state, final goal) | AWR on 25-step advantage |
| `low_actor` π_low | `(s, φ_sub) → 𝓝` on actions | Gaussian over actions given state and subgoal rep | AWR on 1-step advantage |

Note what ISN'T here: there is **no separate V_high and V_low**. One shared V serves both advantages. The hierarchy is entirely in the actors.

### 5.1 Why the high-level takes g, not just s

Your second question. A goal-conditioned policy must behave differently for different goals:

- π_high(s = center of maze, g = top-left) → subgoal rep pointing northwest
- π_high(s = center of maze, g = bottom-right) → subgoal rep pointing southeast

If π_high only saw s, it would have to average over all possible g's seen in training and produce the same subgoal for both. There's no way to recover "which direction am I supposed to be going" from just the current state. **All HIQL networks are goal-conditioned** — that's what the "GC" in GCRL means.

---

## 6. How HIQL trains (three losses, summed)

At each gradient step, HIQL samples a batch of transitions `(s, a, s', masks)` along with three types of relabeled goals — future state in the same trajectory (`trajgoal`), random state (`randomgoal`), current state (`curgoal`). It then computes:

### 6.1 Value loss (IQL expectile regression)

[hiql.py:27-58](external/ogbench/impls/agents/hiql.py#L27-L58). V(s, g) learns to satisfy Bellman consistency under the expectile operator:

```
q = r + γ · min(V_target(s', g))
V(s, g) = expectile_regression_toward q   at τ=0.7
```

The "min(V_target)" is the standard double-Q min-of-2 trick for conservatism. This is where the 2-head ensemble comes from in vanilla HIQL — it's a bias-reduction trick, NOT uncertainty quantification.

### 6.2 Low-level AWR loss

[hiql.py:60-97](external/ogbench/impls/agents/hiql.py#L60-L97). For a relabeled "low-level goal" g_low (a short-horizon target, typically 25 steps ahead), compute the 1-step advantage:

```
adv_low = V(s', g_low) − V(s, g_low)
w_low   = exp(α_low · adv_low)                        # α_low = 3
loss_low = − w_low · log π_low(a | s, φ([s, g_low]))  # weighted BC
```

### 6.3 High-level AWR loss

[hiql.py:99-124](external/ogbench/impls/agents/hiql.py#L99-L124). Teach π_high to predict "the φ-representation of a state 25 steps in the future on trajectories that ended successfully at g."

```
target = φ([s, s_{t+25}])                              # what a good subgoal rep was
adv_high = V(s_{t+25}, g) − V(s, g)                    # 25-step advantage toward g
w_high   = exp(α_high · adv_high)
loss_high = − w_high · log π_high(target | s, g)
```

The three losses are summed and optimized jointly.

---

## 7. How HIQL runs at inference

[hiql.py:168-192](external/ogbench/impls/agents/hiql.py#L168-L192):

```python
# At each environment step:
φ_sub    = π_high(s, g).sample()              # ONE subgoal rep, stochastic
φ_sub    = length_normalize(φ_sub)
a        = π_low(s, φ_sub, goal_encoded=True).sample()
step(a)
```

This repeats every step. Note: the high-level is queried **every single step**. The agent doesn't commit to a subgoal for 25 steps and then re-plan — it re-samples a subgoal each step, just with the expectation that subgoal "drifts" slowly.

---

## 8. Where HIQL fails: stochastic transitions

Your Phase 1 results:

| Env | HIQL | Best competitor |
|---|---|---|
| antmaze-large-navigate (deterministic) | **0.889** | CRL 0.865 |
| humanoidmaze-medium-navigate (deterministic) | **0.907** | CRL 0.576 |
| antmaze-large-stitch (deterministic) | **0.677** | QRL 0.212 |
| **antmaze-teleport-navigate (STOCHASTIC)** | **0.404** | CRL **0.539** |

HIQL wins everything deterministic, loses to non-hierarchical CRL on teleport. The question is *why.*

### 8.1 The teleport env

`antmaze-teleport` has tiles that, when stepped on, randomly teleport the agent to a different location. So transitions become stochastic: the same `(s, a)` can lead to many different `s'`.

### 8.2 The optimism cascade

Expectile regression with τ = 0.7 biases V *upward*. Specifically, V learns to predict approximately the 70th percentile of TD backups from a given state:

> "What's a reasonably-good outcome from s toward g?"

This is *fine in deterministic envs* — the 70th percentile ≈ the 50th percentile ≈ the actual outcome. But in a teleport env, the 70th percentile of `V_target(s', g)` is pulled up by the rare good teleports: occasionally the teleport lands you right next to g. The value function remembers those lucky transitions and overvalues the tiles where teleports happen.

Now the cascade:

1. V(s, g_sub) is overestimated for subgoals near teleport tiles (because lucky teleports pulled their value up).
2. The high-level AWR loss upweights subgoal reps that correspond to those high-V regions.
3. π_high learns to place mass on subgoal reps that look like teleport destinations.
4. At inference, a single sample from π_high lands on one of these "too-good-to-be-true" subgoals.
5. The low-level dutifully steers toward it, gets teleported, and ends up somewhere uncorrelated with g.

The single-sample commitment is the final nail: even if π_high assigns non-trivial probability to a *safe* subgoal, the single draw might still land on a dangerous one. HIQL has no mechanism to prefer reliability over expected value.

### 8.3 Why CRL wins here

CRL doesn't use a value function at all — it uses a contrastive estimator that learns `"is s' reachable from s?"` directly from the dataset statistics. Stochastic transitions dilute the contrastive signal more uniformly, which turns out to be less damaging than expectile regression's selective upward bias. CRL's 0.539 isn't a great score — it's just that HIQL drops harder.

---

## 9. What C-HIQL changes

C-HIQL targets exactly the failure mode in §8.2: **high-level selection trusts an optimistic V too much, with no uncertainty filter.**

### 9.1 Training change: 5-head value ensemble

Replace the 2-head `V` with **N = 5 independent heads**, each trained on the same expectile regression loss with different random initialization. Heads diverge wherever the data is ambiguous, and they converge wherever the data is consistent.

Disagreement among heads (the standard deviation σ across head outputs) is an **epistemic uncertainty estimate**: high σ = "we're not sure about this value."

This is a training-time change — the ensemble is baked into the trained checkpoint.

### 9.2 Inference change: K-sample pessimistic selection

At each environment step, instead of one sample from π_high:

```python
# C-HIQL inference
φ_candidates = [π_high(s, g).sample() for _ in range(K=16)]   # K candidate subgoal reps
for each φ_k:
    V_all = [V_head_i(s, φ_k) for i in 1..5]                   # 5 scalar values
    score_k = mean(V_all) − β · std(V_all)                     # pessimistic utility
φ_sub = argmax(score_k)                                        # pick the most reliable
a     = π_low(s, φ_sub).sample()
```

Three things to notice:

- **β = 0** recovers "ensemble-mean K-sample" — picking the highest-expected-value subgoal among K draws. This is NOT vanilla HIQL (which is single-sample).
- **β > 0** penalizes disagreement. Subgoals in regions the heads agree on (deterministic parts of the env) are preferred over subgoals in regions the heads disagree (teleport tiles, out-of-distribution goals).
- **One checkpoint serves all β.** β only enters inference, so we sweep it cheaply after training once.

### 9.3 What does NOT change

- Every loss function (value, high-actor, low-actor)
- The goal_rep encoder φ
- π_high and π_low architectures
- All hyperparameters except the three new knobs (`num_value_heads=5, num_subgoal_candidates=16, pessimism_beta=0.5`)

This is intentional — the narrower the surgery, the more credibly any improvement can be attributed to uncertainty-aware subgoal selection specifically.

---

## 10. The clean-ablation table

To prove that the *pessimism* is what helps (not just "sampling K and picking the best"), the experimental table needs three rows:

| Method | # V heads | K (candidates) | β (pessimism) | Isolates |
|---|---|---|---|---|
| HIQL baseline | 2 | 1 | — | vanilla |
| C-HIQL, β = 0 | 5 | 16 | 0 | ensemble-mean selection (no pessimism) |
| C-HIQL, β > 0 | 5 | 16 | {0.25, 0.5, 1.0, 2.0} | full method |

If β=0 alone improves on HIQL, the benefit is "more samples + higher expected-value selection," not uncertainty. If only β>0 improves, pessimism is doing real work. Most likely result: β=0 gives some bump, β>0 gives more — both are meaningful, and the paper should report both honestly.

---

## 11. Why this *should* work (the intuition)

On deterministic envs, the heads agree (σ ≈ 0), so `μ − β · σ ≈ μ`. C-HIQL degenerates to ensemble-mean HIQL, which should match or very slightly beat HIQL. That's the "no regression" claim.

On teleport, heads *disagree specifically on teleport-adjacent subgoals* — because their training signal from those tiles is noisy (different seeds pick up different lucky backups). σ is high there. The pessimism term downweights exactly those bad subgoals. The agent gets routed around teleport tiles.

The mechanism is **epistemic uncertainty as a planning filter** — a classical risk-sensitive RL idea, applied at the subgoal-selection layer.

---

## 12. What this won't solve

Be intellectually honest with the report:

- **Manipulation collapse** (cube-double, scene, puzzle-3x3): HIQL under-performs GCIVL/GCIQL here. The failure isn't stochasticity — it's that the hierarchy itself is miscalibrated for manipulation (objects, not paths). C-HIQL won't touch this.
- **Powderworld-hard**: everything fails. Not a uncertainty problem; probably a representation/exploration problem we can't address in 3 days.
- **"No regression"** is a claim, not a guarantee. If ensemble heads collapse to identical functions (σ ≈ 0 everywhere), C-HIQL has no leverage. Watch `v_std_across_heads` during training.

The paper should scope the contribution carefully: *"Lightweight uncertainty-aware subgoal selection fixes HIQL's known failure mode on stochastic transitions without disrupting its strengths on deterministic navigation."* That's a defensible claim. "C-HIQL is universally better than HIQL" is not.

---

## 13. Pointers to the code

| Concept | File / lines |
|---|---|
| HIQL agent (read this first) | [external/ogbench/impls/agents/hiql.py](external/ogbench/impls/agents/hiql.py) |
| Value loss (expectile regression) | [hiql.py:27-58](external/ogbench/impls/agents/hiql.py#L27-L58) |
| High-actor AWR loss | [hiql.py:99-124](external/ogbench/impls/agents/hiql.py#L99-L124) |
| Inference (single-sample subgoal) | [hiql.py:168-192](external/ogbench/impls/agents/hiql.py#L168-L192) |
| goal_rep φ encoder | [hiql.py:219-233](external/ogbench/impls/agents/hiql.py#L219-L233) |
| GCValue (2-head hardcoded) | [external/ogbench/impls/utils/networks.py:269-314](external/ogbench/impls/utils/networks.py#L269-L314) |
| ensemblize helper | [networks.py:15-25](external/ogbench/impls/utils/networks.py#L15-L25) |
| Agent registry | [external/ogbench/impls/agents/__init__.py](external/ogbench/impls/agents/__init__.py) |
| OGBench main loop | [external/ogbench/impls/main.py](external/ogbench/impls/main.py) |
| C-HIQL proposal | [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md) |
| C-HIQL execution plan | [PHASE2_TRAINING_PLAN.md](PHASE2_TRAINING_PLAN.md) |
