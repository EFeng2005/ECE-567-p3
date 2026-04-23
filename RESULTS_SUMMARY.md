# ECE 567 Project 3 — Results Summary (2026-04-23)

**Repo**: `EFeng2005/ECE-567-p3`
**Env**: `antmaze-teleport-navigate-v0` (OGBench)
**Training budget**: 3 seeds × 1M steps per configuration, on a single RTX 5090 laptop inside WSL2.

This is a point-in-time, consolidated summary of every completed configuration. Detailed per-phase reports remain in `PHASE*_REPORT.md`, `EXPECTILE_HIQL.md`, and `PHASE3B_IMPROVEMENT_PLAN.md`. This file is the top-level readout.

---

## 1. Executive summary

Five configurations, laid out as a 2×2 matrix of **architecture × head supervision**, plus the HIQL single-head baseline:

|                   | Independent trunks (5 full MLPs)       | Shared trunk (1 MLP + 5 last-layer heads) |
|-------------------|----------------------------------------|-------------------------------------------|
| **Same τ=0.7**    | Phase-2 **CHIQL**  (0.384 step-1M)     | shared-trunk **CHIQL**  (0.392 step-1M)   |
| **Per-head τ**    | Phase-3a wide (0.427), Phase-3b tight (0.417) | shared-trunk EX-HIQL  — *training in progress* |

Reference: **HIQL baseline 0.404 step-1M** (single-head, same τ=0.7).

**Headline result**: indep-trunks + per-head-τ beats HIQL at step 1M; nothing else reliably does. Specifically **Phase-3b (0.6, 0.65, 0.7, 0.75, 0.8) + β=0.5 pessimism at inference** is the best *trustworthy* result — +1.3 pt over HIQL at step 1M, +1.5 pt at its step-400k peak, with clean training throughout.

Phase-3a (wide-τ) scores slightly higher at step 1M (+2.3 pt) and at peak (+3.7 pt), but its value network is **numerically broken** (grad/norm → 4.7M, V → −14,700); the "success" is the actor running off head 3 (τ=0.7) as if it were plain HIQL, not the designed pessimism mechanism doing its job.

---

## 2. Full results table

All numbers are 3-seed means of `evaluation/overall_success`. Inference β=0.5 everywhere. The "step-1M" column is the standard OGBench reporting convention; the "peak" column reports the best step at which all 3 seeds had an eval row, with the step index in parentheses.

| Config | Architecture | head_expectiles | step-1M mean | peak mean (step) | Δ vs HIQL @1M |
|---|---|---|---|---|---|
| HIQL (baseline) | 1 MLP | τ=0.7 | **0.4040** | 0.4240 (600k) | — |
| Phase-2 CHIQL | 5 indep MLPs | same τ=0.7 | 0.3840 | 0.4440 (600k) | −2.0 pts |
| **Phase-3a** (wide-τ) | 5 indep MLPs | 0.1, 0.3, 0.5, 0.7, 0.9 | **0.4267** | **0.4613 (400k)** | **+2.3 pts** *(see caveat §3)* |
| **Phase-3b** (tight-τ) | 5 indep MLPs | 0.6, 0.65, 0.7, 0.75, 0.8 | **0.4173** | 0.4387 (400k) | **+1.3 pts** *(reliable)* |
| Shared-trunk CHIQL | 1 trunk + 5 heads | same τ=0.7 | 0.3920 | 0.4107 (800k) | −1.2 pts |
| Shared-trunk EX-HIQL | 1 trunk + 5 heads | 0.6, 0.65, 0.7, 0.75, 0.8 | *(training)* | *(training)* | — |

**Per-seed breakdowns** in `scripts/compare_results.py` output (run it for the latest numbers):

- Phase-3a step-1M: `0.460 / 0.384 / 0.436`
- Phase-3b step-1M: `0.452 / 0.432 / 0.368`
- Shared-trunk CHIQL step-1M: `0.396 / 0.404 / 0.376`

---

## 3. Best checkpoint — details and caveats

### 3.1 Nominally highest: Phase-3a wide-τ

- Path: [results/antmaze-teleport-navigate-v0/ex_chiql_phase3_design_a/seed{0,1,2}/params_1000000.pkl](results/antmaze-teleport-navigate-v0/ex_chiql_phase3_design_a/) (82 MiB each)
- Step-1M 3-seed mean: **0.4267** (highest across all runs)
- Peak 3-seed mean at step 400k: **0.4613**
- **Caveat**: this run's value network diverged during training. `v_std_across_heads ≈ 2,900`, `grad/norm ≈ 4.7M`, `v_min ≈ −14,700` (physical bound is ≈ −500). At inference, `β·σ ≫ |μ|` so the pessimism scoring is effectively random noise. The eval success rate is driven by head 3 (τ=0.7) whose expectile fixed point happened to stay usable despite the divergence elsewhere in the ensemble. See [PHASE3_DESIGN_A_REPORT.md §4](PHASE3_DESIGN_A_REPORT.md) for the full mechanism.

### 3.2 Reliable best: Phase-3b tight-τ

- Path: [results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/seed{0,1,2}/params_1000000.pkl](results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/) (82 MiB each)
- Step-1M 3-seed mean: **0.4173**
- Peak 3-seed mean at step 400k: **0.4387**
- Training was clean: `grad/norm` stayed flat 280–650, `v_std_across_heads` drifted 4 → 17 over training but never exploded, V stayed within physical bounds.
- β-sweep on step-1M checkpoints (partial): β=0 → 0.409, β=0.25 → 0.413, β=0.5 → 0.423, close to the eval-CSV numbers within noise.

**Recommended for any downstream use**: Phase-3b, because it's the only +Δ-over-HIQL run where the mechanism the method is designed around (σ-based pessimism) is actually engaging rather than being swamped by numerical chaos.

### 3.3 β configuration

- Every training run used `pessimism_beta=0.5` (visible in each `flags.json`).
- β is **inference-only** — it appears solely in `EXCHIQLAgent.sample_actions` / `CHIQLAgent.sample_actions` at the subgoal-scoring step, never in training gradients. Actors use single-head indexed advantage, not μ−β·σ.
- So any saved checkpoint can be re-scored at any β without retraining. See [scripts/launch_beta_sweep_phase3b.sh](scripts/launch_beta_sweep_phase3b.sh).

---

## 4. Mechanism findings, one cell at a time

### 4.1 Architecture note (important — corrected 2026-04-23)

`GCValue(ensemble=True, num_ensemble=5)` uses `nn.vmap` with `variable_axes={'params': 0}` and `split_rngs={'params': True}` (see [utils/networks.py:15-25](external/ogbench/impls/utils/networks.py#L15-L25)). Every weight has leading axis N=5 and each replica is independently initialized. **This is 5 fully independent MLPs, not "shared trunk + 5 independent heads"** as earlier drafts of C-HIQL.md, EXPECTILE_HIQL.md and Phase-3 reports incorrectly claimed. All docs have been rewritten to match the actual architecture. The "shared trunk" variant (`SharedTrunkGCValue` — 1 trunk + 5 `Dense(1)` heads) is a separate module now used on the two `shared-trunk-*` branches.

### 4.2 Phase-2 CHIQL (indep trunks, same τ)

5 identically-configured MLPs, same τ=0.7 loss. All 5 converge to approximately the same value function; residual disagreement propagates init noise through deep nets. The σ diagnostic (`per_state_mu_sig_corr_mean = −0.44`) showed the residual σ anti-correlates with μ — probably because deep-net init-noise magnifies along directions of largest feature activation, which correlate with |μ|. Pessimism scoring `μ − β·σ` therefore *penalizes candidates that are already bad*, giving β=0.5 a −2 pt penalty vs β=0. Mechanism fails; this is what motivated EX-HIQL.

### 4.3 Phase-3a (indep trunks, wide per-head τ)

Each MLP trained on a different τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}. At extreme τ (0.1 and 0.9), the asymmetric expectile weights drive the individual MLP to a far-tail expectile fixed point. Combined with each MLP's own target-network EMA (a positive feedback loop: as V_9 grows, V_9_target catches up, which makes the Bellman target r+γV_9_target grow, which the τ=0.9 loss chases even higher), heads 1 and 5 run away in opposite directions. Middle heads (τ=0.3, 0.5, 0.7) stay stable. Adam's per-parameter normalization keeps step size bounded (no NaN), but parameters drift unboundedly.

The σ diagnostic confirmed the failure mode: `per_state_mu_sig_corr = +0.99`, because at every candidate state μ and σ are both dominated by the two runaway heads. The scoring `μ − β·σ` becomes noise at β=0.5.

Policy survives via head 3 (τ=0.7), unaffected by the runaway in heads 1/5 because the 5 MLPs share no parameters. Final eval ≈ HIQL. Not a demonstration of pessimism.

### 4.4 Phase-3b (indep trunks, tight per-head τ)

Same architecture as 3a, but τ spread narrowed to {0.6, 0.65, 0.7, 0.75, 0.8}. No head is close enough to the extreme asymmetry region to enter the feedback loop. Training stays numerically clean for the full 1M steps. σ exists (grows 4 → 17 over training — σ-creep, caused by the 5 independent expectile fixed points slowly drifting apart as V learns a wider dynamic range), but never blows up.

Under the σ<10 early-stopping rule (derived from keeping `β·σ/|μ| ≤ 15%`), the method's peak is at step 400k (mean 0.439, +3.5 pt over HIQL). At step 1M, σ has crossed the threshold and scoring quality degrades modestly, leaving the eval mean at 0.417 (+1.3 pt). See [PHASE3B_REPORT.md](PHASE3B_REPORT.md).

### 4.5 Shared-trunk CHIQL (1 trunk + 5 Dense(1) heads, same τ)

`SharedTrunkGCValue`: 1 shared 3×512 trunk feeding 5 independent `Dense(1)` output heads. All 5 heads trained on the same τ=0.7 loss. Training stays stable throughout, but `v_std_across_heads` at step 1M is **0.002** — three orders of magnitude smaller than Phase-2's ~1.2. The 5 heads read identical features and learn effectively identical projections; init-noise disagreement washes out entirely. `β·σ / |μ| ≈ 0.005%` at β=0.5, so the pessimism term is numerically negligible. Eval mean 0.392 is essentially "HIQL with 5 redundant output heads," confirming the shared-trunk + same-supervision case degenerates to the single-head baseline.

### 4.6 Shared-trunk EX-HIQL (training)

Same shared-trunk architecture with per-head τ ∈ {0.6..0.8}. The question this tests: per-head τ provides a *structural* lower bound on σ (different τ must produce different expectile fixed points even on shared features) — does that bound stay meaningful, or does the shared trunk still compress it?

Training launched 2026-04-23 ~13:01 local; expected complete by ~16:00-16:30 (3-seed parallel on full GPU after other concurrent jobs were killed).

---

## 5. Open / not-yet-run experiments

The following exist as code branches but don't have completed results:

- **`phase3c-l2reg`** — adds λ·Σ_k ‖w_k − w̄‖² penalty pulling the 5 head parameter tensors toward their ensemble mean. Code is in place (see commit `69d448e`), but the 3-seed experiment has not been launched. Motivation: if σ-creep is the only late-training degradation mechanism, L2 regularization should bound the drift and let Phase-3b's step-400k peak survive to step 1M.

- **`wide-tau-clip`** — EX-HIQL + `optax.clip_by_global_norm(10.0)` on two τ spreads:
  - wide (0.1–0.9): tests whether Phase-3a is rescuable with clipping.
  - mid (0.4–0.8): intermediate point between Phase-3b's tight (0.6–0.8) and Phase-3a's wide.
  Both were launched 2026-04-23 ~13:10 but killed manually within an hour; no 1M checkpoints. Code exists; re-launching is a matter of invoking the same script.

- **β schedule at inference** — since β is inference-only, a schedule `β(σ) = 0.5·min(1, 8/σ)` can be applied to existing checkpoints without retraining, keeping `β·σ/|μ|` near the 15% target even as σ creeps. Proposed in [PHASE3B_IMPROVEMENT_PLAN.md §3 Option 1](PHASE3B_IMPROVEMENT_PLAN.md); not yet implemented.

- **State-dependent σ verification** — the σ diagnostics so far give aggregate statistics (`sigma_mean`, `sigma_std`, `per_state_mu_sig_corr`). They do *not* separate teleport-adjacent states from ordinary deterministic ones. The theory of EX-HIQL predicts σ should be specifically larger at stochastic transitions; this has not been directly measured.

---

## 6. Git branch layout (as of 2026-04-23 mid-afternoon)

All branches pushed to `origin`.

| Branch | Contents | Remote |
|---|---|---|
| `main` | **Consolidated.** HIQL/CRL/QRL/GCIQL/GCIVL baselines + Phase-2 CHIQL + Phase-3a + Phase-3b + shared-trunk CHIQL (results only) + all reports. Indep-trunks `chiql.py` + indep-trunks `ex_chiql.py`. | `origin/main` |
| `indep-trunk-5head` | Phase-3 EX-HIQL line (indep trunks, per-head τ). Reports, Phase-3a+3b results and 82 MiB × 6 checkpoints, `ex_chiql.py`. | `origin/indep-trunk-5head` |
| `shared-trunk-chiql` | `SharedTrunkGCValue` added to `chiql.py`; shared-trunk same-τ results + 32 MiB × 3 checkpoints. | `origin/shared-trunk-chiql` |
| `shared-trunk-5head` | `SharedTrunkGCValue` added to `ex_chiql.py`; per-head-τ variant — **currently training**, no results committed yet. | `origin/shared-trunk-5head` |
| `phase3c-l2reg` | L2 head-divergence regularizer patch to `ex_chiql.py` (indep trunks). Experiment not run. | `origin/phase3c-l2reg` |
| `wide-tau-clip` | New agent file `ex_chiql_clip.py` (indep trunks + grad clip). Wide (0.1–0.9) and mid (0.4–0.8) τ configs. Experiments killed mid-run; no step-1M checkpoints. | `origin/wide-tau-clip` |

Main is the authoritative read-out. The feature branches are preserved for re-running specific variants or rebuilding the shared-trunk agents without collision with main's code tree.

---

## 7. Pointers to detailed reports

| Doc | What it covers |
|---|---|
| [C-HIQL.md](C-HIQL.md) | Phase-2 idea spec: 5-head ensemble + μ − β·σ scoring |
| [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md) | Original one-page proposal |
| [EXPECTILE_HIQL.md](EXPECTILE_HIQL.md) | Phase-3 idea spec: per-head τ as structural diversity |
| [PHASE2_TRAINING_PLAN.md](PHASE2_TRAINING_PLAN.md), [PHASE2_VERIFICATION_PLAN.md](PHASE2_VERIFICATION_PLAN.md) | Phase-2 execution plan |
| [PHASE3_TRAINING_PLAN.md](PHASE3_TRAINING_PLAN.md) | Phase-3 execution plan (Design A / Design B) |
| [PHASE3_DESIGN_A_REPORT.md](PHASE3_DESIGN_A_REPORT.md) | Phase-3a (wide-τ) run report + mechanism analysis of the explosion |
| [PHASE3B_REPORT.md](PHASE3B_REPORT.md) | Phase-3b (tight-τ) run report + σ<10 early-stopping analysis |
| [PHASE3B_INTERIM_NOTES.md](PHASE3B_INTERIM_NOTES.md) | Live-run notes during Phase-3b training |
| [PHASE3B_IMPROVEMENT_PLAN.md](PHASE3B_IMPROVEMENT_PLAN.md) | Next-experiment options: β-schedule, L2 reg, shared-trunk ablation, save_interval |
| [HIQL_EXPLAINER.md](HIQL_EXPLAINER.md) | Tutorial-style explainer of HIQL and the extensions |

---

## 8. What this summary is NOT

- It is not a claim of publishable improvement. The +1.3 pt over HIQL from Phase-3b at step 1M (or +3.5 pt at peak with early stopping) is within 1σ of the 50-episode eval noise floor (~3 pts). For a paper, we would need more seeds, more benchmark environments, and likely better tuning before making a confident claim.
- It is not the end state. The shared-trunk EX-HIQL cell is still training; the `phase3c-l2reg`, `wide-tau-clip`, β-schedule, and state-dependent-σ diagnostic are open directions. This summary will be stale once any of those completes.
- It is not a mechanism proof. We have reasonable narratives for why each cell behaved the way it did, but most have not been directly measured — in particular, whether σ actually concentrates at teleport-adjacent states (EX-HIQL's core theoretical claim) remains unverified. Running the state-dependent-σ diagnostic on Phase-3b's step-400k checkpoint is the cheapest next experiment that would test the entire method's premise.
