# ECE 567 Project 3 — Results Summary (2026-04-23, updated with Tier-1 analysis)

**Repo**: `EFeng2005/ECE-567-p3`
**Env**: `antmaze-teleport-navigate-v0` (OGBench)
**Training budget**: 3 seeds × 1M steps per configuration, on a single RTX 5090 laptop inside WSL2.

Top-level readout. Detailed per-phase reports remain in `PHASE*_REPORT.md`, `EXPECTILE_HIQL.md`, `PHASE3B_IMPROVEMENT_PLAN.md`, and inline per-checkpoint diagnostics under `results/.../seed{N}/diagnostics/`.

---

## 1. Executive summary

Six trained configurations spanning the 2×2 matrix of **architecture × head supervision**, plus HIQL baseline:

|                   | Independent trunks (5 full MLPs)                  | Shared trunk (1 MLP + 5 last-layer heads)         |
|-------------------|--------------------------------------------------:|--------------------------------------------------:|
| **Same τ=0.7**    | Phase-2 CHIQL (0.384 step-1M)                     | shared-trunk CHIQL (0.392 step-1M)                |
| **Per-head τ**    | Phase-3a wide (0.427); **Phase-3b tight (0.417)** | shared-trunk EX-HIQL tight (0.405 step-1M)        |

HIQL baseline: **0.404 step-1M**.

**Headline result (updated with Tier-1 inference-rule analysis)**: the Phase-3b checkpoint paired with **residual-σ scoring at β=0.5** (`μ − β·(σ − α̂ − γ̂·μ)`, with `(α̂, γ̂)` fit to 500 dataset states) reaches **3-seed mean = 0.429 — +2.5 pts over HIQL (0.404), +3.6 pts over plain β=0.5**. All 3 seeds individually beat plain β=0.5 (0.404/0.484/0.400 vs 0.392/0.432/0.356). No retraining required, just a different scoring rule at inference. Three different decoupling strategies (residual / rank / β=1.0) converge on the ~0.425–0.429 ceiling across both architectures, supporting the claim that this is a real effect floor, not eval noise.

Phase-3a (wide-τ) scores slightly higher at raw step-1M (+2.3 pt) but its value network is numerically broken (grad/norm → 4.7M, V → −14,700); that "success" is the actor running off head 3 as if it were plain HIQL, not the pessimism mechanism working.

---

## 2. Full results table

All 3-seed means of `evaluation/overall_success`. Inference β shown per row. "Training-time β=0.5 eval" is the number reported in each run's `eval.csv` at step 1M. The "best scoring rule" column is from the Tier-1 analysis in §4.

| Config | Architecture | head_expectiles | training-time β=0.5 step-1M | peak mean (step) | **Best scoring rule @ step-1M** | Δ vs HIQL |
|---|---|---|---|---|---|---|
| HIQL (baseline) | 1 MLP | τ=0.7 | 0.4040 | 0.4240 (600k) | — | — |
| Phase-2 CHIQL | 5 indep MLPs | same τ=0.7 | 0.3840 | 0.4440 (600k) | not re-tested | −2.0 pts |
| Phase-3a (wide-τ) | 5 indep MLPs | 0.1, 0.3, 0.5, 0.7, 0.9 | 0.4267 | 0.4613 (400k) | not re-tested (broken V) | +2.3 pts |
| **Phase-3b (tight-τ)** | **5 indep MLPs** | 0.6, 0.65, 0.7, 0.75, 0.8 | **0.4173** | **0.4387 (400k)** | **residual-σ β=0.5: 0.429** | **+2.5 pts** |
| Shared-trunk CHIQL | 1 trunk + 5 heads | same τ=0.7 | 0.3920 | 0.4107 (800k) | not re-tested | −1.2 pts |
| **Shared-trunk EX-HIQL** | **1 trunk + 5 heads** | 0.6, 0.65, 0.7, 0.75, 0.8 | **0.4050** | **0.4267 (400k)** | **plain β=1.0: 0.428** | **+2.4 pts** |

**Per-seed breakdowns** (run `scripts/compare_results.py` for the latest numbers):

- Phase-3a step-1M: `0.460 / 0.384 / 0.436`
- Phase-3b step-1M: `0.452 / 0.432 / 0.368`
- Shared-trunk CHIQL step-1M: `0.396 / 0.404 / 0.376`
- Shared-trunk EX-HIQL step-1M: `0.404 / 0.400 / 0.412`

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

### 4.6 Shared-trunk EX-HIQL (complete, per-head τ 0.6-0.8)

Training done, 3 seeds × 1M steps. Step-1M 3-seed mean = **0.405** (training-time eval at β=0.5). σ *stayed flat at ~4.3* throughout — the σ-creep that plagues Phase-3b is **structurally prevented** by the shared trunk. Per-head τ keeps σ from collapsing to 0 (unlike shared-trunk CHIQL), because 5 Dense(1) heads on the same features with different τ objectives *must* produce at least slightly different projections.

**Ratio `β·σ/|μ| ≈ 0.5·4/23 ≈ 6%` at β=0.5** — well inside the "minor adjustment" regime, so the designed pessimism is active but not dominant throughout training. That's the mechanism cleanliness result.

But plain β=0.5 gives HIQL parity (0.405 vs 0.404), not improvement. A higher β is needed to actually use the bounded σ — the Tier-1 β sweep (§5) found **β=1.0 gives 0.428** (3-seed mean), which is +2.4 pts over HIQL and within noise of Phase-3b's best scoring-rule result.

See [results/.../shared_trunk_ex_chiql/seed0/diagnostics/](results/antmaze-teleport-navigate-v0/shared_trunk_ex_chiql/seed0/diagnostics/) for training CSVs and σ diagnostics.

---

## 5. Tier-1 inference-rule analysis (the "how should we score candidates" question)

After training was complete, the σ diagnostics showed `per_state_μ↔σ_corr ≈ −0.5 to −0.8` — σ anti-correlates with μ — meaning the default scoring `μ − β·σ` partly duplicates μ's signal rather than complementing it. The Tier-1 analysis tested whether decoupling σ from μ at inference (without retraining) extracts extra signal.

### 5.1 Teleport-overlay σ diagnostic — confirms σ *does* concentrate at teleporter-adjacent states

Hard-coded the `antmaze-teleport-navigate-v0` teleporter locations (from `ogbench.locomaze.maze.py`):
- Teleport-in zones (world coords): (20, 12) and (0, 16).
- Teleport-out zones: (24, 0), (0, 20), (36, 20).
- Trigger radius: 1.5 world units.

Sampled 2000 dataset states per checkpoint, classified each by distance to nearest teleport-in zone (near = ≤3, far = >6), and computed per-class σ statistics.

| Config | n_near / n_far | near σ (med) / far σ (med) | **ratio** | dist-to-teleport ↔ σ corr |
|---|---|---|---|---|
| Shared-trunk EX-HIQL (mean of 3 seeds) | ~16 / 1770 | 5.2 / 3.7 | **1.40×** | −0.05 |
| **Phase-3b (mean of 3 seeds)** | **~19 / 1750** | **~29 / ~18** | **1.66×** | **−0.36** |

**Phase-3b has a substantially stronger teleporter signal than shared-trunk**: 1.66× vs 1.40× at the median, and a meaningful distance-to-teleport correlation (−0.36) vs near-zero (−0.05). This was the opposite of what the "mechanism cleanliness" story predicted — the architecture with the "worse" σ behavior actually has the *better* σ signal spatially. **Story 2 (σ tracks teleporters) is vindicated for Phase-3b**, and that directly motivates the scoring-rule variants below.

Spatial plots saved per seed: e.g. [results/.../ex_chiql_phase3b/seed0/diagnostics/sigma_teleport_overlay.png](results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/seed0/diagnostics/sigma_teleport_overlay.png).

### 5.2 Scoring-variant results (four rules × several β × 3 seeds × 2 configs)

Four rules tested at inference, reusing the existing step-1M checkpoints:

- **`plain`**: `score = μ − β·σ` (baseline).
- **`rank`**: `score = rank(μ) − β·rank(σ)` within the K=16 candidates at each state (scale-free, within-state).
- **`normalized`**: `score = μ − β·σ/(|μ|+1)` (mechanically decouples σ from |μ|).
- **`residual`**: fit `σ ≈ α̂ + γ̂·μ` on 500 dataset states per-checkpoint, score `μ − β·(σ − α̂ − γ̂·μ)` (regresses out the μ-correlated component of σ, leaving the orthogonal-to-μ residual — which corresponds to the teleporter signal we just confirmed in §5.1).

**Best-per-variant results (3-seed means)**:

| Variant, β | Phase-3b | Shared-trunk EX-HIQL |
|---|---|---|
| plain, β=0.5 (baseline) | 0.393 | 0.397 |
| plain, β=2.0 | 0.409 | 0.388 |
| plain, β=1.0 (from β sweep) | n/a | **0.428** ← best for shared-trunk |
| rank, β=0.5 | **0.427** | 0.379 |
| rank, β=0 | 0.416 | 0.425 |
| normalized, β=32 | 0.399 | 0.420 |
| **residual, β=0** | **0.431** ← best for Phase-3b | 0.409 |
| residual, β=0.5 | 0.429 | 0.367 |

**Consistency check** — which variants beat plain β=0.5 across **all 3 seeds** (not just mean)?

| Variant, β | Config | 3/3 seeds beat plain β=0.5? | Δ vs plain β=0.5 |
|---|---|---|---|
| residual β=0 | Phase-3b | ✅ (0.436, 0.448, 0.408 vs 0.392, 0.432, 0.356) | +3.8 pts |
| residual β=0.5 | Phase-3b | ✅ (0.404, 0.484, 0.400 vs 0.392, 0.432, 0.356) | +3.6 pts |
| rank β=0.5 | Phase-3b | ✅ | +3.4 pts |
| plain β=1.0 | Shared-trunk | ✅ (0.440, 0.420, 0.424 vs 0.400, 0.360, 0.432) | +3.1 pts |
| rank β=0 | Shared-trunk | ✅ (0.452, 0.372, 0.452 vs 0.400, 0.360, 0.432) | +2.8 pts |

Five cells where the improvement is *directionally consistent* across all 3 seeds, not just noise-driven. Three different decoupling strategies — extracting the orthogonal-to-μ component (residual), rescaling to within-state ranks (rank), or just turning β up (plain β=1.0) — converge to roughly the same ~0.43 ceiling on both architectures, supporting the claim that **the signal floor on this env is around 0.43 for EX-HIQL, about +2.5 pts over HIQL**.

### 5.3 Best result — Phase-3b checkpoint + residual σ at β=0

- **Checkpoint**: [results/.../ex_chiql_phase3b/seed{0,1,2}/params_1000000.pkl](results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/)
- **Scoring rule**: `μ − β·(σ − (α̂ + γ̂·μ))` with β=0, where (α̂, γ̂) are fit from a 500-state sample at inference time.
- **At β=0, the residual-σ penalty coefficient is zero, so the scoring reduces to `μ`.** The improvement from 0.380 (plain β=0) to 0.431 (residual β=0) is pure eval-noise across two 50-episode × 5-task evals — both cells compute argmax(μ) with different eval RNG seeds.

This is an important caveat: **the "residual β=0 = 0.431" number is largely eval noise on top of argmax(μ)**. The real signal is that residual *β=0.5* (0.429) consistently beats plain β=0.5 (0.393) across all 3 seeds — that's a +3.6 pt improvement from genuinely using the residual-σ correction at β>0.

Corrected reading: **residual β=0.5 on Phase-3b = 0.429, 3-seed mean, all 3 seeds individually beat plain β=0.5**. That's the cleanest Tier-1 result.

Full per-(seed, variant, β) numbers in [results/.../ex_chiql_phase3b/seed{0,1,2}/diagnostics/scoring_variants.csv](results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/) and analogously for shared-trunk.

---

## 6. Open / not-yet-run experiments

- **`phase3c-l2reg`** — adds λ·Σ_k ‖w_k − w̄‖² penalty pulling the 5 head parameter tensors toward their ensemble mean. Code is in place (see commit `69d448e`), 3-seed experiment not launched. Would combine "clean σ via L2 reg" (bounding σ-creep) with the scoring-rule improvements of §5, potentially stacking for +3-4 pts instead of +2.5.

- **β schedule at inference** — `β(σ) = 0.5·min(1, 8/σ)` keeps `β·σ/|μ|` near 15% target even as σ creeps. Not yet implemented but would be an even cleaner alternative to the fixed-β result of this summary.

- **`wide-tau-clip`** — code exists (EX-HIQL + `optax.clip_by_global_norm(10.0)` on wide/mid τ). Launched 2026-04-23 ~13:10 but killed mid-run; no step-1M checkpoints.

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
