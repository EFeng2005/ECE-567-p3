# Paper Outline — ECE 567 Phase 2 Report

**Title (working)**: *EX-HIQL: Structural Ensemble Diversity for Stochastic Offline Hierarchical Goal-Conditioned RL*
**Target length**: 10 pages, 11 pt, single-column, single-spaced (excluding references)
**Env evaluated**: `antmaze-teleport-navigate-v0` (OGBench)

---

## 1. Central Thesis

HIQL collapses on stochastic goal-conditioned RL environments because
expectile regression inflates the learned value function at
stochastic-transition-adjacent states, and the resulting miscalibration
propagates through both levels of the hierarchy. The obvious remedy —
C-HIQL-style ensemble pessimism — fails because random-initialisation
disagreement is anti-correlated with the value itself, inverting the
method's intent. We replace random-init diversity with **per-head expectile
diversity** (EX-HIQL), giving the ensemble disagreement signal σ a
provable structural link to the spread of the per-state Bellman-target
distribution. We empirically verify, using the environment's teleporter
geometry, that σ does concentrate at stochastic-transition-adjacent
states (1.66× near/far ratio); and we propose a **residual-σ scoring
rule** that extracts the teleporter signal from σ's μ-correlated bulk
without retraining, yielding +2.5 pt over the HIQL baseline with 3/3
seeds individually beating baseline.

---

## 2. Three Novelty Hooks

Use these explicitly in the Introduction as the contribution list.

| # | Contribution | Status |
|---|---|---|
| 1 | **Mechanism-level diagnosis** of C-HIQL's failure via per-state σ-μ anti-correlation (−0.44) — the first quantitative explanation for why vanilla ensemble pessimism breaks | Data in hand |
| 2 | **EX-HIQL**: per-head expectile diversity as a structural replacement for random-init diversity. Provable fixed-point argument: σ → 0 at deterministic transitions, σ > 0 at stochastic ones | Method complete; 3-seed result |
| 3 | **Geometric validation + inference-time improvement**: direct empirical confirmation that EX-HIQL's σ concentrates at known teleporter zones (first such measurement in the ensemble-pessimism family); residual-σ scoring rule as cheap retraining-free improvement | Data in hand; +2.5 pt over HIQL |

---

## 3. Narrative Spine: Symmetric Diagnostic Framework

The paper's structural hook is a **three-act symmetric diagnosis**:

| | §3 HIQL — Diagnosing the Disease | §4 C-HIQL — Diagnosing the First Attempt | §6 EX-HIQL — Validating the Cure |
|---|---|---|---|
| Purpose | Why HIQL collapses on stochastic envs | Why C-HIQL's remedy also fails | EX-HIQL's σ is state-dependent and exploitable |
| Diagnostic A | **Diag 1**: V-G calibration, bimodal error, $r: 0.84 → 0.36$ | **Diag 4**: σ-μ anti-correlation, $r_s = -0.44$ | **Diag 5**: σ spatial concentration, near/far ratio 1.66× |
| Diagnostic B | **Diag 2**: Oracle subgoal ablation recovers only +0.07 | | **Diag 6**: σ⊥ (residual) amplifies signal → ratio ↑ [NEW] |
| Diagnostic C | **Diag 3**: Subgoal density collapses onto teleport-out tile | | **Diag 7**: Scoring-filtered density no longer collapses [OPTIONAL] |

Each HIQL failure diagnostic has a matching EX-HIQL validation diagnostic.
This lets the paper present methodology as: **"we broke down the disease;
now we break down the cure in the same language"**.

---

## 4. Section-by-Section Plan

### §1 Introduction — 0.5 pg
- Open with HIQL's 40-point collapse from `antmaze-large` (0.83--0.91) to `antmaze-teleport` (0.42--0.52)
- 2-sentence synopsis of the three-act structure
- Contribution bullet list (the three novelty hooks)
- Headline result preview: HIQL 0.404 → EX-HIQL Phase-3b 0.417 → + residual-σ 0.429 (+2.5 pt, 3/3 seeds)

### §2 Related Works — 0.5 pg
Five compressed paragraphs (2–3 sentences each):
- **HIQL** (Park et al., NeurIPS 2023) — the base method we extend
- **IQL / expectile regression** (Kostrikov et al., ICLR 2022) — the V-learning primitive per-head τ generalises
- **Ensemble pessimism in offline RL** (EDAC, SAC-N) — same pessimism-via-disagreement motif, applied to flat online/offline RL rather than hierarchical GCRL
- **Distributional RL** (QR-DQN, Rowland et al.) — per-quantile-head diversity, the intellectual predecessor of per-expectile diversity
- **CQL / conservative Q-learning** — alternative pessimism family via loss-level regularisation rather than scoring-level aggregation

Positioning sentence: *"To our knowledge, ensemble-based pessimism for offline hierarchical GCRL, and direct geometric validation of the stochasticity-σ correspondence, are unexplored."*

### §3 Phase 1: Diagnosing HIQL's Failure — 3.0 pg [already written]
Keep as-is. Already contains:
- §3.1 Motivation and Diagnostic Plan
- §3.2 Diag 1: V-Function Calibration (bimodal error, ranking collapse $r = 0.84 → 0.36$)
- §3.3 Diag 2: Oracle Subgoal Ablation (+0.07 recovery only)
- §3.4 Diag 3: Subgoal Distribution Analysis (collapse onto teleport-out)

### §4 C-HIQL: Design & Failure Diagnosis — 1.25 pg [already written]
- **§4.1** Design: $N = 5$ independent MLPs, shared $\tau = 0.7$, scoring rule $\bar V - \beta_{\text{pes}} \sigma_V$
- **§4.2 Diag 4: σ-μ Anti-Correlation** — per-state Pearson $r_s = -0.44$ across 500 states × 16 candidates. Mechanism: identically-trained deep nets converge to approximately the same function; residual variance amplifies along largest-activation directions, which correlate with $|\bar V|$. Consequence: C-HIQL at $\beta = 0.5$ scores 0.384, *below* HIQL baseline (0.404).

**Editing note**: explicitly label the σ-μ analysis as "Diag 4" to match the symmetric framework.

### §5 EX-HIQL — 2.5 pg [already written, needs compression]
- **§5.1 Per-Head Expectile Diversity**. Expectile fixed-point argument: different $\tau_i$ coincide iff $Y$ is a point mass, so σ → 0 at deterministic transitions and σ > 0 at stochastic ones. Loss is one-tensor change. Actor indexing: pin to τ=0.7 head so training is bit-identical to HIQL (Design A).
- **§5.2 Training Stability + 2×2 Ablation**. Wide τ (0.1–0.9) diverges via per-head target-network EMA feedback loop (grad_norm → 4.7M, V_min → −14,700, corr +0.99). Tight τ (0.6–0.8) stays clean through 1M steps. 2×2 matrix over architecture {indep trunks, shared trunk} × supervision {same τ, per-head τ}: only indep-trunks × tight per-head τ (Phase-3b) is simultaneously stable, has meaningful σ, and beats HIQL.
- **§5.3 Residual-σ Scoring Rule**. Decomposition hypothesis: σ = (α + γ·V̄) + σ⊥. Fit $(\hat\alpha, \hat\gamma)$ on 500 dataset states per checkpoint via linear regression (takes <1 second). Modified scoring: $\arg\max_z\, \bar V - \beta_{\text{pes}} \cdot [\sigma - (\hat\alpha + \hat\gamma \bar V)]$. No retraining required.

**Editing note**: compress actor-indexing paragraph (currently 0.3 pg, target 0.1 pg). Condense "Where the gains come from" algebra in §5.3 to a single paragraph instead of a multi-step derivation.

### §6 Validating the Cure — 1.5 pg [partially written, needs Diag 5/6/7]

- **§6.1 Diag 5: σ Concentrates at Teleporter Zones** [data in hand, needs re-framing from current §R2]
  Hardcode teleporter geometry: teleport-in at world (20, 12), (0, 16); trigger radius 1.5. Sample $N = 2000$ dataset states per seed; classify by distance to nearest teleporter.
  **Result**: Phase-3b median near/far σ ratio = 1.66×; dist-σ correlation = −0.36.
  This is, to our knowledge, the first direct geometric confirmation that ensemble disagreement concentrates at a stochastic environment's known-stochastic transitions.

- **§6.2 Diag 6: The Residual σ⊥ Amplifies the Signal** [NEW — data to be generated]
  Compute σ⊥ = σ − (α̂ + γ̂·μ) from the same 2000 states; re-run the same spatial analysis.
  **Predicted result**: σ⊥'s near/far ratio exceeds σ's (>1.66×); dist-σ⊥ correlation is more negative than −0.36.
  This diagnostic directly validates the hypothesis underlying the residual-σ scoring rule of §5.3: σ's useful stochasticity content is its μ-orthogonal component, not σ itself.

- **§6.3 Diag 7: Scoring-Filtered Subgoal Density** [NEW, optional — data to be generated]
  Mirror §3 Diag 3 on the Phase-3b checkpoint. For the same $(s, g)$ used in Fig 5:
  Panel A: sample $N = 1000$ subgoals from $\pi_{\text{high}}$ and visualise density (expected: still collapses onto teleport-out, because $\pi_{\text{high}}$ is trained bit-identically to HIQL).
  Panel B: score each candidate by $\bar V - \beta \sigma^\perp$ and take top-K argmax; visualise their density (expected: subgoals redirect away from teleport-out toward goal).
  This produces a direct visual counterpart to §3 Fig 5 — the disease figure and the cure figure at identical $(s, g)$.

### §7 Benchmark Results — 0.5 pg [already written, needs compression]
Single headline table:

| Method | Architecture | Supervision | step-1M 3-seed mean | Δ vs HIQL |
|---|---|---|---|---|
| HIQL | 1 MLP | τ=0.7 | 0.404 | — |
| C-HIQL | 5 indep MLPs | τ=0.7 | 0.384 | −2.0 |
| EX-HIQL Phase-3b | 5 indep MLPs | τ=(0.6..0.8) | 0.417 | +1.3 |
| + residual-σ scoring | same checkpoint | — | 0.429 | +2.5 |

Plus a compact scoring-rule ablation table showing plain / rank / residual at β=0/0.5/2.0. Plus 1–2 sentences noting three convergent decoupling strategies collapse to ~0.43 ceiling.

**Editing note**: cut the current detailed R1/R2/R3 narrative — the diagnostics live in §6 now; §7 is just the numerical summary.

### §8 Discussion & Limitations — 0.25 pg
- **Single-env evaluation framed as depth-over-breadth**: "We deliberately trade breadth for depth: a single-environment evaluation with systematic geometric diagnostics is more informative about the mechanism than a broader shallow evaluation would be."
- **Effect size within 50-episode noise floor** (~3 pt), but **3/3 seeds individually exceed baseline** across three convergent decoupling strategies — directional evidence, not a single-draw coincidence.
- **Future work**: (a) L2 head regularisation to bound σ-creep (`phase3c-l2reg` branch, code in repo, experiment not run); (b) σ-aware β schedule at inference; (c) multi-env evaluation on `antmaze-large`, `pointmaze-teleport`, and `humanoidmaze-teleport`.

---

## 5. Status Tracker

### Already written (as of commit on `main`)
- §3 Phase 1 Diagnosis (complete, ~3 pg)
- §4 C-HIQL Design + Diag 4 (complete, ~1.25 pg)
- §5 EX-HIQL (complete skeleton, ~2.5 pg; needs light compression)
- §6 R1/R2/R3 (partially; R2 needs to be reframed as Diag 5, R3 mostly moves to §7)
- §7 Benchmark Results (in draft form inside §6, needs extraction)

### To write (pure writing, no experiments)
- §1 Introduction (0.5 pg)
- §2 Related Works (0.5 pg)
- §8 Discussion & Limitations (0.25 pg)
- Reframe of §6 around Diag 5/6/7 labels
- Extract §7 from current §6 body

### To run (experiments, then integrate)
- **Diag 6: σ⊥ spatial analysis** — 20 min python script, P0, strongly recommended
- **Diag 7: Scoring-filtered subgoal density** — 2–3 hour scripting, P1, optional
- **Multi-env generalisation** (post-paper or supplementary): `antmaze-large-navigate-v0` 3 seeds (~5 h) and/or `pointmaze-teleport-navigate-v0` 3 seeds (~2 h)

---

## 6. Page Budget Summary

| Section | Page budget | Current | Notes |
|---|---|---|---|
| §1 Intro | 0.50 | 0 | write |
| §2 Related Works | 0.50 | 0 | write |
| §3 HIQL Diagnosis | 3.00 | ~3.0 | done |
| §4 C-HIQL | 1.25 | ~1.25 | done, minor relabel |
| §5 EX-HIQL | 2.50 | ~2.7 | compress by 0.2 |
| §6 Validating the Cure | 1.50 | ~1.0 | reframe + add Diag 6 (and optionally Diag 7) |
| §7 Benchmark Results | 0.50 | embedded | extract + compress |
| §8 Discussion | 0.25 | 0 | write |
| **Total** | **10.00** | **8.0** | **~2 pg to write + Diag 6 figure** |

---

## 7. Execution Order

Recommended sequence, with no experiments until the writing is firm:

1. **(writing, ~1 h)** §4.2 relabel → "Diag 4"; §6 reframe → §6.1/§6.2/§6.3 Diag 5/6/7 structure; §7 extraction from §6; §5 compression.
2. **(writing, ~1 h)** §1 Introduction + §2 Related Works + §8 Discussion.
3. **(experiment, ~20 min)** Diag 6 σ⊥ spatial script; inject result numbers and figure into §6.2.
4. **(decision point)** If time permits: Diag 7 subgoal density (~2–3 h); multi-env run (~5–12 h).

Goal: a near-final draft after steps 1–3 (within 2 h of focused work), and optional experiments to strengthen.

---

## 8. Framing Do's and Don'ts

### Safe to claim strongly
- "First mechanism-level diagnosis of C-HIQL's σ-μ anti-correlation failure"
- "First geometric-level validation of σ-stochasticity correspondence in the ensemble-pessimism family"
- "Three mechanistically distinct scoring rules converge on the same performance ceiling"
- "3/3 seeds individually exceed the plain-β=0.5 baseline"
- "Stable training on 1M gradient steps across 3 seeds"

### Safe to claim with qualification
- "+2.5 pt over HIQL" → pair with: "within the 50-episode eval noise floor (~3 pt), but directionally consistent across 3 seeds and 3 scoring rules"
- "σ captures stochasticity" → say instead: "σ has a stochasticity-correlated component, empirically confirmed as a 1.66× near/far ratio and −0.36 dist-σ correlation"

### Avoid
- "Significantly outperforms" / "substantial improvement" (effect size at noise edge)
- "Generalises across stochastic RL benchmarks" (single env)
- "SOTA" (no comparison to current SOTA methods beyond HIQL/C-HIQL)
- "Calibrated uncertainty estimates" (no calibration measurement)
- "σ equals dynamics stochasticity" (at most partial correlation)
