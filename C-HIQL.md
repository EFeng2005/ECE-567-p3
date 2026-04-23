# Conservative HIQL (C-HIQL): Pessimistic Subgoal Selection for Offline Hierarchical Goal-Conditioned RL

> **Elaborated proposal** — the original `FINAL_PROPOSAL.md` was a one-page sketch; this document expands every section so a reader unfamiliar with the project can understand *what we are doing, why it should work, and how we will validate it* without consulting other files.

---

## 1. One-Sentence Summary

> **C-HIQL = HIQL + $N$ value heads on the high level + pessimistic subgoal scoring $\bar V - \beta_{\text{pes}}\,\sigma_V$.**
> Nothing else changes (low-level controller, training losses, representation, hyperparameters all preserved).

This is a **minimal, surgical fix** to a known failure mode of HIQL on stochastic environments — implementable in roughly 30 lines on top of the OGBench HIQL codebase.

---

## 2. Background and Motivation

### 2.1 What HIQL is

HIQL (Park et al., NeurIPS 2023) is the current strongest offline goal-conditioned RL (GCRL) method. It trains:

1. A goal-conditioned **value function** $V(s, g)$ via *action-free* IQL (expectile regression on the residual $r + \gamma\,V(s', g) - V(s, g)$).
2. A **high-level policy** $\pi^h(s_{t+k} \mid s_t, g)$ that selects a $k$-step subgoal by AWR weighting using $V(s_{t+k}, g)$.
3. A **low-level policy** $\pi^l(a \mid s_t, s_{t+k})$ that produces primitive actions toward the chosen subgoal.

All three components share *the same* $V$. The hierarchy mitigates the **signal-to-noise collapse** that plagues flat policies on long-horizon goals (HIQL's Proposition 4.1).

### 2.2 The known failure mode

HIQL's action-free value loss is **unbiased only under deterministic dynamics**. The paper itself states (§3, §7):

> "In stochastic environments, … one cannot tell whether a good outcome is due to a good action or due to noise … applying action-free IQL to stochastic environments often leads to overestimation."

In OGBench's **teleport mazes** (`pointmaze-teleport-medium`, `antmaze-teleport-medium`), an agent stepping onto a teleporter is randomly transported to one of several destinations. The dataset occasionally contains "lucky" teleports that landed close to the goal. Expectile regression (with $\tau \to 1$) treats these lucky outcomes as evidence of *good actions*, inflating $V(s, g)$ for states near teleporters.

### 2.3 How that bias breaks HIQL specifically

The high-level policy is essentially:

$$\pi^h(s_{t+k} \mid s_t, g) \;\propto\; \exp\!\big(\beta \cdot V(s_{t+k}, g)\big).$$

As $\beta$ grows, this tends to $\arg\max_{s_{t+k}} V(s_{t+k}, g)$. **Whichever subgoal is most overestimated gets selected.** The low-level controller then faithfully tries to reach an unreliable waypoint, and the whole hierarchy collapses.

This is the failure mode C-HIQL targets.

---

## 3. The Method

### 3.1 Architectural change (one diagram)

```
HIQL high-level value:          C-HIQL high-level value:
  s ─┐                            s ─┐
     ├─ MLP(s, g) ─→ V(s,g)          ├─► MLP_1(s, g) ─→ V_1(s,g)
  g ─┘                            g ─┤
                                     ├─► MLP_2(s, g) ─→ V_2(s,g)
                                     ├─► MLP_3(s, g) ─→ V_3(s,g)
                                     ├─► MLP_4(s, g) ─→ V_4(s,g)
                                     └─► MLP_5(s, g) ─→ V_5(s,g)

  (single MLP)                    (N=5 completely independent MLPs,
                                   each with its own trunk + output)
```

Two — and only two — code-level changes:

1. Replace the high-level value network with $N$ fully independent MLPs $\{V_i(s, g)\}_{i=1}^N$ (implemented as `GCValue(ensemble=True, num_ensemble=5)`, which uses `nn.vmap` with `split_rngs={'params': True}` — so every weight has leading axis $N$ and each replica is independently initialized; nothing is shared across the 5 networks).
2. Replace $V(s_{t+k}, g)$ in the subgoal scoring step with $V_{\text{pes}}(s_{t+k}, g) = \bar V(s_{t+k}, g) - \beta_{\text{pes}}\,\sigma_V(s_{t+k}, g)$, where

$$\bar V(s, g) = \frac{1}{N}\sum_{i=1}^{N} V_i(s, g),\qquad \sigma_V(s, g) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\big(V_i(s, g) - \bar V(s, g)\big)^2}.$$

### 3.2 Training (almost identical to HIQL)

Each head is trained independently with the **same** action-free expectile loss:

$$\mathcal{L}_V^{(i)}(\theta_i) \;=\; \mathbb{E}_{(s,s',g)\sim\mathcal{D}_S}\!\left[L_2^\tau\!\big(r(s,g) + \gamma\,V_i(s', g) - V_i(s, g)\big)\right],\qquad i = 1,\ldots,N,$$

with the asymmetric squared loss

$$L_2^\tau(u) = |\tau - \mathbf{1}(u<0)|\,u^2.$$

Sources of head diversity:

- **Required:** different random initialization per head.
- **Optional:** independent bootstrap masks (each transition is held out from each head with probability $1-p_{\text{boot}} = 0.2$).

> Important: bootstrap targets $V_i(s', g)$ use the *same* head's prediction (per-head TD target). We do **not** inject pessimism into the training target — only into the inference-time subgoal scoring.

### 3.3 Inference (subgoal selection)

```python
def select_subgoal(s, candidate_subgoals, beta_pes):
    scores = []
    for g_sub in candidate_subgoals:
        z = high_level_trunk(s, g_sub)
        vals = [head_i(z) for head_i in value_heads]   # N scalars
        mu, sigma = mean(vals), std(vals)
        score = mu - beta_pes * sigma                  # pessimistic
        scores.append(score)
    return candidate_subgoals[argmax(scores)]
```

Equivalently, in the AWR formulation: replace the high-level advantage estimator

$$\hat A^h(s_t, s_{t+k}, g) \;=\; V(s_{t+k}, g)\quad\text{(HIQL, Eq. 10)}$$

with

$$\hat A^h(s_t, s_{t+k}, g) \;=\; V_{\text{pes}}(s_{t+k}, g) \;=\; \bar V(s_{t+k}, g) - \beta_{\text{pes}}\,\sigma_V(s_{t+k}, g)\quad\text{(C-HIQL)}.$$

Candidate subgoals come from HIQL's existing sampling distribution — we do not change *what* we score, only *how* we score it.

### 3.4 What does **not** change

| Component                                              | Modified? |
|--------------------------------------------------------|-----------|
| Low-level policy $\pi^l$ training (Eq. 7 of HIQL)      | ❌        |
| Low-level critic / actor network                       | ❌        |
| Value-function training **target** (still optimistic)  | ❌        |
| Representation $\phi(g)$ learning                      | ❌        |
| Candidate subgoal generation                           | ❌        |
| All HIQL hyperparameters ($\gamma, \beta_{\text{AWR}}, \tau, k$) | ❌  |

The "**train optimistically, decide pessimistically**" split is the project's central engineering choice. It preserves HIQL's training dynamics (which are well-tuned and we do not want to disturb) while adding a safety layer at the decision interface.

---

## 4. Why This Should Work — Three Mechanisms

### 4.1 Disagreement is a proxy for overestimation bias

In regions where data is dense and dynamics are deterministic, all heads converge to similar values, so $\sigma_V$ is small. In regions of stochastic transitions or sparse data, different heads latch onto different "lucky" samples, producing divergent estimates, so $\sigma_V$ is large.

> **The locations where overestimation is worst are exactly the locations where $\sigma_V$ is largest.**

### 4.2 $\bar V - \beta_{\text{pes}}\,\sigma_V$ is a soft, calibratable form of pessimism

Lineage of related ideas:

- **CQL** adds a regularizer to push down OOD-action $Q$-values.
- **EDAC / SAC-N** uses $\min_i Q_i$ over an ensemble — a hard form of pessimism.
- **C-HIQL** uses $\bar V - \beta_{\text{pes}}\,\sigma_V$ — a smooth, temperature-controlled relaxation of $\min$.

The **novelty** is *where* the pessimism is applied: not on $Q$-values for OOD actions, but on $V$-values for **subgoal candidates**. This is the natural pessimism interface for hierarchical GCRL, and (to the best of our literature search) unexplored.

### 4.3 In GCRL, ensemble disagreement implicitly encodes reachability uncertainty

Deeper interpretation: when a candidate subgoal sits near a teleporter, the return-to-go from $s_t$ to that subgoal is a high-variance random variable. Different heads, trained with different bootstrap masks/initializations, see different "lucky vs. unlucky" trajectories and disagree.

So $\sigma_V$ doubles as a **reachability uncertainty signal**. Pessimistic scoring effectively tells the high-level planner: *"avoid waypoints that you can only reach by luck."* This matches the safe-planning intuition that has been formalized in robust MDPs but has been missing from offline hierarchical GCRL.

---

## 5. Significance — Three Levels

### 5.1 Engineering: a reliability patch for the strongest offline GCRL baseline

HIQL is the de-facto strongest baseline on OGBench, but its known instability on stochastic environments limits real-world applicability (sensor noise, human disturbance, environment perturbations are ubiquitous). C-HIQL provides a drop-in fix — provided we verify Claim 2 (no regression on deterministic envs), the upgrade has no downside.

### 5.2 Conceptual: bridging two underconnected research traditions

| Tradition 1: Hierarchical GCRL    | Tradition 2: Uncertainty-based offline RL    |
|-----------------------------------|----------------------------------------------|
| HIQL, RIS, RPL, POR, …            | EDAC, SAC-N, PBRL, MOPO, …                   |
| Concerned with long-horizon       | Concerned with OOD pessimism                 |

Almost no prior work explicitly applies ensemble uncertainty *at the subgoal-selection interface* of a hierarchical GCRL method. C-HIQL fills that intersection.

### 5.3 Research positioning: "risk-sensitive subgoal selection," not a grand framework

Deliberately scoped narrow:

> **A minimal, interpretable correction for a single, well-defined failure mode (HIQL on stochastic envs).**

This focused framing avoids the typical "uncertainty framework" critique (overclaim, underdeliver) and gives reviewers a clean story to follow.

---

## 6. Hyperparameters

| Parameter                 | Default | Sweep range                 | Justification                                    |
|---------------------------|---------|-----------------------------|--------------------------------------------------|
| $N$ (ensemble size)       | $5$     | $\{1, 3, 5\}$               | $5$ balances diversity and compute; $1$ = HIQL  |
| $\beta_{\text{pes}}$ (pessimism) | $0.5$   | $\{0,\ 0.25,\ 0.5,\ 1,\ 2\}$ | $0$ = ensemble-mean (control); $1$–$2$ = strong  |
| Bootstrap mask $p_{\text{boot}}$ | None    | $\{\text{None},\ 0.8\}$       | Optional diversity boost                         |

All other HIQL hyperparameters ($\gamma=0.99,\ \beta_{\text{AWR}}=1,\ \tau=0.7,\ k=25$ or $50$) inherited unchanged.

---

## 7. Claims and Evidence Plan

| # | Claim                                                          | Experiment                                                                            | Success Criterion                          |
|---|----------------------------------------------------------------|---------------------------------------------------------------------------------------|--------------------------------------------|
| 1 | C-HIQL improves over HIQL on stochastic environments           | C-HIQL vs HIQL on `pointmaze-teleport-medium`, `antmaze-teleport-medium`, $\geq 3$ seeds | $> 5\%$ absolute success-rate improvement  |
| 2 | C-HIQL does **not** regress on deterministic environments      | C-HIQL vs HIQL on `pointmaze-medium`, `antmaze-large`, `cube-single`, $\geq 3$ seeds   | Drop $< 3\text{–}5\%$                        |
| 3 | The gain comes from *uncertainty*, not from ensembling per se  | C-HIQL($\beta_{\text{pes}}>0$) vs Ensemble-Mean($\beta_{\text{pes}}=0$) vs HIQL on 2 stochastic envs | Only $\beta_{\text{pes}}>0$ improves on stochastic envs |

**Claim 3 is critical** — if Ensemble-Mean ($\beta_{\text{pes}}=0$) already matches C-HIQL, then the contribution reduces to "ensembles average out noise," which is well-known and unpublishable. Confirming $\beta_{\text{pes}}>0$ is specifically necessary is what makes C-HIQL a novel contribution.

---

## 8. Run Matrix and Compute

| Block                | Content                                                                         | Envs | Seeds | Runs |
|----------------------|---------------------------------------------------------------------------------|------|-------|------|
| 1. Main results      | HIQL + C-HIQL(best $\beta_{\text{pes}}$) on 5 environments                      | 5    | 3     | 30   |
| 2. Mechanism ablation| Ensemble-Mean($\beta_{\text{pes}}=0$) + C-HIQL($\beta_{\text{pes}}=0.5$) on 2 stochastic envs | 2    | 3     | 12   |
| 3. $\beta_{\text{pes}}$ sweep | $\beta_{\text{pes}} \in \{0,\ 0.25,\ 0.5,\ 1,\ 2\}$ on 2 stochastic envs | 2    | 3     | 30   |
| 4. $N$ sweep         | $N \in \{1, 3, 5\}$ on 1 stochastic env                                         | 1    | 3     | 9    |
| **Total**            |                                                                                  |      |       | **$\approx 81$** |

- **Per-run cost:** $\sim 6$ GPU-hours on A40 (C-HIQL overhead is $\sim 1.3\text{–}1.5\times$ HIQL since only the high-level head is ensembled).
- **Total budget:** $\sim 486$ GPU-hours.
- **Wall time on 8 GPUs:** $\sim 2.5$ days.
- **Reduced fallback plan** (if compute is tight): 4 envs $\times$ 3 seeds + 2-env ablations $\approx 250\text{–}300$ GPU-hours.

---

## 9. Eight-Day Schedule and Go/No-Go Checkpoints

| Day | Task                                                                | Deliverable                  |
|-----|---------------------------------------------------------------------|------------------------------|
| 1   | Read OGBench JAX codebase, locate HIQL implementation               | Code walkthrough notes       |
| 2   | Implement C-HIQL (ensemble heads + pessimistic scoring)             | Working code, unit tests     |
| 3   | Smoke test: 1 stochastic + 1 deterministic env, 1 seed each         | Learning curves, **go/no-go**|
| 4   | $\beta_{\text{pes}}$ sweep on stochastic env (15 runs)              | Best $\beta_{\text{pes}}$, **go/no-go** |
| 5   | Main stochastic experiments with best $\beta_{\text{pes}}$ (6 runs) | Success-rate tables          |
| 6   | Deterministic + manipulation no-regression (9 runs)                 | Full results table           |
| 7   | Mechanism ablation + ensemble-size ablation + diagnostics (21 runs) | Ablation tables, figures     |
| 8   | Analysis, writing, presentation prep                                | Report draft                 |

**Go/No-Go gates:**

| Day | Checkpoint        | Go criterion                                                 | No-Go action                            |
|-----|-------------------|--------------------------------------------------------------|-----------------------------------------|
| 3   | Smoke test        | Learning curves look sane, no NaN                            | Debug implementation                    |
| 4   | $\beta_{\text{pes}}$ sweep | $\beta_{\text{pes}}>0$ beats $\beta_{\text{pes}}=0$ on the stochastic env | Switch to fallback variant (§11)        |
| 5   | Main stochastic   | $> 5\%$ improvement over HIQL                                | Analyze failure, consider fallback      |

---

## 10. Metrics

- **Primary:** average success rate ($\%$) over 5 goal pairs, $\geq 3$ seeds, with $95\%$ CIs.
- **Secondary:** subgoal-selection stability (variance of chosen subgoals across seeds).
- **Diagnostic:**
  - Ensemble-disagreement histograms (chosen vs. rejected subgoals) — verify that high-$\sigma_V$ subgoals are indeed filtered out.
  - Value-function calibration plot (predicted $V$ vs. realized return, binned) — verify that pessimistic scoring effectively reduces optimistic bias.

---

## 11. Fallback Variants

If by Day 4–5 the ensemble-pessimism gain is weak, try in order:

| Variant | Description                                                                                       | Rationale                                            |
|---------|---------------------------------------------------------------------------------------------------|------------------------------------------------------|
| A       | Replace $\bar V - \beta_{\text{pes}}\,\sigma_V$ with $\min_i V_i$                                | Stronger, harder pessimism (à la EDAC)               |
| B       | Apply pessimism only among top-$k$ mean-valued subgoals                                          | Avoid over-conservatism in low-value regions         |
| C       | Switch to **Hierarchical Contrastive IQL** (CRL score for subgoal ranking)                       | Backup idea from the discovery report (Score 6/10)   |
| D       | Use MC-dropout at inference for uncertainty (single network, dropout sampling)                   | If ensembles are too memory-heavy                    |

---

## 12. Risks and Mitigations

| Risk                                                                 | Likelihood | Mitigation                                                                            |
|----------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------|
| Ensemble disagreement poorly correlated with true overestimation     | Medium     | Variant D (MC-dropout) or Variant C (CRL-based scoring)                                |
| OGBench teleport tasks too benign $\Rightarrow$ improvements not significant | Medium     | Construct custom stochastic perturbations on AntMaze                                  |
| Deterministic-env regression ($\beta_{\text{pes}}>0$ hurts good waypoints in clean $V$) | Low        | Per-environment $\beta_{\text{pes}}$ tuning, or Variant B to localize pessimism  |
| Reviewer: "just EDAC applied to GCRL — too small a contribution"     | Medium     | Stress that the **interface** (subgoal scoring) is novel; Claim 3 shows non-triviality|

---

## 13. Relation to Prior Work

| Method                       | Relationship to C-HIQL                                                              |
|------------------------------|-------------------------------------------------------------------------------------|
| **IQL** (Kostrikov et al. '22) | Provides expectile-regression backbone — C-HIQL's training loss is unchanged       |
| **HIQL** (Park et al. '23)     | Direct base method — C-HIQL adds ensemble heads on top                             |
| **EDAC / SAC-N** (An et al. '21) | Canonical "ensemble + pessimism" for offline RL — C-HIQL ports the idea to subgoals |
| **CQL** (Kumar et al. '20)     | Alternative pessimism via loss regularizer — C-HIQL is lighter and more focused    |
| **POR** (Xu et al. '22)        | One-step subgoal version ($k=1$) of HIQL — C-HIQL transfers to it directly         |
| **TMD** (NeurIPS 2025)         | Quasimetric + contrastive for stochastic envs — orthogonal, different attack angle |
| **NF-HIQL** (2026)             | Closely related; differs in *what* uncertainty is modeled (normalizing flows vs. ensembles) and *where* it is applied |

---

## 14. Positioning Statement

C-HIQL is a **risk-sensitive subgoal-selection method for offline hierarchical GCRL**. It is *not* a grand uncertainty framework. The contribution is showing that a **lightweight pessimistic correction at the planning interface** fixes a known, paper-acknowledged failure mode of HIQL on stochastic environments — without disrupting any of HIQL's strengths on deterministic, long-horizon, or pixel-based tasks.

---

## 15. 60-Second Pitch

> HIQL is the strongest offline goal-conditioned RL baseline, but it overestimates value in stochastic environments — and because the high-level policy is essentially $\arg\max_{s_{t+k}} V(s_{t+k}, g)$, that overestimation directly biases subgoal selection. C-HIQL replaces HIQL's single value head with $5$ ensemble heads and scores subgoals by $\bar V - \beta_{\text{pes}}\,\sigma_V$ instead of $\bar V$. Two lines of inference code change, training is unchanged. We expect **$> 5\%$ gain on teleport mazes**, **no regression on deterministic mazes / manipulation**, and an ablation against ensemble-mean ($\beta_{\text{pes}}=0$) confirming that the gain comes from uncertainty, not from averaging. Total budget $\sim 80$ runs $\times\,6$ GPU-hours $=\,2.5$ days on $8$ GPUs.
