# Final Proposal: Conservative HIQL (C-HIQL)

## Method Thesis

C-HIQL improves offline hierarchical goal-conditioned RL in stochastic environments by making HIQL's high-level subgoal selection pessimistic via ensemble uncertainty, while leaving the low-level controller and training targets unchanged.

## Problem Anchor

HIQL's high-level planner selects subgoals by maximizing V(s, g_sub), but in stochastic OGBench environments (e.g., teleport mazes) this leads to value overestimation and unreliable subgoal choices. The core issue: optimistic value estimates compound through the hierarchy, causing the planner to select subgoals that appear reachable but are not reliably achievable.

## Algorithm Changes

### Architecture
- **Shared trunk**: Keep HIQL's high-level encoder unchanged
- **Ensemble heads**: Replace single value head with N=5 independent scalar heads V_i(s, g)
- **Low-level**: Completely unchanged (actor, critic, policy extraction)

### Training
- Each head trained on the same HIQL high-level value loss (expectile regression)
- Different random initialization per head for diversity
- Optional: bootstrap masks (p=0.8) per head for additional diversity

### Inference (subgoal selection only)
```python
def select_subgoal(s, candidate_subgoals, beta):
    scores = []
    for g_sub in candidate_subgoals:
        z = high_level_trunk(s, g_sub)
        vals = [value_head[i](z) for i in range(N)]
        mu = mean(vals)
        sigma = std(vals)
        score = mu - beta * sigma
        scores.append(score)
    return candidate_subgoals[argmax(scores)]
```

### What does NOT change
- Low-level policy and value function
- Training targets (no pessimism in bootstrap)
- Candidate subgoal generation procedure
- All other HIQL hyperparameters

## Hyperparameters

| Parameter | Default | Sweep Range | Justification |
|-----------|---------|-------------|---------------|
| N (ensemble size) | 5 | {1, 3, 5} | 5 balances diversity and compute; 3 is cheaper backup |
| beta (pessimism) | 0.5 | {0.0, 0.25, 0.5, 1.0, 2.0} | 0.0 = ensemble mean control; 1-2 = strong pessimism |
| Bootstrap mask p | None | {None, 0.8} | Optional diversity boost |

## Claims and Evidence

| Claim | Experiment | Success Criterion |
|-------|-----------|-------------------|
| C-HIQL improves on stochastic envs | C-HIQL vs HIQL on teleport mazes, 3+ seeds | >5% success rate improvement |
| No regression on deterministic envs | C-HIQL vs HIQL on standard mazes + manipulation | Performance drop <3-5% |
| Improvement from uncertainty, not ensembling | C-HIQL(beta>0) vs Ensemble-Mean(beta=0) vs HIQL | Only beta>0 improves stochastic |

## Positioning

This is a **risk-sensitive subgoal selection** method for offline hierarchical GCRL, not a grand uncertainty framework. The contribution is showing that a lightweight pessimistic correction at the planning level fixes a known failure mode without disrupting HIQL's strengths.
