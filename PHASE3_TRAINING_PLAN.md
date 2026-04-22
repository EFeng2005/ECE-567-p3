# Phase 3 Training Plan — EX-HIQL (Expectile-Heads C-HIQL)

> **Parallel to `PHASE2_TRAINING_PLAN.md`.** Implements the idea in [EXPECTILE_HIQL.md](EXPECTILE_HIQL.md): train the existing C-HIQL ensemble with per-head expectiles `(0.1, 0.3, 0.5, 0.7, 0.9)` and keep inference scoring as μ − β·σ (Option B). Subgoal scoring uses Option B because it (a) preserves the C-HIQL framework exactly and (b) gives us a β sweep for free, so we can check whether the *fixed* σ helps beyond ensemble-mean (β=0) alone.

---

## Goal & deliverables

- Produce 3 EX-HIQL training seeds on `antmaze-teleport-navigate-v0` with `params_1000000.pkl` saved.
- Re-run the β sweep from Phase 2 on the new checkpoints.
- Re-run `scripts/diagnose_sigma.py` on the new checkpoints. **Success gate**: per-state μ↔σ correlation flips from Phase-2's ≈ −0.44 to near 0 or positive.
- (Optional, if time permits) 3-seed deterministic-env control on `antmaze-large-navigate-v0`.

---

## Timeline

| Day | Task |
|---|---|
| **Day 1 AM** | Tasks 1–3: implement agent, smoke test |
| **Day 1 PM** | Task 4: launch 3-seed parallel training (~5h) |
| **Day 2 AM** | Tasks 5–6: β sweep + σ diagnostic |
| **Day 2 PM** | Task 7: compare-and-decide. If E3 passes, optional Task 8 (deterministic control) |
| **Day 3** | Writeup / slides |

Training parallelism, dataset caching, wandb-off, and all other machine-level choices inherited from Phase 2 — see [PHASE2_TRAINING_PLAN.md](PHASE2_TRAINING_PLAN.md).

---

## File structure

**New files (tracked):**
- `external/ogbench/impls/agents/ex_chiql.py` — new agent (fork of `chiql.py` with per-head expectiles + actor pinned to τ=0.7 head).
- `scripts/run_ex_chiql_local.sh` — per-seed runner (parallel to `run_chiql_local.sh`).
- `scripts/launch_3seeds_ex.sh` — 3-seed staggered launcher (parallel to `launch_3seeds.sh`).

**Modified files:**
- `external/ogbench/impls/agents/__init__.py` — register `ex_chiql`.

**Re-used without modification:**
- `scripts/eval_beta_sweep.py` — already `agent_name`-agnostic (uses `CHIQLAgent.create`); just needs to import `EXCHIQLAgent` for EX-HIQL checkpoints. We'll add a `--agent` flag or a sibling script.
- `scripts/diagnose_sigma.py` — same; one-line import swap.

**Not modified:** `main.py`, `evaluation.py`, `networks.py`, all other agents.

---

## Task 1 — `EXCHIQLAgent` (new file `ex_chiql.py`)

**File:** `external/ogbench/impls/agents/ex_chiql.py`

Fork of `chiql.py`. Changes, in order:

### 1a. Rename class and config

```python
class EXCHIQLAgent(flax.struct.PyTreeNode):
    # rest of class ...

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ex_chiql',
            num_value_heads=5,
            head_expectiles=(0.1, 0.3, 0.5, 0.7, 0.9),  # NEW — per-head τ
            actor_expectile_index=3,                     # NEW — which head feeds the actor
            num_subgoal_candidates=16,
            pessimism_beta=0.5,
            # ... rest unchanged ...
        )
    )
```

Important: `num_value_heads` **must** equal `len(head_expectiles)`. Assert this in `create()`.

### 1b. `value_loss` — broadcast τ over heads

Current (`chiql.py`):
```python
per_head_loss = self.expectile_loss(advs, qs - vs, self.config['expectile'])
```
becomes:
```python
expectiles = jnp.array(self.config['head_expectiles']).reshape(-1, 1)   # (N, 1)
per_head_loss = self.expectile_loss(advs, qs - vs, expectiles)          # broadcasts (N, B)
```

`expectile_loss` is already broadcast-safe: `jnp.where(adv >= 0, expectile, 1-expectile)` handles array `expectile`.

Also remove the top-level `expectile` key from the config (no longer used; keep if backwards-compat is needed).

### 1c. Actor losses — pin to one head

In `low_actor_loss`:
```python
# was:  v = vs.mean(axis=0);  nv = nvs.mean(axis=0)
idx = self.config['actor_expectile_index']
v = vs[idx]
nv = nvs[idx]
```

Same change in `high_actor_loss`. No other modifications to the actor losses.

### 1d. `sample_actions` — unchanged

Inference already computes μ and σ over the ensemble (chiql.py line 245-249). That code works identically on the new ensemble; σ now reflects per-head-expectile spread instead of init-noise spread. **Do not touch `sample_actions`.**

### 1e. Docstrings

Update the class docstring to point at `EXPECTILE_HIQL.md §3` and note that this is a mechanism-fix over C-HIQL — per-head τ, not per-head random init.

---

## Task 2 — register agent

**File:** `external/ogbench/impls/agents/__init__.py`

```python
from agents.chiql import CHIQLAgent
from agents.ex_chiql import EXCHIQLAgent   # NEW
# ... other imports ...

agents = dict(
    chiql=CHIQLAgent,
    ex_chiql=EXCHIQLAgent,                  # NEW
    # ... others ...
)
```

This file currently lives only as a patched-in version at `external/ogbench_full/impls/agents/__init__.py`. Either:
- Add it to the tracked patched files in `external/ogbench/impls/agents/__init__.py` (preferred — tracks the change in git), or
- Modify upstream copy each rebootstrap.

Track the full `__init__.py` and adjust `.gitignore` if needed.

### Verify agent builds

```bash
cd external/ogbench_full/impls
.venv/bin/python -c "
from agents import ex_chiql
cfg = ex_chiql.get_config()
print('expectiles:', cfg['head_expectiles'])
print('actor_idx:', cfg['actor_expectile_index'])
"
```
Expected: `expectiles: (0.1, 0.3, 0.5, 0.7, 0.9)` and `actor_idx: 3`.

---

## Task 3 — smoke test

**File:** `scripts/smoke_ex_chiql.sh` (copy of `smoke_chiql.sh` with `--agent=agents/ex_chiql.py`).

Run 500-step smoke on one seed with `eval_episodes=3`. Required outcome:
- Training completes end-to-end, no NaN.
- `train.csv` shows the 5 head `v_mean` values (or ensemble-mean v_mean, per current logging).
- Eval at step 1 runs (→ ~0% success, as expected).
- Checkpoint saves.

Bonus sanity check: inspect `train.csv` a few rows in. Expect `v_std_across_heads` to **grow** over training (heads diverging), not shrink.

If smoke fails: investigate and fix **before** launching 3-seed.

---

## Task 4 — 3-seed parallel training

**File:** `scripts/launch_3seeds_ex.sh` — identical to `launch_3seeds.sh` but points at a new runner script and different run group.

Per-seed runner `scripts/run_ex_chiql_local.sh`:
```bash
--agent=agents/ex_chiql.py \
--run_group=ex_chiql_phase3 \
--agent.num_value_heads=5 \
--agent.num_subgoal_candidates=16 \
--agent.pessimism_beta=0.5 \
--agent.high_alpha=3.0 --agent.low_alpha=3.0
# do NOT pass --agent.head_expectiles; default (0.1, 0.3, 0.5, 0.7, 0.9) is fine
# do NOT pass --agent.actor_expectile_index; default 3 is fine
```

Launch:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/c/.../scripts/launch_3seeds_ex.sh'
```

Expected wall clock: 5–6 h, same as Phase-2. Schedule a 15-min cron monitor (copy/adapt from the Phase-2 one).

---

## Task 5 — β sweep

Extend `scripts/eval_beta_sweep.py` to accept an `--agent_name` or `--agent_class` flag. Simplest: one-line import + class-lookup, e.g.

```python
from agents import agents as agent_classes
agent_cls = agent_classes[args.agent_name]   # 'ex_chiql' or 'chiql'
```

Run on the 3 EX-HIQL checkpoints with β ∈ {0, 0.25, 0.5, 1.0, 2.0}. Output: `beta_sweep_ex_seed{N}.csv`.

Expected wall clock: ~50 min (parallel 3-seed).

---

## Task 6 — σ diagnostic

Re-run `scripts/diagnose_sigma.py` on each EX-HIQL seed 0/1/2 checkpoint. Critical output:

```json
{
  "per_state_mu_sig_corr_mean": <value>,   // was -0.44 for C-HIQL
  ...
}
```

Also inspect `sigma_diagnostic.png` — does σ concentrate near teleporter tiles now (the spatial scatter)?

---

## Task 7 — compare-and-decide

Aggregate table (fill after Tasks 4-6):

| β | HIQL (Phase 1) | C-HIQL (Phase 2) | EX-HIQL (Phase 3) |
|---|---|---|---|
| — | 0.404 | — | — |
| 0 | — | 0.404 | ?? |
| 0.25 | — | ?? | ?? |
| 0.5 | — | 0.384 | ?? |
| 1.0 | — | ?? | ?? |
| 2.0 | — | ?? | ?? |

Decision gate (per Claim E1+E2+E3 in `EXPECTILE_HIQL.md §7`):

- **GO to paper** if: at some β > 0, EX-HIQL mean ≥ HIQL + 5 pts AND σ correlation flipped (E3 positive).
- **Partial result** if: some β > 0 beats HIQL but σ correlation still negative — method worked for a reason we didn't predict; needs more investigation but still publishable.
- **Pivot** if: no β beats HIQL significantly.

---

## Task 8 (Optional) — deterministic-env control

If E1–E3 all pass, run 3 seeds on `antmaze-large-navigate-v0` (~5h training × 3 parallel) to check Claim E4 (no regression on deterministic env). Reuse `run_chiql_local.sh` with `--env_name=antmaze-large-navigate-v0 --agent=agents/ex_chiql.py` and default β=0.5.

Expected: matches Phase-1 HIQL's 0.889 within 3 pts. If EX-HIQL drops deterministic-env performance substantially, the story becomes "tradeoff between stochastic and deterministic robustness" — still publishable, different framing.

---

## Risk checkpoints

- **After Task 3 (smoke)**: go/no-go. If any NaN or convergence pathology, halt and debug.
- **After Task 4 (training done)**: sanity-check `v_std_across_heads` in `train.csv`. If it's ≈ 0 (heads collapsed despite different τ) — training didn't separate the heads, something is wrong with the expectile vector broadcast. Check `value_loss` code and re-run.
- **After Task 6 (diagnostic)**: if σ correlation is still strongly negative, STOP. The mechanism still doesn't work; further retraining with this architecture won't help. Pivot: either to independent trunks (5× params) or CRL-based scoring (Variant C in `C-HIQL.md §11`).

---

## Fallback variants (from `EXPECTILE_HIQL.md §9` risks table)

| Fallback | Description | When to deploy |
|---|---|---|
| F1 | `head_expectiles=(0.3, 0.5, 0.7, 0.9, 0.95)` (no τ=0.1) | If Task 7 shows E1 partial positive but low-τ head is noisy or hurts |
| F2 | Per-head independent trunks (no shared trunk) | If Task 6 σ correlation is still negative — need real ensemble diversity |
| F3 | Swap Option B → Option A (pick `vs[0]`, the τ=0.1 head, directly as score) | If β sweep is flat but low-τ head gives good direct-pessimism scores |
| F4 | Switch to CRL-based subgoal scoring (Variant C in `C-HIQL.md §11`) | If the whole ensemble-pessimism family doesn't work on this env |

---

## Metrics & deliverables summary

- **Primary**: mean success rate across 3 seeds × 5 β values on `antmaze-teleport-navigate-v0` (5×3 matrix).
- **Secondary**: σ diagnostic stats (correlation, spatial plot) for each EX-HIQL seed.
- **Comparison table**: HIQL baseline / C-HIQL / EX-HIQL on the same rows.
- **Optional**: deterministic-env (`antmaze-large-navigate-v0`) 3-seed control.
- **Writeup**: incorporate the above into the final project report as the Phase 3 section.
