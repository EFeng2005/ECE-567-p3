# C-HIQL Phase 2 Training Plan

**Goal:** Implement Conservative HIQL (per [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md)), train 3 seeds on `antmaze-teleport-navigate-v0`, evaluate with a β sweep, and produce a comparison table vs the Phase 1 HIQL baseline — all by 2026-04-24.

**Architecture:** Add an N-head value ensemble to HIQL's `GCValue` (N=5). Keep all HIQL training losses and actor code unchanged. Replace inference-time single-sample subgoal selection with: (1) draw K=16 subgoal representations from `π_high`, (2) score each with the N value heads, (3) pick `argmax (μ − β·σ)`. β is an inference-time knob — one trained checkpoint serves all β values.

**Tech stack:** JAX/Flax (OGBench `impls/`), SLURM on Great Lakes (`spgpu` partition), Python 3.10.

---

## Timeline (3 days to deadline 2026-04-24)

| Date | What lands |
|---|---|
| **Tue 04-21 (today)** | Tasks 1–4: code changes + smoke test + submit 3 training seeds before midnight |
| **Wed 04-22** | Trainings finish (≤ 8 h each, parallel). Task 5–7: run β sweep, aggregate results |
| **Thu 04-23** | Write the 10-page report. Optional Task 8 (deterministic no-regression env) if time allows |
| **Fri 04-24** | Submit |

**Hard constraint:** trainings must be in the queue by end of day Tuesday. If there's a bug found Wednesday, one emergency retrain fits; two do not.

---

## File structure

Changes live in `external/ogbench/impls/` (the OGBench clone) and in `scripts/`. At the end we regenerate `patches/ogbench_local.patch` so all edits are captured in the replication repo.

**Modified files:**
- `external/ogbench/impls/utils/networks.py` — parameterize `GCValue` ensemble size (2 → N, default 2 for back-compat)
- `external/ogbench/impls/agents/__init__.py` — register `chiql`

**New files:**
- `external/ogbench/impls/agents/chiql.py` — C-HIQL agent (fork of `hiql.py` with ensemble V + K-sample pessimistic subgoal selection)
- `scripts/submit_chiql.sh` — SLURM submission for 3 training seeds
- `scripts/eval_chiql_sweep.py` — β-sweep evaluation on trained checkpoints
- `scripts/extract_chiql_results.py` — build the comparison table

**Not modified:** `main.py` (already supports `--restore_path` / `--restore_epoch`), `evaluation.py`, `datasets.py`, all other agents.

---

## Task 1: Parameterize `GCValue` for N-head ensemble

**Why:** HIQL's `GCValue` hard-codes `ensemblize(MLP, 2)`. C-HIQL needs N=5. Keeping the default at 2 preserves all Phase 1 behavior for other agents.

**File:** `external/ogbench/impls/utils/networks.py` — class `GCValue` at [networks.py:269](external/ogbench/impls/utils/networks.py#L269).

**Change:** add a `num_ensemble: int = 2` attribute and plumb it through.

```python
class GCValue(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    num_ensemble: int = 2        # NEW
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, self.num_ensemble)   # was: 2
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net
```

**Verify:**
```bash
cd external/ogbench/impls
python -c "
import jax, jax.numpy as jnp
from utils.networks import GCValue
from utils.encoders import GCEncoder
from utils.networks import Identity
m = GCValue(hidden_dims=(64, 64), num_ensemble=5, gc_encoder=GCEncoder(state_encoder=Identity(), concat_encoder=Identity()))
params = m.init(jax.random.PRNGKey(0), jnp.zeros((4, 10)), jnp.zeros((4, 10)))
v = m.apply(params, jnp.zeros((4, 10)), jnp.zeros((4, 10)))
print('shape:', v.shape)  # expect (5, 4)
"
```
Expected stdout: `shape: (5, 4)`.

---

## Task 2: Create `chiql.py` agent

**File:** new `external/ogbench/impls/agents/chiql.py`. Start by copying `hiql.py` and then applying the diffs below.

```bash
cp external/ogbench/impls/agents/hiql.py external/ogbench/impls/agents/chiql.py
```

**Changes in `chiql.py`:**

### 2a. Class + config

At the top, rename `HIQLAgent` → `CHIQLAgent`. Then in `get_config()` at the bottom:

```python
def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='chiql',                  # was: 'hiql'
            # ... keep every other HIQL hyperparameter identical ...
            num_value_heads=5,                   # NEW (N)
            num_subgoal_candidates=16,           # NEW (K)
            pessimism_beta=0.5,                  # NEW (β)
            # ... rest unchanged ...
        )
    )
    return config
```

(Copy the full HIQL `get_config` dict verbatim, then splice the three new keys in. Do NOT delete any existing HIQL keys.)

### 2b. Wire `num_value_heads` into the value networks

In `CHIQLAgent.create`, find the two `GCValue(...)` constructors ([hiql.py:262-273](external/ogbench/impls/agents/hiql.py#L262-L273)) and add the new attribute:

```python
value_def = GCValue(
    hidden_dims=config['value_hidden_dims'],
    layer_norm=config['layer_norm'],
    ensemble=True,
    num_ensemble=config['num_value_heads'],       # NEW
    gc_encoder=value_encoder_def,
)
target_value_def = GCValue(
    hidden_dims=config['value_hidden_dims'],
    layer_norm=config['layer_norm'],
    ensemble=True,
    num_ensemble=config['num_value_heads'],       # NEW
    gc_encoder=target_value_encoder_def,
)
```

### 2c. Generalize `value_loss` to N heads

Current HIQL `value_loss` unpacks `(v1, v2) = network.select('value')(...)`. With N heads, `network.select('value')(...)` returns a tensor of shape `(N, batch)`. Rewrite:

```python
def value_loss(self, batch, grad_params):
    # Target values: shape (N, B)
    next_vs_t = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
    vs_t      = self.network.select('target_value')(batch['observations'],      batch['value_goals'])

    # Conservative aggregate for the advantage weight (matches HIQL's min-of-2 spirit).
    next_v_t_agg = jnp.min(next_vs_t, axis=0)        # (B,)
    v_t_agg      = jnp.mean(vs_t,     axis=0)        # (B,)
    q_agg = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t_agg
    adv   = q_agg - v_t_agg                           # (B,)

    # Per-head TD targets: each head sees its own bootstrapped next-value.
    qs = batch['rewards'][None, :] + self.config['discount'] * batch['masks'][None, :] * next_vs_t   # (N, B)

    # Current values (grad): shape (N, B)
    vs = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)

    # Expectile loss per head, broadcasting adv across the head axis.
    per_head_loss = self.expectile_loss(adv[None, :], qs - vs, self.config['expectile'])   # (N, B)
    value_loss = per_head_loss.mean()

    v_mean_across_heads = vs.mean(axis=0)
    return value_loss, {
        'value_loss': value_loss,
        'v_mean': v_mean_across_heads.mean(),
        'v_max':  vs.max(),
        'v_min':  vs.min(),
        'v_std_across_heads': vs.std(axis=0).mean(),
    }
```

### 2d. Update `low_actor_loss` and `high_actor_loss` to handle N-head V

Both methods currently unpack `(v1, v2) = ...` and do `v = (v1 + v2) / 2`. Replace with mean over all N heads:

```python
# low_actor_loss — [hiql.py:60]
vs  = self.network.select('value')(batch['observations'],      batch['low_actor_goals'])   # (N, B)
nvs = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])   # (N, B)
v   = vs.mean(axis=0)
nv  = nvs.mean(axis=0)
adv = nv - v
# ... rest unchanged ...

# high_actor_loss — [hiql.py:99]
vs  = self.network.select('value')(batch['observations'],       batch['high_actor_goals'])   # (N, B)
nvs = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])   # (N, B)
v   = vs.mean(axis=0)
nv  = nvs.mean(axis=0)
adv = nv - v
# ... rest unchanged ...
```

### 2e. Pessimistic K-sample subgoal selection in `sample_actions`

This is the core inference change. Replace [hiql.py:168-192](external/ogbench/impls/agents/hiql.py#L168-L192):

```python
@jax.jit
def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
    """Sample actions using ensemble-pessimistic subgoal selection.

    1. Draw K candidate subgoal representations from the high-level policy.
    2. Score each candidate with the N-head value ensemble:  μ(s, g_k) - β · σ(s, g_k).
    3. Pick the argmax candidate.
    4. Feed it to the low-level policy as before.
    """
    high_seed, low_seed = jax.random.split(seed)

    K    = self.config['num_subgoal_candidates']
    beta = self.config['pessimism_beta']

    # Per-candidate RNG keys.
    cand_seeds = jax.random.split(high_seed, K)                                 # (K, 2)

    # Broadcast observations and goals across K candidates.
    # observations shape: (obs_dim,) -> (K, obs_dim)
    obs_rep  = jnp.broadcast_to(observations, (K,) + observations.shape)
    goal_rep_in = jnp.broadcast_to(goals, (K,) + goals.shape)

    high_dist   = self.network.select('high_actor')(obs_rep, goal_rep_in, temperature=temperature)
    goal_reps   = jax.vmap(lambda d, s: d.sample(seed=s))(high_dist, cand_seeds)   # (K, rep_dim)
    # Length-normalize to match HIQL's representation constraint.
    goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

    # Score each candidate against the N-head value ensemble:
    # network.select('value')(obs, goal_rep) returns (N,) for a single (obs, goal_rep).
    # vmap over the K-axis to get (K, N).
    def score_one(gr):
        vs = self.network.select('value')(observations, gr)   # (N,)
        return vs
    vs_all = jax.vmap(score_one)(goal_reps)                   # (K, N)
    mu  = vs_all.mean(axis=-1)                                # (K,)
    sig = vs_all.std(axis=-1)                                 # (K,)
    scores = mu - beta * sig                                  # (K,)
    best_idx = jnp.argmax(scores)
    selected_goal_rep = goal_reps[best_idx]                   # (rep_dim,)

    # Low-level actor as before.
    low_dist = self.network.select('low_actor')(observations, selected_goal_rep, goal_encoded=True, temperature=temperature)
    actions = low_dist.sample(seed=low_seed)
    if not self.config['discrete']:
        actions = jnp.clip(actions, -1, 1)
    return actions
```

**Note:** `evaluate()` in [utils/evaluation.py](external/ogbench/impls/utils/evaluation.py) calls `sample_actions` once per step, expecting a single action tensor. The shapes above assume single-observation inputs (no batch dim), which matches how OGBench's `evaluate` calls it. Verify by inspecting `evaluate.py` during Task 3.

### 2f. Register the agent

Edit `external/ogbench/impls/agents/__init__.py`:

```python
from agents.chiql import CHIQLAgent   # NEW
# ...
agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    chiql=CHIQLAgent,     # NEW
    qrl=QRLAgent,
    sac=SACAgent,
)
```

---

## Task 3: Smoke test (must pass before submitting full trainings)

**Why:** Full trainings are 8 h each × 3 seeds. A shape bug discovered post-submission wastes all three.

Run a tiny training run **on a laptop or an interactive GPU session**, NOT via sbatch:

```bash
cd external/ogbench/impls
python main.py \
  --env_name=antmaze-teleport-navigate-v0 \
  --agent=agents/chiql.py \
  --seed=0 \
  --train_steps=2000 \
  --eval_interval=2000 \
  --eval_episodes=5 \
  --video_episodes=0 \
  --save_interval=2000 \
  --run_group=smoke \
  --save_dir=/tmp/chiql_smoke \
  --wandb_mode=disabled \
  --dataset_dir="$HOME/.ogbench/data"
```

**Expected:**
- No shape/JIT errors.
- `value_loss` decreases over the 2000 steps.
- `train.csv` contains a `value/v_std_across_heads` column that is nonzero (heads diversified).
- A final `params_2000.pkl` file is written.
- `eval.csv` contains one row with an `evaluation/overall_success` number (likely near 0 — this is just a smoke test).

**If this fails:** fix before Task 4. Do not submit 8-hour jobs until this passes.

---

## Task 4: Submit 3 C-HIQL training seeds on teleport

**File:** new `scripts/submit_chiql.sh`.

```bash
#!/usr/bin/env bash
# Submit 3 C-HIQL training seeds on antmaze-teleport-navigate-v0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="$REPO_ROOT/scripts/train_ogbench.sbatch"

ACCOUNT="${ACCOUNT:-ece567w26_class}"
PARTITION="${PARTITION:-spgpu}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/chiql_runs}"
DATASET_DIR="${DATASET_DIR:-$HOME/.ogbench/data}"
VENV_PATH="${VENV_PATH:?VENV_PATH must be set}"

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

# Match HIQL hyperparameters for the teleport/navigate dataset (alphas = 3.0) plus C-HIQL additions.
EXTRA="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.num_value_heads=5 --agent.num_subgoal_candidates=16 --agent.pessimism_beta=0.5 --video_episodes=0 --save_interval=1000000"

for seed in 0 1 2; do
  exports="SEED=$seed,RUN_GROUP=chiql_phase2,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH,ENV_NAME=antmaze-teleport-navigate-v0,AGENT_PATH=agents/chiql.py,EXTRA_ARGS=$EXTRA"
  sbatch --account="$ACCOUNT" --partition="$PARTITION" \
    --job-name="chiql_at_s${seed}" --mem=32G --time=08:00:00 \
    --export="ALL,$exports" "$SBATCH"
done
```

Make executable and submit:

```bash
chmod +x scripts/submit_chiql.sh
export VENV_PATH="/path/to/ogbench_venv"    # your venv
bash scripts/submit_chiql.sh
squeue -u $USER
```

**Expected:** three `chiql_at_s0`, `chiql_at_s1`, `chiql_at_s2` jobs in the queue.

**Note on β during training:** `pessimism_beta=0.5` during training is harmless — β only affects `sample_actions`, which is used by periodic eval during training but not by the training loss. The trained checkpoint is independent of β, so Task 6 can freely sweep it.

---

## Task 5: β-sweep eval script

**File:** new `scripts/eval_chiql_sweep.py`. Runs once per seed checkpoint, sweeping β and writing one CSV row per (seed, β).

```python
#!/usr/bin/env python3
"""Evaluate trained C-HIQL checkpoints over a β sweep.

Usage:
    python scripts/eval_chiql_sweep.py \
        --ckpt_glob "results/raw/chiql_runs/OGBench/chiql_phase2/sd*" \
        --output    results/chiql_sweep/summary.csv \
        --betas     0.0,0.25,0.5,1.0,2.0 \
        --k         16 \
        --eval_episodes 50
"""
import argparse, csv, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "ogbench", "impls"))

import json
import numpy as np
import jax
import flax
from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--betas", default="0.0,0.25,0.5,1.0,2.0")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--epoch", type=int, default=1000000)
    args = ap.parse_args()

    betas = [float(b) for b in args.betas.split(",")]
    run_dirs = sorted(glob.glob(args.ckpt_glob))
    print(f"Found {len(run_dirs)} run dirs")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = []
    for run_dir in run_dirs:
        flags = json.load(open(os.path.join(run_dir, "flags.json")))
        env_name = flags["env_name"]
        seed = int(flags["seed"])
        config = flags["agent"]

        env, train_ds, _ = make_env_and_datasets(env_name, frame_stack=config.get("frame_stack"),
                                                 dataset_dir=os.environ.get("DATASET_DIR"))
        # Example batch for agent.create.
        import numpy as np
        obs_ex = np.zeros((1,) + env.observation_space.shape, dtype=np.float32)
        act_ex = np.zeros((1,) + env.action_space.shape, dtype=np.float32) if hasattr(env.action_space, 'shape') \
                 else np.zeros((1,), dtype=np.int32)

        agent = agent_registry[config["agent_name"]].create(seed, obs_ex, act_ex, config)
        agent = restore_agent(agent, run_dir, args.epoch)

        for beta in betas:
            cfg = flax.core.FrozenDict({**agent.config, "pessimism_beta": beta, "num_subgoal_candidates": args.k})
            agent_beta = agent.replace(config=cfg)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            overall = []
            for t in range(1, len(task_infos) + 1):
                info, _, _ = evaluate(agent=agent_beta, env=env, task_id=t, config=cfg,
                                      num_eval_episodes=args.eval_episodes, num_video_episodes=0,
                                      video_frame_skip=1, eval_temperature=0.0)
                overall.append(info.get("success", 0.0))
            mean_success = float(np.mean(overall))
            print(f"  seed={seed} β={beta}  overall_success={mean_success:.4f}")
            rows.append({"env": env_name, "seed": seed, "beta": beta,
                         "k": args.k, "overall_success": mean_success})

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "seed", "beta", "k", "overall_success"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
```

**Why this works:** `agent.replace(config=new_config)` rebinds β and K without touching trained params (Flax `PyTreeNode` pattern). The JIT cache of `sample_actions` re-traces once per config change, which is cheap.

---

## Task 6: Run the β sweep

Once all 3 trainings are done (check `squeue -u $USER` shows none queued):

```bash
source "$VENV_PATH/bin/activate"
export DATASET_DIR="$HOME/.ogbench/data"   # or $SCRATCH/ogbench/data
python scripts/eval_chiql_sweep.py \
  --ckpt_glob "results/raw/chiql_runs/OGBench/chiql_phase2/sd*" \
  --output    results/chiql_sweep/summary.csv \
  --betas     0.0,0.25,0.5,1.0,2.0 \
  --k         16 \
  --eval_episodes 50
```

**Expected runtime:** ~3 seeds × 5 β × 50 episodes per task × ~5 tasks ≈ 3750 rollouts. On CPU in eval mode (OGBench defaults to `eval_on_cpu=1`), budget ~30–60 minutes total. Run on a login node or as a short SLURM job.

---

## Task 7: Aggregate + comparison table

**File:** new `scripts/extract_chiql_results.py`.

Reads:
- Phase 1 HIQL baseline: `results/antmaze-teleport-navigate-v0/hiql/seed{0,1,2}/eval.csv` → last row's `evaluation/overall_success`.
- Phase 2 C-HIQL sweep: `results/chiql_sweep/summary.csv`.

Writes `results/chiql_sweep/comparison.csv` and prints this table:

```
Method                 Seed 0   Seed 1   Seed 2   Mean    Std
HIQL (baseline)        0.440    0.356    0.416    0.404   0.035
C-HIQL β=0.00          ...
C-HIQL β=0.25          ...
C-HIQL β=0.50          ...
C-HIQL β=1.00          ...
C-HIQL β=2.00          ...
```

Minimal implementation:

```python
#!/usr/bin/env python3
import csv, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV = "antmaze-teleport-navigate-v0"

# HIQL baseline
hiql_seeds = []
for s in [0, 1, 2]:
    p = REPO / "results" / ENV / "hiql" / f"seed{s}" / "eval.csv"
    with open(p) as f: rows = list(csv.DictReader(f))
    hiql_seeds.append(float(rows[-1]["evaluation/overall_success"]))

# C-HIQL sweep
rows = list(csv.DictReader(open(REPO / "results" / "chiql_sweep" / "summary.csv")))
betas = sorted({float(r["beta"]) for r in rows})
chiql_by_beta = {b: sorted([float(r["overall_success"]) for r in rows if float(r["beta"]) == b and int(r["seed"]) in (0,1,2)]) for b in betas}

def fmt(seeds):
    m = statistics.mean(seeds); s = statistics.pstdev(seeds) if len(seeds) > 1 else 0.0
    return seeds, m, s

print(f"{'Method':<22}{'Seed 0':>8}{'Seed 1':>8}{'Seed 2':>8}{'Mean':>8}{'Std':>8}")
print("-" * 62)
for name, seeds in [("HIQL (baseline)", hiql_seeds)] + [(f"C-HIQL β={b:.2f}", chiql_by_beta[b]) for b in betas]:
    vals, m, s = fmt(seeds)
    vals_str = "".join(f"{v:>8.3f}" for v in vals)
    print(f"{name:<22}{vals_str}{m:>8.3f}{s:>8.3f}")
```

Run it, paste the table into the report.

---

## Task 8 (optional, only if Wed evening has time): No-regression check on `antmaze-large-navigate`

Phase 1 HIQL scores 0.889 on this deterministic env. The proposal claims "performance drop <3–5%" (i.e., C-HIQL should score ≥ 0.85). To defend that claim, repeat Tasks 4 and 6 with `ENV_NAME=antmaze-large-navigate-v0` and `EXTRA` including `--agent.alpha=0.3` removed (HIQL doesn't take it — just keep `high_alpha=3.0, low_alpha=3.0`). Three more trainings, 8 h each.

If queue is slow Tuesday night, skip this and frame the paper as "headline claim is on teleport; deterministic-env check is future work." Do not burn Wednesday debugging a new env — the teleport result is the thesis.

---

## Final commit + patch refresh

At the end, capture all OGBench edits in the replication repo so it's reproducible:

```bash
cd external/ogbench
git diff HEAD > ../../patches/ogbench_local.patch
cd ../..
git add patches/ogbench_local.patch \
        scripts/submit_chiql.sh scripts/eval_chiql_sweep.py scripts/extract_chiql_results.py \
        results/chiql_sweep/ PHASE2_TRAINING_PLAN.md FINAL_PROPOSAL.md
git commit -m "Phase 2: Conservative HIQL (C-HIQL) on antmaze-teleport"
git push
```

---

## Experiment budget (summary)

| Resource | Count |
|---|---|
| Training jobs (SLURM, 8 h spgpu) | **3** (teleport × 3 seeds) |
| + optional deterministic-env check | **+3** |
| Eval configs (β sweep, no training) | **15** (3 seeds × 5 β), one command |
| Baselines reused from Phase 1 | HIQL × 3 seeds (already in `results/`) |

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Shape bug in N-head value loss | Medium | Task 3 smoke test catches it before 8-h jobs |
| `sample_actions` shape mismatch with `evaluate()` | Medium | Inspect `evaluation.py` during Task 3; add a single-step `agent.sample_actions(obs_ex, goal_ex, key)` call at end of smoke test |
| Heads collapse to identical functions (σ ≈ 0) → pessimism does nothing | Low | Task 3 logs `v_std_across_heads`; if ~0, add bootstrap masking (proposal's optional `p=0.8`) |
| C-HIQL shows no improvement over HIQL on teleport | Medium | Report the negative result honestly; framing becomes "when uncertainty helps and when it doesn't" rather than "we improved X" |
| SLURM queue slow | Medium | Submit Tuesday night. If still queued Wednesday AM, email the course staff / use `squeue` to check partition availability |
