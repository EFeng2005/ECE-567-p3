# C-HIQL Verification Plan (before full training)

**Goal:** Before committing 24 GPU-hours to a 3-seed C-HIQL training, run three cheap diagnostics on the existing HIQL teleport checkpoints to validate the proposal's mechanism. If diagnostics green-light the hypothesis, execute [PHASE2_TRAINING_PLAN.md](PHASE2_TRAINING_PLAN.md). If not, pivot the paper before wasting compute.

**Principle:** each diagnostic tests one assumption the method relies on. If an assumption fails, fix the mechanism or change the scope — don't just run the method anyway.

---

## Timeline

| Date | Work |
|---|---|
| **Tue 04-21 evening** | Phase A: run diagnostics V0–V3 (≈3 h total, mostly compute-bound) |
| **Wed 04-22 morning** | **Decision gate**: go / pivot based on V1–V3 results |
| Wed 04-22 (if go) | Implement C-HIQL code + smoke test; submit 3 training seeds by EOD |
| Thu 04-23 | Trainings finish, run β sweep, aggregate |
| Fri 04-24 | Write report + submit |

Miss Wednesday's decision gate and you miss the paper. So Phase A must finish tonight.

---

## What you need on the machine running Phase A

All diagnostics need:
1. A machine with the `ogbench_venv` Python environment activated (`$VENV_PATH`)
2. The existing HIQL checkpoint: `params_1000000.pkl` from the Phase 1 teleport runs
3. (V2 only) HIQL checkpoint on `antmaze-large-navigate-v0` as a deterministic control

If those live on Great Lakes scratch, run Phase A as an interactive SLURM session (`srun --partition=spgpu --gres=gpu:1 --mem=16G --time=03:00:00 --pty bash`) OR submit each diagnostic as a short sbatch job. V0 and V1 are CPU-only; V2 uses GPU briefly; V3 should use GPU for the rollout speed.

---

## Task V0: Prerequisite check (5 min)

Before writing any code, confirm the checkpoints exist. Each organized run has a `source.txt` pointer to the raw dir written by [organize_results.py:87-88](scripts/organize_results.py#L87-L88).

```bash
cd "$REPO_ROOT"

# Read the raw paths recorded by organize_results.py.
for seed in 0 1 2; do
  src=$(cat "results/antmaze-teleport-navigate-v0/hiql/seed${seed}/source.txt")
  ckpt="${src}/params_1000000.pkl"
  if [ -f "$ckpt" ]; then
    size=$(stat -f%z "$ckpt" 2>/dev/null || stat -c%s "$ckpt")
    echo "seed${seed}: OK  ($ckpt, ${size} bytes)"
  else
    echo "seed${seed}: MISSING  ($ckpt)"
  fi
done

# Control env for V2
for seed in 0; do
  src=$(cat "results/antmaze-large-navigate-v0/hiql/seed${seed}/source.txt")
  ckpt="${src}/params_1000000.pkl"
  [ -f "$ckpt" ] && echo "control_seed${seed}: OK" || echo "control_seed${seed}: MISSING"
done
```

**Decision rule:**
- All OK → proceed to V1.
- Teleport checkpoints missing → you must retrain 1 HIQL teleport seed before any diagnostic can run. That's 8 h — submit it now as `bash scripts/submit_experiments.sh` filtered to just one job, and continue this plan tomorrow.
- Only the control (antmaze-large-navigate) is missing → V2 gets skipped; V1 and V3 still work.

---

## Task V1: Does π_high produce diverse K-sample candidates? (15 min)

**Hypothesis under test:** `π_high(s, g)` has enough variance that 16 independent samples actually span different subgoal directions on the 10-sphere. If the policy is near-deterministic, C-HIQL's K-sample pessimism is fighting a ghost.

**File:** new `scripts/diag1_pihigh_spread.py`

```python
#!/usr/bin/env python3
"""Diagnostic 1: How diverse are subgoal rep samples from HIQL's high-level policy?

Loads one HIQL teleport checkpoint, samples K=16 subgoal representations from pi_high
at each of ~30 randomly chosen states, and reports the spread of those samples on the
10-sphere.

Run:
    python scripts/diag1_pihigh_spread.py \
        --run_dir "$(cat results/antmaze-teleport-navigate-v0/hiql/seed0/source.txt)"
"""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "ogbench", "impls"))

import jax
from agents import agents as agent_registry
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_states", type=int, default=30)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epoch", type=int, default=1000000)
    ap.add_argument("--dataset_dir", default=os.environ.get("DATASET_DIR"))
    args = ap.parse_args()

    flags = json.load(open(os.path.join(args.run_dir, "flags.json")))
    config = flags["agent"]
    env, train_ds_raw, _ = make_env_and_datasets(
        flags["env_name"], frame_stack=config.get("frame_stack"), dataset_dir=args.dataset_dir
    )
    train_ds = HGCDataset(Dataset.create(**train_ds_raw), config)

    # Build an agent with the right shapes, then restore parameters.
    example = train_ds.sample(1)
    agent = agent_registry[config["agent_name"]].create(
        flags["seed"], example["observations"], example["actions"], config
    )
    agent = restore_agent(agent, args.run_dir, args.epoch)

    # Draw n_states (s, g) pairs.
    batch = train_ds.sample(args.n_states)
    rng = jax.random.PRNGKey(0)

    stds, mean_norms = [], []
    for i in range(args.n_states):
        obs = batch["observations"][i]
        goal = batch["high_actor_goals"][i]

        rng, sub = jax.random.split(rng)
        K = args.k
        obs_rep = np.broadcast_to(obs, (K,) + obs.shape)
        goal_rep = np.broadcast_to(goal, (K,) + goal.shape)

        high_dist = agent.network.select("high_actor")(obs_rep, goal_rep, temperature=1.0)
        samples = high_dist.sample(seed=sub)  # (K, rep_dim)
        samples = np.asarray(samples)

        # Spread = mean across dims of std across K samples.
        stds.append(float(samples.std(axis=0).mean()))
        mean_norms.append(float(np.linalg.norm(samples, axis=-1).mean()))

    stds = np.array(stds); norms = np.array(mean_norms)
    print(f"K = {args.k}  |  n_states = {args.n_states}  |  env = {flags['env_name']}")
    print(f"std across K samples (per-dim avg):  median={np.median(stds):.4f}  p10={np.percentile(stds,10):.4f}  p90={np.percentile(stds,90):.4f}")
    print(f"avg sample norm (should be ~sqrt(10)≈3.16): median={np.median(norms):.3f}")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
source "$VENV_PATH/bin/activate"
export DATASET_DIR="$HOME/.ogbench/data"   # or $SCRATCH/ogbench/data on Great Lakes
python scripts/diag1_pihigh_spread.py \
  --run_dir "$(cat results/antmaze-teleport-navigate-v0/hiql/seed0/source.txt)"
```

**Decision rule:**
- `median std ≥ 0.05` → π_high has real spread; K=16 candidates will genuinely differ. Proceed.
- `median std ≈ 0.01` or below → π_high is essentially deterministic. K=16 is no better than K=1. **Mechanism broken.** Options: increase `eval_temperature` in C-HIQL to add noise, OR kill the method.

Expected in practice: HIQL's high_actor has `const_std=True` ([hiql.py:293](external/ogbench/impls/agents/hiql.py#L293)), so there's a fixed-stdev noise baked in. Should give reasonable spread.

---

## Task V2: Do HIQL's 2 value heads already disagree more on teleport? (30 min)

**Hypothesis under test:** Epistemic uncertainty about value is higher on states near teleport tiles than on clean navigation states. HIQL already trains a 2-head ensemble for min-of-2 conservatism. The 2 heads have different random inits, so any real epistemic signal should already leak through. If disagreement is indistinguishable between teleport and non-teleport envs, 5 heads probably won't magically create signal.

**File:** new `scripts/diag2_head_disagreement.py`

```python
#!/usr/bin/env python3
"""Diagnostic 2: Compare V-head disagreement between a stochastic and a deterministic env.

Loads HIQL teleport and HIQL antmaze-large-navigate checkpoints. For each, samples ~500
(s, g) pairs from its own dataset and computes |V_head_0 - V_head_1|. Prints
distributions side by side.

Run:
    python scripts/diag2_head_disagreement.py \
        --teleport_run "$(cat results/antmaze-teleport-navigate-v0/hiql/seed0/source.txt)" \
        --control_run  "$(cat results/antmaze-large-navigate-v0/hiql/seed0/source.txt)"
"""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "ogbench", "impls"))

import jax
from agents import agents as agent_registry
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent

def head_disagreement(run_dir, n, epoch, dataset_dir):
    flags = json.load(open(os.path.join(run_dir, "flags.json")))
    config = flags["agent"]
    env, train_ds_raw, _ = make_env_and_datasets(
        flags["env_name"], frame_stack=config.get("frame_stack"), dataset_dir=dataset_dir
    )
    train_ds = HGCDataset(Dataset.create(**train_ds_raw), config)
    example = train_ds.sample(1)
    agent = agent_registry[config["agent_name"]].create(
        flags["seed"], example["observations"], example["actions"], config
    )
    agent = restore_agent(agent, run_dir, epoch)

    batch = train_ds.sample(n)
    # V shape with 2-head ensemble: (2, n)
    vs = agent.network.select("value")(batch["observations"], batch["value_goals"])
    vs = np.asarray(vs)
    abs_diff = np.abs(vs[0] - vs[1])
    return flags["env_name"], abs_diff

def summarize(label, diff):
    print(f"{label:40s}  "
          f"median={np.median(diff):.4f}  "
          f"p25={np.percentile(diff,25):.4f}  "
          f"p75={np.percentile(diff,75):.4f}  "
          f"p90={np.percentile(diff,90):.4f}  "
          f"mean={diff.mean():.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleport_run", required=True)
    ap.add_argument("--control_run",  required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--epoch", type=int, default=1000000)
    ap.add_argument("--dataset_dir", default=os.environ.get("DATASET_DIR"))
    args = ap.parse_args()

    env_t, diff_t = head_disagreement(args.teleport_run, args.n, args.epoch, args.dataset_dir)
    env_c, diff_c = head_disagreement(args.control_run,  args.n, args.epoch, args.dataset_dir)

    print(f"Sampled n={args.n} (s, g) pairs from each env's own training distribution.")
    print()
    summarize(env_c + " (control)",    diff_c)
    summarize(env_t + " (teleport)",   diff_t)

    ratio = np.median(diff_t) / max(np.median(diff_c), 1e-8)
    print(f"\nratio_teleport_over_control (median |V1 - V2|) = {ratio:.2f}")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python scripts/diag2_head_disagreement.py \
  --teleport_run "$(cat results/antmaze-teleport-navigate-v0/hiql/seed0/source.txt)" \
  --control_run  "$(cat results/antmaze-large-navigate-v0/hiql/seed0/source.txt)"
```

**Decision rule:**
- `ratio ≥ 1.5` (teleport disagreement clearly higher) → epistemic signal present; 5 heads will amplify. **Strong go.**
- `1.0 < ratio < 1.5` → signal present but weak. Proceed with caution; the 5-head training might extract more but it's not certain.
- `ratio ≈ 1.0` → **warning**. Disagreement is the same on stochastic and deterministic envs. Either HIQL's 2 heads collapsed to identical behavior (inadequate diversity), or teleport's stochasticity isn't actually expressed as value disagreement. 5 heads with explicit bootstrap masks (`p=0.8` from the proposal's optional knob) might help; without that, C-HIQL likely won't work.
- `ratio < 1.0` (teleport LOWER) → hypothesis rejected. **Pivot.**

---

## Task V3: Does post-hoc pessimistic selection improve HIQL on teleport? (1–2 h)

**Hypothesis under test:** the C-HIQL inference mechanism (K-sample + μ − β·σ) produces a real improvement even with HIQL's existing 2 heads (not 5). If yes, the mechanism is validated and 5 heads should only make it cleaner. If no, retraining with 5 heads is unlikely to fix a mechanism problem.

This is the **primary go/no-go diagnostic.** It's essentially a zero-training preview of C-HIQL, using the checkpoints you already own.

**File:** new `scripts/diag3_posthoc_chiql.py`

```python
#!/usr/bin/env python3
"""Diagnostic 3: Post-hoc pessimistic subgoal selection on an existing HIQL checkpoint.

Wraps HIQL's sample_actions with: (1) draw K=16 subgoal reps from pi_high,
(2) score each with the existing 2-head V via (mean - beta*std), (3) pick argmax.
Then runs the standard OGBench evaluate() loop and compares to vanilla HIQL.

Run:
    python scripts/diag3_posthoc_chiql.py \
        --run_dir "$(cat results/antmaze-teleport-navigate-v0/hiql/seed0/source.txt)" \
        --betas 0.0,0.5,1.0,2.0 \
        --k 16 \
        --eval_episodes 50
"""
import argparse, functools, json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "ogbench", "impls"))

import jax, jax.numpy as jnp
from agents import agents as agent_registry
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent

def make_posthoc_sampler(agent, K, beta):
    """Returns a fn with HIQLAgent.sample_actions' signature but using pessimistic selection."""

    @jax.jit
    def sample_actions(observations, goals=None, seed=None, temperature=1.0):
        high_seed, low_seed = jax.random.split(seed)

        # Broadcast (obs, goals) across K candidates.
        obs_b  = jnp.broadcast_to(observations, (K,) + observations.shape)
        goal_b = jnp.broadcast_to(goals,        (K,) + goals.shape)

        high_dist = agent.network.select("high_actor")(obs_b, goal_b, temperature=temperature)
        cands = high_dist.sample(seed=high_seed)                         # (K, rep_dim)
        # HIQL length-normalizes to sqrt(rep_dim).
        cands = cands / jnp.linalg.norm(cands, axis=-1, keepdims=True) * jnp.sqrt(cands.shape[-1])

        # Score each candidate with the 2-head value ensemble.
        def score_one(g_rep):
            vs = agent.network.select("value")(observations, g_rep)      # (2,)
            return vs
        vs_all = jax.vmap(score_one)(cands)                              # (K, 2)
        mu  = vs_all.mean(axis=-1)                                       # (K,)
        sig = vs_all.std(axis=-1)                                        # (K,)
        scores = mu - beta * sig
        best = jnp.argmax(scores)
        selected = cands[best]

        low_dist = agent.network.select("low_actor")(
            observations, selected, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)
        if not agent.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    return sample_actions


class WrappedAgent:
    """Has .sample_actions + .config so OGBench's evaluate() is happy."""
    def __init__(self, base_agent, K, beta):
        self._base = base_agent
        self.config = base_agent.config
        self.sample_actions = make_posthoc_sampler(base_agent, K, beta)


def run_eval(agent_like, env, config, num_episodes):
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    successes = []
    for t in range(1, len(task_infos) + 1):
        info, _, _ = evaluate(
            agent=agent_like, env=env, task_id=t, config=config,
            num_eval_episodes=num_episodes, num_video_episodes=0,
            video_frame_skip=1, eval_temperature=0.0,
        )
        successes.append(info.get("success", 0.0))
    return float(np.mean(successes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--betas", default="0.0,0.5,1.0,2.0")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--epoch", type=int, default=1000000)
    ap.add_argument("--dataset_dir", default=os.environ.get("DATASET_DIR"))
    args = ap.parse_args()

    betas = [float(b) for b in args.betas.split(",")]

    flags  = json.load(open(os.path.join(args.run_dir, "flags.json")))
    config = flags["agent"]
    env, train_ds_raw, _ = make_env_and_datasets(
        flags["env_name"], frame_stack=config.get("frame_stack"), dataset_dir=args.dataset_dir
    )
    train_ds = HGCDataset(Dataset.create(**train_ds_raw), config)
    example = train_ds.sample(1)
    agent = agent_registry[config["agent_name"]].create(
        flags["seed"], example["observations"], example["actions"], config
    )
    agent = restore_agent(agent, args.run_dir, args.epoch)

    # Baseline: vanilla HIQL (same checkpoint, original sample_actions).
    baseline = run_eval(agent, env, config, args.eval_episodes)
    print(f"[baseline vanilla HIQL]   overall_success = {baseline:.4f}")

    for beta in betas:
        wrapped = WrappedAgent(agent, K=args.k, beta=beta)
        s = run_eval(wrapped, env, config, args.eval_episodes)
        mark = "  <== BETTER" if s > baseline + 0.02 else ("  <== worse" if s < baseline - 0.02 else "")
        print(f"[post-hoc K={args.k} beta={beta:<4}]  overall_success = {s:.4f}{mark}")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python scripts/diag3_posthoc_chiql.py \
  --run_dir "$(cat results/antmaze-teleport-navigate-v0/hiql/seed0/source.txt)" \
  --betas 0.0,0.5,1.0,2.0 \
  --k 16 \
  --eval_episodes 50
```

For extra robustness, run on all 3 seeds (seed0/1/2) and average.

**Expected runtime per seed:** ~5 tasks × 50 episodes × 4 β values ≈ 1000 episodes ≈ 30–45 min on GPU.

**Decision rule (THIS is the paper's go/no-go):**

| Result | Interpretation | Action |
|---|---|---|
| β=0 alone gives > baseline + 0.03 | K-sample-argmax helps (pessimism may or may not) | Train 5-head C-HIQL; paper story includes both K-sampling and pessimism |
| β>0 gives > β=0 by ≥ 0.02 | Pessimism is the active ingredient | **Strong green light** — train 5-head with confidence |
| All β within ±0.02 of baseline | Mechanism doesn't help on this env | **Pivot.** Don't train C-HIQL. See §Pivot options. |
| Any β gives < baseline − 0.05 | Method actively hurts, possibly for reasons the proposal didn't anticipate | Pivot. Investigate why (maybe log σ distributions) before writing up |

The 2-head post-hoc test is a strict lower bound on what 5-head C-HIQL can achieve. If this doesn't move the needle, retraining with more heads probably won't either.

---

## Decision gate (Wednesday morning)

Collect results in a short table:

| Diagnostic | Pass threshold | Actual | Pass? |
|---|---|---|---|
| V1: π_high spread (median std) | ≥ 0.05 | ? | |
| V2: disagreement ratio (teleport/control) | ≥ 1.2 | ? | |
| V3: best β success rate vs vanilla baseline (0.404-ish) | ≥ baseline + 0.03 | ? | |

**If all three pass:** execute [PHASE2_TRAINING_PLAN.md](PHASE2_TRAINING_PLAN.md) starting at Task 1 (code changes). Submit 3 training seeds by Wednesday evening.

**If V3 passes but V2 is weak (ratio 1.0–1.2):** execute the training plan with the optional bootstrap-mask knob turned on (`--agent.bootstrap_mask_p=0.8`) to force head diversity. Note this is an *additional* code change; only do it if V3 was strong enough to justify the extra complexity.

**If V3 fails but V1/V2 pass:** the mechanism as proposed doesn't land. Two writable options:
1. **Negative-result paper** (riskier for a grade): honestly report that ensemble pessimism at the subgoal-selection layer does not improve HIQL on teleport-navigate; hypothesize why. Requires careful framing but is publishable.
2. **Different intervention on the same failure mode**: e.g., train-time conservatism rather than inference-time pessimism (modify the expectile target). More code changes, but V2's ensemble data is still useful framing.

**If V3 fails and V1/V2 fail:** the "teleport is a stochastic-transition problem solvable by uncertainty-aware planning" framing is probably wrong. Pivot to a different project-scope claim, e.g., "HIQL's hierarchy is miscalibrated for stochastic envs; [some other mechanism] fixes it." This is the worst case and it's what the diagnostic exists to catch early.

---

## Commit checkpoint at end of Phase A

```bash
git add scripts/diag1_pihigh_spread.py scripts/diag2_head_disagreement.py scripts/diag3_posthoc_chiql.py \
        PHASE2_VERIFICATION_PLAN.md
# If you saved diagnostic outputs, also add them, e.g.:
# git add results/diagnostics/*.txt
git commit -m "Phase 2 verification: diagnostics for C-HIQL mechanism"
```

This way if V3 passes, the diagnostic scripts are reproducible; if V3 fails, they are the honest methodology section of a pivoted paper.

---

## Risk register for Phase A

| Risk | Likelihood | Mitigation |
|---|---|---|
| `params_1000000.pkl` missing on scratch | Medium | V0 catches it. Fallback: one 8-h HIQL retrain, then start diagnostics. |
| `restore_agent` fails due to config key mismatch | Low | `flags.json` stores the exact config used; restore should work. If it fails, reconstruct config from `flags.json["agent"]` and try again. |
| Diagnostic 3 wrapper JIT errors from shape mismatches | Medium | Test `make_posthoc_sampler` with a single env step before the full eval loop. If shape issues, check whether `observations` has an extra batch dim (it shouldn't — OGBench's evaluate passes per-step). |
| V3 takes >2 h per seed | Low | Reduce `eval_episodes` to 25 for diagnostic purposes; full 50 is only needed for the final training-plan evaluation. |
| OGBench dataset not cached locally | Medium | `make_env_and_datasets` will download on first call (~100-500 MB for antmaze). Ensure `$DATASET_DIR` is set and has ≥ 1 GB free. |
