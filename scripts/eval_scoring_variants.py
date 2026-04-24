"""Level-1 scoring-rule variants evaluation (no retraining).

Tests whether an existing EX-HIQL checkpoint's (μ, σ) signal can be
rescued by changing the inference-time scoring rule.

Four variants:
  - plain:      score = μ − β·σ                     (baseline)
  - rank:       score = rank(μ)/K − β·rank(σ)/K     (state-relative)
  - normalized: score = μ − β·σ / (|μ|+1)           (scale-free)
  - residual:   score = μ − β·(σ − (α + γ·μ))       (regress out μ-dep)

For the residual variant, α/γ are fit from a sample of 500 states ×
K candidates taken from the dataset at script start.

β values are chosen per variant to span the "barely affects argmax"
→ "dominates scoring" range after accounting for the different units.

Writes one CSV per (seed, variant, β) to --out_csv.
"""
import argparse
import csv
import os
import pathlib
import pickle
import sys

_DEFAULT_IMPLS = "/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567/external/ogbench_full/impls"
_IMPLS = os.environ.get("OGBENCH_IMPLS_DIR", _DEFAULT_IMPLS)
if _IMPLS and _IMPLS not in sys.path:
    sys.path.insert(0, _IMPLS)

import dataclasses

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tqdm

from agents.ex_chiql import EXCHIQLAgent, get_config
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate

EVAL_EPISODES = 50

# Scale-adjusted β schedules so each variant spans "mild" → "dominant".
# Adjusted for typical |μ| ≈ 23, σ ≈ 4 observations from the diagnostic.
BETAS = {
    "plain":      [0.0, 0.5, 2.0],          # σ is ~4, |μ| is ~23 → β·σ/|μ| ≈ 9% / 35%
    "rank":       [0.0, 0.5, 2.0],          # ranks ∈ [0, 1] so β moves argmax by up to β ranks
    "normalized": [0.0, 8.0, 32.0],         # σ/|μ| is ~0.15 so β needs to be ~|μ| larger
    "residual":   [0.0, 0.5, 2.0],          # residual σ has ~same scale as σ itself
}


def build_agent(seed, example_batch, ckpt_path, beta=0.5):
    cfg = get_config().to_dict()
    cfg['high_alpha'] = 3.0
    cfg['low_alpha'] = 3.0
    cfg['num_value_heads'] = 5
    cfg['num_subgoal_candidates'] = 16
    cfg['pessimism_beta'] = float(beta)
    agent = EXCHIQLAgent.create(
        seed=seed,
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=cfg,
    )
    with open(ckpt_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    return agent


def fit_residual_coefs(agent, train_dataset, n_states=500, K=16, rng_seed=0):
    """Fit σ ≈ α + γ·μ on the dataset. Returns (α, γ)."""
    batch = train_dataset.sample(n_states)
    states = np.asarray(batch['observations'])
    goals = np.asarray(batch['value_goals'])

    @jax.jit
    def per_state_mu_sig(s, g, rng_key):
        high_dist = agent.network.select('high_actor')(s, g, temperature=1.0)
        cand_seeds = jax.random.split(rng_key, K)
        goal_reps = jax.vmap(lambda sd: high_dist.sample(seed=sd))(cand_seeds)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])
        def score_one(gr):
            return agent.network.select('value')(s, gr, goal_encoded=True)
        vs = jax.vmap(score_one)(goal_reps)
        return vs.mean(axis=-1), vs.std(axis=-1)

    keys = jax.random.split(jax.random.PRNGKey(rng_seed), n_states)
    mus, sigs = [], []
    for i in range(n_states):
        m, s = per_state_mu_sig(states[i], goals[i], keys[i])
        mus.append(np.asarray(m))
        sigs.append(np.asarray(s))
    mus = np.concatenate(mus)
    sigs = np.concatenate(sigs)
    # σ = γ·μ + α
    gamma, alpha = np.polyfit(mus, sigs, 1)
    print(f"[residual fit] σ ≈ {alpha:.3f} + {gamma:.4f}·μ", flush=True)
    return float(alpha), float(gamma)


def build_variant_agent(base_agent, variant, beta, residual_coefs=None):
    """Return an object with a .sample_actions method using the chosen scoring rule.

    Shim-wraps base_agent so the rest of EXCHIQLAgent (actor nets, value net)
    is unchanged; only the subgoal scoring step changes.
    """
    K = base_agent.config['num_subgoal_candidates']
    alpha, gamma = residual_coefs or (0.0, 0.0)

    @jax.jit
    def sample_actions(observations, goals=None, seed=None, temperature=1.0):
        high_seed, low_seed = jax.random.split(seed)
        high_dist = base_agent.network.select('high_actor')(observations, goals, temperature=temperature)
        cand_seeds = jax.random.split(high_seed, K)
        goal_reps = jax.vmap(lambda s: high_dist.sample(seed=s))(cand_seeds)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        def score_one(gr):
            return base_agent.network.select('value')(observations, gr, goal_encoded=True)

        vs_all = jax.vmap(score_one)(goal_reps)     # (K, N)
        mu = vs_all.mean(axis=-1)                    # (K,)
        sig = vs_all.std(axis=-1)                    # (K,)

        if variant == "plain":
            scores = mu - beta * sig
        elif variant == "rank":
            # argsort of argsort gives ranks in [0, K-1]; normalize to [0, 1]
            mu_rank = jnp.argsort(jnp.argsort(mu)).astype(jnp.float32) / jnp.float32(K - 1)
            sig_rank = jnp.argsort(jnp.argsort(sig)).astype(jnp.float32) / jnp.float32(K - 1)
            scores = mu_rank - beta * sig_rank
        elif variant == "normalized":
            scores = mu - beta * sig / (jnp.abs(mu) + 1.0)
        elif variant == "residual":
            sig_residual = sig - (alpha + gamma * mu)
            scores = mu - beta * sig_residual
        else:
            raise ValueError(f"unknown variant {variant}")

        best_idx = jnp.argmax(scores)
        selected_goal_rep = goal_reps[best_idx]
        low_dist = base_agent.network.select('low_actor')(
            observations, selected_goal_rep, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)
        if not base_agent.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    class ShimAgent:
        pass

    shim = ShimAgent()
    shim.sample_actions = sample_actions
    shim.config = base_agent.config
    return shim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--env', type=str, default='antmaze-teleport-navigate-v0')
    ap.add_argument('--dataset_dir', type=str, default=os.environ.get('OGBENCH_DATASET_DIR'))
    args = ap.parse_args()

    env, train_ds_dict, _ = make_env_and_datasets(args.env, dataset_dir=args.dataset_dir)
    base_cfg = get_config().to_dict()
    base_cfg['high_alpha'] = 3.0
    base_cfg['low_alpha'] = 3.0
    base_cfg['num_value_heads'] = 5
    base_cfg['num_subgoal_candidates'] = 16

    ds_cfg = ml_collections.ConfigDict(base_cfg)
    train_dataset = HGCDataset(Dataset.create(**train_ds_dict), ds_cfg)
    example_batch = train_dataset.sample(1)

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = len(task_infos)

    # Build a base agent with beta=0 (inference β is overridden in the scoring variants)
    base_agent = build_agent(args.seed, example_batch, args.ckpt, beta=0.0)
    base_agent = jax.device_put(base_agent, device=jax.devices('cpu')[0])

    # Fit residual coefficients once per checkpoint
    residual_coefs = fit_residual_coefs(base_agent, train_dataset, n_states=500, K=16, rng_seed=args.seed)

    rows = []
    for variant in ("plain", "rank", "normalized", "residual"):
        for beta in BETAS[variant]:
            shim = build_variant_agent(base_agent, variant, float(beta), residual_coefs)

            per_task = []
            for task_id in tqdm.trange(1, num_tasks + 1, desc=f'seed{args.seed} {variant} β={beta}'):
                ev_info, _, _ = evaluate(
                    agent=shim,
                    env=env,
                    task_id=task_id,
                    config=base_cfg,  # config used for env/eval, not scoring
                    num_eval_episodes=EVAL_EPISODES,
                    num_video_episodes=0,
                    video_frame_skip=3,
                    eval_temperature=0.0,
                    eval_gaussian=None,
                )
                per_task.append(float(ev_info['success']))
            overall = float(np.mean(per_task))
            row = {'seed': args.seed, 'variant': variant, 'beta': beta, 'overall_success': overall}
            for i, v in enumerate(per_task):
                row[f'task{i+1}_success'] = v
            rows.append(row)
            print(f"seed={args.seed} {variant} β={beta} overall={overall:.4f}", flush=True)

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out_csv}", flush=True)


if __name__ == '__main__':
    main()
