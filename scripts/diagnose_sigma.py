"""Diagnostic: how informative is σ (ensemble disagreement) for C-HIQL's
subgoal scoring?

Tests three things on a single trained checkpoint:
  (a) DISTRIBUTION of σ across (state, candidate) pairs. If σ is narrow,
      disagreement carries almost no signal.
  (b) SPATIAL LOCALISATION: map mean σ per state onto (x, y). If σ
      concentrates near specific regions (ideally teleporters), the
      "disagreement = stochastic-dynamics signal" assumption holds.
  (c) ARGMAX OVERLAP: fraction of states where argmax(μ) == argmax(μ−βσ)
      for β ∈ {0.25, 0.5, 1, 2}. Tells us if β is actually moving the
      pick — and paired with success-rate evidence, whether it's moving
      it in the right direction.

Writes sigma_stats.json and sigma_diagnostic.png in --out_dir.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Ensure the OGBench impls/ dir is on sys.path so `agents.*` and `utils.*`
# import correctly regardless of which directory we're run from.
_DEFAULT_IMPLS = "/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567/external/ogbench_full/impls"
_IMPLS = os.environ.get("OGBENCH_IMPLS_DIR", _DEFAULT_IMPLS)
if _IMPLS and _IMPLS not in sys.path:
    sys.path.insert(0, _IMPLS)

import flax
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ml_collections
import numpy as np

from agents.chiql import CHIQLAgent, get_config
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets


def build_agent(seed, example_batch, ckpt_path, beta=0.5):
    cfg = get_config().to_dict()
    cfg['high_alpha'] = 3.0
    cfg['low_alpha'] = 3.0
    cfg['num_value_heads'] = 5
    cfg['num_subgoal_candidates'] = 16
    cfg['pessimism_beta'] = float(beta)
    agent = CHIQLAgent.create(
        seed=seed,
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=cfg,
    )
    with open(ckpt_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--env', type=str, default='antmaze-teleport-navigate-v0')
    ap.add_argument('--dataset_dir', type=str, default=os.environ.get('OGBENCH_DATASET_DIR'))
    ap.add_argument('--n_states', type=int, default=500)
    ap.add_argument('--K', type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env, train_ds_dict, _ = make_env_and_datasets(args.env, dataset_dir=args.dataset_dir)
    ds_cfg = ml_collections.ConfigDict({
        **get_config().to_dict(),
        'high_alpha': 3.0, 'low_alpha': 3.0,
        'num_value_heads': 5, 'num_subgoal_candidates': 16,
        'pessimism_beta': 0.5,
    })
    train_dataset = HGCDataset(Dataset.create(**train_ds_dict), ds_cfg)

    example_batch = train_dataset.sample(1)
    agent = build_agent(args.seed, example_batch, args.ckpt, beta=0.5)

    # Sample states + paired value-goals.
    batch = train_dataset.sample(args.n_states)
    states = np.asarray(batch['observations'])
    goals = np.asarray(batch['value_goals'])

    K = args.K

    @jax.jit
    def per_state(s, g, rng_key):
        high_dist = agent.network.select('high_actor')(s, g, temperature=1.0)
        cand_seeds = jax.random.split(rng_key, K)
        goal_reps = jax.vmap(lambda sd: high_dist.sample(seed=sd))(cand_seeds)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])
        def score_one(gr):
            return agent.network.select('value')(s, gr, goal_encoded=True)  # (N_heads,)
        vs_all = jax.vmap(score_one)(goal_reps)  # (K, N)
        mu = vs_all.mean(axis=-1)
        sig = vs_all.std(axis=-1)
        return mu, sig

    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.n_states)
    mus = np.zeros((args.n_states, K))
    sigs = np.zeros((args.n_states, K))
    xy = states[:, :2]  # antmaze obs has (x, y) as the first 2 dims

    print(f"Diagnosing {args.n_states} states × K={K} candidates...")
    for i in range(args.n_states):
        mu, sig = per_state(states[i], goals[i], keys[i])
        mus[i] = np.asarray(mu)
        sigs[i] = np.asarray(sig)
        if i % 100 == 0:
            print(f"  {i}/{args.n_states}", flush=True)

    # Stats
    sig_flat = sigs.flatten()
    mu_flat = mus.flatten()

    idx_beta0 = mus.argmax(axis=1)
    overlaps = {}
    for beta in [0.25, 0.5, 1.0, 2.0]:
        idx_beta = (mus - beta * sigs).argmax(axis=1)
        overlaps[f'overlap_beta0_vs_beta{beta}'] = float((idx_beta0 == idx_beta).mean())

    per_state_corr = []
    for i in range(args.n_states):
        if mus[i].std() > 1e-8 and sigs[i].std() > 1e-8:
            per_state_corr.append(float(np.corrcoef(mus[i], sigs[i])[0, 1]))

    sig_per_state = sigs.mean(axis=1)

    stats = {
        'n_states': args.n_states,
        'K': K,
        'ckpt': args.ckpt,
        'sigma_mean': float(sig_flat.mean()),
        'sigma_std': float(sig_flat.std()),
        'sigma_p10': float(np.quantile(sig_flat, 0.10)),
        'sigma_p50': float(np.quantile(sig_flat, 0.50)),
        'sigma_p90': float(np.quantile(sig_flat, 0.90)),
        'mu_mean': float(mu_flat.mean()),
        'mu_std': float(mu_flat.std()),
        'per_state_mu_sig_corr_mean': float(np.mean(per_state_corr)) if per_state_corr else None,
        'per_state_mu_sig_corr_std': float(np.std(per_state_corr)) if per_state_corr else None,
        **overlaps,
    }
    with open(out / 'sigma_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=sig_per_state, cmap='viridis', s=12)
    axes[0].set_xlabel('agent x'); axes[0].set_ylabel('agent y')
    axes[0].set_title(f'mean σ per state, seed {args.seed}')
    plt.colorbar(sc, ax=axes[0], label='mean σ over K candidates')

    axes[1].hist(sig_flat, bins=60)
    axes[1].set_xlabel('σ (5-head disagreement)')
    axes[1].set_ylabel('count')
    axes[1].set_title(f'σ distribution ({args.n_states}×{K} points)')

    scatter = axes[2].scatter(mu_flat, sig_flat, s=2, alpha=0.2)
    axes[2].set_xlabel('μ')
    axes[2].set_ylabel('σ')
    axes[2].set_title('μ vs σ across all (state, candidate)')

    fig.tight_layout()
    fig.savefig(out / 'sigma_diagnostic.png', dpi=110)
    print(f"Saved {out}/sigma_diagnostic.png")


if __name__ == '__main__':
    main()
