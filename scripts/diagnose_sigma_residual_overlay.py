"""Diag 6: residual σ⊥ teleport-overlay diagnostic.

Extends diagnose_sigma_teleport_overlay.py with the residual-σ
decomposition: at each (state, candidate) pair we compute
    σ⊥ = σ - (α̂ + γ̂·μ)
using linear-regression coefficients (α̂, γ̂) fit on (μ, σ) pairs flattened
over all 2000 states × 16 candidates = 32k points.

We then re-run the teleport-overlay geometry with σ⊥ replacing σ:
  - near/far median ratio of σ⊥ (expected > σ's 1.66×)
  - Pearson correlation between dist-to-teleport and σ⊥ (expected more
    negative than σ's -0.36)

Writes sigma_residual_stats.json and sigma_residual_overlay.png alongside
the existing Diag-5 outputs.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

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

from agents.ex_chiql import EXCHIQLAgent, get_config
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets

# Teleport geometry — see ogbench.locomaze.maze (maze_unit=4.0, offset=4)
TELEPORT_IN_XYS = np.array([[20.0, 12.0], [0.0, 16.0]])
TELEPORT_OUT_XYS = np.array([[24.0, 0.0], [0.0, 20.0], [36.0, 20.0]])
TRIGGER_RADIUS = 1.5
NEAR_RADIUS = 3.0
FAR_RADIUS = 6.0


def build_agent(seed, example_batch, ckpt_path):
    cfg = get_config().to_dict()
    cfg['high_alpha'] = 3.0
    cfg['low_alpha'] = 3.0
    cfg['num_value_heads'] = 5
    cfg['num_subgoal_candidates'] = 16
    cfg['pessimism_beta'] = 0.5
    agent = EXCHIQLAgent.create(
        seed=seed, ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'], config=cfg,
    )
    with open(ckpt_path, 'rb') as f:
        load_dict = pickle.load(f)
    return flax.serialization.from_state_dict(agent, load_dict['agent'])


def min_dist_to_teleport_in(xy):
    return float(np.linalg.norm(xy[None, :] - TELEPORT_IN_XYS, axis=1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--env', type=str, default='antmaze-teleport-navigate-v0')
    ap.add_argument('--dataset_dir', type=str, default=os.environ.get('OGBENCH_DATASET_DIR'))
    ap.add_argument('--n_states', type=int, default=2000)
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
    agent = build_agent(args.seed, example_batch, args.ckpt)

    batch = train_dataset.sample(args.n_states)
    states = np.asarray(batch['observations'])
    goals = np.asarray(batch['value_goals'])
    xy = states[:, :2]
    K = args.K

    @jax.jit
    def per_state(s, g, rng_key):
        high_dist = agent.network.select('high_actor')(s, g, temperature=1.0)
        cand_seeds = jax.random.split(rng_key, K)
        goal_reps = jax.vmap(lambda sd: high_dist.sample(seed=sd))(cand_seeds)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])
        def score_one(gr):
            return agent.network.select('value')(s, gr, goal_encoded=True)
        vs = jax.vmap(score_one)(goal_reps)
        return vs.mean(axis=-1), vs.std(axis=-1)

    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.n_states)
    mus = np.zeros((args.n_states, K))
    sigs = np.zeros((args.n_states, K))
    print(f"Diagnosing {args.n_states} states x K={K} candidates...", flush=True)
    for i in range(args.n_states):
        m, s = per_state(states[i], goals[i], keys[i])
        mus[i] = np.asarray(m)
        sigs[i] = np.asarray(s)
        if i % 200 == 0:
            print(f"  {i}/{args.n_states}", flush=True)

    # Fit linear regression sigma ~ gamma*mu + alpha on flattened arrays.
    mus_flat = mus.flatten()
    sigs_flat = sigs.flatten()
    gamma, alpha = np.polyfit(mus_flat, sigs_flat, 1)
    print(f"[residual fit] sigma ~ {alpha:.3f} + {gamma:.4f} * mu", flush=True)

    # Compute residual per-candidate, then average per-state.
    sigs_residual = sigs - (alpha + gamma * mus)       # (N_states, K)
    sig_per_state = sigs.mean(axis=1)                  # raw sigma
    sig_perp_per_state = sigs_residual.mean(axis=1)    # residual

    dists = np.array([min_dist_to_teleport_in(xy[i]) for i in range(args.n_states)])
    near_mask = dists <= NEAR_RADIUS
    far_mask = dists > FAR_RADIUS

    def cat_stats(values, mask):
        if mask.sum() == 0:
            return None
        return {
            'n': int(mask.sum()),
            'mean': float(values[mask].mean()),
            'median': float(np.median(values[mask])),
            'p90': float(np.quantile(values[mask], 0.90)),
        }

    sigma_near = cat_stats(sig_per_state, near_mask)
    sigma_far = cat_stats(sig_per_state, far_mask)
    perp_near = cat_stats(sig_perp_per_state, near_mask)
    perp_far = cat_stats(sig_perp_per_state, far_mask)

    # Ratios and correlations
    def safe_ratio(n, f, key):
        return n[key] / f[key] if (n and f and f[key] != 0) else None

    def safe_corr(x, y):
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            return float(np.corrcoef(x, y)[0, 1])
        return None

    summary = {
        'n_states': args.n_states,
        'K': K,
        'ckpt': args.ckpt,
        'residual_fit': {'alpha': float(alpha), 'gamma': float(gamma)},
        'sigma': {
            'near': sigma_near, 'far': sigma_far,
            'ratio_mean': safe_ratio(sigma_near, sigma_far, 'mean'),
            'ratio_median': safe_ratio(sigma_near, sigma_far, 'median'),
            'dist_sigma_corr': safe_corr(dists, sig_per_state),
        },
        'sigma_perp': {
            'near': perp_near, 'far': perp_far,
            'ratio_mean': safe_ratio(perp_near, perp_far, 'mean'),
            'ratio_median': safe_ratio(perp_near, perp_far, 'median'),
            'dist_sigma_corr': safe_corr(dists, sig_perp_per_state),
        },
    }
    with open(out / 'sigma_residual_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    # --- Plot: 2x2 side-by-side ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) sigma spatial + teleporter overlay
    ax = axes[0, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=sig_per_state, cmap='viridis', s=10, alpha=0.8)
    for tx, ty in TELEPORT_IN_XYS:
        ax.scatter(tx, ty, c='red', marker='X', s=280, edgecolors='black', linewidths=1.5, zorder=10)
        circ = plt.Circle((tx, ty), TRIGGER_RADIUS, fill=False, edgecolor='red', linewidth=1.5, zorder=9)
        ax.add_patch(circ)
    for tx, ty in TELEPORT_OUT_XYS:
        ax.scatter(tx, ty, c='orange', marker='s', s=180, edgecolors='black', linewidths=1.0, zorder=10)
    ax.set_xlabel('agent x'); ax.set_ylabel('agent y')
    ax.set_title(f'raw $\\sigma$ per state (seed {args.seed})')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label=r'mean $\sigma$')

    # (b) sigma_perp spatial + teleporter overlay
    ax = axes[0, 1]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=sig_perp_per_state, cmap='viridis', s=10, alpha=0.8)
    for tx, ty in TELEPORT_IN_XYS:
        ax.scatter(tx, ty, c='red', marker='X', s=280, edgecolors='black', linewidths=1.5, zorder=10)
        circ = plt.Circle((tx, ty), TRIGGER_RADIUS, fill=False, edgecolor='red', linewidth=1.5, zorder=9)
        ax.add_patch(circ)
    for tx, ty in TELEPORT_OUT_XYS:
        ax.scatter(tx, ty, c='orange', marker='s', s=180, edgecolors='black', linewidths=1.0, zorder=10)
    ax.set_xlabel('agent x'); ax.set_ylabel('agent y')
    ax.set_title(f'residual $\\sigma^\\perp$ per state (seed {args.seed})')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label=r'mean $\sigma^\perp$')

    # (c) sigma vs distance
    ax = axes[1, 0]
    ax.scatter(dists, sig_per_state, s=14, alpha=0.45)
    ax.axvline(NEAR_RADIUS, color='red', linestyle='--', label=f'NEAR={NEAR_RADIUS}')
    ax.axvline(FAR_RADIUS, color='orange', linestyle='--', label=f'FAR={FAR_RADIUS}')
    c_raw = summary['sigma']['dist_sigma_corr']
    ax.set_xlabel('min distance to teleport-in zone'); ax.set_ylabel(r'per-state mean $\sigma$')
    ax.set_title(f'raw $\\sigma$ vs distance  (corr = {c_raw:.3f})' if c_raw is not None else 'raw sigma vs distance')
    ax.legend()

    # (d) sigma_perp vs distance
    ax = axes[1, 1]
    ax.scatter(dists, sig_perp_per_state, s=14, alpha=0.45, color='C2')
    ax.axvline(NEAR_RADIUS, color='red', linestyle='--', label=f'NEAR={NEAR_RADIUS}')
    ax.axvline(FAR_RADIUS, color='orange', linestyle='--', label=f'FAR={FAR_RADIUS}')
    c_res = summary['sigma_perp']['dist_sigma_corr']
    ax.set_xlabel('min distance to teleport-in zone'); ax.set_ylabel(r'per-state mean $\sigma^\perp$')
    ax.set_title(f'residual $\\sigma^\\perp$ vs distance  (corr = {c_res:.3f})' if c_res is not None else 'sigma_perp vs distance')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / 'sigma_residual_overlay.png', dpi=110)
    print(f"Saved {out}/sigma_residual_overlay.png", flush=True)


if __name__ == '__main__':
    main()
