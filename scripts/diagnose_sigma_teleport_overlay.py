"""σ diagnostic with teleporter-zone overlay.

Extends diagnose_sigma_exhiql.py by:
  1. Hard-coding the `antmaze-teleport-navigate-v0` teleport-in/out zones.
  2. Classifying each of the N=500 sampled states by distance to the nearest
     teleport-in zone: "near" (within NEAR_RADIUS), "boundary" (within
     FAR_RADIUS), "far" (beyond).
  3. Reporting mean/median σ per category — the decisive test of whether
     σ concentrates at teleporter-adjacent states.
  4. Saving a 2×2 plot:
     - (a) σ spatial map + teleport-in zones (red) + teleport-out zones (orange)
     - (b) σ histogram colored by category
     - (c) μ vs σ scatter colored by category
     - (d) distance-to-teleport vs σ scatter

Teleport zone info from ogbench.locomaze.maze.py (maze_unit=4.0, offset=4):
  teleport_in_xys = [(20, 12), (0, 16)]
  teleport_out_xys = [(24, 0), (0, 20), (36, 20)]
  trigger radius = 1.5 world units
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

# Teleport geometry (world-space (x, y) coords); from ogbench.locomaze.maze.py
TELEPORT_IN_XYS = np.array([[20.0, 12.0], [0.0, 16.0]])
TELEPORT_OUT_XYS = np.array([[24.0, 0.0], [0.0, 20.0], [36.0, 20.0]])
TRIGGER_RADIUS = 1.5         # actual teleport trigger in the env
NEAR_RADIUS = 3.0            # classification: "near" (inside or on fringe)
FAR_RADIUS = 6.0             # classification: "boundary" out to here


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
    """Euclidean distance from xy to the nearest teleport-in zone."""
    diffs = xy[None, :] - TELEPORT_IN_XYS  # (2, 2)
    return float(np.linalg.norm(diffs, axis=1).min())


def classify(dist):
    if dist <= NEAR_RADIUS:
        return "near"
    elif dist <= FAR_RADIUS:
        return "boundary"
    else:
        return "far"


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

    sig_per_state = sigs.mean(axis=1)  # (N,)
    mu_per_state = mus.mean(axis=1)    # (N,)
    dists = np.array([min_dist_to_teleport_in(xy[i]) for i in range(args.n_states)])
    categories = np.array([classify(d) for d in dists])

    stats_by_cat = {}
    for cat in ("near", "boundary", "far"):
        mask = categories == cat
        if mask.sum() == 0:
            continue
        stats_by_cat[cat] = {
            'n_states': int(mask.sum()),
            'sigma_mean': float(sig_per_state[mask].mean()),
            'sigma_median': float(np.median(sig_per_state[mask])),
            'sigma_p90': float(np.quantile(sig_per_state[mask], 0.90)),
            'mu_mean': float(mu_per_state[mask].mean()),
            'dist_mean': float(dists[mask].mean()),
        }

    near_sig = sig_per_state[categories == 'near']
    far_sig = sig_per_state[categories == 'far']
    if len(near_sig) > 0 and len(far_sig) > 0:
        ratio_mean = float(near_sig.mean() / far_sig.mean())
        ratio_median = float(np.median(near_sig) / max(np.median(far_sig), 1e-6))
    else:
        ratio_mean = ratio_median = None

    # Correlation of per-state σ with distance to nearest teleport-in
    dist_sig_corr = float(np.corrcoef(dists, sig_per_state)[0, 1]) if len(dists) > 1 else None

    summary = {
        'n_states': args.n_states,
        'K': K,
        'ckpt': args.ckpt,
        'teleport_in_xys': TELEPORT_IN_XYS.tolist(),
        'teleport_out_xys': TELEPORT_OUT_XYS.tolist(),
        'NEAR_RADIUS': NEAR_RADIUS,
        'FAR_RADIUS': FAR_RADIUS,
        'overall': {
            'sigma_mean': float(sig_per_state.mean()),
            'sigma_median': float(np.median(sig_per_state)),
            'mu_mean': float(mu_per_state.mean()),
            'dist_to_teleport_mean': float(dists.mean()),
        },
        'by_category': stats_by_cat,
        'near_vs_far_ratio_mean': ratio_mean,
        'near_vs_far_ratio_median': ratio_median,
        'dist_to_teleport_vs_sigma_corr': dist_sig_corr,
    }
    with open(out / 'sigma_teleport_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cat_colors = {'near': 'red', 'boundary': 'orange', 'far': 'C0'}

    # (a) spatial + teleporter overlay
    ax = axes[0, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=sig_per_state, cmap='viridis', s=10, alpha=0.8)
    for tx, ty in TELEPORT_IN_XYS:
        ax.scatter(tx, ty, c='red', marker='X', s=280, edgecolors='black', linewidths=1.5, zorder=10, label='teleport_in' if (tx, ty) == tuple(TELEPORT_IN_XYS[0]) else None)
        circ = plt.Circle((tx, ty), TRIGGER_RADIUS, fill=False, edgecolor='red', linewidth=1.5, zorder=9)
        ax.add_patch(circ)
        circ2 = plt.Circle((tx, ty), NEAR_RADIUS, fill=False, edgecolor='red', linewidth=0.8, linestyle='--', zorder=8)
        ax.add_patch(circ2)
    for tx, ty in TELEPORT_OUT_XYS:
        ax.scatter(tx, ty, c='orange', marker='s', s=180, edgecolors='black', linewidths=1.0, zorder=10, label='teleport_out' if (tx, ty) == tuple(TELEPORT_OUT_XYS[0]) else None)
    ax.set_xlabel('agent x')
    ax.set_ylabel('agent y')
    ax.set_title(f'σ spatial + teleporter zones (seed {args.seed})')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='mean σ over K cand')
    ax.legend(loc='upper right', fontsize=8)

    # (b) σ distribution by category
    ax = axes[0, 1]
    for cat in ("near", "boundary", "far"):
        mask = categories == cat
        if mask.sum() > 0:
            ax.hist(sig_per_state[mask], bins=30, alpha=0.6, label=f'{cat} (n={mask.sum()})', color=cat_colors[cat])
    ax.set_xlabel('per-state mean σ')
    ax.set_ylabel('count')
    ax.set_title('σ distribution by teleporter distance class')
    ax.legend()

    # (c) μ vs σ colored by category
    ax = axes[1, 0]
    for cat in ("far", "boundary", "near"):  # draw "near" on top
        mask = categories == cat
        if mask.sum() > 0:
            ax.scatter(mu_per_state[mask], sig_per_state[mask], s=20, alpha=0.6, c=cat_colors[cat], label=f'{cat}')
    ax.set_xlabel('per-state mean μ')
    ax.set_ylabel('per-state mean σ')
    ax.set_title('μ vs σ, colored by teleporter distance class')
    ax.legend()

    # (d) distance-to-teleport vs σ
    ax = axes[1, 1]
    ax.scatter(dists, sig_per_state, s=20, alpha=0.5)
    ax.axvline(NEAR_RADIUS, color='red', linestyle='--', label=f'NEAR_RADIUS={NEAR_RADIUS}')
    ax.axvline(FAR_RADIUS, color='orange', linestyle='--', label=f'FAR_RADIUS={FAR_RADIUS}')
    ax.set_xlabel('min distance to teleport-in zone')
    ax.set_ylabel('per-state mean σ')
    ax.set_title(f'σ vs teleport distance  (corr = {dist_sig_corr:.3f})' if dist_sig_corr else 'σ vs teleport distance')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / 'sigma_teleport_overlay.png', dpi=110)
    print(f"Saved {out}/sigma_teleport_overlay.png", flush=True)


if __name__ == '__main__':
    main()
