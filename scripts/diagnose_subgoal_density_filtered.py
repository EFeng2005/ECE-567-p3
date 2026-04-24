"""Diag 7: scoring-filtered subgoal density on the Phase-3b checkpoint.

Mirrors the Phase-1 Diag 3 analysis (subgoal_density_comparison.png) but
on EX-HIQL Phase-3b rather than HIQL, with an additional "filtered"
panel showing the top-K candidates retained after residual-σ scoring.

For a fixed (s, g) pair (task 1 by default: agent at start (36, 0),
goal at (0, 24) world coords), we:

  1. Sample N=1000 goal representations z from π_high(. | s, g).
  2. Build a nearest-neighbour bank of φ([o; o]) embeddings on a
     10,000-observation dataset subsample; for each sampled z find the
     dataset observation whose bank embedding maximises z·φ, and take
     its XY position.
  3. Compute μ̄(s, z_k, g) and σ(s, z_k, g) across the 5 value heads
     for each z_k, fit the residual-σ coefficients (α̂, γ̂), and apply
     the residual scoring rule to select the top-K (K=50) candidates.
  4. Emit a 2-panel figure: (A) raw density of all 1000 NN-decoded
     XYs; (B) density of the top-K post-filter XYs.

Teleport geometry (hard-coded, from ogbench.locomaze.maze):
  teleport-in: (20, 12), (0, 16)  |  teleport-out: (24, 0), (0, 20), (36, 20)
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

# Teleport geometry (world coordinates)
TELEPORT_IN_XYS = np.array([[20.0, 12.0], [0.0, 16.0]])
TELEPORT_OUT_XYS = np.array([[24.0, 0.0], [0.0, 20.0], [36.0, 20.0]])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--env', type=str, default='antmaze-teleport-navigate-v0')
    ap.add_argument('--dataset_dir', type=str, default=os.environ.get('OGBENCH_DATASET_DIR'))
    ap.add_argument('--task_id', type=int, default=1,
                    help='env task id (1..5); paper convention task 1 = opposite-corner')
    ap.add_argument('--n_samples', type=int, default=1000,
                    help='number of subgoal representations to sample from pi_high')
    ap.add_argument('--top_k', type=int, default=50,
                    help='top-K candidates retained after residual-σ scoring')
    ap.add_argument('--nn_bank_size', type=int, default=10000,
                    help='size of dataset subsample used for NN decode')
    ap.add_argument('--beta_pes', type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env, train_ds_dict, _ = make_env_and_datasets(args.env, dataset_dir=args.dataset_dir)
    ds_cfg = ml_collections.ConfigDict({
        **get_config().to_dict(),
        'high_alpha': 3.0, 'low_alpha': 3.0,
        'num_value_heads': 5, 'num_subgoal_candidates': 16,
        'pessimism_beta': args.beta_pes,
    })
    train_dataset = HGCDataset(Dataset.create(**train_ds_dict), ds_cfg)

    example_batch = train_dataset.sample(1)
    agent = build_agent(args.seed, example_batch, args.ckpt)

    # --- Reset env for task_id; grab the initial obs (s) and goal obs (g) ---
    reset_obs, reset_info = env.reset(options={'task_id': args.task_id, 'render_goal': True})
    s = np.asarray(reset_obs)
    if 'goal' in reset_info:
        g = np.asarray(reset_info['goal'])
    elif 'goal_rendered' in reset_info:
        g = np.asarray(reset_info['goal_rendered'])
    else:
        # Fallback: use task_infos' goal_xy with agent's current kinematic state zeroed.
        ti = env.unwrapped.task_infos[args.task_id - 1]
        g = s.copy()
        g[:2] = np.asarray(ti['goal_xy'])
    agent_xy = s[:2].copy()
    goal_xy = g[:2].copy()
    print(f"[task {args.task_id}] agent=({agent_xy[0]:.2f}, {agent_xy[1]:.2f})  "
          f"goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})", flush=True)

    # --- Build the nearest-neighbour embedding bank from the dataset ---
    print(f"Building NN bank of size {args.nn_bank_size}...", flush=True)
    bank_batch = train_dataset.sample(args.nn_bank_size)
    bank_obs = np.asarray(bank_batch['observations'])
    bank_xy = bank_obs[:, :2]

    @jax.jit
    def phi_self(o):
        # φ([o; o]): feed the observation as both obs and goal.
        return agent.network.select('goal_rep')(
            jnp.concatenate([o, o], axis=-1)
        )

    bank_emb = np.asarray(jax.vmap(phi_self)(bank_obs))
    print(f"  bank emb shape: {bank_emb.shape}", flush=True)

    # --- Sample N subgoal reps from pi_high, produce μ̄ and σ per candidate ---
    print(f"Sampling {args.n_samples} subgoals from pi_high...", flush=True)
    rng = jax.random.PRNGKey(args.seed + 7)
    high_dist = agent.network.select('high_actor')(s, g, temperature=1.0)

    @jax.jit
    def sample_and_score(rng_key):
        seeds = jax.random.split(rng_key, args.n_samples)
        reps = jax.vmap(lambda k: high_dist.sample(seed=k))(seeds)
        reps = reps / jnp.linalg.norm(reps, axis=-1, keepdims=True) * jnp.sqrt(reps.shape[-1])
        def val(z):
            return agent.network.select('value')(s, z, goal_encoded=True)  # (N_heads,)
        vs = jax.vmap(val)(reps)  # (n_samples, N_heads)
        mu = vs.mean(axis=-1)
        sig = vs.std(axis=-1)
        return reps, mu, sig

    reps, mus, sigs = sample_and_score(rng)
    reps = np.asarray(reps)     # (n_samples, rep_dim)
    mus = np.asarray(mus)       # (n_samples,)
    sigs = np.asarray(sigs)     # (n_samples,)

    # --- Fit residual-σ coefficients (α̂, γ̂) on a separate 500-state sample ---
    print("Fitting residual-σ coefficients on a 500-state sample...", flush=True)
    fit_batch = train_dataset.sample(500)
    fit_states = np.asarray(fit_batch['observations'])
    fit_goals = np.asarray(fit_batch['value_goals'])

    K = 16

    @jax.jit
    def fit_per_state(s_, g_, rng_key):
        hd = agent.network.select('high_actor')(s_, g_, temperature=1.0)
        seeds = jax.random.split(rng_key, K)
        rs = jax.vmap(lambda k: hd.sample(seed=k))(seeds)
        rs = rs / jnp.linalg.norm(rs, axis=-1, keepdims=True) * jnp.sqrt(rs.shape[-1])
        def v(z):
            return agent.network.select('value')(s_, z, goal_encoded=True)
        vs = jax.vmap(v)(rs)
        return vs.mean(axis=-1), vs.std(axis=-1)

    fit_keys = jax.random.split(jax.random.PRNGKey(args.seed + 11), 500)
    fit_mus = np.zeros((500, K))
    fit_sigs = np.zeros((500, K))
    for i in range(500):
        m, s_ = fit_per_state(fit_states[i], fit_goals[i], fit_keys[i])
        fit_mus[i] = np.asarray(m)
        fit_sigs[i] = np.asarray(s_)
    gamma, alpha = np.polyfit(fit_mus.flatten(), fit_sigs.flatten(), 1)
    print(f"  (α̂, γ̂) = ({alpha:.4f}, {gamma:.4f})", flush=True)

    # --- Apply residual scoring, select top-K ---
    sig_perp = sigs - (alpha + gamma * mus)
    scores = mus - args.beta_pes * sig_perp          # (n_samples,)
    top_k_idx = np.argsort(scores)[-args.top_k:]
    print(f"Applied residual-σ scoring; kept top-{args.top_k}/{args.n_samples} candidates "
          f"(score range [{scores[top_k_idx].min():.2f}, {scores[top_k_idx].max():.2f}])",
          flush=True)

    # --- Decode each rep via NN lookup on bank ---
    print("Decoding via NN lookup...", flush=True)
    # reps: (n_samples, rep_dim), bank_emb: (nn_bank_size, rep_dim)
    # dot product -> (n_samples, nn_bank_size)
    dots = reps @ bank_emb.T
    nn_idx = np.argmax(dots, axis=1)       # (n_samples,)
    decoded_xy = bank_xy[nn_idx]           # (n_samples, 2)

    raw_xy = decoded_xy                    # Panel A
    filtered_xy = decoded_xy[top_k_idx]    # Panel B

    # --- Quantitative metric: fraction of candidates near each teleport-out ---
    # Main pathology (per §3 Diag 3): density collapses onto the (24, 0)
    # teleport-out tile. We measure fraction of candidates within 3 world
    # units of (24, 0), and also vs (0, 20) / (36, 20).
    def frac_near(xys, point, radius=3.0):
        d = np.linalg.norm(xys - np.asarray(point), axis=1)
        return float((d <= radius).mean())

    def frac_moved_toward_goal(xys, thresh_y=10.0):
        # y in [10, 25] is meaningfully "toward the goal" (goal at y=24).
        return float((xys[:, 1] >= thresh_y).mean())

    raw_near_to_24_0 = frac_near(raw_xy, [24.0, 0.0])
    filt_near_to_24_0 = frac_near(filtered_xy, [24.0, 0.0])
    raw_toward_goal = frac_moved_toward_goal(raw_xy)
    filt_toward_goal = frac_moved_toward_goal(filtered_xy)

    # --- Save raw data for reproducibility ---
    summary = {
        'task_id': args.task_id,
        'agent_xy': agent_xy.tolist(),
        'goal_xy': goal_xy.tolist(),
        'n_samples': args.n_samples,
        'top_k': args.top_k,
        'nn_bank_size': args.nn_bank_size,
        'beta_pes': args.beta_pes,
        'residual_fit': {'alpha': float(alpha), 'gamma': float(gamma)},
        'raw_xy_mean': raw_xy.mean(axis=0).tolist(),
        'raw_xy_std': raw_xy.std(axis=0).tolist(),
        'filtered_xy_mean': filtered_xy.mean(axis=0).tolist(),
        'filtered_xy_std': filtered_xy.std(axis=0).tolist(),
        'score_range': [float(scores.min()), float(scores.max())],
        'filtered_score_range': [float(scores[top_k_idx].min()), float(scores[top_k_idx].max())],
        'frac_near_bad_teleport_out': {
            'raw': raw_near_to_24_0,
            'filtered': filt_near_to_24_0,
            'reduction': raw_near_to_24_0 - filt_near_to_24_0,
        },
        'frac_toward_goal_y_ge_10': {
            'raw': raw_toward_goal,
            'filtered': filt_toward_goal,
            'increase': filt_toward_goal - raw_toward_goal,
        },
    }
    with open(out / 'subgoal_density_filtered_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    np.savez(out / 'subgoal_density_filtered_raw.npz',
             raw_xy=raw_xy, filtered_xy=filtered_xy,
             mus=mus, sigs=sigs, scores=scores,
             alpha=alpha, gamma=gamma)

    # --- Make 2-panel plot with hexbin density + scatter + metric annotations ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    panel_configs = [
        (axes[0], raw_xy,
         f'Panel A: raw $\\pi_\\text{{high}}$ density  ($N={args.n_samples}$)',
         f'frac within 3u of (24,0): $\\mathbf{{{raw_near_to_24_0:.2f}}}$  |  '
         f'frac $y{{\\geq}}10$: {raw_toward_goal:.2f}'),
        (axes[1], filtered_xy,
         f'Panel B: residual-$\\sigma$ filtered  (top-{args.top_k})',
         f'frac within 3u of (24,0): $\\mathbf{{{filt_near_to_24_0:.2f}}}$  |  '
         f'frac $y{{\\geq}}10$: {filt_toward_goal:.2f}'),
    ]
    for ax, xy, title, metric_line in panel_configs:
        # hexbin density background
        hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=26,
                       extent=(-6, 42, -6, 26), cmap='Blues',
                       mincnt=1, alpha=0.85)
        # scatter overlay for individual points
        ax.scatter(xy[:, 0], xy[:, 1], s=14, alpha=0.35, c='C0',
                   edgecolors='none', zorder=5)
        # agent + goal markers
        ax.scatter(agent_xy[0], agent_xy[1], marker='o', s=260, facecolors='none',
                   edgecolors='blue', linewidths=2.2, zorder=10, label='agent')
        ax.scatter(goal_xy[0], goal_xy[1], marker='*', s=400, color='gold',
                   edgecolors='black', linewidths=1.8, zorder=10, label='goal')
        # teleport markers
        for tx, ty in TELEPORT_IN_XYS:
            ax.scatter(tx, ty, marker='X', s=240, color='red',
                       edgecolors='black', linewidths=1.8, zorder=9,
                       label='teleport-in' if (tx == TELEPORT_IN_XYS[0][0] and ty == TELEPORT_IN_XYS[0][1]) else None)
        for tx, ty in TELEPORT_OUT_XYS:
            ax.scatter(tx, ty, marker='s', s=180, color='orange',
                       edgecolors='black', linewidths=1.2, zorder=9,
                       label='teleport-out' if (tx == TELEPORT_OUT_XYS[0][0] and ty == TELEPORT_OUT_XYS[0][1]) else None)
        ax.set_xlabel('agent x')
        ax.set_ylabel('agent y')
        ax.set_title(title + '\n' + metric_line, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-6, 42)
        ax.set_ylim(-6, 26)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc='upper right', fontsize=8)

    fig.suptitle(
        f'EX-HIQL Phase-3b (seed {args.seed}, task {args.task_id}): '
        f'$\\pi_\\text{{high}}$ samples before vs after residual-$\\sigma$ scoring',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out / 'subgoal_density_filtered.png', dpi=110)
    print(f"Saved {out}/subgoal_density_filtered.png", flush=True)


if __name__ == '__main__':
    main()
