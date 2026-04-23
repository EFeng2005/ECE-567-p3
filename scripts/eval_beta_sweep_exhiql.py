"""β sweep for an EX-HIQL checkpoint. Mirror of eval_beta_sweep.py but for
EXCHIQLAgent — works with Phase-3a/3b (indep trunks) and shared-trunk
EX-HIQL (shared trunk) checkpoints alike, since the agent class is the
same and SharedTrunkGCValue is defined module-locally in ex_chiql.py on
the shared-trunk branches.

Loads params_1000000.pkl, rebuilds the agent with a new pessimism_beta,
runs eval_episodes*num_tasks eval, writes per-seed CSV.

BETAS sweep widened to include β=4.0 since shared-trunk runs have
σ ≈ 4 at step 1M and need larger β to see pessimism effect.
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

import flax
import jax
import numpy as np
import tqdm

from agents.ex_chiql import EXCHIQLAgent, get_config
from utils.datasets import Dataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate

BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
EVAL_EPISODES = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--env', type=str, default='antmaze-teleport-navigate-v0')
    ap.add_argument('--dataset_dir', type=str, default=os.environ.get('OGBENCH_DATASET_DIR'))
    args = ap.parse_args()

    env, train_ds_dict, _ = make_env_and_datasets(args.env, dataset_dir=args.dataset_dir)

    base_cfg = get_config().to_dict()
    base_cfg['high_alpha'] = 3.0
    base_cfg['low_alpha'] = 3.0
    base_cfg['num_value_heads'] = 5
    base_cfg['num_subgoal_candidates'] = 16

    import ml_collections
    ds_cfg = ml_collections.ConfigDict(base_cfg)
    train_dataset = HGCDataset(Dataset.create(**train_ds_dict), ds_cfg)
    example_batch = train_dataset.sample(1)
    del train_ds_dict, train_dataset

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = len(task_infos)

    with open(args.ckpt, 'rb') as f:
        load_dict = pickle.load(f)

    rows = []
    for beta in BETAS:
        cfg = dict(base_cfg)
        cfg['pessimism_beta'] = float(beta)
        agent = EXCHIQLAgent.create(
            seed=args.seed,
            ex_observations=example_batch['observations'],
            ex_actions=example_batch['actions'],
            config=cfg,
        )
        agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
        eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])

        per_task = []
        for task_id in tqdm.trange(1, num_tasks + 1, desc=f'seed{args.seed} beta={beta}'):
            ev_info, _, _ = evaluate(
                agent=eval_agent,
                env=env,
                task_id=task_id,
                config=cfg,
                num_eval_episodes=EVAL_EPISODES,
                num_video_episodes=0,
                video_frame_skip=3,
                eval_temperature=0.0,
                eval_gaussian=None,
            )
            per_task.append(float(ev_info['success']))
        overall = float(np.mean(per_task))
        row = {'seed': args.seed, 'beta': beta, 'overall_success': overall}
        for i, v in enumerate(per_task):
            row[f'task{i+1}_success'] = v
        rows.append(row)
        print(f'seed={args.seed} beta={beta} overall={overall:.4f} per_task={per_task}', flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
