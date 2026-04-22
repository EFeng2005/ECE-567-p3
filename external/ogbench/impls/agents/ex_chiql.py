"""Expectile-Heads C-HIQL (EX-HIQL).

Response to the Phase-2 C-HIQL diagnostic finding (scripts/diagnose_sigma.py,
per-state μ↔σ correlation ≈ −0.44): the shared-trunk random-init ensemble's
disagreement σ tracks feature magnitude, not bootstrap-target stochasticity,
so the pessimistic scoring `μ − β·σ` pulls in the wrong direction.

EX-HIQL fixes this with a single change to the supervision signal: each of
the N=5 value heads is trained with a DIFFERENT expectile τ_i drawn from
`head_expectiles = (0.1, 0.3, 0.5, 0.7, 0.9)`. At a deterministic (s, g) pair
the Bellman-target distribution is a point; all expectile fixed-points
coincide and the heads collapse to the same value (σ → 0). At a stochastic
(s, g) pair the distribution is spread; the heads fan out proportionally to
its spread (σ > 0). σ is now a structural estimator of local stochasticity
rather than an epiphenomenon of initialisation.

What does NOT change vs C-HIQL (spec: EXPECTILE_HIQL.md §3.5):
  - Architecture: same GCValue(ensemble=True, num_ensemble=5), shared trunk
    plus 5 independent last-layer projections.
  - Inference rule: μ − β·σ scoring across heads, same as C-HIQL
    (`sample_actions` is bit-identical to `chiql.py`).
  - β_pes is inference-only — one trained checkpoint serves any β.
  - Dataset pipeline, goal-rep, low- and high-actor architectures.
  - All HIQL hyperparameters (γ, α_AWR, subgoal_steps, rep_dim, …).

What DOES change vs C-HIQL:
  (1) `value_loss`: expectile goes from a scalar to a length-N vector that
      broadcasts across the head axis. One head per element.
  (2) `low_actor_loss` / `high_actor_loss`: advantage uses a SINGLE
      designated head (not the mean over heads) so the actor sees an
      HIQL-equivalent advantage signal rather than a mixture-of-expectiles.
      Default indices `(low=3, high=3)` pin both to τ=0.7 (HIQL's default
      — "Design A"). Setting `actor_expectile_index_high=2` (τ=0.5) gives
      Design B, which additionally de-biases the high-actor's advantage.

Spec: see EXPECTILE_HIQL.md §3 and PHASE3_TRAINING_PLAN.md §Task 1.
"""
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize


class EXCHIQLAgent(flax.struct.PyTreeNode):
    """Expectile-Heads Conservative HIQL agent. See module docstring."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Asymmetric expectile squared loss. `expectile` may be a scalar or
        broadcast-compatible array (e.g. (N, 1) to apply per-head τ).
        """
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Per-head IVL value loss with per-head expectile (EXPECTILE_HIQL.md §3.2).

        Each head i is trained independently with its own τ_i from
        `head_expectiles` and its own per-head target bootstrap. We do NOT
        mix heads via min/mean during training — that would collapse head
        diversity. The diversity source is structural (different loss
        fixed-points from different τ), not init-noise-based.
        """
        vs_t = self.network.select('target_value')(batch['observations'], batch['value_goals'])          # (N, B)
        next_vs_t = self.network.select('target_value')(batch['next_observations'], batch['value_goals']) # (N, B)

        qs = batch['rewards'][None, :] + self.config['discount'] * batch['masks'][None, :] * next_vs_t
        advs = qs - vs_t

        vs = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)

        # Per-head expectiles broadcast over the head axis.
        head_expectiles = jnp.asarray(self.config['head_expectiles']).reshape(-1, 1)  # (N, 1)
        per_head_loss = self.expectile_loss(advs, qs - vs, head_expectiles)           # (N, B)
        value_loss = per_head_loss.sum(axis=0).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'v_std_across_heads': vs.std(axis=0).mean(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Low-level AWR loss. Uses `actor_expectile_index_low` head for the
        advantage — default 3 (τ=0.7), HIQL-equivalent.
        """
        vs = self.network.select('value')(batch['observations'], batch['low_actor_goals'])        # (N, B)
        nvs = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])  # (N, B)

        idx = self.config['actor_expectile_index_low']
        v = vs[idx]
        nv = nvs[idx]
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params):
        """High-level AWR loss. Uses `actor_expectile_index_high` head for
        the advantage — default 3 (τ=0.7) = Design A (HIQL-equivalent).
        Override to 2 (τ=0.5) for Design B (de-biased high-actor advantage).
        """
        vs = self.network.select('value')(batch['observations'], batch['high_actor_goals'])          # (N, B)
        nvs = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])   # (N, B)

        idx = self.config['actor_expectile_index_high']
        v = vs[idx]
        nv = nvs[idx]
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v
        loss = value_loss + low_actor_loss + high_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """K-sample pessimistic subgoal selection. Bit-identical to
        `chiql.py.sample_actions` — only the σ *source* has changed.
        """
        high_seed, low_seed = jax.random.split(seed)

        K = self.config['num_subgoal_candidates']
        beta = self.config['pessimism_beta']

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        cand_seeds = jax.random.split(high_seed, K)
        goal_reps = jax.vmap(lambda s: high_dist.sample(seed=s))(cand_seeds)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        def score_one(gr):
            return self.network.select('value')(observations, gr, goal_encoded=True)

        vs_all = jax.vmap(score_one)(goal_reps)  # (K, N)
        mu = vs_all.mean(axis=-1)
        sig = vs_all.std(axis=-1)
        scores = mu - beta * sig

        best_idx = jnp.argmax(scores)
        selected_goal_rep = goal_reps[best_idx]

        low_dist = self.network.select('low_actor')(
            observations, selected_goal_rep, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Consistency checks.
        n_heads = config['num_value_heads']
        head_expectiles = tuple(config['head_expectiles'])
        assert len(head_expectiles) == n_heads, (
            f"len(head_expectiles)={len(head_expectiles)} must equal num_value_heads={n_heads}"
        )
        assert 0 <= config['actor_expectile_index_low'] < n_heads, (
            f"actor_expectile_index_low={config['actor_expectile_index_low']} out of range"
        )
        assert 0 <= config['actor_expectile_index_high'] < n_heads, (
            f"actor_expectile_index_high={config['actor_expectile_index_high']} out of range"
        )

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if config['encoder'] is not None:
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            num_ensemble=n_heads,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            num_ensemble=n_heads,
            gc_encoder=target_value_encoder_def,
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent identity.
            agent_name='ex_chiql',

            # EX-HIQL new knobs (EXPECTILE_HIQL.md §6).
            # Phase-3b default: tight spread around HIQL's natural τ=0.7.
            # The wide-spread (0.1, 0.3, 0.5, 0.7, 0.9) configuration was
            # tried in Phase-3a and diverged numerically — see
            # PHASE3_DESIGN_A_REPORT.md.
            num_value_heads=5,
            head_expectiles=(0.6, 0.65, 0.7, 0.75, 0.8),
            actor_expectile_index_low=2,   # index of τ=0.7 head in the new tuple
            actor_expectile_index_high=2,  # index of τ=0.7 head in the new tuple
            num_subgoal_candidates=16,
            pessimism_beta=0.5,

            # HIQL-inherited hyperparameters (unchanged).
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            low_alpha=3.0,
            high_alpha=3.0,
            subgoal_steps=25,
            rep_dim=10,
            low_actor_rep_grad=False,
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),

            # Dataset hyperparameters (unchanged from HIQL / C-HIQL).
            dataset_class='HGCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
