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


class CHIQLAgent(flax.struct.PyTreeNode):
    """Conservative Hierarchical implicit Q-learning (C-HIQL) agent.

    Minimal-surgery extension of HIQL (Park et al., 2023) per C-HIQL.md §3 and
    HIQL_EXPLAINER.md §9. Two and only two changes relative to HIQL:

      (1) Training: the high-level value network has N=5 independent heads
          instead of HIQL's 2. Each head is trained with the SAME action-free
          expectile loss and its OWN per-head TD target (§3.2). No pessimism
          is injected into any training loss — the training target stays
          optimistic (§3.4, §9.3 "Every loss function does not change").

      (2) Inference (sample_actions): draw K=16 candidate subgoal reps from
          pi_high, score each by mu - beta_pes * sigma over the N heads,
          argmax, and feed to pi_low (§9.2). Beta_pes is inference-only, so
          one checkpoint serves any value of beta_pes (§9.2 bullet 3).

    Everything else is HIQL verbatim: goal representation phi, pi_low/pi_high
    architectures, dataset pipeline, hyperparameters.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the N-head IVL value loss, per C-HIQL.md §3.2.

        Each head is trained independently with the same action-free expectile
        loss and its own per-head TD target. We do NOT mix heads via min/mean
        during training — that would collapse head diversity and violate the
        "train optimistically, decide pessimistically" split (§3.4: value-
        function training target is unmodified). Pessimism enters only at
        inference time via sample_actions.

        Per-head loss:
          q_i     = r + gamma * mask * V_i_target(s', g)   [stop-grad via target net]
          adv_i   = q_i - V_i_target(s, g)
          L_i     = E[ L_tau^2( adv_i, q_i - V_i(s, g) ) ]
          L_total = sum_i L_i
        """
        # Target-network predictions (shape (N, B), stop-grad).
        vs_t = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        next_vs_t = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])

        # Per-head TD target and per-head advantage weight.
        qs = batch['rewards'][None, :] + self.config['discount'] * batch['masks'][None, :] * next_vs_t
        advs = qs - vs_t

        # Live predictions (the ones that receive gradients).
        vs = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)

        per_head_loss = self.expectile_loss(advs, qs - vs, self.config['expectile'])
        value_loss = per_head_loss.sum(axis=0).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': vs.mean(),
            'v_max': vs.max(),
            'v_min': vs.min(),
            'v_std_across_heads': vs.std(axis=0).mean(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss.

        Identical to HIQL — the spec's §3.4 / §9.3 mark low-level training as
        unmodified. For N>2 heads we reduce with vs.mean(axis=0), which is the
        natural generalization of HIQL's `(v1 + v2) / 2`. No pessimism here:
        beta_pes is inference-only (§9.2).
        """
        vs = self.network.select('value')(batch['observations'], batch['low_actor_goals'])        # (N, B)
        nvs = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])  # (N, B)
        v = vs.mean(axis=0)
        nv = nvs.mean(axis=0)
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
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
        """Compute the high-level actor loss.

        Identical to HIQL — the high-level AWR advantage uses the ensemble
        mean, NOT V_pes. Spec §3.4 / §9.3 are explicit that every loss
        function is unchanged and pessimism is inference-only. Training
        pi_high with V_pes would make beta_pes a training-time hyper and
        break the "one checkpoint serves all beta_pes" property (§9.2).
        """
        vs = self.network.select('value')(batch['observations'], batch['high_actor_goals'])          # (N, B)
        nvs = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])   # (N, B)
        v = vs.mean(axis=0)
        nv = nvs.mean(axis=0)
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
        """Compute the total loss."""
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
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
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
        """K-sample pessimistic subgoal selection (HIQL_EXPLAINER.md §9.2).

        Implements, verbatim:

            phi_candidates = [pi_high(s, g).sample() for _ in range(K)]
            for phi_k in phi_candidates:
                V_all = [V_head_i(s, phi_k) for i in 1..N]      # N scalars
                score_k = mean(V_all) - beta_pes * std(V_all)
            phi_sub = phi_candidates[argmax(score_k)]
            a = pi_low(s, phi_sub).sample()

        Shape assumption: single (obs, goal) per call, as produced by
        utils.evaluation.evaluate() at each env step. `vs_all` is (K, N).
        beta_pes enters only here, so one trained checkpoint can be evaluated
        at any beta_pes (§9.2 bullet 3).
        """
        high_seed, low_seed = jax.random.split(seed)

        K = self.config['num_subgoal_candidates']
        beta = self.config['pessimism_beta']

        # Step 1: draw K candidate subgoal reps from pi_high. Length-normalize
        # onto the 10-sphere to match phi's output geometry.
        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        cand_seeds = jax.random.split(high_seed, K)
        goal_reps = jax.vmap(lambda s: high_dist.sample(seed=s))(cand_seeds)  # (K, rep_dim)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        # Step 2: score each candidate by mu - beta * sigma over the N heads.
        # goal_encoded=True tells GCValue that `gr` is already the 10-sphere
        # rep so it bypasses the goal_rep encoder (matches how pi_low consumes it).
        def score_one(gr):
            return self.network.select('value')(observations, gr, goal_encoded=True)

        vs_all = jax.vmap(score_one)(goal_reps)  # (K, N)
        mu = vs_all.mean(axis=-1)                # (K,)
        sig = vs_all.std(axis=-1)                # (K,)
        scores = mu - beta * sig                 # (K,)

        # Step 3: argmax and feed the winner to pi_low.
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
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
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

        # Define the encoders that handle the inputs to the value and actor networks.
        # The subgoal representation phi([s; g]) is trained by the parameterized value function V(s, phi([s; g])).
        # The high-level actor predicts the subgoal representation phi([s; w]) for subgoal w given s and g.
        # The low-level actor predicts actions given the current state s and the subgoal representation phi([s; w]).
        if config['encoder'] is not None:
            # Pixel-based environments require visual encoders for state inputs, in addition to the pre-defined shared
            # encoder for subgoal representations.

            # Value: V(encoder^V(s), phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | encoder^h([s; g]))
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.

            # Value: V(s, phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = None

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            num_ensemble=config['num_value_heads'],
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            num_ensemble=config['num_value_heads'],
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
            # Agent hyperparameters.
            agent_name='chiql',  # Agent name.
            # C-HIQL: three and only three new knobs vs HIQL (HIQL_EXPLAINER §9.3).
            num_value_heads=5,  # N: ensemble size on the value network. §6 default.
            num_subgoal_candidates=16,  # K: candidates drawn from pi_high at inference (§9.2).
            pessimism_beta=0.5,  # beta_pes: inference-only coefficient in mu - beta * sigma (§9.2).
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
