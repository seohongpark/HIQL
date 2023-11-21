"""Implementations of algorithms for continuous control."""
import functools
from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, ValueCritic, Critic, ensemblize

import flax
import ml_collections

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class IQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    value: TrainState
    target_value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def critic_loss(agent, batch, critic_params):
        next_v = agent.target_value(batch['next_observations'])
        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v
        q1, q2 = agent.critic(batch['observations'], batch['actions'], params=critic_params)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
        }
    
    def value_loss(agent, batch, value_params):
        q1, q2 = agent.critic(batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = agent.value(batch['observations'], params=value_params)
        value_loss = expectile_loss(q-v, agent.config['expectile']).mean()
        advantage = q - v
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'abs adv mean': jnp.abs(advantage).mean(),
            'adv mean': advantage.mean(),
            'adv max': advantage.max(),
            'adv min': advantage.min(),
        }

    def actor_loss(agent, batch, actor_params):
        v = agent.value(batch['observations'])
        q1, q2 = agent.critic(batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * agent.config['temperature'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = agent.actor(batch['observations'], params=actor_params)
        log_probs = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_probs).mean()

        sorted_adv = jnp.sort(q-v)[::-1]
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': q - v,
            'bc_log_probs': log_probs.mean(),
            'adv median': jnp.median(q - v),
            'adv top 1%': sorted_adv[int(len(sorted_adv) * 0.01)],
            'adv top 10%': sorted_adv[int(len(sorted_adv) * 0.1)],
            'adv top 25%': sorted_adv[int(len(sorted_adv) * 0.25)],
            'adv top 25%': sorted_adv[int(len(sorted_adv) * 0.25)],
            'adv top 75%': sorted_adv[int(len(sorted_adv) * 0.75)],
        }        

    @jax.jit
    def update(agent, batch: Batch) -> InfoDict:
        def critic_loss_fn(critic_params):
            return agent.critic_loss(batch, critic_params)
        
        def value_loss_fn(value_params):
            return agent.value_loss(batch, value_params)

        def actor_loss_fn(actor_params):
            return agent.actor_loss(batch, actor_params)

        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_value = target_update(agent.value, agent.target_value, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        return agent.replace(critic=new_critic, target_value=new_target_value, value=new_value, actor=new_actor), {
            **critic_info, **value_info, **actor_info
        }


def create_learner(
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 value_def,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 value_tx=None,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = Policy(hidden_dims, action_dim=action_dim, 
            log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            actor_tx = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=actor_lr)

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        critic_def = ensemblize(Critic, num_qs=2)(hidden_dims)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
        # target_critic = TrainState.create(critic_def, critic_params)
        
        value_params = value_def.init(value_key, observations)['params']
        if value_tx is None:
            value_tx = optax.adam(learning_rate=value_lr)
        value = TrainState.create(value_def, value_params, tx=value_tx)
        target_value = TrainState.create(value_def, value_params)

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, expectile=expectile, target_update_rate=tau, 
        ))

        return IQLAgent(rng, critic=critic, value=value, target_value=target_value, actor=actor, config=config)

def get_default_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 10.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    return config