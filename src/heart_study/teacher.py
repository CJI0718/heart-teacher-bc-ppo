"""Linear expected costs and on-policy, semi-gradient Sarsa(lambda).

This is our observation/action adaptation, not the paper's V + max-n engine.
Cost is minimized. Frozen auxiliary heads predict four terminal components
under a shared behavior policy; the final head controls effective deal cost.
"""

import jax
import jax.numpy as jnp


def costs(weights, phi):
    return phi @ weights


def greedy_action(weights, phi, legal):
    return jnp.argmin(jnp.where(legal, costs(weights, phi), jnp.inf)).astype(jnp.int32)


def epsilon_action(weights, phi, legal, key, epsilon):
    explore_key, action_key = jax.random.split(key)
    random = jax.random.categorical(action_key, jnp.where(legal, 0.0, -jnp.inf))
    return jnp.where(
        jax.random.uniform(explore_key) < epsilon, random, greedy_action(weights, phi, legal)
    ).astype(jnp.int32)


def sarsa_error(weights, phi, next_phi, cost, terminal, gamma=1.0):
    """next_phi belongs to the ACTUAL next action; terminal suppresses bootstrap."""
    bootstrap = jnp.where(terminal, 0.0, gamma * jnp.dot(weights, next_phi))
    return cost + bootstrap - jnp.dot(weights, phi)


def sarsa_update(weights, trace, phi, next_phi, cost, terminal, alpha=0.01, gamma=1.0, lam=0.75):
    """Training operation. Never called by the default verification suite."""
    delta = sarsa_error(weights, phi, next_phi, cost, terminal, gamma)
    eligibility = gamma * lam * trace + phi
    # Fixed modest step size, with active-feature normalization.
    rate = alpha / jnp.maximum(1.0, jnp.dot(phi, phi))
    updated = weights + rate * delta * eligibility
    return updated, jnp.where(terminal, jnp.zeros_like(eligibility), eligibility), delta


component_update = jax.vmap(sarsa_update, in_axes=(0, 0, None, None, 0, None, None, None, None))
