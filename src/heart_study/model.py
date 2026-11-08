"""Small pure-JAX attention actor-critic, without persistent hidden state.

Shared card encoder -> masked hand pooling + public-history cross-attention
-> shared per-card actor and invariant pooled critic. No hand-slot embeddings.
The output follows a hand permutation; the value stays invariant.
"""

import math
from functools import lru_cache
import jax
import jax.numpy as jnp

from .config import ModelConfig


def init_params(key, sample, feature_count, config=ModelConfig()):
    keys = iter(jax.random.split(key, 20))
    d = config.width

    def dense(inputs, outputs):
        return {
            "w": jax.random.normal(next(keys), (inputs, outputs)) / math.sqrt(inputs),
            "b": jnp.zeros(outputs),
        }

    return {
        "card": dense(sample.cards.shape[-1], d),
        "history": dense(sample.history.shape[-1], d),
        "context": dense(sample.context.shape[-1], d),
        "query": dense(d, d),
        "key": dense(d, d),
        "value": dense(d, d),
        "mix": dense(4 * d + feature_count, d),
        "actor": dense(d, 1),
        "critic_hidden": dense(2 * d, d),
        "critic": dense(d, 1),
    }


def linear(p, x):
    return x @ p["w"] + p["b"]


def layer_norm(x):
    return (x - jnp.mean(x, axis=-1, keepdims=True)) * jax.lax.rsqrt(
        jnp.var(x, axis=-1, keepdims=True) + 1e-5
    )


def masked_mean(x, mask):
    return jnp.sum(x * mask[:, None], axis=0) / jnp.maximum(jnp.sum(mask), 1)


def forward(params, obs, features, config=ModelConfig()):
    d, heads = config.width, config.heads
    card = jax.nn.gelu(linear(params["card"], obs.cards))
    context = jax.nn.gelu(linear(params["context"], obs.context))
    history = jax.nn.gelu(linear(params["history"], obs.history))
    memory = jnp.concatenate([context[None, :], history], axis=0)
    memory_mask = jnp.concatenate([jnp.ones(1, dtype=bool), obs.history_mask])
    q = linear(params["query"], card).reshape(13, heads, d // heads)
    k = linear(params["key"], memory).reshape(-1, heads, d // heads)
    v = linear(params["value"], memory).reshape(-1, heads, d // heads)
    scores = jnp.einsum("ihd,jhd->hij", q, k) / math.sqrt(d // heads)
    attention = jax.nn.softmax(jnp.where(memory_mask[None, None, :], scores, -1e9), -1)
    attended = jnp.einsum("hij,jhd->ihd", attention, v).reshape(13, d)
    pooled_hand = masked_mean(card, obs.present)
    combined = jnp.concatenate(
        [
            card,
            attended,
            jnp.broadcast_to(pooled_hand, (13, d)),
            jnp.broadcast_to(context, (13, d)),
            features,
        ],
        axis=-1,
    )
    hidden = jax.nn.gelu(layer_norm(linear(params["mix"], combined)))
    logits = linear(params["actor"], hidden)[:, 0]
    critic_input = jnp.concatenate([masked_mean(hidden, obs.present), context])
    value = linear(params["critic"], jnp.tanh(linear(params["critic_hidden"], critic_input)))[0]
    # Raw outputs remain finite. Mask at action selection and in ALL losses.
    return logits, value


def batched_forward(params, obs, features, config=ModelConfig()):
    return jax.vmap(lambda o, f: forward(params, o, f, config))(obs, features)


@lru_cache(maxsize=8)
def compiled_observation_policy(spec, config=ModelConfig()):
    """Cache one executable by architecture; learned params remain dynamic inputs."""
    from .observations import encode

    return jax.jit(lambda params, obs: forward(params, encode(obs), spec(obs), config))


def log_probabilities(logits, legal):
    # The runner guarantees at least one legal action for PLAY. The fallback
    # makes inactive PASS/terminal heads finite for export, not a legal PLAY.
    fallback = jnp.arange(logits.shape[-1]) == 0
    safe = jnp.where(jnp.any(legal, axis=-1, keepdims=True), legal, fallback)
    return jax.nn.log_softmax(jnp.where(safe, logits, -1e9), axis=-1)


def action_log_prob(logits, legal, actions):
    return jnp.take_along_axis(log_probabilities(logits, legal), actions[..., None], -1)[..., 0]


def entropy(logits, legal):
    logp = log_probabilities(logits, legal)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)
