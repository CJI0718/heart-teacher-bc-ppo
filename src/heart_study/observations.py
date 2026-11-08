"""Observation-only encoding. No function in this module accepts an engine state.

HEART trick arrays are in PLAY ORDER, not absolute seat order. The previous
winner identifies the next leader. Unseen history is masked, including the
empty current trick. A permanent context token keeps attention well-defined.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Encoded(NamedTuple):
    cards: jax.Array  # [13, 19]: per-card identity, ownership, suit counts
    history: jax.Array  # [56, 26]: ordered public cards, relative seat, position
    history_mask: jax.Array
    context: jax.Array  # public facts, scores, voids, passed/received cards
    legal: jax.Array  # [13]
    present: jax.Array  # a held card, not just a stable nonnegative slot ID


def onehot(x, n):
    return jax.nn.one_hot(x, n, dtype=jnp.float32)


def public_history(obs):
    g = obs.match.game
    rows = jnp.concatenate([g.trick_history, g.current_trick[None, :]], axis=0)
    valid = rows >= 0
    # In the first completed trick, 2C was first, but its seat is not in the
    # card sequence. Reconstruct first leader from the winner and winning offset.
    first = jnp.clip(rows[0], 0, 51)
    eligible = (first // 13) == (first[0] // 13)
    offset = jnp.argmax(jnp.where(eligible, first % 13, -1))
    first_leader = (g.trick_winners[0] - offset) % 4
    leaders = jnp.concatenate([first_leader[None], g.trick_winners[:-1], g.leader[None]])
    seats = (leaders[:, None] + jnp.arange(4)[None, :]) % 4
    relative_seats = (seats - g.player) % 4
    led_suits = jnp.clip(rows[:, 0], 0, 51) // 13
    off_suit = valid & ((jnp.clip(rows, 0, 51) // 13) != led_suits[:, None])
    # Missing rows do not establish a void; only actual off-suit plays do.
    voids = jnp.any(
        off_suit[:, :, None, None]
        & (relative_seats[:, :, None, None] == jnp.arange(4)[None, None, :, None])
        & (led_suits[:, None, None, None] == jnp.arange(4)[None, None, None, :]),
        axis=(0, 1),
    )
    return rows, valid, relative_seats, voids


def encode(obs):
    g = obs.match.game
    ids = jnp.clip(obs.play_hand_cards.astype(jnp.int32), 0, 51)
    present = (obs.play_hand_cards >= 0) & g.hand[ids]
    suits, ranks = ids // 13, ids % 13
    counts = jnp.sum(g.hand.reshape(4, 13), axis=1).astype(jnp.float32)
    cards = jnp.concatenate(
        [
            onehot(suits, 4),
            onehot(ranks, 13),
            present[:, None].astype(jnp.float32),
            (counts[suits] / 13)[:, None],
        ],
        axis=-1,
    )
    cards *= present[:, None]
    rows, valid, relative_seats, voids = public_history(obs)
    hid = jnp.clip(rows.reshape(-1), 0, 51)
    trick_ids = jnp.concatenate([jnp.arange(13), g.trick_index[None]])
    history = jnp.concatenate(
        [
            onehot(hid // 13, 4),
            onehot(hid % 13, 13),
            onehot(relative_seats.reshape(-1), 4),
            onehot(jnp.tile(jnp.arange(4), 14), 4),
            jnp.repeat(trick_ids / 13, 4)[:, None],
        ],
        axis=-1,
    )
    history *= valid.reshape(-1, 1)
    seats = (g.player + jnp.arange(4)) % 4
    passed = jnp.sum(onehot(obs.match.cards_passed, 52), axis=0)
    received = jnp.sum(onehot(obs.match.cards_received, 52), axis=0)
    context = jnp.concatenate(
        [
            counts / 13,
            g.penalties[seats] / 26,
            g.hand_sizes[seats] / 13,
            obs.match.match_scores[seats] / 100,
            jnp.asarray([g.trick_index / 13, g.trick_position / 4, g.hearts_broken]),
            onehot((g.leader - g.player) % 4, 4),
            onehot(obs.match.pass_direction, 4),
            voids.reshape(-1).astype(jnp.float32),
            passed,
            received,
        ]
    ).astype(jnp.float32)
    return Encoded(cards, history, valid.reshape(-1), context, obs.play_action_mask, present)


def stack_encoded(items):
    if not items:
        raise ValueError("at least one observation is required")
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *items)
