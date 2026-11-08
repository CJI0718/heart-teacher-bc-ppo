"""Small observable feature basis plus pairwise products, not 500k copied features."""

from dataclasses import dataclass
from itertools import combinations

import jax.numpy as jnp
import numpy as np

from .observations import public_history

BASE_NAMES = (
    "bias",
    "rank",
    "is_club",
    "is_diamond",
    "is_spade",
    "is_heart",
    "is_queen",
    "has_queen",
    "low_spades",
    "high_spades",
    "suit_count",
    "makes_void",
    "heart_count",
    "high_cards",
    "leading",
    "follows_suit",
    "beats_table",
    "table_penalty",
    "own_raw_penalty",
    "max_opponent_penalty",
    "point_owners",
    "remaining_points",
    "unseen_higher_in_suit",
    "unseen_lower_in_suit",
    "opponents_void_in_suit",
    "trick_progress",
    "queen_seen",
    "only_self_points",
    "one_opponent_owns_points",
    "hearts_broken",
)


def base_features(obs):
    g = obs.match.game
    ids = jnp.clip(obs.play_hand_cards.astype(jnp.int32), 0, 51)
    rank, suit = ids % 13, ids // 13
    counts = jnp.sum(g.hand.reshape(4, 13), axis=1)
    rows, valid, _, voids = public_history(obs)
    card_ids = jnp.arange(52)
    seen = jnp.any((rows[..., None] == card_ids) & valid[..., None], axis=(0, 1))
    unseen = ~seen & ~g.hand
    table = g.current_trick
    safe = jnp.clip(table, 0, 51)
    leading = g.trick_position == 0
    led_suit = safe[0] // 13
    high = jnp.max(jnp.where((table >= 0) & (safe // 13 == led_suit), safe % 13, -1))
    table_points = jnp.sum(jnp.where(table >= 0, (safe // 13 == 3) + 13 * (safe == 36), 0))
    own = g.penalties[g.player]
    others = jnp.where(jnp.arange(4) != g.player, g.penalties, 0)
    same = card_ids[None, :] // 13 == suit[:, None]
    higher = jnp.sum(same & unseen & (card_ids[None, :] % 13 > rank[:, None]), axis=1)
    lower = jnp.sum(same & unseen & (card_ids[None, :] % 13 < rank[:, None]), axis=1)
    total = jnp.sum(g.penalties)
    v = [
        1.0,
        rank / 12,
        suit == 0,
        suit == 1,
        suit == 2,
        suit == 3,
        ids == 36,
        g.hand[36],
        jnp.sum(g.hand[26:36]) / 10,
        jnp.sum(g.hand[37:39]) / 2,
        counts[suit] / 13,
        counts[suit] == 1,
        counts[3] / 13,
        jnp.sum(g.hand.reshape(4, 13)[:, 10:]) / 12,
        leading,
        (~leading) & (suit == led_suit),
        (~leading) & (suit == led_suit) & (rank > high),
        table_points / 26,
        own / 26,
        jnp.max(others) / 26,
        jnp.sum(g.penalties > 0) / 4,
        (26 - total) / 26,
        higher / 12,
        lower / 12,
        jnp.sum(voids[1:], axis=0)[suit] / 3,
        g.trick_index / 13,
        seen[36],
        (own > 0) & (jnp.sum(others) == 0),
        (own == 0) & (jnp.sum(others > 0) == 1),
        g.hearts_broken,
    ]
    return jnp.stack([jnp.broadcast_to(jnp.asarray(x, jnp.float32), (13,)) for x in v], -1)


@dataclass(frozen=True)
class FeatureSpec:
    pairs: tuple[tuple[int, int], ...] = tuple(combinations(range(1, len(BASE_NAMES)), 2))
    selected: tuple[int, ...] | None = None

    def __post_init__(self):
        if any(a < 0 or b >= len(BASE_NAMES) or a >= b for a, b in self.pairs):
            raise ValueError("invalid feature pair")
        size = len(BASE_NAMES) + len(self.pairs)
        if self.selected is not None and (
            not self.selected
            or len(set(self.selected)) != len(self.selected)
            or min(self.selected) < 0
            or max(self.selected) >= size
        ):
            raise ValueError("selected features must be unique valid indices")

    @property
    def all_names(self):
        return list(BASE_NAMES) + [f"{BASE_NAMES[a]}*{BASE_NAMES[b]}" for a, b in self.pairs]

    @property
    def names(self):
        names = self.all_names
        return names if self.selected is None else [names[i] for i in self.selected]

    def __call__(self, obs):
        x = base_features(obs)
        if self.pairs:
            pairs = np.asarray(self.pairs)
            x = jnp.concatenate([x, x[:, pairs[:, 0]] * x[:, pairs[:, 1]]], axis=-1)
        return x if self.selected is None else x[:, np.asarray(self.selected)]

    def to_dict(self):
        return {"pairs": self.pairs, "selected": self.selected, "names": self.names}

    @classmethod
    def from_dict(cls, d):
        return cls(
            tuple(tuple(x) for x in d["pairs"]),
            None if d["selected"] is None else tuple(d["selected"]),
        )


def select_union(weights, activity, spec, top_per_task=32):
    """Coefficient x RMS screening; a predictive proxy, not causal importance.

    Retain the base features, rank nonconstant active interactions per task,
    and merge their indices. No training or weight averaging happens here.
    """
    w, activity = np.asarray(weights), np.asarray(activity)
    if spec.selected is not None or w.ndim != 2 or w.shape[1] != len(spec.names):
        raise ValueError("selection requires all-feature component weights")
    if activity.shape != (w.shape[1],) or not np.isfinite(w).all():
        raise ValueError("invalid feature activity or weights")
    if top_per_task < 1:
        raise ValueError("top_per_task must be positive")
    score = np.abs(w) * np.sqrt(np.maximum(activity, 0))[None, :]
    keep = set(range(len(BASE_NAMES)))
    per_task = []
    for row in score:
        candidates = [
            i
            for i in np.argsort(-row, kind="stable")
            if i >= len(BASE_NAMES) and activity[i] > 1e-6 and row[i] > 0
        ]
        chosen = candidates[:top_per_task]
        keep.update(chosen)
        per_task.append(chosen)
    return FeatureSpec(spec.pairs, tuple(sorted(keep))), per_task
