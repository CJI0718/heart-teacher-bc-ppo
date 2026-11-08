"""Slow, transparent reference runner around the official single-agent API.

Single CPU environment first. Policy inputs are only observations. Terminal
component labels use the publicly revealed final trick, not opponent hands.
"""

import jax
import jax.numpy as jnp
import numpy as np
import heart
from heart.classic import PASS, PLAY


def fixed_pass_logits(obs):
    """Simple deterministic high-card/point-card passing rule, fixed at all stages."""
    from heart.classic import PASS_COMBINATIONS

    ids = jnp.clip(obs.pass_hand_cards.astype(jnp.int32), 0, 51)
    danger = ids % 13 + 13 * (ids == 36) + 2 * (ids // 13 == 3)
    return jnp.sum(danger[PASS_COMBINATIONS.astype(jnp.int32)], axis=1).astype(jnp.float32)


def fixed_pass(obs):
    return int(jnp.argmax(jnp.where(obs.pass_action_mask, fixed_pass_logits(obs), -jnp.inf)))


def terminal_components(before_obs, final_trick, final_winner, effective_cost, moon_shooter):
    """Unnormalized labels [hearts count, queen cost, own moon, opponent moon]."""
    g = before_obs.match.game
    player = int(g.player)
    rows = np.asarray(g.trick_history)
    winners = np.asarray(g.trick_winners)
    owned = rows[winners == player].reshape(-1)
    if int(final_winner) == player:
        owned = np.concatenate([owned, np.asarray(final_trick)])
    owned = owned[owned >= 0]
    hearts = int(np.sum(owned // 13 == 3))
    queen = 13 * int(np.any(owned == 36))
    own_moon = int(int(moon_shooter) == player)
    opponent_moon = int(int(moon_shooter) >= 0 and not own_moon)
    labels = np.array([hearts, queen, own_moon, opponent_moon], np.float32)
    reconstructed = hearts + queen - 26 * own_moon + 26 * opponent_moon
    if reconstructed != int(effective_cost):
        raise RuntimeError("terminal labels disagree with official effective score")
    return labels


class PlayRunner:
    def __init__(self, seed=7, opponents=("easy", "medium", "hard")):
        self.env = heart.make_single_agent(controlled_player=0, opponents=opponents)
        self._step = jax.jit(self.env.step)
        self._reset = jax.jit(self.env.reset)
        self.key = jax.random.key(seed)
        self.opponents = opponents
        self.reset()

    def reset(self):
        self.key, key = jax.random.split(self.key)
        self.state, self.obs = self._reset(key)
        self._skip_pass()
        return self.obs

    def _skip_pass(self):
        if int(self.obs.match.phase) == PASS:
            self.state, self.obs, reward, done, info = self._step(
                self.state, jnp.asarray(fixed_pass(self.obs), jnp.int32)
            )
            if bool(done) or bool(info.deal_completed) or float(reward) != 0:
                raise RuntimeError("PASS unexpectedly ended a deal")
        if int(self.obs.match.phase) != PLAY:
            raise RuntimeError("expected a PLAY decision")

    def step(self, action):
        if not 0 <= int(action) < 13 or not bool(self.obs.play_action_mask[int(action)]):
            raise ValueError("action must be a legal hand slot")
        before = self.obs
        self.state, self.obs, reward, match_done, info = self._step(
            self.state, jnp.asarray(action, jnp.int32)
        )
        if bool(info.core.invalid_action):
            raise RuntimeError("official environment rejected a masked action")
        terminal = bool(info.deal_completed)
        labels = None
        if terminal:
            labels = terminal_components(
                before,
                self.state.match.last_trick_cards,
                self.state.match.last_trick_winner,
                info.core.deal_scores[0],
                info.core.moon_shooter,
            )
            if not np.isclose(float(reward), -float(info.core.deal_scores[0])):
                raise RuntimeError("reward mismatch")
        if bool(match_done):
            self.reset()
        else:
            self._skip_pass()
        return self.obs, float(reward), terminal, labels


def rule_play(obs, key):
    """Official rule policy returns a card ID; convert it to the stable hand slot."""
    from heart.agents import make_rule_policy

    card = make_rule_policy("medium")(obs.match.game, key)
    return jnp.argmax((obs.play_hand_cards == card) & obs.play_action_mask).astype(jnp.int32)
