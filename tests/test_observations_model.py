import jax
import jax.numpy as jnp
import numpy as np

from heart_study.observations import encode, public_history, stack_encoded
from heart_study.features import FeatureSpec
from heart_study.model import init_params, forward, log_probabilities
from heart_study.losses import bc_loss, ppo_loss, Batch
from heart_study.config import ModelConfig
from heart_study.export import submission_policy


def test_empty_history_and_spent_slots_are_masked(play_obs):
    g = play_obs.match.game
    slot = int(jnp.argmax(play_obs.play_action_mask))
    card = play_obs.play_hand_cards[slot]
    obs = play_obs._replace(
        match=play_obs.match._replace(game=g._replace(hand=g.hand.at[card].set(False)))
    )
    encoded = encode(obs)
    assert not bool(encoded.present[slot])
    assert np.isfinite(encoded.context).all()


def test_history_uses_play_order_and_reconstructs_seat(play_obs):
    g = play_obs.match.game
    # Lead seat 2: C2, D2, C4, C3. Seat 0 wins (offset 2), seat 3 is void in clubs.
    rows = jnp.full((13, 4), -1, jnp.int8).at[0].set(jnp.array([0, 13, 2, 1], jnp.int8))
    winners = jnp.full(13, -1, jnp.int8).at[0].set(0)
    obs = play_obs._replace(
        match=play_obs.match._replace(
            game=g._replace(
                player=jnp.array(0),
                trick_history=rows,
                trick_winners=winners,
                current_trick=jnp.full(4, -1, jnp.int8),
                trick_index=jnp.array(1),
                leader=jnp.array(0),
            )
        )
    )
    _, _, seats, voids = public_history(obs)
    np.testing.assert_array_equal(seats[0], [2, 3, 0, 1])
    assert bool(voids[3, 0])
    assert not bool(voids[2, 0])


def test_hand_permutation_equivariance_and_value_invariance(play_obs):
    spec = FeatureSpec(pairs=())
    mc = ModelConfig()
    encoded = encode(play_obs)
    features = spec(play_obs)
    params = init_params(jax.random.key(0), encoded, len(spec.names), mc)
    logits, value = forward(params, encoded, features, mc)
    perm = jnp.array([4, 2, 0, 8, 7, 12, 11, 10, 9, 6, 5, 3, 1])
    perm_obs = play_obs._replace(
        play_hand_cards=play_obs.play_hand_cards[perm],
        play_action_mask=play_obs.play_action_mask[perm],
    )
    pl, pv = forward(params, encode(perm_obs), spec(perm_obs), mc)
    np.testing.assert_allclose(pl, logits[perm], atol=2e-6)
    np.testing.assert_allclose(pv, value, atol=2e-6)


def test_mask_and_finite_inactive_head():
    logp = log_probabilities(jnp.array([100.0, 1.0, 2.0]), jnp.array([False, True, False]))
    np.testing.assert_allclose(jnp.exp(logp), [0.0, 1.0, 0.0])
    assert np.isfinite(log_probabilities(jnp.ones(13), jnp.zeros(13, bool))).all()


def test_losses_and_gradients_without_parameter_updates(play_obs):
    spec = FeatureSpec(pairs=())
    mc = ModelConfig()
    encoded = encode(play_obs)
    features = spec(play_obs)
    params = init_params(jax.random.key(1), encoded, len(spec.names), mc)
    before = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(params)]
    batch_obs = stack_encoded([encoded, encoded])
    fs = jnp.stack([features, features])
    action = int(jnp.argmax(encoded.legal))
    actions = jnp.array([action, action])
    loss, grad = jax.value_and_grad(lambda p: bc_loss(p, batch_obs, fs, actions, mc))(params)
    assert np.isfinite(loss)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad))
    logits, _ = forward(params, encoded, features, mc)
    lp = log_probabilities(logits, encoded.legal)[action]
    batch = Batch(
        batch_obs, fs, actions, jnp.array([lp, lp]), jnp.array([1.0, -1.0]), jnp.array([-2.0, -3.0])
    )
    (loss, _), grad = jax.value_and_grad(lambda p: ppo_loss(p, batch), has_aux=True)(params)
    assert np.isfinite(loss)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad))
    for old, new in zip(before, jax.tree_util.tree_leaves(params), strict=True):
        np.testing.assert_array_equal(old, new)


def test_submission_contract_forward(play_obs):
    spec = FeatureSpec(pairs=())
    mc = ModelConfig()
    p = init_params(jax.random.key(2), encode(play_obs), len(spec.names), mc)
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), play_obs)
    passes, plays = jax.jit(submission_policy(p, spec, mc))(batched)
    assert passes.shape == (2, 286) and plays.shape == (2, 13)
    assert np.isfinite(passes).all() and np.isfinite(plays).all()
