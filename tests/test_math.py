import jax.numpy as jnp
import numpy as np

from heart_study.losses import gae, clipped_surrogate
from heart_study.teacher import sarsa_error
from heart_study.features import FeatureSpec, select_union, BASE_NAMES


def test_terminal_zero_reward_does_not_bootstrap():
    adv, returns = gae(jnp.array([0.0]), jnp.array([-2.0]), jnp.array([100.0]), jnp.array([True]))
    np.testing.assert_allclose(adv, [2.0])
    np.testing.assert_allclose(returns, [0.0])


def test_rollout_cut_bootstraps_but_deal_boundary_stops_trace():
    # First transition ends a deal. Second belongs to the next deal and is
    # truncated at the rollout edge, so its next value remains relevant.
    adv, ret = gae(
        jnp.array([-3.0, 0.0]),
        jnp.array([-2.0, -4.0]),
        jnp.array([99.0, -2.0]),
        jnp.array([True, False]),
        gamma=1.0,
        lam=0.95,
    )
    np.testing.assert_allclose(adv, [-1.0, 2.0])
    np.testing.assert_allclose(ret, [-3.0, -2.0])


def test_two_step_gae_hand_calculation():
    adv, _ = gae(
        jnp.array([0.0, -6.0]),
        jnp.array([-4.0, -5.0]),
        jnp.array([-5.0, 0.0]),
        jnp.array([False, True]),
        1.0,
        0.5,
    )
    np.testing.assert_allclose(adv, [-1.5, -1.0])


def test_ppo_clipping_both_advantage_signs():
    new = jnp.log(jnp.array([1.5, 0.5, 1.5, 0.5]))
    old = jnp.zeros(4)
    adv = jnp.array([2.0, 2.0, -2.0, -2.0])
    np.testing.assert_allclose(clipped_surrogate(new, old, adv), [2.4, 1.0, -3.0, -1.6], rtol=1e-6)


def test_sarsa_uses_actual_next_action_and_terminal():
    w = jnp.array([2.0, 5.0])
    x = jnp.array([1.0, 0.0])
    n = jnp.array([0.0, 1.0])
    assert float(sarsa_error(w, x, n, 0.0, False)) == 3.0
    assert float(sarsa_error(w, x, n, 7.0, True)) == 5.0


def test_feature_union_does_not_average_weights():
    spec = FeatureSpec()
    n = len(spec.names)
    weights = np.zeros((4, n))
    weights[:, len(BASE_NAMES) : len(BASE_NAMES) + 4] = np.eye(4)
    selected, groups = select_union(weights, np.ones(n), spec, top_per_task=1)
    assert set(range(len(BASE_NAMES), len(BASE_NAMES) + 4)).issubset(selected.selected)
    assert len(set(x[0] for x in groups)) == 4
