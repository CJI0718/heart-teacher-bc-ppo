import jax
import jax.numpy as jnp
import pytest
import heart
from heart.classic import PLAY


@pytest.fixture(scope="session")
def play_obs():
    env = heart.make_single_agent(controlled_player=0, opponents=("easy", "medium", "hard"))
    state, obs = jax.jit(env.reset)(jax.random.key(6))
    if int(obs.match.phase) != PLAY:
        action = jnp.argmax(obs.pass_action_mask).astype(jnp.int32)
        state, obs, *_ = jax.jit(env.step)(state, action)
    assert int(obs.match.phase) == PLAY
    return obs
