import json
import numpy as np
import jax.numpy as jnp
import pytest

from heart_study.cli import main
from heart_study.checkpoints import save_arrays, load_arrays
from heart_study.environment import terminal_components


@pytest.mark.parametrize(
    "command,extra",
    [
        ("components", []),
        ("teacher", ["--features", "missing.json"]),
        ("bc", ["--data", "missing"]),
        ("ppo", ["--checkpoint", "missing"]),
    ],
)
def test_training_commands_default_to_plan(tmp_path, capsys, command, extra):
    output = tmp_path / "must_not_exist"
    main([command, "--output", str(output), *extra])
    result = json.loads(capsys.readouterr().out)
    assert result["training_executed"] is False
    assert not output.exists()


def test_checkpoint_roundtrip_without_training(tmp_path):
    root = tmp_path / "fixture"
    save_arrays(
        root,
        {"weights": np.array([1.0, 2.0])},
        {"stage": "test_fixture", "training_executed": False},
    )
    arrays, m = load_arrays(root)
    np.testing.assert_array_equal(arrays["weights"], [1.0, 2.0])
    assert m["training_executed"] is False


def test_moon_terminal_labels(play_obs):
    g = play_obs.match.game
    # Public completed tricks contain all point cards. This is a label fixture,
    # not a legal hand trajectory or a model-training dataset.
    points = np.array(list(range(39, 52)) + [36], dtype=np.int8)
    rows = np.full((13, 4), -1, np.int8)
    rows.reshape(-1)[: len(points)] = points
    winners = jnp.full(13, 0, jnp.int8)
    obs = play_obs._replace(
        match=play_obs.match._replace(
            game=g._replace(
                player=jnp.array(0), trick_history=jnp.asarray(rows), trick_winners=winners
            )
        )
    )
    own = terminal_components(obs, np.array([0, 1, 2, 3]), 1, 0, 0)
    np.testing.assert_array_equal(own, [13, 13, 1, 0])
    other = obs._replace(
        match=obs.match._replace(game=obs.match.game._replace(player=jnp.array(2)))
    )
    np.testing.assert_array_equal(terminal_components(other, [0, 1, 2, 3], 1, 26, 0), [0, 0, 0, 1])
