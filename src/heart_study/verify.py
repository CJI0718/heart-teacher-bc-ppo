"""A one-deal inference smoke check; never call an optimizer or TD update."""

from pathlib import Path
import json
import hashlib
import jax
import numpy as np

from .config import ModelConfig, HEART_COMMIT
from .environment import PlayRunner
from .features import FeatureSpec
from .observations import encode
from .model import init_params, forward


def digest(params):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(params):
        h.update(np.asarray(x).tobytes())
    return h.hexdigest()


def run_checks(report=None):
    runner = PlayRunner(seed=6)
    spec = FeatureSpec(pairs=())
    mc = ModelConfig()
    params = init_params(jax.random.key(9), encode(runner.obs), len(spec.names), mc)
    before = digest(params)
    call = jax.jit(lambda o: forward(params, encode(o), spec(o), mc))
    decisions = 0
    while True:
        logits, value = call(runner.obs)
        assert np.isfinite(logits).all() and np.isfinite(value)
        action = int(np.argmax(np.where(runner.obs.play_action_mask, np.asarray(logits), -np.inf)))
        _, reward, terminal, labels = runner.step(action)
        decisions += 1
        if terminal:
            break
        if decisions > 13:
            raise AssertionError("deal horizon exceeded")
    assert decisions == 13 and before == digest(params)
    result = {
        "status": "passed",
        "heart_commit": HEART_COMMIT,
        "training_executed": False,
        "parameter_updates": 0,
        "weights_unchanged": True,
        "completed_deals": 1,
        "play_decisions": decisions,
        "purpose": "interface smoke check, not performance evaluation",
        "reward": reward,
        "component_labels": labels.tolist(),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "parameter_count": sum(x.size for x in jax.tree_util.tree_leaves(params)),
    }
    if report:
        path = Path(report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
