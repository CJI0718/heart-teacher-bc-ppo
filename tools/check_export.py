"""Compile/export randomly initialized weights; no fitting and no public entry.

Artifacts are written under the ignored .cache directory, not submissions/.
"""

from pathlib import Path
import json
import jax
import heart
from heart.contest import sample_observation

from heart_study.config import ModelConfig
from heart_study.features import FeatureSpec
from heart_study.observations import encode
from heart_study.model import init_params
from heart_study.export import submission_policy

root = Path(__file__).resolve().parents[1]
obs = sample_observation()
spec = FeatureSpec(selected=tuple(range(158)))
mc = ModelConfig()
params = init_params(jax.random.key(0), encode(obs), len(spec.names), mc)
entry = heart.save_submission(
    submission_policy(params, spec, mc),
    str(root / ".cache" / "export-check"),
    name="untrained-contract-check",
    description="Random initialization; no learning; not a competitive entry",
)
report = {
    "training_executed": False,
    "parameter_updates": 0,
    "weights": "random initialization",
    "size_bytes": entry.size_bytes,
    "feature_count": len(spec.names),
    "flops_per_decision": entry.flops_per_decision,
    "purpose": "official serialized inference contract only",
}
(root / "verification" / "export_check.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
