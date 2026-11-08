"""Array-only checkpoints, with explicit stage and feature/model metadata.

No pickle, no training at load. These snapshots are stage hand-off weights;
they do not include environment or optimizer state for exact resumption.
"""

from pathlib import Path
import json
import jax
import numpy as np


def save_arrays(directory, arrays, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "arrays.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_arrays(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "arrays.npz", allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("checkpoint contains nonfinite arrays")
    return arrays, metadata


def pack_params(params):
    return {
        f"{layer}.{key}": np.asarray(value)
        for layer, entries in params.items()
        for key, value in entries.items()
    }


def unpack_params(arrays):
    params = {}
    for name, value in arrays.items():
        layer, key = name.split(".")
        params.setdefault(layer, {})[key] = jax.numpy.asarray(value)
    return params
