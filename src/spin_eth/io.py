"""Reading and writing cached results (JSON for numbers, NPZ for arrays)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), indent=2) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
