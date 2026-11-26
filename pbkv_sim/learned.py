"""Torch-free, recorded GraphSAGE probabilities for causal offline replay."""

import json
from pathlib import Path

import numpy as np

from .predictors import DEFAULT_ROLES, RoleTable


class PredictionBank:
    def __init__(self, path: Path, roles: RoleTable | None = None):
        self.path = Path(path)
        content = json.loads(self.path.read_text())
        if content.get("schema") != 1:
            raise ValueError("Unsupported prediction bank schema")
        self.provenance = content["provenance"]
        self.roles = RoleTable(tuple(content.get("roles", DEFAULT_ROLES.roles)))
        if roles is not None and roles != self.roles:
            raise ValueError("Prediction bank role order differs from dataset metadata")
        self.checkpoint_sha256 = content["checkpoint_sha256"]
        self.rows = {}
        for key, value in content["probabilities"].items():
            p = np.asarray(value, dtype=float)
            if (p.ndim != 2 or p.shape[1] != len(self.roles.roles) or not np.isfinite(p).all()
                    or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6)):
                raise ValueError(f"Invalid probability table: {key}")
            self.rows[key] = p

    def predict(self, workflow, idx: int, k: int):
        key = f"{workflow.setting}|{workflow.workflow_id}|{idx}"
        if key not in self.rows:
            raise KeyError(f"No learned prediction for {key}; refusing silent fallback")
        if k > len(self.rows[key]):
            raise ValueError("Requested horizon exceeds trained predictor")
        return self.rows[key][:k].copy()
