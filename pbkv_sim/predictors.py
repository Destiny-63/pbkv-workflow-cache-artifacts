"""Step-level role predictors: first-order Markov (P1), oracle (P2), uniform."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path

STEP_ROLES = ("Decomposer", "Reader", "Aggregator", "Verifier", "Writer", "END")
ROLE_INDEX = {r: i for i, r in enumerate(STEP_ROLES)}
END_ID = ROLE_INDEX["END"]
N_ROLES = len(STEP_ROLES)


@dataclass(frozen=True)
class RoleTable:
    roles: tuple[str, ...] = STEP_ROLES

    def __post_init__(self):
        if len(self.roles) < 2 or self.roles[-1] != "END" or len(set(self.roles)) != len(self.roles):
            raise ValueError("Role table needs unique roles with END last")

    @property
    def index(self):
        return {r: i for i, r in enumerate(self.roles)}

    @property
    def end_id(self):
        return len(self.roles) - 1

    @classmethod
    def from_metadata(cls, path):
        content = json.loads(Path(path).read_text())
        return cls(tuple(content["roles"]))


DEFAULT_ROLES = RoleTable()


class MarkovPredictor:
    def __init__(self, alpha: float = 0.5, roles: RoleTable = DEFAULT_ROLES):
        self.alpha = alpha
        self.roles = roles
        self.T: np.ndarray | None = None

    def fit(self, sequences: list[list[str]]) -> "MarkovPredictor":
        index, end = self.roles.index, self.roles.end_id
        counts = np.zeros((len(index), len(index)))
        for seq in sequences:
            for a, b in zip(seq, seq[1:] + ["END"]):
                counts[index[a], index[b]] += 1
        counts += self.alpha
        counts[end] = 0.0
        counts[end, end] = 1.0
        self.T = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict(self, current: str, k_max: int) -> np.ndarray:
        end = self.roles.end_id
        alive = np.zeros(len(self.roles.roles))
        alive[self.roles.index[current]] = 1.0
        rows = []
        for _ in range(k_max):
            total = alive.sum()
            nxt = alive @ self.T
            if total <= 0:
                row = np.zeros(len(self.roles.roles))
                row[end] = 1.0
            else:
                row = nxt / total
            rows.append(row)
            alive = nxt.copy()
            alive[end] = 0.0
        return np.stack(rows)

    def sample_paths(self, current: str, k_max: int, m: int, rng: np.random.Generator) -> np.ndarray:
        end = self.roles.end_id
        out = np.full((m, k_max), end, dtype=np.int64)
        state = np.full(m, self.roles.index[current], dtype=np.int64)
        alive = np.ones(m, dtype=bool)
        cdf = np.cumsum(self.T, axis=1)
        for k in range(k_max):
            u = rng.random(m)
            nxt = np.minimum((u[:, None] > cdf[state]).sum(axis=1), end)
            nxt = np.where(alive, nxt, end)
            out[:, k] = nxt
            alive &= nxt != end
            state = np.where(alive, nxt, state)
        return out


def oracle_rows(future: list[str], k_max: int, roles: RoleTable = DEFAULT_ROLES) -> np.ndarray:
    rows = np.zeros((k_max, len(roles.roles)))
    for k in range(k_max):
        rows[k, roles.index[future[k] if k < len(future) else "END"]] = 1.0
    return rows


def uniform_rows(k_max: int, roles: RoleTable = DEFAULT_ROLES) -> np.ndarray:
    rows = np.full((k_max, len(roles.roles)), 1.0 / roles.end_id)
    rows[:, roles.end_id] = 0.0
    return rows


def _ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    conf_a = np.asarray(conf)
    corr_a = np.asarray(correct, dtype=float)
    if conf_a.size == 0:
        return 0.0
    idx = np.minimum((conf_a * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(corr_a[sel].mean() - conf_a[sel].mean())
    return float(total)


def evaluate(pred_fn, sequences: list[list[str]], k_max: int = 3, roles: RoleTable = DEFAULT_ROLES) -> dict:
    conf = [[] for _ in range(k_max)]
    correct = [[] for _ in range(k_max)]
    for seq in sequences:
        full = seq + ["END"]
        for t in range(len(seq)):
            p = pred_fn(seq[: t + 1])
            for k in range(k_max):
                j = t + 1 + k
                if j >= len(full):
                    break
                top = int(np.argmax(p[k]))
                conf[k].append(float(p[k, top]))
                correct[k].append(top == roles.index[full[j]])
    out = {}
    for k in range(k_max):
        out[f"acc_{k + 1}"] = float(np.mean(correct[k])) if correct[k] else 0.0
        out[f"ece_{k + 1}"] = _ece(conf[k], correct[k])
        out[f"n_{k + 1}"] = len(correct[k])
    return out
