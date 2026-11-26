"""PBKV equation (2) with an optional per-(step, role) urgency factor, and eviction priorities."""

from __future__ import annotations

import numpy as np

from .predictors import END_ID, ROLE_INDEX

POLICIES = ("lru", "lae", "he_count", "he_step", "he_urg", "he_urg_rank", "he_time", "cand", "ref_offline", "nu_min")


def workflow_vector(p: np.ndarray, gamma: float, urgency: np.ndarray | None = None,
                    end_id: int = END_ID) -> np.ndarray:
    k = p.shape[0]
    survival = np.ones(k)
    survival[1:] = np.cumprod(1 - p[:-1, end_id])
    weights = survival * gamma ** np.arange(k)
    terms = p if urgency is None else p * urgency[:k]
    v = (weights[:, None] * terms).sum(axis=0)
    v[end_id] = 0.0
    return v


def node_score(roles_by_wf: dict[str, set[str]], vectors: dict[str, np.ndarray],
               role_index: dict[str, int] = ROLE_INDEX) -> float:
    total = 0.0
    for w, roles in sorted(roles_by_wf.items()):
        v = vectors.get(w)
        if v is None:
            continue
        total += sum(v[role_index[r]] for r in sorted(roles) if r in role_index)
    return max(0.0, float(total))


def candidate_score(roles_by_wf, predictions, discounts, role_index=ROLE_INDEX, end_id=END_ID) -> float:
    """Independent-step first-use probability, discounted then unioned over workflows."""
    not_used = 1.0
    for wid, roles in sorted(roles_by_wf.items()):
        if wid not in predictions:
            continue
        p = predictions[wid]
        ids = sorted(role_index[r] for r in roles if r in role_index and role_index[r] != end_id)
        q = np.clip(p[:, ids].sum(axis=1), 0, 1) if ids else np.zeros(len(p))
        survival = np.ones(len(p))
        survival[1:] = np.cumprod(1 - p[:-1, end_id])
        first = np.ones(len(p))
        first[1:] = np.cumprod(1 - q[:-1])
        u = float(np.dot(survival * q * first, discounts[wid]))
        not_used *= 1 - np.clip(u, 0, 1)
    return float(1 - not_used)


def priority(policy: str, retired: bool, n_members: int, score: float, last_access: float) -> tuple:
    if policy == "lru":
        return (last_access,)
    if retired:
        return (0, n_members, last_access)
    if policy == "lae":
        return (1, 0.0, last_access)
    return (1, score, last_access)
