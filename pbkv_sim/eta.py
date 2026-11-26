"""Per-role duration model and urgency tables U[k, a] = E[u(T(k)) | step k is a]."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import numpy as np

from .predictors import DEFAULT_ROLES, END_ID, N_ROLES, ROLE_INDEX, STEP_ROLES, RoleTable


def duration_key(role: str, thinking: bool) -> tuple[str, bool]:
    return (role, bool(thinking) and role == "Verifier")


@dataclass
class DurationModel:
    samples: dict[tuple[str, bool], np.ndarray]
    reader_group: np.ndarray
    tool_samples: dict = field(default_factory=dict)
    role_tools: dict = field(default_factory=dict)
    roles: RoleTable = DEFAULT_ROLES
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def fit(cls, workflows, roles: RoleTable = DEFAULT_ROLES) -> "DurationModel":
        buckets: dict[tuple[str, bool], list[float]] = {}
        groups: list[int] = []
        tools, role_tools = {}, {}
        for wf in workflows:
            for step in wf.steps():
                following = wf.tools_after(step)
                role_tools.setdefault(duration_key(step.role, wf.thinking), []).append(
                    tuple(t.tool_type for t in following))
                if step.role == "Reader":
                    groups.append(len(step.calls))
                for c in step.calls:
                    buckets.setdefault(duration_key(c.role, wf.thinking), []).append(c.service_time)
            for tool in wf.tools:
                tools.setdefault((tool.tool_type, None), []).append(tool.duration)
                if wf.repo:
                    tools.setdefault((tool.tool_type, wf.repo), []).append(tool.duration)
                tools.setdefault((None, None), []).append(tool.duration)
        return cls({k: np.asarray(v, dtype=float) for k, v in buckets.items()}, np.asarray(groups or [1]),
                   {k: np.asarray(v, dtype=float) for k, v in tools.items()}, role_tools, roles)

    def _pool(self, role: str, thinking: bool) -> np.ndarray:
        return self.samples[duration_key(role, thinking)]

    def tool_draw(self, tool_type, repo, rng, n, elapsed=None, visible=True):
        keys = [(tool_type, repo), (tool_type, None), (None, None)] if visible else [(None, None)]
        pool = next((self.tool_samples[k] for k in keys if k in self.tool_samples), np.array([0.0]))
        if elapsed is None:
            return rng.choice(pool, size=n)
        tail = pool[pool > elapsed]
        return rng.choice(tail, size=n) - elapsed if tail.size else np.zeros(n)

    def wait_after(self, role, thinking, repo, rng, n, visible=True):
        options = self.role_tools.get(duration_key(role, thinking), [()])
        if not any(options):
            return np.zeros(n)
        choices = rng.integers(len(options), size=n)
        out = np.zeros(n)
        for i in np.unique(choices):
            selected = choices == i
            for tool_type in options[int(i)]:
                out[selected] += self.tool_draw(tool_type, repo, rng, int(selected.sum()), visible=visible)
        return out

    def draw(self, role: str, thinking: bool, rng: np.random.Generator, n: int) -> np.ndarray:
        pool = self._pool(role, thinking)
        if role != "Reader":
            return rng.choice(pool, size=n)
        sizes = rng.choice(self.reader_group, size=n)
        width = int(sizes.max())
        draws = rng.choice(pool, size=(n, width))
        mask = np.arange(width)[None, :] < sizes[:, None]
        return np.where(mask, draws, 0.0).max(axis=1)

    def remaining(self, role: str, thinking: bool, elapsed: float, rng: np.random.Generator,
                  n: int, group_size: int = 1) -> np.ndarray:
        pool = self._pool(role, thinking)
        tail = pool[pool > elapsed]
        if tail.size == 0:
            return np.zeros(n)
        return rng.choice(tail, size=(n, max(group_size, 1))).max(axis=1) - elapsed


def urgency_tables(model: DurationModel, paths: np.ndarray, current_role: str, thinking: bool,
                   elapsed: float, group_size: int, tau: float | None,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    m, k_max = paths.shape
    n_roles, end = len(model.roles.roles), model.roles.end_id
    U = np.zeros((k_max, n_roles))
    ET = np.zeros((k_max, n_roles))
    t = model.remaining(current_role, thinking, elapsed, rng, m, group_size)
    for k in range(k_max):
        roles_k = paths[:, k]
        alive = roles_k != end
        uk = np.exp(-t / tau) if tau else np.ones(m)
        fallback_u = float(uk[alive].mean()) if alive.any() else 0.0
        fallback_t = float(t[alive].mean()) if alive.any() else 0.0
        for a in range(n_roles):
            if a == end:
                continue
            sel = roles_k == a
            U[k, a] = float(uk[sel].mean()) if sel.any() else fallback_u
            ET[k, a] = float(t[sel].mean()) if sel.any() else fallback_t
        step = np.zeros(m)
        for a in np.unique(roles_k[alive]):
            sel = roles_k == a
            step[sel] = model.draw(model.roles.roles[int(a)], thinking, rng, int(sel.sum()))
        t = t + step
    return U, ET


def oracle_tables(future_starts: list[tuple[str, float]], now: float, k_max: int,
                  tau: float | None, roles: RoleTable = DEFAULT_ROLES) -> tuple[np.ndarray, np.ndarray]:
    U = np.zeros((k_max, len(roles.roles)))
    ET = np.zeros((k_max, len(roles.roles)))
    for k, (role, start) in enumerate(future_starts[:k_max]):
        if role == "END":
            break
        a = roles.index[role]
        ET[k, a] = max(start - now, 0.0)
        U[k, a] = float(np.exp(-ET[k, a] / tau)) if tau else 1.0
    return U, ET


def oracle_time_tables(future_starts: list[tuple[str, float]], now: float, k_max: int,
                       tau: float | None, step_len: float, roles: RoleTable = DEFAULT_ROLES) -> np.ndarray:
    known = [start for role, start in future_starts if role != "END"]
    end = next((start for role, start in future_starts if role == "END"), None)
    times = known[:k_max]
    last = times[-1] if times else (end if end is not None else now)
    while len(times) < k_max:
        last += step_len
        times.append(last)
    U = np.ones((k_max, len(roles.roles)))
    if tau:
        U *= np.exp(-np.maximum(np.array(times) - now, 0.0) / tau)[:, None]
    return U


def candidate_discounts(model, p, current_role, thinking, elapsed, group_size, tau, rng, n,
                        repo=None, waiting_tools=(), wait_elapsed=0.0, visible=True, current_complete=False):
    """Sample durations using the same role marginals supplied to the score.

    Non-END draws describe the elapsed time conditional on reaching each step.
    Termination probabilities enter the score separately through survival.
    """
    if tau is not None and tau <= 0:
        raise ValueError("tau must be positive")
    if waiting_tools:
        # Condition the sum of a serial tool sequence on time already waited.
        pool = sum((model.tool_draw(t.tool_type, repo, rng, n * 4, visible=visible)
                    for t in waiting_tools), start=np.zeros(n * 4))
        tail = pool[pool > wait_elapsed]
        t = rng.choice(tail, size=n) - wait_elapsed if tail.size else np.zeros(n)
    elif current_complete:
        t = np.zeros(n)
    else:
        t = model.remaining(current_role, thinking, elapsed, rng, n, group_size)
        t += model.wait_after(current_role, thinking, repo, rng, n, visible)
    out = np.zeros(len(p))
    end = model.roles.end_id
    for k, row in enumerate(p):
        out[k] = float(np.exp(-t / tau).mean()) if tau else 1.0
        if k == len(p) - 1:
            break
        probs = row[:end]
        if probs.sum() == 0:
            break
        sampled = rng.choice(end, p=probs / probs.sum(), size=n)
        for a in np.unique(sampled):
            sel = sampled == a
            role = model.roles.roles[int(a)]
            count = int(sel.sum())
            t[sel] += model.draw(role, thinking, rng, count)
            t[sel] += model.wait_after(role, thinking, repo, rng, count, visible)
    return out


def service_distribution(model, role, thinking, group_size=None):
    """Exact discrete distribution of a role's service time, including Reader maxima."""
    key = ("service", role, thinking, group_size)
    if key in model._cache:
        return model._cache[key]
    values, counts = np.unique(model._pool(role, thinking), return_counts=True)
    probabilities = counts / counts.sum()
    if role == "Reader":
        sizes = [group_size] if group_size is not None else model.reader_group
        cdf = np.cumsum(probabilities)
        max_cdf = np.mean(cdf[:, None] ** np.asarray(sizes)[None, :], axis=1)
        probabilities = np.diff(np.r_[0.0, max_cdf])
    model._cache[key] = (values, probabilities)
    return values, probabilities


def tool_distribution(model, tool_types, repo, visible):
    key = ("tools", tuple(tool_types), repo, visible)
    if key in model._cache:
        return model._cache[key]
    values, probabilities = np.array([0.0]), np.array([1.0])
    for tool_type in tool_types:
        keys = [(tool_type, repo), (tool_type, None), (None, None)] if visible else [(None, None)]
        pool = next((model.tool_samples[k] for k in keys if k in model.tool_samples), np.array([0.0]))
        x, counts = np.unique(pool, return_counts=True)
        if len(values) * len(x) > 1_000_000:
            raise ValueError("Exact serial-tool convolution exceeds one million terms")
        sums = np.add.outer(values, x).ravel()
        mass = np.multiply.outer(probabilities, counts / counts.sum()).ravel()
        values, inverse = np.unique(sums, return_inverse=True)
        probabilities = np.bincount(inverse, weights=mass, minlength=len(values))
    model._cache[key] = (values, probabilities)
    return values, probabilities


def discounted_remaining(distribution, elapsed, tau):
    values, probabilities = distribution
    keep = values > elapsed
    if not keep.any():
        return 1.0
    probabilities = probabilities[keep] / probabilities[keep].sum()
    return float(np.dot(probabilities, np.exp(-(values[keep] - elapsed) / tau)))


def exact_candidate_discounts(model, p, current_role, thinking, elapsed, group_size, tau,
                              repo=None, waiting_tools=(), wait_elapsed=0.0, visible=True, current_complete=False):
    """Exact Laplace transform under the same independent empirical duration model.

    E[exp(-(X+Y)/tau)] factors for independent durations. Future role mixtures
    use the predictor's non-END row; survival is handled by candidate_score.
    This evaluates the candidate's expectation without Monte Carlo error.
    """
    if tau is None:
        return np.ones(len(p))
    if tau <= 0:
        raise ValueError("tau must be positive")

    def wait_discount(role):
        key = ("wait_discount", role, thinking, repo, tau, visible)
        if key in model._cache:
            return model._cache[key]
        options = model.role_tools.get(duration_key(role, thinking), [()])
        # Options may have different lengths; count tuples without coercing to an array.
        options = Counter(options)
        total = sum(options.values())
        value = 0.0
        for types, count in options.items():
            x, mass = tool_distribution(model, types, repo, visible)
            value += count / total * float(np.dot(mass, np.exp(-x / tau)))
        model._cache[key] = value
        return value

    if waiting_tools:
        current = discounted_remaining(tool_distribution(model, tuple(t.tool_type for t in waiting_tools),
                                                          repo, visible), wait_elapsed, tau)
    elif current_complete:
        current = 1.0
    else:
        current = discounted_remaining(service_distribution(model, current_role, thinking, group_size),
                                       elapsed, tau) * wait_discount(current_role)
    out = np.zeros(len(p))
    out[0] = current
    transforms = {}
    for k in range(1, len(p)):
        row = p[k - 1, :model.roles.end_id]
        if row.sum() == 0:
            break
        mixture = 0.0
        for a in np.flatnonzero(row):
            role = model.roles.roles[int(a)]
            if role not in transforms:
                key = ("role_discount", role, thinking, repo, tau, visible)
                if key not in model._cache:
                    x, mass = service_distribution(model, role, thinking)
                    model._cache[key] = float(np.dot(mass, np.exp(-x / tau))) * wait_discount(role)
                transforms[role] = model._cache[key]
            mixture += row[a] / row.sum() * transforms[role]
        out[k] = out[k - 1] * mixture
    return out


def rank_urgency(et: dict[str, np.ndarray], mask: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = np.concatenate([et[w][mask[w]] for w in et]) if et else np.zeros(0)
    if values.size == 0:
        return {w: np.zeros_like(et[w]) for w in et}
    order = np.sort(values)
    out = {}
    for w, table in et.items():
        ranks = np.searchsorted(order, table, side="left")
        out[w] = np.where(mask[w], 1.0 - ranks / values.size, 0.0)
    return out
