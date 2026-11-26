import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pbkv_sim.scores import node_score, priority, workflow_vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "pbkv_0213e13"))
from pbkv_policy import PBKVPolicy  # noqa: E402

P1 = np.array([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1], [0.1, 0.2, 0.7]])
P2 = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
IDX = {"<END>": 0, "A": 1, "B": 2}


def test_matches_repo_equation_2_example():
    expected = (
        0.3 + 0.7 * 0.8 * 0.5 + 0.7**2 * 0.8 * 0.6 * 0.2
        + 0.1 + 0.7 * 0.2 * 0.7 + 0.7**2 * 0.2 * 0.9 * 0.1
    )
    vecs = {"w1": workflow_vector(P1, 0.7, end_id=0), "w2": workflow_vector(P2, 0.7, end_id=0)}
    got = node_score({"w1": {"A"}, "w2": {"B"}}, vecs, role_index=IDX)
    assert abs(got - expected) < 1e-12


def test_matches_vendored_policy_on_random_fixtures():
    rng = np.random.default_rng(0)
    for _ in range(50):
        policy = PBKVPolicy(lambda n: n.last_access_time)
        policy.configure(["<END>", "A", "B"], 3, 0.7, 0)
        root = SimpleNamespace(id=-1, parent=None, children={})
        pairs = {(f"w{i}", r) for i in range(3) for r in ("A", "B") if rng.random() < 0.5} or {("w0", "A")}
        node = SimpleNamespace(
            id=1, parent=root, children={}, value=[0], host_value=None, lock_ref=0, last_access_time=0.0,
            chain_agent_pairs=set(pairs), visited_chain_ids={w for w, _ in pairs}, active_chain_ids=set(),
            evicted=False, backuped=False,
        )
        policy.sync(node)
        vecs = {}
        for w in {w for w, _ in pairs}:
            p = rng.dirichlet(np.ones(3), size=3)
            policy.update(w, p)
            vecs[w] = workflow_vector(p, 0.7, end_id=0)
        roles = {}
        for w, r in pairs:
            roles.setdefault(w, set()).add(r)
        assert abs(node_score(roles, vecs, role_index=IDX) - policy.score(node)) < 1e-12


def test_urgency_scales_each_term():
    u = np.array([[1.0, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    base = workflow_vector(P1, 0.7, end_id=0)
    scaled = workflow_vector(P1, 0.7, urgency=u, end_id=0)
    assert abs(scaled[1] - (base[1] - 0.5 * 0.3)) < 1e-12
    assert scaled[0] == 0.0


def test_priority_tiers():
    assert priority("lru", True, 3, 9.0, 5.0) == (5.0,)
    assert priority("lae", True, 2, 9.0, 5.0) == (0, 2, 5.0)
    assert priority("lae", False, 2, 9.0, 5.0) == (1, 0.0, 5.0)
    assert priority("he_urg", False, 2, 0.4, 5.0) == (1, 0.4, 5.0)
    assert priority("he_step", True, 1, 0.4, 5.0) < priority("he_step", False, 1, 0.0, 0.0)
