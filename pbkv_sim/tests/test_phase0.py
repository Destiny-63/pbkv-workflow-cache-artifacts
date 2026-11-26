"""Acceptance checks for the heterogeneous-tool-wait experiment's replay layer."""

import heapq
import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from pbkv_sim.belady import build_next_use
from pbkv_sim.eta import DurationModel, candidate_discounts, exact_candidate_discounts, service_distribution
from pbkv_sim.tests.test_eta import model
from pbkv_sim.learned import PredictionBank
from pbkv_sim.predictors import DEFAULT_ROLES, MarkovPredictor, RoleTable, oracle_rows
from pbkv_sim.prefix_tree import PrefixTree
from pbkv_sim.replay import Replay, ReplayConfig, replay
from pbkv_sim.scores import candidate_score, workflow_vector
from pbkv_sim.tests.test_replay import fixtures, mk
from pbkv_sim.tests.test_trace import call, write
from pbkv_sim.trace import ToolEvent, load_workflows


def waiting_workflow(seconds=30):
    wf = mk("wait", [("Decomposer", 10), ("Reader", 1)])
    wf.tools = [ToolEvent("wait", 1, "test", 10, 10, 10 + seconds)]
    wf.calls[1] = replace(wf.calls[1], submit=10 + seconds,
                          first_token=10.1 + seconds, end=11 + seconds)
    wf.end += seconds
    return wf


def test_tool_wait_keeps_workflow_live_unlocked_and_evictable():
    wf = waiting_workflow()
    other = mk("other", [("Writer", 1)])
    other.calls[0] = replace(other.calls[0], segments=(("manual", 100), ("other-1", 200)), prompt_tokens=200)
    markov = MarkovPredictor().fit([wf.roles(), other.roles()])
    sim = Replay([wf, other], ReplayConfig("lru", "none", 2, 200), markov, DurationModel.fit([wf, other]))
    sim.on_admit(None)
    sim.push(15, "admit", None)
    inspected = False
    while sim.heap:
        sim.now, _, kind, payload = heapq.heappop(sim.heap)
        getattr(sim, f"on_{kind}")(payload)
        if kind == "call_end" and payload[0] == "wait" and payload[1].call_index == 1:
            assert sim.now == 10
            assert sim.live["wait"].timeline[1] == ("Reader", 40)
            nodes = [n for n in sim.tree._resident if "wait" in n.roles_by_wf]
            assert nodes and all(n.lock == 0 for n in nodes)
            assert sim.tree.select(1, sim.key_fn("lru"))
            inspected = True
        if kind == "call_start" and payload[0] == "other":
            assert "wait" in sim.live
            assert sim.events and sim.events[-1]["victims"]
    assert inspected
    assert [e[0] for e in sim.access_log if e[1] == "wait"] == [0, 40]


def test_zero_wait_is_identical_to_existing_replay():
    wf = waiting_workflow(0)
    plain = replace(wf, tools=[])
    m = MarkovPredictor().fit([wf.roles()])
    d = fixtures()[2]
    cfg = ReplayConfig("he_urg", "markov", 1, 200, tau=5)
    assert replay([wf], cfg, m, d) == replay([plain], cfg, m, d)


def test_embedded_reader_tool_offset_is_preserved():
    wf = mk("reader", [("Decomposer", 10), ("Reader", 5)])
    wf.calls[1] = replace(wf.calls[1], tool_start=10, tool_end=12, submit=12, first_token=12.1)
    m = MarkovPredictor().fit([wf.roles()])
    sim = Replay([wf], ReplayConfig("lru", "none", 1, None), m, DurationModel.fit([wf]))
    sim.on_admit(None)
    while sim.heap:
        sim.now, _, kind, payload = heapq.heappop(sim.heap)
        getattr(sim, f"on_{kind}")(payload)
    assert [e[0] for e in sim.access_log] == [0, 12]


@pytest.mark.parametrize("with_tools", [False, True])
def test_access_sequence_is_policy_independent(with_tools):
    wfs, m, d = fixtures()
    if with_tools:
        wfs = [waiting_workflow()] + wfs
        d = DurationModel.fit(wfs)
    logs = []
    for policy in ("lru", "lae", "he_step", "he_urg", "cand", "ref_offline"):
        cfg = ReplayConfig(policy, "markov", 4, 500, tau=5, step_len=1)
        logs.append(replay(wfs, cfg, m, d)["access_log"])
    assert all(log == logs[0] for log in logs)


def test_shadow_does_not_consume_primary_rng():
    wfs, m, d = fixtures()
    cfg = ReplayConfig("he_urg", "markov", 4, 700, tau=5)
    plain = replay(wfs, cfg, m, d)
    shadow = replay(wfs, replace(cfg, shadow=("he_urg", "he_urg_oracleT")), m, d)
    assert [e["victims"] for e in plain["events"]] == [e["victims"] for e in shadow["events"]]
    assert all(shadow["summary"][k] == v for k, v in plain["summary"].items())


def test_cross_process_hash_seeds_are_identical():
    code = '''import json
from pbkv_sim.tests.test_replay import fixtures
from pbkv_sim.replay import ReplayConfig,replay
w,m,d=fixtures()
out=[]
for policy,shadow in [('he_step',()),('he_urg',()),('he_step',('he_urg','he_urg_oracleT'))]:
 out.append(replay(w,ReplayConfig(policy,'markov',6,900,tau=5,seed=2,shadow=shadow),m,d)['summary'])
print(json.dumps(out,sort_keys=True))'''
    outputs = [subprocess.check_output([sys.executable, "-c", code],
               env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True) for seed in (0, 1)]
    assert outputs[0] == outputs[1]


def test_offline_three_nodes_two_slots_matches_hand_calculation():
    seq = ["a", "b", "c", "a", "b"]
    nxt = build_next_use([(t, "w", t, ((key,),)) for t, key in enumerate(seq)])
    assert nxt.after(("a",), 0) == 3
    assert nxt.after(("a",), 3) == float("inf")
    tree, victims, misses = PrefixTree(2), [], 0
    for t, key in enumerate(seq):
        def need(n):
            gone = tree.select(n, lambda node: (-nxt.after(node.prefix_key, t), node.last_access))
            victims.append([v.key for v in gone])
            tree.apply(gone, t)
        acc = tree.access("w", "Reader", ((key, 1),), t, need)
        misses += acc.refetch_tokens
        tree.release(acc.path, t)
    assert victims == [["b"], ["c"]]
    assert misses == 1


def test_candidate_hand_calculation_union_and_infinite_tau():
    r = DEFAULT_ROLES.index
    p = np.zeros((2, len(r)))
    p[:, r["Reader"]] = 0.5
    p[:, r["Writer"]] = 0.5
    value = candidate_score({"w": {"Reader"}}, {"w": p}, {"w": np.array([0.8, 0.6])})
    assert value == pytest.approx(0.5 * 0.8 + 0.25 * 0.6)
    assert candidate_score({"w": {"Reader"}}, {"w": p}, {"w": np.ones(2)}) == pytest.approx(0.75)
    assert candidate_score({"w": {"Reader"}, "z": {"Reader"}}, {"w": p, "z": p},
                           {"w": np.ones(2), "z": np.ones(2)}) == pytest.approx(1 - 0.25**2)


def test_oracle_candidate_keeps_soon_prefix_when_step_scores_tie():
    p = oracle_rows(["Reader"], 1)
    assert workflow_vector(p, 0.5)[DEFAULT_ROLES.index["Reader"]] == 1
    tree = PrefixTree(2)
    for wid in ("soon", "late"):
        acc = tree.access(wid, "Reader", ((wid, 1),), 0, None)
        tree.release(acc.path, 0)
    pred = {w: p for w in ("soon", "late")}
    discounts = {"soon": np.exp(-np.array([1.0]) / 10), "late": np.exp(-np.array([30.0]) / 10)}
    gone = tree.select(1, lambda node: (candidate_score(node.roles_by_wf, pred, discounts), node.id))
    assert [n.key for n in gone] == ["late"]


def test_wait_model_conditions_on_elapsed():
    wf = waiting_workflow()
    d = DurationModel.fit([wf])
    out = d.tool_draw("test", None, np.random.default_rng(1), 10, elapsed=12)
    np.testing.assert_allclose(out, 18)
    discount = candidate_discounts(d, oracle_rows(["Reader"], 1), "Decomposer", False,
        22, 1, 10, np.random.default_rng(1), 256, waiting_tools=tuple(wf.tools), wait_elapsed=12)
    assert discount[0] == pytest.approx(np.exp(-1.8))


def test_exact_time_expectation_matches_closed_form_and_large_mc():
    d = model()
    p = oracle_rows(["Writer", "Decomposer"], 2)
    exact = exact_candidate_discounts(d, p, "Verifier", False, 0, 1, 10)
    np.testing.assert_allclose(exact, np.exp(-np.array([1, 3]) / 10))
    sampled = candidate_discounts(d, p, "Verifier", False, 0, 1, 10,
                                  np.random.default_rng(0), 100_000)
    np.testing.assert_allclose(exact, sampled, atol=1e-10)
    x, mass = service_distribution(d, "Reader", False, 2)
    np.testing.assert_allclose(x, [1, 3])
    np.testing.assert_allclose(mass, [.25, .75])
    p = oracle_rows(["Writer"], 1)
    exact = exact_candidate_discounts(d, p, "Reader", False, 2, 2, 10)
    np.testing.assert_allclose(exact, [np.exp(-.1)])


def test_exact_tool_wait_expectation_matches_enumeration():
    wf = waiting_workflow()
    d = DurationModel.fit([wf])
    d.tool_samples[("test", None)] = np.array([20., 40.])
    p = oracle_rows(["Reader"], 1)
    exact = exact_candidate_discounts(d, p, "Decomposer", False, 0, 1, 10)
    assert exact[0] == pytest.approx(np.exp(-1) * (np.exp(-2) + np.exp(-4)) / 2)
    remaining = exact_candidate_discounts(d, p, "Decomposer", False, 22, 1, 10,
                                        waiting_tools=tuple(wf.tools), wait_elapsed=22)
    assert remaining[0] == pytest.approx(np.exp(-1.8))


def test_completed_step_has_zero_remaining_service_before_output_admission():
    p = oracle_rows(["Writer"], 1)
    assert exact_candidate_discounts(model(), p, "Verifier", True, 1, 1, 10,
                                     current_complete=True)[0] == 1
    wfs, m, d = fixtures()

    class Inspect(Replay):
        checked = False

        def on_need(self, need):
            live = self.live.get("f0")
            if live is not None and live.idx == 0 and self.now == 1:
                assert live.step_finished and live.running == 0
                self.checked = True
            return super().on_need(need)

    sim = Inspect(wfs[:1], ReplayConfig("he_step", "markov", 1, 150), m, d)
    sim.on_admit(None)
    while sim.heap:
        sim.now, _, kind, payload = heapq.heappop(sim.heap)
        getattr(sim, f"on_{kind}")(payload)
    assert sim.checked


def test_trace_loads_tools_and_rejects_bad_timestamps(tmp_path):
    rows = [call("w", "Decomposer", 1, 1, 0, 10), call("w", "Reader", 2, 1, 40, 1),
            {"event": "tool", "workflow_id": "w", "after_call_index": 1, "tool_type": "test",
             "queue_start": 10, "exec_start": 12, "exec_end": 40},
            {"event": "workflow_end", "workflow_id": "w", "setting": "default", "decode_end": 41}]
    assert load_workflows(write(tmp_path, rows))["w"].delay_after(load_workflows(write(tmp_path, rows))["w"].steps()[0]) == 30
    rows[2]["exec_end"] = 9
    with pytest.raises(ValueError, match="timestamps"):
        load_workflows(write(tmp_path, rows))


def test_custom_roles_and_bank_order_validation(tmp_path):
    table = RoleTable(("Planner", "Coder", "Committer", "Tester", "Analyzer", "END"))
    m = MarkovPredictor(roles=table).fit([["Planner", "Coder", "Committer", "Tester", "Analyzer", "Coder"]])
    assert m.predict("Planner", 3).shape == (3, 6)
    path = tmp_path / "roles.json"
    path.write_text(json.dumps({"roles": list(table.roles)}))
    assert RoleTable.from_metadata(path) == table
    path.write_text(json.dumps({"schema": 1, "roles": list(table.roles), "provenance": {},
        "checkpoint_sha256": "test", "probabilities": {}}))
    assert PredictionBank(path, table).roles == table
    with pytest.raises(ValueError, match="role order"):
        PredictionBank(path, DEFAULT_ROLES)
