"""True time must add urgency without silently providing the true role."""
from dataclasses import replace
import json

import numpy as np

from pbkv_sim.eta import oracle_time_tables
from pbkv_sim.learned import PredictionBank
from pbkv_sim.replay import Replay, ReplayConfig, replay
from pbkv_sim.run_swe_comparison import GROUPS
from pbkv_sim.tests.test_learned import bank_file
from pbkv_sim.tests.test_replay import fixtures


def test_true_time_does_not_read_future_roles_and_matches_formula():
    wfs, m, d = fixtures()
    cfg = ReplayConfig("he_urg", "markov", 1, 500, gamma=.5, tau=10, oracle_time=True, step_len=2)
    sim = Replay(wfs, cfg, m, d)
    sim.on_admit(None)
    live = sim.live[wfs[0].workflow_id]
    sim.now = .2
    p = np.zeros((3, len(m.roles.roles)))
    reader, writer, end = (m.roles.index[r] for r in ("Reader", "Writer", "END"))
    p[:, reader], p[:, writer], p[:, end] = .4, .5, .1
    live.p = p
    u, _ = sim.urgency(live)
    np.testing.assert_allclose(u, np.broadcast_to(np.exp(-np.array([.8, 1.8, 2.8])/10)[:, None], u.shape))
    v = sim.vectors("he_urg")[wfs[0].workflow_id]
    expected = sum(.5**k * .9**k * .4 * np.exp(-(k+.8)/10) for k in range(3))
    assert np.isclose(v[reader], expected)
    live.timeline = [("Writer" if role != "END" else role, t) for role, t in live.timeline]
    np.testing.assert_array_equal(sim.urgency(live)[0], u)
    np.testing.assert_array_equal(sim.vectors("he_urg")[wfs[0].workflow_id], v)
    # The existing G1 padding rule is preserved even when the prediction has mass beyond END.
    tail = oracle_time_tables([("Reader", 3), ("END", 9)], 1, 3, 10, 2, m.roles)
    np.testing.assert_allclose(tail[:, reader], np.exp(-np.array([2, 4, 6])/10))


def test_five_groups_pair_workloads_and_never_estimate_eta(tmp_path, monkeypatch):
    wfs, m, d = fixtures()
    path, _ = bank_file(tmp_path, wfs)
    content = json.loads(path.read_text())
    for wf in wfs:
        for i, role in enumerate(wf.roles()):
            content["probabilities"][f"{wf.setting}|{wf.workflow_id}|{i}"] = m.predict(role, 3).tolist()
    path.write_text(json.dumps(content))
    bank = PredictionBank(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("True-time group attempted ETA sampling")
    monkeypatch.setattr(d, "draw", forbidden)
    monkeypatch.setattr(d, "remaining", forbidden)
    monkeypatch.setattr(m, "sample_paths", forbidden)
    base = replay(wfs, ReplayConfig("lae", "none", 4, None, drop_retired=True), m, d)
    logs = []
    assert [(g["group"], g["oracle_time"]) for g in GROUPS] == [
        ("step+predictor", False), ("step+markov", False), ("eta+predictor", True),
        ("eta+markov", True), ("oracle", True)]
    for group in GROUPS:
        cfg = ReplayConfig(group["policy"], group["predictor"], 4, 1500,
                           gamma=.5, tau=10, oracle_time=group["oracle_time"], step_len=2)
        out = replay(wfs, cfg, m, d, bank if group["predictor"] == "learned" else None)
        assert out["summary"]["overflow_tokens"] == 0
        assert out["access_log"] == base["access_log"]
        logs.append(out["access_log"])
