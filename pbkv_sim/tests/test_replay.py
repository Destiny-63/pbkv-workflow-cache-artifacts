import numpy as np

from pbkv_sim.eta import DurationModel
from pbkv_sim.predictors import MarkovPredictor
from pbkv_sim.replay import ReplayConfig, replay
from pbkv_sim.trace import Call, Workflow


def mk(wid, roles_durs, slow=False):
    calls, t, cum = [], 0.0, 100
    segs = [("manual", 100)]
    for i, (role, dur) in enumerate(roles_durs, start=1):
        cum += 50
        segs = segs + [(f"{wid}-{i}", cum)]
        calls.append(Call(wid, "highdiff" if slow else "default", role, 1, i, t, t + 0.1, t + dur,
                          cum, 10, 0, None, None, slow and role == "Verifier", tuple(segs), None))
        t += dur
    return Workflow(wid, "highdiff" if slow else "default", slow, calls, t, "SUPPORTED")


def fixtures():
    fast = [mk(f"f{i}", [("Decomposer", 1), ("Reader", 1), ("Aggregator", 1), ("Verifier", 1), ("Writer", 1)])
            for i in range(6)]
    slow = [mk(f"s{i}", [("Decomposer", 1), ("Reader", 1), ("Aggregator", 1), ("Verifier", 20), ("Writer", 1)], True)
            for i in range(6)]
    wfs = fast + slow
    markov = MarkovPredictor().fit([w.roles() for w in wfs])
    return wfs, markov, DurationModel.fit(wfs)


def test_infinite_capacity_has_no_refetch_and_finishes_all():
    wfs, markov, dur = fixtures()
    out = replay(wfs, ReplayConfig("lru", "none", 4, None), markov, dur)
    s = out["summary"]
    assert s["workflows"] == 12 and s["refetch_tokens"] == 0 and s["evictions"] == 0
    assert len(out["calls"]) == sum(len(w.calls) for w in wfs)


def test_pressure_produces_evictions_and_shadow_logs():
    wfs, markov, dur = fixtures()
    cfg = ReplayConfig("he_step", "markov", 6, 900, tau=5.0, shadow=("he_urg",), seed=1)
    out = replay(wfs, cfg, markov, dur)
    assert out["summary"]["evictions"] > 0
    ev = out["events"][0]
    assert "he_urg" in ev["shadow"] and "diverge" in ev["shadow"]["he_urg"]


def test_peak_with_drop_retired_is_not_larger():
    wfs, markov, dur = fixtures()
    keep = replay(wfs, ReplayConfig("lae", "none", 4, None), markov, dur)["summary"]["peak_tokens"]
    drop = replay(wfs, ReplayConfig("lae", "none", 4, None, drop_retired=True), markov, dur)["summary"]["peak_tokens"]
    assert drop <= keep


def test_deterministic_given_seed():
    wfs, markov, dur = fixtures()
    cfg = ReplayConfig("he_urg", "markov", 6, 900, tau=5.0, seed=3)
    a = replay(wfs, cfg, markov, dur)["summary"]
    b = replay(wfs, cfg, markov, dur)["summary"]
    assert a == b


def test_oracle_time_shadow_runs():
    wfs, markov, dur = fixtures()
    cfg = ReplayConfig("he_step", "markov", 6, 900, tau=5.0, shadow=("he_urg", "he_urg_oracleT"), seed=1)
    out = replay(wfs, cfg, markov, dur)
    assert "diverge_rate_he_urg_oracleT" in out["summary"]
