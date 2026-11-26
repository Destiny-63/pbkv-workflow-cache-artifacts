import json

import numpy as np
import pytest

from pbkv_sim.learned import PredictionBank
from pbkv_sim.predictors import END_ID, N_ROLES, MarkovPredictor
from pbkv_sim.replay import Live, Replay, ReplayConfig, replay
from pbkv_sim.eta import DurationModel
from pbkv_sim.tests.test_replay import fixtures


def bank_file(tmp_path, wfs):
    p = np.zeros((3, N_ROLES))
    p[:, END_ID] = 1
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"schema": 1, "provenance": {}, "checkpoint_sha256": "test",
        "probabilities": {f"{w.setting}|{w.workflow_id}|{i}": p.tolist()
                          for w in wfs for i, _ in enumerate(w.steps())}}))
    return path, p


def test_hidden_prediction_is_unavailable_until_prefill(tmp_path):
    wfs, markov, durations = fixtures()
    path, p = bank_file(tmp_path, wfs)
    sim = Replay(wfs, ReplayConfig("he_step", "learned", 1, None), markov,
                 durations, PredictionBank(path))
    wf = wfs[0]
    live = Live(wf, wf.steps())
    sim.live[wf.workflow_id] = live
    sim.start_step(live)
    # Prefill has not finished, so the bank's END prediction must not be used.
    np.testing.assert_allclose(live.p, markov.predict("Decomposer", 3))
    ready = next(e for e in sim.heap if e[2] == "prediction_ready")
    assert ready[0] == pytest.approx(wf.calls[0].first_token - wf.steps()[0].start)
    assert ready[0] > 0
    sim.on_prediction_ready(ready[3])
    np.testing.assert_array_equal(live.p, p)


def test_all_steps_receive_predictions_in_real_replay(tmp_path):
    wfs, markov, durations = fixtures()
    path, _ = bank_file(tmp_path, wfs)
    result = replay(wfs, ReplayConfig("he_step", "learned", 4, None), markov,
                    durations, PredictionBank(path))
    assert result["summary"]["learned_updates"] == sum(len(w.steps()) for w in wfs)
    assert result["summary"]["workflows"] == len(wfs)
    assert result["summary"]["refetch_tokens"] == 0


def test_missing_and_corrupt_predictions_fail_loudly(tmp_path):
    wfs, _, _ = fixtures()
    path, _ = bank_file(tmp_path, wfs[:1])
    with pytest.raises(KeyError, match="No learned prediction"):
        PredictionBank(path).predict(wfs[1], 0, 3)
    with pytest.raises(ValueError, match="horizon"):
        PredictionBank(path).predict(wfs[0], 0, 4)
    content = json.loads(path.read_text())
    first = next(iter(content["probabilities"]))
    content["probabilities"][first][0][0] = 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Invalid probability"):
        PredictionBank(path)


def test_learned_mode_cannot_silently_run_uniform():
    with pytest.raises(ValueError, match="prediction bank"):
        Replay([], ReplayConfig("he_step", "learned", 1, None), MarkovPredictor(),
               DurationModel({}, np.ones(1)))
