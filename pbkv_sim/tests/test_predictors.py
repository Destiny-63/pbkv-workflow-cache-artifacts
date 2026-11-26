import numpy as np

from pbkv_sim.predictors import END_ID, ROLE_INDEX, MarkovPredictor, evaluate, oracle_rows, uniform_rows

SEQS = [
    ["Decomposer", "Reader", "Aggregator", "Verifier", "Writer"],
    ["Decomposer", "Reader", "Aggregator", "Verifier", "Decomposer", "Reader", "Aggregator", "Verifier", "Writer"],
]


def test_rows_are_distributions_and_survival_matches():
    m = MarkovPredictor(alpha=0.0).fit(SEQS)
    p = m.predict("Verifier", 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert abs(p[0, ROLE_INDEX["Writer"]] - 2 / 3) < 1e-9
    assert abs(p[0, ROLE_INDEX["Decomposer"]] - 1 / 3) < 1e-9
    assert abs(p[1, END_ID] - 2 / 3) < 1e-9


def test_sample_paths_follow_transitions():
    m = MarkovPredictor(alpha=0.0).fit(SEQS)
    paths = m.sample_paths("Aggregator", 2, 500, np.random.default_rng(0))
    assert paths.shape == (500, 2)
    assert (paths[:, 0] == ROLE_INDEX["Verifier"]).all()
    after_writer = paths[paths[:, 0] == ROLE_INDEX["Writer"]]
    assert (after_writer[:, 1] == END_ID).all() if len(after_writer) else True


def test_oracle_and_uniform_rows():
    o = oracle_rows(["Writer"], 3)
    assert o[0, ROLE_INDEX["Writer"]] == 1 and o[1, END_ID] == 1 and o[2, END_ID] == 1
    u = uniform_rows(2)
    assert np.allclose(u.sum(axis=1), 1.0) and (u[:, END_ID] == 0).all()


def test_evaluate_oracle_is_perfect():
    seq = SEQS[0]
    future = {tuple(seq[: i + 1]): seq[i + 1:] for i in range(len(seq))}

    def pred(history):
        return oracle_rows(future[tuple(history)], 3)

    res = evaluate(pred, [SEQS[0]], k_max=3)
    assert res["acc_1"] == 1.0 and res["acc_3"] == 1.0
    assert res["n_1"] == 5
