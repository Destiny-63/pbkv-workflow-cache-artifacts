import numpy as np

from pbkv_sim.eta import DurationModel, oracle_tables, oracle_time_tables, rank_urgency, urgency_tables
from pbkv_sim.predictors import END_ID, ROLE_INDEX


def model():
    return DurationModel(
        samples={
            ("Verifier", False): np.array([1.0]),
            ("Verifier", True): np.array([20.0]),
            ("Writer", False): np.array([2.0]),
            ("Decomposer", False): np.array([1.0]),
            ("Reader", False): np.array([1.0, 3.0]),
            ("Aggregator", False): np.array([1.0]),
        },
        reader_group=np.array([2]),
    )


def test_conditional_not_marginal():
    w, d = ROLE_INDEX["Writer"], ROLE_INDEX["Decomposer"]
    paths = np.array([[w, END_ID]] * 50 + [[d, ROLE_INDEX["Reader"]]] * 50)
    rng = np.random.default_rng(0)
    U, ET = urgency_tables(model(), paths, "Verifier", True, 5.0, 1, 10.0, rng)
    assert abs(ET[0, w] - 15.0) < 1e-9
    assert abs(U[0, w] - np.exp(-1.5)) < 1e-9
    assert abs(ET[1, ROLE_INDEX["Reader"]] - 16.0) < 1e-9


def test_tau_none_gives_ones():
    paths = np.array([[ROLE_INDEX["Writer"], END_ID]])
    U, _ = urgency_tables(model(), paths, "Verifier", False, 0.0, 1, None, np.random.default_rng(0))
    assert U[0, ROLE_INDEX["Writer"]] == 1.0


def test_reader_group_takes_max():
    draws = model().draw("Reader", False, np.random.default_rng(0), 2000)
    assert set(np.unique(draws)) <= {1.0, 3.0}
    assert abs((draws == 3.0).mean() - 0.75) < 0.05


def test_remaining_conditions_on_elapsed():
    rem = model().remaining("Reader", False, 2.0, np.random.default_rng(0), 100)
    assert np.allclose(rem, 1.0)


def test_oracle_tables_and_rank():
    U, ET = oracle_tables([("Writer", 13.0), ("END", 15.0)], now=10.0, k_max=3, tau=10.0)
    assert ET[0, ROLE_INDEX["Writer"]] == 3.0 and abs(U[0, ROLE_INDEX["Writer"]] - np.exp(-0.3)) < 1e-12
    assert U[1].sum() == 0
    et = {"a": np.array([[1.0, 9.0]]), "b": np.array([[5.0, 0.0]])}
    mask = {"a": np.array([[True, True]]), "b": np.array([[True, False]])}
    r = rank_urgency(et, mask)
    assert r["a"][0, 0] > r["b"][0, 0] > r["a"][0, 1]
    assert r["b"][0, 1] == 0.0


def test_oracle_time_tables_carry_time_but_not_role():
    U = oracle_time_tables([("Writer", 13.0), ("END", 15.0)], now=10.0, k_max=3, tau=10.0, step_len=2.0)
    assert np.allclose(U[0], np.exp(-0.3))
    assert np.allclose(U[1], np.exp(-0.5)) and np.allclose(U[2], np.exp(-0.7))
    assert U[0].min() == U[0].max()
