"""kappa_m precision-study arithmetic on synthetic errors."""
import numpy as np
import pytest
from noise_module_lucid.precision_study import kappa_from_sd, null_k, placement_errors, summarise


def test_null_k_uses_disjoint_pairs():
    e = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [2.0, 2.0], [9.0, 9.0]])  # odd R: last row unused
    np.testing.assert_allclose(null_k(e), [2.0, 0.5])
    with pytest.raises(ValueError): null_k(np.ones((1, 3)))


def test_kappa_rule():
    assert kappa_from_sd(0.01) == 1.10 and kappa_from_sd(0.08) == pytest.approx(1.16)


def test_identical_realisations_give_zero_spread_and_floor():
    v = np.ones((3, 8, 50)); e = 0.1*np.ones((3, 8, 50))
    s = summarise(v, e)
    assert s["K_vertex"]["sd"] == 0 and s["K"]["mean"] == 1 and s["kappa_m"] == 1.10 and s["K"]["n_pairs"] == 12


def test_noise_spread_matches_expectation():
    rng = np.random.default_rng(0)
    v = 0.3 + 0.1*rng.standard_normal((3, 64, 400)); e = 0.1 + 0.02*rng.standard_normal((3, 64, 400))
    s = summarise(np.abs(v), np.abs(e))
    # mean of 400 errors: relative SD ~ (0.1/0.3)/20; ratio of two such means: x sqrt(2)
    assert s["K_vertex"]["sd"] == pytest.approx(np.sqrt(2)*(0.1/0.3)/20, rel=0.3)


def test_placement_errors():
    t = np.zeros((2, 7)); p = t.copy(); p[:, 3] = [0.3, 0.4]; p[:, 6] = np.log(1.1)
    v, e = placement_errors(p, t)
    np.testing.assert_allclose(v, [0.3, 0.4]); np.testing.assert_allclose(e, [0.1, 0.1])
