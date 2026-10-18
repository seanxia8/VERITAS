"""D-E development endpoint on analytic tracks: E_dep = T(entry or vertex) - T(exit), 0 exit if stopped."""
import numpy as np
import pytest
from noise_module_lucid.deposited_energy import M_MU_MEV, deposited_energy, kinetic_from_beta

R, H = 1.0, 2.0


def beta_of(t):
    g = 1 + np.asarray(t)/M_MU_MEV
    return np.sqrt(1 - 1/g**2)


def straight(x0, x1, t0, t_end, n=200, axis=0):
    """Segments along one axis with T linear in path length from t0 to t_end (t_end=0: stops)."""
    x = np.linspace(x0, x1, n+1)
    p = np.zeros((n+1, 3)); p[:, axis] = x
    t = np.linspace(t0, t_end, n+1)
    return p[:-1], p[1:], beta_of(t[:-1])


def test_kinetic_from_beta_round_trip():
    np.testing.assert_allclose(kinetic_from_beta(beta_of([1, 100, 500])), [1, 100, 500], rtol=1e-12)
    assert kinetic_from_beta(0.0) == 0.0
    with pytest.raises(ValueError): kinetic_from_beta(1.0)


def test_escaping_track_deposits_entry_minus_exit():
    s, e, b = straight(0.0, 2.0, 500.0, 300.0)          # leaves at x = 1, where T = 400
    r = deposited_energy(s, e, b, 500.0, R, H)
    assert r["exited"] and r["t_entry_mev"] == 500.0
    assert r["t_exit_mev"] == pytest.approx(400.0, abs=1e-6) and r["e_dep_mev"] == pytest.approx(100.0, abs=1e-6)


def test_stopped_track_deposits_everything():
    s, e, b = straight(0.0, 0.8, 200.0, 0.0)
    r = deposited_energy(s, e, b, 200.0, R, H)
    assert r["stopped_inside"] and not r["exited"] and r["t_exit_mev"] == 0.0 and r["e_dep_mev"] == 200.0


def test_vertex_uses_generator_energy_not_first_beta():
    s, e, b = straight(0.0, 2.0, 500.0, 300.0)
    r = deposited_energy(s, e, b, 501.0, R, H)
    assert r["t_entry_mev"] == 501.0 and r["e_dep_mev"] == pytest.approx(101.0, abs=1e-6)


def test_vertex_outside_enters_through_the_cap():
    s, e, b = straight(-2.0, 0.5, 600.0, 350.0, axis=2)   # enters at z = -1 where T = 500
    r = deposited_energy(s, e, b, 600.0, R, H)
    # The record ends inside, so by convention the track stopped there: E_dep = T(entry).
    assert r["entries"] == 1 and r["t_entry_mev"] == pytest.approx(500.0, abs=1e-6)
    assert r["stopped_inside"] and r["e_dep_mev"] == pytest.approx(500.0, abs=1e-6)


def test_reentry_is_counted_not_added():
    # Out through x=+1, back in, out again: only the first exit counts.
    p = np.array([[0, 0, 0], [1.5, 0, 0], [0.5, 0, 0], [2.0, 0, 0]], float)
    t = np.array([400.0, 250.0, 200.0, 100.0])
    s, e = p[:-1], p[1:]
    r = deposited_energy(s, e, beta_of(t[:-1]), 400.0, R, H)
    assert r["entries"] == 2 and r["exited"]
    assert r["t_exit_mev"] == pytest.approx(400.0 + (1/1.5)*(250.0 - 400.0), abs=1e-6)


def test_bad_segments_rejected():
    s, e, b = straight(0.0, 1.0, 100.0, 0.0, n=4)
    with pytest.raises(ValueError, match="continuous"): deposited_energy(s[[0, 2, 1, 3]], e[[0, 2, 1, 3]], b, 100.0, R, H)
    with pytest.raises(ValueError, match="one beta"): deposited_energy(s, e, b[:2], 100.0, R, H)
