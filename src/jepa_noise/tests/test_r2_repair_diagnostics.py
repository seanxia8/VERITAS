"""Unit tests of the round-2 diagnostics renderer and tracer (runbook 2026-11-17, Part A)."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import abs_render as ar
from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_generator as gen
from jepa_noise import r2_geometry as rg


@pytest.fixture(scope="module")
def setup():
    try:
        geometry = rg.load_r2_geometry("wcte")
    except FileNotFoundError:
        pytest.skip("WCTE geometry not available")
    cfg = gen.load_r2_config()
    state = SimpleNamespace(gains=np.ones(geometry.n_pmt), attenuation_length_m=lb.LAMBDA0_M)
    return geometry, cfg, ar.PhotonBombRenderer(geometry, cfg, state)


def test_qe_quadrature_matches_mean_qe():
    cfg = gen.load_r2_config()
    nodes, w = ar.qe_quadrature(cfg)
    lam = np.linspace(cfg.spectrum_nm[0], cfg.spectrum_nm[1], 200001)
    assert abs(np.sum(w) - 1.0) < 1e-12
    assert abs(np.sum(nodes * w) - np.mean(lb.qe_at(lam, cfg))) < 1e-6
    assert abs(np.sum(nodes ** 2 * w) - np.mean(lb.qe_at(lam, cfg) ** 2)) < 1e-6


def test_renderer_mu_equals_generator_mu(setup):
    geometry, cfg, r = setup
    v = np.array([0.3, -0.2, 0.4])
    qe = np.full(geometry.n_pmt, 0.2)
    ref = lb.generator_reference(cfg)
    want = lb.expected_charge(geometry, v, 1e5, qe, lb.LAMBDA0_M, cfg) * lb.fixed_charge_gain(geometry, ref)
    base, _ = r.mu_noqe(torch.tensor([*v, np.log(1e5), 0.0], dtype=torch.float64))
    # Torch and NumPy round the distance before the small solid-angle subtraction.
    np.testing.assert_allclose(base.numpy() * 0.2, want, rtol=5e-12, atol=0)


def test_nll_is_minimal_near_truth_on_noise_free_like_event(setup):
    geometry, cfg, r = setup
    theta = torch.tensor([0.2, 0.1, -0.3, np.log(2e5), 5.0], dtype=torch.float64)
    eq, _, p_hit, base, d = r.moments(theta)
    hit = np.flatnonzero(p_hit.numpy() > 0.5)
    ev = ar.EventData(hit=hit, charge=eq.numpy()[hit],
                      time=r.predicted_time(theta, d).numpy()[hit])
    f0 = float(r.nll(theta, ev))
    for k, step in ((0, 0.05), (2, 0.05), (3, 0.1), (4, 1.0)):
        t = theta.clone()
        t[k] += step
        assert float(r.nll(t, ev)) > f0


def test_tracer_hits_sensor_straight_ahead():
    from jepa_noise.ext_r2_repair_diagnostic_helpers import trace_spheres
    geometry = SimpleNamespace(positions_m=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                               sensor_radius_m=0.1, radius_m=1.0, half_height_m=1.0,
                               shape=rg.SHAPE_CYLINDER)
    i, s, t = trace_spheres(geometry, np.zeros((3, 3)),
                            np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    assert i.tolist() == [0, 1] and s.tolist() == [0, 1]
    assert np.allclose(t, 0.9)
