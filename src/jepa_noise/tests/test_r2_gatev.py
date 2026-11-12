"""Addendum §2 tests for the Gate V photon-statistics ceiling."""
import numpy as np

from jepa_noise import cherenkov_validation as cv
from jepa_noise import ext_r2_gatev as g
from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_generator as gen
from jepa_noise import r2_geometry as rg


def _synthetic(n=2000, n_mod=100, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.gamma(0.7, 2.0, n)
    mu[rng.random(n) < 0.2] = 0.0
    module = np.repeat(np.arange(n_mod), n // n_mod)
    return mu, module, n_mod


def test_draw_keeps_lambda():
    mu, _, _ = _synthetic()
    for d in range(5):
        counts, detected = g.perfect_draw(mu, 31_248, 0.16, lb.keyed_rng(1, "t", d))
        assert int(counts.sum()) == 31_248
        assert np.allclose(detected, counts * 0.16)
        assert np.all(counts[mu == 0] == 0)


def test_perfect_model_scores_at_its_ceiling():
    mu, module, n_mod = _synthetic()
    lam, w = 20_000, 0.15
    c = g.event_ceiling(mu, lam, w, module, n_mod, detector="synthetic", event=0, k=200)
    # an observation that *is* a perfect-model draw (independent stream) lands inside p05..p95
    obs_counts, obs_det = g.perfect_draw(mu, lam, w, lb.keyed_rng(99, "obs"))
    s = g.score(mu, obs_counts, obs_det, module, n_mod)
    for m in g.METRICS:
        assert c[m]["p05"] <= s[m] <= c[m]["p95"]
        assert abs(s[m] / c[m]["c"] - 1.0) < 0.05
    # more photons -> a higher module ceiling (the log metric is not monotone: keep grows)
    c_hi = g.event_ceiling(mu, 50 * lam, w, module, n_mod, detector="synthetic", event=0, k=50)
    assert c_hi["module_pattern_pearson"]["c"] > c["module_pattern_pearson"]["c"]


def test_score_matches_compare_event_and_mu_reproduces_expected_detected():
    geometry = rg.load_r2_geometry("wcte")
    cfg = gen.load_r2_config()
    rng = np.random.default_rng(3)
    n_ph = 4000
    ev = cv.ValidationEvent(
        event_id=0, domain="muon", vertex_m=np.array([0.1, -0.2, 0.3]),
        sensor_id=rng.integers(0, geometry.n_pmt, n_ph), time_ns=rng.uniform(0, 30, n_ph),
        wavelength_nm=rng.uniform(300, 600, n_ph), direction_m=np.array([0.0, 0.6, 0.8]),
        energy_MeV=500.0, deposited_MeV=500.0, contained=True, track_length_m=2.15)
    row = cv.compare_event(geometry, cfg, ev)
    mu = g.expected_mu(geometry, cfg, ev)
    assert abs(mu.sum() - row["expected_detected"]) <= 1e-12 * row["expected_detected"]
    counts = np.bincount(ev.sensor_id, minlength=geometry.n_pmt).astype(float)
    detected = np.bincount(ev.sensor_id, weights=lb.qe_at(ev.wavelength_nm, cfg) * cfg.ce,
                           minlength=geometry.n_pmt)
    s = g.score(mu, counts, detected, np.asarray(geometry.module_index, np.int64),
                geometry.n_modules)
    for m in g.METRICS:
        assert abs(s[m] - row[m]) < 1e-12
