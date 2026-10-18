"""R-6 acceptance: keyed dark-count stage and dark-noise-only readout windows.

Tolerances declared before the first run: dark-hit counts within 3 Poisson sigma
of rate x window x n_PMT. The rate-0 golden hash was produced by the pre-R-6 code
(commit 0fba034) on the same batch. Synthetic validation only.
"""
import hashlib
from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import kstest
from noise_module_lucid import response_manifest
from noise_module_lucid.photon_contract import PhotonBatch
from noise_module_lucid.readout import ReadoutParams, dark_noise_window, readout_chain

N = 1995
GOLDEN_0FBA034 = "161563c89f812817362c12c8c7f15c9d073ba45b66bec59854ca8e20d20971bb"


def reference():
    return response_manifest.reference_response(N, response_manifest.module_groups(np.arange(N)//19))


def physics_batch(event=424242):
    rng = np.random.default_rng(20261018)
    sensor = rng.integers(0, N, 30000)
    return PhotonBatch(event, np.arange(30000), sensor, rng.exponential(5, 30000), rng.uniform(250, 750, 30000))


def read(b, p, seed=7):
    return readout_chain(b.sensor_id, b.arrival_time_ns, N, p, seed, photon_batch=b)


def digest(d):
    h = hashlib.sha256()
    for a in (d.sensor, d.charge, d.time, d.n_pe_true):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_manifest_declares_synthetic_dark_rate_and_window():
    m = response_manifest.load_manifest()
    assert m["parameters"]["dark_rate_hz"]["value"] == 1000.0 and m["parameters"]["dark_rate_hz"]["status"] == "synthetic_dev"
    assert m["stage_owners"]["dark_counts"].startswith("readout")
    p = reference()
    assert p.dark_rate_hz == 1000.0 and p.readout_window_ns == (-200.0, 800.0)


def test_rate_zero_is_identical_to_the_pre_r6_reference():
    d = read(physics_batch(), replace(reference(), dark_rate_hz=0.0))
    assert digest(d) == GOLDEN_0FBA034
    assert "dark_pe" not in d.counts and np.all(d.n_dark_true == 0)


def test_dark_hit_count_matches_rate_window_and_pmts():
    p = reference()
    windows = 200
    total = sum(dark_noise_window(p, N, seed=3, run=11, window=k).counts["dark_pe"] for k in range(windows))
    expected = p.dark_rate_hz * (p.readout_window_ns[1] - p.readout_window_ns[0]) * 1e-9 * N * windows
    assert abs(total - expected) < 3*np.sqrt(expected)


def test_dark_times_uniform_in_window_at_high_rate():
    p = replace(reference(), dark_rate_hz=2e5)
    from noise_module_lucid.readout import dark_photoelectrons
    sensor, t = dark_photoelectrons(N, p.dark_rate_hz, p.readout_window_ns, 3, 5)
    lo, hi = p.readout_window_ns
    expected = p.dark_rate_hz * (hi - lo) * 1e-9 * N
    assert abs(len(t) - expected) < 3*np.sqrt(expected)
    assert np.all((t >= lo) & (t <= hi)) and kstest((t - lo)/(hi - lo), "uniform").pvalue > 1e-3


def test_dark_only_window_contains_no_source_hits():
    d = dark_noise_window(replace(reference(), dark_rate_hz=2e4), N, seed=3, run=11)
    assert d.counts["photons_in"] == 0 and d.counts["photons_detected"] == 0 and len(d) > 0
    np.testing.assert_array_equal(d.n_pe_true, d.n_dark_true)
    assert d.counts["digits_with_dark"] == len(d)


def test_dark_draws_are_deterministic_and_keyed():
    p = replace(reference(), dark_rate_hz=2e4)
    a, b = dark_noise_window(p, N, seed=3, run=11), dark_noise_window(p, N, seed=3, run=11)
    for k in ("sensor", "charge", "time"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    for other in (dict(seed=4, run=11), dict(seed=3, run=12), dict(seed=3, run=11, window=1)):
        c = dark_noise_window(p, N, **other)
        assert not (np.array_equal(a.sensor, c.sensor) and np.array_equal(a.time, c.time))
    # A physics event with the same ID as the run never shares the dark window's draws.
    event = read(PhotonBatch(11, np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0)), p, 3)
    assert not (np.array_equal(a.sensor, event.sensor) and np.array_equal(a.time, event.time))


def test_dark_counts_do_not_shift_source_draws():
    b = physics_batch()
    clean, noisy = read(b, replace(reference(), dark_rate_hz=0.0)), read(b, replace(reference(), dark_rate_hz=5e4))
    assert clean.counts["photons_detected"] == noisy.counts["photons_detected"]
    pure = noisy.n_dark_true == 0
    common, ia, ib = np.intersect1d(clean.sensor, noisy.sensor[pure], return_indices=True)
    np.testing.assert_array_equal(clean.time[ia], noisy.time[pure][ib])
    np.testing.assert_array_equal(clean.charge[ia], noisy.charge[pure][ib])


def test_dark_digits_share_the_tq_walk():
    p = replace(reference(), dark_rate_hz=2e4)
    ref = dark_noise_window(p, N, seed=3, run=11)
    up = dark_noise_window(replace(p, walk_curve=p.walk_curve.scaled(1.2)), N, seed=3, run=11)
    np.testing.assert_array_equal(ref.sensor, up.sensor)
    np.testing.assert_allclose(up.time - ref.time, 0.2*p.walk_curve.b_ns/np.sqrt(np.minimum(ref.charge, 50.0)), atol=1e-9, rtol=0)


def test_invalid_dark_configurations_raise():
    with pytest.raises(ValueError, match="photon-stage batch"):
        readout_chain(np.array([0]), np.array([0.0]), 2, ReadoutParams(dark_rate_hz=1e3), 0)
    with pytest.raises(ValueError, match="readout_window_ns"):
        dark_noise_window(replace(reference(), readout_window_ns=None), N, seed=1, run=1)
    with pytest.raises(ValueError, match="dark_rate_hz > 0"):
        dark_noise_window(replace(reference(), dark_rate_hz=0.0), N, seed=1, run=1)
    with pytest.raises(ValueError, match="invalid dark rate"):
        read(physics_batch(), replace(reference(), dark_rate_hz=-1.0))


def test_dark_window_without_qe_and_integer_keys():
    p = ReadoutParams(dark_rate_hz=1e5, readout_window_ns=(0.0, 1000.0))
    d = dark_noise_window(p, 200, seed=1, run=2)
    assert d.counts["photons_in"] == 0 and d.counts["dark_pe"] > 0
    with pytest.raises(ValueError, match="integers"):
        dark_noise_window(p, 200, seed=1, run=2, window=1.7)


def test_time_feature_origin_is_the_window_not_the_first_digit():
    from noise_module_lucid.lucid_integration import TIME_FEATURE_UNIT_NS, time_feature
    # Author rule (18 Oct 2026): origin at the window start, fixed 20 ns unit.
    assert TIME_FEATURE_UNIT_NS == 20.0
    np.testing.assert_allclose(time_feature([-200.0, -180.0, 800.0], (-200.0, 800.0)), [0.0, 1.0, 50.0])
    # R-4 (E33) legacy unit: the window length.
    np.testing.assert_allclose(time_feature([-200.0, 300.0, 800.0], (-200.0, 800.0), None), [0.0, 0.5, 1.0])
    source = np.array([10.0, 20.0])
    with_dark = np.r_[-150.0, source]
    for unit in (20.0, None):
        np.testing.assert_array_equal(time_feature(with_dark, (-200, 800), unit)[1:], time_feature(source, (-200, 800), unit))
    with pytest.raises(ValueError): time_feature(source, (-200, 800), 0.0)


def test_diagnostic_time_origins():
    from noise_module_lucid.lucid_integration import time_feature
    t = np.array([3.0, 13.0, 43.0])
    # E40 diagnostic origins: trigger (t = 0), window start (the default), first digit (56179eb).
    np.testing.assert_allclose(time_feature(t, (-200.0, 800.0), origin="trigger"), t / 20)
    np.testing.assert_allclose(time_feature(t, (-200.0, 800.0)) - time_feature(t, (-200.0, 800.0), origin="trigger"), 10.0)
    # The first-digit origin is bit-identical to the 56179eb formula.
    np.testing.assert_array_equal(time_feature(t, (-200.0, 800.0), origin="first_digit"), (t - t.min()) / 20)
    assert time_feature(np.array([]), (-200.0, 800.0), origin="first_digit").size == 0
    with pytest.raises(ValueError, match="origin"): time_feature(t, (-200.0, 800.0), origin="vertex")


def test_nhits_time_origin_falls_back_to_the_window_start():
    from noise_module_lucid.lucid_integration import time_feature
    t = np.array([-150.0, 100.0, 101.0, 103.0])
    # The isolated early (dark-like) digit does not set the origin; the prompt cluster does.
    np.testing.assert_allclose(time_feature(t, (-200.0, 800.0), origin="nhits"), (t - 100.0) / 20)
    sparse = np.array([10.0, 500.0])
    np.testing.assert_allclose(time_feature(sparse, (-200.0, 800.0), origin="nhits"), (sparse + 200.0) / 20)
