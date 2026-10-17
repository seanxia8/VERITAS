"""R-2 acceptance: D-TQ walk a+b/sqrt(q), D-Q overflow clamp, D-J early-digit count.

Tolerances declared before the first run: conditional means within 3 sigma_MC
(sigma_MC = configured width / sqrt(N)); early-digit fraction within 6 binomial
SE. Seed 71, 20000 channels per charge point. Each charge point uses its own
event ID, so its per-channel jitter draws are independent of the other points.
Synthetic validation only.
"""
from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import norm
from noise_module_lucid import response_manifest
from noise_module_lucid.lucid_controls import paired_checks, passed
from noise_module_lucid.photon_contract import PhotonBatch
from noise_module_lucid.quantum_efficiency import QECurve
from noise_module_lucid.readout import readout_chain
from noise_module_lucid.timing_response import InverseSqrtWalk

N = 20000


def flat(n):
    """Manifest response with unit detection, unit group factors and no dark counts (tested in
    test_dark_counts.py), so charge is set by gain alone."""
    p = response_manifest.reference_response(n, np.zeros(n, dtype=int))
    return replace(p, qe_curve=QECurve(p.qe_curve.wavelength_nm, (1.0,)*len(p.qe_curve.wavelength_nm)), collection_efficiency=1.0, spe_sigma=0.0, dark_rate_hz=0.0)


def event_for(q):
    return 1000 + int(round(100*q))


def one_photon_per_channel(n=N, event=71):
    return PhotonBatch(event, np.arange(n), np.arange(n), np.zeros(n), np.full(n, 400.0))


def read(b, p, seed=71):
    return readout_chain(b.sensor_id, b.arrival_time_ns, int(b.sensor_id.max())+1, p, seed, photon_batch=b)


def raw_view(p):
    """Zero reference walk: digit time = arrival + configured walk + jitter."""
    zero = replace(p.reference_walk, b_ns=0.0)
    return replace(p, reference_walk=zero)


@pytest.mark.parametrize("q", [0.25, 1, 2, 5, 20])
def test_conditional_mean_follows_configured_curve(q):
    p = raw_view(replace(flat(N), gain=float(q)))
    d = read(one_photon_per_channel(event=event_for(q)), p)
    assert len(d) == N and np.all(d.charge == q)
    sigma_mc = float(p.timing_width(q)) / np.sqrt(N)
    configured = float(p.walk_curve(q))
    assert configured == pytest.approx(2/np.sqrt(q))
    assert abs(d.time.mean() - configured) < 3*sigma_mc


def test_walk_is_constant_above_saturation_and_rejects_below_domain():
    w = response_manifest.reference_response(1, np.zeros(1, dtype=int)).walk_curve
    np.testing.assert_allclose(w([50, 100, 1000]), 2/np.sqrt(50))
    assert w(0.25) == pytest.approx(4) and w(16) == pytest.approx(0.5)
    with pytest.raises(ValueError): w([0.2])
    with pytest.raises(ValueError): w([1000.5])
    for bad in (dict(q_sat_pe=2000.0), dict(q_sat_pe=0.1), dict(domain_pe=(0.0, 1000.0)), dict(b_ns=np.nan)):
        with pytest.raises(ValueError): InverseSqrtWalk(**{**dict(a_ns=0.0, b_ns=2.0, q_sat_pe=50.0, domain_pe=(0.25, 1000.0)), **bad})


def test_slope_change_is_charge_dependent_and_preserves_charge_membership():
    n = 400
    b = PhotonBatch(5, np.arange(n*6), np.repeat(np.arange(n), 6) % n, np.zeros(n*6), np.full(n*6, 400.0))
    p = response_manifest.reference_response(n, np.arange(n) // 19)
    a = read(b, p, 5)
    up = read(b, replace(p, walk_curve=p.walk_curve.scaled(1.2)), 5)
    np.testing.assert_array_equal(a.sensor, up.sensor)
    np.testing.assert_array_equal(a.charge, up.charge)
    expected = 0.2*2.0/np.sqrt(np.minimum(a.charge, 50.0))
    np.testing.assert_allclose(up.time - a.time, expected, atol=1e-12, rtol=0)
    assert np.ptp(up.time - a.time) > 0.1
    offset = read(b, replace(p, walk_curve=p.walk_curve.scaled(offset_ns=1.0)), 5)
    np.testing.assert_allclose(offset.time - a.time, 1.0, atol=1e-12, rtol=0)


def test_zero_change_identity():
    b = one_photon_per_channel(2000)
    p = replace(flat(2000), gain=3.0)
    a = read(b, p)
    for same in (p.walk_curve.scaled(1.0), p.walk_curve.scaled(offset_ns=0.0)):
        d = read(b, replace(p, walk_curve=same))
        for k in ("sensor", "charge", "time"):
            np.testing.assert_array_equal(getattr(a, k), getattr(d, k))


def test_overflow_is_clamped_and_counted_not_raised():
    b = one_photon_per_channel(10)
    gain = np.where(np.arange(10) < 3, 2500.0, 10.0)
    d = read(b, replace(flat(10), gain=gain))
    assert len(d) == 10 and d.counts["overflow_clamped"] == 3
    np.testing.assert_array_equal(d.charge, np.where(np.arange(10) < 3, 1000.0, 10.0))
    legacy = replace(flat(10), gain=gain, charge_domain_pe=None)
    with pytest.raises(ValueError, match="outside calibrated domain"):
        read(b, legacy)


@pytest.mark.parametrize("q", [1, 5])
def test_digits_before_arrival_fraction_matches_gaussian(q):
    p = replace(flat(N), gain=float(q))
    d = read(one_photon_per_channel(event=event_for(q)), p)
    expected = norm.cdf(-float(p.walk_curve(q)) / float(p.timing_width(q)))
    se = np.sqrt(expected*(1-expected)/N)
    assert abs(d.counts["digits_before_arrival"]/len(d) - expected) < 6*se


def test_paired_real_photon_control_on_synthetic_batch():
    n = 300
    b = PhotonBatch(9, np.arange(n*4), np.arange(n*4) % n, np.linspace(0, 3, n*4), np.linspace(250, 750, n*4))
    p = response_manifest.reference_response(n, np.arange(n) // 19)
    row = paired_checks(lambda params, batch: read(batch, params, 9), p, b)
    assert passed([row]) and row["n_digits"] > 0
    assert {"tq_slope_1.2_shift_matches_curve", "tq_slope_0.8_charge_dependent", "tq_offset_control_time_plus_offset"} <= set(row)


def test_paired_control_rejects_an_offset_posing_as_slope(monkeypatch):
    n = 300
    b = PhotonBatch(9, np.arange(n*4), np.arange(n*4) % n, np.zeros(n*4), np.full(n*4, 400.0))
    p = response_manifest.reference_response(n, np.arange(n) // 19)
    monkeypatch.setattr(InverseSqrtWalk, "scaled", lambda self, slope_factor=1.0, offset_ns=0.0:
                        replace(self, a_ns=self.a_ns + offset_ns + (slope_factor - 1)))
    row = paired_checks(lambda params, batch: read(batch, params, 9), p, b)
    assert not row["tq_slope_1.2_shift_matches_curve"] and not row["tq_slope_1.2_charge_dependent"]
    assert not passed([row])


def test_threshold_below_charge_domain_is_counted_not_raised():
    """E26: a sub-0.25 PE threshold rejects and counts below-domain digits instead of raising."""
    b = one_photon_per_channel(10)
    gain = np.where(np.arange(10) < 4, 0.2, 2.0)
    p = replace(flat(10), gain=gain, threshold_pe=0.1)
    d = read(b, p)
    assert d.counts["below_charge_domain"] == 4 and len(d) == 6
    np.testing.assert_array_equal(d.sensor, np.arange(4, 10))
    assert np.all(d.charge >= 0.25)
    at_domain = read(b, replace(p, threshold_pe=0.25))
    assert at_domain.counts["below_charge_domain"] == 0 and at_domain.counts["lost_threshold"] == 4
