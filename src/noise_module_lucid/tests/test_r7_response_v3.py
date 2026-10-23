"""R-7 acceptance (author message, 23 Oct 2026): literature-proxy response v3.

The adopted values are proxies from J. Xia, *Upgrade and Calibration of
Super-Kamiokande's Inner Photodetectors* (MSc thesis, University of Tokyo, July
2019): 50 cm Super-K R3600 and Hyper-K R12860 20-inch PMTs, not WCTE's 3-inch
PMTs. They are labelled ``literature_proxy_hk_20in`` or ``literature_sk``.

Tolerances are fixed before the first run of this file:

* QE: the thesis table is reproduced exactly and interpolation is piecewise
  linear between the declared anchors;
* peak QE x CE within the required 0.25-0.29 band, and equal to 0.305 x 0.87;
* split-normal sample SDs within 3 standard errors of the table values, the
  early-half probability within 3 binomial SE of sigma_e/(sigma_e+sigma_l), the
  sample mean within 3 SE of the recorded mean shift, a Gaussian-KDE mode within
  0.5 ns of arrival + walk, and a Kolmogorov-Smirnov test of the whole sampled
  distribution against the analytic Eq. 6.3 CDF at p > 1e-3;
* the v2 manifest is still reproducible by its pinned file hash.

CPU only. Synthetic/proxy validation only.
"""
import numpy as np
import pytest
from scipy.stats import gaussian_kde, kstest
from dataclasses import replace

from noise_module_lucid import response_manifest
from noise_module_lucid.photon_contract import PhotonBatch
from noise_module_lucid.quantum_efficiency import QECurve
from noise_module_lucid.readout import dark_noise_window, readout_chain
from noise_module_lucid.timing_response import SplitNormalWidth, WidthCurve

N = 20000
SEED = 71

#: Fig. 4.3 of the thesis, HK R12860 average QE, exactly as the author transcribed it.
THESIS_QE = {
    280: 0.0, 300: 0.01, 320: 0.10, 340: 0.23, 350: 0.27, 370: 0.29, 400: 0.305,
    420: 0.29, 450: 0.25, 480: 0.19, 500: 0.165, 520: 0.13, 540: 0.07, 560: 0.045,
    600: 0.015, 650: 0.002, 700: 0.0,
}
#: Fig. 6.4(b)/6.3 of the thesis, sigma_e (early) and sigma_l (late), exactly as transcribed.
THESIS_WIDTH = {
    "charge_pe": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1000.0],
    "sigma_e_ns": [2.0, 1.53, 1.26, 1.04, 0.98, 0.86, 0.72, 0.65, 0.61, 0.75],
    "sigma_l_ns": [2.9, 1.82, 1.52, 1.76, 1.44, 1.13, 0.83, 0.60, 0.47, 0.49],
}


def reference(n=1995, groups=None, path=None):
    groups = np.arange(n) // 19 if groups is None else groups
    return response_manifest.reference_response(n, groups, manifest=path)


def one_photon_per_channel(n, q, *, arrival=0.0):
    return PhotonBatch(SEED, np.arange(n), np.arange(n), np.full(n, arrival), np.full(n, 400.0))


def read(b, p):
    n_channels = int(b.sensor_id.max()) + 1
    return readout_chain(b.sensor_id, b.arrival_time_ns, n_channels, p, SEED, photon_batch=b)


def flat_unit_detection(n):
    """Manifest v3 response with unit detection, zero charge smearing and no dark counts."""
    p = reference(n, np.zeros(n, dtype=int))
    return replace(p, qe_curve=QECurve(p.qe_curve.wavelength_nm, (1.0,) * len(p.qe_curve.wavelength_nm)),
                   collection_efficiency=1.0, spe_sigma=0.0, dark_rate_hz=0.0)


def raw_flat(n, q):
    """Unit detection, charge exactly ``q``, zero frozen reference walk: time = walk(q) + jitter."""
    p = flat_unit_detection(n)
    return replace(p, gain=float(q), reference_walk=replace(p.reference_walk, b_ns=0.0))


# --------------------------------------------------------------------- 1. QE curve


def test_qe_table_interpolation_and_zero_anchors():
    m = response_manifest.load_manifest()
    value = m["parameters"]["qe_curve"]["value"]
    w = np.asarray(value["wavelength_nm"], dtype=float)
    e = np.asarray(value["efficiency"], dtype=float)
    # The thesis table, verbatim.
    for lam, qe in THESIS_QE.items():
        assert float(e[w.tolist().index(lam)]) == pytest.approx(qe), lam
    # Zero-QE anchors outside 280-700 nm are kept, as the author asks.
    assert 200 in w.tolist() and 800 in w.tolist()
    assert float(e[w.tolist().index(200)]) == 0.0 and float(e[w.tolist().index(800)]) == 0.0

    q = reference(1, np.zeros(1, dtype=int)).qe_curve
    assert q(280.0) == pytest.approx(0.0) and q(700.0) == pytest.approx(0.0)
    assert q(200.0) == pytest.approx(0.0) and q(800.0) == pytest.approx(0.0)
    # Piecewise-linear in wavelength between the declared anchors.
    assert q(290.0) == pytest.approx(0.005)                         # 280:0 -> 300:0.01
    assert q(410.0) == pytest.approx((0.305 + 0.29) / 2)            # 400:0.305 -> 420:0.29
    assert q(580.0) == pytest.approx((0.045 + 0.015) / 2)           # 560:0.045 -> 600:0.015
    # Outside the calibrated domain: rejected, never extrapolated.
    for bad in (199.0, 801.0, np.nan):
        with pytest.raises(ValueError):
            q([bad])


def test_peak_qe_times_ce_matches_the_hk_requirement():
    m = response_manifest.load_manifest()
    peak = max(m["parameters"]["qe_curve"]["value"]["efficiency"])
    ce = m["parameters"]["collection_efficiency"]["value"]
    product = peak * ce
    assert peak == pytest.approx(0.305) and ce == pytest.approx(0.87)
    assert product == pytest.approx(0.305 * 0.87, rel=1e-12)
    assert 0.25 <= product <= 0.29, product
    # HK requirement 26% (Table 4.1); the check the author asks R-7 to record.
    assert abs(product - 0.26) <= 0.01
    p = reference(1, np.zeros(1, dtype=int))
    assert p.collection_efficiency == pytest.approx(0.87)


# ------------------------------------------------------- 2. split-normal kernel


def test_split_normal_analytic_properties():
    tw = reference(1, np.zeros(1, dtype=int)).timing_width
    assert isinstance(tw, SplitNormalWidth)
    q = 2.0   # the measured 2 PE bump sits here
    x = np.linspace(-60.0, 60.0, 240001)
    pdf = tw.pdf(x, q)
    assert np.trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-8)
    # The mean sits at the mode plus sqrt(2/pi) * (sigma_l - sigma_e).
    assert np.trapezoid(x * pdf, x) == pytest.approx(float(tw.mean_shift(q)), abs=2e-4)
    assert float(tw.mean_shift(q)) == pytest.approx(np.sqrt(2.0 / np.pi) * (float(tw.sigma_l(q)) - float(tw.sigma_e(q))), rel=1e-12)
    # Half probability at the mode; the mode is the density maximum.
    w = float(tw.sigma_e(q)) / (float(tw.sigma_e(q)) + float(tw.sigma_l(q)))
    assert float(tw.cdf(0.0, q)) == pytest.approx(w, rel=1e-12)
    grid = np.linspace(-4.0, 4.0, 801)
    assert abs(float(grid[np.argmax(tw.pdf(grid, q))])) <= 1e-12
    assert float(tw.cdf(-1e3, q)) == 0.0 and float(tw.cdf(1e3, q)) == 1.0
    # Inverse CDF reproduces the uniform.
    u = np.linspace(1e-4, 1.0 - 1e-4, 201)
    z = tw.jitter(np.full(201, q), u)
    np.testing.assert_allclose(tw.cdf(z, q), u, atol=1e-10)
    # Linear interpolation in log q, at the geometric midpoint of 1 PE and 2 PE.
    np.testing.assert_allclose(tw.sigma_e(np.sqrt(2.0)), (1.26 + 1.04) / 2, rtol=1e-12)
    np.testing.assert_allclose(tw.sigma_l(np.sqrt(2.0)), (1.52 + 1.76) / 2, rtol=1e-12)
    # The 2 PE bump is real in the data: sigma_l(2) > sigma_l(1) > sigma_l(3).
    assert float(tw.sigma_l(2.0)) > float(tw.sigma_l(1.0)) > float(tw.sigma_l(3.0))


@pytest.mark.parametrize("q", [0.25, 1.0, 2.0, 10.0, 100.0])
def test_split_normal_sampled_widths_mode_and_mean(q):
    p = raw_flat(N, q)
    b = one_photon_per_channel(N, q)
    d = read(b, p)
    assert len(d) == N and np.allclose(d.charge, q)
    tw = p.timing_width
    walk = float(p.walk_curve(q))
    z = d.time - walk                      # arrival is 0, so t0 = walk(q)
    e, l = float(tw.sigma_e(q)), float(tw.sigma_l(q))
    early = z < 0
    # Sampled early and late widths match the thesis table within 3 standard errors.
    # Each side is a half-normal about the mode, so the root-mean-square deviation
    # about t0 estimates that side's sigma (a plain conditional SD would be smaller
    # by the half-normal factor sqrt(1 - 2/pi)).
    for mask, expected in ((early, e), (~early, l)):
        assert mask.sum() > 0.2 * N
        sigma_hat = float(np.sqrt(np.mean(z[mask] ** 2)))
        se = expected / np.sqrt(2.0 * mask.sum())
        assert abs(sigma_hat - expected) <= 3.0 * se, (q, sigma_hat, expected, se)
    # The half probability is sigma_e / (sigma_e + sigma_l).
    w = e / (e + l)
    assert abs(early.mean() - w) < 3.0 * np.sqrt(w * (1 - w) / N)
    # The sampled mean is the mode plus the recorded shift.
    se_mean = max(e, l) / np.sqrt(N)
    assert abs(z.mean() - float(tw.mean_shift(q))) < 3.0 * se_mean
    # Mode: a Gaussian KDE peaks within 0.5 ns of arrival + walk.
    bandwidth_ns = 0.4
    kde = gaussian_kde(z, bw_method=bandwidth_ns / float(z.std(ddof=1)))
    grid = np.linspace(-3.0, 3.0, 601)
    assert abs(float(grid[np.argmax(kde(grid))])) <= 0.5
    # Whole distribution (mode, both sigma sides, tails) against the analytic Eq. 6.3 CDF.
    assert kstest(z, lambda v: tw.cdf(v, q)).pvalue > 1e-3


def test_split_normal_mode_tracks_a_nonzero_arrival():
    """The kernel's mode is the photon arrival plus the walk, not a fixed time origin."""
    arrival = 12.5
    q = 2.0
    p = raw_flat(N, q)
    d = read(one_photon_per_channel(N, q, arrival=arrival), p)
    tw = p.timing_width
    z = d.time - (arrival + float(p.walk_curve(q)))
    w = float(tw.sigma_e(q)) / (float(tw.sigma_e(q)) + float(tw.sigma_l(q)))
    assert abs((z < 0).mean() - w) < 3.0 * np.sqrt(w * (1 - w) / N)
    assert abs(z.mean() - float(tw.mean_shift(q))) < 3.0 * max(float(tw.sigma_e(q)), float(tw.sigma_l(q))) / np.sqrt(N)
    kde = gaussian_kde(z, bw_method=0.4 / float(z.std(ddof=1)))
    grid = np.linspace(-3.0, 3.0, 601)
    assert abs(float(grid[np.argmax(kde(grid))])) <= 0.5
    assert kstest(z, lambda v: tw.cdf(v, q)).pvalue > 1e-3


def test_width_factor_scales_both_split_normal_sigmas():
    p = raw_flat(N, 2.0)
    wide = replace(p, timing_width_factor=2.0)
    b = one_photon_per_channel(N, 2.0)
    a = read(b, p)
    w = read(b, wide)
    walk = float(p.walk_curve(2.0))
    za, zw = a.time - walk, w.time - walk
    # Identical keyed uniforms, so a factor of 2 doubles the draw exactly.
    np.testing.assert_allclose(zw, 2.0 * za, atol=1e-9, rtol=0)
    # Both sampled sides scale, and so does the recorded mean shift.
    assert abs(float(zw[zw < 0].std()) - 2.0 * float(za[za < 0].std())) < 1e-9
    assert abs(float(zw[zw >= 0].std()) - 2.0 * float(za[za >= 0].std())) < 1e-9
    assert abs(zw.mean() - 2.0 * za.mean()) < 1e-9


def test_split_normal_rejects_invalid_tables():
    with pytest.raises(ValueError): SplitNormalWidth((1.0,), (1.0,), (1.0,))                 # one anchor
    with pytest.raises(ValueError): SplitNormalWidth((1.0, 2.0), (1.0,), (1.0, 2.0))        # length mismatch
    with pytest.raises(ValueError): SplitNormalWidth((2.0, 1.0), (1.0, 1.0), (1.0, 1.0))    # not increasing
    with pytest.raises(ValueError): SplitNormalWidth((1.0, 2.0), (1.0, 0.0), (1.0, 1.0))    # zero sigma
    with pytest.raises(ValueError): SplitNormalWidth((1.0, 2.0), (1.0, np.nan), (1.0, 1.0))
    tw = reference(1, np.zeros(1, dtype=int)).timing_width
    with pytest.raises(ValueError): tw.sigma_e(0.2)                                          # outside domain
    with pytest.raises(ValueError): tw.jitter(np.array([1.0]), np.array([1.0]))              # u outside (0,1)


# --------------------------------------------------- 3. T-Q and dark-hit coupling


def test_tq_intervention_still_changes_only_the_walk():
    n = 400
    b = PhotonBatch(5, np.arange(n * 6), np.repeat(np.arange(n), 6) % n, np.zeros(n * 6), np.full(n * 6, 400.0))
    p = reference(n, np.arange(n) // 19)
    a = readout_chain(b.sensor_id, b.arrival_time_ns, n, p, 5, photon_batch=b)
    up = readout_chain(b.sensor_id, b.arrival_time_ns, n, replace(p, walk_curve=p.walk_curve.scaled(1.2)), 5, photon_batch=b)
    # Charge and membership are untouched; only the walk moves, by the declared curve.
    np.testing.assert_array_equal(a.sensor, up.sensor)
    np.testing.assert_array_equal(a.charge, up.charge)
    expected = 0.2 * p.walk_curve.b_ns / np.sqrt(np.minimum(a.charge, p.walk_curve.q_sat_pe))
    np.testing.assert_allclose(up.time - a.time, expected, atol=1e-12, rtol=0)
    # The split-normal draw itself is unchanged, so the offset control is exactly +1 ns.
    offset = readout_chain(b.sensor_id, b.arrival_time_ns, n, replace(p, walk_curve=p.walk_curve.scaled(offset_ns=1.0)), 5, photon_batch=b)
    np.testing.assert_allclose(offset.time - a.time, 1.0, atol=1e-12, rtol=0)


def test_dark_digits_use_the_same_split_normal_kernel():
    n = 1995
    p = replace(reference(n), dark_rate_hz=5.0e5, timing_width_factor=1.0)
    no_width = replace(p, timing_width=None)          # digit time = earliest dark PE time
    a = dark_noise_window(p, n, seed=3, run=11)
    b = dark_noise_window(no_width, n, seed=3, run=11)
    assert len(a) > 500
    np.testing.assert_array_equal(a.sensor, b.sensor)
    np.testing.assert_array_equal(a.charge, b.charge)
    jitter = a.time - b.time                          # exactly the split-normal draw
    tw = p.timing_width
    e, l = tw.sigma_e(a.charge), tw.sigma_l(a.charge)
    w = e / (e + l)
    early = jitter < 0
    assert abs(early.mean() - w.mean()) < 3.0 * np.sqrt(w.mean() * (1 - w.mean()) / len(jitter))
    for mask, sigma in ((early, e), (~early, l)):
        assert mask.sum() > 100
        expected = float(np.sqrt(np.mean(sigma[mask] ** 2)))   # pooled across charges
        sigma_hat = float(np.sqrt(np.mean(jitter[mask] ** 2)))  # RMS about the mode
        assert abs(sigma_hat - expected) <= 3.0 * expected / np.sqrt(2.0 * mask.sum()), (sigma_hat, expected)
    assert abs(jitter.mean() - float(tw.mean_shift(a.charge).mean())) < 3.0 * float(max(e.max(), l.max())) / np.sqrt(len(jitter))


# ------------------------------------------------------- 4. versioning / hashes


def test_v3_is_the_default_and_declares_literature_proxies():
    m = response_manifest.load_manifest()
    assert m["schema"] == "pmt-response-v3" and response_manifest.SCHEMA == "pmt-response-v3"
    assert m["status"] == "literature_proxy_dev"
    labels = {"threshold_pe": "literature_sk"}
    for name in ("spe_sigma", "qe_curve", "collection_efficiency", "timing_width"):
        labels[name] = "literature_proxy_hk_20in"
    for name, label in labels.items():
        assert m["parameters"][name]["status"] == label, name
        assert "Xia" in m["parameters"][name]["source"], name
        assert "2019" in m["parameters"][name]["source"], name
    assert m["parameters"]["threshold_pe"]["value"] == 0.25
    assert m["parameters"]["spe_sigma"]["value"] == 0.30
    assert m["parameters"]["timing_width"]["value"]["kernel"] == "split_normal"
    assert isinstance(reference(10, np.zeros(10, dtype=int)).timing_width, SplitNormalWidth)
    assert m["literature_provenance"]["citation"].startswith("J. Xia")


def test_v2_is_still_reproducible_by_its_manifest_hash():
    assert response_manifest.manifest_sha256(response_manifest.V2_PATH) == response_manifest.V2_SHA256
    assert response_manifest.V2_SHA256 == "17636f9bba69e10ea59bcbd83a884c003b4709ea5e1a6fabd94178a0be402e68"
    m = response_manifest.load_manifest(response_manifest.V2_PATH)
    assert m["schema"] == "pmt-response-v2" and m["status"] == "synthetic_dev"
    p = reference(76, np.arange(76) // 12, path=response_manifest.V2_PATH)
    assert isinstance(p.timing_width, WidthCurve)
    assert p.collection_efficiency == 0.8
    assert float(p.qe_curve(400.0)) == pytest.approx(0.5)        # the old synthetic peak
    assert float(p.timing_width(1.0)) == pytest.approx(1.5)
    # The v3 manifest names the pinned v2 file hash it supersedes.
    assert response_manifest.V2_SHA256 in response_manifest.load_manifest()["supersedes"]


def test_v2_and_v3_walk_and_charge_domain_are_unchanged():
    for path in (None, response_manifest.V2_PATH):
        p = reference(1995, None, path=path)
        assert p.charge_domain_pe == (0.25, 1000.0)
        assert p.walk_curve == p.reference_walk
        assert float(p.walk_curve(0.25)) == pytest.approx(4.0)
        assert float(p.walk_curve(1000.0)) == pytest.approx(2.0 / np.sqrt(50.0))
        assert p.threshold_pe == 0.25 and p.spe_sigma == 0.3
