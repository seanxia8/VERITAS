"""R-7 v2 acceptance (author message, 23 Oct 2026; decisions 3 Nov 2026): WCSim 3-inch response v3.

The default ``pmt-response-v3`` carries WCTE's 3-inch Hamamatsu R14374 response from
WCSim commit c5ed0b1 (``src/WCSimPMTObject.cc``, class ``PMT3inchR14374_WCTE``, file
sha256 caa9ad51...). Author decisions, 3 Nov 2026: the SPE charge is sampled directly
from the 501-point table (the Gaussian+exponential fit missed the declared KS
tolerance), and sigma at 0.25 PE is WCSim's spline value 0.760 ns (not 0.77).

R-7 v2 section 2 lists the tests; each has its own function below. Tolerances fixed
before the first run of this file:

* sigma(Q) within 1% of the R-7 v2 values (0.81 / 0.67 / 0.65 / 0.64 ns at 0.2 / 0.5
  / 1 / >=4 PE) and of the author's 0.760 ns at 0.25 PE, and within 1e-4 ns of an
  independent not-a-knot spline of the transcribed WCSim table;
* sampled digit-time SD within 3 standard errors (sigma / sqrt(2N)) of sigma(Q) at
  0.25, 1 and 4 PE, and a KS test against the Gaussian at p > 1e-3;
* QE: the WCSim table verbatim, piecewise linear, zero outside 300-700 nm;
* peak QE x CE within 0.24-0.28;
* SPE: the declared fit KS tolerance (0.02) is recorded as failed (0.031), and the
  directly sampled spectrum matches the table CDF (KS p > 1e-3), WCSim's own two-uniform
  rn1pe draw (two-sample KS p > 1e-3) and the table mean / relative sigma;
* the threshold cuts the smeared charge: exactly the channels whose sampled charge is
  below 0.25 PE are lost, and the lost fraction is within 6 binomial SE of the table's;
* T-Q interventions move only the walk; dark digits use the same width and SPE kernels;
* v2 and the thesis cross-check stay reproducible by their manifest hashes.

CPU only. Synthetic validation of simulation-level values, not WCTE data.
"""
import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.stats import kstest, ks_2samp, norm

from noise_module_lucid import response_manifest
from noise_module_lucid.photon_contract import PhotonBatch
from noise_module_lucid.quantum_efficiency import QECurve, keyed_uniform
from noise_module_lucid.readout import dark_noise_window, readout_chain
from noise_module_lucid.spe_charge import TabulatedSPE
from noise_module_lucid.timing_response import WCSimTTSWidth

N = 20000
SEED = 72

#: WCSim c5ed0b1 src/WCSimPMTObject.cc lines 3811-3822 (PMT3inchR14374_WCTE constructor), transcribed.
WCSIM_CHARGE = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
WCSIM_RESOL = [1.1654, 0.61088, 0.4186, 0.32532, 0.26484, 0.23084, 0.20969, 0.19297, 0.17716, 0.17046,
               0.15455, 0.1427, 0.13699, 0.13229]
#: WCSim c5ed0b1 GetQEWavelength / GetQE (raw, without the 1/0.73 correctionFactor), transcribed.
WCSIM_QE = dict(zip(range(300, 701, 20), [0.0787, 0.1838, 0.2401, 0.2521, 0.2695, 0.2676, 0.2593, 0.2472, 0.2276, 0.1970,
                                          0.1777, 0.1547, 0.1033, 0.0727, 0.0587, 0.0470, 0.0372, 0.0285, 0.0220, 0.0130, 0.0084]))
#: R(Q) (ns FWHM) from a separate port of de Boor's CUBSPL with not-a-knot ends (what ROOT's TSpline3
#: does with no end conditions), computed outside this package by the independent reviewer, 3 Nov 2026.
#: Hard-coded so a change of spline convention in the code fails this test.
DEBOOR_R = {0.25: 0.9757710, 0.5: 0.4889332}
#: R-7 v2 section 1.1 expected widths, with the author's 3 Nov 2026 value at 0.25 PE.
R7V2_SIGMA = {0.2: 0.81, 0.25: 0.760, 0.5: 0.67, 1.0: 0.65, 4.0: 0.64, 10.0: 0.64, 1000.0: 0.64}
#: sha256 of json.dumps(the 500-value Getqpe CDF) as committed; a different table must fail.
QPE_CDF_SHA256 = "671cf70f6c7b2faf44b9ab0ec28d6e8a3f2a89962ac8648e9e5996343d3d9196"


def reference(n=1995, groups=None, path=None):
    groups = np.arange(n) // 19 if groups is None else groups
    return response_manifest.reference_response(n, groups, manifest=path)


def one_photon_per_channel(n, event, *, arrival=0.0):
    return PhotonBatch(int(event), np.arange(n), np.arange(n), np.full(n, arrival), np.full(n, 400.0))


def read(b, p, seed=SEED):
    return readout_chain(b.sensor_id, b.arrival_time_ns, int(b.sensor_id.max()) + 1, p, seed, photon_batch=b)


def unit_detection(n):
    """Default v3 response with every incident photon detected, unit group factors and no dark counts."""
    p = reference(n, np.zeros(n, dtype=int))
    return replace(p, qe_curve=QECurve(p.qe_curve.wavelength_nm, (1.0,) * len(p.qe_curve.wavelength_nm)),
                   collection_efficiency=1.0, dark_rate_hz=0.0)


def raw_flat(n, q):
    """Unit detection, charge exactly ``q`` (no SPE smearing), zero reference walk: time = walk(q) + jitter."""
    p = unit_detection(n)
    return replace(p, spe_charge=None, spe_sigma=0.0, gain=float(q), reference_walk=replace(p.reference_walk, b_ns=0.0))


def rn1pe(cdf, u1, u2):
    """WCSim WCSimWCPMT::rn1pe, written out independently: the first bin i with u1 <= cdf[i]."""
    c = np.asarray(cdf)
    i = np.argmax(u1[:, None] <= c[None, :], axis=1)
    return (i - 50 + u2) / 22.83


# --------------------------------------------------------------- 1. timing kernel


def test_sigma_matches_r7v2_values_and_an_independent_spline():
    m = response_manifest.load_manifest()
    value = m["parameters"]["timing_width"]["value"]
    assert value["kernel"] == "wcsim_tts_spline"
    assert value["charge_pe"] == WCSIM_CHARGE and value["resolution_fwhm_ns"] == WCSIM_RESOL
    assert value["tts_fwhm_ns"] == 1.5 and value["q_clamp_pe"] == 4.0 and value["fwhm_to_sigma"] == 2.355
    tw = reference(1, np.zeros(1, dtype=int)).timing_width
    assert isinstance(tw, WCSimTTSWidth) and tw.domain_pe == (0.25, 1000.0)
    # The 0.2 PE value sits below the charge domain, so check it on the same kernel with a wider domain.
    wide = replace(tw, domain_pe=(0.2, 1000.0))
    spline = CubicSpline(WCSIM_CHARGE, WCSIM_RESOL, bc_type="not-a-knot")
    for q, expected in R7V2_SIGMA.items():
        got = float(wide(q))
        assert abs(got - expected) <= 0.01 * expected, (q, got, expected)
        independent = np.sqrt(1.5 ** 2 + spline(min(q, 4.0)) ** 2) / 2.355
        assert got == pytest.approx(independent, abs=1e-4)
    assert float(tw(0.25)) == pytest.approx(0.7599, abs=5e-5)     # author decision: spline, not 0.77
    for q, r in DEBOOR_R.items():
        assert float(tw.resolution_fwhm(q)) == pytest.approx(r, abs=1e-6), q
    # A natural spline (the 24 Oct brief's word) would give a different value at 0.25 PE.
    natural = CubicSpline(WCSIM_CHARGE, WCSIM_RESOL, bc_type="natural")
    assert abs(np.sqrt(1.5 ** 2 + natural(0.25) ** 2) / 2.355 - float(tw(0.25))) > 0.005
    # The R-7 v2 text's 0.77 is linear interpolation; the spline differs from it by more than 1%.
    linear = np.sqrt(1.5 ** 2 + np.interp(0.25, WCSIM_CHARGE, WCSIM_RESOL) ** 2) / 2.355
    assert abs(linear - 0.77) < 0.005 and abs(float(tw(0.25)) - linear) > 0.009
    # Constant above 4 PE, as WCSim clamps Q.
    np.testing.assert_allclose(tw([4.0, 5.0, 50.0, 1000.0]), float(tw(4.0)), rtol=0, atol=1e-15)
    assert float(np.asarray(tw.mean_shift(1.0))) == 0.0
    # No extrapolation: outside the declared domain, or a domain below the first knot, is rejected.
    for bad in (0.2, 1000.5, np.nan):
        with pytest.raises(ValueError):
            tw([bad])
    with pytest.raises(ValueError):
        replace(tw, domain_pe=(0.1, 1000.0))


@pytest.mark.parametrize("q", [0.25, 1.0, 4.0])
def test_sampled_width_matches_sigma(q):
    p = raw_flat(N, q)
    d = read(one_photon_per_channel(N, 2000 + int(round(100 * q))), p)
    assert len(d) == N and np.allclose(d.charge, q)
    z = d.time - float(p.walk_curve(q))                 # arrival 0, so t0 = walk(q)
    sigma = R7V2_SIGMA[q]
    sd = float(np.sqrt(np.mean(z ** 2)))                # about the known mode/mean
    assert abs(sd - float(p.timing_width(q))) <= 3.0 * sigma / np.sqrt(2.0 * N), (q, sd)
    assert abs(sd - sigma) <= 0.01 * sigma + 3.0 * sigma / np.sqrt(2.0 * N)
    assert abs(z.mean()) < 3.0 * sigma / np.sqrt(N)
    assert kstest(z / float(p.timing_width(q)), "norm").pvalue > 1e-3


def test_width_factor_multiplies_sigma_exactly():
    p = raw_flat(N, 1.0)
    b = one_photon_per_channel(N, 2100)
    a, w = read(b, p), read(b, replace(p, timing_width_factor=1.7))
    walk = float(p.walk_curve(1.0))
    np.testing.assert_allclose(w.time - walk, 1.7 * (a.time - walk), atol=1e-9, rtol=0)


# --------------------------------------------------------------------- 2. QE, CE


def test_qe_table_interpolation_and_anchors():
    m = response_manifest.load_manifest()
    v = m["parameters"]["qe_curve"]["value"]
    w, e = list(v["wavelength_nm"]), list(v["efficiency"])
    for lam, qe in WCSIM_QE.items():                    # raw WCSim table, verbatim
        assert e[w.index(float(lam))] == qe, lam
    assert w[:2] == [200.0, 299.99] and w[-2:] == [700.01, 800.0] and e[:2] == [0.0, 0.0] and e[-2:] == [0.0, 0.0]
    q = reference(1, np.zeros(1, dtype=int)).qe_curve
    assert float(q(380.0)) == 0.2695 and max(e) == 0.2695
    assert float(q(310.0)) == pytest.approx((0.0787 + 0.1838) / 2)          # piecewise linear
    assert float(q(690.0)) == pytest.approx((0.0130 + 0.0084) / 2)
    # Zero outside 300-700 nm, as WCSimPMTQE returns 0 there; the 200-800 nm domain is kept.
    for lam in (200.0, 250.0, 299.98, 700.02, 750.0, 800.0):
        assert float(q(lam)) == 0.0, lam
    for bad in (199.0, 801.0, np.nan):
        with pytest.raises(ValueError):
            q([bad])
    # The legacy 1/0.73 factor is recorded, not applied.
    assert "1/0.73" in m["wcsim_provenance"]["recorded_not_applied"]["qe_correction_factor"]
    assert max(e) < 0.2695 / 0.73 - 0.05


def test_peak_qe_times_ce_in_band():
    m = response_manifest.load_manifest()
    peak = max(m["parameters"]["qe_curve"]["value"]["efficiency"])
    ce = m["parameters"]["collection_efficiency"]["value"]
    assert ce == 0.95 and reference(1, np.zeros(1, dtype=int)).collection_efficiency == 0.95
    assert 0.24 <= peak * ce <= 0.28 and peak * ce == pytest.approx(0.256025, abs=1e-9)


# ------------------------------------------------------------------- 3. SPE charge


def test_spe_fit_missed_the_declared_tolerance_so_the_table_is_sampled():
    m = response_manifest.load_manifest()
    fit = m["wcsim_provenance"]["spe_fit_record"]
    assert fit["ks_tolerance_declared_before_fit"] == 0.02
    assert fit["ks_distance_to_table"] == pytest.approx(0.0313, abs=5e-4) and fit["ks_pass"] is False and fit["adopted"] is False
    v = m["parameters"]["spe_charge"]["value"]
    assert v["kernel"] == "wcsim_qpe_table" and v["bin_offset"] == 50 and v["bins_per_pe"] == 22.83 and v["normalise_mean"] is True
    assert "spe_sigma" not in m["parameters"]
    cdf = v["cdf"]
    assert len(cdf) == 500 and cdf[-1] == 1.0 and np.all(np.diff(cdf) >= 0)
    assert int(np.argmax(np.asarray(cdf) > 0)) == 50      # no negative raw charge in this table
    assert hashlib.sha256(json.dumps(cdf).encode()).hexdigest() == QPE_CDF_SHA256
    spe = reference(1, np.zeros(1, dtype=int)).spe_charge
    assert isinstance(spe, TabulatedSPE)
    assert spe.raw_mean_pe == pytest.approx(1.0648, abs=5e-5) and spe.relative_sigma == pytest.approx(0.4525, abs=5e-5)
    assert spe.mean_pe == pytest.approx(1.0, abs=1e-12)


def test_sampled_spe_matches_the_table_and_wcsim_rn1pe():
    spe = reference(1, np.zeros(1, dtype=int)).spe_charge
    x = spe.sample(keyed_uniform(SEED, 3000, np.arange(N), "spe_test"))
    assert kstest(x, spe.cdf_at).pvalue > 1e-3
    assert abs(x.mean() - 1.0) < 3.0 * spe.sigma_pe / np.sqrt(N)
    assert abs(x.std() / x.mean() - spe.relative_sigma) < 0.02
    rng = np.random.default_rng(20261103)
    wcsim = rn1pe(spe.cdf, rng.random(N), rng.random(N)) / spe.raw_mean_pe
    assert ks_2samp(x, wcsim).pvalue > 1e-3
    # The sampler is the exact inverse of the declared CDF.
    u = np.linspace(1e-4, 1 - 1e-4, 501)
    np.testing.assert_allclose(spe.cdf_at(spe.sample(u)), u, atol=1e-12)


def test_threshold_applies_after_charge_smearing():
    """Every channel has exactly one true PE; only those whose *sampled* charge is below 0.25 PE are lost."""
    n, event = N, 3100
    p = replace(unit_detection(n), gain=1.0)
    b = one_photon_per_channel(n, event)
    d = read(b, p)
    x = p.spe_charge.sample(keyed_uniform(SEED, event, b.photon_id, "spe_charge"))   # the readout's own keyed draw
    lost = np.flatnonzero(x < 0.25)
    assert d.counts["lost_threshold"] == lost.size and np.all(d.n_pe_true == 1)
    np.testing.assert_array_equal(np.setdiff1d(np.arange(n), d.sensor), lost)
    np.testing.assert_allclose(d.charge, x[d.sensor], rtol=0, atol=1e-12)
    expected = float(p.spe_charge.cdf_at(0.25))                                       # 0.0296 of 1-PE hits
    assert abs(lost.size / n - expected) < 6.0 * np.sqrt(expected * (1 - expected) / n)
    assert d.counts["lost_threshold"] > 0                 # a cut on the true PE count would lose none


def test_removing_a_photon_leaves_every_other_photoelectron_charge_unchanged():
    n = 300
    ids = np.arange(3 * n)
    b = PhotonBatch(31, ids, ids % n, np.zeros(3 * n), np.full(3 * n, 400.0))
    keep = ids != 5                                      # drop one photon of channel 5
    b2 = PhotonBatch(31, ids[keep], (ids % n)[keep], np.zeros(keep.sum()), np.full(keep.sum(), 400.0))
    p = replace(unit_detection(n), threshold_pe=0.25)
    a, c = read(b, p, 31), read(b2, p, 31)
    x = p.spe_charge.sample(keyed_uniform(31, 31, ids, "spe_charge"))
    q_full = np.bincount(ids % n, weights=x, minlength=n)
    q_drop = np.bincount((ids % n)[keep], weights=x[keep], minlength=n)
    np.testing.assert_allclose(a.charge, q_full[a.sensor], atol=1e-12, rtol=0)
    np.testing.assert_allclose(c.charge, q_drop[c.sensor], atol=1e-12, rtol=0)
    other = np.intersect1d(a.sensor[a.sensor != 5], c.sensor)
    np.testing.assert_array_equal(a.charge[np.isin(a.sensor, other)], c.charge[np.isin(c.sensor, other)])


# --------------------------------------------------- 4. T-Q and dark-hit coupling


def test_tq_intervention_still_changes_only_the_walk():
    n = 400
    b = PhotonBatch(5, np.arange(n * 6), np.repeat(np.arange(n), 6) % n, np.zeros(n * 6), np.full(n * 6, 400.0))
    p = reference(n, np.arange(n) // 19)
    a = readout_chain(b.sensor_id, b.arrival_time_ns, n, p, 5, photon_batch=b)
    up = readout_chain(b.sensor_id, b.arrival_time_ns, n, replace(p, walk_curve=p.walk_curve.scaled(1.2)), 5, photon_batch=b)
    np.testing.assert_array_equal(a.sensor, up.sensor)
    np.testing.assert_array_equal(a.charge, up.charge)
    expected = 0.2 * p.walk_curve.b_ns / np.sqrt(np.minimum(a.charge, p.walk_curve.q_sat_pe))
    np.testing.assert_allclose(up.time - a.time, expected, atol=1e-12, rtol=0)
    offset = readout_chain(b.sensor_id, b.arrival_time_ns, n, replace(p, walk_curve=p.walk_curve.scaled(offset_ns=1.0)), 5, photon_batch=b)
    np.testing.assert_allclose(offset.time - a.time, 1.0, atol=1e-12, rtol=0)


def test_dark_digits_use_the_same_width_and_spe_kernels():
    n = 1995
    p = replace(reference(n), dark_rate_hz=5.0e5, timing_width_factor=1.0)
    a = dark_noise_window(p, n, seed=3, run=11)
    b = dark_noise_window(replace(p, timing_width=None), n, seed=3, run=11)
    assert len(a) > 300
    np.testing.assert_array_equal(a.sensor, b.sensor)
    np.testing.assert_array_equal(a.charge, b.charge)
    jitter = a.time - b.time                              # exactly the width draw
    assert kstest(jitter / p.timing_width(a.charge), "norm").pvalue > 1e-3
    # Single-dark-PE digits carry one draw from the same SPE table (truncated at the 0.25 PE threshold).
    one = a.n_dark_true == 1
    assert one.sum() > 200 and np.all(a.n_pe_true[one] == 1)
    # Undo the per-module gain (1.03 on odd modules); the threshold acts on the gain-scaled charge.
    g = np.asarray(p.gain)[a.sensor[one]]
    x = a.charge[one] / g
    spe, thr = p.spe_charge, p.threshold_pe
    lo = spe.cdf_at(thr / g)                              # per-digit truncation point in SPE units
    u = (spe.cdf_at(x) - lo) / (1 - lo)                   # probability integral transform of a truncated draw
    assert kstest(u, "uniform").pvalue > 1e-3


# ------------------------------------------------------- 5. versioning / hashes


def test_v3_default_is_the_wcsim_3inch_response_with_provenance():
    m = response_manifest.load_manifest()
    assert m["schema"] == "pmt-response-v3" and m["status"] == "literature_proxy_wcsim_wcte_3in"
    prov = m["wcsim_provenance"]
    assert prov["commit"] == "c5ed0b127fd4ef13e5c34436c2c5385fcc9b26b3"
    assert prov["file_sha256"] == "caa9ad51f1e59f236905927102ff775c33d11aeb5b51a320c884b3392bf18b9f"
    assert prov["class"] == "PMT3inchR14374_WCTE" and prov["file"] == "src/WCSimPMTObject.cc"
    assert prov["recorded_not_applied"]["dark_rate_conversion_factor"] == 1.126
    assert prov["GetNbOfQEDefined"]["value"] == 20 and prov["GetNbOfQEDefined"]["table_points"] == 21
    assert prov["wcte_fork_differences"]["decision"].startswith("reported, not chosen")
    for name in ("qe_curve", "collection_efficiency", "spe_charge", "timing_width", "dark_rate_hz"):
        entry = m["parameters"][name]
        assert entry["status"] == "literature_proxy_wcsim_wcte_3in" and "c5ed0b1" in entry["source"], name
    assert m["parameters"]["dark_rate_hz"]["value"] == 1000.0 and m["parameters"]["threshold_pe"]["value"] == 0.25
    assert "digitised" in m["parameters"]["threshold_pe"]["source"]
    assert m["parameters"]["walk"]["value"] == {"form": "inverse_sqrt", "a_ns": 0.0, "b_ns": 2.0, "q_sat_pe": 50.0}


def test_v2_thesis_xcheck_and_v3_are_reproducible_by_their_manifest_hashes():
    assert response_manifest.manifest_sha256(response_manifest.V2_PATH) == response_manifest.V2_SHA256
    assert response_manifest.manifest_sha256() == response_manifest.V3_SHA256
    assert response_manifest.manifest_sha256(response_manifest.THESIS_XCHECK_PATH) == response_manifest.THESIS_XCHECK_SHA256
    sup = response_manifest.load_manifest()["supersedes"]
    assert response_manifest.V2_SHA256 in sup and response_manifest.THESIS_XCHECK_SHA256 in sup
    v2 = reference(76, np.arange(76) // 12, path=response_manifest.V2_PATH)
    assert v2.spe_charge is None and v2.spe_sigma == 0.3 and float(v2.timing_width(1.0)) == pytest.approx(1.5)


@pytest.mark.skipif(not os.environ.get("WCSIM_PMT_OBJECT"), reason="set WCSIM_PMT_OBJECT to WCSim c5ed0b1 src/WCSimPMTObject.cc")
def test_manifest_rebuilds_byte_identically_from_the_pinned_wcsim_file(tmp_path):
    import subprocess
    import sys
    root = response_manifest.DEFAULT_PATH.parents[2]
    out = tmp_path / "rebuilt.json"
    subprocess.run([sys.executable, str(root / "scripts" / "pmt_r7v2_wcsim_manifest.py"), os.environ["WCSIM_PMT_OBJECT"], "--out", str(out)],
                   check=True, env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert out.read_bytes() == response_manifest.DEFAULT_PATH.read_bytes()
