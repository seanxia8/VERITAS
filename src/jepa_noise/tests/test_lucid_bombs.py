"""Tests for the WP4b development-only LUCiD generator contract."""
import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb

GEOM = lb.synthetic_cylinder_geometry()
BIG = lb.synthetic_cylinder_geometry(64, 3)
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lucid_bombs_v0.json"
THRESHOLD_PE = 0.25
TTS_SIGMA_NS = 1.0


def _draws(geometry=GEOM, event_id=7, run_id="lucid/probe_train/0000", n_photons=400,
           dark_mean=0.0, seed=lb.DEFAULT_SEED):
    """Test-only Draws; the real generator fills these from LUCiD."""
    def rng(role):
        return lb.keyed_rng(seed, lb.NS_DRAWS, run_id, event_id, role)

    photon = rng("photons")
    return lb.Draws(
        key=f"{lb.NS_DRAWS}/{run_id}/{event_id}", run_id=run_id, event_id=event_id,
        photon_sensor=photon.integers(0, geometry.n_pmt, n_photons),
        photon_time_ns=photon.uniform(0.0, 1000.0, n_photons),
        charge_z=rng("charge").standard_normal(geometry.n_pmt),
        transit_z=rng("transit").standard_normal(geometry.n_pmt),
        dark_counts=rng("dark").poisson(dark_mean, geometry.n_pmt).astype(np.int64))


def _fraction_draws(geometry, cfg, fractions, *, vertex=None, yield_photons=1.0):
    """SensorDraws whose Poisson arrivals sit at fixed fractions of the raw mean.

    ``fractions[pmt] = [f0, f1, ...]`` puts the k-th arrival at ``f_k * mu_p`` of the
    **no-fixed-response** mean, so an arrival at ``1.02 * mu_p`` is counted on a +3%
    module and not on a 1.00 module: the fixed response is applied to the expected
    charge, exactly and deterministically (charge_z is zero, so the realised charge is
    the count).
    """
    vertex = np.array([0.10, -0.20, 0.05]) if vertex is None else np.asarray(vertex, float)
    qe = np.full(geometry.n_pmt, 0.2)
    mu_raw = lb.expected_charge(geometry, vertex, yield_photons, qe, lb.LAMBDA0_M, cfg)
    capacity = max((len(f) for f in fractions.values()), default=0) + 1
    u = np.full((geometry.n_pmt, capacity), 0.9)  # a large increment: no further arrival
    for pmt, fracs in fractions.items():
        for j, fraction in enumerate(fracs):
            u[int(pmt), j] = np.exp(-float(fraction) * mu_raw[int(pmt)])
    return lb.SensorDraws(
        key=f"{lb.NS_DRAWS}/crafted", run_id="lucid/probe_train/0000", event_id=0,
        vertex_m=vertex, yield_photons=float(yield_photons), qe_eff=qe, poisson_u=u,
        charge_z=np.zeros(geometry.n_pmt), transit_z=np.zeros(geometry.n_pmt),
        dark_counts=np.zeros(geometry.n_pmt, dtype=np.int64), dark_times=np.zeros(0),
        window_ns=float(cfg.window_ns), capacity=capacity)


def _standin(state, draws):
    """Test-only stand-in: gains act on charge before the threshold."""
    n_pmt = int(state.gains.size)
    n = np.bincount(draws.photon_sensor, minlength=n_pmt) + draws.dark_counts
    charge = state.gains * (n + np.sqrt(n) * lb.SIGMA_SPE * draws.charge_z)
    hit = charge >= THRESHOLD_PE
    first = np.full(n_pmt, 0.9 * draws.window_ns)
    np.minimum.at(first, draws.photon_sensor, draws.photon_time_ns)
    time_ns = first + TTS_SIGMA_NS * draws.transit_z
    return lb.HitReadout(np.flatnonzero(hit), charge[hit], time_ns[hit])


# --------------------------------------------------------------------------- laws
def test_f1_distribution_and_clipping():
    u = np.random.default_rng(1).standard_normal(200_000)
    gain, clipped = lb.f1_module_gain(u, 0.05)
    assert clipped == 0
    assert gain.mean() == pytest.approx(1.0, abs=1e-3)
    assert gain.std() == pytest.approx(0.05, rel=0.02)
    forced, n = lb.f1_module_gain(np.array([-100.0, 0.0, 100.0]), 0.10)
    assert n == 2 and np.allclose(forced, [0.5, 1.0, 1.5])


def test_f3_distribution_and_clipping():
    w = np.random.default_rng(2).standard_normal(200_000)
    gain, clipped = lb.f3_pmt_gain(w, 0.05)
    assert clipped == 0
    assert gain.mean() == pytest.approx(1.0, abs=1e-3)
    assert gain.std() == pytest.approx(0.05, rel=0.02)
    forced, n = lb.f3_pmt_gain(np.array([-50.0, 0.0, 50.0]), 0.05)
    assert n == 2 and np.allclose(forced, [0.5, 1.0, 1.5])


def test_f2_distribution_is_positive():
    v = np.random.default_rng(3).standard_normal(200_000)
    lam = lb.f2_attenuation_length(v, 0.20)
    assert np.all(lam > 0)
    assert lam.mean() / lb.LAMBDA0_M == pytest.approx(np.exp(0.2 ** 2 / 2), rel=3e-3)
    assert lb.f2_attenuation_length(np.array([-1.0, 1.0]), 0.2) == pytest.approx(
        lb.LAMBDA0_M * np.exp(np.array([-0.2, 0.2])))


def test_f2p_gradient_and_clipping():
    z = np.array([-GEOM.half_height_m, 0.0, GEOM.half_height_m])
    factor, clipped = lb.f2p_gradient(1.0, 0.10, z, GEOM.half_height_m)
    assert clipped == 0 and np.allclose(factor, [0.9, 1.0, 1.1])
    forced, n = lb.f2p_gradient(100.0, 0.10, np.array([-1.0, 1.0]), 1.0)
    assert n == 2 and np.allclose(forced, [0.5, 1.5])
    v = np.random.default_rng(4).standard_normal(100_000)
    top, _ = lb.f2p_gradient(v, 0.10, 1.0, 1.0)  # top of the detector
    assert top.mean() == pytest.approx(1.0, abs=2e-3)
    assert top.std() == pytest.approx(0.10, rel=0.02)


def test_r11b_covariation_and_namespace_independence():
    state = lb.state_for_run("R11b", 0.20, "lucid/train/0000", GEOM)
    c = np.log(state.attenuation_length_m / lb.LAMBDA0_M) / 0.20
    k = lb.k_R11b(GEOM)
    assert np.allclose(state.module_gain, np.clip(1.0 + 0.05 * c * k, *lb.GAIN_CLIP))
    assert np.corrcoef(state.module_gain, k)[0, 1] == pytest.approx(np.sign(c))
    assert 0.03 < state.pmt_gain.std() < 0.08  # F3 present at the restarted s_p = 0.05
    # the two persistent vectors are independent draws
    k_r11b, k_t = lb.k_R11b(BIG), lb.k_T(BIG)
    assert not np.allclose(k_r11b, k_t)
    assert abs(np.corrcoef(k_r11b, k_t)[0, 1]) < 0.5
    # k_R11b is fixed across runs; only the scalar c changes
    other = lb.state_for_run("R11b", 0.20, "lucid/train/0001", GEOM)
    assert np.allclose(lb.k_R11b(GEOM), k)
    assert not np.allclose(state.module_gain, other.module_gain)


def test_law_t_linear_trend_and_independence():
    k = lb.k_T(GEOM)
    slow = lb.state_for_run("T", 0.10, "lucid/dev_train/0000", GEOM)
    fast = lb.state_for_run("T", 0.50, "lucid/dev_train/0000", GEOM)
    assert np.allclose(fast.module_gain - 1.0, 5.0 * (slow.module_gain - 1.0))
    assert np.allclose(slow.module_gain, np.clip(1.0 + 0.10 * k * 0.10, *lb.GAIN_CLIP))
    assert not np.allclose(lb.k_T(GEOM), lb.k_R11b(GEOM))
    in_range = lb.law_T_ratios(200, kind="in_range")
    extrap = lb.law_T_ratios(200, kind="extrapolation")
    assert np.all((in_range >= 0) & (in_range < 1))
    assert np.all((extrap >= 1.9) & (extrap <= 2.1))
    assert np.array_equal(in_range, lb.law_T_ratios(200, kind="in_range"))
    training = lb.law_T_training_ratios()
    assert training.shape == (lb.T_RUNS,) and training[0] == 0.0
    assert training[-1] == pytest.approx((lb.T_RUNS - 1) / lb.T_RUNS)
    with pytest.raises(ValueError):
        lb.law_T_ratios(10, kind="unknown")


def test_state_determinism_and_run_dependence():
    cases = [("F1", 0.05), ("F3", 0.05), ("F2", 0.20), ("F2p", 0.10),
             ("R11b", 0.20), ("T", 0.50)]
    for law, setting in cases:
        first = lb.state_for_run(law, setting, "lucid/train/0000", GEOM, seed=11)
        again = lb.state_for_run(law, setting, "lucid/train/0000", GEOM, seed=11)
        assert first.digest == again.digest
        other = lb.state_for_run(law, setting, "lucid/train/0001", GEOM, seed=11)
        assert other.digest != first.digest
        reseeded = lb.state_for_run(law, setting, "lucid/train/0000", GEOM, seed=12)
        assert reseeded.digest != first.digest
        if law != "T":  # T is a function of r/R, not run identity
            assert not np.allclose(first.gains, other.gains)


def test_every_cell_carries_f1_and_f3_as_registered():
    f1 = lb.state_for_run("F1", 0.05, "lucid/train/0000", GEOM)
    assert np.allclose(f1.gains, f1.module_gain[GEOM.module_of_pmt] * f1.pmt_gain)
    assert f1.pmt_gain.std() > 0.02
    f2 = lb.state_for_run("F2", 0.20, "lucid/probe_train/0000", GEOM)
    assert f2.attenuation_length_m != lb.LAMBDA0_M
    assert f2.module_gain.std() > 0.02 and f2.pmt_gain.std() > 0.02
    f2p = lb.state_for_run("F2p", 0.10, "lucid/probe_train/0000", GEOM)
    assert not np.allclose(f2p.gradient, 1.0)
    assert np.corrcoef(f2p.gradient, GEOM.z_pmt_m)[0, 1] != 0.0
    no_f3 = lb.state_for_run("F1", 0.05, "lucid/train/0000", GEOM, include_f3=False)
    assert np.allclose(no_f3.pmt_gain, 1.0)


def test_geometry_digest_tracks_layout():
    assert lb.synthetic_cylinder_geometry().digest == GEOM.digest
    assert lb.synthetic_cylinder_geometry(half_height_m=5.0).digest != GEOM.digest
    assert lb.synthetic_cylinder_geometry(8, 7, name="other").digest != GEOM.digest


# --------------------------------------------------------------------------- twin contract
def test_twin_is_private_and_reuses_draws():
    draws = _draws()
    with pytest.raises(PermissionError):
        lb.nominal_twin(draws, GEOM, _standin, arm="sigreg_jepa")
    twin = lb.nominal_twin(draws, GEOM, _standin, arm=lb.ORACLE_ARM)
    reference = _standin(lb.nominal_state(GEOM, run_id=draws.run_id), draws)
    assert np.array_equal(twin.pmt_id, reference.pmt_id)
    assert np.allclose(twin.charge_pe, reference.charge_pe)
    assert np.allclose(twin.time_ns, reference.time_ns)


def test_twin_shares_draws_and_gain_acts_before_threshold():
    draws = _draws(n_photons=600)
    drifted = lb.state_for_run("F1", 0.10, draws.run_id, GEOM)
    measured = _standin(drifted, draws)
    twin = lb.nominal_twin(draws, GEOM, _standin, arm=lb.ORACLE_ARM)
    common = np.intersect1d(measured.pmt_id, twin.pmt_id)
    assert common.size > 0
    ratio = measured.charge_pe[np.searchsorted(measured.pmt_id, common)] / \
        twin.charge_pe[np.searchsorted(twin.pmt_id, common)]
    assert np.allclose(ratio, drifted.gains[common])  # same charge draw, scaled by gain
    assert not np.array_equal(measured.pmt_id, twin.pmt_id) or \
        not np.allclose(measured.charge_pe, twin.charge_pe)


# --------------------------------------------------------------------------- runs / IDs / manifest
def test_run_structure_counts_and_registered_lengths():
    train = lb.COHORTS["train"]
    probe = lb.COHORTS["probe_train"]
    assert (train.n_runs, train.run_length) == (100, 1000)
    assert (probe.n_runs, probe.run_length) == (200, 50)
    for cohort in lb.COHORTS.values():
        runs = lb.make_runs(cohort, "F1", 0.05)
        assert len(runs) == cohort.n_runs
        assert sum(r.n_events for r in runs) == cohort.n_events
        assert {r.n_events for r in runs} == {cohort.run_length}


def test_run_sets_are_disjoint_and_ids_stay_in_range():
    runs = tuple(r for c in lb.COHORTS.values() for r in lb.make_runs(c, "F1", 0.05))
    assert len({r.run_id for r in runs}) == len(runs)
    spans = sorted((r.event_start, r.event_stop) for r in runs)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    by_namespace = {c.event_namespace: c for c in lb.COHORTS.values()}
    for run in runs:
        cohort = by_namespace[run.event_namespace]
        assert cohort.event_start <= run.event_start < run.event_stop <= cohort.event_start + cohort.n_events


def test_law_t_probe_runs_carry_distinct_settings():
    ratios = lb.law_T_ratios(lb.COHORTS["probe_train"].n_runs, kind="extrapolation")
    runs = lb.make_runs(lb.COHORTS["probe_train"], "T", settings=ratios)
    assert len(runs) == 200
    assert len({round(r.setting, 12) for r in runs}) == 200
    assert len({r.state(GEOM).digest for r in runs}) == 200


def test_manifest_round_trip_and_hash_verification(tmp_path):
    runs = tuple(r for c in lb.COHORTS.values() for r in lb.make_runs(c, "F1", 0.05))
    payload = lb.write_manifest(tmp_path / "manifest.json", runs, GEOM, seed=7,
                                code_sha="deadbeef")
    loaded = json.loads((tmp_path / "manifest.json").read_text())
    assert loaded == payload
    assert loaded["readout_label"] == lb.READOUT_LABEL
    assert loaded["reported_results"] is False
    assert loaded["geometry_sha256"] == GEOM.digest
    assert loaded["code_sha"] == "deadbeef"
    assert loaded["n_events"] == sum(r.n_events for r in runs)
    lb.verify_manifest(loaded, runs, GEOM, seed=7)
    loaded["runs"][3]["state_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        lb.verify_manifest(loaded, runs, GEOM, seed=7)


# --------------------------------------------------------------------------- sparse storage
def test_sparse_storage_round_trip_and_sha256sums(tmp_path):
    events = [100, 101, 102]
    readouts = [_standin(lb.state_for_run("F1", 0.05, "lucid/probe_test/0000", GEOM), _draws(event_id=e))
                for e in events]
    store = lb.sparse_from_readouts(events, readouts)
    assert store.offsets[-1] == len(store.pmt_id)
    assert np.all(np.diff(store.offsets) >= 0)
    path = lb.write_sparse_hits(tmp_path / "hits.npz", store)
    back = lb.read_sparse_hits(path)
    assert np.array_equal(back.event_ids, store.event_ids)
    assert np.array_equal(back.offsets, store.offsets)
    assert np.array_equal(back.pmt_id, store.pmt_id)
    assert np.allclose(back.charge_pe, store.charge_pe)
    assert back.readout_label == lb.READOUT_LABEL
    lb.write_sha256sums(tmp_path)
    lb.verify_sha256sums(tmp_path)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError):
        lb.verify_sha256sums(tmp_path)


# --------------------------------------------------------------------------- N_min accounting
def test_exclusion_and_redraw_accounting():
    counter = lb.ViewExclusionCounter()
    assert counter.record_across([5, 20]) is True
    assert (counter.n_across, counter.across_redraws, counter.across_accepted) == (1, 1, 1)
    assert counter.record_across([1, 2, 3, 4]) is False
    assert (counter.n_across, counter.across_redraws, counter.across_excluded) == (2, 4, 1)
    assert counter.record_across([20]) is True
    with pytest.raises(ValueError):
        counter.record_across([1, 2, 3, 4, 5])  # initial + 4 > max_redraws
    assert counter.record_within(19) is False
    assert counter.record_within(20) is True
    rates = counter.rates()
    assert rates["across_redraw_rate"] == pytest.approx(4 / 3)
    assert rates["across_exclusion_rate"] == pytest.approx(1 / 3)
    assert rates["within_exclusion_rate"] == pytest.approx(0.5)


def test_view_masks_are_a_documented_geometry_stub():
    with pytest.raises(NotImplementedError):
        lb.view_masks(GEOM, "within")


# --------------------------------------------------------------------------- detectability
def test_fisher_analytic_fixture():
    mu, dmu = np.array([100.0]), np.array([10.0])
    # s_p = 0 recovers the Poisson + SPE variance; the F3 term is s_p^2 mu^2
    info = lb.fisher_event(mu, dmu, sigma_spe=0.3, v_dark=0.0, s_p=0.0)
    assert info == pytest.approx(100.0 / (100.0 * 1.09))
    assert lb.fisher_event(mu, dmu, sigma_spe=0.3, v_dark=0.0, s_p=0.1) == pytest.approx(
        100.0 / (100.0 * 1.09 + 0.01 * 100.0 ** 2))
    assert lb.fisher_event(mu, dmu, sigma_spe=0.3, v_dark=10.0, s_p=0.0) == pytest.approx(
        100.0 / 119.0)
    # one PMT, sigma_spe = v_dark = 0: weight = mu^2/(mu + s_p^2 mu^2) = mu/(1 + s_p^2 mu)
    assert lb.fisher_event(np.array([5.0]), np.array([5.0]), sigma_spe=0.0, v_dark=0.0,
                           s_p=0.1) == pytest.approx(5.0 / (1.0 + 0.01 * 5.0))
    assert lb.r2_max(info, 0.05) == pytest.approx(0.05 ** 2 / (0.05 ** 2 + 1.0 / info))
    assert lb.r2_max(0.0, 0.05) == 0.0
    assert lb.r2_max(np.inf, 0.05) == 1.0


def test_attenuation_fisher_matches_analytic_sum():
    yield_pe, lambda0 = 100.0, lb.LAMBDA0_M
    distance = np.array([5.0, 10.0, 20.0])
    mu = yield_pe * np.exp(-distance / lambda0)
    dmu = mu * distance / lambda0  # derivative with respect to ln(lambda)
    expected = float(np.sum(mu * (distance / lambda0) ** 2))
    assert lb.fisher_event(mu, dmu, sigma_spe=0.0, v_dark=0.0, s_p=0.0) == pytest.approx(expected)


def test_fisher_monotonicity_and_run_ceiling():
    mu, dmu = np.array([10.0, 20.0, 5.0]), np.array([1.0, 2.0, 0.5])
    base = lb.fisher_event(mu, dmu)
    assert lb.fisher_event(mu, 2.0 * dmu) == pytest.approx(4.0 * base)
    assert lb.fisher_event(mu, dmu, sigma_spe=0.5) < base
    assert lb.fisher_event(mu, dmu, v_dark=5.0) < base
    assert lb.r2_max(2.0 * base, 0.1) > lb.r2_max(base, 0.1)
    assert lb.r2_max(base, 0.2) > lb.r2_max(base, 0.1)
    result = lb.detectability([mu] * lb.RUN_EVENTS, [dmu] * lb.RUN_EVENTS, 0.05)
    assert result.i_run == pytest.approx(lb.RUN_EVENTS * base)
    assert result.r2_run > result.r2_event[0]
    assert result.n_events == lb.RUN_EVENTS and not result.profiled
    assert result.s_p == lb.F3_SPREAD  # F3 is marginalised with the spread in force
    profiled = lb.detectability([mu], [dmu], 0.05, profiled_information=1.0)
    assert profiled.profiled and profiled.i_run == 1.0


def test_invalid_detectability_inputs_raise():
    with pytest.raises(ValueError):
        lb.fisher_event(np.array([-1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        lb.fisher_event(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        lb.fisher_event(np.array([1.0]), np.array([1.0]), s_p=-0.1)
    # A zero-mean PMT is skipped: no NaN and no division by zero, no information.
    assert lb.fisher_event(np.array([0.0]), np.array([1.0]), v_dark=0.0) == 0.0
    assert lb.fisher_event(np.array([0.0]), np.array([1.0])) == 0.0  # the readout's v_dark = 0
    assert np.isfinite(lb.fisher_event(np.array([0.0]), np.array([1.0]), v_dark=1.0))
    with pytest.raises(ValueError):
        lb.detectability([], [], 0.05)
    with pytest.raises(ValueError):
        lb.r2_max(1.0, 0.0)


# --------------------------------------------------------------------------- hygiene / config
def test_nothing_imports_lucid():
    root = Path(lb.__file__).resolve().parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert module != "lucid" and not module.startswith("lucid."), (path, module)


def test_config_matches_module_constants():
    config = json.loads(CONFIG.read_text())
    assert config["readout_label"] == lb.READOUT_LABEL == "readout_pmt_r6"
    assert config["dark_counts"] == lb.DARK_COUNTS_ON == "on"
    assert config["fixed_response"]["dark_counts"] == "on"
    assert config["fixed_response"]["sha256"] == lb.REFERENCE_RESPONSE_SHA256
    assert config["fixed_response"]["readout_window_ns"] == list(lb.R6_READOUT_WINDOW_NS)
    assert config["lambda0_m"] == lb.LAMBDA0_M
    assert config["gain_clip"] == list(lb.GAIN_CLIP)
    assert config["views"] == {"n_min": lb.N_MIN, "max_redraws": lb.MAX_REDRAWS}
    assert config["detectability"]["sigma_spe"] == lb.SIGMA_SPE
    assert config["detectability"]["v_dark"] == lb.V_DARK_READOUT > 0.0
    assert config["detectability"]["dark_counts"] == "on"
    assert config["detectability"]["f3_marginalised"] is True
    assert config["detectability"]["f3_marginalised_spread"] == lb.F3_SPREAD == 0.05
    assert config["detectability"]["readout_window_ns"] == list(lb.R6_READOUT_WINDOW_NS)
    assert (config["detectability"]["variance_formula"] == lb.VARIANCE_FORMULA
            == "mu_p * (1 + sigma_spe^2) + v_dark + s_p^2 * mu_p^2")
    assert config["detector_state_laws"]["F1"]["spreads"] == list(lb.F1_SPREADS)
    assert config["detector_state_laws"]["F2"]["spreads"] == list(lb.F2_SPREADS)
    assert config["detector_state_laws"]["F2p"]["spreads"] == list(lb.F2P_SPREADS)
    assert config["detector_state_laws"]["F3"]["spread"] == lb.F3_SPREAD
    assert config["detector_state_laws"]["R11b"]["spread"] == lb.R11B_SPREAD
    assert config["detector_state_laws"]["T"]["spread"] == lb.T_SPREAD
    assert config["detector_state_laws"]["T"]["runs"] == lb.T_RUNS
    for law in ("F1", "F3", "F2", "F2p"):
        assert config["detector_state_laws"][law]["decision_status"] == \
            lb.SUPERSEDED_PENDING_R6_FREEZE
    assert config["superseded_decisions"]["status"] == lb.SUPERSEDED_PENDING_R6_FREEZE
    assert "superseded_by_r6_dark_counts" in config["superseded_decisions"]["reason"]
    assert config["generator"]["yield_range_decision"]["status"] == \
        lb.SUPERSEDED_PENDING_R6_FREEZE
    for name, cohort in lb.COHORTS.items():
        entry = config["cohorts"][name]
        assert (entry["namespace"], entry["event_start"], entry["n_events"],
                entry["run_length"]) == (cohort.event_namespace, cohort.event_start,
                                         cohort.n_events, cohort.run_length)
    geo, generator = config["geometry"], config["generator"]
    assert geo["lucid_pin"] == lb.LUCID_PIN
    assert geo["config_sha256"] == lb.WCTE_CONFIG_SHA256
    assert geo["npz_sha256_prefix"] == lb.WCTE_NPZ_SHA256_PREFIX
    assert (geo["n_sensors"], geo["n_modules"], geo["pmts_per_module"]) == (
        lb.WCTE_N_SENSORS, lb.WCTE_N_MODULES, lb.WCTE_PMTS_PER_MODULE)
    assert geo["surface_counts"] == lb.WCTE_SURFACE_COUNTS
    assert geo["mpmt_id_gaps"] == [45] and geo["pmt_id_in_mpmt_base"] == 1
    assert generator["y_lo"] == lb.Y_LO_PHOTONS and generator["y_hi"] == lb.Y_HI_PHOTONS
    assert generator["yield_range_status"] == lb.YIELD_RANGE_STATUS_IN_FORCE
    assert generator["dark_rate_hz"] == lb.DARK_RATE_HZ_ON == 1000.0
    assert generator["dark_counts"] == lb.DARK_COUNTS_ON
    assert generator["readout_window_ns"] == list(lb.R6_READOUT_WINDOW_NS)
    assert generator["window_ns"] == lb.R6_WINDOW_NS == 1000.0
    assert generator["sigma_spe"] == lb.R6_SPE_SIGMA == 0.3
    assert generator["threshold_pe"] == lb.R6_THRESHOLD_PE == 0.25
    assert generator["reference_response_sha256"] == lb.REFERENCE_RESPONSE_SHA256
    assert lb.load_generator_config(CONFIG) == lb.GeneratorConfig()
    assert config["benchmark"]["volume_events"] == lb.REGISTERED_VOLUME_EVENTS
    assert config["benchmark"]["cpu_cap_wall_hours"] == lb.CPU_CAP_WALL_HOURS
    assert config["benchmark"]["cores_at_most"] == lb.BENCHMARK_CORES


# ===================================================================== WP4b: WCTE geometry
SYNTH = lb.from_synthetic()
GEN_CFG = lb.load_generator_config(CONFIG)


def test_wcte_schema_fixture_matches_registered_facts():
    g = SYNTH
    assert (g.n_pmt, g.n_modules, g.pmts_per_module) == (1995, 105, 19)
    assert dict(g.surface_counts) == lb.WCTE_SURFACE_COUNTS
    assert g.synthetic and "NOT WCTE" in g.name
    assert g.module_ids.tolist() == [m for m in range(106) if m != 45]
    assert g.module_ids_contiguous is False and 45 not in g.module_ids.tolist()
    assert np.array_equal(np.bincount(g.module_index), np.full(105, 19))
    assert np.array_equal(g.pmt_id, np.arange(1, 1996))
    assert g.pmt_id_in_mpmt.min() == 1 and g.pmt_id_in_mpmt.max() == 19
    for module in (0, 45, 104):
        inside = g.module_index == module
        assert np.array_equal(np.sort(g.pmt_id_in_mpmt[inside]), np.arange(1, 20))
    assert np.allclose(np.linalg.norm(g.directions, axis=1), 1.0)
    assert np.allclose(g.module_centres_m, np.stack(
        [g.positions_m[g.module_index == m].mean(axis=0) for m in range(g.n_modules)]))
    assert len(g.digest) == 64 and g.digest == lb.from_synthetic().digest


def test_geometry_loader_round_trip_from_config_pointer(tmp_path):
    g = SYNTH
    npz = tmp_path / "wcte_geometry.npz"
    np.savez(npz, positions_mm=g.positions_m * 1000.0, directions=g.directions,
             surfaces=g.surfaces, pmt_id=g.pmt_id, mpmt_id=g.mpmt_id,
             pmt_id_in_mpmt=g.pmt_id_in_mpmt, radius=g.radius_m, height=g.height_m,
             sensor_radius=g.sensor_radius_m)
    config = tmp_path / "geom_config_test.json"  # not the pinned filename: no implicit pin
    config.write_text(json.dumps({"material": "water", "detector_type": "cylinder",
                                  "geometry_definitions": {"npz_file_path": npz.name}}))
    loaded = lb.load_wcte_geometry(config)
    assert loaded.config_sha256 == hashlib.sha256(config.read_bytes()).hexdigest()
    assert loaded.source_sha256 and loaded.source_path == str(npz.resolve())
    assert (loaded.n_pmt, loaded.n_modules) == (1995, 105)
    assert np.allclose(loaded.positions_m, g.positions_m, atol=1e-12)
    assert np.array_equal(loaded.pmt_id_in_mpmt, g.pmt_id_in_mpmt)
    assert np.array_equal(loaded.mpmt_id, g.mpmt_id)
    assert loaded.digest == lb.load_wcte_geometry(npz).digest  # stable across loads
    assert loaded.synthetic is False and len(loaded.source_sha256) == 64
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"synthetic": True, "geometry_definitions":
                                   {"npz_file_path": npz.name}}))
    assert lb.load_wcte_geometry(fixture).synthetic is True
    with pytest.raises(ValueError, match="sha256"):
        lb.load_wcte_geometry(config, expected_config_sha256="0" * 64)
    with pytest.raises(ValueError, match="pinned"):
        lb.load_wcte_geometry(npz, expected_npz_sha256_prefix="deadbeef")
    with pytest.raises(FileNotFoundError):
        lb.load_wcte_geometry(tmp_path / "missing.npz")


def test_geometry_validation_refuses_bad_schemas():
    with pytest.raises(ValueError, match="1-based"):
        lb.from_synthetic({"pmt_id_in_mpmt_start": 0})
    with pytest.raises(ValueError, match="carries 18"):
        lb.from_synthetic({"surface_counts": {"top": 398, "barrel": 1197, "bottom": 399},
                           "drop_last_pmt_of_module": 3})
    with pytest.raises(ValueError, match="unknown surface"):
        lb.from_synthetic({"surface_names": ("top", "side", "bottom"),
                           "surface_counts": {"top": 399, "side": 1197, "bottom": 399}})
    with pytest.raises(ValueError, match="surface counts"):
        replace(SYNTH, surface_counts=(("barrel", 1111), ("bottom", 399), ("top", 399)))
    with pytest.raises(ValueError, match="unit vectors"):
        replace(SYNTH, directions=SYNTH.directions * 1.01)
    with pytest.raises(ValueError, match="positions_m"):
        replace(SYNTH, positions_m=SYNTH.positions_m[:-1])
    with pytest.raises(ValueError, match="pmt_id"):
        replace(SYNTH, pmt_id=np.zeros(SYNTH.n_pmt, dtype=np.int64))


# ===================================================================== WP4b: generator physics
def test_expected_charge_scales_with_yield_and_attenuates_far_sensors():
    vertex = np.array([0.2, -0.1, 0.05])
    qe = np.full(SYNTH.n_pmt, 0.2)
    low = lb.expected_charge(SYNTH, vertex, 1_000.0, qe, 50.0, GEN_CFG)
    high = lb.expected_charge(SYNTH, vertex, 2_000.0, qe, 50.0, GEN_CFG)
    assert low.sum() > 0.0
    assert np.allclose(high, 2.0 * low, rtol=1e-12)  # signal conservation in the total
    assert high.sum() == pytest.approx(2.0 * low.sum(), rel=1e-12)
    short = lb.expected_charge(SYNTH, vertex, 1_000.0, qe, 0.2, GEN_CFG)
    dist = np.linalg.norm(SYNTH.positions_m - vertex, axis=1)
    ratio = short / low  # exp(-d/5)/exp(-d/50) < 1, falling with distance
    assert np.corrcoef(ratio, dist)[0, 1] < -0.5
    far, near = dist > np.percentile(dist, 90), dist < np.percentile(dist, 10)
    assert ratio[far].mean() < 0.1 * ratio[near].mean()
    assert short.sum() < low.sum()
    # disc approximation: reduced to pi r^2 cos(theta)/d^2 for d >> r
    omega = lb.solid_angle_fraction(SYNTH, vertex)
    assert np.all(omega > 0) and omega.sum() < 1.0
    chosen = int(np.argmax(dist))
    d = dist[chosen]
    cos = abs(float(np.dot(SYNTH.positions_m[chosen] - vertex, SYNTH.directions[chosen]))) / d
    assert omega[chosen] == pytest.approx(SYNTH.sensor_radius_m ** 2 / (4.0 * d * d) * cos, rel=2e-2)


def test_poisson_inversion_is_exact_and_shared_across_means():
    u = lb.keyed_rng(1, "poisson-test").random((8000, 400))
    for mean in (3.0, 10.0):
        counts = lb.poisson_counts_from_uniform(u, np.full(8000, mean))
        assert counts.mean() == pytest.approx(mean, rel=0.04)
        assert counts.var() == pytest.approx(mean, rel=0.06)
    assert lb.poisson_counts_from_uniform(u, np.zeros(8000)).max() == 0
    lower = lb.poisson_counts_from_uniform(u, np.full(8000, 5.0))
    upper = lb.poisson_counts_from_uniform(u, np.full(8000, 8.0))
    assert np.all(lower <= upper)  # one shared draw, ordered by the mean
    assert lb.count_capacity(10.0) > 10.0 + lb.count_capacity(0.0) - 10.0
    with pytest.raises(ValueError):
        lb.poisson_counts_from_uniform(u, np.full(7999, 1.0))


def test_generation_is_reproducible_and_shares_one_state_per_run():
    first_run = lb.RunSpec("lucid/probe_train/0000", "lucid/probe_train", "F1", 0.05, 0, 30, 7)
    second_run = lb.RunSpec("lucid/probe_train/0001", "lucid/probe_train", "F1", 0.05, 30, 30, 7)
    a = lb.generate_run(SYNTH, GEN_CFG, first_run, with_twin=False)
    b = lb.generate_run(SYNTH, GEN_CFG, first_run, with_twin=False)
    assert np.array_equal(a.event_ids, np.arange(0, 30))
    assert a.state.digest == b.state.digest == first_run.state(SYNTH).digest
    for left, right in zip(a.hits, b.hits):
        assert np.array_equal(left.pmt_id, right.pmt_id)
        assert np.allclose(left.charge_pe, right.charge_pe)
        assert np.allclose(left.time_ns, right.time_ns)
    c = lb.generate_run(SYNTH, GEN_CFG, second_run, with_twin=False)
    assert c.state.digest != a.state.digest
    assert np.array_equal(c.event_ids, np.arange(30, 60))
    # a run of 50 events is the registered probe length and every event shares the state
    probe = lb.COHORTS["probe_train"]
    assert probe.run_length == 50 and lb.COHORTS["train"].run_length == 1000
    assert a.expected_pe.shape == (30,) and np.all(a.expected_pe > 0)
    assert a.expected_pe.sum() > 0 and a.n_hits > 0


def test_nominal_twin_equals_the_undrifted_readout():
    readout = lb.make_readout(SYNTH, GEN_CFG)
    draws = lb.draw_event(SYNTH, GEN_CFG, "lucid/probe_train/0000", 5, seed=11)
    undrifted = lb.state_for_run("F1", 0.0, draws.run_id, SYNTH, include_f3=False)
    assert np.allclose(undrifted.gains, 1.0)
    assert undrifted.attenuation_length_m == lb.LAMBDA0_M
    measured = readout(undrifted, draws)
    twin = lb.nominal_twin(draws, SYNTH, readout, arm=lb.ORACLE_ARM,
                           lambda0_m=GEN_CFG.lambda0_m)
    assert np.array_equal(measured.pmt_id, twin.pmt_id)
    assert np.array_equal(measured.charge_pe, twin.charge_pe)
    assert np.array_equal(measured.time_ns, twin.time_ns)
    with pytest.raises(PermissionError):
        lb.nominal_twin(draws, SYNTH, readout, arm="sigreg_jepa")
    drifted = lb.state_for_run("F2", 0.20, draws.run_id, SYNTH)
    other = readout(drifted, draws)
    assert not np.array_equal(measured.pmt_id, other.pmt_id) or \
        not np.allclose(measured.charge_pe, other.charge_pe)


def test_threshold_is_applied_after_gain_and_unhit_hits_are_not_recorded():
    cfg = replace(GEN_CFG, threshold_pe=1.5, walk_calibration_pe=1.5)
    odd = int(np.flatnonzero(SYNTH.module_index == 45)[0])
    even = int(np.flatnonzero(SYNTH.module_index == 44)[0])
    # exact 1 PE and 2 PE pulses, no SPE smearing: the 1 PE pulse is below 1.5 PE
    draws = _fraction_draws(SYNTH, cfg, {odd: [0.5], even: [0.3, 0.3]})
    nominal = lb.nominal_state(SYNTH, draws.run_id)
    hits = lb.readout_synthetic_unrepaired(nominal, draws, SYNTH, cfg)
    assert hits.pmt_id.tolist() == [even]
    assert hits.charge_pe[0] == pytest.approx(2.0)
    gains = np.ones(SYNTH.n_pmt)
    gains[odd] = 1.5  # gain acts before the threshold: 1 PE x 1.03 x 1.5 now crosses
    lifted = lb.readout_synthetic_unrepaired(replace(nominal, gains=gains), draws, SYNTH, cfg)
    assert odd in lifted.pmt_id.tolist() and even in lifted.pmt_id.tolist()


# ===================================================================== WP4b: fixed response
def test_fixed_parity_gain_uses_the_dense_module_index():
    ref = lb.reference_response()
    assert ref.gain_rule == {"rule": "group_parity", "base": 1.0, "odd_group_delta": 0.03}
    assert ref.timing_width_rule == {"rule": "group_parity", "base": 1.0,
                                     "odd_group_delta": 0.05}
    gain = lb.fixed_charge_gain(SYNTH)
    assert np.all(gain[SYNTH.module_index % 2 == 1] == pytest.approx(1.03))
    assert np.all(gain[SYNTH.module_index % 2 == 0] == pytest.approx(1.0))
    width = lb.fixed_timing_width_factor(SYNTH)
    assert np.all(width[SYNTH.module_index % 2 == 1] == pytest.approx(1.05))
    assert np.all(width[SYNTH.module_index % 2 == 0] == pytest.approx(1.0))
    # dense index 45 is the module whose raw mpmt_id is 46 (id 45 is absent from WCTE)
    module_45 = SYNTH.module_index == 45
    assert SYNTH.module_ids[45] == 46
    assert np.all(gain[module_45] == pytest.approx(1.03))
    assert np.array_equal(lb.dense_module_index_from_mpmt_id(SYNTH.mpmt_id),
                          SYNTH.module_index)
    # a rule on the raw sparse id would put that module at 1.0: not interchangeable
    assert not np.allclose(gain[module_45], 1.0 + 0.03 * (SYNTH.mpmt_id[module_45] % 2))
    with pytest.raises(ValueError):
        replace(ref, gain_rule={"rule": "unknown"}).gain(np.array([0, 1]))


def test_fixed_response_values_come_from_the_manifest(tmp_path):
    from noise_module_lucid.response_manifest import value_sha256
    payload = json.loads(Path(lb.reference_response().path).read_text())
    payload["parameters"]["gain"]["value"]["odd_group_delta"] = 0.07
    payload["parameters"]["gain"]["sha256"] = value_sha256(
        payload["parameters"]["gain"]["value"])
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload))
    ref = lb.reference_response(path)
    assert ref.gain_rule["odd_group_delta"] == 0.07 and ref.sha256 != lb.REFERENCE_RESPONSE_SHA256
    assert np.allclose(ref.gain(np.array([0, 1, 2, 3])), [1.0, 1.07, 1.0, 1.07])
    with pytest.raises(ValueError):  # a tampered value fails the recorded parameter hash
        bad = dict(payload)
        bad["parameters"] = dict(payload["parameters"])
        bad["parameters"]["gain"] = dict(payload["parameters"]["gain"])
        bad["parameters"]["gain"]["value"] = dict(payload["parameters"]["gain"]["value"],
                                                  odd_group_delta=0.5)
        (tmp_path / "tampered.json").write_text(json.dumps(bad))
        lb.reference_response(tmp_path / "tampered.json")


def test_readout_applies_the_parity_gain_and_the_twin_keeps_it(monkeypatch):
    cfg = replace(GEN_CFG, threshold_pe=0.25, walk_calibration_pe=0.25)
    odd = int(np.flatnonzero(SYNTH.module_index == 45)[0])   # dense 45 -> x1.03
    even = int(np.flatnonzero(SYNTH.module_index == 44)[0])  # dense 44 -> x1.00
    # arrive at 1.02 x the gain-included mean: only the +3% module counts the arrival
    draws = _fraction_draws(SYNTH, cfg, {odd: [1.02], even: [1.02]})
    nominal = lb.nominal_state(SYNTH, draws.run_id)
    readout = lb.make_readout(SYNTH, cfg)
    measured = readout(nominal, draws)
    assert measured.pmt_id.tolist() == [odd]
    assert measured.charge_pe[0] == pytest.approx(1.0)  # 1 PE, no SPE smearing
    # the nominal twin keeps the fixed response: it is detector response, not drift
    twin = lb.nominal_twin(draws, SYNTH, readout, arm=lb.ORACLE_ARM)
    assert twin.pmt_id.tolist() == [odd]
    assert twin.charge_pe[0] == pytest.approx(1.0)
    # without the fixed gain the same arrival is lost: the gain moved the expected charge
    monkeypatch.setattr(lb, "fixed_charge_gain",
                        lambda geometry, reference=None: np.ones(geometry.n_pmt))
    assert readout(nominal, draws).pmt_id.size == 0
    monkeypatch.undo()
    # the twin cancels exactly the injected F1/F3 gain: an injected 0.2 drops the hit
    drifted = replace(nominal, gains=np.where(np.arange(SYNTH.n_pmt) == odd, 0.2, 1.0))
    assert readout(drifted, draws).pmt_id.size == 0
    assert lb.nominal_twin(draws, SYNTH, readout, arm=lb.ORACLE_ARM).pmt_id.tolist() == [odd]


def test_twin_ratios_cancel_only_the_injected_gain():
    run = lb.RunSpec("lucid/probe_train/0000", "lucid/probe_train", "F1", 0.10, 0, 20, 7)
    generated = lb.generate_run(SYNTH, GEN_CFG, run, with_twin=True)
    assert not np.allclose(generated.state.gains, 1.0)
    common_hits = 0
    for measured, twin in zip(generated.hits, generated.twin_hits):
        common = np.intersect1d(measured.pmt_id, twin.pmt_id)
        common_hits += common.size
        if common.size:
            ratio = measured.charge_pe[np.searchsorted(measured.pmt_id, common)] / \
                twin.charge_pe[np.searchsorted(twin.pmt_id, common)]
            assert np.allclose(ratio, generated.state.gains[common], rtol=1e-6)
    assert common_hits > 0


def test_dark_counts_are_on_and_derived_from_the_manifest():
    """R-6: dark counts on, every value derived from configs/lucid_pmt/reference.json."""
    ref = lb.reference_response()
    assert GEN_CFG.dark_counts == lb.DARK_COUNTS_ON == "on"
    assert GEN_CFG.dark_rate_hz == ref.dark_rate_hz == 1000.0
    assert GEN_CFG.readout_window_ns == ref.readout_window_ns == (-200.0, 800.0)
    assert GEN_CFG.window_ns == ref.window_ns == 1000.0
    assert GEN_CFG.sigma_spe == ref.spe_sigma == 0.3
    assert GEN_CFG.threshold_pe == ref.threshold_pe == 0.25
    assert "R-6" in ref.dark_stage_owner and "bypass QE/CE" in ref.dark_stage_owner
    # v_dark = lambda (1 + sigma_spe^2) = 0.001 x 1.09 = 0.00109 PE^2, not a literal
    lam = GEN_CFG.dark_rate_hz * GEN_CFG.window_ns * 1e-9
    assert lam == pytest.approx(0.001)
    assert lb.V_DARK_READOUT == pytest.approx(0.00109)
    assert lb.V_DARK_READOUT == pytest.approx(lam * (1.0 + GEN_CFG.sigma_spe ** 2))
    assert lb.V_DARK_READOUT != lb.V_DARK  # the 4200 Hz scaffold value is history
    with pytest.raises(ValueError):  # a rate that is not the manifest's is refused
        replace(GEN_CFG, dark_rate_hz=4200.0)
    with pytest.raises(ValueError):  # R-6 has no dark-off mode in this module
        replace(GEN_CFG, dark_counts="off")
    with pytest.raises(ValueError):  # the window is the manifest's, not a choice
        replace(GEN_CFG, readout_window_ns=(-100.0, 900.0))
    with pytest.raises(ValueError):  # an inconsistent pair is refused too
        replace(GEN_CFG, readout_window_ns=(0.0, 0.0))


def test_dark_hits_per_event_are_about_two_on_a_wcte_sized_geometry():
    """R-6 expectation: 1995 PMTs x 0.001 = ~2 dark hits per event (zero is common)."""
    # a small yield range keeps the shared Poisson columns small; dark draws do not depend on it
    cfg = replace(GEN_CFG, y_lo=100.0, y_hi=200.0)
    totals, times = [], []
    for event_id in range(200):
        draws = lb.draw_event(SYNTH, cfg, "lucid/probe_train/0000", event_id)
        assert draws.dark_counts.shape == (SYNTH.n_pmt,)
        assert np.all(draws.dark_counts >= 0)
        assert draws.dark_times.shape == (int(draws.dark_counts.sum()),)
        assert draws.dark_counts.max() <= 2  # a zero count on ~99.9% of PMTs
        totals.append(int(draws.dark_counts.sum()))
        times.append(draws.dark_times)
    mean = float(np.mean(totals))
    expected = SYNTH.n_pmt * cfg.dark_rate_hz * cfg.window_ns * 1e-9
    assert 1.5 <= mean <= 2.6, (mean, expected)  # ~2/event, Poisson scatter over 200 events
    assert mean == pytest.approx(expected, rel=0.25)
    times = np.concatenate(times)
    start, end = cfg.readout_window_ns
    assert times.size > 300 and np.all(times >= start) and np.all(times <= end)
    # uniform in the window, measured from the window start
    assert times.mean() == pytest.approx(0.5 * (start + end), abs=0.12 * (end - start))
    # the draw is keyed: the same event reproduces, a different event does not
    again = lb.draw_event(SYNTH, cfg, "lucid/probe_train/0000", 0)
    other = lb.draw_event(SYNTH, cfg, "lucid/probe_train/0000", 1)
    assert np.array_equal(again.dark_counts, lb.draw_event(SYNTH, cfg, "lucid/probe_train/0000", 0).dark_counts)
    assert not (np.array_equal(again.dark_counts, other.dark_counts)
                and np.array_equal(again.dark_times, other.dark_times))


def test_a_dark_hit_can_make_a_pmt_fire_and_sets_its_time():
    """Dark PE join before digitisation and share the gain/threshold/walk path."""
    cfg = replace(GEN_CFG, threshold_pe=0.25, walk_calibration_pe=0.25)
    # no source photon anywhere; one dark PE on a +3% module and one on a 1.00 module
    odd = int(np.flatnonzero(SYNTH.module_index == 45)[0])   # dense 45 -> x1.03
    even = int(np.flatnonzero(SYNTH.module_index == 44)[0])  # dense 44 -> x1.00
    assert even < odd                       # dark times are ordered like the PMT indices
    early, late = -150.0, 640.0             # both inside the R-6 window [-200, 800]
    draws = _fraction_draws(SYNTH, cfg, {})
    dark = np.zeros(SYNTH.n_pmt, dtype=np.int64)
    dark[[even, odd]] = 1
    draws = replace(draws, dark_counts=dark, dark_times=np.array([early, late]))
    nominal = lb.nominal_state(SYNTH, draws.run_id)
    hits = lb.readout_synthetic_unrepaired(nominal, draws, SYNTH, cfg)
    assert sorted(hits.pmt_id.tolist()) == sorted([odd, even])  # dark alone crosses it
    q = dict(zip(hits.pmt_id.tolist(), hits.charge_pe.tolist()))
    # one dark PE is one recorded PE on either module (the fixed parity gain scales the
    # *expected source* charge, exactly as for every other recorded digit in this readout)
    assert q[odd] == pytest.approx(1.0) and q[even] == pytest.approx(1.0)
    t = dict(zip(hits.pmt_id.tolist(), hits.time_ns.tolist()))
    # the dark PE sets the digit time, measured from the window start (transit_z = 0 here)
    assert t[even] == pytest.approx(early) and t[odd] == pytest.approx(late)
    # the injected gain is applied to dark charge too: a 0.1 gain loses the dark hit
    killed = replace(nominal, gains=np.where(np.arange(SYNTH.n_pmt) == even, 0.1, 1.0))
    assert lb.readout_synthetic_unrepaired(killed, draws, SYNTH, cfg).pmt_id.tolist() == [odd]
    # a dark PE on an already-hit PMT adds charge like any other photoelectron
    with_source = _fraction_draws(SYNTH, cfg, {even: [0.30]})
    both = replace(with_source, dark_counts=dark, dark_times=np.array([early, late]))
    hits_both = lb.readout_synthetic_unrepaired(nominal, both, SYNTH, cfg)
    charge = dict(zip(hits_both.pmt_id.tolist(), hits_both.charge_pe.tolist()))
    assert charge[even] == pytest.approx(2.0)   # 1 source + 1 dark PE, no SPE smearing
    # times outside the manifest window are refused rather than silently accepted
    with pytest.raises(ValueError, match="window"):
        lb.readout_synthetic_unrepaired(nominal, replace(both, dark_times=np.array([900.0, 0.0])),
                                        SYNTH, cfg)
    with pytest.raises(ValueError, match="dark_times"):
        lb.readout_synthetic_unrepaired(nominal, replace(both, dark_times=np.zeros(1)), SYNTH, cfg)


def test_readout_label_is_r6_everywhere(tmp_path):
    assert lb.READOUT_LABEL == "readout_pmt_r6"
    assert lb.READOUT_LABEL_LEGACY == "readout_synthetic_unrepaired"
    assert lb.READOUT_LABEL_LEGACY_DARK_OFF == "readout_dark_off"
    config = json.loads(CONFIG.read_text())
    assert config["readout_label"] == "readout_pmt_r6"
    run = lb.RunSpec("lucid/probe_train/0000", "lucid/probe_train", "F1", 0.05, 0, 4, 7)
    generated = lb.generate_run(SYNTH, GEN_CFG, run, with_twin=False)
    payload = lb.write_generated_run(tmp_path / "run", generated, SYNTH, GEN_CFG,
                                     code_sha="deadbeef")
    assert payload["readout_label"] == "readout_pmt_r6"
    assert payload["dark_counts"] == "on"
    assert payload["dark_rate_hz"] == 1000.0
    assert payload["readout_window_ns"] == list(lb.R6_READOUT_WINDOW_NS)
    assert payload["window_ns"] == 1000.0
    assert payload["v_dark"] == lb.V_DARK_READOUT
    assert payload["fixed_response"]["sha256"] == lb.reference_response().sha256
    assert payload["fixed_response"]["relpath"] == lb.FIXED_RESPONSE_RELPATH
    assert payload["fixed_response"]["dark_counts"] == "on"
    assert payload["yield_range_decision_status"] == lb.SUPERSEDED_PENDING_R6_FREEZE
    assert lb.read_sparse_hits(tmp_path / "run" / "hits.npz").readout_label == "readout_pmt_r6"
    # R7b depends on the dark bank being real: the run records dark hits, not zeros
    assert payload["n_hits"] == generated.n_hits
    benchmark = lb.benchmark_generation(SYNTH, GEN_CFG, n_events=4, with_twin=False)
    assert benchmark["readout_label"] == "readout_pmt_r6"
    assert benchmark["dark_counts"] == "on" and benchmark["dark_rate_hz"] == 1000.0
    assert benchmark["readout_window_ns"] == list(lb.R6_READOUT_WINDOW_NS)
    assert benchmark["v_dark"] == lb.V_DARK_READOUT
    assert benchmark["fixed_response"]["sha256"] == lb.reference_response().sha256
    # the legacy labels are constants only: no written artefact carries them as its label
    text = json.dumps([payload, benchmark])
    assert "readout_synthetic_unrepaired" not in text
    assert "readout_dark_off" not in text
    assert "readout_pmt_r6" in text


def test_verify_manifest_checks_the_r6_dark_provenance(tmp_path):
    runs = (lb.RunSpec("lucid/probe_train/0000", "lucid/probe_train", "F1", 0.05, 0, 2, 7),)
    payload = lb.write_manifest(tmp_path / "manifest.json", runs, GEOM, seed=7,
                                code_sha="deadbeef")
    lb.verify_manifest(payload, runs, GEOM, seed=7)
    for key, value in (("dark_counts", "off"), ("dark_rate_hz", 0.0),
                       ("readout_window_ns", [-200.0, 900.0]), ("window_ns", 900.0),
                       ("v_dark", 0.0), ("readout_label", "readout_dark_off")):
        broken = dict(payload)
        broken[key] = value
        with pytest.raises(ValueError):
            lb.verify_manifest(broken, runs, GEOM, seed=7)


def test_generator_config_records_the_fixed_response_source_and_hash():
    config = json.loads(CONFIG.read_text())
    block, generator = config["fixed_response"], config["generator"]
    source = Path(lb.resolve_reference_path(block["path"]))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.is_file() and source.name == "reference.json"
    assert block["sha256"] == generator["reference_response_sha256"] == actual
    assert block["sha256"] == lb.REFERENCE_RESPONSE_SHA256
    assert block["path"] == generator["reference_response_path"] == lb.FIXED_RESPONSE_RELPATH
    ref = lb.reference_response()
    assert ref.sha256 == actual and ref.dark_counts == "on"
    assert ref.gain_rule == block["gain"] == ref.parameters["gain"]
    assert ref.timing_width_rule == block["timing_width_factor"]
    assert GEN_CFG.reference_response_sha256 == actual
    with pytest.raises(ValueError):
        replace(GEN_CFG, reference_response_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(GEN_CFG, reference_response_path="configs/lucid_pmt/missing.json")


def test_timing_width_factor_scales_the_digit_gaussian_width():
    cfg = replace(GEN_CFG, threshold_pe=0.25, walk_calibration_pe=0.25)
    odd = int(np.flatnonzero(SYNTH.module_index == 45)[0])   # dense 45 -> x1.05
    even = int(np.flatnonzero(SYNTH.module_index == 44)[0])  # dense 44 -> x1.00
    draws = _fraction_draws(SYNTH, cfg, {odd: [0.5], even: [0.5]})
    draws = replace(draws, transit_z=np.ones(SYNTH.n_pmt))
    hits = lb.readout_synthetic_unrepaired(lb.nominal_state(SYNTH, draws.run_id), draws,
                                           SYNTH, cfg)
    tau = lb.propagation_time_ns(SYNTH, draws.vertex_m, cfg)
    times = dict(zip(hits.pmt_id.tolist(), hits.time_ns.tolist()))
    sigma = GEN_CFG.tts_sigma_ns
    assert times[odd] == pytest.approx(tau[odd] + sigma * 1.05)
    assert times[even] == pytest.approx(tau[even] + sigma * 1.00)


# ===================================================================== WP4b: storage/benchmark
def test_generated_run_storage_manifest_and_sha256sums(tmp_path):
    run = lb.RunSpec("lucid/probe_train/0000", "lucid/probe_train", "F2", 0.20, 200_000, 20, 7)
    generated = lb.generate_run(SYNTH, GEN_CFG, run, with_twin=True)
    payload = lb.write_generated_run(tmp_path / "run", generated, SYNTH, GEN_CFG, code_sha="deadbeef")
    assert payload["readout_label"] == lb.READOUT_LABEL
    assert payload["reported_results"] is False
    assert payload["geometry_sha256"] == SYNTH.digest and payload["code_sha"] == "deadbeef"
    assert payload["generator_config_sha256"] == GEN_CFG.digest
    assert payload["geometry_config_sha256"] is None and payload["geometry_synthetic"] is True
    assert payload["yield_range_photons"] == [GEN_CFG.y_lo, GEN_CFG.y_hi]
    assert payload["yield_range_status"] == lb.YIELD_RANGE_STATUS_IN_FORCE
    assert payload["runs"][0]["law"] == "F2" and payload["runs"][0]["setting"] == 0.20
    assert payload["n_events"] == 20 and payload["n_hits"] == generated.n_hits
    assert payload["twin_present"] is True
    lb.verify_manifest(payload, (run,), SYNTH, seed=7)
    lb.verify_sha256sums(tmp_path / "run")
    back = lb.read_sparse_hits(tmp_path / "run" / "hits.npz")
    assert back.readout_label == lb.READOUT_LABEL
    assert np.array_equal(back.event_ids, generated.event_ids)
    assert np.array_equal(back.offsets,
                          lb.sparse_from_readouts(generated.event_ids, generated.hits).offsets)
    assert lb.read_sparse_hits(tmp_path / "run" / "twin_hits.npz").offsets[-1] > 0
    truth = np.load(tmp_path / "run" / "events.npz")
    assert np.allclose(truth["yield_photons"], generated.yields_photons)
    assert np.allclose(truth["vertex_m"], generated.vertices_m)
    twins = np.load(tmp_path / "run" / "twin_hits.npz")["offsets"]
    hits = np.load(tmp_path / "run" / "hits.npz")["offsets"]
    assert twins.shape == hits.shape == (21,)


def test_benchmark_reports_rate_memory_and_yield_terciles():
    result = lb.benchmark_generation(SYNTH, GEN_CFG, n_events=40, with_twin=True)
    assert result["benchmark_events"] == 40 and result["events_per_s"] > 0.0
    assert result["seconds"] > 0.0 and result["peak_rss_mb"] > 0.0
    assert result["is_synthetic_geometry"] is True and result["readout_label"] == lb.READOUT_LABEL
    assert [row["tercile"] for row in result["terciles"]] == ["low", "mid", "high"]
    assert sum(row["n_events"] for row in result["terciles"]) == 40
    assert result["terciles"][0]["yield_photons_median"] < result["terciles"][2]["yield_photons_median"]
    assert result["volume_events"] == lb.REGISTERED_VOLUME_EVENTS
    assert sum(lb.REGISTERED_VOLUME_BREAKDOWN.values()) == lb.REGISTERED_VOLUME_EVENTS
    assert result["cpu_cap_wall_hours"] == lb.CPU_CAP_WALL_HOURS == 48.0
    assert result["projected_hours_one_process"] > 0.0
    assert result["fraction_of_cap_16_cores"] > 0.0
