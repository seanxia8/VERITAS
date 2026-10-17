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
    assert 0.05 < state.pmt_gain.std() < 0.12  # F3 present at the frozen s_p = 0.10
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
    info = lb.fisher_event(mu, dmu, sigma_spe=0.3, v_dark=0.0)
    assert info == pytest.approx(100.0 / (100.0 * 1.09))
    assert lb.fisher_event(mu, dmu, sigma_spe=0.3, v_dark=10.0) == pytest.approx(100.0 / 119.0)
    assert lb.r2_max(info, 0.05) == pytest.approx(0.05 ** 2 / (0.05 ** 2 + 1.0 / info))
    assert lb.r2_max(0.0, 0.05) == 0.0
    assert lb.r2_max(np.inf, 0.05) == 1.0


def test_attenuation_fisher_matches_analytic_sum():
    yield_pe, lambda0 = 100.0, lb.LAMBDA0_M
    distance = np.array([5.0, 10.0, 20.0])
    mu = yield_pe * np.exp(-distance / lambda0)
    dmu = mu * distance / lambda0  # derivative with respect to ln(lambda)
    expected = float(np.sum(mu * (distance / lambda0) ** 2))
    assert lb.fisher_event(mu, dmu, sigma_spe=0.0, v_dark=0.0) == pytest.approx(expected)


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
    profiled = lb.detectability([mu], [dmu], 0.05, profiled_information=1.0)
    assert profiled.profiled and profiled.i_run == 1.0


def test_invalid_detectability_inputs_raise():
    with pytest.raises(ValueError):
        lb.fisher_event(np.array([-1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        lb.fisher_event(np.array([1.0, 2.0]), np.array([1.0]))
    # mu = 0 is a dark-only PMT and is informative once v_dark > 0; only a
    # zero-variance (v_dark = 0) configuration is invalid.
    assert np.isfinite(lb.fisher_event(np.array([0.0]), np.array([1.0])))
    with pytest.raises(ValueError):
        lb.fisher_event(np.array([0.0]), np.array([1.0]), v_dark=0.0)
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
    assert config["readout_label"] == lb.READOUT_LABEL
    assert config["lambda0_m"] == lb.LAMBDA0_M
    assert config["gain_clip"] == list(lb.GAIN_CLIP)
    assert config["views"] == {"n_min": lb.N_MIN, "max_redraws": lb.MAX_REDRAWS}
    assert config["detectability"]["sigma_spe"] == lb.SIGMA_SPE
    assert config["detectability"]["v_dark"] == lb.V_DARK
    assert config["detector_state_laws"]["F1"]["spreads"] == list(lb.F1_SPREADS)
    assert config["detector_state_laws"]["F2"]["spreads"] == list(lb.F2_SPREADS)
    assert config["detector_state_laws"]["F2p"]["spreads"] == list(lb.F2P_SPREADS)
    assert config["detector_state_laws"]["F3"]["spread"] == lb.F3_SPREAD
    assert config["detector_state_laws"]["R11b"]["spread"] == lb.R11B_SPREAD
    assert config["detector_state_laws"]["T"]["spread"] == lb.T_SPREAD
    assert config["detector_state_laws"]["T"]["runs"] == lb.T_RUNS
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
    draws = lb.draw_event(SYNTH, cfg, "lucid/probe_train/0000", 9, seed=3)
    dark = np.zeros(SYNTH.n_pmt, dtype=np.int64)
    dark[3], dark[7] = 1, 2  # 1 PE and 2 PE, no smearing
    draws = replace(draws, yield_photons=0.0, dark_counts=dark,
                    charge_z=np.zeros(SYNTH.n_pmt))
    nominal = lb.nominal_state(SYNTH, draws.run_id)
    hits = lb.readout_synthetic_unrepaired(nominal, draws, SYNTH, cfg)
    assert hits.pmt_id.tolist() == [7]  # the 1 PE pulse never crosses 1.5 PE
    assert hits.charge_pe[0] == pytest.approx(2.0)
    gains = np.ones(SYNTH.n_pmt)
    gains[3] = 1.5  # gain acts before the threshold: 1 PE x 1.5 now crosses
    lifted = lb.readout_synthetic_unrepaired(replace(nominal, gains=gains), draws, SYNTH, cfg)
    assert 3 in lifted.pmt_id.tolist() and 7 in lifted.pmt_id.tolist()


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
