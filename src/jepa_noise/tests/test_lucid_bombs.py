"""Tests for the WP4b development-only LUCiD generator contract."""
import ast
import json
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
    assert 0.03 < state.pmt_gain.std() < 0.08  # F3 is present
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
    with pytest.raises(ValueError):
        lb.fisher_event(np.array([0.0]), np.array([1.0]))
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
