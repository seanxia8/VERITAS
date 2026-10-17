"""Tests for the WP4b §3.5.3a detectability check (CPU, tiny)."""
import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb
from jepa_noise import lucid_detectability as ld

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lucid_detectability_v0.json"
GEN_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lucid_bombs_v0.json"
GEOM = lb.from_synthetic()
GEN = lb.load_generator_config(GEN_CONFIG)


def _row(factor, setting, r2, threshold=0.6):
    return {"factor": factor, "setting": float(setting), "law": factor,
            "sigma_f": 0.05, "n_events": 4, "i_event_median": 1.0, "i_run_median": 1.0,
            "r2_event_median": r2, "r2_run_median": r2, "r2_run_min": r2,
            "r2_run_max": r2, "passes": bool(r2 >= threshold), "threshold": threshold}


def _table(f1=None, f3=0.9, f2=0.9, t=0.9, law="F2", setting=0.10, threshold=0.6):
    f1 = {0.02: 0.9, 0.05: 0.9, 0.10: 0.9} if f1 is None else f1
    return ld.CeilingTable(
        median_yield=50000.0,
        f1={s: _row("F1", s, r, threshold) for s, r in f1.items()},
        f3=_row("F3", 0.05, f3, threshold), f2=_row(law, setting, f2, threshold),
        t=_row("T", 0.5, t, threshold), f2_law=law, f2_setting=setting, n_events=4)


def _fake_fn(mapping, default=None):
    def fn(y_hi, law, setting, spread):
        key = (law, round(float(setting), 6), round(float(spread), 6))
        return mapping.get(key, default if default is not None else _table())
    return fn


# --------------------------------------------------------------------------- analytic toys
def test_single_pmt_fisher_is_analytic():
    mu, dmu, sigma = 5.0, 2.0, 0.3
    weights = ld.poisson_weights(np.array([mu]), sigma, 0.0)
    matrix = ld.fisher_matrix(np.array([[dmu / mu]]), weights)
    expected = dmu ** 2 / (mu * (1.0 + sigma ** 2))
    assert matrix[0, 0] == pytest.approx(expected)
    assert ld.r2_max(expected, 0.05) == pytest.approx(0.05 ** 2 / (0.05 ** 2 + 1.0 / expected))
    assert ld.r2_max(0.0, 0.05) == 0.0 and ld.r2_max(np.inf, 0.05) == 1.0
    assert ld.r2_max(2 * expected, 0.1) > ld.r2_max(expected, 0.1) > 0.0
    assert ld.poisson_weights(np.array([mu]), 0.5, 0.0)[0] < weights[0]


def test_profiling_reduces_to_naive_when_nuisance_is_orthogonal():
    weights = np.array([2.0, 3.0])
    signal = np.array([[1.0], [0.0]])
    orthogonal = np.array([[0.0], [1.0]])
    naive = ld.fisher_matrix(signal, weights)[0, 0]
    joint = ld.fisher_matrix(np.hstack([signal, orthogonal]), weights)
    assert ld.profiled_information(joint, np.array([1.0, 0.0])) == pytest.approx(naive)
    zero_nuisance = np.hstack([signal, np.zeros((2, 1))])
    assert ld.profiled_information(ld.fisher_matrix(zero_nuisance, weights),
                                   np.array([1.0, 0.0])) == pytest.approx(naive)


def test_profiling_lowers_information_and_ceiling_is_monotone_in_runs():
    weights = np.array([4.0, 9.0])
    signal = np.array([[1.0], [1.0]])
    nuisance = np.array([[1.0], [0.0]])
    joint = ld.fisher_matrix(np.hstack([signal, nuisance]), weights)
    profiled = ld.profiled_information(joint, np.array([1.0, 0.0]))
    assert profiled < ld.fisher_matrix(signal, weights)[0, 0]
    one = ld.r2_max(profiled, 0.05)
    two = ld.r2_max(2.0 * profiled, 0.05)
    assert 0.0 < one < two < 1.0


def test_vertex_derivative_matches_finite_difference():
    check = ld.load_check_config(CONFIG)
    event = ld.sample_events(GEOM, GEN, 50000.0, 1, check.seed)[0]
    state = lb.state_for_run("F1", 0.05, "test/vertex", GEOM, seed=check.seed)
    mu0 = ld.true_expected_charge(GEOM, GEN, state, event)
    analytic = ld.vertex_log_derivatives(GEOM, event.vertex_m, state.attenuation_length_m)
    eps = 1e-5
    for axis in range(3):
        vertex = event.vertex_m.copy()
        vertex[axis] += eps
        mu = lb.expected_charge(GEOM, vertex, event.yield_photons, event.qe_eff,
                                state.attenuation_length_m, GEN) * state.gains
        finite = (np.log(np.maximum(mu, 1e-300)) - np.log(np.maximum(mu0, 1e-300))) / eps
        mask = mu0 > 1e-3
        assert np.max(np.abs(finite[mask] - analytic[mask, axis])) < 1e-3


def test_median_yield_policy_and_config_consistency():
    assert ld.median_yield(6000.0, 54000.0) == pytest.approx(18000.0)
    with pytest.raises(ValueError):
        ld.median_yield(100.0, 10.0)
    check = ld.load_check_config(CONFIG)
    payload = json.loads(CONFIG.read_text())
    assert payload["schema"] == ld.SCHEMA and payload["readout_label"] == lb.READOUT_LABEL
    assert payload["reported_results"] is False
    assert check.f1_spreads == lb.F1_SPREADS and check.f2_spreads == lb.F2_SPREADS
    assert check.f2p_spreads == lb.F2P_SPREADS and check.f3_spread == lb.F3_SPREAD_DEFAULT
    assert check.y_lo == lb.Y_LO_PHOTONS_PLACEHOLDER
    assert check.y_hi_initial == lb.Y_HI_PHOTONS_PLACEHOLDER
    assert check.lambda0_m == lb.LAMBDA0_M and check.n_min == lb.N_MIN
    assert check.median_policy == "population_median_of_log_uniform"
    assert ld.CheckConfig().digest == ld.CheckConfig().digest != ""


# --------------------------------------------------------------------------- decision branches
def test_decision_all_pass_has_no_steps_and_no_g6():
    table = _table()
    _, steps, br = ld.apply_fallbacks(_fake_fn({}), ld.CheckConfig(), y_lo=6000.0,
                                      y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert steps == [] and br["g6"] is False
    assert br["f1_eligible_settings"] == [0.02, 0.05, 0.10]
    assert br["r12_status"] == "evaluable" and br["f2_choice"]["setting"] == 0.10


def test_decision_f1_shortfall_excludes_cell_and_can_write_g6():
    short = _table(f1={0.02: 0.1, 0.05: 0.2, 0.10: 0.3})
    _, steps, br = ld.apply_fallbacks(_fake_fn({}, short), ld.CheckConfig(), y_lo=6000.0,
                                      y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert steps[0]["action"] == "raise_y_hi" and br["g6"] is True
    assert br["r6_status"] == "g6" and br["r7_status"] == "g6"
    assert set(br["g6_dependent_predictions"]) >= {"R6", "R7", "R11b"}
    good = _table(f1={0.02: 0.1, 0.05: 0.9, 0.10: 0.9})
    _, _, br2 = ld.apply_fallbacks(_fake_fn({}, good), ld.CheckConfig(), y_lo=6000.0,
                                   y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert br2["f1_eligible_settings"] == [0.05, 0.10]
    assert br2["f1_excluded_settings"] == [0.02] and br2["g6"] is False


def test_decision_f2_fallback_sequence_is_recorded_exactly():
    tables = {}
    for a, r2 in ((0.10, 0.1), (0.20, 0.2), (0.50, 0.9)):
        tables[("F2", a, 0.05)] = _table(f1=None, f3=0.9, f2=r2, law="F2", setting=a)
    _, steps, br = ld.apply_fallbacks(_fake_fn(tables), ld.CheckConfig(), y_lo=6000.0,
                                      y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert [s["action"] for s in steps] == ["raise_y_hi", "set_F2_a", "set_F2_a"]
    assert [s["to"] for s in steps[1:]] == [0.20, 0.50]
    assert br["f2_choice"] == {"law": "F2", "setting": 0.50}
    assert br["f2_attempted"] == [0.10, 0.20, 0.50]


def test_decision_f2p_fallback_and_g6_when_nothing_passes():
    short = {("F2", a, 0.05): _table(f2=0.1, law="F2", setting=a)
             for a in (0.10, 0.20, 0.50)}
    short.update({("F2p", 0.05, 0.05): _table(f2=0.2, law="F2p", setting=0.05),
                  ("F2p", 0.10, 0.05): _table(f2=0.9, law="F2p", setting=0.10)})
    _, steps, br = ld.apply_fallbacks(_fake_fn(short), ld.CheckConfig(), y_lo=6000.0,
                                      y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert [s["action"] for s in steps] == ["raise_y_hi", "set_F2_a", "set_F2_a",
                                            "switch_F2p_b", "switch_F2p_b"]
    assert br["f2_choice"] == {"law": "F2p", "setting": 0.10} and br["g6"] is False
    all_short = dict(short)
    all_short[("F2p", 0.10, 0.05)] = _table(f2=0.2, law="F2p", setting=0.10)
    _, _, br2 = ld.apply_fallbacks(_fake_fn(all_short), ld.CheckConfig(), y_lo=6000.0,
                                   y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert br2["f2_choice"] is None and br2["g6"] is True
    assert "R11" in br2["g6_dependent_predictions"]


def test_decision_f3_shortfall_raises_spread_then_r12_not_evaluable():
    tables = {("F2", 0.10, 0.05): _table(f2=0.9, f3=0.2),
              ("F2", 0.10, 0.10): _table(f2=0.9, f3=0.9)}
    tables[("F2", 0.10, 0.10)].f3["setting"] = 0.10
    _, steps, br = ld.apply_fallbacks(_fake_fn(tables), ld.CheckConfig(), y_lo=6000.0,
                                      y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert [s["action"] for s in steps][-1] == "raise_F3_spread"
    assert br["f3_choice"] == {"spread": 0.10} and br["r12_status"] == "evaluable"
    tables[("F2", 0.10, 0.10)].f3["r2_run_median"] = 0.2
    _, _, br2 = ld.apply_fallbacks(_fake_fn(tables), ld.CheckConfig(), y_lo=6000.0,
                                   y_hi_initial=320000.0, y_hi_bound=380000.0)
    assert br2["r12_status"] == "not_evaluable" and br2["r12_gate"] is None


# --------------------------------------------------------------------------- frozen JSON
def test_frozen_decision_json_round_trip_and_schema(tmp_path):
    check = ld.load_check_config(CONFIG)
    table = _table()
    yields = {"y_lo": 6000.0, "y_hi": 380000.0, "y_hi_initial": 320000.0,
              "y_hi_bound": 380000.0, "y_hi_bound_source": "hit_fraction_0.95",
              "median_yield": table.median_yield,
              "median_policy": check.median_policy}
    occupancy = {"n_min": 20, "within_target_median_hits": 100.0,
                 "within_exclusion_rate": 0.0, "across_target_median_hits": 90.0,
                 "across_initial_below_nmin_rate": 0.0}
    payload = ld.build_decision_payload(GEOM, GEN, check, table, [], {"g6": False,
                                    "f1_eligible_settings": [0.02, 0.05, 0.10],
                                    "f1_excluded_settings": [], "f2_ok": True,
                                    "f2_choice": {"law": "F2", "setting": 0.10},
                                    "f2_attempted": [0.10], "f3_ok": True,
                                    "f3_choice": {"spread": 0.05}, "f3_attempted": [0.05],
                                    "r12_status": "evaluable", "r12_gate": None,
                                    "law_T_ok": True, "g6_dependent_predictions": [],
                                    "reason": "all pass"},
                                    yields, occupancy, frozen=False, code_sha="deadbeef")
    path = ld.write_decision(tmp_path / "decision.json", payload)
    back = ld.read_decision(path)
    assert back["content_sha256"] == payload["content_sha256"]
    assert back["reported_results"] is False and back["readout_label"] == lb.READOUT_LABEL
    assert back["schema"] == ld.DECISION_SCHEMA and back["geometry"]["synthetic"] is True
    assert "§3.5.3a" in back["decisions_md"] and back["rebenchmark"]["y_lo"] == 6000.0
    bad = dict(back, content_sha256="0" * 64)
    with pytest.raises(ValueError):
        ld.validate_decision(bad)
    with pytest.raises(ValueError):
        ld.validate_decision({**back, "reported_results": True})


# --------------------------------------------------------------------------- determinism / geometry
def test_evaluate_table_is_deterministic_for_same_config_and_seed():
    check = replace(ld.load_check_config(CONFIG), n_events=4)
    events = ld.sample_events(GEOM, GEN, 50000.0, check.n_events, check.seed)
    a = ld.evaluate_table(GEOM, GEN, check, 50000.0, seed=check.seed, events=events)
    b = ld.evaluate_table(GEOM, GEN, check, 50000.0, seed=check.seed, events=events)
    assert [dict(r) for r in a.rows()] == [dict(r) for r in b.rows()]
    assert a.rows()[0]["i_run_median"] == pytest.approx(b.rows()[0]["i_run_median"])


def test_geometry_loaded_from_npz_path_without_lucid(tmp_path):
    npz = tmp_path / "wcte_geometry.npz"
    np.savez(npz, positions_mm=GEOM.positions_m * 1000.0, directions=GEOM.directions,
             surfaces=GEOM.surfaces, pmt_id=GEOM.pmt_id, mpmt_id=GEOM.mpmt_id,
             pmt_id_in_mpmt=GEOM.pmt_id_in_mpmt, radius=GEOM.radius_m,
             height=GEOM.height_m, sensor_radius=GEOM.sensor_radius_m)
    pointer = tmp_path / "geom_config_test.json"
    pointer.write_text(json.dumps({"synthetic": True,
                                   "geometry_definitions": {"npz_file_path": npz.name}}))
    loaded = lb.load_wcte_geometry(pointer)
    assert (loaded.n_pmt, loaded.n_modules) == (1995, 105) and loaded.synthetic
    source = Path(ld.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                   else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        assert all(m != "lucid" and not m.startswith("lucid.") for m in modules)
    assert "noise_module_lucid" not in source.split("from . import")[0]


def test_occupancy_n_min_check_at_low_yield():
    check = replace(ld.load_check_config(CONFIG), occupancy_events=4)
    result = ld.occupancy_n_min(GEOM, GEN, check, 6000.0, seed=check.seed)
    assert result["n_min"] == 20 and result["within_target_median_hits"] > 20
    assert result["within_exclusion_rate"] == 0.0
    assert result["across_target_median_hits"] > 20


def test_f2_branch_records_the_joint_f1_ceiling():
    """v4 §3.5.3a profiles the other factors; the chosen F2 law must carry f1_joint."""
    check = ld.load_check_config(CONFIG)
    events = ld.sample_events(GEOM, GEN, 12000.0, 6, 7)
    table = ld.evaluate_table(
        GEOM, GEN, check, 12000.0, f2_law="F2", f2_setting=0.5, events=events)
    assert table.f2_law == "F2"
    assert "f1_joint" in table.f2
    rows = table.f2["f1_joint"]
    assert len(rows) == 105
    assert set(rows[0]) == {"module", "r2_run"}
    assert all(0.0 <= row["r2_run"] <= 1.0 for row in rows)
