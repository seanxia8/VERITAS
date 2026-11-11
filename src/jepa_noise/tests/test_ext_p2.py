"""Paper 3 P2 extension tests (§8.3).

The exactness tests compare the §2.3 transforms against the generator itself: a
second synthetic run is generated with shared keyed draws and the factor built
into the readout (a ``dataclasses.replace``d ``GeneratorConfig`` for the
threshold, a zero-gain ``DetectorState`` for the dead channels).  No test
compares a transform with itself.
"""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import ext_p2 as p2x
from jepa_noise import lucid_bombs as lb

ROOT = Path(__file__).resolve().parents[3]
GEOM = lb.from_synthetic({"module_ids": tuple(range(4)),
                          "surface_counts": {"top": 16, "barrel": 44, "bottom": 16}})
BASE = dataclasses.replace(lb.GeneratorConfig(), y_lo=8000.0, y_hi=25000.0)
RUN_ID = "lucid/synthetic/p2/0000"
EVENTS = tuple(range(8))

pytestmark = pytest.mark.skipif(
    not lb.pmt_time_rule_available(),
    reason="pending the author-named PMT SHA (NHits time rule); no synthetic digits")


def _hits(cfg, state, *, events=EVENTS, seed=7):
    readout = lb.make_readout(GEOM, cfg)
    out = []
    for event_id in events:
        draws = lb.draw_event(GEOM, cfg, RUN_ID, int(event_id), seed=seed)
        out.append(readout(state, draws))
    return out


def _base_state():
    return lb.nominal_state(GEOM, RUN_ID)


def _threshold_run(tau, cal):
    cfg = dataclasses.replace(BASE, threshold_pe=float(tau),
                              walk_calibration_pe=float(cal))
    return _hits(BASE, _base_state()), _hits(cfg, _base_state()), cfg


def _dead_run(fraction):
    condition = p2x.DriftCondition("K", "dead", fraction=float(fraction),
                                   unit="module", namespace=p2x.MODULE_NAMESPACE)
    dead = condition.dead_pmts(RUN_ID, GEOM)
    state = dataclasses.replace(_base_state(),
                                gains=np.where(dead, 0.0, _base_state().gains))
    return _hits(BASE, _base_state()), _hits(BASE, state), condition


def _compare(base_hits, factor_hits, condition, cfg, *, tolerance=1e-9):
    """Compare the transform of the base run against the generated factor run."""
    rule = lb.pmt_time_rule()
    prompt = tuple(float(v) for v in rule[4])
    compared = flagged = 0
    failures = []
    for index, (base, factor) in enumerate(zip(base_hits, factor_hits)):
        result = p2x.transform_event(condition, RUN_ID, base.pmt_id, base.charge_pe,
                                     base.time_ns, GEOM, cfg, rule)
        if base.trigger_ns is None or factor.trigger_ns is None:
            flagged += 1
            continue
        absolute = np.asarray(factor.absolute_time_ns, float)
        lower = float(base.trigger_ns) + prompt[0]
        upper = float(base.trigger_ns) + prompt[1]
        # a factor-run digit outside the OLD prompt window was never stored in the
        # base store: that event is a window-edge case (§2.3 item 3)
        # The exactness clause is about hits that would enter the new prompt window
        # from beyond the OLD edge (t in (60, 60 + trig'] ns): those were never
        # stored.  A trigger that merely moves without such a hit is still exact
        # (the transform re-derives the shifted trigger on the stored digits).
        unobservable = bool(np.any((absolute < lower) | (absolute > upper)))
        if unobservable:
            flagged += 1
            continue
        if result is None:
            failures.append((index, "transform dropped a triggered event"))
            continue
        compared += 1
        times = np.asarray(result["time_ns"], float)
        factor_times = np.asarray(factor.time_ns, float)
        same_hits = bool(np.array_equal(result["pmt_id"], np.asarray(factor.pmt_id))
                         and np.array_equal(np.asarray(result["charge_pe"], float),
                                            np.asarray(factor.charge_pe, float)))
        same_time = bool(times.shape == factor_times.shape
                         and np.max(np.abs(times - factor_times)) <= tolerance
                         if times.size else times.shape == factor_times.shape)
        trigger = float(result["trig_rel"])
        factor_trigger = float(factor.trigger_ns) - float(base.trigger_ns)
        if not (same_hits and same_time and abs(trigger - factor_trigger) <= tolerance):
            failures.append((index, {"same_hits": same_hits, "same_time": same_time,
                                     "trigger": trigger,
                                     "factor_trigger": factor_trigger}))
    return {"compared": compared, "flagged": flagged, "failures": failures}


# ---------------------------------------------------------- generator exactness
def test_threshold_transform_matches_the_generator():
    base, factor, cfg = _threshold_run(1.0, 0.25)
    condition = p2x.DriftCondition("T1.0w", "threshold", tau=1.0, cal=0.25)
    outcome = _compare(base, factor, condition, BASE)
    assert outcome["compared"] >= 1, "no event was comparable"
    assert not outcome["failures"], outcome["failures"]


def test_pure_hit_loss_transform_matches_the_generator():
    base, factor, _cfg = _threshold_run(1.0, 1.0)
    condition = p2x.DriftCondition("T1.0", "threshold", tau=1.0, cal=1.0)
    outcome = _compare(base, factor, condition, BASE)
    assert outcome["compared"] >= 1, "no event was compared"
    assert not outcome["failures"], outcome["failures"]


def test_dead_channel_transform_matches_the_generator():
    base, factor, condition = _dead_run(0.5)
    outcome = _compare(base, factor, condition, BASE)
    assert outcome["compared"] >= 1, "no event was comparable"
    assert not outcome["failures"], outcome["failures"]


def test_r1_maps_t1_0w_to_t1_0():
    base, factor_w, _cfg_w = _threshold_run(1.0, 0.25)
    _base2, factor_0, _cfg_0 = _threshold_run(1.0, 1.0)
    condition = p2x.DriftCondition(p2x.R1_NAME, "r1", tau=1.0, cal=0.25)
    rule = lb.pmt_time_rule()
    compared = flagged = 0
    failures = []
    for index, (base_i, w_i, zero_i) in enumerate(zip(base, factor_w, factor_0)):
        w = p2x.transform_event(p2x.DriftCondition("T1.0w", "threshold", 1.0, 0.25),
                                RUN_ID, base_i.pmt_id, base_i.charge_pe,
                                base_i.time_ns, GEOM, BASE, rule)
        if (w is None or w_i.trigger_ns is None
                or not np.array_equal(w["pmt_id"], np.asarray(w_i.pmt_id))):
            flagged += 1
            continue
        if zero_i.trigger_ns is None:
            flagged += 1
            continue
        if not set(np.asarray(zero_i.pmt_id).tolist()).issubset(
                set(np.asarray(w_i.pmt_id).tolist())):
            flagged += 1
            continue
        r1 = p2x.transform_event(condition, RUN_ID, base_i.pmt_id, base_i.charge_pe,
                                 base_i.time_ns, GEOM, BASE, rule)
        if r1 is None:
            failures.append((index, "R1 dropped a triggered event"))
            continue
        compared += 1
        same = bool(np.array_equal(r1["pmt_id"], np.asarray(zero_i.pmt_id))
                    and np.array_equal(np.asarray(r1["charge_pe"], float),
                                       np.asarray(zero_i.charge_pe, float)))
        difference = (float(np.max(np.abs(np.asarray(r1["time_ns"], float)
                                         - np.asarray(zero_i.time_ns, float))))
                      if same else None)
        if not (same and difference is not None and difference <= 1e-9):
            failures.append((index, {"same_hits": same, "time_difference": difference}))
    assert compared >= 1
    assert not failures, failures


# ---------------------------------------------------------------- identity/nesting
def _write_run(tmp_path):
    cfg = BASE
    state = _base_state()
    readout = lb.make_readout(GEOM, cfg)
    hits = [readout(state, lb.draw_event(GEOM, cfg, RUN_ID, int(event_id), seed=7))
            for event_id in EVENTS]
    store = lb.sparse_from_readouts(np.asarray(EVENTS, np.int64), hits)
    path = tmp_path / "hits.npz"
    lb.write_sparse_hits(path, store)
    return path, store


def test_identity_transform_reproduces_the_stored_hits(tmp_path):
    path, _store = _write_run(tmp_path)
    store = lb.read_sparse_hits(path)
    rule = lb.pmt_time_rule()
    condition = p2x.identity_condition()
    offsets = np.asarray(store.offsets, np.int64)
    compared = 0
    for index, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
        span = slice(int(offsets[index]), int(offsets[index + 1]))
        pmt = np.asarray(store.pmt_id[span]).copy()
        charge = np.asarray(store.charge_pe[span]).copy()
        time = np.asarray(store.time_ns[span]).copy()
        before = (pmt.copy(), charge.copy(), time.copy())
        result = p2x.transform_event(condition, RUN_ID, pmt, charge, time, GEOM, BASE, rule)
        assert result is not None
        assert np.array_equal(result["pmt_id"], before[0])
        assert np.array_equal(np.asarray(result["charge_pe"], float),
                              np.asarray(before[1], float))
        assert np.max(np.abs(np.asarray(result["time_ns"], float) - before[2])) <= 1e-9
        assert result["trig_rel"] == 0.0
        # the transform never mutates its input
        assert np.array_equal(pmt, before[0])
        assert np.array_equal(charge, before[1])
        assert np.array_equal(time, before[2])
        compared += 1
    assert compared == len(EVENTS)


def test_dead_sets_are_nested_across_k_levels():
    conditions = p2x.build_drift_conditions()
    for run in ("run/a", "run/b"):
        k5 = conditions["K5"].dead_pmts(run, GEOM)
        k15 = conditions["K15"].dead_pmts(run, GEOM)
        k30 = conditions["K30"].dead_pmts(run, GEOM)
        assert not np.any(k5 & ~k15)
        assert not np.any(k15 & ~k30)
        assert int(k5.sum()) <= int(k15.sum()) <= int(k30.sum())
        assert int(np.ceil(0.05 * GEOM.n_modules)) <= int(k5.sum())
    p10 = conditions["P10"].dead_pmts("run/a", GEOM)
    assert int(p10.sum()) == int(np.ceil(0.1 * GEOM.n_pmt))


def test_transform_bookkeeping_counts_window_edges():
    hits = _hits(BASE, _base_state())
    for name, expect_edges in (("T1.0w", True), ("T1.0", None)):
        condition = p2x.build_drift_conditions()[name]
        records = []
        for base in hits:
            records.append(p2x.transform_event(condition, RUN_ID, base.pmt_id,
                                               base.charge_pe, base.time_ns, GEOM, BASE,
                                               lb.pmt_time_rule()))
        kept = [r for r in records if r is not None]
        n_edge = sum(1 for r in kept if r["trig_rel"] > 0)
        assert kept, name
        assert n_edge == sum(1 for r in kept if r["window_edge"])
        if expect_edges:
            # the walk moves the trigger later for essentially every kept event
            assert n_edge >= len(kept) - 1
        assert all(np.isfinite(r["trig_final"]) for r in kept)


# --------------------------------------------------------------------- R2 / P2-e
def test_r2_design_uses_dead_fraction_and_hit_count():
    condition = p2x.build_drift_conditions()["K15"]
    rows = {"run_ids": np.asarray(["r1", "r1", "r2"], dtype=object),
            "event_ids": np.asarray([0, 1, 0]),
            "n_hit": np.asarray([10.0, 20.0, 5.0])}
    dead = {run: np.zeros(GEOM.n_modules) for run in ("r1", "r2")}
    dead["r1"][:2] = 0.5
    design = p2x.r2_design(rows, condition, dead, 10.0, GEOM)
    assert design.shape == (3, GEOM.n_modules)
    assert design[0, 0] == pytest.approx(0.5 * 1.0)
    assert design[1, 0] == pytest.approx(0.5 * 2.0)
    assert design[2, 0] == 0.0
    threshold_design = p2x.r2_design(rows, p2x.build_drift_conditions()["T1.0"],
                                     None, 10.0, GEOM)
    assert threshold_design.shape == (3, 2)
    assert threshold_design[0, 0] == pytest.approx(1.0)
    assert np.all(threshold_design[:, 1] == 1.0)


def test_raw_map_identity_holds():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((80, 4))
    y = x @ rng.standard_normal((4, 6)) + rng.standard_normal(6)
    from jepa_noise import lucid_arms as la
    train = la.ProbeDataset(kind="delta_z", form="occupancy", x=x, y=y,
                            run_ids=np.repeat(np.arange(5), 16).astype(str),
                            target_names=tuple(f"z{i}" for i in range(6)), meta={})
    fitted = la.fit_probe(train)
    A = p2x.er.raw_map(fitted)
    prediction = la.probe_predictions(fitted, train)
    zero = la.probe_predictions(fitted, dataclasses.replace(train, x=np.zeros_like(x)))
    assert np.max(np.abs(x @ A.T - (prediction - zero))) <= 1e-8


def test_p2e_statistic_and_class_rule():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    y = x + 0.01
    block = p2x.permutation_spearman(x, y, level="test", permutations=199)
    assert block["rho_s"] == pytest.approx(1.0)
    assert block["p"] <= 0.02
    supported = p2x.p2e_class({level: {"rho_s": 0.7, "p": 0.01, "n": 10}
                               for level in ("K5", "K15", "K30")})
    assert supported["verdict"] == "supported"
    refuted = p2x.p2e_class({level: {"rho_s": 0.05, "p": 0.4, "n": 10}
                             for level in ("K5", "K15", "K30")})
    assert refuted["verdict"] == "refuted"
    mixed = p2x.p2e_class({"K5": {"rho_s": 0.7, "p": 0.01, "n": 10},
                           "K15": {"rho_s": 0.2, "p": 0.4, "n": 10},
                           "K30": {"rho_s": 0.3, "p": 0.2, "n": 10}})
    assert mixed["verdict"] == "mixed"
    tiny = p2x.p2e_class({"K5": {"rho_s": 0.7, "p": 0.01, "n": 2},
                          "K15": {"rho_s": 0.2, "p": 0.4, "n": 10},
                          "K30": {"rho_s": 0.3, "p": 0.2, "n": 10}})
    assert tiny["verdict"] == "not evaluable"


# --------------------------------------------------------------------- X3-P
def test_erasure_removes_the_rank_r_whitened_subspace_and_controls_have_rank_r():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((200, 12))
    z = x @ rng.standard_normal((12, 2)) + 0.1 * rng.standard_normal((200, 2))
    eraser = p2x.er.leace_fit(x, z, rank=2)
    erased = p2x.er.leace_erase(eraser, x)
    U = eraser["U_r"]; W = eraser["W"]; mu = eraser["mu_x"]
    projection = U @ U.T
    lhs = (W @ (x - erased).T)
    rhs = projection @ (W @ (x - mu).T)
    assert np.max(np.abs(lhs - rhs)) <= 1e-10
    assert eraser["rank"] == 2
    # idempotent and the control bases are orthonormal of rank r
    assert np.allclose(p2x.er.leace_erase(eraser, erased), erased, atol=1e-10)
    for index in range(3):
        basis = p2x.er.random_control_basis(rng, 12, 2)
        assert np.linalg.matrix_rank(basis) == 2
        assert np.max(np.abs(basis.T @ basis - np.eye(2))) <= 1e-10


# --------------------------------------------------------------------- gate
def _fake_repo(tmp_path, *, record=True, sha=None, verdict="PASS"):
    root = tmp_path / "repo"
    (root / "docs/jepa").mkdir(parents=True)
    runbook = ROOT / p2x.RUNBOOK_REL
    (root / p2x.RUNBOOK_REL).write_bytes(runbook.read_bytes())
    (root / p2x.REVIEW_REL).write_text("PASS\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "fixture"], check=True)
    if record:
        impl = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True).strip()
        payload = {"runbook_sha256": sha or p2x.RUNBOOK_SHA256,
                   "impl_commit": impl, "review_path": p2x.REVIEW_REL,
                   "review_verdict": verdict, "lead": "lead", "date": "2026-11-09"}
        (root / p2x.ACCEPTANCE_REL).write_text(json.dumps(payload))
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "-m", "acceptance"], check=True)
    return root


def test_gate_pending_and_denied(tmp_path):
    pending = _fake_repo(tmp_path / "pending", record=False)
    checks = p2x.assert_authorised("dry-run", root=pending, dry_run=True)
    assert checks["status"] == "gate_pending_acceptance"
    assert not checks["authorised"]
    with pytest.raises(SystemExit):
        p2x.assert_authorised("p2", root=pending)
    granted = _fake_repo(tmp_path / "granted")
    checks = p2x.assert_authorised("p2", root=granted)
    assert checks["authorised"], checks["problems"]
    with pytest.raises(ValueError):
        p2x.assert_authorised("train", root=granted, dry_run=True)


def test_gate_denies_a_wrong_runbook_sha_or_verdict(tmp_path):
    wrong = _fake_repo(tmp_path / "wrong", sha="0" * 64)
    check = p2x._gate_checks("p2", root=wrong)
    assert not check["authorised"]
    blocked = _fake_repo(tmp_path / "blocked", verdict="BLOCK")
    check = p2x._gate_checks("p2", root=blocked)
    assert not check["authorised"]
    assert any("review_verdict" in problem for problem in check["problems"])


def test_gate_denies_an_untracked_p_file(tmp_path):
    root = _fake_repo(tmp_path / "untracked")
    (root / "src/jepa_noise").mkdir(parents=True, exist_ok=True)
    (root / "src/jepa_noise/ext_p2.py").write_text("# untracked\n")
    check = p2x._gate_checks("p2", root=root)
    assert not check["authorised"]
    assert check["checks"]["c"]["untracked"]


def test_closure_flags_a_module_outside_p():
    root = _fake_repo(_tmp())
    # a module outside <P> resolved under the fake root is a violation
    import types
    (root / "src/jepa_noise").mkdir(parents=True, exist_ok=True)
    module = types.ModuleType("outside_p_module")
    module.__file__ = str(root / "src/jepa_noise/outside_p_module.py")
    (root / "src/jepa_noise/outside_p_module.py").write_text("x = 1\n")
    sys.modules["outside_p_module"] = module
    try:
        closure = p2x.import_closure(root)
    finally:
        sys.modules.pop("outside_p_module", None)
    assert not closure["pass"]
    assert any(row["module"] == "outside_p_module" for row in closure["violations"])


def _tmp():
    import tempfile
    return Path(tempfile.mkdtemp())


# ------------------------------------------------------- bookkeeping / forbidden
def test_projections_s1_entries_and_budget(tmp_path, monkeypatch):
    projection = p2x.gpu_set_projection()
    assert projection["total_sets"] == 249
    assert projection["projected_l40s_hours"] <= p2x.GPU_CAP_H
    assert not projection["with_optional_pass"]
    consistent_ledger = {"v4_caps": {
        "E2_lucid_l40s_deepthought2": {"charged_gpu_hours": 133.6708},
        "recent": {"base": {"l40s_charged_after": 130.0},
                   "p2": {"part_a": {"gpu_hours": 3.6708}}}}}
    check = p2x.p2_s1_check(None, ledger=consistent_ledger)
    assert check["ledger_consistent"] and check["pass"]
    inconsistent = {"v4_caps": {
        "E2_lucid_l40s_deepthought2": {"charged_gpu_hours": 133.0},
        "recent": {"base": {"l40s_charged_after": 130.0},
                   "p2": {"part_a": {"gpu_hours": 3.6708}}}}}
    assert not p2x.p2_s1_check(None, ledger=inconsistent)["pass"]
    assert p2x.entries_preflight(free_entries=40133)["pass"]
    assert not p2x.entries_preflight(free_entries=30500)["pass"]
    monkeypatch.setattr(p2x, "JOBS_RECORD", tmp_path / "jobs.jsonl")
    p2x.record_job("gpu-test", "gpu", 3600.0)
    assert p2x.job_hours()["gpu"] == pytest.approx(1.0)
    p2x.assert_budget("gpu", 0.2)
    with pytest.raises(SystemExit):
        p2x.assert_budget("gpu", 0.5)


def test_write_json_never_overwrites():
    path = _tmp() / "one.json"
    p2x.write_json(path, {"a": 1})
    with pytest.raises(RuntimeError):
        p2x.write_json(path, {"a": 2}, overwrite=False)


def test_forbidden_generation_and_training_paths_are_never_reached(tmp_path, monkeypatch):
    path, _store = _write_run(tmp_path)
    reached = []

    def boom(*args, **kwargs):
        reached.append(1)
        raise AssertionError("forbidden generator/training path reached")

    for module, name in ((lb, "draw_event"), (lb, "generate_run"),
                         (lb, "write_generated_run"), (lb, "generate_events"),
                         (p2x.lr, "ensure_dataset"), (p2x.lr, "ensure_dark_bank")):
        monkeypatch.setattr(module, name, boom)
    import torch
    monkeypatch.setattr(torch.optim.Optimizer, "step", boom)
    # the transform and its bookkeeping only ever read the stored hits
    store = lb.read_sparse_hits(path)
    condition = p2x.build_drift_conditions()["T1.0w"]
    rule = lb.pmt_time_rule()
    offsets = np.asarray(store.offsets, np.int64)
    for index in range(len(store.event_ids)):
        span = slice(int(offsets[index]), int(offsets[index + 1]))
        p2x.transform_event(condition, RUN_ID, store.pmt_id[span], store.charge_pe[span],
                            store.time_ns[span], GEOM, BASE, rule)
    assert reached == []
    assert "train" not in p2x.MODES and "generate" not in p2x.MODES
    assert set(p2x.MODES) == {"dry-run", "preflight", "reproduce", "new-models",
                              "conditions", "p2", "x3p", "report"}

# ------------------------------------------------- end-to-end cell / X3-P helpers
def _toy_rows(runs, *, offset=0.0, seed=0, n_events=60):
    rng = np.random.default_rng(seed)
    events = rng.standard_normal((n_events, 8)) + offset
    return {"events": events, "run_ids": np.repeat(np.asarray(runs), n_events // len(runs)),
            "event_ids": np.arange(n_events),
            "vertex": rng.standard_normal((n_events, 3)),
            "log_yield": rng.standard_normal(n_events),
            "n_hit": rng.integers(0, 30, n_events).astype(float),
            "particle": np.full(n_events, "none"),
            "contained": np.zeros(n_events, bool),
            "deposited_MeV": np.full(n_events, np.nan)}


def test_cell_reading_dead_condition_uses_both_run_roles():
    """R2's dead design must cover the disjoint probe-test runs (review B1)."""
    n_train = _toy_rows([f"tr{i}" for i in range(5)], seed=1)
    n_test = _toy_rows([f"te{i}" for i in range(5)], seed=2)
    c_train = dict(n_train, events=n_train["events"] + 0.3)
    c_test = dict(n_test, events=n_test["events"] + 0.3)
    condition = p2x.build_drift_conditions()["K5"]
    data = {"condition": condition, "n_train": n_train, "n_test": n_test,
            "c_train": c_train, "c_test": c_test,
            "hbar": float(np.mean(n_train["n_hit"])), "geometry": GEOM}
    frozen_vertex = p2x.er.fit_vertex_probe(n_train)
    frozen_yield = p2x.er.fit_log_yield_probe(n_train)
    reading = p2x.cell_reading(data, frozen_vertex=frozen_vertex,
                               frozen_yield=frozen_yield)
    assert reading["condition"] == "K5"
    assert reading["delta_c_m"] is not None
    assert reading["latent"]["status"] == "evaluated", reading["latent"]
    assert reading["latent"]["r2_eq"] is not None
    assert reading["refit"]["information_destroying"] is not None
    # the exact-subset re-run must not crash either
    exact = dict(data)
    for key in ("c_train", "c_test"):
        exact[key] = p2x._subset_rows(exact[key], np.ones(len(exact[key]["run_ids"]), bool))
    p2x.cell_reading(exact, frozen_vertex=frozen_vertex, frozen_yield=frozen_yield)


def test_x3p_dominance_evaluates_with_vertex_blocks():
    """Review B3: vertex blocks must carry mean_error_m for the Part A X3 helper."""
    rng = np.random.default_rng(9)
    n = 60
    rows = {"events": rng.standard_normal((n, 8)), "run_ids": np.repeat(np.arange(5), 12),
            "event_ids": np.arange(n), "vertex": rng.standard_normal((n, 3)),
            "log_yield": rng.standard_normal(n), "n_hit": np.full(n, 5.0)}
    block = {"N": p2x._metric_block(rows, rows, "vertex"),
             "D": p2x._metric_block(rows, rows, "vertex")}
    assert block["N"]["mean_error_m"] is not None
    assert p2x.er._metric_rows({"vertex": {"N": block["N"]}}, "vertex", "N") is not None
    scalar = p2x._metric_block(rows, rows, "log_yield")
    calibrated = {"vertex": {"D": block["D"]}, "energy": {"D": scalar},
                  "log_yield": {"D": scalar}}
    variant = {"vertex": {"N": block["N"], "D": block["D"]},
               "energy": {"N": scalar, "D": scalar},
               "log_yield": {"N": scalar, "D": scalar}}
    dominance = p2x.er._x3_dominance(calibrated, variant)
    assert dominance["status"] == "evaluated"


def test_x0b_target_table_covers_every_reproduction_model():
    assert set(p2x.X0B_MODELS) <= set(p2x.X0B_EXPECTED)
    assert p2x.X0B_EXPECTED["fix9000"] == pytest.approx(0.219258)
    for label in p2x.NEW_MODELS:
        assert (p2x.X0B_EXPECTED[label]
                == pytest.approx(p2x.NEW_MODEL_SPECS[label]["full_input_vertex_m"]))


def test_mirror_dataset_round_trip(tmp_path, monkeypatch):
    """The GPU-mode data mirror copies and verifies a written synthetic run."""
    source = tmp_path / "data"
    run = source / "runs" / "run0"
    run.mkdir(parents=True)
    lb.write_sparse_hits(run / "hits.npz", lb.SparseHitStore(
        np.asarray(EVENTS, np.int64), np.zeros(len(EVENTS) + 1, np.int64),
        np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.float32)))
    np.savez(run / "events.npz", event_ids=np.asarray(EVENTS))
    (run / "manifest.json").write_text("{}")
    lb.write_sha256sums(run)
    (source / "manifest.json").write_text("{}")
    monkeypatch.setenv("JEPA_P2_MIRROR", str(tmp_path / "mirror"))
    dataset = p2x.lr.Dataset(
        key="toy__F1__0.05__s7", directory=source, cohort="toy", law="F1",
        setting=0.05, setting_kind=None, seed=7, n_events=len(EVENTS),
        run_length=len(EVENTS), runs=tuple([{"run_id": RUN_ID, "run_dir": "run0",
                                             "event_id_start": 0,
                                             "n_events": len(EVENTS)}]),
        manifest={})
    with pytest.raises(RuntimeError):
        p2x.mirror_dataset(dataset, log=lambda *a, **k: None)
    mirrored = p2x.mirror_dataset(dataset, base=str(tmp_path / "mirror" / "p2_data_mirror"),
                                  stage=True, log=lambda *a, **k: None)
    assert str(mirrored.directory).startswith(str(tmp_path / "mirror"))
    assert (mirrored.directory / "runs" / "run0" / "hits.npz").is_file()
    assert mirrored is not dataset
    again = p2x.mirror_dataset(dataset, log=lambda *a, **k: None)
    assert again.directory == mirrored.directory
    # corrupting a mirrored file must make the verify-only read fail closed
    (mirrored.directory / "runs" / "run0" / "hits.npz").write_bytes(b"corrupt")
    with pytest.raises((RuntimeError, ValueError)):
        p2x.mirror_dataset(dataset, log=lambda *a, **k: None)

def test_later_stages_reuse_the_verified_inventory(tmp_path, monkeypatch):
    """GPU stages read the X0-a inventory rather than re-hashing checkpoints."""
    monkeypatch.setattr(p2x, "EXT_ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        p2x.load_verified_inventory()
    (tmp_path / "x0").mkdir()
    p2x.write_json(tmp_path / "x0" / "inventory.json", {"hash_files": False})
    with pytest.raises(RuntimeError):
        p2x.load_verified_inventory()
    p2x.write_json(tmp_path / "x0" / "inventory.json", {"hash_files": True,
                                                       "rows": {}})
    assert p2x.load_verified_inventory()["hash_files"] is True

def test_p10_dead_pmt_transform_matches_the_generator():
    """The §2.3 dead-PMT condition (10 % of the 1995 PMTs) is generator-exact."""
    condition = p2x.build_drift_conditions()["P10"]
    dead = condition.dead_pmts(RUN_ID, GEOM)
    state = dataclasses.replace(_base_state(),
                                gains=np.where(dead, 0.0, _base_state().gains))
    base = _hits(BASE, _base_state())
    factor = _hits(BASE, state)
    outcome = _compare(base, factor, condition, BASE)
    assert outcome["compared"] >= 1, "no event was comparable"
    assert not outcome["failures"], outcome["failures"]


def test_cell_reading_r1_path_evaluates():
    """The §5 P2-c R1 branch runs on the T1.0w cell (cell_reading, not transform-only)."""
    n_train = _toy_rows([f"tr{i}" for i in range(5)], seed=11)
    n_test = _toy_rows([f"te{i}" for i in range(5)], seed=12)
    c_train = dict(n_train, events=n_train["events"] + 0.3)
    c_test = dict(n_test, events=n_test["events"] + 0.3)
    condition = p2x.build_drift_conditions()["T1.0w"]
    data = {"condition": condition, "n_train": n_train, "n_test": n_test,
            "c_train": c_train, "c_test": c_test,
            "r1_test": dict(c_test, events=c_test["events"] - 0.1),
            "t10_test": dict(c_test, events=c_test["events"] - 0.2),
            "hbar": float(np.mean(n_train["n_hit"])), "geometry": GEOM}
    frozen_vertex = p2x.er.fit_vertex_probe(n_train)
    frozen_yield = p2x.er.fit_log_yield_probe(n_train)
    reading = p2x.cell_reading(data, frozen_vertex=frozen_vertex,
                               frozen_yield=frozen_yield)
    assert reading["r1"]["status"] == "evaluated", reading["r1"]
    assert reading["r1"]["walk_share"] is not None
    assert reading["r1"]["rho_R1"] is not None

def test_gpu_stages_prefer_an_explicit_mirror_base(tmp_path, monkeypatch):
    """JEPA_P2_MIRROR (node-local RAM) wins over the plan-recorded base."""
    monkeypatch.setattr(p2x, "EXT_ROOT", tmp_path)
    (tmp_path / "x0").mkdir()
    p2x.write_json(tmp_path / "x0" / "plan.json",
                   {"data_mirror": {"base": "/work/recorded/p2_data_mirror"}})
    monkeypatch.delenv("JEPA_P2_MIRROR", raising=False)
    assert p2x.require_mirror_base() == "/work/recorded/p2_data_mirror"
    monkeypatch.setenv("JEPA_P2_MIRROR", str(tmp_path / "shm"))
    assert p2x.require_mirror_base() == str(tmp_path / "shm" / "p2_data_mirror")
    audit = p2x.mirror_audit()
    assert audit["used_base"] == str(tmp_path / "shm" / "p2_data_mirror")
    assert audit["plan_base"] == "/work/recorded/p2_data_mirror"
