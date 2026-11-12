# SPDX-License-Identifier: MIT
"""Round-2 budget enforcement: ledger, S1 cap check, HK fallback record, seed-9002
derivation, S5/S7, X0 clearance (runbook §8, §9); three-seed readings (§7)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from jepa_noise import ext_r2 as x


def _smoke(root, det, arm, sps, micro=8, peak=10.0):
    d = root / "smokes" / f"{x.run_name(det, arm, 9000, True)}__mb{micro}"
    d.mkdir(parents=True)
    (d / "run_record.json").write_text(json.dumps(
        {"micro_batch": micro, "seconds_per_step": sps, "peak_memory_gib": peak}))


def test_smoke_projection_and_micro_batch(tmp_path):
    _smoke(tmp_path, "sk", "mae_mse", 0.9, micro=8, peak=30.0)   # > 45 % of the card
    _smoke(tmp_path, "sk", "mae_mse", 1.0, micro=4, peak=12.0)
    p = x.smoke_projection_h(tmp_path, "sk", "mae_mse")
    assert p["micro_batch"] == 4 and abs(p["projected_h"] - 10000 / 3600) < 1e-9
    assert x.smoke_projection_h(tmp_path, "sk", "sigreg_jepa") is None
    assert x.run_projection_h(tmp_path, "sk", "sigreg_jepa") == x.DESIGN_MAIN_RUN_H["sk"]


def test_ledger_books_and_s1_refuses(tmp_path):
    for arm in x.TRAINED_ARMS:
        _smoke(tmp_path, "sk", arm, 1.0)
    ok = x.gpu_precheck(tmp_path, "sk", "mae_mse", 9000, 2.8)
    assert ok["total_h"] <= x.GPU_CAP_H
    x.ledger_add(tmp_path, {"kind": "train", "detector": "sk", "arm": "mae_mse", "seed": 9000,
                            "gpu_h": 2.8, "complete": True})
    ledger = x.ledger_load(tmp_path)
    assert x.booked_gpu_h(ledger) == 2.8 and x.run_complete(ledger, "sk", "mae_mse", 9000)
    x.ledger_add(tmp_path, {"kind": "train", "detector": "x", "arm": "a", "seed": 0,
                            "gpu_h": 40.0, "complete": True})
    with pytest.raises(x.StopError) as err:
        x.gpu_precheck(tmp_path, "sk", "mae_mse", 9001, 2.8)
    assert err.value.stop == "S1"


def test_hk_fallback_recorded_once_and_cuts_runs(tmp_path):
    for det in ("sk", "juno", "hk"):
        for arm in x.TRAINED_ARMS:
            _smoke(tmp_path, det, arm, 1.0 if det != "hk" else 2.0)
    for det in ("sk", "juno"):
        for arm, seed in x.planned_runs(x.ledger_load(tmp_path), det):
            x.ledger_add(tmp_path, {"kind": "train", "detector": det, "arm": arm, "seed": seed,
                                    "gpu_h": 3.0, "complete": True})
        x.ledger_add(tmp_path, {"kind": "eval", "detector": det, "arm": "all", "seed": 0,
                                "gpu_h": x.DESIGN_EVAL_H})
    d = x.hk_fallback_decision(tmp_path)
    # booked 2 x (7 x 3 + 5/3) = 45.33 h -> remaining 14.67 h; HK main run = max(2.78 x
    # 19746/10000, 5.56) = 5.56 h -> six runs do not fit, three (16.7 h + eval) do not either
    assert d["step"] == 3 and d["stop"] == "S1"
    assert x.ledger_load(tmp_path)["hk_fallback"] == d
    assert x.planned_runs(x.ledger_load(tmp_path), "hk") == []


def test_storage_and_cpu_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(x, "ceph_free_entries", lambda root=None: 100_000)
    ok = x.storage_check(1e3, work_root=tmp_path, ceph_root=tmp_path)
    assert ok["pass"] and ok["stop"] is None
    big = x.storage_check(1e18, work_root=tmp_path, ceph_root=tmp_path)
    assert big["stop"] == "S5"
    monkeypatch.setattr(x, "ceph_free_entries", lambda root=None: 30_100)
    assert x.storage_check(1e3, entries_needed=500, work_root=tmp_path,
                           ceph_root=tmp_path)["stop"] == "S5"
    assert x.cpu_check(tmp_path, 10.0)["pass"]
    x.ledger_add(tmp_path, {"kind": "generation", "wall_h": 25.0}, cpu=True)
    c = x.cpu_check(tmp_path, 10.0)
    assert c["stop"] == "S7" and abs(c["remaining_wall_h"] - (48 - 18.43 - 25)) < 1e-9


def test_x0_clearance():
    good = {k: {"stop": None} for k in ("geometry", "trigger", "gate_v", "storage", "cpu")}
    good["yield_range"] = {"stop": None, "y_lo": 1e4, "y_hi": 1e6}
    assert x.x0_clearance(good) == {"y_lo": 1e4, "y_hi": 1e6}
    # addendum: Gate V is informational, its S2 no longer blocks generation
    assert x.x0_clearance({**good, "gate_v": {"stop": "S2"}}) == {"y_lo": 1e4, "y_hi": 1e6}
    # S3 is re-judged from the efficiencies (dark-only triggers are recorded, not a stop)
    old_rec = {"stop": "S3", "efficiency": {"photon_y_lo": 1.0, "muon": 1.0, "electron": 1.0}}
    assert x.x0_clearance({**good, "trigger": old_rec})["y_lo"] == 1e4
    with pytest.raises(x.StopError) as err:
        x.x0_clearance({**good, "trigger": {"stop": None, "efficiency": {"muon": 0.98}}})
    assert err.value.stop == "S3"
    with pytest.raises(x.StopError) as err:
        x.x0_clearance({**good, "storage": {"stop": "S5"}})
    assert err.value.stop == "S5"
    with pytest.raises(ValueError):
        x.x0_clearance({k: v for k, v in good.items() if k != "trigger"})


def _write_evals(root, det, spec):
    runs = np.repeat([f"r{i}" for i in range(20)], 10)
    rng = np.random.default_rng(5)
    for (arm, seed), (vertex, f_event) in spec.items():
        d = root / "eval" / x.run_name(det, arm, seed)
        d.mkdir(parents=True)
        err = np.abs(rng.normal(vertex, 0.002, 200))
        (d / "metrics.json").write_text(json.dumps({"metrics": {
            "vertex_full_m": float(err.mean()), "token_log_charge_r2": 0.9, "f_event": f_event,
            "module_gain_r2": 0.2, "particle": {}}}))
        np.savez(d / "per_event.npz", vertex_errors=err, run_ids=runs, event_ids=np.arange(200))


def test_three_seed_version_for_a_single_arm_trigger(tmp_path):
    spec = {("init", s): (0.30, 0.05) for s in (9000, 9001, 9002, 9003, 9004)}
    spec.update({("sigreg_jepa", s): (0.40, 5e-4) for s in (9000, 9001)})
    spec.update({("target_posfree", s): (0.38, 0.05) for s in (9000, 9001)})  # no range overlap
    # MAE: seed 9000 better than init, seed 9001 worse -> R2-c unclear -> seed 9002 for mae_mse
    spec.update({("mae_mse", 9000): (0.20, 0.05), ("mae_mse", 9001): (0.35, 0.05)})
    _write_evals(tmp_path, "sk", spec)
    two = x.detector_readings(tmp_path, "sk")
    assert two["R2-c"]["class"] == "unclear"
    assert [t["arm"] for t in two["seed9002_triggers"]] == ["mae_mse"]
    # with design-time projections for JUNO and HK the cap is already full -> skipped
    assert x.seed9002_decision(tmp_path, "sk", "mae_mse")["entry"]["decision"] == "skip"
    for det in ("sk", "juno", "hk"):
        for arm in x.TRAINED_ARMS:
            _smoke(tmp_path, det, arm, 0.5)
    dec = x.seed9002_decision(tmp_path, "sk", "mae_mse")
    assert dec["entry"]["decision"] == "train"
    assert x.seed9002_decision(tmp_path, "sk", "sigreg_jepa")["entry"] is None
    _write_evals(tmp_path, "sk", {("mae_mse", 9002): (0.21, 0.05)})
    both = x.readings_both_versions(tmp_path, "sk")
    three = both["three_seed"]
    assert three["seeds_used"]["R2-c"] == [9000, 9001, 9002]
    assert three["seeds_used"]["R2-a"] == [9000, 9001]
    assert three["seeds_used"]["R2-b"] == [9000, 9001]
    assert sorted(three["R2-c"]["per_seed"]) == ["9000", "9001", "9002"]
    assert three["R2-c"]["class"] == "unclear"
    assert both["two_seed"]["seeds_used"]["R2-c"] == [9000, 9001]


def test_train_refuses_untriggered_seed_9002(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        x.train_r2(detector="sk", arm="mae_mse", seed=9002, h5_path=tmp_path / "x.h5",
                   out_root=tmp_path, device="cpu")


def test_stopped_detector_owes_nothing(tmp_path):
    before = x.reserved_gpu_h(tmp_path, x.ledger_load(tmp_path))
    x.mark_stopped(tmp_path, "sk", "S2", "Gate V")
    ledger = x.ledger_load(tmp_path)
    assert x.planned_runs(ledger, "sk") == []
    after = x.reserved_gpu_h(tmp_path, ledger)
    expected = 6 * x.DESIGN_MAIN_RUN_H["sk"] + x.DESIGN_RF_RUN_H["sk"] + x.DESIGN_EVAL_H
    assert abs((before - after) - expected) < 1e-9
    pytest.importorskip("torch")
    with pytest.raises(x.StopError):
        x.train_r2(detector="sk", arm="mae_mse", seed=9000, h5_path=tmp_path / "x.h5",
                   out_root=tmp_path, device="cpu")


def test_resume_stopped_keeps_record(tmp_path):
    x.mark_stopped(tmp_path, "sk", "S2", "Gate V")
    x.resume_stopped(tmp_path, "sk", "addendum d237ec2b")
    ledger = x.ledger_load(tmp_path)
    assert not x.is_stopped(ledger, "sk")
    assert ledger["resumed"]["sk"]["original_stop"]["stop"] == "S2"
    assert len(x.planned_runs(ledger, "sk")) == 7
    with pytest.raises(ValueError):
        x.resume_stopped(tmp_path, "sk", "again")


def test_run_stop_removes_one_run(tmp_path):
    before = x.reserved_gpu_h(tmp_path, x.ledger_load(tmp_path))
    x.mark_run_stopped(tmp_path, "sk", x.RF_ARM, 9000, "S1", "cap")
    ledger = x.ledger_load(tmp_path)
    assert (x.RF_ARM, 9000) not in x.planned_runs(ledger, "sk")
    assert len(x.planned_runs(ledger, "sk")) == 6
    after = x.reserved_gpu_h(tmp_path, ledger)
    assert abs((before - after) - x.DESIGN_RF_RUN_H["sk"]) < 1e-9
    pytest.importorskip("torch")
    with pytest.raises(x.StopError) as err:
        x.train_r2(detector="sk", arm=x.RF_ARM, seed=9000, h5_path=tmp_path / "x.h5",
                   out_root=tmp_path, device="cpu")
    assert err.value.stop == "S1"
