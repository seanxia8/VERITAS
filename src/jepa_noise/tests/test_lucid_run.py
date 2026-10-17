"""WP5 LUCiD runner tests (CPU, tiny synthetic fixture, §3.5.6)."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, copy, json, signal, time
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_noise import lucid_run as lr, lucid_bombs as lb, lucid_arms as la

CONFIG = lr.load_run_config()
GEOM = lb.from_synthetic({"module_ids": tuple(range(4)),
                          "surface_counts": {"top": 16, "barrel": 44, "bottom": 16}})
GEN = lb.load_generator_config(lr.ROOT / CONFIG["data"]["generator_config"])
ARMS = la.load_config()
CPU = torch.device("cpu")


def _args(**kw):
    base = dict(arm="sigreg_jepa", runs=2, run_events=4, steps=2, micro_batch=None, resume=None,
                stop_after=None, evaluate=False, probe_events=8, probe_run_events=4,
                probe_max_events=None, sizes="2048", batch=2, hit_fraction=0.2,
                memory_cap_mb=None, step=1, detectability=None, plain_reference=None,
                cells=None, no_evaluate=False, beta_decision=None, arms=None, seeds=None, dry_run=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _smoke(tmp: Path, tag="out", **kw):
    return lr.run_smoke(_args(**kw), CONFIG, geometry=GEOM, generator_config=GEN, arms_config=ARMS,
                        data_root=tmp / "data", out_root=tmp / tag, config_sha="cfg", code_sha="code",
                        status_text="", device=CPU, log=lambda *_: None)


# ---------------------------------------------------------------- enumeration
def test_cell_and_development_enumeration_matches_the_registered_numbers():
    cells = lr.enumerate_cells(CONFIG)
    main, sec = [c for c in cells if c.role == "main"], [c for c in cells if c.role != "main"]
    assert (len(main), len(sec)) == (6, 5)
    assert sum(c.n_runs for c in main) == 162      # 3 x 6 x 5 + 4 x 6 x 3
    assert sum(c.n_runs for c in sec) == 39        # 15 + 9 + 3 + 9 + 3
    assert sum(c.n_runs for c in cells) == 201
    dev = lr.development_runs(CONFIG)
    assert len(dev) == 17 and sum(r["step"] == 2 for r in dev) == 2
    assert {round(c.setting, 2) for c in main} == {0.02, 0.05, 0.10}
    assert all(c.s_p == 0.10 for c in cells)       # frozen F3 spread in every cell
    assert not any(c.s_p == 0 for c in cells)      # no unregistered s_p = 0 cell
    assert [c.cell_id for c in main if c.r6_r7_eligible] == ["main_s005_across", "main_s005_within",
                                                             "main_s010_across", "main_s010_within"]
    r11 = next(c for c in sec if c.role == "r11")
    assert (r11.law, r11.setting, r11.split) == ("F2", 0.5, "within")
    assert r11.arms == ("sigreg_jepa", "sigreg_jepa_anchored", "mae_mse") and r11.n_runs == 15
    assert next(c for c in sec if c.role == "r11b").law == "R11b"
    assert next(c for c in sec if c.role == "dark_anchor").n_runs == 3
    assert next(c for c in sec if c.role == "law_t").settings_source == "law_T_training_ratios"
    assert next(c for c in sec if c.role == "occupancy").occupancy == "balanced"
    assert not any("R12" in c.cell_id or c.criterion == "R12" for c in cells)  # no R12 training cell
    assert any("s_p = 0" in q for q in CONFIG["unresolved_questions"])
    assert any("write-once" in q for q in CONFIG["unresolved_questions"])
    assert any("step 4" in q for q in CONFIG["unresolved_questions"])


# ---------------------------------------------------------------- dry-run / gates
def test_grid_dry_run_writes_nothing_and_pending_optimizer_refuses(tmp_path):
    out, data = tmp_path / "out", tmp_path / "data"
    rc = lr.main(["grid", "--out", str(out), "--data-root", str(data), "--dry-run",
                  "--cells", "main_s005_within"])
    assert rc == 0 and not out.exists() and not data.exists()
    with pytest.raises(RuntimeError, match="pending_review"):
        lr.run_grid(_args(cells="main_s005_within"), CONFIG, geometry=GEOM,
                    generator_config=GEN, arms_config=ARMS, data_root=data, out_root=out,
                    config_sha="cfg", code_sha="code", status_text="", device=CPU,
                    log=lambda *_: None)
    assert not out.exists()


# ---------------------------------------------------------------- smoke / resume / determinism
def test_smoke_writes_manifest_status_checkpoint_and_is_deterministic(tmp_path):
    a = _smoke(tmp_path, "a")
    run_dir = tmp_path / "a" / "runs" / a["run_id"]
    assert a["outcome"] == "complete" and a["steps_completed"] == 2 and a["registered"] is False
    for name in ("run_status.json", "checkpoint.pt", "final.pt", "SHA256SUMS"):
        assert (run_dir / name).is_file()
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["outcome"] == "complete" and status["steps_completed"] == 2
    row = json.loads((tmp_path / "a" / "manifest.jsonl").read_text().splitlines()[-1])
    for field in ("arm", "cell_id", "seed", "config_sha256", "code_sha", "clean_tree", "device",
                  "wall_seconds", "steps", "geometry_sha256", "loader_dataset", "readout_label"):
        assert field in row
    b = _smoke(tmp_path, "b")
    ma = torch.load(run_dir / "final.pt", weights_only=False)["model"]
    mb = torch.load(tmp_path / "b" / "runs" / b["run_id"] / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(ma[k], mb[k]) for k in ma)   # same seed -> bit-identical
    assert sorted(p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*")) and \
        {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*")} <= {"a", "b", "data"}


def test_resume_is_bit_identical(tmp_path):
    direct = _smoke(tmp_path, "direct")
    stopped = _smoke(tmp_path, "stopped", stop_after=1)
    assert stopped["outcome"] == "interrupted" and stopped["steps_completed"] == 1
    checkpoint = tmp_path / "stopped" / "runs" / stopped["run_id"] / "checkpoint.pt"
    resumed = _smoke(tmp_path, "stopped", resume=str(checkpoint))
    assert resumed["outcome"] == "complete" and resumed["steps_completed"] == 2
    a = torch.load(tmp_path / "direct" / "runs" / direct["run_id"] / "final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "stopped" / "runs" / resumed["run_id"] / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_sigterm_checkpoints_and_interrupts(tmp_path):
    def callback(step):
        if step == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    ds = lr.ensure_dataset(data_root=tmp_path / "data", geometry=GEOM, generator_config=GEN,
                           cohort_spec={"namespace": "lucid/sig", "event_start": 0, "n_events": 8,
                                        "run_length": 4}, cohort_name="sig", law="F1", setting=0.05,
                           seed=1, registered=False, log=lambda *_: None)
    cell = next(c for c in lr.enumerate_cells(CONFIG) if c.cell_id == "main_s005_within")
    result = lr.run_one(arm_name="sigreg_jepa", cell=lr.replace(cell, cell_id="sig_cell"), seed=0,
                        beta=None, config=CONFIG, arms_config=ARMS, geometry=GEOM,
                        loader_dataset=ds, probe_datasets={}, dark_bank=None, out_root=tmp_path / "out",
                        data_root=tmp_path / "data", config_sha="cfg", code_sha="code",
                        status_text="", device=CPU, steps=2, allow_pending=True,
                        step_callback=callback, log=lambda *_: None)
    assert result["manifest"]["outcome"] == "interrupted"
    assert result["manifest"]["interrupted_by"] == "sigterm"
    assert (result["run_dir"] / "checkpoint.pt").is_file()


# ---------------------------------------------------------------- scaling
def test_scaling_records_does_not_fit_and_fits_a_tiny_layout(tmp_path):
    payload = lr.run_scaling(_args(sizes="2048", memory_cap_mb=1), CONFIG, out_root=tmp_path,
                             device=CPU, log=lambda *_: None)
    assert payload["results"][0]["status"] == "does_not_fit"
    assert payload["results"][0]["does_not_fit"] is True
    assert (tmp_path / "scaling.json").is_file()
    tiny = lr.run_scaling(_args(sizes="80", memory_cap_mb=1e9), CONFIG, out_root=tmp_path / "tiny",
                          device=CPU, log=lambda *_: None)
    assert tiny["results"][0]["status"] == "fit" and tiny["results"][0]["parameter_count"] > 0


# ---------------------------------------------------------------- decisions
def test_physics_probe_and_beta_rules():
    assert lr.decide_physics_probe({"sigreg_jepa": 0.31, "mae_mse": 0.32}, 0.10)["decision"] == "mlp"
    assert lr.decide_physics_probe({"sigreg_jepa": 0.31, "mae_mse": 0.10}, 0.10)["decision"] == "ridge"
    assert lr.decide_physics_probe({"sigreg_jepa": 0.30, "mae_mse": 0.31}, None)["decision"] == "not_evaluable"
    table = [{"beta": b, "resolution": r} for b, r in
             [(0.1, 0.105), (1, 0.108), (10, 0.109), (100, 0.2), (1000, 0.3)]]
    assert lr.select_beta(table, 0.10)["selected"] == 10       # largest within 1.1x
    assert lr.select_beta(table, 0.001)["status"] == "none_qualified"


# ---------------------------------------------------------------- verdicts
def _tables():
    rows, probes, floors = {}, {}, {}
    def add(cell, arm, seed, r2, phys=None, r11=None):
        rid = lr.run_id_for(cell, arm, seed)
        rows[rid] = {"cell_id": cell, "arm": arm, "seed": seed, "outcome": "complete", "registered": False}
        probes[rid] = {"results": {k: {"status": "evaluated", "r2_mean": v} for k, v in r2.items()},
                       "physics": {"resolution": phys} if phys is not None else {}, "r11": r11 or {}}
    for spread in (5, 10):
        for split, base in (("within", 0.55), ("across", 0.15)):
            cell = f"main_s{spread:03d}_{split}"
            for seed in range(5):
                add(cell, "sigreg_jepa", seed, {"module_gain::propagated": base + 0.001 * seed,
                                                "module_gain::run_mean": base, "pmt_gain::propagated": 0.02,
                                                "pmt_gain::run_mean": 0.6}, phys=0.10 + 0.001 * seed)
            floors[f"sigreg_jepa::{cell}"] = {"results": {
                "module_gain::propagated": {"status": "evaluated", "r2_mean": 0.05},
                "module_gain::run_mean": {"status": "evaluated", "r2_mean": 0.05},
                "pmt_gain::propagated": {"status": "evaluated", "r2_mean": 0.02}}}
    cell = "main_s010_within"
    for seed in range(5):
        add(cell, "sigreg_jepa_anchored", seed, {"module_gain::run_mean": 0.30}, phys=0.105)
    for seed in range(3):
        add(cell, "sigreg_jepa_anchored_dark", seed, {"module_gain::run_mean": 0.55})
    for seed in range(5):
        add("r11_f1f2f3_within", "sigreg_jepa", seed, {"module_gain::run_mean": 0.70,
                                                       "f2_variable::run_mean": 0.60},
            r11={"leakage_i": 0.01, "leakage_ii": 0.02})
    for arm in ("sigreg_jepa_anchored", "mae_mse"):
        for seed in range(5):
            add("r11_f1f2f3_within", arm, seed, {"module_gain::run_mean": 0.6, "f2_variable::run_mean": 0.5})
    for seed in range(3):
        add("r11b_law_within", "sigreg_jepa", seed, {"module_gain::run_mean": 0.7,
                                                     "f2_variable::run_mean": 0.6},
            r11={"leakage_i": 0.01, "leakage_ii": 0.02})
    for arm, cell in (("sigreg_jepa", "r11_f1f2f3_within"), ("sigreg_jepa", "r11b_law_within")):
        floors[f"{arm}::{cell}"] = {"results": {"module_gain::run_mean": {"status": "evaluated", "r2_mean": 0.1},
                                                "f2_variable::run_mean": {"status": "evaluated", "r2_mean": 0.1}}}
    return {"runs": rows, "probes": probes, "floors": floors, "detectability": {"f3_ok": False}}


def test_verdict_wiring_pass_fail_and_not_evaluable():
    tables = _tables()
    verdicts = lr.evaluate_verdicts(tables, CONFIG)
    assert verdicts["R6"]["status"] == "pass"
    assert verdicts["R7"]["status"] == "pass"
    assert verdicts["R7b"]["status"] == "pass"
    assert verdicts["R11"]["status"] == "pass"
    assert verdicts["R11b"]["status"] == "fail"       # co-trained models keep both conditions
    assert verdicts["R12"]["status"] == "not_evaluable"  # F3 failed the frozen check
    failed = copy.deepcopy(tables)
    for rid, probe in failed["probes"].items():
        if rid.startswith("main_s005_across__sigreg_jepa") or rid.startswith("main_s010_across__sigreg_jepa"):
            probe["results"]["module_gain::propagated"]["r2_mean"] = 0.55
    assert lr.evaluate_verdicts(failed, CONFIG)["R6"]["status"] == "fail"
    incomplete = copy.deepcopy(tables)
    incomplete["runs"].pop(lr.run_id_for("main_s010_within", "sigreg_jepa", 4))
    verdict = lr.evaluate_verdicts(incomplete, CONFIG)["R6"]
    assert verdict["status"] == "not_evaluable" and verdict["missing"]
    passing_r12 = copy.deepcopy(tables)
    passing_r12["detectability"] = {"f3_ok": True}
    assert lr.evaluate_verdicts(passing_r12, CONFIG)["R12"]["status"] == "pass"


# ---------------------------------------------------------------- dev cohort
def test_dev_step4_refused_and_step1_writes_a_frozen_decision(tmp_path):
    with pytest.raises(RuntimeError, match="step 4"):
        lr.run_dev(_args(step=4), CONFIG, geometry=GEOM, gen_config=GEN, arms_config=ARMS,
                   data_root=tmp_path / "data", out_root=tmp_path / "out", config_sha="cfg",
                   code_sha="code", status_text="", device=CPU, dry_run=False, log=lambda *_: None)
    assert not (tmp_path / "out").exists()
    dry = lr.run_dev(_args(step=2), CONFIG, geometry=GEOM, gen_config=GEN, arms_config=ARMS,
                     data_root=tmp_path / "data", out_root=tmp_path / "out", config_sha="cfg",
                     code_sha="code", status_text="", device=CPU, dry_run=True, log=lambda *_: None)
    assert len(dry["runs"]) == 2 and not (tmp_path / "out").exists()
    summary = lr.run_dev(_args(step=1), CONFIG, geometry=GEOM, gen_config=GEN, arms_config=ARMS,
                         data_root=tmp_path / "data", out_root=tmp_path / "out", config_sha="cfg",
                         code_sha="code", status_text="", device=CPU, dry_run=False, log=lambda *_: None)
    path = tmp_path / "out" / "dev" / "step1_detectability_decision.json"
    payload = json.loads(path.read_text())
    assert payload["registered"] is False and payload["decisions"]["f3_ok"] is False
    assert payload["decisions"]["f1_eligible_settings"] == [0.05, 0.1]
    assert summary["decisions"]["r12_status"] == "not_evaluable"
    with pytest.raises(RuntimeError, match="step 4"):
        lr.run_dev(_args(step=4), CONFIG, geometry=GEOM, gen_config=GEN, arms_config=ARMS,
                   data_root=tmp_path / "data", out_root=tmp_path / "out", config_sha="cfg",
                   code_sha="code", status_text="", device=CPU, dry_run=True, log=lambda *_: None)
