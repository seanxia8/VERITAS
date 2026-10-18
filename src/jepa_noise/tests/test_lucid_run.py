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
from jepa_noise import point_encoder as pe

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
                cells=None, no_evaluate=False, beta_decision=None, arms=None, seeds=None,
                dry_run=False, physics_probe_decision=None)
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
    assert all(c.s_p == 0.05 for c in cells)       # s_p restarted at the registered default
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


def _pending_config():
    config = copy.deepcopy(CONFIG)
    config["training"]["optimizer"] = "pending_review"
    config["training"]["optimizer_detail"] = {
        key: "pending_review" for key in ("family", "learning_rate", "betas", "weight_decay",
                                          "schedule", "warmup_steps", "gradient_clip",
                                          "muon_ns_steps")}
    return config


# ---------------------------------------------------------------- dry-run / gates
def test_grid_dry_run_writes_nothing_and_pending_optimizer_still_refuses(tmp_path):
    out, data = tmp_path / "out", tmp_path / "data"
    rc = lr.main(["grid", "--out", str(out), "--data-root", str(data), "--dry-run",
                  "--cells", "main_s005_within"])
    assert rc == 0 and not out.exists() and not data.exists()
    with pytest.raises(RuntimeError, match="pending_review"):
        lr.run_grid(_args(cells="main_s005_within"), _pending_config(), geometry=GEOM,
                    generator_config=GEN, arms_config=ARMS, data_root=data, out_root=out,
                    config_sha="cfg", code_sha="code", status_text="", device=CPU,
                    log=lambda *_: None)
    assert not out.exists()


# ---------------------------------------------------------------- probe score (P0.1)
def _probe_table(offset=0.0, rows=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 4))
    y = (x @ np.array([1.0, -2.0, 0.5, 0.0]) + offset + 0.01 * rng.normal(size=rows))[:, None]
    return la.ProbeDataset("module_gain", "run_mean", x, y, np.arange(rows), ["gain"],
                           {"key": np.arange(rows)})


def test_score_computes_a_probe_score():
    train, test = _probe_table(), _probe_table(0.1, seed=1)
    scores = lr._score({("module_gain", "run_mean"): train}, {("module_gain", "run_mean"): test})
    assert set(scores) == {"module_gain::run_mean"}
    assert scores["module_gain::run_mean"]["status"] == "evaluated"
    assert scores["module_gain::run_mean"]["r2_mean"] > 0.9


# ---------------------------------------------------------------- frozen optimizer (P0.5)
def test_frozen_optimizer_values_groups_and_constant_schedule():
    cell = next(c for c in lr.enumerate_cells(CONFIG) if c.cell_id == "main_s005_within")
    arm = lr.build_arm("sigreg_jepa", GEOM, ARMS, cell, 0)
    opts, info, scheduler = lr.build_optimizers(arm, CONFIG)
    adamw = next(o for o in opts if isinstance(o, torch.optim.AdamW))
    group = adamw.param_groups[0]
    assert group["lr"] == 1e-3 and tuple(group["betas"]) == (0.9, 0.999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0.0 and group["fused"] is False
    muon = next(o for o in opts if type(o).__name__ == "Muon")
    mgroup = muon.param_groups[0]
    assert (mgroup["lr"], mgroup["momentum"], mgroup["nesterov"], mgroup["ns_steps"]) == \
        (1e-3, 0.95, True, 5)
    groups = arm.parameter_groups()
    assert info["groups"]["muon"] == len(groups["muon"])
    assert info["groups"]["adamw"] == len(groups["adamw"])
    assert info["schedule"] == "constant" and info["warmup_steps"] == 0
    assert info["gradient_clip"] is None and info["fused_requested"] is True
    before = [g["lr"] for o in opts for g in o.param_groups]
    scheduler.step()
    assert [g["lr"] for o in opts for g in o.param_groups] == before   # implemented no-op
    bad = copy.deepcopy(CONFIG)
    bad["training"]["optimizer_detail"]["schedule"] = "cosine"
    with pytest.raises(RuntimeError, match="schedule"):
        lr.build_optimizers(arm, bad)
    bad_warmup = copy.deepcopy(CONFIG)
    bad_warmup["training"]["optimizer_detail"]["warmup_steps"] = 100
    with pytest.raises(RuntimeError, match="warmup"):
        lr.build_optimizers(arm, bad_warmup)


def test_frozen_numerics_are_refused_if_the_config_changes(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["optimizer_detail"]["eps"] = 1e-6
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="frozen optimizer numerics"):
        lr.load_run_config(path)


# ---------------------------------------------------------------- grid end-to-end (P0.2)
def _tiny_config():
    config = copy.deepcopy(CONFIG)
    config["data"]["dataset_seed"] = 7
    config["data"]["cohorts"] = {
        "train": {"namespace": "lucid/ttrain", "event_start": 0, "n_events": 20, "run_length": 4},
        "probe_train": {"namespace": "lucid/tptrain", "event_start": 1000, "n_events": 20,
                        "run_length": 4},
        "probe_test": {"namespace": "lucid/tptest", "event_start": 2000, "n_events": 20,
                       "run_length": 4},
        "robustness": {"namespace": "lucid/trob", "event_start": 3000, "n_events": 20,
                       "run_length": 4},
    }
    return config


def test_grid_runs_end_to_end_and_threads_physics_method(tmp_path):
    config = _tiny_config()
    decision = tmp_path / "step2_decision.json"
    decision.write_text(json.dumps({"physics_probe": {"decision": "ridge"}}))
    out, data = tmp_path / "out", tmp_path / "data"
    payload = lr.run_grid(
        _args(cells="main_s005_within", arms="sigreg_jepa", seeds="0", steps=2,
              physics_probe_decision=str(decision)), config, geometry=GEOM,
        generator_config=GEN, arms_config=ARMS, data_root=data, out_root=out,
        config_sha="cfg", code_sha="code", status_text="", device=CPU, log=lambda *_: None)
    run_id = lr.run_id_for("main_s005_within", "sigreg_jepa", 0)
    assert payload["completed"] == [run_id]
    row = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()][-1]
    assert row["run_id"] == run_id and row["outcome"] == "complete"
    assert row["steps_completed"] == 2
    assert row["physics"]["model_vertex_probe"] == "ridge"
    assert "module_gain::run_mean" in row["probes"]           # exercises _score
    assert row["optimizer"]["eps"] == 1e-8 and row["optimizer"]["schedule"] == "constant"
    assert row["fallback"]["active"] is False
    assert (out / "floors" / "sigreg_jepa__main_s005_within.json").is_file()
    probes = json.loads((out / "runs" / run_id / "probes.json").read_text())
    assert probes["results"]["module_gain::run_mean"]["status"] == "evaluated"


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
    # P0.5: the manifest records the applied, frozen numerics (not just the config text).
    optimizer = row["optimizer"]
    assert optimizer["learning_rate"] == 1e-3 and optimizer["eps"] == 1e-8
    assert optimizer["betas"] == [0.9, 0.999] and optimizer["weight_decay"] == 0.0
    assert optimizer["muon_momentum"] == 0.95 and optimizer["muon_nesterov"] is True
    assert optimizer["muon_ns_steps"] == 5 and optimizer["schedule"] == "constant"
    assert optimizer["gradient_clip"] is None and optimizer["fused"] is False
    assert row["fallback"]["active"] is False
    b = _smoke(tmp_path, "b")
    ma = torch.load(run_dir / "final.pt", weights_only=False)["model"]
    mb = torch.load(tmp_path / "b" / "runs" / b["run_id"] / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(ma[k], mb[k]) for k in ma)   # same seed -> bit-identical
    assert sorted(p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*")) and \
        {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*")} <= {"a", "b", "data"}


def test_resume_is_bit_identical_and_automatic(tmp_path):
    direct = _smoke(tmp_path, "direct")
    stopped = _smoke(tmp_path, "stopped", stop_after=1)
    assert stopped["outcome"] == "interrupted" and stopped["steps_completed"] == 1
    # documented behaviour: a checkpoint in the run dir resumes without an explicit flag
    resumed = _smoke(tmp_path, "stopped")
    assert resumed["outcome"] == "complete" and resumed["steps_completed"] == 2
    assert resumed["resumed_from_step"] == 1
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
def _add(tables, cell, arm, seed, r2, phys=None, r11=None):
    rid = lr.run_id_for(cell, arm, seed)
    tables["runs"][rid] = {"cell_id": cell, "arm": arm, "seed": seed,
                           "outcome": "complete", "registered": False}
    tables["probes"][rid] = {"results": {k: {"status": "evaluated", "r2_mean": v} for k, v in r2.items()},
                             "physics": {"resolution": phys} if phys is not None else {},
                             "r11": r11 or {}}


def _floor(tables, arm, cell, **metrics):
    tables["floors"][f"{arm}::{cell}"] = {"results": {
        k: {"status": "evaluated", "r2_mean": v} for k, v in metrics.items()}}


def _tables():
    """A complete synthetic table for the six verdicts (§3.5.9)."""
    tables = {"runs": {}, "probes": {}, "floors": {}, "detectability": {"f3_ok": False}}
    main_arms = {"sigreg_jepa": 5, "sigreg_jepa_anchored": 5, "mae_mse": 5, "mae_chi2": 3,
                 "ijepa_ema": 3, "anchor_only": 3, "oracle_nominal_twin": 3}
    for spread in (5, 10):
        for split, base in (("within", 0.55), ("across", 0.15)):
            cell = f"main_s{spread:03d}_{split}"
            for arm, seeds in main_arms.items():
                for seed in range(seeds):
                    anchored = arm == "sigreg_jepa_anchored" and cell == "main_s010_within"
                    _add(tables, cell, arm, seed,
                         {"module_gain::propagated": base + 0.001 * seed,
                          "module_gain::run_mean": 0.30 if anchored else
                          (base if arm == "sigreg_jepa" else 0.6),
                          "pmt_gain::propagated": 0.02, "pmt_gain::run_mean": 0.6},
                         phys=0.105 if anchored else 0.10 + 0.001 * seed)
                _floor(tables, arm, cell,
                       **{"module_gain::propagated": 0.05, "module_gain::run_mean": 0.05,
                          "pmt_gain::propagated": 0.02, "pmt_gain::run_mean": 0.02})
    cell = "main_s010_within"
    for seed in range(3):
        _add(tables, cell, "sigreg_jepa_anchored_dark", seed, {"module_gain::run_mean": 0.55})
    for seed in range(5):
        _add(tables, "r11_f1f2f3_within", "sigreg_jepa", seed,
             {"module_gain::run_mean": 0.70, "f2_variable::run_mean": 0.60},
             r11={"leakage_i": 0.01, "leakage_ii": 0.02})
    for arm in ("sigreg_jepa_anchored", "mae_mse"):
        for seed in range(5):
            _add(tables, "r11_f1f2f3_within", arm, seed,
                 {"module_gain::run_mean": 0.6, "f2_variable::run_mean": 0.5})
    for arm in ("sigreg_jepa", "sigreg_jepa_anchored", "mae_mse"):
        _floor(tables, arm, "r11_f1f2f3_within",
               **{"module_gain::run_mean": 0.1, "f2_variable::run_mean": 0.1})
    for seed in range(3):
        _add(tables, "r11b_law_within", "sigreg_jepa", seed,
             {"module_gain::run_mean": 0.7, "f2_variable::run_mean": 0.6},
             r11={"leakage_i": 0.01, "leakage_ii": 0.02})
    for arm in ("sigreg_jepa_anchored", "mae_mse"):
        for seed in range(3):
            _add(tables, "r11b_law_within", arm, seed,
                 {"module_gain::run_mean": 0.6, "f2_variable::run_mean": 0.5})
    for arm in ("sigreg_jepa", "sigreg_jepa_anchored", "mae_mse"):
        _floor(tables, arm, "r11b_law_within",
               **{"module_gain::run_mean": 0.1, "f2_variable::run_mean": 0.1})
    return tables


def test_verdict_wiring_pass_fail_and_not_evaluable():
    tables = _tables()
    verdicts = lr.evaluate_verdicts(tables, CONFIG)
    assert verdicts["R6"]["status"] == "pass"
    assert verdicts["R7"]["status"] == "pass"
    # R-6: dark counts are on, so the dark-anchor verdict is evaluable again
    assert verdicts["R7b"]["status"] == "pass"
    assert verdicts["R7b"]["values"]["mean_dark"] >= verdicts["R7b"]["values"]["threshold"]
    # the companion number is reported beside R7b (absent in this synthetic table)
    assert verdicts["R7b"]["values"]["dark_only_embedding_change"]["status"] == "not_evaluable"
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


def test_missing_metric_or_floor_is_not_evaluable_for_every_verdict():
    """P0.3: a missing required target/metric/floor is not_evaluable, never fail/pass."""
    # R6: a non-test arm's random-init floor is required for the excess median SD.
    tables = _tables()
    tables["floors"].pop("mae_chi2::main_s005_within")
    verdict = lr.evaluate_verdicts(tables, CONFIG)["R6"]
    assert verdict["status"] == "not_evaluable"
    assert any("floor:mae_chi2" in item for item in verdict["missing"])
    # R6: a missing propagated metric on one seed.
    tables = _tables()
    tables["probes"][lr.run_id_for("main_s005_across", "sigreg_jepa", 3)]["results"] \
        .pop("module_gain::propagated")
    assert lr.evaluate_verdicts(tables, CONFIG)["R6"]["status"] == "not_evaluable"
    # R7: the physics resolution is a required target of the R7 criterion.
    tables = _tables()
    tables["probes"][lr.run_id_for("main_s010_within", "sigreg_jepa_anchored", 2)]["physics"] = {}
    verdict = lr.evaluate_verdicts(tables, CONFIG)["R7"]
    assert verdict["status"] == "not_evaluable" and verdict["missing"]
    # R7b: missing dark-anchor metric.
    tables = _tables()
    tables["probes"][lr.run_id_for("main_s010_within", "sigreg_jepa_anchored_dark", 1)]["results"] \
        .pop("module_gain::run_mean")
    assert lr.evaluate_verdicts(tables, CONFIG)["R7b"]["status"] == "not_evaluable"
    # R11: missing F2 (attenuation) metric used by its own margin.
    tables = _tables()
    tables["probes"][lr.run_id_for("r11_f1f2f3_within", "sigreg_jepa", 1)]["results"] \
        .pop("f2_variable::run_mean")
    verdict = lr.evaluate_verdicts(tables, CONFIG)["R11"]
    assert verdict["status"] == "not_evaluable"
    assert any("f2_variable::run_mean" in item for item in verdict["missing"])
    # R11: missing cross-leakage.
    tables = _tables()
    tables["probes"][lr.run_id_for("r11_f1f2f3_within", "sigreg_jepa", 0)]["r11"] = {}
    assert lr.evaluate_verdicts(tables, CONFIG)["R11"]["status"] == "not_evaluable"
    # R11b: a non-sigreg arm's metric is required for the all-arm median.
    tables = _tables()
    tables["probes"][lr.run_id_for("r11b_law_within", "mae_mse", 2)]["results"] \
        .pop("f2_variable::run_mean")
    assert lr.evaluate_verdicts(tables, CONFIG)["R11b"]["status"] == "not_evaluable"
    # R12: missing propagated per-PMT gain (with F3 and R6 passing).
    tables = _tables()
    tables["detectability"] = {"f3_ok": True}
    tables["probes"][lr.run_id_for("main_s010_across", "sigreg_jepa", 2)]["results"] \
        .pop("pmt_gain::propagated")
    verdict = lr.evaluate_verdicts(tables, CONFIG)["R12"]
    assert verdict["status"] == "not_evaluable" and verdict["missing"]


def test_r11_f2_floor_uses_the_f2_metric_and_its_own_median_sd():
    """P0.4: the F2 seed floor must not be the module-gain median SD."""
    tables = _tables()
    values = [0.6, 0.9, 1.2, 0.6, 0.9]            # SD 0.27 on F1; F2 itself is constant
    for seed in range(5):
        for arm in ("sigreg_jepa", "sigreg_jepa_anchored", "mae_mse"):
            tables["probes"][lr.run_id_for("r11_f1f2f3_within", arm, seed)]["results"] \
                ["module_gain::run_mean"]["r2_mean"] = values[seed]
    verdict = lr.evaluate_verdicts(tables, CONFIG)["R11"]
    assert verdict["status"] == "pass", verdict["values"]["f2_margin"]
    assert verdict["values"]["seed_floor_f2"] == 0.0
    assert verdict["values"]["seed_floor_f1"] > 0.2


def test_r11b_median_sd_uses_all_three_arms():
    """P0.4 / §3.5.8: R11b's median SD is over all three arms, not sigreg only."""
    tables = _tables()
    noisy = [0.2, 0.7, 1.2]
    for seed in range(3):
        for arm in ("sigreg_jepa_anchored", "mae_mse"):
            tables["probes"][lr.run_id_for("r11b_law_within", arm, seed)]["results"] \
                ["module_gain::run_mean"]["r2_mean"] = noisy[seed]
    verdict = lr.evaluate_verdicts(tables, CONFIG)["R11b"]
    assert verdict["status"] == "pass"            # at least one R11 condition now fails
    assert verdict["values"]["seed_floor_f1"] == pytest.approx(0.5)
    assert not verdict["values"]["f1_margin"]


def test_r6_median_sd_is_over_excess_and_requires_every_arm_floor():
    """P0.4: the median SD stabiliser is the excess R² (R² - floor) spread."""
    tables = _tables()
    med = lr._median_sd_excess(tables, "main_s005_within",
                               {"sigreg_jepa": 5, "mae_chi2": 3},
                               "module_gain", "propagated")
    assert med is not None and med > 0
    # an arm with no floor cannot contribute an excess SD -> not_evaluable
    tables["floors"].pop("anchor_only::main_s010_across")
    verdict = lr.evaluate_verdicts(tables, CONFIG)["R6"]
    assert verdict["status"] == "not_evaluable"
    assert any("floor:anchor_only" in item for item in verdict["missing"])


def test_registered_reports_are_recorded_without_changing_verdicts():
    reports = lr.registered_reports(_tables(), CONFIG)
    first = reports["R6_full_input_module_gain_minus_floor"][0]
    assert first["metric"] == "module_gain::run_mean" and first["status"] == "evaluated"
    assert first["mean_excess"] is not None
    attenuation = reports["R7b_attenuation_anchored_vs_plain"]
    assert attenuation["status"] == "reported_without_criterion"
    assert attenuation["plain"]["values"] and attenuation["anchored"]["values"]
    assert reports["R12_full_input_per_pmt_gain"][0]["metric"] == "pmt_gain::run_mean"
    assert reports["occupancy_natural_vs_balanced"]["natural"][("module_gain", "run_mean")]["cell"] \
        == "main_s005_within"
    assert reports["R8_robustness"]["status"] == "not_evaluable"
    assert "ratio" in reports["R8_robustness"]["reason"]
    assert reports["law_T_extrapolation"]["status"] == "not_evaluable"


# ---------------------------------------------------------------- budget fallbacks
def test_budget_fallbacks_are_accepted_and_marked_non_registered(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["steps"] = 5000
    for arm in config["cells"]["main"]["arms_seeds"]:
        config["cells"]["main"]["arms_seeds"][arm] = 3
    path = tmp_path / "fallback.json"
    path.write_text(json.dumps(config))
    loaded = lr.load_run_config(path)
    marker = lr.fallback_active(loaded)
    assert marker["active"] is True and marker["steps_fallback"] and marker["seed_fallback"]
    # a mixed, unrecorded seed set is still rejected
    bad = copy.deepcopy(config)
    bad["cells"]["main"]["arms_seeds"]["sigreg_jepa"] = 4
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="fallback"):
        lr.load_run_config(bad_path)


# ---------------------------------------------------------------- floor cache / twin probes
def _tiny_dataset(tmp_path, tag="tiny", runs=5, run_events=4):
    return lr.ensure_dataset(
        data_root=tmp_path / "data", geometry=GEOM, generator_config=GEN,
        cohort_spec={"namespace": f"lucid/{tag}", "event_start": 0,
                     "n_events": runs * run_events, "run_length": run_events},
        cohort_name=tag, law="F1", setting=0.05, seed=0, registered=False,
        log=lambda *_: None)


def test_floor_cache_identity_includes_mask_seed_and_probe_cap(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    cell = next(c for c in lr.enumerate_cells(CONFIG) if c.cell_id == "main_s005_within")
    probes = {"train": dataset, "test": dataset}
    base = dict(arm_name="sigreg_jepa", cell=cell, geometry=GEOM, config=CONFIG,
                probe_datasets=probes, mask_seed=0, config_sha="cfg",
                out_root=tmp_path / "out", data_root=tmp_path / "data", log=lambda *_: None)
    lr.ensure_floor(**base)
    for changed in (dict(mask_seed=1), dict(max_events=4)):
        with pytest.raises(RuntimeError, match="different arm/cell/config"):
            lr.ensure_floor(**dict(base, **changed))


def test_oracle_nominal_twin_probes_read_the_twin_input(tmp_path):
    dataset = _tiny_dataset(tmp_path, "twin")
    model = la.random_init_floor("oracle_nominal_twin", GEOM, seed=0)
    kinds = {"run_mean": ["module_gain"]}
    real = lr.probe_tables(model, dataset, GEOM, split="within", kinds_by_form=kinds,
                           mask_seed=0)[("module_gain", "run_mean")].x
    twin = lr.probe_tables(model, dataset, GEOM, split="within", kinds_by_form=kinds,
                           mask_seed=0, twin=True)[("module_gain", "run_mean")].x
    assert not np.allclose(real, twin)


# ---------------------------------------------------------------- R-6 dark bank / R7b
def test_dark_bank_draws_real_dark_noise_only_windows(tmp_path):
    """R-6: the bank is a real dark-noise-only draw, not the legacy all-zero window."""
    directory = lr.ensure_dark_bank(data_root=tmp_path, geometry=GEOM, generator_config=GEN,
                                    seed=3, records=150)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["readout_method"] == "noise_module_lucid.readout.dark_noise_window"
    assert "dark_noise_window" in manifest["method"] and "keyed" in manifest["method"]
    assert manifest["readout_label"] == lr.READOUT_LABEL == "readout_pmt_r6"
    assert manifest["status"] == "real_dark_noise_only_windows"
    assert manifest["v_dark"] == lb.V_DARK_READOUT == 0.00109
    assert manifest["readout_params"]["dark_rate_hz"] == 1000.0
    assert manifest["readout_params"]["readout_window_ns"] == [-200.0, 800.0]
    assert manifest["expected_dark_hits_per_record"] == pytest.approx(GEOM.n_pmt * 0.001)
    store = lb.read_sparse_hits(directory / "hits.npz")
    counts = np.diff(store.offsets)
    assert counts.sum() > 0                       # the legacy bank was all zero
    assert manifest["measured_dark_hits_per_record_mean"] == pytest.approx(counts.mean())
    assert np.all(store.charge_pe > 0.0) and np.all(np.isfinite(store.time_ns))
    assert np.abs(store.time_ns).max() < 1000.0   # drawn in [-200, 800] plus the digit width
    # keyed by seed/run/window: the cache is reused, another seed is a different draw
    assert lr.ensure_dark_bank(data_root=tmp_path, geometry=GEOM, generator_config=GEN,
                               seed=3, records=150) == directory
    other = lr.ensure_dark_bank(data_root=tmp_path, geometry=GEOM, generator_config=GEN,
                                seed=4, records=150)
    assert not np.array_equal(lb.read_sparse_hits(other / "hits.npz").pmt_id, store.pmt_id)
    # a cached all-zero (legacy) bank is refused rather than silently overlaid
    legacy = tmp_path / "lucid" / "dark_noise__s9__n2"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text(json.dumps(
        {"geometry_sha256": GEOM.digest, "readout_label": lr.READOUT_LABEL,
         "method": "generator readout at yield_photons = 0, nominal state",
         "readout_method": "lucid_bombs.make_readout"}))
    with pytest.raises(RuntimeError, match="method"):
        lr.ensure_dark_bank(data_root=tmp_path, geometry=GEOM, generator_config=GEN,
                            seed=9, records=2)


def test_r7b_dark_only_embedding_change_on_a_tiny_fixture(tmp_path):
    """The companion number next to R7b: the dark-only draw's measured embedding change."""
    bank_dir = lr.ensure_dark_bank(data_root=tmp_path / "data", geometry=GEOM,
                                   generator_config=GEN, seed=3, records=400)
    store = lb.read_sparse_hits(bank_dir / "hits.npz")
    full_bank = la.dark_bank_from_store(store, GEOM)
    assert full_bank.hit.any()
    # keep only the records that carry dark hits (~2 hits/event means many are empty)
    keep = full_bank.hit.any(axis=1)
    bank = la.DarkBank(full_bank.charge_pe[keep], full_bank.time_100ns[keep], full_bank.hit[keep])
    assert bank.hit.shape[0] >= 5
    dataset = _tiny_dataset(tmp_path, "r7b")
    cell = next(c for c in lr.enumerate_cells(CONFIG) if c.role == "dark_anchor")
    arm = lr.build_arm(la.DARK_VARIANT, GEOM, ARMS, cell, 0, beta=10.0, dark_bank=bank)
    scaling = pe.FeatureScaling.from_geometry(GEOM)
    batches = [lr._probe_batch(dataset.directory / "runs" / row["run_dir"], GEOM, split="within",
                               mask_seed=0, occupancy="natural", scaling=scaling)
               for row in dataset.runs]
    change = lr.dark_only_embedding_change(arm, batches, device=CPU)
    assert change["status"] == "evaluated" and change["n_dark_records"] == bank.hit.shape[0]
    assert change["n_events"] > 0
    assert 0.0 < change["module_pooled_relative_l2_mean"] < 1.0
    assert change["full_input_relative_l2_mean"] > 0.0
    assert change["dark_hits_per_event_mean"] > 0.0
    assert "module-pooled" in change["definition"] and "full-input" in change["definition"]
    # the legacy all-zero bank is the identity: exactly zero change
    empty = lr.build_arm(la.DARK_VARIANT, GEOM, ARMS, cell, 0, beta=10.0,
                         dark_bank=la.DarkBank.empty(GEOM.n_pmt))
    zero = lr.dark_only_embedding_change(empty, batches, device=CPU)
    assert zero["module_pooled_relative_l2_mean"] == 0.0
    assert zero["full_input_relative_l2_mean"] == 0.0
    assert zero["dark_hits_per_event_mean"] == 0.0
    # evaluate_arm_run records it in the R7b run's probes.json
    (tmp_path / "run").mkdir()
    payload = lr.evaluate_arm_run(arm, arm_name=la.DARK_VARIANT, cell=cell, geometry=GEOM,
                                  probe_datasets={"train": dataset, "test": dataset},
                                  config=CONFIG, device=CPU, mask_seed=0,
                                  run_dir=tmp_path / "run")
    assert lr.DARK_CHANGE_KEY in payload
    back = json.loads((tmp_path / "run" / "probes.json").read_text())
    assert back[lr.DARK_CHANGE_KEY]["status"] == "evaluated"
    assert back[lr.DARK_CHANGE_KEY]["module_pooled_relative_l2_mean"] == pytest.approx(
        payload[lr.DARK_CHANGE_KEY]["module_pooled_relative_l2_mean"])
    # a non-anchored arm has no dark bank and no diagnostic
    plain = lr.build_arm("sigreg_jepa", GEOM, ARMS, cell, 0)
    assert lr.evaluate_arm_run(plain, arm_name="sigreg_jepa", cell=cell, geometry=GEOM,
                               probe_datasets={"train": dataset, "test": dataset},
                               config=CONFIG, device=CPU, mask_seed=0) .get(lr.DARK_CHANGE_KEY) is None


# ---------------------------------------------------------------- dev cohort
def test_physics_decision_flow_reads_the_nested_step2_decision(tmp_path):
    path = tmp_path / "step2.json"
    path.write_text(json.dumps({"physics_probe": {"decision": "mlp"}}))
    assert lr.physics_method_from_decision(path) == ("mlp", str(path))
    path.write_text(json.dumps({"physics_probe": {"decision": "not_evaluable"}}))
    assert lr.physics_method_from_decision(path)[0] == "ridge"
    path.write_text(json.dumps({"decision": "ridge"}))     # flat grid-facing decision
    assert lr.physics_method_from_decision(path) == ("ridge", str(path))
    with pytest.raises(RuntimeError, match="does not exist"):
        lr.physics_method_from_decision(tmp_path / "missing.json")
    # dev step 3 refuses without step 2's decision instead of silently using ridge
    with pytest.raises(RuntimeError, match="does not exist"):
        lr.run_dev(_args(step=3), CONFIG, geometry=GEOM, gen_config=GEN, arms_config=ARMS,
                   data_root=tmp_path / "data", out_root=tmp_path / "out", config_sha="cfg",
                   code_sha="code", status_text="", device=CPU, dry_run=False, log=lambda *_: None)


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
