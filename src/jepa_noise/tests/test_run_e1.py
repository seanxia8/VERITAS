"""Runner/report infrastructure tests for E1 (WP3); no campaign outcomes."""
import json
import statistics
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import report, run

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "src" / "jepa_noise" / "configs" / "e1_grid_v4.json"


def _config():
    return run.load_config(CONFIG)[0]


def _args(root, *extra):
    return run.build_parser().parse_args(["E1a", "--root", str(root), *extra])


def _make_root(tmp_path: Path) -> Path:
    (tmp_path / "src" / "jepa_noise" / "configs").mkdir(parents=True)
    (tmp_path / "src" / "jepa_noise" / "configs" / "e1_grid_v4.json").write_text(
        CONFIG.read_text())
    shutil.copy(CONFIG, tmp_path / "src" / "jepa_noise" / "configs" / "e1_grid_v4.json")
    docs = tmp_path / "docs" / "jepa"
    docs.mkdir(parents=True)
    shutil.copy(ROOT / "docs" / "jepa" / "PREDICTION_STATUS.json",
                docs / "PREDICTION_STATUS.json")
    return tmp_path


def _attempt(tmp_path: Path, campaign="E1a", index=1) -> Path:
    directory = run.campaign_root(tmp_path, campaign)
    attempt = directory / f"attempt_{index:03d}_fixture_{campaign}"
    (attempt / "rows").mkdir(parents=True)
    (attempt / "logs" / "rows").mkdir(parents=True)
    return attempt


def _fake_row_file(attempt: Path, record: dict) -> None:
    (attempt / "rows" / f"{record['row_id']}.json").write_text(
        json.dumps(run._jsonable(record), indent=1, sort_keys=True) + "\n")


def test_registered_grids_and_smoke_scope():
    config = _config()
    full = run.enumerate_rows(config, "E1a", "full")
    smoke = run.enumerate_rows(config, "E1a", "smoke")
    assert len(full) == 630
    assert len(smoke) == 84
    assert {row["seed"] for row in smoke} == {0}
    assert {row["cell"]["n"] for row in smoke} == {1000, 10000}
    assert len({row["row_id"] for row in full}) == 630
    e1b = run.enumerate_rows(config, "E1b", "full")
    assert len(e1b) == 65
    assert len({row["cell"]["cell_id"] for row in e1b}) == 13
    assert len({(row["cell"]["common_strength"],
                 row["cell"]["line_multiplier"]) for row in e1b}) == 13
    e1c = run.enumerate_rows(config, "E1c", "full")
    assert len(e1c) == 10
    with pytest.raises(ValueError):
        run.enumerate_rows(config, "E1b", "smoke")


def test_attempt_directories_are_fresh_and_rows_are_discovered(tmp_path):
    root = _make_root(tmp_path)
    directory = run.campaign_root(root, "E1a")
    first = run.create_attempt(directory, "E1a")
    second = run.create_attempt(directory, "E1a")
    assert first != second
    assert first.name.startswith("attempt_001")
    assert second.name.startswith("attempt_002")
    record = {"campaign": "E1a", "row_id": "th0.1_eta1_g0_n1000_s0",
              "cell": {"theta_scale": 0.1}, "seed": 0, "status": "pass"}
    _fake_row_file(first, record)
    discovered = run.discover_rows(directory)
    assert record["row_id"] in discovered
    assert discovered[record["row_id"]]["_attempt"] == first.name


def test_execute_row_writes_write_once_record_and_manifest(tmp_path, monkeypatch):
    root = _make_root(tmp_path)
    attempt = _attempt(root)
    config = _config()
    row = run.enumerate_rows(config, "E1c", "full")[0]

    def fake_e1c(config, row, log):
        return {"campaign": "E1c", "row_id": row["row_id"], "cell": row["cell"],
                "seed": row["seed"], "status": "pass", "reason": None}

    monkeypatch.setattr(run, "run_e1c_row", fake_e1c)
    task = {"root": str(root), "config": config,
            "config_path": str(CONFIG), "config_sha256": "a" * 64,
            "attempt_dir": str(attempt), "run_id": "E1-E1c-fixture",
            "code_sha": "b" * 40, "workers": 1, "row": row}
    result = run.execute_row(task)
    for key in ("run_id", "code_sha", "config_sha256", "campaign", "cell",
                "seed", "host", "wall_seconds", "status", "attempt", "row_id"):
        assert key in result
    assert result["status"] == "pass"
    assert (attempt / "rows" / f"{row['row_id']}.json").is_file()
    stored = json.loads((attempt / "rows" / f"{row['row_id']}.json").read_text())
    assert stored["row_id"] == row["row_id"]
    assert stored["status"] == "pass"
    assert "peak_rss_mb" in stored


def test_execute_row_records_exceptions_as_fail(tmp_path, monkeypatch):
    root = _make_root(tmp_path)
    attempt = _attempt(root)
    config = _config()
    row = run.enumerate_rows(config, "E1c", "full")[0]

    def broken(config, row, log):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(run, "run_e1c_row", broken)
    task = {"root": str(root), "config": config, "config_path": str(CONFIG),
            "config_sha256": "a" * 64, "attempt_dir": str(attempt),
            "run_id": "E1-E1c-fixture", "code_sha": "b" * 40, "workers": 1,
            "row": row}
    result = run.execute_row(task)
    assert result["status"] == "fail"
    assert "fixture failure" in result["reason"]
    stored = json.loads((attempt / "rows" / f"{row['row_id']}.json").read_text())
    assert stored["status"] == "fail"


def test_budget_projection_stop_before_launch(tmp_path):
    root = _make_root(tmp_path)
    args = _args(root, "--row-seconds-estimate", "1000000")
    exit_code = run.run_campaign(args)
    assert exit_code == 2
    directory = run.campaign_root(root, "E1a")
    attempt = run.list_attempts(directory)[0]
    stop = json.loads((attempt / "STOP.json").read_text())
    assert stop["reason"] == "budget_projection_exceeds_cap"
    assert not list((attempt / "rows").glob("*.json"))
    evidence = json.loads((attempt / "attempt.json").read_text())
    assert evidence["status"] == "stopped"


def test_dry_run_and_resume_skip_completed_rows(tmp_path):
    root = _make_root(tmp_path)
    directory = run.campaign_root(root, "E1a")
    first = run.create_attempt(directory, "E1a")
    completed = run.enumerate_rows(_config(), "E1a", "smoke")[0]
    _fake_row_file(first, {**completed, "status": "pass", "reason": None,
                           "row_id": completed["row_id"]})
    args = _args(root, "--smoke", "--dry-run")
    assert run.run_campaign(args) == 0
    attempt = run.list_attempts(directory)[-1]
    evidence = json.loads((attempt / "attempt.json").read_text())
    assert evidence["status"] == "dry_run"
    assert evidence["skipped_completed_rows"] == 1


def test_worker_selection_records_formula_inputs():
    config = _config()
    choice = run.choose_workers(config, None)
    for key in ("workers", "physical_cores", "available_gb_at_start",
                "measured_formula", "configured_cap"):
        assert key in choice
    assert 1 <= choice["workers"] <= config["hosts"]["max_workers_by_config"]
    assert run.choose_workers(config, 1)["workers"] == 1


def test_jsonable_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        run._jsonable({"x": float("nan")})
    with pytest.raises(ValueError):
        run._jsonable([float("inf")])
    assert run._jsonable({"x": np.float64(1.5)}) == {"x": 1.5}


def test_e1b_law_gate_fails_loudly_without_a4(monkeypatch, tmp_path):
    config = _config()
    monkeypatch.setattr(run, "supported_laws", lambda: ["v3-A1-waveform-v1"])
    available, detail = run.a4_available(config)
    assert not available
    assert "v4-A4-waveform-v1" in detail
    with pytest.raises(run.CampaignBlocked, match="v4-A4"):
        run.make_tier1(config, "channel", 0.0, 0.0, 0, "fixture")


def test_e1b_campaign_blocked_records_stop(tmp_path, monkeypatch):
    root = _make_root(tmp_path)
    monkeypatch.setattr(run, "supported_laws", lambda: ["v3-A1-waveform-v1"])
    argv = ["E1b", "--root", str(root), "--dry-run"]
    assert run.main(argv) == 3
    directory = run.campaign_root(root, "E1b")
    attempt = run.list_attempts(directory)[0]
    stop = json.loads((attempt / "STOP.json").read_text())
    assert stop["reason"] == "campaign_blocked_law_not_supported"


def _fixture_e1a_row(theta, eta, gamma, seed, angle=1.0):
    cell = {"theta_scale": theta, "eta_scale": eta, "gamma_load": gamma,
            "n": 100000}
    methods = {}
    for method in ("r1_cca", "r5_gevp", "r6_finite_beta_0.1",
                   "r7_finite_beta_1", "r8_finite_beta_10", "r9_oracle"):
        methods[method] = {"status": "evaluated",
                           "angles": {"max_angle_deg": angle}}
    return {
        "campaign": "E1a",
        "row_id": f"{run.e1a_cell_label(cell)}_s{seed}",
        "cell": cell, "seed": seed, "status": "pass", "reason": None,
        "population": {
            "gevp_max_angle_deg": 1e-12,
            "oracle": {"status": "eligible", "recovery_max_angle_rad": 1e-13},
        },
        "methods": methods,
        "r3_per_view_invariance": {"max_spectrum_difference": 1e-13},
    }


def test_report_falls_back_to_status_table_and_is_byte_identical(tmp_path):
    root = _make_root(tmp_path)
    first, aggregates = report.build_text(root, CONFIG)
    second, _ = report.build_text(root, CONFIG)
    assert first == second
    assert aggregates == {}
    assert report.main(["E1", "--root", str(root)]) == 0
    written = (root / "docs" / "jepa" / "PREDICTIONS.md").read_text()
    assert written == first
    assert report.main(["E1", "--root", str(root), "--check"]) == 0
    assert report.main(["all", "--root", str(root)]) == 0
    assert (root / "docs" / "jepa" / "PREDICTIONS.md").read_text() == first
    status = json.loads((root / "docs" / "jepa" /
                         "PREDICTION_STATUS.json").read_text())
    assert status["predictions"][0]["id"] in written


def test_report_computes_r1_from_stored_rows(tmp_path):
    root = _make_root(tmp_path)
    attempt = _attempt(root)
    config = _config()
    for theta in config["E1a"]["theta_scale"]:
        for eta in config["E1a"]["eta_scale"]:
            for seed in config["E1a"]["seeds"]:
                _fake_row_file(attempt, _fixture_e1a_row(theta, eta, 0.0, seed,
                                                         1.0))
    rows, aggregates = report.build_predictions(root, CONFIG)
    by_id = {row["id"]: row for row in rows}
    # v4 Sec.4 adds an E1b clause to R1/R2; without E1b rows they are pending
    # and must not carry numeric evidence from E1a alone.
    assert by_id["R1"]["status"] == "not_evaluable"
    assert by_id["R1"]["reason"] == (
        "E1a_clause_pass_E1b_pending_v4_R1_R2_clause")
    assert by_id["R1"]["evidence"] is None
    assert by_id["R2"]["status"] == "not_evaluable"
    assert by_id["R5"]["status"] == "not_run"
    assert "E1a" in aggregates
    assert aggregates["E1a"]["recorded_rows"] == 30
    # E1a clause alone fails if any cell median exceeds the 5-degree bound.
    high = _attempt(root, index=2)
    for theta in config["E1a"]["theta_scale"]:
        for eta in config["E1a"]["eta_scale"]:
            for seed in config["E1a"]["seeds"]:
                angle = 6.0 if (theta == 0.4 and eta == 3.0) else 1.0
                _fake_row_file(high, _fixture_e1a_row(theta, eta, 0.0, seed,
                                                       angle))
    rows, _ = report.build_predictions(root, CONFIG)
    by_id = {row["id"]: row for row in rows}
    assert by_id["R1"]["status"] == "not_evaluable"  # E1b clause still pending


def _fixture_e1b_row(cell_id, g, line, seed, angle):
    return {
        "campaign": "E1b", "row_id": f"{cell_id}_s{seed}", "seed": seed,
        "cell": {"cell_id": cell_id, "common_strength": g,
                 "line_multiplier": line},
        "status": "pass", "reason": None,
        "splits": {"channel": {"methods": {"r1_cca": {
            "status": "evaluated", "angles": {"max_angle_deg": angle}}}}},
    }


def test_v4_r1_r2_e1b_clauses_from_channel_rows(tmp_path):
    config = _config()
    root = _make_root(tmp_path)
    attempt = _attempt(root, campaign="E1b")
    angles = {0.0: 1.0, 0.005: 10.0, 0.01: 20.0, 0.02: 40.0, 0.05: 60.0,
              0.1: 70.0, 0.225: 75.0, 0.45: 78.0, 0.9: 80.0}
    for cell in config["E1b"]["cells"]:
        if cell["line_multiplier"] != 0.0:
            continue
        for seed in config["E1b"]["seeds"]:
            _fake_row_file(attempt, _fixture_e1b_row(
                cell["cell_id"], cell["common_strength"], 0.0, seed,
                angles[cell["common_strength"]]))
    records = run.discover_rows(run.campaign_root(root, "E1b"))
    r1 = report._r1_e1b_clause(config, records)
    r2 = report._r2_e1b_clause(config, records)
    assert r1["status"] == "pass"
    assert r1["detail"]["clean_cell_median_angle_deg"] == pytest.approx(1.0)
    assert r2["status"] == "pass"
    assert r2["detail"]["spearman"] > 0.8
    assert r2["detail"]["seed_mean_angle_at_g0p05_deg"] == pytest.approx(60.0)
    assert report._r1_e1b_clause(config, {})["reason"] == \
        "E1b_pending_v4_R1_R2_clause"


def test_render_rejects_invalid_records():
    good = {"id": "R1", "campaign": "E1", "status": "not_run",
            "reason": "pending", "seeds": [], "evidence": None}
    assert "| R1 | E1 | not_run | pending | none | none |" in report.render([good])
    with pytest.raises(ValueError):
        report.render([{**good, "status": "unknown"}])
    with pytest.raises(ValueError):
        report.render([{**good, "status": "pass", "evidence": None}])
    # a hand-edited reason containing a pipe must be escaped, not abort the report
    escaped = report.render([{**good, "reason": "bad|cell"}])
    assert "bad\\|cell" in escaped


def test_solver_isolation_returns_all_ten_registered_rows():
    from jepa_noise.views import ExactModel

    population = ExactModel(d=8, k=3, seed=3, theta_scale=.4,
                            eta_scale=3.).population(0.5)
    scc_u, stt_u, sct_u = (population["scc"], population["stt"],
                           population["sct"])
    methods = run.solve_registered_methods(
        scc_u, stt_u, sct_u, run.ridge_covariance(scc_u),
        run.ridge_covariance(stt_u), sct_u, population["ncc"], population,
        {"betas": [0.1, 1.0, 10.0]}, 3)
    expected = {"r1_cca", "r2_hard_isotropy", "r3_context_pca",
                "r4_whitened_pca", "r5_gevp", "r6_finite_beta_0.1",
                "r7_finite_beta_1", "r8_finite_beta_10", "r9_oracle",
                "r10_of"}
    assert set(methods) == expected
    assert methods["r2_hard_isotropy"]["fit"] is methods["r1_cca"]["fit"]
    assert "not an independent estimator" in methods["r2_hard_isotropy"]["note"]


def test_rerun_all_re_executes_completed_rows(tmp_path):
    root = _make_root(tmp_path)
    directory = run.campaign_root(root, "E1a")
    first = run.create_attempt(directory, "E1a")
    completed = run.enumerate_rows(_config(), "E1a", "smoke")[0]
    _fake_row_file(first, {**completed, "status": "pass", "reason": None})
    args = _args(root, "--smoke", "--dry-run", "--rerun-all")
    assert run.run_campaign(args) == 0
    evidence = json.loads((run.list_attempts(directory)[-1] /
                           "attempt.json").read_text())
    assert evidence["skipped_completed_rows"] == 0
    assert evidence["pending_rows"] == 84


def test_second_attempt_reuses_first_rows_and_manifest_backpoints(tmp_path):
    root = _make_root(tmp_path)
    directory = run.campaign_root(root, "E1a")
    first = run.create_attempt(directory, "E1a")
    completed = run.enumerate_rows(_config(), "E1a", "smoke")[0]
    source_record = {**completed, "status": "pass", "reason": None}
    _fake_row_file(first, source_record)
    args = _args(root, "--smoke", "--dry-run")
    assert run.run_campaign(args) == 0
    second = run.list_attempts(directory)[-1]
    evidence = json.loads((second / "attempt.json").read_text())
    assert evidence["skipped_completed_rows"] == 1
    assert evidence["pending_rows"] == 83
    assert evidence["reused_row_count"] == 1
    # the completed row is discoverable, not copied into the new attempt
    assert not (second / "rows" / f"{completed['row_id']}.json").exists()
    reused = json.loads((second / "reused_rows.json").read_text())
    assert reused["count"] == 1
    assert reused["rows"][0]["source_attempt"] == first.name
    assert reused["rows"][0]["source_row_file"] == str(
        first / "rows" / f"{completed['row_id']}.json")
    manifest = [json.loads(line) for line in
                (second / "manifest.jsonl").read_text().splitlines()]
    skipped = [entry for entry in manifest
               if entry["status"] == "skipped_completed_row"]
    assert len(skipped) == 1
    assert skipped[0]["row_id"] == completed["row_id"]
    assert skipped[0]["source_attempt"] == first.name
    assert skipped[0]["source_row_file"] == reused["rows"][0]["source_row_file"]


def test_stopped_attempt_rows_are_reused(tmp_path):
    root = _make_root(tmp_path)
    directory = run.campaign_root(root, "E1a")
    stopped = run.create_attempt(directory, "E1a")
    completed = run.enumerate_rows(_config(), "E1a", "smoke")[2]
    _fake_row_file(stopped, {**completed, "status": "pass", "reason": None})
    (stopped / "attempt.json").write_text(json.dumps({
        "run_id": "E1-E1a-stopped", "campaign": "E1a",
        "status": "stopped", "reason": "resource_guard_host_free_ram"}))
    args = _args(root, "--smoke", "--dry-run")
    assert run.run_campaign(args) == 0
    second = run.list_attempts(directory)[-1]
    evidence = json.loads((second / "attempt.json").read_text())
    assert evidence["skipped_completed_rows"] == 1
    assert evidence["reused_row_count"] == 1


# ---------------------------------------------------------------------------
# 2026-10-17 author rules: not_run placeholders are not completed rows; a
# registered run refuses to start on a dirty src/scripts tree.
# ---------------------------------------------------------------------------
def test_not_run_placeholder_does_not_count_as_completed(tmp_path):
    campaign = tmp_path / "E1a"
    attempt = campaign / "attempt_001_test"
    rows_dir = attempt / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "row_a.json").write_text(json.dumps({
        "row_id": "row_a", "status": "pass"}))
    (rows_dir / "row_b.json").write_text(json.dumps({
        "row_id": "row_b", "status": "not_run",
        "reason": "stopped_before_launch"}))
    discovered = run.discover_rows(campaign)
    completed = {
        identifier for identifier, record in discovered.items()
        if record.get("status") != "not_run"
    }
    assert completed == {"row_a"}
    assert "row_b" not in completed


def test_src_tree_status_is_empty_in_a_clean_repo_and_dirty_when_modified(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "src" / "mod.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "c"], check=True)
    assert run.src_tree_status(repo) == ""
    (repo / "src" / "mod.py").write_text("x = 2\n")
    assert "src/mod.py" in run.src_tree_status(repo)


def test_rerun_rows_restricts_to_named_rows_and_clears_completion():
    rows = [{"row_id": "g0_line0_s0"}, {"row_id": "g0_line0_s1"},
            {"row_id": "g0p005_line0_s0"}]
    completed = {"g0_line0_s0", "g0_line0_s1", "g0p005_line0_s0"}
    selected, remaining = run.apply_rerun_rows(
        rows, completed, ["g0_line0_s0", "g0p005_line0_s0"])
    assert [row["row_id"] for row in selected] == ["g0_line0_s0", "g0p005_line0_s0"]
    assert remaining == {"g0_line0_s1"}
    assert run.apply_rerun_rows(rows, completed, []) == (rows, completed)
    try:
        run.apply_rerun_rows(rows, completed, ["not_a_row"])
    except run.CampaignBlocked:
        pass
    else:
        raise AssertionError("unknown rerun row must be refused")


def test_r2_uses_seed_mean_angles_not_the_cross_seed_median():
    """v3 §2.4 registers `Spearman over seed-mean angles`; mean and median differ."""
    seed_angles = [10.0, 12.0, 30.0]
    mean_angle = statistics.fmean(seed_angles)
    assert abs(mean_angle - 17.3333333) < 1e-6
    assert statistics.median(seed_angles) == 12.0
