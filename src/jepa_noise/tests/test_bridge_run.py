"""WP4a bridge execution-driver tests (runbook v4 §3.5.7, §3.5.8, §1.3).

CPU only, tiny.  Four 20-step/batch-4 runs (one per arm) exercise the whole
pipeline (data cache, training, checkpoint, embeddings, probes, manifest,
status); the remaining tests cover resume determinism, the ridge protocol, the
beta rule, the seed floor and B1/R7 verdict logic, the cohort ID table, CRLB,
and the write footprint.  Total budget <= 60 s.
"""

import os

# Runbook v3 §3.1: no implicit compilation for CPU checks; must precede any
# reconstruction_model import.
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_noise import bridge_run
from jepa_noise.crlb import crlb as reference_crlb
from jepa_noise.jepa import ALL_ARMS, CONFIG_PATH, load_config
from jepa_noise.probes import ALPHAS

CONFIG = load_config()
CONFIG_SHA = bridge_run.sha256_file(CONFIG_PATH)
MANIFEST_KEYS = {
    "run_id",
    "arm",
    "seed",
    "cohort",
    "config_sha256",
    "code_sha",
    "git_status",
    "host",
    "device",
    "started_at_utc",
    "finished_at_utc",
    "steps",
    "wall_seconds",
}


def grid_args(tmp_path, name, *, steps=20, batch=4, max_events=20):
    return [
        "grid",
        "--out",
        str(tmp_path / name / "out"),
        "--data-root",
        str(tmp_path / name / "data"),
        "--fixture",
        "synthetic",
        "--device",
        "cpu",
        "--steps",
        str(steps),
        "--batch",
        str(batch),
        "--max-events",
        str(max_events),
    ]


def write_decision(tmp_path, beta=1.0):
    grid = [float(value) for value in
            CONFIG["objective"]["anchor"]["beta_grid"]]
    table = [
        {
            "beta": value,
            "score": 0.5 + 0.1 * index,
            "physics_mse": {par: 1.0 for par in bridge_run.PHYSICS_TARGETS},
            "probe_train_variance": {
                par: 1.0 for par in bridge_run.PHYSICS_TARGETS
            },
        }
        for index, value in enumerate(grid)
    ]
    path = tmp_path / "beta_decision.json"
    bridge_run.write_beta_decision(
        path,
        selected_beta=beta,
        table=table,
        config=CONFIG,
        config_sha=CONFIG_SHA,
        code_sha="test",
        cohort_prov={"name": "development"},
        seed=100,
        steps=10000,
        fixture="synthetic",
        registered=False,
    )
    return path


# ---------------------------------------------------------------------------
# 1. each arm completes a 20-step/batch-4 run and writes manifest + status
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", list(ALL_ARMS))
def test_each_arm_tiny_run_writes_manifest_and_status(tmp_path, arm):
    args = grid_args(tmp_path, arm) + ["--arms", arm, "--seeds", "0"]
    if arm == "sigreg_jepa_anchored":
        args += ["--beta-decision", str(write_decision(tmp_path, beta=1.0))]
    assert bridge_run.main(args) == 0
    out = tmp_path / arm / "out"
    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert MANIFEST_KEYS <= set(row)
    assert row["run_id"] == f"wp4a_eval_{arm}_s0"
    assert row["arm"] == arm and row["seed"] == 0
    assert row["cohort"] == "evaluation"
    assert row["steps_completed"] == 20
    assert row["steps"] == 20
    assert row["outcome"] == "complete"
    assert row["registered"] is False  # smoke fixture, truncated cohort
    provenance = row["cohort_provenance"]
    assert provenance["roles"]["subject_training"]["ids"] == [0, 199999]
    assert provenance["roles"]["probe_test"]["ids"] == [210000, 219999]
    assert provenance["truncated"] is True
    assert provenance["views"]["law"] == "v4-A4-waveform-v1"
    assert provenance["views"]["theta_fixture"] == "v4-a4-e1b"
    assert provenance["views"]["base_rho"] == 0.0
    assert "split_hash" in provenance["views"]
    assert "noise_covariance_sha256" in provenance["views"]
    assert row["beta"] == (1.0 if arm == "sigreg_jepa_anchored" else None)
    run_dir = out / "runs" / row["run_id"]
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["outcome"] == "complete"
    assert status["exit_status"] == 0
    assert status["steps_completed"] == 20
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "SHA256SUMS").exists()
    assert (run_dir / "probes.json").exists()
    probes = json.loads((run_dir / "probes.json").read_text())
    assert probes["probes"]["physics"]["targets"] == list(
        bridge_run.PHYSICS_TARGETS
    )
    assert probes["probes"]["nuisance"]["undefined"]["line_I"] == (
        "line_target_undefined_at_line_zero"
    )
    assert "random_init_floor" in probes
    assert (out / "aggregate.json").exists()
    assert (out / "PREDICTIONS_UPDATE.md").exists()


# ---------------------------------------------------------------------------
# 2. checkpoint/resume gives identical final parameters
# ---------------------------------------------------------------------------
def test_checkpoint_resume_is_bitwise_identical(tmp_path):
    base = dict(steps=20, batch=4, max_events=12)
    straight = grid_args(tmp_path, "straight", **base) + [
        "--arms", "sigreg_jepa", "--seeds", "0", "--no-evaluate",
    ]
    assert bridge_run.main(straight) == 0
    partial = grid_args(tmp_path, "partial", **base) + [
        "--arms", "sigreg_jepa", "--seeds", "0", "--no-evaluate",
        "--stop-after", "10",
    ]
    assert bridge_run.main(partial) == 0
    run_dir = tmp_path / "partial" / "out" / "runs" / "wp4a_eval_sigreg_jepa_s0"
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["outcome"] == "interrupted"
    assert status["steps_completed"] == 10
    resumed = grid_args(tmp_path, "partial", **base) + [
        "--arms", "sigreg_jepa", "--seeds", "0", "--no-evaluate",
        "--resume", str(run_dir / "checkpoint.pt"),
    ]
    assert bridge_run.main(resumed) == 0
    final = json.loads((run_dir / "run_status.json").read_text())
    assert final["outcome"] == "complete"
    first = torch.load(
        tmp_path / "straight" / "out" / "runs" / "wp4a_eval_sigreg_jepa_s0"
        / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    second = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                        weights_only=False)
    assert first["step"] == second["step"] == 20
    assert set(first["model"]) == set(second["model"])
    for name in first["model"]:
        assert torch.equal(first["model"][name], second["model"][name]), name


# ---------------------------------------------------------------------------
# 3. ridge protocol: linear recovery and a constant target
# ---------------------------------------------------------------------------
def test_probe_fitting_recovers_linear_target_and_handles_constant():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(200, 6))
    x_test = rng.normal(size=(80, 6))
    weights = np.asarray([0.5, -1.0, 2.0, 0.0, 0.25, -0.75])
    y_train = x_train @ weights + 0.3
    y_test = x_test @ weights + 0.3
    result = bridge_run.fit_ridge_probes(
        x_train, y_train[:, None], x_test, y_test[:, None]
    )
    row = result["rows"][0]
    assert row["status"] == "evaluated"
    assert row["test_r2"] > 0.99
    assert row["alpha"] in ALPHAS
    assert set(row["validation_mse"]) == {str(alpha) for alpha in ALPHAS}
    constant = bridge_run.fit_ridge_probes(
        x_train, np.full((200, 1), 3.0), x_test, np.full((80, 1), 3.0)
    )
    assert constant["rows"][0]["status"] == "not_evaluable"
    assert constant["rows"][0]["reason"] == "constant_probe_train_target"


def test_crlb_helper_matches_reference():
    rng = np.random.default_rng(11)
    n_obs = 64
    noise = np.diag(rng.uniform(0.5, 2.0, n_obs))
    jacobian = rng.normal(size=(n_obs, 3))
    variances = bridge_run.crlb_variances(noise, [jacobian])
    reference = np.diag(reference_crlb(jacobian, noise))
    assert np.allclose(variances[0], reference, rtol=1e-8, atol=0)


# ---------------------------------------------------------------------------
# 4. beta rule
# ---------------------------------------------------------------------------
def test_beta_rule_picks_the_minimum_and_breaks_ties_to_smaller():
    table = [
        {"beta": 0.1, "score": 0.50},
        {"beta": 1.0, "score": 0.40},
        {"beta": 10.0, "score": 0.60},
    ]
    selection = bridge_run.select_beta(table)
    assert selection["selected_beta"] == 1.0
    assert selection["tied_betas"] == [1.0]
    tie_table = [
        {"beta": 0.1, "score": 0.50},
        {"beta": 1.0, "score": 0.50 * (1.0 + 1e-9)},
        {"beta": 10.0, "score": 0.60},
    ]
    tie = bridge_run.select_beta(tie_table)
    assert tie["selected_beta"] == 0.1
    assert tie["tied_betas"] == [0.1, 1.0]


def test_beta_decision_roundtrip_and_grid_validation(tmp_path):
    path = write_decision(tmp_path, beta=1.0)
    loaded = bridge_run.load_beta_decision(path, CONFIG, CONFIG_SHA)
    assert loaded["selected_beta"] == 1.0
    with pytest.raises(RuntimeError):
        bridge_run.load_beta_decision(path, CONFIG, "0" * 64)
    with pytest.raises(ValueError):
        bridge_run.write_beta_decision(
            tmp_path / "bad.json",
            selected_beta=2.0,
            table=[],
            config=CONFIG,
            config_sha=CONFIG_SHA,
            code_sha="test",
            cohort_prov={"name": "development"},
            seed=100,
            steps=10000,
            fixture="synthetic",
            registered=False,
        )


def test_anchored_grid_refuses_without_a_beta_decision(tmp_path):
    args = grid_args(tmp_path, "refuse", steps=1, max_events=8) + [
        "--arms", "sigreg_jepa_anchored", "--seeds", "0",
    ]
    with pytest.raises(SystemExit) as excinfo:
        bridge_run.main(args)
    assert excinfo.value.code == 2
    assert not (tmp_path / "refuse" / "out" / "manifest.jsonl").exists()


# ---------------------------------------------------------------------------
# 5. seed floor and B1/R7 bridge verdicts
# ---------------------------------------------------------------------------
def _sds(*values):
    return [float(np.std(value, ddof=1)) for value in values]


def test_seed_floor_is_the_largest_of_the_three_terms():
    floor = bridge_run.seed_floor(0.01, 0.02, [0.01, 0.02, 0.03, 0.005])
    assert floor == pytest.approx(0.02)
    # the median can dominate when both arm SDs are small
    floor = bridge_run.seed_floor(0.001, 0.002, [0.01, 0.002, 0.5, 0.6])
    assert floor == pytest.approx(0.255)


def test_b1_verdict_logic():
    plain = {0: 0.50, 1: 0.52, 2: 0.48}
    anchored = {0: 0.30, 1: 0.31, 2: 0.29}
    all_sds = _sds(list(plain.values()), list(anchored.values()))
    passing = bridge_run.b1_verdict(plain, anchored, all_sds)
    assert passing["status"] == "pass"
    assert passing["all_seeds_plain_greater"] is True
    assert passing["exceeds_two_floors"] is True
    assert passing["seed_floor"] == pytest.approx(max(
        all_sds[0], all_sds[1], float(np.median(all_sds))
    ))
    # one seed where the anchored arm wins
    broken = {0: 0.30, 1: 0.31, 2: 0.55}
    assert bridge_run.b1_verdict(plain, broken, all_sds)["status"] == "fail"
    # a difference inside the seed floor
    inside = {seed: value - 0.001 for seed, value in plain.items()}
    verdict = bridge_run.b1_verdict(plain, inside, all_sds)
    assert verdict["status"] == "fail"
    assert verdict["reason"] == "delta_within_two_seed_floors"
    empty = bridge_run.b1_verdict({}, {}, all_sds)
    assert empty["status"] == "not_evaluable"


def test_r7_bridge_verdict_logic():
    plain = {0: 0.50, 1: 0.52, 2: 0.48}
    anchored = {0: 0.30, 1: 0.31, 2: 0.29}
    plain_physics = {par: [1.0, 1.1, 0.9] for par in bridge_run.PHYSICS_TARGETS}
    good = {
        par: [value * 1.05 for value in values]
        for par, values in plain_physics.items()
    }
    passing = bridge_run.r7_bridge_verdict(plain, anchored, plain_physics, good)
    assert passing["status"] == "pass"
    assert passing["all_seeds_anchored_lower"] is True
    assert passing["physics_all_within_tolerance"] is True
    for par in bridge_run.PHYSICS_TARGETS:
        assert passing["physics"][par]["ratio"] == pytest.approx(1.05)
    bad_physics = {
        par: [value * 1.2 for value in values]
        for par, values in plain_physics.items()
    }
    failed = bridge_run.r7_bridge_verdict(plain, anchored, plain_physics,
                                          bad_physics)
    assert failed["status"] == "fail"
    assert failed["reason"] == "physics_tolerance_exceeded"
    broken = {0: 0.30, 1: 0.31, 2: 0.55}
    failed_nuisance = bridge_run.r7_bridge_verdict(
        plain, broken, plain_physics, good
    )
    assert failed_nuisance["status"] == "fail"
    assert failed_nuisance["reason"] == "not_every_seed"
    missing = bridge_run.r7_bridge_verdict(
        {}, {}, {}, {}
    )
    assert missing["status"] == "not_evaluable"


# ---------------------------------------------------------------------------
# 6. cohort ID table
# ---------------------------------------------------------------------------
def test_cohort_id_ranges_are_registered_and_disjoint():
    assert bridge_run.COHORT_IDS["evaluation"] == {
        "subject_training": (0, 199999),
        "probe_training": (200000, 209999),
        "probe_test": (210000, 219999),
    }
    assert bridge_run.COHORT_IDS["development"] == {
        "subject_training": (1000000, 1199999),
        "probe_training": (1200000, 1209999),
        "probe_validation": (1210000, 1219999),
    }
    ranges = []
    for cohort, roles in bridge_run.COHORT_IDS.items():
        for role, (start, stop) in roles.items():
            assert start <= stop
            ranges.append((cohort, role, start, stop))
    for index, (cohort_a, role_a, start_a, stop_a) in enumerate(ranges):
        for cohort_b, role_b, start_b, stop_b in ranges[index + 1:]:
            assert stop_a < start_b or stop_b < start_a, (
                f"{cohort_a}/{role_a} overlaps {cohort_b}/{role_b}"
            )
    # The M development calibration namespace stays clear of the bridge.
    assert max(stop for _, _, _, stop in ranges) < 1300000
    specs = bridge_run.build_cohort_specs(CONFIG)
    assert specs["evaluation"].roles == bridge_run.COHORT_IDS["evaluation"]
    assert specs["development"].roles == bridge_run.COHORT_IDS["development"]
    assert specs["evaluation"].views_constructor["seed"] == 0
    assert specs["development"].views_constructor["seed"] == 100
    assert specs["development"].views_constructor["cell"] == "development"


# ---------------------------------------------------------------------------
# 7. write footprint
# ---------------------------------------------------------------------------
def test_no_writes_outside_out_and_data_root(tmp_path):
    sandbox = tmp_path / "sandbox"
    out = sandbox / "out"
    data = sandbox / "data"
    args = [
        "grid",
        "--out", str(out),
        "--data-root", str(data),
        "--fixture", "synthetic",
        "--device", "cpu",
        "--steps", "2",
        "--batch", "4",
        "--max-events", "8",
        "--arms", "sigreg_jepa",
        "--seeds", "0",
        "--no-evaluate",
    ]
    assert bridge_run.main(args) == 0
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written
    for path in written:
        assert sandbox in path.parents, f"{path} is outside the sandbox"
    assert all(path.relative_to(sandbox).parts[0] in {"out", "data"}
               for path in written)
    assert not (tmp_path / "outside").exists()


# ---------------------------------------------------------------------------
# aggregate-only mode: read-only over run directories, never re-runs
# ---------------------------------------------------------------------------
def _run_file_hashes(out_root):
    import hashlib
    hashes = {}
    for path in sorted((Path(out_root) / "runs").rglob("*")):
        if path.is_file():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _tiny_two_arm_tree(tmp_path):
    args = grid_args(tmp_path, "agg") + ["--arms", "sigreg_jepa,mae_chi2",
                                          "--seeds", "0"]
    assert bridge_run.main(args) == 0
    return Path(args[2])


def test_aggregate_is_read_only_and_reports_missing_arm(tmp_path):
    out_root = _tiny_two_arm_tree(tmp_path)
    before = _run_file_hashes(out_root)
    dest = tmp_path / "combined"
    rc = bridge_run.main([
        "aggregate", "--out", str(dest), "--runs-root", str(out_root),
        "--config", str(CONFIG_PATH), "--data-root", str(tmp_path / "agg" / "data"),
        "--fixture", "synthetic", "--allow-partial",
    ])
    assert rc == 0
    assert before == _run_file_hashes(out_root), "aggregate must not write run dirs"
    for name in ("aggregate.json", "PREDICTIONS_UPDATE.md", "run_status.json",
                 "manifest.jsonl"):
        assert (dest / name).exists()
    payload = json.loads((dest / "aggregate.json").read_text())
    assert payload["aggregation_mode"].startswith("aggregate-only")
    b1 = payload["verdicts"]["B1"]
    assert b1["status"] == "not_evaluable"
    assert b1["reason"] == "missing_required_arm:sigreg_jepa_anchored"
    assert payload["status_anomalies"] == []


def test_aggregate_records_a_damaged_run_status_as_anomaly(tmp_path):
    out_root = _tiny_two_arm_tree(tmp_path)
    status_path = (out_root / "runs" / "wp4a_eval_sigreg_jepa_s0"
                   / "run_status.json")
    status = json.loads(status_path.read_text())
    status["outcome"] = "interrupted"
    status_path.write_text(json.dumps(status))
    dest = tmp_path / "combined"
    rc = bridge_run.main([
        "aggregate", "--out", str(dest), "--runs-root", str(out_root),
        "--config", str(CONFIG_PATH), "--data-root", str(tmp_path / "agg" / "data"),
        "--fixture", "synthetic", "--allow-partial",
    ])
    assert rc == 0
    payload = json.loads((dest / "aggregate.json").read_text())
    assert [entry["run_id"] for entry in payload["status_anomalies"]] == [
        "wp4a_eval_sigreg_jepa_s0"
    ]
    assert payload["status_anomalies"][0]["manifest_outcome"] == "complete"


def test_aggregate_refuses_an_incomplete_registered_set(tmp_path):
    out_root = _tiny_two_arm_tree(tmp_path)
    rc = bridge_run.main([
        "aggregate", "--out", str(tmp_path / "combined"),
        "--runs-root", str(out_root), "--config", str(CONFIG_PATH),
        "--data-root", str(tmp_path / "agg" / "data"),
        "--fixture", "synthetic",
    ])
    assert rc == 2
