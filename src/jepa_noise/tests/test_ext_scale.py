# SPDX-License-Identifier: MIT
"""Acceptance tests for the scale-control extension (runbook 2 Nov 2026 §3.3).

Covers: the full-line authorisation gate (not the "我授权" substring); the
``dev_train_300k`` cohort spec/key/manifest validation; ``load_run_config`` refusing
a non-standard ``training.steps`` while ``run_one(steps=30000)`` is accepted; the
``eh.evaluate_run`` / ``lr.evaluate_arm_run`` equivalence on a deployed run
checkpoint (skipped where the scale checkpoints are absent); run-directory refusal +
``resumed_from_step == 0``; the three-valued ``scale_r``; the preflight/entries
failure paths; and a finite synthetic training step.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_scale as es  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_scale", ROOT / "scripts" / "jepa_ext_scale.py")
scale_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scale_cli)


# ---------------------------------------------------------------- gate (1)
def _fake_root(tmp_path, *, runbook_line=None, decisions_line=None, with_row=True):
    root = tmp_path / "root"
    (root / "docs/jepa").mkdir(parents=True)
    line = es.AUTHOR_LINE if runbook_line is None else runbook_line
    row = es.B4A_ROW if with_row else "no row"
    tail = es.B4A_ROW_TAIL if with_row else "no tail"
    (root / es.RUNBOOK_REL).write_text(
        "## 0. Authority\n\n> \"" + line + "\"\n\n" + row + "\n> " + tail +
        "\n\n## 1. Why\n")
    (root / es.DECISIONS_REL).write_text(
        "history\n" + (es.AUTHOR_LINE if decisions_line is None else decisions_line) + "\n")
    return root


def test_gate_requires_the_full_line_and_the_b4a_row(tmp_path):
    assert es.assert_authorised(_fake_root(tmp_path))["authorised"] is True
    with pytest.raises(SystemExit) as caught:
        es.assert_authorised(_fake_root(tmp_path / "a", runbook_line="我授权"))
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        es.assert_authorised(_fake_root(tmp_path / "b", with_row=False))
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        es.assert_authorised(_fake_root(tmp_path / "c", decisions_line="我授权"))
    assert caught.value.code == 2
    # the committed repository passes the gate
    assert es.assert_authorised(ROOT)["authorised"] is True


# --------------------------------------------------- cohort spec/key/manifest (2)
def _cohort_manifest(**overrides):
    runs = [{"event_id_start": es.COHORT["event_start"] + i * es.COHORT["run_length"],
             "run_id": f"run/{i:04d}", "run_dir": f"r{i}", "n_events": es.COHORT["run_length"]}
            for i in range(es.COHORT["n_events"] // es.COHORT["run_length"])]
    payload = {"key": es.expected_key(), "n_events": es.COHORT["n_events"],
               "n_runs": len(runs), "run_length": es.COHORT["run_length"],
               "registered": False, "law": es.LAW, "setting": es.SETTING,
               "runs": runs}
    payload.update(overrides)
    return payload


def test_cohort_spec_key_and_manifest_validation():
    assert es.cohort_spec() == es.COHORT
    assert es.expected_key() == "dev_train_300k__F1__0.05__s20261017__n300000x1000"
    valid = es.validate_cohort_manifest(_cohort_manifest())
    assert valid["pass"] is True
    assert valid["n_events"] == 300000 and valid["n_runs"] == 300
    assert valid["namespace"] == "lucid/dev_train_300k"
    assert valid["event_start"] == 1300000
    for broken in (_cohort_manifest(n_events=100000),
                   _cohort_manifest(registered=True),
                   _cohort_manifest(run_length=50),
                   _cohort_manifest(runs=[])):
        assert es.validate_cohort_manifest(broken)["pass"] is False
    wrong_start = _cohort_manifest()
    wrong_start["runs"][0]["event_id_start"] = 1000000
    assert es.validate_cohort_manifest(wrong_start)["pass"] is False


def test_cohort_usage_counts_entries_and_bytes(tmp_path):
    directory = tmp_path / "cohort"
    (directory / "runs" / "r0").mkdir(parents=True)
    (directory / "manifest.json").write_text("{}")
    (directory / "runs" / "r0" / "hits.npz").write_bytes(b"12345")
    usage = es.cohort_usage(directory)
    assert usage["entries"] == 4 and usage["bytes"] == 7


# --------------------------------------- config steps vs run_one override (3)
def test_load_run_config_rejects_30000_steps_but_run_one_accepts_the_override(
        monkeypatch, tmp_path):
    config = json.loads((ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json")
                        .read_text())
    assert int(config["training"]["steps"]) == es.CONFIG_STEPS
    bad = tmp_path / "config_bad.json"
    config["training"]["steps"] = 30000
    bad.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        lr.load_run_config(bad)
    # the registered run_one receives the explicit 30 000-step override
    captured = {}
    run_dir = tmp_path / "run"

    def fake_prepare(out_root, seed, steps=None, device=None):
        run_dir.mkdir(parents=True, exist_ok=True)
        return es.run_id_for(seed), run_dir

    def fake_run_one(**kwargs):
        captured.update(kwargs)
        (run_dir / "run_status.json").write_text(json.dumps(
            {"outcome": "complete", "steps_completed": es.STEPS,
             "resumed_from_step": None}))
        (run_dir / "train_log.jsonl").write_text(json.dumps(
            {"step": es.STEPS, "metrics": {"loss": 1.0}}) + "\n")
        (run_dir / "final.pt").write_bytes(b"checkpoint")
        return {"run_dir": run_dir, "manifest": {}}

    monkeypatch.setattr(es, "assert_authorised", lambda root=None: {"authorised": True})
    import types
    monkeypatch.setattr(es, "load_cohort_and_probes",
                        lambda *a, **k: ({"key": "k", "manifest_sha256": "a" * 64,
                                          "validation": {"pass": True}},
                                         types.SimpleNamespace(runs=[{}] * 300), {},
                                         {"preflight": {"pass": True}}))
    monkeypatch.setattr(es, "prepare_run_dir", fake_prepare)
    monkeypatch.setattr(lr, "run_one", fake_run_one)
    real_config = lr.load_run_config(
        ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json")
    result = es.run_one_training(seed=9000, out_root=tmp_path / "train",
                                 config=real_config, gen=None, arms_config=None,
                                 geometry=None, device="cpu", data_root=str(tmp_path),
                                 config_path=bad, code_sha="0" * 40, status_text="",
                                 steps=es.STEPS, log=lambda *a, **k: None)
    assert captured["steps"] == es.STEPS
    assert result["summary"]["resumed_from_step"] == 0
    assert result["summary"]["metrics_finite"] is True
    with pytest.raises(ValueError):
        es.run_one_training(seed=9000, out_root=tmp_path / "train2",
                            config=real_config, gen=None, arms_config=None,
                            geometry=None, device="cpu",
                            data_root=str(tmp_path), config_path=bad, code_sha="0" * 40,
                            status_text="", steps=12345, log=lambda *a, **k: None)


# --------------------------- evaluate_run == evaluate_arm_run (4, host only)
def _scale_run_finals():
    return sorted((es.EXT_ROOT / "train" / "scale300k" / "runs").glob("*/final.pt"))


@pytest.mark.skipif(not _scale_run_finals(), reason="scale run checkpoints are not on this host")
def test_evaluate_run_equals_registered_evaluate_arm_run():
    from jepa_noise import ext_readout as er
    config = lr.load_run_config(ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json")
    gen = lb.load_generator_config(
        str(ROOT / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"))
    arms_config = la.load_config()
    geometry = lb.load_wcte_geometry(
        "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json")
    probes = eh.load_probes(config, gen, geometry, es.DATA_ROOT_DEFAULT)
    run_dir = _scale_run_finals()[0].parent
    seed = int(json.loads((run_dir / "variant.json").read_text())["seed"])
    item = {"label": f"scale300k_{seed}", "arm": es.ARM_NAME, "split": "within",
            "seed": seed, "beta": None, "source": str(run_dir / "final.pt")}
    via_item = er.evaluate_item(item, probes, geometry, config, arms_config,
                                device="cpu", max_events=200)
    arm = er.item_arm(item, config, arms_config, geometry, device="cpu")
    via_arm = lr.evaluate_arm_run(arm, arm_name=es.ARM_NAME, cell=lr.dev_cell(config),
                                  geometry=geometry, probe_datasets=probes,
                                  config=config, device=torch.device("cpu"),
                                  mask_seed=int(config["views"]["mask_seed"]),
                                  max_events=200, save_embeddings=False)
    assert via_item["forms"]["full_input"]["vertex_resolution_m"] == pytest.approx(
        via_arm["physics"]["resolution"], abs=1e-6)
    assert via_item["forms"]["full_input"]["effective_rank"] == pytest.approx(
        via_item["forms"]["full_input"]["effective_rank"])


# -------------------------------------------------- run dir / resume (5)
def test_run_dir_refusal_and_fresh_resume(tmp_path):
    run_id, run_dir = es.prepare_run_dir(tmp_path / "train", 9000, steps=es.STEPS,
                                         device="cpu")
    assert run_id == "dev_s005_within__sigreg_jepa__s9000"
    sidecar = json.loads((run_dir / "variant.json").read_text())
    assert sidecar["variant"] == "scale300k" and sidecar["cohort"] == es.expected_key()
    with pytest.raises(RuntimeError, match="already exists"):
        es.prepare_run_dir(tmp_path / "train", 9000)
    from jepa_noise import ext_h1clean as ec
    assert ec.assert_fresh_resume({"resumed_from_step": 0}) == 0
    assert ec.assert_fresh_resume({"resumed_from_step": None}) == 0
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({"resumed_from_step": 1234})


# ------------------------------------------------------ scale_r verdicts (6)
def _run_row(full_v, full_r2, half_v, half_r2):
    return {"full_input": {"vertex_resolution_m": full_v,
                           "token_log_charge_r2": {"r2": full_r2}},
            "half_pass": {"vertex_resolution_m": half_v,
                          "token_log_charge_r2": {"r2": half_r2}}}


def test_scale_r_three_valued():
    recovered = es.scale_r({"9000": _run_row(0.14, 0.9, 0.13, 0.9),
                            "9001": _run_row(0.15, 0.9, 0.14, 0.9)})
    assert recovered["verdict"] == "recovered"
    not_recovered = es.scale_r({"9000": _run_row(0.22, 0.02, 0.20, 0.1),
                                "9001": _run_row(0.24, 0.03, 0.21, 0.1)})
    assert not_recovered["verdict"] == "not recovered"
    charge_fail = es.scale_r({"9000": _run_row(0.14, 0.2, 0.13, 0.9),
                              "9001": _run_row(0.15, 0.2, 0.14, 0.9)})
    assert charge_fail["verdict"] == "not recovered"
    disagreement = es.scale_r({"9000": _run_row(0.14, 0.9, 0.13, 0.9),
                               "9001": _run_row(0.22, 0.9, 0.13, 0.9)})
    assert disagreement["verdict"] == "unclear"


# ------------------------------- preflight / entries failure paths (7)
def test_entries_projection_and_preflight_failure_paths(tmp_path):
    assert es.entries_projection(40959)["pass"] is True
    assert es.entries_projection(40959)["projected_after"] == 40959 - (
        es.ENTRIES_COHORT + es.ENTRIES_CLONE + es.ENTRIES_SUITE + es.ENTRIES_EXT)
    assert es.entries_projection(35000)["pass"] is False
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"level": "OK", "caps": {"free_entries": 40000}}))
    live = es.live_free_entries(guard)
    assert live["free_entries"] == 40000 and live["guard_level"] == "OK"
    with pytest.raises(RuntimeError):
        es.live_free_entries(tmp_path / "missing.json")
    empty = tmp_path / "data"
    (empty / "lucid").mkdir(parents=True)
    status = es.probe_sets_status(empty)
    assert status["pass"] is False and status["regeneration"] == "forbidden"
    with pytest.raises(SystemExit):
        es._require_cohort(empty)


# ------------------------------------------------------ smoke training finite (8)
def test_smoke_level_training_is_finite():
    geometry = lb.from_synthetic({"module_ids": tuple(range(6)),
                                  "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                                  "pmts_per_module": 19})
    settings = la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))
    arm = la.LucidArm("sigreg_jepa", geometry, settings=settings, seed=9000,
                      encoder_config=pe.EncoderConfig(window=16, shift=8))
    rng = np.random.default_rng(0)
    n_pmt = geometry.n_pmt
    features = rng.normal(size=(2, n_pmt, pe.N_FEATURES)).astype(np.float32)
    within = eh.er.within_target_mask(geometry) if hasattr(eh, "er") else None
    from jepa_noise import ext_readout as er
    within = er.within_target_mask(geometry)
    hidden = np.zeros((2, n_pmt), bool)
    hidden[:, within] = True
    hit = np.zeros((2, n_pmt), bool)
    hit[:, 0::3] = True
    batch = pe.LucidBatch(
        features=torch.as_tensor(features), target=torch.as_tensor(hidden),
        hidden=torch.as_tensor(hidden),
        context_index=torch.as_tensor(np.flatnonzero(~within)).unsqueeze(0).expand(2, -1),
        target_index=torch.as_tensor(np.flatnonzero(within)).unsqueeze(0).expand(2, -1),
        hit_flags=hit, log_charge=features[..., pe.LOG_CHARGE_INDEX].copy(),
        event_ids=np.arange(2), keys=tuple({} for _ in range(2)), report={},
        scaling=pe.FeatureScaling.from_geometry(geometry), split="within",
        occupancy="natural")
    optimizer = torch.optim.Adam(arm.parameters(), lr=1e-3)
    out = arm._forward_sigreg(batch)
    assert np.isfinite(float(out["loss"]))
    out["loss"].backward()
    optimizer.step()
    metrics = arm._forward_sigreg(batch)["metrics"]
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert es.STEPS == 30000 and es.CONFIG_STEPS == 10000


def test_loader_cache_override_covers_the_300_run_cohort():
    config = lr.load_run_config(
        ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json")
    assert int(config["data"]["loader_cache_runs"]) == 128
    overridden = es.loader_config(config, 300)
    assert int(overridden["data"]["loader_cache_runs"]) == 300
    assert int(config["data"]["loader_cache_runs"]) == 128        # frozen config untouched
    assert overridden["training"] == config["training"]
    assert es.loader_config(config, 100) is config                 # small cohorts unaffected
