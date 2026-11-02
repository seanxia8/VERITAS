# SPDX-License-Identifier: MIT
"""Acceptance tests for the architecture/masking extension (runbook 2 Nov 2026 §3.3).

Covers: the full-line + H4-row authorisation gate; the ``build_arm`` wrapper
(injection, pass-through, restoration); the wrapper being active during evaluation
(``pred4`` strict-loads, window variants evaluate at their own window vs the
default); ``eh.evaluate_run`` / ``lr.evaluate_arm_run`` equivalence with the wrapper
(host-only); run-directory refusal + resumed_from_step; three-valued per-variant
``arch_r``; dry-run failure paths; and a finite synthetic training step.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_arch as ea  # noqa: E402
from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GEN_PATH = ROOT / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_arch", ROOT / "scripts" / "jepa_ext_arch.py")
arch_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arch_cli)


@pytest.fixture(scope="module")
def geometry():
    return lb.from_synthetic({"module_ids": tuple(range(6)),
                              "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                              "pmts_per_module": 19})


@pytest.fixture(scope="module")
def config():
    return lr.load_run_config(CONFIG_PATH)


def _fake_root(tmp_path, *, runbook_line=None, decisions_line=None, with_row=True):
    root = tmp_path / "root"
    (root / "docs/jepa").mkdir(parents=True)
    line = ea.AUTHOR_LINE if runbook_line is None else runbook_line
    row = ea.H4_ROW if with_row else "no row"
    (root / ea.RUNBOOK_REL).write_text(
        "## 0. Authority\n\n> \"" + line + "\"\n\n" + row + "\n\n## 1. Why\n")
    (root / ea.DECISIONS_REL).write_text(
        "history\n" + (ea.AUTHOR_LINE if decisions_line is None else decisions_line) + "\n")
    return root


# ---------------------------------------------------------------- gate (1)
def test_gate_requires_the_full_line_and_h4_row(tmp_path):
    assert ea.assert_authorised(_fake_root(tmp_path))["authorised"] is True
    with pytest.raises(SystemExit) as caught:
        ea.assert_authorised(_fake_root(tmp_path / "a", runbook_line="我授权"))
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        ea.assert_authorised(_fake_root(tmp_path / "b", with_row=False))
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        ea.assert_authorised(_fake_root(tmp_path / "c", decisions_line="我授权"))
    assert caught.value.code == 2
    assert ea.assert_authorised(ROOT)["authorised"] is True


# ------------------------------------------------------------- wrapper (2)
def test_wrapper_injects_the_override_and_leaves_a_plain_build_unchanged(monkeypatch,
                                                                        geometry,
                                                                        config):
    calls = []

    def fake_build(arm_name, geometry_, arms_config, cell, seed, beta=None,
                   dark_bank=None, encoder_config=None):
        calls.append({"arm": arm_name, "seed": seed, "encoder_config": encoder_config})
        return {"arm": arm_name, "encoder_config": encoder_config}

    monkeypatch.setattr(lr, "build_arm", fake_build)
    cell = lr.dev_cell(config)
    arms_config = la.load_config()
    with ea.build_arm_patch("window32", log=lambda *a, **k: None):
        lr.build_arm("sigreg_jepa", geometry, arms_config, cell, 9000)
        explicit = pe.EncoderConfig(window=16, shift=8)
        lr.build_arm("sigreg_jepa", geometry, arms_config, cell, 9000,
                     encoder_config=explicit)
        assert lr.build_arm is not fake_build
    assert lr.build_arm is fake_build                       # restored
    assert calls[0]["encoder_config"].window == 32
    assert calls[0]["encoder_config"].shift == 16
    assert calls[1]["encoder_config"] is explicit           # pass-through
    # plain build without the wrapper still gets None
    lr.build_arm("sigreg_jepa", geometry, arms_config, cell, 9000)
    assert calls[2]["encoder_config"] is None
    assert list(ea.VARIANTS) == ["window128", "window32", "pred4"]
    assert ea.variant_config("window128").window == 128
    assert ea.variant_config("pred4").n_pred_blocks == 4


# ---------------------------------------- wrapper active during evaluation (3)
def test_pred4_strict_loads_and_window_variants_use_their_own_window(
        tmp_path, geometry, config):
    cell = lr.dev_cell(config)
    arms_config = la.load_config()
    device = torch.device("cpu")
    # window128 arm reports 128/64, the default arm 64/32
    with ea.build_arm_patch("window128", log=lambda *a, **k: None):
        arm128 = lr.build_arm("sigreg_jepa", geometry, arms_config, cell, 9000)
    assert arm128.jepa.config.window == 128 and arm128.jepa.config.shift == 64
    plain = lr.build_arm("sigreg_jepa", geometry, arms_config, cell, 9000)
    assert plain.jepa.config.window == 64 and plain.jepa.config.shift == 32
    # pred4 checkpoint strict-loads only with the wrapper active
    with ea.build_arm_patch("pred4", log=lambda *a, **k: None):
        arm4 = lr.build_arm("sigreg_jepa", geometry, arms_config, cell, 9000)
    assert arm4.jepa.config.n_pred_blocks == 4
    checkpoint = tmp_path / "pred4.pt"
    torch.save({"identity": {"arm": "sigreg_jepa"}, "model": arm4.state_dict()}, checkpoint)
    item = {"label": "pred4_9000", "arm": "sigreg_jepa", "split": "within",
            "seed": 9000, "beta": None, "source": str(checkpoint)}
    from jepa_noise import ext_readout as readout
    with pytest.raises(RuntimeError):
        readout.item_arm(item, config, arms_config, geometry, device=device)
    with ea.build_arm_patch("pred4", log=lambda *a, **k: None):
        loaded = readout.item_arm(item, config, arms_config, geometry, device=device)
    assert loaded.jepa.config.n_pred_blocks == 4
    with ea.build_arm_patch("window128", log=lambda *a, **k: None):
        item128 = dict(item, source=None, label="window128_9000")
        arm_eval = readout.item_arm(item128, config, arms_config, geometry, device=device)
    assert arm_eval.jepa.config.window == 128
    default_arm = readout.item_arm(item128, config, arms_config, geometry, device=device)
    assert default_arm.jepa.config.window == 64
    # encoder signature helper
    assert ea.encoder_signature(ea.variant_config("window32"))["window"] == 32
    assert ea.encoder_signature(None)["window"] == 64


# -------------------- evaluate_run == evaluate_arm_run with wrapper (4, host only)
def _arch_run_finals():
    return sorted((ea.EXT_ROOT / "train" / "arch").glob("*/runs/*/final.pt"))


@pytest.mark.skipif(not _arch_run_finals(), reason="arch run checkpoints are not on this host")
def test_evaluate_run_equals_registered_evaluate_arm_run_with_wrapper():
    from jepa_noise import ext_readout as readout
    run_dir = _arch_run_finals()[0].parent
    sidecar = json.loads((run_dir / "variant.json").read_text())
    variant, seed = sidecar["variant"], int(sidecar["seed"])
    config = lr.load_run_config(CONFIG_PATH)
    gen = lb.load_generator_config(str(GEN_PATH))
    arms_config = la.load_config()
    geometry = lb.load_wcte_geometry(
        "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json")
    probes = eh.load_probes(config, gen, geometry, ea.DATA_ROOT_DEFAULT)
    device = torch.device("cpu")
    item = {"label": f"arch_{variant}_{seed}", "arm": ea.ARM_NAME, "split": "within",
            "seed": seed, "beta": None, "source": str(run_dir / "final.pt")}
    with ea.build_arm_patch(variant, log=lambda *a, **k: None):
        via_item = readout.evaluate_item(item, probes, geometry, config, arms_config,
                                         device=device, max_events=200)
        arm = readout.item_arm(item, config, arms_config, geometry, device=device)
        via_arm = lr.evaluate_arm_run(arm, arm_name=ea.ARM_NAME,
                                      cell=lr.dev_cell(config), geometry=geometry,
                                      probe_datasets=probes, config=config,
                                      device=device,
                                      mask_seed=int(config["views"]["mask_seed"]),
                                      max_events=200, save_embeddings=False)
    assert via_item["forms"]["full_input"]["vertex_resolution_m"] == pytest.approx(
        via_arm["physics"]["resolution"], abs=1e-6)


# ------------------------------------------------- run dir / resume (5)
def test_run_dir_refusal_and_fresh_resume(tmp_path):
    run_id, run_dir = ea.prepare_run_dir(tmp_path / "arch" / "window128", "window128",
                                         9000, steps=ea.STEPS, device="cpu")
    assert run_id == "dev_s005_within__sigreg_jepa__s9000"
    sidecar = json.loads((run_dir / "variant.json").read_text())
    assert sidecar["variant"] == "window128" and sidecar["extension"] == ea.EXTENSION
    with pytest.raises(RuntimeError, match="already exists"):
        ea.prepare_run_dir(tmp_path / "arch" / "window128", "window128", 9000)
    from jepa_noise import ext_h1clean as ec
    assert ec.assert_fresh_resume({"resumed_from_step": None}) == 0
    assert ec.assert_fresh_resume({"resumed_from_step": 0}) == 0
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({"resumed_from_step": 1234})
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({})


# ------------------------------------------------------ arch_r verdicts (6)
def _row(full_v, full_r2, half_v, half_r2):
    return {"full_input": {"vertex_resolution_m": full_v,
                           "token_log_charge_r2": {"r2": full_r2}},
            "half_pass": {"vertex_resolution_m": half_v,
                          "token_log_charge_r2": {"r2": half_r2}}}


def test_arch_r_three_valued_per_variant():
    recovered = ea.arch_r({"9000": _row(0.14, 0.9, 0.13, 0.9),
                           "9001": _row(0.15, 0.9, 0.14, 0.9)})
    assert recovered["verdict"] == "recovered"
    not_recovered = ea.arch_r({"9000": _row(0.22, 0.02, 0.20, 0.1),
                               "9001": _row(0.24, 0.03, 0.21, 0.1)})
    assert not_recovered["verdict"] == "not recovered"
    # full vertex at the floor is not a recovery (strict <)
    at_floor = ea.arch_r({"9000": _row(0.160072, 0.9, 0.13, 0.9),
                          "9001": _row(0.160072, 0.9, 0.13, 0.9)})
    assert at_floor["verdict"] == "not recovered"
    charge_fail = ea.arch_r({"9000": _row(0.14, 0.3, 0.13, 0.9),
                             "9001": _row(0.14, 0.3, 0.13, 0.9)})
    assert charge_fail["verdict"] == "not recovered"
    disagreement = ea.arch_r({"9000": _row(0.14, 0.9, 0.13, 0.9),
                              "9001": _row(0.22, 0.9, 0.13, 0.9)})
    assert disagreement["verdict"] == "unclear"


# ------------------------------------------- dry-run failure paths (7)
def test_preflight_failure_paths(tmp_path):
    assert ea.entries_projection(44949)["pass"] is True
    assert ea.entries_projection(35000)["pass"] is False
    empty = tmp_path / "data"
    (empty / "lucid").mkdir(parents=True)
    assert ea.probe_status(empty)["pass"] is False
    with pytest.raises(RuntimeError):
        ea.live_free_entries(tmp_path / "missing_guard.json")
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"level": "HOLD",
                                 "caps": {"free_entries": 31000}}))
    live = ea.live_free_entries(guard)
    assert live["guard_level"] == "HOLD"
    parser = arch_cli.parser()
    args = parser.parse_args(["dry-run", "--out", "/tmp/plan.json"])
    assert args.mode == "dry-run"
    args = parser.parse_args(["train", "--variant", "window128", "--seed", "9000",
                              "--out", "/ceph/ext/train/arch/window128"])
    assert args.variant == "window128"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--variant", "window64", "--seed", "9000",
                           "--out", "/tmp/x"])


# ------------------------------------------------------ smoke training finite (8)
def test_smoke_level_training_is_finite(geometry):
    settings = la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))
    arm = la.LucidArm("sigreg_jepa", geometry, settings=settings, seed=9000,
                      encoder_config=pe.EncoderConfig(window=16, shift=8))
    rng = np.random.default_rng(0)
    n_pmt = geometry.n_pmt
    features = rng.normal(size=(2, n_pmt, pe.N_FEATURES)).astype(np.float32)
    from jepa_noise import ext_readout as readout
    within = readout.within_target_mask(geometry)
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
    assert ea.STEPS == 10000
