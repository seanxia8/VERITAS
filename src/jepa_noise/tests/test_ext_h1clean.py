# SPDX-License-Identifier: MIT
"""Acceptance tests for the clean H1 causal test (runbook 2 Nov 2026 §3.1/§4).

(1) the encoder input tensor is byte-identical with the patch installed (the noise
    touches only the predictor's query tensor);
(2) with a fixed reseed before each pass, two patched passes draw the same noise and
    the patched prediction differs from the unpatched one;
(3) evaluation outside the patch context equals an unpatched evaluation;
(4) the run-directory refusal triggers and ``resumed_from_step == 0`` is enforced;
(5) smoke-level training with the patch gives a finite loss and finite metrics.
Plus the hard authorisation gate and the §3.4 manipulation-ratio helper.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_h1clean as ec  # noqa: E402
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_h1clean", ROOT / "scripts" / "jepa_ext_h1clean.py")
h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1)

SIGMA = 0.10


@pytest.fixture(scope="module")
def geometry():
    return lb.from_synthetic({"module_ids": tuple(range(6)),
                              "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                              "pmts_per_module": 19})


def _settings():
    return la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))


def _arm(geometry):
    return la.LucidArm("sigreg_jepa", geometry, settings=_settings(), seed=9000,
                       encoder_config=pe.EncoderConfig(window=16, shift=8))


def _batch(geometry, n_events=2, seed=1):
    rng = np.random.default_rng(seed)
    n_pmt = geometry.n_pmt
    features = rng.normal(size=(n_events, n_pmt, pe.N_FEATURES)).astype(np.float32)
    hit = np.zeros((n_events, n_pmt), bool)
    hit[:, 0::3] = True
    hidden = np.zeros((n_events, n_pmt), bool)
    within = er.within_target_mask(geometry)
    hidden[:, within] = True
    return pe.LucidBatch(
        features=torch.as_tensor(features), target=torch.as_tensor(hidden),
        hidden=torch.as_tensor(hidden),
        context_index=torch.as_tensor(np.flatnonzero(~within)).unsqueeze(0)
        .expand(n_events, -1),
        target_index=torch.as_tensor(np.flatnonzero(within)).unsqueeze(0)
        .expand(n_events, -1),
        hit_flags=hit, log_charge=features[..., pe.LOG_CHARGE_INDEX].copy(),
        event_ids=np.arange(n_events), keys=tuple({} for _ in range(n_events)),
        report={}, scaling=pe.FeatureScaling.from_geometry(geometry),
        split="within", occupancy="natural")


# ---------------------------------------------------------------- the patch (1)(2)
def test_patch_touches_only_the_query_tensor_and_keeps_encoder_input(geometry):
    arm = _arm(geometry)
    batch = _batch(geometry)
    features = batch.features.clone()
    captured = {}
    predictor = arm.jepa.predictor

    class Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, context, queries):
            captured["context"] = context.detach().clone()
            captured["queries"] = queries.detach().clone()
            return self.inner(context, queries)

    arm.jepa.predictor = Recorder(predictor)
    with torch.no_grad():
        readout = arm.jepa.encode(batch.features)
        clean_queries = arm.jepa.query_features[batch.target_index]
        clean_context = readout.gather(
            1, batch.context_index.unsqueeze(-1).expand(-1, -1, readout.size(-1)))
        original = ec.install_query_noise(arm, SIGMA)
        try:
            arm.jepa.predict(readout, batch.context_index, batch.target_index)
        finally:
            ec.restore_predict(arm, original)
    torch.testing.assert_close(batch.features, features)          # encoder input untouched
    torch.testing.assert_close(captured["context"], clean_context)  # context unchanged
    delta = captured["queries"] - clean_queries
    assert float(delta.abs().max()) > 0.0
    assert float(delta.std()) == pytest.approx(SIGMA, rel=0.5)
    arm.jepa.predictor = predictor


def test_fixed_reseed_reproduces_the_noise_and_changes_the_prediction(geometry):
    arm = _arm(geometry)
    batch = _batch(geometry)
    with torch.no_grad():
        baseline = arm._encode_views(batch)[1]
    with ec.query_noise_patch(arm, SIGMA):
        torch.manual_seed(1234)
        with torch.no_grad():
            first = arm._encode_views(batch)[1]
        torch.manual_seed(1234)
        with torch.no_grad():
            second = arm._encode_views(batch)[1]
    torch.testing.assert_close(first, second)                      # same noise
    assert float((first - baseline).abs().max()) > 0.0             # differs from unpatched


def test_evaluation_outside_the_patch_equals_the_unpatched_evaluation(geometry):
    arm = _arm(geometry)
    arm.eval()
    batch = _batch(geometry)
    with torch.no_grad():
        clean = arm.jepa.encode(batch.features)
        clean_prediction = arm._forward_sigreg(batch)["prediction"]
    with ec.query_noise_patch(arm, SIGMA):
        with torch.no_grad():
            inside = arm.jepa.encode(batch.features)               # encode cannot see noise
    with torch.no_grad():
        restored = arm.jepa.encode(batch.features)
        restored_prediction = arm._forward_sigreg(batch)["prediction"]
    torch.testing.assert_close(inside, clean)
    torch.testing.assert_close(restored, clean)
    torch.testing.assert_close(restored_prediction, clean_prediction)


# ------------------------------------------------------- run dirs / resume (4)
def test_run_dir_refusal_and_fresh_resume(tmp_path):
    run_id, run_dir = ec.prepare_run_dir(tmp_path / "train", 9000, sigma=SIGMA,
                                         steps=10000, device="cpu")
    assert run_id == "dev_s005_within__sigreg_jepa__s9000"
    sidecar = json.loads((run_dir / "variant.json").read_text())
    assert sidecar["variant"] == "query_noise" and sidecar["sigma"] == SIGMA
    assert sidecar["extension"] == ec.EXTENSION
    with pytest.raises(RuntimeError, match="already exists"):
        ec.prepare_run_dir(tmp_path / "train", 9000, sigma=SIGMA)
    assert ec.assert_fresh_resume({"resumed_from_step": 0}) == 0
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({"resumed_from_step": 3})
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({})


# ------------------------------------------------------ smoke training finite (5)
def test_smoke_level_training_with_patch_is_finite(geometry):
    arm = _arm(geometry)
    batch = _batch(geometry)
    features = batch.features.clone()
    optimizer = torch.optim.Adam(arm.parameters(), lr=1e-3)
    with ec.query_noise_patch(arm, SIGMA):
        out = arm._forward_sigreg(batch)
        assert np.isfinite(float(out["loss"]))
        out["loss"].backward()
        optimizer.step()
        metrics = arm._forward_sigreg(batch)["metrics"]
    assert all(np.isfinite(float(value)) for value in metrics.values())
    torch.testing.assert_close(batch.features, features)


# ------------------------------------------------------- manipulation check §3.4
def test_manipulation_ratio_is_one_without_noise_and_grows_with_noise(geometry):
    arm = _arm(geometry)
    batch = _batch(geometry)
    zero = ec._mse_with_and_without_noise(arm, batch, 0.0)
    assert zero["ratio"] == pytest.approx(1.0, abs=1e-12)
    assert zero["pass"] is False
    noisy = ec._mse_with_and_without_noise(arm, batch, 1.0)
    assert noisy["ratio"] is not None and noisy["ratio"] > 1.0
    again = ec._mse_with_and_without_noise(arm, batch, 1.0)
    assert again["ratio"] == pytest.approx(noisy["ratio"])        # deterministic reseed


# ---------------------------------------------------------------- hard gate §0
def _fake_root(tmp_path, *, runbook_line="我授权", decisions_line=None):
    root = tmp_path / "root"
    (root / "docs/jepa").mkdir(parents=True)
    (root / "docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md").write_text(
        "## 0. Authority\n\n" + runbook_line + "\n\n## 1. Why\n")
    (root / "docs/jepa/DECISIONS.md").write_text(
        "history\n" + (ec.AUTHOR_LINE if decisions_line is None else decisions_line) + "\n")
    return root


def test_hard_gate_requires_the_author_line_in_both_files(tmp_path):
    root = _fake_root(tmp_path)
    assert ec.assert_authorised(root)["authorised"] is True
    with pytest.raises(ec.AuthorisationError):
        ec.assert_authorised(_fake_root(tmp_path / "a", runbook_line="TRAINING NOT AUTHORISED"))
    with pytest.raises(ec.AuthorisationError):
        ec.assert_authorised(_fake_root(tmp_path / "b", decisions_line="no line"))
    # the committed repository carries the author line (§0 gate at run time)
    assert ec.assert_authorised(ROOT)["authorised"] is True


def test_committed_runbook_hash_and_author_line():
    runbook = ROOT / "docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md"
    assert runbook.is_file()
    assert ec._section_zero(runbook.read_text()).count(ec.AUTHORISATION_SUBSTRING) >= 1
    assert ec.RUNBOOK_REL in str(runbook)


# ---------------------------------------------------------------------- parser
def test_cli_parses_all_modes_with_the_runbook_paths():
    args = h1.parser().parse_args(["dry-run", "--out", "/ceph/ext/plan.json"])
    assert args.mode == "dry-run" and args.data_root == ec.DATA_ROOT_DEFAULT
    args = h1.parser().parse_args(["smoke", "--seed", "9000", "--out", "/ceph/ext/smoke"])
    assert args.steps == 20 and args.probe_max_events == 200
    args = h1.parser().parse_args(["train", "--seed", "9001",
                                   "--out", "/ceph/ext/train/query_noise"])
    assert args.seed == 9001
    args = h1.parser().parse_args(["evaluate", "--run", "/a", "--run", "/b",
                                   "--out", "/ceph/ext/eval"])
    assert args.run == ["/a", "/b"]
    args = h1.parser().parse_args(["report", "--out", "/tmp/r", "--eval", "/ceph/ext/eval"])
    assert args.eval == "/ceph/ext/eval"
