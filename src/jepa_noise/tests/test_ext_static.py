# SPDX-License-Identifier: MIT
"""Acceptance tests for the frozen static-target extension (runbook 2 Nov 2026 §3.3).

Covers: the authorisation gate; the frozen net's device move and bit-identity with
E3's seed-0 net; frozen parameters/state_dict after one synthetic training step;
the zeroed-clone target pass and the untouched context pass; the §2 manipulation
check on a natural-like batch (pass) and an across-like batch (fail); run-directory
refusal + ``resumed_from_step == 0``; smoke-level training finiteness; and the
two-sided ``static_target_r`` verdicts.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import ext_static as es  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_static", ROOT / "scripts" / "jepa_ext_static.py")
static_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(static_cli)


@pytest.fixture(scope="module")
def geometry():
    return lb.from_synthetic({"module_ids": tuple(range(6)),
                              "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                              "pmts_per_module": 19})


def _settings():
    return la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))


def _arm(geometry, seed=9000):
    return la.LucidArm("sigreg_jepa", geometry, settings=_settings(), seed=seed,
                       encoder_config=pe.EncoderConfig(window=16, shift=8))


def _natural_batch(geometry, n_events=4, seed=1):
    """A within+natural-like batch: identical geometry columns, event-varying 6-8."""
    rng = np.random.default_rng(seed)
    n_pmt = geometry.n_pmt
    base = np.zeros((n_pmt, pe.N_FEATURES), np.float32)
    base[:, :6] = pe.geometry_features(geometry,
                                       pe.FeatureScaling.from_geometry(geometry))
    features = np.broadcast_to(base, (n_events, n_pmt, pe.N_FEATURES)).copy()
    features[..., 6] = rng.normal(size=(n_events, n_pmt))       # ln q
    features[..., 7] = rng.normal(size=(n_events, n_pmt)) * 3.0  # time
    features[..., 8] = rng.integers(0, 2, size=(n_events, n_pmt))  # hit
    features = features.astype(np.float32)
    hit = features[..., 8] > 0.5
    within = er.within_target_mask(geometry)
    hidden = np.broadcast_to(within, (n_events, n_pmt)).copy()
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


def _across_like_batch(geometry, n_events=4, seed=2):
    """Same event-varied features but a different target mask per event."""
    batch = _natural_batch(geometry, n_events=n_events, seed=seed)
    n_pmt = geometry.n_pmt
    within = er.within_target_mask(geometry)
    even = ~within
    hidden = np.zeros((n_events, n_pmt), bool)
    for event in range(n_events):
        hidden[event] = within if event % 2 == 0 else even
    return pe.LucidBatch(
        features=batch.features, target=torch.as_tensor(hidden),
        hidden=torch.as_tensor(hidden),
        context_index=torch.as_tensor(np.flatnonzero(~within)).unsqueeze(0)
        .expand(n_events, -1),
        target_index=torch.as_tensor(np.flatnonzero(within)).unsqueeze(0)
        .expand(n_events, -1),
        hit_flags=batch.hit_flags, log_charge=batch.log_charge,
        event_ids=batch.event_ids, keys=batch.keys, report=batch.report,
        scaling=batch.scaling, split="across", occupancy="natural")


# ---------------------------------------------------------------- gate (1)
def test_gate_requires_the_line_without_a_banner(tmp_path):
    root = tmp_path / "root"
    (root / "docs/jepa").mkdir(parents=True)
    runbook = root / es.RUNBOOK_REL
    decisions = root / es.DECISIONS_REL
    runbook.write_text("## 0. Authority\n\n> \"" + es.AUTHOR_LINE + "\"\n\n## 1. Why\n")
    decisions.write_text("history\n> \"" + es.AUTHOR_LINE + "\"\n")
    assert es.assert_authorised(root)["authorised"] is True
    # banner present -> exit 2 even with the line
    runbook.write_text("## 0. Authority\n\nTRAINING NOT AUTHORISED\n\n"
                       "> \"" + es.AUTHOR_LINE + "\"\n\n## 1. Why\n")
    with pytest.raises(SystemExit) as caught:
        es.assert_authorised(root)
    assert caught.value.code == 2
    # line missing from the runbook §0 -> exit 2
    runbook.write_text("## 0. Authority\n\nnothing here\n\n## 1. Why\n")
    with pytest.raises(SystemExit) as caught:
        es.assert_authorised(root)
    assert caught.value.code == 2
    # line missing from DECISIONS -> exit 2
    runbook.write_text("## 0. Authority\n\n> \"" + es.AUTHOR_LINE + "\"\n\n## 1. Why\n")
    decisions.write_text("history without the line\n")
    with pytest.raises(SystemExit) as caught:
        es.assert_authorised(root)
    assert caught.value.code == 2
    # the committed repository passes the gate
    assert es.assert_authorised(ROOT)["authorised"] is True


# ------------------------------------------------------- frozen net / device (2)
def test_frozen_net_is_e3s_seed_zero_net_and_moves_with_the_arm(geometry):
    arm = _arm(geometry)
    es.attach_static_target(arm, geometry)
    frozen = arm.__dict__["_ext_frozen_target"]
    e3_arm = _arm(geometry)
    eh.attach_frozen_target(e3_arm, geometry)
    e3_frozen = e3_arm.__dict__["_ext_frozen_target"]
    params = list(frozen.named_parameters())
    e3_params = list(e3_frozen.named_parameters())
    assert [name for name, _ in params] == [name for name, _ in e3_params]
    for (_, left), (_, right) in zip(params, e3_params):
        torch.testing.assert_close(left.detach().cpu(), right.detach().cpu(),
                                   rtol=0, atol=0)
    assert all(parameter.requires_grad is False for parameter in frozen.parameters())
    # the _apply override moves the frozen net with the arm (throwaway meta move)
    throwaway = _arm(geometry)
    es.attach_static_target(throwaway, geometry)
    throwaway_frozen = throwaway.__dict__["_ext_frozen_target"]
    throwaway.to(torch.device("meta"))
    assert next(throwaway_frozen.parameters()).device.type == "meta"
    assert next(throwaway.parameters()).device.type == "meta"


# ------------------------------------------------- training step / state_dict (3)
def test_one_synthetic_training_step_keeps_frozen_params_and_state_dict(geometry):
    arm = _arm(geometry)
    plain = _arm(geometry)
    es.attach_static_target(arm, geometry)
    frozen = arm.__dict__["_ext_frozen_target"]
    snapshot = {name: value.detach().clone() for name, value in frozen.named_parameters()}
    assert set(arm.state_dict()) == set(plain.state_dict())
    batch = _natural_batch(geometry)
    optimizer = torch.optim.Adam(arm.parameters(), lr=1e-3)
    out = arm._forward_sigreg(batch)
    assert np.isfinite(float(out["loss"]))
    assert out["target_tokens"].requires_grad is False
    out["loss"].backward()
    optimizer.step()
    for name, parameter in frozen.named_parameters():
        torch.testing.assert_close(parameter, snapshot[name], rtol=0, atol=0)
        assert parameter.grad is None


# --------------------------------------------- zeroed clone / context pass (4)
def test_target_pass_zeroes_event_columns_and_context_keeps_full_features(geometry):
    arm = _arm(geometry)
    es.attach_static_target(arm, geometry)
    frozen = arm.__dict__["_ext_frozen_target"]
    batch = _natural_batch(geometry)
    original = batch.features.clone()
    seen = {"frozen": [], "context": []}
    frozen_encode, context_encode = frozen.encode, arm.jepa.encode

    def recording_frozen(features, hidden_mask=None):
        seen["frozen"].append(features.detach().clone())
        return frozen_encode(features, hidden_mask=hidden_mask)

    def recording_context(features, hidden_mask=None):
        seen["context"].append(features.detach().clone())
        return context_encode(features, hidden_mask=hidden_mask)

    frozen.encode = recording_frozen
    arm.jepa.encode = recording_context
    try:
        out = arm._encode_views(batch)
    finally:
        frozen.encode = frozen_encode
        arm.jepa.encode = context_encode
    label_features = seen["frozen"][0]
    torch.testing.assert_close(label_features[..., 6:9],
                               torch.zeros_like(label_features[..., 6:9]))
    torch.testing.assert_close(label_features[..., :6], original[..., :6])
    torch.testing.assert_close(batch.features, original)          # never mutated
    torch.testing.assert_close(seen["context"][0], original)      # context sees all
    assert out[2].requires_grad is False                          # detached label


# ----------------------------------------------- manipulation check natural/across (5)
def test_manipulation_check_passes_on_natural_and_fails_on_across_like(geometry):
    arm = _arm(geometry)
    es.attach_static_target(arm, geometry)
    natural = es.static_label_manipulation(arm, _natural_batch(geometry))
    assert natural["pass"] is True
    assert natural["checks"]["event_columns_zero"] is True
    assert natural["checks"]["geometry_columns_unchanged"] is True
    assert natural["checks"]["original_features_unchanged"] is True
    assert natural["label_event_fraction"] is not None
    assert natural["label_event_fraction"] <= 1e-3
    assert natural["per_pmt_variance"] is not None and natural["per_pmt_variance"] > 0
    across = es.static_label_manipulation(arm, _across_like_batch(geometry))
    assert across["pass"] is False
    assert across["label_event_fraction"] > 1e-3


# ------------------------------------------------------ run dir / resume (6)
def test_run_dir_refusal_and_fresh_resume(tmp_path):
    run_id, run_dir = es.prepare_run_dir(tmp_path / "train", 9000, steps=10000,
                                         device="cpu")
    assert run_id == "dev_s005_within__sigreg_jepa__s9000"
    sidecar = json.loads((run_dir / "variant.json").read_text())
    assert sidecar["variant"] == "static_target" and sidecar["extension"] == es.EXTENSION
    with pytest.raises(RuntimeError, match="already exists"):
        es.prepare_run_dir(tmp_path / "train", 9000)
    from jepa_noise import ext_h1clean as ec
    assert ec.assert_fresh_resume({"resumed_from_step": 0}) == 0
    assert ec.assert_fresh_resume({"resumed_from_step": None}) == 0   # fresh run
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({"resumed_from_step": 5})
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({})


# ------------------------------------------------------ smoke training finite (7)
def test_smoke_level_training_with_static_target_is_finite(geometry):
    arm = _arm(geometry)
    es.attach_static_target(arm, geometry)
    batch = _natural_batch(geometry)
    features = batch.features.clone()
    optimizer = torch.optim.Adam(arm.parameters(), lr=1e-3)
    out = arm._forward_sigreg(batch)
    assert np.isfinite(float(out["loss"]))
    out["loss"].backward()
    optimizer.step()
    metrics = arm._forward_sigreg(batch)["metrics"]
    assert all(np.isfinite(float(value)) for value in metrics.values())
    torch.testing.assert_close(batch.features, features)


# ------------------------------------------------------- reading verdicts §2
def _run_row(full_v, full_r2, half_v, half_r2):
    return {"full_input": {"vertex_resolution_m": full_v,
                           "token_log_charge_r2": {"r2": full_r2}},
            "half_pass": {"vertex_resolution_m": half_v,
                          "token_log_charge_r2": {"r2": half_r2}}}


def test_static_target_r_three_valued():
    recovered = es.static_target_r({"9000": _run_row(0.14, 0.9, 0.13, 0.9),
                                    "9001": _run_row(0.15, 0.9, 0.14, 0.9)})
    assert recovered["verdict"] == "recovered"
    not_recovered = es.static_target_r({"9000": _run_row(0.22, 0.02, 0.20, 0.1),
                                        "9001": _run_row(0.24, 0.03, 0.21, 0.1)})
    assert not_recovered["verdict"] == "not recovered"
    charge_fail = es.static_target_r({"9000": _run_row(0.14, 0.2, 0.13, 0.9),
                                      "9001": _run_row(0.15, 0.2, 0.14, 0.9)})
    assert charge_fail["verdict"] == "not recovered"
    disagreement = es.static_target_r({"9000": _run_row(0.14, 0.9, 0.13, 0.9),
                                       "9001": _run_row(0.22, 0.9, 0.13, 0.9)})
    assert disagreement["verdict"] == "unclear"


# ---------------------------------------------------------------------- parser
def test_cli_parses_the_runbook_commands():
    args = static_cli.parser().parse_args(["dry-run", "--out", "/ceph/ext/plan.json"])
    assert args.mode == "dry-run" and args.data_root == es.DATA_ROOT_DEFAULT
    args = static_cli.parser().parse_args(["smoke", "--seed", "9000",
                                           "--out", "/ceph/ext/smoke"])
    assert args.steps == 20 and args.probe_max_events == 200
    args = static_cli.parser().parse_args(["train", "--seed", "9001",
                                           "--out", "/ceph/ext/train/static_target"])
    assert args.seed == 9001
    args = static_cli.parser().parse_args(["evaluate", "--run", "/a", "--run", "/b",
                                           "--out", "/ceph/ext/eval"])
    assert args.run == ["/a", "/b"]
    args = static_cli.parser().parse_args(["report", "--out", "/tmp/r",
                                           "--eval", "/ceph/ext/eval"])
    assert args.eval == "/ceph/ext/eval"
