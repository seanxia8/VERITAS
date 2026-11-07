# SPDX-License-Identifier: MIT
"""Acceptance tests for the shortcut-collapse follow-up extension (Part B §9.3).

Covers every bullet of ``docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md``
Part B §9.3: the ``target_posfree`` variant (zeroed clone, bit-identity of the
rebound ``_encode_views`` with the zeroing disabled, label not detached); the
§5.2 identity blindness on a synthetic geometry; the ``build_arm`` patch
(restored in a ``finally``, also after an exception; applies only to
``sigreg_jepa``); the training guard; the R-1 family table / response size /
reading rule / ``not_run`` / cache-hash refusal; the §4 class and recovery rules
on synthetic values (including a seed disagreement); the forbidden §2.5 call
paths (sentinel patches and a static source check); and the §9.2 gate negative
cases (a), (b), (c).

No real data and no GPU are touched: the model tests use synthetic geometries and
tiny tensor batches in pytest temporaries only.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_followup as ef  # noqa: E402
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import ext_shortcut as xs  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_shortcut", ROOT / "scripts" / "jepa_ext_shortcut.py")
shortcut_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shortcut_cli)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def geometry():
    return lb.from_synthetic({"module_ids": tuple(range(6)),
                              "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                              "pmts_per_module": 19})


@pytest.fixture(scope="module")
def config():
    return lr.load_run_config(ROOT / xs.CONFIG_REL)


@pytest.fixture(scope="module")
def cell(config):
    return lr.dev_cell(config)


def _settings():
    return la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))


def _arm(geometry, seed=9000):
    return la.LucidArm("sigreg_jepa", geometry=geometry, settings=_settings(), seed=seed,
                       encoder_config=pe.EncoderConfig(window=16, shift=8))


def _natural_batch(geometry, n_events=8, seed=1):
    """A ``within`` + natural-like batch: identical geometry columns, event-varying 6-8."""
    rng = np.random.default_rng(seed)
    n_pmt = geometry.n_pmt
    base = np.zeros((n_pmt, pe.N_FEATURES), np.float32)
    base[:, :6] = pe.geometry_features(geometry,
                                       pe.FeatureScaling.from_geometry(geometry))
    features = np.broadcast_to(base, (n_events, n_pmt, pe.N_FEATURES)).copy()
    features[..., 6] = rng.normal(size=(n_events, n_pmt))
    features[..., 7] = rng.normal(size=(n_events, n_pmt)) * 3.0
    features[..., 8] = rng.integers(0, 2, size=(n_events, n_pmt))
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


# ============================================================ variant §5.1/§9.3
def test_posfree_features_zeroes_columns_0_to_5_of_a_clone(geometry):
    batch = _natural_batch(geometry)
    features = batch.features
    original = features.detach().clone()
    out = xs.posfree_features(features)
    assert torch.equal(out[..., 0:6], torch.zeros_like(out[..., 0:6]))
    assert torch.equal(out[..., 6:9], original[..., 6:9])
    assert torch.equal(features, original)          # never mutates its input
    assert out.data_ptr() != features.data_ptr()    # a real clone
    with pytest.raises(ValueError):
        xs.posfree_features(torch.zeros(2, 3, 5))


def test_rebound_encode_views_is_bit_identical_with_zeroing_disabled(geometry):
    arm = _arm(geometry)
    arm.eval()
    batch = _natural_batch(geometry, n_events=4, seed=3)
    with torch.no_grad():
        patched = xs._encode_views(arm, batch, zero_positions=False)
        original = la.LucidArm._encode_views(arm, batch)
    for index, name in enumerate(("context", "prediction", "labels", "hidden_index",
                                  "full")):
        if patched[index] is None or original[index] is None:
            assert patched[index] is None and original[index] is None, name
            continue
        assert torch.equal(patched[index], original[index]), name
    # the zeroing changes only the label pass
    zeroed = xs._encode_views(arm, batch, zero_positions=True)
    assert torch.equal(zeroed[0], original[0])       # context unchanged
    assert torch.equal(zeroed[4], original[4])       # full pass unchanged
    assert not torch.equal(zeroed[2], original[2])   # label changed


def test_label_is_not_detached_and_gradients_reach_the_encoder(geometry):
    arm = _arm(geometry)
    xs.attach_posfree_target(arm)
    arm.eval()
    batch = _natural_batch(geometry)
    out = arm._encode_views(batch)
    labels = out[2]
    assert labels.requires_grad is True
    assert labels.grad_fn is not None
    loss = torch.nn.functional.mse_loss(labels, torch.zeros_like(labels))
    loss.backward()
    grad = xs._input_projection(arm).weight.grad
    assert grad is not None and float(grad.abs().max()) > 0.0


# ==================================================== identity blindness §5.2
def test_identity_blindness_permutation_on_a_synthetic_geometry(geometry):
    arm = _arm(geometry)
    xs.attach_posfree_target(arm)
    arm.eval()
    check = xs.manipulation_check(arm, _natural_batch(geometry))
    blind = check["identity_blindness"]
    assert blind["label_max_abs_delta"] <= 1e-6
    assert blind["context_max_abs_delta"] > 1e-3
    assert check["pass"] is True
    assert check["checks"]["label_position_columns_zero"] is True
    assert check["checks"]["label_event_columns_equal"] is True
    assert check["checks"]["original_features_unchanged"] is True
    assert check["checks"]["context_receives_original"] is True
    assert check["checks"]["full_pass_receives_original"] is True
    assert all(check["masks"].values())
    assert check["label_gradient"]["non_zero"] is True
    fractions = check["label_fractions"]
    assert fractions["n_events"] > 0 and fractions["f_layout"] is not None
    assert fractions["f_event"] is not None


# ========================================================= build patch §9.3
def test_build_arm_patch_restores_and_applies_only_to_sigreg_jepa(geometry, cell):
    original = lr.build_arm
    with xs.build_arm_patch(log=lambda *a, **k: None) as state:
        assert lr.build_arm is not original
        sigreg = lr.build_arm(xs.ARM_NAME, geometry, la.load_config(), cell, 9000)
        other = lr.build_arm("mae_mse", geometry, la.load_config(), cell, 9000)
        assert "_encode_views" in sigreg.__dict__      # variant installed
        assert "_encode_views" not in other.__dict__   # other arms untouched
        assert state == {"installed": 1, "skipped": 1}
    assert lr.build_arm is original
    # restored in a finally, also after an exception
    with pytest.raises(RuntimeError, match="boom"):
        with xs.build_arm_patch(log=lambda *a, **k: None):
            raise RuntimeError("boom")
    assert lr.build_arm is original


# ============================================================ training guard
def test_training_guard_refuses_unauthorised_launches(tmp_path):
    out = tmp_path / "train"
    # authorised launches pass
    assert xs.assert_launch_allowed("train", seed=9000, steps=10000, out_root=out)
    assert xs.assert_launch_allowed("train", seed=9001, steps=10000, out_root=out)
    assert xs.assert_launch_allowed("smoke", seed=9000, steps=20, out_root=out)
    # any other seed or step count is refused
    for seed, steps in ((9002, 10000), (0, 10000), (9000, 5000), (9000, 20)):
        with pytest.raises(SystemExit) as caught:
            xs.assert_launch_allowed("train", seed=seed, steps=steps, out_root=out)
        assert caught.value.code == 2
    for seed, steps in ((9000, 21), (9000, 10000), (9001, 20)):
        with pytest.raises(SystemExit):
            xs.assert_launch_allowed("smoke", seed=seed, steps=steps, out_root=out)
    # an existing run dir/root is refused (never resumed)
    xs.prepare_run_dir(out, 9000, steps=10000, device="cpu")
    with pytest.raises(SystemExit):
        xs.assert_launch_allowed("train", seed=9000, steps=10000, out_root=out)
    with pytest.raises(RuntimeError, match="already exists"):
        xs.prepare_run_dir(out, 9000, steps=10000, device="cpu")
    sidecar = json.loads((xs.run_dir_for(out, 9000) / "variant.json").read_text())
    assert sidecar["variant"] == xs.VARIANT and sidecar["extension"] == xs.EXTENSION
    assert sidecar["seed"] == 9000


# ==================================================================== R-1 §3
def test_r1_family_table_rejects_an_unknown_label():
    assert xs.family_of("mae_within_trigger9000") == "generative"
    assert xs.family_of("fix9000") == "joint_embedding_collapsed"
    assert xs.family_of("init9000") == "untrained_reference"
    assert xs.family_of("ijepa9001") == "joint_embedding_working"   # optional row
    with pytest.raises(ValueError, match="unknown R-1 model label"):
        xs.family_of("mystery9000")
    assert xs.family_table(["fix9000", "init9000"]) == {
        "fix9000": "joint_embedding_collapsed", "init9000": "untrained_reference"}
    with pytest.raises(ValueError):
        xs.family_table(["fix9000", "mystery9000"])


def test_r1_response_size_definition_and_pairing():
    n = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    d = n + np.asarray([1.0, 0.0])
    result = xs.response_size(n, d)
    # numerator = 1, denominator = mean(||N - mean N||^2) = 1
    assert result["response_size"] == pytest.approx(1.0)
    assert result["numerator"] == pytest.approx(1.0)
    assert result["denominator"] == pytest.approx(1.0)
    assert result["n_paired"] == 2
    # paired on (run_id, event_id), independent of row order
    n_keys = [("r0", 11), ("r0", 12)]
    d_keys = [("r0", 12), ("r0", 11)]
    shuffled = xs.response_size(n, d[::-1], n_keys=n_keys, d_keys=d_keys)
    assert shuffled["response_size"] == pytest.approx(1.0)
    # a model far from N has a large S
    far = xs.response_size(n, n * 3.0 + 5.0)
    assert far["response_size"] > 1.0
    with pytest.raises(ValueError):
        xs.response_size(n, np.zeros((3, 2)))


def _r1_rows(generative, frozen):
    rows = {}
    for index, (r2_eq, rho_lr) in enumerate(generative):
        rows[f"mae_within_trigger{9000 + index:04d}"] = {
            "status": "available", "r2_eq": r2_eq, "rho_lr": rho_lr}
    for seed, (r2_eq, rho_lr) in zip((9000, 9001), frozen):
        rows[f"frozen_target{seed}"] = {"status": "available", "r2_eq": r2_eq,
                                        "rho_lr": rho_lr}
    return rows


def test_r1_reading_rule_supports_contradicts_mixed():
    supports = xs.r1_reading(_r1_rows([(0.9, 0.8)] * 8, [(0.5, 0.4), (0.55, 0.45)]))
    assert supports["verdict"] == "supports"
    contradicts = xs.r1_reading(_r1_rows([(0.2, 0.1)] * 8, [(0.5, 0.4), (0.55, 0.45)]))
    assert contradicts["verdict"] == "contradicts"
    mixed = xs.r1_reading(_r1_rows([(0.9, 0.1)] * 8, [(0.5, 0.4), (0.55, 0.45)]))
    assert mixed["verdict"] == "mixed"
    # incomplete values do not crash, and the reading is still descriptive
    incomplete = xs.r1_reading(_r1_rows([(None, 0.8)] * 8,
                                       [(0.5, 0.4), (0.55, 0.45)]))
    assert incomplete["verdict"] in ("mixed", "supports", "contradicts")


def test_r1_is_not_run_when_the_pilot_outputs_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(xs, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})
    payload = xs.r1_payload(ROOT, results_dir=tmp_path / "missing",
                            log=lambda *a, **k: None)
    assert payload["status"] == "not_run"
    assert "pilot outputs missing" in payload["reason"]
    assert payload["reading"] is None
    assert payload["table"] == {}


def test_r1_cache_with_a_sha256_mismatch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(xs, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})
    directory = tmp_path / "x1" / "fix9000"
    directory.mkdir(parents=True)
    n_path = directory / "probe_test__N.npz"
    np.savez(n_path, embeddings=np.eye(2), event_ids=np.arange(2),
             run_ids=np.asarray(["r0", "r0"], dtype=object))
    correct = lr.sha256_file(n_path)
    # no sidecar at all -> refused (accepted only by hash)
    with pytest.raises(RuntimeError, match="no sha256 sidecar"):
        xs.load_cached_embedding(n_path)
    # a wrong sidecar -> refused
    (directory / "probe_test__N.npz.sha256").write_text("f" * 64 + "  n\n")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        xs.load_cached_embedding(n_path)
    # the matching sidecar -> accepted, and the arrays load
    (directory / "probe_test__N.npz.sha256").write_text(correct + "  n\n")
    loaded = xs.load_cached_embedding(n_path)
    assert loaded["verification"]["pass"] is True
    assert loaded["verification"]["sha256"] == correct
    assert loaded["embeddings"].shape == (2, 2)
    assert loaded["pairing"] == [("r0", "0"), ("r0", "1")]
    # LARGE_ARRAYS.json is an accepted alternative source of the expected hash
    (directory / "probe_test__N.npz.sha256").unlink()
    large = {"files": {"x1/fix9000/probe_test__N.npz": "0" * 64}}
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        xs.load_cached_embedding(n_path, large_arrays=large, ext_root=tmp_path)
    large_ok = {"files": {"x1/fix9000/probe_test__N.npz": correct}}
    loaded_ok = xs.load_cached_embedding(n_path, large_arrays=large_ok,
                                         ext_root=tmp_path)
    assert loaded_ok["verification"]["source"].startswith("LARGE_ARRAYS")


def test_cache_condition_and_cached_file_discovery(tmp_path):
    directory = tmp_path / "x1" / "fix9000"
    directory.mkdir(parents=True)
    for name in ("probe_test__N.npz", "probe_test__D.npz", "notes.npz"):
        np.savez(directory / name, embeddings=np.eye(2))
    assert xs.cache_condition("x__N.npz") == "N"
    assert xs.cache_condition("x__D.npz") == "D"
    assert xs.cache_condition("x.npz") is None
    found = xs.cache_files(tmp_path, "fix9000")
    assert sorted(found["conditions"]) == ["D", "N"]


# ============================================================ readings §4/§5.4
def test_event_class_and_recovery_rules_on_synthetic_values():
    assert xs.event_class({9000: 0.05, 9001: 0.02})["class"] == "event-dependent"
    assert xs.event_class({9000: 1e-6, 9001: 5e-4})["class"] == "event-invariant"
    assert xs.event_class({9000: 0.05, 9001: 1e-5})["class"] == "intermediate"
    single = xs.event_class({9000: 0.05})
    assert single["class"] == "single-seed (not classified)"

    def run(full_v, full_r2, half_v, half_r2):
        return {"full_input": {"vertex_resolution_m": full_v,
                               "token_log_charge_r2": {"r2": full_r2}},
                "half_pass": {"vertex_resolution_m": half_v,
                              "token_log_charge_r2": {"r2": half_r2}}}

    recovered = xs.recovery_rule({"9000": run(0.14, 0.9, 0.13, 0.9),
                                  "9001": run(0.15, 0.9, 0.14, 0.9)})
    assert recovered["verdict"] == "recovered"
    not_recovered = xs.recovery_rule({"9000": run(0.22, 0.02, 0.20, 0.1),
                                      "9001": run(0.24, 0.03, 0.21, 0.1)})
    assert not_recovered["verdict"] == "not recovered"
    charge_fail = xs.recovery_rule({"9000": run(0.14, 0.2, 0.13, 0.9),
                                    "9001": run(0.15, 0.2, 0.14, 0.9)})
    assert charge_fail["verdict"] == "not recovered"
    # a seed disagreement is unclear, by the same rule as static_target_r
    disagreement = xs.recovery_rule({"9000": run(0.14, 0.9, 0.13, 0.9),
                                     "9001": run(0.22, 0.9, 0.13, 0.9)})
    assert disagreement["verdict"] == "unclear"
    assert recovered["floors"]["full_input_m"] == xs.INIT_FULL_VERTEX_M

    def _eval_run(full_v, full_r2, half_v, half_r2, *, f_event, rank):
        row = run(full_v, full_r2, half_v, half_r2)
        row["full_input"].update(layout={"f_layout": 0.2, "f_event": f_event},
                                 effective_rank=rank,
                                 module_gain_r2={"value": 0.8})
        row["label_fractions"] = {"f_event": f_event, "f_layout": 0.2,
                                  "n_events": 64}
        return row

    runs = {"9000": _eval_run(0.14, 0.9, 0.13, 0.9, f_event=0.5, rank=10),
            "9001": _eval_run(0.15, 0.9, 0.14, 0.9, f_event=0.6, rank=11)}
    readings = xs.r2_readings(runs)
    assert readings["r2_a"]["class"] == "event-dependent"
    assert readings["r2_b"]["verdict"] == "recovered"
    assert readings["r2_c"]["seeds"]["9000"]["label_pass_f_event"] == 0.5
    assert readings["r2_c"]["seeds"]["9000"]["module_gain_r2"] == 0.8


# ==================================================== forbidden paths §2.5
def _boom(*args, **kwargs):
    raise AssertionError("forbidden call path reached")


def test_source_has_no_forbidden_opener_calls():
    source = Path(xs.__file__).read_text()
    for pattern in (r"ensure_dark_bank\s*\(", r"ensure_dataset\s*\(",
                    r"generate_run\s*\(", r"generate_cohort\s*\(",
                    r"ext_static\s*\(", r"ext_repair\s*\("):
        assert not re.search(pattern, source), pattern
    # the only dataset opener is the hash-checked ext_h12 helper; the evaluation
    # and R-1 paths read the pilot's outputs only
    assert "eh.datasets_for(" in source
    cli_source = (ROOT / "scripts" / "jepa_ext_shortcut.py").read_text()
    for pattern in (r"ensure_dark_bank\s*\(", r"ensure_dataset\s*\(",
                    r"generate_run\s*\("):
        assert not re.search(pattern, cli_source), pattern
    assert "eh.load_probes(" in cli_source


def test_forbidden_call_paths_are_not_reached_by_the_setup_paths(tmp_path,
                                                                 monkeypatch):
    """§2.5 sentinels: data generation, dark bank and other gates are never hit."""
    calls = []

    class _Dataset:
        def __init__(self, key):
            self.key = key

    monkeypatch.setattr(lr, "ensure_dark_bank", _boom)
    monkeypatch.setattr(lb, "generate_run", _boom)
    monkeypatch.setattr(xs.eh, "datasets_for", _boom)
    monkeypatch.setattr(xs.eh, "load_probes", _boom)
    monkeypatch.setattr(ef, "assert_authorised", _boom)
    monkeypatch.setattr(ef, "train_t1", _boom)
    monkeypatch.setattr(ef, "assemble_report", _boom)
    for name in ("readout_a1", "readout_a1b", "readout_a2", "readout_a3",
                 "readout_a4"):
        if hasattr(ef, name):
            monkeypatch.setattr(ef, name, _boom)
    monkeypatch.setattr(lr, "ensure_dataset",
                        lambda **kwargs: calls.append(kwargs) or _Dataset("k"))
    # the light setup paths of preflight / R-1 / r1 / the caps report must not
    # touch any forbidden path (the data openers are sentinelled above)
    plan = xs.preflight(ROOT, dry_run=True, log=lambda *a, **k: None)
    assert "data" in plan and plan["s1"]["id"] == "S1"
    assert xs.s1_check(ROOT)["ceiling_hours"] == 160.0
    assert xs.cost_projection()["cap_hours"] == 1.5
    assert xs.entries_preflight()["registered"] is False
    assert xs.r1_payload(ROOT, results_dir=tmp_path / "missing", dry_run=True,
                         log=lambda *a, **k: None)["status"] == "not_run"
    assert calls == []          # the unapproved ensure_dataset path was not used
    # the gate is called from the library entry points (not only the CLI): before
    # the acceptance record exists the gate raises SystemExit(2) without a
    # dry-run flag.  The repository now carries its own acceptance record, so the
    # pending state is pinned explicitly here (state-independent test); whether a
    # real accepted tree grants is covered by test_gate_positive_and_pending_acceptance.
    monkeypatch.setattr(xs, "acceptance_record", lambda root=None: None)
    for call in (lambda: xs.r1_payload(ROOT, results_dir=tmp_path / "missing",
                                       log=lambda *a, **k: None),
                 lambda: xs.assemble_report(root=ROOT, data_dir=tmp_path,
                                            results_dir=tmp_path / "out",
                                            log=lambda *a, **k: None),
                 # the gate raises before any of these arguments is used
                 lambda: xs.reproduce(None, None, None, None, tmp_path / "items.json",
                                      device="cpu"),
                 lambda: xs.evaluate_run_dirs([], None, None, {}, {}, device="cpu"),
                 lambda: xs.manipulation_run(None, None, None, None,
                                             tmp_path / "no-data")):
        with pytest.raises(SystemExit) as caught:
            call()
        assert caught.value.code == 2


# ======================================================= gate §9.2 negative cases
def _git(root, *args):
    import subprocess
    return subprocess.run(["git", *args], cwd=str(root), text=True,
                          capture_output=True)


RUNBOOK_TEXT = ("# Combined runbook\n\n## 0. Authority\n\n> \"" + xs.AUTHOR_LINE
                + "\"\n\n## 1. Background\n")


def _fake_repo(tmp_path, *, acceptance=True, review_verdict="PASS",
               addendum_record=True):
    """A pinned fake repo: the gate must not depend on the real repo's state.

    The committed addendum (the real, unchanged addendum file) is copied in so
    its sha256 equals ``xs.ADDENDUM_SHA256``; the fake acceptance record carries
    the addendum §7 fields unless ``addendum_record`` is False.
    """
    root = tmp_path / "root"
    (root / "docs/jepa").mkdir(parents=True)
    (root / xs.RUNBOOK_REL).write_text(RUNBOOK_TEXT)
    shutil.copyfile(ROOT / xs.ADDENDUM_REL, root / xs.ADDENDUM_REL)
    (root / "src/jepa_noise").mkdir(parents=True)
    (root / "src/jepa_noise/ext_h12.py").write_text("# tracked P file\n")
    (root / "src/noise_module_lucid").mkdir(parents=True)
    (root / "src/noise_module_lucid/readout.py").write_text("# tracked P dir file\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@example.com", "-c", "user.name=test",
         "commit", "-q", "-m", "init")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if acceptance:
        record = {"runbook_sha256": lr.sha256_file(root / xs.RUNBOOK_REL),
                  "impl_commit": head, "review_path": "docs/jepa/REVIEW.md",
                  "review_verdict": review_verdict, "lead": "lead",
                  "date": "2026-11-07"}
        if addendum_record:
            record.update(addendum_path=xs.ADDENDUM_REL,
                          addendum_sha256=xs.ADDENDUM_SHA256)
        (root / xs.ACCEPTANCE_REL).write_text(json.dumps(record))
    return root, head


def test_gate_positive_and_pending_acceptance(tmp_path):
    root, _head = _fake_repo(tmp_path)
    result = xs.assert_authorised("preflight", root=root)
    assert result["authorised"] is True and result["status"] == "granted"
    assert result["checks"]["a"]["pass"] is True
    assert result["checks"]["c"]["pass"] is True
    # pre-acceptance: the record is absent -> gate_pending_acceptance, no raise
    pending_root, _ = _fake_repo(tmp_path / "pending", acceptance=False)
    pending = xs.assert_authorised("preflight", root=pending_root, dry_run=True)
    assert pending["status"] == "gate_pending_acceptance"
    assert pending["authorised"] is False
    with pytest.raises(SystemExit) as caught:
        xs.assert_authorised("preflight", root=pending_root)
    assert caught.value.code == 2


def test_gate_a_runbook_sha_and_author_line(tmp_path):
    root, _head = _fake_repo(tmp_path)
    runbook = root / xs.RUNBOOK_REL
    runbook.write_text(runbook.read_text() + "\nappendix after acceptance\n")
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["checks"]["a"]["pass"] is False
    with pytest.raises(SystemExit) as caught:
        xs.assert_authorised("preflight", root=root)
    assert caught.value.code == 2
    # the sha can match while the §0 author line is missing: (a) still fails
    root2, _ = _fake_repo(tmp_path / "no-line")
    runbook2 = root2 / xs.RUNBOOK_REL
    runbook2.write_text("# Combined runbook\n\n## 0. Authority\n\nnothing here\n\n"
                        "## 1. Background\n")
    record = json.loads((root2 / xs.ACCEPTANCE_REL).read_text())
    record["runbook_sha256"] = lr.sha256_file(runbook2)
    (root2 / xs.ACCEPTANCE_REL).write_text(json.dumps(record))
    result = xs.assert_authorised("preflight", root=root2, dry_run=True)
    assert result["checks"]["a"]["runbook_sha256"] == record["runbook_sha256"]
    assert result["checks"]["a"]["author_line_in_section_zero"] is False
    assert result["checks"]["a"]["pass"] is False


def test_gate_b_acceptance_record_schema_and_verdict(tmp_path):
    for verdict in ("BLOCK", "FAIL", None):
        root, _head = _fake_repo(tmp_path / f"v-{verdict}", review_verdict=verdict)
        with pytest.raises(SystemExit) as caught:
            xs.assert_authorised("preflight", root=root)
        assert caught.value.code == 2
    # a missing key also fails (b)
    root, _head = _fake_repo(tmp_path / "missing-key")
    record = json.loads((root / xs.ACCEPTANCE_REL).read_text())
    record.pop("lead")
    (root / xs.ACCEPTANCE_REL).write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        xs.assert_authorised("preflight", root=root)


def test_gate_c_untracked_and_modified_p_paths(tmp_path):
    root, _head = _fake_repo(tmp_path)
    # untracked <P> implementation file
    untracked = root / "src/jepa_noise/ext_shortcut.py"
    untracked.write_text("# untracked implementation file\n")
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["checks"]["c"]["pass"] is False
    assert "src/jepa_noise/ext_shortcut.py" in result["checks"]["c"]["untracked"]
    with pytest.raises(SystemExit):
        xs.assert_authorised("preflight", root=root)
    untracked.unlink()
    # a tracked but modified <P> file
    tracked = root / "src/jepa_noise/ext_h12.py"
    tracked.write_text(tracked.read_text() + "# change\n")
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["checks"]["c"]["pass"] is False
    _git(root, "checkout", "--", "src/jepa_noise/ext_h12.py")
    # the pathspec covers the <P> directories as well as the <P> files
    paths = result["checks"]["c"]["paths"]
    assert "src/noise_module_lucid/" in paths
    assert "src/jepa_noise/configs/" in paths
    extra = root / "src/noise_module_lucid/extra.py"
    extra.write_text("# new\n")
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["checks"]["c"]["pass"] is False
    assert "src/noise_module_lucid/extra.py" in result["checks"]["c"]["untracked"]


def test_forced_import_classification():
    """B9: test the classifier itself, not the whole pytest process's closure.

    In the full suite other extensions' tests import repository modules outside
    Part B's <P>, so a whole-process "all violations are test files" assertion is
    only valid in a one-file run.  The per-mode closure stays a dry-run check; this
    test pins the forced-import classification deterministically.
    """
    assert xs.in_p("src/jepa_noise/ext_shortcut.py")
    assert xs.in_p("src/noise_module_lucid/readout.py")
    assert xs.in_p(xs.RUNBOOK_REL)
    assert not xs.in_p("src/jepa_noise/ext_view.py")
    # ext_arch/ext_dejitter are the Part B forced transitive imports
    for relative in xs.PART_B_FORCED_TRANSITIVE_IMPORTS:
        assert not xs.in_p(relative)
        assert xs.forced_import(relative)
    # ext_followup's own exceptions stay forced as well
    for relative in ("src/reconstruction_model/model.py",
                     "src/noise_module/core/generator.py"):
        assert not xs.in_p(relative)
        assert xs.forced_import(relative)
    assert not xs.forced_import("src/jepa_noise/ext_view.py")

    def _insert(module_name, relative):
        module = types.ModuleType(module_name)
        module.__file__ = str(ROOT / relative)
        saved = sys.modules.get(module_name)
        sys.modules[module_name] = module
        return saved

    # deterministic synthetic modules exercise the classifier in every context
    cases = (("probe_outside_p", "src/jepa_noise/ext_view.py", "violations"),
             ("probe_forced_arch", "src/jepa_noise/ext_arch.py",
              "part_b_forced_transitive_imports"),
             ("probe_forced_dejitter", "src/jepa_noise/ext_dejitter.py",
              "part_b_forced_transitive_imports"),
             ("probe_forced_recon", "src/reconstruction_model/model.py",
              "forced_import_exceptions"),
             ("probe_forced_noise", "src/noise_module/core/generator.py",
              "forced_import_exceptions"))
    try:
        for module_name, relative, bucket in cases:
            saved = _insert(module_name, relative)
            try:
                closure = xs.import_closure(ROOT)
                assert relative in {row["path"] for row in closure[bucket]}, \
                    (relative, bucket)
            finally:
                if saved is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = saved
    finally:
        for module_name, _relative, _bucket in cases:
            sys.modules.pop(module_name, None)
    # forced imports are never counted as violations (whatever else the pytest
    # process has imported); the modes' own closure is asserted by the dry run
    closure = xs.import_closure(ROOT)
    violations = {row["path"] for row in closure["violations"]}
    forced = ({row["path"] for row in closure["forced_import_exceptions"]}
              | {row["path"] for row in closure["part_b_forced_transitive_imports"]})
    assert not (violations & forced)
    # the classification itself (suite-context independent)
    assert xs.forced_import("src/jepa_noise/ext_arch.py")
    assert xs.forced_import("src/jepa_noise/ext_dejitter.py")
    assert xs.forced_import("src/reconstruction_model/model.py")
    assert xs.forced_import("src/noise_module/core/generator.py")
    assert "src/jepa_noise/ext_arch.py" not in violations
    assert "src/reconstruction_model/model.py" not in violations


# ==================================== R-1 pilot payloads (B7 / B8 / cross-contract / N11)
def _noop(*_args, **_kwargs):
    return None


def _part_a_payloads(generative_r2, generative_rho, *, frozen_r2=0.5, frozen_rho=0.5):
    """Part-A-shaped ``x1.json``/``x2.json`` bodies (``ext_repair`` canonical keys).

    X1 writes ``delta_d_m``/``delta_d_eligible``/``rho_swtw``/``rho_swunit``/
    ``share_cs`` into x1.json; X2 writes ``r2_eq``/``class``/``rho_lr`` into
    x2.json.  One payload alone therefore cannot supply the R-1 reading (B7).
    """
    x1_models, x2_models = {}, {}
    for label in xs.R1_GENERATIVE_CORE:
        x1_models[label] = {"delta_d_m": 0.03, "delta_d_eligible": False,
                            "rho_swtw": 0.9, "rho_swunit": 0.8, "share_cs": 0.8}
        x2_models[label] = {"r2_eq": generative_r2, "class": "equivariant",
                            "rho_lr": generative_rho}
    for seed in (9000, 9001):
        label = f"frozen_target{seed}"
        x1_models[label] = {"delta_d_m": 0.01, "delta_d_eligible": False,
                            "rho_swtw": 0.5, "rho_swunit": 0.5, "share_cs": 0.5}
        x2_models[label] = {"r2_eq": frozen_r2, "class": "partly equivariant",
                            "rho_lr": frozen_rho}
    return {"models": x1_models}, {"models": x2_models}


def _write_pilot(directory, x1, x2, *, flag_key="completed", x1_flag=True,
                 x2_flag=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "x1.json").write_text(json.dumps(x1))
    (directory / "x2.json").write_text(json.dumps(x2))
    (directory / "readings.json").write_text(
        json.dumps({"x1": {flag_key: x1_flag}, "x2": {flag_key: x2_flag}}))
    return directory


def _save_cache(directory, name, embeddings, run_ids, event_ids):
    path = Path(directory) / name
    np.savez(path, embeddings=np.asarray(embeddings, dtype=np.float64),
             run_ids=np.asarray(run_ids, dtype=object),
             event_ids=np.asarray(event_ids, dtype=np.int64))
    (Path(str(path) + ".sha256")).write_text(lr.sha256_file(path) + "  " + name + "\n")
    return path


def test_r1_merges_both_pilot_payloads_and_produces_the_fixed_reading(tmp_path,
                                                                     monkeypatch):
    """B7: X1 fields from x1.json and X2 fields from x2.json are both read."""
    monkeypatch.setattr(xs, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})
    label = xs.R1_GENERATIVE_CORE[0]
    for verdict, generative_r2, generative_rho in (("supports", 0.9, 0.9),
                                                   ("contradicts", 0.2, 0.2)):
        pilot = tmp_path / verdict
        x1, x2 = _part_a_payloads(generative_r2, generative_rho)
        _write_pilot(pilot, x1, x2)
        payload = xs.r1_payload(ROOT, results_dir=pilot, log=_noop)
        assert payload["status"] == "complete"
        row = payload["table"][label]
        # the X2 fields come from x2.json (they are absent from x1.json)
        assert row["r2_eq"] == pytest.approx(generative_r2)
        assert row["rho_lr"] == pytest.approx(generative_rho)
        assert row["x2_class"] == "equivariant"
        # the X1 fields come from x1.json (they are absent from x2.json)
        assert row["rho_swtw"] == pytest.approx(0.9)
        assert row["share_cs"] == pytest.approx(0.8)
        assert row["delta_d"] == pytest.approx(0.03)
        assert row["payloads"] == ["x1", "x2"]
        assert payload["reading"]["generative_median"]["n_models"] == 8
        expected = (["above", "above"] if verdict == "supports"
                    else ["below", "below"])
        assert payload["reading"]["verdict"] == verdict
        assert payload["reading"]["comparisons"] == expected


def test_r1_field_falls_back_to_the_other_pilot_payload(tmp_path, monkeypatch):
    """B7: each field falls back to the other payload when absent."""
    monkeypatch.setattr(xs, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})
    pilot = tmp_path / "pilot"
    x1, x2 = _part_a_payloads(0.9, 0.9)
    label = xs.R1_GENERATIVE_CORE[0]
    x1["models"][label]["r2_eq"] = 0.88          # only in x1
    del x2["models"][label]["r2_eq"]             # force the fallback to x1
    del x1["models"][label]["delta_d_m"]         # absent everywhere
    x1["models"][label]["rho_lr"] = 0.77         # both present: own payload wins
    _write_pilot(pilot, x1, x2)
    row = xs.r1_payload(ROOT, results_dir=pilot, log=_noop)["table"][label]
    assert row["r2_eq"] == pytest.approx(0.88)   # fell back to x1
    assert row["rho_lr"] == pytest.approx(0.9)   # x2 (its own payload) wins
    assert row["delta_d"] is None
    assert row["payloads"] == ["x1", "x2"]


def test_r1_response_size_uses_the_photon_cache_not_the_particle_cache(tmp_path,
                                                                      monkeypatch):
    """B8: S is computed from the photon probe-test embeddings only."""
    monkeypatch.setattr(xs, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})
    pilot = tmp_path / "pilot"
    x1, x2 = _part_a_payloads(0.9, 0.9)
    _write_pilot(pilot, x1, x2)
    label = xs.R1_GENERATIVE_CORE[0]
    cache_dir = tmp_path / "ext" / "x1" / label
    cache_dir.mkdir(parents=True)
    n = np.asarray([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0]])
    d = n + 1.0       # numerator 4.0, N-row denominator 8/9 -> S = 4.5
    _save_cache(cache_dir, "test__N.npz", n, ["r0", "r0", "r0"], [1, 2, 3])
    _save_cache(cache_dir, "test__D.npz", d, ["r0", "r0", "r0"], [1, 2, 3])
    rng = np.random.default_rng(0)
    particle_n = rng.normal(size=(5, 16))
    particle_d = rng.normal(size=(5, 16))
    _save_cache(cache_dir, "particle__test__N.npz", particle_n, ["p"] * 5,
                [1, 2, 3, 4, 5])
    _save_cache(cache_dir, "particle__test__D.npz", particle_d, ["p"] * 5,
                [1, 2, 3, 4, 5])
    found = xs.cache_files(tmp_path / "ext", label)
    assert sorted(found["excluded_particle"]) == ["particle__test__D.npz",
                                                  "particle__test__N.npz"]
    for rows in found["conditions"].values():
        assert rows and all("particle" not in row["path"] for row in rows)
    assert {Path(row["path"]).name for row in found["conditions"]["N"]} == {
        "test__N.npz"}
    first = xs._first_verified(found["conditions"]["N"], None, tmp_path / "ext")
    assert Path(first["path"]).name == "test__N.npz"
    row = xs.r1_payload(ROOT, results_dir=pilot, ext_root=tmp_path / "ext",
                        log=_noop)["table"][label]
    assert row["response_size"] == pytest.approx(4.5)
    assert row["response_size_detail"]["dim"] == 4          # photon dim, not 16
    assert row["response_size_detail"]["n_paired"] == 3
    assert xs.is_particle_cache("particle__test__N.npz") is True
    assert xs.is_particle_cache("test__N.npz") is False


def test_pilot_status_accepts_completed_or_complete(tmp_path):
    """Cross-contract: Part A writes ``completed``; accept ``complete`` too."""
    x1, x2 = _part_a_payloads(0.9, 0.9)
    for key in ("completed", "complete"):
        pilot = _write_pilot(tmp_path / key, x1, x2, flag_key=key)
        status = xs.pilot_status(ROOT, results_dir=pilot)
        assert status["complete"] is True
    pilot = _write_pilot(tmp_path / "incomplete", x1, x2)
    (pilot / "readings.json").write_text(json.dumps({"x1": {"completed": False},
                                                     "x2": {"completed": True}}))
    status = xs.pilot_status(ROOT, results_dir=pilot)
    assert status["complete"] is False
    assert "not completed" in status["reason"]
    (pilot / "readings.json").write_text(json.dumps({
        "x1": {"completed": True, "status": "failed"}, "x2": {"completed": True}}))
    assert xs.pilot_status(ROOT, results_dir=pilot)["complete"] is False


def test_r1_median_uses_exactly_the_eight_core_generative_models():
    """N11: the optional ``mae_within_median`` rows never enter the median."""
    rows = {}
    core_values = [0.5] * 4 + [0.9] * 4
    for label, value in zip(xs.R1_GENERATIVE_CORE, core_values):
        rows[label] = {"status": "available", "r2_eq": value, "rho_lr": value}
    for label in xs.R1_OPTIONAL["generative"]:
        rows[label] = {"status": "available", "r2_eq": 0.0, "rho_lr": 0.0}
    for seed in (9000, 9001):
        rows[f"frozen_target{seed}"] = {"status": "available", "r2_eq": 0.6,
                                        "rho_lr": 0.6}
    reading = xs.r1_reading(rows)
    assert reading["generative_median"]["n_models"] == 8
    assert reading["generative_median"]["r2_eq"] == pytest.approx(0.7)
    assert reading["generative_median"]["rho_lr"] == pytest.approx(0.7)
    # including the two optional zero rows would have moved the median to 0.5
    assert reading["verdict"] == "supports"


# ==================================================================== parser
def test_cli_parses_the_runbook_commands():
    args = shortcut_cli.parser().parse_args(["dry-run", "--out", "/tmp/plan"])
    assert args.mode == "dry-run" and args.data_root == xs.DATA_ROOT_DEFAULT
    train = shortcut_cli.parser().parse_args(["train", "--seed", "9001"])
    assert train.seed == 9001 and train.steps == 10000
    smoke = shortcut_cli.parser().parse_args(["smoke"])
    assert smoke.steps == 20 and smoke.seed == 9000
    evaluate = shortcut_cli.parser().parse_args(["evaluate", "--run", "/a",
                                                 "--run", "/b"])
    assert evaluate.run == ["/a", "/b"]
    report = shortcut_cli.parser().parse_args(["report", "--r1-only"])
    assert report.r1_only is True
    assert set(shortcut_cli.BODIES) == set(xs.ALL_MODES)
    assert set(xs.MODES) <= set(xs.ALL_MODES)
    assert set(xs.ADDENDUM_MODES) <= set(xs.ALL_MODES)
    # addendum modes (7 Nov 2026, v1)
    evaluate = shortcut_cli.parser().parse_args(
        ["addendum-evaluate", "--run", "/ceph/a", "--run", "/ceph/b"])
    assert evaluate.run == ["/ceph/a", "/ceph/b"] and evaluate.device == "auto"
    assert evaluate.out is None
    r1 = shortcut_cli.parser().parse_args(["addendum-r1"])
    assert r1.out is None and r1.device == "cpu"
    addendum_report = shortcut_cli.parser().parse_args(["addendum-report"])
    assert addendum_report.out is None and addendum_report.device == "cpu"
    help_text = shortcut_cli.parser().format_help()
    for mode in xs.ADDENDUM_MODES:
        assert mode in help_text
    # the --help text documents the addendum budget and the closed-cap rule
    evaluate_help = _subparser("addendum-evaluate").format_help()
    assert "0.35" in evaluate_help and "closed Part B" in evaluate_help
    r1_help = _subparser("addendum-r1").format_help()
    assert "read-only" in r1_help and "rho_LR" in r1_help
    report_help = _subparser("addendum-report").format_help()
    assert "ADDENDUM_2026-11-07.json" in report_help


def _subparser(name):
    """The argparse subparser for ``name`` (help-text assertions)."""
    for action in shortcut_cli.parser()._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and name in choices:
            return choices[name]
    raise AssertionError(f"no subparser {name!r}")


# ================================== addendum (7 Nov 2026, v1): contract constants
def test_addendum_contract_constants_and_committed_addendum():
    """The shared contract: the committed addendum sha256 and the caps."""
    assert xs.ADDENDUM_REL == "docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md"
    assert xs.ADDENDUM_SHA256 == (
        "abe0904af0ed0b5a6586cee459c507ae0fa2a7c60c65ffb58192c71b7180621a")
    assert lr.sha256_file(ROOT / xs.ADDENDUM_REL) == xs.ADDENDUM_SHA256
    assert xs.ADDENDUM_GPU_CAP_H == 0.35
    assert xs.ADDENDUM_CPU_CAP_H == 3.0
    assert str(xs.ADDENDUM_JOBS) == (
        "/ceph/dwong/jepa/returns/addendum_20261107_jobs.jsonl")
    assert xs.ADDENDUM_AUTHOR_LINE == "写进weekly report，然后我选A，你写一个小runbook"
    assert len(xs.ACCEPTANCE_KEYS) == 8
    for key in ("addendum_path", "addendum_sha256"):
        assert key in xs.ACCEPTANCE_KEYS
    assert xs.ADDENDUM_REL in xs.P_FILES


def test_addendum_author_line_is_in_its_section_zero():
    """The addendum's §0 block carries the addendum author line verbatim."""
    text = (ROOT / xs.ADDENDUM_REL).read_text()
    block = xs.author_section(text, xs.ADDENDUM_AUTHOR_LINE)
    assert block and "## 0." in block
    assert xs.ADDENDUM_AUTHOR_LINE in block


# ============================ addendum gate §7: the record fields are required
def test_gate_addendum_record_fields_missing_denies_not_pending(tmp_path):
    """A record missing addendum_path/addendum_sha256 must DENY, not be pending."""
    root, _head = _fake_repo(tmp_path, addendum_record=False)
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["status"] == "denied"
    assert result["status"] != "gate_pending_acceptance"
    assert result["authorised"] is False
    assert result["checks"]["b"]["pass"] is False
    assert set(result["checks"]["b"]["missing_keys"]) == {"addendum_path",
                                                         "addendum_sha256"}
    with pytest.raises(SystemExit) as caught:
        xs.assert_authorised("preflight", root=root)
    assert caught.value.code == 2


def test_gate_addendum_sha_path_and_unchanged_file(tmp_path):
    root, _head = _fake_repo(tmp_path)
    granted = xs.assert_authorised("preflight", root=root)
    assert granted["authorised"] is True and granted["status"] == "granted"
    a = granted["checks"]["a"]
    assert a["addendum_sha256"] == xs.ADDENDUM_SHA256
    assert a["addendum_sha256_pass"] is True
    assert a["addendum_author_line_in_section_zero"] is True
    assert a["addendum_path_match"] is True
    record_path = root / xs.ACCEPTANCE_REL
    record = json.loads(record_path.read_text())

    # wrong addendum_sha256 in the record -> (a) fails, gate denies
    record["addendum_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record))
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["checks"]["a"]["addendum_sha256_pass"] is False
    assert result["checks"]["a"]["pass"] is False
    with pytest.raises(SystemExit) as caught:
        xs.assert_authorised("preflight", root=root)
    assert caught.value.code == 2

    # wrong addendum_path -> (a) fails
    record["addendum_sha256"] = xs.ADDENDUM_SHA256
    record["addendum_path"] = "docs/jepa/SOMETHING_ELSE.md"
    record_path.write_text(json.dumps(record))
    assert xs.assert_authorised("preflight", root=root,
                                dry_run=True)["checks"]["a"]["addendum_path_match"] is False
    with pytest.raises(SystemExit):
        xs.assert_authorised("preflight", root=root)

    # the committed addendum changed -> sha mismatch and (c) diff fails
    record["addendum_path"] = xs.ADDENDUM_REL
    record_path.write_text(json.dumps(record))
    addendum = root / xs.ADDENDUM_REL
    addendum.write_text(addendum.read_text() + "\nappendix after acceptance\n")
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["checks"]["a"]["addendum_sha256"] != xs.ADDENDUM_SHA256
    assert result["checks"]["a"]["addendum_sha256_pass"] is False
    assert result["checks"]["c"]["pass"] is False
    assert result["checks"]["c"]["diff_clean"] is False
    with pytest.raises(SystemExit):
        xs.assert_authorised("preflight", root=root)


def test_pre_acceptance_dry_run_returns_the_checks_without_raising(tmp_path):
    root, _head = _fake_repo(tmp_path, acceptance=False)
    result = xs.assert_authorised("preflight", root=root, dry_run=True)
    assert result["status"] == "gate_pending_acceptance"
    assert result["authorised"] is False
    assert set(result["checks"]) >= {"a", "b", "c"}
    assert result["checks"]["a"]["addendum_sha256"] == xs.ADDENDUM_SHA256
    # a committed record that is missing the addendum fields is denied, but the
    # dry run still returns the checks (only closure/setup failures exit 2)
    denied_root, _ = _fake_repo(tmp_path / "denied", addendum_record=False)
    denied = xs.assert_authorised("preflight", root=denied_root, dry_run=True)
    assert denied["status"] == "denied"
    assert denied["checks"]["b"]["pass"] is False


# ================================= addendum §6: the shared budget and its ledger
def test_addendum_budget_ledger_one_line_per_job_and_hours(tmp_path):
    ledger = tmp_path / "addendum_jobs.jsonl"
    xs.record_addendum_job("A", mode="est-latent", kind="gpu", wall_seconds=216.0,
                           path=ledger)
    xs.record_addendum_job("B", mode="addendum-evaluate", kind="gpu",
                           wall_seconds=360.0, path=ledger)
    xs.record_addendum_job("B", mode="addendum-r1", kind="cpu", wall_seconds=36.0,
                           path=ledger)
    lines = ledger.read_text().splitlines()
    assert len(lines) == 3                       # one JSON line per job (O_APPEND)
    expected = {"utc", "part", "mode", "kind", "wall_seconds", "hours"}
    for line in lines:
        row = json.loads(line)
        assert set(row) == expected
        assert row["kind"] in ("gpu", "cpu")
        assert row["hours"] == pytest.approx(row["wall_seconds"] / 3600.0)
    hours = xs.addendum_hours(path=ledger)
    assert hours["n_rows"] == 3
    assert hours["gpu_hours"] == pytest.approx((216.0 + 360.0) / 3600.0)
    assert hours["cpu_hours"] == pytest.approx(36.0 / 3600.0)
    assert hours["parts"] == ["A", "B"]
    assert hours["gpu_cap_hours"] == 0.35 and hours["cpu_cap_hours"] == 3.0


def test_addendum_budget_guard_refuses_a_job_that_would_cross_the_cap(tmp_path,
                                                                     capsys):
    ledger = tmp_path / "addendum_jobs.jsonl"
    xs.record_addendum_job("B", mode="addendum-evaluate", kind="gpu",
                           wall_seconds=0.30 * 3600.0, path=ledger)
    # exactly at the cap is allowed: 0.30 + 0.05 == 0.35
    allowed = xs.assert_addendum_budget("gpu", 0.05, part="B", path=ledger)
    assert allowed["pass"] is True
    assert allowed["projected_total_hours"] == pytest.approx(0.35)
    with pytest.raises(SystemExit) as caught:
        xs.assert_addendum_budget("gpu", 0.06, part="B", path=ledger)
    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "not_run (cap)" in stderr and "0.35" in stderr
    # the CPU cap is 3.0 wall-hours
    xs.record_addendum_job("A", mode="x2-latent", kind="cpu",
                           wall_seconds=2.99 * 3600.0, path=ledger)
    with pytest.raises(SystemExit) as caught:
        xs.assert_addendum_budget("cpu", 0.02, part="B", path=ledger)
    assert caught.value.code == 2
    assert "not_run (cap)" in capsys.readouterr().err
    with pytest.raises(ValueError):
        xs.assert_addendum_budget("wall", 0.01)


def test_addendum_s1_uses_only_the_addendum_cap():
    result = xs.addendum_s1_check(ROOT)
    assert result["pass"] is True
    assert result["addendum_cap_hours"] == 0.35
    assert result["sum_hours"] == pytest.approx(
        float(result["ledger_charged_gpu_hours"]) + 0.35)
    assert "closed Part A and Part B caps are not consulted" in result["rule"]


# ===================================== D5: the R2 readings and the not_run label
def _eval_run(*, f_event=0.02, full_vertex=0.15, half_vertex=0.16,
              full_r2=0.8, half_r2=0.8, label_f_event=0.01):
    """A synthetic, evaluation-shaped run entry (Part B §5.4 fields)."""
    return {
        "run_id": "dev_s005_within__sigreg_jepa__s9000",
        "run_dir": "/ceph/dwong/jepa/returns/ext_shortcut_20261107/train/runs/x",
        "checkpoint_sha256": "a" * 64,
        "full_input": {"vertex_resolution_m": full_vertex,
                       "token_log_charge_r2": {"r2": full_r2},
                       "effective_rank": 40.0,
                       "module_gain_r2": {"value": 0.9},
                       "layout": {"f_event": f_event, "f_layout": 0.1}},
        "half_pass": {"vertex_resolution_m": half_vertex,
                      "token_log_charge_r2": {"r2": half_r2},
                      "effective_rank": 30.0,
                      "module_gain_r2": {"value": 0.8},
                      "layout": None},
        "label_fractions": {"f_event": label_f_event, "f_layout": 0.05,
                            "n_events": 64},
    }


def test_r2_readings_are_not_run_without_an_evaluation():
    """Addendum §6 label fix: an empty evaluation is ``not_run``, never single-seed."""
    for runs in ({}, {"9000": {"run_id": "x"}},
                 {"9000": {"full_input": {}, "half_pass": {}}},
                 {"9000": None}):
        readings = xs.r2_readings(runs)
        assert readings["status"] == "not_run"
        assert readings["r2_a"]["class"] == "not_run"
        assert readings["r2_a"]["status"] == "not_run"
        assert readings["r2_b"]["verdict"] == "not_run"
        assert readings["r2_c"]["status"] == "not_run"
        assert readings["r2_c"]["seeds"] == {}
        assert "single-seed (not classified)" not in json.dumps(readings)
        assert "not_run" in json.dumps(readings)


def test_r2_readings_follow_the_fixed_rules_when_evaluated():
    runs = {"9000": _eval_run(f_event=0.02), "9001": _eval_run(f_event=0.03)}
    readings = xs.r2_readings(runs)
    assert readings["status"] == "complete"
    assert readings["r2_a"]["class"] == "event-dependent"
    assert readings["r2_a"]["seeds"] == [9000, 9001]
    assert readings["r2_b"]["verdict"] == "recovered"
    assert readings["r2_c"]["seeds"]["9000"]["full_input_vertex_m"] == 0.15
    assert readings["r2_c"]["seeds"]["9000"]["label_pass_f_event"] == 0.01
    assert readings["r2_c"]["seeds"]["9001"]["half_pass_token_r2"] == 0.8
    # event-invariant and a seed disagreement (the A-1 rule is unchanged)
    invariant = {"9000": _eval_run(f_event=5e-4), "9001": _eval_run(f_event=5e-4)}
    assert xs.r2_readings(invariant)["r2_a"]["class"] == "event-invariant"
    intermediate = {"9000": _eval_run(f_event=0.02),
                    "9001": _eval_run(f_event=5e-4)}
    assert xs.r2_readings(intermediate)["r2_a"]["class"] == "intermediate"
    # the §4 recovery rule on a non-recovering pair
    not_recovered = {"9000": _eval_run(full_vertex=0.20),
                     "9001": _eval_run(full_vertex=0.21)}
    assert xs.r2_readings(not_recovered)["r2_b"]["verdict"] == "not recovered"


def test_r2_evaluation_record_is_the_addendum_d5_record():
    runs = {"9000": _eval_run(), "9001": _eval_run(f_event=0.03)}
    record = xs.r2_evaluation_record(runs, run_dirs=["/ceph/a", "/ceph/b"],
                                     evaluation="/ceph/eval.json")
    assert record["id"] == "D5" and record["status"] == "complete"
    assert record["seeds"] == ["9000", "9001"] and record["n_runs"] == 2
    assert "ext_h12.evaluate_run" in record["protocol"]
    assert record["readings"]["r2_a"]["class"] == "event-dependent"
    assert record["statistic_audit"]["pass"] is True
    assert xs.r2_evaluation_record({})["status"] == "not_run"


# ============================== D2: the mean-vs-RMS statistic audit (Part B side)
def test_d2_vertex_point_is_the_mean_error():
    """The vertex point is the mean 3-D error, not the RMS (addendum D2)."""
    predictions = np.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    truth = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    scored = lr._vertex_error(predictions, truth)
    errors = np.linalg.norm(predictions - truth, axis=1)
    assert scored["resolution"] == pytest.approx(float(errors.mean()))
    assert scored["resolution"] != pytest.approx(float(np.sqrt(np.mean(errors ** 2))))
    assert scored["median"] == pytest.approx(float(np.median(errors)))
    assert "errors.mean()" in xs.statistic_audit({})["vertex_point_source"]


def test_d2_statistic_audit_flags_a_mean_point_with_an_rms_interval():
    good = {"9000": _eval_run(), "9001": _eval_run()}
    audit = xs.statistic_audit(good)
    assert audit["pass"] is True and audit["findings"] == []
    assert audit["vertex_point_statistic"] == "mean"
    assert audit["decisions"][0]["id"] == "R2-a"
    assert all(row["interval_paired"] is False for row in audit["decisions"])

    # the addendum D2 defect: a mean vertex point beside an RMS bootstrap interval
    bad = {"9000": _eval_run()}
    bad["9000"]["full_input"]["vertex_resolution_m_ci95"] = {
        "statistic": "rms", "low": 0.1147, "high": 0.1208}
    audit = xs.statistic_audit(bad)
    assert audit["pass"] is False
    finding = audit["findings"][0]
    assert finding["point_statistic"] == "mean"
    assert finding["interval_statistic"] == "rms"
    assert "point statistic 'mean'" in finding["reason"]
    assert "rms" in finding["reason"]

    # a mean interval beside the mean point is consistent
    consistent = {"9000": _eval_run()}
    consistent["9000"]["full_input"]["vertex_resolution_m_ci95"] = {
        "statistic": "mean", "low": 0.14, "high": 0.16}
    assert xs.statistic_audit(consistent)["pass"] is True

    # a nested interval is checked against its own point metric (token R²)
    nested = {"9000": _eval_run()}
    nested["9000"]["full_input"]["token_log_charge_r2"]["ci95"] = {
        "statistic": "r2", "low": 0.7, "high": 0.9}
    assert xs.statistic_audit(nested)["pass"] is True
    nested["9000"]["full_input"]["token_log_charge_r2"]["ci95"] = {
        "statistic": "rms", "low": 0.1, "high": 0.2}
    assert xs.statistic_audit(nested)["pass"] is False


# ======================= addendum §2: the R-1 rerun with the corrected rho_LR
def test_addendum_r1_rerun_reads_the_corrected_pilot_rho_lr(tmp_path, monkeypatch):
    monkeypatch.setattr(xs, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})
    x1, x2 = _part_a_payloads(0.9, 0.9)
    # the D1-corrected pilot writes a very different rho_LR for every model
    for generative in xs.R1_GENERATIVE_CORE:
        x2["models"][generative]["rho_lr"] = -0.42
    for seed in (9000, 9001):
        x2["models"][f"frozen_target{seed}"]["rho_lr"] = -0.12
    pilot = _write_pilot(tmp_path / "pilot-d1", x1, x2)
    payload = xs.r1_payload(ROOT, results_dir=pilot, ext_root=tmp_path / "ext",
                            mode="addendum-r1", log=_noop)
    assert payload["status"] == "complete"
    reading = payload["reading"]
    assert reading["generative_median"]["rho_lr"] == pytest.approx(-0.42)
    assert reading["frozen_target"]["rho_lr"]["9000"] == pytest.approx(-0.12)
    assert reading["frozen_target"]["rho_lr"]["9001"] == pytest.approx(-0.12)
    # the §3 rule and families are unchanged: the verdict follows the corrected
    # values (R²_eq above both seeds, ρ_LR below both -> mixed)
    assert reading["verdict"] == "mixed"


def test_addendum_superseded_lists_the_replaced_rho_lr_and_empty_r2_values():
    old_readings = {
        "r2_a": {"class": "single-seed (not classified)"},
        "r2_b": {"verdict": "unclear"},
        "r2_c": {"seeds": {}},
        "r1": {"reading": {"generative_median": {"rho_lr": -0.67786},
                           "frozen_target": {"rho_lr": {"9000": -0.53384,
                                                        "9001": -0.586497}}}},
    }
    new_readings = {"r2_a": {"class": "event-dependent"},
                    "r2_b": {"verdict": "recovered"},
                    "r2_c": {"seeds": {"9000": {}, "9001": {}}}}
    r1 = {"status": "complete",
          "reading": {"generative_median": {"rho_lr": -0.42},
                      "frozen_target": {"rho_lr": {"9000": -0.12,
                                                   "9001": -0.13}}}}
    rows = xs.addendum_superseded(old_readings, {}, new_readings, r1)
    quantities = [row["quantity"] for row in rows]
    assert "R-1 generative-median ρ_LR" in quantities
    assert "R-1 frozen_target9000 ρ_LR" in quantities
    assert "R-1 frozen_target9001 ρ_LR" in quantities
    assert "R2-a class" in quantities and "R2-b verdict" in quantities
    assert "R2-c per-seed rows" in quantities
    rho = next(row for row in rows if row["quantity"] == "R-1 generative-median ρ_LR")
    assert rho["superseded_7_nov"] == pytest.approx(-0.67786)
    assert rho["new_value"] == pytest.approx(-0.42)
    assert rho["defect"] == "D1" and rho["changed"] is True
    # empty R2 rows are not "replaced" when the evaluation is still missing
    rows = xs.addendum_superseded(old_readings, {}, old_readings, old_readings["r1"])
    assert "R2-a class" not in [row["quantity"] for row in rows]


# ============================ addendum §5/§10: the in-place report (D4 for Part B)
def _write_addendum_inputs(tmp_path):
    ext = tmp_path / "ext"
    (ext / "eval").mkdir(parents=True)
    runs = {"9000": _eval_run(), "9001": _eval_run(f_event=0.03,
                                                   full_vertex=0.14)}
    runs["9000"]["run_dir"] = "/ceph/dwong/jepa/returns/ext_shortcut_20261107/" \
                              "train/runs/dev_s005_within__sigreg_jepa__s9000"
    runs["9001"]["run_dir"] = "/ceph/dwong/jepa/returns/ext_shortcut_20261107/" \
                              "train/runs/dev_s005_within__sigreg_jepa__s9001"
    evaluation = {"runs": runs,
                  "run_dirs": [runs["9000"]["run_dir"], runs["9001"]["run_dir"]]}
    (ext / "eval" / "addendum_eval.json").write_text(json.dumps(evaluation))
    reading = {
        "generative_median": {"rho_lr": -0.42, "r2_eq": 0.31, "n_models": 8},
        "frozen_target": {"rho_lr": {"9000": -0.12, "9001": -0.13},
                          "r2_eq": {"9000": 0.7, "9001": 0.72}},
        "verdict": "contradicts", "caveat": "descriptive, not a test",
    }
    r1 = {"registered": False, "extension": xs.EXTENSION, "id": "R-1",
          "status": "complete", "results_dir": "/ceph/pilot/results",
          "reading": reading, "table": {}, "n_models": 19}
    (ext / "r1_addendum.json").write_text(json.dumps(r1))
    results = tmp_path / "results"
    results.mkdir()
    old_r1 = {"reading": {"generative_median": {"rho_lr": -0.67786},
                          "frozen_target": {"rho_lr": {"9000": -0.53384,
                                                       "9001": -0.586497}}}}
    (results / "readings.json").write_text(json.dumps(
        {"registered": False, "extension": xs.EXTENSION,
         "variant": xs.VARIANT,
         "r2_a": {"class": "single-seed (not classified)"},
         "r2_b": {"verdict": "unclear"}, "r2_c": {"seeds": {}}, "r1": old_r1}))
    (results / "OUTCOME.json").write_text(json.dumps(
        {"registered": False, "extension": xs.EXTENSION, "status": "executed",
         "items": {"evaluate": "not_run (cap)"},
         "r1_reading": old_r1["reading"],
         "resources": {"gpu_hours": 1.5651}}))
    (results / "runs.json").write_text(json.dumps(
        {"registered": False, "extension": xs.EXTENSION, "runs": {},
         "readings": {"r2_a": {"class": "single-seed (not classified)"}}}))
    return ext, results


def test_addendum_report_updates_the_part_b_results_in_place(tmp_path, monkeypatch):
    ext, results = _write_addendum_inputs(tmp_path)
    monkeypatch.setattr(xs, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})
    monkeypatch.setattr(xs, "ADDENDUM_JOBS", tmp_path / "addendum_jobs.jsonl")
    xs.record_addendum_job("B", mode="addendum-evaluate", kind="gpu",
                           wall_seconds=180.0, path=xs.ADDENDUM_JOBS)
    xs.record_addendum_job("B", mode="addendum-r1", kind="cpu",
                           wall_seconds=30.0, path=xs.ADDENDUM_JOBS)
    payload = xs.assemble_addendum_report(
        root=ROOT, data_dir=ext, results_dir=results, ext_root=ext,
        budget={"pass": True}, log=_noop)
    assert payload["r2_status"] == "complete"
    assert payload["r1_status"] == "complete"
    assert payload["complete"] is True
    for name in ("README.md", "readings.json", "report.json", "runs.json",
                 "OUTCOME.json", xs.ADDENDUM_RESULTS_FILE):
        assert (results / name).is_file(), name
    readings = json.loads((results / "readings.json").read_text())
    assert readings["r2_a"]["class"] == "event-dependent"
    assert readings["r2_b"]["verdict"] == "recovered"
    assert readings["r2_c"]["status"] == "complete"
    assert readings["r1"]["reading"]["generative_median"]["rho_lr"] == pytest.approx(-0.42)
    assert readings["addendum"]["sha256"] == xs.ADDENDUM_SHA256
    assert readings["addendum"]["complete"] is True
    quantities = [row["quantity"] for row in readings["addendum"]["superseded"]]
    assert "R-1 generative-median ρ_LR" in quantities
    assert "R2-a class" in quantities and "R2-b verdict" in quantities
    # runs.json is updated with the two evaluated run directories
    runs_payload = json.loads((results / "runs.json").read_text())
    assert sorted(runs_payload["runs"]) == ["9000", "9001"]
    assert runs_payload["runs"]["9000"]["full_input"]["vertex_resolution_m"] == 0.15
    # OUTCOME.json keeps the 7 Nov record and gains the addendum record in place
    outcome = json.loads((results / "OUTCOME.json").read_text())
    assert outcome["addendum"]["sha256"] == xs.ADDENDUM_SHA256
    assert outcome["addendum"]["path"] == xs.ADDENDUM_REL
    assert outcome["r1_reading"]["verdict"] == "contradicts"
    assert outcome["resources"]["addendum"]["gpu_hours"] == pytest.approx(0.05)
    assert outcome["resources"]["addendum"]["cpu_hours"] == pytest.approx(
        0.0083)                       # 30 s rounded to four decimals like job_hours
    # the ADDENDUM file carries the D5 R-2 evaluation record
    addendum = json.loads((results / xs.ADDENDUM_RESULTS_FILE).read_text())
    assert addendum["part"] == "B"
    assert addendum["items"]["D5_r2_evaluation"]["id"] == "D5"
    assert addendum["items"]["D5_r2_evaluation"]["seeds"] == ["9000", "9001"]
    assert addendum["items"]["D5_r2_evaluation"]["readings"]["r2_a"]["class"] == (
        "event-dependent")
    assert addendum["items"]["D2_statistic_audit"]["pass"] is True
    assert "no figures" in addendum["figures"]
    # the README states the D4 readings completeness, the superseded table and
    # that Part B has no figures
    readme = (results / "README.md").read_text()
    for needle in ("Addendum (7 Nov 2026", "Superseded 7 Nov values",
                   "Readings completeness (Part B)", "Part B produces **no figures**",
                   "generative-median ρ_LR", "D5 R-2 evaluation"):
        assert needle in readme, needle


# ======================= addendum modes never consult the closed Part B caps
def _function_source(source, name):
    start = source.index(f"def {name}(")
    rest = source.index("\ndef ", start + 1)
    return source[start:rest]


def test_addendum_modes_do_not_consult_the_closed_part_b_caps():
    cli_source = (ROOT / "scripts" / "jepa_ext_shortcut.py").read_text()
    for name in xs.ADDENDUM_MODES:
        function = {"addendum-evaluate": "mode_addendum_evaluate",
                    "addendum-r1": "mode_addendum_r1",
                    "addendum-report": "mode_addendum_report"}[name]
        body = _function_source(cli_source, function)
        assert "xs.assert_addendum_budget(" in body, name
        assert "xs.assert_gpu_budget(" not in body, name
        assert "xs.assert_cpu_budget(" not in body, name
        assert "xs.s1_check_or_stop(" not in body, name
        assert "xs.GPU_CAP_H" not in body and "xs.CPU_CAP_H" not in body, name
    # the addendum budget helper reads the addendum caps only
    library = (ROOT / "src" / "jepa_noise" / "ext_shortcut.py").read_text()
    helper = _function_source(library, "assert_addendum_budget")
    assert "ADDENDUM_GPU_CAP_H" in helper and "ADDENDUM_CPU_CAP_H" in helper
    stripped = helper.replace("ADDENDUM_GPU_CAP_H", "").replace(
        "ADDENDUM_CPU_CAP_H", "")
    assert "GPU_CAP_H" not in stripped and "CPU_CAP_H" not in stripped
