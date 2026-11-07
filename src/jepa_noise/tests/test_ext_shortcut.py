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
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_followup as ef  # noqa: E402
from jepa_noise import ext_h12 as eh  # noqa: E402
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
    # the gate is called from the library entry points (not only the CLI): the
    # pre-acceptance gate raises SystemExit(2) without a dry-run flag
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


def _fake_repo(tmp_path, *, acceptance=True, review_verdict="PASS"):
    root = tmp_path / "root"
    (root / "docs/jepa").mkdir(parents=True)
    (root / xs.RUNBOOK_REL).write_text(RUNBOOK_TEXT)
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
    assert xs.in_p("src/jepa_noise/ext_shortcut.py")
    assert xs.in_p("src/noise_module_lucid/readout.py")
    assert xs.in_p(xs.RUNBOOK_REL)
    assert not xs.in_p("src/jepa_noise/ext_view.py")
    # ext_arch/ext_dejitter are the Part B forced transitive imports
    for relative in xs.PART_B_FORCED_TRANSITIVE_IMPORTS:
        assert not xs.in_p(relative)
        assert xs.forced_import(relative)
    assert xs.forced_import("src/reconstruction_model/__init__.py")
    # the real closure passes once the pytest modules are set aside
    closure = xs.import_closure(ROOT)
    assert closure["pass"] is False or closure["violations"] == []
    non_test = [row for row in closure["violations"]
                if not row["path"].startswith("src/jepa_noise/tests/")]
    assert non_test == []
    assert any(row["path"] == "src/jepa_noise/ext_arch.py"
               for row in closure["part_b_forced_transitive_imports"])
    assert any(row["path"].startswith("src/reconstruction_model/")
               for row in closure["forced_import_exceptions"])


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
    assert set(shortcut_cli.BODIES) == set(xs.MODES)
