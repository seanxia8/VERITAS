# SPDX-License-Identifier: MIT
"""Acceptance tests for the MAE/JEPA follow-up extension (runbook 3 Nov 2026 §6.3).

Covers: the (a)-(e) gate negative cases; the executed-import closure; the pinned
hash checks; the A0 projection and inventory; the s = 0.30 read-only openers and
every forbidden call path of §2; the A-1b marginal reader (no ``ensure_dataset``),
the within/across membership facts, the ``across_mask`` offset uniformity, the
scale300k hash/A0 domain assertion, the committed fix9000 C-group reproduction
and the label-row collector reproduction; A-2 pairing; the A-4 invariance test;
the T-1 cell/keyword-argument equality against ``jepa_ext_dejitter.mode_train``
and the T-1 host manifest comparison; the anc100 dark-bank ordering; hash
mismatch; and no overwrite.

Data-dependent tests are skipped with a reason when the pinned datasets (or the
committed label cache/checkpoint) are absent; on deepthought2 they actually run.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_arch as ea  # noqa: E402
from jepa_noise import ext_dejitter as ed  # noqa: E402
from jepa_noise import ext_followup as ef  # noqa: E402
from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import lucid_views as lv  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_followup", ROOT / "scripts" / "jepa_ext_followup.py")
followup_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(followup_cli)

DATA_ROOT = os.environ.get("JEPA_DATA_ROOT", ef.DATA_ROOT_DEFAULT)
PROBE_TEST_PIN = ef.manifest_path(DATA_ROOT, ef.PROBE_KEYS_S005["test"])
DATA_PRESENT = PROBE_TEST_PIN.is_file()
FIX_CHECKPOINT = ef.READOUT_EXT / "ckpt" / "fix9000.pt"
LABEL_CACHE_PRESENT = ef.A1B_LABEL_CACHE.is_file()
COLLECTOR_PRESENT = bool(DATA_PRESENT and FIX_CHECKPOINT.is_file() and LABEL_CACHE_PRESENT)
needs_data = pytest.mark.skipif(
    not DATA_PRESENT, reason=f"pinned probe-test manifest absent ({PROBE_TEST_PIN})")
needs_collector = pytest.mark.skipif(
    not COLLECTOR_PRESENT,
    reason="committed fix9000 label cache / checkpoint absent on this host")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def small_geometry():
    return lb.from_synthetic({"module_ids": tuple(range(6)),
                              "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                              "pmts_per_module": 19})


@pytest.fixture(scope="module")
def full_geometry():
    """The real 1 995-PMT layout: 105 modules, dense module_index 0..104."""
    return lb.from_synthetic({"module_ids": tuple(range(105)),
                              "surface_counts": {"top": 399, "barrel": 1197,
                                                 "bottom": 399},
                              "pmts_per_module": 19})


@pytest.fixture(scope="module")
def config():
    return lr.load_run_config(ROOT / ef.CONFIG_REL)


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=str(root), text=True,
                          capture_output=True)


def _fake_repo(tmp_path, *, runbook_text=None, decisions_text=None,
               acceptance=None, commit=True):
    """A throw-away repository with the committed runbook/decisions and a record."""
    root = tmp_path / "repo"
    (root / "docs/jepa").mkdir(parents=True)
    text = runbook_text if runbook_text is not None else (
        "## 0. Authority\n\n> \"" + ef.AUTHOR_LINE + "\"\n\n## 1. Next\n")
    (root / ef.RUNBOOK_REL).write_text(text)
    (root / ef.DECISIONS_REL).write_text(
        decisions_text if decisions_text is not None else "history\n" + ef.AUTHOR_LINE)
    sha = lr.sha256_file(root / ef.RUNBOOK_REL)
    record = acceptance if acceptance is not None else {
        "runbook_sha256": sha, "impl_commit": None, "review_path": "docs/jepa/REVIEW.md",
        "review_verdict": "PASS", "lead": "lead", "date": "2026-11-03"}
    if commit:
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "-c", "user.email=t@example.com", "-c",
             "user.name=test", "commit", "-q", "-m", "init")
        record = dict(record)
        record["impl_commit"] = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / ef.ACCEPTANCE_REL).write_text(json.dumps(record))
    return root


def _tiny_run(directory, geometry, *, events=6, hits=200, seed=0):
    rng = np.random.default_rng(seed)
    reads = [lb.HitReadout(
        np.sort(rng.choice(geometry.n_pmt, hits, replace=False)).astype(np.int32),
        (rng.exponential(3.0, hits) + 0.3).astype(np.float32),
        rng.normal(50.0, 5.0, hits).astype(np.float32)) for _ in range(events)]
    directory.mkdir(parents=True, exist_ok=True)
    lb.write_sparse_hits(directory / "hits.npz",
                         lb.sparse_from_readouts(np.arange(events, dtype=np.int64),
                                                 reads))
    return directory


def _frozen_arm(geometry, *, seed=9000):
    settings = la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))
    return la.LucidArm("sigreg_jepa", geometry=geometry, settings=settings, seed=seed,
                       encoder_config=pe.EncoderConfig(window=16, shift=8))


# ================================================================ gate §6.2
def test_gate_positive_and_pending_acceptance(tmp_path):
    root = _fake_repo(tmp_path)
    result = ef.assert_authorised("inventory", root=root)
    assert result["authorised"] is True and result["status"] == "granted"
    # every check passes, including the runbook sha and the clean <P> tree
    assert all(result["checks"][key]["pass"] for key in ("a", "c", "d"))
    # pre-acceptance: the record is absent -> gate_pending_acceptance, no raise
    no_record = _fake_repo(tmp_path / "pending")
    (no_record / ef.ACCEPTANCE_REL).unlink()
    pending = ef.assert_authorised("readout", root=no_record, dry_run=True)
    assert pending["status"] == "gate_pending_acceptance"
    assert pending["authorised"] is False
    with pytest.raises(SystemExit) as caught:
        ef.assert_authorised("readout", root=no_record)
    assert caught.value.code == 2


def test_gate_a_runbook_sha_mismatch(tmp_path):
    root = _fake_repo(tmp_path)
    runbook = root / ef.RUNBOOK_REL
    runbook.write_text(runbook.read_text() + "\nappendix after acceptance\n")
    result = ef.assert_authorised("inventory", root=root, dry_run=True)
    assert result["checks"]["a"]["pass"] is False
    with pytest.raises(SystemExit) as caught:
        ef.assert_authorised("inventory", root=root)
    assert caught.value.code == 2


def test_gate_b_train_t1_author_line(tmp_path):
    root = _fake_repo(tmp_path, decisions_text="no authority line here")
    result = ef.assert_authorised("train-t1", root=root, dry_run=True, arm=ef.T1_ARM,
                                  variant="s030_within_trigger", seed=9000,
                                  steps=ef.T1_STEPS, out_root=tmp_path / "train")
    assert result["checks"]["b"]["pass"] is False
    with pytest.raises(SystemExit):
        ef.assert_authorised("train-t1", root=root, arm=ef.T1_ARM,
                             variant="s030_within_trigger", seed=9000,
                             steps=ef.T1_STEPS, out_root=tmp_path / "train")


def test_gate_c_review_verdict(tmp_path):
    root = _fake_repo(tmp_path)
    payload = json.loads((root / ef.ACCEPTANCE_REL).read_text())
    payload["review_verdict"] = "BLOCK"
    (root / ef.ACCEPTANCE_REL).write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as caught:
        ef.assert_authorised("inventory", root=root)
    assert caught.value.code == 2
    assert ef.acceptance_record(root)["review_verdict"] == "BLOCK"


def test_gate_d_untracked_and_modified_p_files(tmp_path):
    root = _fake_repo(tmp_path)
    untracked = root / "src/jepa_noise/ext_followup.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("# untracked implementation file\n")
    result = ef.assert_authorised("inventory", root=root, dry_run=True)
    assert result["checks"]["d"]["pass"] is False
    assert "src/jepa_noise/ext_followup.py" in result["checks"]["d"]["untracked"]
    with pytest.raises(SystemExit):
        ef.assert_authorised("inventory", root=root)
    # a modified tracked <P> file also fails (d)
    untracked.unlink()
    tracked = root / "docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md"
    tracked.write_text(tracked.read_text() + "\nchange\n")
    assert ef.assert_authorised("inventory", root=root,
                                dry_run=True)["checks"]["d"]["pass"] is False


def test_gate_e_only_the_pinned_launches(tmp_path):
    root = _fake_repo(tmp_path)
    out = tmp_path / "ext/train"
    # a valid launch is accepted
    result = ef.assert_authorised("train-t1", root=root, arm=ef.T1_ARM,
                                  variant="s030_across_trigger", seed=9001,
                                  steps=ef.T1_STEPS, out_root=out, dry_run=True)
    assert result["checks"]["e"]["pass"] is True
    # an unregistered tuple is refused
    result = ef.assert_authorised("train-t1", root=root, arm=ef.T1_ARM,
                                  variant="s030_across_trigger", seed=9002,
                                  steps=ef.T1_STEPS, out_root=out, dry_run=True)
    assert result["checks"]["e"]["pass"] is False
    # a second smoke (existing smoke run directory) is refused
    smoke_root = tmp_path / "ext/smoke"
    run_id = lr.run_id_for(ef.T1_CELL_IDS["s030_within_trigger"], ef.T1_ARM, 9000)
    (smoke_root / "runs" / run_id).mkdir(parents=True)
    result = ef.assert_authorised("train-t1", root=root, arm=ef.T1_ARM,
                                  variant="s030_within_trigger", seed=9000,
                                  steps=ef.T1_SMOKE_STEPS, out_root=smoke_root,
                                  dry_run=True)
    assert result["checks"]["e"]["pass"] is False
    with pytest.raises(SystemExit):
        ef.assert_authorised("train-t1", root=root, arm=ef.T1_ARM,
                             variant="s030_within_trigger", seed=9000,
                             steps=ef.T1_SMOKE_STEPS, out_root=smoke_root)


def test_import_closure_flags_a_module_outside_p(tmp_path):
    import types
    assert ef.in_p("src/jepa_noise/ext_followup.py")
    assert ef.in_p("src/noise_module_lucid/readout.py")
    assert ef.in_p("src/jepa_noise/configs/lucid_arms_v0.json")
    assert ef.in_p(ef.RUNBOOK_REL)
    assert not ef.in_p("src/jepa_noise/ext_view.py")
    assert not ef.in_p("src/noise_module")
    module = types.ModuleType("synthetic_outside_p")
    module.__file__ = str(ROOT / "src" / "jepa_noise" / "ext_followup.py")
    saved = sys.modules.get("synthetic_outside_p")
    sys.modules["synthetic_outside_p"] = module
    try:
        closure = ef.import_closure(ROOT)
        assert not [row for row in closure["violations"]
                    if row["path"] == "src/jepa_noise/ext_followup.py"]
        # the pytest test package is itself outside <P>; the real CLI dry run does
        # not import it, so only count the synthetic module's own path here.
        module.__file__ = str(ROOT / "src" / "jepa_noise" / "ext_view.py")
        closure = ef.import_closure(ROOT)
        assert any(row["path"] == "src/jepa_noise/ext_view.py"
                   for row in closure["violations"])
        assert any(row["path"].startswith("src/reconstruction_model/")
                   for row in closure["forced_import_exceptions"])
        assert not any(row["path"].startswith("src/reconstruction_model/")
                       for row in closure["violations"])
    finally:
        if saved is None:
            sys.modules.pop("synthetic_outside_p", None)
        else:
            sys.modules["synthetic_outside_p"] = saved


# ============================================================ pinned data §2
def test_hash_mismatch_is_refused(tmp_path):
    keys = [ef.PROBE_KEYS_S030["test"]]
    with pytest.raises(SystemExit):
        ef.require_pinned(tmp_path / "empty", keys=keys, context="test")
    data_root = tmp_path / "data"
    directory = data_root / "lucid" / keys[0]
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}")
    status = ef.data_status(data_root, keys=keys)
    assert status["pass"] is False
    calls = []
    import jepa_noise.lucid_run as lucid_run
    original = lucid_run.ensure_dataset
    lucid_run.ensure_dataset = lambda **kwargs: calls.append(kwargs)
    try:
        with pytest.raises(SystemExit):
            ef.open_probe_sets({}, None, None, data_root, setting=ef.S030)
    finally:
        lucid_run.ensure_dataset = original
    assert calls == []          # the hash check fails before any dataset opener


_COHORTS = {"dev_probe_train": {"namespace": "lucid/dev_probe_train",
                               "event_start": 200000, "n_events": 10000,
                               "run_length": 50},
            "dev_probe_test": {"namespace": "lucid/dev_probe_test",
                              "event_start": 210000, "n_events": 10000,
                              "run_length": 50}}


def test_s030_openers_use_registered_false_after_the_hash_check(tmp_path, config):
    data_root = tmp_path / "data"
    pins = {}
    for role, name in (("train", "dev_probe_train"), ("test", "dev_probe_test")):
        key = lr.dataset_key(name, "F1", ef.S030, 20261017, None, 10000, 50, False)
        directory = data_root / "lucid" / key
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"key": key}))
        pins[key] = lr.sha256_file(directory / "manifest.json")
    fake_config = {"data": {"dataset_seed": 20261017, "cohorts": _COHORTS}}
    recorded = []

    class _Dataset:
        def __init__(self, key):
            self.key = key

    import jepa_noise.lucid_run as lucid_run

    def fake_ensure(**kwargs):
        # the read-only manifest hash check has already passed for this key
        key = lr.dataset_key(kwargs["cohort_name"], "F1", kwargs["setting"], 20261017,
                             None, 10000, 50, kwargs["registered"])
        manifest = data_root / "lucid" / key / "manifest.json"
        assert manifest.is_file()
        assert pins[key] == lr.sha256_file(manifest)
        recorded.append(dict(kwargs))
        return _Dataset(key)

    original = lucid_run.ensure_dataset
    lucid_run.ensure_dataset = fake_ensure
    try:
        import unittest.mock as mock
        with mock.patch.dict(ef.PINNED_DATA, pins):
            probes = ef.open_probe_sets(fake_config, None, None, data_root,
                                        setting=ef.S030)
    finally:
        lucid_run.ensure_dataset = original
    assert sorted(probes) == ["test", "train"]
    assert len(recorded) == 2
    for call in recorded:
        assert call["registered"] is False
        assert float(call["setting"]) == ef.S030
    assert {call["cohort_name"] for call in recorded} == {"dev_probe_train",
                                                          "dev_probe_test"}


def test_forbidden_call_paths_are_not_used(tmp_path, config, monkeypatch):
    """§2: the readout/h12 loaders, preflight(regenerate=True) and the dejitter
    report are never called by this extension's data path."""
    from scripts import jepa_ext_readout as readout_cli
    from scripts import jepa_ext_dejitter as dejitter_cli
    monkeypatch.setattr(ef, "assert_authorised",
                        lambda *a, **k: {"authorised": True, "status": "granted"})

    def boom(*args, **kwargs):
        raise AssertionError("forbidden call path reached")

    saved = {}
    for module, name in ((eh, "load_probes"), (er, "load_probes"),
                         (readout_cli, "load_probes"),
                         (readout_cli, "preflight_data"),
                         (readout_cli, "_regenerate_probe_sets"),
                         (dejitter_cli, "mode_report")):
        if hasattr(module, name):
            saved[(module, name)] = getattr(module, name)
            setattr(module, name, boom)
    import jepa_noise.lucid_run as lucid_run
    saved[(lucid_run, "ensure_dataset")] = lucid_run.ensure_dataset
    lucid_run.ensure_dataset = boom
    try:
        # the marginal reader is a direct read and must not reach ensure_dataset
        with pytest.raises(SystemExit):
            ef.read_dev_marginals(config, lb.from_synthetic(), tmp_path / "empty")
        # the report assembly never calls the dejitter report
        payload = ef.assemble_report(root=ROOT, ext_root=tmp_path / "ext",
                                     results_dir=tmp_path / "results")
        assert (tmp_path / "results" / "README.md").is_file()
        assert payload["out"]
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)


# ================================================================ A-2 pairing
def _pair_rows(run_ids, event_ids, vertex):
    return {"run_ids": np.asarray(run_ids, dtype=object),
            "event_ids": np.asarray(event_ids, np.int64),
            "vertex": np.asarray(vertex, float)}


def test_pair_events_on_run_and_event_id_with_a_mismatch():
    left = _pair_rows(["r0", "r0", "r1"], [1, 2, 1],
                      [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    right = _pair_rows(["r1", "r0", "r0"], [1, 2, 1],
                       [[2, 2, 2], [1, 1, 1], [0, 0, 0]])
    paired = ef.pair_events(left, right)
    assert paired["n_events"] == 3 and paired["n_runs"] == 2
    assert np.array_equal(paired["event_ids"], [1, 2, 1])
    # a deliberately mismatched key set
    bad_keys = _pair_rows(["r0", "r0", "r9"], [1, 2, 1],
                          [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    with pytest.raises(ValueError, match="key sets differ"):
        ef.pair_events(left, bad_keys)
    # same keys, different truth
    bad_truth = _pair_rows(["r0", "r0", "r1"], [1, 2, 1],
                           [[0, 0, 0], [9, 9, 9], [2, 2, 2]])
    with pytest.raises(ValueError, match="different truth"):
        ef.pair_events(left, bad_truth)
    # duplicate keys are refused
    duplicate = _pair_rows(["r0", "r0", "r0"], [1, 1, 2],
                           [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    with pytest.raises(ValueError, match="duplicate"):
        ef.pair_events(left, duplicate)


def test_run_grouped_split_disjointness():
    runs = [f"r{index}" for index in range(10) for _ in range(3)]
    folds = ef._run_grouped_folds(runs, k=5, seed=0)
    for run in set(runs):
        assert len(set(folds[np.asarray(runs) == run])) == 1, "a run leaks across folds"
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
    train = folds != 0
    test = folds == 0
    assert not (set(np.asarray(runs)[train]) & set(np.asarray(runs)[test]))


def test_paired_run_bootstrap_is_keyed_and_reports_a_ci():
    rng = np.random.default_rng(0)
    runs = np.repeat([f"r{index}" for index in range(20)], 5)
    left = rng.normal(0.2, 0.01, len(runs))
    right = left + 0.01
    first = ef.paired_run_bootstrap(left, right, runs, resamples=200, seed=7)
    second = ef.paired_run_bootstrap(left, right, runs, resamples=200, seed=7)
    assert first["mean_difference_m"] == pytest.approx(0.01, abs=1e-12)
    assert first["ci95_m"] == second["ci95_m"]
    assert first["n_runs"] == 20 and first["n_events"] == 100


# ================================================================ A-4 §3
def _a4_batch(geometry, *, seed=0):
    rng = np.random.default_rng(seed)
    n_pmt = geometry.n_pmt
    pmt_id = np.asarray(geometry.pmt_id_in_mpmt)
    visible_pmts = np.flatnonzero((pmt_id % 2) == 0)      # within context
    hidden_pmts = np.flatnonzero((pmt_id % 2) == 1)      # within target
    features = np.zeros((2, n_pmt, pe.N_FEATURES), np.float32)
    features[..., 0:6] = np.asarray(pe.geometry_features(
        geometry, pe.FeatureScaling.from_geometry(geometry)), np.float32)
    hit = np.zeros((2, n_pmt), bool)
    # event 0: 6 visible hits at t = 50 and 6 hidden hits at t = 10 (so the
    # all-token median/trigger differ from the context-only reference)
    hit[0, visible_pmts[:6]] = True
    hit[0, hidden_pmts[:6]] = True
    features[0, hit[0], pe.LOG_CHARGE_INDEX] = 2.0
    features[0, visible_pmts[:6], pe.TIME_INDEX] = 50.0
    features[0, hidden_pmts[:6], pe.TIME_INDEX] = 10.0
    features[0, hit[0], pe.HIT_INDEX] = 1.0
    # event 1: all hits hidden (zero visible hit -> median 0 / trigger None)
    hit[1, hidden_pmts[:5]] = True
    features[1, hit[1], pe.LOG_CHARGE_INDEX] = 2.0
    features[1, hit[1], pe.TIME_INDEX] = 30.0
    features[1, hit[1], pe.HIT_INDEX] = 1.0
    hidden = np.zeros((2, n_pmt), bool)
    hidden[:, hidden_pmts] = True
    return features, hit, hidden


def test_a4_context_reference_is_invariant_to_hidden_time_perturbations(small_geometry):
    arm = _frozen_arm(small_geometry)
    features, hit, hidden = _a4_batch(small_geometry)
    features_t = torch.as_tensor(features)
    hidden_t = torch.as_tensor(hidden)
    for convention in (ef.A4_TOKEN_MEDIAN, ef.A4_TOKEN_TRIGGER):
        context_info = ef.context_time_shift(features_t, hidden, hit, convention,
                                             visible=hit & ~hidden)
        shifted = ef.apply_time_shift(features_t, hit, context_info["shift_ns"])
        # unhit token times stay 0; only hit tokens are reduced
        assert torch.equal(shifted[..., pe.TIME_INDEX][~torch.as_tensor(hit)],
                           torch.zeros(int((~hit).sum()), dtype=shifted.dtype))
        # perturb only the hidden (target) hit token times
        perturbed = features.copy()
        perturbed[..., pe.TIME_INDEX][hidden & hit] += 900.0
        perturbed_t = torch.as_tensor(perturbed)
        context_info_2 = ef.context_time_shift(perturbed_t, hidden, hit, convention,
                                               visible=hit & ~hidden)
        shifted_2 = ef.apply_time_shift(perturbed_t, hit, context_info_2["shift_ns"])
        assert np.array_equal(context_info["shift_ns"], context_info_2["shift_ns"])
        # the context-position features are bit-identical
        assert torch.equal(shifted[..., :][~hidden_t], shifted_2[..., :][~hidden_t])
        with torch.no_grad():
            out_a = arm.jepa.encode(shifted, hidden_mask=hidden_t)
            out_b = arm.jepa.encode(shifted_2, hidden_mask=hidden_t)
        assert torch.equal(out_a, out_b)                   # context_only + propagated
        # the all-token reference recomputed from the perturbed tokens does change
        all_shift = ef.context_time_shift(features_t, hidden, hit, convention,
                                          visible=hit)
        all_shift_2 = ef.context_time_shift(perturbed_t, hidden, hit, convention,
                                            visible=hit)
        assert not np.array_equal(all_shift["shift_ns"], all_shift_2["shift_ns"])
        shifted_all = ef.apply_time_shift(features_t, hit, all_shift["shift_ns"])
        shifted_all_2 = ef.apply_time_shift(perturbed_t, hit, all_shift_2["shift_ns"])
        with torch.no_grad():
            out_a = arm.jepa.encode(shifted_all, hidden_mask=hidden_t)
            out_b = arm.jepa.encode(shifted_all_2, hidden_mask=hidden_t)
        assert not torch.equal(out_a, out_b)


def test_a4_zero_visible_hit_and_none_trigger_rules(small_geometry):
    features, hit, hidden = _a4_batch(small_geometry)
    features_t = torch.as_tensor(features)
    # median: event 1 has no visible hit -> shift 0
    median = ef.context_time_shift(features_t, hidden, hit, ef.A4_TOKEN_MEDIAN,
                                   visible=hit & ~hidden)
    assert median["shift_ns"][1] == 0.0 and median["n_ctx_no_hit"] == 1
    # trigger: event 1 has no visible hit -> None -> shift 0 and counted
    trigger = ef.context_time_shift(features_t, hidden, hit, ef.A4_TOKEN_TRIGGER,
                                    visible=hit & ~hidden)
    assert trigger["shift_ns"][1] == 0.0
    assert trigger["n_ctx_no_trigger"] == 1
    # fewer than NHITS_N (4) visible hits -> None trigger, shift 0
    few = np.zeros_like(hit)
    few[0, np.flatnonzero((np.asarray(small_geometry.pmt_id_in_mpmt) % 2) == 0)[:2]] = True
    sparse = ef.context_time_shift(features_t, np.zeros_like(hidden), few,
                                   ef.A4_TOKEN_TRIGGER, visible=few)
    assert sparse["shift_ns"][0] == 0.0 and sparse["n_ctx_no_trigger"] >= 1
    # a context reference is nonzero when a visible cluster exists
    assert trigger["shift_ns"][0] != 0.0


def test_a4_median_differs_from_all_token_when_hidden_times_move(small_geometry):
    features, hit, hidden = _a4_batch(small_geometry)
    context = ef.context_time_shift(features, hidden, hit, ef.A4_TOKEN_MEDIAN,
                                    visible=hit & ~hidden)
    all_tokens = ef.context_time_shift(features, hidden, hit, ef.A4_TOKEN_MEDIAN,
                                       visible=hit)
    assert context["shift_ns"][0] != all_tokens["shift_ns"][0]


# =============================================================== A-1b §3
def _write_marginal_dataset(data_root, geometry, *, events=4, seed=0):
    key = "test_marginal_key"
    directory = data_root / "lucid" / key
    run_dir = directory / "runs" / "run0"
    rng = np.random.default_rng(seed)
    hits = 40
    reads = [lb.HitReadout(
        np.sort(rng.choice(geometry.n_pmt, hits, replace=False)).astype(np.int32),
        (rng.exponential(3.0, hits) + 0.3).astype(np.float32),
        rng.normal(50.0, 5.0, hits).astype(np.float32)) for _ in range(events)]
    _tiny_run(run_dir, geometry, events=events, hits=hits, seed=seed)
    np.savez(run_dir / "events.npz",
             event_ids=np.arange(events, dtype=np.int64),
             domain=np.asarray(["photon"] * events),
             particle=np.asarray(["none"] * events),
             vertex_m=np.zeros((events, 3)))
    manifest = {"key": key, "registered": False,
                "runs": [{"run_id": "test/0000", "run_dir": "run0",
                          "event_id_start": 0, "n_events": events}]}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return key, lr.sha256_file(directory / "manifest.json")


def test_marginal_reader_never_reaches_ensure_dataset(tmp_path, config, small_geometry):
    key, sha = _write_marginal_dataset(tmp_path / "data", small_geometry)
    calls = []
    import jepa_noise.lucid_run as lucid_run
    original = lucid_run.ensure_dataset
    lucid_run.ensure_dataset = lambda **kwargs: calls.append(kwargs)
    try:
        marginals = ef.read_dev_marginals(config, small_geometry, tmp_path / "data",
                                          key=key, expected_sha256=sha)
    finally:
        lucid_run.ensure_dataset = original
    assert calls == []
    assert marginals["key"] == key and marginals["n_pmt"] == small_geometry.n_pmt
    assert set(marginals["feature_names"]) == set(ef.M_FEATURE_NAMES)
    matrix = ef.marginal_matrix(marginals)
    assert matrix.shape == (small_geometry.n_pmt, 20)
    # a wrong pin is refused before any read
    with pytest.raises(SystemExit):
        ef.read_dev_marginals(config, small_geometry, tmp_path / "data", key=key,
                              expected_sha256="0" * 64)


def test_within_target_set_equals_the_odd_parity_set(full_geometry):
    hit = np.zeros(full_geometry.n_pmt, bool)
    hit[::3] = True
    mask = lv.within_mask(full_geometry, 0, hit)
    odd = np.flatnonzero((np.asarray(full_geometry.pmt_id_in_mpmt) % 2) == 1)
    assert np.array_equal(np.sort(mask.target), odd)
    assert np.array_equal(er.within_target_mask(full_geometry),
                          (np.asarray(full_geometry.pmt_id_in_mpmt) % 2) == 1)
    pair = mask.fractions(hit)
    assert pair["context"] == pytest.approx(hit[mask.context].mean())
    assert pair["target"] == pytest.approx(hit[mask.target].mean())


def test_across_membership_is_exactly_half_for_every_module(full_geometry):
    order = lv.module_hilbert_order(full_geometry)
    n_modules = int(full_geometry.n_modules)
    counts = np.zeros(n_modules, np.int64)
    for offset in range(n_modules):
        target, context, modules = lv._across_partition(
            full_geometry, offset, n_modules // 2, order)
        in_target = np.zeros(n_modules, bool)
        in_target[modules] = True
        counts += in_target
        assert target.size + context.size == full_geometry.n_pmt
    assert np.all(counts == n_modules // 2), "every module must be a target exactly n/2 times"


def test_across_mask_offsets_pass_a_uniformity_test(full_geometry):
    from scipy.stats import chisquare
    n_modules = int(full_geometry.n_modules)
    hit = np.ones(full_geometry.n_pmt, bool)
    offsets = np.empty(20000, np.int64)
    for event in range(20000):
        mask = lv.across_mask(full_geometry, event, hit, seed=0, replicate=0, n_min=0)
        offsets[event] = mask.offset
    counts = np.bincount(offsets, minlength=n_modules)
    # 105 bins, 20000 draws: every bin must be populated and the p-value must pass
    assert counts.size == n_modules and counts.min() > 0
    p_value = float(chisquare(counts).pvalue)
    assert p_value > 1e-3, f"across_mask offsets are not uniform (p={p_value:.3e})"


@needs_data
def test_scale300k_manifest_hash_and_domain_keys():
    geometry = lb.load_wcte_geometry(ef.GEOMETRY_DEFAULT)
    status = ef.data_status(DATA_ROOT, keys=[ef.TRAIN_KEY_300K, ef.DARK_BANK_KEY])
    assert status["sets"][ef.TRAIN_KEY_300K]["pass"] is True
    assert status["sets"][ef.DARK_BANK_KEY]["pass"] is True
    manifest = ef.read_json(ef.manifest_path(DATA_ROOT, ef.TRAIN_KEY_300K), {})
    assert manifest["key"] == ef.TRAIN_KEY_300K
    assert manifest["registered"] is False and manifest["setting"] == ef.S005
    assert manifest["n_events"] == 300000 and manifest["n_runs"] == 300
    assert manifest["runs"][0]["event_id_start"] == 1300000
    # §3 A0 domain-key assertion: every dev_train run has events.npz with `domain`
    for key in (ef.TRAIN_KEY_S005, ef.TRAIN_KEY_300K):
        directory = ef.manifest_path(DATA_ROOT, key).parent
        rows = (ef.read_json(ef.manifest_path(DATA_ROOT, key), {}) or {}).get("runs") or []
        for row in rows:
            events = directory / "runs" / row["run_dir"] / "events.npz"
            with np.load(events, allow_pickle=False) as data:
                assert "domain" in data.files, f"{key} {row['run_dir']} has no domain key"
                assert set(np.asarray(data["domain"]).astype(str)) >= {
                    "photon", "muon", "electron"}
    assert int(geometry.n_pmt) == 1995


@needs_collector
def test_label_collector_reproduces_the_committed_fix9000_rows():
    config = lr.load_run_config(ROOT / ef.CONFIG_REL)
    gen = lb.load_generator_config(ROOT / ef.GEN_REL)
    arms_config = la.load_config(ROOT / ef.ARMS_REL)
    geometry = lb.load_wcte_geometry(ef.GEOMETRY_DEFAULT)
    probes = ef.open_probe_sets(config, gen, geometry, DATA_ROOT, setting=ef.S005,
                                log=lambda *a, **k: None)
    reference = eh._read_label_cache(ef.A1B_LABEL_CACHE, fix_sha=ef.A1B_LABEL_FIX_SHA256,
                                     max_label_events=ef.A1B_LABEL_EVENTS)
    assert reference is not None, "the committed fix9000 label cache was not accepted"
    row = ef._row("fix9000", family="matrix", arm="sigreg_jepa", split="within",
                  seed=9000, beta=None, source=str(FIX_CHECKPOINT),
                  sha256=ef.A1B_LABEL_FIX_SHA256)
    rows = ef._item_label_rows(row, config=config, gen=gen, arms_config=arms_config,
                              geometry=geometry, test_dataset=probes["test"],
                              data_root=DATA_ROOT, device="cpu",
                              log=lambda *a, **k: None)
    outcome = ef.verify_label_collector(rows, reference)
    assert outcome["pass"] is True, (outcome.get("reason"), outcome.get("checks"))
    assert rows["n_events"] == ef.A1B_LABEL_EVENTS
    assert rows["y"].shape[0] == reference["y"].shape[0]


@needs_collector
def test_a1b_c_group_reproduces_the_committed_1043():
    config = lr.load_run_config(ROOT / ef.CONFIG_REL)
    geometry = lb.load_wcte_geometry(ef.GEOMETRY_DEFAULT)
    reference = eh._read_label_cache(ef.A1B_LABEL_CACHE, fix_sha=ef.A1B_LABEL_FIX_SHA256,
                                     max_label_events=ef.A1B_LABEL_EVENTS)
    assert reference is not None
    gen = lb.load_generator_config(ROOT / ef.GEN_REL)
    probes = ef.open_probe_sets(config, gen, geometry, DATA_ROOT, setting=ef.S005,
                                log=lambda *a, **k: None)
    features = ef.collect_probe_test_features(config, geometry, probes["test"],
                                              log=lambda *a, **k: None)
    payload = eh.code_origin_probes(reference, features, geometry,
                                    probes=("linear",), seed=er.SAMPLE_SEED,
                                    log=lambda *a, **k: None)
    got = payload["probes_by_label"]["fix9000"]["all"]["linear"]["r2"]
    assert abs(float(got) - ef.A1B_REPRODUCTION_R2) <= ef.A1B_REPRODUCTION_TOLERANCE, got


@needs_data
def test_scale300k_marginals_use_the_pinned_300k_training_set(config):
    geometry = lb.load_wcte_geometry(ef.GEOMETRY_DEFAULT)
    marginals = ef.read_dev_marginals(config, geometry, DATA_ROOT,
                                      key=ef.TRAIN_KEY_300K, max_runs=2,
                                      log=lambda *a, **k: None)
    assert marginals["key"] == ef.TRAIN_KEY_300K
    assert marginals["manifest_sha256"] == ef.PINNED_DATA[ef.TRAIN_KEY_300K]
    assert marginals["registered_false"] is True
    assert "photon" in marginals["domains_present"]
    assert set(marginals["domains_present"]) >= {"photon", "muon", "electron"}


# ================================================================ T-1 §4
def test_t1_new_cells_differ_only_in_cell_id_setting_and_s_arm(config):
    from scripts import jepa_pregrid_checks as pregrid
    pairs = {
        "s030_within_trigger": "within_trigger",
        "s030_across_trigger": "across",
    }
    old_cells = {"within_trigger": pregrid.variant_cell(config, "trigger"),
                 "across": pregrid.variant_cell(config, "across")}
    for new_variant, old_variant in pairs.items():
        new_cell = ef.t1_cell(config, new_variant)
        old_cell = old_cells[old_variant]
        new, old = dataclasses.asdict(new_cell), dataclasses.asdict(old_cell)
        changed = {key for key in set(new) | set(old) if new.get(key) != old.get(key)}
        assert changed == {"cell_id", "setting", "s_arm"}, (new_variant, changed)
        assert new_cell.cell_id == ef.T1_CELL_IDS[new_variant]
        assert float(new_cell.setting) == ef.S030 and float(new_cell.s_arm) == ef.S030
        assert new_cell.split == old_cell.split


def test_t1_run_one_kwargs_equal_mode_train(tmp_path, config, monkeypatch):
    """§4 required check 1: the kwargs equal ``mode_train``'s except four."""
    from scripts import jepa_ext_dejitter as dejitter_cli
    from scripts import jepa_pregrid_checks as pregrid

    sentinel = {"generator": object(), "arms": object(), "geometry": object()}
    device = "cpu"
    cell = pregrid.variant_cell(config, "trigger")
    monkeypatch.setattr(dejitter_cli, "setup",
                        lambda args: (config, sentinel["generator"], sentinel["arms"],
                                      sentinel["geometry"], device))
    monkeypatch.setattr(dejitter_cli, "cell_of", lambda config_, variant: cell)
    monkeypatch.setattr(dejitter_cli, "refuse_existing_run_dir",
                        lambda out_root, run_id: None)
    monkeypatch.setattr(pregrid, "datasets_for", lambda *a, **k: ("loader", "probes"))
    captured = {}
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run_one(**kwargs):
        captured.update(kwargs)
        (run_dir / "train_log.jsonl").write_text("")
        (run_dir / "final.pt").write_bytes(b"checkpoint")
        return {"run_dir": run_dir, "manifest": {"wall_seconds": 0.0},
                "training": {}}

    monkeypatch.setattr(lr, "run_one", fake_run_one)
    monkeypatch.setattr(pregrid, "build_loaded",
                        lambda *a, **k: (type("Arm", (), {"jepa": None})(), None))
    monkeypatch.setattr(pregrid, "evaluate_forms", lambda *a, **k: {})
    config_path = ROOT / ef.CONFIG_REL
    args = type("Args", (), {"variant": "within_trigger", "arm": "mae_mse", "seed": 9000,
                             "steps": 10000, "probe_max_events": None,
                             "data_root": str(tmp_path), "out": str(tmp_path),
                             "config": str(config_path), "transform": "none"})()
    dejitter_cli.mode_train(args)
    mode_kwargs = {key: value for key, value in captured.items()
                   if key not in ("cell", "out_root", "loader_dataset",
                                  "probe_datasets")}
    new_cell = ef.t1_cell(config, "s030_within_trigger")
    ours = ef.t1_run_kwargs(
        cell=new_cell, seed=9000, config=config, arms_config=sentinel["arms"],
        geometry=sentinel["geometry"], loader_dataset="<loader>",
        probe_datasets="<probes>", out_root="<out>", data_root=str(tmp_path),
        config_sha=lr.sha256_file(config_path), code_sha=lr.git_head(ROOT),
        status_text=lr.git_status(ROOT), device=device, steps=10000)
    ours = {key: value for key, value in ours.items()
            if key not in ("cell", "out_root", "loader_dataset", "probe_datasets")}
    assert set(ours) == set(mode_kwargs)
    for key in ours:
        assert ours[key] == mode_kwargs[key], key
    fixed = {"arm_name": "mae_mse", "beta": None, "steps": 10000, "evaluate": True,
             "physics_method": "ridge", "save_embeddings": False, "dark_bank": None}
    for key, value in fixed.items():
        assert ours[key] == value
    assert ours["config"] is config and ours["arms_config"] is sentinel["arms"]
    assert ours["geometry"] is sentinel["geometry"]


def _manifest_fixture(tmp_path, *, mutate=None, new_missing=False):
    ext = tmp_path / "ext"
    (ext / "train").mkdir(parents=True)
    reference = {
        "run_id": "ext_dev_s005_within_trigger__mae_mse__s9000",
        "arm": "mae_mse", "outcome": "complete", "resumed_from_step": 0,
        "loader_dataset_manifest_sha256": "0" * 64,
        "probe_dataset_sha256": {"train": "1" * 64, "test": "2" * 64},
        "optimizer": "muon_adamw", "batch": 32,
    }
    (tmp_path / "reference.jsonl").write_text(json.dumps(reference) + "\n")
    new = dict(reference)
    new.update({"run_id": "ext_dev_s030_within_trigger__mae_mse__s9000",
                "cell_id": "ext_dev_s030_within_trigger", "setting": 0.3, "s_arm": 0.3,
                "loader_dataset_manifest_sha256": ef.T1_REQUIRED_LOADER_SHA,
                "probe_dataset_sha256": {
                    "train": sorted(ef.T1_REQUIRED_PROBE_SHAS)[0],
                    "test": sorted(ef.T1_REQUIRED_PROBE_SHAS)[1]}})
    if mutate:
        new.update(mutate)
    if not new_missing:
        (ext / "train" / "manifest.jsonl").write_text(json.dumps(new) + "\n")
    return ext, tmp_path / "reference.jsonl"


def test_t1_host_manifest_check(tmp_path):
    ext, reference = _manifest_fixture(tmp_path)
    payload = ef.t1_host_manifest_check(ext_root=ext, variant="s030_within_trigger",
                                        seed=9000, reference_manifest=reference,
                                        log=lambda *a, **k: None)
    assert payload["pass"] is True
    assert "cell_id" in payload["differences"]
    assert payload["unexpected_differences"] == []
    # a difference outside the allowed set is S9
    bad = tmp_path / "bad"
    ext_bad, reference_bad = _manifest_fixture(bad, mutate={"optimizer": "sgd"})
    payload = ef.t1_host_manifest_check(ext_root=ext_bad,
                                        variant="s030_within_trigger", seed=9000,
                                        reference_manifest=reference_bad,
                                        log=lambda *a, **k: None)
    assert payload["pass"] is False
    assert payload["unexpected_differences"] == ["optimizer"]
    # missing rows fail
    missing = tmp_path / "missing"
    ext_missing, reference_missing = _manifest_fixture(missing, new_missing=True)
    payload = ef.t1_host_manifest_check(ext_root=ext_missing,
                                        variant="s030_within_trigger", seed=9000,
                                        reference_manifest=reference_missing,
                                        log=lambda *a, **k: None)
    assert payload["pass"] is False
    # the smoke also allows the step-count differences
    smoke = tmp_path / "smoke"
    ext_smoke, reference_smoke = _manifest_fixture(
        smoke, mutate={"steps": 20, "steps_completed": 20, "final_step": 20})
    (ext_smoke / "smoke").mkdir()
    (ext_smoke / "smoke" / "manifest.jsonl").write_text(
        json.dumps(dict(json.loads((ext_smoke / "train" / "manifest.jsonl")
                                   .read_text()),
                        run_id="ext_dev_s030_within_trigger__mae_mse__s9000",
                        steps=20, steps_completed=20, final_step=20)) + "\n")
    payload = ef.t1_host_manifest_check(ext_root=ext_smoke,
                                        variant="s030_within_trigger", seed=9000,
                                        smoke=True, reference_manifest=reference_smoke,
                                        log=lambda *a, **k: None)
    assert payload["pass"] is True


# ================================================================ anc100 §3
def test_anc100_build_checks_the_dark_bank_before_ensure_dark_bank(tmp_path, config,
                                                                   monkeypatch):
    events = []
    monkeypatch.setattr(ef, "require_pinned",
                        lambda *a, **k: events.append(("require_pinned", k.get("keys"))))
    from scripts import jepa_pregrid_checks as pregrid
    def fake_build_loaded(*a, **k):
        events.append(("build_loaded", a[0]))
        return type("Arm", (), {"to": lambda self, device: self})(), None

    monkeypatch.setattr(pregrid, "build_loaded", fake_build_loaded)
    row = ef._row("anc100", family="predictability", arm="sigreg_jepa_anchored",
                  split="within", seed=9103, beta=100.0,
                  source="/nonexistent/anc100.pt", sha256="0" * 64,
                  eval={"anchor": True, "time_convention": ef.A4_TOKEN_TRIGGER})
    arm = ef.build_item_arm(row, config=config, gen=None, arms_config={},
                            geometry=None, data_root=tmp_path, device="cpu",
                            log=lambda *a, **k: None)
    assert arm is not None
    assert [name for name, _ in events] == ["require_pinned", "build_loaded"]
    assert ef.DARK_BANK_KEY in events[0][1]


def test_require_dark_bank_hash_check_precedes_ensure_dark_bank(tmp_path, monkeypatch):
    directory = tmp_path / "data/lucid" / ef.DARK_BANK_KEY
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}")
    sha = lr.sha256_file(directory / "manifest.json")
    calls = []
    monkeypatch.setitem(ef.PINNED_DATA, ef.DARK_BANK_KEY, sha)
    monkeypatch.setattr(lr, "ensure_dark_bank",
                        lambda **kwargs: calls.append(kwargs) or directory)
    monkeypatch.setattr(la, "dark_bank_from_store", lambda *a, **k: "bank")
    bank = ef.require_dark_bank({"data": {"dataset_seed": 20261017,
                                          "dark_noise": {"records": 512}}},
                                None, None, tmp_path / "data")
    assert bank == "bank" and len(calls) == 1
    # a mismatched dark bank is refused before ensure_dark_bank is reached
    calls.clear()
    monkeypatch.setitem(ef.PINNED_DATA, ef.DARK_BANK_KEY, "f" * 64)
    with pytest.raises(SystemExit):
        ef.require_dark_bank({"data": {"dataset_seed": 20261017,
                                       "dark_noise": {"records": 512}}},
                             None, None, tmp_path / "data")
    assert calls == []


# ============================================================ A0 plan/cost
def test_a0_projection_is_inside_the_stage_a_cap():
    projection = ef.stage_a_projection()
    assert projection["item_equivalents"] == pytest.approx(87.8, abs=0.05)
    assert 3.5 <= projection["projected_l40s_hours"] <= 4.9
    assert projection["cap_hours"] == 5.5 and projection["pass"] is True
    cpu = ef.cpu_projection()
    assert cpu["cap_hours"] == 8.0 and cpu["pass"] is True
    # the task table names every Stage A task and no T-1 item
    tasks = {item["task"] for item in ef.stage_a_items()}
    assert tasks == {"A1", "A-1", "A-1b", "A-2", "A-3", "A-4"}
    assert len(ef.A1_COMPUTE_LABELS) == 23 and len(ef.A1_MLP_FILL_LABELS) == 16
    assert len(ef.A1B_LABELS) == 13 and len(ef.A2_LABELS) == 4
    assert len(ef.A3_LABELS) == 11 and len(ef.A4_LABELS) == 8


def test_inventory_is_built_read_only_and_marks_unmatched_rows(tmp_path):
    inventory = ef.build_inventory(ROOT, hash_files=False,
                                   log=lambda *a, **k: None)
    assert inventory["registered"] is False
    assert inventory["rows"]["init9000"]["status"] == "available"
    assert inventory["rows"]["init9000"]["source"] is None
    assert inventory["rows"]["strong_drift_init"]["data_setting"] == ef.S030
    # §2.1 view design rows are pending while results/jepa/20261103_view/ is absent
    if not (ROOT / "results/jepa/20261103_view/runs.json").is_file():
        assert inventory["rows"]["view_within9000"]["status"] == "pending (view design)"
    assert inventory["rows"]["anc100"]["status"] in ("available", "unverified (dry-run)",
                                                    "unavailable")
    # a row whose sha cannot be matched (or whose file is missing) is unavailable
    assert ef._verify("/nonexistent/final.pt", "0" * 64)[0] == "unavailable"
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"not the committed checkpoint")
    assert ef._verify(checkpoint, "0" * 64)[0] == "unavailable"
    assert ef._verify(checkpoint, None)[0] == "available"


# ================================================================ no overwrite
def test_no_overwrite(tmp_path):
    out_root = tmp_path / "ext/train"
    run_id = lr.run_id_for(ef.T1_CELL_IDS["s030_within_trigger"], ef.T1_ARM, 9000)
    run_dir = ef.prepare_run_dir(out_root, run_id, {"variant": "s030_within_trigger"},
                                 steps=ef.T1_STEPS, device="cpu")
    assert (run_dir / "variant.json").is_file()
    with pytest.raises(RuntimeError, match="refusing existing run directory"):
        ef.prepare_run_dir(out_root, run_id)
    with pytest.raises(RuntimeError, match="refusing existing run directory"):
        ef.refuse_existing_run_dir(out_root, run_id)
    payload = tmp_path / "plan.json"
    ef.write_json(payload, {"a": 1})
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ef.write_json(payload, {"a": 2}, overwrite=False)
    assert json.loads(payload.read_text())["a"] == 1


# ============================================================= CLI / parser
def test_cli_modes_and_defaults():
    parser = followup_cli.parser()
    args = parser.parse_args(["dry-run", "--out", "/tmp/plan.json"])
    assert args.mode == "dry-run" and args.pairing_limit == 8
    args = parser.parse_args(["readout", "--item", "A-1b"])
    assert args.item == "A-1b" and args.include_t1 is False
    args = parser.parse_args(["train-t1", "--variant", "s030_within_trigger",
                              "--seed", "9000"])
    assert args.steps == ef.T1_STEPS and args.smoke is False
    args = parser.parse_args(["train-t1", "--variant", "s030_across_trigger",
                              "--seed", "9001", "--smoke"])
    assert args.smoke is True
    with pytest.raises(SystemExit):
        parser.parse_args(["readout", "--item", "A-9"])
    with pytest.raises(SystemExit):
        parser.parse_args(["train-t1", "--variant", "s050_within_trigger",
                           "--seed", "9000"])
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown-mode"])

# ==================================================== readout-core unit checks
def test_a1_classification_rules_and_spearman():
    assert ef._classify_event({9000: 0.02, 9001: 0.03})["class"] == "event-dependent"
    assert ef._classify_event({9000: 1e-4, 9001: 5e-4})["class"] == "event-invariant"
    assert ef._classify_event({9000: 0.02, 9001: 1e-4})["class"] == "intermediate"
    assert ef._classify_event({9000: None, 9001: 0.03})["class"] == \
        "single-seed (not classified)"
    assert ef._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ef._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ef._h12_label("geomblind_9000") == "geomblind9000"
    assert ef._h12_label("frozen_target_9001") == "frozen_target9001"


def test_a1b_class_order_is_the_fixed_order():
    assert ef.a1b_class(0.8, 0.2) == "input-statistics code"
    assert ef.a1b_class(0.5, 0.8) == \
        "fixed-feature code (geometry or probe-test raw)"
    assert ef.a1b_class(0.1, 0.4) == "not determined by fixed per-PMT features"
    assert ef.a1b_class(0.1, 0.6) == "intermediate"
    assert ef.a1b_class(None, 0.6) == "unavailable"


def test_a3_angular_logistic_and_quintile_helpers():
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(40, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    exact = ef.angular_errors(direction, direction)
    assert exact["median_deg"] == pytest.approx(0.0, abs=1e-6)
    orthogonal = ef.angular_errors(np.tile([1.0, 0.0, 0.0], (10, 1)),
                                   np.tile([0.0, 1.0, 0.0], (10, 1)))
    assert orthogonal["median_deg"] == pytest.approx(90.0, abs=1e-4)
    # a separable L2-logistic problem with 12 run groups
    x = np.vstack([rng.normal(-1.0, 0.4, (60, 4)),
                   rng.normal(1.0, 0.4, (60, 4))])
    y = np.r_[np.zeros(60, int), np.ones(60, int)]
    runs = np.repeat([f"r{index}" for index in range(12)], 10)
    scored = ef._logistic_auc(x, y, runs, x, y)
    assert scored["auc"] > 0.9 and 0.0 <= scored["balanced_accuracy"] <= 1.0
    assert scored["C"] in (0.01, 0.1, 1.0, 10.0)
    quintiles = ef._nhits_quintile_auc(scored["probability"], y,
                                       rng.integers(0, 100, 120))
    assert set(quintiles["quintiles"]) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert len(quintiles["edges"]) == 4
    prediction = ef._mlp_direction(rng.normal(size=(80, 6)),
                                   rng.normal(size=(80, 3)),
                                   rng.normal(size=(20, 6)), epochs=3)
    assert prediction.shape == (20, 3) and np.isfinite(prediction).all()


def test_a3_energy_500_rule_and_run_grouped_probe():
    rng = np.random.default_rng(1)

    def rows(n, runs, d=4):
        run_ids = np.repeat([f"r{index}" for index in range(runs)], n // runs)
        return (rng.normal(size=(n, d)), rng.normal(size=(n,)) * 10.0, run_ids,
                np.ones(n, bool))

    train_x, train_y, train_runs, train_contained = rows(400, 20)
    test_x, test_y, test_runs, test_contained = rows(400, 20)
    small_contained = test_contained.copy()
    small_contained[100:] = False                     # only 100 contained -> < 500
    not_evaluable = ef._energy_task(train_x, train_y, train_runs, train_contained,
                                    test_x, test_y, test_runs, small_contained)
    assert not_evaluable["status"] == "not_evaluable"
    assert not_evaluable["n_contained_test"] == 100
    big_x, big_y, big_runs, big_contained = rows(600, 20)
    evaluated = ef._energy_task(big_x, big_y, big_runs, big_contained,
                                big_x, big_y, big_runs, big_contained)
    assert evaluated["status"] == "evaluated"
    assert evaluated["n_contained_test"] == 600


def test_entries_preflight_distinguishes_entries_from_a_gpu_slot(tmp_path):
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"level": "HOLD", "caps": {"free_entries": 45000}}))
    preflight = ef.entries_preflight(guard_json=guard)
    assert preflight["pass"] is True and preflight["gpu_slot_available"] is False
    assert preflight["projection"]["projected_after"] == 45000 - 5884
    guard.write_text(json.dumps({"level": "RUN", "caps": {"free_entries": 33000}}))
    preflight = ef.entries_preflight(guard_json=guard)
    assert preflight["pass"] is False and preflight["gpu_slot_available"] is True

def test_a0_domain_key_assertion_detects_a_missing_domain(tmp_path):
    geometry = lb.from_synthetic()
    data_root = tmp_path / "data"
    key = "synthetic_train_key"
    directory = data_root / "lucid" / key
    run_dir = directory / "runs" / "run0"
    _tiny_run(run_dir, geometry, events=3, hits=20)
    manifest = {"key": key, "runs": [{"run_id": "synthetic/train/0000",
                                      "run_dir": "run0", "n_events": 3}]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    np.savez(run_dir / "events.npz", event_ids=np.arange(3),
             domain=np.asarray(["photon", "muon", "electron"]))
    payload = ef.domain_key_assertion(data_root, keys=(key,), log=lambda *a, **k: None)
    assert payload["pass"] is True and payload["sets"][key]["n_checked"] == 1
    np.savez(run_dir / "events.npz", event_ids=np.arange(3))     # no domain key
    payload = ef.domain_key_assertion(data_root, keys=(key,), log=lambda *a, **k: None)
    assert payload["pass"] is False
    assert payload["sets"][key]["missing"] == ["run0"]
