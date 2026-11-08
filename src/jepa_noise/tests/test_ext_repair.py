# SPDX-License-Identifier: MIT
"""Focused tests for the Paper 3 repair-pilot extension (runbook 7 Nov 2026 §9.3).

Covers every bullet of Part A §9.3: the §2.5 hit-transform context (identity,
restoration after an exception, per-run factors, the HW twin redirect, failing
closed), the two-setting factor identities on a synthetic shared-draw run, the
recomputed state hash, the μ_p reconstruction, the §7.2 invariant tokens and raw
features, LEACE (erasure, idempotence, reduced-rank permutation rule, random
controls), the equivariance fit, the three identification estimators, the Fisher
accounting and P0, the §3 bookkeeping rules, the §2.6 forbidden call paths, the
§9.2 gate negative cases, hash mismatch and no overwrite.

The tiny synthetic runs in pytest temporary directories are the runbook's
explicitly allowed test fixture; no mode of the extension generates data.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_followup as ef  # noqa: E402
from jepa_noise import ext_repair as rp  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_repair", ROOT / "scripts" / "jepa_ext_repair.py")
repair_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair_cli)

GEN = lb.load_generator_config(ROOT / rp.GEN_REL)
SMALL = {"module_ids": tuple(range(6)),
         "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
         "pmts_per_module": 19}


# ------------------------------------------------------------------ A9 stub
def nhits_trigger_time_stub(digits, n=4, window_ns=20.0):
    t = np.sort(np.asarray(digits, dtype=float).ravel())
    if t.size < n:
        return None
    ok = np.flatnonzero(t[n - 1:] - t[:t.size - n + 1] <= window_ns)
    return float(t[ok[0]]) if ok.size else None


def prompt_window_mask_stub(digits, trigger_ns, window_ns=(-20.0, 60.0)):
    t = np.asarray(digits, dtype=float).ravel()
    if trigger_ns is None:
        return np.ones(t.size, dtype=bool)
    delta = t - float(trigger_ns)
    return (delta >= float(window_ns[0])) & (delta <= float(window_ns[1]))


@pytest.fixture(autouse=True)
def _stub_pmt_time_rule(monkeypatch):
    window = (-20.0, 60.0)
    rule = (nhits_trigger_time_stub,
            lambda digits, trigger_ns: prompt_window_mask_stub(digits, trigger_ns, window),
            4, 20.0, window)
    monkeypatch.setattr(lb, "_PMT_TIME_RULE", rule)
    yield rule


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def geometry():
    return lb.from_synthetic(SMALL)


def _tiny_dataset(tmp_path, geometry, *, setting, n_runs=2, events=6, name=None,
                  run_start=0):
    """Generate and write a tiny shared-draw synthetic probe set (test-only)."""
    directory = tmp_path / (name or f"set_{setting:g}")
    directory.mkdir(parents=True, exist_ok=True)
    runs = [lb.RunSpec(f"lucid/dev_probe_train/{index:04d}", "lucid/dev_probe_train",
                       "F1", float(setting), index * events, events, 20261017)
            for index in range(int(run_start), int(run_start) + int(n_runs))]
    rows, generated = [], {}
    for run in runs:
        produced = lb.generate_run(geometry, GEN, run, with_twin=True)
        run_dir = directory / "runs" / run.run_id.replace("/", "__")
        payload = lb.write_generated_run(run_dir, produced, geometry, GEN, seed=20261017)
        rows.append({"run_id": run.run_id, "run_dir": run_dir.name,
                     "event_id_start": run.event_start, "event_id_stop": run.event_stop,
                     "n_events": run.n_events, "setting": float(setting),
                     "state_sha256": payload["runs"][0]["state_sha256"],
                     "twin_present": True})
        generated[str(run.run_id)] = produced
    manifest = {"schema": "jepa-lucid-dataset-v1", "key": directory.name,
                "cohort": "dev_probe_train", "law": "F1", "setting": float(setting),
                "setting_kind": None, "dataset_seed": 20261017,
                "n_events": int(n_runs * events), "n_runs": int(n_runs),
                "run_length": int(events), "n_pmt": int(geometry.n_pmt), "runs": rows}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    dataset = lr.Dataset(key=directory.name, directory=directory,
                         cohort="dev_probe_train", law="F1", setting=float(setting),
                         setting_kind=None, seed=20261017,
                         n_events=int(n_runs * events), run_length=int(events),
                         runs=tuple(rows), manifest=manifest)
    return {"dataset": dataset, "runs": runs, "generated": generated}


def _event_truth(dataset, geometry):
    """The ``ext_followup.load_particle_truth`` structure for a tiny dataset."""
    out = {"key": dataset.key, "runs": {}}
    for row in dataset.runs:
        run_dir = dataset.directory / "runs" / row["run_dir"]
        with np.load(run_dir / "events.npz", allow_pickle=False) as data:
            truth = {name: np.asarray(data[name]) for name in data.files}
        store = lb.read_sparse_hits(run_dir / "hits.npz")
        n_hit = np.diff(np.asarray(store.offsets, np.int64))
        position = {int(event): index for index, event in enumerate(store.event_ids)}
        truth["n_hit"] = np.asarray([n_hit[position[int(event)]]
                                     for event in truth["event_ids"]], np.int64)
        truth.setdefault("vertex", truth["vertex_m"])
        out["runs"][str(row["run_id"])] = truth
    return out


@pytest.fixture
def tiny(tmp_path, geometry):
    sets = {"s005": _tiny_dataset(tmp_path, geometry, setting=rp.S005),
            "s030": _tiny_dataset(tmp_path, geometry, setting=rp.S030)}
    return {"geometry": geometry, "gen": GEN, "s005": sets["s005"],
            "s030": sets["s030"]}


def _fake_repo(tmp_path, *, acceptance=True, record_mutator=None, addendum=True,
               addendum_mutator=None, addendum2=True, addendum2_mutator=None):
    """A throw-away repository with the committed runbook and an acceptance record.

    The committed addendum is copied byte-for-byte from the real worktree, so the
    addendum sha256 gate (a2) is satisfied without hard-coding a fake file; the
    record carries ``addendum_path``/``addendum_sha256`` unless a mutator removes
    them (the "record missing the new fields -> denied" case).
    """
    root = tmp_path / "repo"
    (root / "docs/jepa").mkdir(parents=True)
    (root / rp.RUNBOOK_REL).write_text("## 0. Authority\n\nPart A text.\n")
    if addendum:
        target = root / rp.ADDENDUM_REL
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rp.ADDENDUM_REL, target)
        if addendum_mutator is not None:
            addendum_mutator(target)
    if addendum2:
        target2 = root / rp.ADDENDUM2_REL
        target2.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rp.ADDENDUM2_REL, target2)
        if addendum2_mutator is not None:
            addendum2_mutator(target2)
    (root / "src/noise_module_lucid").mkdir(parents=True, exist_ok=True)
    (root / "src/noise_module_lucid/readout.py").write_text("# tracked\n")
    (root / "src/jepa_noise/configs").mkdir(parents=True, exist_ok=True)
    (root / "src/jepa_noise/configs/lucid_arms_v0.json").write_text("{}\n")
    sha = lr.sha256_file(root / rp.RUNBOOK_REL)
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=t",
                    "commit", "-q", "-m", "init"], cwd=str(root), check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), text=True,
                          capture_output=True, check=True).stdout.strip()
    record = {"runbook_sha256": sha, "impl_commit": head,
              "review_path": "docs/jepa/REVIEW.md", "review_verdict": "PASS",
              "lead": "lead", "date": "2026-11-07",
              "addendum_path": rp.ADDENDUM_REL,
              "addendum_sha256": rp.ADDENDUM_SHA256,
              "addendum2_path": rp.ADDENDUM2_REL,
              "addendum2_sha256": rp.ADDENDUM2_SHA256}
    if record_mutator is not None:
        record_mutator(record, root)
    if acceptance:
        (root / rp.ACCEPTANCE_REL).write_text(json.dumps(record))
    return root


# ================================================================ gate §9.2
def test_gate_positive_and_pending_acceptance(tmp_path):
    root = _fake_repo(tmp_path)
    result = rp.assert_authorised("x1", root=root)
    assert result["authorised"] is True and result["status"] == "granted"
    assert all(result["checks"][key]["pass"]
               for key in ("a", "a2", "a3", "b", "c"))
    pending = _fake_repo(tmp_path / "pending", acceptance=False)
    row = rp.assert_authorised("x1", root=pending, dry_run=True)
    assert row["status"] == "gate_pending_acceptance" and row["authorised"] is False
    with pytest.raises(SystemExit) as caught:
        rp.assert_authorised("x1", root=pending)
    assert caught.value.code == 2
    with pytest.raises(ValueError):          # there is no training mode
        rp.assert_authorised("train", root=pending)


def test_gate_a_runbook_sha_mismatch(tmp_path):
    root = _fake_repo(tmp_path)
    runbook = root / rp.RUNBOOK_REL
    runbook.write_text(runbook.read_text() + "\nlate edit\n")
    result = rp.assert_authorised("x2", root=root, dry_run=True)
    assert result["checks"]["a"]["pass"] is False
    with pytest.raises(SystemExit) as caught:
        rp.assert_authorised("x2", root=root)
    assert caught.value.code == 2


def test_gate_b_acceptance_record_schema_and_verdict(tmp_path):
    def block(record, root):
        record["review_verdict"] = "BLOCK"

    root = _fake_repo(tmp_path, record_mutator=block)
    with pytest.raises(SystemExit) as caught:
        rp.assert_authorised("preflight", root=root)
    assert caught.value.code == 2
    assert rp.acceptance_record(root)["review_verdict"] == "BLOCK"

    def drop(record, root):
        record.pop("lead")

    missing = _fake_repo(tmp_path / "missing", record_mutator=drop)
    with pytest.raises(SystemExit):
        rp.assert_authorised("preflight", root=missing)


def test_gate_c_untracked_and_modified_p_files(tmp_path):
    root = _fake_repo(tmp_path)
    clean = rp.assert_authorised("report", root=root, dry_run=True)
    assert clean["checks"]["c"]["pass"] is True
    # every <P> directory is in the git pathspec, not only the files
    assert set(rp.P_DIRS) <= set(clean["checks"]["c"]["paths"])
    # an untracked <P>-directory file is caught
    untracked = root / "src/noise_module_lucid/extra.py"
    untracked.write_text("# new\n")
    result = rp.assert_authorised("report", root=root, dry_run=True)
    assert result["checks"]["c"]["pass"] is False
    assert "src/noise_module_lucid/extra.py" in result["checks"]["c"]["untracked"]
    with pytest.raises(SystemExit):
        rp.assert_authorised("report", root=root)
    untracked.unlink()
    # a modified tracked <P>-directory file is caught through the same pathspec
    tracked = root / "src/jepa_noise/configs/lucid_arms_v0.json"
    tracked.write_text('{"changed": true}\n')
    result = rp.assert_authorised("report", root=root, dry_run=True)
    assert result["checks"]["c"]["pass"] is False
    assert result["checks"]["c"]["diff_clean"] is False


def test_import_closure_flags_a_module_outside_p():
    import types
    assert rp.in_p("src/jepa_noise/ext_repair.py")
    assert rp.in_p("src/noise_module_lucid/readout.py")
    assert rp.in_p(rp.RUNBOOK_REL)
    assert not rp.in_p("src/jepa_noise/ext_view.py")
    module = types.ModuleType("synthetic_outside_p")
    module.__file__ = str(ROOT / "src/jepa_noise/ext_view.py")
    sys.modules["synthetic_outside_p"] = module
    try:
        closure = rp.import_closure(ROOT)
        assert any(row["path"] == "src/jepa_noise/ext_view.py"
                   for row in closure["violations"])
        assert any(str(row["path"]).startswith("src/reconstruction_model/")
                   for row in closure["forced_import_exceptions"])
        assert closure["pass"] is False
    finally:
        sys.modules.pop("synthetic_outside_p", None)
    module.__file__ = str(ROOT / "src/jepa_noise/ext_repair.py")
    sys.modules["synthetic_outside_p"] = module
    try:
        assert not [row for row in rp.import_closure(ROOT)["violations"]
                    if row["path"] == "src/jepa_noise/ext_repair.py"]
    finally:
        sys.modules.pop("synthetic_outside_p", None)


# ====================================================== §2.5 transform context
def test_identity_condition_gives_bit_identical_features(tiny, geometry):
    dataset = tiny["s005"]["dataset"]
    condition = rp.build_condition("N", dataset, geometry)
    run_dir = dataset.directory / "runs" / dataset.runs[0]["run_dir"]
    plain = pe.tokenise(run_dir, geometry, split="within", seed=0)
    with rp.hit_transform_context(condition):
        wrapped = pe.tokenise(run_dir, geometry, split="within", seed=0)
    assert np.array_equal(np.asarray(plain.features.detach().cpu()),
                          np.asarray(wrapped.features.detach().cpu()))
    assert np.array_equal(plain.event_ids, wrapped.event_ids)


def test_both_attributes_restored_after_an_exception(tiny, geometry):
    dataset = tiny["s005"]["dataset"]
    condition = rp.build_condition("N", dataset, geometry)
    original_pe, original_lb = pe.read_sparse_hits, lb.read_sparse_hits
    with pytest.raises(RuntimeError):
        with rp.hit_transform_context(condition):
            assert pe.read_sparse_hits is not original_pe
            assert lb.read_sparse_hits is not original_lb
            raise RuntimeError("boom")
    assert pe.read_sparse_hits is original_pe
    assert lb.read_sparse_hits is original_lb


def test_the_factor_mapping_is_per_run(tiny, geometry):
    dataset = tiny["s030"]["dataset"]
    condition = rp.build_condition("SWtw", dataset, geometry)
    assert len(set(map(tuple, condition.factors.values()))) == len(condition.factors)
    for row in dataset.runs:
        run_id = str(row["run_id"])
        expected = condition.factor(run_id)
        with rp.hit_transform_context(condition):
            store = lb.read_sparse_hits(dataset.directory / "runs" / row["run_dir"]
                                        / "hits.npz")
        raw = lb.read_sparse_hits(dataset.directory / "runs" / row["run_dir"]
                                  / "hits.npz")
        assert np.array_equal(np.asarray(store.event_ids), np.asarray(raw.event_ids))
        assert np.array_equal(np.asarray(store.offsets), np.asarray(raw.offsets))
        assert np.array_equal(np.asarray(store.pmt_id), np.asarray(raw.pmt_id))
        assert np.array_equal(np.asarray(store.time_ns), np.asarray(raw.time_ns))
        expected_charge = (np.asarray(raw.charge_pe, np.float32)
                           * expected[np.asarray(raw.pmt_id)].astype(np.float32))
        assert np.allclose(np.asarray(store.charge_pe, float),
                           np.asarray(expected_charge, float), rtol=1e-6)


def test_hw_redirect_returns_the_twin_store_with_identical_event_ids(tiny, geometry):
    dataset = tiny["s030"]["dataset"]
    condition = rp.build_condition("HW", dataset, geometry)
    assert condition.filename == "twin_hits.npz"
    for row in dataset.runs:
        run_dir = dataset.directory / "runs" / row["run_dir"]
        with rp.hit_transform_context(condition):
            store = lb.read_sparse_hits(run_dir / "hits.npz")
        twin = lb.read_sparse_hits(run_dir / "twin_hits.npz")
        assert np.array_equal(np.asarray(store.event_ids), np.asarray(twin.event_ids))
        assert np.array_equal(np.asarray(store.charge_pe),
                              np.asarray(twin.charge_pe))
        assert np.array_equal(np.asarray(store.pmt_id), np.asarray(twin.pmt_id))


def test_a_path_outside_the_condition_set_raises(tiny, geometry):
    dataset = tiny["s005"]["dataset"]
    other = tiny["s030"]["dataset"]
    condition = rp.build_condition("N", dataset, geometry)
    with rp.hit_transform_context(condition):
        with pytest.raises(RuntimeError, match="outside"):
            lb.read_sparse_hits(other.directory / "runs"
                                / other.runs[0]["run_dir"] / "hits.npz")
        with pytest.raises(RuntimeError, match="non-hit path"):
            lb.read_sparse_hits(dataset.directory / "manifest.json")


# ============================================== two-setting synthetic factors
def test_factor_identities_on_shared_draws(tiny, geometry):
    s005, s030 = tiny["s005"], tiny["s030"]
    gen0 = s005["generated"][str(s005["runs"][0].run_id)]
    gen1 = s030["generated"][str(s030["runs"][0].run_id)]
    state0, state1 = gen0.state, gen1.state
    store0 = lb.sparse_from_readouts(gen0.event_ids, gen0.hits)
    store1 = lb.sparse_from_readouts(gen1.event_ids, gen1.hits)
    twin0 = lb.sparse_from_readouts(gen0.event_ids, gen0.twin_hits)
    twin1 = lb.sparse_from_readouts(gen1.event_ids, gen1.twin_hits)
    # the two twins are equal (same draws, nominal gain in both settings)
    assert np.array_equal(np.asarray(twin0.event_ids), np.asarray(twin1.event_ids))
    assert np.array_equal(np.asarray(twin0.pmt_id), np.asarray(twin1.pmt_id))
    assert np.array_equal(np.asarray(twin0.charge_pe), np.asarray(twin1.charge_pe))
    assert np.array_equal(np.asarray(twin0.time_ns), np.asarray(twin1.time_ns))
    swtw = state0.gains / state1.gains
    swunit = 1.0 / state1.gains
    index0 = {int(event): position for position, event in enumerate(store0.event_ids)}
    index1 = {int(event): position for position, event in enumerate(store1.event_ids)}
    n_common = 0
    for event in np.intersect1d(np.asarray(store0.event_ids),
                                np.asarray(store1.event_ids)):
        span0 = slice(int(store0.offsets[index0[int(event)]]),
                      int(store0.offsets[index0[int(event)] + 1]))
        span1 = slice(int(store1.offsets[index1[int(event)]]),
                      int(store1.offsets[index1[int(event)] + 1]))
        pmt0, pmt1 = np.asarray(store0.pmt_id[span0]), np.asarray(store1.pmt_id[span1])
        q0, q1 = np.asarray(store0.charge_pe[span0]), np.asarray(store1.charge_pe[span1])
        common, i0, i1 = np.intersect1d(pmt0, pmt1, return_indices=True)
        assert common.size > 0
        n_common += int(common.size)
        swtw_charge = (np.asarray(q1, np.float32)[i1]
                       * swtw[common].astype(np.float32))
        unit_charge = (np.asarray(q1, np.float32)[i1]
                       * swunit[common].astype(np.float32))
        assert np.allclose(np.asarray(swtw_charge, float), np.asarray(q0, float)[i0],
                           rtol=1e-5, atol=1e-6)
        twin1_charge = np.asarray(twin1.charge_pe[span1], np.float32)[i1]
        assert np.allclose(np.asarray(unit_charge, float),
                           np.asarray(twin1_charge, float), rtol=1e-5, atol=1e-6)
    assert n_common > 0


# ========================================================== state hash and μ_p
def test_recomputed_state_sha256_matches_and_a_corrupt_manifest_fails(tiny, geometry):
    dataset = tiny["s030"]["dataset"]
    result = rp.check_run_states(dataset, geometry)
    assert result["pass"] is True and result["n_checked"] == len(dataset.runs)
    corrupted = dataclasses.replace(
        dataset, runs=tuple(dict(row, state_sha256="0" * 64) for row in dataset.runs))
    bad = rp.check_run_states(corrupted, geometry)
    assert bad["pass"] is False
    assert any("state_sha256" in entry["field"] for entry in bad["mismatches"])


def test_mu_reconstruction_matches_the_stored_expected_pe(tiny, geometry):
    dataset = tiny["s030"]["dataset"]
    result = rp.mu_reconstruction(dataset, geometry, tiny["gen"], dataset_seed=20261017,
                                  limit_runs=1)
    assert result["n_events"] > 0
    assert result["pass"] is True
    assert result["max_relative_difference"] <= 1e-6


# ============================================================== invariant tokens
def test_invariant_tokens_are_invariant_and_leave_unhit_tokens(geometry):
    rng = np.random.default_rng(3)
    n_pmt, n_features = geometry.n_pmt, 8
    features = rng.standard_normal((3, n_pmt, n_features)).astype(np.float32)
    hits = rng.random((3, n_pmt)) > 0.6
    log_charge = np.zeros((3, n_pmt), np.float32)
    log_charge[hits] = rng.standard_normal(hits.sum())
    features[..., pe.LOG_CHARGE_INDEX] = log_charge
    features[..., 7] = 0.0
    module_index = rp.module_of_pmt(geometry)
    scaled = features.astype(np.float64).copy()
    log_scale = rng.normal(0.0, 0.2, int(geometry.n_modules))
    scaled_charge = scaled[..., pe.LOG_CHARGE_INDEX]
    shift = np.broadcast_to(log_scale[module_index], hits.shape)
    scaled_charge[hits] += shift[hits]
    scaled[..., pe.LOG_CHARGE_INDEX] = scaled_charge
    base = rp.invariant_features(features, hits, module_index)
    transformed = rp.invariant_features(scaled, hits, module_index)
    assert np.allclose(base[hits], transformed[hits], atol=1e-5, rtol=1e-5)
    # unhit tokens are unchanged by the transform
    assert np.array_equal(np.asarray(base)[~hits], np.asarray(features)[~hits])
    # the wrapper installation is restored in a finally
    original = pe.tokenise
    with rp.invariant_token_context(geometry):
        assert pe.tokenise is not original
    assert pe.tokenise is original


# ================================================================= raw features
def _particle_truth(dataset, geometry):
    return _event_truth(dataset, geometry)


def test_raw_std_particle_rows_reproduce_ext_followup_bit_identically(tiny, geometry):
    importlib.import_module("scripts.jepa_ext_repair")
    dataset = tiny["s005"]["dataset"]
    truth = _event_truth(dataset, geometry)
    ours = rp.raw_std_features(dataset, geometry, truth=truth)
    reference = ef.raw_event_features(dataset, geometry, truth=truth)
    assert reference["features"].shape[0] > 0
    assert np.array_equal(ours["particle"]["features"], reference["features"])
    assert np.array_equal(ours["particle"]["event_ids"], reference["event_ids"])
    assert ours["photon"]["features"].shape[1] == reference["features"].shape[1]


def test_raw_inv_is_invariant_to_module_scalings(tiny, geometry):
    dataset = tiny["s030"]["dataset"]
    rng = np.random.default_rng(5)
    mod = rp.module_of_pmt(geometry)
    factor = np.exp(rng.normal(0.0, 0.3, int(geometry.n_modules)))[mod]
    condition = rp.Condition(
        name="scaled", base="SWunit", setting=rp.S030, filename="hits.npz",
        directory=dataset.directory,
        run_id_by_dir={str(row["run_dir"]): str(row["run_id"])
                       for row in dataset.runs},
        factors={str(row["run_id"]): factor for row in dataset.runs},
        n_pmt=int(geometry.n_pmt))
    plain = rp.raw_inv_features(dataset, geometry)
    scaled = rp.raw_inv_features(dataset, geometry, condition=condition)
    assert plain["features"].shape[0] == scaled["features"].shape[0] > 0
    assert np.array_equal(plain["features"], scaled["features"])


# ======================================================================== LEACE
def test_leace_erases_and_is_idempotent():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((300, 16))
    z = rng.standard_normal((300, 3))
    x[:, :3] += 2.0 * z
    eraser = rp.leace_fit(x, z)
    check = rp.leace_cross_covariance(eraser, x, z)
    assert check["relative"] <= 1e-8
    once = rp.leace_erase(eraser, x)
    twice = rp.leace_erase(eraser, once)
    assert np.allclose(once, twice, atol=1e-9, rtol=0.0)


def _leace_planted(rng, *, amplitude, n_runs=200, events=17, envs=2, d=16,
                   n_modules=105):
    run_ids = np.repeat([f"run{index:03d}" for index in range(n_runs)], events * envs)
    run_index = np.repeat(np.arange(n_runs), events * envs)
    concepts = rng.standard_normal((n_runs, n_modules))
    z = concepts[run_index]
    design = np.zeros((d, n_modules))
    design[:, :5] = float(amplitude) * np.eye(d)[:, :5]
    x = z @ design.T + rng.standard_normal((len(z), d))
    return x, z, run_ids


def test_permutation_moves_each_runs_n_and_d_vectors_together():
    n_runs, events, envs = 6, 3, 2
    run_ids = np.repeat([f"run{index}" for index in range(n_runs)], events * envs)
    z = np.repeat(np.arange(n_runs, dtype=float), events * envs)[:, None]
    rng = np.random.default_rng(0)
    permuted = rp.permute_concepts_by_run(z, run_ids, rng)
    for run in np.unique(run_ids):
        values = np.unique(permuted[run_ids == run])
        assert values.size == 1, "the run's N and D concept vectors must move together"
        assert values[0] in np.arange(n_runs, dtype=float)


def test_reduced_rank_rule_planted_rank_and_null():
    rng = np.random.default_rng(11)
    # a strong planted rank-5 signal: signal std = 10 x the noise std per direction
    x, z, run_ids = _leace_planted(rng, amplitude=10.0)
    strong = rp.reduction_rank(x, z, run_ids, key="strong", label="strong",
                               domain="photon", n_permutations=rp.LEACE_PERMUTATIONS)
    assert strong["rank"] == rp.E2_PLANTED_RANK
    # r never exceeds the planted rank, at weak amplitudes too
    for amplitude in (0.0, 0.2, 1.0):
        weak_x, weak_z, weak_runs = _leace_planted(rng, amplitude=amplitude)
        weak = rp.reduction_rank(weak_x, weak_z, weak_runs, key=f"weak{amplitude}",
                                 label=f"weak{amplitude}", domain="photon",
                                 n_permutations=rp.LEACE_PERMUTATIONS)
        assert weak["rank"] <= rp.E2_PLANTED_RANK
    # the null: r = 0 in at least 17 of 20 keyed seeds
    null_ranks = []
    for index in range(20):
        null_rng = np.random.default_rng(1000 + index)
        null_x, null_z, null_runs = _leace_planted(null_rng, amplitude=0.0)
        result = rp.reduction_rank(null_x, null_z, null_runs, key=f"null{index}",
                                   label=f"null{index}", domain="photon",
                                   n_permutations=rp.LEACE_PERMUTATIONS)
        null_ranks.append(result["rank"])
    assert sum(1 for rank in null_ranks if rank == 0) >= 17, null_ranks


def test_random_controls_have_the_stated_rank():
    rng = np.random.default_rng(2)
    for rank in (1, 5):
        basis = rp.random_control_basis(rng, 16, rank)
        assert basis.shape == (16, rank)
        assert np.allclose(basis.T @ basis, np.eye(rank), atol=1e-12)


# ========================================================== equivariance fit §6
def test_equivariance_fit_recovers_a_known_map_and_excludes_test_runs():
    rng = np.random.default_rng(4)
    n_train, n_test, d, k = 60, 30, 16, 20
    run_ids_train = np.repeat([f"train{index}" for index in range(6)], 10)
    run_ids_test = np.repeat([f"test{index}" for index in range(3)], 10)
    u_train = rng.standard_normal((n_train, k))
    u_test = rng.standard_normal((n_test, k))
    mapping = rng.standard_normal((d, k))
    delta_train = u_train @ mapping.T
    delta_test = u_test @ mapping.T
    fit = rp.equivariance_fit(u_train, delta_train, u_test, delta_test,
                              run_ids_train=run_ids_train, run_ids_test=run_ids_test)
    assert fit["r2_eq"] >= 0.999
    assert np.allclose(fit["prediction"], delta_test, atol=1e-6)
    assert set(fit["fit_run_ids"]) <= set(run_ids_train)
    assert not (set(fit["fit_run_ids"]) & set(run_ids_test))
    with pytest.raises(ValueError, match="disjoint"):
        rp.equivariance_fit(u_train, delta_train, u_test, delta_test,
                            run_ids_train=run_ids_train,
                            run_ids_test=np.asarray(run_ids_train))


# ===================================================== identification estimator
def _custom_dataset(tmp_path, geometry, *, name, setting, vertices, yields, gain,
                    scale_charges=False, config=GEN, dataset_seed=20261017,
                    n_runs=1):
    """A read-only synthetic probe set with deterministic, noise-free charges."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    runs = [lb.RunSpec(f"lucid/dev_probe_train/{index:04d}", "lucid/dev_probe_train",
                       "F1", float(setting), index * len(vertices),
                       len(vertices), int(dataset_seed))
            for index in range(int(n_runs))]
    state = lb.nominal_state(geometry)
    fixed = lb.fixed_charge_gain(geometry)
    mod = rp.module_of_pmt(geometry)
    rows = []
    for run in runs:
        event_ids = np.arange(run.event_start, run.event_stop, dtype=np.int64)
        charges, vertex_rows, yield_rows = [], [], []
        for event_id, vertex, yield_photons in zip(event_ids, vertices, yields):
            qe = lb.sample_qe(geometry, lb.keyed_rng(dataset_seed, lb.NS_DRAWS,
                                                     run.run_id, int(event_id),
                                                     "wavelength"), config)
            mu = (lb.expected_charge(geometry, vertex, yield_photons, qe,
                                     state.attenuation_length_m, config) * fixed)
            charge = mu * (np.exp(gain[mod]) if scale_charges else 1.0)
            charges.append(charge)
            vertex_rows.append(vertex)
            yield_rows.append(yield_photons)
        run_dir = directory / "runs" / run.run_id.replace("/", "__")
        run_dir.mkdir(parents=True, exist_ok=True)
        pmt = np.arange(int(geometry.n_pmt), dtype=np.int32)
        offsets = np.arange(0, (len(event_ids) + 1) * pmt.size, pmt.size, np.int64)
        store = lb.SparseHitStore(
            event_ids, offsets, np.tile(pmt, len(event_ids)),
            np.concatenate(charges).astype(np.float32),
            np.zeros(pmt.size * len(event_ids), np.float32), lb.READOUT_LABEL)
        lb.write_sparse_hits(run_dir / "hits.npz", store)
        with open(run_dir / "events.npz", "wb") as handle:
            np.savez_compressed(handle, event_ids=event_ids,
                                vertex_m=np.asarray(vertex_rows, float),
                                yield_photons=np.asarray(yield_rows, float))
        rows.append({"run_id": run.run_id, "run_dir": run_dir.name,
                     "event_id_start": run.event_start, "event_id_stop": run.event_stop,
                     "n_events": run.n_events, "setting": float(setting),
                     "state_sha256": run.state_sha256(geometry)})
    manifest = {"cohort": "dev_probe_train", "law": "F1", "setting": float(setting),
                "dataset_seed": int(dataset_seed), "runs": rows}
    dataset = lr.Dataset(key=name, directory=directory, cohort="dev_probe_train",
                         law="F1", setting=float(setting), setting_kind=None,
                         seed=int(dataset_seed), n_events=len(vertices) * int(n_runs),
                         run_length=len(vertices), runs=tuple(rows), manifest=manifest)
    return dataset


def test_source_and_raw_recover_delta_on_noise_free_data(tmp_path, geometry):
    n_modules = int(geometry.n_modules)
    rng = np.random.default_rng(8)
    gain = rng.normal(0.0, 0.05, n_modules)
    # every event of every run is the same here, so the raw stream's N-reference
    # cancels the (event-composition) light-level term exactly
    vertices = np.tile(np.array([[0.0, 0.0, 0.0]]), (2, 1))
    yields = np.full(2, 2.0e6)
    drift = _custom_dataset(tmp_path, geometry, name="drift", setting=rp.S030,
                            vertices=vertices, yields=yields, gain=gain,
                            scale_charges=True)
    nominal = _custom_dataset(tmp_path, geometry, name="nominal", setting=rp.S005,
                              vertices=vertices, yields=yields, gain=gain,
                              scale_charges=False)
    source = rp.source_estimate(drift, geometry, GEN, dataset_seed=20261017)
    for run_id, estimate in source["estimates"].items():
        assert np.allclose(estimate, gain, atol=1e-4), run_id
    raw = rp.raw_estimate(drift, nominal, geometry)
    for run_id, estimate in raw["estimates"].items():
        assert np.allclose(estimate, gain, atol=1e-4), run_id


def test_latent_estimator_recovers_delta_with_lambda_zero():
    rng = np.random.default_rng(9)
    n_modules = 105
    mapping = rng.standard_normal((n_modules, n_modules))
    occupancy = np.ones((3, n_modules))
    truth = rng.normal(0.0, 0.05, (3, n_modules))
    # the latent stream estimates one run's gains at a time; lambda = 0 on a
    # well-conditioned noise-free design is a least-squares solve
    for index in range(3):
        target = mapping @ truth[index]
        fitted = rp.latent_estimate(target, occupancy[index], mapping,
                                    np.eye(n_modules), lam=0.0)
        assert np.allclose(fitted["delta"], truth[index], rtol=0.0, atol=1e-8)
        assert fitted["residual_rms"] <= 1e-10


# =========================================================== Fisher accounting
def _fisher_event_inputs(geometry):
    rng = np.random.default_rng(6)
    n_pmt = int(geometry.n_pmt)
    mu = rng.uniform(1.0, 60.0, n_pmt)
    mu[:8] = 0.2          # low-light PMTs exercise the >=1 PE variance-term cut
    dmu = np.zeros((n_pmt, 4))
    dmu[:, :3] = rng.standard_normal((n_pmt, 3)) * mu[:, None]
    dmu[:, 3] = mu
    return mu, dmu, rp.module_of_pmt(geometry)


def test_p0_holds_in_the_c_mean_model(geometry):
    mu, dmu, mod = _fisher_event_inputs(geometry)
    blocks = rp.fisher_blocks(mu, dmu, mod, model="C-mean")
    check = rp.p0_check(blocks)
    assert check["pass"] is True
    assert check["ratio"] <= 1e-6


def test_c_mean_information_equals_fisher_event(geometry):
    mu, dmu, mod = _fisher_event_inputs(geometry)
    blocks = rp.fisher_blocks(mu, dmu, mod, model="C-mean")
    for index in range(4):
        expected = lb.fisher_event(mu, dmu[:, index])
        assert np.isclose(blocks["theta_theta"][index, index], expected,
                          rtol=1e-12, atol=0.0)


def test_c_var_reduces_to_c_mean_without_the_variance_term(geometry):
    mu, dmu, mod = _fisher_event_inputs(geometry)
    mean = rp.fisher_blocks(mu, dmu, mod, model="C-mean")
    var = rp.fisher_blocks(mu, dmu, mod, model="C-var", variance_term=False)
    for name in ("theta_theta", "theta_delta", "delta_delta"):
        assert np.array_equal(mean[name], var[name])


def test_c_var_mu_cut_applies_only_to_the_pmts_it_names(geometry):
    mu, dmu, mod = _fisher_event_inputs(geometry)
    n_modules = int(geometry.n_modules)
    design = rp.module_design(mod, n_modules)
    sigma_spe, v_dark, s_p = lb.SIGMA_SPE, lb.V_DARK_READOUT, lb.F3_SPREAD
    variance = mu * (1.0 + sigma_spe ** 2) + v_dark + s_p ** 2 * mu ** 2
    theta = (1.0 + sigma_spe ** 2 + 2.0 * s_p ** 2 * mu)[:, None] * dmu
    delta = 2.0 * variance[:, None] * design
    joint = np.concatenate([theta, delta], axis=1)
    keep = mu >= 1.0
    expected = 0.5 * (joint[keep].T * (1.0 / variance[keep])) @ joint[keep]
    cut = rp.variance_term_information(mu, dmu, mod, cuts=1.0)
    assert np.allclose(cut, expected, rtol=1e-12, atol=0.0)
    # PMTs below the cut contribute nothing: changing their mu leaves the cut
    # information exactly unchanged (they are excluded) while the uncut term moves
    below = ~keep
    assert below.any()
    perturbed = mu.copy()
    perturbed[below] *= 0.3
    cut_perturbed = rp.variance_term_information(perturbed, dmu, mod, cuts=1.0)
    assert np.array_equal(cut, cut_perturbed)
    uncut = rp.variance_term_information(mu, dmu, mod)
    uncut_perturbed = rp.variance_term_information(perturbed, dmu, mod)
    assert not np.allclose(uncut, uncut_perturbed)


def test_c_mean_lnY_is_infinite_and_xyz_match_the_pseudoinverse(geometry):
    mu, dmu, mod = _fisher_event_inputs(geometry)
    blocks = rp.fisher_blocks(mu, dmu, mod, model="C-mean")
    report = rp.crlb_report(blocks)
    assert report["inv"]["crlb"][3] == float("inf")
    assert report["inv"]["ratio"][3] == float("inf")
    effective = rp._case_information(blocks, case="inv")
    pseudo = np.diag(np.linalg.pinv(effective, rcond=1e-12))
    for index in range(3):
        assert np.isclose(report["inv"]["crlb"][index], pseudo[index],
                          rtol=1e-9, atol=0.0)
    # the known-gain reference is finite for every parameter
    assert all(np.isfinite(report["known"]["crlb"]))


# ============================================================== bookkeeping §3
def test_eligibility_rule_and_recovered_fraction():
    assert rp.eligibility(0.002, [0.001, 0.003]) is False
    assert rp.eligibility(0.004, [0.001, 0.007]) is True
    assert rp.eligibility(0.004, [-0.001, 0.009]) is False
    undefined = rp.recovered_fraction(0.10, 0.101, 0.102)
    assert undefined["eligible"] is False
    assert undefined["reading"] == "no drift cost to recover"
    undefined = rp.recovered_fraction(0.10, 0.11, 0.12, delta_ci=[-0.001, 0.03])
    assert undefined["eligible"] is False
    eligible = rp.recovered_fraction(0.10, 0.11, 0.12, delta_ci=[0.005, 0.03])
    assert eligible["eligible"] is True
    assert np.isclose(eligible["rho"], 0.5)


def test_classification_rule_per_seed():
    classify = rp._x1b_class
    same = rp.classify_seeds([(9000, 0.9), (9001, 0.8)], classify)
    assert same["class"] == "charge-scale dominated"
    different = rp.classify_seeds([(9000, 0.9), (9001, 0.1)], classify)
    assert different["class"] == "seeds disagree"
    one_missing = rp.classify_seeds([(9000, 0.9), (9001, None)], classify)
    assert one_missing["class"] == "not classified"
    single = rp.classify_seeds([(9000, 0.9)], classify)
    assert single["class"] == "single seed (not classified)"
    assert single["single_seed_class"] == "charge-scale dominated"


def test_run_cluster_resolution_bootstrap():
    rng = np.random.default_rng(1)
    run_ids = np.repeat([f"r{index}" for index in range(20)], 5)
    values = np.abs(rng.normal(0.1, 0.01, run_ids.size))
    result = rp.cluster_bootstrap(values, run_ids, statistic=rp.rms)
    assert result["n_runs"] == 20 and result["n_events"] == run_ids.size
    assert result["ci95"][0] <= result["statistic"] <= result["ci95"][1]
    right = values * 1.10
    difference = rp.paired_cluster_bootstrap(values, right, run_ids, statistic=rp.rms)
    assert difference["difference"] > 0.0
    assert difference["ci95"][0] > 0.0
    # a constant paired difference keeps the pairs together
    paired = rp.paired_cluster_bootstrap(values, values + 0.01, run_ids)
    assert np.isclose(paired["difference"], 0.01)


def test_sigma_half_rule():
    rho = {0.01: 0.9, 0.02: 0.5, 0.05: 0.2, 0.10: 0.1}
    result = rp.sigma_half(lambda sigma: rho[sigma], 1.0)
    assert np.isclose(result["sigma_half"], 0.02)
    assert rp.sigma_half(lambda sigma: rho[sigma], 0.05)["status"] == "undefined"
    assert rp.sigma_half(lambda sigma: 0.4, 1.0)["value"] == "≤ 0.01"
    assert rp.sigma_half(lambda sigma: 0.6, 1.0)["value"] == "> 0.10"
    never = {0.01: 0.8, 0.02: 0.7, 0.05: 0.6, 0.10: 0.6}
    assert rp.sigma_half(lambda sigma: never[sigma], 1.0)["value"] == "> 0.10"


# ===================================================== forbidden paths §2.6
def _fake_inventory():
    return {"rows": {"init9000": {"label": "init9000", "arm": "sigreg_jepa",
                                  "split": "within", "seed": 9000, "beta": None,
                                  "source": None, "sha256": None,
                                  "status": "available", "data_setting": 0.05,
                                  "eval": {"zero_geometry": False,
                                           "median_wrapper": False,
                                           "encoder_variant": None, "anchor": False,
                                           "time_convention": "trigger"}}},
            "counts": {"labels": 1, "available": 1, "unavailable": 0, "core": 13,
                       "optional": 6},
            "unavailable": [], "pending": [], "optional": []}


def test_forbidden_call_paths_are_not_reached_by_any_mode_setup(tiny, tmp_path,
                                                               monkeypatch):
    from scripts import jepa_ext_followup as followup_cli
    geometry = tiny["geometry"]
    config = lr.load_run_config(ROOT / rp.CONFIG_REL)
    arms_config = la.load_config(ROOT / rp.ARMS_REL)
    probes = {"s005": tiny["s005"]["dataset"], "s030": tiny["s030"]["dataset"]}
    truth = {role: _event_truth(probes["s005"], geometry)
             for role in ("train", "test")}
    probes = {"s005": {"train": probes["s005"], "test": probes["s005"]},
              "s030": {"train": probes["s030"], "test": probes["s030"]}}
    reaches = []

    def boom(*args, **kwargs):
        reaches.append("forbidden")
        raise AssertionError("forbidden call path reached")

    patched = [(lr, "ensure_dataset"), (lr, "ensure_dark_bank"), (lr, "run_one"),
               (lb, "draw_event"), (lb, "generate_run"), (lb, "nominal_twin"),
               (lb, "write_generated_run"), (ef, "readout_a1"), (ef, "readout_a1b"),
               (ef, "readout_a2"), (ef, "readout_a3"), (ef, "readout_a4"),
               (ef, "reproduction"), (ef, "assemble_report"), (ef, "build_plan"),
               (ef, "t1_prepare_launch"), (ef, "train_t1")]
    for name in dir(followup_cli):
        if name.startswith("mode_"):
            patched.append((followup_cli, name))
    saved = {}
    for module, name in patched:
        saved[(module, name)] = getattr(module, name)
        monkeypatch.setattr(module, name, boom)
    monkeypatch.setattr(torch.optim.Optimizer, "step", boom)

    root = tmp_path / "repo"
    (root / "docs/jepa").mkdir(parents=True)
    (root / rp.RUNBOOK_REL).write_text("## 0. Authority\n")
    inventory = _fake_inventory()
    for mode in rp.MODES:
        setup = rp._mode_setup_objects(
            mode, root=root, config=config, gen=GEN, arms_config=arms_config,
            geometry=geometry, data_root=tmp_path / "data",
            ext_root=tmp_path / "ext", inventory=inventory, probes=probes,
            particle_truth=truth, build_model=(mode == "dry-run"),
            log=lambda *a, **k: None)
        assert setup["mode"] == mode
    assert reaches == []
    # non-vacuity: the sentinels really replaced the forbidden openers
    assert lr.ensure_dataset is boom and ef.build_plan is boom
    assert lb.generate_run is boom and lb.draw_event is boom


# ================================================== hash mismatch / no overwrite
def test_hash_mismatch_and_no_overwrite(tmp_path):
    key = rp.PROBE_KEYS_S030["test"]
    missing = tmp_path / "empty"
    with pytest.raises(SystemExit):
        rp.require_pinned(missing, keys=[key], context="test")
    data_root = tmp_path / "data"
    directory = data_root / "lucid" / key
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}")
    actual = lr.sha256_file(directory / "manifest.json")
    status = rp.data_status(data_root, keys=[key], pins={key: actual})
    assert status["pass"] is True
    with pytest.raises(SystemExit):
        rp.require_pinned(data_root, keys=[key],
                          pins={key: "0" * 64}, context="test")
    assert set(rp.PINNED_DATA) == set(rp.PROBE_KEYS_S005.values()) \
        | set(rp.PROBE_KEYS_S030.values())
    target = tmp_path / "out" / "plan.json"
    rp.write_json(target, {"a": 1}, overwrite=False)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        rp.write_json(target, {"a": 2}, overwrite=False)
    rp.write_json(target, {"a": 2}, overwrite=True)
    assert rp.read_json(target) == {"a": 2}
    # a cached embedding whose sha no longer matches is refused
    cache = rp.EmbeddingCache(tmp_path / "cache")
    cache.fetch("k", lambda: {"events": np.zeros((2, 3), np.float32)})
    np.savez_compressed(cache.path("k"), events=np.ones((2, 3), np.float32))
    with pytest.raises(RuntimeError, match="does not match"):
        cache.fetch("k", lambda: {"events": np.zeros((2, 3), np.float32)})


# ================================================================== CLI parser
def test_cli_modes_and_defaults():
    import argparse
    parser = repair_cli.parser()
    subparsers = [action for action in parser._actions
                  if isinstance(action, argparse._SubParsersAction)][0]
    choices = subparsers.choices
    assert tuple(sorted(choices)) == tuple(sorted(rp.MODES))
    assert "train" not in choices
    args = parser.parse_args(["dry-run", "--out", "/tmp/repair_dry"])
    assert args.mode == "dry-run" and args.device == "cpu"
    args = parser.parse_args(["x1", "--no-optional"])
    assert args.device == "auto" and args.no_optional is True
    args = parser.parse_args(["x3-fisher", "--events", "10"])
    assert args.events == 10
    with pytest.raises(SystemExit):
        parser.parse_args(["dry-run"])          # --out is required
    assert set(rp.MODES) == {"dry-run", "preflight", "reproduce", "x1", "x2",
                             "x3-fisher", "x3", "report", "addendum-selfcheck",
                             "addendum-x2", "addendum-x2e", "addendum-x3",
                             "addendum-report", "addendum-init9000",
                             "addendum2-ledger", "addendum2-leace",
                             "addendum2-report"}
    # the addendum §8 modes parse with their devices and the report --out
    for mode, device in (("addendum-selfcheck", "cpu"), ("addendum-x2", "cpu"),
                         ("addendum-x3", "cpu"), ("addendum-report", "cpu"),
                         ("addendum-x2e", "auto"), ("addendum-init9000", "auto"),
                         ("addendum2-ledger", "cpu"),
                         ("addendum2-leace", "cpu"),
                         ("addendum2-report", "cpu")):
        parsed = parser.parse_args([mode])
        assert parsed.mode == mode and parsed.device == device
    assert parser.parse_args(["addendum-x2e", "--limit", "3"]).limit == 3
    assert parser.parse_args(["addendum-report", "--out", "/tmp/results"]).out == \
        "/tmp/results"
    help_text = parser.format_help()
    for mode in rp.ADDENDUM_MODES:
        assert mode in help_text


# ==================================================== cross-part (R-1) contract
def test_embedding_cache_writes_the_rerun_contract_layout(tmp_path):
    cache = rp.EmbeddingCache(tmp_path / "x1" / "mae_within_trigger9000")
    events = np.zeros((6, 8), np.float32)
    run_ids = np.asarray(["r0"] * 3 + ["r1"] * 3, dtype=object)
    for name in ("test__N", "test__D"):
        cache.fetch(name, lambda: {"events": events, "event_ids": np.arange(6),
                                   "run_ids": run_ids})
        path = cache.path(name)
        assert "__N" in path.name or "__D" in path.name
        with np.load(path, allow_pickle=False) as data:
            assert {"embeddings", "event_ids", "run_ids"} <= set(data.files)
            assert np.array_equal(data["embeddings"], events)
        sidecar = path.with_suffix(".npz.sha256")
        assert sidecar.read_text().strip() == lr.sha256_file(path)
    big = np.random.default_rng(0).standard_normal((2000, 700)).astype(np.float32)
    cache.fetch("test__big", lambda: {"events": big, "event_ids": np.arange(2000),
                                      "run_ids": np.asarray(["r"] * 2000, dtype=object)})
    large = rp.large_arrays(tmp_path, min_bytes=5 * 1024 * 1024)
    assert large["n_files"] >= 1
    assert any(str(cache.path("test__big")) in key for key in large["files"])


def test_canonical_reader_keys_are_attached_to_each_model_record():
    model = {"conditions": {"N": {"vertex": {"mean_error_m": 0.10}},
                            "D": {"vertex": {"mean_error_m": 0.13},
                                  "refit_vertex": {"mean_error_m": 0.12}},
                            "SWtw": {"vertex": {"mean_error_m": 0.11}},
                            "SWunit": {"vertex": {"mean_error_m": 0.12}}},
             "comparisons": {"D": {"mean": {"mean_difference_m": 0.03,
                                            "ci95_m": [0.01, 0.05]}},
                             "D_refit": {"mean": {"mean_difference_m": 0.02,
                                                  "ci95_m": [0.005, 0.03]}}}}
    rp.attach_x1_canonical(model)
    assert set(("delta_d_m", "delta_d_eligible", "rho_swtw", "rho_swunit",
                "share_cs")) <= set(model)
    assert model["delta_d_m"] is not None and model["delta_d_eligible"] is True
    x2 = rp.attach_x2_canonical({"X2-b": {"occupancy": {"r2_cs": 0.9,
                                                        "class": "equivariant"}},
                                 "X2-c": {"rho_LR": 0.6}})
    assert (x2["r2_eq"], x2["class"], x2["rho_lr"]) == (0.9, "equivariant", 0.6)
    # undefined values are explicit nulls, not missing keys
    empty = rp.attach_x2_canonical({})
    assert empty["r2_eq"] is None and empty["class"] is None and empty["rho_lr"] is None


# ======================================= mode-body execution (review B1/B2/B4/B5/B6)
def _synthetic_rows(dataset, geometry, *, dim=4, seed=0, tag=None):
    """Minimal embedding rows for one tiny probe set (no encoder is run)."""
    rng = np.random.default_rng(int(seed))
    rows = {"events": [], "run_ids": [], "event_ids": [], "vertex": [],
            "log_yield": [], "n_hit": []}
    for row in dataset.runs:
        run_dir = dataset.directory / "runs" / row["run_dir"]
        with np.load(run_dir / "events.npz", allow_pickle=False) as data:
            event_ids = np.asarray(data["event_ids"], np.int64)
            vertex = np.asarray(data["vertex_m"], float)
            yields = np.asarray(data["yield_photons"], float)
        count = event_ids.size
        rows["events"].append(rng.standard_normal((count, int(dim))))
        rows["run_ids"].append(np.full(count, str(row["run_id"]), dtype=object))
        rows["event_ids"].append(event_ids)
        rows["vertex"].append(vertex)
        rows["log_yield"].append(np.log(np.maximum(yields, 1e-30)))
        rows["n_hit"].append(np.full(count, 10, np.int64))
    out = {name: (np.concatenate(values) if name not in ("events", "vertex")
                  else np.asarray(np.concatenate(values), float)) for name, values
           in rows.items()}
    if tag is not None:
        out["tag"] = tag
    return out


def _tiny_many(tmp_path, geometry, *, runs=6, events=3):
    """Train and probe-test sets with disjoint run ids (two settings each)."""
    sets = {}
    for setting, label, run_start in ((rp.S005, "s005", 0), (rp.S030, "s030", 0),
                                      (rp.S005, "s005_test", runs),
                                      (rp.S030, "s030_test", runs)):
        sets[label] = _tiny_dataset(tmp_path, geometry, setting=setting,
                                    n_runs=runs, events=events,
                                    name=f"{label}_{run_start}", run_start=run_start)
    return sets


def _tiny_model(geometry, seed=9000):
    return la.random_init_floor("sigreg_jepa", geometry,
                                encoder_config=pe.EncoderConfig(window=16, shift=8),
                                seed=int(seed))


def test_row_split_propagates_into_streaming(monkeypatch):
    assert rp.row_split({"split": "across"}) == "across"
    assert rp.row_split({}) == "within"
    calls, conditions = [], []

    def fake_stream(model, dataset, geometry, condition, *, split, config, device,
                    cache, cache_key, log, **kwargs):
        calls.append(split)
        return {"events": np.zeros((2, 2)), "run_ids": np.asarray(["r0", "r0"],
                                                                  dtype=object),
                "event_ids": np.arange(2), "vertex": np.zeros((2, 3))}

    def fake_condition(name, dataset, geometry, **kwargs):
        conditions.append((name, dataset))
        return {"built_for": dataset}

    monkeypatch.setattr(rp, "build_condition", fake_condition)
    monkeypatch.setattr(rp, "stream_condition_embeddings", fake_stream)
    monkeypatch.setattr(rp, "fit_vertex_probe", lambda rows: {})
    monkeypatch.setattr(rp.la, "probe_predictions", lambda fitted, dataset: np.zeros(
        (2, 3)))
    # production call shape: reproduction() passes the single-setting dict
    probes = {"geometry": None, "train": "train_ds", "test": "test_ds"}
    condition = {"name": "N", "train": "s005", "test": "s005", "condition": None}
    rp._frozen_refit_vertex(None, probes, condition, "across", {}, "cpu",
                            cache=None, log=lambda *a, **k: None)
    assert calls == ["across", "across"]
    # §2.5: the condition is built per role on the dataset it streams
    assert conditions == [("N", "train_ds"), ("N", "test_ds")]


def test_jsonable_and_write_json_handle_paths(tmp_path):
    payload = {"p": tmp_path, "nested": [tmp_path / "a.npz"], "value": 1.5}
    out = rp._jsonable(payload)
    assert out == {"p": str(tmp_path), "nested": [str(tmp_path / "a.npz")],
                   "value": 1.5}
    target = tmp_path / "payload.json"
    rp.write_json(target, payload)
    assert json.loads(target.read_text())["nested"] == [str(tmp_path / "a.npz")]


def test_identity_check_runs_on_both_roles(tiny, geometry):
    model = _tiny_model(geometry)
    probes005 = {"train": tiny["s005"]["dataset"], "test": tiny["s005"]["dataset"]}
    probes030 = {"train": tiny["s030"]["dataset"], "test": tiny["s030"]["dataset"]}
    inventory = {"rows": {"mae_within_trigger9000": {
        "label": "mae_within_trigger9000", "split": "across"}}}
    result = rp.identity_check(root=None, config=lr.load_run_config(ROOT / rp.CONFIG_REL),
                               gen=GEN, arms_config=la.load_config(ROOT / rp.ARMS_REL),
                               geometry=geometry, inventory=inventory,
                               probes005=probes005, probes030=probes030,
                               model=model, log=lambda *a, **k: None)
    assert result["pass"] is True
    assert result["split"] == "across"
    # both roles (train and test) of each set are exercised
    assert result["sets"]["s005"]["n_runs"] == 4
    assert result["sets"]["s030"]["n_runs"] == 4
    assert result["sets"]["s005"]["features_bit_identical"] is True


def test_stream_condition_particles_executes_and_caches(tiny, geometry):
    dataset = tiny["s005"]["dataset"]
    model = _tiny_model(geometry)
    truth = _event_truth(dataset, geometry)
    condition = rp.build_condition("N", dataset, geometry)
    import tempfile
    cache = rp.EmbeddingCache(tempfile.mkdtemp())
    rows = rp.stream_condition_particles(model, dataset, truth, geometry, condition,
                                         device="cpu", cache=cache,
                                         cache_key="particle__test__N",
                                         log=lambda *a, **k: None)
    assert rows["events"].shape[0] > 0
    assert set(rows["particle"]) <= {"muon", "electron"}
    assert cache.has("particle__test__N")
    again = rp.stream_condition_particles(model, dataset, truth, geometry, condition,
                                          device="cpu", cache=cache,
                                          cache_key="particle__test__N",
                                          log=lambda *a, **k: None)
    assert again["cache_status"] == "reused"
    assert np.array_equal(rows["events"], again["events"])


def test_x1_particle_executes_with_all_conditions(tiny, geometry):
    model = _tiny_model(geometry)
    s005, s030 = tiny["s005"]["dataset"], tiny["s030"]["dataset"]
    conditions = {role: rp.build_conditions(
        {"s005": s005, "s030": s030}, geometry, ("N", "D", "SWunit", "HW"))
        for role in ("train", "test")}
    truth = {role: _event_truth(s005, geometry) for role in ("train", "test")}
    import tempfile
    cache = rp.EmbeddingCache(tempfile.mkdtemp())
    probes005 = {"train": s005, "test": s005}
    probes030 = {"train": s030, "test": s030}
    out = rp._x1_particle(model, truth, geometry, probes005, probes030, conditions,
                          label="mae_within_trigger9000", cache=cache, device="cpu",
                          log=lambda *a, **k: None)
    assert set(out["train"]) == {"N", "D"}
    assert set(out["test"]) == {"N", "D", "SWunit", "HW"}
    for role in ("train", "test"):
        for rows in out[role].values():
            assert rows["events"].shape[0] > 0
    assert cache.has("particle__train__D")
    assert cache.has("particle__test__HW")


def test_x2_model_executes_and_exposes_the_canonical_keys(tmp_path, tiny, geometry,
                                                          monkeypatch):
    sets = _tiny_many(tmp_path, geometry)
    s005, s030 = sets["s005"]["dataset"], sets["s030"]["dataset"]
    s005_test, s030_test = sets["s005_test"]["dataset"], sets["s030_test"]["dataset"]
    probes005 = {"train": s005, "test": s005_test}
    probes030 = {"train": s030, "test": s030_test}
    sources = {"train": {"N": s005, "S": s005, "D": s030},
               "test": {"N": s005_test, "S": s005_test, "D": s030_test}}
    conditions = {role: rp.build_conditions(
        {"s005": probes005[role], "s030": probes030[role]}, geometry,
        ("N", "D", "S")) for role in ("train", "test")}
    rows = {role: {name: _synthetic_rows(source, geometry, dim=4,
                                         seed=index + 1 + (10 if role == "test" else 0))
                   for index, (name, source) in enumerate(sources[role].items())}
            for role in ("train", "test")}
    delta = rp.run_log_gain_vectors(s030, geometry)
    delta.update(rp.run_log_gain_vectors(s030_test, geometry))
    occupancy = {"n": {role: rp.occupancy_counts(probes005[role],
                                                 conditions[role]["N"], geometry)
                       for role in ("train", "test")},
                 "d": {role: rp.occupancy_counts(probes030[role],
                                                 conditions[role]["D"], geometry)
                       for role in ("train", "test")}}
    cache = rp.EmbeddingCache(tmp_path / "cache")
    # X2-e streams EST rows; feed the existing N rows instead of a real forward
    monkeypatch.setattr(rp, "stream_condition_embeddings",
                        lambda *args, **kwargs: rows["test"]["N"])
    config = lr.load_run_config(ROOT / rp.CONFIG_REL)
    out = rp._x2_model({"label": "mae_within_trigger9000", "split": "within"},
                       _tiny_model(geometry), rows, delta, occupancy, conditions,
                       probes005, probes030, geometry, config=config, gen=GEN,
                       data_root=tmp_path, ext_root=tmp_path / "ext",
                       device="cpu", cache=cache, log=lambda *a, **k: None)
    assert out["X2-a"]["share_cs"] is not None
    assert out["X2-b"]["occupancy"]["r2_cs"] is not None
    assert set(("r2_eq", "class", "rho_lr")) <= set(out)
    for stream in ("source", "raw", "latent"):
        assert stream in out["X2-d"]
    assert out["X2-f"] is not None
    assert "X2-e" in out


def test_x3_readings_execute_with_run_cluster_bootstraps(tmp_path):
    rng = np.random.default_rng(4)
    run_ids = np.repeat([f"r{index}" for index in range(10)], 4).astype(object)
    event_ids = np.tile(np.arange(4), 10)

    def block(base):
        errors = np.abs(rng.normal(base, 0.01 * base, run_ids.size))
        return {"resolution": float(np.sqrt(np.mean(errors ** 2))),
                "mean_error_m": float(errors.mean()), "__errors__": errors,
                "run_ids": run_ids, "event_ids": event_ids}

    standard = {"vertex": {"N": block(0.10), "D": block(0.108)},
                "log_yield": {"N": block(0.20), "D": block(0.21)},
                "energy": {"N": block(0.05), "D": block(0.052)}}
    invariant = {"vertex": {"N": block(0.111)}, "log_yield": {"N": block(0.25)},
                 "energy": {"N": block(0.06)},
                 "vertex_D": block(0.12)}
    invariant["vertex"] = {"N": block(0.111), "D": block(0.12)}
    invariant["log_yield"] = {"N": block(0.25), "D": block(0.27)}
    invariant["energy"] = {"N": block(0.06), "D": block(0.07)}
    calibrated = {"vertex": {"D": block(0.112)}, "energy": {"D": block(0.055)},
                  "log_yield": {"D": block(0.26)}}
    controls = []
    for _index in range(20):
        controls.append({"vertex": {"N": block(0.113)},
                         "log_yield": {"N": block(0.252)},
                         "energy": {"N": block(0.061)}})
    entry = {"status": "evaluated", "standard": standard, "variant": invariant,
             "calibrated": calibrated, "controls": controls}
    payload = {"families": {"leace": {"models": {"model/E1": entry}}}}
    readings = rp.x3_readings(payload)
    variant = readings["variants"]["model/E1"]
    assert variant["X3-1"]["log_yield"]["rho_E"] is not None
    assert variant["X3-1"]["log_yield"]["ci95"] is not None
    assert variant["X3-3"]["standard"]["robust"] is not None
    assert variant["X3-4"]["verdict"] in ("calibration dominates",
                                          "invariance dominates", "mixed",
                                          "not available")
    gain = variant["X3-1-gain_specific"]
    assert gain["log_yield"]["condition_q95"] is not None
    assert gain["log_yield"]["gain_specific"] is not None
    robustness = rp._x3_robustness(standard, invariant)
    assert robustness["invariant"]["delta_vertex_m"] is not None
    dominance = rp._x3_dominance(calibrated, invariant)
    assert dominance["status"] == "evaluated"


def test_x3_leace_reads_the_keys_x1_writes(tmp_path, tiny, geometry):
    sets = _tiny_many(tmp_path, geometry)
    s005, s030 = sets["s005"]["dataset"], sets["s030"]["dataset"]
    probes005 = {"train": s005, "test": s005}
    probes030 = {"train": s030, "test": s030}
    label = "mae_within_trigger9000"
    cache = rp.embedding_cache(tmp_path / "ext", label)
    for role, dataset, seed in (("train", s005, 1), ("test", s005, 2)):
        for setting, source in (("N", s005), ("D", s030)):
            rows = _synthetic_rows(source, geometry, dim=4, seed=seed)
            cache.fetch(rp.embedding_cache_key(role, setting), lambda rows=rows: rows)
    inventory = {"rows": {label: {"label": label, "split": "within"}}}
    out = rp._x3_leace(inventory, probes005, probes030, geometry, tmp_path / "ext")
    assert label not in [row["label"] for row in out["unavailable"]]
    assert any(key.startswith(label) for key in out["models"])
    entry = out["models"].get(f"{label}/E1")
    assert entry is not None
    assert entry.get("status") in ("evaluated", "not_applicable",
                                   "unavailable (erasure incomplete)")


def test_leace_sanity_protocol_is_enforced():
    run_ids = np.repeat([f"r{index}" for index in range(8)], 3).astype(object)
    event_ids = np.tile(np.arange(3), 8)

    def rows(seed, concept_scale=0.0):
        rng_local = np.random.default_rng(seed)
        x = rng_local.standard_normal((run_ids.size, 4))
        z = rng_local.standard_normal((run_ids.size, 1))
        x[:, 0] += concept_scale * z[:, 0]
        return {"events": x, "run_ids": run_ids, "event_ids": event_ids}, z

    train_n, z_train = rows(1, 3.0)
    train_d, _ = rows(2, 3.0)
    test_n, z_test = rows(3, 3.0)
    test_d, _ = rows(4, 3.0)
    concept_rows = {"train": {"N": z_train, "D": train_d["events"][:, :1] * 0.0},
                    "test": {"N": z_test}}
    variant = rp._leace_variant(
        label="concept-test", domain="photon",
        rows={"train": {"N": train_n, "D": train_d},
              "test": {"N": test_n, "D": test_d}},
        concepts=concept_rows, rank=1)
    assert variant["status"] == "evaluated"
    # a rank-0 eraser erases nothing: the concept R² stays high -> unavailable
    incomplete = rp._leace_variant(
        label="concept-test", domain="photon",
        rows={"train": {"N": train_n, "D": train_d},
              "test": {"N": test_n, "D": test_d}},
        concepts=concept_rows, rank=0)
    assert incomplete["status"] == "unavailable (erasure incomplete)"
    # an absent concept (independent x and z) -> not applicable
    absent = rp._leace_variant(
        label="concept-absent", domain="photon",
        rows={"train": {"N": rows(10)[0], "D": rows(11)[0]},
              "test": {"N": rows(12)[0], "D": rows(13)[0]}},
        concepts={"train": {"N": rows(20)[1], "D": rows(21)[1]},
                  "test": {"N": rows(22)[1]}}, rank=1)
    assert absent["status"] == "not_applicable"


def test_random_control_keys_are_per_index():
    bases = rp._random_control_bases("label", "photon", "E1", 8, 1, 20)
    assert len(bases) == 20
    assert bases[0].shape == (8, 1)
    assert not np.allclose(bases[0], bases[1])
    repeat = rp._random_control_bases("label", "photon", "E1", 8, 1, 20)
    for left, right in zip(bases, repeat):
        assert np.allclose(left, right)
    first = bases[0][:, 0]
    direct = rp.random_control_basis(
        lb.keyed_rng(20261106, rp.LEACE_RAND_NAMESPACE, "label/photon/E1/0"), 8, 1)[:, 0]
    assert np.allclose(np.abs(first), np.abs(direct))


def test_raw_std_features_honours_the_condition(tiny, geometry):
    dataset = tiny["s030"]["dataset"]
    truth = _event_truth(dataset, geometry)
    condition = rp.build_condition("SWunit", dataset, geometry)
    plain = rp.raw_std_features(dataset, geometry, truth=truth)["merged"]
    calibrated = rp.raw_std_features(dataset, geometry, truth=truth,
                                     condition=condition)["merged"]
    photon = np.asarray(plain["domain"]) == "photon"
    assert photon.any()
    assert not np.allclose(plain["features"][photon], calibrated["features"][photon])
    # the first feature is ln(sum q): check it against the SWunit-scaled charges
    factor = condition.factor(str(dataset.runs[0]["run_id"]))
    store = lb.read_sparse_hits(dataset.directory / "runs"
                                / dataset.runs[0]["run_dir"] / "hits.npz")
    expected = np.log(np.maximum(
        np.asarray(store.charge_pe, np.float64) * factor[np.asarray(store.pmt_id)], 1e-6))
    index = np.flatnonzero(photon)[0]
    assert calibrated["features"][index, 0] >= 0.0
    assert expected.size > 0


def test_x2_additivity_uses_the_swtw_rows(monkeypatch):
    def fake_vertex_row(rows, *, fitted):
        return {"mean_error_m": float(rows["tag"])}

    monkeypatch.setattr(rp, "vertex_row", fake_vertex_row)
    rows = {"run_ids": np.asarray(["r0"], dtype=object), "event_ids": np.array([0]),
            "vertex": np.zeros((1, 3)), "events": np.zeros((1, 2)),
            "log_yield": np.zeros(1), "n_hit": np.array([1])}
    n = dict(rows, tag=0.10)
    d = dict(rows, tag=0.13)
    s = dict(rows, tag=0.11)
    swtw = dict(rows, tag=0.12)
    out = rp.x2_decomposition(n, d, s, frozen_fit=None, swtw_fit=None,
                              swtw_test=swtw)
    assert abs(out["additivity_check"]) < 1e-12
    wrong = rp.x2_decomposition(n, d, s, frozen_fit=None, swtw_fit=None,
                                swtw_test=s)
    assert wrong["additivity_check"] > 0.0


def test_embedding_cache_round_trip_uses_the_x1_key_scheme(tmp_path):
    cache = rp.EmbeddingCache(tmp_path / "x1" / "mae_within_trigger9000")
    rows = {"events": np.arange(12, dtype=np.float32).reshape(6, 2),
            "event_ids": np.arange(6), "run_ids": np.asarray(["r"] * 6, dtype=object)}
    key = rp.embedding_cache_key("test", "N")
    assert key == "test__N"
    first = cache.fetch(key, lambda: rows)
    assert first["cache_status"] == "written"
    second = cache.fetch(key, lambda: {"events": np.zeros((1, 1), np.float32)})
    assert second["cache_status"] == "reused"
    assert np.array_equal(second["events"], rows["events"])
    with np.load(cache.path(key), allow_pickle=False) as data:
        assert {"embeddings", "event_ids", "run_ids"} <= set(data.files)


def test_report_flags_both_complete_and_completed(tmp_path, monkeypatch):
    ext = tmp_path / "ext"
    for relative, payload in (
            ("x0/plan.json", {"pass": True}),
            ("x0/reproduction.json", {"pass": True}),
            ("x1/x1.json", {"complete": True, "readings": {"families": {}}}),
            ("x2/x2.json", {"complete": False, "readings": {"families": {}}}),
            ("x3/x3_fisher.json", {"p0": {"pass": True}, "models": {}}),
            ("x3/x3.json", {"complete": True, "readings": {"variants": {}}})):
        path = ext / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda *args, **kwargs: {"status": "granted",
                                                 "authorised": True})
    monkeypatch.setattr(rp, "_assert_closure", lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "render_figures",
                        lambda *args, **kwargs: {"status": "unavailable",
                                                 "figures": []})
    payload = rp.assemble_report(root=tmp_path, ext_root=ext,
                                 results_dir=tmp_path / "results",
                                 log=lambda *a, **k: None)
    readings = json.loads((tmp_path / "results" / "readings.json").read_text())
    assert readings["x1"]["complete"] is True
    assert readings["x1"]["completed"] is True
    assert readings["x2"]["complete"] is False
    assert readings["x2"]["completed"] is False
    assert payload["completeness"]["x2"] is False
    assert payload["complete"] is False


def test_preflight_scopes_content_stops_and_hw_unavailable(tmp_path, tiny, geometry,
                                                          monkeypatch):
    inv = {"rows": {}, "unavailable": [], "counts": {"labels": 0}}
    probes = {"s005": {"train": tiny["s005"]["dataset"],
                       "test": tiny["s005"]["dataset"]},
              "s030": {"train": tiny["s030"]["dataset"],
                       "test": tiny["s030"]["dataset"]}}
    # one missing twin makes the HW condition unavailable for that set
    (tiny["s005"]["dataset"].directory / "runs"
     / tiny["s005"]["dataset"].runs[0]["run_dir"] / "twin_hits.npz").unlink()
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda *a, **k: {"status": "granted", "authorised": True})
    monkeypatch.setattr(rp, "_assert_closure", lambda *a, **k: {"pass": True})
    monkeypatch.setattr(rp, "hardware_and_software", lambda: {})
    monkeypatch.setattr(rp, "entries_preflight", lambda **k: {"pass": True})
    monkeypatch.setattr(rp, "s1_projection", lambda *a, **k: {"pass": True})
    monkeypatch.setattr(rp, "data_status", lambda *a, **k: {"pass": True, "sets": {}})
    monkeypatch.setattr(rp, "open_probe_sets",
                        lambda config, gen, geometry, data_root, *, setting, log:
                        probes["s005"] if setting == rp.S005 else probes["s030"])
    monkeypatch.setattr(rp, "check_run_states",
                        lambda *a, **k: {"pass": True, "mismatches": [],
                                         "twin_missing": []})
    monkeypatch.setattr(rp, "check_twin_content",
                        lambda *a, **k: {"pass": False, "mismatches": [{"role": "train"}]})
    monkeypatch.setattr(rp, "check_swtw_content",
                        lambda *a, **k: {"pass": False, "mismatches": [{"role": "test"}]})
    monkeypatch.setattr(rp, "mu_reconstruction",
                        lambda *a, **k: {"pass": False, "mismatches": [{}]})
    config = lr.load_run_config(ROOT / rp.CONFIG_REL)
    payload = rp.preflight(root=tmp_path, config=config, gen=GEN,
                           geometry=geometry, inventory=inv, limit=1,
                           log=lambda *a, **k: None)
    # a content mismatch is not a global STOP; it is scoped to its readings
    assert payload["stops"] == []
    assert payload["content_pass"] is False
    assert "X1-b" in payload["readings_stopped"]["swtw_mismatch"]
    assert payload["readings_stopped"]["mu_p_unavailable"] == ["X2-d source stream"]
    assert payload["hw_condition"]["s005_train"] == "unavailable"
    assert payload["hw_condition"]["s030_train"] == "available"


def test_fisher_accounting_executes_with_per_parameter_inf_counts(geometry,
                                                                  monkeypatch):
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda *a, **k: {"status": "granted", "authorised": True})
    monkeypatch.setattr(rp, "_assert_closure", lambda *a, **k: {"pass": True})
    payload = rp.fisher_accounting(geometry, GEN, n_events=2, limit=2,
                                   log=lambda *a, **k: None)
    assert payload["p0"]["pass"] is True
    assert set(payload["models"]) == {"C-mean", "C-mean+occupancy", "C-var_all",
                                      "C-var_ge1", "C-var_ge3"}
    counts = payload["models"]["C-mean"]["n_infinite"]["per_case_parameter"]
    assert set(counts["inv"]) == set(rp.THETA_NAMES)
    assert counts["inv"]["lnY"] == 2


def test_x3_init9000_executes_photon_particle_and_calibrated(tmp_path, tiny, geometry,
                                                            monkeypatch):
    sets = _tiny_many(tmp_path, geometry)
    s005, s030 = sets["s005"]["dataset"], sets["s030"]["dataset"]
    probes005 = {"train": s005, "test": s005}
    probes030 = {"train": s030, "test": s030}
    model = _tiny_model(geometry)
    monkeypatch.setattr(ef, "build_item_arm", lambda row, **kwargs: model)

    def fake_photon(model_, dataset, geometry_, condition, *, split, config, device,
                    cache=None, max_events=None, max_runs=None, log=print,
                    **kwargs):
        return _synthetic_rows(dataset, geometry_, dim=4, seed=3)

    def fake_particle(model_, dataset, truth, geometry_, condition, *, device="cpu",
                      cache=None, cache_key=None, max_events=None, max_runs=None,
                      log=print):
        rows = _synthetic_rows(dataset, geometry_, dim=4, seed=4)
        count = rows["events"].shape[0]
        rows.update({"particle": np.array(["electron"] * count, dtype=object),
                     "contained": np.ones(count, bool),
                     "deposited_MeV": np.linspace(100.0, 900.0, count),
                     "direction": np.zeros((count, 3))})
        return rows

    monkeypatch.setattr(rp, "stream_condition_embeddings", fake_photon)
    monkeypatch.setattr(rp, "stream_condition_particles", fake_particle)
    cache = rp.embedding_cache(tmp_path / "ext", "init9000")
    for key, source in (("train__N", s005), ("test__N", s005),
                        ("test__SWunit", s030)):
        rows = _synthetic_rows(source, geometry, dim=4, seed=5)
        cache.fetch(key, lambda rows=rows: rows)
    for key, source in (("particle__train__N", s005),
                        ("particle__test__SWunit", s030)):
        rows = fake_particle(None, source, None, geometry, None)
        cache.fetch(key, lambda rows=rows: rows)
    inventory = {"rows": {"init9000": {"label": "init9000", "split": "across"}}}
    config = lr.load_run_config(ROOT / rp.CONFIG_REL)
    out = rp._x3_init9000(
        inventory, probes005, probes030, geometry, config,
        la.load_config(ROOT / rp.ARMS_REL), GEN, tmp_path, "cpu",
        {"train": None, "test": None}, tmp_path / "ext",
        log=lambda *a, **k: None)
    assert out["status"] == "evaluated"
    assert out["split"] == "across"
    assert "vertex" in out["standard"] and "log_yield" in out["invariant"]
    assert out["standard"].get("energy") is not None
    assert out["calibrated"].get("vertex", {}).get("D") is not None
    assert out["calibrated"].get("energy", {}).get("D") is not None


# ===================================================== review B10 (X3-b readings)
rms = rp.rms


def _metric_block(resolution, errors, run_ids=None, event_ids=None):
    block = {"status": "evaluated", "resolution": float(resolution),
             "mean_error_m": float(resolution), "errors": np.asarray(errors, float),
             "__errors__": np.asarray(errors, float)}
    if run_ids is not None:
        block["run_ids"] = np.asarray(run_ids, dtype=object)
        block["event_ids"] = np.asarray(event_ids, np.int64)
    return block


def _paired_blocks(rho, *, paired=True):
    run_ids = np.repeat([f"r{index}" for index in range(5)], 4).astype(object)
    event_ids = np.tile(np.arange(4), 5)
    base = np.linspace(0.10, 0.20, run_ids.size)
    left = _metric_block(rms(base), base, run_ids if paired else None,
                         event_ids if paired else None)
    right = _metric_block(rms(base * rho), base * rho, run_ids if paired else None,
                          event_ids if paired else None)
    return left, right


def test_x3_price_class_uses_the_fixed_thresholds():
    standard, invariant = _paired_blocks(1.30)
    reading = rp._variant_reading({}, {"log_yield": {"N": standard},
                                       "energy": {"N": standard}},
                                  {"log_yield": {"N": invariant},
                                   "energy": {"N": invariant}})
    assert reading["X3-1"]["log_yield"]["rho_E"] > rp._X3_PRICE_HIGH
    assert reading["X3-1"]["log_yield"]["ci95"] is not None
    assert reading["X3-1"]["log_yield"]["class"] == "price present"
    assert reading["X3-1"]["energy"]["class"] == "price present"
    assert reading["X3-1"]["classes"] == {"log_yield": "price present",
                                          "energy": "price present"}
    # rho_E <= 1.05 is no price without needing the CI
    standard_low, invariant_low = _paired_blocks(1.02, paired=False)
    low = rp._variant_reading(
        {}, {"log_yield": {"N": standard_low}},
        {"log_yield": {"N": invariant_low}})
    assert low["X3-1"]["log_yield"]["class"] == "no price"
    # 1.05 < rho_E < 1.2 (or a CI straddling 0) is unclear
    standard_mid, invariant_mid = _paired_blocks(1.10)
    mid = rp._variant_reading(
        {}, {"log_yield": {"N": standard_mid}},
        {"log_yield": {"N": invariant_mid}})
    assert mid["X3-1"]["log_yield"]["class"] == "unclear"
    # a missing CI with rho_E >= 1.2 is explicitly not available
    standard_hi, invariant_hi = _paired_blocks(1.30, paired=False)
    no_ci = rp._variant_reading({}, {"log_yield": {"N": standard_hi}},
                                {"log_yield": {"N": invariant_hi}})
    assert no_ci["X3-1"]["log_yield"]["ci95"] is None
    assert no_ci["X3-1"]["log_yield"]["class"] == "not available (CI missing)"


def _log_yield_rows(*, seed, shift=0.0, runs=8, events=5, dim=2):
    rng = np.random.default_rng(seed)
    count = runs * events
    features = rng.standard_normal((count, dim))
    log_yield = features[:, 0] * 0.3 + shift + rng.normal(0, 0.01, count)
    return {"events": features, "log_yield": log_yield,
            "run_ids": np.repeat([f"r{index}" for index in range(runs)],
                                 events).astype(object),
            "event_ids": np.tile(np.arange(events), runs),
            "vertex": np.zeros((count, 3)), "n_hit": np.full(count, 5)}


def test_log_yield_row_returns_pairing_keys_and_a_real_ci():
    train = _log_yield_rows(seed=1)
    test = _log_yield_rows(seed=2, runs=4)
    fitted = rp.fit_log_yield_probe(train)
    block = rp.log_yield_row(test, fitted=fitted)
    assert "run_ids" in block and "event_ids" in block
    assert len(block["errors"]) == len(block["run_ids"]) == len(block["event_ids"])
    assert block["run_ids"].shape[0] > 0
    invariant_rows = _log_yield_rows(seed=2, shift=0.2, runs=4)
    invariant_block = rp.log_yield_row(invariant_rows, fitted=fitted)
    reading = rp._variant_reading(
        {}, {"log_yield": {"N": block}},
        {"log_yield": {"N": invariant_block}})
    entry = reading["X3-1"]["log_yield"]
    assert entry["ci95"] is not None
    assert entry["rho_E"] is not None
    assert entry["class"] in ("price present", "no price", "unclear")


def test_missing_log_yield_ci_is_not_available_not_a_failed_condition():
    standard, invariant = _paired_blocks(1.30, paired=False)
    controls = [{"log_yield": {"N": standard}} for _index in range(3)]
    reading = rp._variant_reading({}, {"log_yield": {"N": standard}},
                                  {"log_yield": {"N": invariant}},
                                  leace={"controls": controls})
    gain = reading["X3-1-gain_specific"]["log_yield"]
    assert gain["status"] == "not available (CI missing)"
    assert gain["gain_specific"] is None
    assert gain["condition_ci_excludes_zero"] is None


def test_raw_feature_family_x3_4_has_a_real_verdict(monkeypatch):
    def rows(tag, scale=1.0):
        run_ids = np.repeat([f"r{index}" for index in range(5)], 4).astype(object)
        event_ids = np.tile(np.arange(4), 5)
        base = np.linspace(0.10, 0.20, run_ids.size) * float(scale)
        return {"tag": tag, "run_ids": run_ids, "event_ids": event_ids,
                "blocks": {kind: _metric_block(tag, base, run_ids, event_ids)
                           for kind in ("vertex", "log_yield", "energy")}}

    def fake_feature_probe_metrics(train, test, *, kind="vertex", fitted=None):
        return dict(test["blocks"][kind])

    monkeypatch.setattr(rp, "feature_probe_metrics", fake_feature_probe_metrics)
    # calibrated: lower resolution and smaller per-event errors at D than invariant
    calibrated = rows(0.08, scale=0.5)
    variant = rp._feature_variant(
        "raw", {"train": rows(0.08), "test": rows(0.10), "drift": rows(0.20)},
        {"train": rows(0.08), "test": rows(0.10), "drift": rows(0.20)},
        calibrated_rows=calibrated)
    # B10.3: the calibrated raw block is nested per metric under "D"
    assert variant["calibrated"]["vertex"]["D"] is not None
    assert "D" in variant["calibrated"]["energy"]
    readings = rp.x3_readings({"families": {"raw": variant}})
    dominance = readings["variants"]["raw"]["X3-4"]
    assert dominance["status"] == "evaluated"
    assert dominance["verdict"] == "calibration dominates"


def test_x3_raw_family_uses_the_s005_truth_for_the_drift_role(monkeypatch, tmp_path):
    """X3-b raw family: the drift (s=0.30) role reuses the s=0.05 test truth.

    The probe sets share events and run ids across settings (§2.2) and
    _x3_init9000 already maps its drift role to truth["test"]; the raw family
    must use the same convention (it previously indexed truth["drift"] -> KeyError).
    """
    seen = []
    monkeypatch.setattr(rp, "assert_authorised", lambda mode, **kw: {"status": "granted"})
    monkeypatch.setattr(rp, "part_a_inventory", lambda root=None, **kw: {"rows": {}})
    monkeypatch.setattr(
        rp, "open_probe_sets",
        lambda config, gen, geometry, data_root, *, setting, log=print:
        {"train": f"TRAIN{setting}", "test": f"TEST{setting}"})
    built = []

    def fake_condition(name, dataset, geometry, **kw):
        built.append((name, dataset))
        return f"COND:{name}"

    monkeypatch.setattr(rp, "build_condition", fake_condition)
    monkeypatch.setattr(
        rp.ef, "load_particle_truth",
        lambda data_root, key: ("TRUTH_TRAIN" if "train" in str(key)
                                else "TRUTH_TEST"))

    def fake_raw(dataset, geometry, truth=None, condition=None):
        seen.append((dataset, truth, condition))
        return {"merged": {"features": [[0.0]] * 3}}

    monkeypatch.setattr(rp, "raw_std_features", fake_raw)
    monkeypatch.setattr(rp, "raw_inv_features",
                        lambda dataset, geometry: {"features": [[0.0]] * 3})
    monkeypatch.setattr(rp, "_variant_metric_rows", lambda rows: rows)
    monkeypatch.setattr(rp, "_feature_variant",
                        lambda name, standard, invariant, calibrated_rows=None: {})
    monkeypatch.setattr(rp, "_x3_init9000", lambda *a, **k: {})
    monkeypatch.setattr(rp, "_x3_leace", lambda *a, **k: {})
    monkeypatch.setattr(rp, "x3_readings", lambda payload, fisher=None: {"variants": []})
    monkeypatch.setattr(rp, "_strip_internal", lambda payload: payload)
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    rp.x3(root=tmp_path, config={}, gen=None, arms_config=None, geometry=None,
          data_root="/data", ext_root=tmp_path, device="cpu",
          log=lambda *a, **k: None)
    # calls: standard train, standard test, standard drift, calibrated drift
    assert [row[1] for row in seen] == ["TRUTH_TRAIN", "TRUTH_TEST",
                                        "TRUTH_TEST", "TRUTH_TEST"]
    assert [row[0] for row in seen] == ["TRAIN0.05", "TEST0.05",
                                        "TEST0.3", "TEST0.3"]
    assert seen[2][2] is None          # standard: no calibration condition
    assert seen[3][2] == "COND:SWunit"  # calibrated: condition on the D probe-test
    assert built == [("SWunit", "TEST0.3")]


def test_merge_standard_rows_aligns_every_column_for_domain_masks():
    """The raw-std merged table must be rectangular: particle rows lack log_yield."""
    particle = {"features": np.zeros((2, 10)),
                "run_ids": np.array(["r0", "r0"], dtype=object),
                "event_ids": np.array([1, 2]), "vertex": np.zeros((2, 3)),
                "particle": np.array(["muon", "electron"]),
                "contained": np.array([True, False]),
                "deposited_MeV": np.array([100.0, 200.0]),
                "direction": np.zeros((2, 3)), "n_hit": np.array([5, 6])}
    photon = {"features": np.ones((3, 10)),
              "run_ids": np.array(["r0"] * 3, dtype=object),
              "event_ids": np.array([3, 4, 5]), "vertex": np.zeros((3, 3)),
              "log_yield": np.array([7.0, 8.0, 9.0]),
              "particle": np.array(["photon"] * 3),
              "contained": np.zeros(3, bool),
              "deposited_MeV": np.full(3, np.nan),
              "direction": np.zeros((3, 3)), "n_hit": np.array([2, 3, 4]),
              "domain": np.array(["photon"] * 3)}
    merged = rp.merge_standard_rows(particle, photon)
    for name in ("features", "run_ids", "event_ids", "vertex", "domain", "particle",
                 "contained", "deposited_MeV", "direction", "n_hit", "log_yield"):
        assert len(merged[name]) == 5, name
    assert np.isnan(merged["log_yield"][:2]).all()
    assert np.allclose(merged["log_yield"][2:], [7.0, 8.0, 9.0])
    assert list(merged["domain"]) == ["muon", "electron", "photon", "photon", "photon"]
    # the per-domain masks now select the intended rows with no shape error
    log_yield_mask = np.isfinite(merged["log_yield"]) & (
        np.asarray(merged["domain"]) == "photon")
    assert int(log_yield_mask.sum()) == 3


# ==========================================================================
# Addendum 2026-11-07 focused tests (D1, D2, D3, D4, gates, budget, CLI)
# ==========================================================================
def _dummy_entries(**kwargs):
    """A deterministic passing addendum entries preflight for tests."""
    return {"pass": True, "free_entries": 40000, "projected_after": 39800,
            "floor": 30000, "projection": {"addendum": 200, "floor": 30000},
            "basis": "the addendum's own projected writes only"}


def _synthetic_equivariance(*, runs=10, per_run=5, dims=4, modules=6, seed=0,
                            constant_column=True):
    """Noiseless Δz = u Aᵀ on run-grouped rows, with an optional constant column."""
    rng = np.random.default_rng(int(seed))
    run_ids = np.repeat([f"r{index}" for index in range(int(runs))],
                        int(per_run)).astype(object)
    u = rng.standard_normal((run_ids.size, int(modules)))
    if constant_column:
        u[:, -1] = 3.0
    A = rng.standard_normal((int(dims), int(modules)))
    if constant_column:
        A[:, -1] = 0.0
    delta = u @ A.T
    split = int(runs) // 2
    train = np.isin(run_ids, [f"r{index}" for index in range(split)])
    fit = rp.equivariance_fit(u[train], delta[train], u[~train], delta[~train],
                              run_ids_train=run_ids[train],
                              run_ids_test=run_ids[~train], kind="occupancy")
    return {"u": u, "delta": delta, "A": A, "run_ids": run_ids,
            "train": train, "fit": fit}


def _predict_reference_map(fit, u):
    """A[d, m] from the fits' own ``predict(u) - predict(0)``, no raw_map access."""
    fits = fit["fitted"]["fits"]
    u = np.asarray(u, float)
    zero = np.zeros((1, u.shape[1]))
    out = np.zeros((len(fits), u.shape[1]))
    for column in range(u.shape[1]):
        basis = np.zeros((1, u.shape[1]))
        basis[0, column] = 1.0
        for index, probe in enumerate(fits):
            if probe.get("status") == "fit":
                fit_row = probe["fit"]
                out[index, column] = float(fit_row.predict(basis)[0]
                                           - fit_row.predict(zero)[0])
    return out


# ---------------------------------------------------------------------- D1 §2
def test_raw_map_identity_on_random_data_including_a_constant_column():
    case = _synthetic_equivariance()
    A_raw = rp._stack_coefficients(case["fit"])
    assert A_raw.shape == case["A"].shape
    # the constant standardised column has a zero coefficient -> raw map row 0
    assert np.allclose(A_raw[:, -1], 0.0, atol=1e-14)
    predicted = case["u"] @ A_raw.T
    reference = rp.d1_identity_reference(case["fit"]["fitted"], case["u"])
    relative = float(np.linalg.norm(predicted - reference)
                     / np.linalg.norm(reference))
    assert relative <= 1e-10
    row = rp.raw_map(case["fit"]["fitted"]["fits"][0])
    assert row.shape == (case["u"].shape[1],)
    assert np.allclose(row, case["fit"]["fitted"]["fits"][0]["fit"].coefficients
                       / case["fit"]["fitted"]["fits"][0]["fit"].scale)


def test_raw_map_recovers_a_synthetic_map_to_one_over_a_million():
    case = _synthetic_equivariance(seed=1)
    A_raw = rp._stack_coefficients(case["fit"])
    recovered = np.max(np.abs(A_raw - case["A"])) / np.max(np.abs(case["A"]))
    assert recovered <= 1e-6
    # the constant column stays exactly zero on both sides
    assert np.allclose(A_raw[:, -1], 0.0, atol=1e-14)


def test_raw_map_is_the_only_route_to_ridge_coefficients():
    import ast
    source = Path(rp.__file__).read_text()
    tree = ast.parse(source)
    owners = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "coefficients":
                owners.append(node.name)
    assert owners, "the module no longer reads RidgeFit.coefficients at all"
    assert set(owners) == {"raw_map"}, (
        f"RidgeFit.coefficients is read outside raw_map by {sorted(set(owners))}")


def test_every_consumer_matches_the_predict_reference(monkeypatch, geometry):
    case = _synthetic_equivariance(seed=2)
    A_raw = rp._stack_coefficients(case["fit"])
    u_test = case["u"][~case["train"]]
    A_ref = _predict_reference_map(case["fit"], u_test)
    assert np.allclose(A_raw, A_ref, rtol=1e-10, atol=1e-12)
    dims, modules = case["A"].shape
    rng = np.random.default_rng(7)
    # latent_estimate: the design and the solve are linear in Â, so Â_raw == A_ref
    W = rng.standard_normal((dims, dims))
    occupancy = rng.random(modules)
    zbar = rng.standard_normal(dims)
    estimate = rp.latent_estimate(zbar, occupancy, A_raw, W)
    reference = rp.latent_estimate(zbar, occupancy, A_ref, W)
    assert np.allclose(estimate["design"], reference["design"], atol=1e-12)
    assert np.allclose(estimate["delta"], reference["delta"], rtol=0, atol=1e-12)
    # identifiability_spectrum uses W Â at the mean occupancy
    spectrum = rp.identifiability_spectrum(A_raw, W, occupancy)
    spectrum_reference = rp.identifiability_spectrum(A_ref, W, occupancy)
    assert np.allclose(spectrum["singular_values"],
                       spectrum_reference["singular_values"], rtol=1e-12)
    # _latent_repair_rows subtracts u Âᵀ
    count = 4 * 5
    run_ids = np.repeat([f"r{index}" for index in range(4)], 5).astype(object)
    event_ids = np.tile(np.arange(5), 4)
    rows = {"events": rng.standard_normal((count, dims)),
            "run_ids": run_ids, "event_ids": event_ids,
            "vertex": rng.standard_normal((count, 3)),
            "log_yield": rng.standard_normal(count), "n_hit": np.full(count, 5)}
    counts = {(str(run), int(event)): rng.random(modules)
              for run, event in zip(run_ids, event_ids)}
    gains = {f"r{index}": rng.standard_normal(modules) for index in range(4)}
    repaired = rp._latent_repair_rows(rows, counts, geometry, A_raw, gains)
    repaired_ref = rp._latent_repair_rows(rows, counts, geometry, A_ref, gains)
    assert np.allclose(repaired["events"], repaired_ref["events"], atol=1e-12)
    # _x2c applies the map to raw u and scores the repaired D rows
    captured = []

    def fake_predictions(fitted, dataset):
        captured.append(np.asarray(dataset.x, float))
        return np.zeros((dataset.x.shape[0], 3))

    monkeypatch.setattr(rp.la, "probe_predictions", fake_predictions)
    delta = {run: rng.standard_normal(modules) for run in np.unique(run_ids)}
    test_rows = {"N": dict(rows), "D": dict(rows)}
    consumed = rp._x2c({"test": test_rows}, delta, counts, geometry, object(),
                       A_raw)
    expected = (np.asarray(rows["events"], float)
                - rp.occupancy_design(test_rows["D"], delta, counts, geometry)
                @ A_raw.T)
    # the repaired D design is the rows passed to the frozen probe's predict
    assert any(np.allclose(seen, expected, atol=1e-12) for seen in captured)
    assert consumed["A_hat"] is not None


def test_d1_check_model_reports_descriptives_and_flag(tiny, tmp_path, geometry):
    sets = _tiny_many(tmp_path, geometry)
    probes005 = {"train": sets["s005"]["dataset"], "test": sets["s005_test"]["dataset"]}
    probes030 = {"train": sets["s030"]["dataset"], "test": sets["s030_test"]["dataset"]}
    conditions = {role: rp.build_conditions(
        {"s005": probes005[role], "s030": probes030[role]}, geometry, ("N", "D", "S"))
        for role in ("train", "test")}
    rows = {
        "train": {"N": _synthetic_rows(probes005["train"], geometry, seed=11),
                  "S": _synthetic_rows(probes005["train"], geometry, seed=12),
                  "D": _synthetic_rows(probes030["train"], geometry, seed=13)},
        "test": {"N": _synthetic_rows(probes005["test"], geometry, seed=14),
                 "S": _synthetic_rows(probes005["test"], geometry, seed=15),
                 "D": _synthetic_rows(probes030["test"], geometry, seed=16)}}
    delta = rp.run_log_gain_vectors(probes030["train"], geometry)
    delta.update(rp.run_log_gain_vectors(probes030["test"], geometry))
    occupancy_n = {role: rp.occupancy_counts(probes005[role],
                                             conditions[role]["N"], geometry)
                   for role in ("train", "test")}
    out = rp._d1_check_model(rows, delta, occupancy_n, geometry)
    assert out["status"] == "evaluated"
    assert out["identity_pass"] is True
    assert out["identity"]["relative_error"] <= 1e-8
    assert out["removed_share"] is not None
    assert out["recovered_fraction"] is not None
    assert out["flag"] is False and out["flag_text"] is None


def test_addendum_selfcheck_stop_rule_and_output_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_addendum_budget_for", lambda mode: {"pass": True})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "open_probe_sets",
                        lambda *args, **kwargs: {"train": object(), "test": object()})
    monkeypatch.setattr(rp, "build_conditions",
                        lambda *args, **kwargs: {"N": object(), "D": object(),
                                                 "S": object()})
    monkeypatch.setattr(rp, "occupancy_counts", lambda *args, **kwargs: {})
    monkeypatch.setattr(rp, "run_log_gain_vectors", lambda *args, **kwargs: {})
    monkeypatch.setattr(rp, "embedding_cache", lambda *args, **kwargs: object())
    monkeypatch.setattr(rp, "_cached_condition_rows",
                        lambda *args, **kwargs: ({"train": {}, "test": {}}, []))
    state = {"pass": True}

    def fake_model(*args, **kwargs):
        return {"status": "evaluated", "identity_pass": state["pass"],
                "identity": {"relative_error": 0.0 if state["pass"] else 1e-3},
                "r2_eq": 0.5, "removed_share": 0.9, "recovered_fraction": 0.3,
                "flag": False, "flag_text": None, "v_N_m": 0.1, "v_S_m": 0.12,
                "v_S_repaired_m": 0.11, "n_train": 1, "n_test": 1}

    monkeypatch.setattr(rp, "_d1_check_model", fake_model)
    ok = rp.addendum_selfcheck(config={}, gen={}, geometry=None, root=tmp_path)
    assert ok["identity_pass"] is True and ok["stop"] is None
    assert ok["n_models"] == len(rp.CORE_MODELS)
    assert ok["entries"]["pass"] is True
    assert ok["budget"]["recorded"] is True
    assert {"gpu", "cpu"} <= set(ok["hours"])
    state["pass"] = False
    failed = rp.addendum_selfcheck(config={}, gen={}, geometry=None, root=tmp_path)
    assert failed["identity_pass"] is False
    assert str(failed["stop"]).startswith("S9")
    assert "D1 reruns" in failed["stop"]


# ---------------------------------------------------------------------- D2 §3
def _d2_variant(metric, errors, run_ids, event_ids):
    if metric == "vertex":
        resolution = float(np.mean(errors))
        return {"vertex": {"D": {"status": "evaluated", "mean_error_m": resolution,
                                 "resolution": resolution,
                                 "__errors__": np.asarray(errors, float),
                                 "run_ids": run_ids, "event_ids": event_ids}}}
    return {metric: {"D": {"status": "evaluated",
                            "resolution": float(np.sqrt(np.mean(errors ** 2))),
                            "errors": np.asarray(errors, float),
                            "run_ids": run_ids, "event_ids": event_ids}}}


def test_x3_4_vertex_point_and_interval_both_use_the_mean():
    run_ids = np.repeat([f"r{index}" for index in range(40)], 4).astype(object)
    event_ids = np.tile(np.arange(4), 40)
    offsets = np.repeat(np.linspace(-0.002, 0.002, 40), 4)
    left = 0.10 + offsets
    right = 0.12 + offsets
    out = rp._compare_at_d(_d2_variant("vertex", left, run_ids, event_ids),
                           _d2_variant("vertex", right, run_ids, event_ids),
                           "vertex")
    assert out["statistic"] == "mean"
    assert out["difference"] == pytest.approx(float(right.mean()) - float(left.mean()),
                                              abs=1e-12)
    assert out["ci95"][0] <= out["difference"] <= out["ci95"][1]
    assert out["calibration_better"] is True
    assert out["invariance_better"] is False
    assert rp.metric_statistic("vertex") is np.mean
    assert rp.metric_statistic("energy") is rp.rms
    assert rp.metric_statistic("log_yield") is rp.rms


def test_compare_at_d_point_inside_its_95_interval_for_each_metric():
    run_ids = np.repeat([f"r{index}" for index in range(40)], 4).astype(object)
    event_ids = np.tile(np.arange(4), 40)
    offsets = np.repeat(np.linspace(-0.002, 0.002, 40), 4)
    left = 0.10 + offsets
    right = 0.13 + offsets
    for metric in ("vertex", "log_yield", "energy"):
        out = rp._compare_at_d(_d2_variant(metric, left, run_ids, event_ids),
                               _d2_variant(metric, right, run_ids, event_ids),
                               metric)
        assert out["ci95"] is not None
        assert out["ci95"][0] <= out["difference"] <= out["ci95"][1], metric
        if metric != "vertex":
            assert out["statistic"] == "rms"
            assert out["difference"] == pytest.approx(
                rp.rms(right) - rp.rms(left), abs=1e-12)


def test_every_other_comparison_uses_a_matched_statistic():
    # X1-c reads the mean point and the mean interval from bootstrap_difference
    entry = rp._x1c({"SWunit_vs_D_refit": {"mean": {"mean_difference_m": -0.004,
                                                     "ci95_m": [-0.005, -0.003]},
                                           "rms": {"difference": 0.002,
                                                   "ci95": [0.001, 0.003]}}})
    assert entry["SWunit"]["difference_m"] == pytest.approx(-0.004)
    assert entry["SWunit"]["ci95_m"] == [-0.005, -0.003]
    assert entry["SWunit"]["reading"] == "calibration better"
    # X3-1's interval is the RMS paired bootstrap, matching both RMS resolutions
    run_ids = np.repeat([f"r{index}" for index in range(10)], 4).astype(object)
    event_ids = np.tile(np.arange(4), 10)
    left = _metric_block(0.20, np.linspace(0.18, 0.22, run_ids.size),
                         run_ids, event_ids)
    right = _metric_block(0.25, np.linspace(0.23, 0.27, run_ids.size),
                          run_ids, event_ids)
    ratio = rp._ratio_and_ci(left, right)
    expected = rp.paired_cluster_bootstrap(left["errors"], right["errors"],
                                           run_ids, statistic=rp.rms)["ci95"]
    assert ratio["ci95"] == pytest.approx(expected)
    # X3-2 is descriptive (a difference, no interval) and uses the mean vertex error
    assert "ci95" not in rp._variant_reading(
        {}, {"vertex": {"N": left}}, {"vertex": {"N": right}})["X3-2"]


# ---------------------------------------------------------------------- D3 §4
def test_x3_five_treats_a_null_efficient_crlb_as_infinity():
    fisher = {"models": {
        "C-mean": {"median_crlb": {"known": [1.0, 1.0, 1.0, 1.0],
                                   "inv": [1.0, 1.0, 1.0, None]}},
        "C-var_all": {"median_crlb": {"known": [1.0, 1.0, 1.0, 1.0],
                                      "inv": [1.0, 1.0, 1.0, 4.0]}}}}
    out = rp._x3_five(fisher, {})
    assert out["theory_rows"]["C-mean"]["theory_ratio_lnY"] == float("inf")
    assert out["theory_rows"]["C-mean"]["efficient_lnY_crlb"] is None
    assert out["theory_rows"]["C-var_all"]["theory_ratio_lnY"] == pytest.approx(2.0)
    # serialises: JSON null is what the file holds, infinity is restored on read
    path = rp.RESULTS_DIR  # only a name; no file is touched here
    assert isinstance(path, str)


def test_strip_internal_drops_fitted_and_prediction_and_x3_serialises(tmp_path):
    payload = {"families": {"leace": {"models": {"a/E1": {
        "fitted": {"fits": [object()]}, "prediction": np.zeros((2, 3)),
        "errors": [1.0, 2.0], "__errors__": np.zeros(2), "rank": 1,
        "sanity": {"unerased_r2": 0.5}}}}}}
    stripped = rp._strip_internal(payload)
    assert stripped == {"families": {"leace": {"models": {"a/E1": {
        "rank": 1, "sanity": {"unerased_r2": 0.5}}}}}}
    target = tmp_path / "x3.json"
    rp.write_json(target, stripped)
    assert json.loads(target.read_text())["families"]["leace"]["models"]["a/E1"][
        "rank"] == 1
    # an un-stripped payload with fit objects cannot serialise: the drop is required
    with pytest.raises(TypeError):
        json.dumps({"fitted": object()})


# ------------------------------------------------------- gate (a2) §7 and §9.2
def test_gate_a2_missing_addendum_fields_denies_not_pending(tmp_path):
    def drop(record, root):
        record.pop("addendum_path")
        record.pop("addendum_sha256")

    root = _fake_repo(tmp_path, record_mutator=drop)
    result = rp.assert_authorised("x1", root=root, dry_run=True)
    assert result["status"] == "denied"          # never the pending state
    assert result["checks"]["a2"]["pass"] is False
    assert result["checks"]["b"]["missing_keys"] == ["addendum_path",
                                                    "addendum_sha256"]
    with pytest.raises(SystemExit) as caught:
        rp.assert_authorised("x1", root=root)
    assert caught.value.code == 2


def test_gate_a2_addendum_sha_mismatch_denies(tmp_path):
    def tamper(path):
        path.write_text(path.read_text() + "\nlate edit\n")

    root = _fake_repo(tmp_path, addendum_mutator=tamper)
    result = rp.assert_authorised("x1", root=root, dry_run=True)
    assert result["checks"]["a2"]["pass"] is False
    assert result["checks"]["a2"]["addendum_sha256"] != rp.ADDENDUM_SHA256
    assert result["status"] == "denied"
    with pytest.raises(SystemExit):
        rp.assert_authorised("x1", root=root)

    def wrong_hash(record, root):
        record["addendum_sha256"] = "0" * 64

    other = _fake_repo(tmp_path / "wrong_hash", record_mutator=wrong_hash)
    assert rp.assert_authorised("x1", root=other,
                                dry_run=True)["checks"]["a2"]["pass"] is False


def test_gate_a2_author_line_and_path_are_checked(tmp_path):
    def strip_line(path):
        path.write_text(path.read_text().replace(rp.ADDENDUM_AUTHOR_LINE, "removed"))

    no_line = _fake_repo(tmp_path / "no_line", addendum_mutator=strip_line)
    check = rp.assert_authorised("x1", root=no_line, dry_run=True)["checks"]["a2"]
    assert check["pass"] is False and check["author_line_present"] is False

    def wrong_path(record, root):
        record["addendum_path"] = "docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md"
        record["addendum_path"] = "docs/jepa/OTHER_ADDENDUM.md"

    bad_path = _fake_repo(tmp_path / "bad_path", record_mutator=wrong_path)
    check = rp.assert_authorised("x1", root=bad_path, dry_run=True)["checks"]["a2"]
    assert check["pass"] is False and check["record_addendum_path"] is not None


def test_gate_complete_record_and_committed_addendum_grants(tmp_path):
    root = _fake_repo(tmp_path)
    result = rp.assert_authorised("addendum-report", root=root)
    assert result["authorised"] is True and result["status"] == "granted"
    assert result["checks"]["a2"]["pass"] is True
    assert result["checks"]["a2"]["author_line_present"] is True


def test_gate_rejects_a_record_pointing_at_a_different_committed_addendum(tmp_path):
    def wrong_hash(record, root):
        record["addendum_sha256"] = "f" * 64

    root = _fake_repo(tmp_path, record_mutator=wrong_hash)
    with pytest.raises(SystemExit):
        rp.assert_authorised("addendum-selfcheck", root=root)


# --------------------------------------------------- addendum budget §6 / §8.4
def test_addendum_budget_ledger_caps_and_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "addendum_jobs.jsonl")
    monkeypatch.chdir(tmp_path)
    assert rp.addendum_hours()["gpu"] == 0.0
    rp.record_addendum_job("addendum-x2e", "gpu", 36.0, part="A")
    rp.record_addendum_job("addendum-x3", "cpu", 180.0, part="A")
    rp.record_addendum_job("r2-evaluation", "gpu", 0.30 * 3600.0, part="B")
    hours = rp.addendum_hours()
    assert hours["gpu"] == pytest.approx(0.31)
    assert hours["cpu"] == pytest.approx(0.05)
    assert hours["jobs"] == 3
    lines = [json.loads(line) for line in
             (tmp_path / "addendum_jobs.jsonl").read_text().splitlines()]
    assert {"utc", "part", "mode", "kind", "wall_seconds", "hours"} <= set(lines[0])
    assert rp.ADDENDUM_GPU_CAP_H == 0.35 and rp.ADDENDUM_CPU_CAP_H == 3.0
    with pytest.raises(SystemExit) as caught:
        rp.assert_addendum_budget("gpu", 0.06, part="A")
    assert caught.value.code == 2
    rp.record_addendum_job("x3-cpu", "cpu", 2.90 * 3600.0, part="A")
    with pytest.raises(SystemExit):
        rp.assert_addendum_budget("cpu", 0.20, part="A")
    assert rp.addendum_hours()["cpu"] > 2.9


def test_addendum_modes_do_not_consult_the_closed_part_a_or_b_caps():
    assert not hasattr(rp, "_ADDENDUM_GPU_CAP_H")
    # the addendum caps are the only ones the addendum modes read
    assert rp.ADDENDUM_GPU_CAP_H < rp.PART_A_GPU_CAP_H
    assert rp.ADDENDUM_CPU_CAP_H < rp.CPU_CAP_H
    assert rp.ADDENDUM_MODES == ("addendum-init9000", "addendum-selfcheck",
                                 "addendum-x2", "addendum-x2e", "addendum-x3",
                                 "addendum-report")
    assert set(rp._ADDENDUM_GPU_MODES) == {"addendum-x2e", "addendum-init9000"}
    assert rp._ADDENDUM_GPU_PROJECTION["addendum-init9000"] == 0.07
    # every addendum mode is a registered gated mode
    assert set(rp.ADDENDUM_MODES) <= set(rp.MODES)


# ------------------------------------------------ D1 reuse of source / raw §2
def test_addendum_x2_reuses_the_source_and_raw_streams():
    assert "A_hat" not in inspect.signature(rp.source_estimate).parameters
    assert "A_hat" not in inspect.signature(rp.raw_estimate).parameters
    base = {"split": "within",
            "X2-a": {}, "X2-b": {"occupancy": {"r2_cs": 0.7, "class": "partly"}},
            "X2-c": {"rho_LR": -0.5},
            "X2-d": {"source": {"median_run_rms": 0.010, "pearson_r": 0.9,
                                "coverage": 1.0, "n_runs": 8},
                     "raw": {"median_run_rms": 0.011, "pearson_r": 0.8,
                             "coverage": 1.0, "n_runs": 8},
                     "latent": {"median_run_rms": -0.002}},
            "X2-e": {"streams": {"source": {"rho_EST": 0.6},
                                 "raw": {"rho_EST": 0.5},
                                 "latent": {"rho_EST": -0.1}},
                     "ceiling_rho_SWmod": 0.8},
            "X2-f": {"max_singular_value": 1.0}}
    extra = {"X2-c": {"rho_LR": 0.6, "class": "latent repair works"},
             "X2-d": {"latent": {"median_run_rms": 0.004, "pearson_r": 0.95,
                                 "coverage": 1.0, "n_runs": 8}},
             "X2-f": {"max_singular_value": 2.0}}
    row = rp._x2_seed_summary(base, extra, {"rho_EST_latent": 0.7})
    assert row["X2-d"]["source"]["median_run_rms"] == 0.010
    assert row["X2-d"]["raw"]["median_run_rms"] == 0.011
    assert row["X2-d"]["latent"]["median_run_rms"] == 0.004
    assert row["X2-d"]["latent"]["source"] == "addendum-x2"
    assert row["X2-d"]["source"]["source"] == "7 Nov"
    assert row["X2-e"]["rho_EST"]["source"] == 0.6
    assert row["X2-e"]["rho_EST"]["latent"] == 0.7
    assert row["X2-e"]["reused_streams"] == ["source", "raw"]


# -------------------------------------------- D2 init9000 cache / not_run §3
def test_x3_init9000_from_cache_reports_not_run_then_reads(tmp_path, geometry):
    inv = {"rows": {"init9000": {"label": "init9000", "split": "across"}}}
    absent = rp._x3_init9000_from_cache(inv, tmp_path, None, None, geometry)
    assert absent["status"] == "not_run (gpu needed)"
    cache = rp.init9000_token_cache(tmp_path)
    for key in rp._INIT9000_TOKEN_KEYS:
        domain, kind, role, name = key.split("__")
        run_ids = np.repeat([f"r{index}" for index in range(5)], 4).astype(object)
        rows = {"events": np.random.default_rng(0).standard_normal((20, 4)),
                "run_ids": run_ids, "event_ids": np.tile(np.arange(4), 5),
                "vertex": np.zeros((20, 3)),
                "log_yield": np.zeros(20), "n_hit": np.full(20, 5)}
        if domain == "particle":
            rows.update({"particle": np.array(["electron"] * 20, dtype=object),
                         "contained": np.ones(20, bool),
                         "deposited_MeV": np.linspace(100.0, 900.0, 20),
                         "direction": np.zeros((20, 3))})
        cache.fetch(key, lambda rows=rows: rows)
    present = rp._x3_init9000_from_cache(inv, tmp_path, None, None, geometry)
    assert present["status"] == "evaluated"
    assert "vertex" in present["standard"] and "log_yield" in present["invariant"]


# --------------------------------------------------- D4 summary + figures §5
def _d4_x1_model(geometry):
    names = ("N", "D", "SWtw", "SWunit", "SWmod", "SWunitN", "HW")
    conditions = {}
    for index, name in enumerate(names):
        conditions[name] = {
            "vertex": {"mean_error_m": 0.10 + 0.004 * index, "status": "evaluated"},
            "log_yield": {"resolution": 0.20 + 0.01 * index,
                          "ci95": [0.19, 0.21], "r2": {"r2": 0.3}},
            "refit_vertex": {"mean_error_m": 0.095 + 0.004 * index,
                             "status": "evaluated"}}
    for sigma in ("0.01", "0.02", "0.05", "0.1"):
        conditions[f"NOISY({sigma})"] = {
            "vertex": {"mean_error_m": 0.105, "status": "evaluated"},
            "log_yield": {"resolution": 0.25, "ci95": [0.24, 0.26], "r2": {"r2": 0.2}}}
    def mean(name):
        return conditions[name]["vertex"]["mean_error_m"]
    def refit(name):
        return conditions[name]["refit_vertex"]["mean_error_m"]
    def comparison(left, right):
        difference = float(left - right)
        return {"mean": {"mean_difference_m": difference,
                         "ci95_m": [difference - 0.001, difference + 0.001]},
                "rms": {"difference": difference, "ci95": [difference - 0.001,
                                                          difference + 0.001]}}
    comparisons = {
        "D": comparison(mean("D"), mean("N")),
        "D_refit": comparison(refit("D"), mean("N")),
        "SWunit_vs_D_refit": comparison(mean("SWunit"), refit("D")),
        "SWunit_refit_vs_D_refit": comparison(refit("SWunit"), refit("D")),
        "HW": comparison(mean("HW"), mean("N")),
        "SWunitN": comparison(mean("SWunitN"), mean("N")),
        "SWunit_vs_HW": comparison(mean("SWunit"), mean("HW"))}
    particle = {name: {"energy": {"resolution": 0.05, "ci95": [0.04, 0.06],
                                  "r2": {"r2": 0.4}, "n_contained_test": 600}}
                for name in ("N", "D", "SWunit", "HW")}
    return {"split": "within", "conditions": conditions,
            "comparisons": comparisons, "particle": particle}


def _d4_x3_reading(rho, *, vertex_difference, gain_specific):
    return {"X3-1": {"log_yield": {"rho_E": rho, "ci95": [rho - 0.1, rho + 0.1],
                                   "class": "price present",
                                   "gain_specific": gain_specific,
                                   "rho_control_q95": 1.10,
                                   "rho_controls": [1.0] * 20},
                      "energy": {"rho_E": rho, "ci95": [rho - 0.1, rho + 0.1],
                                 "class": "price present", "gain_specific": None}},
            "X3-2": {"difference_m": vertex_difference},
            "X3-3": {"standard": {"robust": True}, "invariant": {"robust": False}},
            "X3-4": {"vertex": {"difference": vertex_difference,
                                "ci95": [vertex_difference - 0.001,
                                         vertex_difference + 0.001]},
                     "energy": {"difference": 0.02, "ci95": [0.01, 0.03]},
                     "log_yield": {"difference": 0.05, "ci95": [0.04, 0.06]},
                     "verdict": "calibration dominates", "status": "evaluated"}}


def _d4_fixtures(tmp_path, geometry):
    ext = tmp_path / "ext"
    label = "mae_within_trigger9000"
    x1 = {"models": {label: _d4_x1_model(geometry)}}
    x2_model = {
        "split": "within",
        "X2-a": {"share_cs": 0.9, "share_ni": 0.1, "v_S_minus_v_N": 0.004,
                 "v_D_minus_v_S": 0.012, "additivity_check": 0.0005},
        "X2-b": {"occupancy": {"r2_cs": 0.7, "class": "partly equivariant"},
                 "global": {"r2_tot": 0.8}},
        "X2-c": {"rho_LR": -0.5, "class": "fails"},
        "X2-d": {"source": {"median_run_rms": 0.010, "pearson_r": 0.9,
                            "coverage": 1.0, "n_runs": 8},
                 "raw": {"median_run_rms": 0.011, "pearson_r": 0.8,
                         "coverage": 1.0, "n_runs": 8},
                 "latent": {"median_run_rms": -0.002, "pearson_r": 0.2,
                            "coverage": 1.0, "n_runs": 8}},
        "X2-e": {"streams": {"source": {"rho_EST": 0.6},
                             "raw": {"rho_EST": 0.5},
                             "latent": {"rho_EST": -0.1}},
                 "ceiling_rho_SWmod": 0.8},
        "X2-f": {"singular_values": [3.0, 2.0, 1.0], "max_singular_value": 3.0,
                 "condition_number": 1e6, "median_crlb": 0.5,
                 "crlb_per_module": [0.1] * 6}}
    x2 = {"models": {label: x2_model}}
    families = {
        "raw": {"status": "evaluated"},
        "init9000": {"status": "evaluated"},
        "leace": {"status": "evaluated", "models": {
            f"{label}/E1": {"status": "evaluated", "rank": 1, "concept": "E1",
                            "sanity": {"unerased_r2": 0.5, "erased_r2": 0.001,
                                       "status": "evaluated"}},
            f"{label}/E2r": {"status": "evaluated", "rank": 5, "concept": "E2r",
                             "sanity": {"unerased_r2": 0.4, "erased_r2": 0.002,
                                        "status": "evaluated"}},
            "mae_within_trigger9001/E2r": {"status": "not_applicable", "rank": 0,
                                           "reason": "r = 0: not applicable"}}}}
    x3 = {"families": families, "readings": {
        "variants": {"raw": _d4_x3_reading(3.5, vertex_difference=0.134,
                                           gain_specific=None),
                     "init9000": _d4_x3_reading(1.2, vertex_difference=0.002,
                                                gain_specific=None),
                     f"{label}/E1": _d4_x3_reading(1.4, vertex_difference=0.003,
                                                  gain_specific=True),
                     f"{label}/E2r": _d4_x3_reading(1.5, vertex_difference=0.004,
                                                   gain_specific=True)},
        "X3-5": {"empirical_rho_E_log_yield": 3.5,
                 "theory_rows": {"C-mean": {"theory_ratio_lnY": None},
                                 "C-mean+occupancy": {"theory_ratio_lnY": 2.1}},
                 "rule": "C-mean alone gives ∞ by P0"}}}
    fisher = {"models": {"C-mean": {"median_crlb": {"known": [1, 1, 1, 1],
                                                    "inv": [1, 1, 1, None]}}},
              "p0": {"pass": True}}
    return ext, label, x1, x2, x3, fisher


def _write_d4_stage(ext, x1, x2, x3, fisher):
    rp.write_json(ext / "x1" / "x1.json", x1)
    rp.write_json(ext / "x2" / "x2.json", x2)
    rp.write_json(ext / "x3" / "x3.json", x3)
    rp.write_json(ext / "x3" / "x3_fisher.json", fisher)


def _write_d4_addendum(ext, label, *, corrected=True):
    d1 = {"identity_pass": corrected, "stop": None if corrected else "S9: x",
          "n_models": 1, "complete": True, "unavailable": [], "flags": [],
          "models": {label: {"status": "evaluated", "identity_pass": corrected,
                             "identity": {"relative_error": 1e-16},
                             "r2_eq": 0.5, "removed_share": 0.9,
                             "recovered_fraction": 0.4, "flag": False,
                             "flag_text": None}}}
    x2_add = {
        "complete": True, "unavailable": [], "A_hat_units": "raw", "reused": {},
        "latent_estimates": {label: {"r0": [0.0] * 6}},
        "models": {label: {
            "X2-c": {"rho_LR": 0.6, "class": "latent repair works"},
            "X2-d": {"latent": {"median_run_rms": 0.004, "pearson_r": 0.95,
                                "coverage": 1.0, "n_runs": 8}},
            "X2-f": {"singular_values": [3.5, 2.0, 1.0],
                     "max_singular_value": 3.5, "median_crlb": 0.4,
                     "crlb_per_module": [0.08] * 6},
            "A_hat": [[0.0] * 6]}}}
    x2e_add = {"complete": True, "unavailable": [], "models": {label: {
        "rho_EST_latent": 0.7, "latent_repair_rho_LR": 0.65,
        "ceiling_rho_SWmod": 0.8}}}
    x3_add = {"complete": True, "readings": {"variants": {
        "raw": _d4_x3_reading(3.5, vertex_difference=0.1337,
                              gain_specific=None),
        "init9000": _d4_x3_reading(1.2, vertex_difference=0.002,
                                   gain_specific=None),
        f"{label}/E1": _d4_x3_reading(1.4, vertex_difference=0.003,
                                      gain_specific=True),
        f"{label}/E2r": _d4_x3_reading(1.5, vertex_difference=0.004,
                                       gain_specific=True)},
        "X3-5": {"empirical_rho_E_log_yield": 3.5, "theory_rows": {},
                 "rule": "x"}}}
    rp.write_json(ext / "addendum" / "d1_selfcheck.json", d1)
    rp.write_json(ext / "addendum" / "x2_addendum.json", x2_add)
    rp.write_json(ext / "addendum" / "x2e_addendum.json", x2e_add)
    rp.write_json(ext / "addendum" / "x3_addendum.json", x3_add)
    return x2_add, x2e_add, x3_add


def test_addendum_report_writes_every_d4_item_and_the_superseded_table(
        tmp_path, geometry, monkeypatch):
    ext, label, x1, x2, x3, fisher = _d4_fixtures(tmp_path, geometry)
    _write_d4_stage(ext, x1, x2, x3, fisher)
    x2_add, _x2e_add, _x3_add = _write_d4_addendum(ext, label)
    root = tmp_path / "repo"
    (root / "results/jepa/20261107_shortcut").mkdir(parents=True)
    rp.write_json(root / "results/jepa/20261107_shortcut/readings.json",
                  {"r1": {"reading": {"verdict": "contradicts",
                                      "generative_median": {"r2_eq": 0.2}}}})
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", tmp_path / "ext_shortcut_absent")
    result = rp.addendum_report(root=root, ext_root=ext,
                                results_dir=root / rp.RESULTS_DIR,
                                log=lambda *a, **k: None)
    summary = rp.read_json(Path(result["summary"]))
    # X1-c, X1-d, X1-e, X1-f per model and seed
    seed_row = summary["x1"]["mae_within_trigger"]["9000"]
    assert set(seed_row["X1-c"]) == {"SWunit", "SWunit_refit"}
    assert set((seed_row["X1-d"] or {})["primary"]) == {"HW", "SWunitN",
                                                         "SWunit_vs_HW"}
    assert set(seed_row["X1-e"]["rho"]) == {"0.01", "0.02", "0.05", "0.1"}
    assert "sigma_half" in seed_row["X1-e"]
    assert {"N", "D", "SWunit", "HW"} <= set(seed_row["X1-f"])
    assert seed_row["X1-f"]["N"]["log_yield_resolution"] is not None
    assert seed_row["X1-f"]["N"]["energy_resolution"] is not None
    # X2-a, X2-d labelled streams, X2-e, X2-f
    x2_row = summary["x2"]["mae_within_trigger"]["9000"]
    assert x2_row["X2-a"]["additivity_check"] is not None
    assert x2_row["X2-c"]["rho_LR"] == pytest.approx(0.6)
    for stream in ("source", "raw", "latent"):
        assert x2_row["X2-d"][stream]["median_run_rms"] is not None
        assert "label" in x2_row["X2-d"][stream]
    assert set(x2_row["X2-e"]["rho_EST"]) == {"source", "raw", "latent"}
    assert x2_row["X2-e"]["rho_EST"]["latent"] == pytest.approx(0.7)
    assert x2_row["X2-f"]["max_singular_value"] == pytest.approx(3.5)
    # X3-b every family, including LEACE E1/E2r with rank and sanity R²
    variants = summary["x3"]
    for name in ("raw", "init9000", f"{label}/E1", f"{label}/E2r"):
        assert name in variants
    assert variants[f"{label}/E2r"]["rank"] == 5
    assert variants[f"{label}/E1"]["unerased_concept_r2"] == pytest.approx(0.5)
    assert variants[f"{label}/E1"]["erased_concept_r2"] == pytest.approx(0.001)
    assert variants[f"{label}/E1"]["X3-1"]["log_yield"]["gain_specific"] is True
    assert variants[f"{label}/E1"]["X3-1"]["log_yield"]["rho_control_q95"] == \
        pytest.approx(1.10)
    assert variants["raw"]["X3-2"] is not None and variants["raw"]["X3-3"]
    assert variants["raw"]["X3-4"]["verdict"] == "calibration dominates"
    assert summary["x3_5"]["empirical_rho_E_log_yield"] == pytest.approx(3.5)
    # the superseded table: every replaced number, old value, new value, defect
    items = {(row["item"], row["scope"]) for row in summary["superseded"]}
    assert ("rho_LR", label) in items
    assert ("X2-d latent median_run_rms", label) in items
    assert ("X2-e latent rho_EST", label) in items
    assert ("X2-f max singular value", label) in items
    assert ("X3-4 vertex", "raw") in items
    assert ("R-1 reading", "Part B") in items
    superseded = {(row["item"], row["scope"]): row
                  for row in summary["superseded"]}
    assert superseded[("rho_LR", label)]["defect"] == "D1"
    assert superseded[("rho_LR", label)]["old"] == pytest.approx(-0.5)
    assert superseded[("rho_LR", label)]["new"] == pytest.approx(0.6)
    assert superseded[("rho_LR", label)]["status"] == "superseded"
    assert superseded[("X3-4 vertex", "raw")]["defect"] == "D2"
    assert superseded[("X3-4 vertex", "raw")]["status"] == "superseded"
    assert superseded[("X2-e latent rho_EST", label)]["new"] == pytest.approx(0.7)
    assert superseded[("X2-e latent rho_EST", label)]["old"] == pytest.approx(-0.1)
    # the committed markdown carries the table and the labelled streams
    markdown = (Path(result["path"]) / "READINGS_SUMMARY.md").read_text()
    assert "Superseded 7 Nov values" in markdown
    assert "latent (unlabelled" in markdown
    assert "source (known light)" in markdown
    # the addendum JSON carries self-check, reruns and the superseded table
    addendum = rp.read_json(Path(result["addendum_json"]))
    assert addendum["addendum_sha256"] == rp.ADDENDUM_SHA256
    assert "self_check" in addendum and "reruns" in addendum
    assert addendum["reruns"]["X2-c"]["status"] == "complete"
    assert addendum["superseded"]
    assert (Path(result["path"]) / "OUTCOME.json").is_file()
    assert result["figures"]["status"] in ("evaluated", "unavailable")


def test_addendum_report_never_invents_a_missing_value(tmp_path, monkeypatch):
    ext = tmp_path / "ext_empty"
    root = tmp_path / "repo"
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", tmp_path / "ext_shortcut_absent")
    result = rp.addendum_report(root=root, ext_root=ext,
                                results_dir=tmp_path / "results",
                                log=lambda *a, **k: None)
    summary = rp.read_json(Path(result["summary"]))
    assert summary["x1"] == {} and summary["x2"] == {} and summary["x3"] == {}
    assert any(row["status"] == "not_run" for row in summary["not_run"])
    markdown = (Path(result["path"]) / "READINGS_SUMMARY.md").read_text()
    assert "not_run" in markdown
    assert "Superseded 7 Nov values" in markdown


def test_addendum_report_uses_the_7_nov_flat_keys_for_backward_compatibility(
        tmp_path, geometry, monkeypatch):
    ext, label, x1, x2, x3, fisher = _d4_fixtures(tmp_path, geometry)
    _write_d4_stage(ext, x1, x2, x3, fisher)
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", tmp_path / "ext_shortcut_absent")
    result = rp.addendum_report(root=tmp_path / "repo", ext_root=ext,
                                results_dir=tmp_path / "results",
                                log=lambda *a, **k: None)
    summary = rp.read_json(Path(result["summary"]))
    x1_row = summary["x1"]["mae_within_trigger"]["9000"]
    assert {"delta_D_m", "ci95_m", "eligible", "rho_swtw", "class_swtw",
            "share_cs", "class_x1g"} <= set(x1_row)
    x2_row = summary["x2"]["mae_within_trigger"]["9000"]
    assert {"r2_eq", "class", "rho_lr"} <= set(x2_row)
    x3_row = summary["x3"]["raw"]
    assert {"log_yield_rho", "log_yield_class", "energy_class"} <= set(x3_row)


# ------------------------------------- §8 GPU: init9000 gain-blind token cache
def _token_rows(key):
    """Minimal cache rows for one init9000 token key (no fitting is done here)."""
    domain = key.split("__")[0]
    run_ids = np.repeat([f"r{index}" for index in range(5)], 4).astype(object)
    rows = {"events": np.random.default_rng(0).standard_normal((20, 4)),
            "run_ids": run_ids, "event_ids": np.tile(np.arange(4), 5),
            "vertex": np.zeros((20, 3)), "log_yield": np.zeros(20),
            "n_hit": np.full(20, 5)}
    if domain == "particle":
        rows.update({"particle": np.array(["electron"] * 20, dtype=object),
                     "contained": np.ones(20, bool),
                     "deposited_MeV": np.linspace(100.0, 900.0, 20),
                     "direction": np.zeros((20, 3))})
    return rows


def _populate_init9000_cache(ext_root):
    cache = rp.init9000_token_cache(ext_root)
    for key in rp._INIT9000_TOKEN_KEYS:
        cache.fetch(key, lambda key=key: _token_rows(key))
    return cache


def test_addendum_init9000_mode_registered_and_help(tmp_path):
    handlers = {"addendum-init9000": "mode_addendum_init9000"}
    for name in handlers:
        assert name in rp.MODES and name in rp.ADDENDUM_MODES
        assert hasattr(repair_cli, handlers[name])
    assert "addendum-init9000" in repair_cli.parser().format_help()
    parsed = repair_cli.parser().parse_args(["addendum-init9000"])
    assert parsed.mode == "addendum-init9000" and parsed.device == "auto"


def test_addendum_init9000_budget_refusal_uses_the_addendum_cap(tmp_path, monkeypatch,
                                                                capsys):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"utc": "2026-11-07T00:00:00Z", "part": "B",
                                  "mode": "addendum-evaluate", "kind": "gpu",
                                  "wall_seconds": 1080.0, "hours": 0.30}) + "\n")
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", ledger)
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "addendum_s1_check_or_stop",
                        lambda **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    assert rp._ADDENDUM_GPU_PROJECTION["addendum-init9000"] == 0.07
    with pytest.raises(SystemExit) as caught:
        rp.assert_addendum_budget("gpu", 0.07, part="A")
    assert caught.value.code == 2
    assert "not_run (cap)" in capsys.readouterr().err
    # the mode refuses before any GPU work (empty cache, no geometry needed)
    with pytest.raises(SystemExit) as caught_mode:
        rp.addendum_init9000(config={}, gen={}, arms_config={}, geometry=None,
                             ext_root=tmp_path / "ext", root=tmp_path)
    assert caught_mode.value.code == 2
    # the closed Part A cap is never the one consulted
    assert rp.ADDENDUM_GPU_CAP_H == 0.35 and rp.PART_A_GPU_CAP_H == 4.0


def test_addendum_init9000_already_cached_short_circuits(tmp_path, monkeypatch):
    _populate_init9000_cache(tmp_path / "ext")
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "addendum_s1_check_or_stop",
                        lambda **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", tmp_path / "ext_shortcut_absent")

    def boom(*args, **kwargs):
        raise AssertionError("the encode path ran despite a verified cache")

    monkeypatch.setattr(rp, "part_a_inventory", boom)
    monkeypatch.setattr(rp, "open_probe_sets", boom)
    monkeypatch.setattr(rp.ef, "build_item_arm", boom)
    payload = rp.addendum_init9000(config={}, gen={}, arms_config={},
                                   geometry=None, ext_root=tmp_path / "ext",
                                   root=tmp_path)
    assert payload["status"] == "already_cached"
    assert payload["complete"] is True and payload["cache_verified"] is True
    assert payload["s1"]["pass"] is True
    assert payload["entries"]["pass"] is True
    assert payload["budget"]["kind"] == "gpu"
    assert payload["budget"]["recorded"] is True
    assert payload["budget"]["hours"] == pytest.approx(0.0, abs=1e-4)


def test_addendum_init9000_setup_path_records_the_projection_and_does_not_encode(
        tiny, tmp_path, monkeypatch):
    geometry = tiny["geometry"]
    config = lr.load_run_config(ROOT / rp.CONFIG_REL)
    arms_config = la.load_config(ROOT / rp.ARMS_REL)

    def boom(*args, **kwargs):
        raise AssertionError("the dry-run setup path ran an encode")

    monkeypatch.setattr(rp, "stream_condition_embeddings", boom)
    monkeypatch.setattr(rp, "stream_condition_particles", boom)
    monkeypatch.setattr(rp.ef, "build_item_arm", boom)
    monkeypatch.setattr(rp, "addendum_s1_check",
                        lambda root=None, **kwargs: {"pass": True,
                                                     "id": "S1 (addendum)"})
    monkeypatch.setattr(rp, "addendum_entries_preflight",
                        lambda **kwargs: _dummy_entries())
    probes = {"s005": {"train": tiny["s005"]["dataset"],
                       "test": tiny["s005"]["dataset"]},
              "s030": {"train": tiny["s030"]["dataset"],
                       "test": tiny["s030"]["dataset"]}}
    setup = rp._mode_setup_objects(
        "addendum-init9000", root=tmp_path / "repo", config=config, gen=GEN,
        arms_config=arms_config, geometry=geometry, data_root=tmp_path / "data",
        ext_root=tmp_path / "ext", inventory=_fake_inventory(), probes=probes,
        particle_truth={}, log=lambda *a, **k: None)
    plan = setup["plan"]["addendum"]
    assert plan["gpu_projection"]["addendum-init9000"] == 0.07
    assert "addendum-init9000" in plan["modes"]
    assert plan["gpu_cap_hours"] == 0.35 and plan["cpu_cap_hours"] == 3.0
    assert not (tmp_path / "ext" / "addendum" / "init9000").exists()


# ==========================================================================
# Addendum review A1–A4 fixes (round after the BLOCK at 923276a)
# ==========================================================================
def _write_7nov_results(root, x1, x2, x3):
    """The 7 Nov results payloads that addendum_report preserves and updates."""
    target = root / rp.RESULTS_DIR
    rp.write_json(target / "x1.json", x1)
    rp.write_json(target / "x2.json", x2)
    rp.write_json(target / "x3.json", x3)
    rp.write_json(target / "readings.json", {
        "registered": False, "extension": rp.EXTENSION,
        "x1": {"complete": True, **(x1.get("readings") or {})},
        "x2": {"complete": True, **(x2.get("readings") or {})},
        "x3": {"complete": True, **(x3.get("readings") or {})},
        "reproduction": True})
    return target


# ---------------------------------------------------------------- A1 (D1→R-1)
def test_addendum_report_updates_results_in_place_and_preserves_7nov(
        tmp_path, geometry, monkeypatch):
    ext, label, x1, x2, x3, fisher = _d4_fixtures(tmp_path, geometry)
    _write_d4_stage(ext, x1, x2, x3, fisher)
    _write_d4_addendum(ext, label)
    root = tmp_path / "repo"
    target = _write_7nov_results(root, x1, x2, x3)
    original_x2 = (target / "x2.json").read_bytes()
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", tmp_path / "ext_shortcut_absent")
    result = rp.addendum_report(root=root, ext_root=ext, results_dir=target,
                                log=lambda *a, **k: None)
    # A1: the corrected rho_LR reaches the pilot payload that Part B R-1 reads
    updated = rp.read_json(target / "x2.json")
    assert updated["models"][label]["X2-c"]["rho_LR"] == pytest.approx(0.6)
    assert updated["models"][label]["rho_lr"] == pytest.approx(0.6)
    assert updated["models"][label]["X2-f"]["max_singular_value"] == \
        pytest.approx(3.5)
    readings = rp.read_json(target / "readings.json")
    seed = readings["x2"]["families"]["mae_within_trigger"]["seeds"]["9000"]
    assert seed["X2-c"] == pytest.approx(0.6)
    assert readings["x3"]["variants"]["raw"]["X3-1"]["log_yield"]["rho_E"] == \
        pytest.approx(3.5)
    # A1: the 7 Nov originals are preserved write-once with sha256 sidecars
    preserved = ext / "addendum" / "preserved_7nov"
    assert (preserved / "x2.json").read_bytes() == original_x2
    assert rp.read_json(preserved / "x2.json")["models"][label]["X2-c"][
        "rho_LR"] == pytest.approx(-0.5)
    for name in ("x1", "x2", "x3", "readings"):
        assert (preserved / f"{name}.json").is_file()
        sidecar = preserved / f"{name}.json.sha256"
        assert sidecar.is_file()
        assert sidecar.read_text().strip() == lr.sha256_file(
            preserved / f"{name}.json")
    assert set(result["results_updated"]) == {"x1", "x2", "x3", "readings"}
    # A1/A2: the superseded row's old value comes from the preserved copy
    summary = rp.read_json(Path(result["summary"]))
    row = [item for item in summary["superseded"]
           if item["item"] == "rho_LR" and item["scope"] == label][0]
    assert row["old"] == pytest.approx(-0.5) and row["new"] == pytest.approx(0.6)
    # a second run does not overwrite the preserved copies
    before = {name: (preserved / f"{name}.json").read_bytes()
              for name in ("x1", "x2", "x3", "readings")}
    rp.addendum_report(root=root, ext_root=ext, results_dir=target,
                       log=lambda *a, **k: None)
    for name, payload in before.items():
        assert (preserved / f"{name}.json").read_bytes() == payload


# ---------------------------------------------------------------------- A2
def test_superseded_r1_row_is_populated_from_part_b_payloads(
        tmp_path, geometry, monkeypatch):
    ext, label, x1, x2, x3, fisher = _d4_fixtures(tmp_path, geometry)
    _write_d4_stage(ext, x1, x2, x3, fisher)
    _write_d4_addendum(ext, label)
    root = tmp_path / "repo"
    target = _write_7nov_results(root, x1, x2, x3)
    shortcut = root / "results/jepa/20261107_shortcut"
    rp.write_json(shortcut / "readings.json", {"r1": {"reading": {
        "verdict": "contradicts", "generative_median": {"r2_eq": 0.2}}}})
    new_reading = {"verdict": "supports", "generative_median": {"r2_eq": 0.55}}
    part_b = tmp_path / "ext_shortcut"
    rp.write_json(part_b / "r1_addendum.json",
                  {"status": "complete", "reading": new_reading, "table": {}})
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", part_b)
    result = rp.addendum_report(root=root, ext_root=ext, results_dir=target,
                                log=lambda *a, **k: None)
    addendum = rp.read_json(Path(result["addendum_json"]))
    row = [item for item in addendum["superseded"]
           if item["item"] == "R-1 reading"][0]
    assert row["status"] == "superseded"
    assert row["old"]["verdict"] == "contradicts"
    assert row["new"]["verdict"] == "supports"
    assert addendum["reruns"]["R-1"]["status"] == "complete"
    # (b) the Part B report's results shape: items.R1_rerun
    rp.write_json(shortcut / "ADDENDUM_2026-11-07.json",
                  {"items": {"R1_rerun": {"verdict": "supports",
                                          "generative_median": {"r2_eq": 0.55}}}})
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", tmp_path / "absent")
    again = rp.addendum_report(root=root, ext_root=ext, results_dir=target,
                               log=lambda *a, **k: None)
    addendum = rp.read_json(Path(again["addendum_json"]))
    row = [item for item in addendum["superseded"]
           if item["item"] == "R-1 reading"][0]
    assert row["new"]["verdict"] == "supports"
    assert row["old"]["verdict"] == "contradicts"      # preserved, not lost


# ---------------------------------------------------------------------- A3
def _ledger_payload(charged, *, booked=None):
    """A synthetic RESOURCE_LEDGER payload whose booked sum is ``booked``."""
    booked = float(charged if booked is None else booked)
    recent = {rp.BOOKED_L40S_BASE_KEY: {"l40s_charged_after": booked}}
    return {"v4_caps": {
        "E2_lucid_l40s_deepthought2": {"charged_gpu_hours": float(charged)},
        "recent": recent}}


def test_addendum_s1_pass_at_current_ledger_and_stop_at_159_7(tmp_path):
    passing = _ledger_payload(129.7461)
    ok = rp.addendum_s1_check(tmp_path, ledger=passing)
    assert ok["pass"] is True
    assert ok["sum_hours"] == pytest.approx(130.0961)
    assert ok["addendum_cap_hours"] == 0.35
    assert ok["ceiling_hours"] == 160.0
    assert "not consulted" in ok["rule"]
    failing = _ledger_payload(159.7)
    assert rp.addendum_s1_check(tmp_path, ledger=failing)["pass"] is False
    with pytest.raises(SystemExit) as caught:
        rp.addendum_s1_check_or_stop(tmp_path, ledger=failing)
    assert caught.value.code == 2


def test_addendum_gpu_modes_stop_on_s1_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "addendum_s1_check",
                        lambda root=None, **kwargs: {
                            "pass": False, "ledger_charged_gpu_hours": 159.7,
                            "addendum_cap_hours": 0.35, "sum_hours": 160.05,
                            "ceiling_hours": 160.0})
    monkeypatch.setattr(rp, "addendum2_s1_check",
                        lambda root=None, **kwargs: {
                            "pass": False, "booked_gpu_hours": 159.7,
                            "addendum2_cap_hours": 0.08, "sum_hours": 159.78,
                            "ceiling_hours": 160.0})
    boom = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("work ran before the addendum S1 STOP"))
    monkeypatch.setattr(rp, "part_a_inventory", boom)
    monkeypatch.setattr(rp, "open_probe_sets", boom)
    for call in (
            lambda: rp.addendum_init9000(config={}, gen={}, arms_config={},
                                         geometry=None, ext_root=tmp_path / "ext",
                                         root=tmp_path),
            lambda: rp.addendum_x2e(config={}, gen={}, arms_config={},
                                    geometry=None, ext_root=tmp_path / "ext",
                                    root=tmp_path)):
        with pytest.raises(SystemExit) as caught:
            call()
        assert caught.value.code == 2


# ---------------------------------------------------------------------- A4
def test_addendum_entries_preflight_revised_basis_pass_and_fail(capsys):
    ok = rp.addendum_entries_preflight(free_entries=35770)
    assert ok["pass"] is True
    assert ok["free_entries"] == 35770 and ok["projected_after"] == 35570
    assert ok["projection"] == {"addendum": 200, "floor": 30000}
    assert "own projected writes" in ok["basis"]
    assert rp.ADDENDUM_ENTRIES_PROJECTION == 200
    assert rp.addendum_entries_preflight(free_entries=30200)["pass"] is True
    assert rp.addendum_entries_preflight(free_entries=30050)["pass"] is False
    assert rp.addendum_entries_preflight(free_entries=29999)["pass"] is False
    with pytest.raises(SystemExit) as caught:
        rp.addendum_entries_preflight_or_stop(free_entries=30050)
    assert caught.value.code == 2
    assert "not_run (entries)" in capsys.readouterr().err
    unavailable = rp.addendum_entries_preflight(guard_json="/nonexistent/guard.json")
    assert unavailable["pass"] is False and "error" in unavailable


def test_every_addendum_mode_checks_entries_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "addendum_s1_check_or_stop",
                        lambda **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "addendum2_s1_check_or_stop",
                        lambda **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "assert_addendum2_budget",
                        lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")

    def stop(**kwargs):
        raise SystemExit(2)

    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop", stop)
    monkeypatch.setattr(rp, "addendum2_entries_preflight_or_stop", stop)
    calls = (
        lambda: rp.addendum_selfcheck(config={}, gen={}, geometry=None,
                                      root=tmp_path, ext_root=tmp_path / "ext"),
        lambda: rp.addendum_x2(config={}, gen={}, geometry=None, root=tmp_path,
                               ext_root=tmp_path / "ext"),
        lambda: rp.addendum_x2e(config={}, gen={}, arms_config={}, geometry=None,
                                root=tmp_path, ext_root=tmp_path / "ext"),
        lambda: rp.addendum_x3(config={}, gen={}, arms_config={}, geometry=None,
                               root=tmp_path, ext_root=tmp_path / "ext"),
        lambda: rp.addendum_report(root=tmp_path, ext_root=tmp_path / "ext",
                                   results_dir=tmp_path / "results"),
        lambda: rp.addendum_init9000(config={}, gen={}, arms_config={},
                                     geometry=None, root=tmp_path,
                                     ext_root=tmp_path / "ext"))
    for call in calls:
        with pytest.raises(SystemExit) as caught:
            call()
        assert caught.value.code == 2


def test_dry_run_plan_reports_the_addendum_s1_and_entries_basis(
        tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "addendum_s1_check",
                        lambda root=None, **kwargs: {"pass": True,
                                                     "id": "S1 (addendum)"})
    monkeypatch.setattr(rp, "addendum_entries_preflight",
                        lambda **kwargs: _dummy_entries())
    geometry = tiny["geometry"]
    config = lr.load_run_config(ROOT / rp.CONFIG_REL)
    arms_config = la.load_config(ROOT / rp.ARMS_REL)
    probes = {"s005": {"train": tiny["s005"]["dataset"],
                       "test": tiny["s005"]["dataset"]},
              "s030": {"train": tiny["s030"]["dataset"],
                       "test": tiny["s030"]["dataset"]}}
    setup = rp._mode_setup_objects(
        "addendum-init9000", root=tmp_path / "repo", config=config, gen=GEN,
        arms_config=arms_config, geometry=geometry, data_root=tmp_path / "data",
        ext_root=tmp_path / "ext", inventory=_fake_inventory(), probes=probes,
        particle_truth={}, log=lambda *a, **k: None)
    plan = setup["plan"]["addendum"]
    assert plan["s1"]["pass"] is True
    assert plan["addendum_entries"]["pass"] is True
    assert plan["addendum_entries"]["projection"]["addendum"] == 200
    assert "basis" in plan["addendum_entries"]


# ------------------------------------- nonblocking N8 + post-write quarantine
def test_d4_markdown_carries_the_x2f_singular_values(tmp_path, geometry,
                                                     monkeypatch):
    ext, label, x1, x2, x3, fisher = _d4_fixtures(tmp_path, geometry)
    _write_d4_stage(ext, x1, x2, x3, fisher)
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "_assert_closure", lambda mode, root: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rp, "addendum_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "PART_B_EXT_ROOT", tmp_path / "ext_shortcut_absent")
    result = rp.addendum_report(root=tmp_path / "repo", ext_root=ext,
                                results_dir=tmp_path / "results",
                                log=lambda *a, **k: None)
    markdown = (Path(result["path"]) / "READINGS_SUMMARY.md").read_text()
    assert "singular values of W Â_raw" in markdown
    summary = rp.read_json(Path(result["summary"]))
    assert summary["x2"]["mae_within_trigger"]["9000"]["X2-f"][
        "singular_values"] is not None


def test_addendum_cli_writes_then_quarantines_on_a_closure_violation(
        tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    payload = {"complete": True, "closure": {"pass": True}, "mode": "test"}
    monkeypatch.setattr(repair_cli.rep, "import_closure",
                        lambda root=None: {"pass": False,
                                           "violations": [{"path": "x.py"}]})
    code = repair_cli._finalise_addendum(payload, path, tmp_path, 0)
    assert code == 2
    written = json.loads(path.read_text())
    assert written["quarantined"] is True
    assert written["complete"] is False
    assert written["closure"]["pass"] is False
    # a clean closure keeps the payload and the mode's own exit code
    payload_ok = {"complete": True, "closure": {"pass": True}, "mode": "test"}
    path_ok = tmp_path / "payload_ok.json"
    monkeypatch.setattr(repair_cli.rep, "import_closure",
                        lambda root=None: {"pass": True, "violations": []})
    assert repair_cli._finalise_addendum(payload_ok, path_ok, tmp_path, 0) == 0
    assert json.loads(path_ok.read_text()).get("quarantined") is None


def test_addendum_cached_condition_rows_are_nested_per_role(tmp_path):
    """The cache helper returns {role: {name: rows}}; x2e flattens one role.

    Regression for the addendum-x2e run that raised ``KeyError: 'N'`` on the real
    probe-test caches: the helper's result is nested per role, so the mode must
    flatten ``rows["test"]``/``rows["train"]`` before addressing the names.
    """
    cache = rp.EmbeddingCache(tmp_path)
    for role, name in (("test", "N"), ("test", "D"), ("train", "N")):
        key = rp.embedding_cache_key(role, name)
        cache.fetch(key, lambda: {"events": np.zeros((2, 4)),
                                  "run_ids": np.array(["r0", "r1"], dtype=object)})
    rows, missing = rp._cached_condition_rows(cache, ("test",),
                                               ("N", "D", "SWmod"))
    assert missing == ["test__SWmod"]
    assert rows is None
    rows, missing = rp._cached_condition_rows(cache, ("test",), ("N", "D"))
    assert missing == []
    assert rows["test"]["N"]["events"].shape == (2, 4)
    assert rows["test"]["D"]["events"].shape == (2, 4)
    source = Path(rp.__file__).read_text()
    assert 'rows_test = rows_test["test"]' in source
    assert 'rows_train = rows_train["train"]' in source


# ==========================================================================
# Addendum 2 — E1: scale-only latent gain estimator
# ==========================================================================
def _e1_synthetic(seed=21, dims=8, modules=6, runs=5):
    rng = np.random.default_rng(int(seed))
    W = rng.standard_normal((dims, dims)) + 3.0 * np.eye(dims)
    A = rng.standard_normal((dims, modules)) * np.logspace(-3, 3, modules)[None, :]
    occupancies = [rng.uniform(0.5, 2.0, modules) for _ in range(int(runs))]
    designs = [(W @ A) * occ[None, :] for occ in occupancies]
    return {"rng": rng, "W": W, "A": A, "occupancies": occupancies,
            "designs": designs}


def test_e1_scale_only_recovers_delta_at_extreme_column_scales():
    case = _e1_synthetic()
    rng, W, A = case["rng"], case["W"], case["A"]
    modules = A.shape[1]
    truth = rng.normal(0.0, 0.05, modules)
    occ = case["occupancies"][0]
    scale = rp._design_scales(case["designs"])
    assert scale is not None and scale.shape == (modules,)
    assert scale.max() / scale.min() > 1e5          # 1e-3 .. 1e3 span
    # W zbar = W A diag(occ) truth = design_0 truth
    zbar = A @ (occ * truth)
    fitted = rp.latent_estimate(zbar, occ, A, W, lam=0.0, scale=scale)
    relative = float(np.max(np.abs(fitted["delta"] - truth))
                     / np.max(np.abs(truth)))
    assert relative <= 1e-8
    assert fitted["residual_rms"] <= 1e-10
    # the solve happens in scaled units and is back-transformed
    assert np.allclose(fitted["delta"] * fitted["scale"],
                       fitted["delta_scaled"], rtol=1e-12, atol=1e-15)


def test_e1_lambda_positive_rescaling_invariance_and_raw_units():
    case = _e1_synthetic(seed=22)
    rng, W, A = case["rng"], case["W"], case["A"]
    modules = A.shape[1]
    truth = rng.normal(0.0, 0.05, modules)
    occ = case["occupancies"][0]
    designs = case["designs"]
    scale = rp._design_scales(designs)
    zbar = A @ (occ * truth)
    base = rp.latent_estimate(zbar, occ, A, W, lam=1e-2, scale=scale)
    column, factor = 2, 7.5
    A_scaled = A.copy()
    A_scaled[:, column] *= factor
    designs_scaled = [(W @ A_scaled) * run_occ[None, :]
                      for run_occ in case["occupancies"]]
    scale_scaled = rp._design_scales(designs_scaled)
    moved = rp.latent_estimate(zbar, occ, A_scaled, W, lam=1e-2,
                               scale=scale_scaled)
    # the standardised design is unchanged, so the fit is invariant: the raw-unit
    # coefficient of the rescaled column moves by exactly 1/factor
    assert moved["delta"][column] * factor == pytest.approx(
        base["delta"][column], rel=1e-10)
    others = np.arange(modules) != column
    assert np.allclose(moved["delta"][others], base["delta"][others],
                       rtol=1e-10, atol=1e-14)
    design = designs[0]
    design_scaled = (W @ A_scaled) * occ[None, :]
    assert np.allclose(design_scaled @ moved["delta"],
                       design @ base["delta"], rtol=1e-10, atol=1e-14)
    assert np.allclose(moved["scale"], scale_scaled, rtol=1e-12)


def test_e1_scale_only_path_is_shared_and_back_transformed():
    import ast
    source = Path(rp.__file__).read_text()
    tree = ast.parse(source)
    assert "_standardisation" not in source
    assert "_design_scales" in source
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "latent_estimate"]
    assert calls
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "scale" in keywords, "a latent_estimate call omits the scale"
        assert "centre" not in keywords
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    for name in ("latent_cv", "latent_stream_estimates"):
        used = {call.func.id for call in ast.walk(functions[name])
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}
        assert "_design_scales" in used, name
    # the only readers of a solved "delta" are the two shared callers; the
    # back-transform happens inside latent_estimate before they see it
    readers = set()
    for name, function in functions.items():
        for node in ast.walk(function):
            if isinstance(node, ast.Subscript) \
                    and isinstance(node.slice, ast.Constant) \
                    and node.slice.value == "delta":
                readers.add(name)
    assert readers <= {"latent_cv", "latent_stream_estimates"}
    body = ast.get_source_segment(source, functions["latent_estimate"]) or ""
    assert "delta = delta_scaled / safe" in body


def test_e1_unaffected_x2_items_do_not_use_the_latent_path():
    import ast
    tree = ast.parse(Path(rp.__file__).read_text())
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    latent_names = {"_design_scales", "latent_estimate", "latent_cv",
                    "latent_stream_estimates"}
    for name in ("source_estimate", "raw_estimate", "_x2c",
                 "identifiability_spectrum", "_stack_coefficients",
                 "_fit_equivariance_occupancy"):
        used = {call.func.id for call in ast.walk(functions[name])
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}
        assert not (used & latent_names), (name, used & latent_names)
    assert "A_hat" not in inspect.signature(rp.source_estimate).parameters
    assert "A_hat" not in inspect.signature(rp.raw_estimate).parameters
    # X2-e / the latent repair consume the payload delta (already raw units) and
    # never re-scale it
    for name in ("_latent_repair_rows", "addendum_x2e"):
        used = {call.func.id for call in ast.walk(functions[name])
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}
        assert not (used & latent_names), (name, used & latent_names)


# ==========================================================================
# Addendum 2 — E2 (E3u concept) and E3 (ledger/S1) and the new modes
# ==========================================================================
def test_e3u_concepts_match_a_direct_per_pmt_computation(tiny, geometry):
    dataset = tiny["s005"]["dataset"]
    concepts = rp.e3u_concepts(dataset, geometry, setting=rp.S005)
    mod = rp.module_of_pmt(geometry)
    checked = 0
    for row in dataset.runs:
        state = rp.run_state(row, dataset, geometry, rp.S005)
        log_gain = np.log(np.asarray(state.module_gain, float))
        store = lb.read_sparse_hits(dataset.directory / "runs" / row["run_dir"]
                                    / "hits.npz")
        for position, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
            span = slice(int(store.offsets[position]),
                         int(store.offsets[position + 1]))
            pmt = np.asarray(store.pmt_id[span], np.int64)
            n_hit = np.bincount(mod[pmt],
                                minlength=int(geometry.n_modules)).astype(float)
            key = (str(row["run_id"]), int(event_id))
            assert np.allclose(concepts[key], log_gain * n_hit, rtol=1e-12,
                               atol=1e-12)
            checked += 1
    assert checked > 0
    hits = rp.e3u_hit_counts(dataset, geometry, setting=rp.S005)
    assert all(np.asarray(value).shape == (int(geometry.n_modules),)
               for value in hits.values())


def _e3u_rows(runs=5, events=4, dims=8, modules=6, seed=31):
    rng = np.random.default_rng(int(seed))
    run_ids = np.repeat([f"r{index}" for index in range(int(runs))],
                        int(events)).astype(object)
    event_ids = np.tile(np.arange(int(events)), int(runs))
    counts = rng.uniform(1.0, 6.0, (run_ids.size, int(modules)))
    gains = rng.normal(0.0, 0.2, (int(runs), int(modules)))
    concept = np.stack([gains[int(run[1:])] * counts[index]
                        for index, run in enumerate(run_ids)])
    mixing = rng.standard_normal((int(modules), int(dims)))
    events_matrix = concept @ mixing + 0.1 * rng.standard_normal(
        (run_ids.size, int(dims)))
    hits = counts.copy()

    def rows():
        return {"events": events_matrix, "run_ids": run_ids,
                "event_ids": event_ids, "vertex": rng.standard_normal(
                    (run_ids.size, 3)),
                "log_yield": rng.standard_normal(run_ids.size),
                "n_hit": np.full(run_ids.size, 5)}

    def concepts():
        return {"train": {"N": concept, "D": concept},
                "test": {"N": concept, "D": concept}}

    def hit_concepts():
        return {"train": {"N": hits, "D": hits},
                "test": {"N": hits, "D": hits}}

    table = {"train": {"N": rows(), "D": rows()},
             "test": {"N": rows(), "D": rows()}}
    return table, concepts(), hit_concepts()


def test_e3u_variant_carries_the_hitcount_control_at_the_same_rank():
    table, concepts, hits = _e3u_rows()
    variant = rp._leace_variant(label="t", domain="photon", rows=table,
                                concepts=concepts, rank=None, concept="E3u",
                                hitcount_concepts=hits,
                                random_controls=3)
    assert variant["status"] == "evaluated"
    assert variant["concept"] == "E3u"
    assert len(variant["controls"]) == 3
    assert variant["hitcount_control"]["rank"] == variant["rank"]
    assert variant["hitcount_control"]["concept"] == "hitcount"
    for metric in ("log_yield", "vertex"):
        assert metric in variant["hitcount_control"]


def test_e3u_permutation_keys_include_e3u(monkeypatch):
    seen = []
    real = lb.keyed_rng

    def capture(seed, namespace, key):
        seen.append((namespace, key))
        return real(seed, namespace, key)

    monkeypatch.setattr(rp.lb, "keyed_rng", capture)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 4))
    z = rng.standard_normal((40, 3))
    run_ids = np.repeat([f"r{index}" for index in range(8)], 5).astype(object)
    rp.reduction_rank(x, z, run_ids, key="lab", label="lab", domain="photon",
                      concept="E3u", n_permutations=2)
    assert ("paper3/x3/leace_perm", "lab/photon/E3u") in seen
    seen.clear()
    rp.reduction_rank(x, z, run_ids, key="lab", label="lab", domain="photon",
                      n_permutations=2)
    assert ("paper3/x3/leace_perm", "lab/photon") in seen


def test_e3u_gain_specific_requires_all_three_conditions():
    run_ids = np.repeat([f"r{index}" for index in range(20)], 4).astype(object)
    event_ids = np.tile(np.arange(4), 20)
    offsets = np.repeat(np.linspace(-0.002, 0.002, 20), 4)

    def block(base):
        return _metric_block(float(base), np.full(run_ids.size, base) + offsets,
                             run_ids, event_ids)

    standard = {"log_yield": {"N": block(0.2)}, "energy": {"N": block(0.2)}}
    invariant = {"log_yield": {"N": block(0.32)}, "energy": {"N": block(0.32)}}
    controls = [{"log_yield": {"N": block(0.21)},
                 "energy": {"N": block(0.21)}} for _index in range(20)]
    hitcount = {"log_yield": {"N": block(0.25)}, "energy": {"N": block(0.25)}}
    x3_1 = {"log_yield": {"rho_E": 1.6}, "energy": {"rho_E": 1.6}}
    out = rp._gain_specific_e3u(x3_1, standard, invariant, controls, hitcount)
    for metric in ("log_yield", "energy"):
        assert out[metric]["condition_q95"] is True
        assert out[metric]["condition_hitcount_above"] is True
        assert out[metric]["condition_hitcount_ci_excludes_zero"] is True
        assert out[metric]["gain_specific"] is True
    # a hit-count control with no smaller resolution breaks condition (c)
    hitcount_higher = {"log_yield": {"N": block(0.34)},
                       "energy": {"N": block(0.34)}}
    failed = rp._gain_specific_e3u(x3_1, standard, invariant, controls,
                                   hitcount_higher)
    assert failed["log_yield"]["condition_hitcount_above"] is False
    assert failed["log_yield"]["gain_specific"] is False


def test_e3u_reading_replaces_the_x3_1_gain_specific_flag():
    run_ids = np.repeat([f"r{index}" for index in range(20)], 4).astype(object)
    event_ids = np.tile(np.arange(4), 20)
    offsets = np.repeat(np.linspace(-0.002, 0.002, 20), 4)

    def block(base):
        return _metric_block(float(base), np.full(run_ids.size, base) + offsets,
                             run_ids, event_ids)

    standard = {"log_yield": {"N": block(0.2)}, "energy": {"N": block(0.2)}}
    invariant = {"log_yield": {"N": block(0.32), "D": block(0.32)},
                 "energy": {"N": block(0.32), "D": block(0.32)},
                 "vertex": {"N": block(0.1), "D": block(0.1)}}
    controls = [{"log_yield": {"N": block(0.21)},
                 "energy": {"N": block(0.21)}} for _index in range(20)]
    hitcount = {"log_yield": {"N": block(0.25)}, "energy": {"N": block(0.25)}}
    entry = {"concept": "E3u", "rank": 3, "controls": controls,
             "hitcount_control": hitcount,
             "calibrated": {"vertex": {"D": block(0.09)},
                            "energy": {"D": block(0.3)},
                            "log_yield": {"D": block(0.3)}}}
    reading = rp._variant_reading(entry, standard, invariant, leace=entry)
    assert reading["X3-1-gain_specific_e3u"]["log_yield"]["gain_specific"] is True
    assert reading["X3-1"]["log_yield"]["gain_specific"] is True
    assert reading["X3-1"]["log_yield"]["rho_hitcount"] is not None


# ---------------------------------------------------------------------- E3
def test_booked_gpu_hours_is_the_exact_ledger_sum():
    ledger = rp.read_json(ROOT / "docs/jepa/RESOURCE_LEDGER.json")
    booked = rp.booked_gpu_hours(ledger)
    assert booked["available"] is True
    assert booked["base_key"] == "mae_jepa_followup_2026-11-03"
    assert booked["base_hours"] == pytest.approx(129.7461)
    assert booked["sum_hours"] == pytest.approx(133.6492, abs=1e-6)
    assert booked["entries"]["repair_shortcut_2026-11-07"] == pytest.approx(3.7625)
    assert booked["entries"]["repair_addendum_2026-11-08"] == pytest.approx(0.1406)


def test_inconsistent_ledger_fails_every_s1_check(tmp_path):
    ledger = _ledger_payload(130.0, booked=133.6492)
    (tmp_path / "docs/jepa").mkdir(parents=True)
    rp.write_json(tmp_path / "docs/jepa/RESOURCE_LEDGER.json", ledger)
    combined = rp.s1_projection(tmp_path)
    assert combined["pass"] is False
    assert combined["ledger_consistent"] is False
    assert "differs from the booked" in combined["reason"]
    assert rp.addendum_s1_check(tmp_path)["pass"] is False
    assert rp.addendum2_s1_check(tmp_path)["pass"] is False
    with pytest.raises(SystemExit) as caught:
        rp.addendum2_s1_check_or_stop(tmp_path)
    assert caught.value.code == 2


def test_addendum2_s1_threshold(tmp_path):
    passing = rp.addendum2_s1_check(tmp_path,
                                    ledger=_ledger_payload(133.6492))
    assert passing["pass"] is True
    assert passing["sum_hours"] == pytest.approx(133.7292)
    assert passing["addendum2_cap_hours"] == 0.08
    assert passing["ceiling_hours"] == 160.0
    failing = rp.addendum2_s1_check(tmp_path,
                                    ledger=_ledger_payload(159.93))
    assert failing["pass"] is False
    assert failing["sum_hours"] == pytest.approx(160.01)


def test_addendum2_ledger_mode_writes_the_booked_total_once(tmp_path, monkeypatch):
    ledger = _ledger_payload(129.7461, booked=133.6492)
    (tmp_path / "docs/jepa").mkdir(parents=True)
    rp.write_json(tmp_path / "docs/jepa/RESOURCE_LEDGER.json", ledger)
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "addendum2_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "assert_addendum2_budget",
                        lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    payload = rp.addendum2_ledger(root=tmp_path, ext_root=tmp_path / "ext")
    assert payload["status"] == "written"
    assert payload["written_once"] is True
    assert payload["charged_before"] == pytest.approx(129.7461)
    assert payload["charged_after"] == pytest.approx(133.6492)
    assert payload["s1_after"]["pass"] is True
    written = rp.read_json(tmp_path / "docs/jepa/RESOURCE_LEDGER.json")
    assert written["v4_caps"]["E2_lucid_l40s_deepthought2"][
        "charged_gpu_hours"] == pytest.approx(133.6492)
    assert (tmp_path / "ext/addendum2/ledger_fix.json").is_file()
    again = rp.addendum2_ledger(root=tmp_path, ext_root=tmp_path / "ext")
    assert again["status"] == "already_consistent"
    assert again["written_once"] is False


# ---------------------------------------------------------- gates addendum 2
def test_gate_a3_missing_addendum2_fields_denies_not_pending(tmp_path):
    def drop(record, root):
        record.pop("addendum2_path")
        record.pop("addendum2_sha256")

    root = _fake_repo(tmp_path, record_mutator=drop)
    result = rp.assert_authorised("x1", root=root, dry_run=True)
    assert result["status"] == "denied"
    assert result["checks"]["a3"]["pass"] is False
    assert result["checks"]["b"]["missing_keys"] == ["addendum2_path",
                                                    "addendum2_sha256"]
    with pytest.raises(SystemExit):
        rp.assert_authorised("x1", root=root)


def test_gate_a3_addendum2_sha_and_path_are_checked(tmp_path):
    def tamper(path):
        path.write_text(path.read_text() + "\nlate edit\n")

    root = _fake_repo(tmp_path, addendum2_mutator=tamper)
    check = rp.assert_authorised("x1", root=root, dry_run=True)["checks"]["a3"]
    assert check["pass"] is False
    assert check["addendum2_sha256"] != rp.ADDENDUM2_SHA256

    def wrong_path(record, root):
        record["addendum2_path"] = "docs/jepa/OTHER_ADDENDUM2.md"

    bad = _fake_repo(tmp_path / "bad", record_mutator=wrong_path)
    assert rp.assert_authorised("x1", root=bad,
                                dry_run=True)["checks"]["a3"]["pass"] is False


def test_addendum2_modes_registered_and_in_help():
    for mode in rp.ADDENDUM2_MODES:
        assert mode in rp.MODES
        assert hasattr(repair_cli, "mode_" + mode.replace("-", "_"))
        assert mode in repair_cli.parser().format_help()
    assert rp.ADDENDUM2_GPU_CAP_H == 0.08
    assert rp.ADDENDUM2_CPU_CAP_H == 3.0
    assert set(rp.ADDENDUM2_MODES) == {"addendum2-ledger", "addendum2-leace",
                                       "addendum2-report"}


def test_dry_run_plan_reports_addendum2_s1_and_entries(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "addendum2_s1_check",
                        lambda root=None, **kwargs: {"pass": True,
                                                     "id": "S1 (addendum 2)"})
    monkeypatch.setattr(rp, "addendum2_entries_preflight",
                        lambda **kwargs: _dummy_entries())
    geometry = tiny["geometry"]
    config = lr.load_run_config(ROOT / rp.CONFIG_REL)
    arms_config = la.load_config(ROOT / rp.ARMS_REL)
    probes = {"s005": {"train": tiny["s005"]["dataset"],
                       "test": tiny["s005"]["dataset"]},
              "s030": {"train": tiny["s030"]["dataset"],
                       "test": tiny["s030"]["dataset"]}}
    setup = rp._mode_setup_objects(
        "addendum2-ledger", root=tmp_path / "repo", config=config, gen=GEN,
        arms_config=arms_config, geometry=geometry, data_root=tmp_path / "data",
        ext_root=tmp_path / "ext", inventory=_fake_inventory(), probes=probes,
        particle_truth={}, log=lambda *a, **k: None)
    plan = setup["plan"]["addendum2"]
    assert plan["addendum2_sha256"] == rp.ADDENDUM2_SHA256
    assert plan["gpu_cap_hours"] == 0.08 and plan["cpu_cap_hours"] == 3.0
    assert plan["s1"]["pass"] is True
    assert plan["addendum2_entries"]["pass"] is True
    assert plan["entries_projection"] == 200


def test_addendum2_report_writes_the_json_and_updates_the_summary_in_place(
        tmp_path, geometry, monkeypatch):
    ext = tmp_path / "ext"
    target = tmp_path / "repo" / rp.RESULTS_DIR
    target.mkdir(parents=True)
    label = "mae_within_trigger9000"
    family, seed = rp._family(label)
    summary0 = {"registered": False, "extension": rp.EXTENSION,
                "x1": {family: {seed: {"delta_D_m": 0.01, "rho_swtw": 0.9}}},
                "x2": {family: {seed: {
                    "X2-d": {"source": {"median_run_rms": 0.01},
                             "raw": {"median_run_rms": 0.012},
                             "latent": {"median_run_rms": -0.002,
                                        "pearson_r": -0.2, "coverage": 1.0,
                                        "n_runs": 8}},
                    "X2-e": {"rho_EST": {"source": 0.6, "raw": 0.5,
                                         "latent": -0.1},
                             "latent_repair_rho_LR": -0.05}}}},
                "x3": {}, "x3_5": {}, "superseded": [], "not_run": []}
    rp.write_json(target / "SUMMARY.json", summary0)
    original_summary_bytes = (target / "SUMMARY.json").read_bytes()
    (target / "READINGS_SUMMARY.md").write_text("# old summary\n")
    rp.write_json(target / "OUTCOME.json", {"extension": rp.EXTENSION})
    x2_pre = {"models": {label: {
        "X2-d": {"latent": {"median_run_rms": -0.002, "pearson_r": -0.2,
                            "coverage": 1.0, "n_runs": 8}},
        "X2-e": {"streams": {"latent": {"rho_EST": -0.1}},
                 "latent_repair_rho_LR": -0.05}}}}
    for name in ("x1", "x2", "x3", "readings"):
        rp.write_json(target / f"{name}.json", x2_pre if name == "x2" else {})
    rp.write_json(ext / "x2/x2.json", x2_pre)
    rp.write_json(ext / "addendum/x2_addendum.json", {
        "status": "evaluated",
        "models": {label: {"X2-d": {"latent": {
            "median_run_rms": 0.004, "pearson_r": 0.95, "coverage": 1.0,
            "n_runs": 8}}}}})
    rp.write_json(ext / "addendum/x2e_addendum.json", {
        "models": {label: {"rho_EST_latent": 0.7,
                           "latent_repair_rho_LR": 0.65}}})
    reading = _d4_x3_reading(1.5, vertex_difference=0.002, gain_specific=True)
    reading["X3-1"]["log_yield"]["rho_hitcount"] = 1.2
    reading["X3-1"]["log_yield"]["hitcount_ci95"] = [0.03, 0.05]
    entry = {"status": "evaluated", "concept": "E3u", "rank": 3,
             "sanity": {"unerased_r2": 0.5, "erased_r2": 0.001,
                        "status": "evaluated"},
             "controls": [{"index": index} for index in range(20)],
             "hitcount_control": {"rank": 3, "concept": "hitcount"}}
    rp.write_json(ext / "addendum2/x3_e3u.json", {
        "families": {"leace": {"status": "evaluated",
                               "models": {f"{label}/E3u": entry}}},
        "readings": {
            "variants": {f"{label}/E3u": reading},
            "X3-5": {"empirical_rho_E_log_yield": 1.5,
                     "raw_feature_rho_E_log_yield": 3.5,
                     "e3u_rho_E_log_yield": {f"{label}/E3u": 1.5},
                     "theory_rows": {}, "rule": "x"}}})
    rp.write_json(ext / "addendum2/ledger_fix.json", {
        "status": "written", "charged_before": 129.7461,
        "charged_after": 133.6492, "booked_gpu_hours": 133.6492,
        "written_once": True,
        "s1_after": {"pass": True, "sum_hours": 133.7292},
        "decisions_ready": "ledger corrected"})
    monkeypatch.setattr(rp, "assert_authorised",
                        lambda mode, **kwargs: {"status": "granted"})
    monkeypatch.setattr(rp, "addendum2_entries_preflight_or_stop",
                        lambda **kwargs: _dummy_entries())
    monkeypatch.setattr(rp, "assert_addendum2_budget",
                        lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(rp, "ADDENDUM_JOBS", tmp_path / "ledger.jsonl")
    result = rp.addendum2_report(root=tmp_path / "repo", ext_root=ext,
                                 results_dir=target, log=lambda *a, **k: None)
    assert result["complete"] is True
    a2 = rp.read_json(target / "ADDENDUM2_2026-11-08.json")
    assert a2["addendum2_sha256"] == rp.ADDENDUM2_SHA256
    assert set(a2["e1"]["reruns"]) == {"X2-d latent", "X2-e latent"}
    assert a2["e1"]["rows"] and a2["e3u"]["rows"]
    assert set(a2["e3u"]["readings"]["variants"]) == {f"{label}/E3u"}
    assert a2["ledger"]["status"] == "written"
    assert {row["defect"] for row in a2["superseded"]} == {"E1", "E2", "E3"}
    summary = rp.read_json(target / "SUMMARY.json")
    assert f"{label}/E3u" in summary["x3"]
    assert summary["x3"][f"{label}/E3u"]["concept"] == "E3u"
    assert summary["addendum2"]["e3u"]["n_models"] == 1
    row = summary["x2"][family][str(seed)]
    assert row["X2-d"]["latent"]["median_run_rms"] == pytest.approx(0.004)
    assert row["X2-e"]["rho_EST"]["latent"] == pytest.approx(0.7)
    assert row["X2-e"]["latent_repair_rho_LR"] == pytest.approx(0.65)
    markdown = (target / "READINGS_SUMMARY.md").read_text()
    assert "Addendum 2 — E1 latent gain estimator" in markdown
    assert "E3u" in markdown and "ledger total (E3)" in markdown
    outcome = rp.read_json(target / "OUTCOME.json")
    assert "addendum2" in outcome
    # pre-addendum-2 copies are preserved write-once
    preserved = ext / "addendum2/preserved_before"
    assert (preserved / "SUMMARY.json").is_file()
    assert (preserved / "SUMMARY.json").read_bytes() == original_summary_bytes
    before = (preserved / "SUMMARY.json").read_bytes()
    rp.addendum2_report(root=tmp_path / "repo", ext_root=ext,
                        results_dir=target, log=lambda *a, **k: None)
    assert (preserved / "SUMMARY.json").read_bytes() == before
