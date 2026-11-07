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
import json
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


def _fake_repo(tmp_path, *, acceptance=True, record_mutator=None):
    """A throw-away repository with the committed runbook and an acceptance record."""
    root = tmp_path / "repo"
    (root / "docs/jepa").mkdir(parents=True)
    (root / rp.RUNBOOK_REL).write_text("## 0. Authority\n\nPart A text.\n")
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
              "lead": "lead", "date": "2026-11-07"}
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
    assert all(result["checks"][key]["pass"] for key in ("a", "b", "c"))
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
    from scripts import jepa_ext_repair  # noqa: F401  (import side of the CLI)
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
                             "x3-fisher", "x3", "report"}


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
    calls = []

    def fake_stream(model, dataset, geometry, condition, *, split, config, device,
                    cache, cache_key, log, **kwargs):
        calls.append(split)
        return {"events": np.zeros((2, 2)), "run_ids": np.asarray(["r0", "r0"],
                                                                  dtype=object),
                "event_ids": np.arange(2), "vertex": np.zeros((2, 3))}

    monkeypatch.setattr(rp, "stream_condition_embeddings", fake_stream)
    monkeypatch.setattr(rp, "fit_vertex_probe", lambda rows: {})
    monkeypatch.setattr(rp.la, "probe_predictions", lambda fitted, dataset: np.zeros(
        (2, 3)))
    probes = {"geometry": None, "s005": {"train": "x", "test": "y"}}
    condition = {"name": "N", "train": "s005", "test": "s005", "condition": None}
    rp._frozen_refit_vertex(None, probes, condition, "across", {}, "cpu",
                            cache=None, log=lambda *a, **k: None)
    assert calls == ["across", "across"]


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
    probes = {"s005": {"train": s005, "test": s005},
              "s030": {"train": s030, "test": s030}}
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
    rng = np.random.default_rng(5)
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
