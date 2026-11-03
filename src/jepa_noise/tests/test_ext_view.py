# SPDX-License-Identifier: MIT
"""Acceptance tests for the view-design extension (runbook 3 Nov 2026 §3.4).

Covers: the full-line + C-row authorisation gate; the ``SplitMasker`` patch
(``within``/``across`` bit-identical to the unpatched masker, ``module_parity``
whole modules by dense ``module_index`` parity, ``random_half`` fixed 997/998
partition with the recorded salt, non-zero counters, both import-time bindings
patched); run-directory refusal + resumed_from_step; three-valued ``view_r``;
dry-run failure paths; a finite synthetic training step with the patch active; and
the evaluate item pinned to ``split="within"`` for a ``module_parity`` run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_view as ev  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import lucid_views as lv  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_view", ROOT / "scripts" / "jepa_ext_view.py")
view_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(view_cli)


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


def _hit(geometry, *, seed=0, n_hits=None, all_hits=False):
    rng = np.random.default_rng(seed)
    hit = np.zeros(geometry.n_pmt, dtype=bool)
    if all_hits:
        hit[:] = True
    else:
        hit[rng.choice(geometry.n_pmt, n_hits or geometry.n_pmt // 2,
                       replace=False)] = True
    return hit


# ---------------------------------------------------------------- gate (1)
def _fake_root(tmp_path, *, runbook_line=None, decisions_line=None, with_row=True):
    root = tmp_path / "root"
    (root / "docs/jepa").mkdir(parents=True)
    line = ev.AUTHOR_LINE if runbook_line is None else runbook_line
    row = ev.C_ROW if with_row else "no row"
    (root / ev.RUNBOOK_REL).write_text(
        "## 0. Authority\n\n> \"" + line + "\"\n\n" + row + "\n\n## 1. Why\n")
    (root / ev.DECISIONS_REL).write_text(
        "history\n" + (ev.AUTHOR_LINE if decisions_line is None else decisions_line) + "\n")
    return root


def test_gate_requires_the_full_line_and_c_row(tmp_path):
    assert ev.assert_authorised(_fake_root(tmp_path))["authorised"] is True
    with pytest.raises(SystemExit) as caught:
        ev.assert_authorised(_fake_root(tmp_path / "a", runbook_line="我授权"))
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        ev.assert_authorised(_fake_root(tmp_path / "b", with_row=False))
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        ev.assert_authorised(_fake_root(tmp_path / "c", decisions_line="我授权"))
    assert caught.value.code == 2
    assert ev.assert_authorised(ROOT)["authorised"] is True


# ----------------------------------------------- SplitMasker equivalence (2)
def test_within_and_across_are_bit_identical_to_the_unpatched_masker(small_geometry):
    for split in ("within", "across"):
        hit = _hit(small_geometry, seed=3, n_hits=80)
        reference = lv.ViewMasker(small_geometry, split, seed=1, n_min=20)
        patched = ev.SplitMasker(small_geometry, split, seed=1, n_min=20)
        left = reference.mask(7, hit)
        right = patched.mask(7, hit)
        assert np.array_equal(left.context, right.context)
        assert np.array_equal(left.target, right.target)
        assert left.accepted == right.accepted
        assert left.context_hits == right.context_hits
        assert left.target_hits == right.target_hits
        assert left.key == right.key


def test_module_parity_puts_whole_modules_on_one_side_with_dense_parity(full_geometry):
    masker = ev.SplitMasker(full_geometry, "module_parity", seed=9000, n_min=20)
    hit = _hit(full_geometry, seed=1, n_hits=600)
    mask = masker.mask(11, hit)
    in_context = np.zeros(full_geometry.n_pmt, dtype=bool)
    in_context[mask.context] = True
    expected = (full_geometry.module_index % 2) == 0
    assert np.array_equal(in_context, expected)
    assert mask.accepted is True
    # whole modules: each module's 19 PMTs are all on one side
    for module in range(full_geometry.n_modules):
        members = in_context[full_geometry.module_index == module]
        assert members.all() or not members.any()
    assert mask.context_hits + mask.target_hits == int(hit.sum())
    assert mask.key["module_index_parity"] == "even=context,odd=target"
    # dense parity differs from raw mpmt_id parity because raw id 45 is absent
    assert full_geometry.n_pmt == 1995


def test_random_half_is_a_fixed_997_998_partition_with_the_recorded_salt(full_geometry):
    masker = ev.SplitMasker(full_geometry, "random_half", seed=9000, n_min=20)
    hit = _hit(full_geometry, seed=2, n_hits=800)
    first = masker.mask(0, hit)
    second = masker.mask(1, hit)
    assert first.context.size == 997 and first.target.size == 998
    assert np.array_equal(first.context, second.context)
    assert np.array_equal(first.target, second.target)
    assert first.key["salt"] == ev.RANDOM_HALF_SALT == 20260101
    assert first.key["seed"] == 9000
    assert sorted(np.concatenate([first.context, first.target]).tolist()) == list(
        range(full_geometry.n_pmt))
    report = masker.report()
    assert report["n_events"] == 2 and report["n_accepted"] == 2
    assert report["split"] == "random_half"


def test_patch_counters_are_nonzero_after_a_batch(full_geometry):
    with ev.masker_patch(log=lambda *a, **k: None) as _:
        masker = pe.ViewMasker(full_geometry, "module_parity", seed=9000, n_min=20)
    hit = _hit(full_geometry, seed=5, all_hits=True)
    for event in range(3):
        mask = masker.mask(event, hit)
        assert mask.accepted
    report = masker.report()
    assert report["n_events"] == 3 and report["n_accepted"] == 3
    assert report["context_hit_fraction"] > 0
    assert report["target_hit_fraction"] > 0


# --------------------------------------- pe.ViewMasker also patched (3)
def _write_tiny_run(directory, geometry, *, events=8, hits=300, seed=0):
    rng = np.random.default_rng(seed)
    reads = [lb.HitReadout(
        np.sort(rng.choice(geometry.n_pmt, hits, replace=False)).astype(np.int32),
        (rng.exponential(3.0, hits) + 0.3).astype(np.float32),
        rng.normal(50.0, 5.0, hits).astype(np.float32)) for _ in range(events)]
    directory.mkdir(parents=True, exist_ok=True)
    lb.write_sparse_hits(directory / "hits.npz",
                         lb.sparse_from_readouts(np.arange(events, dtype=np.int64), reads))
    return directory


def test_point_encoder_binding_is_patched_too(tmp_path, full_geometry):
    run_dir = _write_tiny_run(tmp_path / "run", full_geometry, events=8, hits=300)
    scaling = pe.FeatureScaling.from_geometry(full_geometry)
    with ev.masker_patch(log=lambda *a, **k: None):
        assert ev.masker_patch_active() is True
        assert lr.ViewMasker is ev.SplitMasker and pe.ViewMasker is ev.SplitMasker
        batch = pe.tokenise(run_dir, full_geometry, split="module_parity", seed=0,
                            scaling=scaling)
    assert ev.masker_patch_active() is False
    assert lr.ViewMasker is lv.ViewMasker and pe.ViewMasker is lv.ViewMasker
    assert batch.split == "module_parity"
    assert batch.features.shape[1] == full_geometry.n_pmt
    assert int(batch.target[0].sum()) == 988 and int(batch.hidden[0].sum()) == 988
    assert batch.keys and batch.keys[0]["split"] == "module_parity"


# ------------------------------------------------- run dir / resume (4)
def test_run_dir_refusal_and_fresh_resume(tmp_path):
    out = tmp_path / "ext/train/views/module_parity"
    run_id, run_dir = ev.prepare_run_dir(out, "module_parity", 9000, steps=ev.STEPS,
                                         device="cpu")
    assert run_id == "dev_s005_within__sigreg_jepa__s9000"
    sidecar = json.loads((run_dir / "variant.json").read_text())
    assert sidecar["split"] == "module_parity"
    assert sidecar["extension"] == ev.EXTENSION
    assert sidecar["cell_id"] == "dev_s005_within"
    with pytest.raises(RuntimeError, match="already exists"):
        ev.prepare_run_dir(out, "module_parity", 9000)
    from jepa_noise import ext_h1clean as ec
    assert ec.assert_fresh_resume({"resumed_from_step": None}) == 0
    with pytest.raises(RuntimeError, match="resumed_from_step"):
        ec.assert_fresh_resume({"resumed_from_step": 10})


# ------------------------------------------------------ view_r verdicts (5)
def _row(full_v, full_r2, half_v, half_r2):
    return {"full_input": {"vertex_resolution_m": full_v,
                           "token_log_charge_r2": {"r2": full_r2}},
            "half_pass": {"vertex_resolution_m": half_v,
                          "token_log_charge_r2": {"r2": half_r2}}}


def test_view_r_three_valued_per_split():
    recovered = ev.view_r({"9000": _row(0.14, 0.9, 0.13, 0.9),
                           "9001": _row(0.15, 0.9, 0.14, 0.9)})
    assert recovered["verdict"] == "recovered"
    fails = ev.view_r({"9000": _row(0.22, 0.02, 0.20, 0.1),
                       "9001": _row(0.24, 0.03, 0.21, 0.1)})
    assert fails["verdict"] == "not recovered"
    at_floor = ev.view_r({"9000": _row(0.160072, 0.9, 0.13, 0.9),
                          "9001": _row(0.160072, 0.9, 0.13, 0.9)})
    assert at_floor["verdict"] == "not recovered"
    disagreement = ev.view_r({"9000": _row(0.14, 0.9, 0.13, 0.9),
                              "9001": _row(0.22, 0.9, 0.13, 0.9)})
    assert disagreement["verdict"] == "unclear"
    ranked, recommendation = ev.rank_splits(
        {"within": recovered, "across": fails, "module_parity": fails,
         "random_half": fails})
    assert ranked[0]["split"] == "within"
    assert "within" in recommendation


# ------------------------------------------- dry-run failure paths (6)
def test_preflight_failure_paths(tmp_path):
    assert ev.entries_projection(44949)["pass"] is True
    assert ev.entries_projection(35000)["pass"] is False
    empty = tmp_path / "data"
    (empty / "lucid").mkdir(parents=True)
    assert ev.probe_status(empty)["pass"] is False
    with pytest.raises(RuntimeError):
        ev.live_free_entries(tmp_path / "missing_guard.json")
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"level": "HOLD", "caps": {"free_entries": 31000}}))
    assert ev.live_free_entries(guard)["guard_level"] == "HOLD"
    parser = view_cli.parser()
    args = parser.parse_args(["dry-run", "--out", "/tmp/plan.json"])
    assert args.mode == "dry-run"
    args = parser.parse_args(["train", "--split", "module_parity", "--seed", "9000",
                              "--out", "/ceph/ext/train/views/module_parity"])
    assert args.split == "module_parity"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--split", "module_half", "--seed", "9000",
                           "--out", "/tmp/x"])


# --------------------------------------- smoke training finite with patch (7)
def test_smoke_level_training_is_finite_with_the_patch(small_geometry):
    settings = la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))
    arm = la.LucidArm("sigreg_jepa", geometry=small_geometry, settings=settings,
                      seed=9000, encoder_config=pe.EncoderConfig(window=16, shift=8))
    hit_mask = np.zeros((2, small_geometry.n_pmt), bool)
    hit_mask[:, ::3] = True
    hidden = hit_mask.copy()
    batch = pe.LucidBatch(
        features=torch.randn(2, small_geometry.n_pmt, pe.N_FEATURES),
        target=torch.as_tensor(hidden), hidden=torch.as_tensor(hidden),
        context_index=torch.as_tensor(np.tile(np.flatnonzero(~hidden[0]),
                                              (2, 1))),
        target_index=torch.as_tensor(np.tile(np.flatnonzero(hidden[0]), (2, 1))),
        hit_flags=hit_mask,
        log_charge=np.zeros((2, small_geometry.n_pmt), np.float32),
        event_ids=np.arange(2), keys=tuple({} for _ in range(2)), report={},
        scaling=pe.FeatureScaling.from_geometry(small_geometry), split="within",
        occupancy="natural")
    optimizer = torch.optim.Adam(arm.parameters(), lr=1e-3)
    with ev.masker_patch(log=lambda *a, **k: None):
        out = arm._forward_sigreg(batch)
        assert np.isfinite(float(out["loss"]))
        out["loss"].backward()
        optimizer.step()
    metrics = arm._forward_sigreg(batch)["metrics"]
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert ev.masker_patch_active() is False


# ------------------------------- evaluate item pinned to within (8)
def test_evaluate_item_is_pinned_to_within_for_a_module_parity_run(tmp_path,
                                                                   monkeypatch):
    run_dir = tmp_path / "train/views/module_parity/runs/dev_s005_within__sigreg_jepa__s9000"
    run_dir.mkdir(parents=True)
    (run_dir / "variant.json").write_text(json.dumps(
        {"registered": False, "extension": ev.EXTENSION, "split": "module_parity",
         "seed": 9000, "arm": ev.ARM_NAME, "cell_id": ev.CELL_ID,
         "out_root": str(run_dir.parents[2]), "run_id": run_dir.name,
         "steps": ev.STEPS, "device": "cpu"}))
    (run_dir / "final.pt").write_bytes(b"checkpoint")
    captured = {}

    def fake_evaluate_run(item, probes, geometry, config, arms_config, *, device,
                          max_events=None, **kwargs):
        captured.update(item)
        return {"forms": {"full_input": {"vertex_resolution_m": 0.2},
                          "half_pass": {"vertex_resolution_m": 0.3}}}

    monkeypatch.setattr(ev.eh, "evaluate_run", fake_evaluate_run)
    runs = ev.evaluate_run_dirs("module_parity", [run_dir], {}, None, {}, {},
                                device=torch.device("cpu"))
    assert captured["split"] == "within"
    assert captured["arm"] == ev.ARM_NAME and captured["seed"] == 9000
    assert sorted(runs) == ["9000"]
    assert runs["9000"]["probe_split"] == "within"
    # evaluation must not have installed the masker patch
    assert ev.masker_patch_active() is False
