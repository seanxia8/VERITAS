"""Tests for the r2-lucid LUCiD generator adapter (runbook §L3.3, gate G6).

Three environments run these: the native suite (torch, no JAX) runs the schema, pack round-trip,
provenance and import-graph tests on synthetic LUCiD digits; the LUCiD environment (JAX, no torch)
additionally runs the kernel tests (determinism, no toy physics called at run time).
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb
from jepa_noise import lucid_mc_generator as mg

REPO = Path(__file__).resolve().parents[3]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _has_lucid():
    try:
        import lucid
        # the cloned release only (an older pinned LUCiD may sit in another environment)
        return Path(lucid.__file__).resolve().is_relative_to(mg.LUCID_ROOT.resolve())
    except Exception:
        return False


HAS_LUCID = _has_lucid()
TOY_WCTE_PACK = Path("/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/wcte_regression/wcte_probe_r2.h5")


def _cfg():
    return mg.load_config()


def _fake_cache(n_events=4, n_pmt=1995, seed=0):
    """Synthetic LUCiD digits in the adapter's cache format (no physics)."""
    rng = np.random.default_rng(seed)
    run = mg.cohort_runs("dev_probe_train")[0]
    ids = np.arange(run.event_start, run.event_start + n_events, dtype=np.int64)
    truths, digits, stats = [], [], []
    for i, eid in enumerate(ids):
        k = 40 + i
        s = rng.choice(n_pmt, size=k, replace=False)
        s = np.concatenate([s, s[:3]])  # repeated digits on three PMTs
        t = rng.normal(30.0, 3.0, s.size)
        t[-3:] += 300.0
        digits.append({"sensor": s.astype(np.int64), "pe": rng.gamma(2.0, 1.0, s.size),
                       "t": t, "physics": np.ones(s.size, bool), "lucid_triggered": True,
                       "n_gates": 0, "n_dark_pe": 0})
        truths.append(mg.Truth(run.run_id, int(eid), "photon", rng.normal(size=3) * 0.3, 1e4, 1.5))
        stats.append({"event_id": int(eid), "domain": "photon", "true_pe": float(k),
                      "n_photons": 10000, "n_calls": 1, "n_siren_chunks": 0, "n_outside": 0,
                      "n_overflow": 0, "n_digits": int(s.size), "digit_pe": 1.0,
                      "lucid_triggered": True, "n_gates": 0, "late_frac_10ns": 0.0})
    run = lb.RunSpec(run.run_id, run.event_namespace, run.law, run.setting, run.event_start,
                     n_events, run.seed)
    prov = {"detector": "wcte", "code_sha": "test",
            "kernel": {"lucid_sha": mg.LUCID_SHA, "digitizer": {"model": "hk"}}}
    return {"event_ids": ids, "truths": truths, "digits": digits, "stats": stats,
            "provenance": prov}, run


@pytest.fixture(scope="module")
def wcte():
    cfg = _cfg()
    geometry, l2r = mg.lucid_geometry("wcte", cfg)
    return cfg, geometry, l2r


def test_geometry_map_is_a_permutation_and_frozen(wcte):
    cfg, geometry, l2r = wcte
    assert sorted(l2r.tolist()) == list(range(geometry.n_pmt))
    for det in mg.SPECS:
        g, m = mg.lucid_geometry(det, cfg)
        assert g.n_pmt == m.size
    rec = json.loads((REPO / "src/jepa_noise/configs/lucid_v1/geometry_comparison.json").read_text())
    assert rec["hk"]["reuse_round2_positions"] and rec["juno"]["reuse_round2_positions"]
    assert rec["wcte"]["raw_positions_max_abs_dev_m_vs_round2"] < 1e-9


def test_veritas_readout_one_hit_per_pmt_and_token_time(wcte):
    cfg, geometry, _ = wcte
    cache, run = _fake_cache(n_events=1)
    dig = cache["digits"][0]
    rule = mg.time_rule("wcte", cfg)
    h = mg.veritas_readout(dig, geometry.n_pmt, np.ones(geometry.n_pmt), rule, cfg["readout_start_ns"])
    assert np.all(np.diff(h.pmt_id) > 0), "sorted unique PMTs"
    assert h.trigger_ns is not None
    lo, hi = rule["prompt_window_ns"]
    assert np.all(h.time_ns >= lo - 1e-4) and np.all(h.time_ns <= hi + 1e-4)
    # the late repeated digits are not chosen over the earliest digit of the same PMT
    assert np.all(h.time_ns < 100.0)


def test_detector_state_only_scales_charge(wcte):
    cfg, geometry, _ = wcte
    cache, run = _fake_cache()
    rule = mg.time_rule("wcte", cfg)
    a = mg.run_from_cache(cache, run, geometry, cfg, rule, with_twin=True)
    for h, t in zip(a.hits, a.twin_hits):
        np.testing.assert_array_equal(h.pmt_id, t.pmt_id)
        np.testing.assert_array_equal(h.time_ns, t.time_ns)
    b = mg.run_from_cache(cache, run, geometry, cfg, rule, with_twin=True)
    for h1, h2 in zip(a.hits, b.hits):
        np.testing.assert_array_equal(h1.charge_pe, h2.charge_pe)  # deterministic


def _write(tmp_path, wcte, cohort="dev_probe_train"):
    cfg, geometry, _ = wcte
    cache, run = _fake_cache()
    rule = mg.time_rule("wcte", cfg)
    lrun = mg.run_from_cache(cache, run, geometry, cfg, rule, with_twin=mg.cohort_has_twin(cohort))
    d = tmp_path / cohort / run.run_id.replace("/", "__")
    m = mg.write_run(d, lrun, geometry, cfg, rule, cache["provenance"])
    return d, m, run


def test_shard_schema_and_provenance(tmp_path, wcte):
    d, m, run = _write(tmp_path, wcte)
    assert sorted(p.name for p in d.iterdir()) == ["SHA256SUMS", "events.npz", "hits.npz",
                                                   "manifest.json", "twin_hits.npz"]
    assert not d.with_name(d.name + ".tmp").exists()
    hits = np.load(d / "hits.npz")
    assert {k: hits[k].dtype.str for k in ("event_ids", "offsets", "pmt_id", "charge_pe", "time_ns")} == \
        {"event_ids": "<i8", "offsets": "<i8", "pmt_id": "<i4", "charge_pe": "<f4", "time_ns": "<f4"}
    ev = np.load(d / "events.npz")
    for field, status in mg.TRUTH_FIELD_STATUS.items():
        assert status.startswith(("filled", "not applicable")), field
        if status.startswith("filled"):
            assert field in ev.files, field
    for key in ("dataset_kind", "sampling_tier", "lucid", "veritas_steps", "truth_field_status",
                "time_rule", "lucid_v1_config_sha256", "dataset_seed", "runs"):
        assert key in m, key
    assert m["dataset_kind"] == mg.DATASET_KIND and m["lucid"]["lucid_sha"] == mg.LUCID_SHA
    assert m["runs"][0]["state_sha256"] == run.state(wcte[1]).digest


@pytest.mark.skipif(not HAS_TORCH, reason="r2_store needs torch")
def test_r2_store_pack_round_trip_matches_toy_schema(tmp_path, wcte):
    from jepa_noise import r2_store as rs
    _, geometry, _ = wcte
    d, m, run = _write(tmp_path / "shards", wcte)
    h5 = tmp_path / "wcte_lucid.h5"
    rs.pack_detector(h5, tmp_path / "shards", {"dev_probe_train": [run]}, attrs={"detector": "wcte"})
    rs.verify_pack(h5, tmp_path / "shards", {"dev_probe_train": [run]})
    cohort = rs.open_cohort(h5, "dev_probe_train", twin=True)
    store = cohort.run_store(0)
    assert store.event_ids.size == run.n_events
    truth = cohort.run_truth(0)
    if TOY_WCTE_PACK.is_file():
        import h5py
        with h5py.File(TOY_WCTE_PACK, "r") as toy, h5py.File(h5, "r") as new:
            for name in ("event_ids", "offsets", "pmt_id", "charge_pe", "time_ns", "twin_pmt_id"):
                assert toy["dev_probe_train"][name].dtype == new["dev_probe_train"][name].dtype, name
            for name in toy["dev_probe_train/truth"]:
                a, b = toy["dev_probe_train/truth"][name], new["dev_probe_train/truth"][name]
                assert a.dtype.kind == b.dtype.kind and a.shape[1:] == b.shape[1:], name
    assert truth["vertex_m"].shape == (run.n_events, 3)


def test_generation_path_imports_no_toy_physics():
    """G6 (static): only allowed lucid_bombs names, no r2_generator, no toy physics names."""
    for path in (REPO / "src/jepa_noise/lucid_mc_generator.py", REPO / "scripts/lucid_generate.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any("r2_generator" in n for n in names), path
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "lb":
                assert node.attr in mg.ALLOWED_LB_NAMES, f"{path.name}: lb.{node.attr}"
                assert node.attr not in mg.FORBIDDEN_LB_NAMES
    mod_src = (REPO / "src/jepa_noise/lucid_mc_generator.py").read_text()
    for name in mg.FORBIDDEN_LB_NAMES:
        assert f"lb.{name}" not in mod_src


@pytest.mark.skipif(not HAS_LUCID, reason="needs the LUCiD environment (JAX + lucid)")
def test_lucid_kernel_determinism_and_no_toy_physics_called(monkeypatch, wcte):
    """G6 (run time) and determinism: two events through LUCiD with toy physics booby-trapped."""
    cfg, geometry, l2r = wcte
    for name in mg.FORBIDDEN_LB_NAMES:
        if hasattr(lb, name):
            monkeypatch.setattr(lb, name, _trap(name))
    kernel = mg.LucidKernel("wcte", 8192)
    run = mg.cohort_runs("dev_probe_train")[0]
    small = dict(cfg, yield_range_photons=dict(cfg["yield_range_photons"], wcte=[2000.0, 4000.0]))
    a = mg.simulate_run_digits(kernel, geometry, l2r, small, "wcte", run, select=[0, 17, 34])
    b = mg.simulate_run_digits(kernel, geometry, l2r, small, "wcte", run, select=[0, 17, 34])
    assert [t.domain for t in a["truths"]] == ["photon", "muon", "electron"]
    for x, y in zip(a["digits"], b["digits"]):
        np.testing.assert_array_equal(x["sensor"], y["sensor"])
        np.testing.assert_array_equal(x["t"], y["t"])
        np.testing.assert_array_equal(x["pe"], y["pe"])
    assert all(s["true_pe"] > 0 for s in a["stats"])


def _trap(name):
    def boom(*args, **kwargs):
        raise AssertionError(f"toy physics {name} called in the LUCiD generation path")
    return boom
