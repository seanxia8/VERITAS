"""r2-mask (runbook 2026-11-23 §M2.1): ``--patch-size`` for ``--mask spatial_patch``."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import ext_r2_repair_train as rt
from jepa_noise import lucid_views as lv
from jepa_noise import r2_store as rs
from jepa_noise.tests.r2_fixtures import small_cylinder

GOLDEN = Path(__file__).with_name("tier1_mask_golden.json")
LUCID_RUNS = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train/runs")


def _lucid_geometry(det):
    from jepa_noise import data_roots as dr
    try:
        return dr.geometry(det, "lucid_v1")
    except Exception as exc:  # pragma: no cover - host without the LUCiD config
        pytest.skip(f"lucid_v1 geometry unavailable: {exc}")


# ------------------------------------------------------------------ golden: default unchanged
@pytest.mark.parametrize("explicit", [False, True])
def test_default_size_masks_identical_to_base_commit_golden(explicit):
    """Default-size masks (flag absent or equal to the default) equal the base-commit golden
    for the 200 registered event ids on WCTE and SK."""
    golden = json.loads(GOLDEN.read_text())
    for det in ("wcte", "sk"):
        g = _lucid_geometry(det)
        assert g.digest == golden[f"geometry_sha256_{det}"]
        kw = {"patch_size": lv.default_patch_size(g.n_pmt)} if explicit else {}
        m = lv.ViewMasker(g, "spatial_patch", seed=20261017, n_min=0, **kw)
        h = hashlib.sha256()
        for e in golden["event_ids"]:
            vm = m.mask(e, np.zeros(g.n_pmt, bool))
            h.update(np.asarray(vm.target, np.int64).tobytes())
            h.update(np.asarray(vm.context, np.int64).tobytes())
            assert "hidden_patches" not in vm.key
        assert h.hexdigest() == golden["masks"][f"{det}/spatial_patch"], (det, explicit)


def test_registered_default_sizes():
    assert lv.default_patch_size(_lucid_geometry("wcte").n_pmt) == 32
    assert lv.default_patch_size(_lucid_geometry("sk").n_pmt) == 173


# ------------------------------------------------------------------ non-default sizes
@pytest.mark.parametrize("size", [5, 7, 13, 19])
def test_non_default_size_hides_ceil_half_of_complete_patches(size):
    g = small_cylinder()  # 114 PMTs, default size 32
    assert lv.default_patch_size(g.n_pmt) == 32
    count = g.n_pmt // size
    order = lv.token_hilbert_order(g)
    tail = set(order[count * size:].tolist())
    patches = order[:count * size].reshape(count, size)
    sizes, seen = set(), np.zeros(g.n_pmt, bool)
    for e in range(200):
        vm = lv.spatial_patch_mask(g, 1000 + e, np.zeros(g.n_pmt, bool), seed=3, patch_size=size)
        sizes.add(vm.target.size)
        assert not tail & set(vm.target.tolist())
        # the hidden set is a union of whole Hilbert patches
        hidden = np.zeros(g.n_pmt, bool)
        hidden[vm.target] = True
        assert all(hidden[p].all() or not hidden[p].any() for p in patches)
        assert vm.key["patch_size"] == size and vm.key["hidden_patches"] == -(-count // 2)
        assert np.array_equal(np.sort(np.concatenate([vm.target, vm.context])), np.arange(g.n_pmt))
        seen[vm.target] = True
    assert sizes == {(-(-count // 2)) * size}
    assert not seen[list(tail)].any() if tail else True


def test_non_default_size_is_deterministic_per_event_and_round_trips():
    g = small_cylinder()
    hit = np.zeros(g.n_pmt, bool)
    hit[::3] = True
    a = lv.spatial_patch_mask(g, 42, hit, seed=1, patch_size=7)
    b = lv.spatial_patch_mask(g, 42, hit, seed=1, patch_size=7)
    c = lv.spatial_patch_mask(g, 43, hit, seed=1, patch_size=7)
    assert np.array_equal(a.target, b.target) and not np.array_equal(a.target, c.target)
    again = lv.mask_from_key(g, a.key, hit)
    assert np.array_equal(again.target, a.target) and np.array_equal(again.context, a.context)


def test_patch_size_refused_outside_spatial_patch_and_out_of_range():
    g = small_cylinder()
    with pytest.raises(ValueError):
        lv.ViewMasker(g, "random", patch_size=8)
    for bad in (0, g.n_pmt // 2 + 1):
        with pytest.raises(ValueError):
            lv.spatial_patch_mask(g, 1, np.zeros(g.n_pmt, bool), patch_size=bad)


def test_loader_batches_use_the_requested_patch_size():
    """A training batch from the loader hides ceil(count/2) whole patches of the given size."""
    from jepa_noise import lucid_bombs as lb
    g = small_cylinder()
    rng = np.random.default_rng(5)
    reads = []
    for _ in range(8):
        pmt = np.sort(rng.choice(g.n_pmt, 60, replace=False))
        reads.append(lb.HitReadout(pmt, rng.uniform(0.3, 2.0, 60), rng.normal(45.0, 20.0, 60)))
    store = lb.sparse_from_readouts(np.arange(100, 108), reads)
    scaling = rs.scaling_for(g, 100.0)
    for size, hidden in ((None, 32), (32, 32), (19, 3 * 19), (7, 8 * 7)):
        kw = {} if size is None else {"patch_size": size}
        m = lv.ViewMasker(g, "spatial_patch", seed=7, n_min=0, **kw)
        batch = rs.tokenise_sparse(store, g, scaling=scaling, masker=m)
        assert set(batch.hidden.sum(1).tolist()) == {hidden}, size


# ------------------------------------------------------------------ CLI: suffix and recipe
def _ns(**kw):
    base = dict(detector="sk", arm="mae_repaired", seed=9000, time_loss_scale_ns=None, mask=None,
                charge_loss_weight=None, time_reference=None, patch_size=None, data_kind="lucid_v1")
    base.update(kw)
    return argparse.Namespace(**base)


def test_patch_size_suffix_order_and_default():
    _lucid_geometry("sk")
    assert rt.run_suffix(_ns(patch_size=173)) == ""
    assert rt.run_suffix(_ns(patch_size=43)) == "__ps43"
    assert rt.run_suffix(_ns(patch_size=692, mask="spatial_patch", time_loss_scale_ns=450.0)) == \
        "__ps692__ts450"
    assert rt.run_id("sk", "mae_repaired", 9001) + rt.run_suffix(_ns(seed=9001, patch_size=43)) == \
        "sk__mae_repaired__s9001__ps43"
    assert rt.run_suffix(_ns(detector="wcte", patch_size=32)) == ""
    assert rt.run_suffix(_ns(detector="wcte", patch_size=8)) == "__ps8"


@pytest.mark.parametrize("kw", [dict(arm="jepa_repaired", patch_size=43),
                                dict(mask="random", patch_size=43),
                                dict(mask="within", patch_size=43),
                                dict(patch_size=0)])
def test_patch_size_refused_outside_mae_repaired_spatial_patch(kw):
    _lucid_geometry("sk")
    with pytest.raises(ValueError):
        rt.run_suffix(_ns(**kw))


def _prepare(detector, **kw):
    args = argparse.Namespace(detector=detector, arm="mae_repaired", seed=9000, data_kind="lucid_v1",
                              time_loss_scale_ns=None, mask=None, charge_loss_weight=None, data=None,
                              time_reference=None, patch_size=None,
                              micro_batch=8, window=None, smoke=None, device="cpu",
                              memory_fraction=None)
    for k, v in kw.items():
        setattr(args, k, v)
    return rt.prepare(args)


@pytest.mark.parametrize("detector,default", [("sk", 173), ("wcte", 32)])
def test_default_recipe_hash_unchanged_with_and_without_the_flag(detector, default):
    record = LUCID_RUNS / f"{detector}__mae_repaired__s9000" / "run_record.json"
    if not record.is_file():
        pytest.skip("r2-lucid returns unavailable on this host")
    want = json.loads(record.read_text())["config_sha256"]
    assert _prepare(detector)["config_sha"] == want
    assert _prepare(detector, patch_size=default)["config_sha"] == want


def test_patch_size_recipe_is_one_factor():
    if not (LUCID_RUNS / "wcte__mae_repaired__s9000" / "run_record.json").is_file():
        pytest.skip("LUCiD data unavailable on this host")
    base = _prepare("wcte")
    var = _prepare("wcte", patch_size=128)
    changed = {k for k in set(base["recipe"]) | set(var["recipe"])
               if base["recipe"].get(k) != var["recipe"].get(k)}
    assert changed == {"patch_size", "identity"}
    assert var["recipe"]["patch_size"] == 128
    assert var["recipe"]["identity"]["run_id"] == "wcte__mae_repaired__s9000__ps128"
    assert var["loader"].patch_size == 128 and base["loader"].patch_size is None
    assert var["recipe"]["view_mode"] == "spatial_patch"


@pytest.mark.parametrize("det,sizes", [("wcte", (8, 32, 128, 256)), ("sk", (43, 173, 692))])
def test_registered_sizes_on_the_lucid_geometries(det, sizes):
    """Review finding 7: every registered size masks real events with fixed whole-patch counts."""
    g = _lucid_geometry(det)
    default = lv.default_patch_size(g.n_pmt)
    for s in sizes:
        count = g.n_pmt // s
        n_hide = count // 2 if s == default else -(-count // 2)
        m = lv.ViewMasker(g, "spatial_patch", seed=20261017, n_min=0,
                          **({} if s == default else {"patch_size": s}))
        got = {m.mask(e, np.zeros(g.n_pmt, bool)).target.size for e in range(1000, 1020)}
        assert got == {n_hide * s}, (det, s)
        assert 0.4 < n_hide * s / g.n_pmt < 0.6


# ------------------------------------------------------------------ M3 seed starts (no LUCiD)
def test_seed_start_truth_clip_and_t0_median():
    from jepa_noise import tier1_yardstick as ty
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.5], [0.0, 0.0, -1.0]])
    data = {"vertex": np.array([[0.1, 0.0, 0.0]]), "t0": np.array([5.0]), "yield": np.array([1e4]),
            "pmt_lucid": np.array([0, 1, 2, 3]), "t_abs": np.array([10.0, 11.0, 12.0, 13.0])}
    starts = {"start_readout": np.array([[0.0, 0.0, 0.0]]), "start_raw": np.array([[3.0, 4.0, 9.0]]),
              "lnN_start": np.array([9.0])}
    sl = slice(0, 4)
    th, clipped = ty.seed_start(data, starts, 0, "truth", pos, sl, 1.2, 1.0)
    assert np.allclose(th, [0.1, 0.0, 0.0, 5.0, np.log(1e4)]) and not clipped
    th, clipped = ty.seed_start(data, starts, 0, "readout", pos, sl, 1.2, 1.0)
    d = np.linalg.norm(pos, axis=1)
    assert np.allclose(th[:3], 0.0) and not clipped and th[4] == 9.0
    assert th[3] == pytest.approx(np.median(data["t_abs"] - d / ty.C_WATER_M_PER_NS))
    th, clipped = ty.seed_start(data, starts, 0, "raw", pos, sl, 1.2, 1.0)
    assert clipped and np.hypot(th[0], th[1]) == pytest.approx(1.15) and th[2] == pytest.approx(0.95)
