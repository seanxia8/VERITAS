# SPDX-License-Identifier: MIT
"""Round-2 guards: test 8 (forbidden paths, training modes outside §5 / §8)."""
from __future__ import annotations

import pytest

from jepa_noise import ext_r2 as x


@pytest.mark.parametrize("path", [
    "/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/lucid/x/hits.npz",
    "/ceph/dwong/jepa/data/lucid_r6_v2/new.npz",
    "/ceph/dwong/jepa/data/round2_fake/x",
    "/ceph/dwong/jepa/returns/ext_repair_20261106/runs/a/checkpoint.pt",
    "/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11_other/x",
    "/work/dwong/jepa_a10/returns/x.json",
    "/ceph/dwong/jepa/code/jepa_work_20261107/src/x.py",
    "/ceph/dwong/jepa/data/round2/../lucid_r6/x"])
def test_forbidden_paths_refused(path):
    with pytest.raises(x.ForbiddenPath):
        x.check_write_path(path)


@pytest.mark.parametrize("path", [
    "/ceph/dwong/jepa/data/round2/sk/sk_r2.h5",
    "/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/runs/sk__mae_mse__s9000/checkpoint.pt",
    "/work/dwong/r2_gen/sk/dev_train/x/hits.npz"])
def test_allowed_paths(path):
    assert str(x.check_write_path(path)) == path


def test_training_modes():
    for arm in x.MAIN_ARMS:
        for seed in (9000, 9001):
            for det in ("sk", "juno", "hk"):
                x.allowed_training(arm, seed, detector=det)
    x.allowed_training("mae_mse_rf", 9000, detector="hk")
    x.allowed_training("mae_mse", 9002, detector="sk", triggered=("mae_mse",))
    bad = [("init", 9000, "sk", ()), ("mae_mse_rf", 9001, "sk", ()),
           ("mae_mse", 9002, "sk", ()), ("mae_mse", 9002, "sk", ("target_posfree",)),
           ("mae_mse_rf", 9002, "sk", ("mae_mse_rf",)), ("sigreg_jepa", 9003, "sk", ()),
           ("mae_mse", 9000, "wcte", ()), ("ijepa_ema", 9000, "sk", ()),
           ("anchored", 9000, "sk", ())]
    for arm, seed, det, trig in bad:
        with pytest.raises(ValueError):
            x.allowed_training(arm, seed, detector=det, triggered=trig)


def test_train_refuses_unauthorised_before_any_work(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        x.train_r2(detector="sk", arm="mae_mse", seed=9003, h5_path=tmp_path / "x.h5",
                   out_root=x.RETURNS_ROOT, device="cpu")
    with pytest.raises(ValueError):
        x.train_r2(detector="sk", arm="mae_mse", seed=9000, h5_path=tmp_path / "x.h5",
                   out_root=x.RETURNS_ROOT, steps=500, device="cpu")
    with pytest.raises(x.ForbiddenPath):
        x.train_r2(detector="sk", arm="mae_mse", seed=9000, h5_path=tmp_path / "x.h5",
                   out_root="/ceph/dwong/jepa/returns/ext_p2_20261109", device="cpu")
