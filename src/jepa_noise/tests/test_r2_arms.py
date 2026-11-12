# SPDX-License-Identifier: MIT
"""Round-2 arms: test 7 (build on every geometry, position-free label pass, rf control)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_r2 as x
from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_generator as gen
from jepa_noise import r2_store as rs
from jepa_noise.lucid_views import ViewMasker
from jepa_noise.tests.r2_fixtures import lucid_available, small_cylinder, small_sphere


def _batch(geometry):
    cfg = gen.load_r2_config(20000.0, 60000.0)
    run = lb.RunSpec("lucid/dev_train/0000", "lucid/dev_train", "F1", 0.05, 1_000_000, 6,
                     lb.DEFAULT_SEED)
    r = gen.generate_run(geometry, cfg, run, gen.TimeRule(4, 20.0, (-20.0, 60.0), 100.0))
    store = lb.sparse_from_readouts(r.event_ids, r.hits)
    return rs.tokenise_sparse(store, geometry, scaling=rs.scaling_for(geometry, 100.0),
                              masker=ViewMasker(geometry, "within", seed=0, n_min=0))


@pytest.mark.parametrize("make", [small_cylinder, small_sphere])
def test_arms_build_and_train_step_on_every_geometry(make):
    g = make()
    batch = _batch(g)
    for arm in x.ALL_ARMS:
        model = x.build_r2_arm(arm, g, 9000, scaling=rs.scaling_for(g, 100.0))
        if arm == x.INIT_ARM:
            with torch.no_grad():
                z = model.jepa.encode(batch.features)
            assert z.shape == (batch.features.shape[0], g.n_pmt, 128)
            continue
        out = model(batch, step=0)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()


@pytest.mark.skipif(not lucid_available(), reason="LUCiD config at the pin not available")
@pytest.mark.parametrize("detector", ["sk", "juno", "hk"])
def test_arms_build_on_lucid_geometries(detector):
    from jepa_noise import r2_geometry as rg
    g = rg.load_r2_geometry(detector)
    for arm in x.ALL_ARMS:
        model = x.build_r2_arm(arm, g, 9000)
        assert model.n_pmt == g.n_pmt and model.pmts_per_module == g.pmts_per_module
    rf = x.build_r2_arm(x.RF_ARM, g, 9000)
    assert rf.jepa.config.window == rg.DESIGN_TABLE[detector]["w_rf"]


def test_posfree_label_pass_zeroes_exactly_columns_0_to_5():
    g = small_cylinder()
    batch = _batch(g)
    calls = []
    for arm, zero in (("target_posfree", True), ("sigreg_jepa", False)):
        model = x.build_r2_arm(arm, g, 9000)
        original = model.jepa.encode
        calls.clear()

        def spy(features, hidden_mask=None):
            calls.append((features.detach().clone(), None if hidden_mask is None
                          else hidden_mask.detach().clone()))
            return original(features, hidden_mask=hidden_mask)
        model.jepa.encode = spy
        model._encode_views(batch)
        assert len(calls) == 3  # context, full, label
        assert torch.equal(calls[0][0], batch.features) and torch.equal(calls[1][0], batch.features)
        label_features, label_mask = calls[2]
        assert torch.equal(label_mask, ~batch.target)
        if zero:
            assert torch.all(label_features[..., 0:6] == 0)
            assert torch.equal(label_features[..., 6:], batch.features[..., 6:])
            assert not torch.all(batch.features[..., 0:6] == 0)
        else:
            assert torch.equal(label_features, batch.features)


def test_rf_control_differs_only_in_window_and_shift():
    g = small_cylinder()  # 114 PMTs -> W_rf = 64
    big = type("G", (), {"n_pmt": 11096})()
    diff = x.config_diff(x.encoder_config_for(x.RF_ARM, big), x.encoder_config_for("mae_mse", big))
    assert set(diff) == {"window", "shift"} and diff["window"] == (384, 64)
    assert diff["shift"] == (192, 32)
    assert x.config_diff(x.encoder_config_for(x.RF_ARM, g), x.encoder_config_for("mae_mse", g)) == {}
    a = x.build_r2_arm(x.RF_ARM, g, 9000).state_dict()
    b = x.build_r2_arm("mae_mse", g, 9000).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    batch = _batch(g)
    torch.manual_seed(0)
    la_ = x.build_r2_arm(x.RF_ARM, g, 9000)(batch, step=0)["loss"]
    lb_ = x.build_r2_arm("mae_mse", g, 9000)(batch, step=0)["loss"]
    assert float(la_) == float(lb_)
