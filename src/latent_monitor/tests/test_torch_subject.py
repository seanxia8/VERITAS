# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""S1: the compact transformer wrapped as a Subject — hooks, Jacobians, patching, the protocol end to end."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("reconstruction_model.models.current_compact")

from latent_monitor import HOOKS, Subject, attribute, calibrate, cell_statistics, fit_reference
from latent_monitor.tier1 import sigma_covariance_cells
from latent_monitor.torch_subject import TransformerSubject, train_reference

from .conftest import EVAL_IDS, FIT_IDS, NOISE_IDS


@pytest.fixture(scope="module")
def tsubject(ref_cell):
    from reconstruction_model.models.current_compact import Transformer, TransformerConfig
    cfg = TransformerConfig(d_model=32, d_ff=64, max_seq_len=256, patch_len=16, patch_stride=16, n_head=4,
                            n_time_layers=1, n_channel_layers=1)
    torch.manual_seed(0)
    model = Transformer(cfg); model.init_weights()
    # Tier-1 targets are (amplitude, t0, tau_t): map them onto (energy, spatial_z, spatial_y) heads
    sub = TransformerSubject.wrap(model, ref_cell.implied_whitener(), pos_scale=1.0, output_select=(3, 2, 1),
                                  target_names=("amplitude", "t0", "tau_t"))
    X, T = ref_cell.batch(FIT_IDS[:96])
    sub = train_reference(sub, X, T, ref_cell.geometry, steps=60, lr=1e-3)
    return sub.fit_probe_decoder(X, ref_cell.geometry)


def test_wrapped_model_satisfies_the_protocol(tsubject, ref_cell):
    assert isinstance(tsubject, Subject)
    X, _ = ref_cell.batch(np.arange(3))
    rep = tsubject.represent(X, ref_cell.geometry)
    C, N, d = ref_cell.geometry.n_channels, ref_cell.n_samples, tsubject.latent_dim
    assert set(rep) == set(HOOKS)
    assert rep["whitened"].shape == (3, C, N) and rep["channel"].shape == (3, C, d)
    assert rep["token"].shape == (3, C, d) and rep["z"].shape == (3, d) and rep["output"].shape == (3, 3)
    assert tsubject.jac_output(rep["z"][0], ref_cell.geometry).shape == (3, d)
    assert tsubject.jac_recon(rep["z"][0], ref_cell.geometry).shape == (C * N, d)
    assert tsubject.meta["decoder"] == "linear_probe"
    for stage in HOOKS:
        again = tsubject.forward_from_stage(stage, rep[stage], X, ref_cell.geometry)
        np.testing.assert_allclose(again["output"], rep["output"], atol=1e-4)


def test_jac_output_matches_finite_differences(tsubject, ref_cell):
    X, _ = ref_cell.batch(np.arange(1))
    z = tsubject.represent(X, ref_cell.geometry)["z"][0]
    J = tsubject.jac_output(z, ref_cell.geometry)
    eps = 1e-3
    for i in range(3):
        dz = np.zeros_like(z); dz[i] = eps
        fd = (tsubject.forward_from_stage("z", (z + dz)[None], X, ref_cell.geometry)["output"][0]
              - tsubject.forward_from_stage("z", (z - dz)[None], X, ref_cell.geometry)["output"][0]) / (2 * eps)
        np.testing.assert_allclose(J[:, i], fd, atol=2e-3)


def test_geometry_stage_is_explicit_and_only_refit_token_touches_it(tsubject, ref_cell):
    X, T = ref_cell.batch(np.arange(200, 232))
    assert np.all(tsubject.E_pos == 0.0)
    new = tsubject.refit_stage("token", X, T, ref_cell.geometry, steps=5, lr=1e-2)
    assert not np.all(new.E_pos == 0.0)
    assert new.model is tsubject.model, "token refit does not touch the network"
    with pytest.raises(ValueError):
        tsubject.refit_stage("channel", X, T, ref_cell.geometry)


def test_protocol_runs_end_to_end_on_the_transformer(tsubject, ref_cell):
    """Attribution outcomes on a 60-step CPU model are not the experiment; that the protocol runs is."""
    ref = fit_reference(tsubject, ref_cell, EVAL_IDS[:24], NOISE_IDS[:40], n_bootstrap=30)
    thr = calibrate(ref)
    assert ref.fisher_rank >= 1 and ref.k_out == 3
    cell = sigma_covariance_cells(ref_cell)[0]
    s = cell_statistics(ref, tsubject, ref_cell, cell, EVAL_IDS[:24], NOISE_IDS[:40])
    a = attribute(s, thr)
    assert a.label in {"sigma_cov", "sigma_struct", "event", "event_in_span", "geometry", "undeclared", "output_null", "output_aligned"}
    assert np.isfinite(s.mean_shift_norm) and np.isfinite(np.mean(s.z_var_ratio))
