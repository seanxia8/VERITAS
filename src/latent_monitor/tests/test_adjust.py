# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Adjustments that follow from a diagnosis (plan §4)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_monitor import activation_patch, cell_statistics, damage_patch, fit_reference, refit_stage, rewhiten
from latent_monitor.tier1 import geometry_cells, geometry_gain, sigma_covariance_cells, sigma_structural_cells

from .conftest import EVAL_IDS, NOISE_IDS


@pytest.mark.parametrize("i", range(4))
def test_rewhitening_restores_unit_variance_and_keeps_z_meaning(reference, subject, ref_cell, i):
    cell = sigma_covariance_cells(ref_cell)[i]
    adj, info = rewhiten(subject, cell.noise_batch(np.arange(6000, 6200)))
    assert info["kappa_correction"]["kappa"] > 1.3
    # z on a pure signal is preserved: the coefficients on the same raw signal basis
    w, _ = ref_cell.family.waveform(1001, ref_cell.n_samples)
    sig = geometry_gain(ref_cell.geometry)[:, None] * w[None, :]
    z0 = subject.represent(sig[None], ref_cell.geometry)["z"][0]
    z1 = adj.represent(sig[None], ref_cell.geometry)["z"][0]
    np.testing.assert_allclose(z1, z0, rtol=0.03, atol=0.02 * np.abs(z0).max())
    # noise-only variance ratio returns to 1 under the adjusted subject's own reference
    ref2 = fit_reference(adj, cell, EVAL_IDS, NOISE_IDS, seed=0)
    s = cell_statistics(ref2, adj, cell, cell, EVAL_IDS, NOISE_IDS)
    assert abs(np.mean(s.z_var_ratio) - 1.0) < 0.1
    assert s.psd_dev_smooth < 0.05
    # and the consequence is not worse than before the adjustment
    Xc, Tc = cell.batch(EVAL_IDS)
    before = np.mean(np.abs(subject.outputs(Xc, cell.geometry) - Tc))
    after = np.mean(np.abs(adj.outputs(Xc, cell.geometry) - Tc))
    assert after <= before * 1.05


def test_rewhitening_touches_neither_the_signal_basis_nor_the_head(subject, ref_cell):
    cell = sigma_covariance_cells(ref_cell)[0]
    adj, _ = rewhiten(subject, cell.noise_batch(np.arange(6000, 6100)))
    np.testing.assert_allclose(adj.O, subject.O)
    np.testing.assert_allclose(adj.o0, subject.o0)
    S0, S1 = subject.signal_basis_raw(), adj.signal_basis_raw()
    np.testing.assert_allclose(S1, S0, atol=1e-8)


def test_patching_is_flat_for_input_side_corruption(subject, ref_cell):
    """Both patch directions recover fully at every stage: no stage of a linear subject creates damage."""
    cell = sigma_structural_cells(ref_cell)[1]  # channel loss
    Xp, Tp = cell.batch(EVAL_IDS)
    Xc, _ = ref_cell.batch(EVAL_IDS)
    rec = activation_patch(subject, Xp, Xc, Tp, cell.geometry)
    dmg = damage_patch(subject, Xp, Xc, Tp, cell.geometry)
    assert rec["unpatched"]["recovery"] == 0.0
    for stage in ("whitened", "channel", "token", "z"):
        assert rec[stage]["recovery"] == pytest.approx(1.0, abs=1e-9)
        assert dmg[stage]["transmitted"] == pytest.approx(1.0, abs=1e-9)


def test_stage_refit_on_geometry_repairs_through_the_head_not_the_pooling(subject, reference, ref_cell):
    """The finding the plan's G row must carry: for the linear subject a granularity change is a gain on z."""
    cell = geometry_cells(ref_cell)[1]
    Xg, Tg = cell.batch(np.arange(300, 400))
    Xe, Te = cell.batch(EVAL_IDS)
    before = np.mean(np.abs(subject.outputs(Xe, cell.geometry) - Te), axis=0) / reference.consequence_ref
    assert np.max(before) > 1.1
    out = refit_stage(subject, "output", Xg, Tg, cell.geometry)
    after_head = np.mean(np.abs(out.outputs(Xe, cell.geometry) - Te), axis=0) / reference.consequence_ref
    assert np.max(after_head) < 1.05
    tok = refit_stage(subject, "token", Xg, Tg, cell.geometry)
    after_tok = np.mean(np.abs(tok.outputs(Xe, cell.geometry) - Te), axis=0) / reference.consequence_ref
    assert np.max(after_tok) > np.max(after_head), "three pooling weights cannot absorb a channel-count gain"
    np.testing.assert_allclose(tok.A, subject.A)   # the token refit touched nothing else
    np.testing.assert_allclose(tok.O, subject.O)


def test_refit_stage_rejects_unknown_stage(subject, ref_cell):
    X, T = ref_cell.batch(np.arange(4))
    with pytest.raises(ValueError):
        refit_stage(subject, "whitened", X, T, ref_cell.geometry)
