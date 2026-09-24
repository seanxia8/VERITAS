# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
from __future__ import annotations

import numpy as np
import pytest

from latent_monitor import HOOKS, Geometry, Subject
from latent_monitor.statistics import participation_ratio, rank_auroc


def test_linear_subject_satisfies_the_protocol_and_hook_shapes(subject, ref_cell):
    assert isinstance(subject, Subject)
    X, _ = ref_cell.batch(np.arange(3))
    rep = subject.represent(X, ref_cell.geometry)
    assert set(rep) == set(HOOKS)
    C, N, k = ref_cell.geometry.n_channels, ref_cell.n_samples, subject.latent_dim
    assert rep["whitened"].shape == (3, C, N) and rep["channel"].shape == (3, C, k)
    assert rep["token"].shape == (3, C, k + subject.E.shape[0]) and rep["z"].shape == (3, k)
    assert rep["output"].shape == (3, 3)
    assert subject.jac_recon(rep["z"][0], ref_cell.geometry).shape == (C * N, k)
    assert subject.jac_output(rep["z"][0], ref_cell.geometry).shape == (3, k)
    # forward_from_stage with the stage's own value reproduces the pass
    for stage in HOOKS:
        again = subject.forward_from_stage(stage, rep[stage], X, ref_cell.geometry)
        np.testing.assert_allclose(again["output"], rep["output"], atol=1e-10)


def test_decoder_is_the_whitened_reconstruction_jacobian(subject, ref_cell):
    z = np.eye(subject.latent_dim)
    dec = subject.decode(z, ref_cell.geometry)              # (k, C, N)
    J = subject.jac_recon(z[0], ref_cell.geometry)          # (C·N, k)
    np.testing.assert_allclose(dec.reshape(subject.latent_dim, -1).T, J, atol=1e-12)


def test_geometry_validation():
    with pytest.raises(ValueError):
        Geometry(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        Geometry(np.zeros((4, 3)), groups=np.zeros(3, dtype=int))


def test_helpers():
    assert participation_ratio(np.ones(10)) == pytest.approx(1.0)
    assert participation_ratio(np.array([1.0, 0, 0, 0])) == pytest.approx(0.25)
    assert rank_auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True])) == pytest.approx(0.75)
