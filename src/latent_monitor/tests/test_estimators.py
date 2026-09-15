# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The four linear classes (EXPERIMENT_DESIGN.md §IV) on a planted tangent-space problem."""

from __future__ import annotations

import numpy as np
import pytest

from latent_monitor import estimators as est


@pytest.fixture(scope="module")
def planted():
    """Pulses with amplitude, small shift and decay-time spread in AR(1) noise."""
    rng = np.random.default_rng(0)
    d, k, n = 64, 3, 600
    t = np.arange(d) * 1.0
    s = np.exp(-t / 12.0) * (1 - np.exp(-t / 2.5))
    s /= np.linalg.norm(s)
    ds = np.gradient(s)
    dshape = (t / 12.0**2) * np.exp(-t / 12.0) * (1 - np.exp(-t / 2.5))
    tangent = np.vstack([s, ds, dshape])
    Sigma = 0.8 ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    M = np.linalg.inv(Sigma)
    L = np.linalg.cholesky(Sigma)
    a = 5 + rng.standard_normal(n)
    tau = 0.8 * rng.standard_normal(n)
    th = 2.0 * rng.standard_normal(n)
    clean = a[:, None] * s + (a * tau)[:, None] * ds + (a * th)[:, None] * dshape
    X = clean + 0.08 * (rng.standard_normal((n, d)) @ L.T)
    return dict(X=X, s=s, tangent=tangent, Sigma=Sigma, M=M, a=a, tau=tau, th=th, k=k)


def test_of_amplitude_unbiased_and_mismatch_inflates_variance(planted):
    p = planted
    amp = est.gls_amplitude_metric(p["X"], p["s"], p["M"])
    # the derivative directions are not Σ⁻¹-orthogonal to s, so allow a small bias
    assert abs(np.mean(amp - p["a"])) < 0.1
    matched = est.gls_amplitude_variance_metric(p["s"], p["M"])
    white_filter = est.gls_amplitude_variance_metric(p["s"], np.ones(p["s"].size), p["Sigma"])
    assert white_filter >= matched * 0.999
    of = est.OptimalFilter(p["s"], p["M"])
    assert of.reconstruct(p["X"]).shape == p["X"].shape and of.encode(p["X"]).shape == (p["X"].shape[0],)


def test_cw_pca_recovers_tangent_span(planted):
    p = planted
    cw = est.fit_weighted_pca(p["X"], p["M"], p["k"])
    ang = est.principal_angles(cw.components, p["tangent"], weights=p["M"])
    assert ang.max() < 10.0
    z = cw.encode(p["X"])
    assert z.shape == (p["X"].shape[0], p["k"])
    np.testing.assert_allclose(cw.reconstruct(p["X"]), est.project_onto_basis(p["X"], cw.components, weights=p["M"], mean=cw.mean))


@pytest.mark.parametrize("trainer", [est.train_weighted_linear_ae, est.train_whitened_linear_ae])
def test_tied_ae_lbfgs_reaches_empca_optimum(planted, trainer):
    p = planted
    res = trainer(p["X"], p["M"], p["k"], seed=1)
    assert res.relative_gap < 1e-6
    assert res.max_principal_angle_deg < 0.5
    assert res.m_orthonormality_error < 1e-6
    cw = est.fit_weighted_pca(p["X"], p["M"], p["k"])
    np.testing.assert_allclose(res.reconstruct(p["X"]), cw.reconstruct(p["X"]), atol=1e-4)


def test_closed_form_equals_weighted_pca(planted):
    p = planted
    cf = est.tied_linear_ae_closed_form(p["X"], p["k"], weights=p["M"])
    cw = est.fit_weighted_pca(p["X"], p["M"], p["k"])
    np.testing.assert_allclose(cf.explained_variance, cw.explained_variance)


def test_gauge_fix_makes_code_coordinates_physical(planted):
    p = planted
    res = est.train_weighted_linear_ae(p["X"], p["M"], p["k"], seed=3)
    rep = est.mixing_matrix(res.components, p["tangent"], p["M"])
    assert rep.identifiable
    aligned, _ = est.align_to_tangent(res.components, p["tangent"], p["M"])
    z = est.SubspaceFit(aligned, res.mean, np.full(p["k"], np.nan), p["M"]).encode(p["X"])
    assert np.corrcoef(z[:, 0], p["a"])[0, 1] > 0.95
    assert np.corrcoef(z[:, 1], p["a"] * p["tau"])[0, 1] > 0.9
    assert np.corrcoef(z[:, 2], p["a"] * p["th"])[0, 1] > 0.9


def test_nfpa_recovers_separable_directions_and_round_trips():
    rng = np.random.default_rng(1)
    C, T, kc, kt = 6, 48, 2, 2
    tt = np.arange(T) * 1.0
    s = np.exp(-tt / 10.0) * (1 - np.exp(-tt / 2.0))
    s /= np.linalg.norm(s)
    ds = np.gradient(s)
    h = np.array([1.0, 0.8, 0.6, 0.5, 0.3, 0.2])
    tauc = np.array([0.0, 0.1, 0.2, -0.1, 0.3, -0.2])
    Sc = 0.5 ** np.abs(np.subtract.outer(np.arange(C), np.arange(C)))
    St = 0.7 ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    Lc, Lt = np.linalg.cholesky(Sc), np.linalg.cholesky(St)
    m = 400
    amp = 4 + 2.0 * rng.standard_normal(m)
    shift = 2.0 * rng.standard_normal(m)
    D1, D2 = np.outer(h, s), np.outer(h * tauc, ds)
    X = amp[:, None, None] * D1[None] + (amp * shift)[:, None, None] * D2[None]
    X = X + 0.03 * np.einsum("ab,nbt,td->nad", Lc, rng.standard_normal((m, C, T)), Lt.T)

    nf = est.fit_nfpa(X, Sc, St, kc, kt, n_restarts=3, n_iter=80)
    im = est.fit_iso_mpca(X, kc, kt, n_restarts=3, n_iter=80, Sigma_c=Sc, Sigma_t=St)
    assert nf.rank == 4 and nf.n_parameters == kc * (C - kc) + kt * (T - kt)
    assert nf.encode(X).shape == (m, kc, kt)
    assert nf.reconstruct(X[0]).shape == (C, T)

    W = est.whiten_separable(np.stack([D1, D2]), Sc, St)
    Wf = np.stack([w.reshape(-1, order="F") for w in W])
    assert est.principal_angles(Wf, nf.basis_whitened().T).max() < 5.0
    assert est.principal_angles(Wf, im.basis_whitened().T).max() < 5.0

    Z = est.whiten_separable(X, Sc, St)
    np.testing.assert_allclose(est.unwhiten_separable(Z, Sc, St), X, atol=1e-8)
