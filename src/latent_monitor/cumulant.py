# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Connected-cumulant estimators for the first non-Gaussian rung (Bal et al. 2026).

This is the supporting machinery of the Fisher-cumulant tower integrated in
``docs/ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`` §2.1. It estimates the
connected third cumulant tensor of a set of vectors in a **declared chart** and
reports, per window, (a) the per-coordinate third central moments in a
reference PCA basis (``n_pcs`` coordinates) and (b) the scalar Frobenius norm of
the per-window third-moment tensor minus the reference-cell tensor. The
fourth-order companion is provided but is not wired into any arm.

**What this is not.** ``input_quality_features`` already computes a raw-window
kurtosis (``in_kurtosis``) — the fourth central moment of the raw trace. This
module is a *different object*: it is computed on the **whitened residual** and
on the **pooled hooks in the reference chart**, about the *reference* moments,
and its third-order part is the connected cumulant of an exponential-family
sufficient statistic. A raw-window kurtosis is not a cumulant in a natural
chart and the two must not be conflated.

**Chart caveat (Bal et al. Remark 1, App. A).** The cumulant reading is exact
only in a natural (exponential-family) chart. The whitened residual under the
Gaussian reference is such a chart; the pooled latent ``z`` on the *linear*
subject is too (``z`` is linear in ``x``). On a nonlinear subject the pooled
``z`` is not, and the same number is a *deviation from the reference cell's own
third moment*, reported as such (``reading`` in
:class:`latent_monitor.protocol.features.AlarmTimeReference`). The code does
not decide the reading here; callers pass it and label the result.

**Sampling.** All estimators are ordinary numpy; the full tensor costs
``O(N d**3)`` for ``N`` windows of dimension ``d``. ``d`` is ``n_pcs <= 10`` in
the arrays this study forms, so a reference fit is seconds.
"""

from __future__ import annotations

import numpy as np


def _centered(V: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if V.shape[0] == 0:
        raise ValueError("need at least one vector")
    mu = V.mean(axis=0) if mean is None else np.asarray(mean, dtype=float)
    if mu.shape != (V.shape[1],):
        raise ValueError(f"mean has shape {mu.shape}, expected ({V.shape[1]},)")
    return V - mu


def third_central_moment(V: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Population third central-moment tensor ``(d, d, d)`` of the rows of ``V`` (``(N, d)``)."""
    C = _centered(V, mean)
    return np.einsum("na,nb,nc->abc", C, C, C) / C.shape[0]


def third_cumulant(V: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Connected third cumulant tensor of a set of vectors.

    For data centred on its mean (or on ``mean``) the lower-order Fisher terms
    vanish, so the connected third cumulant equals the third central moment:
    ``kappa3 = m3 - 3 m2*mu1 + 2 mu1**3 = m3`` when ``mu1 = 0``. This alias
    exists so the call site reads as the object it is; the arithmetic is shared.
    """
    return third_central_moment(V, mean)


def fourth_central_moment(V: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Population fourth central-moment tensor ``(d, d, d, d)`` — the companion, not wired into any arm."""
    C = _centered(V, mean)
    return np.einsum("na,nb,nc,nd->abcd", C, C, C, C) / C.shape[0]


def fourth_cumulant(V: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Connected fourth cumulant tensor ``(d, d, d, d)`` (Kronecker/circular convention).

    For centred data ``kappa4 = m4 - (m2(x)m2 + m2(x)m2 + m2(x)m2)`` over the
    three pairings. Provided as the fourth-order companion; **not** consumed by
    any arm in this pass (``ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`` §2.1).
    """
    C = _centered(V, mean)
    n = C.shape[0]
    m2 = np.einsum("na,nb->ab", C, C) / n
    m4 = np.einsum("na,nb,nc,nd->abcd", C, C, C, C) / n
    return m4 - (np.einsum("ab,cd->abcd", m2, m2)
                 + np.einsum("ac,bd->abcd", m2, m2)
                 + np.einsum("ad,bc->abcd", m2, m2))


def pooled_third_tensor(V: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Per-window pooled third central-moment tensor over the channel axis.

    ``V`` is ``(N, C, d)``. Each window's channel mean is removed before the
    tensor is formed, so this is the third central moment *pooled over
    channels*, the order-3 companion of the pooled channel covariance returned
    by :func:`latent_monitor.reference.pooled_stage`. Returns ``(N, d, d, d)``.
    """
    V = np.atleast_3d(np.asarray(V, dtype=float))
    C = V.mean(axis=1, keepdims=True) if mean is None else np.asarray(mean, dtype=float)[None, None, :]
    Z = V - C
    return np.einsum("nsa,nsb,nsc->nabc", Z, Z, Z) / max(V.shape[1], 1)


def per_window_third_tensor(V: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Per-window rank-1 third tensor ``(v_i - mean)**otimes3``, shape ``(N, d, d, d)``."""
    C = np.atleast_2d(np.asarray(V, dtype=float)) - np.asarray(mean, dtype=float)
    return np.einsum("na,nb,nc->nabc", C, C, C)


def project_diagonal_third(tensor: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """Diagonal of the third tensor in a basis ``pcs`` ``(d, k)``: returns ``(..., k)``.

    ``T_proj[j, j, j] = P[:, j]**T T P[:, j]`` contracted three times; that is
    the per-coordinate third central moment in the ``pcs`` chart.
    """
    T = np.asarray(tensor, dtype=float)
    P = np.asarray(pcs, dtype=float)
    return np.einsum("...abc,aj,bj,cj->...j", T, P, P, P)


def tensor_frobenius_deviation(tensor: np.ndarray, reference_tensor: np.ndarray) -> np.ndarray:
    """``||T_i - T_ref||_F`` over the trailing three axes; scalar per leading index."""
    T = np.asarray(tensor, dtype=float)
    R = np.asarray(reference_tensor, dtype=float)
    return np.sqrt(np.sum((T - R) ** 2, axis=(-3, -2, -1)))


def pca_basis(V: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference PCA of the rows of ``V``: returns ``(mean, pcs)`` with ``pcs`` ``(d, min(k, n))``."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    mu = V.mean(axis=0)
    C = V - mu
    if C.shape[0] < 2:
        return mu, np.eye(V.shape[1])[:, :k]
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return mu, Vt[: min(k, Vt.shape[0])].T


def truncation_error_ratio(I2: np.ndarray, I3: np.ndarray, delta: np.ndarray) -> float:
    """Bal et al.'s quadratic-vs-quadratic-plus-cubic KL truncation-error ratio.

    Along a displacement ``delta`` the local KL expansion is
    ``D_KL ~ 1/2 delta^T I2 delta + 1/6 I3(delta, delta, delta) + ...``. The
    ratio of the quadratic term to the quadratic-plus-cubic partial sum is
    ``q / (q + c)`` in ``(0, 1]`` (1 = no cubic content). **Development-table
    quantity only** (``ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`` §2.1); it is
    never an arm feature or an estimand.
    """
    d = np.asarray(delta, dtype=float).ravel()
    I2 = np.asarray(I2, dtype=float)
    I3 = np.asarray(I3, dtype=float)
    q = 0.5 * float(d @ I2 @ d)
    c = (1.0 / 6.0) * float(np.einsum("a,b,c,abc->", d, d, d, I3))
    total = q + c
    return float("nan") if total == 0 else float(q / total)
