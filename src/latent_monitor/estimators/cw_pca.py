# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Class 2 — covariance-weighted PCA / EMPCA (CW-PCA): the unrestricted rank-k subspace.

Whiten with Σ̂^{-1/2}, take the top-k right singular vectors of the whitened
data matrix, map back to data units. Reconstruction is the Σ̂⁻¹-orthogonal
projection

    x̂ = μ + P (Pᵀ Σ̂⁻¹ P)⁻¹ Pᵀ Σ̂⁻¹ (x − μ).

Closed form; ``k(d−k)`` Grassmannian parameters. Ordinary Euclidean PCA
(IsoPCA, :func:`fit_pca`) is the same estimator with Σ̂ = I and is always run
as the metric-ablation comparator.

The *subspace* is identified; the *basis inside it* is ordered by population
variance, not by physics, and rotates when the event distribution changes.
Reading physical parameters off the code therefore needs the k×k
re-identification in :mod:`.identification` (EXPERIMENT_DESIGN.md §IV).

Provenance: ``src/noise_geometry/subspace/pca.py`` and
``src/noise_geometry/subspace/angles.py`` in the Paper 1 experiment repository
(``noise-weighted-subspace-reconstruction`` @ ``ea076ff``, 2026-09-10), copied
unchanged apart from the shared metric helpers and :meth:`SubspaceFit.encode`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._metric import as_metric, m_right, metric_sqrt


@dataclass
class SubspaceFit:
    """Container for a fitted linear subspace (components are row vectors, data units)."""

    components: np.ndarray            # (k, d)
    mean: np.ndarray                  # (d,)
    explained_variance: np.ndarray    # (k,)
    weights: np.ndarray | None = None  # the metric the fit was made in (None = identity)

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """Coefficients ``z = (Pᵀ M P)⁻¹ Pᵀ M (x − μ)`` (weighted LS in the fit metric)."""
        X = np.atleast_2d(np.asarray(samples, dtype=np.float64))
        B = self.components
        Xc = X - self.mean[None, :]
        if self.weights is None:
            gram, rhs = B @ B.T, Xc @ B.T
        else:
            w = as_metric(self.weights)
            gram, rhs = B @ m_right(B, w).T, m_right(Xc, w) @ B.T
        z = np.linalg.solve(gram, rhs.T).T
        return z[0] if np.asarray(samples).ndim == 1 else z

    def reconstruct(self, samples: np.ndarray) -> np.ndarray:
        return project_onto_basis(samples, self.components, weights=self.weights, mean=self.mean)


def fit_pca(samples: np.ndarray, rank: int, *, center: bool = True) -> SubspaceFit:
    """Fit an ordinary Euclidean PCA basis with row-wise observations (IsoPCA)."""
    X = np.asarray(samples, dtype=np.float64)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean[None, :]
    _, s, vh = np.linalg.svd(Xc, full_matrices=False)
    r = int(rank)
    return SubspaceFit(vh[:r].copy(), mean, (s[:r] ** 2) / max(X.shape[0] - 1, 1), None)


def fit_weighted_pca(samples: np.ndarray, weights: np.ndarray, rank: int, *, center: bool = True) -> SubspaceFit:
    """Fit PCA in whitened coordinates and map components back to data units (CW-PCA / EMPCA).

    ``weights`` may be a diagonal weight vector or a full positive-definite
    inverse covariance matrix.
    """
    X = np.asarray(samples, dtype=np.float64)
    w = as_metric(weights)
    if X.shape[1] != w.shape[0]:
        raise ValueError("weights length must match feature dimension")
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean[None, :]
    if w.ndim == 1:
        sqrt_w = np.sqrt(np.clip(w, 0.0, None))
        Xw = Xc * sqrt_w[None, :]
        inverse_transform = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        mask = sqrt_w > 0
        inverse_transform[mask, mask] = 1.0 / sqrt_w[mask]
    else:
        vals, vecs = np.linalg.eigh(0.5 * (w + w.T))
        vals = np.clip(vals, np.finfo(float).eps, None)
        transform = vecs * np.sqrt(vals)[None, :]
        inverse_transform = (vecs * (1.0 / np.sqrt(vals))[None, :]).T
        Xw = Xc @ transform
    _, s, vh = np.linalg.svd(Xw, full_matrices=False)
    r = int(rank)
    components = vh[:r] @ inverse_transform
    return SubspaceFit(components, mean, (s[:r] ** 2) / max(X.shape[0] - 1, 1), w)


def project_onto_basis(samples: np.ndarray, basis: np.ndarray, *, weights: np.ndarray | None = None, mean=None) -> np.ndarray:
    """Reconstruct row-wise samples from a basis using LS or weighted LS."""
    X = np.asarray(samples, dtype=np.float64)
    single = X.ndim == 1
    X = np.atleast_2d(X)
    B = np.asarray(basis, dtype=np.float64)
    mu = np.zeros(X.shape[1], dtype=np.float64) if mean is None else np.asarray(mean, dtype=np.float64)
    Xc = X - mu[None, :]
    if weights is None:
        gram = B @ B.T
        rhs = Xc @ B.T
    else:
        w = as_metric(weights)
        if w.ndim == 1:
            gram = (B * w[None, :]) @ B.T
            rhs = (Xc * w[None, :]) @ B.T
        else:
            gram = B @ w @ B.T
            rhs = Xc @ w @ B.T
    coeff = np.linalg.solve(gram, rhs.T).T
    out = coeff @ B + mu[None, :]
    return out[0] if single else out


# --------------------------------------------------------------------------- #
# Subspace angles (from subspace/angles.py)
# --------------------------------------------------------------------------- #
def _orthonormal_rows(basis: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    B = np.asarray(basis, dtype=np.float64)
    if weights is not None:
        w = as_metric(weights)
        if w.ndim == 1:
            B = B * np.sqrt(np.clip(w, 0.0, None))[None, :]
        else:
            sqrt_metric, _ = metric_sqrt(w)
            B = B @ sqrt_metric
    q, _ = np.linalg.qr(B.T)
    return q


def principal_angles(a: np.ndarray, b: np.ndarray, *, weights: np.ndarray | None = None, degrees: bool = True) -> np.ndarray:
    """Principal angles between row-span subspaces, optionally in the metric ``weights``."""
    Qa = _orthonormal_rows(a, weights)
    Qb = _orthonormal_rows(b, weights)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    angles = np.arccos(sv)
    return np.degrees(angles) if degrees else angles
