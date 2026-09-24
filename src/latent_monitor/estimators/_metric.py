# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Shared metric helpers for the linear estimator classes.

Every estimator in this subpackage is the same Σ⁻¹-orthogonal projection
restricted to a different admissible class, so they share one convention for
the metric ``M = Σ̂⁻¹``:

* a 1-D array → diagonal metric (inverse-PSD weights in the rFFT domain, or a
  per-feature inverse variance);
* a 2-D SPD array → full inverse covariance.

Provenance: consolidated from ``noise_geometry.autoencoders.trained`` and
``noise_geometry.subspace.pca`` in the Paper 1 experiment repository
(``noise-weighted-subspace-reconstruction`` @ ``ea076ff``, 2026-09-10).
"""

from __future__ import annotations

import numpy as np


def as_metric(weights: np.ndarray) -> np.ndarray:
    """Validate and return the metric as float64 (vector or square matrix)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim not in (1, 2):
        raise ValueError("weights must be a vector or square matrix")
    if w.ndim == 2 and w.shape[0] != w.shape[1]:
        raise ValueError("matrix weights must be square")
    return w


def metric_dim(w: np.ndarray) -> int:
    return int(w.shape[0])


def m_right(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply the metric on the feature (last) axis of row-wise data: ``A @ M``."""
    return A * w[None, :] if w.ndim == 1 else A @ w


def m_left(B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply the metric on the feature (first) axis of a basis: ``M @ B``."""
    return B * w[:, None] if w.ndim == 1 else w @ B


def metric_sqrt(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M^{1/2}, M^{-1/2})``.

    For a diagonal metric both are returned as 1-D arrays; for a full metric
    as symmetric matrices (the *symmetric* root, not a Cholesky factor — the
    two differ by an orthogonal factor and only the symmetric root inverts the
    whitening used everywhere in this package).
    """
    if w.ndim == 1:
        sqrt_w = np.sqrt(np.clip(w, 0.0, None))
        inv = np.zeros_like(sqrt_w)
        mask = sqrt_w > 0
        inv[mask] = 1.0 / sqrt_w[mask]
        return sqrt_w, inv
    vals, vecs = np.linalg.eigh(0.5 * (w + w.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    msqrt = (vecs * np.sqrt(vals)[None, :]) @ vecs.T
    msqrt_inv = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    return msqrt, msqrt_inv


def whiten_rows(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Map row-wise data to whitened coordinates: ``x ↦ M^{1/2} x``."""
    msqrt, _ = metric_sqrt(w)
    return X * msqrt[None, :] if w.ndim == 1 else X @ msqrt


def unwhiten_rows(Xw: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Inverse of :func:`whiten_rows`."""
    _, msqrt_inv = metric_sqrt(w)
    return Xw * msqrt_inv[None, :] if w.ndim == 1 else Xw @ msqrt_inv


def whiten_basis_columns(D: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Map basis *columns* (d, k) from data units to whitened coordinates."""
    msqrt, _ = metric_sqrt(w)
    return msqrt[:, None] * D if w.ndim == 1 else msqrt @ D


def unwhiten_basis_columns(Dw: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Map basis columns from whitened coordinates back to data units."""
    _, msqrt_inv = metric_sqrt(w)
    return msqrt_inv[:, None] * Dw if w.ndim == 1 else msqrt_inv @ Dw


def sym_sqrt_inv(A: np.ndarray, floor: float | None = None) -> np.ndarray:
    """Symmetric ``A^{-1/2}`` of an SPD matrix via eigendecomposition."""
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    vals = np.clip(vals, np.finfo(float).eps if floor is None else floor, None)
    return (vecs / np.sqrt(vals)[None, :]) @ vecs.T


def sym_sqrt(A: np.ndarray, floor: float | None = None) -> np.ndarray:
    """Symmetric ``A^{1/2}`` — the exact inverse of :func:`sym_sqrt_inv`."""
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    vals = np.clip(vals, np.finfo(float).eps if floor is None else floor, None)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.T
