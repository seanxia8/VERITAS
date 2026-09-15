# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Axis identification: reading physical parameters off a linear code.

**New code (not copied from the Paper 1 repository).** Implements the
diagnostic proposed in EXPERIMENT_DESIGN.md §IV.3.

Around a template ``s(t; a, τ, θ)`` the pulse manifold is, to first order,

    x ≈ a·s + a·τ·∂_t s + a·δθ·∂_θ s + n,

so the signal lives in the tangent space ``span{s, ∂_t s, ∂_θ s}`` and any of
the rank-k classes recovers that *span*. The coordinate along ``s`` is the
amplitude ``a`` (linear, identified); the coordinates along the derivative
directions are ``a·τ`` and ``a·δθ`` (bilinear). Whether the *axes* of a fitted
code coincide with those physical directions depends on the class's gauge:
OF has none, CW-PCA orders by population variance, the tied AE is free up to
``O(k)``, NFPA is free up to ``O(k_c) × O(k_t)``.

This module supplies (i) the tangent basis from a template and its
derivatives, (ii) the ``k×k`` mixing matrix between a recovered basis and the
tangent basis in the Σ̂⁻¹ metric, with its condition number and per-direction
residuals — the quantity that says whether physical recognition is possible —
and (iii) the canonical gauge fix that rotates a fitted basis so that code
coordinate ``i`` tracks tangent direction ``i``.

Regime boundary: a time shift is not low-rank. The first-order picture holds
for ``σ_τ · f_max ≪ 1`` (shifts small against the rise time); outside it the
translation orbit needs rank growing with ``σ_τ × bandwidth`` and no rotation
of the code yields a physical axis. Report the mixing-matrix condition number
as a function of the planted ``σ_τ`` to locate that boundary empirically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._metric import as_metric, m_right


def tangent_basis(template: np.ndarray, *, dt: float = 1.0, shape_derivatives: np.ndarray | None = None) -> np.ndarray:
    """Rows ``[s, ∂_t s, (∂_θ s)…]`` for a sampled template ``s``.

    ``∂_t s`` is the central finite difference (``np.gradient``) scaled by
    ``dt`` so that the code coordinate along it is ``a·τ`` in the same time
    unit; ``shape_derivatives`` (rows) are appended unchanged.
    """
    s = np.asarray(template, dtype=np.float64)
    rows = [s, np.gradient(s, dt)]
    if shape_derivatives is not None:
        rows.extend(np.atleast_2d(np.asarray(shape_derivatives, dtype=np.float64)))
    return np.vstack(rows)


@dataclass
class MixingReport:
    """How a recovered basis expresses the physical tangent directions."""

    mixing: np.ndarray                 # (m, k): tangent_i ≈ Σ_j mixing[i, j] · component_j
    residual_fraction: np.ndarray      # (m,): ‖tangent_i − proj‖²_M / ‖tangent_i‖²_M
    condition_number: float            # of the mixing matrix (m == k) — inf if rank-deficient
    principal_angles_deg: np.ndarray   # between span(tangent) and span(components) in M

    @property
    def identifiable(self) -> bool:
        """All tangent directions lie in the recovered span (residual < 1e-2) and the mixing is well-conditioned."""
        return bool(np.all(self.residual_fraction < 1e-2) and np.isfinite(self.condition_number) and self.condition_number < 1e3)


def mixing_matrix(components: np.ndarray, tangent: np.ndarray, weights: np.ndarray | None = None) -> MixingReport:
    """Weighted least-squares expression of each tangent row in the recovered row basis.

    ``components`` (k, d) and ``tangent`` (m, d) are row vectors in data units;
    ``weights`` is the metric ``M`` (vector or matrix, ``None`` = identity).
    """
    P = np.atleast_2d(np.asarray(components, dtype=np.float64))
    Tg = np.atleast_2d(np.asarray(tangent, dtype=np.float64))
    if weights is None:
        MP, MT = P, Tg
    else:
        w = as_metric(weights)
        MP, MT = m_right(P, w), m_right(Tg, w)
    gram = P @ MP.T                                 # (k, k)  P M Pᵀ
    rhs = Tg @ MP.T                                 # (m, k)  T M Pᵀ
    A = np.linalg.solve(gram, rhs.T).T              # (m, k)
    proj = A @ P
    num = np.einsum("ij,ij->i", (Tg - proj), m_right(Tg - proj, as_metric(weights)) if weights is not None else (Tg - proj))
    den = np.einsum("ij,ij->i", Tg, MT)
    resid = num / np.where(den > 0, den, np.inf)
    cond = float(np.linalg.cond(A)) if A.shape[0] == A.shape[1] else float("nan")
    # principal angles in the metric
    from .cw_pca import principal_angles

    ang = principal_angles(Tg, P, weights=weights, degrees=True)
    return MixingReport(mixing=A, residual_fraction=resid, condition_number=cond, principal_angles_deg=ang)


def align_to_tangent(components: np.ndarray, tangent: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, MixingReport]:
    """Canonical gauge: return a basis of the *same* subspace whose row ``i`` is the
    M-projection of tangent row ``i`` onto the recovered span.

    Requires ``m == k``. The returned rows are not M-orthonormal in general
    (the tangent directions are not orthogonal), so encode with the weighted
    LS projection (:meth:`SubspaceFit.encode`), which then returns
    ``z ≈ (a, a·τ, a·δθ, …)`` directly.
    """
    report = mixing_matrix(components, tangent, weights)
    P = np.atleast_2d(np.asarray(components, dtype=np.float64))
    if report.mixing.shape[0] != P.shape[0]:
        raise ValueError("align_to_tangent needs as many tangent directions as components")
    aligned = report.mixing @ P
    return aligned, report
