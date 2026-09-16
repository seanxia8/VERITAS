# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""A training-support estimator, distinct from resolvability (TWO_CLAIM_REVISION_PLAN §3 M2).

``ReferenceCell.P_resolved`` says which representation directions the
*measurement* resolves at the reference point. Whether an event lies where the
*training distribution* put mass is a different question, answered here by a
density-style score on the reference-fit representation:

    s(r) = max( Mahalanobis(r; μ_ref, Σ_ref + shrinkage) / q_ref ,  d_kNN(r) / q_kNN )

— the larger of a parametric and a non-parametric novelty, each standardised by
its own clean-reference quantile so the two are on one scale. Larger = further
from the excited support. It is fitted on reference-fit windows only and
**validated before use** on constructed controls: held-out clean reference
windows (in support) against far-displaced windows (out of support). If the
control AUROC is below the declared floor the estimator is not used for
abstention and the report says so.

This score is what the abstention rule (``protocol.abstention``) consumes. It
is *not* a proof that an event is physically outside training support — the
6 Sep table already showed valid rare physics inside the resolved span
(``S_in_span``) — and it does not identify the cause of an unknown shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def ledoit_wolf_shrinkage(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf shrinkage of the sample covariance towards a scaled identity (fitted on ``X`` only).

    Returns ``(Σ_shrunk, α)`` with Σ_shrunk = (1−α) S + α μ I, μ = tr(S)/d. The
    analytic α (Ledoit & Wolf 2004, "A well-conditioned estimator for
    large-dimensional covariance matrices") keeps the estimate invertible when d
    is comparable to n.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 2:
        raise ValueError("need at least two samples")
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    mu = float(np.trace(S)) / d
    delta2 = float(np.sum((S - mu * np.eye(d)) ** 2)) / d
    beta2 = 0.0
    for i in range(n):
        xi = Xc[i][:, None]
        beta2 += float(np.sum((xi @ xi.T - S) ** 2))
    beta2 = beta2 / (n * n * d)
    beta2 = min(beta2, delta2)
    alpha = 0.0 if delta2 <= 0 else float(beta2 / delta2)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * S + alpha * mu * np.eye(d), alpha


@dataclass
class SupportEstimator:
    mean: np.ndarray
    inv_cov: np.ndarray
    alpha: float
    ref_points: np.ndarray             # (n_ref, d) reference-fit representations (for kNN)
    k: int
    q_maha: float                      # clean-reference scale of the Mahalanobis term
    q_knn: float                       # clean-reference scale of the kNN term
    validation: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, R_ref: np.ndarray, k: int = 5, quantile: float = 0.9) -> "SupportEstimator":
        R = np.asarray(R_ref, dtype=float)
        n, d = R.shape
        if n < max(2 * k + 2, d + 2):
            raise ValueError(f"need more reference points than max(2k+2, d+2)={max(2*k+2, d+2)}, got {n}")
        cov, alpha = ledoit_wolf_shrinkage(R)
        inv = np.linalg.inv(cov + 1e-12 * np.eye(d))
        est = cls(mean=R.mean(axis=0), inv_cov=inv, alpha=alpha, ref_points=R, k=int(k), q_maha=1.0, q_knn=1.0)
        # leave-one-out scales on the reference itself so a reference window scores ≈ 1
        m = est._maha(R)
        kn = est._knn(R, exclude_self=True)
        est.q_maha = float(np.quantile(m, quantile)) or 1.0
        est.q_knn = float(np.quantile(kn, quantile)) or 1.0
        return est

    def _maha(self, R: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(R) - self.mean
        return np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", d, self.inv_cov, d), 0.0))

    def _knn(self, R: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        R = np.atleast_2d(R)
        D = np.sqrt(np.maximum(np.sum(R**2, axis=1)[:, None] + np.sum(self.ref_points**2, axis=1)[None, :]
                               - 2.0 * R @ self.ref_points.T, 0.0))
        if exclude_self:
            D = D + np.where(D < 1e-12, np.inf, 0.0)
        k = min(self.k, D.shape[1] - (1 if exclude_self else 0))
        part = np.partition(D, k - 1, axis=1)[:, :k]
        return part.mean(axis=1)

    def score(self, R: np.ndarray) -> np.ndarray:
        """Support novelty: max of the two standardised terms; ≈ 1 at the clean reference quantile."""
        return np.maximum(self._maha(R) / self.q_maha, self._knn(R) / self.q_knn)

    def components(self, R: np.ndarray) -> dict[str, np.ndarray]:
        return {"mahalanobis": self._maha(R) / self.q_maha, "knn": self._knn(R) / self.q_knn}


def validate_support_estimator(est: SupportEstimator, R_in: np.ndarray, R_out: np.ndarray, floor: float = 0.9) -> dict:
    """Constructed control: held-out in-support windows against out-of-support windows.

    ``R_in`` are clean reference-family windows *not* used in ``fit``; ``R_out``
    are constructed far-from-support representations (e.g. the reference
    representations displaced by several reference standard deviations in a
    random direction, or an undeclared family known by construction to be off
    support). Returns the AUROC, whether it clears the declared floor, and the
    verdict the abstention rule must obey.
    """
    from .protocol.consequence import weighted_auroc
    s_in, s_out = est.score(R_in), est.score(R_out)
    r = weighted_auroc(np.concatenate([s_in, s_out]), np.concatenate([np.zeros(len(s_in), bool), np.ones(len(s_out), bool)]))
    ok = (not np.isnan(r["auroc"])) and r["auroc"] >= floor
    est.validation = {"auroc_in_vs_out": r["auroc"], "n_in": int(len(s_in)), "n_out": int(len(s_out)), "floor": floor,
                      "usable_for_abstention": bool(ok), "shrinkage_alpha": est.alpha}
    return est.validation


def displaced_controls(R_in: np.ndarray, est: SupportEstimator, sigmas: float = 6.0, seed: int = 0) -> np.ndarray:
    """Out-of-support control by construction: each window displaced by ``sigmas`` reference std in a random direction."""
    rng = np.random.default_rng(seed)
    R = np.atleast_2d(np.asarray(R_in, dtype=float))
    cov = np.linalg.inv(est.inv_cov)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
    u = rng.normal(size=R.shape); u /= np.linalg.norm(u, axis=1, keepdims=True)
    return R + sigmas * (u @ L.T)
