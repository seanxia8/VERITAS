"""Run-grouped Ridge estimators with independently scaled feature blocks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .lucid_arms import _fold_labels
from .probes import ALPHAS

GAMMAS = (0.01, 0.1, 1.0, 10.0, 100.0, np.inf)
TOLERANCE_M = 0.005


def _normalise(x, fit):
    mean = x[fit].mean(axis=0)
    scale = x[fit].std(axis=0)
    constant = np.all(x[fit] == x[fit][0], axis=0)
    mean[constant] = x[fit][0, constant]
    scale[constant] = 1.0
    return (x - mean) / scale, mean, scale


def _solve(gram, cross, penalty):
    matrix = gram.copy()
    matrix.flat[::len(matrix)+1] += penalty
    return cho_solve(cho_factor(matrix, overwrite_a=True, check_finite=False),
                     cross, check_finite=False)


@dataclass(frozen=True)
class BlockRidge:
    mean: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    coef: np.ndarray
    alpha: float
    gamma: float
    cv_mse: float

    def predict(self, base, support):
        x = np.concatenate([base, support], axis=1)
        return (x-self.mean) / self.scale @ self.coef + self.intercept

    def settings(self):
        return {"alpha_z": self.alpha,
                "gamma_u_over_z": float(self.gamma) if np.isfinite(self.gamma) else "inf",
                "cv_mse": self.cv_mse}


def fit_block_ridge(base, support, vertex, runs, *, alphas=ALPHAS, gammas=GAMMAS):
    """Select (alpha, gamma) on five whole-run folds, keeping gamma=infinity exact."""
    base, support, vertex = [np.asarray(x, dtype=float) for x in (base, support, vertex)]
    if (base.ndim != 2 or support.ndim != 2 or vertex.ndim != 2
            or not len(base) or base.shape[1] == 0 or len(base) != len(support)
            or len(base) != len(vertex)):
        raise ValueError("feature blocks, run IDs and vertices must use the same event rows")
    if not all(np.all(np.isfinite(z)) for z in (base, support, vertex)):
        raise ValueError("Ridge needs finite feature blocks and vertices")
    d_base, d_total = base.shape[1], base.shape[1] + support.shape[1]
    x = np.concatenate([base, support], axis=1)
    labels = _fold_labels(np.asarray(runs))
    if np.inf in gammas and np.all(support == support[0]):
        gammas = (np.inf,)
    candidates = [(float(alpha), float(gamma)) for alpha in alphas for gamma in gammas]
    if len(labels) != len(x) or len(np.unique(labels)) != 5:
        raise ValueError("run-grouped ridge requires five nonempty folds")
    if any(alpha <= 0 or not np.isfinite(alpha) or not gamma > 0 for alpha,gamma in candidates):
        raise ValueError("regularization parameters must be positive")
    risk = {pair: 0.0 for pair in candidates}
    for group in np.unique(labels):
        fit = labels != group
        z, _, _ = _normalise(x, fit)
        y_mean = vertex[fit].mean(axis=0)
        fit_z = z[fit]
        gram = fit_z.T @ fit_z
        cross = fit_z.T @ (vertex[fit] - y_mean)
        for alpha, gamma in candidates:
            d = d_total if np.isfinite(gamma) else d_base
            penalty = np.full(d, alpha)
            penalty[d_base:] *= gamma if np.isfinite(gamma) else 1.0
            coef = _solve(gram[:d, :d], cross[:d], penalty)
            errors = z[~fit, :d] @ coef + y_mean - vertex[~fit]
            risk[alpha, gamma] += float(np.square(errors).sum())
    alpha, gamma = min(risk, key=lambda pair: (risk[pair], -pair[1], -pair[0]))
    z, mean, scale = _normalise(x, np.ones(len(x), dtype=bool))
    y_mean = vertex.mean(axis=0)
    d = d_total if np.isfinite(gamma) else d_base
    z_fit = z[:, :d]
    penalty = np.full(d, alpha)
    penalty[d_base:] *= gamma if np.isfinite(gamma) else 1.0
    coef = np.zeros((d_total, vertex.shape[1]))
    coef[:d] = _solve(z_fit.T @ z_fit, z_fit.T @ (vertex-y_mean), penalty)
    return BlockRidge(mean, scale, y_mean, coef, alpha, gamma,
                      risk[alpha, gamma] / vertex.size)
