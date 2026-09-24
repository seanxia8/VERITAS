# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Class 3 — the tied noise-aware linear autoencoder: class 2 as an encoder–decoder.

One linear layer each way, no nonlinearity, no bias beyond the fitted mean:

    decoder  x̂ = D z,        D ∈ ℝ^{d×k}
    encoder  z = Dᵀ M x,     M = Σ̂⁻¹   (the Σ̂⁻¹-adjoint, not the transpose)
    loss     L(D) = (1/N) Σ_n r_nᵀ M r_n,   r_n = x_n − D Dᵀ M x_n.

Its minimiser is the CW-PCA / EMPCA subspace (Paper 1, Bridge Theorem), which
is why the module ships both the closed form (:func:`tied_linear_ae_closed_form`)
and two *independently trained* paths — full-batch L-BFGS-B in float64 by
default, Adam as the first-order evidence — so that the S3 claim "a weighted
tied linear AE recovers the EMPCA subspace" is verified by optimisation, not by
construction. Convergence is judged against the EMPCA optimum ``L*`` via the
optimality gap, the principal angle to :func:`fit_weighted_pca`, and the
M-orthonormality error ``‖Dᵀ M D − I‖``.

Gauge: the loss is invariant under ``D → D Q`` for ``Q ∈ O(k)``, so a trained
``D`` is an arbitrary rotation of the EMPCA basis. Individual code coordinates
carry no physical meaning until the gauge is fixed (:mod:`.identification`);
the subspace, the Jacobians and the reference-cell projectors are gauge-free.

Provenance: ``src/noise_geometry/autoencoders/trained.py`` and
``autoencoders/linear.py`` in the Paper 1 experiment repository
(``noise-weighted-subspace-reconstruction`` @ ``ea076ff``, 2026-09-10). The
``.npz`` persistence and the ``run_s3_bridge`` experiment driver were not
copied; the untied ablation (``autoencoders/untied.py``, P1-E12) is not part of
this class and stays in the source repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ._metric import as_metric, m_left, m_right, metric_sqrt
from .cw_pca import SubspaceFit, fit_pca, fit_weighted_pca, principal_angles, project_onto_basis


# --------------------------------------------------------------------------- #
# Closed form
# --------------------------------------------------------------------------- #
def tied_linear_ae_closed_form(samples: np.ndarray, rank: int, *, weights: np.ndarray | None = None, center: bool = True) -> SubspaceFit:
    """Return the tied linear AE subspace for MSE (``weights=None``) or weighted loss."""
    if weights is None:
        return fit_pca(samples, rank, center=center)
    return fit_weighted_pca(samples, weights, rank, center=center)


# --------------------------------------------------------------------------- #
# Loss, gradient, optimisers
# --------------------------------------------------------------------------- #
def _loss_and_grad(D: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted reconstruction loss and its gradient w.r.t. the basis ``D``.

    ``X`` rows are centred observations. ``D`` has shape ``(d, k)``.
    """
    n = X.shape[0]
    MX = m_right(X, w)               # rows: (M x_n)^T
    Z = MX @ D                       # rows: z_n^T = x_n^T M D
    Xhat = Z @ D.T                   # rows: x_hat_n^T = (D D^T M x_n)^T
    R = X - Xhat
    MR = m_right(R, w)               # rows: (M r_n)^T
    loss = float(np.sum(MR * R) / n)
    grad = (-2.0 / n) * (MR.T @ Z + MX.T @ (MR @ D))
    return loss, grad


def _lbfgs(X: np.ndarray, w: np.ndarray, D0: np.ndarray, max_iter: int) -> tuple[np.ndarray, int]:
    d, k = D0.shape

    def fun(vec: np.ndarray) -> tuple[float, np.ndarray]:
        loss, grad = _loss_and_grad(vec.reshape(d, k), X, w)
        return loss, grad.ravel()

    res = minimize(
        fun,
        D0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "maxfun": max_iter * 10, "ftol": 1e-16, "gtol": 1e-12},
    )
    return res.x.reshape(d, k), int(res.nit)


def _adam(X: np.ndarray, w: np.ndarray, D0: np.ndarray, max_iter: int, lr: float, tol: float) -> tuple[np.ndarray, int, list[float]]:
    D = D0.copy()
    m = np.zeros_like(D)
    v = np.zeros_like(D)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []
    prev = np.inf
    nit = 0
    for t in range(1, max_iter + 1):
        loss, grad = _loss_and_grad(D, X, w)
        history.append(loss)
        nit = t
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        D = D - lr * mhat / (np.sqrt(vhat) + eps)
        if abs(prev - loss) < tol * max(1.0, abs(prev)):
            break
        prev = loss
    return D, nit, history


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def weighted_reconstruction_loss(samples: np.ndarray, components: np.ndarray, weights: np.ndarray, mean: np.ndarray) -> float:
    """Weighted loss of the *optimal* (weighted-LS) reconstruction onto a basis."""
    w = as_metric(weights)
    recon = project_onto_basis(samples, components, weights=w, mean=mean)
    r = np.asarray(samples, dtype=np.float64) - recon
    return float(np.sum(m_right(r, w) * r) / samples.shape[0])


def empca_optimal_loss(samples: np.ndarray, weights: np.ndarray, rank: int) -> tuple[float, SubspaceFit]:
    """Return ``(L*, empca_fit)`` for the weighted rank-``k`` subspace."""
    empca = fit_weighted_pca(samples, weights, rank)
    lstar = weighted_reconstruction_loss(samples, empca.components, weights, empca.mean)
    return lstar, empca


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TrainedAEResult:
    """Outcome of a trained tied linear AE and its bridge diagnostics."""

    method: str                       # "direct" or "whitened"
    optimizer: str                    # "lbfgs" or "adam"
    components: np.ndarray            # (k, d) row vectors in data units
    decoder: np.ndarray               # D (d, k) in data units
    mean: np.ndarray
    weights: np.ndarray
    final_loss: float
    optimal_loss: float               # L* from EMPCA
    optimality_gap: float             # final_loss - optimal_loss
    relative_gap: float               # optimality_gap / |optimal_loss|
    max_principal_angle_deg: float    # vs fit_weighted_pca, in the M-metric
    m_orthonormality_error: float     # ||D^T M D - I||_max in data units
    n_iter: int
    loss_history: np.ndarray | None = field(default=None, repr=False)

    @property
    def rank(self) -> int:
        return int(self.decoder.shape[1])

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """The tied encoder ``z = Dᵀ M (x − μ)`` (no Gram inverse — D is M-orthonormal at optimum)."""
        X = np.atleast_2d(np.asarray(samples, dtype=np.float64)) - self.mean[None, :]
        z = m_right(X, self.weights) @ self.decoder
        return z[0] if np.asarray(samples).ndim == 1 else z

    def reconstruct(self, samples: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(self.encode(samples))
        out = z @ self.decoder.T + self.mean[None, :]
        return out[0] if np.asarray(samples).ndim == 1 else out

    def as_subspace(self) -> SubspaceFit:
        return SubspaceFit(self.components, self.mean, np.full(self.rank, np.nan), self.weights)


# --------------------------------------------------------------------------- #
# Public training entry points
# --------------------------------------------------------------------------- #
def _init_basis(Xc: np.ndarray, w: np.ndarray, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Random (NOT EMPCA-seeded) M-orthonormal initial basis for conditioning."""
    d = Xc.shape[1]
    D = rng.standard_normal((d, rank))
    msqrt, msqrt_inv = metric_sqrt(w)
    Dw = (msqrt[:, None] * D) if w.ndim == 1 else (msqrt @ D)
    Q, _ = np.linalg.qr(Dw)
    return (msqrt_inv[:, None] * Q) if w.ndim == 1 else (msqrt_inv @ Q)


def _finalise(method, optimizer, components, D_data, mean, weights, samples, rank, n_iter, history) -> TrainedAEResult:
    w = as_metric(weights)
    final_loss = weighted_reconstruction_loss(samples, components, w, mean)
    lstar, empca = empca_optimal_loss(samples, w, rank)
    gap = final_loss - lstar
    angle = float(np.max(principal_angles(empca.components, components, weights=w)))
    gram = D_data.T @ m_left(D_data, w)
    ortho_err = float(np.max(np.abs(gram - np.eye(rank))))
    return TrainedAEResult(
        method=method,
        optimizer=optimizer,
        components=components,
        decoder=D_data,
        mean=mean,
        weights=w,
        final_loss=final_loss,
        optimal_loss=lstar,
        optimality_gap=gap,
        relative_gap=gap / max(abs(lstar), np.finfo(float).tiny),
        max_principal_angle_deg=angle,
        m_orthonormality_error=ortho_err,
        n_iter=n_iter,
        loss_history=np.asarray(history) if history is not None else None,
    )


def train_weighted_linear_ae(
    samples: np.ndarray,
    weights: np.ndarray,
    rank: int,
    *,
    optimizer: str = "lbfgs",
    center: bool = True,
    seed: int = 0,
    max_iter: int = 5000,
    adam_lr: float = 5e-2,
    adam_tol: float = 1e-14,
) -> TrainedAEResult:
    """Method (A): train the tied AE directly in data coordinates (weighted loss).

    This is the literal Paper 1 claim: the S3 synthetic runs converge in
    72–77 L-BFGS iterations to the EMPCA subspace.
    """
    X = np.asarray(samples, dtype=np.float64)
    w = as_metric(weights)
    if X.shape[1] != w.shape[0]:
        raise ValueError("weights dimension must match feature dimension")
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean[None, :]
    rng = np.random.default_rng(seed)
    D0 = _init_basis(Xc, w, int(rank), rng)

    if optimizer == "lbfgs":
        D, nit = _lbfgs(Xc, w, D0, max_iter)
        history = None
    elif optimizer == "adam":
        D, nit, history = _adam(Xc, w, D0, max_iter, adam_lr, adam_tol)
    else:
        raise ValueError("optimizer must be 'lbfgs' or 'adam'")

    return _finalise("direct", optimizer, D.T.copy(), D, mean, w, X, int(rank), nit, history)


def train_whitened_linear_ae(
    samples: np.ndarray,
    weights: np.ndarray,
    rank: int,
    *,
    optimizer: str = "lbfgs",
    center: bool = True,
    seed: int = 0,
    max_iter: int = 5000,
    adam_lr: float = 5e-2,
    adam_tol: float = 1e-14,
) -> TrainedAEResult:
    """Method (B): whiten, train a standard MSE tied AE, map the basis back.

    The optimisation core is reused with an identity metric on whitened data,
    so this is a genuine gradient-trained AE (not a closed form), but in the
    coordinates where Baldi–Hornik guarantees the PCA subspace.
    """
    X = np.asarray(samples, dtype=np.float64)
    w = as_metric(weights)
    if X.shape[1] != w.shape[0]:
        raise ValueError("weights dimension must match feature dimension")
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean[None, :]
    msqrt, msqrt_inv = metric_sqrt(w)
    Xw = (Xc * msqrt[None, :]) if w.ndim == 1 else (Xc @ msqrt)
    ident = np.ones(Xw.shape[1], dtype=np.float64)  # identity metric in whitened space
    rng = np.random.default_rng(seed)
    W0 = _init_basis(Xw, ident, int(rank), rng)

    if optimizer == "lbfgs":
        W, nit = _lbfgs(Xw, ident, W0, max_iter)
        history = None
    elif optimizer == "adam":
        W, nit, history = _adam(Xw, ident, W0, max_iter, adam_lr, adam_tol)
    else:
        raise ValueError("optimizer must be 'lbfgs' or 'adam'")

    # Map basis columns from whitened space back to data units: D = M^{-1/2} W.
    D_data = (msqrt_inv[:, None] * W) if w.ndim == 1 else (msqrt_inv @ W)
    return _finalise("whitened", optimizer, D_data.T.copy(), D_data, mean, w, X, int(rank), nit, history)
