# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Frozen-propagator hypergraph subject — candidate third subject (D10, design and interface only).

*Design text: ``EXPERIMENT_DESIGN.md`` §I.6 and ``TESTBEDS.md`` §2. This module
implements the ``Subject`` interface and its Laplacians; it is **not trained**
and produces no result.*

The construction follows Bal et al. 2026 (arXiv:2605.03063v2) §6.5 as bounded by
``docs/ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`` §2.4:

* **vertices** = channels (TES/QP channels on arm B; band-frames on TIDMAD);
* **P2** from the measured noise covariance — the noise-Laplacian positional
  encoding of the companion paper (cited here by the name *Prop. stationary-pe*
  only); adjacency is the partial-correlation (precision) matrix, normalised as
  the symmetric Laplacian ``L = I - D^{-1/2} A D^{-1/2}`` (Eq. 4.3);
* **P3** from the measured third noise cumulant: the sign of each hyperedge is
  carried as an **edge attribute**, and ``|w|`` enters the order-3 Laplacian;
* **P2 and P3 are frozen from the reference cell**; only a small readout is
  trainable. The optional trainable correction is QUIVER's zero-initialised
  residual multiplicative gate (arXiv:2606.02785 Eq. 8: ``x → (1 + alpha *
  Theta) x``, ``alpha = 0`` at initialisation) on top of the tied linear AE, so
  the subject is exactly the tied linear AE at step 0.

**Non-triviality.** For Gaussian noise the connected third cumulant vanishes, so
P3 vanishes identically and the subject reduces to the order-2 (covariance)
graph. It is therefore non-trivial only on non-Gaussian cells — exactly where
the study's whitened-residual cumulant statistic is non-trivial. The two
integrations are one mechanism seen twice.

**Chart rule.** This is a *nonlinear* subject candidate: it does not declare
``is_linear`` and the pooled latent is not a natural chart, so the third-order
statistics computed on it are deviations from the reference cell's third moment,
never called cumulants (``ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`` §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .cumulant import third_central_moment
from .subject import Geometry


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric normalised graph Laplacian ``L = I - D^{-1/2} A D^{-1/2}`` (Eq. 4.3).

    Diagonal of the adjacency is dropped; ``D`` is the row-sum degree. The result
    is symmetric, positive semi-definite, has unit diagonal and eigenvalues in
    ``[0, 2]``.
    """
    A = np.asarray(adjacency, dtype=float)
    A = np.abs(0.5 * (A + A.T))                       # |partial correlation| keeps the normalised Laplacian PSD
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    d_inv = np.zeros_like(deg)
    pos = deg > 0
    d_inv[pos] = deg[pos] ** -0.5
    return np.eye(A.shape[0]) - d_inv[:, None] * A * d_inv[None, :]


def order3_pair_weights(T: np.ndarray) -> np.ndarray:
    """Project the order-3 cumulant onto pairs with ``|w|``: ``Theta_ij = sum_k |T_ijk|``.

    Symmetric, zero diagonal. This is the object whose normalisation the order-3
    Laplacian uses; the sign of each triple is *not* here — it is an edge
    attribute returned by :func:`edge_signs`.
    """
    W = np.abs(np.asarray(T, dtype=float))
    Theta = W.sum(axis=2)
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta


def normalized_laplacian_order3(T: np.ndarray) -> np.ndarray:
    """Normalised order-3 Laplacian ``I - D_v^{-1/2} Theta D_v^{-1/2}`` with ``|w|`` weights (Eq. 4.3).

    Symmetric, positive semi-definite, unit diagonal. This is the normalised
    companion of :meth:`FrozenPropagators.p3`; the propagator the subject uses is
    the combinatorial form, which scales with the cumulant and so vanishes on the
    Gaussian reference.
    """
    Theta = order3_pair_weights(T)
    deg = Theta.sum(axis=1)
    d_inv = np.zeros_like(deg)
    pos = deg > 0
    d_inv[pos] = deg[pos] ** -0.5
    return np.eye(Theta.shape[0]) - d_inv[:, None] * Theta * d_inv[None, :]


def edge_signs(T: np.ndarray) -> np.ndarray:
    """Sign attributes of the order-3 hyperedges, ``sign(T_ijk)`` (0 for |T| below the machine floor)."""
    T = np.asarray(T, dtype=float)
    return np.sign(T)


@dataclass
class FrozenPropagators:
    """Order-2 and order-3 propagators measured on the reference-cell noise and then frozen."""

    channel_cov: np.ndarray          # (C, C) measured noise covariance, unit mean diagonal
    third_cumulant: np.ndarray       # (C, C, C) measured connected third cumulant of the channel means
    n_records: int = 0
    floor: float = 1e-12

    @classmethod
    def fit(cls, noise: np.ndarray, *, floor: float = 1e-12) -> "FrozenPropagators":
        """Measure (Sigma_c, T3) from noise-only records ``(m, C, N)``.

        ``Sigma_c`` is the time-averaged channel covariance, normalised to unit
        mean diagonal. ``T3`` is the connected third cumulant of the per-record
        channel means — a measured, noise-only quantity. Fitting uses no signal,
        no truth and no evaluation label.
        """
        R = np.asarray(noise, dtype=float)
        if R.ndim != 3:
            raise ValueError("noise must be (n_records, C, N)")
        m, C, _ = R.shape
        samples = R.transpose(1, 0, 2).reshape(C, -1).T          # (m*N, C): one vector per time sample
        cov = np.cov(samples.T)
        cov = cov / max(float(np.mean(np.diag(cov))), floor)
        T = third_central_moment(samples - samples.mean(axis=0))
        return cls(channel_cov=cov, third_cumulant=T, n_records=int(m), floor=floor)

    def precision(self) -> np.ndarray:
        """Partial-correlation adjacency: ``pinv(Sigma_c)`` with the diagonal kept off the graph."""
        P = np.linalg.pinv(self.channel_cov + self.floor * np.eye(self.channel_cov.shape[0]))
        return 0.5 * (P + P.T)

    def p2(self) -> np.ndarray:
        """Order-2 propagator: the noise-Laplacian PE from the measured covariance."""
        return normalized_laplacian(self.precision())

    def p3(self) -> np.ndarray:
        """Order-3 combinatorial Laplacian ``D_v - Theta`` with ``|w|``; zero identically on a Gaussian reference.

        The sign of each triple is *not* here — it is an edge attribute
        (:meth:`signs`). Because the Laplacian is linear in the cumulant, it
        vanishes when the third cumulant does; the normalised Laplacian
        (:func:`normalized_laplacian_order3`) is provided for the Eq. 4.3
        normalisation test.
        """
        Theta = order3_pair_weights(self.third_cumulant)
        return np.diag(Theta.sum(axis=1)) - Theta

    def signs(self) -> np.ndarray:
        """Sign attributes of the order-3 hyperedges (carried separately from the Laplacian)."""
        return edge_signs(self.third_cumulant)

    def to_dict(self) -> dict[str, Any]:
        return {"n_channels": int(self.channel_cov.shape[0]), "n_records": self.n_records,
                "edge_signs": "carried as an edge attribute (sign(T3)); |w| enters the Laplacian",
                "gaussian_note": "P3 vanishes identically for Gaussian noise, so the subject is non-trivial only on non-Gaussian cells"}


class HypergraphSubject:
    """A ``Subject``-conforming readout on frozen cumulant propagators. **Untrained.**

    The forward pass is the tied linear AE with the frozen propagators applied to
    the whitened traces and an optional QUIVER zero-initialised residual gate:

        x̃  = (I + alpha * (P2 + P3)) · W x        (channel mixing; alpha = 0 at init)
        h_c = A x̃_c
        z   = pool_c(h_c)
        x̂   = D z                                 (tied if D = Aᵀ)
        y   = O z + o0

    Only ``O, o0`` (the readout) are declared trainable; in this pass they are
    left at their initial values and no training is run. ``is_linear`` is
    deliberately **not** set: this is the candidate *nonlinear* subject, so the
    designed controls refuse it and its latent third moments read as deviations.
    """

    def __init__(self, propagators: FrozenPropagators, whitener: Any, A: np.ndarray, D: np.ndarray,
                 O: np.ndarray, o0: np.ndarray, token_embed: np.ndarray, *,
                 pool_weights: np.ndarray | None = None, gate_alpha: float = 0.0,
                 target_names: tuple[str, ...] = (), meta: dict | None = None):
        self.propagators = propagators
        self._sigma_hat = whitener
        self.A = np.asarray(A, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.O = np.asarray(O, dtype=float)
        self.o0 = np.asarray(o0, dtype=float)
        self.E = np.asarray(token_embed, dtype=float)
        self._pool_weights = None if pool_weights is None else np.asarray(pool_weights, dtype=float)
        self.gate_alpha = float(gate_alpha)
        self.target_names = tuple(target_names)
        self.meta = dict(meta or {})
        #: this candidate nonlinear subject does not declare the tied-linear construction
        self.is_linear = False

    # ----------------------------------------------------------------- protocol
    @property
    def latent_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def sigma_hat(self) -> Any:
        return self._sigma_hat

    def with_sigma_hat(self, sigma_hat: Any) -> "HypergraphSubject":
        return HypergraphSubject(self.propagators, sigma_hat, self.A, self.D, self.O, self.o0, self.E,
                                 pool_weights=self._pool_weights, gate_alpha=self.gate_alpha,
                                 target_names=self.target_names, meta=self.meta)

    def _mixer(self, C: int) -> np.ndarray:
        if C != self.propagators.channel_cov.shape[0]:
            raise ValueError(f"subject frozen on {self.propagators.channel_cov.shape[0]} channels, got {C}")
        return np.eye(C) + self.gate_alpha * (self.propagators.p2() + self.propagators.p3())

    def _pool(self, h: np.ndarray) -> np.ndarray:
        C = h.shape[1]
        w = np.full(C, 1.0 / C) if self._pool_weights is None else self._pool_weights / self._pool_weights.sum()
        return np.einsum("c,bck->bk", w, h)

    def _embed(self, geometry: Geometry) -> np.ndarray:
        scale = float(np.max(np.abs(geometry.positions))) or 1.0
        return (geometry.positions / scale) @ self.E.T

    def represent(self, X: np.ndarray, geometry: Geometry) -> Mapping[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 2
        if squeeze:
            X = X[None]
        xw = self.sigma_hat.whiten(X)                                   # (n, C, N)
        m = self._mixer(xw.shape[1])
        xw = np.einsum("ij,bjn->bin", m, xw)
        h = np.einsum("kn,bcn->bck", self.A, xw)
        e = self._embed(geometry)
        tok = np.concatenate([h, np.broadcast_to(e, (h.shape[0],) + e.shape)], axis=-1)
        z = self._pool(h)
        pre = z @ self.O.T
        y = pre + self.o0
        out = {"whitened": xw, "channel": h, "token": tok, "z": z, "pre_output": pre, "output": y}
        if squeeze:
            out = {k: v[0] for k, v in out.items()}
        return out

    def outputs(self, X: np.ndarray, geometry: Geometry) -> np.ndarray:
        return self.represent(X, geometry)["output"]

    def decode(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        C = geometry.n_channels
        xw_hat = z @ self.D.T
        return np.broadcast_to(xw_hat[:, None, :], (xw_hat.shape[0], C, xw_hat.shape[1])).copy()

    def jac_recon(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        """Whitened ``∂g/∂z`` stacked over channels; constant for a tied decoder."""
        return np.tile(self.D, (geometry.n_channels, 1))

    def jac_output(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        return self.O.copy()

    def noise_variance_scale(self, reference: Geometry, geometry: Geometry) -> float:
        return 1.0

    def forward_from_stage(self, stage: str, value: np.ndarray, X: np.ndarray, geometry: Geometry) -> Mapping[str, np.ndarray]:
        rep = self.represent(X, geometry)
        if stage not in rep:
            raise ValueError(f"unknown stage {stage!r}")
        return {**rep, stage: np.asarray(value, dtype=float)}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "hypergraph_subject", "latent_dim": self.latent_dim, "gate_alpha": self.gate_alpha,
                "tied": bool(np.allclose(self.D, self.A.T)), "trainable": "readout only (O, o0)",
                "nonlinear": "is_linear is not declared; designed controls refuse it",
                **self.propagators.to_dict()}
