# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Class 4 — NFPA, the noise-factored projection autoencoder: the Kronecker-separable restriction.

For a ``C × T`` channel–time matrix ``X`` with separable noise
``Σ̂ = Σ_t ⊗ Σ_c`` (column-major ``vec``):

    whiten          Z̃ = Σ_c^{-1/2} X Σ_t^{-1/2}
    factors         U_c ∈ ℝ^{C×k_c},  U_t ∈ ℝ^{T×k_t}    (orthonormal columns)
    code            Γ = U_cᵀ Z̃ U_t                      (k_c × k_t)
    reconstruction  Ẑ = U_c U_cᵀ Z̃ U_t U_tᵀ,  un-whitened back.

Fitted by alternating Rayleigh–Ritz (covariance-weighted Tucker-2 /
multilinear PCA) with restarts; ``k_c(C−k_c) + k_t(T−k_t)`` parameters versus
``k(CT−k)`` for the unrestricted class. Its isotropic twin **Iso-MPCA** is the
identical algorithm on centred raw traces (Σ̂ = I) and is the metric ablation.
The plan's preferred wording: "a covariance-weighted Tucker-2 projection,
expressed as a tied encoder–decoder and referred to as NFPA."

Physical reading (EXPERIMENT_DESIGN.md §IV): to first order in per-channel
delays, ``X ≈ h sᵀ + (h∘τ)(∂_t s)ᵀ`` is exactly Tucker-2 with
``k_c = k_t = 2`` — ``U_t ⊃ {s, ∂_t s}`` carries amplitude/timing/shape and
``U_c ⊃ {h, h∘τ}`` carries the channel-share and channel-delay patterns
(position). Gauge is two-sided, ``Γ → Q_cᵀ Γ Q_t``.

Provenance: ``als_ut``/``als_uc``/``recon_kron``/``als_fit`` and the
``fit_nfpa``/``fit_iso_mpca`` bodies from
``experiments/synthetic/nfpa_confirmatory/methods.py``, ``whiten_separable``/
``unwhiten_separable`` from ``nfpa_confirmatory/covariance.py`` and
``kron_basis`` from ``nfpa_confirmatory/generator.py`` in the Paper 1
experiment repository (``noise-weighted-subspace-reconstruction`` @
``ea076ff``, 2026-09-10). The numerical core is unchanged; the ``Dataset``
coupling is replaced by explicit ``(Sigma_c, Sigma_t)`` arguments, and the
``NFPAFit`` container adds ``encode``/``reconstruct``. The archival
``basis_whitened`` reshape defect noted in the source is *not* reproduced:
:func:`kron_basis_whitened` follows the corrected ``angles_v2`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._metric import sym_sqrt, sym_sqrt_inv


# --------------------------------------------------------------------------- #
# Separable whitening (from covariance.py)
# --------------------------------------------------------------------------- #
def whiten_separable(X: np.ndarray, Sigma_c: np.ndarray, Sigma_t: np.ndarray) -> np.ndarray:
    """Apply ``Σ^{-1/2}`` as ``Z = Σ_c^{-1/2} X Σ_t^{-1/2}`` on ``(..., C, T)`` arrays."""
    Wc = sym_sqrt_inv(Sigma_c)
    Wt = sym_sqrt_inv(Sigma_t)
    return np.einsum("ab,...bd,de->...ae", Wc, X, Wt)


def unwhiten_separable(Z: np.ndarray, Sigma_c: np.ndarray, Sigma_t: np.ndarray) -> np.ndarray:
    """Invert :func:`whiten_separable` exactly (symmetric roots, not Cholesky factors)."""
    A = sym_sqrt(Sigma_c)
    B = sym_sqrt(Sigma_t)
    return np.einsum("ab,...bd,de->...ae", A, Z, B)


# --------------------------------------------------------------------------- #
# Alternating Rayleigh–Ritz (from methods.py) — numerical core unchanged
# --------------------------------------------------------------------------- #
def als_ut(Z: np.ndarray, U_c: np.ndarray, k_t: int) -> np.ndarray:
    D = np.einsum("nct,ck->nkt", Z, U_c)
    _, _, Vt = np.linalg.svd(D.reshape(Z.shape[0] * U_c.shape[1], Z.shape[2]), full_matrices=False)
    return Vt[:k_t].T


def als_uc(Z: np.ndarray, U_t: np.ndarray, k_c: int) -> np.ndarray:
    E = np.einsum("nct,tl->ncl", Z, U_t)
    _, _, Vt = np.linalg.svd(E.transpose(0, 2, 1).reshape(Z.shape[0] * U_t.shape[1], Z.shape[1]), full_matrices=False)
    return Vt[:k_c].T


def recon_kron(Z: np.ndarray, U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    c = np.einsum("nct,ck,tl->nkl", Z, U_c, U_t)
    return np.einsum("nkl,ck,tl->nct", c, U_c, U_t)


def als_fit(
    Z: np.ndarray,
    k_c: int,
    k_t: int,
    *,
    n_iter: int = 150,
    n_restarts: int = 10,
    seed: int = 0,
    tol: float = 1e-10,
    collect_restarts: bool = False,
) -> dict:
    """Alternating least squares with restarts; returns the best fit diagnostics.

    ``collect_restarts`` records every restart's factors and final loss under
    ``best["restarts"]`` so the optimisation term of an error decomposition can
    be measured. It changes neither the restart sequence, nor the selection
    rule, nor any returned numerical value.
    """
    best = None
    restarts: list[dict] = []
    for restart in range(int(n_restarts)):
        rng = np.random.default_rng(seed * 1000 + restart)
        U_c = np.linalg.qr(rng.standard_normal((Z.shape[1], k_c)))[0]
        U_t = np.linalg.qr(rng.standard_normal((Z.shape[2], k_t)))[0]
        hist = []
        for _ in range(int(n_iter)):
            U_t = als_ut(Z, U_c, k_t)
            U_c = als_uc(Z, U_t, k_c)
            hist.append(float(np.mean((Z - recon_kron(Z, U_c, U_t)) ** 2)))
        converged = len(hist) > 2 and abs(hist[-1] - hist[-2]) < tol
        if collect_restarts:
            restarts.append({"restart": restart, "U_c": U_c, "U_t": U_t, "loss": hist[-1], "converged": converged})
        if best is None or hist[-1] < best["loss"]:
            best = {
                "U_c": U_c,
                "U_t": U_t,
                "loss": hist[-1],
                "loss_history": hist,
                "restart": restart,
                "converged": converged,
            }
    if collect_restarts:
        best["restarts"] = restarts
    return best


def kron_basis(U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    """Orthonormal ``(CT, k_c·k_t)`` basis for ``range(U_t) ⊗ range(U_c)`` under column-major vec.

    Columns are ``kron(u_t[:, b], u_c[:, a])`` for all ``(a, b)``, consistent
    with Paper 1's vectorisation convention.
    """
    C, k_c = U_c.shape
    T, k_t = U_t.shape
    cols = [np.kron(U_t[:, b], U_c[:, a]) for b in range(k_t) for a in range(k_c)]
    B = np.column_stack(cols)
    Q, _ = np.linalg.qr(B)
    return Q


def kron_basis_whitened(U_c: np.ndarray, U_t: np.ndarray, Sigma_c: np.ndarray | None, Sigma_t: np.ndarray | None) -> np.ndarray:
    """Orthonormal whitened basis of the factored subspace, for principal angles.

    For an NFPA fit the factors already live in whitened coordinates and the
    result is :func:`kron_basis`. For an Iso-MPCA fit (raw coordinates) each
    basis column is whitened *as a C×T matrix* before re-orthonormalising —
    the corrected (``angles_v2``) path, not the archival interleaving reshape.
    """
    B = kron_basis(U_c, U_t)
    if Sigma_c is None or Sigma_t is None:
        return B
    C, T = U_c.shape[0], U_t.shape[0]
    cols = []
    for j in range(B.shape[1]):
        M = B[:, j].reshape(C, T, order="F")
        cols.append(whiten_separable(M, Sigma_c, Sigma_t).reshape(-1, order="F"))
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return Q


# --------------------------------------------------------------------------- #
# Fit container and entry points
# --------------------------------------------------------------------------- #
@dataclass
class NFPAFit:
    """A fitted factored subspace; ``metric`` is ``"Sigma^{-1}"`` (NFPA) or ``"I"`` (Iso-MPCA)."""

    name: str
    metric: str
    U_c: np.ndarray                   # (C, k_c), whitened coords for NFPA / raw for Iso-MPCA
    U_t: np.ndarray                   # (T, k_t)
    mean: np.ndarray                  # (C, T) training mean in raw units
    Sigma_c: np.ndarray | None = None
    Sigma_t: np.ndarray | None = None
    diag: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.U_c.shape[1] * self.U_t.shape[1])

    @property
    def n_parameters(self) -> int:
        """Grassmannian parameter count ``k_c(C−k_c) + k_t(T−k_t)``."""
        C, k_c = self.U_c.shape
        T, k_t = self.U_t.shape
        return int(k_c * (C - k_c) + k_t * (T - k_t))

    def _to_fit_coords(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mean
        if self.metric == "I":
            return Xc
        return whiten_separable(Xc, self.Sigma_c, self.Sigma_t)

    def _from_fit_coords(self, Z: np.ndarray) -> np.ndarray:
        if self.metric == "I":
            return Z + self.mean
        return unwhiten_separable(Z, self.Sigma_c, self.Sigma_t) + self.mean

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Code ``Γ = U_cᵀ Z̃ U_t`` of shape ``(..., k_c, k_t)``."""
        Z = self._to_fit_coords(X)
        return np.einsum("...ct,ck,tl->...kl", Z, self.U_c, self.U_t)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z = self._to_fit_coords(X)
        single = Z.ndim == 2
        Zhat = recon_kron(Z[None] if single else Z, self.U_c, self.U_t)
        return self._from_fit_coords(Zhat[0] if single else Zhat)

    def basis_whitened(self) -> np.ndarray:
        """Orthonormal ``(CT, rank)`` basis in whitened coordinates (for angles to CW-PCA).

        NFPA factors already live in whitened coordinates; only an Iso-MPCA fit
        (raw coordinates) is mapped through Σ̂^{-1/2} here.
        """
        if self.metric == "I":
            if self.Sigma_c is None or self.Sigma_t is None:
                raise ValueError("Iso-MPCA fit needs Sigma_c/Sigma_t to express its basis in whitened coordinates")
            return kron_basis_whitened(self.U_c, self.U_t, self.Sigma_c, self.Sigma_t)
        return kron_basis(self.U_c, self.U_t)


def fit_nfpa(
    X: np.ndarray,
    Sigma_c: np.ndarray,
    Sigma_t: np.ndarray,
    k_c: int,
    k_t: int,
    *,
    n_iter: int = 150,
    n_restarts: int = 10,
    seed: int = 0,
    tol: float = 1e-10,
    center: bool = True,
) -> NFPAFit:
    """NFPA: ALS in Σ̂-whitened coordinates restricted to the factored ``(k_c, k_t)`` class.

    ``X`` has shape ``(n, C, T)`` in raw units.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1:], dtype=np.float64)
    Z = whiten_separable(X - mean, Sigma_c, Sigma_t)
    out = als_fit(Z, k_c, k_t, n_iter=n_iter, n_restarts=n_restarts, seed=seed, tol=tol)
    return NFPAFit(
        name="NFPA",
        metric="Sigma^{-1}",
        U_c=out["U_c"],
        U_t=out["U_t"],
        mean=mean,
        Sigma_c=np.asarray(Sigma_c, dtype=np.float64),
        Sigma_t=np.asarray(Sigma_t, dtype=np.float64),
        diag={
            "train_loss": out["loss"],
            "selected_restart": out["restart"],
            "converged": out["converged"],
            "loss_history": out["loss_history"],
        },
    )


def fit_iso_mpca(
    X: np.ndarray,
    k_c: int,
    k_t: int,
    *,
    n_iter: int = 150,
    n_restarts: int = 10,
    seed: int = 0,
    tol: float = 1e-10,
    center: bool = True,
    Sigma_c: np.ndarray | None = None,
    Sigma_t: np.ndarray | None = None,
) -> NFPAFit:
    """Iso-MPCA: the same ALS on centred raw traces (Σ̂ = I) — the metric ablation.

    ``Sigma_c``/``Sigma_t`` are optional and only used by
    :meth:`NFPAFit.basis_whitened` to express the fitted subspace in the common
    whitened coordinates for principal angles.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1:], dtype=np.float64)
    out = als_fit(X - mean, k_c, k_t, n_iter=n_iter, n_restarts=n_restarts, seed=seed, tol=tol)
    return NFPAFit(
        name="Iso-MPCA",
        metric="I",
        U_c=out["U_c"],
        U_t=out["U_t"],
        mean=mean,
        Sigma_c=None if Sigma_c is None else np.asarray(Sigma_c, dtype=np.float64),
        Sigma_t=None if Sigma_t is None else np.asarray(Sigma_t, dtype=np.float64),
        diag={
            "train_loss": out["loss"],
            "selected_restart": out["restart"],
            "converged": out["converged"],
            "loss_history": out["loss_history"],
        },
    )
