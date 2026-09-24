# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Two pullback metrics on the representation, kept apart (TWO_CLAIM_REVISION_PLAN §3 M1).

Let r ∈ ℝ^d be a representation (here the pooled z), g: r ↦ x̂ ∈ ℝ^{C·N} the
reconstruction into *measurement* space, and y: r ↦ ℝ^t the *task* map into
declared physics outputs (amplitude, t0, τ …).

**Measurement metric.** With x = g(r) + n, n ~ 𝒩(0, Σ̂), the Fisher information
of r is I(r) = J_gᵀ Σ̂⁻¹ J_g (the standard Gaussian-model result, e.g. Kay 1993,
ch. 3). ``Subject.jac_recon`` returns the whitened Jacobian J̃_g = Σ̂^{-1/2} J_g,
so

    M_recon(r) = J̃_gᵀ J̃_g            (d × d, unit-free: whitened residual per sample)

measures how far a displacement δr moves the *reconstructed measurement*
relative to what the assumed noise resolves. It is local (evaluated at a
reference point), it depends on Σ̂ (the assumed, operational covariance — never
the realised one), and its rank deficiency is a *local identifiability*
statement, not a statement about training support.

**Task metric.** With W_y a declared positive semi-definite weight in
physics-output units (a diagonal of 1/σ_k² for a resolution requirement, or a
one-hot selecting the consequence target),

    M_task(r) = J_yᵀ W_y J_y            (d × d, units of W_y)

measures how far δr moves the declared physics outputs. Σ never enters: J_y maps
to outputs, not to measurements, so ``J_yᵀ Σ⁻¹ J_y`` is dimensionally invalid
and is not formed anywhere in this package.

Neither metric is a measure of realised scientific harm; Claim 2 tests that
connection empirically. The task-sensitive score of Claim 2 is the
Mahalanobis-type length ‖δr‖_{M_task} of the window's deviation from the
reference mean, standardised on clean windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TaskMetric:
    """M_task = J_yᵀ W_y J_y with a declared output weight W_y."""

    J_y: np.ndarray                 # (t, d) ∂y/∂r at the reference point
    W_y: np.ndarray                 # (t, t) declared PSD weight in output units
    target_names: tuple[str, ...] = ()
    declaration: str = ""           # where W_y comes from (a requirement, or "one-hot on <target>")

    def __post_init__(self) -> None:
        J = np.asarray(self.J_y, dtype=float); W = np.asarray(self.W_y, dtype=float)
        if J.ndim != 2:
            raise ValueError("J_y must be (t, d)")
        if W.shape != (J.shape[0], J.shape[0]):
            raise ValueError(f"W_y must be ({J.shape[0]}, {J.shape[0]}) in output units, got {W.shape}")
        if not np.allclose(W, W.T) or np.any(np.linalg.eigvalsh(0.5 * (W + W.T)) < -1e-12):
            raise ValueError("W_y must be symmetric positive semi-definite")
        object.__setattr__(self, "J_y", J); object.__setattr__(self, "W_y", W)

    @classmethod
    def one_hot(cls, J_y: np.ndarray, target: int, target_names: tuple[str, ...] = ()) -> "TaskMetric":
        """W_y selecting one declared consequence target (units: that output's units squared)."""
        t = np.asarray(J_y).shape[0]
        W = np.zeros((t, t)); W[target, target] = 1.0
        name = target_names[target] if target_names else str(target)
        return cls(J_y, W, target_names, declaration=f"one-hot on target {name!r}")

    @classmethod
    def from_resolutions(cls, J_y: np.ndarray, sigma: np.ndarray, target_names: tuple[str, ...] = ()) -> "TaskMetric":
        """W_y = diag(1/σ_k²) from declared per-output resolution requirements σ_k."""
        s = np.asarray(sigma, dtype=float)
        if np.any(s <= 0):
            raise ValueError("resolution requirements must be positive")
        return cls(J_y, np.diag(1.0 / s**2), target_names, declaration="diag(1/σ_k²) from declared resolutions")

    @property
    def M(self) -> np.ndarray:
        return self.J_y.T @ self.W_y @ self.J_y

    @property
    def latent_dim(self) -> int:
        return int(self.J_y.shape[1])

    def length(self, delta: np.ndarray) -> np.ndarray:
        """‖δ‖_{M_task} = sqrt(δᵀ M δ) for one or many displacements ``(…, d)``."""
        d = np.atleast_2d(np.asarray(delta, dtype=float))
        return np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", d, self.M, d), 0.0))

    def whitening(self) -> np.ndarray:
        """Symmetric ``M_task^{1/2}`` (zeros in its null space) — the ``M_task``-whitened chart.

        Coordinates ``y = A (r - r̄)`` satisfy ``‖y‖² = (r-r̄)ᵀ M_task (r-r̄)``
        (the task length) and the task metric becomes the identity there, so a
        third cumulant estimated on reference-fit ``y`` is the object the aligned
        cubic score contracts against (``ARITRA_CUMULANT_INTEGRATION_2026-09-17.md``
        §2.2).
        """
        w, V = np.linalg.eigh(0.5 * (self.M + self.M.T))
        tol = 1e-10 * max(float(w.max()), 1e-300)
        root = np.zeros_like(w)
        pos = w > tol
        root[pos] = w[pos] ** 0.5
        return (V * root) @ V.T

    def cubic_aligned(self, delta: np.ndarray, I3: np.ndarray) -> np.ndarray:
        """Aligned cubic score ``Δ_a Δ_b Δ_c Î3_abc`` for one or many ``delta`` ``(…, d)``.

        ``I3`` is the connected third cumulant ``(d, d, d)`` estimated on
        reference-fit vectors in the same (``M_task``-whitened) chart as
        ``delta``. The reading is task-aligned skewness: harm carried by an
        asymmetric excursion. It is a supporting intermediate score only — it
        never enters :data:`ΔAUROC_harm <latent_monitor.protocol.consequence.
        paired_delta_auroc>` or the committed generic score.
        """
        d = np.atleast_2d(np.asarray(delta, dtype=float))
        I3 = np.asarray(I3, dtype=float)
        if I3.shape != (d.shape[1], d.shape[1], d.shape[1]):
            raise ValueError(f"I3 must be ({d.shape[1]},) thrice, got {I3.shape}")
        return np.einsum("na,nb,nc,abc->n", d, d, d, I3)

    def aligned_direction(self) -> np.ndarray:
        """Unit vector in r along which the weighted outputs move fastest (top eigenvector of M_task)."""
        w, V = np.linalg.eigh(0.5 * (self.M + self.M.T))
        v = V[:, np.argmax(w)]
        return v / max(np.linalg.norm(v), 1e-300)

    def null_projector(self, tol: float = 1e-10) -> np.ndarray:
        """Projector onto directions that leave the *weighted* outputs unchanged to first order."""
        w, V = np.linalg.eigh(0.5 * (self.M + self.M.T))
        keep = w <= tol * max(float(w.max()), 1e-300)
        Q = V[:, keep]
        return Q @ Q.T

    def null_rank(self, tol: float = 1e-10) -> int:
        return int(np.round(np.trace(self.null_projector(tol))))


def measurement_metric(J_recon_whitened: np.ndarray) -> np.ndarray:
    """M_recon = J̃ᵀ J̃ for a whitened reconstruction Jacobian ``(C·N, d)``."""
    J = np.asarray(J_recon_whitened, dtype=float)
    if J.ndim != 2:
        raise ValueError("jac_recon must be (C·N, d)")
    return J.T @ J


@dataclass
class InvarianceReport:
    """What a statistic is invariant to. Recorded by the tests in ``tests/test_spine.py`` and stated in the paper."""

    statistic: str
    orthogonal: bool
    isotropic_scale: bool
    shear: bool
    note: str = ""
    extra: dict = field(default_factory=dict)
