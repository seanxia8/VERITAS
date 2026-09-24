# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The designed dissociation (plan §1, last row; proposal §6.2; audited 16 Sep, plan §8.3 R4).

Perturb the *input* so that z moves along the null space of the output
Jacobian (large alarm, ≈ zero consequence) or, norm-matched, along its row
space (same alarm, large consequence), or in a norm-matched random direction
(between the two). For a subject with a tied linear decoder in whitened
coordinates, a latent displacement δ is realised in input space as
``x + W⁻¹ Aᵀ δ`` — so all three families are exact, not approximate, and the
predicted sign is unambiguous.

**What this is and is not.** The families are *constructed* from the same
frozen head (``ref.P_out`` from ``subject.jac_output`` at the reference mean)
that a task-aware monitor uses. That a Jacobian-projected or pullback monitor
separates them is therefore a positive control, true by construction, and is
never reported as independent evidence. The falsifiable content is (i) the
*realised* consequence — the physical loss actually measured on the perturbed
events, not the Jacobian prediction — is ≈ 0 on ``output_null`` and large on
``output_aligned`` at the same norm, and (ii) magnitude-only monitors cannot
tell them apart. Both are local-linear statements: for a nonlinear subject
:func:`linearization_check` measures how far the realised output change
departs from the first-order prediction, and the report carries that number.

**Norm matching** is declared: ``match_metric="euclidean"`` matches ‖δ‖ in z
(the metric a naive monitor uses — the 6 Sep artifacts); ``"null_mahalanobis"``
matches the per-event alarm metric of :mod:`latent_monitor.statistics`
(‖δ‖ under the reference null Δz covariance), i.e. the alarm being challenged.

**Unavailable null spaces are explicit skips.** If the output Jacobian has
full row rank at the latent dimension there is no null space and
:class:`NullSpaceUnavailable` is raised — never a zero vector dressed up as a
null perturbation.

**Linear-only (16 Sep, I8).** ``x + W⁻¹ g(δ)`` is an exact local inverse of
the encoder only for the tied linear subject. For a nonlinear subject it is
not, so :meth:`DesignedFamily.perturb` refuses unless the subject declares
``is_linear = True``; the nonlinear route (a constrained local inverse through
the encoder Jacobian with an input-validity check) is not implemented and is
listed as future work, not approximated.

**Task-specific families (I8).** ``output_aligned`` moves along the whole row
space of J_y, which for a three-output head mostly moves timing while leaving
the declared consequence target (amplitude) almost unchanged — the 16 Sep smoke
showed exactly that. ``task_aligned`` / ``task_null`` are built from the
declared consequence's own metric M_task = J_yᵀ W_y J_y (``task_metric``):
aligned = its leading eigenvector, null = the complement in which the
*weighted* outputs are unchanged to first order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import ReferenceCell
from .subject import Geometry, Subject

KINDS = ("output_null", "output_aligned", "random", "task_aligned", "task_null")


class NonlinearSubjectUnsupported(RuntimeError):
    """The designed construction is exact only for a tied linear decoder; a nonlinear subject needs a local inverse that is not implemented."""


class NullSpaceUnavailable(RuntimeError):
    """The requested subspace has zero rank at this reference (e.g. no output-null directions)."""


def _rank(P: np.ndarray, tol: float = 1e-8) -> int:
    return int(np.sum(np.linalg.eigvalsh(0.5 * (P + P.T)) > tol))


@dataclass(frozen=True)
class DesignedFamily:
    kind: str                       # one of KINDS
    norm: float                     # ‖δ‖ in the declared match metric
    seed: int = 0
    match_metric: str = "euclidean" # "euclidean" | "null_mahalanobis"
    task_target: int | None = None  # for task_aligned / task_null: index of the declared consequence output

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.match_metric not in ("euclidean", "null_mahalanobis"):
            raise ValueError("match_metric must be euclidean | null_mahalanobis")
        if self.kind.startswith("task_") and self.task_target is None:
            raise ValueError("task_aligned / task_null need task_target (the declared consequence output index)")

    def _task_projectors(self, ref: ReferenceCell, subject: Subject | None, geometry: Geometry | None):
        from .task_metric import TaskMetric
        if subject is None or geometry is None:
            raise ValueError("task families need the subject and geometry to form J_y at the reference mean")
        J = np.asarray(subject.jac_output(ref.z_mean, geometry), dtype=float)
        tm = TaskMetric.one_hot(J, int(self.task_target))
        a = tm.aligned_direction()
        return np.outer(a, a), tm.null_projector()

    def subspace_rank(self, ref: ReferenceCell, subject: Subject | None = None, geometry: Geometry | None = None) -> int:
        if self.kind == "output_null":
            return _rank(ref.P_null)
        if self.kind == "output_aligned":
            return _rank(ref.P_out)
        if self.kind == "task_aligned":
            return _rank(self._task_projectors(ref, subject, geometry)[0])
        if self.kind == "task_null":
            return _rank(self._task_projectors(ref, subject, geometry)[1])
        return ref.latent_dim

    def latent_directions(self, ref: ReferenceCell, n: int, subject: Subject | None = None, geometry: Geometry | None = None) -> np.ndarray:
        """``n`` random directions in the family's subspace, norm-matched in the declared metric."""
        if self.subspace_rank(ref, subject, geometry) == 0:
            raise NullSpaceUnavailable(
                f"{self.kind}: the subspace has rank 0 at this reference (k_out={ref.k_out}, latent_dim={ref.latent_dim}); "
                "the designed contrast is not constructible here and must be reported as skipped")
        rng = np.random.default_rng([self.seed, {"output_null": 1, "output_aligned": 2, "random": 3, "task_aligned": 4, "task_null": 5}[self.kind]])
        if self.kind in ("task_aligned", "task_null"):
            Pa, Pn = self._task_projectors(ref, subject, geometry)
            P = Pa if self.kind == "task_aligned" else Pn
        else:
            P = {"output_null": ref.P_null, "output_aligned": ref.P_out, "random": np.eye(ref.latent_dim)}[self.kind]
        d = rng.normal(size=(n, ref.latent_dim)) @ P
        if self.match_metric == "euclidean":
            nrm = np.linalg.norm(d, axis=1, keepdims=True)
        else:
            L = np.linalg.cholesky(ref.null_dz_cov + 1e-12 * np.eye(ref.latent_dim))
            nrm = np.sqrt(np.sum(np.linalg.solve(L, d.T) ** 2, axis=0))[:, None]
        return self.norm * d / np.maximum(nrm, 1e-12)

    def perturb(self, subject: Subject, ref: ReferenceCell, X: np.ndarray, geometry: Geometry) -> np.ndarray:
        """X → X + W⁻¹ g(δ) so that z moves by exactly δ. Linear subjects only (see module docstring)."""
        if not getattr(subject, "is_linear", False):
            raise NonlinearSubjectUnsupported(
                f"{type(subject).__name__} does not declare is_linear=True: adding decode(δ) to the input is not a local "
                "inverse of a nonlinear encoder; a constrained local inverse with an input-validity check is not implemented")
        n = X.shape[0]
        delta = self.latent_directions(ref, n, subject, geometry)
        dxw = subject.decode(delta, geometry)                     # whitened-domain displacement, (n, C, N)
        return X + subject.sigma_hat.unwhiten(dxw)


def linearization_check(subject: Subject, ref: ReferenceCell, family: DesignedFamily, X: np.ndarray, geometry: Geometry) -> dict:
    """Realised versus first-order-predicted change, in z and in the output.

    Returns the relative error of the realised Δz against the intended δ and
    of the realised Δy against ``J_o δ``. Both are ≈ 0 for the linear subject
    (the control is exact); on a nonlinear subject they quantify how far the
    local-linear construction holds, and belong in the report next to the
    consequence numbers.
    """
    n = X.shape[0]
    delta = family.latent_directions(ref, n, subject, geometry)
    Xp = X + subject.sigma_hat.unwhiten(subject.decode(delta, geometry))
    r0, r1 = subject.represent(X, geometry), subject.represent(Xp, geometry)
    dz = r1["z"] - r0["z"]
    dy = r1["output"] - r0["output"]
    J = np.asarray(subject.jac_output(ref.z_mean, geometry), dtype=float)
    dy_pred = delta @ J.T
    rel_z = float(np.linalg.norm(dz - delta) / max(np.linalg.norm(delta), 1e-300))
    # normalise by the largest first-order output change the head could produce at this norm, so the
    # error is well defined for the null family (whose predicted change is zero by construction)
    scale = float(np.linalg.norm(J, 2) * np.linalg.norm(delta))
    rel_y = float(np.linalg.norm(dy - dy_pred) / max(scale, 1e-300))
    return {"kind": family.kind, "match_metric": family.match_metric, "norm": family.norm, "task_target": family.task_target,
            "subspace_rank": family.subspace_rank(ref, subject, geometry), "rel_err_dz": rel_z, "rel_err_dy": rel_y,
            "realised_dy_per_output": np.mean(np.abs(dy), axis=0).tolist(),
            "rel_err_dy_normalisation": "‖Δy − J δ‖ / (‖J‖₂ ‖δ‖)",
            "realised_dy_norm": float(np.mean(np.linalg.norm(dy, axis=1))),
            "predicted_dy_norm": float(np.mean(np.linalg.norm(dy_pred, axis=1)))}


class DesignedCell:
    """Wraps a ``tier1.Cell`` so that ``batch`` returns the designed perturbation of the twin."""

    def __init__(self, base, subject: Subject, ref: ReferenceCell, family: DesignedFamily):
        self.base, self.subject, self.ref, self.family = base, subject, ref, family
        self.geometry = base.geometry
        self.label = f"designed:{family.kind}"
        self.moved = "designed"

    def batch(self, event_ids: np.ndarray, replicate: int = 0):
        X, T = self.base.batch(event_ids, replicate)
        return self.family.perturb(self.subject, self.ref, X, self.geometry), T

    def noise_batch(self, record_ids: np.ndarray) -> np.ndarray:
        return self.base.noise_batch(record_ids)
