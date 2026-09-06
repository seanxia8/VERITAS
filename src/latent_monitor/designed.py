# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The designed dissociation (plan §1, last row; proposal §6.2).

Perturb the *input* so that z moves along the null space of the output
Jacobian (large alarm, ≈ zero consequence) or, norm-matched, along its row
space (same alarm, large consequence). For a subject with a tied linear
decoder in whitened coordinates, a latent displacement δ is realised in
input space as ``x + W⁻¹ Aᵀ δ`` — so both families are exact, not
approximate, and the predicted sign is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import ReferenceCell
from .subject import Geometry, Subject


@dataclass(frozen=True)
class DesignedFamily:
    kind: str                       # "output_null" | "output_aligned"
    norm: float                     # ‖δ‖ in z units
    seed: int = 0

    def latent_directions(self, ref: ReferenceCell, n: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, hash(self.kind) & 0xFFFF])
        P = ref.P_null if self.kind == "output_null" else ref.P_out
        d = rng.normal(size=(n, ref.latent_dim)) @ P
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        return self.norm * d / np.maximum(nrm, 1e-12)

    def perturb(self, subject: Subject, ref: ReferenceCell, X: np.ndarray, geometry: Geometry) -> np.ndarray:
        """X → X + W⁻¹ g(δ) so that z moves by exactly δ (linear subject) or approximately (nonlinear)."""
        n = X.shape[0]
        delta = self.latent_directions(ref, n)
        dxw = subject.decode(delta, geometry)                     # whitened-domain displacement, (n, C, N)
        return X + subject.sigma_hat.unwhiten(dxw)


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
