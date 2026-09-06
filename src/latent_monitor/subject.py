# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The ``Subject`` protocol: what a frozen model must expose to be monitored.

This is the interface named in ``docs/LATENT_MONITORING_PLAN_2026-09-05.md``
§2.2 and compatible with ``oracle_cov.subjects.Subject`` (``represent``,
``outputs``, ``jac_recon``, ``jac_output``), extended with the six named hook
points and the whitening layer as a first-class, replaceable parameter.

Hook names, in forward order:

    whitened   x̃ = Σ̂^{-1/2} x             (…, C, N)
    channel    h_c per channel             (…, C, k)
    token      [h_c ‖ e(pos_c, group_c)]   (…, C, k + d_e)
    z          pooled representation       (…, d)
    pre_output decoder / head pre-activation (subject-defined)
    output     physics targets y           (…, n_targets)

Per-channel hooks are compared *pooled* across channels (mean and second
moment) because a geometry cell changes C; see ``statistics.stage_shift``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

HOOKS = ("whitened", "channel", "token", "z", "pre_output", "output")
PER_CHANNEL_HOOKS = frozenset({"whitened", "channel", "token"})


@dataclass(frozen=True)
class Geometry:
    """Channel positions, orientations and group ids — the per-channel geometry features."""

    positions: np.ndarray            # (C, 3)
    groups: np.ndarray | None = None  # (C,) int
    name: str = "unnamed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", np.asarray(self.positions, dtype=float))
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must be (C, 3).")
        if self.groups is not None:
            g = np.asarray(self.groups, dtype=int)
            if g.shape != (self.positions.shape[0],):
                raise ValueError("groups must be (C,).")
            object.__setattr__(self, "groups", g)

    @property
    def n_channels(self) -> int:
        return int(self.positions.shape[0])


@runtime_checkable
class Subject(Protocol):
    """A frozen model with named hooks, a named Σ̂ and exposed Jacobians."""

    @property
    def latent_dim(self) -> int: ...

    @property
    def sigma_hat(self) -> Any:
        """The assumed covariance (a :class:`~latent_monitor.whitening.KroneckerWhitener` or equivalent)."""
        ...

    def with_sigma_hat(self, sigma_hat: Any) -> "Subject":
        """Return a copy with the whitening layer replaced — the re-whitening adjustment."""
        ...

    def represent(self, X: np.ndarray, geometry: Geometry) -> Mapping[str, np.ndarray]:
        """All hooks for a batch ``X`` of shape ``(n, C, N)``."""
        ...

    def outputs(self, X: np.ndarray, geometry: Geometry) -> np.ndarray:
        """Physics targets ``(n, n_targets)``."""
        ...

    def decode(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        """``g(z)``: reconstruction in *whitened* coordinates, ``(n, C, N)``."""
        ...

    def jac_recon(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        """``Σ̂^{-1/2} ∂g/∂z`` at ``z`` — the whitened reconstruction Jacobian ``(C·N, d)``."""
        ...

    def jac_output(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        """``∂y/∂z`` at ``z``, ``(n_targets, d)``."""
        ...

    def forward_from_stage(self, stage: str, value: np.ndarray, X: np.ndarray, geometry: Geometry) -> Mapping[str, np.ndarray]:
        """Resume the forward pass with hook ``stage`` overwritten by ``value`` (activation patching)."""
        ...

    def noise_variance_scale(self, reference: Geometry, geometry: Geometry) -> float:
        """Expected ratio Var[z | noise, geometry] / Var[z | noise, reference] under Σ̂ — the
        subject's *own* prediction of how pooling rescales noise with channel count (1.0 if none)."""
        ...
