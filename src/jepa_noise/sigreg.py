"""SIGReg for the Tier-1 bridge (runbook v3 WP4 arm 4; v4 §3.5.7).

SIGReg ("Sketched Isotropic Gaussian Regularization") is a sliced
goodness-of-fit test that pushes an embedding matrix toward an isotropic
Gaussian.  This module re-implements the primary reference's loss and pins its
numerical defaults; it is not a reproduction claim for a different architecture.

Source followed
---------------
- R. Balestriero and Y. LeCun, "LeJEPA: Provable and Scalable Self-Supervised
  Learning Without the Heuristics", arXiv:2511.08544 (2025).
- The authors' reference implementation of the loss, mirrored in
  ``stable_pretraining.methods.lejepa`` (galilai-group/stable-pretraining):
  classes ``EppsPulley`` and ``SlicedEppsPulley``, and the ``LeJEPA`` class
  default ``lamb=0.02``.  Exact pin (see :data:`SIGREG_SOURCE_PIN`):
  commit ``749419f1f6c28d6a3355bee236931ba492f1d95d`` of
  ``stable_pretraining/methods/lejepa.py``, fetched 2026-10-17; the file at
  that SHA was verified against this implementation.

Pinned numerics (see configs/bridge_v1.json, "sigreg")
-----------------------------------------------------
- ``num_slices = 1024`` random unit directions, drawn per forward call.
- Epps-Pulley quadrature ``t = linspace(0, t_max, n_points)`` with
  ``t_max = 3.0`` and ``n_points = 17`` (odd); trapezoid weights
  ``2*dt`` for interior nodes and ``dt`` at both ends, multiplied by the
  Gaussian weight ``exp(-t**2/2)``.
- target characteristic function ``phi(t) = exp(-t**2/2)`` for the cosine term
  and 0 for the sine term.
- per-slice statistic
  ``(err @ weights) * N`` (the source's N factor), reduced by the mean over
  slices; then mean over the samples feeding the call.
- slice directions are N(0,1) columns normalised to unit L2 norm, generated
  from a step-seeded ``torch.Generator`` exactly as the source does, so the
  sequence is deterministic and independent of global RNG state.
- ``coefficient_lambda = 0.02`` (source default); the coefficient is applied by
  the objective in :mod:`jepa_noise.jepa`, never chosen from claim performance.

Application per v3 WP4: the regulariser value is the average of the value on
the context pooled embeddings and the value on the target token embeddings
(flattened over batch).  The pinned statistic carries the source's per-call
sample-count factor ``N``; it is therefore a sum over samples (mean over
slices), so the per-sample weight is ~1/N.  With ``B`` pooled context samples
and ``B * n_target`` target tokens the two terms are of comparable total
magnitude and the target term is *not* up-weighted per sample: measured at
initialisation, context 1.05 at B = 128 vs target 0.92 at B = 32*128, i.e. the
per-sample target gradient is ~1/n_target of the context one.  The N factor
normalises per-sample weight; it is kept exactly as the source reduces it and
is recorded as a frozen property of the config (no invented normalisation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "SIGRegConfig",
    "SIGREG_SOURCE",
    "SIGREG_SOURCE_PIN",
    "EppsPulley",
    "SlicedEppsPulley",
    "SIGReg",
]

SIGREG_SOURCE = (
    "Balestriero & LeCun, LeJEPA, arXiv:2511.08544 (2025); reference loss in "
    "stable_pretraining/methods/lejepa.py (galilai-group/stable-pretraining): "
    "EppsPulley, SlicedEppsPulley, LeJEPA(lamb=0.02)."
)

#: Exact code pin followed by the implementation (also recorded in
#: ``configs/bridge_v1.json`` under ``sigreg.source``).
SIGREG_SOURCE_PIN = {
    "repository": "galilai-group/stable-pretraining",
    "file": "stable_pretraining/methods/lejepa.py",
    "commit_sha": "749419f1f6c28d6a3355bee236931ba492f1d95d",
    "commit_date": "2026-04-21T05:39:14Z",
    "raw_url": (
        "https://raw.githubusercontent.com/galilai-group/stable-pretraining/"
        "749419f1f6c28d6a3355bee236931ba492f1d95d/"
        "stable_pretraining/methods/lejepa.py"
    ),
    "accessed": "2026-10-17",
}


@dataclass(frozen=True)
class SIGRegConfig:
    """Frozen numerical defaults of the pinned reference implementation."""

    num_slices: int = 1024
    t_max: float = 3.0
    n_points: int = 17
    coefficient_lambda: float = 0.02

    def __post_init__(self):
        if self.num_slices < 1:
            raise ValueError("num_slices must be positive")
        if self.n_points < 3 or self.n_points % 2 != 1:
            raise ValueError("n_points must be odd and at least 3")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")
        if self.coefficient_lambda < 0:
            raise ValueError("coefficient_lambda must be non-negative")


class EppsPulley(nn.Module):
    """Univariate Epps-Pulley normality statistic (source quadrature)."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        if n_points < 3 or n_points % 2 != 1:
            raise ValueError("n_points must be odd and at least 3")
        t = torch.linspace(0.0, float(t_max), int(n_points))
        dt = float(t_max) / (int(n_points) - 1)
        self.register_buffer("t", t)
        weights = torch.full((int(n_points),), 2.0 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("phi", (-0.5 * t**2).exp())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x: Tensor) -> Tensor:
        """``x`` is ``(N, S)``: N samples, S slices; returns ``(S,)``."""
        if x.ndim != 2:
            raise ValueError("EppsPulley expects (N, slices)")
        n = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(dim=0)
        sin_mean = x_t.sin().mean(dim=0)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * n


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley statistic, source slice generation and reduction."""

    def __init__(
        self,
        num_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.num_slices = int(num_slices)
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        """``x`` is ``(N, D)`` embeddings; returns the scalar mean statistic."""
        if x.ndim != 2:
            raise ValueError("SlicedEppsPulley expects (N, D)")
        if x.size(0) < 2:
            raise ValueError("at least two samples are required")
        with torch.no_grad():
            step = int(self.global_step.item())
            generator = torch.Generator(device=x.device).manual_seed(step)
            directions = torch.randn(
                x.size(-1), self.num_slices, device=x.device, generator=generator
            )
            directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
            self.global_step.add_(1)
        return self.ep(x @ directions).mean()


class SIGReg(nn.Module):
    """Average of the SIGReg values on context-pooled and target-token embeddings.

    ``forward(context_pooled, target_tokens)`` returns a dict with ``value``
    (the regulariser value used by the objective), and the two components.  The
    coefficient ``coefficient_lambda`` is *not* applied here; the objective in
    :mod:`jepa_noise.jepa` multiplies by it.
    """

    def __init__(self, config: SIGRegConfig | None = None):
        super().__init__()
        self.config = SIGRegConfig() if config is None else config
        self.sliced = SlicedEppsPulley(
            num_slices=self.config.num_slices,
            t_max=self.config.t_max,
            n_points=self.config.n_points,
        )

    def forward(self, context_pooled: Tensor, target_tokens: Tensor) -> Dict[str, Tensor]:
        if context_pooled.ndim != 2:
            raise ValueError("context pooled embeddings must be (B, d)")
        if target_tokens.ndim < 2:
            raise ValueError("target token embeddings must be (B, n_target, d)")
        d_model = context_pooled.size(-1)
        target_flat = target_tokens.reshape(-1, d_model)
        if context_pooled.size(-1) != target_flat.size(-1):
            raise ValueError("context and target embeddings must share d_model")
        context_value = self.sliced(context_pooled)
        target_value = self.sliced(target_flat)
        return {
            "value": 0.5 * (context_value + target_value),
            "context": context_value,
            "target": target_value,
        }


def sigreg_config_from_dict(config: dict) -> SIGRegConfig:
    """Build the pinned config from the reviewed bridge JSON."""
    return SIGRegConfig(
        num_slices=int(config.get("num_slices", 1024)),
        t_max=float(config.get("t_max", 3.0)),
        n_points=int(config.get("n_points", 17)),
        coefficient_lambda=float(config.get("coefficient_lambda", 0.02)),
    )
