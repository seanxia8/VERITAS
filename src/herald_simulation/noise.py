# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""HeRALD-shaped noise: ``noise_module.TESNoiseBudget`` + multichannel structure.

24 CPDs on one cold stage: a shared bath-temperature fluctuation and shared
SQUID/wiring pickup (mains, vibration) ride on private TES+SQUID white noise.
That is ``shared_private`` with the shared spectrum set by the budget, or
``lowrank`` with a few latent pickup modes. The generator returns both the
implied covariance (Σ̂, what the subject assumes) and the realized one (Σ), so
κ(Σ̂⁻¹Σ) is a number in every provenance record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from noise_module import HERALD_V1_PLACEHOLDER, MultiChannelNoiseGenerator, TESNoiseBudget


@dataclass(frozen=True)
class NoiseSpec:
    budget: TESNoiseBudget = HERALD_V1_PLACEHOLDER
    mode: str = "shared_private"           # shared_private | lowrank | independent
    corr_strength: float = 0.3             # bath fluctuation + pickup share (placeholder)
    n_latent: int = 2                      # pickup modes in lowrank mode
    channel_gain_jitter: float = 0.05
    label: str = "TES_HERALD_V1"

    def base_config(self, sampling_frequency: float, n_samples: int) -> dict[str, Any]:
        return self.budget.to_noise_config(sampling_frequency, n_samples)

    def generator(self, n_channels: int, sampling_frequency: float, n_samples: int, structure_seed: int) -> MultiChannelNoiseGenerator:
        return MultiChannelNoiseGenerator(
            self.base_config(sampling_frequency, n_samples),
            {"mode": self.mode, "n_channels": n_channels, "corr_strength": self.corr_strength,
             "n_latent": self.n_latent, "channel_gain_jitter": self.channel_gain_jitter,
             "freeze_channel_structure": True, "normalize_channel_variance": False},
            seed=structure_seed,
        )

    def provenance(self) -> dict[str, Any]:
        return {"label": self.label, "mode": self.mode, "corr_strength": self.corr_strength, "n_latent": self.n_latent,
                "channel_gain_jitter": self.channel_gain_jitter, "budget": self.budget.to_dict()}


def add_noise(X: np.ndarray, spec: NoiseSpec, sampling_frequency: float, event_seed: int, structure_seed: int = 11
              ) -> tuple[np.ndarray, dict[str, Any]]:
    """X + noise, with the implied and realized channel covariance and their κ in the metadata."""
    C, N = X.shape
    gen = spec.generator(C, sampling_frequency, N, structure_seed)
    gen.generate(8)                                    # pin the channel structure at structure_seed
    gen.rng = np.random.default_rng([int(event_seed), 0x4E])
    noise, meta = gen.generate(N, return_metadata=True)
    Sh, Sg = meta["implied_covariance"], meta["realized_covariance"]
    kappa = float(np.linalg.cond(np.linalg.solve(Sh + 1e-12 * np.eye(C), Sg + 1e-12 * np.eye(C)))) if C > 1 else 1.0
    return X + noise, {"implied_covariance": Sh, "realized_covariance": Sg, "kappa": kappa,
                       "mean_offdiag_corr": float(meta.get("mean_offdiag_corr", 0.0)),
                       "channel_structure_frozen": bool(meta.get("channel_structure_frozen", False)),
                       "spec": spec.label}


# ------------------------------------------------------------------ Σ cells
def sigma_cells(ref: NoiseSpec) -> list[NoiseSpec]:
    """Σ̂ ≠ Σ, covariance-type: bath correlation, pickup modes, SQUID knee, mains amplitude."""
    b = ref.budget
    return [
        replace(ref, corr_strength=min(ref.corr_strength + 0.4, 0.9), label="bath_corr_up"),
        replace(ref, mode="lowrank", n_latent=3, label="pickup_modes"),
        replace(ref, budget=replace(b, squid_knee_hz=b.squid_knee_hz * 10.0,
                                    provenance={**b.provenance, "squid_knee_hz": "design"}), label="squid_knee_up"),
        replace(ref, budget=replace(b, mains_line_psd=b.mains_line_psd * 8.0,
                                    provenance={**b.provenance, "mains_line_psd": "design"}), label="mains_up"),
    ]
