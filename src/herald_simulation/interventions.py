# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Structural N interventions on the acquired traces (applied to signal *and* noise)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Structural:
    kind: str                                    # none | sensor_loss | gain_drift | timing_jitter
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, X: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng([seed, 0x57])
        C = X.shape[0]
        if self.kind == "none":
            return X
        if self.kind == "sensor_loss":
            k = int(self.params.get("n_lost", max(1, C // 4)))
            mask = np.ones(C, dtype=bool)
            mask[rng.choice(C, size=min(k, C), replace=False)] = False
            return X * mask[:, None]
        if self.kind == "gain_drift":
            s = float(self.params.get("sigma", 0.15))
            return X * (1.0 + rng.normal(0.0, s, size=C))[:, None]
        if self.kind == "timing_jitter":
            m = int(self.params.get("max_samples", 8))
            return np.stack([np.roll(X[c], int(rng.integers(-m, m + 1))) for c in range(C)])
        raise ValueError(f"unknown structural intervention {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params}


def structural_cells() -> list[Structural]:
    return [Structural("sensor_loss"), Structural("gain_drift"), Structural("timing_jitter")]
