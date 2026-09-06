# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Event-type cells: ER vs NR at fixed energy, an energy sweep, and a WIMP spectrum (U)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._hest import _load


@dataclass(frozen=True)
class EventFamily:
    label: str
    interaction: str                       # ER | NR
    energy_ev: float | None = None         # fixed energy, or None for a spectrum
    spectrum: str | None = None            # None | "wimp"
    wimp_mass_mev: float = 500.0
    energy_range_ev: tuple[float, float] = (100.0, 2000.0)
    vertex_cm: tuple[float, float, float] = (0.0, 0.0, 2.0)

    def energy(self, event_id: int) -> float:
        if self.energy_ev is not None:
            return float(self.energy_ev)
        rng = np.random.default_rng([int(event_id), 0xE0])
        lo, hi = self.energy_range_ev
        if self.spectrum == "wimp":
            core, _, _ = _load()
            import importlib
            wg = importlib.import_module("HeST.core.WIMP_Generation")
            grid = np.linspace(lo, hi, 400)
            rate = np.asarray([max(float(wg.WIMP_dRate(e / 1000.0, self.wimp_mass_mev)), 0.0) for e in grid])
            if rate.sum() <= 0:
                return float(rng.uniform(lo, hi))
            return float(rng.choice(grid, p=rate / rate.sum()))
        return float(rng.uniform(lo, hi))

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def reference_family(energy_ev: float = 1000.0) -> EventFamily:
    return EventFamily("ER_1keV", "ER", energy_ev=energy_ev)


def event_cells(ref: EventFamily) -> list[EventFamily]:
    e = ref.energy_ev or 1000.0
    return [
        EventFamily("NR_same_E", "NR", energy_ev=e),
        EventFamily("ER_half_E", "ER", energy_ev=0.5 * e),
        EventFamily("ER_double_E", "ER", energy_ev=2.0 * e),
    ]


def undeclared_cells() -> list[EventFamily]:
    return [EventFamily("WIMP_NR_500MeV", "NR", spectrum="wimp", wimp_mass_mev=500.0)]
