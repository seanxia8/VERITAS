# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Seeded, geometry-paired quantum-evaporation events.

``QP_propagation`` samples the whole initial quasiparticle population —
directions then momenta — before any geometry is touched, so seeding NumPy's
global generator from ``event_id`` gives a bit-identical initial population in
every ``VDetector``. That is the pairing contract; ``test_pairing`` enforces it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._hest import _load
from .geometry import HeraldGeometry

INTERACTIONS = ("ER", "NR")


def event_seed(event_id: int, salt: str = "herald") -> int:
    h = hashlib.sha256(f"{salt}:{int(event_id)}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def quanta(energy_ev: float, interaction: str) -> dict[str, int]:
    """HeST's yield partition at ``energy_ev``: quasiparticles, IR, singlet UV, triplet."""
    core, _, _ = _load()
    if interaction not in INTERACTIONS:
        raise ValueError(f"interaction must be one of {INTERACTIONS}")
    q = core.GetQuanta(float(energy_ev), interaction)
    return {"quasiparticles": int(q.Quasiparticles), "ir_photons": int(q.IRPhotons),
            "singlet_photons": int(q.SingletPhotons), "triplet_molecules": int(q.TripletMolecules)}


@dataclass(frozen=True)
class HeraldEvent:
    event_id: int
    geometry_name: str
    geometry_hash: str
    energy_ev: float
    interaction: str
    vertex_cm: tuple[float, float, float]
    n_qp: int
    arrival_times_us: list[np.ndarray]     # per sensor
    energies_ev: list[np.ndarray]          # per sensor
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        return len(self.arrival_times_us)

    @property
    def n_detected(self) -> int:
        return int(sum(len(a) for a in self.arrival_times_us))

    def truth(self) -> dict[str, Any]:
        return {"energy_ev": self.energy_ev, "interaction": self.interaction, "vertex_x_cm": self.vertex_cm[0],
                "vertex_y_cm": self.vertex_cm[1], "vertex_z_cm": self.vertex_cm[2], "n_qp": self.n_qp,
                "n_detected": self.n_detected}


def evaporate(
    geometry: HeraldGeometry,
    event_id: int,
    energy_ev: float,
    interaction: str,
    vertex_cm: tuple[float, float, float] = (0.0, 0.0, 2.0),
    *,
    n_qp: int | None = None,
    qp_fraction: float = 1.0,
    temperature_k: float = 2.0,
    detector=None,
) -> HeraldEvent:
    """One event through one geometry, seeded from ``event_id`` so it pairs across geometries.

    ``n_qp`` overrides the yield model (e.g. for a fixed-population pilot);
    ``qp_fraction`` thins the population uniformly for cost (a 1 keV NR is ~9e5
    QP ≈ 17 s single-core; 0.1 keeps the arrival-time *distribution* and
    scales the amplitude by a known factor recorded in ``meta``).
    """
    _, det, _ = _load()
    q = quanta(energy_ev, interaction)
    n = int(q["quasiparticles"] if n_qp is None else n_qp)
    n_sim = max(1, int(round(n * qp_fraction)))
    seed = event_seed(event_id)
    np.random.seed(seed)                       # HeST samples from NumPy's global generator
    vd = geometry.detector() if detector is None else detector
    sig = det.GetEvaporationSignal(vd, n_sim, float(vertex_cm[0]), float(vertex_cm[1]), float(vertex_cm[2]),
                                   useMap=False, T=float(temperature_k))
    times = [np.asarray(a, dtype=float) for a in sig.arrivalTimes]
    energies = [np.asarray(e, dtype=float) for e in sig.energies]
    return HeraldEvent(event_id=int(event_id), geometry_name=geometry.name, geometry_hash=geometry.geometry_hash,
                       energy_ev=float(energy_ev), interaction=interaction, vertex_cm=tuple(map(float, vertex_cm)),
                       n_qp=n, arrival_times_us=times, energies_ev=energies, seed=seed,
                       meta={"n_qp_simulated": n_sim, "qp_fraction": float(qp_fraction), "yields": q,
                             "temperature_k": float(temperature_k)})


def initial_population(event_id: int, n_qp: int, temperature_k: float = 2.0) -> np.ndarray:
    """The initial QP momenta HeST would draw for this seed — the pairing witness (see tests)."""
    core, _, _ = _load()
    np.random.seed(event_seed(event_id))
    _dx, _dy, _dz = _load()[1].generate_random_direction(n_qp)
    return np.asarray(core.Random_QPmomentum(n_qp, T=float(temperature_k)))
