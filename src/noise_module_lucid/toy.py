# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""A toy water-Cherenkov photon source — DEVELOPMENT ONLY, never a claim substrate.

Exists so the response cells, the readout chain and the consequence variable
can be exercised without a local LUCiD installation. Research use was clarified in issue 6133100584. It is not
LUCiD and must not be described as a detector simulation:

* PMTs on the barrel and caps of a cylinder, facing inward;
* an isotropic point source (a flasher, LUCiD's default dataset source) at a
  vertex inside the volume, emitting at ``t0``;
* each *detected* photon lands on PMT ``i`` with probability proportional to the
  solid angle ``cos(incidence) / r^2``; direct light arrives at
  ``t0 + r / v_g + N(0, tts)``;
* a fraction ``scatter_fraction`` is diffuse late light: a uniformly random PMT
  and an extra exponential delay (mean ``scatter_delay_ns``).

No absorption, no Cherenkov cone, no reflections, no QE (the efficiency stage
of the readout chain holds it). Outputs are non-citable development artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .consequence import V_GROUP_M_PER_NS


@dataclass(frozen=True)
class ToyDetector:
    positions: np.ndarray          # (C, 3) metres
    normals: np.ndarray            # (C, 3) inward unit normals
    radius: float
    half_height: float

    @property
    def n_channels(self) -> int:
        return int(self.positions.shape[0])


def cylinder_detector(n_barrel_rings: int = 16, n_per_ring: int = 24, n_cap_rings: int = 4,
                      radius: float = 2.0, half_height: float = 2.0) -> ToyDetector:
    """A WCTE-scale cylinder (default 384 barrel + caps) with inward-facing PMTs."""
    pos, nrm = [], []
    zs = np.linspace(-half_height, half_height, n_barrel_rings + 2)[1:-1]
    phis = np.linspace(0.0, 2 * np.pi, n_per_ring, endpoint=False)
    for z in zs:
        for ph in phis:
            pos.append([radius * np.cos(ph), radius * np.sin(ph), z])
            nrm.append([-np.cos(ph), -np.sin(ph), 0.0])
    for sign in (+1.0, -1.0):
        for k in range(1, n_cap_rings + 1):
            rr = radius * k / (n_cap_rings + 1)
            m = max(6, int(round(n_per_ring * k / (n_cap_rings + 1))))
            for ph in np.linspace(0.0, 2 * np.pi, m, endpoint=False):
                pos.append([rr * np.cos(ph), rr * np.sin(ph), sign * half_height])
                nrm.append([0.0, 0.0, -sign])
    return ToyDetector(np.asarray(pos), np.asarray(nrm), radius, half_height)


@dataclass(frozen=True)
class ToyEvent:
    event_id: int
    vertex: np.ndarray
    t0: float
    photon_sensor: np.ndarray
    photon_time: np.ndarray
    true_energy: float = 1.0       # the mean photon yield (the flasher's "energy"); the count is Poisson
    endpoint_kind: str = "point_source"


def point_source_event(det: ToyDetector, event_id: int, *, n_photons: int = 3000, tts_ns: float = 1.5,
                       scatter_fraction: float = 0.15, scatter_delay_ns: float = 20.0,
                       fiducial: float = 0.6, t0: float = 10.0) -> ToyEvent:
    """One flasher event; the vertex is drawn uniformly in the fiducial cylinder (seeded by ``event_id``)."""
    rng = np.random.default_rng(10_000 + int(event_id))
    while True:
        v = np.array([rng.uniform(-1, 1) * det.radius, rng.uniform(-1, 1) * det.radius,
                      rng.uniform(-1, 1) * det.half_height]) * fiducial
        if np.hypot(v[0], v[1]) <= fiducial * det.radius:
            break
    d = det.positions - v
    r = np.linalg.norm(d, axis=1)
    cosi = np.clip(np.sum(-d / r[:, None] * det.normals, axis=1), 0.0, None)
    p = cosi / r**2
    p = p / p.sum()
    n_emit = int(rng.poisson(n_photons))              # light yield fluctuates: E_true = the mean yield
    n_sc = rng.binomial(n_emit, scatter_fraction)
    n_dir = n_emit - n_sc
    s_dir = rng.choice(det.n_channels, size=n_dir, p=p)
    t_dir = t0 + r[s_dir] / V_GROUP_M_PER_NS + rng.normal(0.0, tts_ns, n_dir)
    s_sc = rng.integers(0, det.n_channels, n_sc)
    t_sc = t0 + r[s_sc] / V_GROUP_M_PER_NS + rng.exponential(scatter_delay_ns, n_sc) + rng.normal(0.0, tts_ns, n_sc)
    return ToyEvent(int(event_id), v, float(t0), np.concatenate([s_dir, s_sc]), np.concatenate([t_dir, t_sc]),
                    true_energy=float(n_photons))


@dataclass(frozen=True)
class PromptEvent:
    event_id: int
    vertex: np.ndarray
    t0: float
    photon_batch: object
    true_energy: float
    endpoint_kind: str = "point_source"

    @property
    def photon_sensor(self):
        return self.photon_batch.sensor_id

    @property
    def photon_time(self):
        return self.photon_batch.arrival_time_ns


def prompt_source_event(det, event_id, *, n_photons=4000):
    """Incident isotropic flasher; no upstream QE/TTS/scatter/reflection.

    Geometric projected area is retained, angular response is uniform.
    Absorption and dispersion off; constant declared group velocity.
    Synthetic uniform wavelength spectrum; not a Cherenkov event.
    """
    from .photon_contract import PhotonBatch
    base = point_source_event(det, event_id, n_photons=n_photons, tts_ns=0, scatter_fraction=0)
    rng = np.random.default_rng(900000+int(event_id))
    batch = PhotonBatch(int(event_id), np.arange(len(base.photon_sensor)), base.photon_sensor,
                        base.photon_time, rng.uniform(300, 600, len(base.photon_sensor)))
    return PromptEvent(int(event_id), base.vertex, base.t0, batch, base.true_energy)
