# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""HeRALD-style cell geometries: the five HeST builders plus a parametric factory.

HeST describes a detector as five boolean implicit surfaces (top, bottom,
wall, liquid surface, liquid volume) and a list of sensor surfaces. Geometry
is therefore Python, not a file — a sweep is a factory over
``(cell_radius_cm, fill_height_cm, sensor_pitch_cm, array_map)``. This module
mirrors ``HeST/core/Geometry.py::HeRALD_v1`` (MIT) so that a custom cell is
built from the same primitives, records the sensor positions HeST does not
expose, and hashes the parameters so a dataset names its geometry exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ._hest import _load

#: HeRALD_v1's 6×6 sensor mask (24 live 1 cm² CPDs at 1.1 cm pitch).
HERALD_V1_MAP = np.array([[0, 0, 1, 1, 0, 0],
                          [0, 1, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1, 0],
                          [0, 0, 1, 1, 0, 0]])


@dataclass(frozen=True)
class HeraldGeometry:
    name: str
    n_sensors: int
    positions_cm: np.ndarray               # (C, 3) sensor centres
    params: dict[str, Any]
    _build: Callable[[], Any] = field(repr=False, compare=False)

    def detector(self):
        """A fresh HeST ``VDetector`` (they hold numba closures; build per use, do not pickle)."""
        return self._build()

    @property
    def geometry_hash(self) -> str:
        payload = json.dumps({"name": self.name, "params": _jsonable(self.params)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def groups(self) -> np.ndarray:
        """Cold-stage grouping: every CPD in a HeRALD cell shares one stage."""
        return np.zeros(self.n_sensors, dtype=int)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    return o


def make_cell(
    cell_radius_cm: float = 3.5,
    fill_height_cm: float = 4.8,
    sensor_pitch_cm: float = 1.1,
    sensor_width_cm: float = 1.0,
    sensor_height_cm: float = 5.2,
    sensor_thickness_cm: float = 0.1,
    array_map: np.ndarray | None = None,
    wall_reflection: float = 0.3,
    wall_diffuse: float = 1.0,
    name: str | None = None,
) -> HeraldGeometry:
    """A HeRALD_v1-shaped cell with any sensor array (mirrors HeST's builder, parametrised)."""
    amap = HERALD_V1_MAP if array_map is None else np.asarray(array_map)
    rows, cols = amap.shape
    ii, jj = np.nonzero(amap > 0.5)
    x0 = (ii - (rows - 1) / 2.0) * sensor_pitch_cm
    y0 = (jj - (cols - 1) / 2.0) * sensor_pitch_cm
    z0 = sensor_height_cm + 0.5 * sensor_thickness_cm
    positions = np.column_stack([x0, y0, np.full_like(x0, z0, dtype=float)])
    params = dict(cell_radius_cm=cell_radius_cm, fill_height_cm=fill_height_cm, sensor_pitch_cm=sensor_pitch_cm,
                  sensor_width_cm=sensor_width_cm, sensor_height_cm=sensor_height_cm,
                  sensor_thickness_cm=sensor_thickness_cm, array_map=amap, wall_reflection=wall_reflection,
                  wall_diffuse=wall_diffuse)

    def build():
        core, det, _ = _load()
        from numba import njit

        VSensor, VDetector = det.VSensor, det.VDetector
        rad_sq = cell_radius_cm**2
        half_w, half_t = sensor_width_cm / 2.0, sensor_thickness_cm / 2.0

        @njit
        def outside_sensor(x, y, z, sx, sy, sz):
            return (np.abs(x - sx) > half_w) | (np.abs(y - sy) > half_w) | (z < sz - half_t)

        def make_condition(sx, sy, sz, index):
            return lambda x, y, z: (outside_sensor(x, y, z, sx, sy, sz), index)

        sensors = [VSensor(make_condition(float(px), float(py), float(pz), int(i)))
                   for i, (px, py, pz) in enumerate(positions)]

        @njit
        def bottom(x, y, z):
            return z > 0

        @njit
        def top(x, y, z):
            return z < sensor_height_cm + half_t

        @njit
        def wall(x, y, z):
            return x * x + y * y < rad_sq

        @njit
        def liquid(x, y, z):
            return z < fill_height_cm

        @njit
        def liquid_volume(x, y, z):
            return (x * x + y * y < rad_sq) & (z < fill_height_cm) & (z > 0)

        return VDetector(lambda x, y, z: (top(x, y, z), -1),
                         lambda x, y, z: (bottom(x, y, z), -1),
                         lambda x, y, z: (wall(x, y, z), -2),
                         lambda x, y, z: (liquid(x, y, z), -3),
                         liquid_volume,
                         sensors=sensors,
                         QP_wall_reflection_prob=wall_reflection, QP_wall_diffuse_prob=wall_diffuse,
                         QP_wall_Andreev_prob=0.0, QP_sensor_reflection_prob=0, QP_sensor_diffuse_prob=0,
                         QP_sensor_Andreev_prob=0.0)

    return HeraldGeometry(name=name or f"cell_{len(positions)}ch", n_sensors=int(len(positions)),
                          positions_cm=positions, params=params, _build=build)


def shipped(name: str, **kwargs) -> HeraldGeometry:
    """One of HeST's five shipped builders, with positions recorded where HeST defines them."""
    _, _, geo = _load()
    builders = {
        "HeRALD_v1": (geo.HeRALD_v1, 24),
        "HeRALD_v1_monolithic": (geo.HeRALD_v1_monolithic, 1),
        "HeRALD_LBNL": (geo.HeRALD_LBNL, None),
        "HeRALD_UMass_splitCPD": (geo.HeRALD_UMass_splitCPD, 2),
        "HeRALD_UMass_monolithic": (geo.HeRALD_UMass_monolithic, 1),
    }
    if name not in builders:
        raise ValueError(f"unknown HeST builder {name!r}; choose from {sorted(builders)}")
    fn, n = builders[name]
    det = fn(**kwargs)
    n_sensors = int(det.get_nsensors()) if n is None else n
    if name == "HeRALD_v1":
        positions = make_cell(**{k: v for k, v in kwargs.items() if k == "fill_height_cm"}).positions_cm
    else:
        positions = np.zeros((n_sensors, 3))          # HeST does not expose these; monolithic = one centred sensor
    return HeraldGeometry(name=name, n_sensors=n_sensors, positions_cm=positions,
                          params={"builder": name, **kwargs}, _build=lambda: fn(**kwargs))


def granularity_pair(fill_height_cm: float = 4.8) -> tuple[HeraldGeometry, HeraldGeometry]:
    """The designed contrast: 24 sensors vs one, on the identical cell."""
    return shipped("HeRALD_v1", fill_height=fill_height_cm), shipped("HeRALD_v1_monolithic", fill_height=fill_height_cm)
