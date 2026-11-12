# SPDX-License-Identifier: MIT
"""Synthetic round-2 fixtures (not LUCiD data): a small cylinder and a small sphere."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_geometry as rg

ROUND1_DEV_TRAIN = Path("/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/lucid/"
                        "dev_train__F1__0.05__s20261017__n100000x1000")


def lucid_available() -> bool:
    try:
        rg.lucid_config_dir()
        return True
    except FileNotFoundError:
        return False


def small_cylinder(n_modules: int = 6) -> rg.R2Geometry:
    """114 PMTs on a WCTE-sized cylinder, pseudo-modules (k = 19)."""
    g = lb.from_synthetic({"module_ids": tuple(range(n_modules)), "pmts_per_module": 19,
                           "surface_counts": {"top": 19, "barrel": 19 * (n_modules - 2),
                                              "bottom": 19}})
    return rg.build_cylinder("synthetic r2 cylinder (NOT LUCiD)", "synthetic", g.positions_m,
                             g.directions, g.surfaces, g.radius_m, g.height_m, g.sensor_radius_m,
                             synthetic=True)


def fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return radius * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                              np.cos(phi)], axis=1)


def small_sphere(n: int = 200, radius: float = 2.0) -> rg.R2Geometry:
    return rg.build_sphere("synthetic r2 sphere (NOT LUCiD)", "synthetic",
                           fibonacci_sphere(n, radius), radius, 0.08, synthetic=True)
