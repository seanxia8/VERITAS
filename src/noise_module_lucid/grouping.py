# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Channel grouping: the covariance unit for the LUCiD arm.

Coherent noise in a PMT array comes from shared electronics, so the covariance
unit is a *crate / string / sub-array*, never the whole tank
(``docs/EXPERIMENT_DESIGN.md`` §II.7.3). These helpers turn the detector's own
grouping (a ``string_id`` array) or raw positions into a list of index arrays.
"""
from __future__ import annotations

import numpy as np


def contiguous_groups(n_channels: int, size: int) -> list[np.ndarray]:
    """Split ``range(n_channels)`` into contiguous groups of at most ``size``."""
    if n_channels <= 0 or size <= 0:
        raise ValueError("n_channels and size must be positive.")
    return [np.arange(start, min(start + size, n_channels))
            for start in range(0, n_channels, size)]


def groups_from_string_id(string_id: np.ndarray) -> list[np.ndarray]:
    """Group channel indices by ``string_id`` (order follows first appearance)."""
    string_id = np.asarray(string_id)
    if string_id.ndim != 1:
        raise ValueError("string_id must be 1-D, one entry per channel.")
    groups: list[np.ndarray] = []
    for value in dict.fromkeys(string_id.tolist()):
        groups.append(np.flatnonzero(string_id == value))
    return groups


def groups_from_positions(positions: np.ndarray, *, n_sectors: int = 8,
                          n_bands: int = 4) -> list[np.ndarray]:
    """Group channels into angular-sector × height-band cells.

    ``positions`` is ``(C, 3)`` in detector coordinates. Sectors are equal slices
    of the azimuthal angle; bands are equal slices of the vertical extent. Empty
    cells are dropped. This is a stand-in for a crate when the detector ships no
    crate/string id (the LUCiD cylinder case, ``EXPERIMENT_DESIGN.md`` §III.6).
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (C, 3).")
    if n_sectors <= 0 or n_bands <= 0:
        raise ValueError("n_sectors and n_bands must be positive.")
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    phi = np.arctan2(y, x)                                   # (-pi, pi]
    sector = np.floor((phi + np.pi) / (2 * np.pi) * n_sectors).astype(int)
    sector = np.clip(sector, 0, n_sectors - 1)
    z_lo, z_hi = float(z.min()), float(z.max())
    span = z_hi - z_lo
    if span <= 0:
        band = np.zeros_like(sector)
    else:
        band = np.clip(((z - z_lo) / span * n_bands).astype(int), 0, n_bands - 1)
    keys = sector * n_bands + band
    groups: list[np.ndarray] = []
    for value in dict.fromkeys(keys.tolist()):
        groups.append(np.flatnonzero(keys == value))
    return groups
