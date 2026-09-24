# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Which channels share a front-end board: the covariance unit.

Coherent noise in a PMT array comes from **shared electronics** — PMTs on the
same HV supply, front-end board or digitiser crate share the clock pickup and
ripple, but not each other's amplifier thermal noise. So the natural covariance
unit is the physical board/string (16-64 channels), not the whole detector.

LUCiD does not expose a board or string identifier (its output is per-sensor
with positions in ``detector.all_points``). The methods here, in decreasing
physical fidelity:

* ``board_map`` — an explicit per-channel board id (from the real detector
  map). This is the correct grouping; supply it when you have it.
* ``proximity`` — a geometry proxy: greedy nearest-neighbour groups from the
  sensor positions. Use when no board map exists; it is a proxy, and the
  sensitivity of the result to the grouping must be reported.
* ``contiguous`` — the historical default (channel index). Only physically
  right if LUCiD's sensor ordering happens to follow the boards.
* ``z_plane`` — sensors in the same z band, a plausible readout unit for a
  cylindrical array.
"""
from __future__ import annotations

import numpy as np


def channel_groups(
    n_channels: int,
    group_size: int,
    positions: np.ndarray | None = None,
    method: str = "contiguous",
    board_ids: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Partition ``range(n_channels)`` into index arrays.

    Parameters
    ----------
    n_channels, group_size : int
        Number of channels and the maximum channels per group.
    positions : (n_channels, 3) array, optional
        Sensor positions in metres; required for ``proximity`` and ``z_plane``.
    method : {"contiguous", "z_plane", "proximity", "board_map"}
        Grouping method (see the module docstring).
    board_ids : (n_channels,) array, optional
        Physical board id per channel; required for ``board_map``.
    """
    if n_channels <= 0:
        raise ValueError("n_channels must be positive.")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    if method == "contiguous":
        order = np.arange(n_channels)
        return [np.asarray(order[i:i + group_size]) for i in range(0, n_channels, group_size)]

    if method == "z_plane":
        positions = _check_positions(positions, n_channels)
        order = np.argsort(positions[:, 2], kind="stable")
        return [np.asarray(order[i:i + group_size]) for i in range(0, n_channels, group_size)]

    if method == "proximity":
        positions = _check_positions(positions, n_channels)
        return _proximity_groups(positions, group_size)

    if method == "board_map":
        if board_ids is None:
            raise ValueError("method='board_map' requires board_ids.")
        board_ids = np.asarray(board_ids)
        if board_ids.shape != (n_channels,):
            raise ValueError(f"board_ids must have shape ({n_channels},).")
        groups = []
        for bid in np.unique(board_ids):
            idx = np.flatnonzero(board_ids == bid)
            for i in range(0, len(idx), group_size):
                groups.append(idx[i:i + group_size])
        return groups

    raise ValueError("method must be 'contiguous', 'z_plane', 'proximity' or 'board_map'.")


def _check_positions(positions, n_channels):
    if positions is None:
        raise ValueError("this method requires positions.")
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (n_channels, 3):
        raise ValueError(f"positions must have shape ({n_channels}, 3).")
    return positions


def _proximity_groups(positions: np.ndarray, group_size: int) -> list[np.ndarray]:
    """Greedy nearest-neighbour groups: a geometry proxy for the board map."""
    n = len(positions)
    remaining = list(range(n))
    groups = []
    while remaining:
        seed = remaining[0]
        d = np.linalg.norm(positions[remaining] - positions[seed], axis=1)
        chosen = [remaining[i] for i in np.argsort(d, kind="stable")[:group_size]]
        groups.append(np.array(sorted(chosen)))
        chosen_set = set(chosen)
        remaining = [i for i in remaining if i not in chosen_set]
    return groups
