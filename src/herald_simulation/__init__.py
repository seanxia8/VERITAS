# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""HeST → qp_simulator → noise_module: the paired superfluid-helium dark-matter arm."""

__version__ = "0.1.0"

from ._hest import HeSTUnavailable, available, hest_commit
from .events import HeraldEvent, evaporate, event_seed, initial_population, quanta
from .geometry import HERALD_V1_MAP, HeraldGeometry, granularity_pair, make_cell, shipped
from .noise import NoiseSpec, add_noise, sigma_cells
from .traces import TraceConfig, clean_traces

__all__ = ["HERALD_V1_MAP", "HeSTUnavailable", "HeraldEvent", "HeraldGeometry", "NoiseSpec", "TraceConfig",
           "add_noise", "available", "clean_traces", "evaporate", "event_seed", "granularity_pair", "hest_commit",
           "initial_population", "make_cell", "quanta", "shipped", "sigma_cells"]
