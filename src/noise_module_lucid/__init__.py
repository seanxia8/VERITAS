# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work, please
# cite it: see CITATION.cff at the repository root.
"""LUCiD front-end noise: presets, units bridge, grouping, adapter, interventions.

Thin adapter **on top of** :mod:`noise_module`; nothing in ``noise_module`` is
forked and nothing here imports LUCiD, so the package builds and tests without a
LUCiD clone (arm A is licence-gated on A0, ``docs/TESTBEDS.md`` §1.1).

What it supplies
----------------
* :mod:`noise_module_lucid.presets` — ``PMT_FRONTEND_V2`` (512 ns / 1 GHz grid)
  and ``PMT_FRONTEND_LONG`` (the 16–32 µs grid the κ cells need), each with a
  provenance record and a single-channel and a crate (spectral shared/private)
  view.
* :mod:`noise_module_lucid.units` — the photoelectron → mV bridge (SPE template).
* :mod:`noise_module_lucid.grouping` — the covariance unit (crate / string /
  angular-sector × height band).
* :mod:`noise_module_lucid.adapter` — ``add_readout_noise``: crate-wise noise on
  a LUCiD charge waveform, returning the implied and realized covariance per
  group.
* :mod:`noise_module_lucid.interventions` — the declared N families, plus the
  families documented but deliberately not modelled as covariance.

Explanation and physics: ``docs/noise_module_lucid.md``. Design:
``docs/EXPERIMENT_DESIGN.md`` §II.7 and §III.6.
"""

from .adapter import add_pmt_noise, add_readout_noise, crate_noise, kappa
from .grouping import groups_from_positions, groups_from_string_id
from .interventions import (
    DOCUMENTED_NOT_IMPLEMENTED,
    N_FAMILIES,
    add_broadband_common_mode,
    clock_scale,
    decimate_alias_fold,
)
from .presets import (
    GROUP,
    PMT_CRATE_LONG,
    PMT_CRATE_V2,
    PMT_FRONTEND_LONG,
    PMT_FRONTEND_V2,
    PMT_PRIVATE,
    PMT_SHARED,
    PMT_SHARED_LONG,
    PROVENANCE,
    RMS_MV,
    crate_preset,
    preset_for,
)
from .units import FS_L, charge_to_mv, spe_template, to_mv

__all__ = [
    "add_pmt_noise",
    "add_readout_noise",
    "add_broadband_common_mode",
    "charge_to_mv",
    "clock_scale",
    "crate_noise",
    "crate_preset",
    "decimate_alias_fold",
    "DOCUMENTED_NOT_IMPLEMENTED",
    "FS_L",
    "GROUP",
    "groups_from_positions",
    "groups_from_string_id",
    "kappa",
    "N_FAMILIES",
    "PMT_CRATE_LONG",
    "PMT_CRATE_V2",
    "PMT_FRONTEND_LONG",
    "PMT_FRONTEND_V2",
    "PMT_PRIVATE",
    "PMT_SHARED",
    "PMT_SHARED_LONG",
    "PROVENANCE",
    "preset_for",
    "RMS_MV",
    "spe_template",
    "to_mv",
]

__version__ = "0.1.0"
