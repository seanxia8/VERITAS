# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""LUCiD-customised front-end noise for the ORACLE water-Cherenkov arm.

A thin package that depends on ``noise_module`` (never forks it) and adds what
LUCiD does not simulate: the PMT front end between the anode and the digitiser.
See ``README.md`` for the sorted instruction list and what is in scope.

Import either the whole package or a submodule::

    from noise_module_lucid import add_pmt_noise, PMT_FRONTEND_V2
    from noise_module_lucid import digitiser, delay, pulses, clock, interventions
"""
from __future__ import annotations

__version__ = "0.3.0"

from . import clock, dataset, delay, digitiser, grouping, interventions, pulses, readouts, validation
from .adapter import (
    add_pmt_noise,
    add_readout_noise,
    apply_channel_gains,
    crate_preset,
    kappa,
    long_window_preset,
    matched_cell_kappa_floor,
)
from .clock import add_deterministic_clock, line_power
from .delay import cable_delay, delay_samples, group_delay_s
from .digitiser import (
    aperture_jitter_noise,
    high_frequency_fraction,
    quantise,
    sampling_jitter,
)
from .grouping import channel_groups
from .presets import (
    CLOCK_LINE_NAMES,
    CONTRACTS,
    FRONT_END,
    FRONT_END_BANDWIDTH_RATIO,
    FRONT_END_CORNER_HZ,
    FS_L,
    GROUP,
    LUCID_DARK_RATE_HZ,
    PMT_CRATE_V2,
    PMT_FRONTEND_V2,
    PMT_PRIVATE,
    PMT_SHARED,
    PROVENANCE,
    RMS_MV,
    SPE_BANDWIDTH_HZ,
    SPE_LENGTH_NS,
    SPE_MV_PER_PE,
    SPE_TAU_FALL_NS,
    SPE_TAU_RISE_NS,
    long_window_components,
    spe_bandwidth_hz,
    spe_shape,
)
from .pulses import (
    add_afterpulses,
    add_dark_pulses,
    add_prepulses,
    dark_pulse_times,
    place_pulses,
)
from .dataset import (
    CellSpec,
    EventSpec,
    build_dataset,
    covariance_cells,
    intervention_matrix,
    lucid_available,
    lucid_commit,
    reference_cell,
    run_cell,
    run_lucid_cell,
    write_cell,
)
from .readouts import (
    PMT_1GHZ,
    READOUTS,
    UNMODELLED,
    Readout,
    custom_readout,
    get_readout,
    register,
)
from .units import charge_to_mv, spe_bandwidth, spe_template, to_mv
from .validation import (
    bandwidth_report,
    grouping_report,
    grouping_sensitivity,
    kappa_floor_sweep,
    realized_csd_check,
)

__all__ = [
    # submodules
    "clock", "dataset", "delay", "digitiser", "grouping", "interventions", "pulses",
    "readouts", "validation",
    # readouts
    "Readout", "READOUTS", "UNMODELLED", "PMT_1GHZ", "get_readout", "custom_readout", "register",
    # dataset
    "EventSpec", "CellSpec", "run_cell", "run_lucid_cell", "build_dataset", "write_cell",
    "reference_cell", "intervention_matrix", "covariance_cells", "lucid_available", "lucid_commit",
    # adapter
    "add_pmt_noise", "add_readout_noise", "apply_channel_gains", "crate_preset", "kappa",
    "long_window_preset", "matched_cell_kappa_floor",
    # clock
    "add_deterministic_clock", "line_power",
    # delay
    "cable_delay", "delay_samples", "group_delay_s",
    # digitiser
    "aperture_jitter_noise", "high_frequency_fraction", "quantise", "sampling_jitter",
    # grouping
    "channel_groups",
    # units
    "charge_to_mv", "spe_bandwidth", "spe_template", "to_mv",
    # pulses
    "add_afterpulses", "add_dark_pulses", "add_prepulses", "dark_pulse_times", "place_pulses",
    # validation
    "bandwidth_report", "grouping_report", "grouping_sensitivity", "kappa_floor_sweep",
    "realized_csd_check",
    # constants and presets
    "CLOCK_LINE_NAMES", "CONTRACTS", "FRONT_END", "FRONT_END_BANDWIDTH_RATIO",
    "FRONT_END_CORNER_HZ", "FS_L", "GROUP", "LUCID_DARK_RATE_HZ", "PMT_CRATE_V2",
    "PMT_FRONTEND_V2", "PMT_PRIVATE", "PMT_SHARED", "PROVENANCE", "RMS_MV",
    "SPE_BANDWIDTH_HZ", "SPE_LENGTH_NS", "SPE_MV_PER_PE", "SPE_TAU_FALL_NS",
    "SPE_TAU_RISE_NS", "long_window_components", "spe_bandwidth_hz", "spe_shape",
    "__version__",
]
