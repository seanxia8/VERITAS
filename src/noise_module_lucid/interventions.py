# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Declared acquisition-contract interventions (the N families) for LUCiD.

Each intervention holds everything else fixed and moves one declared
acquisition condition. Two kinds are produced:

* **preset** interventions return a ``(base_config, crate_config)`` pair for
  ``MultiChannelNoiseGenerator`` (a changed front-end covariance);
* **trace** interventions transform an already-generated mV trace (a structural
  change that does not enter ``Sigma``).

The registry is deliberately small and named, so a protocol run can say which
families it used and record it.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .adapter import crate_preset
from .delay import cable_delay
from .digitiser import aperture_jitter_noise, quantise
from .presets import PMT_SHARED

#: Re-exported closed-form alias fold (decimation without anti-alias filter).
from noise_module.resampling.psd import alias_fold_psd_density  # noqa: E402


def _scaled(components, scale: float):
    out = deepcopy(components)
    for c in out:
        c["scale"] = float(c.get("scale", 1.0)) * scale
    return out


def clock_amplitude(scale: float):
    """Preset: multiply the shared clock lines by ``scale`` (adds power)."""
    if scale < 0.0:
        raise ValueError("scale must be non-negative.")
    return crate_preset(shared=_scaled(PMT_SHARED, scale))


def deterministic_clock():
    """Preset: phase-locked clock tone instead of the Gaussian line (P1.2)."""
    return crate_preset(clock="deterministic")


def broadband_common_mode(scale: float):
    """Preset: add a shared white term — a broadband coherent mode."""
    if scale < 0.0:
        raise ValueError("scale must be non-negative.")
    shared = deepcopy(PMT_SHARED) + [{"type": "white", "scale": float(scale), "name": "broadband_common_mode"}]
    return crate_preset(shared=shared)


def gain_drift(trace: np.ndarray, per_channel_gain: np.ndarray) -> np.ndarray:
    """Trace: multiply each channel by its gain (a structural N family)."""
    trace = np.asarray(trace, dtype=float)
    gain = np.asarray(per_channel_gain, dtype=float)
    if gain.shape != (trace.shape[0],):
        raise ValueError("per_channel_gain must have one entry per channel.")
    return trace * gain[:, None]


def decimate_no_antialias(trace: np.ndarray, factor: int) -> np.ndarray:
    """Trace: keep every ``factor``-th sample with no anti-alias filter.

    The realized spectrum is the closed-form alias fold of the input PSD; the
    prediction is exact, which makes this the cleanest N family there is.
    """
    if factor < 1:
        raise ValueError("factor must be a positive integer.")
    return np.asarray(trace, dtype=float)[:, ::factor]


#: Name -> (kind, callable). ``kind`` is "preset" or "trace".
INTERVENTIONS = {
    "clock_x2": ("preset", lambda: clock_amplitude(2.0)),
    "clock_x5": ("preset", lambda: clock_amplitude(5.0)),
    "clock_deterministic": ("preset", deterministic_clock),
    "broadband_common_mode_30": ("preset", lambda: broadband_common_mode(0.43)),
    "quantise_1mV": ("trace", lambda x, fs: quantise(x, 1.0)),
    "aperture_jitter_50ps": ("trace", lambda x, fs: aperture_jitter_noise(x, 50e-12, fs)),
    "cable_delay_10ns": ("trace", lambda x, fs: cable_delay(x, 10e-9, fs)),
    "cable_dispersive_10ns": ("trace", lambda x, fs: cable_delay(x, 10e-9, fs, dispersion_s2=1e-17)),
    "alias_fold_4": ("trace", lambda x, fs: decimate_no_antialias(x, 4)),
}


def describe() -> dict[str, str]:
    """Names and kinds of every declared intervention."""
    return {name: kind for name, (kind, _) in INTERVENTIONS.items()}
