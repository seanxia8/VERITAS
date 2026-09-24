# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The LUCiD units bridge: photoelectron histogram -> mV.

LUCiD's waveform bin holds **summed photoelectron charge per 1 ns bin**
(``lucid/simulation/sensor_response.py:build_make_hits_waveform``), not volts.
A PSD in pe^2/Hz means nothing physically, so before adding front-end noise in
mV^2/Hz the charge waveform is convolved with a single-photoelectron voltage
template. The template is the only piece of real physics in the bridge; the
mV-per-pe amplitude is a placeholder like every level in the preset.

The template constants and its power bandwidth live in :mod:`presets` so the
front-end corner can be derived from them without a circular import.
"""
from __future__ import annotations

import numpy as np

from .presets import (
    FS_L,
    SPE_LENGTH_NS,
    SPE_MV_PER_PE,
    SPE_TAU_FALL_NS,
    SPE_TAU_RISE_NS,
    spe_bandwidth_hz,
    spe_shape,
)

#: Re-exported for callers that think of the bandwidth as a units property.
spe_bandwidth = spe_bandwidth_hz


def spe_template(
    fs: float = FS_L,
    tau_rise_ns: float = SPE_TAU_RISE_NS,
    tau_fall_ns: float = SPE_TAU_FALL_NS,
    mv_per_pe: float = SPE_MV_PER_PE,
    length_ns: float = SPE_LENGTH_NS,
) -> np.ndarray:
    """A two-exponential single-photoelectron voltage pulse, peak ``mv_per_pe``.

    ``p(t) = (1 - exp(-t/tau_rise)) * exp(-t/tau_fall)`` sampled at ``fs`` for
    ``length_ns``. The placeholder is a WCTE-like 3-inch PMT: 1 ns rise, 3 ns
    fall, power -3 dB at ~51 MHz.
    """
    if tau_rise_ns <= 0.0 or tau_fall_ns <= 0.0:
        raise ValueError("SPE time constants must be positive.")
    if fs <= 0.0 or length_ns <= 0.0:
        raise ValueError("fs and length_ns must be positive.")
    t_ns = np.arange(int(length_ns * fs / 1e9)) / fs * 1e9
    return mv_per_pe * spe_shape(t_ns, tau_rise_ns, tau_fall_ns)


def to_mv(wf_pe: np.ndarray, spe: np.ndarray | None = None) -> np.ndarray:
    """Convolve each row of a ``(C, N)`` photoelectron histogram with the SPE.

    Keeps the input length (causal, truncated convolution), so the output is in
    mV on the same time grid as LUCiD's waveform.
    """
    wf_pe = np.asarray(wf_pe, dtype=float)
    if wf_pe.ndim != 2:
        raise ValueError("waveform must have shape (n_channels, n_samples).")
    spe = spe_template() if spe is None else np.asarray(spe, dtype=float)
    return np.stack([np.convolve(row, spe)[: wf_pe.shape[1]] for row in wf_pe])


#: Backwards-compatible name used by the plan (``units.charge_to_mV``).
charge_to_mv = to_mv
