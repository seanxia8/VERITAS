# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The units bridge: LUCiD photoelectrons -> mV.

LUCiD's waveform bin holds summed photoelectron charge, not volts. A PSD in
pe²/Hz has no physical meaning, so each channel is convolved with a
single-photoelectron voltage template before noise is added in mV. The
calibration constant (mV per photoelectron) must agree with LUCiD's own gain so
the two layers cannot silently disagree (``docs/EXPERIMENT_DESIGN.md`` §II.7.4).
"""
from __future__ import annotations

import numpy as np

FS_L = 1.0e9                # LUCiD 1 GHz convention
SPE_MV_PER_PE = 4.0         # placeholder (sets the SNR)


def spe_template(fs: float = FS_L, tau_rise_ns: float = 2.0, tau_fall_ns: float = 8.0,
                 mv_per_pe: float = SPE_MV_PER_PE, length_ns: float = 60.0) -> np.ndarray:
    """Two-exponential single-photoelectron voltage pulse, peak ``mv_per_pe``.

    ``p(t) = (1 - exp(-t/tau_rise)) exp(-t/tau_fall)``, normalised so its maximum
    is ``mv_per_pe``. The integral (mV·ns per pe) is returned by
    :func:`spe_integral`.
    """
    if fs <= 0 or tau_rise_ns <= 0 or tau_fall_ns <= 0 or length_ns <= 0:
        raise ValueError("fs, time constants and length_ns must be positive.")
    t = np.arange(int(length_ns * fs / 1e9)) / fs * 1e9
    p = (1.0 - np.exp(-t / tau_rise_ns)) * np.exp(-t / tau_fall_ns)
    peak = p.max()
    if peak <= 0:
        raise ValueError("template is identically zero; check the time constants.")
    return mv_per_pe * p / peak


def spe_integral(spe: np.ndarray, fs: float = FS_L) -> float:
    """Integrated area of the SPE template in mV·ns per photoelectron."""
    return float(np.sum(spe) / fs * 1e9)


def charge_to_mv(wf_pe: np.ndarray, spe: np.ndarray | None = None) -> np.ndarray:
    """Convolve each channel's photoelectron histogram with the SPE template.

    ``wf_pe`` is ``(C, N)``; the result is ``(C, N)`` in mV, truncated to the
    input length (the tail beyond the window is dropped, matching LUCiD's
    fixed-window convention).
    """
    wf_pe = np.asarray(wf_pe, dtype=float)
    if wf_pe.ndim != 2:
        raise ValueError("charge_to_mv expects a (C, N) array.")
    spe = spe_template() if spe is None else np.asarray(spe, dtype=float)
    out = np.empty_like(wf_pe)
    for i, row in enumerate(wf_pe):
        out[i] = np.convolve(row, spe)[: wf_pe.shape[1]]
    return out


#: Compatibility alias for the pre-package notebook helper name.
to_mv = charge_to_mv
