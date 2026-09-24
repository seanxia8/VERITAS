# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Cable length: a lagged, dispersive, attenuating signal on the readout path.

A cable delays the signal by ``tau = L / v`` (group velocity ``v``) with
``|H(f)| = 1`` in the ideal case. This is a **phase-only** transfer function, so
it cannot be a ``noise_module`` PSD component: every spectral primitive there
multiplies the *density* by ``|H_k(f)|^2``, and ``|exp(-2 pi i f tau)|^2 = 1``,
i.e. a delay would be invisible in the power spectrum. The delay belongs on the
**signal path**, applied to the trace before the noise is added.

The optional ``dispersion_s2`` adds a quadratic phase (frequency-dependent group
delay, ``tau_g(f) = tau + D f``) and ``attenuation_per_hz`` a linear-in-frequency
amplitude loss (skin-effect-like). Both default to 0, recovering the pure delay.

The reflection *magnitude* comb of a mismatched cable is a different object and
does live in the PSD grammar (``noise_module.spectral_models.Reflection``); a
delay and a reflection are not interchangeable.
"""
from __future__ import annotations

import numpy as np


def cable_delay(x: np.ndarray, delay_s: float, fs: float, *,
                dispersion_s2: float = 0.0, attenuation_per_hz: float = 0.0) -> np.ndarray:
    """Delay a ``(C, N)`` trace by ``delay_s`` with optional dispersion/attenuation.

    Transfer function ``H(f) = exp(-a f) * exp(-2 pi i (tau f + 0.5 D f^2))``.
    Positive ``delay_s`` shifts the signal later. The DC bin is unchanged. For
    even ``N`` the Nyquist bin is forced real so the output is exactly real; a
    non-integer delay there cannot be represented, so a sub-sample fraction of
    the Nyquist power is lost — a bounded, documented artifact of delaying a
    real signal by an FFT phase ramp.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (n_channels, n_samples).")
    if fs <= 0.0:
        raise ValueError("fs must be positive.")
    if attenuation_per_hz < 0.0:
        raise ValueError("attenuation_per_hz must be non-negative.")
    if delay_s == 0.0 and dispersion_s2 == 0.0 and attenuation_per_hz == 0.0:
        return x.copy()
    N = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(N, 1.0 / fs)
    phase = -2.0 * np.pi * (delay_s * f + 0.5 * dispersion_s2 * f**2)
    amplitude = np.exp(-attenuation_per_hz * f)
    X = X * (amplitude * np.exp(1.0j * phase))
    if N % 2 == 0:
        X[..., -1] = X[..., -1].real
    return np.fft.irfft(X, n=N, axis=-1)


def group_delay_s(f: np.ndarray, delay_s: float, dispersion_s2: float = 0.0) -> np.ndarray:
    """Group delay ``tau_g(f) = tau + D f`` (seconds) for the cable model."""
    return delay_s + dispersion_s2 * np.asarray(f, dtype=float)


def delay_samples(delay_s: float, fs: float) -> float:
    """Convenience: a delay expressed in samples on the ``fs`` grid."""
    if fs <= 0.0:
        raise ValueError("fs must be positive.")
    return float(delay_s) * fs
