# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The digitiser contract: quantisation and aperture/sampling jitter.

One acquisition-side change is unique to a sampled system: the ADC/TDC
conversion. Two declared effects are modelled here, both **structural N
families** that do not enter the front-end covariance ``Sigma``:

* **Quantisation** — uniform rounding to the ADC LSB. The error is white with
  variance ``LSB^2 / 12`` when the signal is not clipping; ``quantise`` returns
  that variance so the monitor can be told the contract changed.
* **Aperture jitter** — the sample instant is ``n/fs + delta_n`` with
  ``delta_n`` a zero-mean random offset. ``aperture_jitter_noise`` adds the
  first-order error ``delta_n * s'(t_n)``, which is what a real aperture jitter
  does to a signal: it converts slew rate into noise. It is valid for a
  **band-limited** trace (which the front-end roll-off guarantees).

The older ``sampling_jitter`` resampled the discrete trace by interpolation.
That is only valid for a band-limited signal and, applied to a white trace,
*removes* variance instead of adding jitter noise (measured ratio ~0.93); it is
kept for band-limited signals with an explicit guard, and the intervention uses
the derivative model.

The alias fold (decimation without an anti-alias filter) is the third member of
this family; it lives in ``noise_module.resampling.psd`` and is re-exported
through ``interventions.alias_fold``.
"""
from __future__ import annotations

import numpy as np


def _as_rng(rng):
    from noise_module.core.utils import resolve_rng

    return resolve_rng(rng=rng)


def quantise(x: np.ndarray, lsb: float, *, offset: float = 0.0,
             return_metadata: bool = False):
    """Uniformly quantise ``x`` to a step ``lsb`` (same units as ``x``).

    The ideal quantisation error of a non-clipping signal is uniform on
    ``[-lsb/2, lsb/2]``, variance ``lsb^2 / 12``. Returns the quantised array,
    and with ``return_metadata`` a record naming the LSB, the theoretical noise
    variance and the clipping fraction.
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(lsb) or lsb <= 0.0:
        raise ValueError("lsb must be finite and positive.")
    q = np.round((x - offset) / lsb) * lsb + offset
    if not return_metadata:
        return q
    return q, {
        "contract": "quantisation",
        "lsb": float(lsb),
        "noise_variance": float(lsb**2 / 12.0),
        "noise_variance_units": "signal^2",
        "clipping_fraction": 0.0,
        "enters_sigma": False,
    }


def high_frequency_fraction(x: np.ndarray, fraction_of_nyquist: float = 0.8) -> float:
    """Fraction of a trace's power above ``fraction_of_nyquist * Nyquist``.

    A smooth, band-limited trace has a small value; a white trace has ~0.2.
    Used to guard the interpolation-based :func:`sampling_jitter`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (n_channels, n_samples).")
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n)
    high = freqs >= fraction_of_nyquist * 0.5
    return float(power[..., high].sum() / max(power.sum(), 1e-300))


def aperture_jitter_noise(x: np.ndarray, sigma_t_s: float, fs: float, rng=None,
                          *, return_metadata: bool = False):
    """First-order aperture-jitter error on a band-limited ``(C, N)`` trace.

    Sample ``n`` is taken at ``n/fs + delta_n``, so to first order the measured
    value is ``x[n] + delta_n * x'(t_n)``. The derivative is the central
    difference ``(x[n+1] - x[n-1]) * fs / 2``. Jitter therefore turns slew rate
    into noise: a flat or slow signal is unaffected, a fast edge is not.

    The trace must be band-limited below Nyquist (the front-end roll-off
    guarantees this); on a white trace the finite-difference derivative is
    aliased and the model overstates the effect.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (n_channels, n_samples).")
    if sigma_t_s < 0.0 or fs <= 0.0:
        raise ValueError("sigma_t_s must be non-negative and fs positive.")
    if sigma_t_s == 0.0:
        out = x.copy()
        jitter = np.zeros_like(x)
    else:
        rng = _as_rng(rng)
        slope = np.gradient(x, axis=-1) * fs          # s'(t_n), central difference
        delta = rng.normal(0.0, sigma_t_s, size=x.shape)
        jitter = delta * slope
        out = x + jitter
    if not return_metadata:
        return out
    return out, {
        "contract": "aperture_jitter",
        "sigma_t_s": float(sigma_t_s),
        "sigma_samples": float(sigma_t_s * fs),
        "added_rms": float(np.sqrt(np.mean(jitter**2))) if jitter.size else 0.0,
        "model": "first_order_derivative",
        "enters_sigma": False,
    }


def sampling_jitter(x: np.ndarray, sigma_t_s: float, fs: float, rng=None,
                    *, strict: bool = True, return_metadata: bool = False):
    """Resample a **band-limited** ``(C, N)`` trace at jittered instants.

    ``delta_n ~ N(0, sigma_t_s)``; values are linearly interpolated from the
    nominal samples. This is valid only when the trace is oversampled relative
    to its bandwidth. With ``strict=True`` (default) it raises if more than 20 %
    of the trace power sits above 0.8 x Nyquist, which is the signature of a
    white trace where interpolation would *remove* variance; use
    :func:`aperture_jitter_noise` there instead.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (n_channels, n_samples).")
    if sigma_t_s < 0.0 or fs <= 0.0:
        raise ValueError("sigma_t_s must be non-negative and fs positive.")
    if strict and high_frequency_fraction(x) > 0.2:
        raise ValueError(
            "sampling_jitter needs a band-limited trace; use aperture_jitter_noise "
            "for a white/wideband trace.")
    rng = _as_rng(rng)
    C, N = x.shape
    nominal = np.arange(N, dtype=float)
    out = np.empty_like(x)
    for c in range(C):
        jittered = np.clip(nominal + rng.normal(0.0, sigma_t_s * fs, size=N), 0.0, N - 1.0)
        out[c] = np.interp(nominal, jittered, x[c])
    if not return_metadata:
        return out
    return out, {
        "contract": "sampling_jitter",
        "sigma_t_s": float(sigma_t_s),
        "sigma_samples": float(sigma_t_s * fs),
        "enters_sigma": False,
    }
