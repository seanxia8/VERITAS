# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Deterministic clock pickup: a phase-locked tone, not a Gaussian line.

A real ADC-clock / switching pickup is a deterministic, phase-locked sinusoid.
The Gaussian ``line`` component in the preset matches its *second-order* PSD but
sets every higher cumulant to zero, which is wrong if the study uses the
connected third cumulant to separate families. This module adds the tone with
the correct integrated power, so the PSD (and the covariance) are unchanged
while the non-Gaussianity is real.

The tone is **shared**: one phase per crate, the same waveform on every channel
of that crate, scaled by the per-channel front-end gains. Its power is taken
from the same ``line`` components the Gaussian preset would have used, so
switching between the two leaves the total noise power fixed.
"""
from __future__ import annotations

import numpy as np

from .presets import PMT_PRIVATE, PMT_SHARED, RMS_MV


def _as_rng(rng):
    from noise_module.core.utils import resolve_rng

    return resolve_rng(rng=rng)


def _abs_integral(components, fs: float, n: int) -> float:
    from noise_module import NoiseGenerator

    base = dict(noise_type="composite", sampling_frequency=fs, noise_power=1.0,
                power_definition="variance", composite_psd_scaling="absolute")
    _, s = NoiseGenerator({**base, "components": list(components)}).build_psd_density(n)
    return float(np.sum(s[1:]) * (fs / n))


def line_power(components, fs: float, n: int, *, all_components=None,
               noise_power: float | None = None) -> np.ndarray:
    """Power of each ``line`` component **as it appears in the normalized preset**.

    The preset is a composite in ``normalize`` mode, so a component's absolute
    density scale is only relative: its actual power is its absolute integral
    divided by the total, times ``noise_power``. Pass ``all_components`` (the
    full preset component list) and ``noise_power`` to get the physical powers;
    without them the raw absolute integrals are returned (and are not mV^2 of
    the preset).
    """
    abs_lines = np.array([_abs_integral([c], fs, n) for c in components])
    if all_components is None or noise_power is None:
        return abs_lines
    total = _abs_integral(all_components, fs, n)
    if total <= 0.0:
        return np.zeros_like(abs_lines)
    return abs_lines * (float(noise_power) / total)


def add_deterministic_clock(trace: np.ndarray, fs: float, components=None, rng=None, *,
                            gains: np.ndarray | None = None, t0_ns: float = 0.0,
                            all_components=None, noise_power: float | None = None,
                            return_metadata: bool = False):
    """Add one deterministic sinusoid per ``line`` component, shared by the crate.

    ``components`` defaults to ``presets.PMT_SHARED``. Each line's amplitude is
    ``sqrt(2 P)`` for its **normalized preset power** ``P`` (see
    :func:`line_power`); ``all_components``/``noise_power`` set that
    normalization (default: private + these lines, at the preset rms). ``gains``
    (per channel) scales the shared tone as the front-end would.
    """
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 2:
        raise ValueError("trace must have shape (n_channels, n_samples).")
    components = PMT_SHARED if components is None else components
    lines = [c for c in components if c.get("type") == "line"]
    C, N = trace.shape
    rng = _as_rng(rng)
    if all_components is None:
        all_components = [*PMT_PRIVATE, *components]
    if noise_power is None:
        noise_power = RMS_MV**2
    powers = (line_power(lines, fs, N, all_components=all_components, noise_power=noise_power)
              if lines else np.zeros(0))
    t = t0_ns * 1e-9 + np.arange(N) / fs
    g = np.ones(C) if gains is None else np.asarray(gains, dtype=float)
    if g.shape != (C,):
        raise ValueError("gains must have one entry per channel.")
    out = trace.copy()
    records = []
    for c, p in zip(lines, powers):
        amp = float(np.sqrt(2.0 * max(p, 0.0)))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        tone = amp * np.sin(2.0 * np.pi * c["frequency_hz"] * t + phase)
        out += g[:, None] * tone[None, :]
        records.append({"name": c.get("name"), "frequency_hz": float(c["frequency_hz"]),
                        "power": float(p), "amplitude": amp, "phase": phase})
    if not return_metadata:
        return out
    return out, {
        "contract": "deterministic_clock", "lines": records,
        "total_power": float(powers.sum()), "enters_sigma": True,
        "note": "shared, phase-locked; PSD matches the Gaussian line, higher cumulants are real",
    }
