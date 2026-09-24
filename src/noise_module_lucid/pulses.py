# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Event-level PMT pulse processes: dark counts, pre-pulses, after-pulses.

These are **not** front-end covariance. Dark counts (thermionic emission from
the photocathode), pre-pulses (light leaking to the photocathode) and
after-pulses (ion feedback) are Poisson-like *event* processes; LUCiD already
generates dark noise (``lucid/simulation/digitizer.py:generate_dark_noise``,
4.2 kHz/PMT). This module exists for the cases the arm needs to *declare and
inject* them on top of an existing trace.

The dark-noise contract (24 Sep 2026 review, P0.2)
--------------------------------------------------
Because LUCiD already adds dark pulses, calling :func:`add_dark_pulses` on a
LUCiD trace would double-count unless the baseline is subtracted. The default
``baseline_rate_hz = presets.LUCID_DARK_RATE_HZ`` therefore makes ``rate_hz`` a
**target total rate** and injects only the delta. To supply the full rate
(e.g. when LUCiD's dark layer is disabled), pass ``baseline_rate_hz=0.0``. The
metadata records both rates and the flag, so a protocol can audit it.

Other objects:
* **pre/after-pulse** contamination of an in-gate hit (a PMT-intrinsic N family,
  distinct from pile-up);
* **pile-up** is a different object and is *not* here: it is a second physical
  event in the gate, breaks exact pairing, and is an S/U family.

The after-pulse probability is charge-dependent (a brighter primary pulse
ionizes more residual gas): ``p(q) = min(1, p0 + p_per_pe * q)`` with an
exponential delay tail (the same geometric-offspring structure as the Hawkes
option in ``noise_module.artifacts.injector``).
"""
from __future__ import annotations

import numpy as np

from .presets import LUCID_DARK_RATE_HZ
from .units import spe_template


def _as_rng(rng):
    from noise_module.core.utils import resolve_rng

    return resolve_rng(rng=rng)


def place_pulses(trace: np.ndarray, sample_times, spe: np.ndarray | None = None,
                 amplitudes=None) -> np.ndarray:
    """Add SPE-shaped pulses to a ``(C, N)`` trace at per-channel sample times.

    ``sample_times`` is a sequence of length ``C`` of sample indices (float,
    rounded to the nearest sample). ``amplitudes`` is a matching sequence of
    per-pulse multipliers (default 1). Pulses that run past the record end are
    truncated. Returns a new array.
    """
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 2:
        raise ValueError("trace must have shape (n_channels, n_samples).")
    C, N = trace.shape
    if len(sample_times) != C:
        raise ValueError("sample_times must have one entry per channel.")
    spe = spe_template() if spe is None else np.asarray(spe, dtype=float)
    L = len(spe)
    out = trace.copy()
    for c in range(C):
        times = np.atleast_1d(np.asarray(sample_times[c], dtype=float))
        amps = np.ones_like(times) if amplitudes is None else np.atleast_1d(
            np.asarray(amplitudes[c], dtype=float))
        for t, a in zip(times, amps):
            start = int(round(t))
            if start < 0:
                continue
            end = min(start + L, N)
            if end <= start:
                continue
            out[c, start:end] += a * spe[: end - start]
    return out


def dark_pulse_times(n_channels: int, n_samples: int, fs: float, rate_hz: float, rng=None):
    """Homogeneous Poisson dark-pulse sample times, one array per channel."""
    rng = _as_rng(rng)
    if rate_hz < 0.0:
        raise ValueError("rate_hz must be non-negative.")
    expected = rate_hz * n_samples / fs
    return [np.sort(rng.uniform(0.0, n_samples, size=rng.poisson(expected)))
            for _ in range(n_channels)]


def add_dark_pulses(trace: np.ndarray, fs: float, rate_hz: float, spe=None, rng=None, *,
                    baseline_rate_hz: float = LUCID_DARK_RATE_HZ,
                    return_metadata: bool = False):
    """Add dark pulses for the **delta** between ``rate_hz`` and the baseline.

    ``rate_hz`` is the target total dark rate; ``baseline_rate_hz`` is what is
    already in the trace (LUCiD's own dark layer, default 4.2 kHz). Only the
    difference is injected, so calling this on a LUCiD trace does not
    double-count. Pass ``baseline_rate_hz=0.0`` to supply the full rate.
    """
    trace = np.asarray(trace, dtype=float)
    C, N = trace.shape
    if rate_hz < 0.0 or baseline_rate_hz < 0.0:
        raise ValueError("rates must be non-negative.")
    injected_rate = max(rate_hz - baseline_rate_hz, 0.0)
    times = dark_pulse_times(C, N, fs, injected_rate, rng=rng)
    out = place_pulses(trace, times, spe=spe)
    if not return_metadata:
        return out
    return out, {
        "contract": "dark_rate",
        "target_rate_hz": float(rate_hz),
        "baseline_rate_hz": float(baseline_rate_hz),
        "injected_rate_hz": float(injected_rate),
        "assumes_lucid_dark": baseline_rate_hz > 0.0,
        "counts_per_channel": [int(len(t)) for t in times],
        "expected_per_channel": float(injected_rate * N / fs),
        "enters_sigma": False,
    }


def add_afterpulses(trace: np.ndarray, fs: float, primary_times, probability: float,
                    tau_s: float, spe=None, rng=None, *, primary_charge=None,
                    probability_per_pe: float = 0.0, return_metadata: bool = False):
    """Ion-feedback after-pulses: per primary, at +Exp(tau), charge-dependent.

    ``primary_times`` is a sequence of length ``C`` of sample indices of the
    primary pulses. The per-primary probability is
    ``min(1, probability + probability_per_pe * q)`` where ``q`` is the primary
    charge in pe (``primary_charge``; default 1 pe each). Returns the trace with
    after-pulses added.
    """
    trace = np.asarray(trace, dtype=float)
    C, N = trace.shape
    if not 0.0 <= probability <= 1.0 or probability_per_pe < 0.0:
        raise ValueError("probability must lie in [0, 1] and probability_per_pe be non-negative.")
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive.")
    rng = _as_rng(rng)
    after, probs = [], []
    for c in range(C):
        prim = np.atleast_1d(np.asarray(primary_times[c], dtype=float))
        if primary_charge is None:
            q = np.ones(prim.size)
        else:
            q = np.atleast_1d(np.asarray(primary_charge[c], dtype=float))
            if q.size != prim.size:
                raise ValueError("primary_charge must match primary_times per channel.")
        p = np.clip(probability + probability_per_pe * q, 0.0, 1.0)
        keep = prim[rng.uniform(size=prim.size) < p]
        after.append(keep + rng.exponential(tau_s * fs, size=keep.size))
        probs.append(p)
    out = place_pulses(trace, after, spe=spe)
    if not return_metadata:
        return out
    return out, {
        "contract": "afterpulse", "probability": float(probability),
        "probability_per_pe": float(probability_per_pe), "tau_s": float(tau_s),
        "counts_per_channel": [int(len(a)) for a in after],
        "mean_probability": float(np.mean([p.mean() for p in probs if p.size])) if any(p.size for p in probs) else float(probability),
        "enters_sigma": False,
    }


def add_prepulses(trace: np.ndarray, fs: float, primary_times, probability: float,
                  lead_s: float, spe=None, rng=None, *, return_metadata: bool = False):
    """Pre-pulses: per primary, with ``probability``, ``lead_s`` before it."""
    trace = np.asarray(trace, dtype=float)
    C, N = trace.shape
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1].")
    if lead_s < 0.0:
        raise ValueError("lead_s must be non-negative.")
    rng = _as_rng(rng)
    pre = []
    for c in range(C):
        prim = np.atleast_1d(np.asarray(primary_times[c], dtype=float))
        keep = prim[rng.uniform(size=prim.size) < probability]
        pre.append(keep - lead_s * fs)
    out = place_pulses(trace, pre, spe=spe)
    if not return_metadata:
        return out
    return out, {
        "contract": "prepulse", "probability": float(probability), "lead_s": float(lead_s),
        "counts_per_channel": [int(len(p)) for p in pre], "enters_sigma": False,
    }
