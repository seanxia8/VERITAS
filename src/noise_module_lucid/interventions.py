# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Declared LUCiD intervention families (the N contract) and what is *not* Σ.

Three kinds, kept apart because the ORACLE signature table separates them
(``docs/EXPERIMENT_DESIGN.md`` §III.1):

* **covariance-type N** — the realized Σ differs from the assumed Σ̂ while the
  transfer function is unchanged (clock amplitude, broadband common mode,
  digitiser alias fold). These carry the κ prediction.
* **structural N** — the transfer function or channel set changes (gain drift,
  channel loss, cable lag, ADC/TDC aperture jitter). These move the
  representation mean and the noise-only statistics together.
* **background / event families** — real unwanted *events* (radon, PMT
  pre-/after-pulse, pileup, DSNB, atmospheric ν, QE-vs-DIS). They are simulated
  as events with a shape, never as covariance; they belong to the S/U axis.

``DOCUMENTED_NOT_IMPLEMENTED`` records the families named in the working note
that are deliberately left to the event/background layer.
"""
from __future__ import annotations

import numpy as np

from .presets import FS_L, PMT_SHARED


def clock_scale(scale: float, shared: list | None = None) -> list:
    """Multiply the ADC-clock / switching line amplitudes by ``scale``.

    A covariance-type N: the shared process grows, the private floor does not.
    Use with :func:`noise_module_lucid.presets.crate_preset` and
    ``keep_private_power=True`` so the added power is the line's.
    """
    if scale < 0:
        raise ValueError("scale must be non-negative.")
    shared = PMT_SHARED if shared is None else shared
    out = []
    for comp in shared:
        comp = dict(comp)
        if comp.get("type") == "line":
            comp["scale"] = float(comp.get("scale", 1.0)) * float(scale)
        out.append(comp)
    return out


def add_broadband_common_mode(scale: float, shared: list | None = None) -> list:
    """Append a white common-mode term to the shared process.

    This is the V1 mistake promoted to a *declared intervention*: a frequency-flat
    crate coherence, physically a ground/HV ripple, not the clock. Covariance-type.
    """
    if scale < 0:
        raise ValueError("scale must be non-negative.")
    shared = PMT_SHARED if shared is None else shared
    return [*shared, {"type": "white", "scale": float(scale),
                      "name": "broadband_common_mode"}]


def decimate_alias_fold(trace: np.ndarray, factor: int = 4,
                        fs: float = FS_L) -> tuple[np.ndarray, float]:
    """Keep every ``factor``-th sample with no anti-alias filter.

    The digitiser-contract N family: nothing is added, the sampling contract
    changes, and :func:`alias_fold_prediction` gives the folded spectrum in
    closed form (``noise_module.psd_resampling``).
    """
    if factor < 1:
        raise ValueError("factor must be a positive integer.")
    trace = np.asarray(trace, dtype=float)
    return trace[:, ::factor], fs / factor


def alias_fold_prediction(f: np.ndarray, S: np.ndarray, fs_new: float, n_new: int):
    """Closed-form alias fold of a one-sided density (wraps ``noise_module``)."""
    from noise_module import alias_fold_psd_density

    return alias_fold_psd_density(f, S, fs_new, n_new)


def gain_drift_apply(trace: np.ndarray, groups: list[np.ndarray], sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Per-channel gain drift (structural N): scale each channel by N(1, sigma)."""
    trace = np.asarray(trace, dtype=float).copy()
    for idx in groups:
        gains = 1.0 + rng.normal(0.0, sigma, size=len(idx))
        trace[idx] *= gains[:, None]
    return trace


def channel_loss_apply(trace: np.ndarray, fraction: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Zero a random ``fraction`` of channels (structural N); return masked trace."""
    trace = np.asarray(trace, dtype=float).copy()
    C = trace.shape[0]
    n_drop = int(round(fraction * C))
    dropped = rng.choice(C, size=n_drop, replace=False) if n_drop else np.array([], int)
    trace[dropped] = 0.0
    return trace, np.sort(dropped)


def cable_lag_apply(trace: np.ndarray, delays_samples: np.ndarray) -> np.ndarray:
    """Apply per-channel integer cable delays (structural N: lagged signal)."""
    trace = np.asarray(trace, dtype=float)
    delays = np.asarray(delays_samples, dtype=int)
    if delays.shape != (trace.shape[0],):
        raise ValueError("delays_samples must have one integer per channel.")
    out = np.zeros_like(trace)
    for i, d in enumerate(delays):
        if d == 0:
            out[i] = trace[i]
        elif d > 0:
            out[i, d:] = trace[i, :-d]
        else:
            out[i, :d] = trace[i, -d:]
    return out


#: The declared N contract, keyed by family name.
N_FAMILIES: dict[str, dict] = {
    "clock_line_scale": {"kind": "covariance", "apply": clock_scale,
                         "note": "ADC clock + switching line amplitude ×k; narrow-band κ"},
    "broadband_common_mode": {"kind": "covariance", "apply": add_broadband_common_mode,
                              "note": "frequency-flat crate coherence; ground/HV ripple"},
    "alias_fold": {"kind": "digitiser", "apply": decimate_alias_fold,
                   "note": "decimate without anti-alias; exact closed-form prediction"},
    "gain_drift": {"kind": "structural", "apply": gain_drift_apply,
                   "note": "per-channel gain; moves the representation mean at the channel stage"},
    "channel_loss": {"kind": "structural", "apply": channel_loss_apply,
                     "note": "dead PMTs; mean shift at the token stage"},
    "cable_lag": {"kind": "structural", "apply": cable_lag_apply,
                  "note": "cable length -> lagged signal; transfer-function change"},
}

#: Named in the working note; simulated as events or deferred, never as Σ.
DOCUMENTED_NOT_IMPLEMENTED: dict[str, str] = {
    "radon_in_water": "radioactive background events (rate/spectrum/position); S/U family, not noise",
    "pmt_pre_after_pulse": "out-of-trigger and in-gate contamination; event/pileup family",
    "pileup_near_detector": "in-gate hit from a different event; mostly IWCD / near detector",
    "dark_rate_fluctuation": "LUCiD models dark counts as events; the rate/electronics fluctuation is the N part",
    "adc_tdc_aperture_jitter": "random sampling in ADC/TDC; digitiser-contract N, to add with quantisation",
    "dsnb_signal": "diffusive supernova neutrino background as signal; physics S family",
    "atmospheric_neutrino": "higher-energy tail leaking into the signal band (Cherenkov); background event family",
    "qe_vs_dis": "neutrino-nucleus interaction channel; physics S family",
    "50hz_mains": "not representable at 1 GHz/512 ns or even 32 µs; needs decimation or a >=1 s record",
}
