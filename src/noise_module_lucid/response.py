# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Detector-response N families for the LUCiD arm (12 Oct 2026 re-scope, R-2/R-3).

Why this module exists
----------------------
Front-end electronics noise is a weak N for an optical detector: it is small
next to the single-photoelectron pulse and hit-finding thresholds remove it
before reconstruction, so harm is close to zero (nothing for Claim 2 to rank)
and whatever is visible is caught by a generic input/output monitor (Claim 1
expected null). Changes of the *detector response* do harm optical
reconstruction, so they are the LUCiD N families
(``docs/RUNBOOK_RESCOPE_N_2026-10-12.md``).

The taxonomy rule (R-2)
-----------------------
N is a **change of the acquisition/response chain after the model was trained**.

* A static per-PMT spread present at training time (e.g. QE varying from PMT
  to PMT) is part of the detector definition — it belongs in the reference
  cell, not in an N family.
* Simulation mis-modelling — the angular-acceptance shape, scattering,
  absorption, the medium — is a separate category **M**. It is not a corrupted
  measurement; it is excluded from both claims and from fitting and
  calibration. See :data:`EXCLUDED`.

The families (all origin ``N_struct``)
--------------------------------------
* ``efficiency_loss`` — photon-level binomial thinning (QE or collection
  efficiency drops). Changes hit counts and therefore the Poisson statistics.
* ``gain_drift`` — charge scale only; the hit count is unchanged. Kept separate
  from efficiency because the two signatures differ.
* ``timing_drift`` — per-channel time offsets (a calibration constant moves).
* ``threshold_walk`` — a discriminator-threshold change. A larger pulse
  crosses a fixed threshold earlier (time-walk); moving the threshold shifts
  each hit by a charge-dependent amount and loses hits that no longer cross.
  Derived from the SPE template by threshold crossing.
* ``module_loss`` — a crate/board goes dead.

Data conventions
----------------
Nothing here imports LUCiD (gate A0). Two input forms are supported:

* **hits** — flat arrays ``pmt_ids``, ``times`` (ns), ``charges`` (pe), one
  entry per detected photoelectron/hit;
* **waveforms** — ``(n_channels, n_bins)`` arrays: an *integer* photoelectron
  count waveform (thinning is exact), a pe-charge waveform (gain), or a mV
  trace (timing, module loss, leading-edge times).

Thinning is refused on a float, charge-smeared waveform: once charges are
smeared, which photons to remove is no longer defined. Thin before smearing,
or scale the per-PMT QE in the LUCiD physics config (:func:`scaled_qe`).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .delay import cable_delay
from .presets import FS_L, SPE_MV_PER_PE, SPE_TAU_FALL_NS, SPE_TAU_RISE_NS

# --------------------------------------------------------------------------- helpers


def _rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _per_channel(values, n_channels: int, name: str) -> np.ndarray:
    """Broadcast a scalar or validate a ``(n_channels,)`` array."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 0:
        return np.full(n_channels, float(v))
    if v.shape != (n_channels,):
        raise ValueError(f"{name} must be a scalar or have shape ({n_channels},); got {v.shape}.")
    return v


def _check_ids(pmt_ids, n_channels: int) -> np.ndarray:
    ids = np.asarray(pmt_ids)
    if ids.size and (not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError("pmt_ids must be integers.")
    if ids.size and (ids.min() < 0 or ids.max() >= n_channels):
        raise ValueError("pmt_ids out of range for the per-channel array.")
    return ids.astype(int, copy=False)


def group_values(groups: Sequence[np.ndarray], n_channels: int, value: float,
                 affected_group_ids: Iterable[int], default: float) -> np.ndarray:
    """Per-channel array: ``value`` on the channels of the affected groups, ``default`` elsewhere.

    Drifts are usually coherent per board/crate; ``groups`` comes from
    :func:`noise_module_lucid.grouping.channel_groups`.
    """
    out = np.full(n_channels, float(default))
    for g in affected_group_ids:
        if g < 0 or g >= len(groups):
            raise ValueError(f"group id {g} out of range (have {len(groups)} groups).")
        out[np.asarray(groups[g], dtype=int)] = float(value)
    return out


# --------------------------------------------------------------------------- efficiency loss


def _check_survival(survival: np.ndarray) -> None:
    if np.any(~np.isfinite(survival)) or np.any(survival < 0.0) or np.any(survival > 1.0):
        raise ValueError("survival probabilities must lie in [0, 1].")


def efficiency_loss_hits(pmt_ids, survival, rng=None, *, n_channels: int | None = None) -> np.ndarray:
    """Keep-mask for photon-level binomial thinning.

    Each hit on PMT ``i`` survives independently with probability
    ``survival[i]`` (a scalar applies to every PMT). Apply the mask to every
    per-hit array (times, charges, ids).
    """
    ids = np.asarray(pmt_ids)
    if np.ndim(survival) == 0:
        p = np.full(ids.shape, float(survival))
        _check_survival(p)
    else:
        s = np.asarray(survival, dtype=float)
        nch = s.shape[0] if n_channels is None else n_channels
        s = _per_channel(s, nch, "survival")
        _check_survival(s)
        p = s[_check_ids(ids, nch)]
    return _rng(rng).random(ids.shape) < p


def efficiency_loss_counts(pe_counts: np.ndarray, survival, rng=None) -> np.ndarray:
    """Exact binomial thinning of an integer photoelectron-count waveform ``(C, N)``.

    Refuses non-integer input: thinning a charge-smeared waveform is not
    defined (see the module docstring).
    """
    x = np.asarray(pe_counts)
    if x.ndim != 2:
        raise ValueError("pe_counts must have shape (n_channels, n_bins).")
    xf = x.astype(float)
    if np.any(xf < 0) or np.any(np.mod(xf, 1.0) != 0.0):
        raise ValueError("efficiency_loss_counts needs a non-negative integer pe-count waveform; "
                         "thin before charge smearing, or use scaled_qe on the LUCiD config.")
    s = _per_channel(survival, x.shape[0], "survival")
    _check_survival(s)
    return _rng(rng).binomial(xf.astype(np.int64), s[:, None]).astype(x.dtype if np.issubdtype(x.dtype, np.integer) else float)


def scaled_qe(per_pmt_qe, survival) -> np.ndarray:
    """The LUCiD-config route: multiply the per-PMT QE array by ``survival``.

    For use once gate A0 allows a LUCiD-dependent run: the thinning then
    happens inside LUCiD's own photon loop, before charge smearing.
    """
    qe = np.asarray(per_pmt_qe, dtype=float)
    if qe.ndim != 1:
        raise ValueError("per_pmt_qe must be one-dimensional.")
    s = _per_channel(survival, qe.shape[0], "survival")
    _check_survival(s)
    return qe * s


# --------------------------------------------------------------------------- gain drift


def gain_drift_hits(charges, pmt_ids, gain) -> np.ndarray:
    """Scale each hit's charge by its PMT's gain factor. The hit count is unchanged."""
    q = np.asarray(charges, dtype=float)
    if np.ndim(gain) == 0:
        return q * float(gain)
    g = np.asarray(gain, dtype=float)
    if np.any(g < 0):
        raise ValueError("gain must be non-negative.")
    return q * g[_check_ids(pmt_ids, g.shape[0])]


def gain_drift_pe(pe_waveform: np.ndarray, gain) -> np.ndarray:
    """Scale a pe-charge waveform ``(C, N)`` channel by channel (signal path only).

    Unlike ``interventions.gain_drift`` (which acts on a mV trace that may
    already contain front-end noise), this scales the photoelectron charge
    *before* the front end, so downstream noise is not scaled with it.
    """
    x = np.asarray(pe_waveform, dtype=float)
    if x.ndim != 2:
        raise ValueError("pe_waveform must have shape (n_channels, n_bins).")
    g = _per_channel(gain, x.shape[0], "gain")
    if np.any(g < 0):
        raise ValueError("gain must be non-negative.")
    return x * g[:, None]


# --------------------------------------------------------------------------- timing drift


def timing_drift_hits(times, pmt_ids, offsets_ns) -> np.ndarray:
    """Add a per-PMT time offset (ns) to each hit time."""
    t = np.asarray(times, dtype=float)
    if np.ndim(offsets_ns) == 0:
        return t + float(offsets_ns)
    o = np.asarray(offsets_ns, dtype=float)
    return t + o[_check_ids(pmt_ids, o.shape[0])]


def timing_drift_trace(trace: np.ndarray, offsets_s, fs: float = FS_L) -> np.ndarray:
    """Delay each channel of a ``(C, N)`` trace by its own offset (seconds).

    Uses the phase-ramp delay of :func:`noise_module_lucid.delay.cable_delay`
    per channel (circular; keep the event away from the window edge).
    """
    x = np.asarray(trace, dtype=float)
    if x.ndim != 2:
        raise ValueError("trace must have shape (n_channels, n_samples).")
    o = _per_channel(offsets_s, x.shape[0], "offsets_s")
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        out[c] = cable_delay(x[c:c + 1], float(o[c]), fs)[0]
    return out


# --------------------------------------------------------------------------- threshold walk


def _rising_edge(tau_rise_ns: float, tau_fall_ns: float, dt_ns: float):
    """Unit-peak SPE shape on its rising edge (monotone increasing), with the time grid."""
    if tau_rise_ns <= 0 or tau_fall_ns <= 0:
        raise ValueError("SPE time constants must be positive.")
    if dt_ns <= 0:
        raise ValueError("dt_ns must be positive.")
    # p(t) = (1 - e^{-t/r}) e^{-t/f} peaks at t* = r ln(1 + f/r); normalise analytically.
    t_peak = tau_rise_ns * np.log1p(tau_fall_ns / tau_rise_ns)
    t = np.append(np.arange(0.0, t_peak, dt_ns), t_peak)

    def p(x):
        return (1.0 - np.exp(-x / tau_rise_ns)) * np.exp(-x / tau_fall_ns)

    s = p(t) / p(t_peak)
    return t, np.maximum.accumulate(s)


def leading_edge_time(charges, threshold_mv: float, *, mv_per_pe: float = SPE_MV_PER_PE,
                      tau_rise_ns: float = SPE_TAU_RISE_NS, tau_fall_ns: float = SPE_TAU_FALL_NS,
                      dt_ns: float = 0.005) -> np.ndarray:
    """Walk curve: time (ns after pulse start) at which ``q * mv_per_pe * s(t)`` first reaches the threshold.

    ``s`` is the unit-peak bi-exponential SPE shape of the preset. Returns
    ``NaN`` where the pulse never reaches the threshold (the hit is lost).
    Monotone: decreasing in charge, increasing in threshold.
    """
    if threshold_mv <= 0:
        raise ValueError("threshold_mv must be positive.")
    if mv_per_pe <= 0:
        raise ValueError("mv_per_pe must be positive.")
    q = np.asarray(charges, dtype=float)
    t, s = _rising_edge(tau_rise_ns, tau_fall_ns, dt_ns)
    level = threshold_mv / (np.where(q > 0, q, np.nan) * mv_per_pe)   # required fraction of peak
    out = np.full(q.shape, np.nan)
    ok = np.isfinite(level) & (level <= s[-1])
    out[ok] = np.interp(level[ok], s, t)
    return out


def walk_readout_times(times, charges, threshold_mv: float, **kw) -> np.ndarray:
    """Readout time of each hit for a leading-edge discriminator at ``threshold_mv``.

    ``t_readout = t + leading_edge_time(q, threshold)``; ``NaN`` for lost hits.
    Use it to build a reference readout from photon arrival times.
    """
    return np.asarray(times, dtype=float) + leading_edge_time(charges, threshold_mv, **kw)


def threshold_walk_hits(times, charges, threshold_ref_mv: float, threshold_new_mv: float, *,
                        drop_subthreshold_reference: bool = False, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Move the discriminator threshold from ``threshold_ref_mv`` to ``threshold_new_mv``.

    ``times`` are the reference readout times (hits that crossed the
    reference threshold). Returns ``(new_times, keep)``: each surviving hit is
    shifted by ``t_c(q, new) - t_c(q, ref)`` (later for a higher threshold, and
    more so for small charges — the time-walk), and ``keep`` is False for hits
    that no longer cross. Lowering the threshold cannot recover hits that are
    absent from the reference list; only the timing of existing hits moves.

    A hit below the reference threshold is a contract violation (it could not
    have been recorded); it raises unless ``drop_subthreshold_reference``.
    """
    t = np.asarray(times, dtype=float)
    q = np.asarray(charges, dtype=float)
    if t.shape != q.shape:
        raise ValueError("times and charges must have the same shape.")
    t_ref = leading_edge_time(q, threshold_ref_mv, **kw)
    bad = ~np.isfinite(t_ref)
    if bad.any() and not drop_subthreshold_reference:
        raise ValueError(f"{int(bad.sum())} hits are below the reference threshold; "
                         "pass drop_subthreshold_reference=True to discard them.")
    t_new = leading_edge_time(q, threshold_new_mv, **kw)
    keep = np.isfinite(t_new) & ~bad
    out = np.full(t.shape, np.nan)
    out[keep] = t[keep] + (t_new[keep] - t_ref[keep])
    return out, keep


def leading_edge_times(trace: np.ndarray, threshold_mv: float, fs: float = FS_L) -> np.ndarray:
    """First upward threshold crossing per channel of a mV trace, in ns (linear interpolation).

    ``NaN`` for channels that never cross. A sample already at or above the
    threshold at index 0 gives time 0.
    """
    x = np.asarray(trace, dtype=float)
    if x.ndim != 2:
        raise ValueError("trace must have shape (n_channels, n_samples).")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    dt_ns = 1e9 / fs
    above = x >= threshold_mv
    out = np.full(x.shape[0], np.nan)
    has = above.any(axis=1)
    idx = np.argmax(above, axis=1)
    for c in np.flatnonzero(has):
        i = idx[c]
        if i == 0:
            out[c] = 0.0
            continue
        x0, x1 = x[c, i - 1], x[c, i]
        frac = (threshold_mv - x0) / (x1 - x0) if x1 != x0 else 0.0
        out[c] = (i - 1 + frac) * dt_ns
    return out


# --------------------------------------------------------------------------- module loss


def dead_channels_from_groups(groups: Sequence[np.ndarray], dead_group_ids: Iterable[int]) -> np.ndarray:
    """Sorted channel indices of the dead crates/boards."""
    ids = list(dead_group_ids)
    for g in ids:
        if g < 0 or g >= len(groups):
            raise ValueError(f"group id {g} out of range (have {len(groups)} groups).")
    if not ids:
        return np.array([], dtype=int)
    return np.unique(np.concatenate([np.asarray(groups[g], dtype=int) for g in ids]))


def module_loss_trace(trace: np.ndarray, dead_channels) -> np.ndarray:
    """Zero the dead channels of a ``(C, N)`` waveform (no signal, no noise read out)."""
    x = np.array(trace, dtype=float, copy=True)
    if x.ndim != 2:
        raise ValueError("trace must have shape (n_channels, n_samples).")
    d = np.asarray(dead_channels, dtype=int)
    if d.size and (d.min() < 0 or d.max() >= x.shape[0]):
        raise ValueError("dead channel index out of range.")
    x[d] = 0.0
    return x


def module_loss_hits(pmt_ids, dead_channels) -> np.ndarray:
    """Keep-mask that removes every hit on a dead channel."""
    return ~np.isin(np.asarray(pmt_ids), np.asarray(dead_channels, dtype=int))


# --------------------------------------------------------------------------- registry

#: Declared detector-response N families. Severity grids are development
#: choices (unregistered); the R-2 rule fixes the origin.
RESPONSE_FAMILIES: dict[str, dict] = {
    "efficiency_loss": {
        "origin": "N_struct", "forms": ("hits", "pe_counts", "lucid_config"),
        "description": "binomial photon thinning: QE or collection efficiency drops after training",
        "severity": {"parameter": "survival", "grid": [0.95, 0.90, 0.80, 0.60], "scope": "per crate"},
    },
    "gain_drift": {
        "origin": "N_struct", "forms": ("hits", "pe_waveform"),
        "description": "charge scale changes after training; hit count unchanged",
        "severity": {"parameter": "gain", "grid": [0.80, 0.90, 1.10, 1.25], "scope": "per crate"},
    },
    "timing_drift": {
        "origin": "N_struct", "forms": ("hits", "trace"),
        "description": "a per-channel timing calibration constant moves",
        "severity": {"parameter": "offset_ns", "grid": [0.5, 1.0, 2.0, 5.0], "scope": "per crate"},
    },
    "threshold_walk": {
        "origin": "N_struct", "forms": ("hits",),
        "description": "discriminator threshold changes: charge-dependent time-walk and lost small hits",
        "severity": {"parameter": "threshold_new_pe", "reference_pe": 0.25,
                     "grid": [0.35, 0.5, 0.75, 1.0], "scope": "detector-wide or per crate"},
    },
    "module_loss": {
        "origin": "N_struct", "forms": ("hits", "trace"),
        "description": "one or more crates/boards dead",
        "severity": {"parameter": "n_dead_crates", "grid": [1, 2, 4], "scope": "crate"},
    },
}

#: Proposed perturbations that are **not** N under R-2, with where they belong.
EXCLUDED: dict[str, dict[str, str]] = {
    "static_qe_spread": {
        "category": "reference",
        "reason": "a per-PMT QE spread present at training time is part of the detector definition; "
                  "put it in the reference cell (LUCiD per-PMT QE array), not in an N family",
    },
    "angular_acceptance_model": {
        "category": "M",
        "reason": "a different acceptance shape is simulation mis-modelling, not a corrupted "
                  "measurement; excluded from both claims and from fitting/calibration (D-M open)",
    },
    "medium_scattering_absorption": {
        "category": "M",
        "reason": "optical-medium mis-modelling is not acquisition corruption; excluded (D-M open)",
    },
    "front_end_electronics_noise": {
        "category": "frozen (R-1)",
        "reason": "removed by hit-finding thresholds before reconstruction: harm ~ 0 and visible to a "
                  "generic monitor; the interventions.INTERVENTIONS families stay but enter no claim cell",
    },
}


def describe_response() -> dict[str, str]:
    """Name -> origin for every declared response family."""
    return {name: spec["origin"] for name, spec in RESPONSE_FAMILIES.items()}
