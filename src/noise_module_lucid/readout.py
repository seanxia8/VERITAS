# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The declared PMT readout chain on which the detector-response N families act.

One function, :func:`readout_chain`, turns detected photons into per-PMT digits
``(sensor, charge, time)`` — the observable LUCiD's own reconstruction uses —
through a fixed sequence of stages. A response N family moves **one**
parameter of :class:`ReadoutParams` away from the reference; everything else,
including every random draw, is held fixed, so a cell is event-paired with the
reference.

Stages (in order)
-----------------
1. **efficiency** — per-photon binomial thinning with per-channel survival
   (``response.efficiency_loss_hits``);
2. **digitisation** — per PMT: photoelectron count and first-arrival time;
3. **charge** — ``n + sqrt(n) * spe_sigma * z`` with one fixed normal ``z`` per
   channel (paired across cells), clipped at 0;
4. **gain** — per-channel charge scale (``response.gain_drift_hits``);
5. **discriminator** — leading-edge at ``threshold_pe`` × the SPE amplitude: a
   digit whose pulse never reaches it is lost; the readout time carries the
   walk at the *actual* threshold and is corrected with the walk curve
   calibrated at ``walk_calibration_pe`` (fixed at training time), so the
   reference has zero residual walk and a threshold change leaves a
   charge-dependent residual (``response.leading_edge_time``);
6. **timing calibration** — per-channel offsets (``response.timing_drift_hits``);
7. **dead channels** — removed (``response.module_loss_hits``).

Because the discriminator acts on the measured (gain-scaled) amplitude, a gain
drift also loses small hits at a fixed-mV threshold — the coupling the review
asked to be modelled rather than idealised away.

NumPy/SciPy; no LUCiD import. Research use is clarified in issue 6133100584. Photons can come from LUCiD with explicit stage ownership, from
:mod:`noise_module_lucid.toy` (development only), or from any other source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from .presets import SPE_MV_PER_PE
from .response import (
    efficiency_loss_hits,
    gain_drift_hits,
    leading_edge_time,
    module_loss_hits,
    timing_drift_hits,
)

_STAGES = ("efficiency", "charge")


@dataclass(frozen=True)
class ReadoutParams:
    """Readout-chain parameters. Per-channel fields accept a scalar or an ``(n_channels,)`` array."""

    survival: Any = 1.0
    gain: Any = 1.0
    timing_offsets_ns: Any = 0.0
    threshold_pe: float = 0.25
    walk_calibration_pe: float = 0.25
    spe_sigma: float = 0.3
    dead_channels: tuple[int, ...] = ()
    mv_per_pe: float = SPE_MV_PER_PE

    qe_curve: Any = None
    collection_efficiency: Any = 1.0
    walk_curve: Any = None
    reference_walk: Any = None
    timing_width: Any = None
    timing_width_factor: Any = 1.0
    reference_offsets_ns: Any = 0.0

    def moved_from(self, ref: "ReadoutParams") -> list[str]:
        """Names of the fields that differ from ``ref`` (a valid cell moves exactly one)."""
        out = []
        for k in self.__dataclass_fields__:
            a, b = getattr(self, k), getattr(ref, k)
            same = (a == b) if hasattr(a, "__dataclass_fields__") or hasattr(b, "__dataclass_fields__") else np.array_equal(np.asarray(a), np.asarray(b))
            if not same:
                out.append(k)
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


@dataclass
class Digits:
    """Per-PMT digits after the readout chain (one entry per surviving PMT)."""

    sensor: np.ndarray            # (D,) int
    charge: np.ndarray            # (D,) float, pe (measured, gain-scaled)
    time: np.ndarray              # (D,) float, ns
    n_pe_true: np.ndarray         # (D,) int, detected photoelectrons before charge smearing
    counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.sensor.size)


def _per_channel(v, n: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have shape ({n},).")
    return a


def stage_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Independent generators per random stage, so moving one stage never shifts another's draws."""
    children = np.random.SeedSequence(int(seed)).spawn(len(_STAGES))
    return {name: np.random.default_rng(s) for name, s in zip(_STAGES, children)}


def readout_chain(photon_sensor, photon_time, n_channels: int, params: ReadoutParams | None = None,
                  seed: int = 0, *, photon_batch=None) -> Digits:
    """Readout preserving the legacy call, with optional explicit PhotonBatch.

    New incident-photon mode uses bounded QE/CE and identity-keyed streams.
    Polynomial timing mode uses a continuous PE peak discriminator; legacy
    threshold_walk retains its sampled-template normalization. A Gaussian
    digit width is a synthetic approximation, not a second per-photon TTS.
    """
    p = params or ReadoutParams()
    raw_ids = np.asarray(photon_sensor)
    if raw_ids.size and not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("photon sensor IDs must be integers")
    if not isinstance(n_channels, (int, np.integer)) or n_channels <= 0:
        raise ValueError("positive channel count required")
    dead = np.asarray(p.dead_channels)
    if dead.size and (len(np.unique(dead)) != len(dead) or not np.issubdtype(dead.dtype, np.integer) or np.any((dead < 0) | (dead >= n_channels))):
        raise ValueError("invalid dead channels")
    ids = raw_ids.astype(int)
    t = np.asarray(photon_time, dtype=float)
    if ids.shape != t.shape:
        raise ValueError("photon_sensor and photon_time must have the same shape.")
    if ids.size and (ids.min() < 0 or ids.max() >= n_channels):
        raise ValueError("photon sensor index out of range.")
    if ids.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("finite one-dimensional photon times required")
    for name in ("gain", "timing_width_factor"):
        value = _per_channel(getattr(p, name), n_channels, name)
        if np.any(~np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"invalid {name}")
    if not np.all(np.isfinite([p.threshold_pe, p.walk_calibration_pe, p.spe_sigma, p.mv_per_pe])) or min(p.threshold_pe, p.walk_calibration_pe, p.spe_sigma) < 0 or p.mv_per_pe <= 0:
        raise ValueError("invalid discriminator/charge parameters")
    if photon_batch is not None:
        photon_batch.validate(n_channels)
        if not np.array_equal(ids, photon_batch.sensor_id) or not np.array_equal(t, photon_batch.arrival_time_ns):
            raise ValueError("photon batch and legacy arguments disagree")
        if p.timing_width is not None and "TTS" in photon_batch.applied_effects:
            raise ValueError("double TTS forbidden")
    elif p.qe_curve is not None or p.timing_width is not None:
        raise ValueError("QE/TTS response requires explicit photon-stage batch")
    rng = stage_rngs(seed)
    counts: dict[str, int] = {"photons_in": int(ids.size)}

    # 1. efficiency (per-channel survival)
    survival = _per_channel(p.survival, n_channels, "survival")
    from .quantum_efficiency import probabilities, keyed_uniform
    probabilities(survival)
    ce = probabilities(_per_channel(p.collection_efficiency, n_channels, "collection_efficiency"))
    detect = survival[ids]
    if photon_batch is not None and photon_batch.stage == "incident_photons":
        if p.qe_curve is None or photon_batch.wavelength_nm is None:
            raise ValueError("incident photons require wavelength QE and wavelengths")
        detect = detect * p.qe_curve(photon_batch.wavelength_nm) * ce[ids]
    elif p.qe_curve is not None or np.any(ce != 1):
        raise ValueError("cannot apply QE/CE to post-QE hits")
    keep = (keyed_uniform(seed, photon_batch.event_id, photon_batch.photon_id, "detection") < probabilities(detect)) if photon_batch is not None else efficiency_loss_hits(ids, survival, rng["efficiency"])
    ids, t = ids[keep], t[keep]
    counts["photons_detected"] = int(ids.size)

    # 2. digitisation
    n_pe = np.bincount(ids, minlength=n_channels)
    t_first = np.full(n_channels, np.inf)
    np.minimum.at(t_first, ids, t)
    hit = n_pe > 0
    counts["pmts_hit"] = int(hit.sum())

    # 3. charge: one normal per channel, drawn for every channel (paired across cells)
    if photon_batch is None:
        z = rng["charge"].standard_normal(n_channels)
    else:
        from scipy.special import ndtri
        z = ndtri(keyed_uniform(seed, photon_batch.event_id, np.arange(n_channels), "charge"))
    q = np.clip(n_pe + np.sqrt(n_pe) * p.spe_sigma * z, 0.0, None)

    # 4. gain
    gain = _per_channel(p.gain, n_channels, "gain")
    chan = np.arange(n_channels)
    q = gain_drift_hits(q, chan, gain)

    # 5. discriminator with walk, corrected by the training-time calibration curve
    thr_mv = p.threshold_pe * p.mv_per_pe
    cal_mv = p.walk_calibration_pe * p.mv_per_pe
    template_kw = {"template_fs": None} if p.walk_curve is not None else {}
    walk = leading_edge_time(np.where(hit, q, 0.0), thr_mv, mv_per_pe=p.mv_per_pe, **template_kw)
    walk_cal = leading_edge_time(np.where(hit, q, 0.0), cal_mv, mv_per_pe=p.mv_per_pe, **template_kw)
    fired = hit & np.isfinite(walk)
    counts["lost_threshold"] = int((hit & ~fired).sum())
    # A threshold *below* the calibration threshold fires on pulses the calibration curve
    # never covered: those hits keep their full, uncorrected walk (declared, counted).
    uncal = fired & ~np.isfinite(walk_cal)
    counts["uncalibrated_walk"] = int(uncal.sum())
    corr = np.where(np.isfinite(walk_cal), walk_cal, 0.0)
    t_read = t_first + np.where(fired, walk - corr, 0.0)

    # New bounded T-Q response is distinct from historical threshold walk.
    if (p.walk_curve is None) != (p.reference_walk is None):
        raise ValueError("both cell and frozen reference walk curves required")
    if p.walk_curve is not None:
        from .timing_response import evaluate_walk
        t_read[fired] += evaluate_walk(p.walk_curve, q[fired], chan[fired]) - evaluate_walk(p.reference_walk, q[fired], chan[fired])
    if p.timing_width is not None:
        from scipy.special import ndtri
        jitter = ndtri(keyed_uniform(seed, photon_batch.event_id, chan, "timing"))
        widths = p.timing_width(q[fired]) * _per_channel(p.timing_width_factor, n_channels, "timing_width_factor")[fired]
        t_read[fired] += widths * jitter[fired]
    elif np.any(_per_channel(p.timing_width_factor, n_channels, "timing_width_factor") != 1):
        raise ValueError("width factor requires a timing width model")
    counts["lost_time_window"] = 0  # this chain does not impose an event window

    # 6. timing calibration
    offsets = _per_channel(p.timing_offsets_ns, n_channels, "timing_offsets_ns")
    ref_offsets = _per_channel(p.reference_offsets_ns, n_channels, "reference_offsets_ns")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(ref_offsets)):
        raise ValueError("finite timing offsets required")
    t_read = timing_drift_hits(t_read, chan, offsets-ref_offsets)

    # 7. dead channels
    alive = module_loss_hits(chan, np.asarray(p.dead_channels, dtype=int))
    counts["lost_dead"] = int((fired & ~alive).sum())
    sel = fired & alive
    counts["digits"] = int(sel.sum())
    return Digits(sensor=chan[sel], charge=q[sel], time=t_read[sel], n_pe_true=n_pe[sel], counts=counts)


def with_change(ref: ReadoutParams, **change) -> ReadoutParams:
    """``ref`` with exactly the given fields changed (one field for a valid N cell)."""
    out = replace(ref, **change)
    moved = out.moved_from(ref)
    if len(moved) > 1:
        raise ValueError(f"a response cell moves one parameter; this moves {moved}.")
    return out
