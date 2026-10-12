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

Pure numpy; no LUCiD import. Photons can come from LUCiD (after gate A0), from
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

    def moved_from(self, ref: "ReadoutParams") -> list[str]:
        """Names of the fields that differ from ``ref`` (a valid cell moves exactly one)."""
        out = []
        for k in self.__dataclass_fields__:
            a, b = getattr(self, k), getattr(ref, k)
            same = np.array_equal(np.asarray(a), np.asarray(b)) if not isinstance(a, tuple) else a == b
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
                  seed: int = 0) -> Digits:
    """Run detected photons through the declared readout chain (see the module docstring)."""
    p = params or ReadoutParams()
    ids = np.asarray(photon_sensor, dtype=int)
    t = np.asarray(photon_time, dtype=float)
    if ids.shape != t.shape:
        raise ValueError("photon_sensor and photon_time must have the same shape.")
    if ids.size and (ids.min() < 0 or ids.max() >= n_channels):
        raise ValueError("photon sensor index out of range.")
    rng = stage_rngs(seed)
    counts: dict[str, int] = {"photons_in": int(ids.size)}

    # 1. efficiency (per-channel survival)
    survival = _per_channel(p.survival, n_channels, "survival")
    keep = efficiency_loss_hits(ids, survival, rng["efficiency"])
    ids, t = ids[keep], t[keep]
    counts["photons_detected"] = int(ids.size)

    # 2. digitisation
    n_pe = np.bincount(ids, minlength=n_channels)
    t_first = np.full(n_channels, np.inf)
    np.minimum.at(t_first, ids, t)
    hit = n_pe > 0
    counts["pmts_hit"] = int(hit.sum())

    # 3. charge: one normal per channel, drawn for every channel (paired across cells)
    z = rng["charge"].standard_normal(n_channels)
    q = np.clip(n_pe + np.sqrt(n_pe) * p.spe_sigma * z, 0.0, None)

    # 4. gain
    gain = _per_channel(p.gain, n_channels, "gain")
    chan = np.arange(n_channels)
    q = gain_drift_hits(q, chan, gain)

    # 5. discriminator with walk, corrected by the training-time calibration curve
    thr_mv = p.threshold_pe * p.mv_per_pe
    cal_mv = p.walk_calibration_pe * p.mv_per_pe
    walk = leading_edge_time(np.where(hit, q, 0.0), thr_mv, mv_per_pe=p.mv_per_pe)
    walk_cal = leading_edge_time(np.where(hit, q, 0.0), cal_mv, mv_per_pe=p.mv_per_pe)
    fired = hit & np.isfinite(walk)
    counts["lost_threshold"] = int((hit & ~fired).sum())
    # A threshold *below* the calibration threshold fires on pulses the calibration curve
    # never covered: those hits keep their full, uncorrected walk (declared, counted).
    uncal = fired & ~np.isfinite(walk_cal)
    counts["uncalibrated_walk"] = int(uncal.sum())
    corr = np.where(np.isfinite(walk_cal), walk_cal, 0.0)
    t_read = t_first + np.where(fired, walk - corr, 0.0)

    # 6. timing calibration
    offsets = _per_channel(p.timing_offsets_ns, n_channels, "timing_offsets_ns")
    t_read = timing_drift_hits(t_read, chan, offsets)

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
