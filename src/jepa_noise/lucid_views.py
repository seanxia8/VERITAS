"""LUCiD views and masks, runbook v4 §3.5.4 (WP4c).

``across``: modules ordered along a 3D Hilbert curve of their centres; per event
a random contiguous half of that order (seed-keyed namespace
``lucid/views/across/<replicate>/<event_id>/attempt/<k>``) is the target, the
rest the context; the half is redrawn up to ``MAX_REDRAWS`` times until the
target view holds ``N_MIN`` hit PMTs, then the event is excluded and counted.
``within``: context = PMTs whose **1-based** ``pmt_id_in_mpmt`` is even, target =
odd (9/10 per module), fixed for every event; a failing event is excluded and
counted (A8).

Masks depend on (geometry, event id, seed/replicate) only -- no hit, charge or
time enters the partition -- and each carries the JSON-serialisable ``key`` it
was built from; :func:`mask_from_key` recomputes the identical partition.

**Occupancy ablation** (§3.5.4, A8; recorded, off by default):
``occupancy="balanced"`` subsamples the *unhit* PMTs of the **context** view to
match its hit count; every target PMT is still queried, so the query set
reveals nothing about the target hit pattern.

Hilbert index: Skilling's *axes -> transpose* algorithm (the formulation of the
MIT-licensed ``hilbertcurve`` package, re-implemented in vectorised numpy) on an
integer grid of the fixed detector box ``[-R, R] x [-R, R] x [-H/2, H/2]`` with
``2**bits`` cells per axis; construction in ``docs/jepa/WP4C_ENCODER.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lucid_bombs import (MAX_REDRAWS, N_MIN, DetectorGeometry, ViewExclusionCounter,
                          keyed_rng)

__all__ = ["NS_VIEWS", "TOKEN_HILBERT_BITS", "MODULE_HILBERT_BITS", "OCCUPANCY_MODES",
           "SPLITS", "hilbert_index_3d", "module_hilbert_order", "token_hilbert_order",
           "ViewMask", "across_mask", "within_mask", "mask_from_key", "ViewMasker"]

NS_VIEWS = "lucid/views"
TOKEN_HILBERT_BITS = 12
MODULE_HILBERT_BITS = 10
OCCUPANCY_MODES = ("natural", "balanced")
SPLITS = ("across", "within")

_ORDER_CACHE: dict = {}


def hilbert_index_3d(points, bits: int) -> np.ndarray:
    """Hilbert distance of integer grid points, ``points`` shape ``(n, 3)``.

    Each coordinate must be in ``[0, 2**bits)``.  Returns ``(n,)`` int64
    distances, the position of each point along the curve.  Skilling (2004):
    invert the excess work of the transpose representation, Gray-encode, then
    interleave the bits back into an integer.
    """
    pts = np.ascontiguousarray(points, dtype=np.int64).copy()
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("hilbert_index_3d needs an (n, 3) integer array")
    if int(bits) < 1:
        raise ValueError("hilbert bits must be >= 1")
    if np.any(pts < 0) or np.any(pts >= (1 << int(bits))):
        raise ValueError(f"hilbert coordinates must lie in [0, 2**{int(bits)})")
    n = 3
    m = 1 << (int(bits) - 1)
    q = m
    while q > 1:
        mask = q - 1
        for i in range(n):
            bit = (pts[:, i] & q) != 0
            t = (pts[:, 0] ^ pts[:, i]) & mask
            pts[:, 0] ^= np.where(bit, mask, t)
            pts[:, i] ^= np.where(bit, 0, t)
        q >>= 1
    for i in range(1, n):
        pts[:, i] ^= pts[:, i - 1]
    flip = np.zeros(pts.shape[0], dtype=np.int64)
    q = m
    while q > 1:
        flip ^= np.where((pts[:, n - 1] & q) != 0, q - 1, 0)
        q >>= 1
    pts ^= flip[:, None]
    distance = np.zeros(pts.shape[0], dtype=np.int64)
    for b in range(int(bits)):
        for i in range(n):
            # Skilling's transpose -> integer packing (the published C code,
            # as in hilbertcurve): bit b of coordinate i sits at position
            # n*b + (n-1-i), so coordinate 0 is the most significant of a row.
            distance |= ((pts[:, i] >> b) & 1) << (n * b + (n - 1 - i))
    return distance


def _curve_order(points_m, geometry: DetectorGeometry, bits: int) -> np.ndarray:
    """Indices sorted along the curve; ties broken by the original index."""
    radius, half = float(geometry.radius_m), float(geometry.half_height_m)
    low = np.array([-radius, -radius, -half], dtype=float)
    unit = np.clip((np.asarray(points_m, dtype=float) - low) / (2.0 * -low), 0.0, 1.0)
    grid = np.clip(np.floor(unit * (1 << int(bits))).astype(np.int64), 0, (1 << int(bits)) - 1)
    return np.lexsort((np.arange(len(grid)), hilbert_index_3d(grid, bits)))


def module_hilbert_order(geometry: DetectorGeometry, bits: int = MODULE_HILBERT_BITS) -> np.ndarray:
    """Dense module indices in Hilbert order of the module centres (cached)."""
    key = ("module", geometry.digest, int(bits))
    if key not in _ORDER_CACHE:
        _ORDER_CACHE[key] = _curve_order(geometry.module_centres_m, geometry, bits)
    return _ORDER_CACHE[key].copy()


def token_hilbert_order(geometry: DetectorGeometry, bits: int = TOKEN_HILBERT_BITS) -> np.ndarray:
    """PMT token indices in Hilbert order of the PMT positions (cached)."""
    key = ("token", geometry.digest, int(bits))
    if key not in _ORDER_CACHE:
        _ORDER_CACHE[key] = _curve_order(geometry.positions_m, geometry, bits)
    return _ORDER_CACHE[key].copy()


@dataclass(frozen=True)
class ViewMask:
    """One event's partition plus the accounting and the recompute key."""

    split: str
    event_id: int
    context: np.ndarray
    target: np.ndarray
    accepted: bool
    attempts: int
    context_hits: int
    target_hits: int
    offset: int | None = None
    attempt_hits: tuple = ()
    extra_hidden: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    key: dict = field(default_factory=dict)

    @property
    def hidden(self) -> np.ndarray:
        """Positions the context encoder replaces with the mask embedding."""
        return np.concatenate([self.target, self.extra_hidden])

    def fractions(self, hit) -> dict:
        hit = np.asarray(hit, dtype=bool)
        return {"context": float(hit[self.context].mean()) if self.context.size else 0.0,
                "target": float(hit[self.target].mean()) if self.target.size else 0.0}


def _hit_array(geometry: DetectorGeometry, hit_pmt) -> np.ndarray:
    values = np.asarray(hit_pmt)
    if values.dtype == bool:
        if values.shape != (geometry.n_pmt,):
            raise ValueError("boolean hit pattern must have shape (n_pmt,)")
        return values.copy()
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("hit PMTs must be a boolean pattern or one-dimensional integers")
    hit = np.zeros(geometry.n_pmt, dtype=bool)
    if values.size:
        if values.min() < 0 or values.max() >= geometry.n_pmt:
            raise ValueError("hit PMT index out of range")
        hit[values] = True
    return hit


def _across_partition(geometry: DetectorGeometry, offset: int, n_target_modules: int,
                      order: np.ndarray):
    """Cyclic contiguous half of the module order -> (target, context, modules)."""
    n_modules = geometry.n_modules
    modules = order[(int(offset) + np.arange(int(n_target_modules))) % n_modules]
    in_target = np.zeros(n_modules, dtype=bool)
    in_target[modules] = True
    selected = in_target[geometry.module_index]
    return np.flatnonzero(selected).astype(np.int64), np.flatnonzero(~selected).astype(np.int64), modules


def across_mask(geometry: DetectorGeometry, event_id: int, hit_pmt, *, seed: int = 0,
                replicate: int = 0, n_min: int = N_MIN, max_redraws: int = MAX_REDRAWS,
                n_target_modules: int | None = None, order=None,
                module_bits: int = MODULE_HILBERT_BITS) -> ViewMask:
    """Random contiguous half of the Hilbert module order, with N_min redraws."""
    order = module_hilbert_order(geometry, module_bits) if order is None else np.asarray(order)
    n_modules = geometry.n_modules
    n_target = n_modules // 2 if n_target_modules is None else int(n_target_modules)
    if not 1 <= n_target < n_modules:
        raise ValueError("across needs at least one target and one context module")
    hit = _hit_array(geometry, hit_pmt)
    attempt_hits, offset, target = [], 0, np.empty(0, dtype=np.int64)
    for attempt in range(int(max_redraws) + 1):
        rng = keyed_rng(seed, NS_VIEWS, "across", replicate, int(event_id), "attempt", attempt)
        offset = int(rng.integers(0, n_modules))
        target, context, modules = _across_partition(geometry, offset, n_target, order)
        attempt_hits.append(int(hit[target].sum()))
        if attempt_hits[-1] >= int(n_min):
            break
    accepted = bool(attempt_hits and attempt_hits[-1] >= int(n_min))
    target, context, modules = _across_partition(geometry, offset, n_target, order)
    context_hits, target_hits = int(hit[context].sum()), int(hit[target].sum())
    key = {"split": "across", "geometry_sha256": geometry.digest, "event_id": int(event_id),
           "seed": int(seed), "replicate": int(replicate), "offset": int(offset),
           "n_target_modules": int(n_target), "n_min": int(n_min),
           "max_redraws": int(max_redraws), "module_hilbert_bits": int(module_bits),
           "occupancy": "natural", "attempts": len(attempt_hits) - 1,
           "context_hits": context_hits, "target_hits": target_hits}
    return ViewMask("across", int(event_id), context, target, accepted, len(attempt_hits) - 1,
                    context_hits, target_hits, int(offset),
                    tuple(attempt_hits), np.empty(0, dtype=np.int64), key)


def within_mask(geometry: DetectorGeometry, event_id: int, hit_pmt, *, n_min: int = N_MIN) -> ViewMask:
    """Fixed parity split: even 1-based ``pmt_id_in_mpmt`` -> context, odd -> target."""
    in_context = (geometry.pmt_id_in_mpmt % 2) == 0
    context = np.flatnonzero(in_context).astype(np.int64)
    target = np.flatnonzero(~in_context).astype(np.int64)
    hit = _hit_array(geometry, hit_pmt)
    accepted = int(hit[target].sum()) >= int(n_min)
    context_hits, target_hits = int(hit[context].sum()), int(hit[target].sum())
    key = {"split": "within", "geometry_sha256": geometry.digest, "event_id": int(event_id),
           "n_min": int(n_min), "parity": "even=context,odd=target",
           "pmt_id_in_mpmt": "1-based", "occupancy": "natural", "attempts": 0,
           "context_hits": context_hits, "target_hits": target_hits}
    return ViewMask("within", int(event_id), context, target, bool(accepted), 0,
                    context_hits, target_hits, None, (),
                    np.empty(0, dtype=np.int64), key)


def mask_from_key(geometry: DetectorGeometry, key: dict, hit_pmt=None) -> ViewMask:
    """Recompute a mask from its recorded key (and check counts if hits are given)."""
    split = key.get("split")
    hit = None if hit_pmt is None else _hit_array(geometry, hit_pmt)
    if key.get("geometry_sha256") != geometry.digest:
        raise ValueError("mask_from_key: key belongs to a different geometry")
    if split == "within":
        mask = within_mask(geometry, int(key["event_id"]),
                           np.zeros(0, dtype=np.int64) if hit is None else hit,
                           n_min=int(key.get("n_min", N_MIN)))
    elif split == "across":
        order = module_hilbert_order(geometry, int(key.get("module_hilbert_bits", MODULE_HILBERT_BITS)))
        target, context, _ = _across_partition(geometry, int(key["offset"]),
                                               int(key["n_target_modules"]), order)
        mask = ViewMask("across", int(key["event_id"]), context, target, True,
                        int(key.get("attempts", 0)), 0, 0, int(key["offset"]), (),
                        np.empty(0, dtype=np.int64), dict(key))
    else:
        raise ValueError(f"unregistered split {split!r}")
    if mask.key.get("split") != split or int(mask.key.get("event_id", -1)) != int(key["event_id"]):
        raise ValueError("mask_from_key: key does not round-trip")
    if split == "across" and int(mask.key.get("offset", -1)) != int(key["offset"]):
        raise ValueError("mask_from_key: offset does not round-trip")
    if hit is not None:
        expected = (key.get("target_hits"), key.get("context_hits"))
        found = (int(hit[mask.target].sum()), int(hit[mask.context].sum()))
        if None not in expected and tuple(expected) != found:
            raise ValueError(f"mask_from_key: recorded hit counts {expected} != recomputed {found}")
    return mask


class ViewMasker:
    """One §3.5.4 cell (split x drift setting): masks plus its own counters."""

    def __init__(self, geometry: DetectorGeometry, split: str, *, seed: int = 0,
                 replicate: int = 0, cell: str = "", n_min: int = N_MIN,
                 max_redraws: int = MAX_REDRAWS, occupancy: str = "natural",
                 n_target_modules: int | None = None, module_bits: int = MODULE_HILBERT_BITS):
        if split not in SPLITS:
            raise ValueError(f"unregistered split {split!r}")
        if occupancy not in OCCUPANCY_MODES:
            raise ValueError(f"unregistered occupancy mode {occupancy!r}")
        self.geometry, self.split, self.seed = geometry, split, int(seed)
        self.replicate, self.cell = int(replicate), cell
        self.n_min, self.max_redraws = int(n_min), int(max_redraws)
        self.occupancy = occupancy
        self.n_target_modules = n_target_modules
        self.module_bits = int(module_bits)
        self.counter = ViewExclusionCounter(n_min=self.n_min, max_redraws=self.max_redraws)
        self.fraction_context = 0.0
        self.fraction_target = 0.0
        self.n_seen = 0
        self.n_balanced = 0

    def mask(self, event_id: int, hit_pmt) -> ViewMask:
        hit = _hit_array(self.geometry, hit_pmt)
        if self.split == "across":
            mask = across_mask(self.geometry, event_id, hit, seed=self.seed,
                               replicate=self.replicate, n_min=self.n_min,
                               max_redraws=self.max_redraws,
                               n_target_modules=self.n_target_modules,
                               module_bits=self.module_bits)
            self.counter.record_across(mask.attempt_hits)
        else:
            mask = within_mask(self.geometry, event_id, hit, n_min=self.n_min)
            self.counter.record_within(mask.target_hits)
        self.n_seen += 1
        if self.occupancy == "balanced":
            hidden = self._balance(mask, hit)
            mask = ViewMask(mask.split, mask.event_id, mask.context, mask.target, mask.accepted,
                            mask.attempts, mask.context_hits, mask.target_hits, mask.offset,
                            mask.attempt_hits, hidden,
                            dict(mask.key, occupancy="balanced", n_balanced_context=int(hidden.size)))
            self.n_balanced += int(hidden.size > 0)
        if mask.accepted:
            fractions = mask.fractions(hit)
            self.fraction_context += fractions["context"]
            self.fraction_target += fractions["target"]
        return mask

    def _balance(self, mask: ViewMask, hit: np.ndarray) -> np.ndarray:
        """Subsample *unhit* context PMTs to match the context hit count (A8)."""
        unhit = mask.context[~hit[mask.context]]
        n_hit = int(hit[mask.context].sum())
        if unhit.size <= n_hit:
            return np.empty(0, dtype=np.int64)
        rng = keyed_rng(self.seed, NS_VIEWS, "balance", self.replicate, mask.event_id)
        keep = np.sort(rng.permutation(unhit.size)[:unhit.size - n_hit])
        return np.asarray(unhit[keep], dtype=np.int64)

    def report(self) -> dict:
        rates = self.counter.rates()
        n_accepted = int(rates["across_accepted"] + rates["within_accepted"])
        accepted = max(1, n_accepted)
        out = {"cell": self.cell, "split": self.split, "occupancy": self.occupancy,
               "seed": self.seed, "replicate": self.replicate,
               "geometry_sha256": self.geometry.digest,
               "n_events": self.n_seen, "n_min": self.n_min,
               "max_redraws": self.max_redraws,
               "n_accepted": n_accepted,
               "n_excluded": int(self.counter.across_excluded + self.counter.within_excluded),
               "across_redraws": int(self.counter.across_redraws),
               "across_redraw_rate": float(rates["across_redraw_rate"]),
               "across_exclusion_rate": float(rates["across_exclusion_rate"]),
               "within_exclusion_rate": float(rates["within_exclusion_rate"]),
               "context_hit_fraction": self.fraction_context / accepted,
               "target_hit_fraction": self.fraction_target / accepted,
               "n_balanced_events": self.n_balanced}
        return out
