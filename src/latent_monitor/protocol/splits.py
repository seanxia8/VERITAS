# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Event-group splits (plan §8.3 R2.4; §0.5 resampling units).

The unit is the *event group*: one underlying event / injection identity and
every variant of it — every geometry it is replayed through, every noise
replicate, every corruption applied to it. A group is assigned to exactly one
partition, so a corrupted copy of an evaluation event can never sit in the
reference fit, and the classifier can never learn an event's identity.

Partitions (all five, separately; I6): ``reference_fit`` (clean subject /
reference / null fitting — **never** supervised examples), ``attribution_train``
(supervised N/S examples for the classifier arms), ``development``
(hyper-parameter tuning, threshold choice), ``calibration`` (clean windows for
the false-alert budget and the conformal threshold), ``evaluation`` (scored once).

Declared hold-outs: families, severity ranges and perturbation seeds listed
in advance go to ``evaluation`` only. Undeclared (unknown) families never
enter ``reference_fit``, ``development`` or ``calibration``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

PARTITIONS = ("reference_fit", "attribution_train", "development", "calibration", "evaluation")


class OverlapError(RuntimeError):
    """An event group appears in more than one partition."""


@dataclass(frozen=True)
class GroupSplit:
    partitions: dict[str, np.ndarray]          # partition -> sorted array of event-group ids
    held_out_families: tuple[str, ...] = ()
    held_out_severities: tuple[tuple[float, float], ...] = ()   # closed intervals
    held_out_seeds: tuple[int, ...] = ()
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def partition_of(self, group_id: int) -> str | None:
        for name, ids in self.partitions.items():
            if np.searchsorted(ids, group_id) < len(ids) and ids[np.searchsorted(ids, group_id)] == group_id:
                return name
        return None

    def to_dict(self) -> dict:
        return {"partitions": {k: v.tolist() for k, v in self.partitions.items()},
                "held_out_families": list(self.held_out_families),
                "held_out_severities": [list(s) for s in self.held_out_severities],
                "held_out_seeds": list(self.held_out_seeds), "seed": self.seed, "meta": dict(self.meta)}


def assert_disjoint(split: GroupSplit) -> None:
    """Raise :class:`OverlapError` if any group id sits in two partitions."""
    seen: dict[int, str] = {}
    for name, ids in split.partitions.items():
        for g in np.asarray(ids).tolist():
            if g in seen:
                raise OverlapError(f"event group {g} is in both {seen[g]!r} and {name!r}")
            seen[g] = name


def split_event_groups(group_ids: Iterable[int], fractions: Mapping[str, float], seed: int = 0, **held_out) -> GroupSplit:
    """Deterministically assign unique event groups to the four partitions by shuffled fractions."""
    ids = np.unique(np.asarray(list(group_ids), dtype=int))
    names = list(fractions)
    unknown = set(names) - set(PARTITIONS)
    if unknown:
        raise ValueError(f"unknown partitions {sorted(unknown)}; use {PARTITIONS}")
    fr = np.asarray([fractions[n] for n in names], dtype=float)
    if fr.sum() <= 0 or np.any(fr < 0):
        raise ValueError("fractions must be non-negative and sum to > 0")
    fr = fr / fr.sum()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    cuts = np.floor(np.cumsum(fr) * len(perm)).astype(int)
    parts, start = {}, 0
    for n, c in zip(names, cuts):
        parts[n] = np.sort(perm[start:c]); start = c
    if start < len(perm):                                   # rounding remainder goes to the last partition
        parts[names[-1]] = np.sort(np.concatenate([parts[names[-1]], perm[start:]]))
    split = GroupSplit(partitions=parts, seed=seed, **held_out)
    assert_disjoint(split)
    return split


def assign_partitions(records: Sequence[Mapping], split: GroupSplit) -> list[str]:
    """Partition for each record ``{"event_group", "family", "declared", "severity"?, "seed"?}``.

    Rules, in order: an undeclared family, or a held-out family / severity /
    seed, is scored **only** on groups that are themselves in ``evaluation``
    (so a corrupted copy of a reference-fit event is never scored); on any
    other group such a record is ``excluded`` — it enters nothing. Otherwise
    the record takes its group's partition. A record whose group is in no
    partition raises ``KeyError`` — nothing is scored by accident.
    """
    out = []
    for r in records:
        g = int(r["event_group"])
        part = split.partition_of(g)
        if part is None:
            raise KeyError(f"event group {g} is in no partition")
        fam = r.get("family")
        sev = r.get("severity")
        sd = r.get("seed")
        held = (fam in split.held_out_families) or (sd is not None and int(sd) in split.held_out_seeds) or any(
            sev is not None and lo <= float(sev) <= hi for lo, hi in split.held_out_severities)
        if not r.get("declared", True) or held:
            out.append("evaluation" if part == "evaluation" else "excluded")
        else:
            out.append(part)
    return out


def check_no_unknown_in_fit(records: Sequence[Mapping], assigned: Sequence[str]) -> None:
    """Raise if an undeclared record landed anywhere but ``evaluation`` (defence in depth for callers that bypass :func:`assign_partitions`)."""
    from .availability import LeakageError
    for r, p in zip(records, assigned):
        if not r.get("declared", True) and p != "evaluation":
            raise LeakageError(f"undeclared family {r.get('family')!r} assigned to {p!r}")
