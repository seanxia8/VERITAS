# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Origin versus harm, and the EF/EV namespace (plan §8.3 R0, R2.5).

Two axes that the old single ``moved`` label conflated:

* **origin** — which declared mechanism produced the window: ``N_cov`` (Σ̂ ≠ Σ,
  covariance-type), ``N_struct`` (gain drift, channel loss, jitter: N by
  contract), ``G`` (geometry), ``S_in_span`` (supported-but-rare physics),
  ``S_support`` (outside the excited support), ``mixture``, ``constructed`` (the
  designed output-null / output-aligned families), ``unknown`` (undeclared: an
  abstention target, never in fitting or calibration), ``clean``.
* **harm** — whether the scientific consequence K crossed the declared
  threshold κ_m: ``benign`` / ``harmful`` / ``undefined`` (κ_m pending, or K
  not measurable for the cell).

A clean rare event can be harmful and a mild corruption benign; C2 scores the
first axis, C4 the second.

**Namespace.** The proposal's **E** is an *evaluation-contract fault* (a
pipeline bug; gate E0). The controlled-variable table's **E** is *event /
physics variation* (a support move). New schemas write ``EF`` and ``EV``
respectively and never a bare ``E``. Likewise ``kappa_cond`` is the condition
number κ(Σ̂⁻¹Σ) and ``kappa_m`` the harm threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ORIGIN = ("clean", "N_cov", "N_struct", "G", "S_in_span", "S_support", "mixture", "constructed", "unknown")
HARM = ("benign", "harmful", "undefined")

#: the contract class each origin belongs to (what the C2 macro-F1 is scored over)
CONTRACT_OF_ORIGIN = {
    "clean": "clean", "N_cov": "N", "N_struct": "N", "G": "G", "S_in_span": "S", "S_support": "S",
    "mixture": "mixture", "constructed": "constructed", "unknown": "U",
}

#: ``tier1.Cell.moved`` (and the 6 Sep artifacts) → origin. The old labels are never rewritten.
LEGACY_MOVED_TO_ORIGIN = {
    "none": "clean", "sigma_cov": "N_cov", "sigma_struct": "N_struct", "geometry": "G",
    "event": "S_support", "event_in_span": "S_in_span", "designed": "constructed", "mixture": "mixture",
}

NAMESPACE = {
    "EF": "evaluation-contract fault — proposal §2 'E', gate E0; a deterministic pipeline bug",
    "EV": "event / physics variation — EXPERIMENT_DESIGN §III.1 'E'; a training-support move (origin S_*)",
    "kappa_cond": "condition number κ(Σ̂⁻¹Σ) of the assumed-versus-realised covariance ratio",
    "kappa_m": "scientific harm threshold on the consequence K (per arm; pending until declared)",
}


@dataclass(frozen=True)
class CellLabel:
    cell: str
    origin: str
    family: str                 # e.g. "corr_up", "line_pickup", "double_pulse"
    declared: bool = True       # False ⇒ abstention target; excluded from fitting / tuning / calibration
    severity: float | None = None
    harm: str = "undefined"
    legacy_moved: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in ORIGIN:
            raise ValueError(f"unknown origin {self.origin!r}; one of {ORIGIN}")
        if self.harm not in HARM:
            raise ValueError(f"unknown harm {self.harm!r}; one of {HARM}")
        if self.origin == "unknown" and self.declared:
            raise ValueError("an 'unknown' origin cannot be declared")

    @property
    def contract(self) -> str:
        return CONTRACT_OF_ORIGIN[self.origin]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contract"] = self.contract
        return d


def label_from_legacy(cell_label: str, moved: str, *, declared: bool = True, severity: float | None = None) -> CellLabel:
    """Map a ``tier1``-style ``(label, moved)`` pair onto the new axes without touching the old strings.

    ``declared=False`` marks the family as an *unknown* (abstention target): its
    origin becomes ``unknown`` for every protocol purpose while ``legacy_moved``
    keeps the generator's own label for the record.
    """
    if moved not in LEGACY_MOVED_TO_ORIGIN:
        raise KeyError(f"no legacy mapping for moved={moved!r}")
    family = cell_label.split(":", 1)[1] if ":" in cell_label else cell_label
    origin = LEGACY_MOVED_TO_ORIGIN[moved] if declared else "unknown"
    return CellLabel(cell=cell_label, origin=origin, family=family, declared=declared, severity=severity, legacy_moved=moved)
