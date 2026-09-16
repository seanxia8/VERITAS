# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Claim 2 — does a task-sensitive score rank scientific harm better than a committed generic score?

Primary (TWO_CLAIM_REVISION_PLAN §2):

    ΔAUROC_harm = AUROC_{K ≥ κ_m}(task-sensitive score) − AUROC_{K ≥ κ_m}(committed generic score)

over all held-out intervention **cells** under a declared cell weighting, with
a paired interval whose outer resampling unit is the cell (the unit the claim
generalises over), event groups nested inside. Supporting: both absolute
AUROCs, AUPRC with prevalence, the four alarm–harm quadrants with counts, the
missed-harm rate at the alert budget, false rejection of benign valid-rare
cells, breakdowns, and the conditional strong-alarm triage as secondary.

Levels are stated on every quantity (I12): a *window* is one observed record;
a *cell* is one intervention (family × severity × seed) and the unit of
generalisation; K and the alarm are aggregated to cell level as declared
(``cell_aggregate``).

Definitions the caller must supply (never invented here):

* ``K``       the consequence per unit in declared units, baseline-normalised
              by the caller; an *independent physical endpoint* is preferred
              to the diagnostic weighted residual.
* ``kappa_m`` a :class:`HarmThreshold`. ``status="pending"`` makes every harm
              label undefined and confirmatory mode fail closed.

Inference is refused — the interval is reported as *descriptive* — when the
number of outer units is below :data:`MIN_OUTER_UNITS_FOR_INFERENCE`
(a declared, unfrozen constant).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

#: below this many outer resampling units an interval is descriptive, never inferential (unfrozen; PREREGISTRATION §6)
MIN_OUTER_UNITS_FOR_INFERENCE = 10


class PendingThresholdError(RuntimeError):
    """κ_m has not been declared; harm labels are undefined."""


class InvalidWeightsError(ValueError):
    """Weights must be finite, non-negative, of the right length and of positive total mass."""


def _weights(w: np.ndarray | None, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n)
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise InvalidWeightsError(f"weights have shape {w.shape}, expected ({n},)")
    if not np.all(np.isfinite(w)):
        raise InvalidWeightsError("weights contain non-finite values")
    if np.any(w < 0):
        raise InvalidWeightsError("weights contain negative values")
    if w.sum() <= 0:
        raise InvalidWeightsError("weights have zero total mass")
    return w


@dataclass(frozen=True)
class HarmThreshold:
    """κ_m for one arm: ``declared`` (frozen, cited requirement), ``provisional_dev`` (development only), ``pending``."""

    value: float | None
    status: str
    units: str = "ratio to reference consequence"
    source: str = ""
    arm: str = ""
    level: str = "cell"                 # the level at which K is compared with κ_m

    def __post_init__(self) -> None:
        if self.status not in ("declared", "provisional_dev", "pending"):
            raise ValueError("status must be declared | provisional_dev | pending")
        if self.status == "pending" and self.value is not None:
            raise ValueError("a pending threshold carries no value")
        if self.status != "pending" and self.value is None:
            raise ValueError("a declared or provisional threshold needs a value")

    @property
    def usable_in(self) -> tuple[str, ...]:
        return {"declared": ("dev", "confirmatory"), "provisional_dev": ("dev",), "pending": ()}[self.status]

    def to_dict(self) -> dict:
        return asdict(self)


def harm_labels(K: np.ndarray, kappa_m: HarmThreshold) -> np.ndarray:
    """Boolean harm labels; raises :class:`PendingThresholdError` when κ_m is pending. NaN K → undefined."""
    if kappa_m.status == "pending":
        raise PendingThresholdError(f"κ_m is pending for arm {kappa_m.arm or '?'}; harm labels are undefined")
    K = np.asarray(K, dtype=float)
    out = np.zeros(K.shape, dtype=object)
    out[np.isnan(K)] = "undefined"
    out[~np.isnan(K) & (K >= kappa_m.value)] = "harmful"
    out[~np.isnan(K) & (K < kappa_m.value)] = "benign"
    return out


def weighted_auroc(score: np.ndarray, positive: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """AUROC by weighted pairwise comparison (ties count ½). ``nan`` with a reason when a class is empty."""
    s = np.asarray(score, dtype=float); y = np.asarray(positive, dtype=bool)
    w = _weights(weights, len(s))
    ok = ~np.isnan(s)
    s, y, w = s[ok], y[ok], w[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return {"auroc": float("nan"), "n_pos": n1, "n_neg": n0,
                "undefined_reason": "one class is empty; the ranking endpoint is undefined for this set"}
    sp, sn = s[y], s[~y]
    wp, wn = w[y], w[~y]
    gt = (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :])
    W = wp[:, None] * wn[None, :]
    return {"auroc": float(np.sum(gt * W) / np.sum(W)), "n_pos": n1, "n_neg": n0, "undefined_reason": None}


def auprc(score: np.ndarray, positive: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Weighted average precision with **tied scores aggregated into one threshold** (permutation invariant)."""
    s = np.asarray(score, dtype=float); y = np.asarray(positive, dtype=bool)
    w = _weights(weights, len(s))
    ok = ~np.isnan(s); s, y, w = s[ok], y[ok], w[ok]
    if len(s) == 0 or y.sum() == 0 or (~y).sum() == 0:
        prev = float(np.sum(w[y]) / np.sum(w)) if len(w) else float("nan")
        return {"auprc": float("nan"), "prevalence": prev, "undefined_reason": "one class is empty"}
    uniq = np.unique(s)[::-1]                          # thresholds, descending
    tp_at = np.array([np.sum(w[(s == t) & y]) for t in uniq])
    fp_at = np.array([np.sum(w[(s == t) & ~y]) for t in uniq])
    tp = np.cumsum(tp_at); fp = np.cumsum(fp_at)
    precision = tp / np.maximum(tp + fp, 1e-300)
    recall = tp / tp[-1]
    prev_r = np.concatenate([[0.0], recall[:-1]])
    return {"auprc": float(np.sum((recall - prev_r) * precision)), "prevalence": float(tp[-1] / np.sum(w)), "undefined_reason": None}


def cell_weights(cells: Sequence[Mapping], scheme: str = "uniform_cell") -> np.ndarray:
    """Declared cell-weighting scheme: ``uniform_cell`` (each cell weight 1) or ``declared`` (``cell["weight"]``)."""
    if scheme == "uniform_cell":
        return np.ones(len(cells))
    if scheme == "declared":
        return _weights(np.asarray([float(c["weight"]) for c in cells]), len(cells))
    raise ValueError("scheme must be uniform_cell | declared")


def alarm_harm_matrix(alarm: np.ndarray, harm: np.ndarray, alarm_threshold: float, weights: np.ndarray | None = None) -> dict:
    """All four quadrants (+ undefined), with counts and weights, at a declared cell-level alarm threshold."""
    a = np.asarray(alarm, dtype=float); h = np.asarray(harm, dtype=object)
    w = _weights(weights, len(a))
    strong = a >= alarm_threshold
    defined = h != "undefined"
    q = {}
    for name, sel in (("low_alarm_benign", ~strong & (h == "benign")), ("low_alarm_harmful", ~strong & (h == "harmful")),
                      ("strong_alarm_benign", strong & (h == "benign")), ("strong_alarm_harmful", strong & (h == "harmful"))):
        q[name] = {"n": int(sel.sum()), "weight": float(w[sel].sum())}
    q["undefined"] = {"n": int((~defined).sum()), "weight": float(w[~defined].sum())}
    q["alarm_threshold"] = float(alarm_threshold)
    q["level"] = "cell"
    return q


def all_cell_ranking(alarm: np.ndarray, harm: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Absolute all-cell ranking of one score — no conditioning on the alarm. Supporting for Claim 2."""
    h = np.asarray(harm, dtype=object)
    defined = h != "undefined"
    a = np.asarray(alarm, dtype=float)[defined]
    y = (h[defined] == "harmful")
    w = _weights(weights, len(h))[defined]
    out = weighted_auroc(a, y, w)
    out.update({f"auprc_{k}" if k != "auprc" else k: v for k, v in auprc(a, y, w).items()})
    out["n_undefined"] = int((~defined).sum())
    out["n_cells"] = int(len(h))
    out["level"] = "cell"
    return out


def paired_delta_auroc(score_task: np.ndarray, score_generic: np.ndarray, harm: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Claim-2 primary point estimate: AUROC(task-sensitive) − AUROC(committed generic) on the same cells."""
    a = all_cell_ranking(score_task, harm, weights)
    g = all_cell_ranking(score_generic, harm, weights)
    d = a["auroc"] - g["auroc"] if a["undefined_reason"] is None and g["undefined_reason"] is None else float("nan")
    return {"delta_auroc": d, "auroc_task_sensitive": a["auroc"], "auroc_generic_committed": g["auroc"],
            "n_pos": a["n_pos"], "n_neg": a["n_neg"], "undefined_reason": a["undefined_reason"] or g["undefined_reason"], "level": "cell"}


def missed_harm_rate_at_budget(alarm: np.ndarray, harm: np.ndarray, alarm_threshold: float, weights: np.ndarray | None = None) -> dict:
    """Fraction of harmful cells *below* the alert threshold."""
    m = alarm_harm_matrix(alarm, harm, alarm_threshold, weights)
    harmful = m["low_alarm_harmful"]["weight"] + m["strong_alarm_harmful"]["weight"]
    return {"missed_harm_rate": (m["low_alarm_harmful"]["weight"] / harmful) if harmful > 0 else float("nan"),
            "n_harmful": m["low_alarm_harmful"]["n"] + m["strong_alarm_harmful"]["n"], "level": "cell",
            "undefined_reason": None if harmful > 0 else "no harmful cell in this set"}


def valid_rare_cell_rejection(alarm: np.ndarray, harm: np.ndarray, is_valid_rare: np.ndarray, alarm_threshold: float) -> dict:
    """Among valid-rare **cells** (origin S_*) that are benign, the fraction the cell-level alarm would reject."""
    a = np.asarray(alarm, dtype=float); h = np.asarray(harm, dtype=object); v = np.asarray(is_valid_rare, dtype=bool)
    sel = v & (h == "benign")
    n = int(sel.sum())
    return {"valid_rare_rejection_rate": float(np.mean(a[sel] >= alarm_threshold)) if n else float("nan"),
            "n_valid_rare_benign": n, "level": "cell",
            "undefined_reason": None if n else "no benign valid-rare cell in this set"}


#: pass-1 name, kept for the 16 Sep artifact readers; the quantity was always cell-level
valid_rare_event_rejection = valid_rare_cell_rejection


def valid_rare_window_rejection(window_alarm: np.ndarray, window_threshold: float, is_valid_rare_benign_cell: np.ndarray) -> dict:
    """The **window**-level estimator: among windows of benign valid-rare cells, the fraction above the window threshold."""
    a = np.asarray(window_alarm, dtype=float); sel = np.asarray(is_valid_rare_benign_cell, dtype=bool)
    n = int(sel.sum())
    return {"valid_rare_window_rejection_rate": float(np.mean(a[sel] >= window_threshold)) if n else float("nan"),
            "n_windows": n, "level": "window", "undefined_reason": None if n else "no window from a benign valid-rare cell"}


def conditional_triage(alarm: np.ndarray, harm: np.ndarray, alarm_threshold: float, weights: np.ndarray | None = None) -> dict:
    """Secondary: among strong-alarm cells only. Reports how many cells the conditioning dropped."""
    a = np.asarray(alarm, dtype=float); h = np.asarray(harm, dtype=object)
    w = _weights(weights, len(a))
    strong = a >= alarm_threshold
    defined = h != "undefined"
    sel = strong & defined
    y = h[sel] == "harmful"
    out = weighted_auroc(a[sel], y, w[sel]) if sel.any() else {"auroc": float("nan"), "n_pos": 0, "n_neg": 0, "undefined_reason": "no strong-alarm cell"}
    out["conditional_harm_risk"] = float(np.sum(w[sel][y]) / np.sum(w[sel])) if sel.any() else float("nan")
    out["n_dropped_by_conditioning"] = int((~strong & defined).sum())
    out["n_dropped_harmful"] = int((~strong & (h == "harmful")).sum())
    out["level"] = "cell"
    return out


def breakdown(alarm: np.ndarray, harm: np.ndarray, by: np.ndarray, weights: np.ndarray | None = None) -> dict[str, dict]:
    """Per-stratum ranking; a stratum with one class is reported as undefined, not dropped."""
    by = np.asarray(by, dtype=object)
    w = _weights(weights, len(by))
    out = {}
    for key in sorted(set(by.tolist()), key=str):
        sel = by == key
        out[str(key)] = all_cell_ranking(np.asarray(alarm)[sel], np.asarray(harm, dtype=object)[sel], w[sel])
    return out


# ------------------------------------------------------------------ level aggregation and thresholds

def cell_aggregate(window_values: np.ndarray, cell_of: np.ndarray, how: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Aggregate window-level values to cell level with a declared rule (``median`` or ``mean``). Returns (cells, values)."""
    c = np.asarray(cell_of, dtype=object); v = np.asarray(window_values, dtype=float)
    cells = sorted(set(c.tolist()), key=str)
    f = {"median": np.nanmedian, "mean": np.nanmean}[how]
    return np.array(cells, dtype=object), np.array([f(v[c == k]) for k in cells])


def cell_alarm_threshold(clean_window_scores: np.ndarray, cell_size: int, far: float, how: str = "median",
                         n_draws: int = 2000, seed: int = 0) -> dict:
    """A **cell-level** alert threshold calibrated on clean windows (I12).

    Draws pseudo-cells of ``cell_size`` clean windows (with replacement),
    aggregates each with the declared rule, and takes the (1 − FAR) quantile of
    the aggregate. Reports the resolution the clean count allows.
    """
    s = np.asarray(clean_window_scores, dtype=float); s = s[np.isfinite(s)]
    if len(s) < 3:
        raise ValueError("need at least three clean windows")
    rng = np.random.default_rng(seed)
    f = {"median": np.median, "mean": np.mean}[how]
    agg = np.array([f(rng.choice(s, size=cell_size, replace=True)) for _ in range(n_draws)])
    return {"threshold": float(np.quantile(agg, 1.0 - far)), "far": far, "aggregate": how, "cell_size": int(cell_size),
            "n_clean_windows": int(len(s)), "n_draws": n_draws,
            "note": "pseudo-cell bootstrap of clean windows; the underlying window count bounds the resolution — "
                    f"1/{len(s)} = {1.0/len(s):.3f} per window",
            "resolvable": len(s) * far >= 1.0}


def far_precision(n_clean: int, n_false: int, far_budget: float) -> dict:
    """Realised FAR with a Clopper–Pearson (binomial, independent windows) interval and the budget's resolvability (I13)."""
    from scipy.stats import beta
    n, x = int(n_clean), int(n_false)
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return {"n_clean_windows": n, "n_false_alerts": x, "realised_far": x / n if n else float("nan"),
            "ci95": [lo, hi], "budget": far_budget, "resolvable_in_one_step": n * far_budget >= 1.0,
            "assumption": "independent, non-overlapping clean windows (binomial); temporal dependence widens the interval",
            "note": "a run-level false-alert rate over several monitors needs a multiplicity-adjusted budget; not computed here"}


# ------------------------------------------------------------------ resampling

def bootstrap_groups(stat: Callable[[np.ndarray], float], groups: np.ndarray, seeds: np.ndarray | None = None,
                     n_boot: int = 200, seed: int = 0, quantiles: tuple[float, float] = (0.025, 0.975)) -> dict:
    """Bootstrap over event groups (× perturbation seeds, crossed). One level; for Claim 2 use :func:`bootstrap_hierarchical`."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    us = None if seeds is None else np.unique(np.asarray(seeds))
    vals, failed = [], 0
    for _ in range(n_boot):
        gs = rng.choice(ug, size=len(ug), replace=True)
        counts_g = {g: c for g, c in zip(*np.unique(gs, return_counts=True))}
        rep = np.array([counts_g.get(g, 0) for g in groups])
        if us is not None:
            ss = rng.choice(us, size=len(us), replace=True)
            counts_s = {s: c for s, c in zip(*np.unique(ss, return_counts=True))}
            rep = rep * np.array([counts_s.get(s, 0) for s in np.asarray(seeds)])
        idx = np.repeat(np.arange(len(groups)), rep)
        if len(idx) == 0:
            failed += 1; continue
        try:
            v = float(stat(idx))
        except Exception:
            failed += 1; continue
        if np.isnan(v):
            failed += 1; continue
        vals.append(v)
    point = float(stat(np.arange(len(groups))))
    if not vals:
        return {"point": point, "low": float("nan"), "high": float("nan"), "n_boot": n_boot, "n_boot_effective": 0, "n_failed": failed,
                "n_groups": int(len(ug)), "n_seeds": None if us is None else int(len(us)), "units": "event_group", "inference": "undefined"}
    return {"point": point, "low": float(np.quantile(vals, quantiles[0])), "high": float(np.quantile(vals, quantiles[1])),
            "n_boot": n_boot, "n_boot_effective": len(vals), "n_failed": failed, "n_groups": int(len(ug)),
            "n_seeds": None if us is None else int(len(us)),
            "units": "event_group × perturbation_seed (crossed)" if us is not None else "event_group",
            "inference": "event-sampling only (conditional on the fixed cells)"}


def bootstrap_hierarchical(stat: Callable[[np.ndarray], float], outer: np.ndarray, inner: np.ndarray, *,
                           n_boot: int = 200, seed: int = 0, quantiles: tuple[float, float] = (0.025, 0.975),
                           min_outer: int = MIN_OUTER_UNITS_FOR_INFERENCE, model_seeds: np.ndarray | None = None) -> dict:
    """Two-level bootstrap: resample **outer** units (cells / families / perturbation seeds) with replacement,
    then event groups (**inner**) within each selected outer unit. Model seeds, when given, are a third,
    outermost level (nested: each model seed carries its own cells).

    ``stat(idx)`` receives an index array into the units. When the number of
    outer units is below ``min_outer`` the interval is returned but labelled
    ``descriptive`` — confirmatory inference about held-out interventions is
    then impossible and the report must say so.
    """
    rng = np.random.default_rng(seed)
    outer = np.asarray(outer, dtype=object); inner = np.asarray(inner)
    ms = None if model_seeds is None else np.asarray(model_seeds)
    uo = np.array(sorted(set(outer.tolist()), key=str), dtype=object)
    um = None if ms is None else np.unique(ms)
    n_outer = len(uo) if um is None else len(um)
    vals, failed = [], 0
    for _ in range(n_boot):
        idx_parts = []
        if um is not None:
            chosen_models = rng.choice(um, size=len(um), replace=True)
        else:
            chosen_models = [None]
        for m in chosen_models:
            sel_m = np.ones(len(outer), bool) if m is None else (ms == m)
            uo_m = np.array(sorted(set(outer[sel_m].tolist()), key=str), dtype=object)
            if len(uo_m) == 0:
                continue
            chosen = rng.choice(uo_m, size=len(uo_m), replace=True)
            for o in chosen:
                rows = np.where(sel_m & (outer == o))[0]
                gi = inner[rows]
                ug = np.unique(gi)
                gs = rng.choice(ug, size=len(ug), replace=True)
                counts = {g: c for g, c in zip(*np.unique(gs, return_counts=True))}
                rep = np.array([counts.get(g, 0) for g in gi])
                idx_parts.append(np.repeat(rows, rep))
        idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
        if len(idx) == 0:
            failed += 1; continue
        try:
            v = float(stat(idx))
        except Exception:
            failed += 1; continue
        if np.isnan(v):
            failed += 1; continue
        vals.append(v)
    point = float(stat(np.arange(len(outer))))
    base = {"point": point, "n_boot": n_boot, "n_boot_effective": len(vals), "n_failed": failed,
            "n_outer": int(len(uo)), "n_inner_groups": int(len(np.unique(inner))),
            "n_model_seeds": None if um is None else int(len(um)),
            "units": ("model_seed ⊃ " if um is not None else "") + "outer (cell/family/seed) ⊃ event_group (nested)",
            "min_outer_for_inference": min_outer}
    if not vals:
        return {**base, "low": float("nan"), "high": float("nan"), "inference": "undefined"}
    base.update({"low": float(np.quantile(vals, quantiles[0])), "high": float(np.quantile(vals, quantiles[1]))})
    if n_outer < min_outer:
        base["inference"] = f"descriptive only — {n_outer} outer units < {min_outer}; confirmatory inference about held-out interventions is impossible at this size"
    else:
        base["inference"] = "outer-unit bootstrap; supports a statement about interventions drawn like these"
    return base
