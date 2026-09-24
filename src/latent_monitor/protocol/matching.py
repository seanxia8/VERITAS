# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Hard matched N/S contrasts and the joint operational decision table (TWO_CLAIM_REVISION_PLAN §4 I7).

**Hard contrasts.** Claim 1's primary estimand is on a frozen *hard*
evaluation set: S windows and N windows whose *generic* signatures overlap.
Matching is 1:1 without replacement on declared matching variables
(standardised by the clean pool) inside a declared caliper; unmatched windows
are retained in the inclusive set and their fraction reported. Matching is a
declared conditional design that removes the easy cases; it is not a causal
identification result.

**Joint operational table.** A good conditional N/S classifier is not a
deployed decision rule. The table applies the whole chain — detect (window
alarm ≥ threshold) → abstain (novelty above the conformal threshold) → attribute
(N/S) — to every evaluation window of every category (clean, N, S, mixture,
unknown) and reports the decision distribution per category, so quiet clean
windows, alarmed unknowns and mislabelled mixtures are all visible.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MATCHING_VARIABLES = ("in_rms", "in_kurtosis", "z_mahalanobis", "gr_output_maha")


def hard_match(features: dict[str, np.ndarray], contract: np.ndarray, clean_mask: np.ndarray, *, variables=DEFAULT_MATCHING_VARIABLES,
               caliper: float = 1.0, seed: int = 0) -> dict:
    """1:1 nearest-neighbour matching of S windows to N windows in standardised generic-signature space.

    Returns the matched row indices (N and S), the retained fractions, the
    distribution of pair distances and the count of overlap failures (S windows
    with no N window inside the caliper).
    """
    contract = np.asarray(contract, dtype=object)
    F = np.column_stack([np.asarray(features[v], dtype=float) for v in variables])
    clean = np.asarray(clean_mask, dtype=bool)
    if clean.sum() < 3:
        raise ValueError("need ≥ 3 clean windows to standardise the matching variables")
    mu = np.nanmean(F[clean], axis=0); sd = np.nanstd(F[clean], axis=0) + 1e-9
    Z = np.where(np.isfinite(F), (F - mu) / sd, 0.0)
    iS = np.where(contract == "S")[0]; iN = np.where(contract == "N")[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(iS)                                # match order does not favour any family
    free = np.ones(len(iN), dtype=bool)
    pairs, dists, failures = [], [], 0
    for s in order:
        if not free.any():
            failures += 1; continue
        d = np.linalg.norm(Z[iN] - Z[s], axis=1)
        d[~free] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper:
            pairs.append((int(iN[j]), int(s))); dists.append(float(d[j])); free[j] = False
        else:
            failures += 1
    matched_N = np.array([p[0] for p in pairs], dtype=int); matched_S = np.array([p[1] for p in pairs], dtype=int)
    return {"variables": list(variables), "caliper": caliper, "n_S": int(len(iS)), "n_N": int(len(iN)), "n_pairs": len(pairs),
            "retained_fraction_S": len(pairs) / max(len(iS), 1), "retained_fraction_N": len(pairs) / max(len(iN), 1),
            "overlap_failures_S": failures, "pair_distance_median": float(np.median(dists)) if dists else float("nan"),
            "rows_N": matched_N, "rows_S": matched_S, "rows": np.concatenate([matched_N, matched_S]) if pairs else np.array([], dtype=int),
            "note": "1:1 nearest neighbour without replacement inside the caliper, standardised by the clean pool; a declared "
                    "conditional design, not a causal identification"}


def joint_decision_table(window_alarm: np.ndarray, alarm_threshold: float, abstain: np.ndarray, attribution: np.ndarray,
                         category: np.ndarray) -> dict:
    """Decision distribution per true category after detect → abstain → attribute.

    Decisions: ``quiet`` (alarm below threshold), ``abstain`` (alarmed, novelty above the conformal threshold),
    ``N`` / ``S`` (alarmed, retained, attributed). Correctness per category: clean → quiet; N → N; S → S;
    mixture → N or S (both acceptable, reported separately); unknown → abstain.
    """
    a = np.asarray(window_alarm, dtype=float); ab = np.asarray(abstain, dtype=bool)
    att = np.asarray(attribution, dtype=object); cat = np.asarray(category, dtype=object)
    decision = np.where(a < alarm_threshold, "quiet", np.where(ab, "abstain", att)).astype(object)
    correct_map = {"clean": ("quiet",), "N": ("N",), "S": ("S",), "mixture": ("N", "S"), "unknown": ("abstain",)}
    table = {}
    for c in ("clean", "N", "S", "mixture", "unknown"):
        sel = cat == c
        n = int(sel.sum())
        if n == 0:
            table[c] = {"n": 0, "note": "no window of this category in the evaluation set"}
            continue
        dist = {d: int(np.sum(decision[sel] == d)) for d in ("quiet", "abstain", "N", "S")}
        ok = np.isin(decision[sel], correct_map[c])
        table[c] = {"n": n, "decisions": dist, "correct_rate": float(np.mean(ok)), "correct_decisions": list(correct_map[c])}
    overall = float(np.mean([np.isin(decision[i], correct_map.get(cat[i], ())) for i in range(len(cat))])) if len(cat) else float("nan")
    return {"alarm_threshold_window": float(alarm_threshold), "per_category": table, "overall_correct_rate": overall,
            "note": "whole-chain decision rule; conditional N/S macro-F1 is not this table"}
