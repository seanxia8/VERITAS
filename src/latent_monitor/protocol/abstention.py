# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Abstention end to end (TWO_CLAIM_REVISION_PLAN §4 I1).

The rule: abstain when a *feature-novelty* score (the training-support novelty
of ``latent_monitor.support``, or any alarm-time score the caller names) exceeds
a threshold set by split-conformal calibration on **clean calibration windows
only**. Under exchangeability of those windows with deployment *clean*
windows, the retained fraction of clean windows is ≥ 1 − α in expectation
(the finite-sample bound of split conformal). That is the only guarantee: it
says nothing about arbitrary unknown families, whose rejection is measured
empirically here (unknown-vs-known AUROC, risk–coverage curve) and never
assumed.

Inputs never include unknown-family windows at calibration; unknown families
appear only in the evaluation set (``protocol.splits`` enforces this).

Reported: unknown AUROC; the risk–coverage curve and its AUC (selective risk =
error rate among retained windows, where a retained *unknown* window counts as
an error because no known label is correct for it); retained coverage at the
conformal threshold; retained-known macro-F1 and counts at declared coverage
points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arms import macro_f1_report
from .consequence import weighted_auroc


@dataclass
class ConformalNovelty:
    alpha: float
    threshold: float
    n_calibration: int
    finite_sample_bound: float          # ≥ 1 − α − 1/(n+1) retained clean windows in expectation
    score_name: str
    note: str = ("split conformal on clean calibration windows; guarantee holds only under exchangeability of clean "
                 "windows and does not cover unknown families")

    @classmethod
    def fit(cls, clean_scores: np.ndarray, alpha: float, score_name: str) -> "ConformalNovelty":
        s = np.asarray(clean_scores, dtype=float); s = s[np.isfinite(s)]
        n = len(s)
        if n < 2:
            raise ValueError("need ≥ 2 clean calibration scores")
        if not 0 < alpha < 1:
            raise ValueError("alpha in (0, 1)")
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        thr = float(np.sort(s)[k - 1])
        return cls(alpha=alpha, threshold=thr, n_calibration=n, finite_sample_bound=float(1 - alpha - 1.0 / (n + 1)), score_name=score_name)

    def abstain(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return (s > self.threshold) | ~np.isfinite(s)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "threshold": self.threshold, "n_calibration": self.n_calibration,
                "finite_sample_clean_retention_bound": self.finite_sample_bound, "score": self.score_name, "note": self.note}


def unknown_auroc(novelty: np.ndarray, is_unknown: np.ndarray) -> dict:
    """Does the novelty score rank unknown-family windows above declared ones? (evaluation windows only)"""
    return weighted_auroc(novelty, is_unknown)


def risk_coverage(novelty: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, is_unknown: np.ndarray, classes,
                  coverage_points=(0.5, 0.8, 0.9, 1.0)) -> dict:
    """Selective risk versus retained coverage, sorting windows by novelty (least novel retained first).

    Risk at coverage c = error rate among the retained windows, where a retained
    unknown-family window is always an error. Returns the curve, its AUC (lower
    is better), and at each declared coverage point the retained-known macro-F1
    and the counts of retained known / unknown windows.
    """
    nov = np.asarray(novelty, dtype=float); yt = np.asarray(y_true, dtype=object); yp = np.asarray(y_pred, dtype=object)
    unk = np.asarray(is_unknown, dtype=bool)
    n = len(nov)
    order = np.argsort(np.where(np.isfinite(nov), nov, np.inf), kind="mergesort")
    err = np.where(unk, 1.0, (yt != yp).astype(float))[order]
    cum_err = np.cumsum(err)
    k = np.arange(1, n + 1)
    cov = k / n
    risk = cum_err / k
    auc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    points = {}
    for c in coverage_points:
        m = max(1, int(np.floor(c * n)))
        keep = order[:m]
        known = keep[~unk[keep]]
        rep = macro_f1_report(yt[known], yp[known], classes) if len(known) else {"macro_f1": float("nan"), "undefined_reason": "no retained known window"}
        points[str(c)] = {"coverage": m / n, "risk": float(risk[m - 1]), "n_retained": int(m), "n_retained_known": int(len(known)),
                          "n_retained_unknown": int(m - len(known)), "retained_known_macro_f1": rep["macro_f1"],
                          "undefined_reason": rep.get("undefined_reason")}
    return {"coverage": cov.tolist(), "risk": risk.tolist(), "risk_coverage_auc": auc, "points": points, "n": int(n)}


@dataclass
class AbstentionReport:
    conformal: dict
    unknown_auroc: dict
    coverage_at_threshold: float
    clean_retention_at_threshold: float | None
    risk_coverage: dict
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"conformal": self.conformal, "unknown_auroc": self.unknown_auroc, "coverage_at_threshold": self.coverage_at_threshold,
                "clean_retention_at_threshold": self.clean_retention_at_threshold, "risk_coverage": self.risk_coverage, "counts": self.counts}


def evaluate_abstention(conf: ConformalNovelty, novelty_eval: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                        is_unknown: np.ndarray, classes, novelty_clean_eval: np.ndarray | None = None) -> AbstentionReport:
    abst = conf.abstain(novelty_eval)
    unk = np.asarray(is_unknown, dtype=bool)
    counts = {"n_eval": int(len(novelty_eval)), "n_unknown": int(unk.sum()), "n_abstained": int(abst.sum()),
              "n_unknown_abstained": int((abst & unk).sum()), "n_known_abstained": int((abst & ~unk).sum()),
              "n_unknown_retained": int((~abst & unk).sum())}
    clean_ret = None if novelty_clean_eval is None else float(np.mean(~conf.abstain(novelty_clean_eval)))
    return AbstentionReport(conformal=conf.to_dict(), unknown_auroc=unknown_auroc(novelty_eval, unk),
                            coverage_at_threshold=float(np.mean(~abst)), clean_retention_at_threshold=clean_ret,
                            risk_coverage=risk_coverage(novelty_eval, y_true, y_pred, unk, classes), counts=counts)
