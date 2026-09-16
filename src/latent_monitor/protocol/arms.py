# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The comparison arms for Claim 1 (TWO_CLAIM_REVISION_PLAN §2, §4 I3, I6, I9).

One classifier (L2-regularised multinomial logistic regression on standardised
features), one tuning grid, identical splits, identical calibration windows,
identical side information — for every arm. Every arm passes the alarm-time
contract (declarative *and* data-flow) before it is fitted.

Primary arms (Claim 1):

    generic_rich        every operational generic quantity (input quality, outputs, the
                        uncertainty proxy, final z, noise-only statistics) *plus* the same
                        reference-distance transforms the intermediate arm enjoys, applied to
                        generic quantities (input, output, pre-output, z-energy splits,
                        out-of-span, support novelty)
    intermediate_only   strictly internal hooks (channel, token) — no whitened/raw input, no
                        final z, no pre-output, no output duplicates
    full_intermediate   generic_rich + intermediate_only

Controls: ``generic_rich_matched`` (generic_rich + a quadratic expansion of its
own scalars, so the two arms compared have the same feature count);
``full_intermediate_drop_channel`` / ``_drop_token`` (hook-drop).

Development diagnostics (the pass-1 arms, kept so the 16 Sep artifact can be
read; never the Claim-1 comparison): ``input_only``, ``output_uncertainty``,
``final_embedding``, ``all_generic``, ``all_generic_no_noise``, ``noise_only``,
``full_layerwise_legacy``.

Partitions: fitted on ``attribution_train``, λ chosen on ``development``, scored
once on ``evaluation``; ``reference_fit`` groups are never supervised examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .availability import AlarmTimeContract, FeatureBatch, FeatureManifest

GENERIC_GROUPS = ("input_quality", "output", "uncertainty", "final_embedding", "noise_only")
ARMS: dict[str, tuple[str, ...]] = {
    # Claim-1 arms
    "generic_rich": GENERIC_GROUPS + ("generic_rich",),
    "intermediate_only": ("intermediate",),
    "full_intermediate": GENERIC_GROUPS + ("generic_rich", "intermediate"),
    # controls
    "generic_rich_matched": GENERIC_GROUPS + ("generic_rich", "generic_quadratic"),
    # development diagnostics (pass-1 arms)
    "input_only": ("input_quality",),
    "output_uncertainty": ("output", "uncertainty"),
    "final_embedding": ("final_embedding",),
    "all_generic": GENERIC_GROUPS,
    "all_generic_no_noise": ("input_quality", "output", "uncertainty", "final_embedding"),
    "noise_only": ("noise_only",),
    "full_layerwise_legacy": GENERIC_GROUPS + ("layerwise_legacy",),
}
PRIMARY_ARMS = ("generic_rich", "intermediate_only", "full_intermediate")
CONTROL_ARMS = ("generic_rich_matched", "full_intermediate_drop_channel", "full_intermediate_drop_token")
DIAGNOSTIC_ARMS = ("input_only", "output_uncertainty", "final_embedding", "all_generic", "all_generic_no_noise", "noise_only", "full_layerwise_legacy")

#: the one tuning grid every arm gets (chosen on the development partition, never on evaluation)
TUNING_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

#: predeclared but UNFROZEN reading margins (proposal §4; PREREGISTRATION §1)
MARGINS = {"benefit_min_delta": 0.10, "equivalence_half_width": 0.05}

FIT_PARTITION, TUNE_PARTITION, EVAL_PARTITION = "attribution_train", "development", "evaluation"


class ArmFitError(RuntimeError):
    """A partition is empty, a required class is absent, or the optimiser failed."""


def arm_feature_names(manifest: FeatureManifest, arm: str) -> list[str]:
    if arm in ARMS:
        names: list[str] = []
        for g in ARMS[arm]:
            names.extend(manifest.group(g))
        return names
    if arm.startswith("full_intermediate_drop_"):
        hook = arm[len("full_intermediate_drop_"):]
        base = arm_feature_names(manifest, "full_intermediate")
        dropped = [n for n in base if not n.startswith(f"im_{hook}_")]
        if len(dropped) == len(base):
            raise KeyError(f"hook-drop arm {arm!r}: no intermediate feature of hook {hook!r} to drop")
        return dropped
    raise KeyError(f"unknown arm {arm!r}")


def check_arms(manifest: FeatureManifest, arms: Sequence[str] = tuple(ARMS) + CONTROL_ARMS[1:], batch: FeatureBatch | None = None) -> dict[str, dict]:
    """Every arm passes the alarm-time contract or the run stops here."""
    c = AlarmTimeContract(manifest)
    return c.check_arms({a: arm_feature_names(manifest, a) for a in arms}, batch)


# ------------------------------------------------------------------ classifier

@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        return cls(mean=np.nanmean(X, axis=0), std=np.nanstd(X, axis=0) + 1e-9)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return np.where(np.isnan(Z), 0.0, np.clip(Z, -20, 20))


@dataclass
class LogisticArm:
    classes: tuple[str, ...]
    W: np.ndarray
    b: np.ndarray
    scaler: Standardizer
    lam: float
    features: tuple[str, ...] = ()
    converged: bool = True
    n_iter: int = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler(X) @ self.W + self.b
        Z = Z - Z.max(axis=1, keepdims=True)
        p = np.exp(Z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes, dtype=object)[np.argmax(self.predict_proba(X), axis=1)]


def fit_logistic(X: np.ndarray, y: Sequence[str], classes: Sequence[str], lam: float, features: Sequence[str] = (),
                 maxiter: int = 500) -> LogisticArm:
    X = np.asarray(X, dtype=float)
    classes = tuple(classes)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ArmFitError("empty design matrix")
    if not np.isfinite(lam) or lam < 0:
        raise ArmFitError(f"invalid λ {lam}")
    idx = {c: i for i, c in enumerate(classes)}
    y = list(y)
    missing = [c for c in classes if c not in set(y)]
    if missing:
        raise ArmFitError(f"training data lacks class(es) {missing}; every declared class must be present")
    unknown = sorted(set(y) - set(classes))
    if unknown:
        raise ArmFitError(f"training labels contain undeclared class(es) {unknown}")
    yi = np.array([idx[c] for c in y])
    scaler = Standardizer.fit(X)
    Z = scaler(X)
    n, d = Z.shape
    k = len(classes)
    Y = np.zeros((n, k)); Y[np.arange(n), yi] = 1.0
    cw = n / (k * np.maximum(np.bincount(yi, minlength=k), 1))
    sw = cw[yi]

    def unpack(theta):
        return theta[: d * k].reshape(d, k), theta[d * k:]

    def f(theta):
        W, b = unpack(theta)
        L = Z @ W + b
        L = L - L.max(axis=1, keepdims=True)
        logp = L - np.log(np.exp(L).sum(axis=1, keepdims=True))
        nll = -np.sum(sw * np.sum(Y * logp, axis=1)) / n
        P = np.exp(logp)
        G = (P - Y) * sw[:, None] / n
        gW = Z.T @ G + lam * W
        gb = G.sum(axis=0)
        return nll + 0.5 * lam * np.sum(W**2), np.concatenate([gW.ravel(), gb])

    res = minimize(f, np.zeros(d * k + k), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    if not np.all(np.isfinite(res.x)):
        raise ArmFitError(f"optimiser returned non-finite parameters: {res.message}")
    converged = bool(res.success) or ("ABNORMAL" not in str(res.message).upper() and np.all(np.isfinite(res.fun)))
    if not converged:
        raise ArmFitError(f"optimiser failed: {res.message}")
    W, b = unpack(res.x)
    return LogisticArm(classes=classes, W=W, b=b, scaler=scaler, lam=float(lam), features=tuple(features),
                       converged=bool(res.success), n_iter=int(res.nit))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> float:
    """Macro-F1 over the *declared* classes.

    Policy (I9): every declared class contributes. A class present in the truth
    but never predicted has F1 = 0; a class absent from the truth makes the
    endpoint **undefined** (``nan``) because its recall does not exist — use
    :func:`macro_f1_report` for the reason. Nothing is silently omitted.
    """
    return macro_f1_report(y_true, y_pred, classes)["macro_f1"]


def macro_f1_report(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> dict:
    yt = np.asarray(list(y_true), dtype=object); yp = np.asarray(list(y_pred), dtype=object)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred differ in length")
    per: dict[str, float | None] = {}
    absent = []
    for c in classes:
        tp = int(np.sum((yt == c) & (yp == c))); fp = int(np.sum((yt != c) & (yp == c))); fn = int(np.sum((yt == c) & (yp != c)))
        if tp + fn == 0:
            absent.append(c); per[c] = None
            continue
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn)
        per[c] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    if absent or len(yt) == 0:
        return {"macro_f1": float("nan"), "per_class": per, "n": int(len(yt)),
                "undefined_reason": "empty evaluation set" if len(yt) == 0 else f"class(es) {absent} absent from the truth; recall undefined"}
    return {"macro_f1": float(np.mean([per[c] for c in classes])), "per_class": per, "n": int(len(yt)), "undefined_reason": None}


def confusion(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> dict:
    yt = np.asarray(list(y_true), dtype=object); yp = np.asarray(list(y_pred), dtype=object)
    return {t: {p: int(np.sum((yt == t) & (yp == p))) for p in classes} for t in classes}


@dataclass
class ArmResult:
    arm: str
    features: tuple[str, ...]
    lam: float
    f1_tune: float
    f1_eval: float
    status: str                       # operational | privileged
    y_pred_eval: np.ndarray
    proba_eval: np.ndarray
    n_train: int
    n_tune: int
    n_eval: int
    model: LogisticArm
    extra: dict = field(default_factory=dict)


def fit_arm(arm: str, manifest: FeatureManifest, batch: FeatureBatch, y: np.ndarray, part: np.ndarray,
            classes: Sequence[str], grid: Sequence[float] = TUNING_GRID) -> ArmResult:
    """Fit on ``attribution_train``, choose λ on ``development``, score once on ``evaluation``.

    The same procedure, grid and partitions for every arm; the batch must be
    locked and every feature's sources within the alarm-time set.
    """
    names = arm_feature_names(manifest, arm)
    status = AlarmTimeContract(manifest).check(arm, names, batch)
    X = batch.matrix(names)
    y = np.asarray(y, dtype=object); part = np.asarray(part, dtype=object)
    tr = part == FIT_PARTITION; dv = part == TUNE_PARTITION; ev = part == EVAL_PARTITION
    for nm, sel in (("attribution_train", tr), ("development", dv), ("evaluation", ev)):
        if not sel.any():
            raise ArmFitError(f"arm {arm!r}: partition {nm!r} is empty")
    if set(part.tolist()) & {"reference_fit"}:
        # reference-fit rows may be present in the table but are never supervised examples
        pass
    best = None
    for lam in grid:
        m = fit_logistic(X[tr], y[tr], classes, lam, names)
        f = macro_f1(y[dv], m.predict(X[dv]), classes)
        if np.isnan(f):
            raise ArmFitError(f"arm {arm!r}: macro-F1 undefined on the tuning partition ({macro_f1_report(y[dv], m.predict(X[dv]), classes)['undefined_reason']})")
        if best is None or f > best[0]:
            best = (f, lam, m)
    f_tune, lam, model = best
    pred = model.predict(X[ev])
    return ArmResult(arm=arm, features=tuple(names), lam=lam, f1_tune=f_tune, f1_eval=macro_f1(y[ev], pred, classes),
                     status=status["status"], y_pred_eval=pred, proba_eval=model.predict_proba(X[ev]),
                     n_train=int(tr.sum()), n_tune=int(dv.sum()), n_eval=int(ev.sum()), model=model,
                     extra={"privileged_features": status["privileged_features"], "grid": list(grid),
                            "converged": model.converged, "n_iter": model.n_iter, "n_features": len(names),
                            "confusion_eval": confusion(y[ev], pred, classes)})


#: below this many evaluation windows (or event groups) no reading is issued — declared, unfrozen
MIN_WINDOWS_FOR_READING = 30
MIN_GROUPS_FOR_READING = 10


def read_delta(delta: float, low: float, high: float, margins: Mapping[str, float] = MARGINS, *,
               n_windows: int | None = None, n_groups: int | None = None) -> str:
    """The predeclared three-way reading of a paired ΔF1 with its interval (plus 'detriment').

    A reading needs a set large enough for the interval to mean something: below
    the declared minima, or when the bootstrap interval is degenerate (zero
    width), the result is ``inconclusive`` with the reason attached.
    """
    if np.isnan(delta) or np.isnan(low) or np.isnan(high):
        return "inconclusive (undefined)"
    if n_windows is not None and n_windows < MIN_WINDOWS_FOR_READING:
        return f"inconclusive (n_windows {n_windows} < {MIN_WINDOWS_FOR_READING})"
    if n_groups is not None and n_groups < MIN_GROUPS_FOR_READING:
        return f"inconclusive (n_groups {n_groups} < {MIN_GROUPS_FOR_READING})"
    if high - low <= 0:
        return "inconclusive (degenerate interval)"
    if low > 0 and delta >= margins["benefit_min_delta"]:
        return "benefit"
    if -margins["equivalence_half_width"] <= low and high <= margins["equivalence_half_width"]:
        return "equivalence"
    if high < 0:
        return "detriment"
    return "inconclusive"
