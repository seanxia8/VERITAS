# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Adjustments that follow from a diagnosis (plan §4).

Three of the four are not gradient steps:

* ``rewhiten``          Σ-covariance: estimate Σ from noise-only records, replace Σ̂ in W.
* ``activation_patch``  Σ-structural: substitute the clean stage-k representation into the
                        perturbed forward pass; the stage whose patch recovers the consequence
                        is the causal one — *before* any stage-restricted retraining.
* ``refit_stage``       geometry: the linear analogue of a stage-restricted LoRA.
* (support shift)       nothing on weights — abstain and extend the training support.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .subject import HOOKS, Geometry, Subject
from .whitening import KroneckerWhitener, estimate_kronecker


def rewhiten(subject: Subject, noise_records: np.ndarray, psd_smoothing: int = 5) -> tuple[Subject, dict[str, Any]]:
    """Replace the whitening layer's Σ̂ with an estimate from ``noise_records`` (m, C, N).

    The decoder, encoder and heads are untouched. Returns the adjusted subject
    and κ(Σ̂_old⁻¹ Σ̂_new), which is the size of the correction applied.
    """
    W: KroneckerWhitener = subject.sigma_hat
    Sc, psd = estimate_kronecker(noise_records, W.sampling_frequency, psd_smoothing=psd_smoothing)
    kappa = W.kappa_against(Sc, psd)
    new = KroneckerWhitener(Sc, psd, W.sampling_frequency, W.n_samples, W.floor)
    return subject.with_sigma_hat(new), {"kappa_correction": kappa, "n_records": int(noise_records.shape[0])}


def activation_patch(
    subject: Subject,
    X_pert: np.ndarray,
    X_clean: np.ndarray,
    targets: np.ndarray,
    geometry: Geometry,
    stages: tuple[str, ...] = ("whitened", "channel", "token", "z"),
) -> dict[str, dict[str, float]]:
    """Per stage: consequence when the clean stage-k value is substituted into the perturbed pass.

    ``recovery`` is the fraction of the perturbed-minus-clean consequence gap
    closed by the patch (1 = fully causal at or before that stage, 0 = none).
    Requires ``X_pert`` and ``X_clean`` to share the geometry.
    """
    rep_p = subject.represent(X_pert, geometry)
    rep_c = subject.represent(X_clean, geometry)
    err_p = float(np.mean(np.abs(rep_p["output"] - targets)))
    err_c = float(np.mean(np.abs(rep_c["output"] - targets)))
    gap = err_p - err_c
    out: dict[str, dict[str, float]] = {"unpatched": {"consequence": err_p, "recovery": 0.0},
                                        "clean": {"consequence": err_c, "recovery": 1.0}}
    for stage in stages:
        if stage not in HOOKS:
            raise ValueError(stage)
        patched = subject.forward_from_stage(stage, rep_c[stage], X_pert, geometry)
        err = float(np.mean(np.abs(patched["output"] - targets)))
        rec = 1.0 if abs(gap) < 1e-12 else float((err_p - err) / gap)
        out[stage] = {"consequence": err, "recovery": rec}
    return out


def damage_patch(
    subject: Subject,
    X_pert: np.ndarray,
    X_clean: np.ndarray,
    targets: np.ndarray,
    geometry: Geometry,
    stages: tuple[str, ...] = ("whitened", "channel", "token", "z"),
) -> dict[str, dict[str, float]]:
    """The reverse patch: substitute the *perturbed* stage-k value into the *clean* pass.

    ``transmitted`` is the fraction of the perturbed-minus-clean consequence
    gap that stage k alone carries forward. Together with
    :func:`activation_patch` it brackets where damage enters: for an
    input-side corruption both are flat (1.0 at every stage) — the linear
    subject has no stage that *creates* damage, which is itself the finding
    that sends C5 to the nonlinear subjects.
    """
    rep_p = subject.represent(X_pert, geometry)
    rep_c = subject.represent(X_clean, geometry)
    err_p = float(np.mean(np.abs(rep_p["output"] - targets)))
    err_c = float(np.mean(np.abs(rep_c["output"] - targets)))
    gap = err_p - err_c
    out: dict[str, dict[str, float]] = {}
    for stage in stages:
        patched = subject.forward_from_stage(stage, rep_p[stage], X_clean, geometry)
        err = float(np.mean(np.abs(patched["output"] - targets)))
        out[stage] = {"consequence": err, "transmitted": 1.0 if abs(gap) < 1e-12 else float((err - err_c) / gap)}
    return out


def refit_stage(subject: Subject, stage: str, X: np.ndarray, targets: np.ndarray, geometry: Geometry) -> Subject:
    """Stage-restricted refit — the subject decides what that means for its class."""
    if not hasattr(subject, "refit_stage"):
        raise TypeError("subject does not support stage-restricted refitting")
    return subject.refit_stage(stage, X, targets, geometry)  # type: ignore[attr-defined]
