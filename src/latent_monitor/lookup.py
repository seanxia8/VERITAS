# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Attribution is a pre-registered lookup, not a classifier (plan §3.3).

    if   isotropic per-event alarm, consequence unchanged, energy in P_null     -> output-null       -> no action (control)
    elif isotropic per-event alarm, consequence up, energy in P_out            -> output-aligned    -> none (control partner)
    elif noise-only statistics off their null and no consistent mean shift      -> sigma, covariance -> re-whiten
    elif consistent shift, noise unchanged, energy mostly OUTSIDE the span      -> event (support)   -> abstain
    elif consistent shift peaking at the token stage and small at z             -> geometry          -> refit P / S2
    elif noise-only statistics off their null and a consistent shift            -> sigma, structural -> patch S1
    elif consistent shift, noise unchanged, energy mostly INSIDE the span       -> event in span (S) -> recalibrate head / abstain
    else                                                                         -> abstain (undeclared)

The discriminator between acquisition (N) and physics (S/E) is whether the
*noise-only* (random-trigger) statistics moved: an acquisition change shows in
records that contain no event; a physics change cannot.

Thresholds are calibrated once on the reference cell's own null and are then
frozen; :func:`calibrate` is the only place they are set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .reference import ReferenceCell
from .statistics import CellStatistics

ADJUSTMENT = {
    "event": "abstain; extend the training support with data — no weight adjustment is valid",
    "sigma_cov": "re-estimate Σ from noise-only records and replace Σ̂ in the whitening layer; no gradient step",
    "sigma_struct": "activation-patch the per-channel stage first; if consequence recovers, LoRA on S1 only",
    "geometry": "LoRA / refit the geometry embedding and pooling only; W, S1, decoder frozen",
    "output_null": "none — alarm without consequence; this is the control",
    "output_aligned": "none — the norm-matched consequential partner of the output-null control",
    "event_in_span": "the representation is intact and the head is not: recalibrate or extend the output head on labelled data from the family, else abstain",
    "undeclared": "abstain",
}


@dataclass(frozen=True)
class Thresholds:
    mean_shift_norm: float          # null quantile of the Mahalanobis mean-shift norm (a consistent shift)
    dz_norm: float                  # null quantile of the per-event Mahalanobis ‖Δz‖ (the alarm)
    var_ratio_low: float            # band on the noise-only z-variance ratio
    var_ratio_high: float
    psd_ratio_dev: float            # max |smoothed residual PSD ratio − 1| the null allows
    psd_line_dev: float             # max |single-bin residual PSD ratio − 1| the null allows
    chan_corr_shift: float          # Frobenius shift of the whitened-residual channel correlation the null allows
    out_of_span: float = 0.5        # residual / (residual + in-span) energy of the paired change above this = out-of-span
    consequence_ratio_null: float = 0.10   # |ratio − 1| below this counts as "no consequence"
    isotropy: float = 0.5           # mean-shift / (per-event ‖Δz‖·√n) below this = random directions (designed families)
    z_small_factor: float = 0.5     # geometry: z-stage shift must be below this × token-stage shift

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calibrate(ref: ReferenceCell, *, quantile: str = "q99", var_band: float = 0.15,
              out_of_span: float = 0.5, consequence_ratio_null: float = 0.10) -> Thresholds:
    """Fix the thresholds from the reference cell's null distributions. Called once, then frozen."""
    return Thresholds(
        mean_shift_norm=float(ref.null_mean_shift_quantiles[quantile]),
        dz_norm=float(ref.null_dz_norm_quantiles[quantile]),
        var_ratio_low=1.0 - var_band,
        var_ratio_high=1.0 + var_band,
        psd_ratio_dev=float(ref.null_psd_ratio_dev[quantile]),
        psd_line_dev=float(ref.null_psd_line_dev[quantile]),
        chan_corr_shift=float(ref.null_chan_corr_shift[quantile]),
        out_of_span=out_of_span,
        consequence_ratio_null=consequence_ratio_null,
    )


@dataclass(frozen=True)
class Attribution:
    label: str                      # event | sigma_cov | sigma_struct | geometry | output_null | undeclared
    reason: str
    adjustment: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "reason": self.reason, "adjustment": self.adjustment, "evidence": self.evidence}


def attribute(stats: CellStatistics, thr: Thresholds) -> Attribution:
    """The pre-registered lookup. Rule order is the decision procedure; see the module docstring."""
    var_mean = float(np.mean(stats.z_var_ratio))
    z_var_off = not (thr.var_ratio_low <= var_mean <= thr.var_ratio_high) or \
        float(np.max(stats.z_var_ratio)) > thr.var_ratio_high * 1.5 or float(np.min(stats.z_var_ratio)) < thr.var_ratio_low * 0.5
    psd_off = stats.psd_dev_smooth > thr.psd_ratio_dev or stats.psd_dev_line > thr.psd_line_dev
    chan_off = (not np.isnan(stats.residual_chan_corr_shift)) and stats.residual_chan_corr_shift > thr.chan_corr_shift
    #: the N-vs-S discriminator: an acquisition change shows in noise-only (random-trigger) records; a physics change never does
    var_off = z_var_off or psd_off or chan_off
    mean_shifted = stats.mean_shift_norm > thr.mean_shift_norm
    alarm = stats.dz_norm_mean > thr.dz_norm
    isotropy = stats.mean_shift_norm / max(stats.dz_norm_mean * np.sqrt(stats.n_events), 1e-12)
    isotropic = isotropy < thr.isotropy
    out_of_span = stats.out_of_span_fraction > thr.out_of_span
    lp = stats.layer_profile
    token_peak = stats.layer_peak == "token" and lp["z"]["total"] < thr.z_small_factor * lp["token"]["total"]
    no_consequence = float(np.max(np.abs(stats.consequence_ratio - 1.0))) < thr.consequence_ratio_null
    consequence_up = float(np.max(stats.consequence_ratio)) > 1.0 + thr.consequence_ratio_null
    ev = {"mean_shift_norm": stats.mean_shift_norm, "thr_mean_shift": thr.mean_shift_norm, "mean_shifted": mean_shifted,
          "alarm": alarm, "dz_norm_mean": stats.dz_norm_mean, "isotropy": isotropy,
          "z_var_ratio_mean": var_mean, "z_var_off": z_var_off, "psd_off": psd_off, "chan_off": chan_off, "var_off": var_off,
          "out_of_span_fraction": stats.out_of_span_fraction, "layer_peak": stats.layer_peak,
          "dz_energy": stats.dz_energy, "dz_energy_event": stats.dz_energy_event,
          "consequence_ratio": stats.consequence_ratio.tolist()}

    if isotropic and alarm and no_consequence and stats.dz_energy_event["null"] > 0.9:
        lab, why = "output_null", "isotropic per-event displacement in the output-null subspace, consequence unchanged"
    elif isotropic and alarm and consequence_up and stats.dz_energy_event["out"] > 0.9:
        lab, why = "output_aligned", "isotropic per-event displacement in the output-aligned subspace, consequence rose"
    elif var_off and not mean_shifted:
        lab, why = "sigma_cov", "noise-only statistics off their null with no consistent mean shift"
    elif mean_shifted and not var_off and out_of_span:
        lab, why = "event", "consistent shift with the noise model unchanged and most of its energy outside the excited span"
    elif mean_shifted and token_peak:
        lab, why = "geometry", "shift concentrated at the token stage and small at pooled z"
    elif var_off and mean_shifted:
        lab, why = "sigma_struct", "consistent mean shift together with a noise-only change (acquisition-side)"
    elif mean_shifted and not var_off:
        lab, why = "event_in_span", "consistent shift, noise model unchanged, energy inside the excited span: supported-but-rare physics"
    else:
        lab, why = "undeclared", "no statistic crossed its threshold"
    return Attribution(lab, why, ADJUSTMENT[lab], ev)
