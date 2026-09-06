# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Per-cell statistics on per-event Δz against the paired twin (plan §3.2).

Every number here is a *per-event* quantity aggregated over paired events, or
a noise-only quantity against the reference noise-only null — never a
distributional distance between unpaired samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.fft import rfft

from .reference import ReferenceCell, pooled_stage, residual_coherence, whitened_residual
from .subject import HOOKS, Geometry, Subject


PSD_SMOOTH = 4


def smooth(v: np.ndarray, width: int) -> np.ndarray:
    """Boxcar-smooth a spectrum (edge-padded) so a broad deviation is seen with less periodogram noise."""
    if width <= 1:
        return np.asarray(v, dtype=float)
    k = np.ones(width) / width
    pad = np.pad(np.asarray(v, dtype=float), (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def participation_ratio(v: np.ndarray) -> float:
    """(Σ|v|)² / (n Σ v²) ∈ (0, 1]: 1 = spread over every component, →0 = concentrated in one."""
    v = np.abs(np.asarray(v, dtype=float)).ravel()
    if v.size == 0 or np.sum(v**2) == 0:
        return 1.0
    return float(np.sum(v) ** 2 / (v.size * np.sum(v**2)))


def rank_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC of ``score`` for boolean ``label`` by rank statistics (ties averaged)."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = sums[inv] / counts[inv]
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@dataclass
class CellStatistics:
    label: str
    moved: str
    n_events: int
    # z displacement
    mean_shift_norm: float                  # Mahalanobis norm of mean Δz against the null
    dz_energy: dict[str, float]             # fraction of ‖mean Δz‖² in P_out / P_null / P_exc / P_unexc
    dz_energy_event: dict[str, float]       # the same split averaged per event (random-direction families)
    dz_norm_mean: float                     # mean per-event ‖Δz‖ in the null metric (alarm magnitude)
    dz_euclid_mean: float                   # mean per-event Euclidean ‖Δz‖ (the metric a naive monitor uses)
    # whitened variance (noise-only)
    z_var_ratio: np.ndarray                 # (k,) in the Fisher basis, vs reference noise-only
    residual_psd_ratio: np.ndarray          # (N//2+1,)
    residual_psd_ratio_participation: float
    psd_dev_smooth: float                   # max |smoothed ratio − 1|  (broad deviations)
    psd_dev_line: float                     # max |single-bin ratio − 1| (narrow lines)
    residual_chan_corr_shift: float         # Frobenius norm of Δ channel correlation of whitened residual
    # event residual
    residual_norm_ratio: float              # mean ‖r̃‖² cell / reference, on paired events
    residual_coherence: float               # ‖mean Δr̃‖²/mean‖Δr̃‖²
    residual_shift_ratio: float             # mean Δr̃² per sample vs the ref-twin null
    residual_shift_z: float                 # z-score of the mean per-event residual-difference energy vs its null
    out_of_span_fraction: float             # residual energy / (residual + in-span) of the paired change
    fisher_rank: int
    fisher_rank_ref: int
    # layer profile
    layer_profile: dict[str, dict[str, float]]
    layer_peak: str
    # consequence
    consequence: np.ndarray                 # (n_targets,) mean |y − target|
    consequence_ratio: np.ndarray           # vs reference
    consequence_auroc_given_alarm: float    # AUROC(alarm magnitude → consequence increased), events with alarm > q90
    alarm_rate: float                       # fraction of events with per-event Mahalanobis(z) above reference q99
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label, "moved": self.moved, "n": self.n_events,
            "mean_shift_norm": round(self.mean_shift_norm, 3),
            "dz_energy": {k: round(v, 3) for k, v in self.dz_energy.items()},
            "dz_energy_event": {k: round(v, 3) for k, v in self.dz_energy_event.items()},
            "dz_norm_mean": round(self.dz_norm_mean, 4), "dz_euclid_mean": round(self.dz_euclid_mean, 4),
            "z_var_ratio_mean": round(float(np.mean(self.z_var_ratio)), 3),
            "z_var_ratio_max": round(float(np.max(self.z_var_ratio)), 3),
            "z_var_ratio_min": round(float(np.min(self.z_var_ratio)), 3),
            "residual_psd_participation": round(self.residual_psd_ratio_participation, 3),
            "psd_dev_smooth": round(self.psd_dev_smooth, 3), "psd_dev_line": round(self.psd_dev_line, 3),
            "residual_chan_corr_shift": round(self.residual_chan_corr_shift, 3),
            "residual_norm_ratio": round(self.residual_norm_ratio, 3),
            "residual_coherence": round(self.residual_coherence, 3),
            "residual_shift_ratio": round(self.residual_shift_ratio, 3),
            "residual_shift_z": round(self.residual_shift_z, 2), "out_of_span_fraction": round(self.out_of_span_fraction, 3),
            "fisher_rank": self.fisher_rank,
            "layer_peak": self.layer_peak,
            "layer_profile": {h: round(v["total"], 2) for h, v in self.layer_profile.items()},
            "consequence_ratio": [round(float(c), 3) for c in self.consequence_ratio],
            "consequence_auroc_given_alarm": None if np.isnan(self.consequence_auroc_given_alarm) else round(self.consequence_auroc_given_alarm, 3),
            "alarm_rate": round(self.alarm_rate, 3),
        }


def cell_statistics(
    ref: ReferenceCell,
    subject: Subject,
    ref_cell: Any,
    cell: Any,
    event_ids: np.ndarray,
    noise_ids: np.ndarray,
    *,
    alarm_quantile: float = 0.9,
) -> CellStatistics:
    """All §3.2 statistics for ``cell`` against its paired twins in ``ref_cell``."""
    geometry: Geometry = cell.geometry
    X_cell, T = cell.batch(event_ids, replicate=0)
    X_twin, _ = ref_cell.batch(event_ids, replicate=0)
    rep_c = subject.represent(X_cell, geometry)
    rep_t = subject.represent(X_twin, ref_cell.geometry)
    n, k = rep_c["z"].shape

    # --- z displacement
    dz = rep_c["z"] - rep_t["z"]
    dz_mean = dz.mean(axis=0)
    mean_shift_norm = ref.null_norm(dz_mean, n)
    e_total = float(dz_mean @ dz_mean) + 1e-300
    dz_energy = {
        "out": float(dz_mean @ ref.P_out @ dz_mean) / e_total,
        "null": float(dz_mean @ ref.P_null @ dz_mean) / e_total,
        "exc": float(dz_mean @ ref.P_exc @ dz_mean) / e_total,
        "unexc": float(dz_mean @ ref.P_unexc @ dz_mean) / e_total,
    }
    Li = np.linalg.cholesky(ref.null_dz_cov + 1e-12 * np.eye(k))
    dz_null_norm = np.sqrt(np.sum(np.linalg.solve(Li, dz.T) ** 2, axis=0))     # per-event alarm magnitude
    e_ev = np.sum(dz**2, axis=1) + 1e-300
    dz_energy_event = {
        "out": float(np.mean(np.einsum("bi,ij,bj->b", dz, ref.P_out, dz) / e_ev)),
        "null": float(np.mean(np.einsum("bi,ij,bj->b", dz, ref.P_null, dz) / e_ev)),
        "exc": float(np.mean(np.einsum("bi,ij,bj->b", dz, ref.P_exc, dz) / e_ev)),
        "unexc": float(np.mean(np.einsum("bi,ij,bj->b", dz, ref.P_unexc, dz) / e_ev)),
    }

    # --- whitened variance on noise-only records
    Xn = cell.noise_batch(noise_ids)
    repn = subject.represent(Xn, geometry)
    zn = repn["z"] @ ref.exc_basis
    scale = float(getattr(subject, "noise_variance_scale", lambda a, b: 1.0)(ref.geometry, geometry))
    z_var_ratio = zn.var(axis=0) / np.maximum(ref.noise_z_var * scale, 1e-15)
    rn = whitened_residual(subject, repn, geometry)
    psd = np.mean(np.abs(rfft(rn, axis=-1)) ** 2, axis=(0, 1))
    residual_psd_ratio = psd / np.maximum(ref.noise_residual_psd, 1e-15)
    part = participation_ratio(residual_psd_ratio[1:] - 1.0)
    psd_dev_smooth = float(np.max(np.abs(smooth(residual_psd_ratio[1:], PSD_SMOOTH) - 1.0)))
    psd_dev_line = float(np.max(np.abs(residual_psd_ratio[1:] - 1.0)))
    chan_cov = np.cov(rn.transpose(1, 0, 2).reshape(rn.shape[1], -1))
    if chan_cov.shape == ref.noise_residual_chan_cov.shape:
        def corr(c):
            s = np.sqrt(np.clip(np.diag(c), 1e-15, None)); return c / np.outer(s, s)
        chan_corr_shift = float(np.linalg.norm(corr(chan_cov) - corr(ref.noise_residual_chan_cov)))
    else:
        chan_corr_shift = float("nan")

    # --- event residual
    r_c = whitened_residual(subject, rep_c, geometry)
    r_t = whitened_residual(subject, rep_t, ref_cell.geometry)
    residual_norm_ratio = float(np.mean(np.sum(r_c.reshape(n, -1) ** 2, axis=1)) / (np.mean(np.sum(r_t.reshape(n, -1) ** 2, axis=1)) + 1e-300))
    if r_c.shape == r_t.shape:
        dr = r_c - r_t
    else:  # geometry cell: compare the channel-mean residual trace
        dr = r_c.mean(axis=1) - r_t.mean(axis=1)
    coherence = residual_coherence(dr)
    per_sample = np.mean(dr**2)
    residual_shift_ratio = float(per_sample / max(ref.null_residual_shift_per_sample, 1e-15))
    # per-event residual-difference energy against its null (z-score of the mean over events)
    e_event = np.sum(dr.reshape(n, -1) ** 2, axis=1)
    e_null_mean = ref.null_residual_shift_per_sample * dr[0].size
    e_null_std = ref.null_residual_shift_event_std * np.sqrt(dr[0].size / ref.null_residual_shift_event_size)
    residual_shift_z = float((np.mean(e_event) - e_null_mean) / max(e_null_std / np.sqrt(n), 1e-15))
    # out-of-span fraction of the paired *signal* change: residual energy per channel vs in-span energy ‖Δz‖²
    e_res_per_channel = float(np.mean(np.sum(dr**2, axis=-1)))          # mean over events and channels
    e_span = float(np.mean(np.sum(dz**2, axis=1)))
    out_of_span_fraction = e_res_per_channel / max(e_res_per_channel + e_span, 1e-300)
    J_g = np.asarray(subject.jac_recon(rep_c["z"].mean(axis=0), geometry), dtype=float)
    ev = np.linalg.eigvalsh(J_g.T @ J_g)
    fisher_rank = int(np.sum(ev > ref.fisher_rank_threshold))

    # --- layer profile
    layer_profile: dict[str, dict[str, float]] = {}
    for hook in HOOKS:
        mc, sc = pooled_stage(rep_c, hook)
        mt, st = pooled_stage(rep_t, hook)
        null = ref.null_stage[hook]
        if mc.shape == mt.shape:
            dm = (mc - mt).mean(axis=0) / null["mean_std"] * np.sqrt(n)
            mean_shift = float(np.sqrt(np.mean(dm**2)))
        else:
            mean_shift = float("nan")
        if sc is not None and st is not None and sc.shape == st.shape and null["second_std"] is not None:
            ds = (sc - st).mean(axis=0) / null["second_std"] * np.sqrt(n)
            second = float(np.sqrt(np.mean(ds**2)))
        else:
            second = 0.0
        layer_profile[hook] = {"mean": mean_shift, "second_moment": second,
                               "total": float(np.hypot(0.0 if np.isnan(mean_shift) else mean_shift, second))}
    layer_peak = max(layer_profile, key=lambda h: layer_profile[h]["total"])

    # --- consequence
    err_c = np.abs(rep_c["output"] - T)
    err_t = np.abs(rep_t["output"] - T)
    consequence = err_c.mean(axis=0)
    consequence_ratio = consequence / np.maximum(ref.consequence_ref, 1e-15)
    worsened = err_c.sum(axis=1) > err_t.sum(axis=1)
    alarm_thr = np.quantile(dz_null_norm, alarm_quantile)
    strong = dz_null_norm >= alarm_thr
    auroc = rank_auroc(dz_null_norm[strong], worsened[strong]) if strong.sum() >= 4 else float("nan")
    maha = ref.mahalanobis(rep_c["z"])
    ref_maha_q99 = float(np.quantile(ref.mahalanobis(rep_t["z"]), 0.99))
    alarm_rate = float(np.mean(maha > ref_maha_q99))

    return CellStatistics(
        label=cell.label, moved=cell.moved, n_events=int(n),
        mean_shift_norm=mean_shift_norm, dz_energy=dz_energy, dz_energy_event=dz_energy_event,
        dz_norm_mean=float(np.mean(dz_null_norm)), dz_euclid_mean=float(np.mean(np.linalg.norm(dz, axis=1))),
        z_var_ratio=z_var_ratio, residual_psd_ratio=residual_psd_ratio,
        residual_psd_ratio_participation=part, psd_dev_smooth=psd_dev_smooth, psd_dev_line=psd_dev_line,
        residual_chan_corr_shift=chan_corr_shift,
        residual_norm_ratio=residual_norm_ratio, residual_coherence=coherence, residual_shift_ratio=residual_shift_ratio,
        residual_shift_z=residual_shift_z, out_of_span_fraction=out_of_span_fraction,
        fisher_rank=fisher_rank, fisher_rank_ref=ref.fisher_rank,
        layer_profile=layer_profile, layer_peak=layer_peak,
        consequence=consequence, consequence_ratio=consequence_ratio,
        consequence_auroc_given_alarm=auroc, alarm_rate=alarm_rate,
        extra={"n_channels": geometry.n_channels, "dz_null_norm": dz_null_norm, "maha": maha},
    )
