# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The reference cell: projectors and null distributions, fitted once (plan §3.1).

Four projectors, all from the frozen subject at the reference cell:

    P_out / P_null   row space of J_o = ∂y/∂z and its complement
    P_exc / P_unexc  eigenvectors of the pullback Fisher I = J_gᵀ Σ̂⁻¹ J_g above / below a rank threshold

plus the null distributions every later statistic is measured against —
reference-vs-reference paired twins (independent noise realisations of the
same events) for the mean-shift and layer-profile statistics, and
noise-only records for the whitened-variance and residual-spectrum ones.
``k_out`` and the Fisher rank threshold are fixed here and never re-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.fft import rfft

from .subject import HOOKS, PER_CHANNEL_HOOKS, Geometry, Subject

PSD_SMOOTH = 4


def _smooth(v: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(v, dtype=float)
    k = np.ones(width) / width
    pad = np.pad(np.asarray(v, dtype=float), (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def pooled_stage(rep: dict[str, np.ndarray], hook: str) -> tuple[np.ndarray, np.ndarray | None]:
    """(per-event mean over channels, per-event channel second moment) for a hook.

    Non-per-channel hooks return (value, None). Per-channel hooks are pooled so
    that cells with different channel counts remain comparable.
    """
    v = np.asarray(rep[hook], dtype=float)
    if hook not in PER_CHANNEL_HOOKS:
        return v.reshape(v.shape[0], -1), None
    if hook == "whitened":
        # (n, C, N): pool to the per-event channel mean trace; second moment = mean per-channel power
        return v.mean(axis=1), (v**2).mean(axis=(1, 2))[:, None]
    # (n, C, D): mean over channels and the flattened channel covariance
    mean = v.mean(axis=1)
    centred = v - mean[:, None, :]
    cov = np.einsum("bci,bcj->bij", centred, centred) / max(v.shape[1] - 1, 1)
    return mean, cov.reshape(v.shape[0], -1)


def whitened_residual(subject: Subject, rep: dict[str, np.ndarray], geometry: Geometry) -> np.ndarray:
    """r̃ = x̃ − g(z): (n, C, N)."""
    return rep["whitened"] - subject.decode(rep["z"], geometry)


@dataclass
class ReferenceCell:
    geometry: Geometry
    z_mean: np.ndarray                      # (k,)
    z_cov: np.ndarray                       # (k, k)  reference z covariance (for Mahalanobis abstention)
    null_dz_cov: np.ndarray                 # (k, k)  covariance of ref-vs-ref twin Δz
    P_out: np.ndarray
    P_null: np.ndarray
    P_exc: np.ndarray
    P_unexc: np.ndarray
    fisher_eigvals: np.ndarray
    fisher_rank: int
    fisher_rank_threshold: float
    k_out: int
    noise_z_var: np.ndarray                 # (k,) variance of z on noise-only records, per excited direction
    exc_basis: np.ndarray                   # (k, k) columns = Fisher eigenvectors (excited first)
    noise_residual_psd: np.ndarray          # (N//2+1,) whitened-residual PSD on noise records, channel-averaged
    noise_residual_chan_cov: np.ndarray     # (C, C) whitened-residual channel covariance on noise records
    null_stage: dict[str, dict[str, np.ndarray]]   # per hook: {"mean_std": (D,), "second_std": (D2,) or None}
    null_mean_shift_quantiles: dict[str, float]
    null_dz_norm_quantiles: dict[str, float]        # per-event Mahalanobis ‖Δz‖ under the null
    null_residual_coherence_quantiles: dict[str, float]
    null_residual_shift_per_sample: float           # mean Δr̃² per sample between ref twins
    null_residual_shift_event_std: float            # std over events of the per-event Δr̃ energy
    null_residual_shift_event_size: int             # elements per event that std was measured on
    null_residual_shift_quantiles: dict[str, float] # bootstrap of (that ratio) — ≈1 with spread
    null_psd_ratio_dev: dict[str, float]          # max |smoothed PSD ratio − 1| between two halves of the noise records
    null_psd_line_dev: dict[str, float]           # max |single-bin PSD ratio − 1|, same construction
    null_chan_corr_shift: dict[str, float]
    consequence_ref: np.ndarray             # (n_targets,) mean |y − target| on reference events
    n_fit: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def latent_dim(self) -> int:
        return int(self.z_mean.shape[0])

    # -- Mahalanobis in the reference z-metric (abstention) --------------
    def mahalanobis(self, z: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(z) - self.z_mean
        Li = np.linalg.cholesky(self.z_cov + 1e-9 * np.eye(self.latent_dim))
        w = np.linalg.solve(Li, d.T)
        return np.sqrt(np.sum(w**2, axis=0))

    def null_norm(self, dz_mean: np.ndarray, n_events: int) -> float:
        """Mahalanobis norm of a mean Δz against the null, scaled for the number of events averaged."""
        Li = np.linalg.cholesky(self.null_dz_cov / max(n_events, 1) + 1e-12 * np.eye(self.latent_dim))
        w = np.linalg.solve(Li, np.asarray(dz_mean, dtype=float))
        return float(np.sqrt(np.sum(w**2)))


def _projector(basis: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the column space of ``basis``."""
    if basis.size == 0:
        k = basis.shape[0]
        return np.zeros((k, k))
    Q, _ = np.linalg.qr(basis)
    return Q @ Q.T


def residual_coherence(dr: np.ndarray) -> float:
    """‖mean_events Δr̃‖² / mean_events ‖Δr̃‖² — 1 for a deterministic shift, ~1/n for noise."""
    dr = dr.reshape(dr.shape[0], -1)
    num = float(np.sum(dr.mean(axis=0) ** 2))
    den = float(np.mean(np.sum(dr**2, axis=1))) + 1e-300
    return num / den


def fit_reference(
    subject: Subject,
    cell: Any,
    event_ids: np.ndarray,
    noise_ids: np.ndarray,
    *,
    k_out: int | None = None,
    fisher_rank_rel: float = 1e-3,
    null_replicate: int = 1,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99),
    n_bootstrap: int = 200,
    seed: int = 0,
) -> ReferenceCell:
    """Fit projectors and nulls on the reference cell (``cell`` is a ``tier1.Cell``-like object)."""
    rng = np.random.default_rng(seed)
    geometry = cell.geometry
    X0, T0 = cell.batch(event_ids, replicate=0)
    X1, _ = cell.batch(event_ids, replicate=null_replicate)
    rep0 = subject.represent(X0, geometry)
    rep1 = subject.represent(X1, geometry)
    z0, z1 = rep0["z"], rep1["z"]
    n, k = z0.shape
    dz_null = z1 - z0

    # projectors
    z_mean = z0.mean(axis=0)
    J_o = np.asarray(subject.jac_output(z_mean, geometry), dtype=float)   # (t, k)
    if k_out is None:
        k_out = int(np.linalg.matrix_rank(J_o))
    _, _, Vt = np.linalg.svd(J_o, full_matrices=False)
    P_out = _projector(Vt[:k_out].T)
    P_null = np.eye(k) - P_out
    J_g = np.asarray(subject.jac_recon(z_mean, geometry), dtype=float)    # (C·N, k), already whitened
    fisher = J_g.T @ J_g
    ev, V = np.linalg.eigh(fisher)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    thr = fisher_rank_rel * float(ev[0])
    excited = ev > thr
    P_exc = _projector(V[:, excited]) if excited.any() else np.zeros((k, k))
    P_unexc = np.eye(k) - P_exc

    # noise-only references
    Xn = cell.noise_batch(noise_ids)
    repn = subject.represent(Xn, geometry)
    zn_exc = repn["z"] @ V                                                # coordinates in the Fisher basis
    noise_z_var = zn_exc.var(axis=0)
    rn = whitened_residual(subject, repn, geometry)                        # (m, C, N)
    F = rfft(rn, axis=-1)
    noise_residual_psd = np.mean(np.abs(F) ** 2, axis=(0, 1))
    noise_residual_chan_cov = np.cov(rn.transpose(1, 0, 2).reshape(rn.shape[1], -1))

    # stage nulls
    null_stage: dict[str, dict[str, np.ndarray]] = {}
    for hook in HOOKS:
        m0, s0 = pooled_stage(rep0, hook)
        m1, s1 = pooled_stage(rep1, hook)
        entry = {"mean_std": (m1 - m0).std(axis=0) + 1e-12}
        entry["second_std"] = None if s0 is None else (s1 - s0).std(axis=0) + 1e-12
        null_stage[hook] = entry

    # null quantiles of the mean-shift norm (bootstrap over event subsets of size n)
    Li = np.linalg.cholesky(np.cov(dz_null.T) / n + 1e-12 * np.eye(k))
    norms = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = dz_null[idx].mean(axis=0)
        norms.append(float(np.sqrt(np.sum(np.linalg.solve(Li, m) ** 2))))
    null_mean_shift_quantiles = {f"q{int(q*100)}": float(np.quantile(norms, q)) for q in quantiles}
    Lc = np.linalg.cholesky(np.cov(dz_null.T) + 1e-12 * np.eye(k))
    per_event = np.sqrt(np.sum(np.linalg.solve(Lc, dz_null.T) ** 2, axis=0))
    null_dz_norm_quantiles = {f"q{int(q*100)}": float(np.quantile(per_event, q)) for q in quantiles}

    # null residual coherence (ref twins: the residual difference is noise-like)
    r0 = whitened_residual(subject, rep0, geometry)
    r1 = whitened_residual(subject, rep1, geometry)
    cohs, shifts = [], []
    dr_null = r1 - r0
    null_shift = float(np.mean(dr_null**2))
    null_event_energy = np.sum(dr_null.reshape(n, -1) ** 2, axis=1)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cohs.append(residual_coherence(dr_null[idx]))
        shifts.append(float(np.mean(dr_null[idx] ** 2)) / null_shift)
    null_residual_coherence_quantiles = {f"q{int(q*100)}": float(np.quantile(cohs, q)) for q in quantiles}
    null_residual_shift_quantiles = {f"q{int(q*100)}": float(np.quantile(shifts, q)) for q in quantiles}

    # null bands for the noise-only spectral and channel-correlation deviations: bootstrap halves
    def _corr(c):
        s = np.sqrt(np.clip(np.diag(c), 1e-15, None)); return c / np.outer(s, s)
    m = rn.shape[0]
    psd_devs, line_devs, corr_shifts = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.permutation(m)
        a, b = rn[idx[: m // 2]], rn[idx[m // 2:]]
        pa = np.mean(np.abs(rfft(a, axis=-1)) ** 2, axis=(0, 1)); pb = np.mean(np.abs(rfft(b, axis=-1)) ** 2, axis=(0, 1))
        ratio = pa[1:] / np.maximum(pb[1:], 1e-15)
        psd_devs.append(float(np.max(np.abs(_smooth(ratio, PSD_SMOOTH) - 1.0))))
        line_devs.append(float(np.max(np.abs(ratio - 1.0))))
        ca = np.cov(a.transpose(1, 0, 2).reshape(a.shape[1], -1)); cb = np.cov(b.transpose(1, 0, 2).reshape(b.shape[1], -1))
        corr_shifts.append(float(np.linalg.norm(_corr(ca) - _corr(cb))))
    null_psd_ratio_dev = {f"q{int(q*100)}": float(np.quantile(psd_devs, q)) for q in quantiles}
    null_psd_line_dev = {f"q{int(q*100)}": float(np.quantile(line_devs, q)) for q in quantiles}
    null_chan_corr_shift = {f"q{int(q*100)}": float(np.quantile(corr_shifts, q)) for q in quantiles}

    consequence_ref = np.mean(np.abs(rep0["output"] - T0), axis=0)

    return ReferenceCell(
        geometry=geometry, z_mean=z_mean, z_cov=np.cov(z0.T), null_dz_cov=np.cov(dz_null.T),
        P_out=P_out, P_null=P_null, P_exc=P_exc, P_unexc=P_unexc,
        fisher_eigvals=ev, fisher_rank=int(excited.sum()), fisher_rank_threshold=thr, k_out=k_out,
        noise_z_var=noise_z_var, exc_basis=V,
        noise_residual_psd=noise_residual_psd, noise_residual_chan_cov=noise_residual_chan_cov,
        null_stage=null_stage, null_mean_shift_quantiles=null_mean_shift_quantiles,
        null_dz_norm_quantiles=null_dz_norm_quantiles,
        null_residual_coherence_quantiles=null_residual_coherence_quantiles,
        null_residual_shift_per_sample=null_shift, null_residual_shift_quantiles=null_residual_shift_quantiles,
        null_residual_shift_event_std=float(np.std(null_event_energy)), null_residual_shift_event_size=int(dr_null[0].size),
        null_psd_ratio_dev=null_psd_ratio_dev, null_psd_line_dev=null_psd_line_dev, null_chan_corr_shift=null_chan_corr_shift,
        consequence_ref=consequence_ref, n_fit=int(n),
        meta={"n_noise": int(len(noise_ids)), "fisher_rank_rel": fisher_rank_rel},
    )
