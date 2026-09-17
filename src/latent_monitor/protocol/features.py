# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Alarm-time features of one observed window (R2; I2, I3, I10).

Everything here is computable from :class:`~latent_monitor.protocol.availability.AlarmTimeInputs`
and the *reference-fitted* statistics only — no paired twin, no truth, no
realised covariance. The builder returns a locked
:class:`~latent_monitor.protocol.availability.FeatureBatch` whose every value
carries its source tags.

Feature groups (names match :func:`~latent_monitor.protocol.availability.tier1_manifest`):

    input_quality      raw-trace summaries, no model
    output             frozen-model outputs
    uncertainty        an output-side proxy (raw-domain reconstruction residual energy)
    final_embedding    z and its (shrunk) Mahalanobis distance from the reference
    noise_only         random-trigger record statistics against the reference noise null
    generic_rich       reference-distance transforms of *generic* quantities: pooled whitened
                       input, output, pre-output, the energy split of z − z̄ in the reference
                       projectors, the out-of-span fraction from the residual, training-support novelty
    intermediate       strictly internal hooks (``channel``, ``token``): Mahalanobis of the pooled
                       mean and of the pooled second moment, plus reference principal-component
                       coordinates — no input, no final z, no output
    generic_quadratic  a capacity control: squares and pairwise products of generic_rich scalars,
                       truncated to the intermediate feature count
    layerwise_legacy   the pass-1 per-hook Mahalanobis (a development diagnostic; contaminated by
                       input/output duplicates by construction)

Reference distances (I10): each hook null is fitted on reference-fit windows
only, with Ledoit–Wolf shrinkage of the covariance and, when the hook
dimension exceeds n_ref / 5, a PCA reduction to that many components fitted on
the same windows; the rule is declared in :class:`HookNull`. Raw Mahalanobis
values are then mapped to a common clean-null scale by :class:`NullCalibrator`
fitted on clean *calibration* windows (never reference-fit, never evaluation)
before any combination across hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import rfft
from scipy.special import ndtri

from ..cumulant import (pca_basis, per_window_third_tensor, project_diagonal_third, tensor_frobenius_deviation,
                        third_central_moment)
from ..reference import ReferenceCell, pooled_stage, whitened_residual
from ..statistics import PSD_SMOOTH, smooth
from ..subject import HOOKS, Geometry, Subject
from ..support import SupportEstimator, ledoit_wolf_shrinkage
from ..task_metric import TaskMetric
from .availability import AlarmTimeInputs, FeatureBatch

INTERMEDIATE_HOOKS = ("channel", "token")
#: generic_rich scalars that feed the quadratic capacity control, in declared order.
#: The residual third-cumulant scalars (D7) are appended dynamically because the
#: principal-coordinate count depends on ``n_pcs``; the control is re-truncated to
#: the intermediate feature count so ``generic_rich_matched`` stays count-matched.
QUADRATIC_BASIS = ("in_rms", "in_kurtosis", "unc_recon_energy", "z_mahalanobis", "gr_input_maha", "gr_output_maha",
                   "gr_out_of_span", "gr_support_novelty", "gr_resid_c3_norm", "gr_resid_c3_maha")
#: the committed generic harm score of Claim 2: max of these calibrated distances (fixed before any evaluation label)
GENERIC_COMMITTED = ("gr_input_maha", "gr_output_maha", "z_mahalanobis")
#: the secondary intermediate score (supporting; not the Claim-2 primary)
INTERMEDIATE_COMMITTED = ("im_channel_mean_maha", "im_token_mean_maha")


@dataclass
class HookNull:
    """Reference-fitted null of a pooled hook value: shrunk covariance, optional PCA reduction, declared rule."""

    mean: np.ndarray
    inv_cov: np.ndarray            # in the (possibly reduced) coordinates
    dim: int                       # original dimension
    reduced_dim: int
    pcs: np.ndarray | None         # (dim, reduced_dim) or None
    alpha: float                   # Ledoit–Wolf shrinkage
    n_ref: int
    pcs_all: np.ndarray | None = None   # (dim, min(n, dim)) reference principal directions, for coordinates
    rule: str = "PCA to min(d, n_ref // 5) components when d > n_ref // 5; Ledoit–Wolf shrinkage; fitted on reference_fit windows only"

    @classmethod
    def fit(cls, V: np.ndarray, min_ratio: int = 5) -> "HookNull":
        V = np.atleast_2d(np.asarray(V, dtype=float))
        n, d = V.shape
        if n < 3:
            raise ValueError("need at least three reference windows per hook null")
        mu = V.mean(axis=0)
        Vc = V - mu
        _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
        pcs_all = Vt.T
        pcs = None
        rd = d
        if d > max(1, n // min_ratio):
            rd = max(1, n // min_ratio)
            pcs = pcs_all[:, :rd]
            Vc = Vc @ pcs
        cov, alpha = ledoit_wolf_shrinkage(Vc)
        return cls(mean=mu, inv_cov=np.linalg.inv(cov + 1e-12 * np.eye(rd)), dim=d, reduced_dim=rd, pcs=pcs, alpha=alpha, n_ref=n, pcs_all=pcs_all)

    def mahalanobis(self, V: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(np.asarray(V, dtype=float)) - self.mean
        if d.shape[1] != self.dim:
            return np.full(d.shape[0], np.nan)
        if self.pcs is not None:
            d = d @ self.pcs
        return np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", d, self.inv_cov, d), 0.0) / max(self.reduced_dim, 1))

    def coordinates(self, V: np.ndarray, k: int) -> np.ndarray:
        """First ``k`` reference principal coordinates of the centred value (zeros beyond the available rank)."""
        d = np.atleast_2d(np.asarray(V, dtype=float)) - self.mean
        if d.shape[1] != self.dim:
            return np.full((d.shape[0], k), np.nan)
        pcs = (np.eye(self.dim) if self.pcs_all is None else self.pcs_all)[:, :k]
        out = np.zeros((d.shape[0], k))
        out[:, : pcs.shape[1]] = d @ pcs
        return out


@dataclass
class CumulantNull:
    """Reference null of a per-window third-moment object, compressed to ``n_pcs`` coordinates + one scalar (D7).

    The object is compressed to the declared ``(n_pcs + 1)``-dimensional chart:
    the per-coordinate third central moments in a reference PCA basis and one
    Frobenius-deviation scalar ``||T_i - T_ref||_F``. Two fit routes:

    * :meth:`fit_residual` — the per-window object is a vector (the flattened
      whitened residual); the basis is the reference residual PCA and ``T_ref``
      is the mean rank-1 third tensor of the projections. The per-window object
      is a single rank-1 tensor, so its reference null is wide; the *population*
      cumulant (the cell-level reading used by the signature runner) is the mean
      of these rank-1 objects, i.e. ``ref_tensor``.
    * :meth:`fit_tensor` — the per-window object is the pooled channel third
      tensor of a per-channel hook; the basis comes from the pooled means and
      ``T_ref`` is the mean third tensor.

    ``reading`` is the chart rule of ``ARITRA_CUMULANT_INTEGRATION_2026-09-17.md``
    §2.1: ``"cumulant"`` where the chart is natural (the whitened residual under
    the Gaussian reference; any hook on the linear subject), ``"deviation"``
    otherwise, where the same number is a deviation from the reference cell's
    own third moment.
    """

    kind: str                       # "residual" | "tensor"
    n_pcs: int
    reading: str
    mean: np.ndarray                # (d,) reference mean of the underlying vectors
    pcs: np.ndarray                 # (d, n_pcs) reference PCA basis
    ref_tensor: np.ndarray          # reference third tensor (residual: (n_pcs,)*3; tensor: (D,)*3)
    null: HookNull

    @classmethod
    def fit_residual(cls, V: np.ndarray, n_pcs: int, *, reading: str = "cumulant") -> "CumulantNull":
        V = np.atleast_2d(np.asarray(V, dtype=float))
        mean, pcs = pca_basis(V, n_pcs)
        coords = (V - mean) @ pcs
        ref_t = third_central_moment(coords)
        null = HookNull.fit(cls._compress_residual(coords, ref_t))
        return cls("residual", pcs.shape[1], reading, mean, pcs, ref_t, null)

    @classmethod
    def fit_tensor(cls, T: np.ndarray, basis_vectors: np.ndarray, n_pcs: int, *, reading: str = "cumulant") -> "CumulantNull":
        T = np.atleast_2d(np.asarray(T, dtype=float))
        D = int(round(T.shape[1] ** (1.0 / 3.0)))
        if D ** 3 != T.shape[1]:
            raise ValueError("pooled third tensors must be flattened from a cubic (D, D, D) array")
        T = T.reshape(T.shape[0], D, D, D)
        mean, pcs = pca_basis(basis_vectors, n_pcs)
        ref_t = T.mean(axis=0)
        null = HookNull.fit(cls._compress_tensor(T, pcs, ref_t))
        return cls("tensor", pcs.shape[1], reading, mean, pcs, ref_t, null)

    @staticmethod
    def _compress_tensor(T: np.ndarray, pcs: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
        diag = project_diagonal_third(T, pcs)
        dev = tensor_frobenius_deviation(T, ref_t)
        return np.column_stack([diag, dev])

    @staticmethod
    def _compress_residual(coords: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
        diag = coords ** 3
        dev = tensor_frobenius_deviation(per_window_third_tensor(coords, np.zeros(coords.shape[1])), ref_t)
        return np.column_stack([diag, dev])

    def compress(self, obj: np.ndarray) -> np.ndarray:
        obj = np.asarray(obj, dtype=float)
        if self.kind == "residual":
            A = np.atleast_2d(obj)
            if A.shape[1] != self.mean.shape[0]:
                # a geometry cell forms the residual in a different (C·N) chart: the reference
                # chart does not apply, so the value is undefined here (NaN), never a silent
                # fallback to another chart.
                return np.full((A.shape[0], self.n_pcs + 1), np.nan)
            return self._compress_residual((A - self.mean) @ self.pcs, self.ref_tensor)
        D = self.ref_tensor.shape[0]
        A = np.atleast_2d(obj)
        if A.shape[1] != D ** 3:
            return np.full((A.shape[0], self.n_pcs + 1), np.nan)
        return self._compress_tensor(A.reshape(-1, D, D, D), self.pcs, self.ref_tensor)

    def mahalanobis(self, obj: np.ndarray) -> np.ndarray:
        return self.null.mahalanobis(self.compress(obj))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n_pcs": self.n_pcs, "reading": self.reading, "dim": self.null.dim,
                "reduced_dim": self.null.reduced_dim, "alpha": self.null.alpha, "n_ref": self.null.n_ref,
                "compression": "n_pcs reference-PC coordinates + one Frobenius-deviation scalar (D7)"}


@dataclass
class AlarmTimeReference:
    """Reference-fitted parameters an alarm-time monitor may use (phase ``reference_fit``)."""

    ref: ReferenceCell
    hook_mean_nulls: dict[str, HookNull] = field(default_factory=dict)
    hook_second_nulls: dict[str, HookNull] = field(default_factory=dict)
    hook_third_nulls: dict[str, CumulantNull] = field(default_factory=dict)
    resid_cumulant: CumulantNull | None = None
    z_null: HookNull | None = None
    support: SupportEstimator | None = None
    task: TaskMetric | None = None
    task_target: int = 0
    n_pcs: int = 3
    #: chart rule (D7): "cumulant" on the linear subject, "deviation" otherwise
    reading: str = "cumulant"
    task_whitening: np.ndarray | None = None
    task_I3: np.ndarray | None = None

    @classmethod
    def fit(cls, subject: Subject, ref: ReferenceCell, X_fit: np.ndarray, geometry: Geometry, *, task_target: int = 0,
            n_pcs: int = 3, support_k: int = 5) -> "AlarmTimeReference":
        rep = subject.represent(X_fit, geometry)
        reading = "cumulant" if bool(getattr(subject, "is_linear", False)) else "deviation"
        means, seconds, thirds = {}, {}, {}
        for hook in HOOKS:
            m, s, t = pooled_stage(rep, hook)
            means[hook] = HookNull.fit(m)
            if s is not None:
                seconds[hook] = HookNull.fit(s)
            if t is not None:
                thirds[hook] = CumulantNull.fit_tensor(t, m, n_pcs, reading=reading)
        z = rep["z"]
        J = np.asarray(subject.jac_output(ref.z_mean, geometry), dtype=float)
        task = TaskMetric.one_hot(J, task_target, getattr(subject, "target_names", ()))
        # whitened-residual third cumulant: exact in the Gaussian-reference chart,
        # so its reading does not depend on subject linearity.
        resid = whitened_residual(subject, rep, geometry).reshape(z.shape[0], -1)
        resid_cumulant = CumulantNull.fit_residual(resid, n_pcs, reading="cumulant")
        # aligned cubic companion: I3 on reference-fit z in the M_task-whitened chart
        A = task.whitening()
        task_I3 = third_central_moment((z - z.mean(axis=0)) @ A.T)
        return cls(ref=ref, hook_mean_nulls=means, hook_second_nulls=seconds, hook_third_nulls=thirds,
                   resid_cumulant=resid_cumulant, z_null=HookNull.fit(z),
                   support=SupportEstimator.fit(z, k=support_k), task=task,
                   task_target=task_target, n_pcs=n_pcs, reading=reading, task_whitening=A, task_I3=task_I3)

    def to_dict(self) -> dict:
        return {"reading": self.reading,
                "reading_note": ("connected third cumulant of the whitened residual in the natural Gaussian-reference chart "
                                 "(exact); pooled-hook third moments read as a cumulant on a linear subject and as a "
                                 "deviation from the reference cell's third moment otherwise (D7)"),
                "hook_mean_nulls": {h: {"dim": n.dim, "reduced_dim": n.reduced_dim, "alpha": n.alpha, "n_ref": n.n_ref, "rule": n.rule}
                                    for h, n in self.hook_mean_nulls.items()},
                "hook_second_nulls": {h: {"dim": n.dim, "reduced_dim": n.reduced_dim, "alpha": n.alpha} for h, n in self.hook_second_nulls.items()},
                "hook_third_nulls": {h: n.to_dict() for h, n in self.hook_third_nulls.items()},
                "resid_cumulant": None if self.resid_cumulant is None else self.resid_cumulant.to_dict(),
                "support": None if self.support is None else {"k": self.support.k, "alpha": self.support.alpha, "validation": self.support.validation},
                "task_metric": None if self.task is None else {"declaration": self.task.declaration, "target": self.task_target,
                                                                 "target_names": list(self.task.target_names)}}


def input_quality_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """Model-free summaries of the raw window ``(n, C, N)``."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    flat = X.reshape(n, -1)
    rms = np.sqrt(np.mean(flat**2, axis=1))
    mu = flat.mean(axis=1, keepdims=True); sd = flat.std(axis=1, keepdims=True) + 1e-12
    kurt = np.mean(((flat - mu) / sd) ** 4, axis=1) - 3.0
    P = np.mean(np.abs(rfft(X, axis=-1)) ** 2, axis=1)[:, 1:]
    f = np.arange(1, P.shape[1] + 1, dtype=float)
    centroid = np.sum(P * f, axis=1) / np.maximum(np.sum(P, axis=1), 1e-300) / P.shape[1]
    flatness = np.exp(np.mean(np.log(P + 1e-300), axis=1)) / np.maximum(np.mean(P, axis=1), 1e-300)
    ch_rms = np.sqrt(np.mean(X**2, axis=2))
    spread = ch_rms.std(axis=1) / np.maximum(ch_rms.mean(axis=1), 1e-12)
    peak = np.max(np.abs(flat), axis=1) / np.maximum(rms, 1e-12)
    return {"in_rms": rms, "in_kurtosis": kurt, "in_spectral_centroid": centroid,
            "in_spectral_flatness": flatness, "in_channel_rms_spread": spread, "in_peak_abs": peak}


def build_alarm_time_features(subject: Subject, atref: AlarmTimeReference, inputs: AlarmTimeInputs) -> FeatureBatch:
    """All alarm-time features for the windows in ``inputs``. The only builder; returns a locked batch.

    The signature admits no truth, twin, label or realised covariance; every
    feature is tagged with the sources it used.
    """
    X = np.asarray(inputs.X, dtype=float)
    geometry = inputs.geometry
    n = X.shape[0]
    b = FeatureBatch(n=n)
    RW, RF, NR = "raw_window", "reference_fit", "noise_record"
    rep = subject.represent(X, geometry)
    ref = atref.ref
    for k, v in input_quality_features(X).items():
        b.add(k, v, RW)
    y = rep["output"]
    for i in range(y.shape[1]):
        b.add(f"out_{i}", y[:, i], {RW, RF})
    xw_hat = subject.decode(rep["z"], geometry)
    resid_raw = X - subject.sigma_hat.unwhiten(xw_hat)
    b.add("unc_recon_energy", np.mean(resid_raw.reshape(n, -1) ** 2, axis=1), {RW, RF})
    z = rep["z"]
    for i in range(z.shape[1]):
        b.add(f"z_{i}", z[:, i], {RW, RF})
    b.add("z_mahalanobis", atref.z_null.mahalanobis(z), {RW, RF})
    # generic_rich
    pooled = {h: pooled_stage(rep, h) for h in HOOKS}
    b.add("gr_input_maha", atref.hook_mean_nulls["whitened"].mahalanobis(pooled["whitened"][0]), {RW, RF})
    b.add("gr_output_maha", atref.hook_mean_nulls["output"].mahalanobis(pooled["output"][0]), {RW, RF})
    b.add("gr_pre_output_maha", atref.hook_mean_nulls["pre_output"].mahalanobis(pooled["pre_output"][0]), {RW, RF})
    d = z - ref.z_mean
    e = np.sum(d**2, axis=1) + 1e-300
    for name, P in (("out", ref.P_out), ("null", ref.P_null), ("resolved", ref.P_resolved), ("weak", ref.P_weak)):
        b.add(f"gr_z_energy_{name}", np.einsum("bi,ij,bj->b", d, P, d) / e, {RW, RF})
    r = whitened_residual(subject, rep, geometry)
    e_res = np.mean(np.sum(r**2, axis=-1), axis=1)
    b.add("gr_out_of_span", e_res / (e_res + np.sum(d**2, axis=1) + 1e-300), {RW, RF})
    b.add("gr_support_novelty", atref.support.score(z), {RW, RF})
    # generic_rich: whitened-residual third-cumulant scalars (D7; chart rule in AlarmTimeReference.reading)
    if atref.resid_cumulant is not None:
        Z = atref.resid_cumulant.compress(r.reshape(n, -1))
        for i in range(atref.n_pcs):
            b.add(f"gr_resid_c3_pc_{i}", Z[:, i], {RW, RF})
        b.add("gr_resid_c3_norm", Z[:, atref.n_pcs], {RW, RF})
        b.add("gr_resid_c3_maha", atref.resid_cumulant.null.mahalanobis(Z), {RW, RF})
    # intermediate_only: strictly internal hooks
    for h in INTERMEDIATE_HOOKS:
        m, s, t = pooled[h]
        b.add(f"im_{h}_mean_maha", atref.hook_mean_nulls[h].mahalanobis(m), {RW, RF})
        b.add(f"im_{h}_second_maha", atref.hook_second_nulls[h].mahalanobis(s) if s is not None else np.full(n, np.nan), {RW, RF})
        if h in atref.hook_third_nulls:
            b.add(f"im_{h}_third_maha", atref.hook_third_nulls[h].mahalanobis(t), {RW, RF})
        pcs = atref.hook_mean_nulls[h].coordinates(m, atref.n_pcs)
        for i in range(atref.n_pcs):
            b.add(f"im_{h}_pc_{i}", pcs[:, i], {RW, RF})
    # capacity control: quadratic expansion of generic_rich scalars (now including the
    # residual cumulant scalars), truncated to the intermediate feature count
    n_im = len(INTERMEDIATE_HOOKS) * (3 + atref.n_pcs)
    quad_names = list(QUADRATIC_BASIS) + [f"gr_resid_c3_pc_{i}" for i in range(atref.n_pcs)]
    basis = [b[k] for k in quad_names if k in b]
    quad: list[np.ndarray] = [v * v for v in basis]
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            quad.append(basis[i] * basis[j])
    for i in range(n_im):
        b.add(f"gq_{i}", quad[i] if i < len(quad) else np.zeros(n), {RW, RF})
    # legacy layerwise (pass-1 diagnostic)
    for h in HOOKS:
        b.add(f"lw_{h}_maha", atref.hook_mean_nulls[h].mahalanobis(pooled[h][0]), {RW, RF})
    # Claim-2 raw scores (calibrated later on clean calibration windows)
    b.add("raw_task_sensitive", atref.task.length(d), {RW, RF})
    if atref.task_I3 is not None and atref.task_whitening is not None:
        d_w = d @ atref.task_whitening.T
        b.add("raw_task_cubic", atref.task.cubic_aligned(d_w, atref.task_I3), {RW, RF})
    if inputs.noise_records is not None:
        for k, v in noise_only_features(subject, atref, inputs.noise_records, geometry).items():
            b.add(k, v, {NR, RF})
    return b.lock()


def noise_only_features(subject: Subject, atref: AlarmTimeReference, records: np.ndarray, geometry: Geometry) -> dict[str, np.ndarray]:
    """Per-window statistics of the random-trigger records ``(n, m, C, N)`` against the reference noise null."""
    ref = atref.ref
    R = np.asarray(records, dtype=float)
    n, m = R.shape[:2]
    rep = subject.represent(R.reshape(n * m, *R.shape[2:]), geometry)
    scale = float(getattr(subject, "noise_variance_scale", lambda a, b: 1.0)(ref.geometry, geometry))
    zn = (rep["z"] @ ref.exc_basis).reshape(n, m, -1)
    var_ratio = np.mean(zn**2, axis=1) / np.maximum(ref.noise_z_var * scale, 1e-15)
    rn = whitened_residual(subject, rep, geometry).reshape(n, m, R.shape[2], R.shape[3])
    psd = np.mean(np.abs(rfft(rn, axis=-1)) ** 2, axis=(1, 2))
    ratio = psd / np.maximum(ref.noise_residual_psd, 1e-15)
    dev_s = np.array([float(np.max(np.abs(smooth(rw[1:], PSD_SMOOTH) - 1.0))) for rw in ratio])
    dev_l = np.max(np.abs(ratio[:, 1:] - 1.0), axis=1)

    def corr(c):
        s = np.sqrt(np.clip(np.diag(c), 1e-15, None)); return c / np.outer(s, s)

    cc = np.full(n, np.nan)
    if rn.shape[2] == ref.noise_residual_chan_cov.shape[0]:
        ref_corr = corr(ref.noise_residual_chan_cov)
        for i in range(n):
            c = np.cov(rn[i].transpose(1, 0, 2).reshape(rn.shape[2], -1))
            cc[i] = float(np.linalg.norm(corr(c) - ref_corr))
    return {"no_var_ratio": np.mean(var_ratio, axis=1), "no_psd_dev_smooth": dev_s, "no_psd_dev_line": dev_l, "no_chan_corr_shift": cc}


class NullCalibrator:
    """Map raw scores to a common clean-null scale: Φ⁻¹ of the empirical clean CDF (linear tails).

    Fitted on clean *calibration* windows only. A calibrated value of 0 is the
    clean median, 1.64 the clean 95th percentile, and so on; scores from
    different hooks become comparable before any max/mean combination.
    """

    def __init__(self):
        self.sorted: dict[str, np.ndarray] = {}

    def fit(self, batch: FeatureBatch, names) -> "NullCalibrator":
        for n in names:
            v = np.asarray(batch[n], dtype=float)
            v = v[np.isfinite(v)]
            if len(v) < 5:
                raise ValueError(f"need ≥ 5 finite clean values to calibrate {n!r}, got {len(v)}")
            self.sorted[n] = np.sort(v)
        return self

    def calibrate(self, name: str, s: np.ndarray) -> np.ndarray:
        ref = self.sorted[name]
        n = len(ref)
        s = np.asarray(s, dtype=float)
        rank = np.searchsorted(ref, s, side="right")
        p = (rank + 0.5) / (n + 1.0)
        p = np.clip(p, 0.5 / (n + 1.0), 1.0 - 0.5 / (n + 1.0))
        out = ndtri(p)
        # linear extrapolation beyond the clean range so that ordering is preserved above the max
        above = s > ref[-1]
        if above.any():
            scale = max(float(ref[-1] - ref[max(n // 2, 0)]), 1e-12)
            out[above] = ndtri(1.0 - 0.5 / (n + 1.0)) + (s[above] - ref[-1]) / scale
        out[~np.isfinite(s)] = np.nan
        return out

    def combine_max(self, batch: FeatureBatch, names) -> np.ndarray:
        cols = np.column_stack([self.calibrate(n, batch[n]) for n in names])
        return np.nanmax(cols, axis=1)
