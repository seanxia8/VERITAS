# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Class 1 — the optimal filter (OF): the fixed rank-1 endpoint.

A known template ``s`` and the GLS / matched-filter amplitude

    â = (sᴴ Σ⁻¹ x) / (sᴴ Σ⁻¹ s),

computed either in the frequency domain with inverse-PSD weights ``1/J_k``
(:func:`gls_amplitude`, the audited Paper 1 form) or in any coordinates with
a diagonal or full metric (:func:`gls_amplitude_metric`). Zero learned
parameters. The one latent *is* the amplitude, and because ``wᴴ s = 1`` the
estimate stays unbiased under Σ̂ ≠ Σ — only its variance inflates. That is
the "no mean shift" row of the monitoring table (EXPERIMENT_DESIGN.md §III.1).

Provenance: ``src/noise_geometry/filters/optimal.py`` in the Paper 1
experiment repository (``noise-weighted-subspace-reconstruction`` @
``ea076ff``, 2026-09-10), copied with the frequency-domain functions
unchanged; :func:`gls_amplitude_metric` and :class:`OptimalFilter` are
additions for use as a ``latent_monitor`` subject stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._metric import as_metric, m_right


# --------------------------------------------------------------------------- #
# Frequency-domain form (rFFT traces, one-sided inverse-PSD weights) — audited
# --------------------------------------------------------------------------- #
def weighted_inner(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> complex:
    """Return ``sum(conj(a) * b * w)``."""
    return np.sum(np.conj(a) * b * w)


def gls_amplitude(X_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray, *, return_complex: bool = False):
    """Estimate OF/GLS amplitudes in the weighted frequency-domain inner product."""
    X_arr = np.asarray(X_f)
    single = X_arr.ndim == 1
    X2 = np.atleast_2d(X_arr)
    s = np.asarray(template_f)
    w = np.asarray(weights, dtype=np.float64)
    denom = np.real(weighted_inner(s, s, w))
    if denom <= 0:
        raise ValueError("template has zero weighted norm")
    amps = np.sum(np.conj(s)[None, :] * X2 * w[None, :], axis=1) / denom
    if not return_complex:
        amps = np.real(amps)
    return amps[0] if single else amps


def project_rank1(X_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray, *, return_amp: bool = False):
    """Project traces onto a fixed rank-1 template under the inverse-noise metric."""
    X_arr = np.asarray(X_f)
    single = X_arr.ndim == 1
    X2 = np.atleast_2d(X_arr)
    amps = np.asarray(gls_amplitude(X2, template_f, weights))
    recon = amps[:, None] * np.asarray(template_f)[None, :]
    if single:
        recon = recon[0]
        amps = float(amps[0])
    return (recon, amps) if return_amp else recon


def matched_filter_score(X_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return normalized matched-filter scores for one or more traces."""
    X_arr = np.atleast_2d(np.asarray(X_f))
    s = np.asarray(template_f)
    w = np.asarray(weights, dtype=np.float64)
    denom = np.sqrt(np.real(weighted_inner(s, s, w)))
    if denom <= 0:
        raise ValueError("template has zero weighted norm")
    score = np.real(np.sum(np.conj(s)[None, :] * X_arr * w[None, :], axis=1)) / denom
    return score[0] if np.asarray(X_f).ndim == 1 else score


def psd_amplitude_variance(
    template_f: np.ndarray,
    psd: np.ndarray,
    weights: np.ndarray,
    *,
    trace_len: int,
    sampling_frequency: float,
) -> float:
    """Predict OF amplitude variance for the Paper 1 rFFT convention.

    Includes the FFT and one-sided-PSD normalization used by the noise
    generator (``generate_colored_noise`` in the source repository) rather than
    assuming dimensionless unit noise. Under Σ̂ ≠ Σ pass the *realized* PSD as
    ``psd`` and the *assumed* inverse PSD as ``weights`` to get the inflated
    variance of the mismatched filter.
    """
    s = np.asarray(template_f)
    J = np.asarray(psd, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if s.shape != J.shape or s.shape != w.shape:
        raise ValueError("template_f, psd, and weights must have matching shapes")
    den = np.real(weighted_inner(s, s, w))
    if den <= 0:
        raise ValueError("template has zero weighted norm")

    scale = float(sampling_frequency) * int(trace_len)
    variance_num = 0.0
    if s.size > 1:
        stop = -1 if trace_len % 2 == 0 else None
        variance_num += float(np.sum((w[1:stop] ** 2) * J[1:stop] * scale * np.abs(s[1:stop]) ** 2 / 4.0))
    if trace_len % 2 == 0 and s.size > 1:
        variance_num += float(w[-1] ** 2 * J[-1] * scale * np.real(s[-1]) ** 2)
    if w[0] != 0:
        variance_num += float(w[0] ** 2 * J[0] * scale / 2.0 * np.real(s[0]) ** 2)
    return variance_num / den**2


# --------------------------------------------------------------------------- #
# Metric form (any coordinates; diagonal or full Σ̂⁻¹) — addition
# --------------------------------------------------------------------------- #
def gls_amplitude_metric(X: np.ndarray, template: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``â = (sᵀ M x) / (sᵀ M s)`` for row-wise real samples and metric ``M``."""
    X2 = np.atleast_2d(np.asarray(X, dtype=np.float64))
    s = np.asarray(template, dtype=np.float64)
    w = as_metric(weights)
    Ms = m_right(s[None, :], w)[0]
    denom = float(s @ Ms)
    if denom <= 0:
        raise ValueError("template has zero weighted norm")
    amps = (X2 @ Ms) / denom
    return amps[0] if np.asarray(X).ndim == 1 else amps


def gls_amplitude_variance_metric(template: np.ndarray, weights: np.ndarray, realized_cov: np.ndarray | None = None) -> float:
    """Variance of :func:`gls_amplitude_metric` under the realized covariance.

    With ``realized_cov=None`` the filter is assumed matched (Σ = M⁻¹) and the
    variance is ``1 / (sᵀ M s)``. Otherwise ``Var â = wᵀ Σ w`` with
    ``w = M s / (sᵀ M s)`` — the mismatched-filter variance, always ≥ matched.
    """
    s = np.asarray(template, dtype=np.float64)
    w = as_metric(weights)
    Ms = m_right(s[None, :], w)[0]
    denom = float(s @ Ms)
    if realized_cov is None:
        return 1.0 / denom
    filt = Ms / denom
    Sigma = np.asarray(realized_cov, dtype=np.float64)
    if Sigma.ndim == 1:
        return float(np.sum(filt**2 * Sigma))
    return float(filt @ Sigma @ filt)


@dataclass
class OptimalFilter:
    """Rank-1 subject stage: ``encode`` → amplitude, ``reconstruct`` → â·s.

    ``template`` and ``weights`` are in the same coordinates as the data
    (time domain with a full/diagonal metric, or rFFT domain with inverse-PSD
    weights via ``frequency_domain=True``).
    """

    template: np.ndarray
    weights: np.ndarray
    frequency_domain: bool = False

    @property
    def rank(self) -> int:
        return 1

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self.frequency_domain:
            return np.atleast_1d(gls_amplitude(X, self.template, self.weights))
        return np.atleast_1d(gls_amplitude_metric(X, self.template, self.weights))

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        amps = self.encode(X)
        recon = amps[:, None] * np.asarray(self.template)[None, :]
        return recon[0] if np.asarray(X).ndim == 1 else recon

    def basis(self) -> np.ndarray:
        """The (d, 1) decoder basis — the template itself."""
        return np.asarray(self.template)[:, None]
