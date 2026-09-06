# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The whitening layer W: an assumed covariance Σ̂ as a *named parameter*.

Σ̂ is modelled as a Kronecker product — a channel covariance Σ_c (C×C) times a
stationary, circulant temporal covariance given by a one-sided PSD S(f) on
the rFFT grid. That is exactly the family :class:`noise_module.MultiChannelNoiseGenerator`
samples from (frequency-domain synthesis is circulant-stationary; the
shared-private and low-rank modes share one spectrum across channels), so the
subject's assumption and the generator's implied covariance live in the same
parametrisation and κ(Σ̂⁻¹Σ) is a matrix ratio, not a metaphor.

Whitening is applied as ``x̃ = Σ_c^{-1/2} · irfft( rfft(x) / sqrt(S_k · scale) )``,
which is orthogonal up to the Kronecker structure, so a tied linear decoder in
whitened coordinates has Fisher information J_gᵀ Σ̂⁻¹ J_g = AAᵀ exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq


def psd_to_bin_variance(psd: np.ndarray, sampling_frequency: float, n_samples: int) -> np.ndarray:
    """Expected |rfft|² per bin for a one-sided PSD in units²/Hz.

    Matches ``NoiseGenerator.psd_density_to_rfft_power``: with the unnormalised
    rfft, ``E|X_k|² = S_k · fs · N / 2`` for interior bins and ``S_k · fs · N``
    at DC and Nyquist (one-sided convention).
    """
    psd = np.asarray(psd, dtype=float)
    power = psd * float(sampling_frequency) * int(n_samples) / 2.0
    power[0] = psd[0] * float(sampling_frequency) * int(n_samples)
    if int(n_samples) % 2 == 0:
        power[-1] = psd[-1] * float(sampling_frequency) * int(n_samples)
    return power


@dataclass
class KroneckerWhitener:
    """Σ̂ = Σ_c ⊗ Circ(S), with forward and inverse square-root operators."""

    channel_cov: np.ndarray          # (C, C)
    psd: np.ndarray                  # (N//2+1,) one-sided, units²/Hz
    sampling_frequency: float
    n_samples: int
    floor: float = 1e-12

    def __post_init__(self) -> None:
        self.channel_cov = np.asarray(self.channel_cov, dtype=float)
        self.psd = np.asarray(self.psd, dtype=float)
        C = self.channel_cov.shape[0]
        if self.channel_cov.shape != (C, C):
            raise ValueError("channel_cov must be square.")
        if self.psd.shape != (self.n_samples // 2 + 1,):
            raise ValueError("psd must be on the rfft grid of n_samples.")
        w, V = np.linalg.eigh(0.5 * (self.channel_cov + self.channel_cov.T))
        w = np.clip(w, self.floor, None)
        self._c_isqrt = (V * w**-0.5) @ V.T
        self._c_sqrt = (V * w**0.5) @ V.T
        bin_var = psd_to_bin_variance(self.psd, self.sampling_frequency, self.n_samples)
        # DC carries no variance under the 'variance' power definition; leave it unscaled.
        bin_var = np.where(bin_var > self.floor, bin_var, 1.0)
        self._t_isqrt = bin_var**-0.5
        self._t_sqrt = bin_var**0.5
        # Per-sample scaling so that whitened noise has unit variance per sample:
        # with unit-variance rfft bins, Var[x_n] = (1/N²)·Σ_k c_k·1 = 1/N (c_k = 1 at DC
        # and Nyquist, 2 elsewhere), hence the factor sqrt(N).
        self._sample_scale = float(np.sqrt(self.n_samples))

    @property
    def n_channels(self) -> int:
        return self.channel_cov.shape[0]

    @property
    def mean_offdiag_correlation(self) -> float:
        """The channel-correlation *assumption* carried to other channel counts."""
        C = self.n_channels
        if C < 2:
            return 0.0
        s = np.sqrt(np.clip(np.diag(self.channel_cov), self.floor, None))
        corr = self.channel_cov / np.outer(s, s)
        return float(corr[np.triu_indices(C, 1)].mean())

    def for_channels(self, C: int) -> "KroneckerWhitener":
        """Σ̂ for a different channel count.

        The whitening layer carries an assumption about channel *structure*, not
        a particular C×C matrix: for a geometry with a different sensor count the
        same mean off-diagonal correlation ρ̄ is assumed, Σ̂_c = ρ̄·11ᵀ + (1−ρ̄)·I,
        with the reference's mean channel variance. That is what "the model was
        trained under Σ̂" means when the geometry moves.
        """
        if int(C) == self.n_channels:
            return self
        rho = self.mean_offdiag_correlation
        var = float(np.mean(np.diag(self.channel_cov)))
        Sc = var * (rho * np.ones((C, C)) + (1.0 - rho) * np.eye(C))
        return KroneckerWhitener(Sc, self.psd, self.sampling_frequency, self.n_samples, self.floor)

    @property
    def frequencies(self) -> np.ndarray:
        return rfftfreq(self.n_samples, d=1.0 / self.sampling_frequency)

    def whiten(self, X: np.ndarray) -> np.ndarray:
        """(…, C, N) → (…, C, N) in whitened units (unit variance per sample under Σ̂)."""
        X = np.asarray(X, dtype=float)
        if X.shape[-2] != self.n_channels:
            return self.for_channels(X.shape[-2]).whiten(X)
        F = rfft(X, axis=-1) * self._t_isqrt
        F = np.einsum("ij,...jk->...ik", self._c_isqrt, F)
        return irfft(F, n=self.n_samples, axis=-1) * self._sample_scale

    def unwhiten(self, Xw: np.ndarray) -> np.ndarray:
        Xw = np.asarray(Xw, dtype=float)
        if Xw.shape[-2] != self.n_channels:
            return self.for_channels(Xw.shape[-2]).unwhiten(Xw)
        Xw = Xw / self._sample_scale
        F = rfft(Xw, axis=-1) * self._t_sqrt
        F = np.einsum("ij,...jk->...ik", self._c_sqrt, F)
        return irfft(F, n=self.n_samples, axis=-1)

    def with_channel_cov(self, channel_cov: np.ndarray) -> "KroneckerWhitener":
        return KroneckerWhitener(channel_cov, self.psd, self.sampling_frequency, self.n_samples, self.floor)

    def with_psd(self, psd: np.ndarray) -> "KroneckerWhitener":
        return KroneckerWhitener(self.channel_cov, psd, self.sampling_frequency, self.n_samples, self.floor)

    def kappa_against(self, channel_cov: np.ndarray, psd: np.ndarray | None = None) -> dict[str, float]:
        """κ(Σ̂⁻¹Σ) split into its Kronecker factors.

        For Σ̂ = Σ̂_c ⊗ T̂ and Σ = Σ_c ⊗ T the eigenvalues of Σ̂⁻¹Σ are the
        products of the two factors' eigenvalues, so κ_total = κ_channel · κ_temporal.
        """
        Sc = np.asarray(channel_cov, dtype=float)
        ratio_c = self._c_isqrt @ Sc @ self._c_isqrt
        ev_c = np.linalg.eigvalsh(0.5 * (ratio_c + ratio_c.T))
        kappa_c = float(np.max(ev_c) / max(np.min(ev_c), self.floor))
        if psd is None:
            kappa_t = 1.0
        else:
            r = np.asarray(psd, dtype=float)[1:] / np.maximum(self.psd[1:], self.floor)
            kappa_t = float(np.max(r) / max(np.min(r), self.floor))
        return {"kappa_channel": kappa_c, "kappa_temporal": kappa_t, "kappa": kappa_c * kappa_t}

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "kronecker",
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "sampling_frequency": self.sampling_frequency,
            "channel_cov": self.channel_cov.tolist(),
            "psd": self.psd.tolist(),
        }


def estimate_kronecker(noise: np.ndarray, sampling_frequency: float, psd_smoothing: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (Σ_c, S(f)) from noise-only records ``(n_records, C, N)``.

    Σ_c is the time-averaged channel covariance normalised so that the
    temporal PSD carries the absolute scale; S(f) is the channel-averaged
    one-sided periodogram, optionally boxcar-smoothed. This is the estimator
    behind the re-whitening adjustment (plan §4, row Σ-covariance).
    """
    R = np.asarray(noise, dtype=float)
    if R.ndim != 3:
        raise ValueError("noise must be (n_records, C, N).")
    n_rec, C, N = R.shape
    fs = float(sampling_frequency)
    F = rfft(R, axis=-1)
    power = np.mean(np.abs(F) ** 2, axis=(0, 1))          # (N//2+1,)
    psd = 2.0 * power / (fs * N)
    if N % 2 == 0:
        psd[-1] = power[-1] / (fs * N)
    if psd_smoothing > 1:
        k = np.ones(psd_smoothing) / psd_smoothing
        pad = np.pad(psd[1:], (psd_smoothing // 2, psd_smoothing - 1 - psd_smoothing // 2), mode="edge")
        psd[1:] = np.convolve(pad, k, mode="valid")
    # 'variance' convention: DC carries no variance and must not be whitened against a
    # near-zero estimate; and no bin may fall so far below the rest that whitening blows up.
    psd[0] = 0.0
    floor_rel = 1e-4 * float(np.median(psd[1:]))
    psd[1:] = np.maximum(psd[1:], floor_rel)
    flat = R.transpose(1, 0, 2).reshape(C, -1)
    cov = np.cov(flat)
    cov = cov / np.mean(np.diag(cov))                      # unit average diagonal
    return cov, psd
