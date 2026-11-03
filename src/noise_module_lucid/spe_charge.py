# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Tabulated single-photoelectron (SPE) charge spectrum (R-7 v2, the ``pmt-response-v3`` default).

WCSim samples one SPE charge per photoelectron from a 500-bin cumulative table
(``WCSimPMTObject::Getqpe``) with ``WCSimWCPMT::rn1pe``:

    find the first bin i with u1 <= cdf[i]; charge = (i - 50 + u2) / 22.83

so the charge is uniform inside bin ``i`` and the bin probabilities are the
table increments. :class:`TabulatedSPE` reproduces that distribution exactly
with one uniform ``u`` per photoelectron, by inverting the piecewise-linear CDF
(the same distribution as WCSim's two-uniform draw, but one keyed identity per
photoelectron, so a response cell never shifts another photoelectron's draw).

Author decision, 3 Nov 2026 (R-7 v2 item 4): the Gaussian+exponential fit of the
WCTE 3-inch table missed the KS tolerance declared before fitting (0.031 > 0.02),
so the table is sampled directly instead of fitted. With ``normalise_mean`` the
charge is divided by the table mean (1.0648 PE for ``PMT3inchR14374_WCTE``), as
R-7 v2 asks ("with the mean divided out"), so the mean SPE charge is 1 PE and the
0.25 PE threshold, the walk and the width act on charge in mean-SPE units.

NumPy only; no WCSim or LUCiD import.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabulatedSPE:
    """Inverse-CDF sampler of a WCSim ``Getqpe`` table.

    ``cdf`` holds the 500 cumulative values (WCSim's trailing dummy element dropped);
    bin ``i`` covers raw charge ``[(i - bin_offset), (i - bin_offset + 1)) / bins_per_pe``.
    """

    cdf: tuple[float, ...]
    bin_offset: int = 50
    bins_per_pe: float = 22.83
    normalise_mean: bool = True

    def __post_init__(self):
        c = np.asarray(self.cdf, dtype=float)
        if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)):
            raise ValueError("SPE table must be a finite one-dimensional CDF")
        if np.any(np.diff(c) < 0) or c[0] < 0 or c[-1] != 1.0:
            raise ValueError("SPE table must be non-decreasing in [0, 1] and end at exactly 1")
        if not (np.isfinite(self.bins_per_pe) and self.bins_per_pe > 0) or not isinstance(self.bin_offset, (int, np.integer)):
            raise ValueError("positive bins_per_pe and an integer bin_offset required")
        pmf = np.diff(np.concatenate([[0.0], c]))
        lo = (np.arange(len(c)) - self.bin_offset) / self.bins_per_pe
        mid = lo + 0.5 / self.bins_per_pe
        mean = float(np.sum(pmf * mid))
        second = float(np.sum(pmf * (mid ** 2 + (1.0 / self.bins_per_pe) ** 2 / 12.0)))
        if self.normalise_mean and not mean > 0:
            raise ValueError("mean normalisation needs a positive table mean")
        object.__setattr__(self, "_pmf", pmf)
        object.__setattr__(self, "_lo", lo)
        object.__setattr__(self, "_raw_mean", mean)
        object.__setattr__(self, "_raw_var", second - mean ** 2)

    @property
    def raw_mean_pe(self) -> float:
        """Mean of the raw WCSim charge ``(i - bin_offset + u) / bins_per_pe``."""
        return self._raw_mean

    @property
    def scale(self) -> float:
        """Multiplier applied to the raw charge (``1 / raw_mean_pe`` when normalised)."""
        return 1.0 / self._raw_mean if self.normalise_mean else 1.0

    @property
    def mean_pe(self) -> float:
        return self._raw_mean * self.scale

    @property
    def sigma_pe(self) -> float:
        return float(np.sqrt(self._raw_var)) * self.scale

    @property
    def relative_sigma(self) -> float:
        return float(np.sqrt(self._raw_var)) / self._raw_mean

    def sample(self, u):
        """One SPE charge (PE) per uniform ``u`` in (0, 1): WCSim's ``rn1pe`` distribution."""
        u = np.asarray(u, dtype=float)
        if np.any(~np.isfinite(u)) or np.any((u <= 0.0) | (u >= 1.0)):
            raise ValueError("uniforms must lie strictly inside (0, 1)")
        c = np.asarray(self.cdf, dtype=float)
        i = np.searchsorted(c, u, side="left")          # first bin with u <= cdf[i], as rn1pe
        below = np.where(i > 0, c[np.maximum(i - 1, 0)], 0.0)
        frac = (u - below) / self._pmf[i]               # position inside the bin, uniform in [0, 1]
        return (self._lo[i] + frac / self.bins_per_pe) * self.scale

    def cdf_at(self, q):
        """Exact CDF of the sampled charge (PE), piecewise linear inside each bin."""
        x = np.asarray(q, dtype=float) / self.scale * self.bins_per_pe + self.bin_offset
        c = np.concatenate([[0.0], np.asarray(self.cdf, dtype=float)])
        k = np.clip(np.floor(x).astype(int), 0, len(self.cdf))
        f = np.clip(x - k, 0.0, 1.0)
        nxt = c[np.minimum(k + 1, len(self.cdf))]
        out = c[k] + (nxt - c[k]) * f
        return np.where(x <= 0, 0.0, np.where(x >= len(self.cdf), 1.0, out))
