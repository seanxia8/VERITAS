# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Physics-validation helpers for the LUCiD front-end preset.

These are the checks a reviewer asked for (24 Sep 2026): the SPE/front-end
bandwidth consistency (P0.3), the realized-vs-implied cross-spectral density
(P2.3) and the sensitivity of the crate grouping to the board map (P1.1).
"""
from __future__ import annotations

import numpy as np

from .adapter import crate_preset, matched_cell_kappa_floor
from .presets import FS_L, FRONT_END_CORNER_HZ, PMT_FRONTEND_V2, spe_bandwidth_hz


def kappa_floor_sweep(Ns=(512, 2048, 8192, 32768, 65536, 262144), C: int = 64, seed: int = 1,
                      target: float = 1.1, **crate_overrides) -> dict:
    """Matched-cell kappa floor vs N/C, and the window that meets ``target``.

    The floor is estimator noise: at the default 512 ns window (N/C = 8) it is
    large, so a covariance cell must lengthen the window (or shrink the crate).
    Returns the table and the first ``N`` whose floor is below ``target``.
    """
    rows = []
    for N in Ns:
        k = matched_cell_kappa_floor(int(N), C=C, seed=seed, **crate_overrides)
        rows.append({"N": int(N), "N_over_C": int(N) / C, "kappa_floor": k})
    recommended = next((r for r in rows if r["kappa_floor"] <= target), None)
    return {"rows": rows, "target": target, "crate_size": C, "recommended": recommended}


def bandwidth_report(preset: dict | None = None, N: int = 512) -> dict:
    """Report the SPE band against the front-end noise band.

    ``signal_f3db_hz`` is the SPE power -3 dB; ``noise_above_signal_fraction``
    is the fraction of front-end noise power above that frequency. A matched
    system keeps this small; the placeholder targets < ~0.2.
    """
    from noise_module import NoiseGenerator

    preset = PMT_FRONTEND_V2 if preset is None else preset
    f, s = NoiseGenerator(preset, seed=0).build_psd_density(N)
    df = float(preset["sampling_frequency"]) / N
    signal_f = spe_bandwidth_hz()
    total = float(np.sum(s[1:]) * df)
    return {
        "signal_f3db_hz": signal_f,
        "noise_corner_hz": FRONT_END_CORNER_HZ,
        "bandwidth_ratio": FRONT_END_CORNER_HZ / signal_f,
        # the truly wasted noise: above 3x the signal -3 dB, where the signal is negligible
        "noise_above_3x_signal_fraction": float(np.sum(s[f >= 3.0 * signal_f]) * df) / total if total > 0 else 0.0,
        "total_power": total,
    }


def realized_csd_check(preset: tuple[dict, dict] | None = None, N: int = 1 << 16, C: int = 2,
                       seed: int = 7) -> dict:
    """Compare the realized Welch coherence to the implied ``rho_ij(f)``.

    Returns the frequency of the coherence peak, the implied and realized rho
    there, and the median absolute difference over the band. The Welch estimate
    needs a long record; the default is 65.5 us.
    """
    from scipy.signal import csd, welch
    from noise_module import MultiChannelNoiseGenerator

    base, crate = crate_preset() if preset is None else preset
    crate = {**crate, "n_channels": C, "channel_gain_jitter": 0.0,
             "private_strength_range": [1.0, 1.0]}
    gen = MultiChannelNoiseGenerator(base, crate, seed=seed)
    X, m = gen.generate(N, return_metadata=True)
    f_rho, rho_implied = MultiChannelNoiseGenerator.implied_correlation_spectrum(m, 0, 1)
    f_w, s01 = csd(X[0], X[1], fs=FS_L, nperseg=512)
    _, s00 = welch(X[0], fs=FS_L, nperseg=512)
    _, s11 = welch(X[1], fs=FS_L, nperseg=512)
    rho_real = np.real(s01) / np.sqrt(s00 * s11)
    implied_on_w = np.interp(f_w, f_rho, rho_implied)
    peak = int(np.argmax(implied_on_w))
    return {
        "peak_frequency_hz": float(f_w[peak]),
        "rho_implied_peak": float(implied_on_w[peak]),
        "rho_realized_peak": float(rho_real[peak]),
        "median_abs_diff": float(np.median(np.abs(rho_real - implied_on_w))),
        "frequencies": f_w,
        "rho_implied": implied_on_w,
        "rho_realized": rho_real,
    }


def grouping_report(positions: np.ndarray, groups: list[np.ndarray]) -> dict:
    """Spatial compactness of a grouping: intra-group vs centroid separation."""
    positions = np.asarray(positions, dtype=float)
    intra = []
    centroids = []
    for g in groups:
        p = positions[np.asarray(g)]
        if len(p) > 1:
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            intra.append(d[np.triu_indices(len(p), 1)].mean())
        centroids.append(p.mean(axis=0))
    centroids = np.asarray(centroids)
    if len(centroids) > 1:
        dc = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        inter = dc[np.triu_indices(len(centroids), 1)].mean()
    else:
        inter = 0.0
    mean_intra = float(np.mean(intra)) if intra else 0.0
    return {"mean_intra_distance_m": mean_intra, "mean_centroid_distance_m": float(inter),
            "compactness": float(inter / mean_intra) if mean_intra > 0 else float("inf")}


def grouping_sensitivity(C: int = 64, group_size: int | None = None,
                         positions: np.ndarray | None = None, **crate_overrides) -> dict:
    """Spatial compactness per grouping method, on ``C`` channels.

    The per-crate covariance is set by the preset, not the grouping, so the
    compactness is the limitation to disclose: a grouping whose ``compactness``
    (mean centroid distance / mean intra-group distance) is low is not a
    physical board map.
    """
    from .grouping import channel_groups

    group_size = max(2, C // 4) if group_size is None else group_size
    if positions is None:
        positions = np.random.default_rng(0).normal(scale=1.0, size=(C, 3))
    out = {}
    for method in ("contiguous", "z_plane", "proximity"):
        groups = channel_groups(C, group_size, positions=positions, method=method)
        out[method] = {**grouping_report(positions, groups), "n_groups": len(groups)}
    return out
