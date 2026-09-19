# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Add LUCiD front-end noise crate by crate, reporting the covariance it drew."""
from __future__ import annotations

from typing import Any

import numpy as np

from .grouping import contiguous_groups
from .presets import GROUP, PMT_CRATE_V2, PMT_FRONTEND_V2, preset_for
from .units import FS_L, charge_to_mv, spe_template


def crate_noise(base: dict, crate: dict, n_samples: int, seed: int = 0,
                n_channels: int | None = None) -> tuple[np.ndarray, dict]:
    """Draw one crate of noise; return ``(noise (C, N), metadata)``."""
    from noise_module import MultiChannelNoiseGenerator

    crate_cfg = dict(crate)
    if n_channels is not None:
        crate_cfg["n_channels"] = int(n_channels)
    gen = MultiChannelNoiseGenerator(base, crate_cfg, seed=seed)
    noise, meta = gen.generate(n_samples, return_metadata=True)
    return noise, meta


def kappa(meta: dict) -> float:
    """cond(Σ̂⁻¹ Σ) for one group's record — the matched-cell estimator floor."""
    implied = np.asarray(meta["implied_covariance"], dtype=float)
    realized = np.asarray(meta["realized_covariance"], dtype=float)
    return float(np.linalg.cond(np.linalg.solve(implied, realized)))


def add_readout_noise(charge_waveform: np.ndarray, groups: list[np.ndarray] | None = None,
                      preset: tuple[dict, dict] | None = None, seed: int = 0, *,
                      in_units: str = "pe", spe: np.ndarray | None = None,
                      window_ns: float | None = None, group_size: int = GROUP,
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    """Add V2/LONG front-end noise to a LUCiD waveform, crate by crate.

    Parameters
    ----------
    charge_waveform : ``(C, N)`` array
        LUCiD's per-sensor waveform in photoelectrons (``in_units="pe"``) or an
        already-calibrated voltage trace in mV (``in_units="mv"``).
    groups : list of index arrays, optional
        The covariance units. Defaults to contiguous crates of ``group_size``.
    preset : ``(base, crate)``, optional
        As returned by :func:`noise_module_lucid.presets.crate_preset` or
        :func:`preset_for`. Defaults to V2 (or the long preset when
        ``window_ns`` is given and large).
    in_units : ``"pe"`` or ``"mv"``
        Whether to run the SPE convolution first.
    window_ns : float, optional
        LUCiD window length; used only to select the default preset and recorded
        in provenance.

    Returns
    -------
    trace_mv : ``(C, N)``
    metadata : dict
        ``{"preset", "window_ns", "in_units", "groups": [...], "covariance":
        [...per-group metadata...], "kappa": [...], "spe_mv_per_pe": ...}``.
    """
    if in_units not in {"pe", "mv"}:
        raise ValueError("in_units must be 'pe' or 'mv'.")
    charge_waveform = np.asarray(charge_waveform, dtype=float)
    if charge_waveform.ndim != 2:
        raise ValueError("charge_waveform must be (C, N).")
    C, N = charge_waveform.shape

    if in_units == "pe":
        spe = spe_template() if spe is None else np.asarray(spe, dtype=float)
        trace = charge_to_mv(charge_waveform, spe)
    else:
        trace = charge_waveform.copy()

    if preset is None:
        if window_ns is not None:
            preset = preset_for(window_ns)
        else:
            preset = (PMT_FRONTEND_V2, PMT_CRATE_V2)
    base, crate = preset

    if groups is None:
        groups = contiguous_groups(C, group_size)
    groups = [np.asarray(g, dtype=int) for g in groups]

    cov_meta: list[dict] = []
    kappas: list[float] = []
    for g0, idx in enumerate(groups):
        noise, m = crate_noise(base, crate, N, seed=seed + int(idx[0]),
                               n_channels=len(idx))
        trace[idx] += noise
        cov_meta.append(m)
        kappas.append(kappa(m))

    metadata = {
        "preset": base.get("noise_type", "composite"),
        "window_ns": float(window_ns) if window_ns is not None else None,
        "in_units": in_units,
        "sampling_frequency": FS_L,
        "n_channels": int(C),
        "n_samples": int(N),
        "groups": [g.tolist() for g in groups],
        "covariance": cov_meta,
        "kappa": kappas,
        "spe_mv_per_pe": float(spe.max()) if in_units == "pe" and spe is not None else None,
    }
    return trace, metadata


def add_pmt_noise(sig_mv: np.ndarray, preset: tuple[dict, dict] | None = None,
                  seed: int = 0, group: int = GROUP):
    """Compatibility helper: add crate-wise noise to an **mV** trace.

    Returns ``(trace_mv, [per-group metadata, ...])`` — the shape the LUCiD
    notebooks used before this package existed. Prefer
    :func:`add_readout_noise` for new code.
    """
    trace, meta = add_readout_noise(sig_mv, preset=preset, seed=seed,
                                    in_units="mv", group_size=group)
    return trace, meta["covariance"]
