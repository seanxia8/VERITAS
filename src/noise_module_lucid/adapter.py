# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Crate-wise noise injection: the adapter between LUCiD and ``noise_module``.

``crate_preset`` builds a ``(base_config, crate_config)`` pair for
``MultiChannelNoiseGenerator``; ``add_pmt_noise`` adds the front-end noise to a
mV waveform crate by crate and returns the per-crate metadata, which carries
the implied and realized covariance. ``kappa`` reads
``cond(Sigma_hat^-1 Sigma)`` from that metadata.

The covariance unit is the *crate* (or string), not the whole detector: PMTs on
one front-end board or digitiser share the clock pickup but not each other's
amplifier thermal noise. ``grouping.py`` chooses the groups.

Per-channel gain (24 Sep 2026 review, P1.3)
-------------------------------------------
The preset's ``channel_gain_jitter`` scales the **noise coupling** only; a real
channel gain scales signal and noise together. Pass ``channel_gains`` (from
LUCiD's ``PerPmtParams.gain``) to ``add_pmt_noise`` and the same gains are
applied to the signal and pinned into the noise structure, so the two are
consistent. When ``channel_gains`` is ``None`` the metadata records
``signal_gain_applied=False`` — the jitter is then declared amplifier-only.

Clock model (P1.2)
------------------
``clock="gaussian"`` (default) keeps the lines in the Gaussian shared process;
``clock="deterministic"`` moves them to a phase-locked tone (``clock.py``) with
the same power; ``clock="none"`` drops them.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .presets import FS_L, GROUP, PMT_CRATE_V2, PMT_FRONTEND_V2, PMT_PRIVATE, PMT_SHARED, RMS_MV
from .grouping import channel_groups

_CLOCK_MODES = ("gaussian", "deterministic", "none")
_SHARED_PLACEHOLDER = {"type": "white", "scale": 1e-9, "name": "shared_placeholder"}


def _split_lines(shared):
    lines = [c for c in shared if c.get("type") == "line"]
    rest = [c for c in shared if c.get("type") != "line"]
    return lines, rest


def crate_preset(shared=None, private=None, *, keep_private_power: bool = True, n_ref: int = 512,
                 clock: str = "gaussian", sampling_frequency: float = FS_L,
                 noise_power: float | None = None, **crate_overrides) -> tuple[dict, dict]:
    """Return ``(base_config, crate_config)`` for a variant of the V2 preset.

    ``shared`` / ``private`` replace the component lists. With
    ``keep_private_power`` the per-PMT *private* power is held at its V2 value
    and the total is re-derived, so "clock line x5" adds line power instead of
    silently re-partitioning a fixed 0.64 mV^2 (the ``normalize`` rule would
    otherwise do the latter, as V1 did). Integrals are taken on the ``n_ref``
    grid; the split is then fixed by the component scales alone.

    ``clock`` is ``"gaussian"`` (lines in the shared process), ``"deterministic"``
    (lines moved to a phase-locked tone, kept in the crate config for
    ``add_pmt_noise``) or ``"none"``.
    """
    from noise_module import NoiseGenerator

    if clock not in _CLOCK_MODES:
        raise ValueError(f"clock must be one of {_CLOCK_MODES}.")
    shared = PMT_SHARED if shared is None else shared
    private = PMT_PRIVATE if private is None else private
    lines, shared_gaussian = _split_lines(shared)
    if clock == "gaussian":
        shared_gaussian = shared

    def integral(components):
        cfg = {**PMT_FRONTEND_V2, "sampling_frequency": sampling_frequency,
               "components": components, "composite_psd_scaling": "absolute"}
        f, s = NoiseGenerator(cfg).build_psd_density(n_ref)
        return float(np.sum(s[1:]) * (sampling_frequency / n_ref))

    power = RMS_MV**2 if noise_power is None else float(noise_power)
    if keep_private_power:
        i_pr0, i_sh0 = integral(PMT_PRIVATE), integral(PMT_SHARED)
        p_private = power * i_pr0 / (i_pr0 + i_sh0)
        i_pr, i_sh = integral(private), integral(shared_gaussian or [_SHARED_PLACEHOLDER])
        power = p_private * (i_pr + i_sh) / i_pr
    base = {**PMT_FRONTEND_V2, "sampling_frequency": sampling_frequency, "noise_power": power,
            "components": [*private, *shared_gaussian]}
    crate = {**PMT_CRATE_V2, "shared_components": shared_gaussian or [_SHARED_PLACEHOLDER],
             "private_components": private, **crate_overrides}
    if clock == "deterministic":
        crate["_clock_lines"] = deepcopy(lines)
    return base, crate


def add_pmt_noise(sig_mv: np.ndarray, preset: tuple[dict, dict] | None = None, seed: int = 0,
                  group: int = GROUP, positions: np.ndarray | None = None,
                  group_method: str = "contiguous", board_ids: np.ndarray | None = None,
                  channel_gains: np.ndarray | None = None, clock: str | None = None):
    """Add V2 front-end noise crate by crate to a ``(C, N)`` mV waveform.

    Returns ``(trace, [metadata per crate])``; each metadata carries the implied
    and realized covariance and ``implied_spectra``. Groups come from
    :func:`grouping.channel_groups`. ``channel_gains`` (per channel) is applied
    to the signal and pinned into the noise; ``clock`` overrides the preset's
    clock mode.
    """
    from noise_module import MultiChannelNoiseGenerator
    from noise_module.core.utils import sample_range
    from . import clock as clock_mod

    base, crate = crate_preset() if preset is None else preset
    crate = deepcopy(crate)
    lines = crate.pop("_clock_lines", None)
    clock_mode = clock or ("deterministic" if lines else "gaussian")
    sig_mv = np.asarray(sig_mv, dtype=float)
    if sig_mv.ndim != 2:
        raise ValueError("sig_mv must have shape (n_channels, n_samples).")
    C, N = sig_mv.shape
    fs = float(base.get("sampling_frequency", FS_L))
    gains_full = None
    if channel_gains is not None:
        gains_full = np.asarray(channel_gains, dtype=float)
        if gains_full.shape != (C,):
            raise ValueError("channel_gains must have one entry per channel.")
    out = sig_mv * gains_full[:, None] if gains_full is not None else sig_mv.copy()
    groups_meta = []
    for gidx, idx in enumerate(channel_groups(C, group, positions=positions,
                                              method=group_method, board_ids=board_ids)):
        c = len(idx)
        gen = MultiChannelNoiseGenerator(base, {**crate, "n_channels": c}, seed=seed + gidx)
        if gains_full is not None:
            strengths = sample_range(gen.rng, crate.get("private_strength_range", [0.8, 1.2]), size=c)
            gen.set_channel_structure("spectral_shared_private", c,
                                      gains=gains_full[idx], private_strengths=strengths)
        noise, m = gen.generate(N, return_metadata=True)
        out[idx] += noise
        if clock_mode == "deterministic" and lines:
            tone, tmeta = clock_mod.add_deterministic_clock(
                np.zeros((c, N)), fs, components=lines,
                rng=np.random.default_rng(seed + 10_000 + gidx), gains=m["gains"],
                return_metadata=True)
            out[idx] += tone
            m = {**m, "deterministic_clock": tmeta}
        m = {**m, "channel_indices": np.asarray(idx),
             "signal_gain_applied": gains_full is not None}
        groups_meta.append(m)
    return out, groups_meta


#: Plan-facing alias (``adapter.add_readout_noise``).
add_readout_noise = add_pmt_noise


def apply_channel_gains(signal: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Multiply each channel by its gain — the per-PMT gain LUCiD also carries."""
    signal = np.asarray(signal, dtype=float)
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (signal.shape[0],):
        raise ValueError("gains must have one entry per channel.")
    return signal * gains[:, None]


def long_window_preset(window_ns: float, **overrides) -> tuple[dict, dict]:
    """Crate preset valid at ``df = 1/window_ns`` (adds DC-DC lines, extends 1/f)."""
    from .presets import long_window_components

    private, shared = long_window_components(window_ns)
    return crate_preset(shared=shared, private=private, **overrides)


def kappa(meta: dict) -> float:
    """cond(Sigma_hat^-1 Sigma) for one crate record (estimator floor on a matched cell)."""
    return float(np.linalg.cond(np.linalg.solve(meta["implied_covariance"], meta["realized_covariance"])))


def matched_cell_kappa_floor(N: int, C: int = GROUP, seed: int = 1, **crate_overrides) -> float:
    """The matched-cell kappa floor: every value above 1.0 is estimator noise."""
    from noise_module import MultiChannelNoiseGenerator

    base, crate = crate_preset(**crate_overrides)
    gen = MultiChannelNoiseGenerator(base, {**crate, "n_channels": C}, seed=seed)
    _, m = gen.generate(N, return_metadata=True)
    return kappa(m)
