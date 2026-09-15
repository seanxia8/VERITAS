# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""PMT front-end noise for the LUCiD arm — preset V2 (11 Sep 2026).

Shared by `one_event_herald_lucid.ipynb`, `noise_models_herald_lucid.ipynb`
and the review figure; deliberately a helper next to the notebooks, not a
package (arm A is gated on LUCiD's licence, gate A0 in docs/TESTBEDS.md).

What changed from V1 (docs/reviews/LUCID_NOISE_REVIEW_2026-09-11.md):

1. The front-end bandwidth and the ringing are *transfer functions*, so they
   multiply the amplifier floor (`Filtered` component) instead of being summed
   with it. V1's additive low-pass left 58 % of the power flat to Nyquist; the
   V2 floor falls as 1/(1 + (f/250 MHz)^2), i.e. to 0.2 at 500 MHz.
2. The crate coherence is a function of frequency
   (`MultiChannelNoiseGenerator` mode `spectral_shared_private`): the ADC clock
   and its harmonics are the *shared* process, the amplifier floor and flicker
   are *private*. V1's `shared_private` mixed one PSD at a flat 0.3, which made
   the thermal floor 30 % coherent and the clock only 30 % coherent.

Every number below is a placeholder like V1's (provenance `placeholder`); the
0.8 mV rms total and the 4 mV/pe SPE amplitude are the two that set the SNR.
"""
from __future__ import annotations

import numpy as np

FS_L = 1.0e9                # LUCiD "1 GHz FADC convention": 1 ns bins
RMS_MV = 0.8                # baseline noise, placeholder
GROUP = 64                  # PMTs per crate = the covariance unit (TESTBEDS.md §1.1)

# --- transfer functions of the private path (amplifier + base/connector) ----
FRONT_END = [
    {"type": "rolloff", "corner_hz": 2.5e8, "order": 2.0, "kind": "lowpass", "name": "frontend_bandwidth"},
    {"type": "peaking", "center_hz": 1.5e8, "half_width_hz": 2.0e7, "gain": 0.5, "name": "base_ringing"},
]

# --- private: one process per PMT, shaped by the front end -------------------
PMT_PRIVATE = [
    {"type": "filtered", "name": "amplifier_floor", "source": {"type": "white", "scale": 1.0}, "filters": FRONT_END},
    {"type": "filtered", "name": "flicker", "source": {"type": "powerlaw", "scale": 0.05, "exponent": -1.0, "reference_hz": 1.0e7},
     "filters": FRONT_END},
]

# --- shared: one process per crate, entering at the digitiser (unfiltered) ---
# A real clock is a deterministic, phase-locked sinusoid; a Gaussian line of
# ~1 bin width is the stationary-Gaussian approximation of it, declared here.
PMT_SHARED = [
    {"type": "line", "scale": 6.0, "frequency_hz": 6.25e7, "width_hz": 2.0e6, "name": "clock_62.5MHz"},
    {"type": "line", "scale": 1.5, "frequency_hz": 1.25e8, "width_hz": 2.0e6, "name": "clock_2nd_harmonic"},
    {"type": "line", "scale": 0.5, "frequency_hz": 1.875e8, "width_hz": 2.0e6, "name": "clock_3rd_harmonic"},
]

# Single-channel view (shared + private summed): what one PMT's random trigger
# looks like. Used for PSD plots and for the `NoiseGenerator` alias-fold demo.
PMT_FRONTEND_V2 = dict(
    noise_type="composite", sampling_frequency=FS_L, noise_power=RMS_MV**2,
    power_definition="variance", composite_psd_scaling="normalize",
    components=[*PMT_PRIVATE, *PMT_SHARED],
)

# Crate view: the same components split into shared / private processes.
PMT_CRATE_V2 = dict(
    mode="spectral_shared_private", n_channels=GROUP,
    shared_components=PMT_SHARED, private_components=PMT_PRIVATE,
    channel_gain_jitter=0.05, private_strength_range=[0.8, 1.2],
    freeze_channel_structure=True, normalize_channel_variance=False,
)

PROVENANCE = {
    "preset": "PMT_FRONTEND_V2", "date": "2026-09-11",
    "rms_mv": ("placeholder", RMS_MV), "spe_mv_per_pe": ("placeholder", 4.0),
    "frontend_corner_hz": ("placeholder", 2.5e8), "ringing_hz": ("placeholder", 1.5e8),
    "clock_hz": ("placeholder", 6.25e7), "group": ("design", GROUP),
    "note": "shared_private (V1) replaced by spectral_shared_private; additive rolloff/resonance replaced by Filtered",
}


# --- units bridge: LUCiD photoelectron histogram -> mV ------------------------
def spe_template(fs: float = FS_L, tau_rise_ns: float = 2.0, tau_fall_ns: float = 8.0,
                 mv_per_pe: float = 4.0, length_ns: float = 60.0) -> np.ndarray:
    t = np.arange(int(length_ns * fs / 1e9)) / fs * 1e9
    p = (1 - np.exp(-t / tau_rise_ns)) * np.exp(-t / tau_fall_ns)
    return mv_per_pe * p / p.max()


def to_mv(wf_pe: np.ndarray, spe: np.ndarray | None = None) -> np.ndarray:
    spe = spe_template() if spe is None else spe
    return np.stack([np.convolve(row, spe)[: wf_pe.shape[1]] for row in wf_pe])


# --- interventions on the crate preset ---------------------------------------
def crate_preset(shared=None, private=None, *, keep_private_power: bool = True, n_ref: int = 512,
                 **crate_overrides) -> tuple[dict, dict]:
    """Return ``(base_config, crate_config)`` for a variant of the V2 preset.

    ``shared`` / ``private`` replace the component lists. With
    ``keep_private_power`` the per-PMT *private* power is held at its V2 value
    and the total is re-derived, so "clock line x5" adds line power instead of
    silently re-partitioning a fixed 0.64 mV^2 (the ``normalize`` rule would
    otherwise do the latter, as V1 did). Integrals are taken on the ``n_ref``
    grid; the split is then fixed by the component scales alone.
    """
    from noise_module import NoiseGenerator

    shared = PMT_SHARED if shared is None else shared
    private = PMT_PRIVATE if private is None else private

    def integral(components):
        cfg = {**PMT_FRONTEND_V2, "components": components, "composite_psd_scaling": "absolute"}
        f, s = NoiseGenerator(cfg).build_psd_density(n_ref)
        return float(np.sum(s[1:]) * (FS_L / n_ref))

    power = RMS_MV**2
    if keep_private_power:
        i_pr0, i_sh0 = integral(PMT_PRIVATE), integral(PMT_SHARED)
        p_private = power * i_pr0 / (i_pr0 + i_sh0)          # V2's private share of 0.64 mV^2
        i_pr, i_sh = integral(private), integral(shared)
        power = p_private * (i_pr + i_sh) / i_pr
    base = {**PMT_FRONTEND_V2, "noise_power": power, "components": [*private, *shared]}
    crate = {**PMT_CRATE_V2, "shared_components": shared, "private_components": private, **crate_overrides}
    return base, crate


# --- crate-wise noise --------------------------------------------------------
def add_pmt_noise(sig_mv: np.ndarray, preset: tuple[dict, dict] | None = None, seed: int = 0,
                  group: int = GROUP):
    """Add V2 front-end noise crate by crate. ``preset = (base_config, crate_config)``
    as returned by :func:`crate_preset` (default: the V2 reference). Returns
    ``(trace, [metadata per crate])``; each metadata carries the implied and
    realized covariance and ``implied_spectra``."""
    from noise_module import MultiChannelNoiseGenerator

    base, crate = crate_preset() if preset is None else preset
    C, N = sig_mv.shape
    out = sig_mv.copy()
    groups = []
    for g0 in range(0, C, group):
        c = min(group, C - g0)
        gen = MultiChannelNoiseGenerator(base, {**crate, "n_channels": c}, seed=seed + g0)
        noise, m = gen.generate(N, return_metadata=True)
        out[g0:g0 + c] += noise
        groups.append(m)
    return out, groups


def kappa(meta: dict) -> float:
    """cond(Σ̂⁻¹ Σ) for one crate record (estimator floor on a matched cell)."""
    return float(np.linalg.cond(np.linalg.solve(meta["implied_covariance"], meta["realized_covariance"])))
