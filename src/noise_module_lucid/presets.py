# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""LUCiD PMT front-end presets (V2, short and long window).

Two grids, because the record length decides which noise exists at all
(``docs/noise_module_lucid.md`` §3):

* ``PMT_FRONTEND_V2`` — 1 GHz, 512 ns (df = 1.95 MHz). The LUCiD convention used
  by the notebooks; 50 Hz mains is 2.6e-5 of one bin and is not representable.
* ``PMT_FRONTEND_LONG`` — 1 GHz, 16–32 µs (df ≈ 30–60 kHz). The grid the
  covariance (κ) cells need: DC-DC switching lines at 100 kHz–2 MHz and more of
  the 1/f decade come into band. 50 Hz mains is *still* not representable; that
  needs decimation or a ≥ 1 s record and belongs to ``psd_resampling`` /
  ``temporal_noise``.

Every number is a placeholder (provenance ``placeholder``); the 0.8 mV rms total
and the 4 mV/pe SPE amplitude are the two that set the SNR.
"""
from __future__ import annotations

import numpy as np

FS_L = 1.0e9                # LUCiD "1 GHz FADC convention": 1 ns bins
RMS_MV = 0.8                # baseline noise, placeholder
GROUP = 64                  # PMTs per crate = the covariance unit

# --- transfer functions of the private path (amplifier + base/connector) -----
FRONT_END = [
    {"type": "rolloff", "corner_hz": 2.5e8, "order": 2.0, "kind": "lowpass",
     "name": "frontend_bandwidth"},
    {"type": "peaking", "center_hz": 1.5e8, "half_width_hz": 2.0e7, "gain": 0.5,
     "name": "base_ringing"},
]

# --- private: one process per PMT, shaped by the front end -------------------
PMT_PRIVATE = [
    {"type": "filtered", "name": "amplifier_floor",
     "source": {"type": "white", "scale": 1.0}, "filters": FRONT_END},
    {"type": "filtered", "name": "flicker",
     "source": {"type": "powerlaw", "scale": 0.05, "exponent": -1.0,
                "reference_hz": 1.0e7}, "filters": FRONT_END},
]

# Long-window private path: the 1/f reference moves down so the extra decades
# are populated, and the same front end shapes the floor.
PMT_PRIVATE_LONG = [
    {"type": "filtered", "name": "amplifier_floor",
     "source": {"type": "white", "scale": 1.0}, "filters": FRONT_END},
    {"type": "filtered", "name": "flicker",
     "source": {"type": "powerlaw", "scale": 0.08, "exponent": -1.0,
                "reference_hz": 1.0e5}, "filters": FRONT_END},
]

# --- shared: one process per crate, entering at the digitiser (unfiltered) ---
# A real clock is a deterministic, phase-locked sinusoid; a Gaussian line of
# ~1 bin width is the stationary-Gaussian approximation of it, declared here.
PMT_SHARED = [
    {"type": "line", "scale": 6.0, "frequency_hz": 6.25e7, "width_hz": 2.0e6,
     "name": "clock_62.5MHz"},
    {"type": "line", "scale": 1.5, "frequency_hz": 1.25e8, "width_hz": 2.0e6,
     "name": "clock_2nd_harmonic"},
    {"type": "line", "scale": 0.5, "frequency_hz": 1.875e8, "width_hz": 2.0e6,
     "name": "clock_3rd_harmonic"},
]

# Long-window shared path: DC-DC / switching pickup, representable at df ~ 30 kHz.
PMT_SHARED_LONG = [
    *PMT_SHARED,
    {"type": "line", "scale": 2.0, "frequency_hz": 1.0e5, "width_hz": 3.0e4,
     "name": "switching_100kHz"},
    {"type": "line", "scale": 1.2, "frequency_hz": 2.5e5, "width_hz": 3.0e4,
     "name": "switching_250kHz"},
    {"type": "line", "scale": 0.8, "frequency_hz": 5.0e5, "width_hz": 3.0e4,
     "name": "switching_500kHz"},
    {"type": "line", "scale": 0.5, "frequency_hz": 1.0e6, "width_hz": 3.0e4,
     "name": "switching_1MHz"},
    {"type": "line", "scale": 0.3, "frequency_hz": 2.0e6, "width_hz": 3.0e4,
     "name": "switching_2MHz"},
]

# Single-channel view (shared + private summed): one PMT's random trigger.
PMT_FRONTEND_V2 = dict(
    noise_type="composite", sampling_frequency=FS_L, noise_power=RMS_MV**2,
    power_definition="variance", composite_psd_scaling="normalize",
    components=[*PMT_PRIVATE, *PMT_SHARED],
)
PMT_FRONTEND_LONG = dict(
    noise_type="composite", sampling_frequency=FS_L, noise_power=RMS_MV**2,
    power_definition="variance", composite_psd_scaling="normalize",
    components=[*PMT_PRIVATE_LONG, *PMT_SHARED_LONG],
)

# Crate view: the same components split into shared / private processes.
PMT_CRATE_V2 = dict(
    mode="spectral_shared_private", n_channels=GROUP,
    shared_components=PMT_SHARED, private_components=PMT_PRIVATE,
    channel_gain_jitter=0.05, private_strength_range=[0.8, 1.2],
    freeze_channel_structure=True, normalize_channel_variance=False,
)
PMT_CRATE_LONG = dict(
    mode="spectral_shared_private", n_channels=GROUP,
    shared_components=PMT_SHARED_LONG, private_components=PMT_PRIVATE_LONG,
    channel_gain_jitter=0.05, private_strength_range=[0.8, 1.2],
    freeze_channel_structure=True, normalize_channel_variance=False,
)

PROVENANCE = {
    "preset": "PMT_FRONTEND_V2", "date": "2026-09-11",
    "rms_mv": ("placeholder", RMS_MV), "spe_mv_per_pe": ("placeholder", 4.0),
    "frontend_corner_hz": ("placeholder", 2.5e8), "ringing_hz": ("placeholder", 1.5e8),
    "clock_hz": ("placeholder", 6.25e7), "group": ("design", GROUP),
    "note": "shared_private (V1) replaced by spectral_shared_private; additive "
            "rolloff/resonance replaced by Filtered",
}

#: ``window_ns`` thresholds that select the long preset.
LONG_WINDOW_NS = 16384.0


def preset_for(window_ns: float) -> tuple[dict, dict]:
    """Return ``(base_config, crate_config)`` for a LUCiD window length.

    Short (≤ 512 ns) uses V2; the 16–32 µs covariance cells use the long preset
    (switching lines and more 1/f in band). The choice is explicit so provenance
    can record it.
    """
    if float(window_ns) >= LONG_WINDOW_NS:
        return PMT_FRONTEND_LONG, PMT_CRATE_LONG
    return PMT_FRONTEND_V2, PMT_CRATE_V2


def crate_preset(base: dict | None = None, crate: dict | None = None,
                 shared: list | None = None, private: list | None = None, *,
                 keep_private_power: bool = True, n_ref: int = 512,
                 **crate_overrides) -> tuple[dict, dict]:
    """Return ``(base_config, crate_config)`` for a variant of a preset.

    ``shared`` / ``private`` replace the component lists. With
    ``keep_private_power`` the per-PMT *private* power is held at its reference
    value and the total is re-derived, so "clock line ×5" adds line power
    instead of silently re-partitioning a fixed 0.64 mV² (the ``normalize`` rule
    would otherwise do the latter, as V1 did). Integrals are taken on the
    ``n_ref`` grid; the split is then fixed by the component scales alone.
    """
    from noise_module import NoiseGenerator

    base = PMT_FRONTEND_V2 if base is None else base
    crate = PMT_CRATE_V2 if crate is None else crate
    # The reference share always comes from the *unmodified* preset, so that
    # overriding the private/shared lists changes the split, not the reference.
    ref_private = crate["private_components"]
    ref_shared = crate["shared_components"]
    shared = ref_shared if shared is None else shared
    private = ref_private if private is None else private

    def integral(components):
        cfg = {**base, "components": components, "composite_psd_scaling": "absolute"}
        f, s = NoiseGenerator(cfg).build_psd_density(n_ref)
        return float(np.sum(s[1:]) * (FS_L / n_ref))

    power = RMS_MV**2
    if keep_private_power:
        i_pr0, i_sh0 = integral(ref_private), integral(ref_shared)
        p_private = power * i_pr0 / (i_pr0 + i_sh0)   # the preset's private share
        i_pr, i_sh = integral(private), integral(shared)
        power = p_private * (i_pr + i_sh) / i_pr
    out_base = {**base, "noise_power": power, "components": [*private, *shared]}
    out_crate = {**crate, "shared_components": shared, "private_components": private,
                 **crate_overrides}
    return out_base, out_crate
