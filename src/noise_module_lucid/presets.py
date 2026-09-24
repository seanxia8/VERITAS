# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""PMT front-end noise presets for the LUCiD arm (preset V2, 11 Sep 2026).

This module is the *spectral* half of ``noise_module_lucid``: the component
lists that describe the stationary front-end covariance of a water-Cherenkov
PMT readout, split into a **private** process (one per PMT) and a **shared**
process (one per crate). It was moved here from
``notebooks/pmt_frontend_v2.py`` so the LUCiD arm has a package, not a
notebook helper (the arm was gated on LUCiD's licence, gate A0; the package
itself is ours, MIT).

What the preset is, and what it is not
--------------------------------------
LUCiD itself models the *event* stochastic layer — quantum efficiency,
single-photoelectron charge smearing, transit-time spread, dark counts and TDC
jitter — all independent per photon and per PMT, with **no electronic noise and
no inter-channel covariance anywhere** (``lucid/simulation/digitizer.py``).
The front end between the anode and the digitiser is what this package adds.

Two rules, learned the hard way (V1 of this preset broke both; see
``docs/LUCID_NOISE_REVIEW_2026-09-11.md``):

1. **A bandwidth is a filter, not a source.** The amplifier/cable roll-off and
   the base/connector ringing *multiply* the white floor (``Filtered``), they
   are not summed with it. V1's additive low-pass left 58 % of the power flat
   to Nyquist.
2. **Coherence belongs to a term, not to the crate.** The ADC clock and its
   harmonics are the *shared* process; the amplifier floor and flicker are
   *private* (``spectral_shared_private``). V1 mixed one PSD at a flat 0.3,
   which made the thermal floor 30 % coherent and the clock only 30 %.

Co-calibration (24 Sep 2026 review, P0.3)
-----------------------------------------
V1/V2 left a factor-~13 gap between the SPE pulse bandwidth and the front-end
corner: the placeholder SPE (2 ns / 8 ns) is -3 dB at ~20 MHz while the corner
was 250 MHz, so ~62 % of the noise power sat above the signal band. The corner
is now **derived** from the SPE template's own power -3 dB bandwidth via
:data:`FRONT_END_BANDWIDTH_RATIO`, and the SPE template is a faster, WCTE-like
3-inch PMT pulse (1 ns rise / 3 ns fall, -3 dB ~51 MHz). The two are consistent
by construction; ``validation.bandwidth_report`` reports the residual overlap.
Every number is still a placeholder (provenance ``placeholder``).
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

# --- LUCiD's readout convention: 1 ns bins ("1 GHz FADC convention") ---------
FS_L = 1.0e9                # sampling frequency [Hz]
RMS_MV = 0.8                # baseline noise, placeholder
GROUP = 64                  # PMTs per crate = the covariance unit

#: LUCiD's own dark-noise rate (``digitizer.generate_dark_noise``, SK/HK 4.2 kHz).
#: Used by ``pulses.add_dark_pulses`` to distinguish a *delta* rate from the
#: full rate (see the dark-noise contract there).
LUCID_DARK_RATE_HZ = 4.2e3

# --- SPE voltage template: WCTE-like 3-inch PMT, placeholders ----------------
SPE_TAU_RISE_NS = 1.0
SPE_TAU_FALL_NS = 3.0
SPE_MV_PER_PE = 4.0
SPE_LENGTH_NS = 60.0


def spe_shape(t_ns: np.ndarray, tau_rise_ns: float = SPE_TAU_RISE_NS,
              tau_fall_ns: float = SPE_TAU_FALL_NS) -> np.ndarray:
    """Unit-peak bi-exponential SPE pulse shape on a time grid in ns."""
    p = (1.0 - np.exp(-t_ns / tau_rise_ns)) * np.exp(-t_ns / tau_fall_ns)
    peak = p.max()
    return p / peak if peak > 0.0 else p


def spe_bandwidth_hz(fs: float = FS_L, tau_rise_ns: float = SPE_TAU_RISE_NS,
                     tau_fall_ns: float = SPE_TAU_FALL_NS,
                     length_ns: float = SPE_LENGTH_NS, nfft: int = 1 << 20) -> float:
    """Power -3 dB frequency of the SPE template (Hz).

    The template is short (``length_ns``), so the FFT is zero-padded to
    ``nfft`` for a frequency resolution fine enough that the -3 dB point is not
    bin-limited.
    """
    t_ns = np.arange(int(length_ns * fs / 1e9)) / fs * 1e9
    p = spe_shape(t_ns, tau_rise_ns, tau_fall_ns)
    power = np.abs(np.fft.rfft(p, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    power = power / power[0]
    below = np.flatnonzero(power < 0.5)
    return float(freqs[below[0]]) if below.size else float(freqs[-1])


#: The front end is this many times the SPE power bandwidth. A matched front
#: end is ~1-3x; 2.5x is a placeholder that preserves the pulse edge without a
#: large wideband noise penalty (noise above 3x the signal band ~14%).
FRONT_END_BANDWIDTH_RATIO = 2.5
SPE_BANDWIDTH_HZ = spe_bandwidth_hz()
FRONT_END_CORNER_HZ = FRONT_END_BANDWIDTH_RATIO * SPE_BANDWIDTH_HZ

# --- transfer functions of the private path (amplifier + base/connector) -----
# Read as |H_k(f)|^2 by noise_module.spectral.models.Filtered: they shape the
# floor, they do not add power of their own.
FRONT_END = [
    {"type": "rolloff", "corner_hz": FRONT_END_CORNER_HZ, "order": 4.0, "kind": "lowpass",
     "name": "frontend_bandwidth"},
    {"type": "peaking", "center_hz": 1.5e8, "half_width_hz": 2.0e7, "gain": 0.5, "name": "base_ringing"},
]

# --- private: one process per PMT, shaped by the front end -------------------
PMT_PRIVATE = [
    {"type": "filtered", "name": "amplifier_floor", "source": {"type": "white", "scale": 1.0}, "filters": FRONT_END},
    {"type": "filtered", "name": "flicker",
     "source": {"type": "powerlaw", "scale": 0.05, "exponent": -1.0, "reference_hz": 1.0e7},
     "filters": FRONT_END},
]

# --- shared: one process per crate, entering at the digitiser (unfiltered) ---
# A real clock is a deterministic, phase-locked sinusoid; a Gaussian line of
# ~1 bin width is the stationary-Gaussian approximation of it. The deterministic
# option is in ``clock.py``; here the default is the Gaussian approximation.
PMT_SHARED = [
    {"type": "line", "scale": 3.7, "frequency_hz": 6.25e7, "width_hz": 2.0e6, "name": "clock_62.5MHz"},
    {"type": "line", "scale": 0.95, "frequency_hz": 1.25e8, "width_hz": 2.0e6, "name": "clock_2nd_harmonic"},
    {"type": "line", "scale": 0.35, "frequency_hz": 1.875e8, "width_hz": 2.0e6, "name": "clock_3rd_harmonic"},
]

#: The clock lines, by name, so ``clock.py`` and ``adapter`` can move them
#: between the Gaussian shared process and a deterministic tone.
CLOCK_LINE_NAMES = tuple(c["name"] for c in PMT_SHARED if c["type"] == "line")

# Single-channel view (shared + private summed): what one PMT's random trigger
# looks like. Used for PSD plots and for the NoiseGenerator alias-fold demo.
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


def long_window_components(window_ns: float, fs: float = FS_L):
    """(private, shared) component lists valid at ``df = 1/window_ns``.

    At the default 512 ns record df = 1.95 MHz, so DC-DC switching pickup
    (100 kHz - 2 MHz) and most of the 1/f decade are unrepresentable. At the
    16-32 us windows the covariance cells require, df is 30-60 kHz and those
    terms become representable. This builder adds them and lets 1/f develop
    down to df. 50 Hz mains still needs a ~ms record and is not added.

    Returns ``(private, shared)``; pass to ``adapter.crate_preset``.
    """
    if window_ns <= 0.0:
        raise ValueError("window_ns must be positive.")
    df = fs / (window_ns * 1e-9)
    width_hz = 3.0 * df
    low_cutoff = df
    private = deepcopy(PMT_PRIVATE)
    for c in private:
        src = c.get("source")
        if src and src.get("type") == "powerlaw":
            src["low_cutoff_hz"] = low_cutoff
    shared = deepcopy(PMT_SHARED) + [
        {"type": "line", "scale": 2.0, "frequency_hz": 1.0e5, "width_hz": width_hz, "name": "dc_dc_100kHz"},
        {"type": "line", "scale": 1.0, "frequency_hz": 5.0e5, "width_hz": width_hz, "name": "dc_dc_500kHz"},
        {"type": "line", "scale": 0.5, "frequency_hz": 1.0e6, "width_hz": width_hz, "name": "dc_dc_1MHz"},
    ]
    return private, shared


#: Contract tags for the instruction list of 24 Sep 2026 (see
#: ``docs/INSTRUCTION_SORTING_2026-09-24.md``): which layer of the stack each
#: item belongs to, and whether it enters the front-end covariance Sigma.
CONTRACTS = {
    "amplifier_floor": {"layer": "noise_module", "contract": "N", "enters_sigma": True},
    "flicker": {"layer": "noise_module", "contract": "N", "enters_sigma": True},
    "clock_lines": {"layer": "noise_module", "contract": "N", "enters_sigma": True,
                    "note": "Gaussian approximation by default; clock.py gives the deterministic tone"},
    "cable_delay": {"layer": "noise_module", "contract": "N", "enters_sigma": False,
                    "note": "phase on the signal path; not expressible as a PSD magnitude"},
    "adc_tdc_sampling": {"layer": "noise_module", "contract": "N", "enters_sigma": False,
                         "note": "quantisation + aperture jitter; digitiser.py"},
    "dark_rate": {"layer": "lucid", "contract": "N", "enters_sigma": False,
                  "note": "Poisson event process inside LUCiD; a rate shift is the N family"},
    "prepulse_afterpulse": {"layer": "lucid", "contract": "N", "enters_sigma": False},
    "pileup_in_gate": {"layer": "lucid", "contract": "S/U", "enters_sigma": False},
    "dsnb_signal": {"layer": "signal", "contract": "signal", "enters_sigma": False},
    "atmospheric_tail": {"layer": "signal", "contract": "S", "enters_sigma": False},
    "qe_vs_dis": {"layer": "signal", "contract": "S", "enters_sigma": False},
    "radon_in_water": {"layer": "signal", "contract": "S/U", "enters_sigma": False},
}

PROVENANCE = {
    "preset": "PMT_FRONTEND_V2", "date": "2026-09-24 (P0.3 co-calibration)",
    "rms_mv": ("placeholder", RMS_MV), "spe_mv_per_pe": ("placeholder", SPE_MV_PER_PE),
    "spe_tau_rise_ns": ("placeholder", SPE_TAU_RISE_NS), "spe_tau_fall_ns": ("placeholder", SPE_TAU_FALL_NS),
    "spe_bandwidth_hz": ("derived", SPE_BANDWIDTH_HZ),
    "frontend_corner_hz": ("derived", FRONT_END_CORNER_HZ),
    "frontend_bandwidth_ratio": ("design", FRONT_END_BANDWIDTH_RATIO),
    "ringing_hz": ("placeholder", 1.5e8), "clock_hz": ("placeholder", 6.25e7),
    "group": ("design", GROUP),
    "note": "corner derived from SPE -3 dB bandwidth (24 Sep 2026 P0.3); moved from notebooks/pmt_frontend_v2.py; "
            "shared_private (V1) replaced by spectral_shared_private; additive rolloff/resonance replaced by Filtered",
}
