# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Final PMT front-end noise spectrum (preset V2) — generated from the module.

Run from the repo root:  python docs/reviews/plot_pmt_frontend_v2_final.py
Writes docs/reviews/pmt_frontend_v2_final.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.fft import rfft, rfftfreq
from scipy.signal import csd, welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from noise_module import MultiChannelNoiseGenerator, NoiseGenerator  # noqa: E402
from noise_module_lucid import (FS_L, GROUP, PMT_CRATE_V2, PMT_FRONTEND_V2, PMT_PRIVATE,  # noqa: E402
                                PMT_SHARED, RMS_MV, kappa)

N = 512                                   # LUCiD default window: 512 ns at 1 ns
f = rfftfreq(N, 1 / FS_L)
df = FS_L / N
fM = f / 1e6

# ---- (a) design spectra: every component + total, in mV²/Hz ----------------
gen = NoiseGenerator(PMT_FRONTEND_V2, seed=0)
_, S_total, meta = gen.build_psd_density(N, return_metadata=True)
contrib = meta["component_contributions"]
factor = RMS_MV**2 / sum(c["integrated_power"] for c in contrib)     # the composite 'normalize' factor
comp_psd = {}
for c in PMT_FRONTEND_V2["components"]:
    _, s = NoiseGenerator({**PMT_FRONTEND_V2, "composite_psd_scaling": "absolute", "components": [c]}).build_psd_density(N)
    comp_psd[c["name"]] = s * factor
share = {c["name"]: c["integrated_power_after_global_scaling"] / RMS_MV**2 for c in contrib}
assert abs(sum(comp_psd.values())[1:].sum() * df - RMS_MV**2) < 1e-9

# ---- (b) crate coherence: implied rho_ij(f) vs realized (Welch on a long record)
crate = MultiChannelNoiseGenerator(PMT_FRONTEND_V2, {**PMT_CRATE_V2, "n_channels": 2,
                                                     "channel_gain_jitter": 0.0, "private_strength_range": [1.0, 1.0]}, seed=7)
X2, m2 = crate.generate(1 << 16, return_metadata=True)
f_rho, rho_implied = MultiChannelNoiseGenerator.implied_correlation_spectrum(m2, 0, 1)
f_w, s01 = csd(X2[0], X2[1], fs=FS_L, nperseg=N)
_, s00 = welch(X2[0], fs=FS_L, nperseg=N)
_, s11 = welch(X2[1], fs=FS_L, nperseg=N)
rho_realized = np.real(s01) / np.sqrt(s00 * s11)

# ---- (c) validation: ensemble PSD of one crate vs the design, plus kappa floors
crate64 = MultiChannelNoiseGenerator(PMT_FRONTEND_V2, PMT_CRATE_V2, seed=1)
X64, m64 = crate64.generate(N, return_metadata=True)
psd_real = np.mean(np.abs(rfft(X64, axis=-1)) ** 2, axis=0) * 2 / (FS_L * N)
k_short = kappa(m64)
X64L, m64L = crate64.generate(32768, return_metadata=True)
k_long = kappa(m64L)

# ---- figure ----------------------------------------------------------------
C = {"amplifier_floor": "#2a78d6", "flicker": "#1baf7a", "clock_62.5MHz": "#eda100",
     "clock_2nd_harmonic": "#eb6834", "clock_3rd_harmonic": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
fig = plt.figure(figsize=(12.5, 9.0))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1], hspace=0.4, wspace=0.26)
axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1])
ticks = [2, 5, 10, 20, 50, 100, 200, 500]

def style(ax, ylabel, log_y=True):
    ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlim(1.9, 520); ax.grid(True, which="major", color=GRID, lw=0.6); ax.grid(False, which="minor")
    ax.set_xlabel("frequency [MHz]"); ax.set_ylabel(ylabel)
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    for x in (62.5, 125, 187.5, 250): ax.axvline(x, color=INK2, lw=0.5, ls=":")

sl = slice(1, None)
labels = {"amplifier_floor": "private · amplifier floor × |H_fe|²·|H_ring|²", "flicker": "private · flicker × |H|²",
          "clock_62.5MHz": "shared · clock 62.5 MHz", "clock_2nd_harmonic": "shared · 125 MHz", "clock_3rd_harmonic": "shared · 187.5 MHz"}
for name, s in comp_psd.items():
    axA.plot(fM[sl], s[sl], color=C[name], lw=1.8)
axA.plot(fM[sl], S_total[sl], color=INK, lw=2.4)
style(axA, "one-sided PSD [mV²/Hz]"); axA.set_ylim(2e-13, 4e-8)
axA.text(2.3, S_total[1] * 1.5, f"TOTAL · {RMS_MV} mV rms on 512 ns (0.64 mV²)", color=INK, fontsize=9, fontweight="bold")
axA.text(2.3, comp_psd["amplifier_floor"][1] * 0.45, f"{labels['amplifier_floor']} · {share['amplifier_floor']:.0%}", color=C["amplifier_floor"], fontsize=8.5, va="top")
axA.text(2.3, comp_psd["flicker"][1] * 0.45, f"{labels['flicker']} · {share['flicker']:.1%}", color=C["flicker"], fontsize=8.5, va="top")
for name, x in (("clock_62.5MHz", 66), ("clock_2nd_harmonic", 131), ("clock_3rd_harmonic", 196)):
    axA.text(x, comp_psd[name].max() * 1.2, f"{labels[name]} · {share[name]:.1%}", color=C[name], fontsize=8)
axA.annotate(f"S(500)/S(2) = {S_total[256] / S_total[1]:.2f}: the floor now rolls off\nas 1/(1+(f/250 MHz)²); ringing is a bump on it, not a source",
             xy=(490, S_total[256]), xytext=(150, 1.5e-11), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
axA.set_title("(a)  PMT_FRONTEND_V2 — final single-PMT noise spectrum on LUCiD's 512 ns / 1 GHz grid: components and total",
              loc="left", fontsize=10, color=INK)

axB.plot(fM[sl], np.interp(f[sl], f_rho, rho_implied), color=INK, lw=2.0)
axB.plot(f_w[sl] / 1e6, rho_realized[sl], color="#eda100", lw=1.2, alpha=0.9)
style(axB, "ρ_ij(f), two PMTs of one crate", log_y=False); axB.set_ylim(-0.15, 1.05)
axB.text(2.3, 0.98, "black: implied ρ_ij(f) = S_sh /(S_sh + S_pr)\nyellow: realized, Welch on a 65 µs record", color=INK, fontsize=8, va="top")
axB.text(2.3, 0.30, "≈ 0 on the private floor\n≈ 0.9 at the clock line and harmonics", color=INK2, fontsize=8)
axB.set_title("(b)  crate coherence is now a function of frequency", loc="left", fontsize=10, color=INK)

axC.plot(fM[sl], psd_real[sl], color="#2a78d6", lw=1.0, alpha=0.9)
axC.plot(fM[sl], S_total[sl], color=INK, lw=2.0)
style(axC, "one-sided PSD [mV²/Hz]"); axC.set_ylim(2e-13, 4e-8)
axC.text(2.3, 2.5e-8, f"black: design total; blue: mean |FFT|² of one generated crate ({GROUP} ch × 512 ns)\n"
         f"matched-cell κ floor: {k_short:.1f} at N/C = 8 (512 ns)  →  {k_long:.2f} at N/C = 512 (32.8 µs)",
         color=INK, fontsize=8, va="top")
axC.set_title("(c)  generated ensemble against the design, and the κ floor rule", loc="left", fontsize=10, color=INK)

fig.text(0.01, 0.005, "Generated by docs/reviews/plot_pmt_frontend_v2_final.py from notebooks/pmt_frontend_v2.py with the patched noise_module "
         "(Filtered component; spectral_shared_private mode). All levels are placeholders; total rms and SPE amplitude set the SNR.",
         fontsize=7, color=INK2)
out = Path(__file__).with_name("pmt_frontend_v2_final.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
print("shares:", {k: round(v, 4) for k, v in share.items()})
print("S(500)/S(2) =", round(S_total[256] / S_total[1], 3), "| rho at 62.5 MHz implied", round(float(np.interp(6.25e7, f_rho, rho_implied)), 3),
      "realized", round(float(rho_realized[np.argmin(abs(f_w - 6.25e7))]), 3), "| kappa floors", round(k_short, 2), round(k_long, 3))
