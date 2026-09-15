"""PMT_FRONTEND_V1 as implemented in notebooks/_build_nb1.py / _build_nb2.py (ORACLE, dev),
re-evaluated with the exact component shapes from src/noise_module/spectral_models.py
and the composite 'normalize' rule from NoiseGenerator.build_psd_density.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

FS, N = 1e9, 512
f = np.fft.rfftfreq(N, 1 / FS)
df = FS / N
fM = f / 1e6

# ---- exact component shapes (spectral_models.py) --------------------------
def white(f): return np.ones_like(f)
def rolloff(f, fc, order): return 1 / (1 + (f / fc) ** order)
def powerlaw(f, ref, ex):
    o = np.zeros_like(f); a = f > 0; o[a] = (f[a] / ref) ** ex; return o
def line(f, f0, w): return np.exp(-0.5 * ((f - f0) / w) ** 2)
def lor(f, c, hw): return 1 / (1 + ((f - c) / hw) ** 2)

NOISE_POWER = 0.8 ** 2          # mV², power_definition='variance' -> DC bin zeroed

def normalize(density):
    d = density.copy(); d[0] = 0.0
    return d * (NOISE_POWER / (d.sum() * df))

# ---- V1 as implemented: additive composite -------------------------------
v1 = {
    "amplifier_floor":    1.0  * white(f),
    "frontend_bandwidth": 1.0  * rolloff(f, 2.5e8, 2.0),
    "flicker":            0.05 * powerlaw(f, 1e7, -1.0),
    "clock_pickup":       6.0  * line(f, 6.25e7, 4e6),
    "cable_ringing":      0.5  * lor(f, 1.5e8, 2e7),
}
for v in v1.values(): v[0] = 0.0
tot1_raw = sum(v1.values())
g1 = NOISE_POWER / (tot1_raw.sum() * df)          # the single global factor 'normalize' applies
v1s = {k: v * g1 for k, v in v1.items()}
tot1 = tot1_raw * g1
frac1 = {k: v.sum() / tot1_raw.sum() for k, v in v1.items()}

# ---- V2 proposal: multiplicative front end, coherent lines ----------------
H_fe   = rolloff(f, 2.5e8, 2.0)                     # amplifier/cable low-pass  |H|^2
H_ring = 1.0 + 0.5 * lor(f, 1.5e8, 2e7)             # ringing as a transfer-function bump, not a source
priv_floor   = 1.0  * white(f)       * H_fe * H_ring
priv_flicker = 0.05 * powerlaw(f, 1e7, -1.0) * H_fe * H_ring
shared_clock = (6.0 * line(f, 6.25e7, 2e6)          # ADC clock and harmonics: enter after the front end,
                + 1.5 * line(f, 1.25e8, 2e6)        # common to a crate
                + 0.5 * line(f, 1.875e8, 2e6))
v2 = {"private floor × |H|²": priv_floor, "private flicker × |H|²": priv_flicker,
      "shared clock + harmonics": shared_clock}
for v in v2.values(): v[0] = 0.0
tot2_raw = sum(v2.values())
g2 = NOISE_POWER / (tot2_raw.sum() * df)
v2s = {k: v * g2 for k, v in v2.items()}
tot2 = tot2_raw * g2

# ---- crate coherence: rho_ij(f) = S_ij / sqrt(S_ii S_jj) -------------------
corr = 0.3
rho1 = np.full_like(f, corr)                        # shared_private: the same PSD shared and private -> flat
S_sh = v2s["shared clock + harmonics"]
S_pr = v2s["private floor × |H|²"] + v2s["private flicker × |H|²"]
rho2 = S_sh / (S_sh + S_pr)                          # unit coupling gains

# ---- palette (dataviz reference instance, light) ---------------------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})

fig = plt.figure(figsize=(12.5, 8.6))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.28)
axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1])

def style(ax, ylabel):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.9, 520); ax.grid(True, which="major", color=GRID, lw=0.6); ax.grid(False, which="minor")
    ax.set_xlabel("frequency [MHz]"); ax.set_ylabel(ylabel)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200", "500"])
    ax.axvline(250, color=INK2, lw=0.6, ls=":"); ax.axvline(62.5, color=INK2, lw=0.6, ls=":")

# (a) V1 components + total
sl = slice(1, None)
for (k, v), c in zip(v1s.items(), C):
    axA.plot(fM[sl], v[sl], color=c, lw=1.8)
axA.plot(fM[sl], tot1[sl], color=INK, lw=2.4)
style(axA, "one-sided PSD [mV²/Hz]")
axA.set_ylim(2e-13, 3e-8)
lab = {
    "amplifier_floor":    (2.3, v1s["amplifier_floor"][1] * 1.25, "amplifier floor (white) · 58 %"),
    "frontend_bandwidth": (2.3, v1s["frontend_bandwidth"][1] * 0.55, "'frontend_bandwidth' (additive low-pass) · 32 %"),
    "flicker":            (2.3, v1s["flicker"][1] * 0.42, "flicker 1/f · 0.4 %"),
    "clock_pickup":       (66, v1s["clock_pickup"].max() * 1.15, "clock pickup 62.5 MHz · 7 %"),
    "cable_ringing":      (95, v1s["cable_ringing"].max() * 0.42, "cable ringing 150 MHz · 3 %"),
}
for (k, (x, y, t)), c in zip(lab.items(), C):
    axA.text(x, y, t, color=c, fontsize=8.5, va="bottom" if k not in ("frontend_bandwidth", "flicker") else "top")
axA.text(2.3, tot1[1] * 1.45, "TOTAL = Σ components, normalised to 0.64 mV² (0.8 mV rms)", color=INK, fontsize=9, fontweight="bold")
axA.text(255, 4e-13, " 250 MHz 'corner'", color=INK2, fontsize=7.5)
axA.annotate(f"S(500 MHz) / S(2 MHz) = {tot1[256] / tot1[1]:.2f}\nthe floor is not band-limited",
             xy=(490, tot1[256]), xytext=(150, 8e-12), fontsize=8, color=INK2,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
axA.set_title("(a)  PMT_FRONTEND_V1 as implemented — five additive terms and their sum, on LUCiD's 512 ns / 1 GHz grid (df = 1.95 MHz)",
              loc="left", fontsize=10, color=INK)

# (b) V2 proposal
for (k, v), c in zip(v2s.items(), [C[0], C[2], C[3]]):
    axB.plot(fM[sl], v[sl], color=c, lw=1.8)
axB.plot(fM[sl], tot2[sl], color=INK, lw=2.4)
style(axB, "one-sided PSD [mV²/Hz]")
axB.set_ylim(2e-13, 3e-8)
axB.text(2.3, v2s["private floor × |H|²"][1] * 0.45, "private: (white) × |H_fe|² × |H_ring|²", color=C[0], fontsize=8, va="top")
axB.text(2.3, v2s["private flicker × |H|²"][1] * 0.5, "private: flicker × |H|²", color=C[2], fontsize=8, va="top")
axB.text(2.3, 1.6e-8, "shared per crate: clock + harmonics (62.5 / 125 / 187.5 MHz)", color=C[3], fontsize=8, va="top")
axB.text(2.3, tot2[1] * 1.5, "TOTAL (same 0.64 mV²)", color=INK, fontsize=8.5, fontweight="bold")
axB.annotate(f"S(500)/S(2) = {tot2[256] / tot2[1]:.2f}", xy=(490, tot2[256]), xytext=(120, 6e-12), fontsize=8, color=INK2,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
axB.set_title("(b)  proposed V2 — front end multiplies the floor, lines are the shared term", loc="left", fontsize=10, color=INK)

# (c) coherence
axC.plot(fM[sl], rho1[sl], color=C[1], lw=2.0)
axC.plot(fM[sl], rho2[sl], color=C[3], lw=2.0)
axC.set_xscale("log"); axC.set_xlim(1.9, 520); axC.set_ylim(-0.02, 1.05)
axC.set_xticks([2, 5, 10, 20, 50, 100, 200, 500]); axC.set_xticklabels(["2", "5", "10", "20", "50", "100", "200", "500"])
axC.grid(True, color=GRID, lw=0.6); axC.grid(False, which="minor")
axC.set_xlabel("frequency [MHz]"); axC.set_ylabel("ρ_ij(f) between two PMTs of one crate")
axC.text(2.3, 0.22, "V1 shared_private, corr_strength = 0.3: flat —\nthe white floor is 30 % coherent", color=C[1], fontsize=8, va="top")
axC.text(2.3, 0.97, "V2: coherent only where the shared term lives\n(clock line + harmonics)", color=C[3], fontsize=8, va="top")
axC.set_title("(c)  what the crate correlation looks like across frequency", loc="left", fontsize=10, color=INK)

fig.text(0.01, 0.005, "Evaluated from the preset dict in notebooks/_build_nb1.py with the component shapes of src/noise_module/spectral_models.py; "
         "percentages are each term's share of the 0.64 mV² total after the composite 'normalize' step. V2 numbers are placeholders like V1's.",
         fontsize=7, color=INK2)
out = "/mnt/user-data/outputs/pmt_frontend_v1_psd.png"
import os; os.makedirs("/mnt/user-data/outputs", exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
print({k: round(float(x), 3) for k, x in frac1.items()})
