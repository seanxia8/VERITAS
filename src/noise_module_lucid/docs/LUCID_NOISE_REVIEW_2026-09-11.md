# LUCiD noise review (PMT_FRONTEND_V1) and the Prometheus question — 11 Sep 2026

_Read on `~/Documents/ORACLE`, branch `dev`: `notebooks/_build_nb1.py` / `_build_nb2.py`
(the source of `one_event_herald_lucid.ipynb` and `noise_models_herald_lucid.ipynb`),
`src/noise_module/spectral_models.py`, `NoiseGenerator.build_psd_density`,
`multichannel_noise.generate_shared_private`, and the vendored Prometheus at
`8c19938` (28 Aug 2026). Figure: `pmt_frontend_v1_psd.png` (script alongside)._

## 1. What the notebook does

LUCiD emits `(n_sensors, 512)` photoelectron counts at 1 ns. The notebook convolves
each row with a two-exponential SPE voltage template (2 ns rise, 8 ns fall, 4 mV peak
per pe), then adds `PMT_FRONTEND_V1` noise per crate of 64 PMTs with
`MultiChannelNoiseGenerator(mode='shared_private', corr_strength=0.3)`, total 0.8 mV rms.
The preset is five **additive** composite terms (all `normalization='density'`),
globally rescaled to 0.64 mV²:

| term | share of the 0.64 mV² |
|---|---|
| amplifier floor (white) | 58 % |
| `frontend_bandwidth` (lowpass, 250 MHz, order 2) — *additive* | 32 % |
| flicker, 0.05·(f/10 MHz)⁻¹ | 0.4 % |
| clock pickup, Gaussian line 62.5 MHz, σ 4 MHz | 7 % |
| cable ringing, Lorentzian 150 MHz, HW 20 MHz | 3 % |

The component *choice* is right for a 2–500 MHz band (the notebook's own table is
correct: 50 Hz mains is 2.6·10⁻⁵ of a bin and cannot exist on this grid; the in-band
physics is amplifier floor, front-end bandwidth, clock/switching lines, ringing, weak
flicker). Levels are placeholders, declared as such. Dark counts, TTS, SPE gain spread
and QE are inside LUCiD and are events, not covariance — correctly kept out of Σ.

## 2. Two things that do not make sense as implemented

**(i) The roll-off is a source, not a filter.** `rolloff` is summed with the white
term instead of multiplying it, so the "250 MHz bandwidth" is a second, weaker white-ish
term. The total PSD falls only to 0.53 of its 2 MHz value at 500 MHz; 58 % of the
power is a floor that is flat to Nyquist. Consequence: the noise carries structure
faster than the SPE pulse (2 ns rise ≈ 80 MHz signal band), so a whitened filter sees
a Σ with energy where no signal ever is — harmless to the lemma, wrong as a detector.
Same issue for `cable_ringing`: physically it is part of the transfer function
(|H_ring|²·floor), not an independent source. The composite grammar has no product
operator; the cheap fix is a `custom` PSD (or a `transfer` wrapper component) so that
S_private(f) = [white + flicker]·|H_fe(f)|²·|H_ring(f)|².

**(ii) The crate coherence is frequency-flat.** `generate_shared_private` draws the
shared and the private process from the *same* composite PSD, so ρ_ij(f) = 0.3 at
every frequency: the amplifier's thermal floor — private by construction in any real
crate — is 30 % coherent, while the clock line, the one term that *is* common to a
crate, is only 30 % coherent. In a real board the picture is the opposite: coherence
≈ 1 at the clock line and its harmonics, ≈ 0 on the floor. The module already supports
this: build S(f) = S_priv(f)·I + S_shared(f)·g gᵀ and call `generate_from_csd`. That
also makes the LUCiD Σ deliberately non-Kronecker (Σ_c ⊗ S(f) fails), which is the
unrestricted comparator the NFPA note says is needed anyway.

Smaller points. A 4 MHz Gaussian line is 2 bins wide; a real clock is a deterministic,
phase-locked sinusoid, i.e. not Gaussian — it belongs to the "structural" family, or
use `width_hz=0` and note the Gaussian-process approximation. "Cable ringing" at
150 MHz is a base/connector resonance; a long-cable reflection would be a comb with
spacing v/2L (≈ 3 MHz for 30 m), not a single Lorentzian. Flicker at 0.4 % is
declared-but-negligible on 512 ns; fine. At the 16–32 µs `window_ns` the plan requires,
df drops to 30–60 kHz and DC-DC switching lines (100 kHz–2 MHz) plus 3 decades of 1/f
become representable — the preset must be re-parameterised for that grid, not reused.
ADC quantisation (LSB²/12, white) and the deterministic clock are both natural
acquisition-contract N-families like the alias fold. Real water-Cherenkov front ends
digitise at ~100–125 MS/s (mPMT) or use QTC/TDC charge-time pairs; 1 GHz is LUCiD's
convention, so the arm should say "LUCiD's readout", not "a WC detector's".

## 3. Prometheus instead of HeST + noise_module?

What Prometheus (vendored `8c19938`) emits: per-photon `(sensor_id, string_id, t)`
plus truth (flavour, CC/NC, E, direction, vertex). Its new mDOM response
(`utils/fadc_digitization.py`) does QE → TTS → dark noise (thermal + correlated ⁴⁰K
bursts, 750 Hz/PMT) → Gaussian SPE template → 3.3 ns FADC bins, then keeps only bins
with q > 0.05 pe and ToT hits; there is no electronic noise anywhere. Our
`prometheus_simulation.response` reproduces NuBench's pulse emulation (QE, merge, smear).

Feasibility per recognised factor (canonical copy in `docs/TESTBEDS.md` §3.3; "bridge" =
histogram per-PMT photon times on a fixed window, SPE-convolve, add `noise_module` per crate;
✅ feasible as the programme defines it, ⚠ feasible with a stated limitation, ✘ not feasible):

| recognised factor | **Prometheus alone** (hits / sparse FADC) | **Prometheus + bridge + `noise_module`** | **LUCiD + `noise_module` V2** (arm A) | **HeST → `qp_simulator` → `noise_module`** (arm B) |
|---|---|---|---|---|
| **geometry** | ✅ `.geo` text file, 6 shipped (ORCA, ARCA, IceCube, GVD, P-ONE, TRIDENT); exact pairing by replaying one injection file | ✅ same | ✅ one JSON file, 16 configs; exact pairing by PRNG key (`apply_translation=False`) | ✅ `make_cell(...)`, 24 → 1 on the identical cell; pairing tested |
| **physics / signal type** | ✅ **strongest**: ν flavour, CC/NC, track vs cascade, E, direction, vertex from LeptonInjector; angular error is a real consequence variable | ✅ same labels, now on a trace | ⚠ flasher / PhotonSim μ, e, π⁰ / SIREN μ, e gun; material (water ↔ WbLS); no neutrino kinematics, consequence variable still ours to define | ⚠ ER vs NR **degenerate with energy** at fixed E (quasiparticle-only readout); WIMP spectrum as held-out U |
| **noise type — covariance Σ** (κ, whitening lemma, `z_noise` branch) | ✘ no electronic noise; only rate/nuisance families (dark rate, QE, TTS, jitter, dead modules, correlated ⁴⁰K bursts); no Σ̂, no realised Σ | ✅ Σ̂ and Σ recorded per crate by `noise_module`; the digitiser is then ours (declare it) | ✅ V2: `filtered` front end, `spectral_shared_private` crate, ρ_ij(f) reported | ✅ per cell, `kappa_floor` in `truth.parquet`; TES budget constants placeholder |
| **noise type — structural** (gain drift, channel loss, jitter, alias fold) | ⚠ hit-level only: N1–N5 in `prometheus_simulation.interventions` | ✅ trace-level families incl. alias fold and quantisation | ✅ | ✅ |
| **readout physics** | PMT array, hit level | PMT array, 1–3.3 ns trace (hypothetical: real DOMs read ATWD/FADC at 300/40 MS/s or ToT) | PMT array, 1 GHz trace (LUCiD's convention) | TES calorimeter, 250 kHz — the second readout physics and the DELight link |
| **granularity axis** (same volume, different sensor count) | ⚠ six real geometries differ in volume too; no dense ↔ sparse of the *same* volume shipped | ⚠ same | ✅ `n_sensors` 2 000 → 20 000 (planned) | ✅ 24 → 1 |
| **licence / release** | ✅ LGPL-2.1; outputs not covered | ✅ | ✘ **gate A0**: no licence as of today | ⚠ **gate B0**: MIT text, PyPA sample copyright line |
| **cost to reach the trace** | — | ~ the LUCiD units bridge + a window convention (days) | notebook exists; `noise_module_lucid` ~8 days once A0 lands | built |
| **status today** | built, licence-clean, DynEdge frozen-public-model arm | not started | notebook only | 14 cells, tests green |

So: Prometheus alone keeps two of the three recognitions and loses the one the paper
is about. Prometheus plus a bridge keeps all three, is licence-clean (LGPL-2.1, outputs
uncovered), and has better physics labels — it is the natural **replacement for LUCiD
if A0 never lands**, not for HeST. Replacing HeST with it collapses the programme to
two PMT arrays and drops the granularity-24→1 dissociation, the second readout physics,
and the DELight relevance. The bridge would be ~the LUCiD units bridge plus a window
convention; the honest caveat is that the digitiser (1 ns or 3.3 ns dense trace) is then
ours, while real IceCube/KM3NeT DOMs read ATWD/FADC at 300/40 MS/s or ToT only.


## 4. Fixed (same day) — what changed in the tree

**`src/noise_module`** (generic, no LUCiD code; 141 existing tests + 10 new all green)

- `spectral_models.Filtered` — `density = source × Π |H_k(f)|²`; the multiplicative operator the
  additive composite lacked. Config: `{"type": "filtered", "source": {...}, "filters": [{...}, ...]}`;
  filter detail is recorded in the composite metadata. Nesting is refused.
- `spectral_models.Peaking` (`|H|² = 1 + gain·L(f)`, a unit-gain resonant bump) and
  `Reflection` (`|1 + r e^{-2πifτ}|²`, the cable comb) as transfer functions for `Filtered`.
- `MultiChannelNoiseGenerator` mode **`spectral_shared_private`** — the shared and private processes
  get their own component lists; the base `noise_power` is split between them in the ratio of their
  absolute integrals so a unit-gain channel still has variance `noise_power`. Implied
  `S(f) = S_sh(f)·ggᵀ + S_pr(f)·diag(p²)`; `implied_correlation_spectrum(meta, i, j)` gives ρ_ij(f);
  `implied_csd(C, N, meta)` the dense (F, C, C); metadata flags `kronecker_separable: False`.
  Shares the WP-N1 freezing contract with `shared_private`.
- `config.MultiChannelConfig` — `shared_components`, `private_components`, mode validation.
- `tests/test_filtered_and_spectral_coherence.py` — 10 tests, including realized Welch coherence
  against the implied ρ_ij(f) on a 65 µs record.

**`notebooks/pmt_frontend_v2.py`** — the preset, units bridge, `crate_preset()` (interventions that
hold the *private* power fixed so "clock ×5" adds power instead of re-partitioning 0.64 mV²) and
`add_pmt_noise()`. Both notebook builders import it; the executed `.ipynb` outputs still show V1
until re-run against a LUCiD clone (Part B of `noise_models_herald_lucid` was executed here with the
patched module and runs clean). The nb1 Σ-cell "group coherence 0.3 → 0.7" is replaced by
"broadband common mode (30 %)" — a shared white term — which is exactly the structure V1 imposed on
every cell, now an intervention.

**Final spectrum** — `docs/reviews/pmt_frontend_v2_final.png` (script alongside). Shares of the
0.64 mV²: amplifier floor × |H|² 87 %, flicker 0.8 %, clock 62.5 MHz 8.8 %, 125 MHz 2.2 %,
187.5 MHz 0.7 %. S(500)/S(2) = 0.16 (V1: 0.53). Implied ρ at the clock 0.86, realized 0.85; ≈ 0 on
the floor. Matched-cell κ floor 6.7 at N/C = 8 → 1.28 at N/C = 512, unchanged rule.

**Follows from the same critique, not done:** the HeRALD arm's `herald_simulation.NoiseSpec` also
uses `shared_private` with a flat `corr_strength`. Physically the bath fluctuation (shared) is
low-frequency and the TFN/Johnson terms are private, so the same `spectral_shared_private` split
applies there; `tes_budget` would need to expose its components as two lists.

## Actions

1. ~~Replace the additive `rolloff`/`resonance` by a transfer function~~ done (`Filtered`).
2. ~~Build the LUCiD Σ with the lines shared and the floor private; report ρ_ij(f)~~ done
   (`spectral_shared_private`).
3. Re-run both notebooks against a LUCiD clone so the stored outputs show V2.
4. Re-parameterise the preset for the 16–32 µs window before any covariance cell
   (switching lines at 100 kHz–2 MHz and three decades of 1/f become representable).
5. Apply the same shared/private split to the HeRALD `NoiseSpec` (§4, last paragraph).
6. Keep Prometheus as the LUCiD fallback with a waveform bridge; do not swap it for HeST.
