# `noise_module_lucid` — the LUCiD front-end noise module

_18 September 2026. Code: `src/noise_module_lucid/`. Design:
`docs/EXPERIMENT_DESIGN.md` §II.7 and §III.6. Review that motivated V2:
`docs/reviews/LUCID_NOISE_REVIEW_2026-09-11.md`. Canonical inventory:
`docs/TESTBEDS.md` §1.1._

This document explains what the package is, what physics each term represents,
which grid it lives on, what is deliberately **not** modelled, and how to use it.
It is the explanation companion to the code; the notebook
`notebooks/noise_models_herald_lucid.ipynb` is the hands-on version.

---

## 1. Scope

LUCiD emits a **photoelectron histogram** per sensor; it has no electronics
model — no amplifier, cable, digitiser, no cross-channel covariance. The
ORACLE study needs an **acquisition contract** (an assumed covariance Σ̂) and a
**realized covariance** Σ to monitor, so the front end is supplied by us.

`noise_module_lucid` is a **thin adapter on top of `noise_module`**:

- it does **not** fork `noise_module` (the generic spectral components,
  `MultiChannelNoiseGenerator`, `psd_resampling` are reused unchanged);
- it does **not import LUCiD**, so it builds and tests without a LUCiD clone
  (arm A is licence-gated on **A0**, `docs/TESTBEDS.md` §1.1).

What it supplies: presets, the pe → mV units bridge, the covariance-unit
grouping, the crate-wise adapter with implied/realized covariance reporting, and
the declared intervention families.

---

## 2. The two layers: what LUCiD models and what we add

Three words are kept apart throughout (the notebook's opening):

- **Noise** — random fluctuation of the readout with **no event in the record**;
  a random trigger measures it. Everything in this package is noise in this
  sense.
- **Background** — a real unwanted *event* (radon, dark count, atmospheric ν).
  It has a shape and lives in the signal; a random trigger sees it only as a
  rate. Simulated as events, never as Σ.
- **Acquisition contract** — the assumptions a trained model carries about the
  noise (its Σ̂). Noise that differs from Σ̂ is what the monitoring package is
  for.

**Inside LUCiD (signal-carrier statistics, per photon and per PMT, independent):**

| term | physics |
|---|---|
| quantum efficiency | Bernoulli conversion, Poisson counting noise on charge |
| single-photoelectron charge | dynode-chain gain fluctuation (Gaussian core + exponential tail) |
| transit-time spread | Gaussian jitter on every photon's arrival (σ ≈ 1 ns) |
| dark noise | thermionic emission from the photocathode, ~4 kHz/PMT, as events |
| time jitter / TDC | charge-dependent timing resolution and quantisation |

**Ours to add (the front end, the environmental, and the covariance):** the
amplifier floor, the front-end bandwidth, flicker, clock/switching pickup, base
ringing, cable effects, and the cross-channel structure. These are the terms in
§3–§5.

The split matters because it decides what a "noise" family is allowed to claim:
dark counts and SPE/TTS are **events** in LUCiD and cannot be part of Σ; the
electronics between the anode and the digitiser is ours and is.

---

## 3. Two grids, because the record length decides what exists

A single number — the frequency resolution `df = fs / N` — decides which terms
are even representable.

| | short (V2) | long |
|---|---|---|
| `window_ns` | 512 ns | 16 384 ns (16 µs) and up to 32 768 ns |
| `fs` | 1 GHz (LUCiD convention) | 1 GHz |
| `df` | 1.95 MHz | 30–61 kHz |
| Nyquist | 500 MHz | 500 MHz |
| clock / switching | 62.5 / 125 / 187.5 MHz | same, **plus** 0.1 / 0.25 / 0.5 / 1 / 2 MHz DC-DC lines |
| flicker (1/f) | ~2 decades, weak | reference moved to 100 kHz, more decades in band |
| 50 Hz mains | 2.6e-5 of one bin — **absent** | still far below one bin — **absent** |

The short grid is the notebook / display convention. The **κ (covariance) cells
need the long grid**: with 64 channels and only 512 samples, N/C = 8 and the
realized covariance of a *matched* cell already has κ ≈ 5–7 against its own
implied one, so an "alarm" would read estimator noise, not the detector. The
rule of thumb is **N/C ≳ 500 for a κ floor below ~1.1** (`EXPERIMENT_DESIGN.md`
§II.7.3). 50 Hz is *still* not representable at 32 µs; it needs decimation
(`psd_resampling.alias_fold_psd_density`) or a ≥ 1 s record, and is declared out
of scope here.

Select the preset explicitly:

```python
from noise_module_lucid.presets import preset_for
base, crate = preset_for(window_ns=32768)      # long preset
```

---

## 4. The preset terms (V2)

Every term is a `noise_module` spectral component. **A bandwidth is a filter,
not a source**: the front-end low-pass and the base ringing *multiply* the
amplifier floor via the `filtered` component, instead of being summed with it.
(V1 summed them, which left 58 % of the power flat to Nyquist and put noise
structure faster than the SPE pulse — the review's point (i).)

**Private path** (one process per PMT; shaped by the front end):

| term | component | physics |
|---|---|---|
| amplifier floor | `filtered(white, [frontend_bandwidth, base_ringing])` | transistor thermal + shot noise, shaped by the ~250 MHz low-pass and the ~150 MHz base/connector resonance |
| flicker | `filtered(powerlaw −1, …)` | transistor 1/f; reference 10 MHz (short) / 100 kHz (long) |

**Shared path** (one process per crate, entering at the digitiser, unfiltered):

| term | component | physics |
|---|---|---|
| clock | `line` 62.5 MHz + 125 + 187.5 MHz | ADC clock and its harmonics, common to a board |
| switching (long grid) | `line` 0.1–2 MHz | DC-DC converters |

A real clock is a deterministic, phase-locked sinusoid; the Gaussian `line` of
finite width is the **stationary-Gaussian approximation** of it, declared as
such. Total single-channel power is 0.8 mV rms (placeholder).

---

## 5. Cross-channel structure: coherence belongs to a term, not to the crate

The crate is built with `MultiChannelNoiseGenerator` mode
**`spectral_shared_private`**: the shared and private processes get their **own**
component lists, so the coherence is a function of frequency,

$$S(f) = S_\text{shared}(f)\, g g^\top + S_\text{private}(f)\,\mathrm{diag}(p^2),$$

with the clock lines shared and the amplifier floor/flicker private. Then
ρ_ij(f) ≈ 0.86 at the clock and ≈ 0 on the floor — the physically correct picture.

V1 used `shared_private` with one PSD and a flat `corr_strength = 0.3`, which
made the thermal floor 30 % coherent and the clock only 30 % coherent — the
wrong way round (the review's point (ii)). That flat common mode is now a
*declared intervention* (`add_broadband_common_mode`), not the default.

`Σ` is deliberately **non-Kronecker** here (Σ_c ⊗ S(f) fails), which is the
unrestricted comparator the NFPA analysis needs anyway.

The **covariance unit** is a crate / string / sub-array of 16–64 PMTs, never the
whole tank (`grouping.py`; `EXPERIMENT_DESIGN.md` §III.6). When the detector
ships no crate id, `groups_from_positions(..., n_sectors, n_bands)` gives
angular-sector × height-band cells.

---

## 6. The units bridge

LUCiD's bin holds summed photoelectron charge; a PSD in pe²/Hz is meaningless.
`units.py` convolves each channel with a two-exponential SPE voltage template
(2 ns rise, 8 ns fall, 4 mV peak per pe — placeholders) to mV, then noise is
added in mV. `charge_to_mv` returns `(C, N)` in mV, truncated to the window.

The mV-per-pe constant must be tied to LUCiD's own gain so the two layers cannot
silently disagree (`EXPERIMENT_DESIGN.md` §II.7.4); that is an open calibration
item.

---

## 7. The adapter

```python
from noise_module_lucid import add_readout_noise, groups_from_positions

trace_mv, meta = add_readout_noise(
    charge_waveform,                     # (C, N) pe, or mV with in_units="mv"
    groups=groups,                       # list of index arrays; default contiguous crates
    window_ns=32768,                     # selects the long preset, recorded
    seed=0,
)
```

`meta` carries, per group: the **implied** covariance Σ̂, the **realized**
covariance Σ, the matched-cell estimator floor `κ = cond(Σ̂⁻¹Σ)`, the group
indices, the preset, `window_ns`, and the SPE amplitude. The realized covariance
is what makes κ a *measured* lever rather than a nominal setting.

---

## 8. Declared intervention families

`interventions.py` registers the N contract. **Covariance-type** (transfer
function unchanged, Σ ≠ Σ̂, carries the κ prediction):

- `clock_scale(k)` — clock/switching line amplitude ×k (narrow-band κ);
- `add_broadband_common_mode(scale)` — frequency-flat crate coherence.

**Digitiser** (the sampling contract changes, exact closed-form prediction):

- `decimate_alias_fold(trace, factor)` + `alias_fold_prediction(...)` — keep every
  factor-th sample with no anti-alias filter; `noise_module.psd_resampling` gives
  the folded spectrum.

**Structural** (transfer function / channel set changes; moves the mean *and* the
noise-only statistics):

- `gain_drift_apply(trace, groups, sigma, rng)`;
- `channel_loss_apply(trace, fraction, rng)`;
- `cable_lag_apply(trace, delays_samples)` — cable length → lagged signal.

Use `presets.crate_preset(shared=..., private=..., keep_private_power=True)` to
build a variant: the per-PMT *private* power is held at its reference value and
the total re-derived, so "clock ×5" **adds** line power instead of silently
re-partitioning a fixed total (the `normalize` rule would otherwise do the
latter, as V1 did).

---

## 9. Documented but deliberately not modelled as Σ

`DOCUMENTED_NOT_IMPLEMENTED` records the families named in the working note that
belong to the **event / background** layer (S/U), not the covariance:

| family | why not Σ |
|---|---|
| radon in water | radioactive **events** (rate/spectrum/position) |
| PMT pre-/after-pulse | out-of-trigger and in-gate contamination; event/pileup |
| pileup near detector (IWCD) | in-gate hit from a different event |
| dark-rate fluctuation | LUCiD's dark counts are events; the rate/electronics part is the N candidate |
| ADC/TDC aperture jitter | digitiser-contract N, to add with quantisation |
| DSNB (as signal) | physics S family |
| atmospheric ν tail | background event family |
| QE vs DIS | neutrino–nucleus interaction channel; physics S family |
| 50 Hz mains | not representable without decimation / long records |

This keeps the axes clean: **origin (N/S/G) and harm are separate**, and a
background is never smuggled into the covariance.

---

## 10. Status, acceptance, provenance

**Implemented and tested:** V2 and the long-window preset, units bridge,
grouping, adapter, interventions; `PYTHONPATH=src python -m pytest
src/noise_module_lucid/tests -q` → 12 passed.

**Open / placeholder:**

- every constant is `placeholder`; read from a measured front-end or LUCiD's gain
  before any dataset is released (provenance record in `presets.PROVENANCE`);
- the 16–32 µs re-parameterisation (switching lines, 1/f reference) is a
  declared decision, not yet validated against a measured CSD;
- the mV-per-pe calibration against LUCiD's gain;
- 50 Hz mains / long-record drift via decimation.

**Acceptance** follows `EXPERIMENT_DESIGN.md` §II.7.6: the preset PSD integrates
to `noise_power`; a matched cell's reported κ equals the estimator floor for that
(C, N) before any mismatch is injected; a swept κ reproduces its requested value
over a decade; `validate_csd_ensemble` passes; the units round-trip; the alias
fold matches the closed form; every dataset writes a provenance record.

**No LUCiD work starts before gate A0** (a permissive licence). The noise package
is independent of that gate by construction.
