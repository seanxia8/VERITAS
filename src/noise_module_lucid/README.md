# noise_module_lucid

LUCiD-customised PMT front-end noise for the ORACLE water-Cherenkov arm
(`docs/EXPERIMENT_DESIGN.md` §II.7, arm A). A thin package that **depends on
`noise_module` and never forks it**: the generic spectral/multichannel machinery
lives there, the LUCiD-specific preset, units bridge, grouping, digitiser and
pulse processes live here.

> **Gate A0 is still open.** LUCiD has no licence (its `README.md` says "the
> license is being finalized"); the clone at `external/LUCiD` (origin
> `https://github.com/dowlingwong/LUCiD`) still ships none. This package is
> ours (MIT) and depends on nothing from LUCiD at import time, so it can be
> released independently later — a fork could not.

## Where this sits in the stack

LUCiD simulates the **event** layer: photon transport, quantum efficiency,
single-photoelectron charge smearing, transit-time spread, dark counts, TDC
jitter — all independent per photon and per PMT, with **no electronic noise and
no inter-channel covariance anywhere**. Three layers must be kept apart:

| layer | owns | examples | enters `Sigma`? |
|---|---|---|---|
| signal / background physics | event generators (LUCiD, PhotonSim, ...) | DSNB, atmospheric tail, QE vs DIS, radon | no |
| LUCiD event stochastic layer | `lucid/simulation/digitizer.py` | dark counts, TTS, SPE charge, pre/after pulse | no |
| **front-end acquisition** | **this package** | amplifier floor + bandwidth, clock lines + coherence, cable delay, digitiser, alias fold | **yes** (and structural N) |

## The sorted instruction list (24 Sep 2026)

The instruction list was a domain-realistic DSNB-analysis checklist, but it
mixes the three layers. Sorted (full version and rationale in
[`docs/INSTRUCTION_SORTING_2026-09-24.md`](docs/INSTRUCTION_SORTING_2026-09-24.md)):

**Taken into this package (front-end acquisition):**
- amplifier white floor + front-end bandwidth + base/connector ringing
  (`presets.py`, already V2)
- ADC clock + harmonics, frequency-dependent crate coherence
  (`presets.py` + `noise_module.spectral_shared_private`, already V2)
- **cable length → lagged signal** (`delay.py`, new)
- **ADC/TDC random sampling: quantisation + aperture jitter**
  (`digitiser.py`, new)
- alias fold / decimation (`noise_module.resampling.psd`, re-exported in
  `interventions.py`)

**Declared and injected, but event-level — not `Sigma` (kept out):**
- PMT thermal dark counts, pre-pulses, after-pulses (`pulses.py`, new) —
  LUCiD already generates dark noise; this is for declaring a rate change or
  adding contamination to an existing trace

**Left to the event generators / LUCiD (out of scope here):**
- DSNB as signal, atmospheric-neutrino tail, QE vs DIS interaction channel,
  radon in water — signal/background physics; they define `J_y` and the S/U
  families, not the front-end covariance
- pile-up / in-gate contamination from a different event — a second physical
  event, breaks exact pairing, an S/U family

**Administrative / geometry:**
- license — gate A0, unresolved
- geometry by JSON — already LUCiD's format (a four-key JSON file)

## Layout

| file | status | purpose |
|---|---|---|
| `presets.py` | V2 + co-calibration | component lists, SPE constants, derived front-end corner, long-window builder, contract tags |
| `units.py` | V2 | SPE template, `to_mv` / `charge_to_mv`, `spe_bandwidth` |
| `adapter.py` | V2 + P1.3/P1.4 | `crate_preset` (clock modes), `add_pmt_noise` (gains, clock, grouping), `long_window_preset`, `kappa` |
| `grouping.py` | P1.1 | `channel_groups` — `board_map` (physical), `proximity`, `z_plane`, `contiguous` |
| `interventions.py` | V2 + P0.1 | declared N families: clock (Gaussian/deterministic), common mode, gain drift, quantisation, aperture jitter, cable delay, alias fold |
| `digitiser.py` | P0.1 | quantisation (LSB²/12), `aperture_jitter_noise` (slew-rate), guarded `sampling_jitter` |
| `delay.py` | P2.1 | cable delay with dispersion and attenuation |
| `pulses.py` | P0.2/P2.2 | dark (delta vs baseline) / pre-pulse / charge-dependent after-pulse |
| `clock.py` | P1.2 | deterministic phase-locked clock tone with the Gaussian line's power |
| `validation.py` | P0.3/P1.1/P2.3 | `bandwidth_report`, `realized_csd_check`, `grouping_sensitivity`, `kappa_floor_sweep` |
| `readouts.py` | new | declared readout registry (1 GHz PMT + `custom_readout`); calibration flags; `UNMODELLED` DOM/SiPM/TES |
| `dataset.py` | new | LUCiD-arm driver: `CellSpec`/`EventSpec`, `intervention_matrix`, `covariance_cells`, `run_cell` (numpy), `run_lucid_cell` (lazy LUCiD), `write_cell` (truth/traces/provenance) |
| `__main__.py` | new | `python -m noise_module_lucid` → the dataset driver |
| `docs/` | moved + new | V2 review, V1/V2 figures, instruction sorting, the 24 Sep review fixes |
| `tests/` | new | 40 unit tests |

The end-to-end procedure (produce the dataset → validate on the linear subject →
train and interpret one frozen transformer) is the runbook
[`docs/RUNBOOK_LUCID_2026-09-24.md`](../../docs/RUNBOOK_LUCID_2026-09-24.md).

## Usage

```python
import numpy as np
from noise_module_lucid import to_mv, add_pmt_noise, kappa, bandwidth_report
from noise_module_lucid import digitiser, delay, pulses, clock, interventions

wf_pe = np.zeros((64, 512))              # LUCiD (n_sensors, n_bins) photoelectrons
sig_mv = to_mv(wf_pe)                    # -> mV on the same 1 ns grid

# signal and noise share LUCiD's per-PMT gain (P1.3); deterministic clock (P1.2)
trace, groups = add_pmt_noise(sig_mv, seed=0, channel_gains=lucid_gain,
                              clock="deterministic")
print("matched-cell kappa floor:", kappa(groups[0]), bandwidth_report())

trace = delay.cable_delay(trace, 10e-9, 1e9, dispersion_s2=1e-17)   # lag + dispersion
trace = digitiser.quantise(trace, lsb=1.0)                          # ADC LSB
trace = digitiser.aperture_jitter_noise(trace, 50e-12, 1e9, rng)    # slew-rate jitter
trace, meta = pulses.add_dark_pulses(trace, 1e9, rate_hz=6e3)       # +1.8 kHz over LUCiD
```

## Placeholders

Every level is a placeholder (`presets.PROVENANCE`): 0.8 mV rms total, 4 mV/pe
SPE (1 ns rise / 3 ns fall), a front-end corner **derived** from the SPE
bandwidth (2.5x, ~127 MHz), 62.5 MHz clock, 64 PMTs/crate. The two that set the
SNR are the total rms and the SPE amplitude. Do not quote this preset as a
validated detector model before it is calibrated.

## Review history

- `docs/LUCID_NOISE_REVIEW_2026-09-11.md` — the V1→V2 fixes.
- `docs/REVIEW_FIXES_2026-09-24.md` — the independent-review P0/P1/P2 fixes
  (aperture jitter, dark contract, SPE/front-end co-calibration, board map,
  deterministic clock, per-channel gain, long window, dispersion, after-pulsing,
  realized-CSD validation).
