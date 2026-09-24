# LUCiD instruction list — sorted, and what the noise module takes in (24 Sep 2026)

_The instruction was a domain-realistic DSNB-analysis checklist for the LUCiD
arm. It is sensible as a detector inventory, but it mixes three different
layers of the simulation stack. This note sorts it, decides what belongs in
`noise_module_lucid`, and records what is deliberately left out._

## 1. The list, sorted

ORACLE classes: **N** acquisition/noise shift (corrupted measurement), **S**
training-support shift (valid but under-represented physics), **U** undeclared
family (routed to abstention), **G** geometry factor. "Enters Σ" means it is
part of the front-end covariance the whitening lemma uses.

| instruction item | class | layer | enters Σ | disposition |
|---|---|---|---|---|
| License | — | admin | — | gate A0, unresolved (LUCiD still ships no licence) |
| Define geometry by JSON | **G** | LUCiD | no | already LUCiD's format (four-key JSON) |
| 改noise定义 (DSNB as signal) | signal | event generator | no | signal model; defines `J_y`, not noise |
| Atmospheric ν (higher E, tail in band) | **S** (hard) / U | event generator | no | the hard S contrast: valid physics overlapping signal |
| QE vs DIS interaction channel | **S** | event generator | no | event morphology; defines `J_y` |
| PMT thermal / dark (workfunction) | N (rate shift) | **LUCiD** | no | already `generate_dark_noise`; a rate change is the N family |
| Radioactive (radon in water) | **S/U** | event generator | no | background population; low-rate monitoring concern |
| Cable length → lagged signal | **N** | **noise_module** | no (phase) | `delay.py` — signal-path phase ramp |
| Electronics random sampling (ADC/TDC) | **N** | **noise_module** | no | `digitiser.py` — quantisation + aperture jitter |
| PMT pre/after pulse | **N** (rate shift) | **LUCiD** | no | `pulses.py` — event-level, declared |
| In-gate contamination / pile-up (IWCD, near det) | **S/U** | event generator | no | a second event in the gate; breaks exact pairing |

## 2. Two category errors to avoid

- **"DSNB as signal" is a signal-model change, not a noise change.** DSNB is a
  diffuse, low-rate *event population*; a 512 ns PMT waveform does not contain a
  "DSNB signal". Reframing the arm as a DSNB search is legitimate and helps
  ORACLE — the atmospheric tail overlapping the DSNB band is exactly the hard
  **S** the N-vs-S claim needs — but it changes `J_y` and the consequence `K`,
  not the front-end noise.
- **Pre/after-pulse and pile-up are different families.** Pre/after-pulse is
  PMT-intrinsic (Poisson, ion feedback); in-gate contamination is pile-up
  (overlapping events, breaks exact pairing). Do not merge them.

## 3. What the noise module takes in

Only two instruction items are genuinely new front-end work; the rest of the
list is already modelled or belongs elsewhere.

1. **Cable delay (`delay.py`).** Every `noise_module` spectral primitive is
   magnitude-only — `Filtered.shape` multiplies `|H_k(f)|²`, and
   `Reflection` puts the delay inside a *magnitude* comb. A cable lag is
   `H(f) = exp(-2πi f τ)`, i.e. pure phase, and `|H|² = 1`, so it cannot be a
   PSD component. It is applied to the trace on the signal path. A delay and a
   reflection are not interchangeable.
2. **Digitiser (`digitiser.py`).** Quantisation (white, variance `LSB²/12`) and
   aperture / sampling jitter, completing the sampling-contract family. The
   alias fold is the third member and already lives in
   `noise_module.resampling.psd`.
3. **PMT pulse processes (`pulses.py`).** Dark counts, pre-pulses and
   after-pulses as declared, injectable event-level processes — kept out of Σ,
   with the after-pulse tail as a geometric/exponential offspring (the same
   shape as the Hawkes option in `noise_module.artifacts.injector`).

## 4. Carry-overs from the V2 review that this list reinforces

- Group crate coherence by a **physical readout unit**, not channel index
  (`grouping.py`; LUCiD exposes no string id, so use positions).
- Re-parameterise the preset for the **16–32 µs windows** the covariance cells
  need (switching lines at 100 kHz–2 MHz and three decades of 1/f become
  representable).
- The matched-cell κ floor at N/C = 8 (512 ns) is ~6.7, so the default window
  cannot measure a covariance change; the alarm would read estimator noise.

## 5. Net effect

The DSNB reframing is a net positive for the N-vs-S story, provided it is
recorded as the arm's **signal model** and the new front-end items (cable,
digitiser, pulses) are tagged with their contract and `enters_sigma` flag in
`presets.CONTRACTS`, so the protocol's typed alarm-time builder cannot leak
truth.
