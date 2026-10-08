# Implementation prompt — `noise_module_lucid` revision to V3 (21 September 2026)

This prompt is for an implementation agent working in the ORACLE repository. The audit behind it
is in the Project doc `claude/lucid-noise-module-audit-2026-09-21.md`, summarised in §0. Before
dispatch, the author ticks or edits the decisions in §1. Work packages R1–R7 run in the order given.

## 0. Revision advice: finding → fix

| # | finding (audit) | fix | why this fix |
|---|---|---|---|
| 1 | Per-crate seed `seed + idx[0]`. The channel structure (Σ̂) is redrawn every event. Seeds collide: event 0 / crate 1 is the same as event 64 / crate 0. | Keep the **structure seed** (per detector, fixed) separate from the **event seed** (per event). Derive every per-crate stream from `SeedSequence`. Add a stateful `LucidReadout` that holds the frozen structure. | This restores WP-N1: a fixed Σ̂ with fresh noise in every event. |
| 2 | κ = cond(Σ̂⁻¹Σ) on the zero-lag C×C covariance. It does not depend on the shared term's spectrum. A clock ×k and a broadband common mode of equal power are indistinguishable. | Add **κ_spectral**, the generalized eigenvalues of (Ŝ(f), S(f)) in every bin, pooled over bins. For stationary noise this is exact in the circulant approximation of the full CN×CN Σ. Keep κ_zero_lag and the Kronecker κ as labelled diagnostics, and report the **Kronecker gap**. | The paper's non-Kronecker claim then enters the number the paper reports. |
| 3 | The floor passes through the 250 MHz low-pass and the 150 MHz base ringing, but the SPE pulse passes through neither. | One `FrontEnd` object provides both the signal impulse response and the noise power response. **Signal:** template ⊗ h_base ⊗ h_amp. **Amplifier noise:** white × \|H_amp\|². Base ringing moves to the signal path. The anti-alias filter becomes a real Butterworth. | Signal and noise share one transfer function by construction, which whitening and matched filtering need. It also gives "structural N" a concrete meaning. |
| 4 | The long preset is different hardware: the 1/f corner is about 500 kHz (V2) versus about 8 kHz (LONG). Switching lines are about 0.1 % of the power. Both presets renormalise to 0.64 mV². | Describe the hardware **once, in absolute units** (mV²/Hz, µV rms per line) and evaluate it on each grid. Drop components the grid cannot represent, and record which ones. Delete `PMT_PRIVATE_LONG`. | Then "same detector, longer record" is literally true. The long record sees more low-frequency power, as it should. |
| 5 | The Gaussian clock line has σ = 2 MHz on every grid: 1 bin at 512 ns, 65 bins at 32 µs. | Default: a deterministic tone (R5). In the legacy Gaussian mode, width = one bin on every grid. | A crystal clock is far narrower than one bin at either record length. |
| 6 | 62.5 MHz = fs/16. A clock locked to the sampler is a fixed-amplitude, 16-sample pattern whose phase is set by the trigger offset. The Gaussian line gives Rayleigh amplitudes instead. | Add a `ClockPickup` process with the models `sample_locked` (default) / `random_phase` / `gaussian_line`. It has per-channel path delays, so the cross-spectrum is complex. The trigger offset is shared across the whole detector. Its second-order statistics are analytic. | This is physically right. It is also the one term where the Fisher-cumulant rung (D7/D8) can tell the model apart from a Gaussian stand-in. |
| 7 | Minor: a single-pole filter labelled "order 2", no alias folding, no clock correlation between crates, grouping by geometry. | The single-pole filter and alias folding are handled in R3. Correlation between crates is handled in R5 via the shared trigger offset. Geometric grouping stays, as a declared stand-in. | — |

Continuity rule: V2 stays importable, frozen and marked superseded, so that the executed
September notebooks remain reproducible. V3 is the new default.

## 1. Author decisions in force (edit before dispatch)

- [x] **L1** default clock model = `sample_locked` (alternatives: `random_phase`, `gaussian_line`).
- [x] **L2** hardware placeholders:
  - 1/f corner f_c = 100 kHz.
  - Switching lines 0.1 / 0.25 / 0.5 / 1 / 2 MHz at 30 µV rms each.
  - Clock tone rms chosen so each tone's power equals V2's line power: 62.5 MHz 0.238 mV, 125 MHz 0.119 mV, 187.5 MHz 0.069 mV (agent recomputes these from V2 and records them).
  - The white density S0 is fixed so that the V3 total on the 512-ns grid is 0.64 mV².
- [x] **L3** the trigger offset, and so the tone phase, is shared across the whole detector in each event.
- [x] **L4** anti-alias filter = 4th-order Butterworth, f_c = 250 MHz.
- [x] **L5** base ringing: an analog resonant section at 150 MHz, Q = 3.75 (continuity with V2's 20 MHz half-width), relative amplitude 0.3. All three values are placeholders.
- [ ] **L6** whether κ_spectral replaces κ_zero_lag as the *registered* κ. The agent implements both and never edits `PREREGISTRATION.md`. Tick only after reading the R2 numbers.
- [ ] **L7** optional `pedestal_subtraction="per_phase"` plus a `stale_pedestal(eps)` intervention (R5.4). If unticked, it is not implemented.

---

```text
Implement the V3 revision of src/noise_module_lucid specified in
docs/IMPLEMENTATION_PROMPT_LUCID_NOISE.md (this file), with the author decisions in its §1.
Do the implementation, the tests and the documentation edits. Do not return another plan.

Repository discipline
- Work in the ORACLE repository on branch dev. Read AGENTS.md if it exists. Check git status
  first and preserve every existing change. Do not reset, clean, commit or push. If git leaves a
  stale .git/index.lock, report it; do not force anything.
- Nothing may import LUCiD (arm A is gated on A0). Generic additions go into src/noise_module
  only where this prompt says so, and must not change the behaviour of any existing noise_module
  test. Only the noise_module_lucid tests that pin V2 numbers may be updated, and each such update
  is listed in the report.
- Read first: src/noise_module_lucid/* (all), docs/noise_module_lucid.md,
  src/noise_module/{multichannel_noise,spectral_models,psd_resampling,validation,utils}.py,
  src/latent_monitor/whitening.py (KroneckerWhitener.kappa_against),
  docs/EXPERIMENT_DESIGN.md §II.7 and §III.6, notebooks/_build_nb1.py and _build_nb2.py
  (call sites), docs/reviews/plot_pmt_frontend_v2_final.py.
- Test command: PYTHONPATH=src python -m pytest src/noise_module src/noise_module_lucid -q
  (Python >= 3.11; test_review_fixes.py needs tomllib). Everything must pass at the end.
- Tolerances below are acceptance thresholds. If one fails, fix the cause; do not loosen it. If
  the physics makes a threshold impossible, stop that package and say so in the report.

R0. Freeze V2
- Move the V2 constants (PMT_PRIVATE, PMT_PRIVATE_LONG, PMT_SHARED, PMT_SHARED_LONG,
  PMT_FRONTEND_V2/LONG, PMT_CRATE_V2/LONG) unchanged into presets_v2.py, with a module docstring
  "superseded 2026-09-21, kept to reproduce the executed notebooks". Re-export them under the
  same names so old imports still work. Add a test that the V2 PSD on the 512 grid is bitwise
  identical to the value before the change (hash the array now, before editing).

R1. Seeding and frozen structure (bug; do first)
1. adapter.py: add class LucidReadout(groups, preset, *, structure_seed: int).
   - __init__: ss = np.random.SeedSequence(structure_seed). For each group g, use the child
     ss.spawn(n_groups)[g] to draw gains = 1 + N(0, channel_gain_jitter) and private_strengths
     ~ U(private_strength_range), with the same distributions as
     MultiChannelNoiseGenerator._draw. Store them, plus a sha256 structure hash per group.
   - add(charge, *, event_seed, in_units="pe", spe=None) -> (trace, meta). For each group, the
     noise rng is np.random.default_rng(np.random.SeedSequence([structure_seed, event_seed, g])).
     Build MultiChannelNoiseGenerator(base, crate, rng=...), call set_channel_structure(...) with
     the stored arrays, then generate. meta records structure_seed, event_seed and the per-group
     structure hashes.
2. add_readout_noise keeps its positional signature and gains keyword-only structure_seed=0 and
   event_seed. The old `seed` becomes an alias of event_seed and emits a DeprecationWarning.
   add_pmt_noise follows. Internally both build a LucidReadout, cached on
   (groups, preset id, structure_seed).
3. Tests (new file tests/test_seeding.py):
   a. Same structure_seed with event seeds 0 and 1: identical gains, private strengths and
      implied Σ (exact); noise differs (|corr| of the flattened noise < 0.05).
   b. No collisions: event seeds 0..127 × 4 groups of 64. All 512 (event, group) noise blocks
      are pairwise different (compare hashes of the first 64 samples).
   c. Bitwise reproducibility for the same (structure_seed, event_seed).
   d. Ensemble: 400 events, one fixed structure, 512 samples. The pooled zero-lag covariance
      against the implied Σ gives κ_zero_lag within the matched-cell floor for
      N_eff = 400·512 (compute the floor by simulation with 20 bootstrap replicas; accept if κ is
      inside the 95 % interval).
4. Update the call sites (_build_nb1.py, _build_nb2.py, the plot script) to event_seed=.
   Do not execute notebooks that need LUCiD.

R2. Spectral κ
1. New module noise_module_lucid/covariance.py:
   - implied_csd(meta) -> (f, S) with S of shape (F, C, C), complex Hermitian. Delegate to
     MultiChannelNoiseGenerator.implied_csd and add the R5 tone terms when present.
   - kappa_spectral(S_hat, S, *, bins=slice(1, -1)) -> dict with
     lam = the generalized eigenvalues from scipy.linalg.eigh(S[k], S_hat[k]) for every bin k;
     kappa = lam.max()/lam.min(); kappa_f = per-bin max/min; argmax_bin.
     Exclude DC and Nyquist. Guard bins where S_hat is singular with a floor of 1e-12 × its
     trace and report how many bins were guarded.
   - kappa_kronecker(S_hat, S): fit the Kronecker factors of each (Σ_c = Σ_f S(f)/Σ_f tr S(f)·C,
     T(f) = tr S(f)/C) and reuse the eigen-ratio logic of KroneckerWhitener.kappa_against.
     kronecker_gap = kappa_kronecker / kappa_spectral.
   - estimate_csd(records (R, C, N), fs, band_bins) -> a CSD averaged over records and over
     `band_bins` adjacent bins. Raise if dof = R·band_bins < 10·C.
   - kappa_spectral_floor(C, dof, n_bands, n_rep=50, seed): the matched-cell floor by
     simulation (a complex Wishart per band), returning the median and the 95 % interval.
2. Keep adapter.kappa(meta), renamed in meta to "kappa_zero_lag". Keep the old "kappa" key as
   an alias for one release. Its docstring must state that it is blind to the shared spectrum.
3. Tests (tests/test_kappa_spectral.py):
   a. kappa_spectral(S, S) == 1 to 1e-10 (matched).
   b. From V3 (R4) with delay_spread_ns = 0, build two cells: clock ×5, and a broadband common
      mode whose added shared power equals the clock cell's added power (1 % tolerance).
      κ_zero_lag must agree to 1e-6 (the documented blindness). The spectral κ must tell them
      apart by *where* it is > 1: in the clock cell, kappa_f > 1 + 1e-9 exactly in the tone
      bins; in the broadband cell, in > 90 % of the bins. Report both kappa values; do not
      assert their ratio (the rank-1 direction scales the broadband one with C).
   c. Proportional shared and private PSDs (Kronecker-separable): kronecker_gap == 1 to 1e-8.
      For V3: gap > 1, and report the value.
   d. Matched realized cell: 256 random-trigger records, C = 64, N = 4096, band_bins = 16.
      κ_spectral(realized vs implied) lies inside kappa_spectral_floor's 95 % interval.
      This is the spectral analogue of EXPERIMENT_DESIGN AC-2.
4. Do not edit PREREGISTRATION.md or the registered estimand. Put the choice κ_spectral vs
   κ_zero_lag (decision L6) into docs/TODO.md as an open decision, with the numbers from 3b–3d.

R3. One front end for signal and noise
1. Generic, in src/noise_module/spectral_models.py: add an AnalogFilter component
   {"type": "analog_filter", "b": [...], "a": [...]} whose shape(f) = |H(j2πf)|², using
   scipy.signal.freqs. Register it in component_from_config and add unit tests. It is a filter
   for Filtered; used as a standalone source, the docstring says it is a white term shaped by H.
2. New noise_module_lucid/frontend.py, class FrontEnd:
   - amp: scipy.signal.butter(order=4, Wn=2π·250e6, analog=True) (L4); unit DC gain.
   - base: H_base(s) = 1 + a·(ω0/Q)s / (s² + (ω0/Q)s + ω0²), with f0 = 150 MHz, Q = 3.75,
     a = 0.3 (L5); unit DC gain.
   - power_response(f, stages) -> the product of |H|² for the chosen stages;
     analog_filter_configs(stages) -> the AnalogFilter configs, from the same (b, a).
   - impulse_response(stages, fs, n_fft=16384): the causal response obtained by evaluating H on
     the rFFT grid of a zero-padded record and inverse-transforming. Assert that the tail
     energy in the last 10 % is < 1e-8 of the total. Truncate at 99.9999 % of the energy.
3. units.py: charge_to_mv(wf_pe, spe=None, frontend=FrontEnd.default()) convolves with
   spe ⊗ h_base ⊗ h_amp. `mv_per_pe` keeps its meaning as the anode pulse peak. Record the
   post-chain peak and area in the metadata. frontend=None reproduces V2 exactly (used by R0).
4. Presets (V3): the private floor is Filtered(white, AnalogFilter(amp)); flicker likewise.
   No base term is in the noise path. The shared clock enters at the digitiser, unfiltered.
5. Aliasing check (a report number only): evaluate the analog floor on an 8 GHz grid, fold it to
   1 GHz with psd_resampling.alias_fold_psd_density, and report the folded-in fraction. If it
   is > 1 %, add the folded PSD to the preset and say so. It is expected to be negligible
   with the 4th-order filter.
6. New structural N family in interventions.py: frontend_bandwidth(k). It scales the amp
   corner by k and applies the change to BOTH the signal (a new impulse response) and the noise
   (a new power response). Register it in N_FAMILIES with kind "structural".
7. Tests (tests/test_frontend.py):
   a. The area of a unit impulse through the chain equals the SPE area to 1e-6
      (unit DC gain).
   b. rfft(pulse_out)/rfft(spe) == H_base·H_amp to 1e-6 relative, in the bins where
      |rfft(spe)| > 1e-3 of its maximum.
   c. The private noise PSD equals S0·|H_amp|² on the 512 and 32768 grids to 1e-12, with no
      150 MHz bump. The pulse spectrum does show the bump.
   d. frontend_bandwidth(0.5) lowers both the signal's high-frequency energy and the noise
      power above 100 MHz. Check the direction and the magnitude against the closed form.

R4. One hardware description on every grid
1. presets.py (V3) in absolute units:
   - S0 is the white density in mV²/Hz. Compute it once so that the V3 total on the 512 grid is
     0.64 mV², store it as a literal, and assert it in a test.
   - PMT_PRIVATE = [Filtered(white S0), Filtered(powerlaw, scale S0, exponent −1,
     reference_hz = F_CORNER = 1e5)], so flicker equals the floor at f_c (L2).
   - Switching lines are defined by rms (30 µV each, L2). Their power is exactly rms². On any grid
     they are tones (R5, random_phase, not shared with the clock), or Gaussian lines one bin
     wide if L1 = gaussian_line.
   - Remove PMT_PRIVATE_LONG and the separate long shared list: one hardware, one list.
2. preset_for(window_ns, fs=1e9) evaluates the hardware on the grid N = window_ns·fs/1e9:
   - It drops components the grid cannot represent (a line with f < 2·df or f > fs/2) and
     lists them in meta["dropped_components"].
   - It sets noise_power to the absolute integral on that grid, so the normalize step is the
     identity. Add an assert inside a test that the renormalisation factor is 1 to 1e-12.
   - crate_preset(...) is re-implemented on absolute scales. An intervention adds power; it never
     re-partitions power.
3. Tests (tests/test_hardware_consistency.py):
   a. The private density on the 32768 grid at bins 64·k equals the 512-grid density at bin k,
      to 1e-12 relative.
   b. The flicker/floor ratio at f_c is 1 to 1e-9. The long-grid total is ≥ the short-grid total.
   c. Each switching line's realized power (from 200 records) equals rms² within 3 %.
   d. clock_scale(5) multiplies the clock-tone power by 25 (1e-12) and leaves the private power
      unchanged (1e-12).

R5. Clock pickup as a deterministic tone
1. New noise_module_lucid/clock.py, class ClockPickup(f0=62.5e6, harmonics=(1, 2, 3),
   rms_mv=(L2 values), model="sample_locked", delay_spread_ns=2.0, amp_jitter=0.0):
   - sample_locked: requires P = fs/f0 to be an integer (16). Per event, draw the trigger offset
     k ~ U{0..P−1} once for the whole detector (L3), from a dedicated stream
     SeedSequence([structure_seed, event_seed, "clock"]). Channel i:
     x_i(t) = g_i · Σ_h √2·rms_h · cos(2π h f0 (t − τ_i) + 2π h k / P + φ_h).
     τ_i ~ U[0, delay_spread_ns] and φ_h are frozen with the structure (R1). g_i are the
     crate gains.
   - random_phase: φ ~ U[0, 2π) per event, per crate.
   - gaussian_line: the V2 behaviour with width_hz = 0 (one bin) on every grid.
   - analytic_second_order(C, N): the zero-lag covariance term
     Σ_h rms_h² g_i g_j cos(2π h f0 (τ_i − τ_j)), and the CSD term, a single-bin complex entry at
     h f0 with phase −2π h f0 (τ_i − τ_j). Note in the docstring that the discrete-phase
     average equals the continuous-phase one for h·k/P not a multiple of 1/2, which holds for
     h = 1..3 and P = 16.
   - analytic excess kurtosis of each channel: Σ_h (−1.5·rms_h⁴·g_i⁴) / var_i². Report it
     honestly: it is small per channel (about −0.015 at a ~10 % share); the non-Gaussianity is
     concentrated in the clock bins.
2. LucidReadout adds ClockPickup to every group, with the shared trigger offset. meta carries
   the tone parameters, k, and "inter_group_tone_covariance" (analytic, groups × groups
   blocks), because Σ is no longer block-diagonal across crates. κ stays per group; say so in
   the docs.
3. implied_csd (R2) includes the tone terms. The tone CSD is complex; confirm that
   kappa_spectral handles complex Hermitian inputs (test with a 2×2 case against a hand
   computation).
4. (Only if L7 is ticked.) pedestal_subtraction="per_phase" subtracts the tone conditioned on
   k. The intervention stale_pedestal(eps) leaves a residual tone of amplitude eps·A.
5. Tests (tests/test_clock.py):
   a. Private noise off, sample_locked: the trace is exactly periodic with period 16 samples.
   b. Over 1600 events, k is uniform (chi², p > 0.01).
   c. The realized cross-phase at f0 between channels i and j, averaged over 400 events, is
      −2π f0 (τ_i − τ_j) within 0.05 rad.
   d. The realized pooled zero-lag covariance of the tone part matches
      analytic_second_order within 3 %.
   e. With private noise off, the coefficient of variation across 400 events of |X(f0)| in one
      channel is < 1e-9 for sample_locked (f0 sits exactly on a bin, so the phase does not
      change the magnitude) and 0.52 ± 0.05 for gaussian_line (Rayleigh, √(4/π − 1)).
      With private noise at its V3 level, the sample_locked CV matches the Rician prediction
      from the analytic tone-to-private bin power ratio within ±0.03.
      This is the check that separates Rayleigh amplitudes from a fixed amplitude.
   f. gaussian_line occupies exactly one bin on both grids.

R6. Interventions and N contract update
- clock_scale scales the tone rms (ClockPickup) or the line scale (gaussian_line).
- add_broadband_common_mode stays: a white shared term in absolute units, defined by rms.
- Register frontend_bandwidth (R3.6) and, if L7 is ticked, stale_pedestal.
- In DOCUMENTED_NOT_IMPLEMENTED, move "adc_tdc_aperture_jitter" next to a new entry for
  "adc_quantisation (LSB²/12, white)" with a one-line pointer, not an implementation.

R7. Documentation, provenance, figures, report
- Rename the default preset PMT_FRONTEND_V3. PROVENANCE records: S0, F_CORNER, the switching
  rms, the tone rms, the clock model, P, delay_spread, the filter (b, a) of both stages, the
  dropped components per grid, structure_seed, and the version "V3 2026-09-21".
- Rewrite docs/noise_module_lucid.md §3–§8 for V3: the signal/noise transfer-function split,
  the one-hardware rule, the clock model and its non-Gaussian signature, the three κ and which
  one answers which question. Update the README table and the test count.
- Copy plot_pmt_frontend_v2_final.py to plot_pmt_frontend_v3_final.py and produce
  docs/reviews/pmt_frontend_v3_final.png with four panels: (a) PSD per component on the
  512 and 32768 grids, overlaid; (b) ρ_ij(f) and the cross-phase at the clock;
  (c) the SPE pulse before and after the front end; (d) κ_spectral per band versus the floor.
- Write docs/reviews/LUCID_NOISE_REVISION_REPORT_2026-09-21.md containing:
  - one row per finding 1–7: what changed, which test, and the numbers;
  - the power shares on both grids, the folded-in alias fraction, κ_zero_lag / κ_spectral /
    Kronecker gap for the matched, clock ×5 and broadband cells, and the tone kurtosis;
  - every existing test that was edited, and why;
  - the edits that EXPERIMENT_DESIGN §II.7 and PREREGISTRATION would need. List these; do not
    make them.

Done means: all tests pass; the report exists; there are no LUCiD imports; the git worktree
contains only the changes above; nothing is committed.
```

## 2. Effort and order

R0 0.25 d → R1 0.5 d → R4 1 d → R3 1.5 d → R5 1.5 d → R2 1.5 d → R6 0.25 d → R7 1 d,
about 7.5 days. R4 comes before R3 and R5 because both build on the absolute-unit hardware.
R2's test 3b needs the V3 clock, so R2 runs last among the physics packages, but its
functions can be written earlier.
