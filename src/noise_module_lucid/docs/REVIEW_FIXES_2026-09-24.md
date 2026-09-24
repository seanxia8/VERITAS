# LUCiD front-end review fixes — 24 Sep 2026

Response to the independent review of `noise_module_lucid`. Every P0/P1/P2 item
is addressed in code, tested, and re-run through the tutorial. Preset numbers
changed because the co-calibration (P0.3) is a real model change; all levels
remain placeholders.

## P0 — must fix

### P0.1 Aperture jitter (`digitiser.py`)
The old `sampling_jitter` resampled the discrete trace by interpolation; on a
white trace it *removed* ~7 % of the variance (measured 0.928) and on a smooth
trace did nothing. **Fix:** `aperture_jitter_noise` adds the first-order error
`delta_n * s'(t_n)` (central difference), i.e. slew rate becomes noise — the
physical aperture-jitter effect. A flat signal is untouched; a fast edge is not.
`sampling_jitter` is kept for band-limited signals and now raises (`strict=True`)
when more than 20 % of the trace power is above 0.8 x Nyquist.

### P0.2 Dark-noise contract (`pulses.py`)
LUCiD already adds dark noise (4.2 kHz/PMT); re-adding would double-count.
**Fix:** `add_dark_pulses(..., baseline_rate_hz=presets.LUCID_DARK_RATE_HZ)`
treats `rate_hz` as the target total and injects only the delta; metadata records
`target_rate_hz`, `baseline_rate_hz`, `injected_rate_hz`, `assumes_lucid_dark`.
Pass `baseline_rate_hz=0.0` to supply the full rate.

### P0.3 SPE / front-end co-calibration (`presets.py`, `units.py`)
The placeholder SPE (2 ns / 8 ns, ~20 MHz) against a 250 MHz corner left ~62 % of
the noise power above the signal band (a factor ~13 mismatch). **Fix:** the SPE
template is now a WCTE-like 3-inch PMT (1 ns rise / 3 ns fall, power -3 dB
~50.9 MHz) and the front-end corner is **derived** from it:
`FRONT_END_CORNER_HZ = 2.5 * SPE_BANDWIDTH_HZ` (~127 MHz), with a 4th-order
roll-off. Noise above 3x the signal band falls to ~14 %. `validation.bandwidth_report`
reports the ratio and the residual overlap.

## P1 — should fix

### P1.1 Physical grouping (`grouping.py`)
Added `board_map` (explicit per-channel board id — the correct grouping) and
`proximity` (greedy nearest-neighbour geometry proxy). `contiguous` is retained
and declared a proxy. `validation.grouping_sensitivity` / `grouping_report`
report spatial compactness so the sensitivity is disclosed.

### P1.2 Non-Gaussian clock (`clock.py`)
`add_deterministic_clock` adds a phase-locked tone per clock line with the same
normalized power as the Gaussian line, so the PSD is unchanged and the higher
cumulants are real. `crate_preset(clock="deterministic")` moves the lines out of
the Gaussian shared process; `add_pmt_noise` injects the tone per crate.

### P1.3 Per-channel gain (`adapter.py`)
`add_pmt_noise(..., channel_gains=...)` applies LUCiD's `PerPmtParams.gain` to
the **signal** and pins the same gains into the noise structure, so signal and
noise share the channel gain. With `channel_gains=None` the metadata records
`signal_gain_applied=False` — the jitter is then declared amplifier-only.

### P1.4 Long window (`presets.long_window_components`, `adapter.long_window_preset`)
A preset valid at `df = 1/window_ns` that adds DC-DC switching lines
(100 kHz / 500 kHz / 1 MHz) and lets 1/f develop down to `df`. 50 Hz still needs
a ~ms record and is not added.

## P2 — nice to have

### P2.1 Cable dispersion (`delay.py`)
`cable_delay(..., dispersion_s2, attenuation_per_hz)` gives
`H(f) = exp(-a f) exp(-2 pi i (tau f + D f^2 / 2))`; `group_delay_s` returns
`tau_g(f) = tau + D f`.

### P2.2 Charge-dependent after-pulsing (`pulses.py`)
`add_afterpulses(..., primary_charge, probability_per_pe)` uses
`p(q) = min(1, p0 + p_per_pe * q)`, a brighter primary ionising more gas.

### P2.3 Realized-CSD validation (`validation.py`)
`realized_csd_check` compares the realized Welch coherence to the implied
`rho_ij(f)`; asserted in the test suite (peak at 62.5 MHz, median |diff| < 0.1).

## Before / after (512 ns, 64 PMTs)

| quantity | before | after |
|---|---|---|
| SPE -3 dB | ~19.5 MHz | ~50.9 MHz |
| front-end corner | 250 MHz (additive order 2) | 127.4 MHz (derived, order 4) |
| noise above 3x signal band | ~62 % above 100 MHz | ~14 % |
| S(500)/S(2) | 0.158 | 0.003 |
| shared fraction | 11.8 % | 14.1 % |
| matched-cell kappa floor, N/C=8 | 6.7 | 13.8 |
| matched-cell kappa floor, N/C=512 | 1.28 | 1.37 |

The higher small-N kappa floor is expected: band-limiting the noise reduces the
number of independent samples per record, so the estimator floor rises. The
covariance cells still need the 16-32 us window (the rule becomes N/C >~ 1000
for a floor below ~1.3).
