# Toy waveform with channel-ordered arrival delays (registered: false)

Author request 24 Oct 2026: "各组 arriving time 逐组变晚，加一个 causal 关系".
Code `src/jepa_noise/delay_views.py` (`DelayedTier1Views`), script
`scripts/jepa_toy_delay_population.py`, tests `src/jepa_noise/tests/test_delay_views.py`.

Channel c's pulse onset is `t0 + c * delta` (record-fraction units; 0.01 = 2.56 samples), so
channels 0-3 (context) always fire before 4-7 (target).
- `fixed`: delta = 0.01 for every event (a known causal structure, 3 physics parameters).
- `per_event`: delta uniform in [0.005, 0.015] per event, a 4th physics parameter (an arrival
  gradient, like a source position).

## Population displacement D (deg), linearised fixture, A4 noise, no sampling

D = max principal angle between the top-k population CCA context directions and R_c
(k = number of physics parameters). `no_delay` reproduces the registered §A table exactly.

| variant | split | g = 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.225 | 0.45 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| no delay | channel | 0.0 | 11.3 | 22.3 | 48.0 | 76.9 | 84.1 | 87.5 | 88.8 | 89.4 |
| fixed 0.01 | channel | 0.0 | 11.0 | 21.7 | 47.2 | 76.9 | 84.2 | 87.7 | 89.0 | 89.7 |
| per-event 0.005-0.015 | channel | 0.0 | 11.0 | 21.8 | 47.4 | 76.9 | 84.2 | 87.3 | 87.8 | 88.2 |
| no delay | time | 90.0 | 90.0 | 90.0 | 90.0 | 89.9 | 89.9 | 89.9 | 89.9 | 89.9 |
| fixed 0.01 | time | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 |

Reading: in the linear population picture the causal ordering changes nothing. CCA is symmetric
in the two views and a fixed delay is a per-view change of template (P2), so the displacement
by shared common mode is unchanged. A per-event delay is a strongly identified extra physics
direction (canonical correlation 0.946) and is displaced only at large g. Any effect of the
causal structure must therefore come from the deep model (attention has to learn the
inter-channel offsets), which is what the bridge run tests.

## Signal variance per parameter, relative to the total noise variance (channel split, g = 0.02)

| parameter | linearised | true nonlinear pulses |
|---|---|---|
| amplitude | 0.024 | 0.025 |
| t0 | 1.57 (no delay) / 1.78 (delayed) | 0.18 |
| tau | 0.008 | 0.008 |
| delay (per-event) | 0.37 | 0.061 |

The linearisation overstates the timing parameters (t0 by ~9x, delay by ~6x), because their
ranges are wide compared with the pulse's fast rise. Amplitude and tau are linear and exact.
