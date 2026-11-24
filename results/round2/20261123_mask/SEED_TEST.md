# SEED_TEST — r2-mask M3 (readout-seeded LUCiD fit)

Runbook `RUNBOOK_2026-11-23_R2_MASK_SCALE.md` §M3; expectation registered in `PREREG_2026-11-23_R2_MASK.md` (appendix M3) before running. Events: the 200 Y2 `dev_probe_test` photon bombs per detector of the r2-tier1 yardstick input. Fits: LUCiD's expected-response model and losses as in the yardstick (`tier1_yardstick.Model`, L-BFGS-B, maxiter 40 for every start); counts-only is primary (the joint time term has an unresolved t₀ offset). Starts (x, y, z) as listed; ln N from a linear fit of ln N on ln(total charge) over `dev_probe_train`; t₀ from the median time residual at the starting vertex (truth start: full true θ). Good basin = final error ≤ 2 × the median final error of the truth start (same detector and mode).

## SK (200 events)

### counts (primary) — good-basin threshold 0.339 m

| Start | start error mean / median (m) | final mean (m) | final median (m) | good basin | iterations | wall (s/fit) | projected into tank |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) truth | 0.000 / 0.000 | 0.180 | 0.170 | 88.5 % ± 2.3 | 4.8 | 1.28 | 0.0 % |
| (b) charge centroid | 3.883 / 3.952 | 3.351 | 3.292 | 0.0 % ± 0.0 | 6.4 | 1.27 | 0.0 % |
| (c) frozen readout, s9000 @10k | 0.359 / 0.236 | 0.463 | 0.349 | 47.5 % ± 3.5 | 5.2 | 1.16 | 0.0 % |
| (d) frozen readout, matched init | 0.547 / 0.407 | 0.623 | 0.455 | 27.5 % ± 3.2 | 5.2 | 1.15 | 0.0 % |
| (e) raw-feature baseline | 0.659 / 0.539 | 0.677 | 0.548 | 21.5 % ± 2.9 | 5.2 | 1.12 | 0.0 % |

### joint — good-basin threshold 0.334 m

| Start | start error mean / median (m) | final mean (m) | final median (m) | good basin | iterations | wall (s/fit) | projected into tank |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) truth | 0.000 / 0.000 | 0.180 | 0.167 | 87.5 % ± 2.3 | 4.9 | 1.48 | 0.0 % |
| (b) charge centroid | 3.883 / 3.952 | 0.479 | 0.247 | 68.5 % ± 3.3 | 7.8 | 1.70 | 0.0 % |
| (c) frozen readout, s9000 @10k | 0.359 / 0.236 | 0.273 | 0.256 | 75.5 % ± 3.0 | 5.4 | 1.35 | 0.0 % |
| (d) frozen readout, matched init | 0.547 / 0.407 | 0.285 | 0.254 | 68.0 % ± 3.3 | 5.9 | 1.39 | 0.0 % |
| (e) raw-feature baseline | 0.659 / 0.539 | 0.273 | 0.244 | 73.0 % ± 3.1 | 6.2 | 1.45 | 0.0 % |

## WCTE (200 events)

### counts (primary) — good-basin threshold 0.033 m

| Start | start error mean / median (m) | final mean (m) | final median (m) | good basin | iterations | wall (s/fit) | projected into tank |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) truth | 0.000 / 0.000 | 0.030 | 0.017 | 68.0 % ± 3.3 | 5.1 | 1.45 | 0.0 % |
| (b) charge centroid | 0.382 / 0.393 | 0.146 | 0.118 | 11.0 % ± 2.2 | 8.3 | 1.58 | 0.0 % |
| (c) frozen readout, s9000 @10k | 0.055 / 0.046 | 0.058 | 0.046 | 30.5 % ± 3.3 | 5.3 | 1.35 | 0.0 % |
| (d) frozen readout, matched init | 0.070 / 0.057 | 0.068 | 0.056 | 20.0 % ± 2.8 | 5.5 | 1.41 | 0.0 % |
| (e) raw-feature baseline | 0.078 / 0.068 | 0.070 | 0.061 | 19.5 % ± 2.8 | 5.0 | 1.27 | 0.0 % |

### joint — good-basin threshold 0.201 m

| Start | start error mean / median (m) | final mean (m) | final median (m) | good basin | iterations | wall (s/fit) | projected into tank |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) truth | 0.000 / 0.000 | 0.125 | 0.101 | 84.0 % ± 2.6 | 10.5 | 3.08 | 0.0 % |
| (b) charge centroid | 0.382 / 0.393 | 0.120 | 0.102 | 86.5 % ± 2.4 | 8.9 | 2.55 | 0.0 % |
| (c) frozen readout, s9000 @10k | 0.055 / 0.046 | 0.101 | 0.088 | 95.5 % ± 1.5 | 8.6 | 2.58 | 0.0 % |
| (d) frozen readout, matched init | 0.070 / 0.057 | 0.100 | 0.093 | 93.5 % ± 1.7 | 8.7 | 2.65 | 0.0 % |
| (e) raw-feature baseline | 0.078 / 0.068 | 0.103 | 0.096 | 95.5 % ± 1.5 | 8.6 | 2.48 | 0.0 % |

## Registered expectation (SK, counts-only)

Readout start (c): 47.5 % in the good basin (expected ≥ 80 %: not met); centroid start (b): 0.0 % (expected ≤ 30 %: met). Expectation **not met**. (c) is below 50 %: **this piece is dropped** (runbook §M3.5).
