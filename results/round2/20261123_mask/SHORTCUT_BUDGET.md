# SHORTCUT_BUDGET — r2-mask M1 (network-free shortcut budget per mask)

Runbook `RUNBOOK_2026-11-23_R2_MASK_SCALE.md` §1.2. Targets of the default `mae_repaired` loss (hit flag and ln q on every hidden PMT, time / 50 ns on hidden hit PMTs; unit weights). S = per-PMT mean over fitting events; L = one Ridge per channel, shared across PMTs, from the k = 8 nearest visible PMTs (h, ln q, t, distance) plus the static means. E = 1 − MSE/Var on scoring events (combined: 1 − ΣMSE/ΣVar). U contains irreducible noise that is the same for every mask of a detector: only differences in U between masks are interpreted. Events: `dev_probe_train`, fitting and scoring runs disjoint; photon bombs 1,700 + 1,700 (the whole cohort), μ and e 1,000 + 1,000. Masks: the training masker (`ViewMasker`, training seed, n_min = 0). WCTE and SK are registered; JUNO and HK are **exploratory**.

## WCTE

N_PMT = 1995, median nearest-neighbour spacing 0.072 m, default patch 32, code 1a2a0ab0.

Correlation length ℓ_c (charge channel): photon 2.48 m (34.5 spacings), muon 0.66 m (9.2 spacings), electron 0.68 m (9.4 spacings).

### WCTE — photon (primary)

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.526 | 0.474 | 0.0058 | 0.7111 | 0.2831 | 0.078 | 1.09 | 0.785 / 0.455 / 0.698 |
| random | 0.500 | 0.000 | 0.0059 | 0.7086 | 0.2855 | 0.083 | 1.16 | 0.781 / 0.459 / 0.699 |
| patch8 | 0.501 | 0.002 | 0.0062 | 0.7033 | 0.2905 | 0.160 | 2.23 | 0.775 / 0.456 / 0.696 |
| patch32 | 0.497 | 0.006 | 0.0058 | 0.6888 | 0.3054 | 0.293 | 4.08 | 0.757 / 0.452 / 0.688 |
| patch128 | 0.513 | 0.038 | 0.0056 | 0.6591 | 0.3352 | 0.474 | 6.60 | 0.722 / 0.441 / 0.664 |
| patch256 | 0.513 | 0.102 | 0.0058 | 0.6303 | 0.3638 | 0.604 | 8.40 | 0.689 / 0.433 / 0.643 |

### WCTE — muon

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.526 | 0.474 | 0.0007 | 0.7571 | 0.2422 | 0.078 | 1.09 | 0.868 / 0.403 / 0.560 |
| random | 0.500 | 0.000 | 0.0008 | 0.7444 | 0.2548 | 0.083 | 1.16 | 0.852 / 0.401 / 0.557 |
| patch8 | 0.501 | 0.002 | 0.0008 | 0.6966 | 0.3026 | 0.161 | 2.24 | 0.794 / 0.385 / 0.548 |
| patch32 | 0.497 | 0.006 | 0.0006 | 0.6134 | 0.3860 | 0.293 | 4.08 | 0.694 / 0.353 / 0.518 |
| patch128 | 0.513 | 0.038 | 0.0003 | 0.5036 | 0.4961 | 0.471 | 6.56 | 0.563 / 0.306 / 0.471 |
| patch256 | 0.513 | 0.102 | 0.0011 | 0.3828 | 0.6161 | 0.603 | 8.39 | 0.423 / 0.246 / 0.414 |

### WCTE — electron

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.526 | 0.474 | 0.0011 | 0.6704 | 0.3285 | 0.078 | 1.09 | 0.784 / 0.373 / 0.569 |
| random | 0.500 | 0.000 | 0.0011 | 0.6615 | 0.3374 | 0.083 | 1.16 | 0.772 / 0.373 / 0.570 |
| patch8 | 0.501 | 0.002 | 0.0012 | 0.6233 | 0.3755 | 0.161 | 2.24 | 0.722 / 0.365 / 0.565 |
| patch32 | 0.497 | 0.006 | 0.0010 | 0.5528 | 0.4462 | 0.293 | 4.07 | 0.632 / 0.343 / 0.549 |
| patch128 | 0.513 | 0.038 | 0.0009 | 0.4472 | 0.5519 | 0.476 | 6.62 | 0.501 / 0.301 / 0.507 |
| patch256 | 0.513 | 0.102 | 0.0009 | 0.3781 | 0.6210 | 0.594 | 8.27 | 0.417 / 0.268 / 0.482 |

## SK

N_PMT = 11096, median nearest-neighbour spacing 0.698 m, default patch 173, code 1a2a0ab0.

Correlation length ℓ_c (charge channel): photon 13.90 m (19.9 spacings), muon 3.26 m (4.7 spacings), electron 4.20 m (6.0 spacings).

### SK — photon (primary)

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.526 | 0.474 | 0.0172 | 0.4335 | 0.5493 | 0.705 | 1.01 | 0.393 / 0.241 / 0.514 |
| random | 0.500 | 0.000 | 0.0174 | 0.4340 | 0.5486 | 0.723 | 1.04 | 0.394 / 0.244 / 0.514 |
| patch43 | 0.500 | 0.000 | 0.0179 | 0.4302 | 0.5519 | 1.503 | 2.15 | 0.388 / 0.242 / 0.512 |
| patch173 | 0.499 | 0.002 | 0.0168 | 0.4183 | 0.5650 | 2.592 | 3.71 | 0.370 / 0.235 / 0.499 |
| patch692 | 0.499 | 0.002 | 0.0168 | 0.3837 | 0.5995 | 4.918 | 7.05 | 0.322 / 0.220 / 0.464 |

### SK — muon

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.526 | 0.474 | 0.0074 | 0.4745 | 0.5181 | 0.705 | 1.01 | 0.662 / 0.335 / 0.454 |
| random | 0.500 | 0.000 | 0.0076 | 0.4731 | 0.5193 | 0.723 | 1.04 | 0.655 / 0.334 / 0.455 |
| patch43 | 0.500 | 0.000 | 0.0077 | 0.4331 | 0.5592 | 1.502 | 2.15 | 0.521 / 0.305 / 0.437 |
| patch173 | 0.499 | 0.002 | 0.0062 | 0.3915 | 0.6023 | 2.592 | 3.71 | 0.401 / 0.268 / 0.413 |
| patch692 | 0.499 | 0.002 | 0.0071 | 0.3158 | 0.6771 | 4.852 | 6.95 | 0.261 / 0.205 / 0.353 |

### SK — electron

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.526 | 0.474 | 0.0105 | 0.4061 | 0.5834 | 0.705 | 1.01 | 0.438 / 0.282 / 0.432 |
| random | 0.500 | 0.000 | 0.0103 | 0.4055 | 0.5842 | 0.723 | 1.04 | 0.434 / 0.282 / 0.432 |
| patch43 | 0.500 | 0.000 | 0.0103 | 0.3886 | 0.6012 | 1.504 | 2.16 | 0.376 / 0.268 / 0.422 |
| patch173 | 0.499 | 0.002 | 0.0103 | 0.3632 | 0.6265 | 2.577 | 3.69 | 0.308 / 0.247 / 0.403 |
| patch692 | 0.499 | 0.002 | 0.0109 | 0.3065 | 0.6826 | 4.860 | 6.97 | 0.215 / 0.201 / 0.352 |

## JUNO (exploratory)

N_PMT = 10000, median nearest-neighbour spacing 0.604 m, default patch 156, code 1a2a0ab0.

Correlation length ℓ_c (charge channel): photon 13.22 m (21.9 spacings), muon 2.68 m (4.4 spacings), electron 2.90 m (4.8 spacings).

### JUNO — photon (primary)

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.500 | 0.500 | -0.0012 | 0.4596 | 0.5415 | 0.611 | 1.01 | 0.454 / 0.230 / 0.528 |
| random | 0.500 | 0.000 | -0.0012 | 0.4598 | 0.5414 | 0.628 | 1.04 | 0.454 / 0.230 / 0.528 |
| patch156 | 0.499 | 0.002 | -0.0013 | 0.4447 | 0.5566 | 2.135 | 3.53 | 0.436 / 0.224 / 0.511 |

### JUNO — muon

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.500 | 0.500 | -0.0050 | 0.5025 | 0.5024 | 0.611 | 1.01 | 0.646 / 0.322 / 0.495 |
| random | 0.500 | 0.000 | -0.0051 | 0.5004 | 0.5047 | 0.628 | 1.04 | 0.639 / 0.321 / 0.494 |
| patch156 | 0.499 | 0.002 | -0.0053 | 0.4161 | 0.5892 | 2.132 | 3.53 | 0.387 / 0.247 / 0.438 |

### JUNO — electron

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.500 | 0.500 | -0.0054 | 0.4020 | 0.6034 | 0.611 | 1.01 | 0.400 / 0.241 / 0.419 |
| random | 0.500 | 0.000 | -0.0052 | 0.4019 | 0.6032 | 0.628 | 1.04 | 0.396 / 0.241 / 0.420 |
| patch156 | 0.499 | 0.002 | -0.0055 | 0.3649 | 0.6407 | 2.122 | 3.51 | 0.263 / 0.207 / 0.392 |

## HK (exploratory)

N_PMT = 19746, median nearest-neighbour spacing 1.000 m, default patch 308, code 1a2a0ab0.

Correlation length ℓ_c (charge channel): photon 1.51 m (1.5 spacings), muon 3.91 m (3.9 spacings), electron 1.54 m (1.5 spacings).

### HK — photon (primary)

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.500 | 0.500 | 0.0174 | 0.4047 | 0.5780 | 1.010 | 1.01 | 0.380 / 0.223 / 0.433 |
| random | 0.500 | 0.000 | 0.0172 | 0.4054 | 0.5774 | 1.028 | 1.03 | 0.381 / 0.222 / 0.434 |
| patch308 | 0.499 | 0.002 | 0.0163 | 0.3901 | 0.5936 | 4.657 | 4.66 | 0.350 / 0.213 / 0.418 |

### HK — muon

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.500 | 0.500 | 0.0075 | 0.4025 | 0.5900 | 1.010 | 1.01 | 0.593 / 0.292 / 0.408 |
| random | 0.500 | 0.000 | 0.0074 | 0.4012 | 0.5913 | 1.028 | 1.03 | 0.585 / 0.290 / 0.407 |
| patch308 | 0.499 | 0.002 | 0.0057 | 0.3457 | 0.6485 | 4.653 | 4.65 | 0.285 / 0.215 / 0.358 |

### HK — electron

| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| parity | 0.500 | 0.500 | 0.0093 | 0.3574 | 0.6333 | 1.010 | 1.01 | 0.362 / 0.223 / 0.371 |
| random | 0.500 | 0.000 | 0.0092 | 0.3574 | 0.6334 | 1.028 | 1.03 | 0.360 / 0.223 / 0.371 |
| patch308 | 0.499 | 0.002 | 0.0087 | 0.3246 | 0.6666 | 4.629 | 4.63 | 0.222 / 0.187 / 0.340 |

## Notes

- `parity` = `within` (fixed odd 1-based `pmt_id_in_mpmt` hidden; 10 of 19 per WCTE/SK module, hence hidden fraction 0.526); `random` = fresh uniform half per event; `patchS` = fresh half of the contiguous Hilbert patches of S PMTs (⌈count/2⌉ for non-default S; the tail of the Hilbert order beyond count × S is never hidden).
- E_S is ≈ 0 for every configuration: the PMT identity alone carries ≤ 2 % of the target variance on these cohorts (slightly negative values are fit/score sampling noise).
- Ridge α chosen by generalised cross-validation over 10⁻³…10³; with 10⁵–10⁶ fitting rows the upper end of the grid is still weak regularisation.
