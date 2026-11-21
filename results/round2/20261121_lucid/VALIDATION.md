# Validation gates (r2-lucid L4)

Pilot: the first 10 runs (500 events, 1:1:1 photon/muon/electron) of each detector's `dev_probe_train` cohort, generated with the production code (`scripts/lucid_generate.py`, LUCiD `e651e836`, tier A1). Validation code: `scripts/lucid_validate.py`; raw outputs under `/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/validation/<det>/`.

| Gate | WCTE | SK | JUNO | HK |
|---|---|---|---|---|
| G1 loading + 2-step MAE/JEPA dry run | PASS | PASS | PASS | PASS |
| G2 photon statistics | PASS | PASS | PASS | PASS |
| G3 LUCiD self-consistency | PASS | PASS | PASS | PASS |
| G4 regime and trigger | PASS | PASS | PASS | PASS |
| G5 displays and ToF fit | PASS | PASS | PASS | PASS |
| G6 no toy leakage | PASS | PASS | PASS | PASS |

## G2 photon statistics (photon-bomb events of the pilot)

| Detector | Events | Slope ratio free/origin | R² | Residual peak (ns) | Core σ (ns) | LUCiD jitter σ at 1 PE (ns) | First hits > 10 ns late |
|---|---:|---:|---:|---:|---:|---:|---:|
| WCTE | 170 | 1.0019 | 0.9974 | +0.3 | 1.10 | 0.95 | 1.5 % |
| SK | 170 | 1.0009 | 0.9968 | -0.9 | 2.21 | 3.49 | 11.6 % |
| JUNO | 170 | 1.0002 | 0.9954 | -0.3 | 1.36 | 0.95 | 17.2 % |
| HK | 170 | 1.0011 | 0.9956 | -0.3 | 1.44 | 0.95 | 20.6 % |

Residual = first-hit digit time − d·n/c − emission offset, with n = 1.33 as a straight-line reference (LUCiD's group velocity is wavelength-dependent, so the peak may sit slightly off zero). LUCiD's `response.tts` is 0 in these configs; the PMT timing spread comes from the digitizer's charge-dependent jitter, whose 1-PE σ is listed. Late light (> 10 ns) is present, as required with scattering and reflection enabled.

## G3 LUCiD self-consistency (sampled kernel vs LUCiD expected mode, 3 positions)

| Detector | Position (m) | Pearson (200 ev) | Ratio (200 ev) | Pearson (2000 ev) | Poisson ceiling (2000) | χ²/PMT (2000) | Ratio (2000) | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WCTE | (0.00, 0.00, 0.00) | 0.928 | 0.9993 | 0.967 | 0.996 | 0.10 | 1.0000 | PASS |
| WCTE | (0.63, 0.00, 0.00) | 0.987 | 0.9986 | 0.994 | 0.999 | 0.12 | 1.0004 | PASS |
| WCTE | (0.00, 1.14, 0.00) | 0.998 | 0.9968 | 0.999 | 1.000 | 0.12 | 0.9969 | PASS |
| SK | (0.00, 0.00, 0.00) | 0.742 | 1.0006 | 0.843 | 0.987 | 0.16 | 0.9993 | PASS |
| SK | (6.78, 0.00, 0.00) | 0.940 | 1.0024 | 0.967 | 0.997 | 0.16 | 0.9995 | PASS |
| SK | (0.00, 12.21, 0.00) | 0.991 | 1.0007 | 0.995 | 1.000 | 0.17 | 1.0003 | PASS |
| JUNO | (0.00, 0.00, 0.00) | 0.001 | 1.0005 | 0.008 | 0.924 | 0.05 | 0.9990 | PASS |
| JUNO | (7.00, 0.00, 0.00) | 0.940 | 1.0014 | 0.975 | 0.996 | 0.05 | 1.0004 | PASS |
| JUNO | (0.00, 12.60, 0.00) | 0.992 | 0.9978 | 0.997 | 0.999 | 0.05 | 0.9977 | PASS |
| HK | (0.00, 0.00, 0.00) | 0.530 | 0.9917 | 0.638 | 0.988 | 0.32 | 0.9932 | PASS |
| HK | (12.97, 0.00, 0.00) | 0.857 | 0.9989 | 0.897 | 0.996 | 0.37 | 0.9979 | PASS |
| HK | (0.00, 23.35, 0.00) | 0.977 | 0.9953 | 0.983 | 0.999 | 0.46 | 0.9973 | PASS |

Criterion: total ratio within ±5 % and Pearson ≥ 0.95; where the 200-event Poisson ceiling is below 0.95 (flat expected pattern near the centre), the 2000-event re-run decides (Pearson ≥ 0.95, or a noise-aware per-PMT χ²/dof ≤ 1.2 that includes the expected-mode Monte Carlo noise). Decision in `docs/round2/DECISIONS.md`.

## G4 regime and trigger

| Detector | Detected PE q5/q50/q95 photon bombs | all domains | Hit PMTs q5/q50/q95 (bombs) | NHits eff. above y5 | LUCiD trigger eff. above y5 |
|---|---|---|---|---:|---:|
| WCTE | 195 / 2,004 / 27,983 | 303 / 3,046 / 17,671 | 157 / 956 / 1,991 | 1.000 | 1.000 |
| SK | 208 / 2,194 / 29,429 | 334 / 5,158 / 20,275 | 192 / 1,462 / 7,867 | 1.000 | 1.000 |
| JUNO | 198 / 2,170 / 29,084 | 334 / 3,082 / 19,587 | 208 / 1,668 / 8,059 | 1.000 | 1.000 |
| HK | 200 / 2,183 / 29,549 | 353 / 4,169 / 19,568 | 251 / 1,865 / 11,299 | 1.000 | 1.000 |

Targets 200 / 3,000 / 30,000 detected PE (photon bombs); every pilot quantile is within a factor 2. No yield-range change was needed. LUCiD's own trigger exists only for SK and HK (WCTE/JUNO: no LUCiD trigger, efficiency 1 by construction).

## G1 loading and dry runs, G5 displays and ToF fit

| Detector | Pilot pack sha256 | MAE (repaired) 2 steps | JEPA (repaired) 2 steps | ToF-only Huber fit: mean / median 3-D error (m) |
|---|---|---|---|---|
| WCTE | `bd9b216523dd904b…` | complete | complete | 0.141 / 0.138 |
| SK | `a072b9920e43998d…` | complete | complete | 0.227 / 0.211 |
| JUNO | `939b4a643f800391…` | complete | complete | 1.410 / 0.295 |
| HK | `13294dc6381f4f7c…` | complete | complete | 0.299 / 0.277 |

The ToF fit is an independent simple estimator (Huber least squares on token first-hit times, straight-line n = 1.33); it is reported, not gated, and is not a likelihood ceiling.

G6: the generation path imports only the allowed `lucid_bombs` helpers (geometry, keyed RNG, run specs, detector state, sparse storage) and never imports `r2_generator`; a run-time test booby-traps every toy physics function and generates a photon, a muon and an electron event through LUCiD (`src/jepa_noise/tests/test_lucid_mc_generator.py`).

Figures: [WCTE displays](figures/validation/event_displays_wcte.png), [WCTE G2](figures/validation/g2_wcte.png), [SK displays](figures/validation/event_displays_sk.png), [SK G2](figures/validation/g2_sk.png), [JUNO displays](figures/validation/event_displays_juno.png), [JUNO G2](figures/validation/g2_juno.png), [HK displays](figures/validation/event_displays_hk.png), [HK G2](figures/validation/g2_hk.png)
