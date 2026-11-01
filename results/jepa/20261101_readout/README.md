# Q1/Q2 readout matrix and nonlinear probes — 1 Nov 2026

`registered: false`. Extension `readout_diagnostic_2026-11-01`.
Runbook: `docs/jepa/RUNBOOK_2026-11-01_READOUT.md` (revision 2, sha256 `0e629141fc90d24cfa5bf3b0d1ac2ac271e7c94738eca68c678393032ee58c83`).
Evaluation only: no encoder was trained, no grid run; photon domain only.

- final code SHA: `02283a632f0926578044539eec31bcbbb17c937e`
- §4.2 reproduction: PASS (tolerance 0.001 m)

## Checkpoint hashes used (§3.3)

| item | arm | split | seed | source sha256 | copy sha256 |
|---|---|---|---:|---|---|
| init9000 | sigreg_jepa | within | 9000 | `n/a` | `n/a` |
| old9000 | sigreg_jepa | within | 9000 | `b1632266f2f8ca9fd49f9a06383734fda77432f5247294e79ed65bc1936f7503` | `b1632266f2f8ca9fd49f9a06383734fda77432f5247294e79ed65bc1936f7503` |
| old9001 | sigreg_jepa | within | 9001 | `14e3590f9ed6649d86f3de751a56c3590bb748f0b4a15984bb056373bff97ceb` | `14e3590f9ed6649d86f3de751a56c3590bb748f0b4a15984bb056373bff97ceb` |
| fix9000 | sigreg_jepa | within | 9000 | `2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714` | `2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714` |
| fix9001 | sigreg_jepa | within | 9001 | `95cb5e3f4f158a719098a4b3dc573eebb8c71645c22290f6483223ccc04c484e` | `95cb5e3f4f158a719098a4b3dc573eebb8c71645c22290f6483223ccc04c484e` |
| mae9000 | mae_mse | within | 9000 | `efd54c2586f8b5378dfe2febffe2ceb954c50185f6b07939c2b892c825270664` | `efd54c2586f8b5378dfe2febffe2ceb954c50185f6b07939c2b892c825270664` |
| across9000 | sigreg_jepa | across | 9000 | `ac332d4b602d28fe2fdc2b4c1094058eed33538be4f0a9bc5c18503eac73bd5f` | `ac332d4b602d28fe2fdc2b4c1094058eed33538be4f0a9bc5c18503eac73bd5f` |
| across9001 | sigreg_jepa | across | 9001 | `749e87376123ff629e8f2bf7f02f012f3ac965c181266b4710ce687c8055651a` | `749e87376123ff629e8f2bf7f02f012f3ac965c181266b4710ce687c8055651a` |

## Readout matrix (photon domain, Part-C-comparable protocol)

| item | form | vertex m | token ln q R2 | eff. rank | part. ratio | event SIGReg | module gain R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| init9000 | full_input | 0.160072 | 0.998299 | 1.53028 | 1.2154 | 2326.76 | 0.200498 |
| init9000 | propagated | 0.206089 | 0.52672 | 1.75896 | 1.32388 | 4633.17 | -0.000687549 |
| init9000 | context_only | 0.210418 | 0.998621 | 1.62 | 1.24954 | 4488.23 | n/a |
| init9000 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| init9000 | half_pass | 0.174821 | 0.998395 | 1.61262 | 1.25018 | 4347.02 | 0.203756 |
| old9000 | full_input | 0.2644 | 0.00406609 | 10.2725 | 5.14987 | 1367.16 | -0.00443271 |
| old9000 | propagated | 0.189382 | 0.469657 | 4.02515 | 2.52559 | 1374.22 | -0.00258125 |
| old9000 | context_only | 0.174225 | 0.655824 | 3.43151 | 2.30464 | 865.46 | n/a |
| old9000 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| old9000 | half_pass | 0.14002 | 0.647803 | 3.38647 | 2.27548 | 878.285 | 0.00183205 |
| old9001 | full_input | 0.233759 | 0.00349151 | 9.20747 | 5.00983 | 1367.17 | -0.00424558 |
| old9001 | propagated | 0.18544 | 0.474941 | 4.26269 | 2.59118 | 1344.58 | -0.00233865 |
| old9001 | context_only | 0.165341 | 0.776586 | 3.60439 | 2.45734 | 702.244 | n/a |
| old9001 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| old9001 | half_pass | 0.130648 | 0.772495 | 3.47527 | 2.35554 | 714.49 | 0.00915159 |
| fix9000 | full_input | 0.219259 | 0.0171499 | 2.49264 | 1.46705 | 1424.54 | -0.00208715 |
| fix9000 | propagated | 0.268534 | 0.446304 | 4.13646 | 2.01743 | 1680.5 | -0.00382932 |
| fix9000 | context_only | 0.36669 | 0.00137734 | 15.367 | 4.37719 | 1367.13 | n/a |
| fix9000 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| fix9000 | half_pass | 0.325457 | 0.000445805 | 12.497 | 3.83403 | 1367.02 | 0.000412327 |
| fix9001 | full_input | 0.204281 | 0.051402 | 2.80461 | 1.61896 | 1408.6 | 0.00550313 |
| fix9001 | propagated | 0.250452 | 0.467436 | 3.04423 | 1.67069 | 1690.15 | -0.00367344 |
| fix9001 | context_only | 0.326661 | 0.00198786 | 8.1452 | 3.05843 | 1367.18 | n/a |
| fix9001 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| fix9001 | half_pass | 0.293429 | 0.00165443 | 7.79002 | 3.20346 | 1367.05 | -0.00380496 |
| mae9000 | full_input | 0.104949 | 0.98785 | 2.91085 | 1.85214 | 1651.04 | 0.18019 |
| mae9000 | propagated | 0.120679 | 0.510082 | 7.05326 | 5.11454 | 2577.04 | 0.000296698 |
| mae9000 | context_only | 0.105943 | 0.983396 | 4.47398 | 2.8199 | 1367.34 | n/a |
| mae9000 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| mae9000 | half_pass | 0.0832159 | 0.983219 | 4.28399 | 2.69787 | 1382.26 | 0.197239 |
| across9000 | full_input | 0.348113 | 0.000914092 | 14.4449 | 4.57463 | 1370.27 | -0.00255096 |
| across9000 | propagated | 0.973556 | -0.000807598 | 3.34326 | 1.90571 | 1656.06 | -0.00333658 |
| across9000 | context_only | 0.891314 | 0.00115573 | 28.2612 | 14.7683 | 1366.06 | n/a |
| across9000 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| across9000 | half_pass | n/a | n/a | n/a | n/a | n/a | n/a |
| across9001 | full_input | 0.333464 | 0.00223358 | 8.99976 | 2.87166 | 1371.08 | 0.00390762 |
| across9001 | propagated | 0.959699 | -0.000947424 | 2.56827 | 1.52618 | 1736.71 | -0.00300209 |
| across9001 | context_only | 0.873362 | 0.00181081 | 33.5206 | 20.9734 | 1366.11 | n/a |
| across9001 | raw_input | 1.06125 | n/a | 1.15352 | 1.06464 | 1350.67 | n/a |
| across9001 | half_pass | n/a | n/a | n/a | n/a | n/a | n/a |

`n/a` cells and their reasons are in `matrix.json`; module gain is only defined for `full_input`/`half_pass` (`module_gain::run_mean`) and `propagated` (`module_gain::propagated`), §2.

## §4.2 reproduction

| item | recorded Part C m | recomputed m | |diff| m | pass |
|---|---:|---:|---:|---|
| old9000 | 0.264401 | 0.2644 | 7.08578e-07 | True |
| old9001 | 0.233759 | 0.233759 | 3.96314e-07 | True |
| fix9000 | 0.219258 | 0.219259 | 5.91047e-07 | True |
| init9000 | 0.160072 | 0.160072 | 3.09078e-07 | True |
| fix9001 | 0.204281 | 0.204281 | 2.336e-07 | True |
| mae9000 | 0.104949 | 0.104949 | 1.5326e-07 | True |

## §6 pre-declared readings

- **Q1-a** — not supported (trained-vs-random verdict is readout-dependent).
- **Q1-b** — supported (half-pass label-version difference is not seed noise).
- **Q1-c** — supported (propagated carries the same information as half_pass).
- **Q1-e** — descriptive (deterministic-layout signal (H1)): {"init9000": 0.8283663520347606, "old9000": 0.9960865117048144, "old9001": 0.9908185626652362, "fix9000": 0.9969886012742276, "fix9001": 0.9937337959586334, "mae9000": 0.882864706557843, "across9000": 1.00034976858557, "across9001": 1.0003623434999447}.
- **Q1-f** — descriptive (event-dependence signal (H2)): {"init9000": 0.07897473178893168, "old9000": 2.7565425395691117e-06, "old9001": 7.260840665402809e-06, "fix9000": 8.016776171518009e-05, "fix9001": 0.00016481596163020083, "mae9000": 0.0454750718853029, "across9000": 4.670153367344038e-07, "across9001": 5.026418045314233e-07}.
- **Q2-a** — not supported (charge is destroyed, not just unreadable).
- **Q2-b** — supported (charge is present but nonlinearly encoded).
- **Q2-c** — not supported (vertex information is destroyed).
- **Q2-d** — unclear (trained representation closer to physics than to a random projection).

## Q2 probe table

| item | form | ridge vertex m | MLP vertex m | ridge token R2 | MLP token R2 |
|---|---|---:|---:|---:|---:|
| init9000 | full_input | 0.160072 | 0.129361 | 0.998299 | 0.999776 |
| init9000 | half_pass | 0.174821 | 0.123493 | 0.998395 | 0.999859 |
| old9000 | full_input | 0.2644 | 0.173724 | 0.00406625 | 0.511619 |
| old9000 | half_pass | 0.14002 | 0.163026 | 0.647798 | 0.925384 |
| old9001 | full_input | 0.233759 | 0.174527 | 0.00349204 | 0.317434 |
| old9001 | half_pass | 0.130648 | 0.155122 | 0.772502 | 0.967032 |
| fix9000 | full_input | 0.219259 | 0.137596 | 0.0171504 | 0.814893 |
| fix9000 | half_pass | 0.325457 | 0.199177 | 0.000444906 | 0.000125364 |
| fix9001 | full_input | 0.204281 | 0.11833 | 0.0514011 | 0.921727 |
| fix9001 | half_pass | 0.293429 | 0.172112 | 0.00165422 | 0.577879 |
| mae9000 | full_input | 0.104949 | 0.0682477 | 0.98785 | 0.999317 |
| mae9000 | half_pass | 0.0832159 | 0.0686506 | 0.983219 | 0.999107 |

## CKA (full_input event-level means, same probe-test rows)

- `raw_physics_summary` = [vertex x, y, z, log yield] per event
- `init9000_vs_raw_physics` = 0.8676
- `old9000_vs_init9000` = 0.563786
- `old9000_vs_raw_physics` = 0.566094
- `old9001_vs_init9000` = 0.623647
- `old9001_vs_raw_physics` = 0.637683
- `fix9000_vs_init9000` = 0.902775
- `fix9000_vs_raw_physics` = 0.876119
- `fix9001_vs_init9000` = 0.962792
- `fix9001_vs_raw_physics` = 0.915001
- `mae9000_vs_init9000` = 0.97365
- `mae9000_vs_raw_physics` = 0.898492

## Outputs

- `matrix.json` — merged matrix_a/matrix_b, every form cell with value or explicit n/a + reason.
- `nonlinear.json` — Q2-a/b/c ridge-vs-MLP on identical cached rows and the Q2-d CKA rows.
- `reproduction.json` — the two §4.2 reproduction blocks.
- `readings.json` — the §6 readings with their numbers (machine-readable).
- `README.md` — this file.


## Wording fixed for the paper (Lead, 1 Nov 2026)

- **Q1-a is "not supported":** the trained-vs-random verdict is **worse in both readouts** — full-input vertex 1.37×/1.28× the init floor and half-pass vertex 1.86×/1.68× the half-pass init floor — so it does not flip under the training-like readout.
- **Q2-b is "supported": "linearly unreadable, not destroyed".** On the identical sampled hit-token rows, `fix9000` full-input charge ridge R² is 0.0172 (0.0514 for `fix9001`) while the MLP reaches 0.8149 (0.9217); the `init9000` ridge is 0.9983. The lost charge information is therefore still recoverable by a small nonlinear probe, not destroyed.
- Q1-e/Q1-f descriptive priors: `f_layout` ≈ 0.99 for old/fix (init 0.83) and `f_event` ≈ 1e-5 for old/fix (init 0.079) — a layout-dominated, event-invariant `fix` representation, exactly the H1/H2 signature that the causal E2/E3 tests now probe.
