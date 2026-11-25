# r2-sparse2 SPARSE2_TABLE (WCTE lucid_v1_w2, seeds A/B/C = 9000/9001/9002)

Primary: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1 A.2), step 10,000. T = e_test / e_base − 1 (negative = test better). SE_boot: paired bootstrap of the seed-mean contrast (2,000 resamples of sealed events); SE_seed = SD_pooled/√3, SD_pooled over r2-sparse R1–R6, r2-follow A1–A4 and P1–P5 (vertex, sealed: 0.0216); N_σ = |T̄|/max(SE_boot, SE_seed); clear = N_σ ≥ 2 and one sign in all three seeds.

## Registered contrasts (primary, sealed)

| ID | Contrast | T̄ | A / B / C | SE_boot | SE_seed | N_σ | clear | prediction | verdict |
|---|---|---|---|---|---|---|---|---|---|
| P1 | S1L vs C2 | -2.9 % | -2.5 % / -2.2 % / -4.1 % | +0.4 % | +1.2 % | 2.34 | yes | S1L not worse by > 5 % | holds |
| P2 | S1L vs S1 | -37.0 % | -37.4 % / -37.0 % / -36.6 % | +0.3 % | +1.2 % | 29.70 | yes | S1L better, clear | holds |
| P3 | S2L vs S2 | -12.5 % | -12.6 % / -14.1 % / -10.8 % | +0.3 % | +1.2 % | 10.04 | yes | S2L better, clear | holds |
| P4 | S2L vs C2 | +0.2 % | -0.2 % / +1.1 % / -0.2 % | +0.4 % | +1.2 % | 0.20 | no | no sign | no sign registered (reported) |
| P5 | S2Lt vs S2L | +15.3 % | +20.4 % / +13.2 % / +12.2 % | +0.4 % | +1.2 % | 12.25 | yes | no sign | no sign registered (reported) |
| P6 | S2Lt vs S2L, f_own/chance | — | 1.24 vs 2.49 / 1.23 vs 2.54 / 1.23 vs 2.48 | — | — | — | — | S2Lt f_own/chance higher in every seed | fails |

## Cells (seed means, sealed, step 10,000)

| Cell | vertex [m] | per seed A / B / C | MLP vertex | attentive vertex | μ dir [°] | e dir [°] | μ mis-ID @99 % e | gain vs own init |
|---|---|---|---|---|---|---|---|---|
| C2 | 0.0549 | 0.0551 / 0.0546 / 0.0550 | 0.0524 | 0.0642 | 6.63 | 5.86 | 0.0543 | +18.4 % |
| S1 | 0.0846 | 0.0857 / 0.0847 / 0.0832 | 0.0697 | 0.0796 | 8.84 | 7.75 | 0.1855 | +23.1 % |
| S2 | 0.0629 | 0.0629 / 0.0642 / 0.0616 | 0.0491 | 0.0599 | 5.92 | 5.33 | 0.1671 | +29.4 % |
| S1L | 0.0533 | 0.0537 / 0.0534 / 0.0528 | 0.0519 | 0.0597 | 5.81 | 5.35 | 0.0655 | +28.1 % |
| S2L | 0.0550 | 0.0550 / 0.0552 / 0.0549 | 0.0518 | 0.0539 | 5.44 | 4.97 | 0.1509 | +0.7 % |
| S2Lt | 0.0634 | 0.0662 / 0.0624 / 0.0616 | 0.0523 | 0.0606 | 6.50 | 5.83 | 0.2913 | -14.5 % |

Gain vs own init = 1 − e(10,000)/e(0), sealed primary readout, seed mean.

## Secondary contrasts (sealed; T̄ (N_σ, clear))

| ID | MLP vertex | attentive vertex | μ dir | e dir | μ mis-ID | vertex (validation) |
|---|---|---|---|---|---|---|
| P1 | -1.1 % (0.4) | -7.0 % (6.0, clear) | -12.4 % (11.6, clear) | -8.7 % (7.8, clear) | +20.6 % (1.8) | -2.8 % (2.0, clear) |
| P2 | -25.6 % (9.9, clear) | -25.0 % (21.5, clear) | -34.3 % (31.8, clear) | -31.0 % (27.7, clear) | -64.7 % (5.5, clear) | -37.1 % (27.1, clear) |
| P3 | +5.4 % (2.1, clear) | -9.9 % (8.6, clear) | -8.1 % (7.5, clear) | -6.7 % (6.0, clear) | -9.6 % (0.8) | -13.0 % (9.5, clear) |
| P4 | -1.3 % (0.5) | -16.0 % (13.8, clear) | -18.0 % (16.7, clear) | -15.2 % (13.6, clear) | +179.3 % (7.5, clear) | +0.1 % (0.0) |
| P5 | +1.0 % (0.4) | +12.3 % (10.6, clear) | +19.6 % (18.2, clear) | +17.4 % (15.6, clear) | +94.5 % (8.1, clear) | +14.9 % (10.9, clear) |
| S2Lt vs C2 (expl.) | -0.2 % (0.1) | -5.7 % (4.9, clear) | -2.0 % (1.8) | -0.5 % (0.4) | +443.3 % (10.0, clear) | +15.0 % (10.9, clear) |
| S2Lt vs S2 (expl.) | +6.5 % (2.5, clear) | +1.1 % (1.0) | +9.9 % (9.1, clear) | +9.5 % (8.5, clear) | +74.8 % (6.4, clear) | +0.0 % (0.0) |
| S2L vs S1L (expl.) | -0.1 % (0.0) | -9.7 % (8.3, clear) | -6.3 % (5.9, clear) | -7.1 % (6.3, clear) | +132.0 % (7.3, clear) | +2.9 % (2.1, clear) |
| S1 vs C2 (expl.) | +33.0 % (12.8, clear) | +24.1 % (20.7, clear) | +33.2 % (30.9, clear) | +32.3 % (28.8, clear) | +241.7 % (9.3, clear) | +54.5 % (39.8, clear) |
| S2 vs C2 (expl.) | -6.4 % (2.5, clear) | -6.7 % (5.8, clear) | -10.8 % (10.0, clear) | -9.1 % (8.1, clear) | +208.8 % (8.3, clear) | +15.0 % (11.0, clear) |

## f_own diagnostic (dev_probe_test, step 10,000; PREREG appendix B)

| Cell | seed | f_own (own / hidden-hit mass) | chance | f_own / chance | own / total mass | mass on hidden hits | mass on unhit |
|---|---|---|---|---|---|---|---|
| S2 | A | 11.6 % | 5.7 % | 2.04 | 7.8 % | 73.7 % | 26.3 % |
| S2 | B | 11.6 % | 5.7 % | 2.05 | 7.9 % | 73.7 % | 26.3 % |
| S2 | C | 11.6 % | 5.7 % | 2.05 | 7.8 % | 73.7 % | 26.3 % |
| S2L | A | 14.1 % | 5.7 % | 2.49 | 10.0 % | 73.6 % | 26.4 % |
| S2L | B | 14.4 % | 5.7 % | 2.54 | 10.2 % | 73.5 % | 26.5 % |
| S2L | C | 14.0 % | 5.7 % | 2.48 | 10.1 % | 74.1 % | 25.9 % |
| S2Lt | A | 74.2 % | 59.9 % | 1.24 | 58.3 % | 75.9 % | 24.1 % |
| S2Lt | B | 73.7 % | 59.9 % | 1.23 | 57.7 % | 75.6 % | 24.4 % |
| S2Lt | C | 73.9 % | 59.9 % | 1.23 | 58.1 % | 75.8 % | 24.2 % |

## How much of the r2-sparse gap to C2 the handicaps explained (seed means, sealed)

- P2: S1 0.0846 m → S1L 0.0533 m (C2 0.0549 m): 105 % of the gap closed.
- P3: S2 0.0629 m → S2L 0.0550 m (C2 0.0549 m): 98 % of the gap closed.

## Decision rule (§2.6, appendix C)

- S1L: primary T̄ vs C2 -2.9 % (clear True); ≤ +2 %: True; better clear (vertex / both directions): True / True; MLP / attentive T̄ -1.1 % / -7.0 %, same sign: True → fires: **True**
- S2L: primary T̄ vs C2 +0.2 % (clear False); ≤ +2 %: True; better clear (vertex / both directions): False / True; MLP / attentive T̄ -1.3 % / -16.0 %, same sign: False → fires: **False**
- S2Lt: primary T̄ vs C2 +15.5 % (clear True); ≤ +2 %: False; better clear (vertex / both directions): False / False; MLP / attentive T̄ -0.2 % / -5.7 %, same sign: False → fires: **False**
- **Verdict: scale the sparse family to SK (follow-up runbook) with S1L.**

Everything outside P1–P6 and the decision rule is exploratory.
