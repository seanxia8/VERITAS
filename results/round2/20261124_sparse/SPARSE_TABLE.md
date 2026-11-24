# r2-sparse SPARSE_TABLE (WCTE, seeds A/B/C = 9000/9001/9002)

Primary: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1 A.2), each model on the sealed pack of its own data kind. T = e_test / e_base − 1 (negative = test better). SE_boot: paired bootstrap of the seed-mean contrast (2,000 resamples of sealed events, shared by all runs); SE_seed = SD_pooled/√3 (SD_pooled per metric over R1–R6); N_σ = |T|/max(SE_boot, SE_seed); clear = N_σ ≥ 2 and one sign in all three seeds.

## Registered contrasts (primary, sealed)

| ID | Contrast | T̄ | A / B / C | SE_boot | SE_seed | N_σ | clear | prediction | verdict |
|---|---|---|---|---|---|---|---|---|---|
| R1 | C1 vs C0 | +1.4 % | +1.8 % / +0.2 % / +2.1 % | +0.2 % | +1.1 % | 1.20 | no | none | no prediction (reported) |
| R2 | C2 vs C1 | +2.3 % | +1.5 % / +3.8 % / +1.8 % | +0.3 % | +1.1 % | 2.08 | yes | ≤ +2 % | fails |
| R2b | C3 vs C2 | +32.3 % | +35.0 % / +34.4 % / +27.5 % | +0.6 % | +1.1 % | 28.65 | yes | ≤ +2 % | fails |
| R3 | S1 vs C3 | +16.5 % | +15.3 % / +15.5 % / +18.6 % | +0.4 % | +1.1 % | 14.61 | yes | ≤ +2 % | fails |
| R4 | S1n vs S1 | -0.2 % | -0.2 % / -0.3 % / -0.1 % | +0.1 % | +1.1 % | 0.17 | no | within ±3 % | holds |
| R5 | S1p vs S1 | +2.5 % | +2.1 % / +2.1 % / +3.1 % | +0.2 % | +1.1 % | 2.18 | yes | S1p worse, clear | holds |
| R6 | S2 vs C3 | -13.4 % | -15.4 % / -12.4 % / -12.3 % | +0.4 % | +1.1 % | 11.84 | yes | no sign | no prediction (reported) |

## Cells (seed means, sealed)

| Cell | vertex [m] | per seed A / B / C | MLP vertex | attentive vertex | μ dir [°] | e dir [°] | μ mis-ID @99 % e | vertex, P = all hits (expl.) | gain vs own init |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.0529 | 0.0533 / 0.0525 / 0.0529 | 0.0514 | 0.0638 | 6.53 | 5.79 | 0.0572 | 0.0529 | -21.6 % |
| C1 | 0.0536 | 0.0543 / 0.0526 / 0.0540 | 0.0490 | 0.0641 | 6.50 | 5.78 | 0.0535 | 0.0541 | -22.9 % |
| C2 | 0.0549 | 0.0551 / 0.0546 / 0.0550 | 0.0524 | 0.0642 | 6.63 | 5.86 | 0.0543 | 0.0553 | -18.4 % |
| C3 | 0.0726 | 0.0743 / 0.0733 / 0.0702 | 0.0599 | 0.0738 | 7.82 | 6.63 | 0.1037 | 0.0728 | -18.2 % |
| S1 | 0.0846 | 0.0857 / 0.0847 / 0.0832 | 0.0697 | 0.0796 | 8.84 | 7.75 | 0.1855 | 0.0846 | -23.1 % |
| S1n | 0.0844 | 0.0855 / 0.0845 / 0.0831 | 0.0689 | 0.0795 | 8.80 | 7.71 | 0.1905 | 0.0844 | -23.2 % |
| S1p | 0.0866 | 0.0875 / 0.0865 / 0.0858 | 0.0719 | 0.0816 | 9.20 | 8.02 | 0.2174 | 0.0867 | -21.2 % |
| S2 | 0.0629 | 0.0629 / 0.0642 / 0.0616 | 0.0491 | 0.0599 | 5.92 | 5.33 | 0.1671 | 0.0633 | -29.4 % |

Raw-feature baselines (spatial-moment Ridge on the 3 raw features, sealed): lucid_v1 0.0753 m; lucid_v1_w2 0.0852 m

## Secondary contrasts (sealed; T̄, N_σ, clear)

| ID | MLP vertex | attentive vertex | μ dir | e dir | μ mis-ID | vertex P = all hits (expl.) | vertex (validation) |
|---|---|---|---|---|---|---|---|
| R1 | -3.9 % (1.3) | +0.4 % (0.4) | -0.4 % (0.4) | -0.1 % (0.1) | -6.3 % (0.8) | +2.2 % (2.1, clear) | +0.7 % (0.6) |
| R2 | +7.2 % (2.3, clear) | +0.2 % (0.2) | +2.0 % (2.4, clear) | +1.4 % (1.5) | +2.1 % (0.3) | +2.3 % (2.3, clear) | +3.6 % (3.0, clear) |
| R2b | +14.2 % (4.5, clear) | +15.0 % (14.4, clear) | +17.9 % (21.1, clear) | +13.1 % (13.7, clear) | +90.1 % (6.7, clear) | +31.6 % (31.1, clear) | +33.4 % (28.0, clear) |
| R3 | +16.5 % (5.3, clear) | +7.9 % (7.5, clear) | +13.1 % (15.5, clear) | +17.0 % (17.7, clear) | +80.3 % (6.7, clear) | +16.2 % (15.9, clear) | +15.8 % (13.2, clear) |
| R4 | -1.2 % (0.4) | -0.2 % (0.2) | -0.4 % (0.5) | -0.5 % (0.5) | +3.1 % (0.4) | -0.2 % (0.2) | +0.1 % (0.1) |
| R5 | +3.1 % (1.0) | +2.4 % (2.3, clear) | +4.1 % (4.9, clear) | +3.5 % (3.6, clear) | +17.6 % (2.2, clear) | +2.5 % (2.5, clear) | +3.4 % (2.9, clear) |
| R6 | -17.9 % (5.7, clear) | -18.9 % (18.0, clear) | -24.3 % (28.8, clear) | -19.6 % (20.4, clear) | +63.2 % (5.2, clear) | -13.1 % (12.9, clear) | -13.8 % (11.5, clear) |
| S1n_vs_C3 (expl.) | +15.1 % (4.8, clear) | +7.6 % (7.3, clear) | +12.6 % (14.9, clear) | +16.4 % (17.1, clear) | +86.5 % (6.6, clear) | +16.0 % (15.7, clear) | +15.9 % (13.3, clear) |
| S1p_vs_C3 (expl.) | +20.2 % (6.4, clear) | +10.5 % (10.0, clear) | +17.7 % (21.0, clear) | +21.0 % (21.9, clear) | +112.8 % (8.0, clear) | +19.1 % (18.8, clear) | +19.8 % (16.6, clear) |
| C3_vs_C1 (expl.) | +22.3 % (7.1, clear) | +15.3 % (14.6, clear) | +20.3 % (24.0, clear) | +14.7 % (15.3, clear) | +95.0 % (6.3, clear) | +34.7 % (34.1, clear) | +38.3 % (32.0, clear) |
| S1_vs_C1 (expl.) | +42.5 % (13.6, clear) | +24.3 % (23.2, clear) | +36.0 % (37.7, clear) | +34.2 % (35.6, clear) | +248.3 % (8.4, clear) | +56.5 % (55.5, clear) | +60.1 % (46.5, clear) |
| S2_vs_S1 (expl.) | -29.6 % (9.4, clear) | -24.8 % (23.7, clear) | -33.0 % (39.1, clear) | -31.3 % (32.6, clear) | -9.7 % (1.2) | -25.2 % (24.8, clear) | -25.6 % (21.4, clear) |

## Decision rule (§2.6)

- S1: better than C3 by ≥ 5 % and clear on vertex / μ dir / e dir = False / False / False; MLP and attentive same sign: False; seed-mean vertex 0.0846 m vs best dense 0.0536 m → fires: **False**
- S1n: better than C3 by ≥ 5 % and clear on vertex / μ dir / e dir = False / False / False; MLP and attentive same sign: False; seed-mean vertex 0.0844 m vs best dense 0.0536 m → fires: **False**
- S2: better than C3 by ≥ 5 % and clear on vertex / μ dir / e dir = True / True / True; MLP and attentive same sign: True; seed-mean vertex 0.0629 m vs best dense 0.0536 m → fires: **False**
- Keep the fixed loss as default (R2 holds): **False**; keep the new embedding (R2b holds): **False**.
- **Verdict: report and stop: the sparse family is a negative result on WCTE (no S arm passes the rule).**

Everything outside R1–R6 and the decision rule is exploratory.
