# FOLLOW_TABLE — r2-follow (PREREG_2026-11-24_R2_FOLLOW.md)

Primary: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1 A.2), step 10,000. T = e_test / e_base − 1 (negative = test better). SE_boot: paired bootstrap of the seed-mean contrast (2,000 resamples of sealed events); SE_seed = SD_pooled/√3; N_σ = |T̄|/max(SE_boot, SE_seed); clear = N_σ ≥ 2 and one sign in all three seeds.

## Part A — WCTE attribution (lucid_v1_w2; seeds A/B/C = 9000/9001/9002)

| Cell | vertex [m] seed mean | A / B / C | MLP | attentive | μ dir [°] | e dir [°] | gain vs own init |
|---|---|---|---|---|---|---|---|
| C2 | 0.0549 | 0.0551 / 0.0546 / 0.0550 | 0.0524 | 0.0642 | 6.63 | 5.86 | +18.4 % |
| C3e | 0.0591 | 0.0603 / 0.0586 / 0.0585 | 0.0539 | 0.0672 | 7.02 | 6.03 | +33.4 % |
| C2dc | 0.0633 | 0.0631 / 0.0635 / 0.0632 | 0.0580 | 0.0709 | 7.41 | 6.31 | +5.9 % |
| C3 | 0.0726 | 0.0743 / 0.0733 / 0.0702 | 0.0599 | 0.0738 | 7.82 | 6.63 | +18.2 % |

| ID | Contrast | measures | T̄ | A / B / C | SE_boot | SE_seed | N_σ | clear |
|---|---|---|---|---|---|---|---|---|
| A1 | C3e vs C2 | the new embedding alone | +7.8 % | +9.6 % / +7.4 % / +6.3 % | +0.36 % | +1.23 % | 6.32 | yes |
| A2 | C2dc vs C2 | the query-isolated decoder and count head alone | +15.3 % | +14.5 % / +16.4 % / +15.0 % | +0.43 % | +1.23 % | 12.46 | yes |
| A3 | C3 vs C3e | decoder and count head, given the new embedding | +22.8 % | +23.2 % / +25.1 % / +20.0 % | +0.47 % | +1.23 % | 18.54 | yes |
| A4 | C3 vs C2dc | the new embedding, given the new decoder | +14.8 % | +17.9 % / +15.4 % / +10.9 % | +0.39 % | +1.23 % | 12.01 | yes |

R2b (C3 vs C2) recomputed with the new SD_pooled: T̄ +32.3 %, N_σ 26.31, clear True (registered +32.3 %). SD_pooled (vertex, sealed) = 0.0213 over A1, A2, A3, A4, R1, R2, R2b, R3, R4, R5, R6.

**Attribution verdict (§2.3): interaction: neither single change accounts for the C3 loss, but A3 does.**

Interaction (exploratory): I = (e_C3·e_C2)/(e_C3e·e_C2dc) − 1 = +6.5 % (A / B / C +7.6 % / +7.5 % / +4.3 %; SE_boot +0.48 %, SE_seed +1.06 %, N_σ 6.1). I = 0 means the two changes combine multiplicatively (A3 = A2, A4 = A1).

Secondary contrasts (sealed; T̄, N_σ, clear):

| ID | MLP vertex | attentive vertex | μ dir | e dir | vertex (validation) |
|---|---|---|---|---|---|
| A1 | +2.8 % (1.0) | +4.6 % (4.2, clear) | +5.8 % (6.2, clear) | +3.0 % (2.5) | +6.8 % (5.1, clear) |
| A2 | +10.6 % (3.6, clear) | +10.4 % (9.4, clear) | +11.8 % (12.5, clear) | +7.7 % (6.5, clear) | +15.4 % (11.4, clear) |
| A3 | +11.1 % (3.8, clear) | +9.9 % (8.9, clear) | +11.4 % (12.1, clear) | +9.9 % (8.3, clear) | +24.9 % (18.4, clear) |
| A4 | +3.3 % (1.1) | +4.2 % (3.7, clear) | +5.4 % (5.8, clear) | +5.0 % (4.2, clear) | +15.7 % (11.6, clear) |

§2.4 diagnostics (exploratory; seed 9000; dev_probe_test; every event fully visible):

| Cell | step | input RMS positional / content (hit tokens) | block-4 variance explained: geometry / content / joint (all tokens) | same, hit tokens | PMT-position probe R² | vertex R² (frozen readout, validation) |
|---|---|---|---|---|---|---|
| C2 | 0 | 0.528 / 0.490 (0.527 / 0.565) | 0.662 / 0.158 / 0.823 | 0.719 / 0.083 / 0.805 | 0.9967 | 0.9947 |
| C2 | 10000 | 0.569 / 0.540 (0.567 / 0.549) | 0.540 / 0.183 / 0.726 | 0.592 / 0.128 / 0.725 | 0.9969 | 0.9963 |
| C3e | 0 | 0.702 / 0.648 (0.702 / 0.819) | 0.580 / 0.211 / 0.794 | 0.621 / 0.165 / 0.790 | 0.9756 | 0.9912 |
| C3e | 10000 | 0.702 / 0.667 (0.702 / 0.761) | 0.543 / 0.196 / 0.742 | 0.606 / 0.143 / 0.752 | 0.9889 | 0.9958 |
| C2dc | 0 | 0.528 / 0.490 (0.527 / 0.565) | 0.662 / 0.158 / 0.823 | 0.719 / 0.083 / 0.805 | 0.9967 | 0.9947 |
| C2dc | 10000 | 0.502 / 0.528 (0.500 / 0.559) | 0.493 / 0.204 / 0.699 | 0.627 / 0.095 / 0.724 | 0.9941 | 0.9953 |
| C3 | 0 | 0.702 / 0.648 (0.702 / 0.819) | 0.580 / 0.211 / 0.794 | 0.621 / 0.165 / 0.790 | 0.9756 | 0.9912 |
| C3 | 10000 | 0.702 / 0.923 (0.702 / 1.040) | 0.538 / 0.168 / 0.708 | 0.650 / 0.101 / 0.751 | 0.9703 | 0.9937 |

## Part B — SK, three seeds (lucid_v1; seeds A, B, C)

| Config | U (M1) | G @10k A / B / C | G @10k mean | G @5k A / B / C | G @5k mean | T vs random @10k (N_σ, clear) | T vs random @5k (N_σ, clear) |
|---|---|---|---|---|---|---|---|
| random | 0.5486 | +21.9 % / +24.6 % / +25.7 % | +24.0 % | +17.4 % / +17.5 % / +16.8 % | +17.2 % | — | — |
| patch43 | 0.5519 | +31.4 % / +28.6 % / +31.1 % | +30.4 % | +24.8 % / +23.3 % / +22.3 % | +23.5 % | -8.3 % (5.6, clear) | -7.6 % (9.7, clear) |
| patch173 | 0.5650 | +32.8 % / +35.0 % / +34.1 % | +34.0 % | +29.3 % / +29.5 % / +28.4 % | +29.1 % | -13.1 % (8.8, clear) | -14.3 % (18.3, clear) |
| patch692 | 0.5995 | +35.0 % / +34.2 % / +35.2 % | +34.8 % | +32.5 % / +29.3 % / +30.5 % | +30.8 % | -14.1 % (9.5, clear) | -16.4 % (21.0, clear) |

| ID | Statement | Result | Holds |
|---|---|---|---|
| B1 | every patch size beats random (T̄ < 0, clear, 3 seeds) | patch43 -8.3 % (N_σ 5.6, clear True); patch173 -13.1 % (N_σ 8.8, clear True); patch692 -14.1 % (N_σ 9.5, clear True) | **True** |
| B2 | U predicts the gain (ρ ≥ 0.8; μ and e direction ρ ≥ 0.8) | ρ vertex 1.00; μ 0.80; e 0.80 | **True** |
| B3 | ordering reproduced by seed C alone (ρ ≥ 0.8) | ρ 1.00 | **True** |

**Part B verdict: SK mask claim holds with three seeds (B1 and B2 hold).** Figure: `results/round2/20261123_mask/MASK_SCALE_3seed.png`.

Everything outside A1–A4, the attribution rule and B1–B3 is secondary or exploratory.
