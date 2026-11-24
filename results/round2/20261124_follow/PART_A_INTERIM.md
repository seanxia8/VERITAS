# Part A interim (24 Nov 2026, 19:35 UTC) — superseded by FOLLOW_TABLE.md

All 6 new WCTE trainings and 12 scorings complete; C2/C3 from the r2-sparse eval tree. Built by `scripts/follow_table.py` (Part B of that run used placeholder inputs and is omitted).

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

