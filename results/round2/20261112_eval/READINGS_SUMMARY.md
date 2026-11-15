# Round 2 readings summary

Extension `r2_multidetector_2026-11-11`, `registered: false`. Gate V is informational (addendum `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md`): it failed with the round-1 absolute thresholds on SK and JUNO, so the analytic model is not validated against PhotonSim + LUCiD at these scales and every number below carries that caveat.

## Vertex error (full input, m), per arm and seed

| Detector | MAE | position-free JEPA | random init | fixed-label JEPA | MAE (RF) 9000 |
|---|---|---|---|---|---|
| WCTE (round 1) | 0.1049 | 0.1260/0.1405 | 0.1601 | 0.2193/0.2043 | – |
| SK | 0.9287/0.8264 | 1.1442/1.4193 | 1.2755/1.2572 | 2.1863/1.9727 | 0.7684 |
| JUNO | 0.6855/0.4972 | 1.0135/0.9716 | 0.9499/0.9682 | 1.3400/1.2834 | 0.5203 |
| HK | 1.5797/1.2582 | 2.1615/1.9736 | 2.0771/2.0311 | 3.7033/3.3839 | – |

## Readings R2-a to R2-e (two seeds; three-seed version where seed 9002 ran)

### SK (fiducial radius 13.57 m)
- R2-a fixed-label collapse: **event-invariant** (f_event 9000: 0.0002613, 9001: 0.0004253).
- R2-b position-free recovery: **unclear** (9000: vertex 1.1442 vs init 1.2755, token R² 0.9360, f_event 0.0270; 9001: vertex 1.4193 vs init 1.2572, token R² 0.9826, f_event 0.0367).
- R2-c MAE above floor: **yes** (9000: MAE − init -0.3467 m, CI [-0.3670, -0.3269]; 9001: MAE − init -0.4308 m, CI [-0.4506, -0.4098]).
- R2-d ranking: mae_mse < init < target_posfree < sigreg_jepa; Kendall τ vs WCTE = 0.667.
- R2-e receptive field (RF − MAE, seed 9000): vertex -0.1604 m, CI [-0.1752, -0.1462]; token R² -0.000936; f_event 0.0165; confounded by receptive field by 0.160 m.
- Floor band (5 init seeds): vertex mean 1.2425 m, range [1.2059, 1.2755]; token R² mean 0.9987.
- Three-seed version: R2-a event-invariant, R2-b unclear, R2-c yes, R2-d mae_mse < init < target_posfree < sigreg_jepa (seeds used {'R2-a': [9000, 9001], 'R2-b': [9000, 9001, 9002], 'R2-c': [9000, 9001], 'R2-d': {'mae_mse': [9000, 9001], 'target_posfree': [9000, 9001, 9002], 'init': [9000, 9001, 9002], 'sigreg_jepa': [9000, 9001]}}).
- Seed 9002: target_posfree (R2-b unclear): train.

### JUNO (fiducial radius 14.00 m)
- R2-a fixed-label collapse: **intermediate** (f_event 9000: 0.0023, 9001: 0.0038).
- R2-b position-free recovery: **not recovered** (9000: vertex 1.0135 vs init 0.9499, token R² 0.9941, f_event 0.0316; 9001: vertex 0.9716 vs init 0.9682, token R² 0.9972, f_event 0.0170).
- R2-c MAE above floor: **yes** (9000: MAE − init -0.2645 m, CI [-0.2792, -0.2510]; 9001: MAE − init -0.4710 m, CI [-0.4872, -0.4554]).
- R2-d ranking: mae_mse < init < target_posfree < sigreg_jepa; Kendall τ vs WCTE = 0.667.
- R2-e receptive field (RF − MAE, seed 9000): vertex -0.1652 m, CI [-0.1767, -0.1538]; token R² -0.0006288; f_event 0.0050; confounded by receptive field by 0.165 m.
- Floor band (5 init seeds): vertex mean 0.9298 m, range [0.8902, 0.9682]; token R² mean 0.9993.
- Seed 9002: sigreg_jepa (R2-a intermediate): train.

### HK (fiducial radius 25.94 m)
- R2-a fixed-label collapse: **event-invariant** (f_event 9000: 0.0003344, 9001: 0.000726).
- R2-b position-free recovery: **unclear** (9000: vertex 2.1615 vs init 2.0771, token R² 0.9893, f_event 0.0431; 9001: vertex 1.9736 vs init 2.0311, token R² 0.9959, f_event 0.0164).
- R2-c MAE above floor: **yes** (9000: MAE − init -0.4974 m, CI [-0.5310, -0.4661]; 9001: MAE − init -0.7729 m, CI [-0.8092, -0.7381]).
- R2-d ranking: mae_mse < init < target_posfree < sigreg_jepa; Kendall τ vs WCTE = 0.667.
- R2-e receptive field: not run.
- Floor band (5 init seeds): vertex mean 1.9953 m, range [1.9458, 2.0771]; token R² mean 0.9990.
- Seed 9002: target_posfree (R2-b unclear): train.

## Particle tasks (seed 9000)

| Detector | Arm | μ/e AUC | e direction (deg) | μ direction (deg) | e energy R² | μ energy R² |
|---|---|---|---|---|---|---|
| SK | MAE | 0.9863 | 3.8311 | 6.0294 | 0.9873 | 0.9706 |
| SK | position-free JEPA | 0.9408 | 3.6356 | 4.9783 | 0.9576 | 0.9120 |
| SK | random init | 0.9425 | 5.0982 | 6.5408 | 0.9812 | 0.9502 |
| SK | fixed-label JEPA | 0.7475 | 7.9355 | 10.8436 | 0.9424 | 0.8707 |
| SK | MAE (RF-matched) | 0.9893 | 3.1611 | 6.1516 | 0.9787 | 0.9516 |
| JUNO | MAE | 0.9949 | 3.6580 | 6.0371 | 0.9924 | 0.9843 |
| JUNO | position-free JEPA | 0.9856 | 3.2037 | 4.1692 | 0.9868 | 0.9707 |
| JUNO | random init | 0.9690 | 4.3132 | 6.3438 | 0.9858 | 0.9668 |
| JUNO | fixed-label JEPA | 0.8660 | 5.4251 | 6.8449 | 0.9503 | 0.8974 |
| JUNO | MAE (RF-matched) | 0.9973 | 3.0096 | 4.4823 | 0.9889 | 0.9750 |
| HK | MAE | 0.9595 | 4.6641 | 4.9335 | 0.9912 | 0.9786 |
| HK | position-free JEPA | 0.9534 | 4.6242 | 4.5883 | 0.9888 | 0.9719 |
| HK | random init | 0.8836 | 6.5168 | 6.2395 | 0.9899 | 0.9737 |
| HK | fixed-label JEPA | 0.6733 | 9.6213 | 11.0114 | 0.9429 | 0.8431 |

## Time-scale diagnostics (§6, logged, no threshold)

- WCTE round-1 MAE loss shares (step 100 / step 10,000 rows): charge 0.773 → 0.663, hit 0.222 → 0.335, time 0.005 → 0.002.
- WCTE round 1: time-feature std 0.0393 (scale 100 ns).
- SK: time-feature std 0.0755 (scale 450.0 ns).
  - mae_mse loss shares (first / last 100 steps): charge 0.655 → 0.576, hit 0.312 → 0.423, time 0.033 → 0.000.
    The time term's share is 7.11× (start) and 0.23× (end) the WCTE share: it moved by more than 2× relative to WCTE.
- JUNO: time-feature std 0.0856 (scale 333.3 ns).
  - mae_mse loss shares (first / last 100 steps): charge 0.669 → 0.595, hit 0.295 → 0.405, time 0.035 → 0.000.
    The time term's share is 7.55× (start) and 0.21× (end) the WCTE share: it moved by more than 2× relative to WCTE.
  - mae_mse_rf loss shares (first / last 100 steps): charge 0.671 → 0.595, hit 0.293 → 0.405, time 0.036 → 0.000.
    The time term's share is 7.70× (start) and 0.21× (end) the WCTE share: it moved by more than 2× relative to WCTE.
- HK: time-feature std 0.0834 (scale 766.7 ns).
  - mae_mse loss shares (first / last 100 steps): charge 0.599 → 0.519, hit 0.352 → 0.480, time 0.050 → 0.001.
    The time term's share is 10.67× (start) and 0.46× (end) the WCTE share: it moved by more than 2× relative to WCTE.

## Gate V (informational) and the photon-statistics ceiling

| Detector | Module median | Log-charge median | Absolute gate | Module ceiling C | Log ceiling C | ρ̃ module | ρ̃ log |
|---|---|---|---|---|---|---|---|
| WCTE | 0.9015 | 0.8150 | passed | 0.9993 | 0.9789 | 0.9022 | 0.8326 |
| SK | 0.8288 | 0.6924 | failed | 0.9972 | 0.9113 | 0.8311 | 0.7607 |
| JUNO | 0.8882 | 0.7609 | failed | 0.9989 | 0.9709 | 0.8891 | 0.7838 |
| HK | 0.7565 | 0.5634 | failed | – | – | – | – |
