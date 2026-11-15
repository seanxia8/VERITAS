# Round 2 readings summary

Extension `r2_multidetector_2026-11-11`, `registered: false`.

**Gate V caveat.** Gate V (round-1 absolute thresholds: module-pooled Pearson ≥ 0.85, PMT log-charge Pearson ≥ 0.70) failed on SK, JUNO and HK (it passed on WCTE in round 1). The analytic model is therefore not validated against PhotonSim + LUCiD at these detector scales; Gate V was made informational by the author's decision of 12 Nov 2026, 10:08 (addendum `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md`), and every number below carries this caveat.

## Vertex error (full input, m), per arm and seed

| Detector | MAE | position-free JEPA | random init | fixed-label JEPA | MAE (RF) 9000 |
|---|---|---|---|---|---|
| WCTE (round 1) | 0.1049 | 0.1260/0.1405 | 0.1601 | 0.2193/0.2043 | – |
| SK | 0.9287/0.8264 | 1.1442/1.4193 | 1.2755/1.2572 | 2.1863/1.9727 | 0.7684 |
| JUNO | 0.6855/0.4972 | 1.0135/0.9716 | 0.9499/0.9682 | 1.3400/1.2834 | 0.5203 |
| HK | 1.5797/1.2582 | 2.1615/1.9736 | 2.0771/2.0311 | 3.7033/3.3839 | 1.2492 |

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
- Three-seed version: R2-a intermediate, R2-b not recovered, R2-c yes, R2-d mae_mse < init < target_posfree < sigreg_jepa (seeds used {'R2-a': [9000, 9001, 9002], 'R2-b': [9000, 9001], 'R2-c': [9000, 9001], 'R2-d': {'mae_mse': [9000, 9001], 'target_posfree': [9000, 9001], 'init': [9000, 9001, 9002], 'sigreg_jepa': [9000, 9001, 9002]}}).
- Seed 9002: sigreg_jepa (R2-a intermediate): train.

### HK (fiducial radius 25.94 m)
- R2-a fixed-label collapse: **event-invariant** (f_event 9000: 0.0003344, 9001: 0.000726).
- R2-b position-free recovery: **unclear** (9000: vertex 2.1615 vs init 2.0771, token R² 0.9893, f_event 0.0431; 9001: vertex 1.9736 vs init 2.0311, token R² 0.9959, f_event 0.0164).
- R2-c MAE above floor: **yes** (9000: MAE − init -0.4974 m, CI [-0.5310, -0.4661]; 9001: MAE − init -0.7729 m, CI [-0.8092, -0.7381]).
- R2-d ranking: mae_mse < init < target_posfree < sigreg_jepa; Kendall τ vs WCTE = 0.667.
- R2-e receptive field (RF − MAE, seed 9000): vertex -0.3305 m, CI [-0.3549, -0.3046]; token R² -0.0005622; f_event 0.0072; confounded by receptive field by 0.330 m.
- Floor band (5 init seeds): vertex mean 1.9953 m, range [1.9458, 2.0771]; token R² mean 0.9990.
- Three-seed version: R2-a event-invariant, R2-b unclear, R2-c yes, R2-d mae_mse < init < target_posfree < sigreg_jepa (seeds used {'R2-a': [9000, 9001], 'R2-b': [9000, 9001, 9002], 'R2-c': [9000, 9001], 'R2-d': {'mae_mse': [9000, 9001], 'target_posfree': [9000, 9001, 9002], 'init': [9000, 9001, 9002], 'sigreg_jepa': [9000, 9001]}}).
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
| HK | MAE (RF-matched) | 0.9492 | 3.9258 | 4.6670 | 0.9891 | 0.9725 |

## Time-scale diagnostics (§6, logged, no threshold)

- **2× rule.** On every detector and both MAE variants the time term's share of the loss moved by more than 2× relative to WCTE: about 7–11× the WCTE share at the start (0.033–0.051 vs 0.0047) and about 0.2–0.5× at the end (0.0004–0.0008 vs 0.0018). The time-scale rule of §2.2 was kept as fixed; this is logged, not acted on.
- WCTE round-1 MAE loss shares (step 100 / step 10,000 rows): charge 0.773 → 0.663, hit 0.222 → 0.335, time 0.005 → 0.002.
- WCTE round 1: time-feature std 0.0393 (scale 100 ns).
- SK: time-feature std 0.0755 (scale 450.0 ns).
  - mae_mse loss shares (first / last 100 steps): charge 0.655 → 0.576, hit 0.312 → 0.423, time 0.033 → 0.000.
    The time term's share is 7.11× (start) and 0.23× (end) the WCTE share: it moved by more than 2× relative to WCTE.
  - mae_mse_rf loss shares (first / last 100 steps): charge 0.656 → 0.576, hit 0.310 → 0.424, time 0.033 → 0.000.
    The time term's share is 7.18× (start) and 0.23× (end) the WCTE share: it moved by more than 2× relative to WCTE. (Added in r2-report from the A100 run log; see `readings.json` `loss_shares_note`.)
- JUNO: time-feature std 0.0856 (scale 333.3 ns).
  - mae_mse loss shares (first / last 100 steps): charge 0.669 → 0.595, hit 0.295 → 0.405, time 0.035 → 0.000.
    The time term's share is 7.55× (start) and 0.21× (end) the WCTE share: it moved by more than 2× relative to WCTE.
  - mae_mse_rf loss shares (first / last 100 steps): charge 0.671 → 0.595, hit 0.293 → 0.405, time 0.036 → 0.000.
    The time term's share is 7.70× (start) and 0.21× (end) the WCTE share: it moved by more than 2× relative to WCTE.
- HK: time-feature std 0.0834 (scale 766.7 ns).
  - mae_mse loss shares (first / last 100 steps): charge 0.599 → 0.519, hit 0.352 → 0.480, time 0.050 → 0.001.
    The time term's share is 10.67× (start) and 0.46× (end) the WCTE share: it moved by more than 2× relative to WCTE.
  - mae_mse_rf loss shares (first / last 100 steps): charge 0.601 → 0.519, hit 0.347 → 0.480, time 0.051 → 0.001.
    The time term's share is 11.04× (start) and 0.47× (end) the WCTE share: it moved by more than 2× relative to WCTE.

## Gate V (informational) and the photon-statistics ceiling

| Detector | Module median | Log-charge median | Absolute gate | Module ceiling C | Log ceiling C | ρ̃ module | ρ̃ log |
|---|---|---|---|---|---|---|---|
| WCTE | 0.9015 | 0.8150 | passed | 0.9993 | 0.9789 | 0.9022 | 0.8326 |
| SK | 0.8288 | 0.6924 | failed | 0.9972 | 0.9113 | 0.8311 | 0.7607 |
| JUNO | 0.8882 | 0.7609 | failed | 0.9989 | 0.9709 | 0.8891 | 0.7838 |
| HK | 0.7565 | 0.5634 | failed | 0.9925 | 0.7297 | 0.7621 | 0.7681 |

Ceilings: `results/round2/20261112_gatev_ceiling/` (HK computed in r2-report without a second transport, scale fixed by the committed Σμ; validated to 1.7e-15 on SK and JUNO; fresh-context review PASS). The module ceiling is ≥ 0.99 everywhere, so the module shortfall is not finite-photon noise; the log-charge ceiling falls to 0.91 (SK) and 0.73 (HK) and explains part of the log-charge shortfall. Gate V (round-1 absolute thresholds: module-pooled Pearson ≥ 0.85, PMT log-charge Pearson ≥ 0.70) failed on SK, JUNO and HK (it passed on WCTE in round 1). The analytic model is therefore not validated against PhotonSim + LUCiD at these detector scales; Gate V was made informational by the author's decision of 12 Nov 2026, 10:08 (addendum `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md`), and every number below carries this caveat.

## Cross-detector table (§7, descriptive; no pass/fail across detectors)

Vertex: full-input mean 3-D error, seed mean of 9000/9001 (WCTE: round-1 committed values; MAE and init single seed). R2 classes use the §7 rules unchanged, two-seed version; the three-seed version gives the same class everywhere.

| | WCTE (round 1) | SK | JUNO | HK |
|---|---|---|---|---|
| N_PMT / fiducial radius (m) | 1,995 / 1.26 | 11,096 / 13.57 | 10,000 / 14.00 | 19,746 / 25.94 |
| vertex MAE, m (/ R_fid) | 0.105 (0.0832) | 0.878 (0.0647) | 0.591 (0.0422) | 1.419 (0.0547) |
| vertex position-free JEPA, m (/ R_fid) | 0.133 (0.1056) | 1.282 (0.0945) | 0.993 (0.0709) | 2.068 (0.0797) |
| vertex random init, m (/ R_fid) | 0.160 (0.1269) | 1.266 (0.0933) | 0.959 (0.0685) | 2.054 (0.0792) |
| vertex fixed-label JEPA, m (/ R_fid) | 0.212 (0.1678) | 2.079 (0.1532) | 1.312 (0.0937) | 3.544 (0.1366) |
| vertex MAE (RF), seed 9000 | – | 0.768 (0.0566) | 0.520 (0.0372) | 1.249 (0.0482) |
| init floor band (5 seeds), mean [min, max] | – | 1.242 [1.206, 1.275] | 0.930 [0.890, 0.968] | 1.995 [1.946, 2.077] |
| R2-a fixed-label f_event (9000/9001) | 8.0e-5 / 1.6e-4 (≤ 1e-3: event-invariant by the rule) | event-invariant (2.6e-04 / 4.3e-04) | intermediate (2.3e-03 / 3.8e-03) | event-invariant (3.3e-04 / 7.3e-04) |
| R2-b position-free | recovered (round-1 R-2 reading) | unclear | not recovered | unclear |
| R2-c MAE − init vertex | -0.055 m (one seed, no CI) | yes (-0.347 [-0.367, -0.327]; -0.431 [-0.451, -0.410]) | yes (-0.264 [-0.279, -0.251]; -0.471 [-0.487, -0.455]) | yes (-0.497 [-0.531, -0.466]; -0.773 [-0.809, -0.738]) |
| R2-d order (τ vs WCTE) | MAE < posfree < init < fixed (reference) | MAE < init < posfree < fixed (0.67) | MAE < init < posfree < fixed (0.67) | MAE < init < posfree < fixed (0.67) |
| R2-e RF − MAE vertex, m [95 % CI] | – (64-token window ≈ 10 % of WCTE) | -0.160 [-0.175, -0.146] | -0.165 [-0.177, -0.154] | -0.330 [-0.355, -0.305] |
| μ/e AUC: MAE / posfree / init / fixed (9000) | 0.905 / – / 0.885 / 0.840 | 0.986 / 0.941 / 0.942 / 0.748 | 0.995 / 0.986 / 0.969 / 0.866 | 0.960 / 0.953 / 0.884 / 0.673 |
| e direction: MAE / posfree / init / fixed (9000) | 3.29° / – / 6.19° / 7.45° | 3.83° / 3.64° / 5.10° / 7.94° | 3.66° / 3.20° / 4.31° / 5.43° | 4.66° / 4.62° / 6.52° / 9.62° |
| μ direction: MAE / posfree / init / fixed (9000) | 4.93° / – / 8.41° / 9.95° | 6.03° / 4.98° / 6.54° / 10.84° | 6.04° / 4.17° / 6.34° / 6.84° | 4.93° / 4.59° / 6.24° / 11.01° |
| e energy R²: MAE / posfree / init / fixed (9000) | 0.953 / – / 0.943 / 0.908 | 0.987 / 0.958 / 0.981 / 0.942 | 0.992 / 0.987 / 0.986 / 0.950 | 0.991 / 0.989 / 0.990 / 0.943 |
| μ energy R²: MAE / posfree / init / fixed (9000) | – / – / – / – | 0.971 / 0.912 / 0.950 / 0.871 | 0.984 / 0.971 / 0.967 / 0.897 | 0.979 / 0.972 / 0.974 / 0.843 |
| Gate V module / log-charge median (absolute) | 0.902 / 0.815 (passed) | 0.829 / 0.692 (failed) | 0.888 / 0.761 (failed) | 0.756 / 0.563 (failed) |
| ceiling C module / log; ρ̃ module / log | 0.9993 / 0.979; 0.902 / 0.833 | 0.9972 / 0.911; 0.831 / 0.761 | 0.9989 / 0.971; 0.889 / 0.784 | 0.9925 / 0.730; 0.762 / 0.768 |

WCTE particle tasks: round-1 A-3 (`results/jepa/20261103_followup/readings.json`; MAE = `mae_within_trigger9000`, init = `init9000`, fixed = `fix9000`); position-free JEPA has no committed A-3 entry and WCTE muon energy was not evaluable (452 contained < 500). Copied to `wcte_particle_round1.json`.


## Floor band, particle tasks and token R² (five `init` seeds 9000–9004: mean [min, max])

| Detector | vertex (m) | token ln q R² | μ/e AUC | e dir (°) | μ dir (°) | e energy R² | μ energy R² | module gain R² |
|---|---|---|---|---|---|---|---|---|
| SK | 1.242 [1.206, 1.275] | 0.9987 [0.9981, 0.9988] | 0.9493 [0.9425, 0.9542] | 4.904 [4.541, 5.463] | 6.611 [6.364, 6.9] | 0.9812 [0.9778, 0.9825] | 0.9549 [0.9502, 0.9596] | 0.1693 [0.1555, 0.1855] |
| JUNO | 0.9298 [0.8902, 0.9682] | 0.9993 [0.9989, 0.9995] | 0.9719 [0.969, 0.9743] | 3.905 [3.563, 4.313] | 5.981 [5.155, 6.51] | 0.9862 [0.9852, 0.9875] | 0.9698 [0.9664, 0.9731] | 0.2824 [0.2698, 0.3045] |
| HK | 1.995 [1.946, 2.077] | 0.999 [0.9986, 0.9992] | 0.9022 [0.8836, 0.914] | 6.39 [5.893, 6.664] | 6.178 [5.834, 6.469] | 0.99 [0.9891, 0.9906] | 0.9736 [0.9701, 0.975] | 0.1801 [0.1632, 0.1925] |

## R2-e particle-task differences (MAE (RF) − MAE, seed 9000)

| Detector | μ/e AUC | e dir (°) | μ dir (°) | e energy R² | μ energy R² |
|---|---|---|---|---|---|
| SK | +0.0030 | -0.6700 | +0.1222 | -0.0086 | -0.0189 |
| JUNO | +0.0024 | -0.6485 | -1.5548 | -0.0036 | -0.0093 |
| HK | -0.0103 | -0.7383 | -0.2666 | -0.0021 | -0.0061 |

## Seed-9002 runs (§8; all triggers trained and evaluated, none skipped)

| Detector | Arm | Trigger | Hardware, GPU h | 9002: vertex (m) vs init 9002 | token R² | f_event | Two-seed class | Three-seed class |
|---|---|---|---|---|---|---|---|---|
| SK | target_posfree | R2-b unclear | TOpAS A100 (batch line), 6.42 | 1.3400 vs 1.2209 | 0.9781 | 0.0251 | R2-b unclear | R2-b unclear |
| JUNO | sigreg_jepa | R2-a intermediate | TOpAS A100 (batch line), 5.43 | 1.3038 vs 0.9007 | 0.7184 | 0.0071 | R2-a intermediate | R2-a intermediate |
| HK | target_posfree | R2-b unclear | TOpAS A100 (batch line), 16.12 | 2.0499 vs 1.9458 | 0.9950 | 0.0251 | R2-b unclear | R2-b unclear |

The overlap trigger (seed ranges of `mae_mse` and `target_posfree` vertex) did not fire on any detector; R2-c read *yes* everywhere, so no `mae_mse` seed 9002 was triggered.


## Figures (`figures/`)

- `vertex_per_arm.png`: vertex per arm and seed per detector, with WCTE (round 1) and the five-seed `init` floor band (grey).
- `f_event_per_arm.png`: full-input f_event per arm.
- `particle_tasks.png`: the particle-task table (seed 9000) with WCTE round-1 values and the `init` floor band.
- `rf_minus_mae.png`: R2-e, `mae_mse_rf` − `mae_mse` per detector (vertex with paired bootstrap CI against the ±1.5 cm round-1 seed spread; token R², f_event; particle tasks).
- `time_scale.png`: §6 time-scale diagnostics (time-feature std; time-term loss share, first/last 100 steps, against WCTE with a ×/÷2 band).
