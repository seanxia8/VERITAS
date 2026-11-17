# Round 2 repair and diagnostics

17 November 2026 · VERITAS · `r2_repair_2026-11-17` · `registered: false`

## 1. Executive summary

All seven execution stages finished. The campaign evaluated 35 frozen checkpoint and initialization cases, completed 14 seed-9000 trainings to 10,000 optimizer steps, and ran both final probe suites on every trained model. The WCTE/SK comparison scores 90 snapshots on matched cohorts. The settled charge is 66.072986 GPU-h against a 100 GPU-h cap. The scientific verdict is **NOT_READY_FOR_ACCEPTANCE** because the WCTE P2a S4 criterion remains open. [GPU ledger](../../../docs/round2/RESOURCE_LEDGER.json)

### Twelve lines for Junjie

1. Q1. The fixed-label code is event-invariant in the original WCTE, SK and HK comparisons. Frozen seed-9000/9001 event variance fractions are 0.000080/0.000160, 0.000261/0.000425 and 0.000334/0.000726.
2. Q1. JUNO is intermediate at 0.0023/0.0038. The new WCTE/SK fixed-label runs finish at 0.000084/0.000217.
3. Q2. Position-free targets met the original recovery rule in WCTE. SK and HK were inconclusive, and JUNO finished at 1.0135/0.9716 m against initialization at 0.9499/0.9682 m.
4. Q3. Original MAE improves over initialization on all four detectors. Seed-9000 errors are 0.1049/0.9287/0.6855/1.5797 m versus 0.1601/1.2755/0.9499/2.0771 m in WCTE/SK/JUNO/HK order.
5. Q4. WCTE ranks MAE, position-free JEPA, initialization, fixed-label JEPA. SK, JUNO and HK rank MAE, initialization, position-free JEPA, fixed-label JEPA, with Kendall τ = 0.667.
6. Reading the frozen MAE through latent spatial moments gives 0.0397/0.1280/0.0964/0.1863 m.
7. Repaired MAE gives 0.1007/0.3863/0.3666/0.6518 m with all-mean Ridge and 0.0411/0.1189/0.0907/0.1674 m with latent spatial-moment Ridge.
8. Repaired JEPA gives event variance fractions of 0.0958/0.1232/0.1318/0.0847 and latent spatial errors of 0.0465/0.1885/0.1487/0.2559 m.
9. WCTE all-posfree reaches 0.3218 event variance fraction, 0.8292 m with all-mean Ridge and 0.0534 m with spatial readout.
10. Joint photon-bomb ML errors average 2.417/1.543/1.642/1.154 cm. The corresponding median position CRLBs are 1.859/1.432/1.509/1.085 cm.
11. HK Gate V yields LUCiD/analytic of 2.073 at 50 m/cosine, 0.900 with attenuation disabled, and 0.706 when spherical acceptance is also applied.
12. Historical S4 violations fell from four to two. The K5 exact stratum remains outside 5 mm on both fixed-label seeds.

[Original Q1–Q4 readings](../20261112_eval/READINGS_SUMMARY.md) · [Frozen readout results](probes/PROBE_REPAIR_TABLE.md) · [All retrained readouts](retrain/retrain_readings.json)

## 2. Photon-bomb ceiling and vertex readouts

Cells show metres followed by error divided by the detector fiducial radius in parentheses. ML errors are means over 500 photon-bomb events per detector. CRLB is the median square root of the position covariance trace under the same response model. Probe errors are means on 3,400 held-out photon-bomb events per detector at seed 9000. The ML model uses calibrated detector response and a Poisson/SPE charge approximation with a timing likelihood.

| Detector | Fiducial radius (m) | ToF-only ML | Charge-only ML | Joint ML | Median CRLB |
| --- | --- | --- | --- | --- | --- |
| WCTE | 1.262 | 0.0364 (0.02886) | 0.0343 (0.02717) | 0.0242 (0.01915) | 0.0186 (0.01473) |
| SK | 13.570 | 0.0152 (0.00112) | 0.1847 (0.01361) | 0.0154 (0.00114) | 0.0143 (0.00105) |
| JUNO | 14.000 | 0.0164 (0.00117) | 0.1934 (0.01381) | 0.0164 (0.00117) | 0.0151 (0.00108) |
| HK | 25.944 | 0.0113 (0.00044) | 0.2350 (0.00906) | 0.0115 (0.00044) | 0.0109 (0.00042) |

| Detector | Frozen MAE all-mean | Frozen MAE spatial | Repaired MAE all-mean | Repaired MAE spatial | Repaired JEPA all-mean | Repaired JEPA spatial |
| --- | --- | --- | --- | --- | --- | --- |
| WCTE | 0.1049 (0.08313) | 0.0397 (0.03144) | 0.1007 (0.07983) | 0.0411 (0.03256) | 0.1345 (0.10656) | 0.0465 (0.03686) |
| SK | 0.9287 (0.06844) | 0.1280 (0.00944) | 0.3863 (0.02846) | 0.1189 (0.00876) | 1.2763 (0.09405) | 0.1885 (0.01389) |
| JUNO | 0.6855 (0.04896) | 0.0964 (0.00688) | 0.3666 (0.02618) | 0.0907 (0.00648) | 1.0026 (0.07162) | 0.1487 (0.01062) |
| HK | 1.5797 (0.06089) | 0.1863 (0.00718) | 0.6518 (0.02512) | 0.1674 (0.00645) | 1.7993 (0.06935) | 0.2559 (0.00986) |

All 2,000 selected events converged under each of the three ML fits and had finite CRLBs. Repaired MAE improved all-mean vertex error on each detector relative to the frozen MAE. With spatial readout, WCTE original MAE retained a small advantage (0.0397 m versus 0.0411 m). Frozen latent spatial MAE also beat the raw spatial features on all four detectors. [Physics results](diagnostics/physics_ceiling.json) · [ML and CRLB versus yield and wall distance](figures/physics_ceiling_vs_yield.png) · [Frozen probe variants](probes/probe_repair_results.json)

## 3. Simulation closure, Gate V and the HK trigger

The Gate V table gives median LUCiD/analytic yield. The four settings change analytic attenuation and angular response. Stage 2 uses ten transport holdouts per detector, with the existing transport references for WCTE, SK and JUNO and a straight-ray sphere-intersection replay for HK.

| Detector | 50 m cosine | Off cosine | 50 m spherical | Off spherical |
| --- | --- | --- | --- | --- |
| WCTE | 1.0061 | 0.9770 | 0.6243 | 0.6058 |
| SK | 1.1863 | 0.7553 | 0.9293 | 0.5907 |
| JUNO | 1.0774 | 0.7594 | 1.0768 | 0.7590 |
| HK | 2.0731 | 0.8996 | 1.6322 | 0.7057 |

The original HK ratio was 2.07295 and the replay gave 2.07305. Matching attenuation produces 0.89965. Spherical acceptance with attenuation disabled produces 0.70572, with module Pearson 0.76412 and log-charge Pearson 0.58883. The original module Pearson was 0.75647 and its photon-statistics ceiling was 0.9925. The normalization discrepancy is sensitive to the optical assumptions. The remaining spatial mismatch is larger than the photon-statistics ceiling can explain. [Gate V results](gatev/gatev_reconciliation.json) · [HK and SK per-PMT ratios](figures/gatev_hk_sk_ratio.png)

| Detector | Surface | PMTs | PMT radius (m) | Nominal area coverage | Prompt window (ns) | Trigger hits | Token time scale (ns) | Dark rate per PMT (Hz) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WCTE | cylinder | 1995 | 0.040 | 23.2% | -20 to 60 | 4 | 100.0 | 1000 |
| SK | cylinder | 11096 | 0.254 | 39.6% | -20 to 270 | 6 | 450.0 | 1000 |
| JUNO | sphere | 10000 | 0.250 | 51.0% | -20 to 200 | 6 | 333.3 | 1000 |
| HK | cylinder | 19746 | 0.254 | 20.0% | -20 to 460 | 8 | 766.7 | 1000 |

Nominal coverage is total PMT disc area divided by the full cylinder or sphere surface area. All four share the 300–600 nm QE curve (0.08–0.25 across tabulated wavelengths), 0.95 collection efficiency and 1.5 ns single-photoelectron timing spread. The analytic reference uses 50 m attenuation and cosine response. The matched grid also tests attenuation off and spherical acceptance. Replayed total mean charge agrees with the stored generator expectation to within 6 × 10⁻⁸ relative in every detector. Charge-versus-yield slopes are 0.9996–1.0027 and the central timing-residual half-widths are 1.520–1.534 ns. [Simulation sanity](figures/sim_sanity_summary.png)

The keyed HK replay identified 152/500 dark-moved events. Mean joint-ML error is 0.01131 m in those 152 events and 0.01164 m in the other 348. Frozen MAE all-mean error is 1.6263 m versus 1.5139 m. The same model with latent spatial readout gives 0.17565 m versus 0.17680 m. These are descriptive comparisons of the two trigger strata. [Trigger and closure records](diagnostics/physics_ceiling.json)

## 4. Event displays and error anatomy

Each detector shows the six worst frozen MAE cases, three near-median MAE cases, and six worst fixed-label JEPA cases. WCTE/SK/HK use an unrolled cylinder and JUNO uses Mollweide. Each grid aligns charge and first-hit time with truth, joint ML and the baseline and spatial-readout vertex markers. The 500-event matched panels quantify variation with radius, height, wall distance, photon yield and run.

| Detector | Low/high-yield MAE all-mean median (m) | Low/high-yield MAE spatial median (m) | Low/high-yield joint ML median (m) |
| --- | --- | --- | --- |
| WCTE | 0.148 / 0.066 | 0.063 / 0.024 | 0.0404 / 0.0134 |
| SK | 1.185 / 0.666 | 0.165 / 0.085 | 0.0206 / 0.0108 |
| JUNO | 0.883 / 0.526 | 0.122 / 0.067 | 0.0211 / 0.0112 |
| HK | 1.983 / 1.182 | 0.228 / 0.131 | 0.0151 / 0.0080 |

Each yield quartile contains 125 matched events. Low yield increases median error in every displayed reconstruction. Spatial readout also shows a near-wall penalty. Comparing the nearest and farthest wall-distance quartiles gives 0.136/0.091 m on SK, 0.101/0.062 m on JUNO and 0.197/0.134 m on HK. [WCTE errors](figures/error_anatomy/wcte.png) · [SK errors](figures/error_anatomy/sk.png) · [JUNO errors](figures/error_anatomy/juno.png) · [HK errors](figures/error_anatomy/hk.png)

[WCTE worst MAE displays](figures/event_displays/wcte_worst_mae.png) · [SK worst MAE displays](figures/event_displays/sk_worst_mae.png) · [JUNO worst MAE displays](figures/event_displays/juno_worst_mae.png) · [HK worst MAE displays](figures/event_displays/hk_worst_mae.png) · [All event displays](figures/event_displays/)

## 5. JEPA collapse, training repairs and latent evolution

The 200-event per-PMT decomposition attributes 99.65%, 98.49%, 89.48% and 96.84% of fixed-label frozen token variance to static layout in WCTE/SK/JUNO/HK. Position-free targets give 82.95%, 78.64%, 90.14% and 85.72%. The all-posfree WCTE control removes encoder position/direction columns from context, label and full evaluation, while retaining the predictor queries and Hilbert ordering. Its high event fraction and 0.8292 m all-mean vertex error show that event signal and spatial vertex information are distinct. Spatial moments recover substantial vertex information when sensor coordinates are supplied at readout. [Variance and CKA](diagnostics/latent_variance_cka.json) · [PCA, t-SNE and UMAP maps](figures/latent/)

The previous loader appended four events per run contiguously, so a micro-batch of four saw one run. Keyed shuffling interleaves runs and the two-pass backward evaluates SIGReg once over the effective batch. In the frozen SK fixed-label replay, the context statistic was 1.60848 on four events and 12.86780 on 32. The repaired model combines a stop-gradient EMA teacher, position-free target inputs, hit-only latent and observable losses, an occupancy head, and event/hit-token regularization. Repaired MAE uses Hilbert spatial patches and rescales timing to 50 ns. HK JEPA uses the runbook's W128 fallback after the W640 L40S smoke projected 12.0 h. HK MAE retains W640. [Held-out training objective checks](diagnostics/a6_training_checks.json)

| Detector | Arm | All-mean (m) | Spatial (m) | Event fraction | Layout fraction | Rank | μ/e AUC | Direction e/μ (°) | Energy R² e/μ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WCTE | Fixed-label JEPA | 0.222 | 0.140 | 0.000084 | 0.997 | 2.52 | 0.845 | 7.59 / 10.26 | 0.908 / — |
| WCTE | Position-free target | 0.127 | 0.058 | 0.063005 | 0.818 | 1.53 | 0.867 | 7.50 / 11.32 | 0.890 / — |
| WCTE | Original MAE | 0.105 | 0.040 | 0.045520 | 0.883 | 2.91 | 0.905 | 3.28 / 4.72 | 0.953 / — |
| WCTE | All-posfree control | 0.829 | 0.053 | 0.321834 | 0.024 | 1.27 | 0.927 | 61.81 / 64.92 | 0.975 / — |
| WCTE | Repaired JEPA | 0.134 | 0.047 | 0.095791 | 0.735 | 3.65 | 0.925 | 4.31 / 5.49 | 0.938 / — |
| WCTE | Repaired MAE | 0.101 | 0.041 | 0.088263 | 0.728 | 3.62 | 0.931 | 3.63 / 5.28 | 0.937 / — |
| SK | Fixed-label JEPA | 2.177 | 0.724 | 0.000217 | 0.985 | 5.01 | 0.752 | 7.83 / 10.65 | 0.928 / 0.863 |
| SK | Position-free target | 1.048 | 0.219 | 0.045515 | 0.681 | 2.81 | 0.945 | 4.17 / 5.42 | 0.947 / 0.890 |
| SK | Repaired JEPA | 1.276 | 0.189 | 0.123230 | 0.658 | 2.72 | 0.983 | 5.12 / 7.16 | 0.965 / 0.933 |
| SK | Repaired MAE | 0.386 | 0.119 | 0.094733 | 0.699 | 3.40 | 0.995 | 3.62 / 4.67 | 0.951 / 0.905 |
| JUNO | Repaired JEPA | 1.003 | 0.149 | 0.131818 | 0.645 | 2.71 | 0.992 | 4.79 / 9.26 | 0.975 / 0.954 |
| JUNO | Repaired MAE | 0.367 | 0.091 | 0.103617 | 0.669 | 3.18 | 0.999 | 2.97 / 4.14 | 0.972 / 0.937 |
| HK | Repaired JEPA | 1.799 | 0.256 | 0.084658 | 0.767 | 2.30 | 0.972 | 5.18 / 7.12 | 0.984 / 0.948 |
| HK | Repaired MAE | 0.652 | 0.167 | 0.090794 | 0.680 | 3.21 | 0.979 | 3.99 / 5.83 | 0.957 / 0.916 |

WCTE muon energy has 452 contained test events and is below the 500-event evaluation floor. Final readout files also give light yield, module gain, both spatial Ridge variants, the MLP, and token charge and time controls. For example, frozen MAE post-LayerNorm charge R² spans 0.9873–0.9980, while the masked-target charge R² spans 0.5816–0.6908 across detectors. These two measurements use different access to target observables. [Final standard and repaired metrics](retrain/retrain_readings.json) · [Frozen token readouts](probes/probe_repair_results.json)

| Detector | Repaired MAE first 100 time share | Last 100 time share |
| --- | --- | --- |
| WCTE | 8.37% | 0.52% |
| SK | 72.32% | 3.10% |
| JUNO | 57.35% | 1.70% |
| HK | 91.19% | 15.24% |

The nine-step curves use 1,000 distinct probe-train and 1,000 distinct probe-test events per detector, fixed across arms and checkpoints. SK fixed-label event fraction falls as effective rank rises. On that detector the repaired JEPA increases event fraction and improves spatial error relative to the fixed-label baseline. The UMAP panels are independently fitted with seed 42, 15 neighbors and common truth color scales per detector. Loss curves use each objective's native units. [WCTE curves](figures/evolution/metrics_vs_step_wcte.png) · [WCTE UMAP](figures/evolution/umap_evolution_wcte.png) · [SK curves](figures/evolution/metrics_vs_step_sk.png) · [SK UMAP](figures/evolution/umap_evolution_sk.png) · [Evolution readings](retrain/latent_evolution.json)

## 6. WCTE P2a split-penalty ceiling and S4

The original 78 model/condition/stratum readings reproduced frozen R2 within 10⁻⁹ m. Split penalties reduced historical S4 violations from four to two. Both remaining comparisons are in K5 exact.

| Model | Condition | Stratum | R2 − split (mm) | Paired 95% CI (mm) |
| --- | --- | --- | --- | --- |
| fix9000 | K5 | exact | -5.421 | [-7.238, -3.748] |
| fix9001 | K5 | exact | -5.437 | [-7.143, -3.823] |

The requested gain/time grid contains six models per drift cell. In this grid the gain/time perturbations keep the known-dead block at zero. The full design includes the complementary-view contrast and reproduces R2 in every cell.

| Gain cell | Time cell | Split − R2 range (mm) | Split violations | Full violations |
| --- | --- | --- | --- | --- |
| N | N | 20.3–54.6 | 6 | 0 |
| N | D2_j25 | 30.0–64.3 | 6 | 0 |
| N | D2_j40 | 32.3–70.1 | 6 | 0 |
| D1_s015 | N | 19.9–55.0 | 6 | 0 |
| D1_s015 | D2_j25 | 29.7–63.8 | 6 | 0 |
| D1_s015 | D2_j40 | 32.2–69.5 | 6 | 0 |
| D1_s030 | N | 18.9–54.5 | 6 | 0 |
| D1_s030 | D2_j25 | 29.2–62.4 | 6 | 0 |
| D1_s030 | D2_j40 | 31.3–67.7 | 6 | 0 |

The grid records 54/54 split violations and 0/54 full-design violations. S4 remains open in the original replay and the restricted split ceiling. [All 54 grid rows](wcte_p2a/WCTE_P2A_REPAIR.md) · [Historical audit and paired intervals](wcte_p2a/p2a_repaired_readings.json)

Deepthought regressions recorded 992 passed, zero failed, two skipped and one pre-existing expected failure. The existing guard fixture passed after rerunning with node-local temp files. At one CPU thread a single legacy gradient component exceeded the requested 10⁻⁵ tolerance by 1.36 × 10⁻⁸. Both exact-gradient fixtures met 10⁻⁵ with two threads and restore the caller's thread setting. Stage 5 was verified against 106 snapshot files, including the 90 scored WCTE/SK snapshots. Training source was `b1d90bf`, the evolution plotting source was `c01eaa4`, and the complete Stage 5 results and accounting are at `f813de2`. [Execution decisions](../../../docs/round2/DECISIONS.md) · [Acceptance record](ACCEPTANCE_2026-11-17_R2_REPAIR.json)
