# r2-lucid interim report — night 1 (21 Nov 2026, 07:50 UTC)

**Status.** L0–L6 complete for all four detectors (not only WCTE and SK): LUCiD `e651e836` regenerated WCTE, SK, JUNO and HK (120,000 simulated events each; tier A1, `lucid_mc_native`), all validation gates pass ([VALIDATION.md](VALIDATION.md)), the analytic toy is archived read-only ([DATA_LINEAGE.md](../../../docs/DATA_LINEAGE.md)). L7/L8 for WCTE priority 1 are done (7 trained models + init × 2 seeds, standard and repaired probe suites). SK (4 arms), JUNO (2) and HK (2) are training on HTCondor A100s; the WCTE time-scale sweep is running on the guarded deepthought2 lanes. GPU booked so far ≈ 25 GPU-h of 120.

**WCTE readings (seed 9000, spatial-moment Ridge, primary metric).**
1. LUCiD data are harder for every arm: init 0.066 m (toy 0.053 m), mae_mse 0.058 (0.040), repaired MAE 0.052 (0.041), repaired JEPA 0.063 (0.047). The LUCiD regime is dimmer (photon-bomb detected PE median ≈ 2,000–2,500 vs the toy's brighter regime) and includes scattering, reflections and LUCiD's PMT jitter.
2. The ordering survives with one change: MAE-family < repaired JEPA < init < fixed-label SIGReg JEPA on both datasets, but on LUCiD the repaired MAE beats plain MAE (0.052 vs 0.058 m; seed 9001: 0.052 vs 0.056), whereas in the toy plain MAE was slightly better (0.040 vs 0.041).
3. Gains over init: repaired MAE +0.0135/+0.0142 m (toy +0.0114), plain MAE +0.0082/+0.0098 (toy +0.0128/+0.0121), repaired JEPA +0.0027/+0.0050 (toy +0.0060); fixed-label SIGReg JEPA is again far worse than init (−0.087 m), i.e. its collapse is not a toy artefact.
4. Event-variance fractions are higher on LUCiD (repaired MAE f_event 0.185 vs 0.088; repaired JEPA 0.171 vs 0.096), and token time R² rises (repaired MAE 0.971 vs 0.941).
5. Particle tasks: μ/e AUC is near 1 on LUCiD for every arm including init (0.98–0.997); direction errors are larger than in the toy (repaired MAE μ 10.2°, e 8.3° vs 5.3°, 3.6°). Energy regression is not evaluable on LUCiD WCTE: LUCiD's containment flag (track inside the tank over the SIREN 99.99 % emission extent) leaves 0 contained electrons and 310 muons, vs the toy's 0.8 m shower-length rule.
6. The simple ToF-only Huber fit (approximate, not a likelihood ceiling) gives a median 0.139 m on LUCiD vs 0.034 m in the toy: the first-hit times are much less informative under LUCiD's PMT jitter, scattering and dark noise at 4.2 kHz.

Mean 3-D vertex error (m) on photon-bomb probe-test events (run-grouped probes, 3,400 train / 3,400 test in the toy; LUCiD counts in the JSON). **Primary metric: latent spatial-moment Ridge.** Toy = analytic-toy world (round 2 / repair); LUCiD = `lucid_v1` (tier A1). Gain over init = init error − arm error (positive is better), same seed and same data. Toy rows marked *frozen* are the original round-1/2 checkpoints (their s9001 repaired arms were never trained in the toy).

## WCTE

| Arm | Seed | Spatial Ridge toy | Spatial Ridge LUCiD | Gain/init toy | Gain/init LUCiD | Spatial MLP toy | Spatial MLP LUCiD | All-mean toy | All-mean LUCiD | Raw spatial LUCiD | Token time R² toy / LUCiD | f_event toy / LUCiD | f_layout toy / LUCiD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| init | 9000 | 0.0525 | 0.0656 | – | – | 0.0552 | 0.0749 | 0.1601 | 0.1962 | 0.0752 | 0.629 / 0.757 | – / 0.1255 | – / 0.772 |
| mae_mse | 9000 | 0.0397 | 0.0575 | 0.0128 | 0.0082 | 0.0385 | 0.0522 | 0.1050 | 0.1521 | 0.0752 | 0.928 / 0.955 | 0.0455 / 0.1034 | 0.883 / 0.765 |
| mae_repaired | 9000 | 0.0411 | 0.0521 | 0.0114 | 0.0135 | 0.0446 | 0.0465 | 0.1007 | 0.1410 | 0.0752 | 0.941 / 0.971 | 0.0883 / 0.1852 | 0.728 / 0.548 |
| jepa_repaired | 9000 | 0.0465 | 0.0629 | 0.0060 | 0.0027 | 0.0408 | 0.0545 | 0.1345 | 0.1926 | 0.0752 | 0.884 / 0.901 | 0.0958 / 0.1706 | 0.735 / 0.593 |
| sigreg_jepa | 9000 | 0.1402 | 0.1525 | -0.0876 | -0.0869 | 0.0687 | 0.0866 | 0.2216 | 0.2636 | 0.0752 | 0.008 / 0.008 | 0.0001 / 0.0002 | 0.997 / 0.993 |
| init | 9001 | 0.0528 | 0.0662 | – | – | 0.0546 | 0.0735 | 0.1560 | 0.1915 | 0.0752 | 0.678 / 0.785 | – / 0.1380 | – / 0.752 |
| mae_mse *frozen* | 9001 | 0.0407 | 0.0564 | 0.0121 | 0.0098 | 0.0383 | 0.0515 | 0.1165 | 0.1574 | 0.0752 | 0.929 / 0.976 | – / 0.1180 | – / 0.743 |
| mae_repaired | 9001 | – | 0.0519 | – | 0.0142 | – | 0.0498 | – | 0.1373 | 0.0752 | – / 0.977 | – / 0.1980 | – / 0.544 |
| jepa_repaired | 9001 | – | 0.0612 | – | 0.0050 | – | 0.0520 | – | 0.1442 | 0.0752 | – / 0.920 | – / 0.1531 | – / 0.685 |
| sigreg_jepa *frozen* | 9001 | 0.1199 | – | -0.0671 | – | 0.0673 | – | 0.2043 | – | – | 0.010 / – | – / – | – / – |

Particle tasks (standard suite, LUCiD; toy where available):

| Arm | Seed | μ/e AUC toy / LUCiD | μ energy R² toy / LUCiD | e energy R² toy / LUCiD | μ dir. (°) toy / LUCiD | e dir. (°) toy / LUCiD |
|---|---:|---|---|---|---|---|
| init | 9000 | – / 0.980 | – / – | – / – | – / 18.81 | – / 14.90 |
| mae_mse | 9000 | 0.905 / 0.996 | – / – | 0.953 / – | 4.72 / 12.24 | 3.28 / 9.84 |
| mae_repaired | 9000 | 0.931 / 0.997 | – / – | 0.937 / – | 5.28 / 10.20 | 3.63 / 8.34 |
| jepa_repaired | 9000 | 0.925 / 0.996 | – / – | 0.938 / – | 5.49 / 12.54 | 4.31 / 10.66 |
| sigreg_jepa | 9000 | 0.845 / 0.884 | – / – | 0.908 / – | 10.26 / 15.77 | 7.59 / 14.90 |
| init | 9001 | – / 0.983 | – / – | – / – | – / 18.91 | – / 15.72 |
| mae_mse | 9001 | – / 0.994 | – / – | – / – | – / 13.43 | – / 10.61 |
| mae_repaired | 9001 | – / 0.995 | – / – | – / – | – / 9.54 | – / 7.85 |
| jepa_repaired | 9001 | – / 0.993 | – / – | – / – | – / 12.50 | – / 9.57 |

Approximate ToF-only ceiling (simple Huber fit on token first-hit times, 1000 probe-test bombs, straight-line n = 1.33; approximate, not a likelihood ceiling, no CRLB claim): toy median 0.034 m, p68 0.044 m, failures 0.1 %; LUCiD median 0.139 m, p68 0.168 m, failures 0.3 % (failure = fit outside 2 R_fid).

