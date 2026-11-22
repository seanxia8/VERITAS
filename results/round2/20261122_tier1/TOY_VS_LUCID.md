# Toy versus LUCiD (rebuilt by r2-tier1 T1.5; keyed by run id)

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

## SK

| Arm | Seed | Spatial Ridge toy | Spatial Ridge LUCiD | Gain/init toy | Gain/init LUCiD | Spatial MLP toy | Spatial MLP LUCiD | All-mean toy | All-mean LUCiD | Raw spatial LUCiD | Token time R² toy / LUCiD | f_event toy / LUCiD | f_layout toy / LUCiD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| init | 9000 | 0.1830 | 0.5274 | – | – | 0.3266 | 0.6206 | 1.2755 | 3.6946 | 0.6671 | 0.900 / 0.932 | – / 0.0369 | – / 0.917 |
| mae_mse *frozen* | 9000 | 0.1280 | 0.4266 | 0.0549 | 0.1009 | 0.2553 | 0.5593 | 0.9287 | 2.0563 | 0.6671 | 0.983 / 0.988 | – / 0.0574 | – / 0.834 |
| mae_repaired | 9000 | 0.1189 | 0.3564 | 0.0641 | 0.1711 | 0.2175 | 0.3713 | 0.3863 | 0.8389 | 0.6671 | 0.993 / 0.983 | 0.0947 / 0.1589 | 0.699 / 0.529 |
| jepa_repaired | 9000 | 0.1885 | 0.4491 | -0.0055 | 0.0783 | 0.2572 | 0.4302 | 1.2763 | 2.4496 | 0.6671 | 0.985 / 0.978 | 0.1232 / 0.1249 | 0.658 / 0.736 |
| sigreg_jepa | 9000 | 0.7241 | 1.1512 | -0.5411 | -0.6237 | 0.3967 | 0.7808 | 2.1770 | 4.8226 | 0.6671 | 0.044 / 0.067 | 0.0002 / 0.0001 | 0.985 / 0.989 |
| init | 9001 | 0.1911 | 0.5258 | – | – | 0.3293 | 0.6288 | 1.2572 | 3.5210 | 0.6671 | 0.908 / 0.925 | – / 0.0381 | – / 0.916 |
| mae_mse *frozen* | 9001 | 0.1312 | – | 0.0598 | – | 0.2529 | – | 0.8264 | – | – | 0.988 / – | – / – | – / – |
| sigreg_jepa *frozen* | 9001 | 0.5830 | – | -0.3919 | – | 0.3708 | – | 1.9727 | – | – | 0.081 / – | – / – | – / – |

Particle tasks (standard suite, LUCiD; toy where available):

| Arm | Seed | μ/e AUC toy / LUCiD | μ energy R² toy / LUCiD | e energy R² toy / LUCiD | μ dir. (°) toy / LUCiD | e dir. (°) toy / LUCiD |
|---|---:|---|---|---|---|---|
| init | 9000 | – / 0.993 | – / 0.984 | – / 0.994 | – / 18.61 | – / 11.17 |
| mae_mse | 9000 | – / 0.999 | – / 0.990 | – / 0.996 | – / 9.08 | – / 7.44 |
| mae_repaired | 9000 | 0.995 / 1.000 | 0.905 / 0.981 | 0.951 / 0.986 | 4.67 / 5.58 | 3.62 / 4.68 |
| jepa_repaired | 9000 | 0.983 / 0.999 | 0.933 / 0.978 | 0.965 / 0.993 | 7.16 / 9.48 | 5.12 / 7.78 |
| sigreg_jepa | 9000 | 0.752 / 0.866 | 0.863 / 0.942 | 0.928 / 0.948 | 10.65 / 18.27 | 7.83 / 16.52 |
| init | 9001 | – / 0.995 | – / 0.985 | – / 0.994 | – / 17.60 | – / 11.81 |

## JUNO

| Arm | Seed | Spatial Ridge toy | Spatial Ridge LUCiD | Gain/init toy | Gain/init LUCiD | Spatial MLP toy | Spatial MLP LUCiD | All-mean toy | All-mean LUCiD | Raw spatial LUCiD | Token time R² toy / LUCiD | f_event toy / LUCiD | f_layout toy / LUCiD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| init | 9000 | 0.1225 | 0.4232 | – | – | 0.2594 | 0.4741 | 0.9499 | 2.6565 | 0.5383 | 0.957 / 0.975 | – / 0.0571 | – / 0.889 |
| mae_mse *frozen* | 9000 | 0.0964 | – | 0.0261 | – | 0.1903 | – | 0.6855 | – | – | 0.995 / – | – / – | – / – |
| mae_repaired | 9000 | 0.0907 | 0.2876 | 0.0318 | 0.1355 | 0.1730 | 0.3671 | 0.3666 | 0.7323 | 0.5383 | 0.997 / 0.993 | 0.1036 / 0.1889 | 0.669 / 0.471 |
| jepa_repaired | 9000 | 0.1487 | 0.3786 | -0.0262 | 0.0445 | 0.2004 | 0.3358 | 1.0026 | 1.9809 | 0.5383 | 0.989 / 0.983 | 0.1318 / 0.1293 | 0.645 / 0.685 |
| sigreg_jepa *frozen* | 9000 | 0.3982 | – | -0.2757 | – | 0.3118 | – | 1.3400 | – | – | 0.121 / – | – / – | – / – |
| init | 9001 | 0.1291 | 0.4228 | – | – | 0.2703 | 0.4810 | 0.9682 | 2.5945 | 0.5383 | 0.959 / 0.975 | – / 0.0536 | – / 0.897 |
| mae_mse *frozen* | 9001 | 0.0923 | – | 0.0368 | – | 0.1814 | – | 0.4972 | – | – | 0.995 / – | – / – | – / – |
| sigreg_jepa *frozen* | 9001 | 0.3531 | – | -0.2240 | – | 0.3050 | – | 1.2834 | – | – | 0.136 / – | – / – | – / – |

Particle tasks (standard suite, LUCiD; toy where available):

| Arm | Seed | μ/e AUC toy / LUCiD | μ energy R² toy / LUCiD | e energy R² toy / LUCiD | μ dir. (°) toy / LUCiD | e dir. (°) toy / LUCiD |
|---|---:|---|---|---|---|---|
| init | 9000 | – / 0.990 | – / 0.992 | – / 0.996 | – / 9.89 | – / 8.54 |
| mae_repaired | 9000 | 0.999 / 1.000 | 0.937 / 0.991 | 0.972 / 0.994 | 4.14 / 5.97 | 2.97 / 5.71 |
| jepa_repaired | 9000 | 0.992 / 0.999 | 0.954 / 0.991 | 0.975 / 0.996 | 9.26 / 6.13 | 4.79 / 5.73 |
| init | 9001 | – / 0.995 | – / 0.992 | – / 0.996 | – / 10.37 | – / 9.58 |

## HK

| Arm | Seed | Spatial Ridge toy | Spatial Ridge LUCiD | Gain/init toy | Gain/init LUCiD | Spatial MLP toy | Spatial MLP LUCiD | All-mean toy | All-mean LUCiD | Raw spatial LUCiD | Token time R² toy / LUCiD | f_event toy / LUCiD | f_layout toy / LUCiD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| init | 9000 | 0.2600 | 0.9072 | – | – | 0.5458 | 1.0964 | 2.0771 | 7.3804 | 1.1884 | 0.923 / 0.951 | – / 0.0229 | – / 0.944 |
| mae_mse *frozen* | 9000 | 0.1863 | – | 0.0737 | – | 0.4535 | – | 1.5797 | – | – | 0.977 / – | – / – | – / – |
| mae_repaired | 9000 | 0.1674 | 0.6247 | 0.0925 | 0.2825 | 0.3862 | 0.6553 | 0.6518 | 1.5775 | 1.1884 | 0.993 / 0.983 | 0.0908 / 0.1352 | 0.680 / 0.569 |
| jepa_repaired | 9000 | 0.2559 | 0.8119 | 0.0040 | 0.0952 | 0.4174 | 0.8574 | 1.7993 | 5.5208 | 1.1884 | 0.986 / 0.979 | 0.0847 / 0.0656 | 0.767 / 0.833 |
| sigreg_jepa *frozen* | 9000 | 0.9180 | – | -0.6580 | – | 0.6198 | – | 3.7033 | – | – | 0.089 / – | – / – | – / – |
| init | 9001 | 0.2779 | 0.8967 | – | – | 0.5511 | 1.1136 | 2.0311 | 7.1678 | 1.1884 | 0.923 / 0.946 | – / 0.0226 | – / 0.945 |
| mae_mse *frozen* | 9001 | 0.1872 | – | 0.0907 | – | 0.4530 | – | 1.2582 | – | – | 0.987 / – | – / – | – / – |
| sigreg_jepa *frozen* | 9001 | 0.7323 | – | -0.4543 | – | 0.5876 | – | 3.3839 | – | – | 0.123 / – | – / – | – / – |

Particle tasks (standard suite, LUCiD; toy where available):

| Arm | Seed | μ/e AUC toy / LUCiD | μ energy R² toy / LUCiD | e energy R² toy / LUCiD | μ dir. (°) toy / LUCiD | e dir. (°) toy / LUCiD |
|---|---:|---|---|---|---|---|
| init | 9000 | – / 0.985 | – / 0.986 | – / 0.994 | – / 16.16 | – / 13.70 |
| mae_repaired | 9000 | 0.979 / 1.000 | 0.916 / 0.977 | 0.957 / 0.982 | 5.83 / 5.58 | 3.99 / 4.58 |
| jepa_repaired | 9000 | 0.972 / 1.000 | 0.948 / 0.981 | 0.984 / 0.993 | 7.12 / 9.30 | 5.18 / 7.32 |
| init | 9001 | – / 0.992 | – / 0.986 | – / 0.994 | – / 17.16 | – / 12.92 |

## Sweep and variant runs (LUCiD; not part of the main tables)

Keyed by full run id (suffixes `__m<mask>`, `__ts<unit>`, `__cw<weight>`).

| Run id | Spatial Ridge LUCiD | Spatial MLP LUCiD | All-mean LUCiD | Raw spatial LUCiD | Token time R² |
|---|---:|---:|---:|---:|---:|
| wcte__mae_repaired__s9000__ts15 | 0.0512 | 0.0527 | 0.1344 | 0.0752 | 0.976 |
| wcte__mae_repaired__s9000__ts1p5 | 0.0485 | 0.0449 | 0.1180 | 0.0752 | 0.986 |
| wcte__mae_repaired__s9000__ts450 | 0.0541 | 0.0469 | 0.1428 | 0.0752 | 0.957 |
| wcte__mae_repaired__s9000__ts5 | 0.0501 | 0.0513 | 0.1218 | 0.0752 | 0.983 |

