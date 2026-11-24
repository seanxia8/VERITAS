# VERIFY_A_REPORT — blind recomputation of the LUCiD headline numbers (r2-tier1v, part A)

Seals: `8b5d2aa (A), 3ecf960 (A2)` ("BLIND SEAL A" / "BLIND SEAL A2"), each pushed before the matching r2-tier1 results were opened. Definitions: runbook §1. Ridge fitted on `dev_probe_train` photon bombs; tests `dev_probe_test` and `sealed_test` on all four detectors (HK/JUNO sealed in A2).

## Invariants (registered, runbook VA2)

| ID | Result |
|---|---|
| J1 | PASS |
| J2 | PASS |
| J3 | PASS |
| J4 | PASS |
| J5 | PASS |
| J6 | PASS |
| J7 | PASS |
| J8 | PASS |
| J9 | PASS |

## Tolerances against the other phase

| Quantity | Tolerance | Cases | Within | Max abs rel. diff |
|---|---|---|---|---|
| c | 5 % | 27 | 27 | 5.23e-07 |
| a | 5 % | 27 | 27 | 5.62e-05 |
| d | 5 % | 27 | 27 | 1.93e-03 |
| f_event | 15 % | 27 | 17 | 4.90e-01 |

Arm ordering per detector and seed identical: (c) True, (a) True, f_event False (f_event under the other phase's definition reproduces their values, so their ordering).

RESCORE_LUCID (228 values, incl. sealed_test): 228 within tolerance, max abs rel. diff 1.82e-03; (a), (c), (d) only; f_event separately (definitional).

## Per claim and case (dev_probe_test unless marked; other phase = rebuilt TOY_VS_LUCID.json)

| Case | Qty | Recomputed | Other phase | Rel. diff | Tol. | Verdict |
|---|---|---|---|---|---|---|
| wcte__legacy_init__s9000 | c | 0.0656 | 0.0656 | -0.00% | pass | V |
| wcte__legacy_init__s9000 | a | 0.1962 | 0.1962 | +0.00% | pass | V |
| wcte__legacy_init__s9000 | d | 0.0752 | 0.0752 | +0.08% | pass | V |
| wcte__legacy_init__s9000 | f_event | 0.1015 | 0.1255 | -19.15% | FAIL | R |
| wcte__mae_mse__s9000__final | c | 0.0575 | 0.0575 | +0.00% | pass | V |
| wcte__mae_mse__s9000__final | a | 0.1521 | 0.1521 | -0.00% | pass | V |
| wcte__mae_mse__s9000__final | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_mse__s9000__final | f_event | 0.0887 | 0.1034 | -14.19% | pass | V |
| wcte__mae_repaired__s9000__final | c | 0.0521 | 0.0521 | +0.00% | pass | V |
| wcte__mae_repaired__s9000__final | a | 0.1410 | 0.1410 | -0.00% | pass | V |
| wcte__mae_repaired__s9000__final | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_repaired__s9000__final | f_event | 0.1604 | 0.1852 | -13.42% | pass | V |
| wcte__jepa_repaired__s9000__final | c | 0.0629 | 0.0629 | +0.00% | pass | V |
| wcte__jepa_repaired__s9000__final | a | 0.1926 | 0.1926 | -0.00% | pass | V |
| wcte__jepa_repaired__s9000__final | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__jepa_repaired__s9000__final | f_event | 0.2083 | 0.1706 | +22.12% | FAIL | R |
| wcte__sigreg_jepa__s9000__final | c | 0.1525 | 0.1525 | -0.00% | pass | V |
| wcte__sigreg_jepa__s9000__final | a | 0.2636 | 0.2636 | -0.00% | pass | V |
| wcte__sigreg_jepa__s9000__final | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__sigreg_jepa__s9000__final | f_event | 0.0002 | 0.0002 | -7.72% | pass | V |
| wcte__legacy_init__s9001 | c | 0.0662 | 0.0662 | -0.00% | pass | V |
| wcte__legacy_init__s9001 | a | 0.1915 | 0.1915 | -0.00% | pass | V |
| wcte__legacy_init__s9001 | d | 0.0752 | 0.0752 | +0.08% | pass | V |
| wcte__legacy_init__s9001 | f_event | 0.1213 | 0.1380 | -12.16% | pass | V |
| wcte__mae_mse__s9001__final | c | 0.0564 | 0.0564 | +0.00% | pass | V |
| wcte__mae_mse__s9001__final | a | 0.1574 | 0.1574 | +0.00% | pass | V |
| wcte__mae_mse__s9001__final | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_mse__s9001__final | f_event | 0.0903 | 0.1180 | -23.51% | FAIL | R |
| wcte__mae_repaired__s9001__final | c | 0.0519 | 0.0519 | +0.00% | pass | V |
| wcte__mae_repaired__s9001__final | a | 0.1373 | 0.1373 | -0.00% | pass | V |
| wcte__mae_repaired__s9001__final | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_repaired__s9001__final | f_event | 0.1852 | 0.1980 | -6.49% | pass | V |
| wcte__jepa_repaired__s9001__final | c | 0.0612 | 0.0612 | +0.00% | pass | V |
| wcte__jepa_repaired__s9001__final | a | 0.1442 | 0.1442 | +0.00% | pass | V |
| wcte__jepa_repaired__s9001__final | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__jepa_repaired__s9001__final | f_event | 0.1589 | 0.1531 | +3.79% | pass | V |
| sk__legacy_init__s9000 | c | 0.5274 | 0.5274 | -0.00% | pass | V |
| sk__legacy_init__s9000 | a | 3.6946 | 3.6946 | +0.00% | pass | V |
| sk__legacy_init__s9000 | d | 0.6673 | 0.6671 | +0.03% | pass | V |
| sk__legacy_init__s9000 | f_event | 0.0220 | 0.0369 | -40.46% | FAIL | R |
| sk__mae_mse__s9000__final | c | 0.4266 | 0.4266 | +0.00% | pass | V |
| sk__mae_mse__s9000__final | a | 2.0562 | 2.0563 | -0.00% | pass | V |
| sk__mae_mse__s9000__final | d | 0.6673 | 0.6671 | +0.03% | pass | V |
| sk__mae_mse__s9000__final | f_event | 0.0293 | 0.0574 | -49.02% | FAIL | R |
| sk__mae_repaired__s9000__final | c | 0.3564 | 0.3564 | +0.00% | pass | V |
| sk__mae_repaired__s9000__final | a | 0.8389 | 0.8389 | -0.00% | pass | V |
| sk__mae_repaired__s9000__final | d | 0.6673 | 0.6671 | +0.03% | pass | V |
| sk__mae_repaired__s9000__final | f_event | 0.1414 | 0.1589 | -11.03% | pass | V |
| sk__jepa_repaired__s9000__final | c | 0.4491 | 0.4491 | +0.00% | pass | V |
| sk__jepa_repaired__s9000__final | a | 2.4496 | 2.4496 | +0.00% | pass | V |
| sk__jepa_repaired__s9000__final | d | 0.6673 | 0.6671 | +0.03% | pass | V |
| sk__jepa_repaired__s9000__final | f_event | 0.1281 | 0.1249 | +2.52% | pass | V |
| sk__sigreg_jepa__s9000__final | c | 1.1512 | 1.1512 | -0.00% | pass | V |
| sk__sigreg_jepa__s9000__final | a | 4.8226 | 4.8226 | +0.00% | pass | V |
| sk__sigreg_jepa__s9000__final | d | 0.6673 | 0.6671 | +0.03% | pass | V |
| sk__sigreg_jepa__s9000__final | f_event | 0.0001 | 0.0001 | -10.35% | pass | V |
| sk__legacy_init__s9001 | c | 0.5258 | 0.5258 | +0.00% | pass | V |
| sk__legacy_init__s9001 | a | 3.5209 | 3.5210 | -0.00% | pass | V |
| sk__legacy_init__s9001 | d | 0.6673 | 0.6671 | +0.03% | pass | V |
| sk__legacy_init__s9001 | f_event | 0.0247 | 0.0381 | -35.32% | FAIL | R |
| juno__legacy_init__s9000 | c | 0.4232 | 0.4232 | -0.00% | pass | V |
| juno__legacy_init__s9000 | a | 2.6565 | 2.6565 | -0.00% | pass | V |
| juno__legacy_init__s9000 | d | 0.5393 | 0.5383 | +0.19% | pass | V |
| juno__legacy_init__s9000 | f_event | 0.0313 | 0.0571 | -45.21% | FAIL | R |
| juno__mae_repaired__s9000__final | c | 0.2876 | 0.2876 | -0.00% | pass | V |
| juno__mae_repaired__s9000__final | a | 0.7323 | 0.7323 | -0.00% | pass | V |
| juno__mae_repaired__s9000__final | d | 0.5393 | 0.5383 | +0.19% | pass | V |
| juno__mae_repaired__s9000__final | f_event | 0.1768 | 0.1889 | -6.44% | pass | V |
| juno__jepa_repaired__s9000__final | c | 0.3786 | 0.3786 | +0.00% | pass | V |
| juno__jepa_repaired__s9000__final | a | 1.9809 | 1.9809 | -0.00% | pass | V |
| juno__jepa_repaired__s9000__final | d | 0.5393 | 0.5383 | +0.19% | pass | V |
| juno__jepa_repaired__s9000__final | f_event | 0.1214 | 0.1293 | -6.10% | pass | V |
| juno__legacy_init__s9001 | c | 0.4228 | 0.4228 | -0.00% | pass | V |
| juno__legacy_init__s9001 | a | 2.5946 | 2.5945 | +0.00% | pass | V |
| juno__legacy_init__s9001 | d | 0.5393 | 0.5383 | +0.19% | pass | V |
| juno__legacy_init__s9001 | f_event | 0.0332 | 0.0536 | -38.05% | FAIL | R |
| hk__legacy_init__s9000 | c | 0.9072 | 0.9072 | -0.00% | pass | V |
| hk__legacy_init__s9000 | a | 7.3803 | 7.3804 | -0.00% | pass | V |
| hk__legacy_init__s9000 | d | 1.1870 | 1.1884 | -0.12% | pass | V |
| hk__legacy_init__s9000 | f_event | 0.0134 | 0.0229 | -41.35% | FAIL | R |
| hk__mae_repaired__s9000__final | c | 0.6247 | 0.6247 | +0.00% | pass | V |
| hk__mae_repaired__s9000__final | a | 1.5775 | 1.5775 | +0.00% | pass | V |
| hk__mae_repaired__s9000__final | d | 1.1870 | 1.1884 | -0.12% | pass | V |
| hk__mae_repaired__s9000__final | f_event | 0.1180 | 0.1352 | -12.75% | pass | V |
| hk__jepa_repaired__s9000__final | c | 0.8119 | 0.8119 | +0.00% | pass | V |
| hk__jepa_repaired__s9000__final | a | 5.5204 | 5.5208 | -0.01% | pass | V |
| hk__jepa_repaired__s9000__final | d | 1.1870 | 1.1884 | -0.12% | pass | V |
| hk__jepa_repaired__s9000__final | f_event | 0.0627 | 0.0656 | -4.37% | pass | V |
| hk__legacy_init__s9001 | c | 0.8967 | 0.8967 | -0.00% | pass | V |
| hk__legacy_init__s9001 | a | 7.1678 | 7.1678 | +0.00% | pass | V |
| hk__legacy_init__s9001 | d | 1.1870 | 1.1884 | -0.12% | pass | V |
| hk__legacy_init__s9001 | f_event | 0.0136 | 0.0226 | -39.69% | FAIL | R |
| wcte__mae_repaired__s9000__ts15__final (expl.) | c | 0.0512 | 0.0512 | -0.00% | pass | V |
| wcte__mae_repaired__s9000__ts15__final (expl.) | a | 0.1344 | 0.1344 | +0.00% | pass | V |
| wcte__mae_repaired__s9000__ts15__final (expl.) | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_repaired__s9000__ts15__final (expl.) | f_event | 0.1570 | 0.1771 | -11.34% | pass | V |
| wcte__mae_repaired__s9000__ts1p5__final (expl.) | c | 0.0485 | 0.0485 | -0.00% | pass | V |
| wcte__mae_repaired__s9000__ts1p5__final (expl.) | a | 0.1180 | 0.1180 | +0.00% | pass | V |
| wcte__mae_repaired__s9000__ts1p5__final (expl.) | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_repaired__s9000__ts1p5__final (expl.) | f_event | 0.1168 | 0.1342 | -12.97% | pass | V |
| wcte__mae_repaired__s9000__ts450__final (expl.) | c | 0.0541 | 0.0541 | +0.00% | pass | V |
| wcte__mae_repaired__s9000__ts450__final (expl.) | a | 0.1428 | 0.1428 | +0.00% | pass | V |
| wcte__mae_repaired__s9000__ts450__final (expl.) | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_repaired__s9000__ts450__final (expl.) | f_event | 0.1912 | 0.1977 | -3.31% | pass | V |
| wcte__mae_repaired__s9000__ts5__final (expl.) | c | 0.0501 | 0.0501 | +0.00% | pass | V |
| wcte__mae_repaired__s9000__ts5__final (expl.) | a | 0.1218 | 0.1218 | -0.00% | pass | V |
| wcte__mae_repaired__s9000__ts5__final (expl.) | d | 0.0752 | 0.0752 | +0.07% | pass | V |
| wcte__mae_repaired__s9000__ts5__final (expl.) | f_event | 0.1402 | 0.1594 | -12.08% | pass | V |

## Per case and test set against RESCORE_LUCID.json (relative differences; f_event under my definition / under theirs)

| Case | Test | (c) m | Δ(c) | Δ(a) | Δ(d) | Δf_event | Δf_event (their def.) | Verdict |
|---|---|---|---|---|---|---|---|---|
| wcte__legacy_init__s9000 | dev | 0.0656 | -7.7e-09 | +2.1e-06 | +8.5e-04 | -19.2% | +1.4e-07 | R |
| wcte__legacy_init__s9000 | sealed | 0.0664 | +5.5e-08 | +1.0e-07 | +2.7e-04 | -18.1% | +4.4e-08 | R |
| wcte__legacy_init__s9001 | dev | 0.0662 | -2.0e-07 | -2.7e-06 | +8.5e-04 | -12.2% | +1.5e-07 | V |
| wcte__legacy_init__s9001 | sealed | 0.0673 | -8.3e-09 | +3.4e-07 | +2.7e-04 | -11.0% | +4.7e-08 | V |
| wcte__jepa_repaired__s9000__final | dev | 0.0629 | +1.5e-07 | -5.5e-08 | +8.5e-04 | +22.1% | +1.4e-07 | R |
| wcte__jepa_repaired__s9000__final | sealed | 0.0636 | +1.3e-07 | -3.2e-07 | +2.7e-04 | +22.8% | +4.6e-08 | R |
| wcte__jepa_repaired__s9000__matched_init | dev | 0.0681 | -8.6e-09 | +1.8e-06 | +8.5e-04 | -20.1% | +1.4e-07 | R |
| wcte__jepa_repaired__s9000__matched_init | sealed | 0.0688 | +4.7e-08 | +1.7e-07 | +2.7e-04 | -18.9% | +4.3e-08 | R |
| wcte__jepa_repaired__s9001__final | dev | 0.0612 | +1.0e-07 | +1.2e-06 | +8.5e-04 | +3.8% | +1.4e-07 | V |
| wcte__jepa_repaired__s9001__final | sealed | 0.0623 | +1.1e-07 | -5.4e-07 | +2.7e-04 | +3.7% | +4.7e-08 | V |
| wcte__jepa_repaired__s9001__matched_init | dev | 0.0702 | -1.4e-07 | -1.7e-07 | +8.5e-04 | -14.9% | +1.4e-07 | V |
| wcte__jepa_repaired__s9001__matched_init | sealed | 0.0711 | -2.2e-07 | -1.1e-06 | +2.7e-04 | -13.9% | +4.5e-08 | V |
| wcte__mae_mse__s9000__final | dev | 0.0575 | +2.2e-07 | -2.1e-07 | +8.5e-04 | -14.2% | +1.4e-07 | V |
| wcte__mae_mse__s9000__final | sealed | 0.0583 | +1.2e-07 | -7.0e-07 | +2.7e-04 | -13.8% | +4.1e-08 | V |
| wcte__mae_mse__s9000__matched_init | dev | 0.0656 | -7.7e-09 | +2.1e-06 | +8.5e-04 | -19.2% | +1.4e-07 | R |
| wcte__mae_mse__s9000__matched_init | sealed | 0.0664 | +4.9e-08 | +1.4e-07 | +2.7e-04 | -18.1% | +4.4e-08 | R |
| wcte__mae_mse__s9001__final | dev | 0.0564 | +4.9e-08 | +9.4e-07 | +8.5e-04 | -23.5% | +1.5e-07 | R |
| wcte__mae_mse__s9001__final | sealed | 0.0574 | +9.6e-08 | -7.5e-07 | +2.7e-04 | -23.5% | +4.7e-08 | R |
| wcte__mae_mse__s9001__matched_init | dev | 0.0662 | -2.0e-07 | -2.7e-06 | +8.5e-04 | -12.2% | +1.5e-07 | V |
| wcte__mae_mse__s9001__matched_init | sealed | 0.0673 | +1.2e-08 | +2.4e-07 | +2.7e-04 | -11.0% | +4.7e-08 | V |
| wcte__mae_repaired__s9000__final | dev | 0.0521 | +2.4e-08 | -3.7e-07 | +8.5e-04 | -13.4% | +1.4e-07 | V |
| wcte__mae_repaired__s9000__final | sealed | 0.0533 | +5.4e-08 | -2.7e-08 | +2.7e-04 | -13.4% | +4.1e-08 | V |
| wcte__mae_repaired__s9000__matched_init | dev | 0.0656 | -7.7e-09 | +2.1e-06 | +8.5e-04 | -19.2% | +1.4e-07 | R |
| wcte__mae_repaired__s9000__matched_init | sealed | 0.0664 | +5.5e-08 | +1.0e-07 | +2.7e-04 | -18.1% | +4.4e-08 | R |
| wcte__mae_repaired__s9001__final | dev | 0.0519 | +1.1e-07 | -4.2e-07 | +8.5e-04 | -6.5% | +1.4e-07 | V |
| wcte__mae_repaired__s9001__final | sealed | 0.0525 | +4.4e-08 | -4.9e-07 | +2.7e-04 | -6.3% | +4.7e-08 | V |
| wcte__mae_repaired__s9001__matched_init | dev | 0.0662 | -2.0e-07 | -2.7e-06 | +8.5e-04 | -12.2% | +1.5e-07 | V |
| wcte__mae_repaired__s9001__matched_init | sealed | 0.0673 | -8.3e-09 | +3.4e-07 | +2.7e-04 | -11.0% | +4.7e-08 | V |
| wcte__sigreg_jepa__s9000__final | dev | 0.1525 | +2.5e-08 | -1.2e-06 | +8.5e-04 | -7.7% | +2.1e-07 | V |
| wcte__sigreg_jepa__s9000__final | sealed | 0.1526 | +2.0e-08 | +5.4e-07 | +2.7e-04 | -5.9% | +1.0e-07 | V |
| wcte__sigreg_jepa__s9000__matched_init | dev | 0.0656 | -7.7e-09 | +2.1e-06 | +8.5e-04 | -19.2% | +1.4e-07 | R |
| wcte__sigreg_jepa__s9000__matched_init | sealed | 0.0664 | +5.5e-08 | +1.0e-07 | +2.7e-04 | -18.1% | +4.4e-08 | R |
| sk__legacy_init__s9000 | dev | 0.5274 | -9.6e-08 | -9.7e-08 | -1.0e-04 | -40.5% | +9.1e-08 | R |
| sk__legacy_init__s9000 | sealed | 0.5433 | -5.9e-08 | -1.3e-06 | -8.6e-04 | -39.1% | +7.7e-08 | R |
| sk__legacy_init__s9001 | dev | 0.5258 | +2.8e-07 | -1.3e-05 | -1.0e-04 | -35.3% | +1.9e-08 | R |
| sk__legacy_init__s9001 | sealed | 0.5407 | +3.9e-08 | +8.7e-06 | -8.6e-04 | -33.6% | +1.3e-09 | R |
| sk__jepa_repaired__s9000__final | dev | 0.4491 | +7.8e-08 | +1.6e-06 | -1.0e-04 | +2.5% | +1.1e-06 | V |
| sk__jepa_repaired__s9000__final | sealed | 0.4600 | -3.9e-09 | +1.4e-06 | -8.6e-04 | +3.7% | +1.1e-06 | V |
| sk__jepa_repaired__s9000__matched_init | dev | 0.5363 | +3.9e-08 | +4.6e-06 | -1.0e-04 | -37.3% | +1.5e-07 | R |
| sk__jepa_repaired__s9000__matched_init | sealed | 0.5526 | -2.5e-07 | -1.5e-06 | -8.6e-04 | -35.8% | +1.4e-07 | R |
| sk__mae_mse__s9000__final | dev | 0.4266 | +1.4e-08 | -4.7e-06 | -1.0e-04 | -49.0% | +9.8e-07 | R |
| sk__mae_mse__s9000__final | sealed | 0.4375 | +1.0e-07 | -5.4e-06 | -8.6e-04 | -48.8% | +9.6e-07 | R |
| sk__mae_mse__s9000__matched_init | dev | 0.5274 | -9.6e-08 | -9.7e-08 | -1.0e-04 | -40.5% | +9.1e-08 | R |
| sk__mae_mse__s9000__matched_init | sealed | 0.5433 | -5.9e-08 | -1.3e-06 | -8.6e-04 | -39.1% | +7.7e-08 | R |
| sk__mae_repaired__s9000__final | dev | 0.3564 | +2.0e-07 | -2.6e-08 | -1.0e-04 | -11.0% | +5.6e-07 | V |
| sk__mae_repaired__s9000__final | sealed | 0.3681 | +1.5e-07 | +2.3e-08 | -8.6e-04 | -11.7% | +5.4e-07 | V |
| sk__mae_repaired__s9000__matched_init | dev | 0.5301 | -3.2e-07 | -3.4e-06 | -1.0e-04 | -36.4% | +1.4e-07 | R |
| sk__mae_repaired__s9000__matched_init | sealed | 0.5477 | -1.3e-07 | -2.8e-06 | -8.6e-04 | -34.9% | +1.3e-07 | R |
| sk__sigreg_jepa__s9000__final | dev | 1.1512 | -2.3e-07 | +2.9e-06 | -1.0e-04 | -10.3% | +9.3e-07 | V |
| sk__sigreg_jepa__s9000__final | sealed | 1.1545 | -1.5e-07 | -2.3e-07 | -8.6e-04 | -7.5% | +9.2e-07 | V |
| sk__sigreg_jepa__s9000__matched_init | dev | 0.5274 | -9.6e-08 | -9.7e-08 | -1.0e-04 | -40.5% | +9.1e-08 | R |
| sk__sigreg_jepa__s9000__matched_init | sealed | 0.5433 | -5.9e-08 | -1.3e-06 | -8.6e-04 | -39.1% | +7.7e-08 | R |
| juno__legacy_init__s9000 | dev | 0.4232 | -4.6e-07 | -1.1e-05 | +1.8e-03 | -45.2% | +3.1e-08 | R |
| juno__legacy_init__s9000 | sealed | 0.4358 | -3.7e-07 | -3.4e-06 | +6.1e-04 | -44.1% | +1.7e-08 | R |
| juno__legacy_init__s9001 | dev | 0.4228 | -8.6e-08 | +1.1e-05 | +1.8e-03 | -38.0% | -2.3e-08 | R |
| juno__legacy_init__s9001 | sealed | 0.4357 | -1.3e-07 | -2.0e-05 | +6.1e-04 | -36.5% | -3.4e-08 | R |
| juno__jepa_repaired__s9000__final | dev | 0.3786 | +6.0e-08 | -1.3e-06 | +1.8e-03 | -6.1% | +1.2e-06 | V |
| juno__jepa_repaired__s9000__final | sealed | 0.3866 | +7.1e-08 | -4.3e-07 | +6.1e-04 | -4.9% | +1.2e-06 | V |
| juno__jepa_repaired__s9000__matched_init | dev | 0.4396 | -4.7e-07 | +2.4e-06 | +1.8e-03 | -42.5% | +6.6e-08 | R |
| juno__jepa_repaired__s9000__matched_init | sealed | 0.4504 | -2.0e-08 | -3.9e-06 | +6.1e-04 | -41.4% | +5.2e-08 | R |
| juno__mae_repaired__s9000__final | dev | 0.2876 | -3.6e-08 | -4.6e-07 | +1.8e-03 | -6.4% | +4.1e-07 | V |
| juno__mae_repaired__s9000__final | sealed | 0.2919 | +6.4e-08 | -2.5e-06 | +6.1e-04 | -7.4% | +3.8e-07 | V |
| juno__mae_repaired__s9000__matched_init | dev | 0.4372 | -2.7e-07 | +5.1e-06 | +1.8e-03 | -40.6% | +5.6e-08 | R |
| juno__mae_repaired__s9000__matched_init | sealed | 0.4475 | +1.3e-07 | +4.1e-06 | +6.1e-04 | -39.4% | +4.0e-08 | R |
| hk__legacy_init__s9000 | dev | 0.9072 | -2.1e-07 | -1.4e-05 | -1.5e-03 | -41.3% | +7.8e-08 | R |
| hk__legacy_init__s9000 | sealed | 0.9321 | -1.1e-07 | -7.5e-06 | -7.8e-04 | -40.0% | +7.3e-08 | R |
| hk__legacy_init__s9001 | dev | 0.8967 | -3.7e-07 | +4.5e-06 | -1.5e-03 | -39.7% | +9.1e-09 | R |
| hk__legacy_init__s9001 | sealed | 0.9211 | -1.5e-07 | +1.1e-06 | -7.8e-04 | -37.9% | +5.1e-09 | R |
| hk__jepa_repaired__s9000__final | dev | 0.8119 | +1.2e-07 | -5.6e-05 | -1.5e-03 | -4.4% | +1.3e-06 | V |
| hk__jepa_repaired__s9000__final | sealed | 0.8374 | -1.6e-07 | -3.5e-05 | -7.8e-04 | -2.5% | +1.3e-06 | V |
| hk__jepa_repaired__s9000__matched_init | dev | 0.9242 | -1.6e-08 | -2.0e-05 | -1.5e-03 | -42.1% | +8.4e-08 | R |
| hk__jepa_repaired__s9000__matched_init | sealed | 0.9517 | -5.2e-08 | -1.6e-05 | -7.8e-04 | -40.8% | +7.5e-08 | R |
| hk__mae_repaired__s9000__final | dev | 0.6247 | +5.2e-07 | +1.1e-06 | -1.5e-03 | -12.7% | +4.6e-07 | V |
| hk__mae_repaired__s9000__final | sealed | 0.6473 | +4.5e-07 | +9.3e-07 | -7.8e-04 | -13.6% | +4.4e-07 | V |
| hk__mae_repaired__s9000__matched_init | dev | 0.9365 | -1.1e-07 | -7.8e-06 | -1.5e-03 | -37.9% | +1.5e-07 | R |
| hk__mae_repaired__s9000__matched_init | sealed | 0.9542 | -7.8e-08 | +1.3e-06 | -7.8e-04 | -36.4% | +1.4e-07 | R |

## Sealed test (recomputed; trained finals)

| Case | (a) m | (c) m | (c) sealed / dev |
|---|---|---|---|
| hk__jepa_repaired__s9000__final | 5.6382 | 0.8374 | 1.031 |
| hk__mae_repaired__s9000__final | 1.6216 | 0.6473 | 1.036 |
| juno__jepa_repaired__s9000__final | 2.0092 | 0.3866 | 1.021 |
| juno__mae_repaired__s9000__final | 0.7397 | 0.2919 | 1.015 |
| sk__jepa_repaired__s9000__final | 2.4836 | 0.4600 | 1.024 |
| sk__mae_mse__s9000__final | 2.0958 | 0.4375 | 1.026 |
| sk__mae_repaired__s9000__final | 0.8669 | 0.3681 | 1.033 |
| sk__sigreg_jepa__s9000__final | 4.9703 | 1.1545 | 1.003 |
| wcte__jepa_repaired__s9000__final | 0.1948 | 0.0636 | 1.011 |
| wcte__jepa_repaired__s9001__final | 0.1466 | 0.0623 | 1.019 |
| wcte__mae_mse__s9000__final | 0.1540 | 0.0583 | 1.014 |
| wcte__mae_mse__s9001__final | 0.1584 | 0.0574 | 1.018 |
| wcte__mae_repaired__s9000__final | 0.1414 | 0.0533 | 1.023 |
| wcte__mae_repaired__s9001__final | 0.1370 | 0.0525 | 1.010 |
| wcte__sigreg_jepa__s9000__final | 0.2682 | 0.1526 | 1.000 |

## Discrepancies and diagnosis

- **f_event** (10 of 27 outside 15 %): definitional. r2-tier1 reports the per-dimension mean of Var_e/Var_{p,e} over all test photon bombs (`ext_readout.layout_fractions`); runbook §1 points to `verify/k2_collapse.py`, the trace ratio over the first 200 events. Recomputed from the same dumps under their definition, all 27 agree to < 1e-4 (`va3_fevent_diagnostic.json`). J4 holds under both.
- (a), (c), (d): no discrepancy; (c) agrees to < 1e-6, i.e. the same encoder outputs and the same Ridge.

## Coverage and inputs

- Cases: the 38 cases of VERIFY_INPUTS_A map one-to-one onto my independently discovered list (46 cases; the 8 extra WCTE time-scale cases are exploratory). Manifest finals are `snapshot_step_10000.pt`, mine `checkpoint.pt`: all tensors equal. Matched-init SHA256, scalings and sealed-pack SHA256 identical (`va1_reconcile.json`, `va1_ckpt_equality.json`). K0: my feature rebuild equals the repository tokeniser bit for bit on 200 events per detector (`k0_lucid.json`).
- HK and JUNO sealed packs appeared after VERIFY_INPUTS_A (listed null). They were computed separately and sealed as **A2** (`3ecf960`) before their RESCORE rows were read.

- f_event under the runbook definition is computed once per case on the first 200 dev_probe_test photon bombs; for sealed rows Δf_event compares that value with the other phase's sealed-cohort value, while the "their def." column uses my sealed dumps.

## Verdict

**V** for (a), (c) and (d) in every case and test set (TOY_VS_LUCID: 27 cases, RESCORE_LUCID: 76 rows; max abs. rel. diff 5e-7 for (c), 6e-5 for (a), 2e-3 for (d)). All nine registered invariants pass, sealed_test included on all four detectors. **R** for f_event: the cause is a diagnosed definitional difference (per-dimension mean vs trace ratio, 200 vs all events). Under the other phase's definition, my dumps reproduce all 76 values to 1.3e-6. Arm orderings for (a) and (c) are identical. f_event ordering differs once, WCTE s9000 jepa_repaired vs mae_repaired, and follows the definition. No flag.

## Independence (part A)

Before seal A (`8b5d2aa`) I read only the runbook, my own `verify/` code, the allowed black-box repository code, the data packs and manifests, checkpoints with the identity, encoder and scaling recipe keys, and, from origin/tier1, VERIFY_INPUTS_A.json, READOUT_SPEC_v1.md and the PREREG file. One disclosure: READOUT_SPEC_v1.md (an allowed file) contains the other phase's validation values for `wcte__mae_repaired__s9000` on dev_probe_test ((c) 0.0521, (a) 0.1410, (d) 0.0752, f_event 0.1852). My dev numbers had already been computed (16:17Z, 22 Nov) when I read it (19:49Z), so it could not influence them. After seal A I opened the rebuilt TOY_VS_LUCID.json, RESCORE_LUCID.json (except the HK/JUNO sealed rows, until seal A2) and, for the f_event diagnosis, `ext_readout.layout_fractions`. No closed file was opened before its seal.

