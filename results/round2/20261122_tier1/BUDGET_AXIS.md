# BUDGET_AXIS — vertex error against training step (r2-tier1 T4, exploratory)

Spatial-moment Ridge readout (READOUT_SPEC_v1), `dev_probe_test` photon bombs, snapshots of each run. Exploratory, not registered (§2.6), except the sealed 5,000/10,000-step values that feed F5.

| Run | 0 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|
| WCTE 450 ns (`wcte__mae_repaired__s9000__ts450`) | 0.0656 | 0.0616 | 0.0596 | 0.0583 | 0.0563 | 0.0541 |
| WCTE 50 ns (`wcte__mae_repaired__s9000`) | 0.0656 | 0.0612 | 0.0587 | 0.0570 | 0.0544 | 0.0521 |
| WCTE 15 ns (`wcte__mae_repaired__s9000__ts15`) | 0.0656 | 0.0600 | 0.0585 | 0.0564 | 0.0525 | 0.0512 |
| WCTE 5 ns (`wcte__mae_repaired__s9000__ts5`) | 0.0656 | 0.0607 | 0.0577 | 0.0551 | 0.0504 | 0.0501 |
| WCTE 1.5 ns (`wcte__mae_repaired__s9000__ts1p5`) | 0.0656 | 0.0615 | 0.0585 | 0.0552 | 0.0495 | 0.0485 |
| SK mae_repaired 50 ns A (`sk__mae_repaired__s9000`) | 0.5301 | 0.5109 | 0.4867 | 0.4481 | 0.3798 | 0.3564 |
| SK mae_mse A (`sk__mae_mse__s9000`) | 0.5274 | 0.5143 | 0.5083 | 0.5051 | 0.4687 | 0.4266 |
| SK jepa_repaired A (`sk__jepa_repaired__s9000`) | 0.5363 | 0.5299 | 0.5196 | 0.5067 | 0.4619 | 0.4491 |
| SK sigreg_jepa A (`sk__sigreg_jepa__s9000`) | 0.5274 | 0.5879 | 0.6215 | 0.6966 | 0.8752 | 1.1512 |
| SK patch 50 ns B (SK-5) (`sk__mae_repaired__s9001`) | – | – | – | – | 0.3709 | 0.3472 |
| SK patch 450 ns A (SK-1) (`sk__mae_repaired__s9000__ts450`) | – | – | – | – | 0.4122 | 0.3620 |
| SK patch 450 ns B (SK-4) (`sk__mae_repaired__s9001__ts450`) | – | – | – | – | 0.4009 | 0.3687 |
| SK patch 5 ns A (SK-8) (`sk__mae_repaired__s9000__ts5`) | – | – | – | – | 0.3809 | 0.3577 |
| SK patch 5 ns B (SK-9) (`sk__mae_repaired__s9001__ts5`) | – | – | – | – | 0.3823 | 0.3538 |

Sealed (F5 inputs):

| Run | 5,000 | 10,000 |
|---|---:|---:|
| WCTE 450 ns | – | 0.0548 |
| WCTE 50 ns | – | 0.0533 |
| WCTE 15 ns | – | 0.0521 |
| WCTE 5 ns | – | 0.0510 |
| WCTE 1.5 ns | – | 0.0497 |
| SK mae_repaired 50 ns A | 0.3870 | 0.3681 |
| SK mae_mse A | – | 0.4375 |
| SK jepa_repaired A | – | 0.4600 |
| SK sigreg_jepa A | – | 1.1545 |
| SK patch 50 ns B (SK-5) | 0.3835 | 0.3534 |
| SK patch 450 ns A (SK-1) | 0.4211 | 0.3709 |
| SK patch 450 ns B (SK-4) | 0.4131 | 0.3755 |
| SK patch 5 ns A (SK-8) | 0.3887 | 0.3678 |
| SK patch 5 ns B (SK-9) | 0.3961 | 0.3607 |

Figure: `BUDGET_AXIS.png`.


Reading (exploratory): on WCTE the time-unit ordering (finer unit → lower error) appears from ≈2,000 steps and widens to 10,000; on SK the patch runs at 450 and 5 ns separate more at 5,000 steps than at 10,000 (the F5 comparison is computed in `GATE_TABLE`). `sigreg_jepa` on SK degrades monotonically with training (0.53 → 1.15 m), the collapse already reported in r2-lucid. The SK snapshots 0–2,000 of the E-cells were not scored (budget); their 5,000- and 10,000-step values come from the full E-cell scores.
