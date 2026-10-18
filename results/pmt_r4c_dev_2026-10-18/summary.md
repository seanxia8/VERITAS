## Arms (validation split only; seeds 20–22; mean [min–max]; model ÷ constant)

| Arm | Time origin | Energy target | Vertex RMSE, pooled | Direction (°) | Energy MAE, pooled | Best epochs | Checkpoints = 56179eb |
|---|---|---|---|---|---|---|---|
| arm0 | first digit | initial | 0.782 [0.725–0.850] | 7.452 [7.070–7.892] | 0.823 [0.803–0.845] | 49, 60, 48 | 3/3 |
| arm1 | trigger | initial | 0.681 [0.654–0.715] | 6.805 [6.466–7.362] | 0.857 [0.812–0.911] | 87, 56, 57 | 0/3 |
| arm2 | trigger | deposited | 0.652 [0.620–0.707] | 6.824 [6.567–7.169] | 0.441 [0.395–0.480] | 85, 98, 96 | 0/3 |
| arm3 | window start | initial | 0.961 [0.949–0.975] | 9.756 [9.060–10.317] | 0.831 [0.762–0.876] | 96, 51, 98 | 0/3 |
| arm4 | window start | deposited | 0.936 [0.933–0.938] | 10.159 [9.394–11.459] | 0.496 [0.483–0.511] | 80, 91, 83 | 0/3 |
| 56179eb checkpoints (reference) | first digit | initial | 0.782 [0.725–0.850] | 7.452 [7.070–7.892] | 0.823 [0.803–0.845] | — | — |

## Target scale (training split; constant-predictor loss per head, vertex in 0.35 m units)

| Arm | Direction | Vertex | Energy | SD of log-energy target | Energy share of constant loss |
|---|---|---|---|---|---|
| arm0 | 0.332 | 0.331 | 0.206 | 0.454 | 0.24 |
| arm1 | 0.332 | 0.331 | 0.206 | 0.454 | 0.24 |
| arm2 | 0.332 | 0.331 | 0.035 | 0.188 | 0.05 |
| arm3 | 0.332 | 0.331 | 0.206 | 0.454 | 0.24 |
| arm4 | 0.332 | 0.331 | 0.035 | 0.188 | 0.05 |

## Per-head loss at initialisation (mean over seeds; validation loss ÷ constant loss, and share of the summed training loss; encoder gradient norm)

| Arm | Vertex val ÷ const | Energy val ÷ const | Direction val | Train share v / e / d | Encoder grad norm v / e / d |
|---|---|---|---|---|---|
| arm0 | 1.193 | 1.098 | 0.674 | 0.31 / 0.17 / 0.52 | 0.54 / 0.198 / 0.537 |
| arm1 | 1.196 | 1.097 | 0.675 | 0.31 / 0.17 / 0.52 | 0.544 / 0.198 / 0.54 |
| arm2 | 1.196 | 43.753 | 0.675 | 0.17 / 0.54 / 0.29 | 0.544 / 1.76 / 0.54 |
| arm3 | 1.432 | 1.080 | 0.668 | 0.35 / 0.16 / 0.49 | 2.61 / 0.554 / 0.764 |
| arm4 | 1.432 | 41.973 | 0.668 | 0.21 / 0.51 / 0.28 | 2.61 / 5.19 / 0.764 |

## Per-head loss at the selected checkpoint (mean over seeds; validation loss ÷ constant loss, and share of the summed training loss; encoder gradient norm)

| Arm | Vertex val ÷ const | Energy val ÷ const | Direction val | Train share v / e / d | Encoder grad norm v / e / d |
|---|---|---|---|---|---|
| arm0 | 0.614 | 0.813 | 0.007 | 0.53 / 0.45 / 0.03 | 0.489 / 0.386 / 0.114 |
| arm1 | 0.465 | 0.846 | 0.006 | 0.47 / 0.50 / 0.03 | 0.792 / 0.557 / 0.133 |
| arm2 | 0.427 | 0.197 | 0.006 | 0.81 / 0.14 / 0.05 | 0.724 / 0.0868 / 0.0637 |
| arm3 | 0.924 | 0.807 | 0.012 | 0.62 / 0.36 / 0.03 | 0.609 / 0.396 / 0.343 |
| arm4 | 0.877 | 0.274 | 0.013 | 0.92 / 0.04 / 0.05 | 0.56 / 0.216 / 0.388 |

## Input comparisons

| Comparison | x identical | x columns identical | y identical | y columns identical |
|---|---|---|---|---|
| inputs_arm0_vs_56179eb | True | — | True | — |
| inputs_arm1_vs_arm3 | False | [True, False, True, True, True, True] | True | — |
| inputs_arm3_vs_r4b_u20_nodark | True | — | False | [True, True, True, True, True, True, False] |
