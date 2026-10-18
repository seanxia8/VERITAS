## 2×2 diagnostic (validation split only; mean [min–max] over seeds 30–32; model ÷ constant)

| Arm | Vertex RMSE, pooled | Vertex, worst energy | Direction (°) | E_dep MAE, pooled | Seeds passing all gates |
|---|---|---|---|---|---|
| 20 ns, dark on | 0.996 [0.991–1.004] | 1.048 [1.023–1.083] | 9.924 [9.566–10.310] | 0.661 [0.591–0.765] | 0/3 |
| 20 ns, dark off | 0.948 [0.936–0.955] | 0.988 [0.962–1.002] | 10.138 [9.404–10.918] | 0.480 [0.460–0.491] | 0/3 |
| 1 µs, dark on | 0.964 [0.949–0.981] | 1.001 [0.980–1.015] | 9.105 [8.664–9.366] | 0.633 [0.550–0.693] | 0/3 |
| 1 µs, dark off | 0.969 [0.957–0.978] | 0.997 [0.972–1.025] | 9.450 [9.236–9.586] | 0.526 [0.514–0.544] | 0/3 |

## Data limitation (20 ns, dark on; validation split only)

| Training events per energy | Training placements | Vertex RMSE, pooled | E_dep MAE, pooled |
|---|---|---|---|
| 4 of 8 | 96 | 1.023 [1.017–1.031] | 0.888 [0.813–1.005] |
| 6 of 8 | 144 | 1.016 [1.002–1.035] | 0.810 [0.741–0.900] |
| 8 of 8 | 192 | 0.996 [0.991–1.004] | 0.661 [0.591–0.765] |

## R-4b beside R-4 (test split; R-4 stays as recorded)

| Run | Seed | Vertex, pooled | Vertex, worst energy | Direction (°) | E_dep, pooled | E_dep, worst energy | Verdict |
|---|---|---|---|---|---|---|---|
| R-4 (1 µs) | 30 | 0.965 | 0.973 | 9.0 | 0.644 | 0.785 | FAIL: pooled vertex RMSE / constant: 0.964511 > 0.8 |
| R-4 (1 µs) | 31 | 0.960 | 0.974 | 9.6 | 0.672 | 0.731 | FAIL: pooled vertex RMSE / constant: 0.960455 > 0.8 |
| R-4 (1 µs) | 32 | 0.998 | 1.033 | 9.5 | 0.877 | 1.315 | FAIL: pooled vertex RMSE / constant: 0.998417 > 0.8; pooled deposited_energy_in_volume relative MAE / constant: 0.876935 > 0.8; subgroup 1500 vertex RMSE / constant: 1.03316 > 1.0; subgroup 500 deposited_energy_in_volume relative MAE / constant: 1.31515 > 1.0 |
| R-4b (20 ns rule) | 30 | 0.992 | 1.017 | 10.0 | 0.943 | 1.125 | FAIL: pooled vertex RMSE / constant: 0.991553 > 0.8; pooled deposited_energy_in_volume relative MAE / constant: 0.943011 > 0.8; subgroup 1000 vertex RMSE / constant: 1.00497 > 1.0; subgroup 1500 vertex RMSE / constant: 1.01659 > 1.0; subgroup 500 deposited_energy_in_volume relative MAE / constant: 1.12513 > 1.0 |
| R-4b (20 ns rule) | 31 | 0.998 | 1.005 | 10.1 | 0.673 | 0.988 | FAIL: pooled vertex RMSE / constant: 0.997651 > 0.8; subgroup 1500 vertex RMSE / constant: 1.00494 > 1.0 |
| R-4b (20 ns rule) | 32 | 1.000 | 1.019 | 10.3 | 0.687 | 0.937 | FAIL: pooled vertex RMSE / constant: 0.999512 > 0.8; subgroup 500 vertex RMSE / constant: 1.01852 > 1.0 |
