# E1a extension: pure-noise and pure-physics cells (registered: false)

Author request 24 Oct 2026. Script `scripts/jepa_e1_ext_pure_cells.py`; registered row runner `run.run_e1a_row` unchanged; n = 1e5, seeds 0-4. Predictions are in the script docstring and were written before the run.

Max principal angle to R_c (deg), mean ± SD over 5 seeds; then nuisance R² (eta) and mean physics R² from the registered ridge probes.

| family | theta | eta | gamma | CCA (linear JEPA) | Context PCA (linear MAE) | Whitened PCA (Paper 1) | Noise-only GEVP | beta 0.1 | beta 10 | CCA nuisance R² | CCA physics R² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noise_only | 0 | 1 | 0 | 85.4 ± 5.8 | 90.0 ± 0.0 | 80.8 ± 3.7 | 83.4 ± 4.4 | 83.8 ± 5.1 | 83.4 ± 4.4 | 0.07 | – |
| noise_only | 0 | 1 | 0.1 | 87.8 ± 1.8 | 89.9 ± 0.0 | 83.9 ± 3.9 | 87.1 ± 1.8 | 87.4 ± 2.3 | 86.5 ± 3.3 | 0.98 | – |
| noise_only | 0 | 1 | 0.25 | 86.5 ± 2.9 | 89.9 ± 0.0 | 87.5 ± 1.7 | 82.3 ± 5.7 | 87.9 ± 1.3 | 83.8 ± 1.9 | 0.98 | – |
| noise_only | 0 | 1 | 0.5 | 87.6 ± 2.2 | 89.9 ± 0.0 | 82.5 ± 4.8 | 87.1 ± 2.7 | 86.9 ± 2.1 | 87.1 ± 2.2 | 0.98 | – |
| noise_only | 0 | 1 | 0.75 | 87.3 ± 2.1 | 89.9 ± 0.0 | 82.1 ± 7.3 | 81.9 ± 5.7 | 87.3 ± 1.5 | 82.3 ± 4.5 | 0.98 | – |
| noise_only | 0 | 1 | 0.9 | 88.9 ± 0.9 | 89.9 ± 0.0 | 86.3 ± 1.7 | 86.6 ± 2.9 | 87.8 ± 1.8 | 84.6 ± 3.9 | 0.98 | – |
| noise_only | 0 | 1 | 1 | 88.2 ± 1.0 | 89.9 ± 0.0 | 85.6 ± 2.4 | 83.0 ± 3.8 | 88.1 ± 1.7 | 85.4 ± 2.1 | 0.98 | – |
| noise_only | 0 | 3 | 0 | 84.3 ± 4.5 | 90.0 ± 0.0 | 84.9 ± 2.4 | 85.4 ± 3.7 | 84.3 ± 1.4 | 85.4 ± 3.6 | 0.07 | – |
| noise_only | 0 | 3 | 0.1 | 86.3 ± 2.1 | 90.0 ± 0.0 | 86.0 ± 2.5 | 88.0 ± 1.3 | 88.4 ± 1.0 | 85.3 ± 5.5 | 1.00 | – |
| noise_only | 0 | 3 | 0.25 | 88.4 ± 0.9 | 90.0 ± 0.0 | 82.2 ± 3.9 | 85.4 ± 3.8 | 87.2 ± 2.3 | 85.6 ± 3.7 | 1.00 | – |
| noise_only | 0 | 3 | 0.5 | 87.9 ± 1.3 | 90.0 ± 0.0 | 84.1 ± 3.9 | 87.4 ± 2.6 | 87.8 ± 1.4 | 82.7 ± 6.2 | 1.00 | – |
| noise_only | 0 | 3 | 0.75 | 87.8 ± 2.2 | 90.0 ± 0.0 | 83.7 ± 2.3 | 87.1 ± 1.9 | 88.3 ± 1.2 | 86.8 ± 2.5 | 1.00 | – |
| noise_only | 0 | 3 | 0.9 | 87.1 ± 2.2 | 90.0 ± 0.0 | 85.3 ± 2.7 | 85.8 ± 5.6 | 88.0 ± 1.5 | 84.6 ± 3.1 | 1.00 | – |
| noise_only | 0 | 3 | 1 | 88.4 ± 1.0 | 90.0 ± 0.0 | 85.8 ± 4.3 | 85.9 ± 2.0 | 88.8 ± 1.6 | 86.3 ± 2.5 | 1.00 | – |
| physics_only | 0.1 | 0 | 0 | 3.7 ± 0.4 | 88.8 ± 0.6 | 3.3 ± 0.7 | 5.4 ± 0.9 | 3.1 ± 0.3 | 5.2 ± 0.9 | – | 0.33 |
| physics_only | 0.1 | 0 | 1 | 3.8 ± 0.3 | 88.8 ± 0.6 | 3.2 ± 0.7 | 5.3 ± 0.9 | 3.1 ± 0.4 | 5.1 ± 0.9 | – | 0.32 |
| physics_only | 0.4 | 0 | 0 | 1.7 ± 0.2 | 53.2 ± 5.4 | 0.5 ± 0.0 | 1.3 ± 0.1 | 1.6 ± 0.2 | 1.2 ± 0.1 | – | 0.88 |
| physics_only | 0.4 | 0 | 1 | 1.6 ± 0.2 | 53.0 ± 4.9 | 0.5 ± 0.1 | 1.5 ± 0.2 | 1.4 ± 0.2 | 1.3 ± 0.2 | – | 0.88 |
| physics_only | 1 | 0 | 0 | 1.6 ± 0.1 | 40.8 ± 1.1 | 0.2 ± 0.0 | 1.1 ± 0.1 | 1.4 ± 0.1 | 1.0 ± 0.1 | – | 0.98 |
| physics_only | 1 | 0 | 1 | 1.5 ± 0.0 | 40.8 ± 1.1 | 0.2 ± 0.0 | 1.2 ± 0.1 | 1.4 ± 0.0 | 1.1 ± 0.1 | – | 0.98 |
| physics_only_lownoise | 0.4 | 0 | 0 | 1.6 ± 0.2 | 39.2 ± 1.2 | 0.0 ± 0.0 | 1.1 ± 0.1 | 1.5 ± 0.2 | 1.0 ± 0.1 | – | 1.00 |
| physics_only_lownoise | 0.4 | 0 | 1 | 1.4 ± 0.1 | 39.2 ± 1.2 | 0.0 ± 0.0 | 1.2 ± 0.1 | 1.3 ± 0.1 | 1.1 ± 0.1 | – | 1.00 |

## Predictions vs outcome

- **N1 (confirmed):** in noise-only data CCA's nuisance R² is 0.07 at gamma = 0 and 0.98-1.00 from gamma = 0.1 on.
- **N2 (confirmed):** no method recovers R_c without physics: every angle is 80-90 deg.
- **N3 (confirmed):** the noise-only GEVP is degenerate (81-88 deg, seed SD up to 6 deg) and ignores eta (R² ≈ 0.05). Finite beta = 10 still picks up eta (R² 0.51-0.94): with k forced to 3 and nothing else to predict, the shared factor is the cheapest gain.
- **P1 (confirmed):** physics-only CCA is at the sampling floor (1.5-3.8 deg) for gamma = 0 and 1 alike.
- **P2 (partly):** removing eta brings context PCA from ~90 deg to 53 deg (theta 0.4) and 41 deg (theta 1), but not at theta 0.1 (89 deg), where the private noise still dominates the variance.
- **P3 (confirmed):** whitened PCA, GEVP and beta values at the floor.
- **L1 (confirmed):** with 1% private noise every method's physics R² is 1.00; PCA's 39 deg angle then reflects R_c ≠ span(A_c) under anisotropic noise, not lost physics.
