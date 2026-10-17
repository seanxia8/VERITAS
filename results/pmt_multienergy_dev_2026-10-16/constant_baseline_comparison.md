# Constant-predictor comparison, training_01 (runbook v2 R-3)

| Seed | Subgroup | n | Vertex RMSE model / constant (m) | Ratio | Direction mean model / constant (deg) | Ratio | Initial-energy relative MAE model / constant | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | pooled | 96 | 0.2927 / 0.3697 | 0.7918 | 7.4121 / 92.8001 | 0.0799 | 0.4176 / 0.4343 | 0.9615 |
| 20 | 500 MeV | 32 | 0.2575 / 0.3612 | 0.7127 | 7.2727 / 88.8554 | 0.0818 | 0.7304 / 0.8171 | 0.8939 |
| 20 | 1000 MeV | 32 | 0.2848 / 0.3513 | 0.8106 | 6.7752 / 95.6979 | 0.0708 | 0.2710 / 0.0914 | 2.9639 |
| 20 | 1500 MeV | 32 | 0.3312 / 0.3952 | 0.8381 | 8.1884 / 93.8470 | 0.0873 | 0.2512 / 0.3943 | 0.6372 |
| 21 | pooled | 96 | 0.2685 / 0.3697 | 0.7262 | 7.1888 / 92.8001 | 0.0775 | 0.3903 / 0.4343 | 0.8988 |
| 21 | 500 MeV | 32 | 0.2548 / 0.3612 | 0.7054 | 6.9042 / 88.8554 | 0.0777 | 0.6855 / 0.8171 | 0.8389 |
| 21 | 1000 MeV | 32 | 0.2569 / 0.3513 | 0.7312 | 6.4995 / 95.6979 | 0.0679 | 0.2470 / 0.0914 | 2.7008 |
| 21 | 1500 MeV | 32 | 0.2921 / 0.3952 | 0.7392 | 8.1628 / 93.8470 | 0.0870 | 0.2386 / 0.3943 | 0.6050 |
| 22 | pooled | 96 | 0.3259 / 0.3697 | 0.8815 | 8.4986 / 92.8001 | 0.0916 | 0.4120 / 0.4343 | 0.9487 |
| 22 | 500 MeV | 32 | 0.2844 / 0.3612 | 0.7873 | 7.5965 / 88.8554 | 0.0855 | 0.7765 / 0.8171 | 0.9503 |
| 22 | 1000 MeV | 32 | 0.3213 / 0.3513 | 0.9148 | 9.0678 / 95.6979 | 0.0948 | 0.2213 / 0.0914 | 2.4198 |
| 22 | 1500 MeV | 32 | 0.3667 / 0.3952 | 0.9280 | 8.8315 / 93.8470 | 0.0941 | 0.2382 / 0.3943 | 0.6042 |

Charge-calibration benchmark (response-fit least squares, recorded in training.json): pooled relative energy MAE 0.4223; 500 MeV 0.8287, 1000 MeV 0.1622, 1500 MeV 0.2759; negative predictions 0.

Energy endpoint: initial kinetic energy (recorded target, log(E/1000 MeV)); the D-E deposited-energy endpoint was not trained. The constant is the pooled training-split mean, evaluated unchanged in every subgroup; ratio = model / constant.

Recorded verdict (unchanged): energy_absolute gate failed for all seeds; R-3 gates apply to future runs only.
