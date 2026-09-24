# Whitened-residual third-cumulant signature row — DEVELOPMENT OUTPUT, NOT CITABLE

_DEVELOPMENT — NOT CITABLE. Tier-1 signature row for the whitened-residual third cumulant (D7) on the linear subject; no scientific conclusion, a negative outcome is reported as recorded._

chart reading `cumulant`; statistic `population connected third cumulant tensor of the whitened residual in the reference residual PCA chart (n_pcs = 3)`; threshold z = 2.33 (q99 of the z-scored bootstrap deviation; reference deviation 31.64, z -0.07). **12 match, 0 documented, 2 mismatch.**

| cell | moved | expected | observed | deviation | z | KL ratio | status |
|---|---|---|---|---:|---:|---:|---|
| reference | none | at_reference | at_reference | 31.64 | -0.07 | 1.003 | match |
| sigma_cov:corr_up | sigma_cov | at_reference | at_reference | 31.63 | -0.12 | 0.900 | match |
| sigma_cov:corr_down | sigma_cov | at_reference | at_reference | 31.47 | -0.63 | 0.953 | match |
| sigma_cov:bandwidth_down | sigma_cov | at_reference | at_reference | 31.62 | -0.14 | 1.000 | match |
| sigma_cov:line_pickup | sigma_cov | at_reference | at_reference | 31.61 | -0.20 | 0.950 | match |
| sigma_struct:gain_drift | sigma_struct | at_reference | at_reference | 31.90 | 0.77 | 1.048 | match |
| sigma_struct:channel_loss | sigma_struct | moves | moves | 161.44 | 429.99 | 0.029 | match |
| sigma_struct:timing_jitter | sigma_struct | at_reference | at_reference | 30.92 | -2.46 | 1.015 | match |
| event:glitch | event | moves | at_reference | 30.98 | -2.28 | 0.981 | MISMATCH |
| event_in_span:oscillation | event_in_span | at_reference | at_reference | 31.73 | 0.22 | 0.999 | match |
| event_in_span:double_pulse | event_in_span | at_reference | at_reference | 31.15 | -1.72 | 0.542 | match |
| sigma_non_gaussian:sparse_burst | sigma_struct | moves | at_reference | 31.85 | 0.60 | 1.006 | MISMATCH |
| geometry:4ch | geometry | skipped | skipped | — | — | — | match |
| geometry:16ch | geometry | skipped | skipped | — | — | — | match |

Notes:

- sigma_cov:corr_up: re-whitened: kappa 4.96 correction
- sigma_cov:corr_down: re-whitened: kappa 5.09 correction
- sigma_cov:bandwidth_down: re-whitened: kappa 1.98 correction
- sigma_cov:line_pickup: re-whitened: kappa 8.11 correction

A MISMATCH is reported, not explained away. On this subject and at this size the population third-cumulant deviation stayed within the reference window-bootstrap band (q99 z = 2.33) for the glitch event and the non-Gaussian sparse-burst family, while gain drift and channel loss matched their predictions (a linear-Gaussian gain operation stays at reference; dead channels leave a deterministic residual that moves it) and the covariance-type cells were at reference after re-whitening (no false positives). The estimator is a population tensor of per-window rank-1 third moments, which is noise-dominated at n_eval = 60 and n_pcs = 3; the note already records that reference distances are unstable at n_ref <~ 100 and that a third-moment tensor is noisier than a second. This is a development negative for the separator at this size, not a confirmed refutation.
