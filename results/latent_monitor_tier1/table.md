# Latent-monitoring table — linear subject, Tier-1 cells (latent_monitor 0.1.0)

C = 8, N = 256, k = 6; 200 fit / 60 eval events, 100 noise-only records; seed 0.
Fisher rank 6, k_out 3. 13 match, 1 documented, 0 mismatch.

| cell | moved | expected | attributed | status | mean-shift | z-var ratio | psd dev | chan-corr | out-of-span | layer peak | consequence ratio |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| sigma_cov:corr_up | sigma_cov | sigma_cov | **sigma_cov** | match | 0.7 | 1.67 | 0.45/0.46 | 2.01 | 1.00 | whitened | [1.013, 1.12, 1.014] |
| sigma_cov:corr_down | sigma_cov | sigma_cov | **sigma_cov** | match | 1.1 | 0.32 | 0.45/0.47 | 0.81 | 0.99 | whitened | [0.988, 0.817, 0.986] |
| sigma_cov:bandwidth_down | sigma_cov | sigma_cov | **sigma_cov** | match | 0.3 | 1.09 | 0.43/0.44 | 0.01 | 1.00 | channel | [1.003, 1.064, 1.001] |
| sigma_cov:line_pickup | sigma_cov | sigma_cov | **sigma_cov** | match | 0.4 | 0.91 | 4.77/7.02 | 0.05 | 1.00 | whitened | [0.986, 0.93, 0.99] |
| sigma_struct:gain_drift | sigma_struct | sigma_struct | **sigma_struct** | match | 648.7 | 0.88 | 0.08/0.11 | 0.38 | 0.98 | channel | [1.045, 0.991, 0.994] |
| sigma_struct:channel_loss | sigma_struct | sigma_struct | **sigma_struct** | match | 2604.5 | 0.56 | 0.14/0.18 | 2.79 | 0.95 | channel | [1.654, 1.322, 1.172] |
| sigma_struct:timing_jitter | sigma_struct | sigma_struct | **sigma_cov** | documented | 1.4 | 0.55 | 0.57/0.66 | 1.42 | 1.00 | whitened | [0.977, 0.985, 0.987] |
| geometry:4ch | geometry | geometry | **geometry** | match | 2949.7 | 0.90 | 0.11/0.19 | nan | 0.12 | token | [1.807, 1.544, 1.416] |
| geometry:16ch | geometry | geometry | **geometry** | match | 2545.5 | 0.92 | 0.04/0.12 | nan | 0.06 | token | [1.625, 1.216, 1.147] |
| event:glitch | event | event | **event** | match | 601.9 | 1.00 | 0.00/0.00 | 0.00 | 0.84 | channel | [1.03, 1.034, 1.097] |
| event_in_span:oscillation | event_in_span | event_in_span | **event_in_span** | match | 950.3 | 1.00 | 0.00/0.00 | 0.00 | 0.17 | channel | [1.06, 1.756, 1.098] |
| event_in_span:double_pulse | event_in_span | event_in_span | **event_in_span** | match | 9412.9 | 1.00 | 0.00/0.00 | 0.00 | 0.02 | whitened | [4.578, 2.104, 3.877] |
| designed:output_null | designed | output_null | **output_null** | match | 9.5 | 1.00 | 0.00/0.00 | 0.00 | 0.00 | whitened | [1.0, 1.0, 1.0] |
| designed:output_aligned | designed | output_aligned | **output_aligned** | match | 10.2 | 1.00 | 0.00/0.00 | 0.00 | 0.00 | whitened | [1.403, 3.66, 1.452] |

Thresholds (calibrated once on the reference null): mean_shift_norm=5.44, dz_norm=4.53, var_ratio_low=0.85, var_ratio_high=1.15, psd_ratio_dev=0.173, psd_line_dev=0.293, chan_corr_shift=0.125, out_of_span=0.5, consequence_ratio_null=0.1, isotropy=0.5, z_small_factor=0.5

## Adjustments

### Re-whitening (Σ-covariance cells)

| cell | κ correction | z-var ratio before → after | consequence ratio before → after |
|---|---:|---|---|
| sigma_cov:corr_up | 4.52 | 1.66 → 1.00 | [1.01, 1.12, 1.01] → [1.02, 1.1, 1.02] |
| sigma_cov:corr_down | 4.94 | 0.32 → 1.00 | [0.99, 0.82, 0.99] → [0.99, 0.81, 0.99] |
| sigma_cov:bandwidth_down | 1.94 | 1.09 → 1.00 | [1.0, 1.06, 1.0] → [1.01, 1.06, 1.01] |
| sigma_cov:line_pickup | 7.47 | 0.91 → 1.00 | [0.99, 0.93, 0.99] → [0.99, 0.93, 0.99] |

### Stage-restricted refit (consequence ratio; 1.0 = reference)

| cell | before | channel | token | output |
|---|---|---|---|---|
| sigma_struct:gain_drift | [1.05, 0.99, 0.99] | [0.97, 0.91, 0.96] | [1.0, 1.03, 1.0] | [1.0, 1.01, 1.0] |
| sigma_struct:channel_loss | [1.65, 1.32, 1.17] | [0.94, 0.97, 0.93] | [1.0, 1.06, 0.98] | [1.0, 1.03, 0.99] |
| sigma_struct:timing_jitter | [0.98, 0.98, 0.99] | [1.0, 0.89, 1.01] | [0.98, 1.05, 0.99] | [0.98, 0.9, 1.0] |
| geometry:4ch | [1.81, 1.54, 1.42] | [0.99, 1.0, 0.98] | [1.81, 1.89, 1.4] | [0.99, 0.99, 0.99] |
| geometry:16ch | [1.62, 1.22, 1.15] | [1.01, 0.93, 1.02] | [1.55, 1.22, 1.12] | [0.99, 0.96, 0.99] |

### Activation patching (fraction of the consequence gap recovered by substituting the clean stage)

- sigma_struct:gain_drift: unpatched=0.00, whitened=1.00, channel=1.00, token=1.00, z=1.00
- sigma_struct:channel_loss: unpatched=0.00, whitened=1.00, channel=1.00, token=1.00, z=1.00
- sigma_struct:timing_jitter: unpatched=0.00, whitened=1.00, channel=1.00, token=1.00, z=1.00

For an input-side corruption every stage recovers fully: the linear subject has no stage that *creates* damage. Stage localisation (C5) is therefore a question for the nonlinear subjects; here the informative repair contrast is the stage refit above.
