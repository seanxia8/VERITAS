# De-jitter extension — Part 1 sensitivity (registered: false)

1 000 events, excluded for dark-moved triggers: 70; fiducial redraws 281, capacity failures 1001; mean per-PMT x_phys time change SD 0.465 ns.

## Counterfactual sensitivities and ratios

| encoder | S_jit | S_noise | S_phys | R_jit/phys | R_jit/noise | Rw_jit/phys | Rw_jit/noise | rank |
|---|---|---|---|---|---|---|---|---|
| init9000 | 0.0002 | 0.0032 | 0.0060 | 0.0399 | 0.0764 | 0.0813 | 0.0184 | 128 |
| fix9000 | 0.0002 | 0.0320 | 0.0158 | 0.0111 | 0.0055 | 0.0197 | 0.0053 | 37 |
| fix9001 | 0.0002 | 0.0224 | 0.0137 | 0.0134 | 0.0082 | 0.0589 | 0.0150 | 48 |
| mae9000 | 0.0296 | 0.0299 | 0.0181 | 1.6388 | 0.9906 | 0.7246 | 0.1254 | 128 |
| ijepa9000 | 0.0002 | 0.0059 | 0.0082 | 0.0258 | 0.0357 | 0.0625 | 0.0139 | 85 |
| ijepa9001 | 0.0002 | 0.0057 | 0.0120 | 0.0197 | 0.0415 | 0.0677 | 0.0154 | 126 |
| across9000 | 0.0021 | 0.3227 | 0.0821 | 0.0250 | 0.0064 | 0.0110 | 0.0037 | 8 |
| across9001 | 0.0031 | 0.2573 | 0.0630 | 0.0495 | 0.0121 | 0.0622 | 0.0212 | 7 |
| raw_pooled | 0.0001 | 0.1138 | 0.0339 | 0.0021 | 0.0006 | 0.1288 | — | 630 |

## 10 000-event detectability probes (robust/mean delta, vertex)

| row | R2(delta_robust) | R2(delta_mean) | R2(vertex) |
|---|---|---|---|
| init9000 | 0.4262 | 0.4458 | 0.9703 |
| fix9000 | 0.0643 | 0.0710 | 0.9428 |
| fix9001 | 0.1872 | 0.1975 | 0.9502 |
| mae9000 | 0.5867 | 0.6161 | 0.9886 |
| ijepa9000 | 0.2207 | 0.2331 | 0.9482 |
| ijepa9001 | 0.3714 | 0.3892 | 0.9543 |
| across9000 | 0.0754 | 0.0781 | 0.8695 |
| across9001 | 0.0699 | 0.0726 | 0.8791 |
| raw_pooled | 0.1662 | 0.1764 | 0.9380 |
| hit_token_mean_time | 0.1798 | 0.1863 | 0.0002 |
