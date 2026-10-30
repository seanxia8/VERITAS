# Timing extension — Part B deep encoders (registered: false)

Events: 1000 F1 s=0.05 photon events (runs 0-99 x 10), 16 scheme-A realisations each; invalid trigger events: 0; dark-moved r=0 triggers: 37.

## Probes (full-input encoder tokens, ridge train runs 0-49 / test 50-99)

| encoder | cell | subset | R2(delta) | R2(mu_t) | R2(vertex) | R2(log yield) | vertex res (m) |
|---|---|---|---|---|---|---|---|
| init9000 | dev | all | 0.6342 | 0.3978 | 0.9646 | 0.9934 | 0.1609 |
| init9000 | dev | without_dark_moved | 0.1360 | 0.4178 | 0.9647 | 0.9940 | 0.1609 |
| fix9000 | dev | all | 0.2127 | 0.2551 | 0.9307 | 0.9863 | 0.2193 |
| fix9000 | dev | without_dark_moved | -0.0211 | 0.2774 | 0.9291 | 0.9876 | 0.2193 |
| fix9001 | dev | all | 0.2899 | 0.2785 | 0.9401 | 0.9883 | 0.2043 |
| fix9001 | dev | without_dark_moved | -0.0190 | 0.3138 | 0.9386 | 0.9892 | 0.2043 |
| mae9000 | dev | all | 0.8349 | 0.5410 | 0.9831 | 0.9966 | 0.1049 |
| mae9000 | dev | without_dark_moved | 0.2240 | 0.5589 | 0.9850 | 0.9972 | 0.1049 |
| ijepa9000 | dev | all | 0.3601 | 0.3054 | 0.9366 | 0.9888 | 0.2091 |
| ijepa9000 | dev | without_dark_moved | 0.0078 | 0.3467 | 0.9358 | 0.9897 | 0.2091 |
| ijepa9001 | dev | all | 0.6243 | 0.3449 | 0.9418 | 0.9907 | 0.1950 |
| ijepa9001 | dev | without_dark_moved | 0.1030 | 0.3482 | 0.9411 | 0.9909 | 0.1950 |
| across9000 | across | all | 0.3499 | 0.2423 | 0.8226 | 0.9786 | 0.3481 |
| across9000 | across | without_dark_moved | -0.0535 | 0.2477 | 0.8217 | 0.9791 | 0.3481 |
| across9001 | across | all | 0.4319 | 0.1781 | 0.8406 | 0.9796 | 0.3335 |
| across9001 | across | without_dark_moved | 0.0115 | 0.1935 | 0.8417 | 0.9834 | 0.3335 |

## Ceilings and floor rows

| row | subset | R2(delta) | R2(mu_t) | R2(vertex) | R2(log yield) |
|---|---|---|---|---|---|
| raw_pooled | all | 0.4773 | 0.2401 | 0.9113 | 0.9798 |
| raw_pooled | without_dark_moved | -0.0215 | 0.3307 | 0.9104 | 0.9827 |
| hit_token_mean_time | all | 0.6835 | 0.2020 | -0.0111 | 0.0611 |
| hit_token_mean_time | without_dark_moved | 0.1242 | 0.3823 | -0.0040 | 0.2100 |
