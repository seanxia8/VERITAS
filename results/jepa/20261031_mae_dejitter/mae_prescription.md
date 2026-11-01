# MAE de-jitter — eight encoders vs references and floors (registered: false)

## Runs (full input)

| variant | seed | vertex (m) | gain R2 | token charge R2 | effective rank | event SIGReg |
|---|---|---|---|---|---|---|
| across_median | 9000 | 0.0970 | 0.2157 | 0.9787 | 4.1207 | 2439.7572 |
| across_median | 9001 | 0.1006 | 0.2016 | 0.9707 | 3.5261 | 2782.7529 |
| across_trigger | 9000 | 0.1210 | 0.1995 | 0.9715 | 4.5833 | 2231.0822 |
| across_trigger | 9001 | 0.1196 | 0.1658 | 0.9598 | 4.3577 | 2345.3108 |
| within_median | 9000 | 0.0852 | 0.1921 | 0.9865 | 2.2684 | 1671.1639 |
| within_median | 9001 | 0.0950 | 0.2151 | 0.9850 | 2.3144 | 2057.5703 |
| within_trigger | 9000 | 0.1049 | 0.1794 | 0.9880 | 2.9101 | 1656.3234 |
| within_trigger | 9001 | 0.1165 | 0.2171 | 0.9869 | 2.8925 | 2014.9322 |

Floors: trigger init9000 0.16007 m (Part C), median init9000 0.159692 m (de-jitter Titan floor); de-jitter SIGReg-JEPA across 0.347506/0.320314 m, within 0.222653/0.207199 m.

## Token time R2 (own convention / median-relative)

| label | transform | R2(own) | R2(median) |
|---|---|---|---|
| init9000_median | median_time | 0.5600 | 0.5600 |
| init9000_trigger | none | 0.6366 | 0.4960 |
| mae_across_median_s9000 | median_time | 0.9444 | 0.9444 |
| mae_across_median_s9001 | median_time | 0.9610 | 0.9610 |
| mae_across_trigger_s9000 | none | 0.9200 | 0.7742 |
| mae_across_trigger_s9001 | none | 0.9299 | 0.8125 |
| mae_within_median_s9000 | median_time | 0.9527 | 0.9527 |
| mae_within_median_s9001 | median_time | 0.9436 | 0.9436 |
| mae_within_trigger_s9000 | none | 0.9289 | 0.7639 |
| mae_within_trigger_s9001 | none | 0.9253 | 0.7614 |
| sigreg_across_median_s9000 | median_time | 0.0043 | 0.0043 |
| sigreg_across_median_s9001 | median_time | 0.0074 | 0.0074 |
| sigreg_within_median_s9000 | median_time | 0.0185 | 0.0185 |
| sigreg_within_median_s9001 | median_time | 0.0263 | 0.0263 |
| sigreg_within_trigger_s9000 | none | 0.0089 | 0.0167 |
| sigreg_within_trigger_s9001 | none | 0.0108 | 0.0258 |

## Trigger-jitter sensitivity

| label | R_jit/phys | Rw_jit/phys | R_jit/noise | rank |
|---|---|---|---|---|
| mae_across_trigger_s9000 | 1.1450 | 0.6460 | 0.7380 | 128 |
| mae_across_trigger_s9001 | 1.3476 | 0.6262 | 0.7851 | 128 |
| mae_across_median_s9000 | 0.0000 | 0.0003 | 0.0000 | 128 |
| mae_across_median_s9001 | 0.0000 | 0.0003 | 0.0000 | 128 |
| mae_within_trigger_s9000 | 1.6392 | 0.7234 | 0.9905 | 128 |
| mae_within_trigger_s9001 | 1.3505 | 0.5941 | 0.9264 | 128 |
| mae_within_median_s9000 | 0.0000 | 0.0005 | 0.0001 | 128 |
| mae_within_median_s9001 | 0.0000 | 0.0009 | 0.0001 | 128 |
| mae9000 | 1.6388 | 0.7246 | 0.9906 | 128 |
| fix9000 | 0.0111 | 0.0197 | 0.0055 | 37 |
| raw_pooled | 0.0021 | 0.1288 | 0.0006 | 630 |

Median-time encoders: their R_jit/phys is edge effects only (a constant trigger shift is removed by construction) and is not scored.

S9 regeneration checks: {"mae9000": {"got": 1.6388379147906313, "expected": 1.6388379147906313, "relative": 0.0}, "fix9000": {"got": 0.011072650646817152, "expected": 0.011072650646817152, "relative": 0.0}, "whitened_mae9000": {"got": 0.7245656314414844, "expected": 0.7245656314414844, "relative": 0.0}, "whitened_fix9000": {"got": 0.01970821348627603, "expected": 0.01970821348627603, "relative": 0.0}}
