# De-jitter extension — Part 2 prescription (registered: false)

## Linear check (§3.2): nu* by split and level, baseline vs median_time

| level | split | nu*(A) baseline | nu*(A) median_time | nu*(B) median_time | nu*(gain) median_time | regeneration A/B max diff |
|---|---|---|---|---|---|---|
| F1_0.00 | within | 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0e+00/0.0e+00 |
| F1_0.00 | across_fixed | 0.5055 | 0.0013 | 0.0013 | -0.0000 | 0.0e+00/0.0e+00 |
| F1_0.05 | within | 0.0007 | 0.0000 | 0.0002 | 0.0002 | 0.0e+00/0.0e+00 |
| F1_0.05 | across_fixed | 0.4920 | 0.0014 | 0.0011 | 0.0000 | 0.0e+00/0.0e+00 |

## Six new runs (§3.3) vs references

| variant | seed | vertex (m) | gain R2 | log yield R2 | effective rank | event SIGReg | token charge R2 |
|---|---|---|---|---|---|---|---|
| across_median | 9000 | 0.3475 | 0.0016 | 0.9771 | 18.0421 | 1370.1956 | 0.0002 |
| across_median | 9001 | 0.3203 | 0.0016 | 0.9862 | 3.5784 | 1371.3195 | 0.0042 |
| within_median | 9000 | 0.2227 | -0.0015 | 0.9872 | 2.1452 | 1427.4448 | 0.0172 |
| within_median | 9001 | 0.2072 | 0.0084 | 0.9895 | 2.9793 | 1430.0955 | 0.0509 |
| within_trigger | 9000 | 0.2244 | -0.0022 | 0.9882 | 2.4500 | 1426.4418 | 0.0194 |
| within_trigger | 9001 | 0.2038 | 0.0058 | 0.9897 | 2.9733 | 1411.0240 | 0.0532 |

Part D across references: 0.348113 / 0.333464 m; Part C fixed-within references: 0.219258 / 0.204281 m.

## median_time random-init floors (Titan, inspect with wrapper)

| cell | vertex (m) | gain R2 | effective rank | token charge R2 |
|---|---|---|---|---|
| across_median | 0.1597 | 0.2014 | 1.5196 | 0.9983 |
| within_median | 0.1597 | 0.2014 | 1.5196 | 0.9983 |
