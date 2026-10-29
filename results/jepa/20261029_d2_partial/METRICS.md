# Part D2 metrics — registered: false

| Variant | Seed | Vertex resolution (m) | Module-gain run-mean R² |
|---|---:|---:|---:|
| across | 9000 | 0.348113 | -0.002551 |
| across | 9001 | 0.333464 | 0.003908 |
| ijepa | 9000 | 0.209150 | 0.030165 |
| ijepa | 9001 | 0.194992 | 0.085530 |
| lambda | 9000 | 0.220055 | 0.064379 |
| lambda | 9001 | 0.199603 | 0.059772 |
| stopgrad | 9000 | 0.219583 | -0.005715 |
| stopgrad | 9001 | 0.189377 | -0.004773 |
| strong_drift | 9000 | 0.233209 | 0.097482 |
| strong_drift | 9001 | 0.217213 | 0.217191 |

Fixed-within vertex references: 0.219259 m (9000), 0.204281 m (9001). Random-init floors: nominal vertex 0.160904 m and gain R² 0.218548; across vertex 0.160072 m and gain R² 0.200498; strong drift vertex 0.170301 m and gain R² 0.841585.

Drift-too-weak is not supported: gains versus strong-drift floor are −0.744103 (9000) and −0.624394 (9001), both below the required +0.10. Other pre-declared readings and individual evidence are in README.md.
