# D2 partial evidence — 29 Oct 2026

`registered: false`. Across seed 9000 completed 10000 steps, exit 0 after 2801.10 s (0.778083333 L40S GPU-h), with finite extension summary; full-input vertex resolution 0.3481133106 m. Host SHA256 checks passed for both copied files. Stopgrad seed 9000 also completed 10000 steps, exit 0 after 3006.82 s (0.835227778 L40S GPU-h), with finite summary and full-input vertex resolution 0.2195829580 m. These single-seed results are not complete D2 comparisons. Remaining jobs and the D1 floors are pending; grid remains paused.

Across seed 9001 completed 10000 steps, exit 0 after 2694.73 s (0.748536111 L40S GPU-h), full-input vertex 0.333463657 m. Across vs fixed within is 0.348113/0.219258 = 1.588 and 0.333464/0.204281 = 1.632; the pre-declared task-too-local reading is not supported by its <=0.85 criterion. Across's own D1 floor is still pending.

Stopgrad seed 9001 completed 10000 steps, exit 0 after 3459.33 s (0.960925 L40S GPU-h), full-input vertex 0.189377155 m. The pre-declared label-shaping reading is not supported: 0.219583 > 0.9 × 0.219258 and 0.189377 > 0.9 × 0.204281. Both results are descriptive extensions.

Lambda seed 9000 completed 10000 steps, exit 0 after 2936.43 s (0.815675 L40S GPU-h). Full-input vertex 0.220055133 m; final training SIGReg 1.4837366 and loss verified as prediction + 0.2 × SIGReg. Seed 9001 remains needed for the pre-declared reading.

Lambda seed 9001 completed 10000 steps, exit 0 after 3099.72 s (0.861033333 L40S GPU-h). Full-input vertex 0.199603069 m; final training SIGReg 1.44313335. The pre-declared SIGReg-overridden reading is not supported: seed 9000 vertex 0.220055133 m exceeds its fixed-within reference 0.219258 m, despite final SIGReg below 3. Both iJEPA seeds are running; D1 remains pending.

IJEPA seed 9000 completed 10000 steps, exit 0 after 1729.56 s (0.480433333 L40S GPU-h), full-input vertex 0.209149763 m; host hashes match. Seed 9001 remains pending; seed 9000 does not beat the 0.160904 m init floor.

IJEPA seed 9001 completed 10000 steps, exit 0 after 1569.29 s (0.435913889 L40S GPU-h), full-input vertex 0.194991722 m; host hashes match. EMA reading is not supported: neither seed beats the 0.160904 m random-init floor (0.209149763 and 0.194991722 m).

Strong-drift seed 9000 completed 10000 steps, exit 0 after 2607.98 s (0.724438889 L40S GPU-h), full-input vertex 0.233208941 m and module-gain R² 0.097482087, 0.744103251 below its 0.841585338 random-init floor; host hashes match. Seed 9001 remains pending.

Strong-drift seed 9001 completed 10000 steps, exit 0 after 2595.06 s (0.72085 L40S GPU-h), full-input vertex 0.217212912 m and module-gain R² 0.217191018, 0.624394320 below its 0.841585338 random-init floor; host hashes match. Drift-too-weak reading is not supported: both gains fall below the floor rather than exceeding it by at least 0.10. All ten D2 runs complete; see METRICS.md.
