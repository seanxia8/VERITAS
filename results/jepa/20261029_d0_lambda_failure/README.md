# D0 lambda smoke failure — 29 Oct 2026

`registered: false`. `pregrid_D0_lambda_20261029`, code `f29137c`, exited 1 after 8.32 s (0.002311111 Titan GPU-h, booked). `patch_lambda` tried to assign to the read-only `LucidArm.coefficient_lambda` property before training. This is an engineering failure; no training or usable smoke result was produced. The five copied files match host SHA256SUMS; original job and output remain untouched. The retry, if tests pass, uses a distinct run and output path.
