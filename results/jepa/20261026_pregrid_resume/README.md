# Part C continuation — 26 Oct 2026

`registered: false`. Grid remains paused. Parts A/B are complete; Part C is partially complete. Parts D–F have not run.

The earlier quota-failed `pregrid_C_init9000` attempt and the historical `20261025_pregrid_checks` report remain preserved. Before guard housekeeping removed the dead PID record, the complete failed job folder was copied to `/ceph/dwong/jepa/returns/ext_pregrid_20261025/recovery_20261026/deepthought/failed_job_snapshot`. Original and snapshot job-log SHA256 are both `50522600570efc0f8b1192f8f01c9289fb08a7687eede99692adc377d987f63f`.

## Completed init check

`pregrid_C_init9000_retry1`: Deepthought Titan GPU0; GitHub execution SHA `3ecdc5cc20663d62f32ad52432643aa0a95db6b0`. Guard was fresh OK with three utilization samples over246.5 seconds at launch. Exit0,785.11 seconds,0.218086111 Titan GPU-hours.

| Quantity | Full input | Half-pass |
|---|---:|---:|
| Vertex resolution m | 0.16007161647162788 | 0.17482120319755615 |
| Module-gain run-mean R2 | 0.20049761158220947 | 0.2037560719974032 |
| Log-yield R2 | 0.9948826346987075 | 0.9952901632713392 |
| Effective rank | 1.53028277354294 | 1.6126226013162415 |
| Participation ratio | 1.2153969569169343 | 1.2501803190434067 |
| Event SIGReg | 2326.7632131451833 | 4347.023510434597 |
| Hit-PMT log-charge R2 | 0.9983181840341251 | 0.9983947587975116 |

Unchanged expectation0.160904 m; absolute difference0.00083238352837212 m <=0.001 m: PASS. No seed, tolerance, reference, dataset, checkpoint or frozen decision changed. These random-init diagnostics do not resolve the predeclared readings for the two fixed trained seeds.

## Pending six checkpoint inspections

`old9000`, `old9001`, `fix9000`, `fix9001`, `mae9000`, `anc100` use the exact Part C paths, arms, expectations, frozen configuration and nominal data. Hashes of all six checkpoints, both configurations and five data manifests were reverified; nominal caches are reused, not regenerated. Source code reaches hosts only through GitHub.

Foreign GPU work started after the accepted init launch. Titan guard HOLD prevents any new launch; the running init evaluation continued within memory limits and completed. No SHED was observed. Full guard history stays on the host; concise events are in INFRA_LOG.md.

A bounded Part-C-only queue waits on the existing guard/front-door checks, then runs these six items in the same `inspect` command, as in the runbook. It accepts reproduction only after reading inspect.json (process exit0 alone is insufficient), writes STOP_S2.md with lettered options for any failed expectation, and never launches Part D or the grid. It refuses to overwrite a partly existing attempt. Runtime is planned100 min (six times the measured13.1 min init runtime, with margin); launcher timeout150 min. All executable code is pushed to GitHub before a pinned clone is used.

Queue job ID: `pregrid_C_checkpoints_20261026`. Remote status: `/ceph/dwong/jepa/returns/ext_pregrid_20261025/part_c_queue_20261026/status.json`. Output: `/ceph/dwong/jepa/returns/ext_pregrid_20261025/inspect_dev/checkpoints_20261026/inspect.json`. The queue stops after Part C for review, even when all expectations pass.

## Budget

Runbook Titan charge after init:0.1125 failed-original +0.218086111 retry =0.330586111 GPU-h of30. Pending queue reserves at most2.5 GPU-h (timeout), so total remains below30. Conservative L40S ledger plus remaining-grid projection plus extensions stays142.977123<160 GPU-h; no L40S run launched. CPU generation stays0.6535/48 wall-h; no new generation.

Six standard-library coordinator tests cover false reproduction flags, independent tolerance verification, incomplete reports, registered-report rejection, valid reports and timeout-inclusive budgets. The prior483-test scientific suite remains the accepted source baseline; only the Part C queue and its tests were added afterward.
