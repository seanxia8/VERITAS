# INFRA_LOG — JEPA infrastructure events (one line per event)

Runbook v4 §6.2: every HOLD or SHED event is logged here as one line. Other infrastructure
incidents that affect provenance are also one line each. Full evidence lives in
`PREFLIGHT_A8_deepthought.md`, `PREFLIGHT_A8_deepthought2.md` and the host job directories.

| UTC time | host | event | one-line record |
|---|---|---|---|
| 2026-10-17 12:18–12:30 | deepthought, deepthought2 | guard dry run | 10-min log-only dry run each; level OK from the third cycle (warm-up HOLD before that, as designed); no status JSON, no signals, no SHED |
| 2026-10-17 12:29–12:30 | deepthought, deepthought2 | guard live | `jepa_guard` started in tmux on both hosts after the dry-run acceptance; level OK, 0 SHED since |
| 2026-10-17 12:35 | deepthought | probe job | `probe_deepthought_6468582` exit 0, 10.55 s, GPU 0; guard OK before/after |
| 2026-10-17 12:36 | deepthought2 | probe job | `probe_deepthought2_6468582` exit 0, 126.07 s (cold CephFS cache), GPU 0; guard OK |
| 2026-10-17 12:57 | deepthought | probe + smoke | corrected probe `probe_deepthought_89af5af_v2` exit 0, 10.52 s; the first bridge smoke `smoke_bridge_deepthought_89af5af` exit 1 in 10.68 s on the platform-brittle frozen hash (no CUDA code ran) |
| 2026-10-17 12:57 | deepthought2 | probe job | corrected probe `probe_deepthought2_89af5af_v2` exit 0, 3.74 s |
| 2026-10-17 13:05 | deepthought | smoke job | `smoke_bridge_deepthought_bbcded2` exit 0, 14.36 s, four arms on cuda:0 after the tolerance fix |
| 2026-10-17 13:43–14:13 | deepthought | training job | `wp4a_bench_sigreg_s0` exit 0, 1782.67 s; guard OK before/after |
| 2026-10-17 14:20–16:06 | deepthought | training job | `wp4a_beta_dev` exit 0, 6402.63 s; β table written |
| 2026-10-17 16:15–18:00 | deepthought | 3 training jobs | `wp4a_grid_sigreg`, `wp4a_grid_anchored`, `wp4a_grid_mae_chi2` on GPUs 0/2/3; 3 training jobs = the §6.2 cap; 0 SHED/HOLD; GPU 1 (foreign kernel) never used |
| 2026-10-17 18:06–18:19 | deepthought | aggregation incident | a `grid` re-invocation for combined aggregation **re-executed** runs instead of aggregating; stopped with SIGKILL after 13 min 17 s on CPU (no GPU, no cohort-data writes); `wp4a_grid_sigreg` s0 `run_status.json` clobbered and `checkpoint.pt`/`SHA256SUMS` overwritten (final-checkpoint artifact lost, sha preserved in the manifest), s1 `run_status.json` reset; `embeddings.npz`/`probes.json` of both and all other runs untouched |
| 2026-10-17 18:58–19:47 | deepthought | training job | `wp4a_grid_parallel` exit 0, 3944.72 s; 12/12 bridge runs complete, total 6.5 Titan GPU-hours including the beta cohort and benchmark |
| 2026-10-23 11:43 | deepthought2 | grid stop | `a10_grid_gpu0_wait` exit 2 after 15 completed runs: `anchor_only requires a beta decision` because the frozen A10 step-3 has `selected: null`; completed runs retained |
| 2026-10-23 21:23 | deepthought2 | grid pause | `a10_grid_gpu1_retry` paused gracefully after the in-progress run completed; `SIGTERM` forwarded by `jepa_launch` (exit 143), no SIGKILL; 10 completed runs retained |
| 2026-10-23 21:23 | deepthought2 | training stopped | no JEPA training job remains on deepthought2; both grid halves stopped for the author's label-shortcut diagnostic |
| 2026-10-23 21:26–22:36 | deepthought2 | diagnostic training | `a10_step2b_old_s9001b` exit 0, 2407.54 s; current-label seed 9001 |
| 2026-10-23 22:47–23:30 | deepthought2 | diagnostic training | `a10_step2b_fixed_s9000` exit 0, 2575.51 s; target-only labels, seed 9000 |
| 2026-10-23 23:34–2026-10-24 00:19 | deepthought2 | diagnostic training | `a10_step2b_fixed_s9001` exit 0, 2655.57 s; target-only labels, seed 9001; branch confirmed |
| 2026-10-24 00:37 | deepthought2 | dev re-run | `a10_dev_step3_fixed` launched on GPU 0 at `c2dce54`; fixed-label A10 dev step 3 (15 runs, ≤15 GPU-h) |

| 2026-10-25 00:41:13Z | deepthought | HOLD | During `pregrid_C_init9000`: guard reports free entries 11,468 <30,000, `write no new data`; no new launches, existing evaluation continues per §6.2. No deletion/config change; monitor SHED threshold10,000. |
| 2026-10-25 <=00:45:17Z | deepthought | QUOTA/HOLD | Part C probe write and launcher/guard atomic status writes fail withOSError122. Quota502724/500000 entries; need>=32724 released for30000 free. Evaluation no longer running; no retry/new launch, no deletion;405s conservative upper-bound charge. |
| 2026-10-25 23:04Z | both hosts | guard restart / HOLD | Fresh Ceph-job guards from3ecdc5c; old failed job+PID and guard snapshots retained under ext_pregrid_20261025/recovery_20261026; utilization warm-up blocks starts, quota headroom195118; grid paused. |
| 2026-10-25 23:16:54Z | deepthought | HOLD | Foreign GPU work started during Part C retry; guard reports foreign utilization73% on GPU0 and no eligible start GPU. No new starts; running evaluation continues per6.2, our GPU memory426 MiB/RSS1240 MiB; no SHED. |
| 2026-10-25 23:18:55Z | deepthought | HOLD | No eligible start GPU after foreign jobs occupied the Titan lane; Part C retry continues within426 MiB GPU memory and1240 MiB RSS. No new starts and no SHED. |
| 2026-10-25 23:20:55Z | deepthought | HOLD | Foreign GPU0 memory3644 MiB and high utilization across Titans block starts; running Part C retry uses428 MiB GPU memory/1240 MiB RSS, full and half-pass probes saved, no SHED. |
| 2026-10-25 23:22:56Z | deepthought | HOLD / retry complete | Init9000 retry exited0 at23:22:17Z; reproduction passed. High foreign GPU utilization/memory still block all starts; six checkpoint inspections pending; no SHED, grid paused. |
| 2026-10-25T23:28:51Z | deepthought | HOLD / Part C queue | Bounded coordinator is WAITING_GUARD; no eligible Titan, no checkpoint job started. Both hosts CPU tests18/18 passed; no D/grid continuation queued. |

- **28 Oct — Part C queue read-only handoff check:** `ENGINEERING_FAILURE`, exit 1 after 3894.59 s, five probe pairs and no complete `inspect.json`; anchored construction raised missing-beta ValueError. Evidence and host SHA256 manifest copied under `results/jepa/20261026_pregrid_resume/failure_20261028/`, 17/17 verified. Deepthought guard snapshot dated 27 Oct is stale. Ceph quota 428702/500000 entries, free 71298 at check. Both hosts reachable; deepthought2 JEPA path present. No relaunch.
- **29 Oct — Part C rerun:** fresh Deepthought guard OK after warm-up, 68551 free Ceph entries at launch, Titan GPU1, 4239.36 s, exit 0; no reported HOLD/SHED during monitored evaluation. Evidence and SHA256 verification in `results/jepa/20261029_part_c/`.
- **29 Oct — D0 lambda smoke engineering failure:** exit 1 after 8.32 s on read-only coefficient property assignment; original files preserved, five copied files SHA256 verified under `results/jepa/20261029_d0_lambda_failure/`. Retry requires tested pushed fix and a new path.
