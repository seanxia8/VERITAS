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
