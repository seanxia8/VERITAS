# WP5 LUCiD grid — execution plan (after the development cohort)

Code: the SHA the development decisions are frozen at (today `f824075`+). Host: deepthought2 (L40S),
≤2 concurrent guard-gated jobs, fp32 (TF32 rejected at 1.224× < 1.5×). Cap: **170 GPU-h** (G2
Option E). Ledger: `docs/jepa/RESOURCE_LEDGER.json`. Overlay: `/ceph/dwong/jepa/env_overlay` with
`PROVENANCE.json` re-verified before every batch.

## Volume and cost (measured 0.639 h/run training; +10–20 % evaluation)

> **18 Oct 2026 correction:** that per-run cost came from a 500-step smoke with only **2** runs, which
> was fully loader-cached. On real 100-run cohorts the loader default (12 cached runs) thrashed and
> trained at 3.07 s/step. `loader_cache_runs: 128` is now set (`327c66d`); the per-run cost **must be
> re-measured on a dev-cohort (100-run) smoke before this table is quoted again**. The cap and the
> fallback rule are unchanged.

| block | runs | GPU-h (training) |
|---|---|---|
| development cohort: step 2 (2) + step 3 (15) | 17 | ~10.9 |
| main cells: 6 cells × 7 arms × seeds (5/5/5/3/3/3/3) | 162 | ~103.6 |
| R11: 3 arms × 5 | 15 | ~9.6 |
| R11b: 3 arms × 3 | 9 | ~5.8 |
| dark-window anchor: 3 | 3 | ~1.9 |
| law T: 3 arms × 3 | 9 | ~5.8 |
| occupancy: 3 | 3 | ~1.9 |
| **total** | **201 + 17** | **~139.4** |

Projection ~139 h training + evaluation (~15 h) ≈ 154 h of 170 h — inside the cap with ~16 h margin.
**Fallback trigger (no new gate):** if ledger + projected cost reaches 170 h, apply option A
(3 seeds for every main arm, 5 000 steps for every LUCiD arm), recorded as a weaker protocol.

## Order (secondary runs in the §3.5.6 order)

1. main cells, one `grid --cells <cell> --arms <arm>` job per cell×arm (42 jobs; the seed list is
   taken from the config so the 5/3 split is automatic); run **≤2 concurrently**.
2. R11 (3 arms × 5 seeds) → R11b (3 × 3) → dark-window anchor (3) → law T (3 × 3) → occupancy (3).
3. `aggregate` after each block (read-only) so the verdicts are checked incrementally; the final
   aggregate emits R6, R7, R7b, R11, R11b, R12 (R12 will be `not_evaluable` from the frozen F3
   outcome; R7b is evaluable on the R-6 readout and carries the dark-only embedding-change number).

## Per-batch provenance (required)

For every batch: `git rev-parse HEAD` + empty `git status --porcelain`; overlay `PROVENANCE.json`
sha256 unchanged; data-root dataset manifests carry `readout_pmt_r6` and generator digest
`61fdafa6…`; run manifests record `tf32: false`, `micro_batch 8`, `gradient_accumulation 4`,
effective batch 32, the frozen optimizer block and the cell/β/seed. Any change → stop the batch and
report.

## Gates that still apply

No registered numeric may change without a gate: the 5 000-step / 3-seed fallback is the only
registered degradation and is recorded as a weaker protocol. R8 is reported `not_evaluable` until
its shifted-set construction is implemented (its criterion text is recorded). R12 stays
`not_evaluable` from the frozen F3 outcome.
