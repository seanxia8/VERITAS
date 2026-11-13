# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook and addenda committed unchanged: Gate V informational (`d237ec2b…`), JUNO on Titans v2 (`0a8d76e9…`), caps raised (`d715eee6…`: L40S 72 h, Titan 160 h).
- SK (L40S, cap 72 h): mae 9001 complete; mae 9000 resuming at step 8,880; posfree 9000 (2,337), posfree 9001, sigreg 9000, sigreg 9001 (restored by the cap raise) and `mae_mse_rf` (restored, last) queued; `init` 9000–9003 evaluated. The guard admits one job at a time while another user shares the GPU.
- JUNO: data packed and verified; Titan feasibility done (CUDA OK, WCTE regression PASS on a Titan); T-S1 kept at 160 h (seed-9000 plan needs 316 Titan GPU-h).
- HK: S1 (§8 fallback step 3), not generated; Gate V (informational) failed.
- Gate V ceiling: WCTE, SK, JUNO done (review PASS); HK skipped (CPU allowance).

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
