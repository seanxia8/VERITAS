# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook and addenda committed unchanged: Gate V informational (`d237ec2b…`), JUNO on Titans v2 (`0a8d76e9…`), caps (`d715eee6…`: L40S 72 h, Titan 160 h), SK order + co-tenancy (`2e9869ea…`).
- SK (L40S, cap 72 h, wall cap 2.5×, shared GPUs allowed since 13 Nov 23:20): mae 9000 and 9001 complete and evaluated; `init` 9000–9004 evaluated; priority order sigreg 9000 → posfree 9000 (3,168) → posfree 9001 (3,548+) → sigreg 9001 → `mae_mse_rf` (S1 unless the cap holds it).
- JUNO: data packed and verified; Titan feasibility done; T-S1 kept at 160 h (seed-9000 plan needs 316 Titan GPU-h).
- HK: S1 (§8 fallback step 3), not generated; Gate V (informational) failed.
- Gate V ceiling: WCTE, SK, JUNO done (review PASS); HK skipped (CPU allowance).

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
