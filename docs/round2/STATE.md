# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook and addenda committed unchanged: Gate V informational (`d237ec2b…`), JUNO on Titans v2 (`0a8d76e9…`).
- SK (L40S): data packed and verified; smokes done; main runs training under co-tenancy (core comparison first; watchdog pauses runs that cannot finish within their cap); `init` evaluations running; `mae_mse_rf` S1.
- JUNO: data packed and verified; Titan feasibility done (CUDA OK, WCTE regression PASS on a Titan) but JEPA arms need ≈ 135 h per run on a TITAN X: **T-S1**, not trained.
- HK: S1 (§8 fallback step 3), not generated.
- Gate V ceiling: WCTE, SK, JUNO done (review PASS); HK pending its Gate V rows.

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
