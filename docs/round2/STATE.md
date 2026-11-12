# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook committed unchanged (sha256 `366f068…`).
- §4.1 implementation and the ten required tests: done (52 tests pass on Deepthought, including
  WCTE identity (test 1) and the bit-for-bit regeneration of round-1 run `lucid/dev_train/0000`
  (test 2)). The recomputed §2.2 table matches the runbook for WCTE, SK, HK and JUNO.
- §4.3 review: round 1 BLOCK (fixed), round 2 PASS at `8db555f`; acceptance recorded. Next: the WCTE regression, X0 and the
  detectors (SK, JUNO, HK). Nothing generated or trained yet.

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
