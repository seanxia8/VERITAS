# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook (sha256 `366f068…`) and addendum `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md` (sha256 `d237ec2b…`, Gate V informational) committed unchanged.
- Implementation accepted (PASS). WCTE regression PASS. X0 recorded for SK and JUNO; HK X0 finishing on Deepthought.
- SK: generated and packed (`sk_r2.h5`, sha256 `89cde55c…`, verified); smokes done; six main runs training on deepthought2 (projection 55.4 L40S h); `mae_mse_rf` stopped (S1).
- JUNO: **S1** (does not fit after SK). Data generated and packed as safe work.
- HK: **S1** (§8 fallback step 3). No generation.
- Gate V ceiling diagnostic: WCTE and SK done; JUNO running.

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
