# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook committed unchanged (sha256 `366f068…`); addendum `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md` committed unchanged (sha256 `d237ec2b…`): Gate V informational, S2 no longer applies.
- Implementation accepted (`ACCEPTANCE_2026-11-11_R2.json`, PASS). WCTE regression PASS.
- X0: SK and JUNO recorded (Gate V failed, informational since the addendum; the S2 stops are lifted with the original records kept). HK X0 running on Deepthought.
- SK: generation running (Deepthought). JUNO, HK: waiting.
- Gate V ceiling diagnostic: code and tests committed; runs on spare CPU.

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
