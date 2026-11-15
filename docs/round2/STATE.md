# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook and addenda committed unchanged (Gate V, JUNO-Titan v2, caps, SK order/co-tenancy, HTCondor batch, staging dwong, batch reopen, A100 equal).
- SK: done — every model evaluated (L40S + A100), readings two/three-seed, R2-e; `EXECUTION_sk.json` final.
- JUNO: done — six main runs, `mae_mse_rf` and seed-9002 `sigreg_jepa` trained and evaluated on TOpAS A100 (42.05 batch GPU-h); `EXECUTION_juno.json` final.
- HK: six main runs, `init` 9000–9004 and `mae_mse_rf` trained/evaluated except: `mae_mse_rf` evaluation and seed-9002 `target_posfree` (training, cluster 759705) pending.
- Batch: GO on TOpAS A100 (WCTE regression PASS); NEMO2 unusable (xrootd unreachable). Ledger lines: L40S 72 h, Titan 160 h (JUNO smokes only), batch 200 h.

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
