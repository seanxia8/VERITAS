# STATE — round 2 (single source of truth)

Updated: 2026-11-12. Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- Runbook and addenda committed unchanged (Gate V, JUNO-Titan v2, caps, SK order/co-tenancy, HTCondor batch, staging dwong, batch reopen, A100 equal).
- SK: mae 9000/9001 done and evaluated; `init` 9000–9004 evaluated; sigreg 9000 and posfree 9001 training on deepthought2 (shared L40S); posfree 9000 (resume from 3,168) and sigreg 9001 on TOpAS A100; `mae_mse_rf` S1 on the L40S line (re-checked under the batch priority).
- JUNO: six main runs training on TOpAS A100 (batch, `R2_batch_gpu`); evaluation follows on A100.
- HK: X0 (yield, trigger, benchmark) running on Deepthought; then generation, pack, A100 smokes and runs.
- Batch: GO on TOpAS A100 (WCTE regression PASS); NEMO2 unusable (xrootd unreachable). Ledger lines: L40S 72 h, Titan 160 h (JUNO smokes only), batch 200 h.

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`,
  `scripts/jepa_r2.py`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Data (planned): shards `/work/dwong/r2_gen/<det>/`, packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`.
- Returns (planned): `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
