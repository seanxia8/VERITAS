# STATE — round 2 (single source of truth)

Updated: 2026-11-16 (phase r2-report, final). Extension `r2_multidetector_2026-11-11`, `registered: false`.

## Where we are

- **Round 2 complete.** Runbook and every addendum committed unchanged (Gate V ceiling, JUNO Titan v1/v2, caps, SK order/co-tenancy, HTCondor batch, staging dwong, batch reopen, A100 equal; batch staging withdrawn).
- SK, JUNO, HK: six main runs (mae_mse, target_posfree, sigreg_jepa × 9000/9001), `mae_mse_rf` 9000 and the triggered seed-9002 run trained; every model plus `init` 9000–9004 evaluated; readings R2-a to R2-e (two- and three-seed) final. No detector ended stopped.
- Gate V failed (absolute) on SK, JUNO and HK and is informational (author, 12 Nov 10:08). Photon-statistics ceilings for WCTE, SK, JUNO and HK are in `results/round2/20261112_gatev_ceiling/` (HK added in r2-report, no second transport).
- Ledger (final): L40S 53.51 / 72 h, Titan 0.81 / 160 h, batch 190.23 / 200 GPU-h; CPU generation 4.25 wall-h (22.68 / 48 in total), ceiling 1.31 / 3 wall-h.
- Report to the author: `~/Documents/VERITAS_offrepo/claude-code/reports/r2-report.md` (off-repo); run report `results/round2/R2_RUN_REPORT.md`.

## Deliverables (§10)

- Data: `results/round2/20261112_data/DATA_MANIFEST.json` (packs `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`).
- X0: `results/round2/20261112_x0/{sk,juno,hk}.json`, `hk_gatev.json`, `EXECUTION_<det>.json`, `R2_PREFLIGHT.md`.
- Training: `results/round2/20261112_train/runs.json`.
- Evaluation: `results/round2/20261112_eval/readings.json`, `READINGS_SUMMARY.md` (readings, cross-detector table, floor band, seed 9002, time-scale, Gate V caveat), `figures/` (vertex, f_event, particle tasks, RF − MAE, time scale).
- Gate V ceiling: `results/round2/20261112_gatev_ceiling/`.
- Records: this file, `DECISIONS.md`, `RESOURCE_LEDGER.json`.

## Paths

- Code: `src/jepa_noise/r2_geometry.py`, `r2_generator.py`, `r2_store.py`, `ext_r2.py`, `ext_r2_results.py`, `ext_r2_gatev.py`, `ext_r2_gatev_scaled.py`, `ext_r2_report_figs.py`, `scripts/jepa_r2.py`, `scripts/r2_batch/`, tests `src/jepa_noise/tests/test_r2_*.py`.
- Returns: `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/` (runs, batch/runs, evals, ledger, gatev_ceiling).
