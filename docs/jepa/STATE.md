# STATE — JEPA study (single source of truth)

Updated: 2026-10-17 12:10Z by Lead. Runbook v4 registered at `151f26e`.

## Where we are

- **WP0-v4**: registration commit `151f26e` (runbook v4 + prompt + check script, unchanged). WP3
  E1 budget artifacts at `dab95d8`. Hygiene (v4 §7) done: 7 audit files (3.11 MB) removed from
  `results/jepa/20261016_storage_recovery/`, README + SHA256SUMS + summary JSON kept, history
  retained. CPAL date check done (Recent Spotlight: **18 Jan 2027 23:59 AoE**). Prediction status
  updated to v4 (R6–R8 `not_run` "v4 A4/A7 redesign"; B1/R7b/R11/R11b/R12 added `not_run`).
- **Phase 0 preflight**: op-deepthought and op-deepthought2 running read-only preflight (§6.1
  environment discovery). Reports go to `PREFLIGHT_A8_deepthought.md` / `..._deepthought2.md`.
- **Guard (§6.2)**: impl-guard writing `scripts/jepa_guard.py` + `jepa_run.sh` + tests. Must pass
  its tests AND a fresh review before the operators dry-run it 10 min per host.
- **WP2-v4 / E1**: impl-wp2v4 on the A4 law in `views.py` + population tables (13 cells × 4 splits,
  D); impl-e1 on the E1 runner, E1a smoke first.
- **WP4a bridge**: impl-bridge on `backbone.py` token grid (read-out before `final_proj_norm`, fixed
  per-channel scaling), arms + ablation + leakage/amplitude tests.

## Running jobs

None on the hosts. Local: only CPU work by implementers (≤ 3 workers total).

## Queue

- deepthought: nothing (preflight only) → first ≤10-min probe after guard review + dry run.
- deepthought2: same. Nothing starts before its guard status is OK and < 2 min old.

## Guard status at last check

- deepthought: no guard yet (not deployed). deepthought2: no guard yet (not deployed).
- No HOLD/SHED events. `INFRA_LOG.md` not yet created.

## Next three actions

1. Harvest preflight reports; write `PREFLIGHT_A8.md`; decide the environment per §6.1 order
   (existing read-only env vs one new env under `/ceph/dwong/jepa/env/`; G-ENV only as last resort).
2. Review guard code + tests; send it to a fresh reviewer; then 10-min dry run per host, then one
   ≤10-min GPU probe per host (imports, one forward pass, env freeze, timings).
3. Merge WP2-v4 + E1 + bridge diffs when their tests pass; commit; push; report to the author.

## Open gates

- None open. Awaiting (not gates): PMT-branch SHA (blocks reported LUCiD results only);
  drift ranges s/s_p/a from Junjie (defaults in force); `docs/LUCID_MODEL_LINEUP_2026-10-09.md`
  untracked by author's instruction.

## Budget (RESOURCE_LEDGER.json)

- E1 CPU: 0.995 h charged of 24 h (projection 13.86 h incl. reserve and charge → fits).
- E2 (v3 row): 0 of 72 CPU-h; LUCiD/Titan: 0 of 300 GPU-h; LUCiD/L40S: 0 of 80 GPU-h.
- LUCiD generation CPU: 0 of 48 wall-h. E3: 0 of 48 CPU-h / 0 of 24 GPU-h. E4: 0 of 4 / 0 of 2.

## Last verified SHA on hosts

- deepthought: not yet (preflight in progress). deepthought2: not yet.

## Environment decisions

- Local: `/private/tmp/jepa-venv-cpu311/bin/python` (accepted v3 env, 75 tests pass).
- Hosts: pending preflight; order per §6.1 step 1 (existing read-only env) → step 2 (one new env,
  needs ≥100k free entries) → G-ENV.

## Research status

`PREDICTION_STATUS.json` v4: R1–R5, R3b, R9, R10 `not_run`; R6–R8 `not_run` (A4/A7 redesign);
B1, R7b, R11, R11b, R12 `not_run` (A7/A8). No reported result exists yet.
