# E1b hand-off (detached continuation running)

State written by impl-e1 before ending its turn. The E1b continuation is **detached** and must
survive the session.

## Live processes (own sessions, not children of an agent turn)

| pid | role |
|---|---|
| 56883 | `/tmp/e1b_chain.py` — detached chain (session leader) |
| 56978 | `python -m jepa_noise.run E1b --workers 1`, attempt_003_20261017T153221Z, W=1, 57 rows, ETA ≈4 h from 17:32 |
| 57774 | `/tmp/e1b_post.py` — waits for 56883, then recomputes the missing M diagnostic, regenerates PREDICTIONS.md, writes `POST_STATUS.json` |

Check `ps -p 56883,56978,57774`. Logs: `detached_attempt.log`, `detached_post.log`.
Status: `CHAIN_STATUS.json`, later `POST_STATUS.json`. PID files: `detached_chain.pid`,
`detached_post.pid`.

## What already happened in this directory

- `attempt_003_20261017T141027Z_E1b` — 8 lean rows from the pre-detach run (kept, valid).
- `attempt_002_20261017T124115Z_E1b` — archived: W=2, stopped by its own resource guard
  (`resource_guard_host_free_ram`, peak worker RSS 1909.9 MB, min host free 2.76 GB), 33 pass +
  33 `not_run` placeholders. It also held the M diagnostic row.
- `attempt_001_20261017T123730Z_E1b` — empty pre-run (archived with 002).
- `SUPERSEDED_ATTEMPTS_001_002.tar.xz` (sha256 `e12ab027b8f109d2297296ae7d161acf5d10345f2832eadd9748344200ea0f80`)
  — bit-exact archive of both superseded attempts (66 row files verified). Contains the M row.
- `attempt_004_20261017T153209Z_E1b` — no-op continuation caused by the `not_run`-placeholder
  resume defect (see below); no rows written.
- `attempt_003_20261017T153221Z_E1b` — the running 57-row continuation.

## Resume defect (cause found)

A stopped attempt records one `not_run` placeholder row per unlaunched row. The default
`completed = set(discovered)` counted those placeholders as completed, so the first continuation
no-op'd (`completed_rows_found=65 pending=0`) and the chain archived attempt 002. Fix prepared in
`/tmp/patch_notrun_resume.py`: completed = recorded rows with `status != "not_run"`; adds a
regression test (not_run placeholder re-attempted) and keeps the two added tests (second attempt
reuses first attempt's rows with manifest back-pointer; stopped attempt's pass rows reused).
Apply **only after the chain and post exit** (no src edits while anything runs), then run
`PYTHONPATH=src .../python -m pytest src/jepa_noise/tests/test_run_e1.py -q` (expect 19 pass).

## Steps after the detached run finishes

1. Check `CHAIN_STATUS.json` (expect status complete, 65 discoverable grid rows from
   attempt_003_141027Z + attempt_003_153221Z) and `POST_STATUS.json` (expect M_development
   restored in a later attempt; all grid rows pass).
2. Apply `/tmp/patch_notrun_resume.py`; re-run the E1 tests (19 expected).
3. Regenerate `docs/jepa/PREDICTIONS.md` (`python -m jepa_noise.report E1`), run `--check` twice;
   confirm byte identity. Final E1b clauses determine R1/R2: E1a clause values already fixed
   (R1 pass, R2 fail 4/6 Spearman); E1b clean-cell median ≈16–19° (>10° → R1 fail) and g-sweep
   Spearman/seed-mean-angle at g=0.05 from the channel split.
4. Add `RESOURCE_LEDGER.json` rows: `20261017_wp3_e1b` final attempt (rows/workers/worker-seconds
   from its `attempt.json`) and the stopped attempt_002 (2.476 worker-h at W=2, status stopped,
   reason resource_guard_host_free_ram). Update E1 `charged_cpu_hours` with the accounted hours
   (worker-seconds/W/3600), failed attempts included.
5. Update `docs/jepa/WP3_E1_RUNNER.md` runtime table and the W=2 deviation/guard line
   (attempt_002: peak 1.91 GB/worker, min free 2.76 GB, stopped by guard; final run W=1), plus the
   E1b size/schema re-execution note.
6. Results size check: this directory must be ≤20 MB after the archive (expected ≈8–10 MB).
7. E1b row records now carry the full A4 `view.manifest()` per split plus the generator manifest,
   and `attempt.json` carries the population-preflight manifest (A4 provenance requirement).
