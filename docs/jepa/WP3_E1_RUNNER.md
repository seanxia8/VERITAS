# WP3 E1 runner — how to run, grid, workers, runtime, outputs

Owner: impl-e1. Scope: `src/jepa_noise/run.py`, `report.py`, `configs/e1_grid_v4.json`,
`tests/test_run_e1.py`; results under `results/jepa/20261017_wp3_e1*/`. Runbook v4 §3.1/§3.2,
§6, §8 WP3 and v3 §5 WP3 are binding. `docs/jepa/PREDICTIONS.md` is generated only by
`python -m jepa_noise.report E1`; it is never hand-edited.

## Commands

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/private/tmp/jepa-venv-cpu311/bin/python

$PY -m jepa_noise.run E1a --smoke        # n in {1e3,1e4}, seed 0: 84 rows
$PY -m jepa_noise.run E1a                # full: 630 rows (skips recorded rows)
$PY -m jepa_noise.run E1a --rerun-all    # new attempt after an explicit code fix
$PY -m jepa_noise.run E1a --retry-failed # re-execute failed/not_evaluable rows
$PY -m jepa_noise.run E1b                # A4 law only; 65 (cell,seed) rows + M diagnostic
$PY -m jepa_noise.run E1c                # 10 (gamma,seed) rows x 3 depths
$PY -m jepa_noise.report E1               # regenerate docs/jepa/PREDICTIONS.md
$PY -m jepa_noise.report E1 --check       # byte-identity check
```

Every run creates a new `attempt_NNN_<UTC>_<campaign>/` directory. Row files are write-once and
never overwritten: a re-run goes to a new attempt, and `report` uses the latest record per
`(campaign, cell, seed)` row while older attempts stay for audit. Each attempt has
`attempt.json`, `manifest.jsonl` (one row per executed row with run_id, code SHA, config hash,
cell, seed, host, wall time, peak RSS, status), per-row logs, and `SHA256SUMS`. The code SHA is
`git rev-parse HEAD` at attempt start; scientific source hashes are recorded at start and end and
a mid-run change is flagged in `attempt.json`.

## Grids (from `configs/e1_grid_v4.json`, exactly as registered)

- **E1a**: d=40, k=3, r=1; θ scale {.1,.4,1}, η scale {1,3}, γ\_load {0,.1,.25,.5,.75,.9,1},
  n {1e3,1e4,1e5}, seeds 0–4 → 630 rows. Ten method rows: CCA, hard-isotropy (recorded as an
  equivalence of CCA in the reviewed solver — not an independent estimator), context PCA, exact-
  noise-whitened PCA, sample-calibration GEVP, finite-β {0.1,1,10}, conditional oracle, OF.
  Probe cohorts 10 000 train + 10 000 test with five 2 000-event grouped folds; independent
  calibration noise with n\_cal = n; angles use n=1e5 for R1/R2/R3/R3b/R4.
- **E1b**: A4 law (`v4-A4-waveform-v1`, base ρ=0), 13 unique cells (g sweep at line 0 + line sweep
  at g=0), seeds 0–4, four splits (channel primary; interleaved M alternative; time/block
  diagnostic only), n=1e5, streaming in ≤10 000-event float64 blocks. M diagnostic runs first at
  g=0.02, line=0, development seed 100, 4096+4096 records.
- **E1c**: `dynamics.py`, d=40, depths 1/2/3, 100 000 full-batch GD steps, lr 1e-3, θ scale .4,
  η scale 3, γ\_load {.5,1}, seeds 0–4 → 10 rows; R5 eligibility/censoring recorded per mode.

## Workers and memory

`W = min(physical cores − 2, floor(free_GB / 4))`, hard-capped at 3, measured at launch from
`vm_stat`; the measured inputs and chosen W are recorded in every `attempt.json`. E1a/E1c peak RSS
is ~0.1–0.3 GB/worker; E1b is streamed and estimated at 1–1.5 GB/worker (limit 4 GB). The runner
refuses to launch when the preflight projection would exceed the 24 h E1 cap (G2 belongs to the
Lead) and stops launching if the measured rate would exceed the remaining allocation.

## Outputs

- `results/jepa/20261017_wp3_e1a/attempt_*/` — row tables (pass/fail/not_evaluable/not_run with
  reason), manifest, logs; `aggregate_E1a.json` written by `report`.
- `results/jepa/20261017_wp3_e1b/`, `..._e1c/` analogous. Each results directory stays ≤ 20 MB
  (v4 §7); E1c curves are stored decimated with the full-curve digest.
- `docs/jepa/PREDICTIONS.md` — full table: E1 rows computed from stored tables, other campaigns
  from `PREDICTION_STATUS.json` only.

## Status semantics

`pass` — all registered per-row representation quantities evaluated; `fail` — uncaught exception
(recorded with traceback); `not_evaluable` — a registered per-row quantity (oracle eligibility,
Fisher rank, probe target) is ineligible; `not_run` — not launched (budget stop, missing campaign).
Prediction-level rows are `not_evaluable` and carry **no numeric evidence** whenever required
inputs are absent; v4 §4 adds an E1b clause to R1 and R2, so both stay
`not_evaluable / E1b_pending_v4_R1_R2_clause` until E1b rows exist.

## Engineering decisions (one line each)

- Workspace worker count is re-measured at each launch (v4 §6) rather than taken from DECISIONS.
- Per-solver exception isolation: a fixed-tolerance numerical failure in one registered solver is
  recorded as that row's failure, not allowed to erase the row's other registered quantities.
- E1a subject/probe/calibration namespaces are explicit `namespace_rng` roles; loading geometry
  stays keyed by the evaluation seed, subject draws include n.
- E1a calibration noise is drawn independently in the `E1a_noise_cal` role with covariance N\_cc
  (the reviewed `ExactModel.sample` already draws its own calibration records, unused here).
- Hard-isotropy is recorded as an explicit equivalence of CCA (reviewed solver maps both to β=0),
  labelled "not an independent estimator"; no second estimator was invented.
- R3 transform frozen before the corrected campaign: per (evaluation seed, physical cell) i.i.d.
  Gaussian matrices + 5·I for context and target (`r3_transform` in the config).
- E1b population displacement uses the committed `views.displacement` (A4.5); no duplicate formula.
- E1b sample-covariance accumulation uses one raw generation pass sliced by all four splits; oracle
  whitening is applied to accumulated float64 covariance blocks (algebraically identical to
  transforming events, avoids a second n=1e5 pass).
- E1b probe cohorts are materialised (10 000+10 000, allowed by A3) and reused across splits.
- Criterion M stores scores and projections (`.npy` + SHA-256) under the attempt directory.
- `report E1` and `report all` share one code path and are byte-identical while only E1 has tables.

## Incidents retained (not repaired by tuning)

### R3b: spec-compliance bug, corrected-implementation outcome (not a clean first outcome)

The first E1a full run reported R3b as fail with oracle sample angles of ~32 degrees on all cells.
Verification found an implementation defect: the conditional-oracle row reported angles and probe
embeddings measured on the **whitened-coordinate** functional instead of the context map pulled
back to original context coordinates. The registered definition (v3 Sec.4 "principal angles are
measured between row spaces, mapped back to original context coordinates"; P2 "map directions back
before comparing") was therefore not implemented for that row.

It was fixed (`sample_oracle` context is now used for angles and probe features), and the whole
E1a grid was re-executed in a fresh write-once attempt. No row was overwritten. Superseded attempts
001–006 are preserved bit-exactly in
`results/jepa/20261017_wp3_e1a/SUPERSEDED_ATTEMPTS_001_006.tar.xz` (sha256
`2c98e2bc36961807c7bbaab61a116374c53cc62688586c3c1b66c7e399aea1cc`); the aggregate records the
same provenance. R3b after the fix: pass, 42/42 theorem-eligible cells, max sample median angle
well inside 5 degrees, population recovery <=1e-6 rad. This is a corrected-implementation outcome
and a fresh reviewer must recompute it from the raw tables; it is not presented as a clean first
outcome.

### R3 numerical solver failure

Two of 210 R3 transformed-CCA diagnostics raise the reviewed solver's fixed isotropy/objective
residual check. This is recorded (`r3_solver_tolerance_failure_2_of_210_rows`); the 208 computable
rows have max spectrum difference 8.2e-10 <= 1e-8. No solver setting or transform was changed.

### E1c initialisation check (v3 Sec.4.2) — no initialisation defect found

`dynamics.initialize` does not use the repository's `init_weights`/muP zero-init: it draws every
square encoder layer and the predictor as independent Gaussian N(0, (1e-3)^2) entries from the
E1c namespaces, exactly as v3 Sec.4.2 requires (no zero entries; layer Frobenius norms 0.040).
The depth-2/3 encoders are frozen by construction of that registered protocol: their encoder norms
are 2.5e-4 and 1.6e-6 (products of the same 1e-3-scale layers) and their gradient scale vanishes
with depth, so 100 000 steps at lr 1e-3 leave their mode contributions at the initialisation floor.
This is a registered-protocol consequence, so E1c was not re-run and no hyperparameter was changed;
R5 is reported `not_evaluable` (censored depth-2/3 modes), consistent with "a finite-budget
inconclusive result remains inconclusive".

## Runtime log

| run | rows | W | wall | worker-hours |
|---|---|---|---|---|
| E1a smoke (attempt_001) | 84 | 3 | 31 s | 0.025 |
| E1a full (attempt_002) | 546 new (630 total) | 3 | 205 s | 0.170 |
| E1a retries (attempt_003–005) | 2+2 | 3 | ≤4 s each | <0.005 |
| E1a corrected full (attempt_006) | 630 | 1 | 731 s | 0.202 |
| E1a final compact (attempt_007) | 630 | 1 | 545 s | 0.151 |
| E1c full (attempt_001) | 10 | 1 | 405 s | 0.112 |
| E1b full (attempt_002, running) | 65 + M | 2 | see attempt.json | see attempt.json |

Budget reference: projection 13.86 h cumulative at 1 worker for the whole E1 campaign; the E1 cap
is 24 h with 0.995 h already charged. Ledger rows are added by the Lead-owned ledger process.
