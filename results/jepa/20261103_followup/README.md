# MAE/JEPA follow-up — Stage A readouts and T-1 control

`registered: false`. Extension `mae_jepa_followup_2026-11-03`; runbook `docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md` (sha256 `063e908774ffe29b3d59ef40ab54ef5d02ff462b6aa1e49eef57c61c49e7af24`).

- code SHA: `feed8c959808041590e64f4f6c56b17305e1197a`
- Stage A projection: 4.9 L40S h (cap 5.5); entries pass: True
- inventory: 54 labels, 0 unavailable, 0 pending
- §3 A1 reproduction: PASS

## Stage A readouts

| item | complete | unavailable |
|---|---|---|
| A-1 | None | `[]` |
| A-1b | False | `[{'label': 'A-1b', 'reason': 'label collector reproduction did not pass'}]` |
| A-2 | True | `[]` |
| A-3 | True | `[]` |
| A-4 | True | `[]` |

## T-1 runs

| label | split | seed | checkpoint sha256 |
|---|---|---:|---|
| t1_s030_within_trigger9000 | within | 9000 | `683f3ef9474ed0be3febcdc005bd4bbb0ed8ecd793b6df580b655e7f25462dbb` |
| t1_s030_within_trigger9001 | within | 9001 | `415b38031182fe4d5dd5f444ff12febe84cdd6a2e9ef9ce3774ab5a0e7bd8611` |
| t1_s030_across_trigger9000 | across | 9000 | `accc9016b9b2270777033f76fc15636ecaca17da6c4531e0853816f78c30a475` |
| t1_s030_across_trigger9001 | across | 9001 | `a8752fde554f7715bc96a67c3f5b717131e313155ae84a6497216710b6d193f8` |
| t1smoke_s030_within_trigger9000 | within | 9000 | `b5993029c93ddb23d3fcbdcebfb320d3fe4053e92459ae89c39b3bce9bf11d90` |

## T-1 GPU cost (N4)

- cap: 3.5 L40S GPU-h
- projected: 2.7 h
- measured (sum of completed-run wall_seconds): 2.13 h
- pass: True

## Readings (§3/§4, fixed in advance)

- A-1: `readings.json` -> `A-1` (J-a variant classes, J-b MAE checkpoints beside init/E3/static, J-c Spearman over the fixed population).
- A-1b: `readings.json` -> `A-1b` (M/C/All MLP R² and the ordered classes; both seeds must agree).
- A-2: `readings.json` -> `A-2` (gain R² minus the environment floor, refit and frozen vertices, log-yield R², paired run-cluster bootstrap).
- A-3: `readings.json` -> `A-3` (μ/e AUC and balanced accuracy overall and in NHits quintiles, direction ridge/MLP angular errors, contained-energy probes under the 500 rule, raw-feature and random-init baselines).
- A-4: `readings.json` -> `A-4` (material/immaterial/unclear per model, Δ token charge R² descriptive).
- T-1: `readings.json` -> `T-1` (G-1 retained/suppressed/intermediate; G-2 paired vertex difference, descriptive).

## Outputs

- `plan.json`, `inventory.json`, `reproduction.json`, `runs.json`, `readings.json`, `README.md` (§6.5 deliverables).


## Acceptance conditions C0-C8 and author decisions D1-D5 — how each was applied

| id | how it was applied |
|---|---|
| C0 | The 6 Nov acceptance-fix patch was applied byte-identically with `git am` as `7214e99` (diff sha256 `9e82ec31…f187`, blobs `4cb938da…`/`9791a792…`). It gives the 20-step T-1 smoke the `t1smoke_` label, which `readout_label_list` never evaluates, so `--include-t1` readouts evaluate the four 10 000-step runs only. Round 3 re-review PASS. |
| C1 | Recorded exactly: focused `test_ext_followup.py` **56 passed** (624.41 s); full suite with only the three long-standing `test_guard` launcher tests deselected **669 passed, 2 skipped, 3 deselected** (815.58 s), exit 0. One DECISIONS line supersedes the unsupported "pass on deepthought" wording without editing old lines. |
| C2 | `plan.json` reports only the conservative Stage A projection **107.8 item-equivalents = 4.9049 L40S h**; the runbook-rule figure (87.8 = 3.9949 h) is written nowhere. One DECISIONS line corrects the S3 record. |
| C3 | Entries re-checked live at A0 and before T-1. A0 snapshot: free **37 986 → projected 32 102 ≥ 30 000**, pass. The dry run's own entries sub-check (free 34 316 → 28 432) had run while the two pytest basetemps were still present; they were removed (3 670 entries, the Phase-1 `clean_mine.sh` practice) before the execution preflight, and nothing else was deleted. |
| C4 | The six D2 s = 0.05 vertices of ijepa/lambda/stopgrad are pinned (`D2_S005_VERTEX`, `d2_s005_committed_vertices`) and reproduce as targets in A-1 (table below); `anc100` and the T-1 checkpoints stay `not_applicable`. One DECISIONS line narrows the 3 Nov S1 record. |
| C5 | The A-3 and A-4 reproduction tables below are built from the per-model records (`models[label].tasks.photon_reproduction`), not from the summary that loses the item label. |
| C6 | T-1 cost is the sum of launcher wall time over every T-1 attempt including the smoke (table below), not `t1_cost()`. |
| C7 | Stage A GPU time is measured from each job's launcher status (the payloads write `wall_seconds = null`); one CLI call is one item and the cap is checked before each call. |
| C8 | Checked read-only before the A-1b launch: `/ceph/dwong/jepa/returns/ext_h12_20261101/e0e1_rows.npz` exists and `ext_h12._read_label_cache` accepts it (`fix_sha256 2d912a2d…`, `max_label_events 2000`, 64 000 rows on 2 000 events). A-1b then failed its own collector-reproduction requirement and is `unavailable` (its own §3 A-1b rule), so no further GPU time was spent on it. |
| D1 | Stage A execution cap 6.5 L40S h; the A0 projection gate stayed at 5.5 (projection 4.9049 passes it). Measured Stage A total is in the table below. |
| D2 | No guard bypass: every GPU job went through `scripts/jepa_run.sh`. Guard HOLD/SHED was treated as wait-and-retry, never a STOP (one A-4 attempt was SHED when another user's 19.6 GB job took GPU 0 to 98.6 %; the attempt is booked and A-4 was rerun). |
| D3 | Deepthought stayed reachable for A0; deepthought2 dropped long-lived SSH sessions during the night, so the CPU steps (focused tests, full suite, dry run, A0 re-checks) ran on deepthought2 CPU with `nice -n 10` and 4 threads. GPU steps stayed on deepthought2 under the guard. |
| D4 | All execution ran from the author's Mac over SSH; the commits were pushed to `JEPA` from the Mac only (the hosts have no GitHub credentials). |
| D5 | A-1 and A-2 ran once each with `--include-t1` as soon as the four T-1 runs completed, ahead of the pending A-1b/A-4/A-3. The Stage A/T-1 split of their wall time is 39/43 + 4/43 and 8/16 + 8/16. No reading, threshold or item changed. |


## C4 — ijepa / lambda / stopgrad D2 s = 0.05 reproduction

| label | pinned (m) | recomputed (m) | abs diff (m) | status |
|---|---:|---:|---:|---|
| ijepa9000 | 0.20915 | 0.20915 | 2.37455e-07 | pass |
| ijepa9001 | 0.194992 | 0.194992 | 2.78461e-07 | pass |
| lambda9000 | 0.220055 | 0.220055 | 1.33466e-07 | pass |
| lambda9001 | 0.199603 | 0.199603 | 6.86096e-08 | pass |
| stopgrad9000 | 0.219583 | 0.219583 | 4.19696e-08 | pass |
| stopgrad9001 | 0.189377 | 0.189377 | 1.54596e-07 | pass |

6/6 pinned D2 s = 0.05 rows reproduced within 1e-3 m; a mismatch would have marked the row `unavailable` and excluded it from J-a and J-c (runbook §3 A1, acceptance C4).

## C5 — A-3 reproduction (per-model records)

| model | expected (m) | recomputed (m) | abs diff (m) | status |
|---|---:|---:|---:|---|
| fix9000 | 0.219259 | 0.219259 | 0 | pass |
| fix9001 | 0.204281 | 0.204281 | 0 | pass |
| frozen_target9000 | 0.138459 | 0.138459 | 0 | pass |
| frozen_target9001 | 0.131436 | 0.131436 | 0 | pass |
| ijepa9000 | 0.20915 | 0.20915 | 2.37455e-07 | pass |
| ijepa9001 | 0.194992 | 0.194992 | 2.78461e-07 | pass |
| init9000 | 0.160072 | 0.160072 | 0 | pass |
| mae_within_median9000 | 0.0852425 | 0.0852425 | 0 | pass |
| mae_within_median9001 | 0.0950288 | 0.0950288 | 0 | pass |
| mae_within_trigger9000 | 0.104916 | 0.104916 | 0 | pass |
| mae_within_trigger9001 | 0.116542 | 0.116542 | 0 | pass |

11/11 per-model rows reproduced their committed full-input vertex within 1e-3 m (11 models). Built from the per-model records, not the summary (C5).

## C5 — A-4 reproduction (per-model records)

| model | expected (m) | recomputed (m) | abs diff (m) | status |
|---|---:|---:|---:|---|
| fix9000 | 0.219259 | 0.219259 | 0 | pass |
| fix9001 | 0.204281 | 0.204281 | 0 | pass |
| frozen_target9000 | 0.138459 | 0.138459 | 0 | pass |
| frozen_target9001 | 0.131436 | 0.131436 | 0 | pass |
| mae_within_median9000 | 0.0852425 | 0.0852425 | 0 | pass |
| mae_within_median9001 | 0.0950288 | 0.0950288 | 0 | pass |
| mae_within_trigger9000 | 0.104916 | 0.104916 | 0 | pass |
| mae_within_trigger9001 | 0.116542 | 0.116542 | 0 | pass |

8/8 per-model rows reproduced their committed full-input vertex within 1e-3 m (8 models). Built from the per-model records, not the summary (C5).

## C6 / C7 — measured L40S hours per job

| job | wall (h) | Stage A share (h) | T-1 share (h) | outcome |
|---|---:|---:|---:|---|
| `a1_incl_t1_20261106` | 2.7365 | 2.4819 | 0.2546 | exited/exit=0 |
| `a1b2_20261106` | 0.0106 | 0.0106 | 0.0000 | exited/exit=2 |
| `a1b_20261106` | 0.0009 | 0.0009 | 0.0000 | exited/exit=1 |
| `a2_incl_t1_20261106` | 0.1126 | 0.0563 | 0.0563 | exited/exit=0 |
| `a32_20261106` | 0.5395 | 0.5395 | 0.0000 | exited/exit=0 |
| `a3_20261106` | 0.0008 | 0.0008 | 0.0000 | exited/exit=1 |
| `a42_20261106` | 0.2630 | 0.2630 | 0.0000 | signalled/exit=143 |
| `a4_20261106` | 0.0008 | 0.0008 | 0.0000 | exited/exit=1 |
| `a4r1_20261106` | 0.6791 | 0.6791 | 0.0000 | exited/exit=0 |
| `followup_a1_reproduce_20261106` | 0.5514 | 0.5514 | 0.0000 | exited/exit=0 |
| `t1_s030_across_trigger_s9000_20261106` | 0.5596 | 0.0000 | 0.5596 | exited/exit=0 |
| `t1_s030_across_trigger_s9001_20261106` | 0.5567 | 0.0000 | 0.5567 | exited/exit=0 |
| `t1_s030_within_trigger_s9000_20261106` | 0.5776 | 0.0000 | 0.5776 | exited/exit=0 |
| `t1_s030_within_trigger_s9001_20261106` | 0.5790 | 0.0000 | 0.5790 | exited/exit=0 |
| `t1_smoke_20261106` | 0.0973 | 0.0000 | 0.0973 | exited/exit=0 |

- **Stage A measured: 4.5844 L40S h of the 6.5 h cap (D1)**
- **T-1 measured: 2.6811 L40S h of the 3.5 h cap**
- Failed attempts are included (A-1b/A-4/A-3 CLI-kwarg failures, the A-4 guard SHED attempt).
- The `--include-t1` walls are split 39/43 + 4/43 (A-1) and 8/16 + 8/16 (A-2) per D5.

## Large arrays and environment

The large arrays stay on `/ceph` and are not committed; `LARGE_ARRAYS.json` records every file > 5 MB with its byte size and sha256 (11 files, 208 593 085 bytes): the five T-1 checkpoints (`final.pt` + `checkpoint.pt` per run, ~7.5 MB each under `<ext>/{train,smoke}/runs/`) and the committed fix9000 label cache `/ceph/dwong/jepa/returns/ext_h12_20261101/e0e1_rows.npz` (133 683 850 bytes) that A-1b reads under C8.

**Interpreter.** The deepthought2 GPU jobs ran `/home/dwong/anaconda3/bin/python3` (Python 3.12.2, conda-forge GCC 12.3.0): the launcher default in `scripts/jepa_run.sh:43` (`JEPA_PYTHON` was never set in this execution), confirmed by the live guard status `guard.interpreter` and by the same default in the latest deepthought2 run manifests (arch/view). A0/report used the same interpreter on the CPU hosts; the A0 `plan.json` records that environment (`python 3.12.2`, numpy 1.26.4, torch 2.5.1+cu121 on Deepthought).
