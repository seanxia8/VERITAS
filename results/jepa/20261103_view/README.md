# View design: four candidate splits — 2 Nov 2026

`registered: false`. Extension `view_design_2026-11-03`; all runs `arm_name = `sigreg_jepa`; only the runtime mask/split differs (process-level `SplitMasker` patch of `lucid_run.ViewMasker` + `point_encoder.ViewMasker`, training-only).
Runbook: `docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md` (sha256 `993755d7dc77b917a29566d4cf9e79b25a0146e1364b3b91b06b76456e2920e6`).
Author-approved training: exactly eight 10 000-step runs (four splits × seeds 9000/9001) on the committed 100k dev cohort and one 20-step `module_parity` seed-9000 smoke; no other training and no new data (masks are runtime only).

- code SHA: `f5b7434c4eed438a3eb0f5a703da086a763d19f1`
- random_half salt: 20260101; probe split for every evaluation: `within` (registered)
- §4.2 reproduction: PASS

## Per-split reading (two-sided, both seeds)

| split | verdict | seed | full vertex m | full token R2 | half vertex m | half token R2 | f_layout | f_event | eff. rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| within | not recovered | 9000 | 0.219284 | 0.016902 | 0.320263 | 0.000463106 | 0.997046 | 7.46629e-05 | 2.66922 |
| within | not recovered | 9001 | 0.206562 | 0.0542486 | 0.282871 | 0.00157434 | 0.99342 | 0.000172535 | 2.88877 |
| across | not recovered | 9000 | 0.349382 | 0.000804 | 0.312176 | 0.00222551 | 1.00035 | 4.74536e-07 | 14.2541 |
| across | not recovered | 9001 | 0.320287 | 0.00136682 | 0.286121 | 0.00220302 | 1.00036 | 4.00855e-07 | 14.2329 |
| module_parity | not recovered | 9000 | 0.185144 | 0.0730999 | 0.220342 | 0.0737055 | 0.980006 | 0.00195129 | 1.17459 |
| module_parity | not recovered | 9001 | 0.164888 | 0.192995 | 0.199977 | 0.100348 | 0.935353 | 0.0112449 | 1.21485 |
| random_half | not recovered | 9000 | 0.216066 | 0.0174948 | 0.294098 | 0.00233613 | 0.99721 | 7.98103e-05 | 2.15343 |
| random_half | not recovered | 9001 | 0.181205 | 0.0793844 | 0.22079 | 0.00680989 | 0.9868 | 0.000507725 | 1.91497 |

| split | wall s | steps | resumed |
|---|---:|---:|---:|
| within | 2e+03 | 10000 | 0 |
| within | 2e+03 | 10000 | 0 |
| across | 2e+03 | 10000 | 0 |
| across | 2e+03 | 10000 | 0 |
| module_parity | 3e+03 | 10000 | 0 |
| module_parity | 2e+03 | 10000 | 0 |
| random_half | 2e+03 | 10000 | 0 |
| random_half | 3e+03 | 10000 | 0 |

## Ranking and recommendation

| rank | split | verdict | mean full vertex m | max | min |
|---:|---|---|---:|---:|---:|
| 1 | module_parity | not recovered | 0.175016 | 0.185144 | 0.164888 |
| 2 | random_half | not recovered | 0.198636 | 0.216066 | 0.181205 |
| 3 | within | not recovered | 0.212923 | 0.219284 | 0.206562 |
| 4 | across | not recovered | 0.334834 | 0.349382 | 0.320287 |

**Recommendation:** no split in this set makes the physics learnable at the development budget; the view is not the lever (the target-content mechanism is) — best-ranked is module_parity (mean full vertex 0.175016 m), still above the floor

Criteria: recovered: both seeds full vertex strictly < 0.160072 m, full token ln q R2 >= 0.5, half vertex < 0.174821 m, half token R2 >= 0.5; not recovered: for each seed full vertex >= 0.160072 m or full token R2 < 0.5 (both seeds); else unclear

Caveat (runbook §2): `half_pass` always uses the registered `within` target mask, so for `module_parity`/`random_half` it is the same fixed probe as for every other run — comparable, but not "the pass the run saw during training".

## §4.2 reproduction

| item | expected m | recomputed m | |diff| m | pass |
|---|---:|---:|---:|---|
| fix9000 | 0.219258 | 0.219259 | 5.91047e-07 | True |
| init9000 | 0.160072 | 0.160072 | 3.09078e-07 | True |

## Outputs

- `runs.json` — keyed `<split>__<seed>`; wall, steps, resumed_from_step, metrics_finite, probe split, full/half vertex, token charge R², f_layout, f_event, eff_rank, readings, ranking, recommendation.
- `readings.json` — per-split `view_r` verdicts/per-seed rows, ranking and recommendation (pinned schema).
- `reproduction.json` — §4.2 check.
- `plan.json` — gate, probes, data and entries projection.
- `README.md` — this file.

