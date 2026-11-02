# Scale control (300k events / 30k steps) — 2 Nov 2026

`registered: false`. Extension `scale_2026-11-02`, variant `sigreg_jepa_scale300k` (`arm_name` stays `sigreg_jepa`).
Runbook: `docs/jepa/RUNBOOK_2026-11-02_SCALE.md` (sha256 `073d1532eb603565b7deede465161c924d8eb1195b9b15dc1dad322e2d8a1c70`).
Author-approved training: exactly two 30 000-step runs (seeds 9000/9001) on the new `registered=False` `dev_train_300k` cohort and one 20-step smoke; no other training.

- code SHA: `57295c46f831b97df26589616c13c549d6a5a012`
- cohort: `dev_train_300k__F1__0.05__s20261017__n300000x1000` (manifest `9a9dbe81ba63cea1c2ccad27e3d5cce7df9cb6e99faf0a9b340c6cec18c23a4a`, present=True, valid=True)
- entries projection: live 44949 -> projected 37253 (floor 30000)
- §4.2 reproduction: PASS

## §2 reading (two-sided, both seeds)

- **scale_r = not recovered**
- not recovered: the collapse persists at 3x data and 3x steps (optional pre-declared follow-up: 300k events at 10k steps)

| seed | wall s | steps | resumed | full vertex m | full token R2 | half vertex m | half token R2 | f_layout | f_event | eff. rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9000 | 7e+03 | 30000 | 0 | 0.32584 | 0.00316449 | 0.320643 | 5.62546e-05 | 1.00033 | 9.94665e-07 | 4.88331 |
| 9001 | 8e+03 | 30000 | 0 | 0.276314 | 0.00414104 | 0.319111 | 0.000249963 | 1.00033 | 1.21474e-06 | 5.28936 |

Criteria: recovered: both seeds full vertex < 0.160072, full token R2 >= 0.5, half vertex < 0.174821, half token R2 >= 0.5; not recovered: both seeds full vertex >= 0.160072 or full token R2 < 0.5

The effective epochs are nominally unchanged (300k x 3.2 = 30k x 32; the loader samples with replacement), so the run separates unique-data/steps scale from the development-scale collapse.

## §4.2 reproduction

| item | expected m | recomputed m | |diff| m | pass |
|---|---:|---:|---:|---|
| fix9000 | 0.219258 | 0.219259 | 5.91047e-07 | True |
| init9000 | 0.160072 | 0.160072 | 3.09078e-07 | True |

## Outputs

- `runs.json` — per seed: wall, steps, resumed_from_step, metrics_finite, cohort manifest sha256, full/half readout, f_layout, f_event, eff_rank.
- `readings.json` — the pinned `scale_r` verdict and per-seed values.
- `reproduction.json` — §4.2 check.
- `plan.json` — gate, cohort, probes, readout and entries projection.
- `README.md` — this file.

