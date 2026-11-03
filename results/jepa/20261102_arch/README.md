# Architecture/masking controls (window 128, window 32, 4-block predictor) — 2 Nov 2026

`registered: false`. Extension `arch_2026-11-02`; all runs `arm_name = `sigreg_jepa`; only the `EncoderConfig` differs (injected through a process-level `lucid_run.build_arm` wrapper, also active during evaluation).
Runbook: `docs/jepa/RUNBOOK_2026-11-02_ARCH.md` (sha256 `e45029ffd788f48f052fa286b0a2951b42a1ee5cbff1bda5eb6ace023f022b47`).
Author-approved training: exactly six 10 000-step runs (variants × seeds 9000/9001) on the committed 100k dev cohort and one 20-step `window128` seed-9000 smoke; no other training; the probe sets and data are reused read-only.

- code SHA: `eafbce6b3d0e9f6d1692cc005e9f825a967537d4`
- §4.2 reproduction: PASS

## Per-variant reading (two-sided, both seeds)

| variant | override | verdict | full vertex m | full token R2 | half vertex m | half token R2 |
|---|---|---|---:|---:|---:|---:|
| window128 | `{'window': 128, 'shift': 64}` | **not recovered** (seed 9000) | 0.163058 | 0.103526 | 0.192221 | 0.00637591 |
| window128 | `{'window': 128, 'shift': 64}` | **not recovered** (seed 9001) | 0.162922 | 0.142309 | 0.190853 | 0.00711801 |
| window32 | `{'window': 32, 'shift': 16}` | **not recovered** (seed 9000) | 0.215058 | 0.0246961 | 0.353435 | 0.00111242 |
| window32 | `{'window': 32, 'shift': 16}` | **not recovered** (seed 9001) | 0.206846 | 0.059136 | 0.331191 | 0.000544853 |
| pred4 | `{'n_pred_blocks': 4}` | **not recovered** (seed 9000) | 0.233639 | 0.0164786 | 0.334189 | 0.000970608 |
| pred4 | `{'n_pred_blocks': 4}` | **not recovered** (seed 9001) | 0.207866 | 0.0446054 | 0.346625 | 0.0031231 |

| variant | seed | wall s | steps | resumed | f_layout | f_event | eff. rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| window128 | 9000 | 3e+03 | 10000 | 0 | 0.967798 | 0.00128925 | 1.46421 |
| window128 | 9001 | 3e+03 | 10000 | 0 | 0.969654 | 0.000966151 | 2.03739 |
| window32 | 9000 | 2e+03 | 10000 | 0 | 0.996646 | 0.000138899 | 2.09904 |
| window32 | 9001 | 2e+03 | 10000 | 0 | 0.994616 | 0.000131256 | 3.32263 |
| pred4 | 9000 | 3e+03 | 10000 | 0 | 0.99611 | 6.88107e-05 | 2.92541 |
| pred4 | 9001 | 3e+03 | 10000 | 0 | 0.992509 | 0.000151547 | 3.50319 |

Criteria: recovered: both seeds full vertex strictly < 0.160072 m, full token ln q R2 >= 0.5, half vertex < 0.174821 m, half token R2 >= 0.5; not recovered: for each seed full vertex >= 0.160072 m or full token R2 < 0.5 (both seeds); else unclear

Interpretation: `any variant recovered` means the collapse is not architecture-robust for that knob; `none recovered` means the collapse is robust to these architecture/masking changes at the development scale (H4 ruled out for these knobs; a larger sweep is a separate decision).

## §4.2 reproduction

| item | expected m | recomputed m | |diff| m | pass |
|---|---:|---:|---:|---|
| fix9000 | 0.219258 | 0.219259 | 5.91047e-07 | True |
| init9000 | 0.160072 | 0.160072 | 3.09078e-07 | True |

## Outputs

- `runs.json` — keyed `<variant>__<seed>`: wall, steps, resumed_from_step, metrics_finite, encoder signature, full/half vertex, token charge R², f_layout, f_event, eff_rank.
- `readings.json` — `arch_r` verdicts and per-seed rows per variant.
- `reproduction.json` — §4.2 check.
- `plan.json` — gate, probes, data and entries projection.
- `README.md` — this file.

