# WP3 E1c results

Campaign `20261017_wp3_e1c`, runbook v3 §4.2 / v4 §8 WP3, approved A2 binding. 10 rows =
γ\_load {0.5, 1} × seeds 0–4; each row runs three depths (L=1/2/3) of the square tied stop-gradient
encoder on d=40 with 100 000 full-batch GD steps at lr 1e-3, fixed n=100 000 empirical moments.

## Attempt

`attempt_001_20261017T121545Z_E1c/` — 10/10 rows complete; **all 10 rows `not_evaluable`**
(`mode_learning_ineligible`). One worker (v4 §6 measured rule), wall 405 s, 0.112 worker-hours,
code SHA `ed3b199ec27016c8dca361f5cbae66b01c0af5c8`. `attempt.json`, `manifest.jsonl`,
`SHA256SUMS` and per-row logs are present.

## Outcome and check

Depths 2 and 3 are effectively frozen: encoder norms 2.5e-4 and 1.6e-6 are the expected products of
the registered N(0,(1e-3)^2) layers, and their mode contributions stay at the initialisation floor
(`insufficient_absolute_growth` / `below_absolute_floor`). Depth 1 learns (contributions grow ~35×,
nuisance/physics learning times 2 395–14 264 steps), but R5 needs both L=1 and L=3 evaluable, so all
seeds are censored and **R5 = `not_evaluable` (`seed_evaluable_lt_4`)**. This is the honest
inconclusive result required by the runbook; no hyperparameter, horizon or schedule was changed.

`dynamics.py` was checked against v3 §4.2: it uses explicit Gaussian SD 1e-3 for every encoder layer
and the predictor and does not inherit the repository's `init_weights` zero-init, so the freezing is
a registered-protocol consequence, not an initialisation defect. No E1c re-run was made.

Curves are stored decimated by 50 steps (plus the terminal step) with the full-curve SHA-256 in each
row record; the full traces are reproducible from the recorded namespaces, config and code SHA.

## Regeneration

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/private/tmp/jepa-venv-cpu311/bin/python -m jepa_noise.report E1
```
