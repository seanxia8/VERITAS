# PMT R-1 to R-3 development evidence (18 October 2026)

Runbook v2 §4 repairs, run on local CPU. Everything here is development output, not citable, `synthetic_dev`.

- `fixture_run_02/` (final): the 76-sensor prompt fixture re-run with the response built from `configs/lucid_pmt/reference.json` (R-1) and the D-TQ slope cells (R-2). It uses 4000 photons per event and the same seeds and closure bounds as `pmt_dev_2026-10-16_wp0_wp5/prompt_run_05`. All four predeclared checks pass. Command:
  `PYTHONPATH=src nice -n 10 $PMT_PYTHON -B -m noise_module_lucid.pmt_development --out results/pmt_r1_r3_dev_2026-10-18/fixture_run_02`
- `fixture_run_01/`: the same run before the independent-review fixes. It differs only in the offset control's family label, the scope of the early-digit and overflow counts, and the code hashes. Its checks and K values are identical to `fixture_run_02`. It is kept as the pre-review record.
- `tq_validation.py` → `tq_validation.json`: the R-2 numbers. The conditional timing mean is checked against the configured curve at 0.25, 1, 2, 5 and 20 PE (within 3σ_MC), along with the slope-cell shifts, charge and membership preservation, and the D-J early-digit fraction. Each charge point has its own event ID, so its jitter draws are independent of the other points: z = −1.49, −0.37, 0.00, 0.88, −0.97.
- The R-3 constant-predictor table for the past multi-energy run is in `../pmt_multienergy_dev_2026-10-16/constant_baseline_comparison.{json,md}`.

`$PMT_PYTHON` = `/Users/wongdowling/Documents/VERITAS/.venv/bin/python` (Python 3.12.7, NumPy 2.5.3, SciPy 1.18.1).
