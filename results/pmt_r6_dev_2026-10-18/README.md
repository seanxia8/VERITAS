# PMT R-6 dark-count development evidence (18 October 2026)

Runbook v2 repair R-6 (author decision, 18 Oct 2026; `docs/PMT_DECISIONS_LOG.md`). Local CPU only. Everything here is development output, not citable, `synthetic_dev`.

- `fixture_run_02/` (final): the same run on the code after the independent R-6 review fixes (E33/E34); all four checks pass.
- `fixture_run_01/`: the pre-review run of the 76-sensor prompt fixture with the manifest's dark rate of 1 kHz per PMT and a 1 µs readout window. Seeds, closure bounds and tolerances are the same as `../pmt_r1_r3_dev_2026-10-18/fixture_run_02`. All four predeclared checks pass. The reference cell averages 0.083 dark photoelectrons per event (expected 0.076). Command:
  `PYTHONPATH=src nice -n 10 $PMT_PYTHON -B -m noise_module_lucid.pmt_development --out results/pmt_r6_dev_2026-10-18/fixture_run_01`
- Dark-noise-only windows come from `noise_module_lucid.readout.dark_noise_window(params, n_channels, seed=..., run=..., window=...)`. The acceptance tests are in `src/noise_module_lucid/tests/test_dark_counts.py`.

`$PMT_PYTHON` = `/Users/wongdowling/Documents/VERITAS/.venv/bin/python` (Python 3.12.7, NumPy 2.5.3, SciPy 1.18.1).
