# Modular noise simulator

Composable noise generation for detector-physics waveform studies.

Copyright (c) 2026 Dowling Wong. Released under the MIT licence (see
[`LICENSE`](LICENSE)). **If you use this package in published work, please cite
it** — see [`CITATION.cff`](../../CITATION.cff) at the repository root.

## What it provides

The package is organised into subpackages; `noise_module/__init__.py` re-exports
the public API so `from noise_module import NoiseGenerator` keeps working.

| Subpackage | Module | Role |
|---|---|---|
| `core` | `generator` | stationary Gaussian single-channel synthesis from an arbitrary one-sided PSD |
| `core` | `config`, `utils`, `templates`, `streaming` | config schema, RNG/array helpers, pulse/burst/glitch templates, chunked generation |
| `spectral` | `models` | composable analytic PSD components (white, power law, Lorentzian, roll-off, line, peaking, reflection, filtered) |
| `multichannel` | `generator` | correlated channels in independent, shared-private, spectral-shared-private and low-rank modes, returning both the implied and the realized covariance |
| `artifacts` | `injector`, `non_gaussian` | spectral lines, glitches, bursts, sparse artifacts; non-Gaussian innovations |
| `temporal` | `wrapper` | non-stationarity, piecewise stationarity, drift, local variance change |
| `resampling` | `psd` | alias-folding and in-band resampling of PSD densities between sampling rates |
| `validation` | `checks`, `calibration` | stationarity, Gaussianity, CSD-ensemble and artifact checks with bootstrap intervals; reference-dataset calibration |
| `budgets` | `reference_budget`, `al2o3_athermal`, `fit_al2o3_athermal`, `tes_budget` | closed-form athermal-calorimeter and HeRALD TES budgets; regenerate the reference tables |

Design rationale is in
[`docs/archive/noise_module/noise_generator_modular_design_spec.md`](../../docs/archive/noise_module/noise_generator_modular_design_spec.md).

## Reference data

The `data/Al2O3_Al_athermal/*.dat` tables are **generated build artifacts**, not
source data, and are git-ignored. Recreate them with:

```bash
python scripts/regenerate_reference_asd.py
```

See [`data/Al2O3_Al_athermal/README.md`](data/Al2O3_Al_athermal/README.md) for
the provenance of the reference budget and the evidence that it is analytic.

## Tests

```bash
pytest src/noise_module/tests
```
