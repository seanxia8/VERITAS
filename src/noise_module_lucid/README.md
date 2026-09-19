# `noise_module_lucid` — LUCiD front-end noise

Thin adapter **on top of** [`noise_module`](../noise_module/): LUCiD PMT
front-end presets, the photoelectron → mV units bridge, the covariance-unit
grouping, and the declared intervention families. Nothing in `noise_module` is
forked, and **nothing here imports LUCiD**, so the package builds and tests
without a LUCiD clone (arm A is licence-gated on A0).

Full physics explanation: [`docs/noise_module_lucid.md`](../../docs/noise_module_lucid.md).
Design: `docs/EXPERIMENT_DESIGN.md` §II.7, §III.6.

## What it supplies

| module | what |
|---|---|
| `presets.py` | `PMT_FRONTEND_V2` (512 ns grid) and `PMT_FRONTEND_LONG` (16–32 µs κ grid), single-channel and crate (`spectral_shared_private`) views, provenance, `crate_preset()` |
| `units.py` | `spe_template`, `charge_to_mv` — the pe → mV bridge |
| `grouping.py` | `groups_from_string_id`, `groups_from_positions` (sector × height band), `contiguous_groups` — the covariance unit |
| `adapter.py` | `add_readout_noise`, `crate_noise`, `kappa` — crate-wise noise with the implied/realized covariance per group |
| `interventions.py` | the declared N families (`clock_scale`, `add_broadband_common_mode`, `decimate_alias_fold`, `gain_drift_apply`, `channel_loss_apply`, `cable_lag_apply`) and `DOCUMENTED_NOT_IMPLEMENTED` |

## Use

```python
import numpy as np
from noise_module_lucid import add_readout_noise, groups_from_positions

charge = np.zeros((64, 32768))          # LUCiD (C, N) photoelectron histogram
groups = groups_from_positions(positions, n_sectors=8, n_bands=4)
trace_mv, meta = add_readout_noise(charge, groups=groups, window_ns=32768, seed=0)
print(meta["kappa"], meta["groups"])
```

```bash
PYTHONPATH=src python -m pytest src/noise_module_lucid/tests -q   # 12 passed
```

## Status

V2 and the long-window preset are implemented and tested; every constant is a
**placeholder** until read from a measured front-end or LUCiD's own gain. The
16–32 µs re-parameterisation is a declared open decision (switching lines, 1/f
reference); 50 Hz mains remains unrepresentable without decimation or a ≥ 1 s
record.
