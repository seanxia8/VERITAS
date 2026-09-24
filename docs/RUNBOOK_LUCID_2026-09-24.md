# Runbook — LUCiD arm: simulation → dataset → frozen, interpreted representation

_24 Sep 2026. Scope: turn the LUCiD arm from a notebook prototype into a
reproducible dataset, validate the monitor stack on the linear subject, then
train and interpret one frozen nonlinear subject. Every command is run from the
repository root with `PYTHONPATH=src`._

This runbook closes the four "missing before it is a data pipeline" items and
then carries the two training steps. The code lives in
`src/noise_module_lucid/` (dataset driver, readouts, validation) and
`src/latent_monitor/` + `src/reconstruction_model/` (subjects).

---

## 0. Gates and scope (read first)

| gate | status | effect |
|---|---|---|
| **A0 — LUCiD licence** | **open** (the fork ships no `LICENSE`) | you may **produce and analyse** data internally; you may **not release** it. Every cell's `provenance.json` records `release.releasable = false` until A0 and calibration are both resolved. |
| **Calibration** | placeholders (`readout.calibrated = false`) | absolute levels (rms, mV/pe) are not validated; the *ratios* and the covariance structure are what the study uses. |
| **Readout coverage** | 1 GHz PMT only | IceCube-DOM / SiPM / TES readouts are declared but **not modelled** (`readouts.UNMODELLED`). |

Acceptance for this runbook: a dataset tree with `truth` + `traces` +
`provenance` per cell, a linear-subject validation, and one trained, frozen,
hooked transformer with its latent diagnostics.

---

## 1. Environment

```bash
# LUCiD clone (already in the tree; this is how to get it fresh)
git clone https://github.com/dowlingwong/LUCiD external/LUCiD
export LUCID_PATH="$PWD/external/LUCiD"

# JAX on CPU is enough to simulate; a GPU is picked up automatically
python -c "import jax; print('jax', jax.__version__, jax.devices())"

# package + tests
PYTHONPATH=src python -m pytest src/noise_module_lucid/tests -q     # 40 passed
```

---

## 2. Declare the readout (fixes "1 GHz PMT only")

The readout is a first-class object: SPE shape, sampling rate, front-end corner,
clock lines, and its **calibration state**.

```python
import noise_module_lucid as nml
ro = nml.PMT_1GHZ                       # the default placeholder
print(ro.calibrated, ro.reference)      # False, None

# a different sensor, once you have its SPE + gain + rms:
dom = nml.custom_readout(
    "icecube_dom", sampling_frequency_hz=3.0e8,   # 300 MS/s
    spe_tau_rise_ns=2.0, spe_tau_fall_ns=8.0,
    mv_per_pe=2.0, rms_mv=0.5, reference="IceCube DOM electronics paper")
nml.register(dom)
```

Gate: do not use an uncalibrated readout for a claim-bearing result; the
provenance carries `readout.calibrated` so this is auditable.

---

## 3. Re-measure the κ floor → choose window and crate size (fixes "covariance cells")

On a matched cell every κ(Σ̂⁻¹Σ) above 1 is estimator noise; it is set by N/C.
At the 512 ns default (N/C = 8) the floor is ~13.8 and the estimator cannot see
a real mismatch. Measure the floor and pick the window:

```python
sweep = nml.kappa_floor_sweep(Ns=(512, 4096, 32768, 65536), C=64, target=1.3)
for r in sweep["rows"]:
    print(f"N={r['N']:6d}  N/C={r['N_over_C']:6.1f}  κ floor={r['kappa_floor']:.2f}")
print("recommended:", sweep["recommended"])
```

Rule of thumb after the 24 Sep co-calibration: **N/C ≳ 1000** for a floor below
~1.3. Use `covariance_cells(...)` (below), which refuses `window_ns < 16 µs` and
switches to the long-window preset (DC-DC switching lines, extended 1/f).

---

## 4. Produce the dataset (fixes "no dataset driver/writer")

One cell → `truth.parquet` (or `.csv`), `traces.npy` `(n, C, N)`,
`provenance.json`, mirroring `herald_simulation.export`.

```bash
# intervention matrix (reference + covariance-type + structural N families),
# plus the long-window covariance cells
PYTHONPATH=src python -m noise_module_lucid \
    --out runs/lucid_pilot --geometry WCTE_like --n-photons 50000 --covariance
```

Or drive it in Python (this path is LUCiD-free and unit-tested; the LUCiD path
is `run_lucid_cell`):

```python
import numpy as np, noise_module_lucid as nml
ev = nml.EventSpec(7, "isotropic", {"position": [0.0, 0.0, 0.0], "intensity": 50_000})
cells = nml.intervention_matrix("WCTE_like", ev) + nml.covariance_cells("WCTE_like", ev)
summary = nml.build_dataset("runs/lucid_pilot", cells, n_photons=50_000)
```

**Provenance schema** (per cell): LUCiD commit, geometry name/hash and
placed-sensor count, `window_ns`/`bin_width_ns`, `readout` (with
`calibrated`/`reference`), preset `PROVENANCE`, `intervention`, `grouping`
(method + compactness), `implied_covariance`, `kappa_floor_mean`, and
`release` (`gate_A0_lucid_licence`, `releasable`).

**Grouping** (fixes "proxy grouping"): `CellSpec.group_method` is
`board_map` (physical, pass `board_ids`), `proximity` (geometry proxy),
`z_plane`, or `contiguous`. Until a board map exists, run `proximity` and report
`validation.grouping_sensitivity` so the proxy is disclosed, not hidden.

Gate: inspect one `provenance.json` and confirm `release.releasable == false`.

---

## 5. Step 0 — validate on the linear subject (no training)

Do **not** train before this. The four linear classes are exact and interpretable,
and they are the mechanism layer the monitor stack is validated against.

```bash
PYTHONPATH=src python -m pytest src/latent_monitor/tests/test_estimators.py -q
PYTHONPATH=src python -m latent_monitor.run_table --help      # the Tier-1 signature table driver
```

Fit a class (OF → CW-PCA → tied linear AE → NFPA) on the LUCiD waveform and read
the interpretable quantities: `P_resolved`/`P_weak` projectors, principal angles,
per-event Δz, and `rank(M_recon)` (`latent_monitor.reference`,
`latent_monitor.statistics`, `latent_monitor.task_metric`).

Gate: the designed controls (`latent_monitor.designed`, output-null /
output-aligned) behave as predicted, and the matched-cell κ floor matches
`kappa_floor_sweep`. Only then proceed.

---

## 6. Step 1 — train one small, frozen, hooked model

Subject: `reconstruction_model.models.current_compact` (two-stage transformer,
outputs `spatial_pred` (3) + `energy_pred` (1) → a declared `y`, hence `J_y` and
`M_task`). Wrapped by `latent_monitor.torch_subject.TorchSubject`, which exposes
`W` (Kronecker whitener), `S1` (temporal → `channel`), `P` (geometry → `token`),
`S2` (spatial → `z`), the `y` head, and autograd Jacobians.

Training is **CUDA-only** (`src/reconstruction_model/train.py` asserts a GPU;
use the L40S/A100 path in `scripts/`). Configure by environment:

```bash
RECONSTRUCTION_MODEL_VARIANT=current_compact \
RECONSTRUCTION_NUM_STEPS=... RECONSTRUCTION_DEVICE_BATCH_SIZE=... \
PYTHONPATH=src python -m reconstruction_model.train
```

Data for the subject: the LUCiD traces from §4, with the target = direction +
energy. **Two adaptations are required and are not yet wired:**
1. *Format* — `reconstruction_model/dataset.py` consumes HDF5/zst shards
   `energy_{E}_{ER|NR}.h5` shaped `(batch_events, n_channels, trace_samples)`,
   while the LUCiD driver writes `traces.npy (n, C, N)` + `truth.parquet`; write
   a small adapter (keep it under `runs/` or `scripts/`).
2. *Architecture* — `current_compact` was built for the DELight/TES traces
   (≈56 channels, 65536 samples); instantiate it with a `TransformerConfig`
   matching the LUCiD geometry (~2444 PMTs for `WCTE_like`) and train from
   scratch, keeping 0.1–1 M params.

Then **freeze** and wrap:

```python
from latent_monitor.torch_subject import TransformerSubject
subject = TransformerSubject.wrap(model, whitener, device="cuda")
```

Gate: the frozen subject beats a trivial baseline on direction+energy, and its
hooks are finite and stable across the reference cell.

---

## 7. Interpret the representation

```python
hooks = subject.hooks(X)     # whitened, channel, token, z, pre_output, output
```

Read, in order:
- **reference-cell projectors** (`latent_monitor.reference`) — resolved vs weak span;
- **per-event Δz** and **principal angles** (`latent_monitor.statistics`);
- **k-NN retention** and the **out-of-span fraction** — the conditional signatures;
- **`rank(M_recon)`** (`latent_monitor.task_metric.measurement_metric`) — resolvability;
- **`M_task` length** — the Claim-2 score;
- **ridge probes** from `z` to vertex/direction/energy — descriptive only.

Cross-check against the conditional-signature table in
`docs/EXPERIMENT_DESIGN.md` §III.1: covariance-type N should move noise-only
residual statistics without a consistent mean shift; structural N should move the
representation; in-span rare S should move `z` with noise-only statistics at
reference.

---

## 8. Still open (not fixed here)

- **Gate A0** — the LUCiD licence (email the maintainers); blocks release.
- **Readout calibration** — replace the placeholders with a measured SPE + PSD
  for whichever PMT you actually claim (`custom_readout(..., reference=...)`).
- **Physical board map** — supply `board_ids` to `channel_groups(method="board_map")`.
- **Event pairing + content-matched clean cells** — the LUCiD arm currently runs
  one isotropic event per cell; add track/cascade `source_factory` and the
  matched clean twin before an attribution run.
- **Transfer arm** — repeat the two claims once on a held-out physical arm
  (per `docs/EXPERIMENT_DESIGN.md` §I.3).

## File map

| what | where |
|---|---|
| dataset driver, cells, matrix, writer | `src/noise_module_lucid/dataset.py` |
| readout registry | `src/noise_module_lucid/readouts.py` |
| κ-floor sweep, bandwidth/CSD/grouping checks | `src/noise_module_lucid/validation.py` |
| front-end preset, modules, tutorial | `src/noise_module_lucid/` |
| subject + hooks | `src/latent_monitor/torch_subject.py`, `subject.py` |
| model + training | `src/reconstruction_model/` |
| design of record | `docs/EXPERIMENT_DESIGN.md` |
