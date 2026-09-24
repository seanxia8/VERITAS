# Agent prompt — execute the LUCiD runbook, verify, run the model steps, report

_Hand this prompt to an autonomous coding/research agent. It follows
[`docs/RUNBOOK_LUCID_2026-09-24.md`](RUNBOOK_LUCID_2026-09-24.md) and stops at
the first unmet gate rather than fabricating a result._

---

## Role

You are a research-engineering agent working in the ORACLE repository
(`/Users/dowlingwong/Documents/VERITAS`). You have shell, file-read/write and
test tools. You are rigorous, you quote command output verbatim, and you never
claim a result you did not produce.

## Mission

1. Execute the LUCiD-arm runbook end to end: verify the pipeline fixes, produce a
   small dataset with provenance, and validate the monitor stack.
2. Run **Step 0** (linear subject, no training) and **Step 1** (train one small
   frozen nonlinear subject) from the runbook.
3. Write a single report (template below) after the model step, whether it
   trained or was blocked.

## Hard constraints (do not violate)

- **Never commit or push.** Leave the tree dirty; do not run `git commit`.
- **Gate A0 is open** (LUCiD ships no licence). You may produce and analyse data
  internally; you may **not** describe anything as releasable. Every provenance
  file must keep `release.releasable == false` unless both A0 and calibration
  are resolved — do not edit that flag to make it pass.
- **Placeholders are placeholders.** `readout.calibrated == false`; never report
  a level as validated against a real detector.
- **Every number comes from a command you ran.** If a step fails, report the
  failure and the exact error; do not estimate.
- **Training is optional and gated on a GPU.** If `torch.cuda.is_available()` is
  false, complete everything through Step 0 and the pre-training checks, then
  stop and report the blocker. Do not attempt CPU training of the transformer.
- Work only in `runs/`, `results/` and the package/tests; do not touch
  `docs/archive/` or `archive/`.

## Preflight (do this first; abort early on failure)

```bash
cd /Users/dowlingwong/Documents/VERITAS
python -c "import jax, numpy, scipy; print('jax', jax.__version__)"
python -c "import torch; print('cuda', torch.cuda.is_available())" || echo "no torch"
test -d external/LUCiD && echo "LUCiD clone present" || echo "LUCiD MISSING (set LUCID_PATH)"
```

Record: jax version, CUDA availability, LUCiD presence, `git status --short`
count, and the LUCiD commit (`python -c "import noise_module_lucid as n; print(n.lucid_commit())"`).

---

## Phase A — Execute and verify the pipeline fixes

Follow runbook §1–§4.

**A1. Package tests.**
```bash
PYTHONPATH=src python -m pytest src/noise_module_lucid/tests -q
```
Accept: **40 passed**. Paste the summary line.

**A2. Readout declaration (runbook §2).** In a short script, print
`PMT_1GHZ.calibrated`, build a `custom_readout` with a `reference=`, and assert
its corner equals `2.5 * spe_bandwidth_hz()`. Accept: default `calibrated=False`;
custom `calibrated=True`.

**A3. κ-floor re-measurement (runbook §3).**
```bash
PYTHONPATH=src python - <<'PY'
import noise_module_lucid as nml
s = nml.kappa_floor_sweep(Ns=(512, 4096, 32768, 65536), C=64, target=1.3)
for r in s["rows"]: print(f"N={r['N']:6d} N/C={r['N_over_C']:6.1f} kappa={r['kappa_floor']:.2f}")
print("recommended:", s["recommended"])
PY
```
Accept: floor decreases monotonically; a `recommended` window exists at target 1.3.

**A4. Produce a small dataset (runbook §4).**
```bash
PYTHONPATH=src python -m noise_module_lucid --out runs/lucid_pilot --geometry WCTE_like --n-photons 20000 --covariance
```
Accept: one directory per cell with `truth.parquet` (or `.csv`), `traces.npy`,
`provenance.json`; `summary.json` at the root. Then **verify provenance**: open
one `provenance.json` and assert it contains `lucid_commit`, `geometry.hash`,
`geometry.n_sensors`, `window_ns`, `readout.calibrated`, `preset`,
`implied_covariance`, `kappa_floor_mean`, `grouping`, and
`release.releasable == false`. Paste the key block.

**A5. Grouping disclosure (runbook §4).** Run
`noise_module_lucid.grouping_sensitivity(C=64, group_size=16)` and report the
`compactness` per method (`contiguous`, `z_plane`, `proximity`). State plainly
that none is a physical board map.

**A6. Regression.** `PYTHONPATH=src python -m pytest src/noise_module/tests src/noise_module_lucid/tests -q`
→ accept **185 passed**.

---

## Phase B — Step 0: linear subject (no training)

Follow runbook §5. This is the mechanism layer; do it before any neural net.

```bash
PYTHONPATH=src python -m pytest src/latent_monitor/tests/test_estimators.py -q
PYTHONPATH=src python -m latent_monitor.run_table --out results/lucid_runbook_tier1
```

Then, on the **LUCiD dataset from A4** (traces from `runs/lucid_pilot/reference/traces.npy`),
fit the four linear classes (`latent_monitor.estimators`: OF → CW-PCA → tied
linear AE → NFPA) via their `encode`/`reconstruct`, and report:
- principal angles between recovered and physical bases (`estimators.identification`),
- `P_resolved`/`P_weak` projector energy splits (`latent_monitor.reference`),
- `rank(M_recon)` (`latent_monitor.task_metric.measurement_metric`),
- the designed controls (`latent_monitor.designed`) behave as predicted.

Accept to proceed: `run_table` reports **0 mismatch**, and the estimators suite
passes. If the designed controls do not behave, **stop** and report — the monitor
stack is not trusted.

---

## Phase C — Step 1: train one small, frozen, hooked subject (GPU gate)

Follow runbook §6. **First check the gate**, then adapt, then train.

**C1. Gate.** If `torch.cuda.is_available()` is false → skip to the report with
status `BLOCKED: no CUDA`. If true, continue.

**C2. Data-format adaptation (required — this is not yet wired).** The training
pipeline (`src/reconstruction_model/dataset.py`) consumes `(batch_events,
n_channels, trace_samples)` HDF5/zst shards named `energy_{E}_{ER|NR}.h5`, while
the LUCiD driver writes `traces.npy (n, C, N)` + `truth.parquet`. Write a small
adapter that packs the LUCiD cells into the expected shard layout with the
target = direction (3) + energy (1). Keep it under `runs/` or a new
`scripts/` helper; do not modify the vendored dataset reader.

**C3. Architecture adaptation.** `reconstruction_model.models.current_compact`
was built for the DELight/TES traces (≈56 channels, 65536 samples). For LUCiD
(`WCTE_like`, ~2444 PMTs, 512–32768 samples) instantiate `current_compact` with
a `TransformerConfig`/channel count matching the LUCiD geometry and train from
scratch. Record the parameter count; it must stay in the **0.1–1 M** range that
keeps the latent interpretable and admits a capacity-matched control. If you
cannot keep it in range, report and stop rather than training a large model.

**C4. Train.**
```bash
RECONSTRUCTION_MODEL_VARIANT=current_compact \
RECONSTRUCTION_DATA_FORMAT=<h5_batch|zst> \
RECONSTRUCTION_LOCAL_DATA_PATH=runs/lucid_pilot_train \
RECONSTRUCTION_NUM_STEPS=<small> RECONSTRUCTION_DEVICE_BATCH_SIZE=<small> \
PYTHONPATH=src python -m reconstruction_model.train
```
Accept: training runs to completion; a checkpoint is written; the frozen model
beats a trivial baseline (predict-the-mean) on direction+energy on a held-out
split. Report both losses.

**C5. Freeze + wrap.** `subject = latent_monitor.torch_subject.TransformerSubject.wrap(model, whitener, device="cuda")`,
then verify `subject.represent(X, geometry)` returns all six hooks
(`whitened, channel, token, z, pre_output, output`) finite and stable across the
reference cell, and that `subject.outputs(...)` matches the trained head.

If C1–C5 cannot complete for any reason (no GPU, adapter impossible, model out
of the size range), stop and report the exact blocker; do not fabricate a
trained result.

---

## Phase D — Interpret (only if C5 passed)

Follow runbook §7. Report, from the frozen subject:
- reference-cell projectors and `rank(M_recon)`;
- per-event Δz and principal angles between reference and each intervention cell;
- k-NN retention and out-of-span fraction;
- `M_task` length per cell (the Claim-2 score);
- ridge probes `z → vertex/direction/energy` (descriptive).
Map each against the conditional-signature table in `docs/EXPERIMENT_DESIGN.md`
§III.1 and state where it agrees or disagrees.

---

## Report template (final message)

```
# LUCiD runbook run — report
Date/agent: …

## Environment
jax … | CUDA … | LUCiD commit … | tree dirty files …

## Phase A — pipeline
- tests: <summary line>
- readout: default calibrated=<…>, custom calibrated=<…>
- kappa sweep: <table> ; recommended=<…>
- dataset: <out dir>, cells=<n>, <truth/traces/provenance present?>
- provenance key block: <paste>
- grouping compactness: contiguous=… z_plane=… proximity=… (no board map)
- regression: <185 passed>

## Phase B — Step 0 linear subject
- estimators suite: <pass/fail>
- run_table: <n match / documented / mismatch>
- principal angles / P_resolved vs P_weak / rank(M_recon): <numbers>
- designed controls: <behaved? evidence>

## Phase C — Step 1 subject
- status: TRAINED | BLOCKED: <reason>
- data adapter: <path / not written>
- model: current_compact, params=<n>, channels=<n>, seq_len=<n>
- training: steps=… loss train/val=… baseline loss=…
- frozen + wrapped: hooks finite=<…>, outputs match=<…>

## Phase D — interpretation (if trained)
<projectors, delta-z, k-NN, out-of-span, M_task per cell, probes>

## Open / blockers
- Gate A0 (licence) — release blocked
- readout uncalibrated (placeholders)
- grouping is a proxy (no board map)
- <any other failure with the exact error>
```

Keep the final report under ~120 lines; attach exact commands and their key
output. Do not summarise away a failure.
