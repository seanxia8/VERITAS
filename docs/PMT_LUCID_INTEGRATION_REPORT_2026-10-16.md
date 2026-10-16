# LUCiD integration and prompt Cherenkov development closeout

16 October 2026. **Engineering pilot complete; scientific issue remains partially complete.**

## Scope and provenance

The user explicitly authorized real LUCiD integration, prompt Cherenkov validation and baseline GPU training on Deepthought. Work stayed on `codex/junjie-pmt-response`, following PMT commit `ac83c79e333fe990f2a2665876680b7cbb2a0916`. Main checkout, JEPA, manuscript and historical PMT results were not edited. No push, merge, collaborator contact or transfer arm occurred.

Upstream LUCiD is pinned at `eaf2e89f579d6679e90020c732ac2c846895d691`. `configs/lucid_pmt/upstream_manifest.json` verifies Python/configuration files before transport. Actual WCTE geometry has 1,995 sensors. `lucid_transport.PromptTransport` invokes its geometry propagator and exports first forward hard sensor encounters **before QE, CE or TTS**, preserving photon IDs, wavelengths and emission times. It does not convert soft SIREN weights into photons. Absorption, dispersion, scattering, reflection and concentrators are disabled; angular acceptance is uniform. The sensor model remains the upstream spherical approximation.

The source is the public PhotonSim water-muon 500 MeV example linked by the pinned upstream `scripts/download_data.sh`. Byte-range reads retained events 0–23: 2,039,166 emitted photons. Including the inspection and source audit, 131,522,269 bytes were read, below the 134,217,728-byte cap. Per-event arrays are hashed. ROOT EventID, total photon count and summed segment Cherenkov count agree for all 24 events. Positions are converted explicitly from mm to metres. This avoids the incompatible `/100` conversion in the older simulator data path.

Full source: [pinned upstream downloader](https://github.com/CIDeR-ML/LUCiD/blob/eaf2e89f579d6679e90020c732ac2c846895d691/scripts/download_data.sh). Local receipts: [source audit](../results/pmt_lucid_integration_dev_2026-10-16/source_audit.json), [source manifest](../results/pmt_lucid_integration_dev_2026-10-16/manifest.json), [runtime/code receipt](../results/pmt_lucid_integration_dev_2026-10-16/runtime_receipt.json).

## Prompt closure and retained failure

The first run failed its predeclared closure gates and did not train. One of 6,144 sampled rays hit a sensor absent from the default four-candidate lookup. Near-unit source directions and float32 calculations also failed the strict timing tolerance. `diagnosis_01.log` identifies the omitted sensor (1308) and actual candidates. The first report and logs remain in `closure_01`.

The corrected adapter normalizes directions, enables float64 and requests 16 candidates through the public upstream API. It uses fixed-size photon chunks to avoid repeated JIT compilation. No acceptance tolerance was loosened and no upstream source was modified.

The second run passed:

- All 6,144 sampled rays agree with independent float64 all-sensor ray–sphere intersections.
- Maximum sensor-surface residual: 2.10e-10 m; maximum flight-time residual: 4.91e-9 ns.
- The local-emission cone diagnostic uses each photon's emitting segment direction and beta, with the pinned medium's constant phase-index approximation. Across 1,576,360 photons in 300–600 nm, median absolute angle residual is 0.45936° and the 95th percentile is 2.35920° (limits 5° and 10°).
- Independent paired response checks pass for all 24 physical events: identity, exact QE×CE product, photon ordering, unchanged charge/membership under pure fixed-threshold T-Q, and the expected +1 ns timing shift.

[Passing closure](../results/pmt_lucid_integration_dev_2026-10-16/closure_02/closure.json), [failed closure](../results/pmt_lucid_integration_dev_2026-10-16/closure_01/closure.json), [paired controls](../results/pmt_lucid_integration_dev_2026-10-16/controls_01/controls.json).

This establishes bounded direct-light integration and a supporting local cone check. It does not validate arbitrary grazing rays, dispersive transport, reconstructed cone-angle inference, track-start/vertex reconstruction, or an energy-loss history.

## Baseline GPU pilot

The configuration was written before training: `configs/lucid_pmt/integration_pilot.json`. Train/validation/test physical IDs are 0–15 / 16–19 / 20–23. Each has 16 deterministic rotations; all rotations stay in their physical-event split. There are 384 samples but only 24 independent source showers and four independent test showers. All photons remain inside the geometry after the declared centring/rotation; spectrum filtering is recorded per sample.

The measured-digit PointNet has 100,483 parameters and masked mean/max pooling. It uses measured charge, time relative to the first measured digit, sensor coordinates and occupancy only. Truth direction is a target, never an input; wavelengths, response labels and clean twins are excluded. The waveform transformer was not silently applied to digit features. This pilot does not select or replace the primary scientific model.

Only the reference response is trained, using three seeds (10, 11, 12), 60 epochs each and validation-MSE checkpoint selection. The JSON's historical key `source_seeds` denotes these **model initialization seeds**; source rotations use the separately coded deterministic seed schedule. Each held-out test is evaluated once per model. The PMT QE/CE, T-Q, width and group factors are synthetic development values, not measured WCTE calibration; sequential groups of 19 channels are not claimed to be hardware crates.

| Model seed | Test direction MSE | Mean angular error | Selected epoch (zero-based) | Gate |
|---|---:|---:|---:|---|
| 10 | 0.00083475 | 2.5742° | 54 | PASS |
| 11 | 0.00083056 | 2.6599° | 52 | PASS |
| 12 | 0.00067841 | 2.2894° | 57 | PASS |

Training-mean constant-vector test MSE is 0.335968. Every seed passes the predeclared gates (MSE < 80% of this baseline; mean angle < 60°). GPU: GTX TITAN X, PyTorch 2.5.1+cu121; total training 40.51 seconds, peak allocated tensor memory 236,953,600 bytes. One GPU was used with a 25% process memory limit, without changing other users' processes. Test metrics were independently recomputed from saved predictions after receipt; runtime code/config hashes match the local sources.

[Training report](../results/pmt_lucid_integration_dev_2026-10-16/training_01/training.json), [receiving checks](../results/pmt_lucid_integration_dev_2026-10-16/receiving_checks.json). Weights and generated digit arrays remain on Deepthought under `/work/dwong/pmt_lucid_20261016/evidence/training_01/` and `closure_02/digits.npz`, with hashes in the receipts. This small single-energy sample establishes trainability only.

## Validation and remaining gates

Local PMT suite: 94 passed, one optional PyTorch test skipped. The remote PyTorch padding/permutation tests both pass. The full remote PMT suite passed: 95 tests, no skips (`remote_pmt_tests.log`). Earlier unchanged shared-noise/latent suites are recorded in the preceding closeout; no shared packages changed here.

Still required for full scientific closure:

1. Target PMT calibration sources and a verified physical group/crate map.
2. Multi-energy and wider physically supported event data, with agreed track/energy targets and independent reconstruction/calibration closure. The current single-energy pilot does not validate energy reconstruction.
3. Frozen S support, held-out families/severities, sample counts, D-K/threshold/dead-map policy and primary model/monitor interfaces.
4. Same-information attribution/harm comparisons, uncertainty across independent units, adverse/null-result reporting and collaborator review.
5. Parent-issue D-T/D-L/D-M/D-S decisions; no new transfer arm has been started.

Every new result is development/non-citable. No scientific superiority, target-detector calibration or issue closure is claimed.

## Execution record

Remote root: `/work/dwong/pmt_lucid_20261016`; interpreter `/home/dwong/anaconda3/bin/python3`. The exact extraction/probe/audit/diagnosis scripts are retained in `execution_scripts/`. The packaged extraction CLI adds streamed-response bounds and performs the same source identity/count checks inline; the executed extraction and subsequent audit are preserved separately.

```sh
# From remote root; pinned source is in upstream/, task source in code/.
PYTHONPATH=code/src:upstream JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  timeout 1800 /home/dwong/anaconda3/bin/python3 -m noise_module_lucid.lucid_integration \
  --upstream upstream --source source_subset \
  --config code/configs/lucid_pmt/integration_pilot.json \
  --manifest code/configs/lucid_pmt/upstream_manifest.json --out evidence/closure_02
PYTHONPATH=code/src:upstream JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  timeout 300 /home/dwong/anaconda3/bin/python3 -m noise_module_lucid.lucid_controls \
  --upstream upstream --source source_subset \
  --manifest code/configs/lucid_pmt/upstream_manifest.json --out evidence/controls_01
# Executed only after source audit, closure and paired controls passed.
PYTHONPATH=code/src CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 1800 \
  /home/dwong/anaconda3/bin/python3 -m noise_module_lucid.lucid_baseline \
  --data evidence/closure_02/digits.npz --config code/configs/lucid_pmt/integration_pilot.json \
  --out evidence/training_01
```

Use new output directories for any reproduction; completed evidence must not be overwritten. An approval-review usage-limit interruption occurred during a read-only audit dispatch; the user asked to continue, review became available, and the audit then completed. Existing bounded jobs continued and were inspected before further execution.
