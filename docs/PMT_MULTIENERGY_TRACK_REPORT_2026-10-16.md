# Multi-energy prompt-track development validation

16 October 2026. **PARTIAL PASS: transport, direction and vertex development gates pass; initial-energy trainability fails in every seed.** Development/non-citable evidence; primary scientific claims are not evaluated. This follows the [single-energy integration pilot](PMT_LUCID_INTEGRATION_REPORT_2026-10-16.md). The [author-decision list](PMT_CALIBRATION_AND_PROTOCOL_DECISIONS_2026-10-16.md) separates remaining calibration inputs and primary protocol choices.

## Scope and source provenance

The public PhotonSim water/muon example files supply 20 physical events each at 500, 1,000 and 1,500 MeV. Each receives eight deterministic rotations and primary-start placements, producing 480 digit samples from 60 physical events. The primary starting position, initial direction and recorded initial kinetic energy are development targets. No shower-centering is used. Primary track segments, photon emission positions/times/wavelengths and photon IDs are retained. All 500 MeV source events were available to the preceding development work; this is not an untouched confirmation set.

The source reader validates chunk contiguity, event identity, photon count and segment count, and explicitly converts positions from millimetres to metres. The [energy-label audit](../results/pmt_multienergy_dev_2026-10-16/energy_source_semantics.md) distinguishes initial kinetic energy from deposited energy. The public files' exact generator revision is unknown; the audited PhotonSim writer revision is not asserted to be their generation revision.

The original plan requested 24 events per energy. Two extraction implementations were interrupted to repair repeated ROOT-basket downloads (contiguous chunk reads, then exact-range caching). Their logs and source snapshots are retained. Resume accounting charged an additional 64 MiB per interruption for in-flight work. The final extractor stopped at its aggregate source-read cap, after completing 24/24/20 events. The retained accounting at the last complete event is 931,869,635 bytes against a 939,524,096-byte cap; it is conservative application accounting, not a measured network transfer total. No source request was made to fill the remaining four events. Before digit reconstruction or model training, the cohort was amended to the first 20 events at every energy; the [original configuration](../results/pmt_multienergy_dev_2026-10-16/planned_24_event_config.json), STOP and final configuration are retained. Model, metric and acceptance thresholds were unchanged.

## Transport, geometry and physical groups

The same hash-verified LUCiD revision `eaf2e89f579d6679e90020c732ac2c846895d691` is used through its public prompt propagator, with normalized rays and float64. The initial 16-candidate run failed one sensor assignment among 15,360 checked rays and blocked training. That failed dataset is retained in `track_closure_01`; timing and surface residuals passed. The isolated diagnosis found that sensor 1672 was absent from the candidate list, which returned sensor 1689 instead. Increasing the public API candidate limit to 64 repaired all 256 checks for that event; the full cohort was regenerated under that setting. Candidate-search diagnosis and repair are recorded separately, with all scientific acceptance thresholds unchanged. A duplicate provenance keyword introduced while making the limit configurable was caught by regression tests and fixed before the repaired generation; those failed test/diagnosis logs are also retained. The independent all-sensor intersection check covers 256 rays per physical event (15,360 rays in total), at the first placement. Surface and flight-time residuals cover all recorded hits at those placements. These finite checks do not prove completeness for arbitrary geometry or rays.

The upstream NPZ contains physical mPMT IDs. Its loader reorders those IDs consistently with sensor positions; this pilot uses that mapping for synthetic group gain/width factors. A physical module map is therefore available. An electronics-crate map and magnetic-field/response calibration remain unavailable. The upstream loader snaps sensor spheres to the cylinder boundary. The map receipt identifies 105 physical mPMT groups and a maximum raw-to-model displacement of 0.05766785 m.

Photon emission points outside the cylinder or inside a sensor sphere are excluded and counted; only the declared 200–800 nm band is transported. Track containment is computed from the retained primary-segment endpoints after transformation. Only 3/160 placements at 500 MeV are contained; none of the 160 placements at either higher energy are contained. Two contained placements belong to response-fit and one to test; none belong to model training or validation. The contained test subgroup therefore has only one placement and cannot support a contained-track performance claim. This is predominantly an escaping-track pilot. This is post-generation cropping of water-simulation emissions, not a full WCTE geometry/material simulation. It does not reproduce boundary-dependent secondary production or a measured energy-loss history. Constant-index prompt propagation omits absorption, dispersion, scattering and reflection. Here prompt means direct optical transport; source Cherenkov emission times and emitting secondary segments are retained, with no additional primary-only or emission-time cut. It is not a measured prompt-time selection. Synthetic QE/CE, charge, T-Q and Gaussian timing width are applied once by the owned readout. The local cone diagnostic uses segment direction/beta and constant phase index, and is supporting evidence rather than a reconstructed-track cone measurement.

## Frozen baseline protocol

Within every energy, source IDs 0–3 are response-fit, 4–11 train, 12–15 validation and 16–19 test. All rotations stay in the same physical-event split: 12/24/12/12 physical events and 96/192/96/96 digit samples. The charge-to-energy least-squares fit uses only response-fit events and is a supporting benchmark, not measured PMT calibration.

The measured-digit PointNet uses charge, time relative to the first measured digit, sensor coordinates and occupancy. Truth vertex/direction/energy are targets only. Seeds 20/21/22 each train for 100 epochs, with a validation-selected checkpoint and one test evaluation. The equal-weight loss combines direction MSE, vertex MSE scaled by 0.35 m, and log-energy MSE. Predeclared gates require mean direction error below 20 degrees, vertex RMSE below 0.35 m, relative energy MAE below 30%, and improvement over a training-mean constant on each endpoint's MSE. A failed endpoint remains a failure; no threshold or checkpoint retuning is permitted after test inspection.

## Results and independent receiving checks

The repaired transport run passes all 15,360 sampled sensor assignments. Maximum flight-time residual is 4.81510e-9 ns and maximum sensor-surface residual is 2.95605e-10 m, both below the unchanged 1e-4 tolerances. Generation took 657.90 seconds (initial failed run: 491.15 seconds). The per-event median absolute local cone residual ranges from 0.42559 to 0.47196 degrees; the largest per-event 95th-percentile residual is 4.75518 degrees. These are per-event diagnostics, not pooled quantiles or reconstructed track angles. Source emission maxima are 8.62/17.56/25.07 ns at 500/1,000/1,500 MeV.

| Model seed | Mean direction error | Vertex RMSE | Relative initial-energy MAE | Selected epoch (zero-based) | Overall trainability |
|---|---:|---:|---:|---:|---|
| 20 | 7.4121 degrees | 0.29274 m | 41.756% | 49 | FAIL: energy absolute gate |
| 21 | 7.1888 degrees | 0.26849 m | 39.033% | 60 | FAIL: energy absolute gate |
| 22 | 8.4986 degrees | 0.32590 m | 41.200% | 48 | FAIL: energy absolute gate |

Every seed improves direction, vertex and log-energy MSE over the training-mean constant predictor and passes the pooled direction/vertex absolute thresholds. Every seed fails the predeclared energy MAE <30% criterion. The combined run therefore terminates with a retained trainability failure after writing all three seeds' results. No extra epochs, seeds, checkpoint selection or threshold changes were used to rescue it.

The constant predictor has vertex RMSE 0.36972 m, direction MSE 0.33555 and log-energy MSE 0.205756. The separate response-fit charge benchmark has 42.225% relative energy MAE; its frozen fit is `E_MeV = 0.171792817 * Q + 33.262014`, with Q in the synthetic measured-charge scale. It has no negative test predictions, but is not a target-detector calibration.

| Test energy | Relative energy MAE across seeds | Vertex RMSE across seeds |
|---|---:|---:|
| 500 MeV | 68.55–77.65% | 0.25483–0.28440 m |
| 1,000 MeV | 22.13–27.10% | 0.25686–0.32133 m |
| 1,500 MeV | 23.82–25.12% | 0.29213–0.36673 m |

The largest energy error is at 500 MeV, and seed 22's 1,500 MeV vertex RMSE is 0.36673 m. The predeclared absolute vertex gate is pooled, so its pass does not imply that every energy subgroup meets 0.35 m. Each test energy has only four physical events and 32 correlated placements; the whole test has 12 physical events and 96 placements. Three initialization seeds do not multiply the number of independent physical events. The single contained test placement cannot establish contained-track accuracy. These results neither validate initial-energy reconstruction on escaping tracks nor prove that a different justified model could not do better.

Each model has 102,023 parameters. Total three-seed GPU training took 48.98 seconds on a GTX TITAN X, with peak allocated tensor memory 236,972,032 bytes under the 25% process cap. All predictions are finite. Independent NumPy recomputation agrees with all reported aggregate direction, vertex and energy metrics. Runtime code/config, dataset and checkpoint hashes match the receipts. The first hash check encountered 45 macOS AppleDouble sidecars copied alongside the package; their binary signatures and hashes were verified separately in `metadata_receipt.json`, and every actual Python source matches the local worktree. No sidecar was treated as executable source.

The source audit verifies all 60 event hashes, finite arrays, contiguous photon IDs, nonempty primary tracks and agreement of nominal energy with `PrimaryEnergy` and primary `TrackInfo_Energy`. Physical-event split intersections are empty. The repaired and failed datasets have identical targets, split membership and containment. Digit arrays differ in 186/480 placements after the candidate repair; this count is not a count of missed rays. Final tests: 96 local passes with two PyTorch-dependent skips, and 98 remote passes without skips. `git diff --check` passes.

Evidence: [passing transport](../results/pmt_multienergy_dev_2026-10-16/track_closure_02/validation.json), [candidate diagnosis](../results/pmt_multienergy_dev_2026-10-16/diagnosis_02.json), [all seed results](../results/pmt_multienergy_dev_2026-10-16/training_01/training.json), [runtime receipt](../results/pmt_multienergy_dev_2026-10-16/runtime_receipt.json), [independent receiving checks](../results/pmt_multienergy_dev_2026-10-16/receiving_checks.json).

The next scientific decision is the energy endpoint and containment policy. A contained initial-energy study needs different event support/geometry; a visible/deposited-energy study needs an explicit new truth definition and corresponding source records. Retaining initial energy for escaping tracks would require a justified inference study and a new, prespecified development protocol. This adverse pilot does not authorize switching the primary endpoint or loosening its acceptance criterion.

## Execution and evidence

Remote root: `/work/dwong/pmt_multienergy_20261016`. Source subsets and digit arrays remain there. The pinned upstream dependency remains under `/work/dwong/pmt_lucid_20261016/upstream`. The repaired config is `multienergy_pilot_repaired.json`, matching the repository's `configs/lucid_pmt/multienergy_pilot.json`. Generation and total three-seed training each have a 1,800-second wall cap. GPU training is limited to one GTX TITAN X, four CPU threads and 25% process GPU memory; other users' processes are untouched.

The result directory retains manifests, stopped-attempt logs/snapshots, validation reports, physical map, calibration coefficients, test predictions, histories, hash receipts and independent receiving checks. Large source arrays, digit arrays and model weights stay on Deepthought and are identified by hashes. Use new output directories for reproductions. No primary harm comparison, architecture replacement, D-K policy, harm-threshold adoption, publication or issue closure is implied.

## Constant-predictor comparison (runbook v2 R-3, added 18 Oct 2026)

This section adds a comparison only; it does not change any verdict above. The training-mean constant was not saved by the run; it is reconstructed from the committed config, source manifest and deterministic placement seeds, and reproduces every recorded `constant_metrics` value of `training.json` exactly. Model metrics are recomputed from the saved test predictions and agree with the recorded seed and per-energy metrics. The constant is the pooled training-split mean of the target vector, applied unchanged in every energy subgroup. Each subgroup has 4 physical events (32 correlated placements). The energy column is the recorded initial-energy target; the D-E deposited-energy endpoint was not trained. Script: [`constant_baseline_table.py`](../results/pmt_multienergy_dev_2026-10-16/constant_baseline_table.py); outputs: [JSON](../results/pmt_multienergy_dev_2026-10-16/constant_baseline_comparison.json), [Markdown](../results/pmt_multienergy_dev_2026-10-16/constant_baseline_comparison.md).

| Seed | Subgroup | n | Vertex RMSE model / constant (m) | Ratio | Direction mean model / constant (deg) | Ratio | Initial-energy relative MAE model / constant | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | pooled | 96 | 0.2927 / 0.3697 | 0.7918 | 7.4121 / 92.8001 | 0.0799 | 0.4176 / 0.4343 | 0.9615 |
| 20 | 500 MeV | 32 | 0.2575 / 0.3612 | 0.7127 | 7.2727 / 88.8554 | 0.0818 | 0.7304 / 0.8171 | 0.8939 |
| 20 | 1000 MeV | 32 | 0.2848 / 0.3513 | 0.8106 | 6.7752 / 95.6979 | 0.0708 | 0.2710 / 0.0914 | 2.9639 |
| 20 | 1500 MeV | 32 | 0.3312 / 0.3952 | 0.8381 | 8.1884 / 93.8470 | 0.0873 | 0.2512 / 0.3943 | 0.6372 |
| 21 | pooled | 96 | 0.2685 / 0.3697 | 0.7262 | 7.1888 / 92.8001 | 0.0775 | 0.3903 / 0.4343 | 0.8988 |
| 21 | 500 MeV | 32 | 0.2548 / 0.3612 | 0.7054 | 6.9042 / 88.8554 | 0.0777 | 0.6855 / 0.8171 | 0.8389 |
| 21 | 1000 MeV | 32 | 0.2569 / 0.3513 | 0.7312 | 6.4995 / 95.6979 | 0.0679 | 0.2470 / 0.0914 | 2.7008 |
| 21 | 1500 MeV | 32 | 0.2921 / 0.3952 | 0.7392 | 8.1628 / 93.8470 | 0.0870 | 0.2386 / 0.3943 | 0.6050 |
| 22 | pooled | 96 | 0.3259 / 0.3697 | 0.8815 | 8.4986 / 92.8001 | 0.0916 | 0.4120 / 0.4343 | 0.9487 |
| 22 | 500 MeV | 32 | 0.2844 / 0.3612 | 0.7873 | 7.5965 / 88.8554 | 0.0855 | 0.7765 / 0.8171 | 0.9503 |
| 22 | 1000 MeV | 32 | 0.3213 / 0.3513 | 0.9148 | 9.0678 / 95.6979 | 0.0948 | 0.2213 / 0.0914 | 2.4198 |
| 22 | 1500 MeV | 32 | 0.3667 / 0.3952 | 0.9280 | 8.8315 / 93.8470 | 0.0941 | 0.2382 / 0.3943 | 0.6042 |

Charge-calibration benchmark (response-fit least squares, recorded in training.json): pooled relative energy MAE 0.4223; 500 MeV 0.8287, 1000 MeV 0.1622, 1500 MeV 0.2759; negative predictions 0.

Energy endpoint: initial kinetic energy (recorded target, log(E/1000 MeV)); the D-E deposited-energy endpoint was not trained. The constant is the pooled training-split mean, evaluated unchanged in every subgroup; ratio = model / constant.

Recorded verdict (unchanged): energy_absolute gate failed for all seeds; R-3 gates apply to future runs only.
