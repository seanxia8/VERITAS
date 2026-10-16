# PMT experiment contract

16 October 2026. **DRAFT — UNFROZEN.** This document prepares WP6; it authorizes no training, remote execution or new transfer arm. Claims, labels and `GENERIC_COMMITTED` remain as in `PREREGISTRATION.md` §§1–8.

## 🔌 Interfaces and baseline

Input is `PhotonBatch` (`pmt-photons-v1`): incident photons with persistent IDs, sensor IDs, forward arrival times in ns, wavelengths in nm and declared upstream effects. Response parameters and their provenance are versioned in `configs/lucid_pmt/reference.json`. Wavelengths and truth are simulation/evaluation data. The deployed model/monitor sees measured sensor charge/time, geometry, grouping and frozen operational calibration only, through an allowlist. All twins and variants preserve physical event identity.

Candidate target: pinned LUCiD WCTE geometry. Actual PMT hardware model, physical group map, wavelength QE/CE and width calibrations remain pending. The current 76-channel cylinder is a synthetic flasher fixture, not WCTE or a Cherenkov event. A real exporter before QE/TTS and prompt Cherenkov closure are required before physics evaluation. Baseline includes nontrivial group gain/width and per-channel T-Q curves. Uniform angular response, prompt light and no concentrators are settled for the first trial.[^request]

Subject interface draft: per-event measured digit arrays plus sensor geometry → frozen reconstruction outputs and declared internal hooks. Padding/masking, architecture, output target, operational noise-reference source and trainability thresholds must be fixed before model fitting. There is no model trained in this pass. Train baseline response only with ≥3 seeds, target 5; no per-variant primary retraining.[^request]

## 🧩 Split and information design

| Protocol role | Use | Isolation |
|---|---|---|
| `reference_fit` (draft 30%) | Distinct subsets for subject training, response/energy/dead-map calibration and monitor reference statistics | No evaluation event; dedicated calibration may use its known source labels, never pass them to alarm features |
| `attribution_train` (20%) | Fit N/S attribution on allowed families | No held-out family/group/severity/seed |
| `development` (15%) | Precision planning, tune valid grids and fixed reconstructor | Independent from final evaluation; never retune to rescue a null result |
| `calibration` (15%) | Clean FAR/conformal/null calibration | No interventions or evaluation labels |
| `evaluation` (20%) | Frozen checkpoints, all paired cells and clean denominators | All twins/variants of an event stay together |

These are existing unfrozen smoke proportions, not a new protocol freeze. `configs/lucid_pmt/split_contract.json` declares roles; sample counts, family holdouts and final severities remain explicit pending values. The CPU engineering fixture has separate IDs 10000–10031 (response-fit), 20000–20015 (closure), 30000–30023 (development evaluation), and does not instantiate the training roles.

Persist a content hash and event-group ID for every real event. Response-fit events, operational noise records, subject training and final evaluation need independently audited membership. Do not infer disjointness by renaming duplicate physical events; current synthetic generator IDs define independent draws.

## 🧪 Cells and comparisons

Candidate N families are QE shape, fixed-threshold T-Q location, charge-dependent width and coupled group gain/width; isolated gain and width, timing offsets and module loss are supporting controls. Identity and exact QE×CE product controls must show identical measured response under shared draws. A separately charge-matched control may retain count/shape/spatial/timing differences and is not an exact degeneracy. Family, severity, group and perturbation-seed holdouts must be frozen before evaluation. The synthetic demonstration grid is not a harm-spanning scientific grid.

Candidate S families are valid event positions, directions and energies rare under baseline training, within the same detector and medium. Junjie/author must confirm support and event physics. Direction-based S is undefined for the current isotropic flasher. Water-to-WbLS, simulator replacement and other simulation-model changes are excluded from these cells; they are not silently relabelled as S or N.

Claim 1 remains paired ΔF1_attr against `generic_rich`, with hard matching, inclusive results, retained fractions and overlap failures. Claim 2 remains cell-level ΔAUROC_harm against the committed generic harm score. Preserve capacity matching, hook-drop, abstention and unknown controls. Use equal operational information for both sides. Charge sums/distributions, occupancy, group asymmetry, charge-conditioned time residuals/widths and operational frozen reconstruction are candidate strong **supporting** diagnostics; changing the primary baseline requires a recorded amendment. The current helper feature projection is not a completed generic baseline.

Model seeds are outermost; cells are Claim-2 resampling units; events are paired within cells. Report every seed/family including adverse or null results. Do not treat events pooled across trained replicas as independent replicas. Exact observable degeneracy implies no representation-level recovery of cause under this observation contract.

## 🚧 Decisions before evaluation

| Decision | Current preparation | Owner / required before |
|---|---|---|
| Target PMT/geometry and calibration sources | WCTE candidate, synthetic mechanisms only | Author/Junjie before physics interpretation |
| S support and track target | Candidate families above, point-source-only implementation | Author/Junjie before event generation for claims |
| D-K | Draft max requires both endpoints; any failed fit makes cell unevaluable; conditional complete-case statistic and both failure rates reported separately | Author before primary harm evaluation |
| κ_m | 1.10 remains `provisional_dev`; no thesis-derived threshold | Author before primary harm evaluation |
| Dead-map correction | Frozen reference-fit channel shares; uncorrected/corrected outputs both retained | Author chooses primary policy |
| Denominator | Same-event clean errors; zero denominator is undefined | Author adopts policy before primary evaluation |
| Counts/severity/held-out units | Development precision study not run | Author after bounded precision work, before held-out evaluation |
| Monitor feature amendment | Supporting diagnostics only | Author before changing primary comparison |
| Compute budget | This pass CPU only; maximum local fixture 32 fit +16 closure +24 evaluation ×8 cells, no model training | Host, memory, wall time, storage and seed budget pending before WP7 |

## 🧭 WP8 parent-issue decisions

| Decision | Preparation and rationale | Status |
|---|---|---|
| Electronics claim cells | Frozen; detector-response mechanisms are this pass's focus | Implemented exclusion |
| D-M | Exclude simulation-model changes; no new origin label | Draft experiment scope, protocol label decision still open |
| D-L | LUCiD attribution control/harm replication versus paper exclusion | Author decision pending; not selected by favorable outcomes |
| D-T | NuRadioMC: planned frozen NuRadioReco direction/energy endpoint, pilot and threshold source absent. JAXTPC: candidate replacement, endpoint/calibration/cost not established. Both availability/resource audits remain unrun | No arm selected or implemented |
| D-S/TIDMAD | Clean rare signal family, injection-before-recovery consequence and disjoint operational reference files required; preserve frozen 20-file evaluation contract | No real-data execution in this pass |
| Future scope | Angular acceptance next iteration; concentrators/scattering/reflection deferred. LIGO future; HeST/DELight conditional on existing Phase-B decision | No automatic arm expansion |

[^request]: Junjie Xia, current correction and acceptance: https://github.com/seanxia8/VERITAS/issues/5#issuecomment-6134579713
