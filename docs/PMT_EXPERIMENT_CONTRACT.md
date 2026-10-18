# PMT experiment contract

## WP6 freeze v1 (18 October 2026)

**FROZEN v1.** The binding contract is `configs/lucid_pmt/wp6_contract_v1.json`, checked by `src/noise_module_lucid/tests/test_wp6_contract.py`. It was frozen under the author's instruction of 18 Oct 2026 to freeze WP6 with every decision adopted that day (`docs/PMT_DECISIONS_LOG.md`). Changes need a recorded, dated amendment before any held-out evaluation. The draft text below (16 Oct) is kept as history; where it conflicts, the JSON wins.

**What is frozen**
- **Adopted decisions:** D-Q, D-J, D-G and D-K, and the form of D-TQ (its values stay `synthetic_dev`).
- **Reference response:** `reference.json` v2, with mPMT groups and 1 kHz synthetic dark counts.
- **Cells:**
  - N cells: QE shape (a = 0.15, 0.30); T-Q slope (b × 0.8, × 1.2); timing width (× 1.25, × 1.5 on affected modules); coupled mPMT gain + width ((0.9, 1.25), (0.8, 1.5)).
  - Supporting cells: isolated gain and width; module loss (1 or 5 modules); the charge-matched, not exact, control.
  - Controls: offset only (+1 ns); identity; exact QE×CE product.
- **Held-out units:**
  - the `timing_width` family;
  - the larger severity point of every N family;
  - mPMT modules with index mod 3 = 2 (35 of 105);
  - evaluation perturbation seeds 301–303.
- **Consequence and gates:** K_vertex, the R-3 gates for vertex and direction, and the split roles.

**Pending D-E (`pending_D-E`):** K_energy, the aggregate K = max(K_vertex, K_energy), the energy gates, and event support (contained initial energy and S families). The development default for D-E is deposited energy (A).

**Not frozen:**
- κ_m = 1.10 (`provisional_dev`; the author must confirm or replace it at this freeze);
- WP7 event counts and the compute plan;
- the precision study, which has not been run, so the severity grids are not yet shown to cross the harm boundary;
- D-T/L/M/S.

---

16 October 2026. **DRAFT — UNFROZEN (superseded by WP6 freeze v1 above).** This document prepares WP6. The user subsequently authorized real LUCiD integration, prompt Cherenkov validation and baseline GPU training on Deepthought on 16 October 2026. The bounded engineering pilot is specified in `configs/lucid_pmt/integration_pilot.json`; this is not a scientific protocol freeze or authorization for a new transfer arm. Claims, labels and `GENERIC_COMMITTED` remain as in `PREREGISTRATION.md` §§1–8.

## 🔌 Interfaces and baseline

Input is `PhotonBatch` (`pmt-photons-v1`): incident photons with persistent IDs, sensor IDs, forward arrival times in ns, wavelengths in nm and declared upstream effects. Response parameters and their provenance are versioned in `configs/lucid_pmt/reference.json`. Wavelengths and truth are simulation/evaluation data. The deployed model/monitor sees measured sensor charge/time, geometry, grouping and frozen operational calibration only, through an allowlist. All twins and variants preserve physical event identity.

Candidate target: pinned LUCiD WCTE geometry. Actual PMT hardware model, electronics-crate map, wavelength QE/CE and width calibrations remain pending. The later multi-energy pilot resolves upstream physical mPMT membership; it does not establish crate membership or measured group response. The current 76-channel cylinder is a synthetic flasher fixture, not WCTE or a Cherenkov event. The follow-up real exporter before QE/TTS and bounded prompt Cherenkov closure now pass; broader target-detector validation is still required before physics evaluation. Baseline includes nontrivial group gain/width and per-channel T-Q curves. Uniform angular response, prompt light and no concentrators are settled for the first trial.[^request]

Subject interface draft: per-event measured digit arrays plus sensor geometry → frozen reconstruction outputs and declared internal hooks. Padding/masking, architecture, output target, operational noise-reference source and trainability thresholds must be fixed before model fitting. The original CPU pass trained no model. The follow-up engineering pilot uses a measured-digit PointNet direction model because the existing compact waveform transformer assumes a different input representation. This does not replace the primary scientific model or baseline by implication. Train baseline response only with ≥3 seeds, target 5; no per-variant primary retraining.[^request]

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
| S support and track target | Candidate families above; primary-start/direction/initial-energy development pilot, scientific targets unfrozen | Author/Junjie before event generation for claims |
| D-K | Draft max requires both endpoints; any failed fit makes cell unevaluable; conditional complete-case statistic and both failure rates reported separately | Author before primary harm evaluation |
| κ_m | 1.10 remains `provisional_dev`; no thesis-derived threshold | Author before primary harm evaluation |
| Dead-map correction | Frozen reference-fit channel shares; uncorrected/corrected outputs both retained | Author chooses primary policy |
| Denominator | Same-event clean errors; zero denominator is undefined | Author adopts policy before primary evaluation |
| Counts/severity/held-out units | Development precision study not run | Author after bounded precision work, before held-out evaluation |
| Monitor feature amendment | Supporting diagnostics only | Author before changing primary comparison |
| Compute budget | Original CPU fixture retained. Follow-up: 24 physical PhotonSim events, 16 rotations each; 3 baseline seeds; 60 epochs/seed; 30-minute generation and total training caps; one GPU at 25% memory fraction | Pilot budget implemented; full scientific WP7 budget pending |

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

## 16 October follow-up integration pilot

The real input is the upstream public 500 MeV water-muon PhotonSim example, events 0–23 read under a 128 MiB network cap. Source emission positions are explicitly converted from mm to m. Physical IDs 0–15 train, 16–19 select checkpoints, and 20–23 provide the one-time pilot test; all 16 rotations of an event remain together. Rotation is data augmentation, not another independent physical event. The target is the primary direction; the single-energy sample cannot validate energy inference. Shower centring and in-volume/spectral selections are generation operations, recorded per event, not model features.

The sensor geometry is pinned WCTE (1,995 channels). The PMT response remains an explicitly synthetic development model. Groups of 19 sequential indices are synthetic groups, not verified hardware crates. Transport uses hard first-forward spherical intersections before QE/CE/TTS, with uniform acceptance and scattering/reflection/absorption/dispersion off. Float64 and 16 sensor candidates address numerical precision and candidate omissions observed during engineering closure. This is not a full optical-detector validation.

Inputs contain only measured charge/time, sensor position and occupancy. The PointNet uses masked mean/max pooling and a unit-vector head; the exact model, optimizer, seeds, splits and gates are recorded before training in the pilot JSON. No primary harm score, attribution claim, S-family selection, transfer decision or target-PMT calibration is resolved by passing this pilot. See `PMT_LUCID_INTEGRATION_REPORT_2026-10-16.md` for results and preserved failures.

## 16 October multi-energy/track follow-up

The user separately authorized multi-energy and track validation and GPU training. The frozen development settings are in `configs/lucid_pmt/multienergy_pilot.json`; the [validation report](PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md) preserves the source-read STOP, balanced 20-event-per-energy amendment and transport candidate repair before training. This expands development to 500/1,000/1,500 MeV, eight placements per event, 60 physical events and 480 samples, with three model seeds and 100 epochs per seed. Generation and total training retain separate 30-minute caps, one GPU and a 25% process memory limit. Response-fit, train, validation and test membership is disjoint at the physical-event level. The prior 500 MeV development use prevents treating this as pristine confirmation.

The starting vertex is placed directly; the shower is not centred. Initial direction and initial kinetic energy are supporting reconstruction targets. Nearly every placement is escaping, and post-generation cropping of water-simulation photons is an explicit approximation. The 105-group physical mPMT map replaces sequential-index synthetic membership in this pilot, while all response amplitudes and timing curves remain synthetic. None of these implementation choices freezes the primary energy target, containment policy, track endpoint, D-K, harm threshold, primary baseline or final held-out campaign. The [author-decision register](PMT_CALIBRATION_AND_PROTOCOL_DECISIONS_2026-10-16.md) lists those choices.
