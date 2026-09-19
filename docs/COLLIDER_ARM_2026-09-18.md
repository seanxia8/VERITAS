# Arm D (Tier 3b) — collider domain transfer: CMS / ATLAS

_Future-work arm, 18 September 2026. **Not** part of the minimum viable paper and
**not claim-bearing**: it cannot carry Claim 1's N-vs-S attribution or the
calibrated Σ̂/Σ mechanism. It is recorded because (i) it is the natural place to
test whether the diagnostics transfer to a second readout regime and a second
detector-layout axis, and (ii) the collider field already has open,
layout-variable simulation and a body of representation-transfer work this study
must cite and position against. Canonical design: `EXPERIMENT_DESIGN.md` §II.10._

## 1. Why it is a separate, later arm

The paper's core distinctions need a controllable acquisition contract (N), an
exactly paired counterfactual, and a calibrated assumed-vs-realized covariance.
Collider data have none of these: there is no electronics/acquisition layer the
study controls, open data are reconstruction/object-level, and CMS vs ATLAS is a
**geometry (G) contrast between two different detectors**, not an N-vs-S design.
So the collider arm is a **G-transfer / representation-diagnostics** arm: does a
frozen reconstruction representation move with detector layout, and do its
intermediate latents order scientific harm across layouts? That is worth a
future-work paragraph and a possible follow-up paper, not a headline claim.

## 2. Sources

### 2.1 Fast simulation — Delphes detector cards (cheapest harness)

One shared HepMC/Pythia/MadGraph event sample pushed through different detector
cards; a layout change is a card change, so pairing is exact by construction.
Cards shipped in `github.com/delphes/delphes/cards` include
`delphes_card_ATLAS.tcl`, `delphes_card_CMS.tcl`, `CMS_PhaseII`, `HLLHC`,
`CLICdet_Stage1/2/3`, `ILD`, `IDEA`, `FCCeeDetWithSiTracking`, `CEPC`,
`MuonCollider`, `LHCb`. GPL-3.0, active. Output is particle-flow/reco level, so
the arm is constructed waveform/representation evidence under a declared fast-sim
contract.

### 2.2 Full simulation — Key4hep / DD4hep detector concepts (richest harness)

Same e+e- physics through different full GEANT4 geometry descriptions: **CLICdet,
CLD, IDEA, SiD, ILD**. Exact by re-simulating one generator file. Key4hep and
DD4hep are Apache-2.0 and open. This is the setup of the MLPF cross-detector
transfer study (arXiv:2503.00131). CMSSW (CMS Phase-2) and Athena (ATLAS ITk)
offer the same property for pp physics but are collaboration-gated and heavy.

### 2.3 Open datasets

| dataset | content | level | note |
|---|---|---|---|
| **CERN Open Data — CMS** | 2010–2015 pp collisions; simulated and real | reco/AOD, not raw traces | CC0/CC-BY; API `opendata.cern.ch/api/records` |
| **ATLAS Open Data** | 2025 beta release (record 93910); 65 TB for research (2024); first open event-generation batch (record 160000) | reco/object + generator level | https://opendata.atlas.cern/ |
| **TrackML** (CERN/Kaggle) | point-cloud tracking hits with detector geometry | hit/point cloud | single geometry |
| **Open Data Detector (ODD) + ACTS** | generic tracking detector in DD4hep for algorithm development | hit/point cloud | open; ACTS Apache-2.0 |

## 3. Prior work this arm must cite and position against

**Representation / transfer precedent (Claim 1's premise, in the collider domain):**

- **arXiv:2606.14373** — *Machine-learned particle flow as a foundation model for
  collider physics.* The per-particle **latent representation learned during
  reconstruction** supports jet flavour, jet energy and missing-momentum
  regression; a single linear layer on the latents is competitive, and beats
  kinematic baselines for MET with ~35× fewer parameters. **Direct precedent that
  intermediate reconstruction representations carry downstream physics
  information** — the premise of Claim 1.
- **arXiv:2604.12364** — *Cross-Domain Transfer with Particle Physics Foundation
  Models: From Jets to Neutrino Interactions.* OmniLearned / ParticleViT
  pretrained on pp/ep collisions transfer to few-GeV MINERvA neutrino events
  across energy scale, detector technology and process; BERT-initialised controls
  do not. Evidence for detector-agnostic inductive bias.
- **arXiv:2512.00187** — *Cross-Geometry Transfer Learning in Fast Electromagnetic
  Shower Simulation.* Geometry generalization for calorimeter generative models.
- **arXiv:2305.11531** — *GAAMs: Generalizing to new geometries with
  Geometry-Aware Autoregressive Models for fast calorimeter simulation.* One
  geometry-aware model replaces per-geometry models; a geometry-conditioning
  precedent for the structured decoder of `TESTBEDS.md` §2.
- **arXiv:2503.00131** — *Fine-tuning machine-learned particle-flow reconstruction
  for new detector geometries in future colliders.* CLICdet → CLD cross-detector
  transfer; an order of magnitude fewer samples matches from-scratch training.
  First full-simulation cross-detector transfer for particle flow.
- **arXiv:2609.00611** — *Panda Diplomacy (Panda V2).* Cross-detector foundation
  model (LArTPC, collider-TPC, water Cherenkov); the paper's cross-detector foil,
  already cited.
- **arXiv:2510.24066** — *OmniLearned: A Foundation Model Framework for All Tasks
  Involving Jet Physics.*
- **arXiv:2605.29283** — *fmbench: Do Physics Foundation Models Learn Generalizable
  Physics?* Regime-dependent scores across distribution shifts without a mechanism
  — the gap this study's mechanism section targets.

**Shift / domain-adaptation caution (what the arm must not over-claim):**

- **arXiv:2608.15166** — *Stress-Testing DANTE under Detector Domain Shift: a
  Representation-Coherent Reanalysis of LIGO O4a.* Measured failure modes of a
  frozen-representation pipeline under detector domain shift; a cautionary
  precedent for representation-coherent monitoring.
- **arXiv:2608.18190** — *Safe Domain Adaptation for Physics: Overcoming Nuisances,
  Label Shifts, and Simulation Priors.* Separates a detector-response nuisance, a
  physical simulation shift and a spectrum shift on a toy air-shower benchmark;
  the N-vs-S-vs-prior distinction ORACLE formalises, and the argument for why
  adversarial adaptation can anchor on a simulation prior.

**Prior art to position against (not reproduce):**

- **arXiv:2501.13789** — CMS ML data-quality monitoring. Deployed ML DQM at scale;
  the N-detection half of this proposal is "DQM by another name" from a HEP
  referee's seat.
- **arXiv:2309.10157** — CMS ECAL autoencoder anomaly detection (production use).
- **arXiv:2407.20278** — CMS ECAL autoencoder follow-up.
- **arXiv:2105.08742** — systematics-aware learning ("train invariance rather than
  monitor"). Needs an explicit answer.

The distinction to draw: DQM asks *"is the detector behaving"*; this study asks
*"has the reconstruction model's internal representation moved in a direction that
costs physics"* — a question about the model, not the apparatus.

## 4. Fit against the ORACLE admission criteria (`DATASET_STRATEGY` §3)

| criterion | Delphes cards | Key4hep concepts | ATLAS/CMS open data |
|---|---|---|---|
| observable contract | yes | yes | yes (reco-level) |
| N and S independently | **no** | partial (digitization config) | no controlled acquisition |
| paired counterfactuals | exact (same HepMC) | exact (same generator) | **no** |
| independent K | yes (JES, MET, mass peak, efficiency) | yes | yes |
| hard contrasts | tunable | tunable | limited |
| unknowns held out | yes | yes | yes |
| operational noise reference | no random triggers | no | no |
| release | GPL-3, open | Apache-2.0, open | open |

## 5. Recommended shape if pursued

Delphes cards (cheap) or Key4hep CLICdet→CLD (full sim), one shared event sample,
a frozen reconstruction subject (MLPF-family), **layout cells as the G
intervention**, and an independent jet/MET/mass K. Report as **G-transfer only**;
do not fold layout into N or S. Keep it out of the minimum viable paper.

## 6. Sources

- Delphes cards: https://github.com/delphes/delphes/tree/master/cards
- ATLAS Open Data: https://opendata.atlas.cern/ (records 93910, 160000)
- CERN Open Data: https://opendata.cern.ch/ (`/api/records`)
- Key4hep: https://key4hep.github.io/key4hep-doc/ · DD4hep: https://dd4hep.web.cern.ch/dd4hep/
- ACTS / ODD: https://github.com/acts-project/acts
- arXiv:2503.00131, 2606.14373, 2604.12364, 2512.00187, 2305.11531, 2609.00611,
  2510.24066, 2605.29283, 2608.15166, 2608.18190, 2501.13789, 2309.10157, 2407.20278
