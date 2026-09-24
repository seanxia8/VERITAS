# ATLAS/CMS same-physics / different-detector-layout resources for a later domain-transfer arm

_Research note, 18 September 2026. Compiled for ORACLE from open scholarly APIs
(arXiv, OpenAlex), the CERN Open Data API and upstream pages. Not a protocol
freeze. URLs and licences must be re-checked before any dependency is pinned._

## Question

For a later **domain-transfer expansion** (not a claim-bearing arm now): is there
an open simulation where the *same physics* can be pushed through *different
detector layouts*, with exact event pairing, so the ORACLE diagnostics
(N/S/G, alarm, harm) can be exercised on collider data — and what open datasets
and prior representation/diagnostics work exist?

## 1. Open simulations with detector-layout variation

| resource | same physics, different layout? | pairing | licence / state |
|---|---|---|---|
| **Delphes detector cards** (`github.com/delphes/delphes/cards`) | yes — identical HepMC/Pythia/MadGraph events through `delphes_card_ATLAS.tcl`, `_CMS.tcl`, `CMS_PhaseII`, `CLICdet_Stage1/2/3`, `ILD`, `IDEA`, `FCCeeDetWithSiTracking`, `CEPC`, `HLLHC`, `MuonCollider` | exact: one HepMC file, N cards | GPL-3; active; pure C++/Python |
| **Key4hep / DD4hep detector concepts** (CLICdet, CLD, IDEA, SiD, ILD) | yes — same e+e- physics, different full GEANT4 geometry descriptions | exact: one generator file, re-simulate | Apache-2.0 (Key4hep/DD4hep); open |
| **Open Data Detector (ODD) + ACTS** | generic tracking detector in DD4hep; used for algorithm development | one geometry, re-runnable | open; ACTS is Apache-2.0 |
| **CMSSW / Athena** (CMS Phase-2, ATLAS ITk) | yes, same pp physics, upgraded layouts | exact if you run your own sim | source open, but collaboration-gated for detector description and heavy to run |
| **TrackML** (CERN/Kaggle) | point-cloud tracking dataset with detector geometry | single geometry | open, CC |
| **ATLAS Open Data / CMS Open Data** (CERN Open Data Portal) | real data from two detectors, same pp collisions — but **not event-paired**, and reco/object level | no | CC0/CC-BY; ATLAS 2025 beta, 65 TB, first event-generation batch (record 160000); CMS 2010–2015 |

**Cheapest usable harness:** Delphes cards + one shared event sample. A layout
change is a card change; pairing is exact by construction. It is reconstruction/
particle-flow level (no raw electronics), so it is a **G (geometry/forward-model)
shift**, not N or S.

**Richest full-simulation harness:** Key4hep/DD4hep detector concepts (CLICdet →
CLD, or CLD vs IDEA) with one shared generator file — exactly the setup of the
MLPF cross-detector transfer paper (arXiv:2503.00131).

## 2. Prior work — representation / diagnostics across detector layouts

- **arXiv:2503.00131** — *Fine-tuning machine-learned particle-flow reconstruction
  for new detector geometries in future colliders.* CLICdet → CLD transfer; order
  of magnitude fewer samples matches training from scratch. First full-simulation
  cross-detector transfer for particle flow. The closest precedent to ORACLE's
  transfer arm, and direct evidence that a reconstruction representation carries
  transferable structure across layouts.
- **arXiv:2606.14373** — *Machine-learned particle flow as a foundation model for
  collider physics.* The per-particle **latent representation learned during
  reconstruction** supports jet flavour, jet energy and missing momentum; a single
  linear layer on the latents is competitive. Direct precedent for ORACLE Claim 1's
  premise (intermediate representations carry physics information).
- **arXiv:2604.12364** — *Cross-Domain Transfer with Particle Physics Foundation
  Models: From Jets to Neutrino Interactions.* OmniLearned / ParticleViT pretrained
  on pp/ep collisions transfer to a few-GeV fixed-target neutrino experiment across
  energy scale, detector technology and process. Detector-agnostic inference claim.
- **arXiv:2512.00187** — *Cross-Geometry Transfer Learning in Fast Electromagnetic
  Shower Simulation.* Geometry generalization in calorimeter generative models.
- **arXiv:2305.11531** — *GAAMs: Generalizing to new geometries with Geometry-Aware
  Autoregressive Models for fast calorimeter simulation.* A single geometry-aware
  model replaces per-geometry models; a geometry-conditioning precedent.
- **arXiv:2608.15166** — *Stress-Testing DANTE under Detector Domain Shift: a
  Representation-Coherent Reanalysis of LIGO O4a.* Detector-domain shift with
  frozen representations; a cautionary measurement of failure modes.
- **arXiv:2608.18190** — *Safe Domain Adaptation for Physics: Overcoming Nuisances,
  Label Shifts, and Simulation Priors.* Separates a detector-response nuisance, a
  physical simulation shift and a spectrum shift on a toy air-shower benchmark —
  the N-vs-S-vs-prior distinction ORACLE formalises.
- **arXiv:2502.07724** — *Contrastive Learning for Robust Representations of
  Neutrino Data.* Controlled augmentations for sim-to-real transfer.
- **Panda V2 (arXiv:2609.00611)** — cross-detector foundation model (LArTPC, TPC,
  water Cherenkov); already cited by ORACLE as the cross-detector foil.
- **OmniLearned (arXiv:2510.24066)**, **fmbench (arXiv:2605.29283)** — foundation
  models and regime-dependent shift benchmarks.
- **CMS ML DQM (arXiv:2501.13789)**, **CMS ECAL autoencoder AD (2309.10157,
  2407.20278)** — the prior art ORACLE must position against, not reproduce.

## 3. Fit against the ORACLE admission criteria (DATASET_STRATEGY §3)

| criterion | Delphes cards | Key4hep concepts | ATLAS/CMS open data |
|---|---|---|---|
| observable contract | yes (truth vs reco separable) | yes | yes but reco-level only |
| N and S independently | **no** — no electronics/acquisition layer | partial (digitization config) | no controlled acquisition |
| paired counterfactuals | exact (same HepMC) | exact (same generator) | **no** |
| independent K | yes (JES, MET, mass peak, efficiency) | yes | yes |
| hard contrasts | can be tuned | can be tuned | limited |
| unknowns held out | yes | yes | yes |
| operational noise reference | no random triggers | no random triggers | no |
| release | GPL-3, open | Apache-2.0, open | open |

**Conclusion.** Collider resources can carry a **G/layout-transfer and
representation-diagnostics** study (does the frozen representation move with
layout? do intermediate latents rank harm across layouts?) and are a natural
"domain transfer" future-work arm — but they **cannot** carry Claim 1's N-vs-S
attribution or the calibrated Σ̂/Σ mechanism, because there is no controllable
acquisition noise and no exact paired counterfactual over N. CMS vs ATLAS is a
geometry contrast between two different detectors, not an N-vs-S design.

**Recommended shape if pursued later:** Delphes cards (cheap) or Key4hep
CLICdet→CLD (full sim) with one shared event sample, a frozen reconstruction
subject (e.g. an MLPF-family model), G-cell layout changes as the intervention,
and an independent jet/MET/mass K. Report it as a G-transfer arm; do not fold it
into N or S.

## 4. Sources

- Delphes cards: https://github.com/delphes/delphes/tree/master/cards
- ATLAS Open Data: https://opendata.atlas.cern/ (2025 beta; CERN record 93910; event-generation batch 160000)
- CERN Open Data portal API: https://opendata.cern.ch/api/records
- Key4hep: https://key4hep.github.io/key4hep-doc/ ; DD4hep: https://dd4hep.web.cern.ch/dd4hep/
- ACTS: https://github.com/acts-project/acts
- MLPF cross-detector: https://arxiv.org/abs/2503.00131
- MLPF foundation model: https://arxiv.org/abs/2606.14373
- Cross-domain FM transfer: https://arxiv.org/abs/2604.12364
- Cross-geometry shower transfer: https://arxiv.org/abs/2512.00187
- GAAMs: https://arxiv.org/abs/2305.11531
- DANTE detector domain shift: https://arxiv.org/abs/2608.15166
- Safe domain adaptation for physics: https://arxiv.org/abs/2608.18190
- Contrastive neutrino representations: https://arxiv.org/abs/2502.07724
