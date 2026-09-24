# Dataset and simulator strategy for the two-claim paper

_Research audit: 16 September 2026. This memo evaluates the current experiment
plan against `TWO_CLAIM_REVISION_PLAN.md`. It is a recommendation, not a protocol
freeze. Upstream status and licences must be rechecked when a dependency is
pinned._

## 1. Decision

The repository has a detailed experiment plan, but its dataset strategy is
conceptually stale. `TESTBEDS.md` and Parts I--II of `EXPERIMENT_DESIGN.md` were
written for the older C0--C5 programme. They optimize for three readout domains,
geometry variation, and a mechanism ladder. The revised paper has two narrower
questions:

1. do intermediate representations add acquisition-versus-valid-physics
   attribution information beyond the same strong generic observations; and
2. does a fixed task-sensitive score rank independently measured scientific
   harm better than a fixed generic score?

The new questions require paired counterfactuals, deliberately overlapping N/S
signatures, and an independent scientific endpoint. A large collection of
detectors is less useful than one controlled simulator and one real-noise
transfer dataset that satisfy those requirements cleanly.

**Recommended evidence ladder**

1. Keep ORACLE-Cov as an analytic and implementation control only.
2. Use **NuRadioMC/NuRadioReco** as the primary controlled physical simulator.
   It is the best immediate substitute for HeST and can also replace LUCiD if
   LUCiD remains unlicensed.
3. Use **GWOSC real strain with PyCBC/LALSimulation injections** as the main
   external, real-noise transfer test.
4. Keep **TIDMAD** as a bounded supporting test of scientific consequence if
   storage and compute permit. It should not carry acquisition-versus-physics
   attribution.
5. Keep **LUCiD** behind its licence gate. Demote **HeST + the home-built QP
   and noise chain** from claim-bearing evidence to an optional domain-motivated
   case study.

If the paper must stay close to cryogenic rare-event detectors, the best
alternative ladder is **SolidStateDetectors.jl/LegendGeSim for controlled
simulation plus Majorana or CRESST pulse data for external validation**. That
route has stronger domain continuity but a less direct real-data Claim 2 than
the GW route.

## 2. What is stale in the current plan

The dates are recent; the scientific logic is stale.

| Current item | Audit | Required change |
|---|---|---|
| `TESTBEDS.md` calls itself canonical and says the programme needs physics, geometry, and project-owned noise simultaneously | This predates the two-claim revision. Project-owned noise is useful for controls but weakens external validity if it supplies most claim-bearing N examples. | Replace the three-axis admission rule with the requirements in section 3 below. |
| `EXPERIMENT_DESIGN.md` still states that three testbeds are fixed and retains C0--C5 | This conflicts with the two-claim plan and gives every arm a different role and endpoint. | Rewrite around one controlled arm and one transfer arm, with common estimands. |
| LUCiD is the planned Arm A | Upstream still says its licence is being finalized. An unlicensed dependency cannot be a core reproducible testbed. | Preserve the adapter design, but do not depend on it until an SPDX licence appears upstream. |
| HeST is the planned Arm B | It is explicitly an early-development package with a tiny maintainer base. The repository supplies the QP trace model, electronics, noise, and harm proxy; therefore much of the apparent transfer is self-authored. ER/NR is also energy-degenerate in the QP-only readout. | Demote it or replace it. Do not use it as the sole evidence that the method transfers to physical detector simulation. |
| TIDMAD is described as external validity for the full programme | It has real unchosen noise and a genuine dark-matter-limit pipeline, but its released labels do not provide a rich acquisition-versus-supported-physics attribution experiment. | Use it for Claim 2/supporting consequence evidence only. |
| Prometheus is the fallback | Its public output is hit-level rather than a native dense electronics trace. A custom waveform bridge would make both trace formation and noise ours. | Keep it only for a frozen public reconstruction model and angular-error endpoint, or label a bridge as constructed evidence. |
| Several arms use reconstruction loss as harm | Reconstruction loss is internal to the monitored model and can make Claim 2 circular. | Require a physical task error or collaboration analysis product in physical units. |

The current plan is still useful as an engineering survey. Its upstream checks,
trace-shape inventory, and pairing analysis should be preserved. It should no
longer be the authoritative scientific selection rule.

## 3. Admission criteria for a claim-bearing testbed

A dataset or simulator must satisfy all mandatory items before it carries a
primary claim.

1. **Observable contract.** Alarm-time inputs, delayed labels, simulator truth,
   paired twins, and realised noise parameters can be separated mechanically.
2. **Two intervention axes.** Acquisition interventions N and valid physical
   variations S can be changed independently. Geometry/forward-model changes
   remain a third label rather than being folded into S.
3. **Paired or matched counterfactuals.** The same physical event can be replayed
   under N, or N and S can be matched on SNR, output confidence, and other generic
   features. Event identity is available for grouped splits.
4. **Independent harm.** K is measured in scientific units by a frozen task
   procedure: parameter error, angular error, energy error, coverage, detection
   efficiency, or a published physics limit. It is not the monitored model's
   reconstruction loss.
5. **Hard contrasts.** The design can create N and S cells whose generic alarms
   overlap; easy separability is insufficient for Claim 1.
6. **Unknowns.** At least one physical family and one acquisition family can be
   excluded from all fitting and used only for abstention evaluation.
7. **Operational noise reference.** Off-source, forced-trigger, random-trigger,
   or neighbouring clean records exist at alarm time, or their absence is an
   explicit limitation shared by the compared arms.
8. **Reproducible release.** A usable licence, immutable version, provenance,
   documented schema, and feasible subset are available.

Maintenance is judged from governance, releases, CI, documentation, and recent
activity. The career stage of individual authors is not evidence for or against
scientific reliability.

## 4. Candidate comparison

Scores are 0 (absent), 1 (possible with a material caveat), or 2 (native/strong).
They are decision aids, not measurements.

| Candidate | C1 attribution | C2 harm | pairing/matching | real or independent noise | release health | engineering cost | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| NuRadioMC/NuRadioReco | 2 | 2 | 2 | 1 | 2 | 1 | **Primary controlled simulator** |
| GWOSC + PyCBC/LALSimulation | 2 | 2 | 2 | 2 | 2 | 1 | **Primary real-noise transfer** |
| SolidStateDetectors.jl/LegendGeSim | 2 | 2 | 2 | 1 | 2 | 1 | Best cryogenic/semiconductor simulation fallback |
| CRESST-II/III pulse data | 2 | 1 | 1 | 2 | 1 | 2 | Real cryogenic Claim 1/supporting validation |
| Majorana AI/ML release | 1 | 2 | 1 | 2 | 2 | 1 | Real pulse-shape and energy validation |
| TIDMAD | 1 | 2 | 1 | 2 | 2 | 0 | Supporting Claim 2 only; large data |
| Project 8 via PhyTS | 1 | 2 | 1 | 1 | 2 | 1 | Useful fixed benchmark; insufficient N-family breadth alone |
| LUCiD + custom noise | 2 | 1 | 2 | 1 | 0 while unlicensed | 1 | Conditional candidate, not a core dependency yet |
| HeST + custom QP/noise | 1 | 1 | 2 | 0 | 1 | 2 because already built | Optional case study; do not let sunk cost make it claim-bearing |
| Prometheus + custom waveform bridge | 2 | 2 | 2 | 0 | 2 | 1 | Useful public-model control; constructed waveform evidence |

## 5. Recommended primary simulation: NuRadioMC/NuRadioReco

### Why it fits

- It generates neutrino events separately from detector simulation. A saved HDF5
  event list contains energy, flavour, interaction, vertex, and direction and
  can be replayed through detector JSON files.
- Detector descriptions expose channel positions, orientations, antenna types,
  sampling, and responses. The pipeline produces channel voltage traces and
  trigger objects.
- Thermal and Galactic noise modules, filtering, amplifiers, triggers, emitters,
  and likelihood reconstruction already exist and can be enabled, disabled, or
  varied without inventing a second detector chain.
- Event-group identifiers and saved truth naturally support grouped splits.
- The project has a GPL-3.0 licence, packaged releases, documentation, tests, and
  an active multi-institution development history. Version 3.0 was a substantial
  refactor and the current documentation identifies a 3.2 development line.

### Intervention table

| Contract | Families | Examples | Matching variables |
|---|---|---|---|
| clean | reference detector and noise configuration | fixed RNO-G-like station subset | energy, direction, vertex, event id |
| N-covariance | acquisition noise changes | thermal RMS/PSD; Galactic coherent component; narrow spectral line; nonstationarity | network SNR, trigger multiplicity, output confidence |
| N-structural | detector/readout faults | gain or phase calibration; cable delay; channel dropout; sampling/clock offset | same event, same physical truth |
| S-supported | valid physics population changes | energy, direction, flavour, CC/NC, inelasticity within declared training support | SNR, zenith, trigger multiplicity |
| G/model | forward-model change, kept separate | antenna response, ice model, detector layout | same event list |
| U | held-out families | emitter/pulser; omitted Askaryan model; unseen N line/coherence family | evaluation only |

Do not call an ice-model or detector-response change a valid physics-population
shift. Those are forward-model or geometry changes and should remain supporting
labels.

### Endpoints

- Claim 1: paired delta macro-F1 for N versus supported S on SNR- and
  confidence-matched cells.
- Claim 2: paired delta AUROC for harmful cells, with K based on frozen
  reconstruction error in neutrino direction or energy. Trigger efficiency can
  be a second physical endpoint.
- Supporting: low-alarm harmful quadrant, false rejection of benign valid S,
  unknown-family risk--coverage, and results by N-covariance/N-structural.

Use a NuRadioReco reconstruction that does not consume the monitored model's
intermediate representation. Otherwise K is not independent.

### Pilot gate

Before adapting the whole pipeline, run 1,000--5,000 events through two detector
configurations and four cells: clean, one N-covariance, one N-structural, and one
supported S. The candidate passes only if:

- identical event truth survives replay across cells;
- clean and noisy traces, operational noise summaries, and reconstruction truth
  can be exported without evaluation leakage;
- at least one matched N/S pair overlaps in generic-score space;
- direction or energy error changes independently of the monitored
  reconstruction loss; and
- a bounded run is feasible on the intended compute.

### NuRadioMC versus LUCiD

The microscopic physics differs substantially, but their role in this paper
overlaps.

| Layer | NuRadioMC/NuRadioReco | LUCiD |
|---|---|---|
| primary interaction | ultra-high-energy neutrino interaction and particle shower in ice | charged-particle/light source in water, WbLS, or ice |
| emitted carrier | coherent Askaryan radio pulse from the shower charge excess | Cherenkov/scintillation optical photons |
| propagation | refractive-index-dependent radio ray tracing, attenuation, focusing, reflections, optional birefringence | optical ray transport with absorption, scattering, reflections, and material response |
| sensor/readout | sparse antennas, analogue chain, filters, voltage sampling and trigger | dense PMT arrays, photon arrival/charge response and waveform convention |
| characteristic noise | thermal and coherent Galactic/radio backgrounds plus electronics | photon counting, dark counts, transit-time spread and added PMT electronics |
| natural task endpoint | neutrino direction/energy and trigger efficiency | track/source parameters and calibration/reconstruction error |

Both are simulated neutrino-detector chains with configurable geometry,
multi-channel time series, exact truth, and project-controlled interventions.
Consequently, using both as equal headline arms would exaggerate the breadth of
the evidence. Their shared causal structure is useful for replication, but it is
not a substitute for real, unchosen noise.

Use **NuRadioMC as the claim-bearing simulator** because it is licensed, has a
native detector/electronics/noise/trigger stack, and supplies an independent
reconstruction endpoint. Keep **LUCiD as a conditional supplementary
replication** only if its licence is resolved and it is run without changing
the hypotheses or score definitions. If both are used, present the contrast as
radio-antennas versus optical-PMT transport within the neutrino-detector domain;
do not call it broad cross-domain validation. The main transfer claim should
still come from real noise such as GWOSC or a cryogenic release.

## 6. Recommended real-noise transfer: GWOSC + PyCBC

### Why it fits

GWOSC supplies public detector strain, data-quality information, injection
segments, event catalogs, and parameter-estimation products through a stable
API. PyCBC is an actively maintained collaboration package that can generate
waveforms, inject them into real or simulated strain, apply calibration models,
and run detection or parameter estimation. The combination supplies real,
unchosen acquisition noise while retaining exact injection truth.

### Intervention table

| Contract | Families | Examples |
|---|---|---|
| N-covariance | off-source noise state | PSD drift across time, line contamination, nonstationary segment |
| N-structural | acquisition fault | amplitude/phase calibration perturbation, dropout of one interferometer, injected glitch or labelled glitch segment |
| S-supported | valid astrophysical change | component masses, aligned spins, distance, inclination at matched network SNR |
| U-physics | held-out waveform morphology | precession, eccentricity, higher modes, or burst waveform omitted from fitting |
| U-acquisition | held-out glitch morphology | one Gravity Spy or data-quality family omitted from fitting |

### Endpoints

- Claim 1: N versus supported S at matched network SNR and output confidence.
- Claim 2: chirp-mass error is the cheapest primary K; recovered network SNR or
  missed-detection status can support it. Sky-localisation area or posterior
  coverage is more scientifically complete but more expensive.
- For the first pilot, avoid claiming real-event ground truth. Inject known
  waveforms into off-source real strain and reserve catalogued detections for a
  qualitative external check.

### Limitations

- It is less close to the project's cryogenic-detector motivation.
- PSD estimation and injection placement must be split by time segment to avoid
  leakage.
- Glitch labels and auxiliary-channel products have their own licences; verify
  each asset rather than assuming the GWOSC strain licence covers it.
- Full Bayesian parameter estimation is expensive. Begin with a frozen matched
  filter or compact regressor and chirp-mass error.

## 7. Cryogenic alternative

If detector-domain continuity is required, use two complementary resources.

### Controlled: SolidStateDetectors.jl/LegendGeSim

This stack provides detector geometry, electric and weighting potentials,
charge drift, per-contact waveforms, event hit tables, and LEGEND detector
metadata. It is licensed and actively developed by established project
organisations. Replay identical hit tables through detector/contact and
electronics configurations. Use single-site versus multi-site deposits and
position/energy changes as S; electronics noise, gain, shaping, and contact loss
as N; energy bias or pulse-shape classification error as K.

The cost is a Julia bridge and, for realistic deposits, a Geant4/remage input
chain. Run a small waveform/export pilot before making it primary.

### Real: Majorana or CRESST

- The Majorana release contains raw HPGe waveforms, calibrated energies, pulse
  shape labels, fixed train/test files, checksums, and a collaboration DOI. It
  is well suited to independent energy/pulse-shape harm, but has no clean
  counterfactual twin.
- The CRESST release contains labelled cryogenic pulse shapes across 68
  detectors and is highly relevant to valid-event versus acquisition-artifact
  attribution. Its strongest use is Claim 1 and false rejection of valid rare
  events; it does not by itself provide the clean paired intervention grid
  needed for the primary Claim 2 comparison. Confirm the dataset's explicit
  reuse licence and stable identifier before adoption.

## 8. Roles of existing candidates

### LUCiD

LUCiD remains scientifically attractive: configurable detector geometry,
differentiable photon transport, waveform output, and reconstruction variables
fit the controlled design. Its upstream README still says the licence is being
finalized. Until that changes, it is unsuitable as a reproducible core
dependency. If a licence lands, compare its pilot against NuRadioMC rather than
automatically adding a third simulated arm.

### HeST

The concern is not that its authors are graduate students. The relevant risks
are its explicit early-development status, small bus factor, ambiguous
copyright line noted in the local audit, limited validation surface, and the
large amount of downstream detector response supplied by this repository.
Because the paper's harm proxy and noise would also be project-authored, HeST
does not give sufficiently independent support for the two headline claims.
Preserve the completed adapter as an optional case study or regression fixture.

### TIDMAD

TIDMAD has a clear CC BY 4.0 release, DOI, real SQUID science data, injected
training/validation data, and a pipeline that ends in a dark-matter limit. That
makes it valuable for the principle that alarm magnitude must be judged against
scientific consequence. It has a weak N-versus-S taxonomy, one main instrument
configuration, and very large files. Use a predeclared subset for feasibility,
then run the full science path only if Claim 2 remains promising.

### PhyTS / Project 8

PhyTS is useful as a ready-made subject and baseline suite. Its Project 8 data
include noisy I/Q time series and electron-energy truth; its LIGO task supplies
chirp-mass regression. This lowers engineering cost but does not automatically
provide the intervention families required by Claim 1. Treat it as a pilot
source or robustness benchmark, and audit upstream provenance/licences of each
component before redistribution.

## 9. Minimal experiments that serve the story

### Experiment A: controlled attribution

Train one compact waveform subject on a NuRadio reference configuration. Freeze
it. Compare `generic_rich`, `intermediate_only`, and `full_intermediate` with the
same operational features and capacity controls. Evaluate N versus supported S
on matched hard cells. Hold out one N family and one S family for abstention.

This directly tests whether intermediate representations contain incremental
origin information. Easy inclusive classification remains supporting evidence.

### Experiment B: harm ranking

For every held-out cell, compute an independent task loss K and both fixed harm
scores. Estimate paired delta AUROC with the intervention family/cell as the
outer resampling unit and events/model seeds inside it. Include low-alarm harmful
cells.

This tests the actual scientific claim. A high absolute AUROC without improvement
over the generic score does not support it.

### Experiment C: real-noise transfer

Repeat the same score definitions on PyCBC injections into held-out GWOSC time
segments. Preserve time-segment grouping. Change only the adapter and the
physical definition of K; do not retune the story after seeing the result.

## 10. Stop/go sequence

1. **One-day metadata audit:** pin candidate versions, licences, citations,
   schemas, example detector files, and data sizes.
2. **Three-day NuRadio pilot:** four cells, 1,000--5,000 events, export clean/noisy
   traces and independent reconstruction K.
3. **Hard-contrast feasibility:** verify generic-score overlap and enough
   independent cells for an interval. Stop if separation is trivial or outer
   sample size is too small.
4. **One-week protocol adapter:** implement provenance-carrying features,
   disjoint splits, arms, abstention, and paired C1/C2 outputs.
5. **GWOSC pilot:** two held-out time blocks per detector, injections matched in
   network SNR, one calibration N family, one nonstationary/glitch N family, and
   one supported S family.
6. Choose the transfer arm before large training. The decision uses feasibility,
   independent K, and hard-contrast quality, not whichever pilot gives the most
   favourable result.

Do not download the full TIDMAD, Majorana, CRESST, or GWOSC releases until the
metadata/pilot gate establishes the exact subset and integrity manifest.

## 11. Upstream sources checked

- NuRadioMC/NuRadioReco repository and licence:
  https://github.com/nu-radio/NuRadioMC
- NuRadio simulation/detector configuration:
  https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/Manuals/simulation_configuration.html
- NuRadio event generation and output structure:
  https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/Manuals/event_generation.html
  and https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html
- NuRadio likelihood reconstruction:
  https://nu-radio.github.io/NuRadioMC/NuRadioReco/pages/how_to/likelihood_reconstruction.html
- GWOSC API, tutorials, and supported analysis software:
  https://gwosc.org/api/ , https://gwosc.org/tutorials/ ,
  https://gwosc.org/software/
- PyCBC: https://github.com/gwastro/pycbc
- TIDMAD code/licence and data mirror:
  https://github.com/jessicafry/TIDMAD and
  https://huggingface.co/datasets/jessicafry/TIDMAD
- SolidStateDetectors.jl and LegendGeSim:
  https://github.com/JuliaPhysics/SolidStateDetectors.jl and
  https://legend-exp.github.io/LegendGeSim.jl/stable/
- Majorana AI/ML release: https://zenodo.org/records/8257027
- CRESST pulse-shape release description: https://arxiv.org/abs/2508.03078
- PhyTS code and data: https://github.com/kyoon-mit/PhyTS and
  https://huggingface.co/datasets/PhyTS-team/PhyTS-bench
- LUCiD status: https://github.com/CIDeR-ML/LUCiD
- HeST status: https://github.com/spice-herald/HeST
