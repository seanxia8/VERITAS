### 3.8 A10 (mixed event domains: photon, muon and electron bombs; author instruction 22 Oct 2026 — DRAFT, NOT IN FORCE)

**Status.** Draft amendment for registration. It is **not in force**: no grid, dev cohort or
outcome-generating run may use it until the orchestrator commits this section unchanged as
`jepa: register runbook amendment A10 (author revision)`. Until then the registered §3.5/§3.7 text
is authoritative. The proposal and its options are in
[GATE_A10_MIXED_DOMAINS_2026-10-22.md](GATE_A10_MIXED_DOMAINS_2026-10-22.md).

**Authority.** Junjie (21 Oct 2026) proposed adding μ/e particle bombs because the isotropic photon
bomb only tests ToF, while Cherenkov adds directionality and particle physics; he leans toward
mixing the datasets. The author (22 Oct 2026) accepted option A and required **both μ⁻ and e⁻**.
Junjie's message is design input, not approval of the mixing ratio, generator mode or criteria.

**A10.1 Event domains (data factor, not a method).** E2 keeps one method — SIGReg-JEPA plus the
noise anchor, one encoder and one probe protocol (§3.5.5–§3.5.6) — and adds two event domains
beside the isotropic photon bomb:
- `photon`: unchanged isotropic point source (§3.5.3); truth vertex and log yield.
- `muon`: μ⁻ track; Cherenkov emission along the track, `cos θ_C = 1/(β n)`; truth vertex,
  direction, particle type, initial energy, deposited energy and containment.
- `electron`: e⁻ shower; Cherenkov emission from the shower axis/profile; the same truth fields.
A domain is a data-domain factor. The photon bomb is a simplified isotropic domain, not a literal
subset of a Cherenkov event; per-domain results are reported, pooled results secondary.

**A10.2 Run structure and mixing.** Each run carries one detector state, shared by every domain in
that run. Default training-run composition is **334 photon + 333 muon + 333 electron per 1000
events (≈1:1:1)**; probe, robustness and dark-only runs are 17 + 17 + 16 per 50 events. The class
label is metadata only and never a model input; batches are class-balanced by construction, with
optional exact per-batch stratification if the development domain classifier shows imbalance.
A **2:1:1** photon:μ:e fallback is allowed only through the budget fallback in A10.6 and is
recorded. The ratio is a pre-registered parameter; changing it needs a further amendment.

**A10.3 Generator.** Two modes, one label per manifest:
- `photonsim`: PhotonSim/Geant4 μ⁻ (and e⁻, if supplied) events transported through the pinned WCTE
  geometry. Existing μ⁻ samples: 500/1000/1500 MeV, 20 events each. No e⁻ generator exists in this
  repository at the time of drafting.
- `analytic_cherenkov`: a fast parameterised track/shower Cherenkov source in the JEPA generator,
  using the pinned geometry, attenuation and per-photon readout; labelled `simplified_cherenkov`.
**Default: hybrid.** Production uses `analytic_cherenkov`; a mandatory validation subset of real
PhotonSim μ⁻ events must reproduce the hit-count, charge and time distributions within pre-declared
tolerances before the grid. Electron production is analytic until Junjie supplies PhotonSim e⁻
samples. Modes are never silently mixed; a combined dataset records both.

**A10.4 Particle truth and probes.** Per-domain probes:
- photon: vertex, log yield;
- muon: vertex, direction, deposited energy;
- electron: vertex, direction, deposited energy, particle type;
- nuisance in every domain: module gain (F1), per-PMT gain (F3), attenuation (F2), age (law T).
Run-mean is primary; the propagated probe is used for R6; the raw-input ceiling and per-domain
random-init floor are reported. Energy probes use the contained subset (deposited energy inside the
instrumented volume; escaping events are diagnostic); direction and vertex probes use all events.
Particle-type probes use the μ and e domains jointly. A domain classifier on frozen embeddings
reports domain AUC as a diagnostic, never as a criterion.

**A10.5 Predictions.** R6 and R7 are evaluated per domain. The **particle domain (μ + e) is the
primary physics-retention domain**; the photon domain is the isotropic control; the pooled result is
secondary. R6 and R7 criteria are otherwise unchanged (§3.5.9). Optional **R13**: in the particle
domain, direction and deposited-energy probes exceed their random-init floors by more than
2 × the seed floor. R11/R12 may be extended to the particle domains by a later amendment; R1–R5 and
B1 are unchanged.

**A10.6 Detectability, budget and fallback.** The §3.5.3a detectability check is re-run on the mixed
sample, per domain, and frozen before training. A domain whose F1 run-level ceiling is < 0.6 is
excluded from R6/R7; if no domain passes, G6 for the dependent predictions. F3/F2/T ceilings are
re-recorded. Training volume is unchanged if mixed replaces photon-only (same rung-1 run count);
generation, storage and per-domain analysis increase and are re-projected against the 160 h start
rule and 170 h cap. Fallbacks, in order: (1) 2:1:1 mixing; (2) electron domain deferred; (3) E3/E4
dropped; (4) G2. Each step is recorded.

**A10.7 What A10 does not change.** E1, R1–R5, B1, the resource guard and shared-server rules, the
no-touch packages, the class-label-free input contract, the one-method framing, and the registered
arms/objectives of §3.5.6. The photon bomb is not reclassified as N or S; domains are a data factor.

**A10.8 Author decisions (22 Oct 2026; registration slots filled).**
- **Generator: hybrid.** Analytic Cherenkov production for the grid; a mandatory PhotonSim μ⁻
  validation subset; electron analytic until Junjie supplies PhotonSim e⁻ samples.
- **Electron source: analytic first.** Replacing it with full PhotonSim e⁻ needs a further amendment.
- **Mixing: 1:1:1** photon:μ:e (334/333/333 per 1000-event training run; 17/17/16 per 50-event
  probe run). The 2:1:1 ratio is only the recorded budget fallback of A10.6.
- **Primary domain: particle (μ + e) primary for R6/R7**; the photon domain is the isotropic
  control; the pooled result is secondary.
- **Remaining values (not design slots):** particle energy ranges/topology (defaults: μ⁻
  300–1500 MeV, e⁻ 100–1000 MeV log-uniform; energy probes on the contained subset) and the drift
  values s, s_p and a (pending Junjie). They do not change the domain or mixing design.
