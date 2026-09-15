# ORACLE experiment design — the canonical document

_10 September 2026, branch `dev`. **This is the one design document.** It merges
the agreed three-tier design (31 Aug), the arm-level plan (5 Sep) and the
controlled-variable latent-monitoring plan (5 Sep, verified 6 Sep) into one
file, and adds the four linear representation classes of Paper 1 as a tentative
plan for the linear subjects (Part IV). The three merged files are kept
verbatim in `docs/archive/` as evidence; their section numbers survive here as
§I.n, §II.n, §III.n so that every existing citation still resolves (Appendix B).
When something changes, edit this file and cite the evidence — do not start
another plan._

**Companions (live):** `TESTBEDS.md` — which simulations and real datasets, what
each controls, what is implemented, and what the *nonlinear* autoencoder should
look like (§2 there is the target for the transformer subject; §IV here is the
linear subjects). `IMPLEMENTATION_PLAN.md` — work packages, interfaces,
acceptance criteria, gates (31 Aug, rev. 2; its WP numbering is still the one
used by `TODO.md`). `RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md` — the table of
§III.1 verified on the linear subject. `REVIEW_PROMPTS.md` — reviewer prompts.
`TODO.md` — open items.

**Archived (evidence, cited not restated):** `docs/archive/EXPERIMENT_PLAN_ARMS_2026-09-05.md`,
`docs/archive/LATENT_MONITORING_PLAN_2026-09-05.md`,
`docs/archive/SIM_TESTBED_SURVEY_2026-09-05.md`, `docs/archive/DEV_UPDATE_2026-09-03.md`,
`docs/archive/THEME_ADJUSTMENT_2026-09-03.md`, `docs/archive/NOVELTY_CHECK_2026-09-03.md`,
`docs/archive/REVIEW_PROMPT_THEME_2026-09-03.md`, and the earlier
`DATASET_PRODUCTION_PLAN.md`, `OPEN_DECISIONS.md` D1–D8, `NOVELTY_REVIEW.md`,
`PAPER3_AUDIT.md`, `REVISION_PLAN.md`.

---

## Contents

- **Part I — The study** (from `EXPERIMENT_DESIGN.md`, 31 Aug): story, tiers, claims, sequencing, release posture, models versus diagnostics.
- **Part II — The arms** (from the arms plan, 5 Sep): why LUCiD, HeST, TIDMAD; how each is driven; what each may prove; the `noise_module_lucid` build; gates and descope.
- **Part III — The controlled-variable protocol** (from the latent-monitoring plan, 5 Sep, corrected by the 6 Sep results): the factor → determinant → signature → adjustment table; subject architecture; projectors, statistics and the lookup; adjustments; per-arm cells; LUCiD integration; the HeST fork; work packages; risks.
- **Part IV — The linear subject classes: tentative plan** (new, 10 Sep): OF → CW-PCA/EMPCA → tied linear AE → NFPA, their latent identifiability, how they slot into the protocol, and the experiments proposed.
- **Appendix A** — programme status and document map. **Appendix B** — section mapping for old citations.

---

# Part I — The study

## I.1 The story in one paragraph

The paper's claim is that a monitoring alarm on a frozen reconstruction model
can be made to *rank scientific consequence* — and that whether it does is
governed by the metric it is computed in: an unweighted representation
displacement is the wrong-metric statistic, and the consequence-relevant
quantity is displacement in the Σ⁻¹-whitened metric pulled back through the
output Jacobian. No public dataset can test this, because none has a known
assumed-versus-realized covariance, and none is event-paired across the factors
being varied. So the study produces its own data — in a controlled simulator
and in two physics simulations with independently written upstream physics —
and uses a public real-data benchmark as the external check.

The 3 Sep theme adjustment (archived) sits on top of this: Paper 3 / ORACLE is
the main paper — *what determines which representation a detector model learns,
and what it lets a physicist reconstruct* — and Paper 1 is the leading service
paper supplying the theory (the Σ⁻¹ metric, the class 𝓕, the excited support
T_S), the instruments (the four linear classes of Part IV) and the controlled
evidence. The two share the mechanism section; ORACLE owns Σ̂ ≠ Σ and support
shift, Paper 1 owns the (Σ⁻¹, 𝓕) pair — "same levers, opposite side of the
freeze".

## I.2 The three tiers

**Tier 1 — ORACLE-Cov (mechanism).** Controlled waveforms from
`src/noise_module/`. The encoder is trained under an explicit assumed
covariance Σ̂ (the inverse-PSD-weighted objective makes the assumption
literal) and deployed under a realized Σ that `MultiChannelNoiseGenerator`
reports exactly. Strata: matched (Σ̂ = Σ); a κ(Σ̂⁻¹Σ) sweep; output-null
perturbations (large alarm, ≈zero consequence); norm-matched output-aligned
perturbations (same alarm, large consequence); random; clean. Everything
theoretical is proved here: the whitening result, the designed C4 dissociation
with its predicted sign, and the D6 power analysis that sizes every other arm.
**Status:** the §III.1 table is verified on the linear subject (13/14 cells,
`RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md`).

**Tier 2 — realism, now three arms (Part II).** The 31 Aug design had one
realism arm, a Prometheus production (ORACLE-Paired). The 5 Sep plan kept the
tier and retargeted it, because Prometheus emits photon arrival times, not a
sampled trace, so the covariance claim cannot be tested there at all:

- **Arm A — LUCiD** (water Cherenkov, waveform-level, *licence-gated*): the
  transfer of the κ prediction into a real detector geometry.
- **Arm B — HeST → `qp_simulator` → `noise_module`** (superfluid-⁴He dark
  matter, TES readout): the designed granularity dissociation 24 → 1 channel
  and a second readout physics. **Built** (`src/herald_simulation/`).
- **Prometheus / DynEdge** is *not cancelled*: it is the only arm with a frozen
  model nobody in this project trained and a physics consequence variable
  (angular error), and it is the fallback if LUCiD's licence does not land. It
  keeps the ORACLE-Paired production as described in the 31 Aug design — one
  seeded LeptonInjector run reused across ORCA and ARCA, our own NuBench §3.2
  detector response, N1–N5 / S1–S5 / U1–U4 families, content-matched clean twins
  (`prometheus_simulation.matching`), `injection_id` as the pairing key. It
  cannot carry the Σ̂-versus-Σ claim.

**Tier 3 — TIDMAD (real data).** Already in hand: the same compact transformer
trained twice, MSE versus inverse-PSD-weighted — two different Σ̂ choices on
identical real electronics noise. One paragraph of the paper; the paragraph
that blocks "it only works in your simulator". Consequence per D2's three-tier
K (`K_rel` as the confirmatory variable). Adding arm B makes TIDMAD *more*
load-bearing, not less (§II.2).

**The bridge that makes it one study, not three demos:** after Tier 1,
pre-register which monitor families will fail on which perturbation families
of the realism arms — then run them once. A mechanism that predicts
out-of-sample failures in a testbed it never saw is the referee-proof form of
every claim.

## I.3 Claims × tiers and arms

One claim ladder, the proposal's (`IMPLEMENTATION_PLAN.md` WP0), with the
arms of Part II filled in. **The bolded row is the paper.** Read left to right
it is: proved, transferred to a geometry, transferred to a second readout
physics, survived real noise.

| claim | Tier 1 ORACLE-Cov | A · LUCiD | B · HeST | C · TIDMAD | Prometheus |
|---|---|---|---|---|---|
| C1 detection @1% FAR | controlled families, power sizing (cov_D) | waveform-level, in geometry | second readout physics | — | E1 on frozen DynEdge, matched cells |
| C2 attribution N vs matched-clean (primary), N vs S (secondary), **with abstention on U** | cov_C: full control, held-out U | material-axis S; U held out; geometry transfer of the layer profile | ER/NR S; WIMP U | — | E2: matched cells, U1–U4 (E5) |
| C3 cost (capture vs sample/sketch) | cov_C on the compact transformer | — | — | — | E3 on DynEdge |
| **C4 alarm ranks consequence, conditional on alarm** | **designed**: output-null vs output-aligned — K prediction registered; κ sweep (cov_A, cov_B) | **transfer test of the registered κ prediction** | **designed granularity dissociation, 24 → 1** | **external validity**: `K_rel` across the two Σ̂ trainings (T1) | **observational**: E4, conditional-on-alarm AUROC on angular error within severity strata |
| C5 stage localization, by activation patching only | cov_E | replication | replication | — | replication on E2's consequential cells |

The 3 Sep theme adjustment added **C0** (physical-variable organisation of the
latent) to the proposal; Part IV is where C0 gets its linear-subject
instruments. Junjie must see and agree the new title and C0 before anything is
frozen (`TODO.md`).

## I.4 Sequencing

1. Tier 1 now (includes the D6 null simulation — the only place it can run).
   The linear-subject table is done; the *trained* transformer table is the
   next experiment (§III.8).
2. Realism arms in the order of §II.8: arm B pilot and build (done) →
   `noise_module_lucid` only after gate A0 → arm A → arm C (WP10) → bridge →
   confirmatory. Tier 2 Prometheus *production* is not gated by anything; its
   *confirmatory run* is gated by D5 (the 0.944° re-inference offset is a
   confound on K until the CPU-vs-GPU test resolves it or it is shown stable
   across severity strata, `IMPLEMENTATION_PLAN.md` §7.2). Quoted NuBench
   numbers and any email to the authors wait for both halves of D5.
3. Everything else — Panda, LIGO, MicroBooNE, JUNO, FASER, and the reserve
   candidates in `TESTBEDS.md` §§3–4 — is one sentence of future work each.
   Ten testbeds against a diagnostics layer that is still mostly unwritten is
   the project's largest schedule risk. **Three is the number.**

## I.5 Release posture

ORACLE-Paired goes to Hugging Face with generation config and seed manifest —
licence-clean (Prometheus LGPL-2.1, our seeds). **ORACLE-Cov does not go out
before the preprint**: publishing its dataset plus generation scripts is
publishing `src/noise_module/`, whose authorship position was only just
established. Release together with the arXiv submission. The same rule covers
arm B's dataset (HeST constants must be `from_paper`, gate B0 resolved) and
arm A's (gate A0). A `noise_module_lucid` package that *depends on*
`noise_module` can be released independently later; a fork cannot (§II.9).

## I.6 Models versus diagnostics — what exists, what we build

The **subject models** — the frozen networks being monitored — all exist: the
compact two-stage transformer in `src/tidmad_transformer/` and
`src/reconstruction_model/` for Tiers 1 and 3 (retrained per assumed
covariance Σ̂; that retraining *is* the experiment), the analytic linear
subject `src/latent_monitor/linear_subject.py` (and, per Part IV, the four
linear classes in `src/latent_monitor/estimators/`), and DynEdge, ParticleNeT,
GRIT and DeepIce from GraphNeT for the Prometheus arm. Nothing to download or
invent; `TESTBEDS.md` §2 says what the transformer subject must expose.

The **diagnostics are protocol code, not models**, and most of it is standard
machinery to be reused and cited. The three bold rows are small in code and are
the paper; build them first, on Tier 1, because the pre-registered predictions
for the realism arms come from them.

| Component | Status | Tiers |
|---|---|---|
| Representation hooks | `latent_monitor.subject` — six hooks `whitened → channel → token → z → pre_output → output`; DynEdge hook points settled (D4) | 1, 2, 3 |
| Standardized displacement, k-NN retention, principal angles | `latent_monitor.statistics`, `estimators.principal_angles`; parts in `scripts/nubench/` | 1, 2, 3 |
| **Σ⁻¹-whitened displacement** | `latent_monitor.whitening.KroneckerWhitener` + `statistics` — trivial once Σ is known | **1, A, B** (needs Σ) |
| **Jacobian-projected displacement** | `latent_monitor.reference` (P_out/P_null from J_o) — the one monitor computable *without* knowing Σ | all |
| **Output-null / output-aligned perturbation generators** | `latent_monitor.designed` — exact for a linear decoder; SVD of the local Jacobian otherwise | 1 (designed C4), realism arms (replication) |
| Baselines: corrected univariate KS, RBF-MMD, classifier two-sample test, embedding mean/covariance distance, output and uncertainty tests | **reuse** — `alibi-detect`; cite, do not reimplement | all |
| Five-arm attribution classifier (input / output+uncertainty / final embedding / all-generic / full layerwise) | reuse — scikit-learn regularized logistic regression, identical splits | all |
| Conformal abstention, risk–coverage AUC | Mahalanobis in the reference-cell z-metric + Fisher-rank (§III.3.4); `MAPIE` or ~50 lines of split conformal for the baseline | 1, 2 |
| Content-matched clean cells (audit P1.1) | **done** — `prometheus_simulation.matching`; exact pairing in arms A/B makes it unnecessary there | Prometheus |
| Consequence variable K | angular error (Prometheus); `K_rel` via TIDMAD's upstream Brazil-band code (D2); whitened reconstruction error on the trace (Tier 1, arm B, declared); arm A's is an open decision (§II.9) | per arm |
| Activation-patching causal check (before any repair claim) | `latent_monitor.adjust.activation_patch` / `damage_patch` | 1, 2 |
| LoRA repair | reuse — `peft`; on the linear subject `refit_stage` | 1, 2 |
| D6 power analysis / null simulation | implement — scripts on ORACLE-Cov; the only place the null is exactly simulable | 1 (sizes every other arm) |

Rough size of the whole layer: 1.5–3k lines, of which about 80% is wiring
standard components. That is consistent with `docs/archive/NOVELTY_REVIEW.md`:
the contribution is the instantiation and the evaluation protocol, not new
monitors.

## I.7 Which datasets, finally

`TESTBEDS.md` is the canonical inventory (simulations that serve and do not,
real datasets that serve and do not, verified 10 Sep). The programme uses:

| | role | why |
|---|---|---|
| **Controlled simulator** (`src/noise_module/`, ORACLE-Cov) | Tier 1 — the headline | the only place Σ̂ and Σ are both known; done on the linear subject |
| **HeST → `qp_simulator` → `noise_module`** (arm B) | realism, dark matter | independently written physics upstream of our noise; exact pairing; 24 → 1 granularity contrast; DELight's target material |
| **LUCiD + `noise_module`** (arm A) | realism, neutrino, *gated on A0* | the only water-Cherenkov package emitting a dense per-channel trace; geometry is a JSON file; exact pairing |
| **Prometheus / NuBench**, via our own production, *not* the released tarballs | frozen public model; angular-error consequence; fallback for A | pairing, matched cells, DynEdge |
| **TIDMAD** (arm C) | real data | two Σ̂ trainings on identical real electronics noise; already in hand |
| CRESST, GWOSC/Gravity Spy, Majorana, NuRadioMC, SSD.jl | reserve | `TESTBEDS.md` §§3.1, 4.1 — CRESST first if a second real arm is ever added; LIGO if a *published* Σ̂ with labelled Σ̂ ≠ Σ events is required |

---

# Part II — The arms

_From the 5 Sep arms plan. Everything marked ✅ was executed on this machine at
the time and the output is quoted; status notes marked **[10 Sep]** are from
`TESTBEDS.md`._

## II.0 What changed on 5 Sep, and what did not

The 31 Aug design names three testbeds — ORACLE-Cov, Prometheus/ORACLE-Paired,
TIDMAD — and says "three is the number." That judgement stands. The arms plan
did not add arms; it **retargeted two of them**:

| | was | is | why |
|---|---|---|---|
| mechanism | ORACLE-Cov (synthetic waveforms) | unchanged | still the only place the whitening lemma is provable |
| realism | Prometheus (6 geometries, hit-level) | **LUCiD** (water Cherenkov, waveform-level), *conditional on a licence*; Prometheus stays as the fallback and keeps the frozen-public-model role | Prometheus emits photon arrival times, not a sampled trace, so the covariance claim cannot be tested there at all |
| dark matter | *(none)* | **HeST → `qp_simulator` → `noise_module`** | survey §5; two of three links already written — **[10 Sep] built** |
| real data | TIDMAD | unchanged | §II.5 — adding HeST makes it *more* load-bearing, not less |

## II.1 The three arms at a glance

| | **A — LUCiD** | **B — HeST** | **C — TIDMAD** |
|---|---|---|---|
| domain | water Cherenkov ν | superfluid-⁴He DM | axion haloscope, real |
| substrate | simulated PMT waveforms | simulated TES traces | recorded electronics noise |
| what varies | geometry (16 configs), material, particle | geometry (5 builders), ER/NR/WIMP | Σ̂ only (the loss) |
| Σ̂ known | yes (ours) | yes (ours) | yes (the loss) |
| Σ known | yes (ours) | yes (ours) | **no — and that is the point** |
| event pairing | exact (photon dict is an argument) | exact ✅ measured, tested | n/a |
| noise source | `noise_module_lucid` (new preset) | `noise_module` + `qp_simulator` | nobody's — it is real |
| upstream physics author | Tufts/SLAC | SPICE/HeRALD | the ABRACADABRA-style DAQ |
| headline claim it carries | C1, C2, C4 at waveform level in a real geometry | C4 dissociation under a granularity change | external validity of C4 |
| blocker | **no licence** (A0, re-checked open 10 Sep) | licence copyright line (B0, re-checked open 10 Sep) | `K_rel` must survive |
| status **[10 Sep]** | notebook only (`notebooks/one_event_herald_lucid.ipynb` Part B) | **built**: `src/herald_simulation/`, 14 cells, tests green, constants placeholder | code exists, needs data + GPU |
| cost | 8–12 d after licence | done; ~1 d to replace placeholders | 3 d (WP10, exists) |

## II.2 Why three, and why these three

The arms are not three helpings of the same evidence. Each answers a question the
other two cannot.

**The independence argument, which is the reason TIDMAD stays.** Tier 1 (ORACLE-Cov)
and arm B (HeST) both draw their noise from `src/noise_module/`. If that generator
has a systematic quirk — a wrong alias fold in `psd_resampling`, a conditioning
artifact in the Cholesky path at high κ, a spectral preset that happens to flatter
the whitened statistic — it contaminates both arms *identically*. Three synthetic
arms of which two share the instrument under test is not three observations. Arm C
is the only place the noise is real electronics noise that nobody in this project
chose, and therefore the only arm that can catch that class of error. Everywhere
else we set both Σ̂ and Σ; in arm C, Σ̂ varies against a fixed, unchosen Σ. That is
the exact shape of the external-validity claim.

**The domain argument.** A referee's first question about a covariance-geometry
result is "does this depend on your detector?" Arm A is a photon-counting PMT array
at 1 GHz; arm B is a phonon calorimeter at 250 kHz with 1–24 channels; arm C is a
single-channel SQUID readout on real data. Three readout physics, three sampling
rates, three channel counts, one predicted sign.

**The granularity argument.** The abstract axis common to A and B — *detector
granularity at fixed active volume* — is instantiated as `n_sensors` 2000→20000 in
LUCiD and as 24→1 sensors on the identical helium cell in HeST. Same axis, two
domains, opposite ends of the channel-count range. This is what replaces the
(abandoned) idea of giving two experiments the same layout.

## II.3 Arm A — LUCiD (water Cherenkov)

### II.3.1 Why this arm

It is the **only** package in the water-Cherenkov world that emits a dense
per-channel uniformly-sampled time series:
`lucid/simulation/sensor_response.py:389 build_make_hits_waveform(n_photons,
window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=1.0, smear_time=True,
smear_charge=True)` returns `(num_detectors, n_time_bins)` by `segment_sum` of
per-photon smeared charge into 1 ns bins — described in-code as the "1 GHz FADC
convention". WCSim, the standard, has no electronics layer at all (a digit is
`map<int,double> pe/time`; open issue #13 confirms it was never written), and
Prometheus stops at photon arrival times.

It also has no covariance concept anywhere — the only `covariance` in the package is
over fit parameters (`lucid/fitting/fisher.py`, whose weight matrix is literally
`diag`). So `noise_module` fills a hole rather than competing with an incumbent
model whose assumptions we would have to defend.

Two further properties earn their keep: geometry is a four-key JSON file and the
event is a **photon dict passed as an argument** (so pairing is exact, not replayed —
`analysis/paper/fig_charge_displays.py` already loops three geometry families over
one event and one PRNG key), and the whole forward model is JAX, so a simulator-side
Jacobian is available by autodiff for the output-null construction.

### II.3.2 The blocker, and the fallback

**LUCiD has no licence.** No `LICENSE*` in the tree, no `license` field in
`pyproject.toml`; `README.md` says "The license is being finalized and will be added
shortly" and `CONTRIBUTING.md:67-68` says contributions will be licensed "once it is
finalized". That is all-rights-reserved today, and the release posture (§I.5) makes
it a hard gate, not a footnote. **[10 Sep] re-checked: still no licence; last
upstream commit 2026-08-21.**

**Action:** one email to Kensuke Terao and Omar Alterkait asking them to add MIT or
BSD-3. They maintain a `CITATION.cff` and a `CONTRIBUTING.md`, so it is almost
certainly an oversight. Budget 1–3 weeks.

**Gate A0.** No LUCiD work starts before a permissive licence is in the repository.
If it has not landed by M3, arm A reverts to Prometheus at hit level, the
covariance-geometry claim is carried by arms B and C alone, and the paper says so.
Until then the LUCiD material stays in the notebook — deliberately no
`src/lucid_simulation` package (`TESTBEDS.md` §1.1).

### II.3.3 How it is driven

```
config/<NAME>_geom_config.json           <- geometry, 4 keys, metres
config/<NAME>_physics_config.json        <- optics, QE, TTS, gain, t0, per-PMT arrays
PhotonSim ROOT file (one, fixed)         <- the physical event
        |
setup_event_simulator(geom, n_photons, is_data=True,
                      hit_mode='waveform', window_ns=..., bin_width_ns=...)
        |
        v
charge_waveform : (n_sensors, n_bins)   [photoelectron charge per bin]
        |
   noise_module_lucid  (§II.7)
        v
voltage_waveform : (n_sensors, n_bins)  [mV, with Sigma_hat / Sigma reported]
```

**Geometry axis.** `analysis/paper/utils/make_geometries.py` already deep-copies
`SK_like_geom_config.json` and overrides only `geometry_definitions.n_sensors` for
2000…20000 in steps of 1000. Reuse it verbatim. Two named cells for the paired claim
(`SK_like` at 11000 and at 4000, say) plus the scan for the trend.

**Physics axis (free).** `SK_like` vs `SK_like_wbls`, `JUNO` vs `JUNO_wbls` — same
geometry, different *material* physics config. That is an S-family generator that
costs nothing. Particle type is a setup-time SIREN string and only `muon` and
`electron` emitters ship, so particle-type S-families come from the PhotonSim path
(`GeV/01_mu`, `03_e`, `05_pi0`, `06_pbomb`), not the differentiable one.

**Traps, all verified:**

- `apply_translation` defaults `true` and draws the vertex from *the detector's own*
  fiducial volume (`lucid/sources/writer.py:323-370`). **Set it `false`** or supply
  the translation, or differently-sized geometries get different vertices at the same
  seed. Same-size scans are unaffected.
- `n_sensors` is a target, not a guarantee: SK_like asks 11000 and places 10764.
  Record the placed count, never the requested one.
- `Cylinder.configure_grid(..., max_candidates_per_ray=4)` **silently drops sensors**
  if the grid is too coarse. Assert placed-sensor count against a reference run.
- `IPython` is an undeclared dependency (`lucid/siren/training/monitor.py:16`,
  imported at module scope via `simulator.py`). `pip install ipython`.
- The soft-overlap lookup table (`lucid/overlap.py:255-365`, defaults
  `n_theta=n_rho=2000`) attempts a **9.5 GB** allocation on first use per sensor
  radius. Budget ≥12 GB RAM, or run `temperature=None` (hard-step mode — what the
  notebook does).
- The GENIE example config the docs reference (`GeV/13_genie_numu_nue.json`) is not
  in the tree. GENIE support is real in code, but unexercised by any shipped example.

### II.3.4 What arm A is allowed to prove

| claim | arm A's contribution | why it is arm A and not another |
|---|---|---|
| **C1** detection @1% FAR | on waveforms, in a real detector geometry, across a granularity change | the only arm with both a real geometry and a sampled trace |
| **C2** attribution N vs S, abstention on U | N from `noise_module_lucid` knobs + LUCiD's own acquisition knobs (QE, dark rate, TTS); S from the material axis and PhotonSim particle types; U held out | the material axis gives an S-family that is unambiguously *physics*, not acquisition |
| **C4** alarm ranks consequence | **the transfer test**: the κ-sweep prediction registered on Tier 1 is re-run here, in a geometry, with a different readout physics | this is the whole point of the arm — Tier 1 proves it, arm A shows it is not an artifact of the synthetic substrate |
| **C5** stage localization | replication only, by activation patching | no new mechanism |

Arm A does **not** carry: the whitening lemma itself (Tier 1), the frozen-public-model
claim (Prometheus/DynEdge), or a physics consequence variable in the NuBench sense —
its consequence variable is reconstruction error from LUCiD's own fitting layer
(`lucid/fitting/recon.py`), which we would be defining, so it must be declared as
such and not dressed up as an external metric (open question, §II.9).

## II.4 Arm B — HeST (dark matter)

### II.4.1 Why this arm

Because **it stops exactly where our code starts.** ✅ Verified: `HestSignal` has two
fields and nothing else — `['energies', 'arrivalTimes']`, each a list of per-sensor
lists. No digitizer, no trace, no electronics, no noise model.
`src/qp_simulator/QPSimulator.py` consumes per-sensor arrival times. Σ̂ and Σ are
ours by construction because nothing else sits between the quasiparticle list and
the sampled trace.

And because the physics upstream is **not ours**. Tier 1 is entirely self-authored;
if the dark-matter arm were too, a referee can say we built the world we tested in.
HeST puts independently written, published detector physics (PRD; arXiv:2307.11877)
upstream of our noise. That is what makes arm B a different kind of evidence from
Tier 1 rather than a second helping of it.

Third, superfluid ⁴He is DELight's own target material, which converts the arm from a
benchmark into a DELight-relevant result — worth something to the authorship position
recorded in `claude/noise-module-ip-and-nubench-reproduction.md`.

### II.4.2 What HeST simulates

Two stages. Not Geant4, no photon transport, no background model — the energy
deposit is an **input**, not an output.

**Yields.** `HeST/core/HeST_Core.py:GetQuanta(energy_eV, interaction)` partitions a
deposit four ways. ✅ Measured at 1 keV:

| | quasiparticles | IR photons | singlet UV | triplet |
|---|---:|---:|---:|---:|
| `"NR"` | 900,238 | 7 | 20 | 1 |
| `"ER"` | 419,925 | 116 | 18 | 11 |

The 2.1× quasiparticle ratio at equal deposit is the ER/NR handle — a one-argument
switch. `core/WIMP_Generation.py` gives `WIMP_spectrum(mass_MeV, ...)` for a recoil
spectrum instead of a fixed energy. **Consequence (`TESTBEDS.md` §1.2): ER/NR is
degenerate with energy in a quasiparticle-only readout**, so "signal type" in arm B
is a fixed-energy contrast unless a photon channel is also transported to a trace.
`events.quanta` records all four yields; `qp_simulator` transports only the
quasiparticles. Adding a photon channel is the one physics extension that would
make type recognition non-degenerate — an open decision.

**Transport.** `core/Detection.py:2078 GetEvaporationSignal(detector, QPs, X, Y, Z,
useMap=False, T=2.0)` launches `QPs` quasiparticles isotropically from a vertex in
cm, ray-traces them against the cell surfaces with per-surface reflection / diffuse /
Andreev probabilities, and records those that evaporate a helium atom onto a sensor.

### II.4.3 Geometry: Python, not a config file — and that is better here

A `VDetector` is five boolean implicit-surface functions plus a list of `VSensor`s
plus the surface-interaction probabilities. The five shipped builders in
`core/Geometry.py` are ~60-line functions; `HeRALD_v1`'s array is a literal
(`sqcm_width = 1; sqcm_pitch = 1.1; cell_rad = 3.5` cm, a 6×6 `array_map` with 24
ones). So a geometry sweep is a factory over `(cell_rad, fill_height, sqcm_pitch,
array_map)` — `herald_simulation.geometry.make_cell`. ✅ Inventory: `HeRALD_v1` 24 ·
`HeRALD_v1_monolithic` 1 · `HeRALD_UMass_splitCPD` 2 · `HeRALD_UMass_monolithic` 1 ·
`HeRALD_LBNL` top+bottom.

### II.4.4 ⭐ Pairing is exact — measured, tested

`QP_propagation` (`core/Detection.py:1171`) samples the **entire** initial population
up front and vectorised — `generate_random_direction(nQPs)` at `:1244`,
`Random_QPmomentum(nQPs, T=T)` at `:1258` — *before any geometry is touched*.
✅ 20,000 QP from (0, 0, 2.0) cm, seed 42: `v1_28` 24 sensors detected 131; `mono`
1 sensor detected 174; `umass_split` 2 sensors detected 281; initial QP momenta
identical across all three geometries. Same event, three geometries, three observed
signals, no replay machinery. `herald_simulation.events` seeds from `event_id` so the
contract is enforced by construction, and `tests/test_pairing.py` asserts it.

### II.4.5 Cost and hazards

✅ Throughput `useMap=False`, one core, post-JIT: 2000 QP → 0.03 s; 10000 QP → 0.19 s,
~0.8 % evaporation efficiency. A 1 keV NR event (900k QP) is **~17 s single-core**
yielding ~7000 detected quasiparticles; `--qp-fraction` thins for development and
records the rescaling. Embarrassingly parallel.

- **Maps are not shipped** (`get_QPEmap()` returns `0.0`). Irrelevant: `useMap=False`
  is full first-principles propagation.
- **Install:** `setup.py` requires `detprocess` → `annoy` → `aplus`, which fails to
  build; `pip install qetpy numba` is enough.
- **Licence:** MIT *text*, but the copyright line is the unedited PyPA sample.
  **Gate B0:** ask Greg Rischbieter (rischbie@umich.edu) to name the real holder before
  a released dataset depends on it. **[10 Sep] re-checked open; last upstream commit
  2026-03-09.**
- **Bus factor:** 89 commits, one maintainer, "early developmental version". Vendor a
  pinned copy once the copyright line is fixed.
- Bugs to report upstream: `VDetector.get_QPEmap()` returns `self.LCEmap_positions`
  (`core/Detection.py:243`); the unbuildable `aplus` dependency.

### II.4.6 The chain (built)

```
GetQuanta(E, "NR"|"ER")          -> n quasiparticles          [HeST]
GetEvaporationSignal(detector,…) -> per-sensor arrival times  [HeST]
QPSimulator.generate(times_ns)   -> clean per-channel trace   [ours]
MultiChannelNoiseGenerator       -> Sigma_hat / Sigma          [ours]
```

Units: HeST reports µs; `QPSimulator` takes ns at 2.5e5 Hz × 16384 samples = 65.5 ms,
against a HeST window of order 5 ms — one conversion, no resampling. Output per cell:
`truth.parquet` + `traces.npy (n, C, N)` + `provenance.json` (HeST commit, geometry
hash and positions, budget with provenance states, trace config).

### II.4.7 What arm B is allowed to prove

| claim | contribution |
|---|---|
| **C4 (headline for this arm)** | the designed dissociation under a *granularity* change: 24 channels → 1 channel on the identical helium cell. With C=1 the multichannel covariance claim is vacuous by construction; with C=24 it is not. A designed contrast with a predicted sign, not an observation. |
| **C1, C2** | replication in a second readout physics; N families from `noise_module` knobs and HeST's surface probabilities, S families from ER/NR and the four yield channels, U from a WIMP spectrum held out |
| **C5** | replication only |

Arm B does **not** carry a physics consequence variable of external provenance. Its
consequence is whitened reconstruction error on the trace — the variable the
2026-09-02 Tier-1 review recommended over the amplitude readout. Declare it. HeST has
no background model; families are designed, not sampled from a background.

## II.5 Arm C — TIDMAD (real data)

### II.5.1 Why it survives, and why arm B makes it *more* necessary

See §II.2. In one line: arms A and B are both simulated and arm B shares its noise
generator with Tier 1, so arm C is the only evidence that the effect is not a
property of `src/noise_module/`. Cost is also decisive: WP10 is **3 days**, depends
only on WP1–WP6 which are needed anyway, and `src/tidmad_transformer/` already exists
with the int8-saturation and uninitialised-tail bugs (C1, C2) fixed on `dev`.

### II.5.2 How it is driven

Unchanged from `IMPLEMENTATION_PLAN.md` WP10: the same ~0.6M-parameter two-stage
transformer trained twice with **identical seeds and data**, differing *only* in the
loss — MSE versus inverse-PSD-weighted. Two Σ̂ on one fixed, unchosen Σ. Monitors on
both; `K_dev` disclosed as non-scientific per D2; `K_rel` on the hash-frozen 20-file
subset with the 328-band truncation and in-band file selection declared. TIDMAD's 208
no-injection science files are the noise-only records.

### II.5.3 What it is allowed to prove, and the exit condition

Arm C proves exactly one thing: **the monitor contrast between the two Σ̂ trainings
has the sign predicted on Tier 1, on noise nobody in this project generated.** It is
one paragraph — the paragraph that blocks "it only works in your simulator".

**Exit condition.** D2 records that TIDMAD's own authors disclaim the denoising score,
so the consequence variable rests entirely on the Brazil-band `K_rel`. If `K_rel`
does not survive scrutiny, arm C degrades to "two Σ̂ trainings, monitor contrast, no
consequence variable" — still worth the paragraph as an external check on the
monitor, but no longer claim-bearing. The replacement, if a claim-bearing real-data
arm is required, is LIGO/Gravity Spy (published ASD as Σ̂, labelled glitches as
Σ̂ ≠ Σ) — budget 2–3 weeks of domain entry; CRESST is the DELight-recognisable
alternative (`TESTBEDS.md` §4.1).

**Paper 1 caveat (10 Sep).** The Paper 1 held-out TIDMAD test on 14 files found no
useful subspace recovery (rank-2 ≈ zero predictor; mechanism hypothesised p/n = 187,
a narrowband multi-bin follow-up is planned there). That is Paper 1's *reconstruction*
claim, not ORACLE's *monitor-contrast* claim, but it means arm C must not be described
as showing that the covariance-aware model reconstructs better on TIDMAD — only that
the two Σ̂ trainings differ in the predicted direction.

## II.6 Claims × arms

Merged into §I.3.

## II.7 `noise_module_lucid` — the customised noise module (gated on A0)

### II.7.1 The finding that makes this cheap

**Almost nothing has to be written.** ✅ The spectral layer is already composable and
physics-agnostic, driven from a registry (`spectral_models._COMPONENT_TYPES`) via
`NoiseConfig.components`, so a PMT front-end ASD is **pure configuration** — no new
`SpectralComponent` subclass, no new module: on a 512 ns record (df = 1.953 MHz,
Nyquist 500 MHz) the PSD integral is 1.0000 against the target and the `(64, 512)`
block carries implied *and* realized covariance in its metadata. So
`noise_module_lucid` is a **thin adapter package plus a preset**, not a fork. The
preset `PMT_FRONTEND_V1` and the units bridge already exist *inside the notebook*
(`TESTBEDS.md` §1.1); the package is the same code with tests and provenance.

### II.7.2 ⚠️ 50 Hz mains is not representable at LUCiD's grid

✅ Over a 512 ns window, 50 Hz sits 2.56e-05 of one bin above DC: a DC offset. The
in-band physics between 2 MHz and 500 MHz is different and richer:

| component | `type` | physical origin | in band? |
|---|---|---|---|
| amplifier white floor | `white` | front-end thermal + shot | dominant |
| front-end bandwidth | `rolloff` (lowpass, corner ~250 MHz, order 2) | preamp / cable | yes |
| clock & switching pickup | `line` (e.g. 62.5 MHz, width 4 MHz) | ADC clock, DC-DC converters and harmonics | **yes — the real coherent line source** |
| cable-reflection ringing | `resonance` (centre ~150 MHz, half-width 20 MHz) | impedance mismatch | yes |
| flicker | `powerlaw` (−1, ref 10 MHz) | front-end 1/f | weak, ~2 decades only |

**1/f, drift and mains need a longer record.** `window_ns` is an argument, so raise
it; to reach the kHz decade at 1 ns bins would need ~10⁶ samples, so the practical
route is **decimation** — and `psd_resampling.alias_fold_psd_density` earns its
place, because decimating without an anti-alias filter folds the >Nyquist/2 content
down. That fold is a declarable acquisition-contract term, i.e. a free and physically
honest **N-family**.

### II.7.3 ⚠️ Two measured constraints that would otherwise poison the κ sweep

**(a) The realized-covariance estimator has a floor set by N/C.** ✅ Matched cells
(Σ̂ = Σ by construction), so every value above 1.0 is estimator noise:

| C | N=512 | N=4096 | N=32768 | N=262144 |
|---:|---:|---:|---:|---:|
| 16 | 1.968 | 1.297 | 1.105 | 1.037 |
| 64 | 8.252 | 2.174 | 1.346 | 1.116 |

Rule of thumb: **N/C ≳ 500 for a κ floor below ~1.1.** Consequences, not optional:
use `window_ns` ≥ 16–32 µs for the covariance cells; **do not use all ~10,764 PMTs as
one covariance unit** — take a crate/string/16–64-PMT sub-array (coherent pickup is
per-crate anyway; `string_id` is already in LUCiD's output); report the matched-cell
κ floor alongside every swept κ. (Arm B at C = 24, N = 16 384 has N/C = 683 and a
measured floor of 1.18, as predicted.)

**(b) Channel gains were redrawn on every `generate()` call.** ✅ Verified at the time;
pooling records across calls mixed covariances (5.26 vs 1.12 for one long call).
**WP-N1 — done:** `MultiChannelNoiseGenerator.freeze_channel_structure=True` lets one
covariance span many records; required for one Σ̂ to span a cell.

### II.7.4 The units bridge — the one piece of real physics

LUCiD's waveform bin holds **summed photoelectron charge**, not volts. Convolve with a
per-channel SPE voltage template (`templates.pulse_template_2`, PMT time constants
rise ~1–3 ns, decay ~5–10 ns) to mV, then add noise with PSD in mV²/Hz. Calibration
constant mV per photoelectron from LUCiD's own `gain` so the two layers cannot
silently disagree. ~40 lines (`spe_template`, `to_mv` in the notebook).

### II.7.5 Proposed layout

```
src/noise_module_lucid/           # new package, MIT, depends on noise_module
  presets.py        PMT_FRONTEND_V1 : NoiseConfig components + provenance record
  units.py          spe_template(fs, tau_rise, tau_decay), charge_to_mV(...)
  adapter.py        add_readout_noise(charge_waveform, detector, cfg, rng)
                       -> (trace_mV, metadata)   Sigma_hat, Sigma, kappa, kappa_floor, grouping
  grouping.py       channel_groups_from_string_id(detector) -> list[np.ndarray]
  interventions.py  N-families: gain drift, dark-rate change, clock-line amplitude,
                    decimation/alias fold, group-coherence change
  tests/
```

Nothing in `src/noise_module/` is forked.

### II.7.6 Acceptance criteria

1. `presets.PMT_FRONTEND_V1` builds a one-sided PSD whose discrete integral equals
   `noise_power` to 1e-12 on the LUCiD grid, every component with a named physical
   origin in the provenance record.
2. **On a matched cell (Σ̂ = Σ), the reported κ equals the measured estimator floor
   for that (C, N) to within its bootstrap interval** — the null is calibrated before
   any mismatch is injected. (Load-bearing.)
3. A swept κ cell reproduces its requested κ to within the matched-cell floor, over
   at least one decade.
4. `validate_csd_ensemble` passes on the generated multichannel block at the LUCiD
   sampling rate.
5. Round-trip: `charge_to_mV` on a unit-charge impulse returns a pulse whose integral
   equals the configured mV·ns per photoelectron.
6. The decimation/alias-fold N-family changes the realized in-band PSD in the
   direction `alias_fold_psd_density` predicts, checked against the closed form.
7. Every generated dataset writes a provenance record naming: LUCiD commit, geom and
   physics config hashes, placed sensor count, `apply_translation`, `window_ns`,
   `bin_width_ns`, the preset version, the grouping, Σ̂, Σ, κ, and the matched-cell
   κ floor.

### II.7.7 Effort

~8 days: units + template 1 · preset + provenance 1 · adapter + grouping 1.5 ·
interventions 2 · tests + κ-floor calibration 2 (WP-N1 already done).

## II.8 Sequencing, gates, descope

**Gates.** A0 — LUCiD licence in the repository. B0 — HeST copyright line names a real
holder. Both are emails (`TODO.md`, Collaboration). Neither blocks Tier 1.

**Order.** Tier 1 → arm B pilot and build (**done**) → `noise_module_lucid` (8 d,
starts only after gate A0) → arm A build → arm C (WP10, 3 d, any time after WP6) →
pre-registration bridge → confirmatory runs.

**Descope order** (replaces `IMPLEMENTATION_PLAN.md` §4's list, which predates arm B):

1. The LUCiD geometry *scan* (keep two named geometries; the trend becomes future work).
2. Arm B's WIMP-spectrum U family (keep ER/NR).
3. cov_E / C5.
4. Prometheus's second geometry — ORCA only.
5. WP10's `K_rel`, keeping the two-Σ̂ monitor contrast.

Arm B and arm C are the last things to cut, for opposite reasons: arm B is the
cheapest claim-bearing arm, and arm C is the only unshared substrate. If the whole of
arm A has to go, the paper still has proved → transferred → survived-real-noise; it
loses "in a real detector geometry", and it must say so.

## II.9 Open questions

- **Does the κ prediction transfer across sampling rate?** Tier 1 runs at the
  athermal-calorimeter rate, arm A at 1 GHz, arm B at 250 kHz. The lemma is
  rate-free, but the *estimator floor* is not — the N/C rule has to be re-measured per
  arm and reported *before* the pre-registration bridge is frozen.
- **What is arm A's consequence variable?** LUCiD's own `recon.py` is ours to define,
  which weakens it relative to Prometheus's angular error. Options: use LUCiD's
  Fisher/CRB machinery (`lucid/fitting/fisher.py`, remembering its documented √12
  "honesty factor"), or keep the physics consequence variable on the Prometheus arm
  and let arm A carry only the waveform-level claims. **Decide before building.**
- **Does arm B transport a photon channel?** Without it ER/NR is a fixed-energy
  contrast (§II.4.2). Decide before the E cells are frozen.
- **Should `noise_module_lucid` live in ORACLE or beside `noise_module`?** A separate
  package that *depends* on it can be released independently later; a fork cannot.

---

# Part III — The controlled-variable protocol

_From the 5 Sep latent-monitoring plan, **corrected by the 6 Sep results** on the
linear subject (`RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md`). The 控制变量
discipline: hold everything fixed, move **one** of three factors — geometry **G**,
noise **Σ**, event type **E** — and watch the frozen model's latent space. Which
architecture, how to drive it, how to tell *which part* of the latent space moved,
what to adjust once you know; then how each arm is used under that protocol._

## III.1 The design in one table

The proposal's §3 says a frozen model's representation is fixed by three
determinants — the **metric** Σ⁻¹ the measurement fixes, the **class** 𝓕 the
architecture fixes, and the **support** T_S the training excites. The
controlled-variable design is that table read backwards: each factor we move is
one determinant, each determinant has a predicted latent signature, and each
signature names its own repair. The table below is the 5 Sep prediction with the
6 Sep corrections written in; the corrections are marked ▸.

| factor moved | determinant | where it shows in the latent space | the adjustment that follows |
|---|---|---|---|
| **Σ** — covariance-type (Σ̂ ≠ Σ: correlation, bandwidth, line, alias fold) | metric Σ⁻¹ | **no mean shift**; the noise-only z-variance ratio, residual PSD or residual channel correlation departs from its calibrated null, scaled by the eigenvalues of Σ̂⁻¹Σ; magnitude tracks κ(Σ̂⁻¹Σ) | re-estimate Σ from noise-only records or residuals, set Σ̂ ← Σ in layer W, and ▸ **re-derive the encoder as the GLS projection onto the same raw signal basis** (the whitening lemma; a bare layer swap multiplied the consequence by ~4–5). No gradient step; decoder and head untouched |
| **Σ** — structural N (gain drift, channel loss) | metric, seen through the channel stage | a **mean** shift in the output-aligned subspace — looks like E in the mean — *and* the noise-only statistics move; layer profile peaks at the per-channel stage | activation-patch stage S1 first; if consequence recovers, LoRA on the per-channel encoder only |
| **Σ** — timing jitter ▸ | metric | ▸ **documented, not a mismatch**: jitter that decorrelates a *shared* cross-channel component is a covariance change in the latent (noise-only variance 0.55, no consistent mean shift) — N by contract, Σ-covariance in the signature | as covariance-type |
| **G** — geometry | class 𝓕, through the geometry embedding | mean shift concentrated at the **geometry-embedding stage**, *small* at the pooled z if the aggregation is geometry-invariant — that smallness is itself the prediction under test | LoRA on the geometry embedding + pooling only; ▸ **on the linear subject a granularity change is a gain on z and repairs through the output head or the channel stage, not the pooling weights** — so the G-row repair is a prediction about nonlinear subjects |
| **E** — supported-but-rare physics (in span: oscillation, double pulse) ▸ | support T_S, inside | ▸ **its own row**: z moves a lot, the residual and every noise-only statistic stay at reference, out-of-span fraction low (0.02–0.17), consequence rises (up to 4.6×) — the representation is intact, the head is not | recalibrate or extend the output head; cheaper than abstaining and different in kind from a support shift |
| **E** — outside training support (glitch, WIMP spectrum) | support T_S, outside | displacement in the **unexcited complement** T_S^⊥; out-of-span fraction high (0.84); Fisher-rank drop | **no weight adjustment is valid** — abstain, then extend the training support with data (outside T_S the representation is fixed by architecture and initialisation, not data — Paper 1 §7.4b) |
| designed output-null family | — | large Euclidean ‖Δz‖, ≈0 output change; ▸ recognised by isotropy (random per-event directions, large per-event alarm, small mean shift) | none — the control that separates alarm from consequence |

▸ **The N-vs-S discriminator is noise-only records.** Every Σ-type cell moves at
least one noise-only statistic off its null; every event-type cell leaves all three
at exactly their reference value (variance ratio 1.000, PSD and channel-correlation
deviation 0.000). A random trigger cannot see physics. The lookup (§III.3.3) is built
on this. ▸ **Patching is flat on the linear subject**: substituting the clean stage
into the perturbed pass recovers 100 % at every stage; the linear subject has no
stage that creates damage, so C5 is a transformer question.

## III.2 Architecture

### III.2.1 The constraint the design imposes

If geometry changes and the model must not be retrained, then the latent space has
to be **the same space** for every geometry. That rules out any encoder whose input
layer is tied to a channel count — a flat MLP or 2-D CNN over `(n_sensors × n_bins)`
gives a *different* z-space per geometry, and "which part of the latent space moved"
is then not a question. The architecture must be a **set encoder over channels, with
geometry as an explicit per-channel feature, pooled to a fixed-dimension z**. DynEdge
and the compact two-stage transformer already are this. **The recommendation is not
a new architecture; it is to make the latent-facing pieces explicit and identical
across arms**, so that every diagnostic in §III.3 is the same code on every arm.

### III.2.2 The shape (minimal form)

```
x  (C channels × N samples)                       raw traces
 │
 ├─ [W]  whitening layer  x̃ = Σ̂^{-1/2} x           Σ̂ is a NAMED PARAMETER, not baked in
 │                                                   (Kronecker Σ̂ = Σ_c ⊗ Circ(S(f)), KroneckerWhitener)
 ├─ [S1] per-channel encoder  h_c = f(x̃_c)          shared weights across channels;
 │        1-D conv or small transformer over time    this is where structural N shows
 ├─ [P]  channel token = [h_c ‖ e(pos_c, group_c)]  geometry embedding e(·): position,
 │                                                    orientation, crate/string id
 ├─ [S2] geometry-aware aggregation                 attention pooling or EdgeConv on
 │        z = pool({token_c})   z ∈ R^d, d fixed     positions; permutation-invariant,
 │                                                    channel-count-invariant
 ├─ [g]  decoder  x̂ = g(z)                          reconstruction head, whitened domain
 │                                                    — makes the inverse-PSD loss literal
 └─ [y]  output head  y = o(z)                      physics targets
```

Training objective: `‖Σ̂^{-1/2}(x − g(z))‖² + λ·L_phys(y)`. **Hook points** (six,
matching D4): `x̃`, `h_c`, post-embedding tokens, `z`, decoder pre-output, `y`.
**Exposed Jacobians**: `J_g = ∂g/∂z` (recon), `J_o = ∂y/∂z` (output) — the `Subject`
interface (`represent`, `outputs`, `jac_recon`, `jac_output`), `latent_monitor.subject`.

**The structured target for the transformer subject** — factor slots
`z_sig = [z_amp | z_pos | z_shape]`, a separate noise branch `z_noise` fed by
whitened residuals and random-trigger records, a structured analysis-by-synthesis
decoder, cross-geometry paired reconstruction and block-wise probe losses — is
specified in `TESTBEDS.md` §2 and is not restated here. It follows from the 6 Sep
findings (the noise-only discriminator; geometry invariance as a prediction) and from
the identifiability argument that disentanglement is identifiable from pairs that
differ in exactly one factor, which is what the one-factor cells provide.

### III.2.3 Analytic baseline first

Paper 1's tied linear autoencoder is the same diagram with linear f, g, o. Every
statement in §III.1 is provable there, and the excited span T_S is exact, not local.
`src/latent_monitor/linear_subject.py` is that subject; Part IV widens it to the four
linear classes. Run the full §III.3 protocol on the linear subject before the
transformer, on every arm. If a signature fails on the linear subject, it is wrong;
if it holds there and fails on the transformer, that is a finding.

### III.2.4 Training protocol — the controlled-variable discipline

Train on the **reference cell only**: (G₀, Σ̂₀ = Σ₀, E₀ ∈ S). Freeze. The model never
sees a perturbed cell. Every perturbed cell is evaluated against its **paired clean
twin** — the same event through the reference cell — so Δz is a **per-event vector**,
not a distributional distance. This is what the pairing machinery buys: per-event Δz
can be projected, per event; a distributional MMD cannot.

## III.3 Recognising which part of the latent space changed

### III.3.1 Four projectors, fitted once on the reference cell

| projector | built from | separates |
|---|---|---|
| **P_out / P_null** | row space of `J_o` (SVD, top-k right singular vectors) vs its orthogonal complement | output-aligned (consequential) vs output-null (harmless) directions of z |
| **P_exc / P_unexc** | eigenvectors of the pullback Fisher `I(z) = J_gᵀ Σ̂⁻¹ J_g` above vs below a rank threshold | excited support T_S vs its complement |
| **Π_ℓ** | per hook ℓ | the layer profile |

`k` and the rank threshold are fixed on the reference cell and pre-registered; they
are not tuned per family (`latent_monitor.reference.fit_reference`).

### III.3.2 The statistics, per cell

For each perturbed cell with paired twins, per event: Δz = z(perturbed) − z(twin).
Then (`latent_monitor.statistics`): the **mean-shift norm** in the null metric and
the **per-event alarm**; **energy splits** `‖P Δz‖² / ‖Δz‖²` for each projector;
the **noise-only statistics** — z-variance ratio along the excited directions,
smoothed and single-bin residual PSD deviation, residual channel-correlation
deviation — computed on random-trigger records; the **out-of-span fraction** of the
paired change; **Fisher rank** at the perturbed cell vs reference; the **layer
profile** `‖Π_ℓ Δh‖` across the six hooks; **consequence** K and its
conditional-on-alarm AUROC against the alarm magnitude (C4); abstention rate.
Thresholds are calibrated once on the reference null (q99) and never re-tuned
(6 Sep: mean-shift 5.44, per-event alarm 4.53, z-variance band 0.85–1.15, smoothed
PSD deviation 0.17, single-bin 0.29, channel-correlation 0.13).

### III.3.3 Attribution is a lookup, not a classifier

The pre-registration bridge (`IMPLEMENTATION_PLAN.md` §5) is the table in §III.1 with
a decision rule per row that two readers score identically. The rule order *is* the
decision procedure (`latent_monitor.lookup.attribute`):

```
if   any noise-only statistic off its null:
        if consistent mean shift in P_out and layer profile peaks at S1  -> Σ structural  -> patch S1, then LoRA S1
        else                                                             -> Σ covariance  -> re-whiten (GLS re-derivation)
elif isotropic per-event directions, large per-event alarm, small mean shift:
        if ≈0 consequence                                                -> output-null   -> no action (control)
        else                                                             -> output-aligned
elif mean shift and layer profile peaks at P, small at z                 -> G geometry    -> LoRA P/S2 (linear: refit head)
elif out-of-span fraction high and Fisher-rank drop                      -> E support     -> abstain
elif out-of-span fraction low and consequence up                         -> E in-span     -> recalibrate / extend head
else                                                                     -> abstain (undeclared)
```

The five-arm attribution classifier and the alibi-detect baselines (MMD, C2ST, KS,
embedding-mean distance) are run **beside** this, not instead of it: they answer "did
something change"; the lookup answers "which determinant". That contrast is C2.

### III.3.4 Abstention

Conformal on classifier margin does not detect undeclared families (Tier-1 dev:
AUROC 0.25); a feature-space Mahalanobis nonconformity in the reference-cell z-metric
does (0.71). Use the Mahalanobis rule, and the Fisher-rank criterion as a second,
mechanism-derived abstention trigger.

## III.4 Adjustment — what to change once you know which part

The point of §III.1 is that **the repair follows from the diagnosis, and three of the
four repairs are not gradient steps** (`latent_monitor.adjust`).

| diagnosis | adjustment | what must be true afterwards (the test) | what must NOT be done |
|---|---|---|---|
| Σ, covariance-type | estimate Σ from noise-only records or residuals at the perturbed cell; set Σ̂ ← Σ in W; **re-derive the encoder as the Σ̂′-weighted (GLS) projection onto the raw signal basis S = W⁻¹D** (temporal GLS per channel, GLS channel weights, gain matched) | noise-only variance ratio returns to 1.00 (verified on all four Σ-covariance cells); z on a pure signal unchanged to 0.1 %; K recovers; decoder and head untouched | retraining anything; a bare layer swap (×4–5 consequence) |
| Σ, structural | activation-patch: substitute the *clean* S1 output into the perturbed forward pass. If K recovers, S1 is causal → LoRA on S1 with the twin pairs | patched K ≈ clean K before any LoRA is trained | LoRA on S2/g "because it helps" — the wrong-stage control, which must repair *less* (C5) |
| G | LoRA on P + S2 only, on a small paired sample from the new geometry; W, S1, g, o frozen. **Linear subject: refit the output head** (1.8 → 0.99); the pooling weights cannot (1.8 → 1.8–1.9) | layer profile flattens at P; z displacement shrinks; K recovers | touching S1 — the per-channel physics did not change |
| E, in span | recalibrate or extend the output head on the rare family | consequence recovers; noise-only statistics still at reference | abstaining (wasteful) or touching the encoder (the representation was intact) |
| E, support | **none on weights.** Abstain; extend training support with data from the new family; retrain | Fisher rank restored after retraining | fine-tuning on the perturbed cell: outside T_S weights fitted there fit noise |

Two honesty constraints, both from the audit: Hase et al. (2301.04213) show
localisation does not predict where editing works, so every stage-localised repair is
validated by **activation patching before LoRA**, never by LoRA success alone; and
"diagnosed-stage LoRA beats wrong-stage LoRA" is Surgical Fine-Tuning's result (Lee
et al.) — cite, and claim only the instantiation.

## III.5 How each arm is used under this protocol

One reference cell per arm; every other cell moves **one** factor and is paired to
the reference. The single permitted two-factor cell is G × Σ, as a pre-registered
robustness check, run last.

| arm | reference cell | G cells | Σ cells | E cells | U (abstain) | status |
|---|---|---|---|---|---|---|
| **Tier 1 ORACLE-Cov** | synthetic box, Σ̂ = Σ, C=8, N=256 | ½C, 2C on the same box | corr ↑/↓; bandwidth; line pickup; gain drift; channel loss; jitter | oscillation, double pulse (in span) | glitch (out of span) | **done**, 13/14 + designed pair (`latent_monitor.tier1`) |
| **B · HeST** | `HeRALD_v1` (24 ch), ER 1 keV, `TES_HERALD_V1` | `_monolithic` (1 ch), `UMass_splitCPD` (2 ch); `fill_height` | bath correlation ↑; low-rank pickup modes; SQUID 1/f knee ×10; mains ×8; sensor loss; gain drift; timing jitter | NR at the same E; ER at ½× and 2× | WIMP spectrum, 500 MeV | **built**, 14 cells (`herald_simulation.simulate.all_cells`); protocol run pending |
| **A · LUCiD** | `SK_like` @ 11 000 placed (`WCTE_like` in the notebook), `PMT_FRONTEND_V1`, muon/flasher | `n_sensors` 4 000 and 20 000 (same cylinder); `MidBox` (same volume) | clock-line ×k; group coherence 0 → 0.6; alias fold 1 GHz → 250 MHz; per-channel gain drift; dark rate; channel loss | water → WbLS at fixed geometry; e⁻ and π⁰ via PhotonSim | supernova burst, pile-up | notebook only; gated on A0 |
| **C · TIDMAD** | real noise, MSE loss | — | the *other* loss (inverse-PSD) — Σ̂ only | — | — | needs data + GPU |
| **Prometheus** | ORCA, DynEdge frozen | ARCA, +4 geofiles | NuBench §3.2 knobs (hit level) | S1–S5 | U1–U4 | built, licence-clean |

**What each arm proves in the table.** Tier 1 proves it. LUCiD shows the same rows
hold when the channel stage is a photon-counting PMT array in a real geometry, and
adds the G row at scale. HeST shows the G row at the other extreme — 24 → 1 channel
makes the multichannel covariance claim vacuous by construction, a designed contrast
— and the E row on ER/NR, which is physics nobody here wrote. TIDMAD shows the Σ row
on noise nobody here generated. Prometheus shows the G row on a model nobody here
trained.

## III.6 LUCiD — integrating `noise_module` (gated on A0)

The build is §II.7; this is the integration into the protocol.

**Pipeline.** PhotonSim ROOT (fixed event) → `setup_event_simulator(geom, n_photons,
is_data=True, hit_mode='waveform', window_ns=32768, bin_width_ns=1.0)` →
`charge_waveform (n_sensors, 32768)` → `noise_module_lucid.adapter.add_readout_noise(
charge_waveform, detector, preset, groups, rng)` → `trace_mV, meta{Sigma_hat, Sigma,
kappa, kappa_floor, groups}` → parquet with `event_id`, `geometry_hash`,
`preset_version`. `window_ns=32768` is not a default; it is the N/C rule for a
64-channel group. Record it in provenance every time.

**Σ̂ enters the model, not only the data.** Layer W is initialised from
`PMT_FRONTEND_V1`'s one-sided PSD **and** its group covariance: Σ̂ = Σ_group ⊗ S(f)
block-diagonal by group. A Σ cell realises a *different* Σ while W keeps Σ̂, and
κ(Σ̂⁻¹Σ) is computed from the two matrices the generator returns.

**The Σ families and their rows.**

| family | generator knob | type | predicted row |
|---|---|---|---|
| clock-line amplitude ×k | `Line.scale` in the preset | covariance | variance ratio ≠ 1 at the line bins only — a *narrow-band* κ |
| group coherence | `corr_strength` 0 → 0.6 | covariance | off-diagonal Σ mismatch; κ grows with C |
| alias fold | decimate 1 GHz → 250 MHz, no anti-alias | covariance | broadband, predicted by `alias_fold_psd_density` in closed form |
| per-channel gain drift | `channel_gain_jitter` | structural | mean shift at S1 |
| dark rate | LUCiD's own `dark_rate_khz` | structural | mean shift at S1, sparse |
| channel loss | mask PMTs | structural | mean shift at P (tokens vanish) |

The first three are Σ̂ ≠ Σ in the strict sense and carry the C4 κ prediction; the
last three are N by contract but mean-shift in the latent, and the table must say so.

**Channel groups.** `string_id` for string telescopes; for cylinders, angular sector ×
height band (16–64 PMTs each), a stand-in for a crate. The **group is the covariance
unit**; cross-group covariance is zero in Σ̂ and may be non-zero in Σ (a low-rank
global pickup) — a clean Σ family in its own right.

**Acceptance for the integration.** The subject trained on the reference cell
reproduces the matched-cell κ floor (§II.7.6 AC-2) in its own residuals; on the
alias-fold cell the whitened residual variance departs from 1 in the bins the closed
form predicts, and nowhere else.

## III.7 HeST — the fork, and HeRALD-shaped noise (built 6 Sep)

**Confirmed about HeRALD** (arXiv:2307.11877): a TES-based calorimeter reading both the
quantum-evaporation signal and helium quasiparticle excitations; threshold 145 eV at
5σ; evaporation gain 0.15 ± 0.01. The TES parameters (T_c, R_n, τ_eff), the SQUID
floor and the baseline resolution are **not** in the abstract; until read from the
paper every such constant in `HERALD_V1_PLACEHOLDER` carries provenance state
`placeholder` (only the two time constants do not), and **no dataset is released
before they are `from_paper`**.

**Fork structure — HeST untouched, mirroring `prometheus_simulation`.**
`src/herald_simulation/`: `geometry.py` (five builders + `make_cell`, `geometry_hash`),
`events.py` (`quanta`, `evaporate`; `event_id` seeds the population — the pairing
contract, enforced by `tests/test_pairing.py`), `traces.py` (QPSimulator per sensor →
`(C, N)`), `noise.py` (`TES_HERALD_V1` + `MultiChannelNoiseGenerator` → Σ̂, Σ, κ,
κ_floor), `interventions.py` (Σ cells and structural N), `strata.py` (E cells),
`export.py`, `simulate.py` (reference + 13 one-factor cells, `--qp-fraction`). **No
edits inside HeST**; bugs go upstream as PRs. That keeps "the physics is theirs" true.

**Noise made HeRALD-shaped** — `noise_module.tes_budget.TESNoiseBudget`, closed-form
and citable (Irwin & Hilton 2005), built from existing components:

| term | form | component |
|---|---|---|
| TES Johnson | white, electrothermally suppressed below the loop roll-off | `white` + `rolloff(highpass)` — suppression factor placeholder |
| thermal-fluctuation noise | white current noise shaped by the responsivity `S_TFN/(1+(2πfτ_eff)²)` | `white` + `rolloff(lowpass)` — dominant in-band |
| load/shunt Johnson | white | `white` |
| SQUID amplifier | white floor + 1/f, knee f_k | `white` + `powerlaw(−1)` |
| vibration / microphonics | narrow lines 5–200 Hz — **in band** at 250 kHz × 16 384 (df = 15.3 Hz, the opposite of LUCiD) | `line` ×n |
| mains pickup | 50 Hz + harmonics, in band | `line` |
| paramagnetic spin | — | removed; not a TES term (the existing `AthermalNoiseBudget` is a *magnetic* calorimeter) |

Multichannel structure for 24 CPDs on one cold stage: shared low-frequency bath
fluctuation (`shared_private`, TFN-shaped shared spectrum); shared lines via
`lowrank` (1–3 latent modes); private white per TES; non-Gaussian sparse bursts via
`ArtifactInjector`, singles-vs-shared as an N-vs-U family. `corr_strength` and the
number of modes are placeholders until a measured CSD exists; the *structure* is the
claim, the constants are declared. N/C = 683 at C = 24, N = 16 384; measured
matched-cell κ floor 1.18.

**Acceptance (status):** (1) bit-reproducible pairing across geometries — **tested**;
(2) `TES_HERALD_V1` integrates to `noise_power`, every term named, placeholders
flagged — **done**; (3) matched-cell κ floor measured and reported — **done, 1.18**;
(4) the 24 → 1 contrast reproduces the designed prediction — **pending** (protocol
run on arm B); (5) ER vs NR lands in P_unexc with a Fisher-rank drop when NR is
outside support and not when inside — **pending**, and subject to the ER/NR–energy
degeneracy of §II.4.2.

## III.8 Work packages and order

| WP | what | days | gate | status (10 Sep) |
|---|---|---|---|---|
| N1 | `freeze_channel_structure` in `noise_module` | 0.5 | — | **done** |
| L0 | linear subject + §III.3 projectors + lookup on Tier 1 | 3 | — | **done** (13/14) |
| H0 | HeST pilot | 0.5 | B0 email sent | **done** |
| H1 | `herald_simulation` fork + `TESNoiseBudget` | 4 | H0 | **done**; constants placeholder |
| S1 | make the transformer subject expose W, P, z, `jac_*` | 2 | — | **done** (`torch_subject.py`, CPU smoke); *trained* table pending (GPU) |
| L1 | the four linear classes as subjects + axis identification (Part IV) | 3 | — | modules copied; experiments tentative |
| A1 | `noise_module_lucid` (§II.7) | 8 | **A0 licence** | blocked |
| A2 | LUCiD cells + integration (§III.6) | 3 | A1 | blocked |
| C1 | TIDMAD WP10 | 3 | data + GPU | pending |
| B | pre-registration bridge frozen | — | L0, S1, per-arm κ floors | pending |
| R | confirmatory runs, all arms | — | B | pending |

Order: N1 → L0 → H0 → S1 → H1 (done) → **arm B protocol run** → L1 → trained
transformer table → C1 → (A1 → A2 when A0 lands) → B → R. The linear subjects come
before the transformer because that is where the table is provable.

## III.9 Risks specific to this protocol

- **Geometry invariance is an assumption of the G row, not a given.** On the linear
  subject it already fails in the predicted way (repair through the head). If S2 is
  not channel-count-invariant on the transformer either, G shows up at z — a finding
  about the architecture, not a failure of the protocol. Say so in the bridge.
- **Structural N and E share a mean-shift signature.** Separated by the noise-only
  statistics (6 Sep: cleanly, on the linear subject at dev SNR); if those are weak at
  realistic SNR, attribution degrades to "N-or-E, not covariance Σ". Report the
  confusion.
- **Estimator floors differ per arm** (1 GHz vs 250 kHz vs real). Re-measure the N/C
  rule on each arm before the bridge is frozen; a κ prediction made against Tier 1's
  floor is transferable as a sign, not as a number.
- **HeRALD constants are placeholders.** Two provenance states, `placeholder` and
  `from_paper`; the paper reports which.
- **The κ mismatch costs a linear subject little in consequence** (6 Sep: κ ≈ 5–12
  cost almost nothing; the alarm is where κ shows). The C4 claim therefore rests on
  the designed dissociation and on the nonlinear subjects, and the paper must not
  imply that re-whitening rescues a large consequence on the linear subject.

---

# Part IV — The linear subject classes: tentative plan

_New, 10 Sep 2026. **Tentative**: nothing here is pre-registered or in the critical
path of Part III; it widens the analytic baseline of §III.2.3 from one linear subject
to the four linear representation classes of Paper 1, so that the same protocol can
say which class's latent is readable as physics, and it is the linear-subject
instrument for the proposal's C0 (physical-variable organisation). Code:
`src/latent_monitor/estimators/` (copied from the Paper 1 experiment repository
`noise-weighted-subspace-reconstruction` @ `ea076ff`; provenance per module).
Paper 1 owns these classes as *reconstruction* results; ORACLE uses them as frozen
*subjects*. The boundary is Paper 1's rule: **no new architecture is built** — every
model below is an existing, verifiable estimator, and each is the same
Σ̂⁻¹-orthogonal projection restricted to a different admissible class._

## IV.1 The four classes

| # | class | concretely | learned parameters | metric ablation | code |
|---|---|---|---|---|---|
| 1 | **OF** — optimal filter | known template s; â = (sᴴΣ̂⁻¹x)/(sᴴΣ̂⁻¹s), GLS / matched filter, frequency domain with inverse-PSD weights 1/J_k | 0 | — (Σ̂ = I is the un-whitened matched filter) | `estimators/of.py` |
| 2 | **CW-PCA / EMPCA** | whiten with Σ̂^{-1/2}, top-k right singular vectors of the whitened data, x̂ = μ + P(PᵀΣ̂⁻¹P)⁻¹PᵀΣ̂⁻¹(x−μ); closed form | k(d−k) (Grassmannian) | **IsoPCA** (Σ̂ = I) | `estimators/cw_pca.py` |
| 3 | **tied noise-aware linear AE** | decoder x̂ = Dz, encoder z = DᵀΣ̂⁻¹x (the Σ̂⁻¹-adjoint), loss (x−x̂)ᵀΣ̂⁻¹(x−x̂); one linear layer each way; trained by L-BFGS-B (S3: 72–77 iterations to the EMPCA subspace) or evaluated in closed form | same subspace as 2; parameterisation differs | untied D, E (P1-E12: tied/untied × Σ̂⁻¹/MSE) — *not copied* | `estimators/tied_linear_ae.py` |
| 4 | **NFPA** — noise-factored projection autoencoder | per-axis whitening Z̃ = Σ_c^{-1/2} X Σ_t^{-1/2}; orthonormal U_c (C×k_c), U_t (T×k_t); code Γ = U_cᵀZ̃U_t; reconstruction U_cU_cᵀZ̃U_tU_tᵀ; alternating Rayleigh–Ritz (covariance-weighted Tucker-2), ten restarts | k_c(C−k_c)+k_t(T−k_t) (771 at (2,3) vs 12 252 unrestricted rank 6) | **Iso-MPCA** (Σ̂ = I) | `estimators/nfpa.py` |

Not part of the linear classes but relevant: the TIDMAD baselines (zero, calibration
mean, bin read, quadrature OF, Wiener gain, calibration shrinkage) are estimators, not
representation classes; the one nonlinear arm is the existing rung-4 factored-attention
transformer (`src/tidmad_transformer/`, 569 352 parameters), which Paper 1 extends to
≥ 10 seeds and ORACLE uses as the Tier-3 subject — shared, not duplicated.

## IV.2 What the protocol can and cannot read off each class

**The common mechanism.** Around a template s(t; a, τ, θ) the pulse manifold is, to
first order, x ≈ a·s + a·τ·∂_t s + a·δθ·∂_θ s + n. The signal lives in the tangent
space span{s, ∂_t s, ∂_θ s}, and every rank-k class recovers that *span* when the
parameter spread is small against the noise (the regime of Paper 1's T19 first-order
displacement result). Two consequences: the coordinate along s is the amplitude a
(linear, identified), but the coordinates along the derivative directions are a·τ and
a·δθ — bilinear, so timing and shape are read off a linear code only as products with
amplitude; and whether the *axes* of the code coincide with those directions depends on
the class's gauge.

| class | latent | identified by construction | gauge freedom | what recognition needs | boundary |
|---|---|---|---|---|---|
| OF | one scalar â | amplitude, exactly; unbiased under Σ̂ ≠ Σ (wᴴs = 1), variance inflates — the "no mean shift" row | none | nothing | everything off-template is residual; CRESST's bank recovers shape/shift only as *discrete* labels |
| CW-PCA | k coordinates, variance-ordered | the subspace (Grassmannian point) | axes rotate with the *population* covariance — an E-factor move rotates the basis even when the physics manifold does not | a k×k re-identification: regress the code on the planted (a, a·τ, a·δθ) or Procrustes-align P to the tangent basis | IsoPCA's axes are biased toward high-noise frequencies — the metric, not the architecture, decides whether axes can be made physical |
| tied AE | k coordinates, arbitrary basis | the subspace | O(k) (U(k)): L-BFGS lands on a rotation of the EMPCA basis fixed by initialisation; untied enlarges it to GL(k) | the same gauge fix; Jacobians, P_out/P_null, P_exc/P_unexc are gauge-covariant, so §III.3 is unaffected — only per-axis readings need alignment | as class 2 |
| NFPA | Γ (k_c × k_t) | the *structure*: to first order in per-channel delays X ≈ h sᵀ + (h∘τ)(∂_t s)ᵀ is Tucker-2 with k_c = k_t = 2 — U_t ⊃ {s, ∂_t s} (amplitude, timing, shape), U_c ⊃ {h, h∘τ} (channel share and channel delay = position by solid angle and triangulation); geometry in the channel factor, pulse physics in the temporal factor, by a linear restriction rather than by design | two-sided, Γ → Q_cᵀ Γ Q_t | alignment on each factor | Kronecker Σ̂ = Σ_t ⊗ Σ_c fails for shared-bath-plus-private-lines noise (ORACLE's side of the freeze); degenerate at C = 1 (TIDMAD) and C = 2 (CRESST phonon/light) |

**The regime boundary all four share.** A time shift is not low-rank. In the
frequency domain a shift multiplies by e^{−iωτ}; the linear regime is σ_τ·f_max ≪ 1
(shifts small against the rise time). Outside it the translation orbit needs rank
growing with σ_τ × bandwidth, the recovered directions are a Fourier-like expansion of
the orbit rather than {s, ∂_t s}, and no rotation of the code yields a physical axis.
Amplitude never has this problem; rise/decay variation has it mildly; arrival time has
it badly — which is why CRESST discretises shifts and why the TIDMAD quadrature OF
reads phase as atan2 of a 2-D code. The general lesson, and the reason the structured
target in `TESTBEDS.md` §2 has a physics head: **the linear code is the sufficient
statistic; the physical parameter is a readout of it.**

## IV.3 How the classes slot into Part III

The linear subject of §III.2.3 is class 2/3 with a geometry-weighted pooling and a
least-squares head (`linear_subject.py`). The plan is to make the class a parameter of
the subject, not to build four subjects:

- **[W]** unchanged (`KroneckerWhitener`). **[S1]+[S2]+[g]** become the class's
  `encode`/`reconstruct`: OF (rank 1, template from the reference cell), CW-PCA (rank
  k), tied AE (rank k, trained, then gauge-fixed), NFPA (k_c, k_t) — with the
  Σ̂⁻¹-adjoint encoder in every case, so `with_sigma_hat` (the GLS re-derivation of
  §III.4) applies verbatim. **[y]** stays the least-squares head.
- The **axis-identification diagnostic** (`estimators/identification.py`) is added to
  the reference-cell fit: the k×k mixing matrix between the recovered basis and the
  tangent basis {s, ∂_t s, ∂_θ s} in the Σ̂⁻¹ metric, its condition number, the
  per-direction residual, and the canonical gauge (rotate so coordinate i tracks
  tangent direction i). It costs nothing beyond what §III.3.1 already computes and
  is not a new architecture.
- **Metric ablations are cells, not models.** IsoPCA and Iso-MPCA are the Σ̂ = I
  twins; running them through the same protocol is the linear-subject form of the
  TIDMAD two-Σ̂ contrast (arm C), and the axis-identification residual is the
  statistic that shows *why* the metric matters for readability.

## IV.4 Proposed experiments (tentative, not pre-registered)

| id | question | design | prediction | reports |
|---|---|---|---|---|
| L1-a | Does the §III.1 table hold for every class? | Tier-1 cells × {OF, CW-PCA, tied AE, NFPA}; same thresholds, calibrated per class on the reference null | every row as on the linear subject; OF has no G row (rank 1) and no in-span E row (off-template → residual); NFPA's G cells show in U_c only | the table, per class |
| L1-b | Which class's axes are physical? | axis identification on the reference cell, per class; planted σ_τ swept across the linear-regime boundary | OF exact for a; CW-PCA/tied AE identifiable after the gauge fix inside the regime, condition number diverging with σ_τ·f_max; NFPA identifiable per factor | mixing-matrix condition number and residual vs σ_τ·f_max — the empirical regime boundary |
| L1-c | Does a population (E) move rotate the axes without moving the physics? | in-span E cells; compare raw code displacement with gauge-fixed displacement | raw PCA/AE axes rotate (apparent latent shift); aligned coordinates and the subspace do not | the E-row confusion with and without the gauge fix |
| L1-d | Is the metric what makes axes readable? | IsoPCA vs CW-PCA, Iso-MPCA vs NFPA, Σ-covariance cells | isotropic twins have larger tangent residuals and their axes drift under Σ moves; whitened classes do not | residual and angle differences; the linear form of the arm-C contrast |
| L1-e | Does the factored code separate G from pulse physics on arm B? | NFPA on `HeRALD_v1` cells: 24 → 1/2 sensors, vertex/`fill_height` vs ER/NR/energy | geometry moves in U_c (channel factor), physics in U_t; a Σ_c change moves the channel whitening only | energy split of ΔΓ across the two factors |

Descope: L1-e first to cut (depends on arm B's protocol run), then L1-c. L1-b is the
one that changes the paper's wording — "the latents are physical" becomes "exact for
amplitude, exact-up-to-a-known-gauge for timing and shape inside σ_τ·f_max ≲ x, with a
stated breakdown" — and is the piece to keep if only one is run.

## IV.5 What is deliberately out of scope

No conv AE, U-Net, transformer or graph model is built for this part (Paper 1's rule;
ORACLE's nonlinear subject is the existing compact transformer). The untied AE
ablation, the CRESST multi-template OF bank and the TIDMAD quadrature OF stay in the
Paper 1 repository. Paper 1's *reconstruction* results (planted-subspace recovery,
NFPA held-out risk, TIDMAD held-out test) are cited, not reproduced here.

---

# Appendix A — Programme status (10 Sep 2026) and document map

| tier / arm | substrate | status | carries |
|---|---|---|---|
| Tier 1 — ORACLE-Cov | `noise_module` synthetic | linear-subject table done (13/14); trained-transformer table pending | whitening lemma; κ sweep; designed dissociation |
| Arm B — HeST → `qp_simulator` → `noise_module` | simulated TES, 1–24 ch, 250 kHz | **built**, 14 cells, tests green; constants placeholder; gate B0 open | granularity dissociation 24 → 1; second readout physics |
| Arm A — LUCiD + `noise_module` | simulated PMT, ~2 400 ch, 1 GHz | **notebook only**; gate A0 open | transfer of the κ prediction into a real geometry; waveform-level C1/C2 |
| Prometheus / DynEdge | hit-level | built, licence-clean | frozen public model; angular-error consequence; fallback for A |
| Arm C — TIDMAD | real SQUID | code exists, needs data + GPU | external validity |
| linear subject classes (Part IV) | Tier 1, arm B | modules copied and tested; experiments tentative | C0 on linear subjects; readability of the latent |

**Document map.** Live: this file; `TESTBEDS.md`; `IMPLEMENTATION_PLAN.md`;
`RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md`; `REVIEW_PROMPTS.md`; `TODO.md`;
`reviews/`. Archived under `docs/archive/` with a README index: the three merged
plans, the testbed survey, the 3 Sep theme/novelty/dev notes and their reviewer
prompt, and the pre-31-Aug material.

# Appendix B — Section mapping for existing citations

Citations in code, notebooks and `TODO.md` to the merged files resolve as follows
(the archived copies keep their original numbering).

| old citation | here |
|---|---|
| `EXPERIMENT_DESIGN.md` (31 Aug) §"story / tiers / claims / sequencing / release / models vs diagnostics / datasets" | §I.1 – §I.7 |
| `EXPERIMENT_PLAN_ARMS_2026-09-05.md` §n | §II.n (§6 → §I.3) |
| `LATENT_MONITORING_PLAN_2026-09-05.md` §n | §III.n (§2.2 structured shape → `TESTBEDS.md` §2) |
| `SIM_TESTBED_SURVEY_2026-09-05.md` | `TESTBEDS.md` (inventory) and `docs/archive/` (evidence) |
| `DEV_UPDATE_2026-09-03.md` findings | §III.1 (▸ corrections), `RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md` |
| `THEME_ADJUSTMENT_2026-09-03.md`, `NOVELTY_CHECK_2026-09-03.md` | §I.1 (second paragraph), §I.3 (C0); `TODO.md` carries the actions |
