# ORACLE implementation plan

**Revision work queued 16 September 2026:** see [§8 — MLST revision](#8-mlst-revision-plan--16-september-2026)
for the current revision sequence and [the implementation-agent prompt](IMPLEMENTATION_PROMPT_MLST.md).
Section 8 is a proposed amendment programme, not a protocol freeze; the canonical
scientific design remains `EXPERIMENT_DESIGN.md` until the amendments are implemented.
**First implementation pass done the same day** (§8.5; `REVISION_REPORT_2026-09-16.md`):
the amendments are in `EXPERIMENT_DESIGN.md` Part V, the unfrozen draft
`PREREGISTRATION.md`, and `src/latent_monitor/protocol/`. No freeze exists;
no scientific claim is validated by it.

_31 August 2026. The buildable version of `EXPERIMENT_DESIGN.md`. That document
says what the study is; this one says what gets written, in what order, with
what interfaces, and what "done" means for each piece. Review it with
`docs/REVIEW_PROMPTS.md` §A before any code is written, and §B after._

_Revision 2, same day: the §A review (`docs/reviews/2026-08-31_planA.md`)
returned twelve blockers; every one is resolved below and marked **[A-Bn]**.
Two scope decisions it forced are flagged for the collaboration in §7._

---

## 0. Ground rules — what "real, not toy" means here

Every rule below is enforced by something a reviewer can check, not by intent.

1. **Protocol before data.** Endpoints, thresholds (false-alert budget, κ_m,
   equivalence margins, power targets) and the family lists live in
   `docs/PREREGISTRATION.md` and are hash-frozen *before* any confirmatory run.
   Confirmatory runners refuse to start unless the config carries the frozen
   hash (WP7).
2. **Two modes, never mixed.** `mode: dev` may touch anything and produces
   nothing citable. `mode: confirmatory` consumes frozen configs, held-out
   seeds, held-out severities and held-out families, and writes results only
   into `experiments/<id>/results/confirmatory/`. The paper cites only the
   latter.
3. **Every metric has an analytic unit test and a positive control.** A
   monitor that cannot be shown to recover a known answer on synthetic data
   with a closed-form solution (WP2 tests) does not enter the frozen set. A
   metric is promoted from descriptive to confirmatory **only** on
   positive-control monotonicity — never on how well it beats a baseline
   **[A-B12]**. Baselines receive the same declared tuning budget on dev
   data (one hyper-parameter grid each, listed in the pre-registration).
4. **Baselines get the same budget.** Identical windows, identical
   false-alert calibration on the same clean null, identical splits. A
   baseline that is not calibrated to 1 % FAR on held-out clean data is not a
   baseline.
5. **Resampling units are the real units.** Intervals resample *event groups*,
   *perturbation seeds* and — where more than one subject model is trained —
   *model seeds*, as separate nested levels; never pulses, never windows that
   share an event. Enforced in `oracle_diag.report`. Experiments with a single
   model seed say so and restrict their claim accordingly.
6. **Seeds and provenance in every artifact.** Each result table carries the
   config hash, the git commit, the data manifest hash, and the seed list.
   Anything without them is dev output.
7. **Toy generators never touch confirmatory paths.** `prometheus_simulation.toy` and
   any `demo_*` helper are import-forbidden under `experiments/` — a test
   greps for it (WP7). The tutorial notebook is documentation, not evidence.
8. **Negative results are reportable.** Every comparison declares an
   equivalence margin in advance, so "no benefit" is a finding, not a failure.
9. **Costs are measured, not asserted.** C3 latency and storage numbers come
   from `oracle_diag.hooks.benchmark` on the real models with p50/p95 over
   ≥ 30 repetitions, wall-clock and bytes, on a named machine.
10. **Nothing is "validated" by adaptation gains.** Any localization claim is
    preceded by activation patching (WP6); LoRA repair is out of scope for
    this paper (§7).
11. **One primary endpoint per claim; everything else is corrected.** Each
    claim names exactly one primary endpoint. All secondary endpoints across
    arms × monitors × families × experiments are controlled with
    Benjamini–Hochberg FDR at 0.05 within each claim, and the prediction table
    (§5) is scored as a single pre-registered summary (fraction of rows
    confirmed, with a binomial CI), not as row-wise tests **[A-B11]**.
12. **Freezing is versioned.** The protocol freezes in numbered parts, each
    with its own hash: `core` (endpoints, thresholds, families, margins) at
    M0; `counts` (from the power analysis) at M3; `bridge` (the prediction
    table) at M3. A confirmatory config lists which parts it depends on; a
    result carries the hashes of exactly those parts **[A-B6]**. cov_B needs
    only `core`, which is why it can be the first citable result.

---

## 1. Target repository layout

_Annotation, 16 Sep 2026: `src/oracle_diag/` and `experiments/` were never
created. The diagnostics layer was built as `src/latent_monitor/` (hooks,
projectors, statistics, lookup, designed contrasts, adjustments; `EXPERIMENT_DESIGN.md`
Part III) and, since 16 Sep, `src/latent_monitor/protocol/` (WP0/WP4/WP5/WP6/WP7
pieces: availability manifest, splits, arms, all-cell C4, mode/freeze, smoke). The
WP names below are kept; read `oracle_diag.<x>` as `latent_monitor.<x>` where one
exists._

```
src/
  noise_module/          exists  — Tier-1 generator, Σ̂/Σ instrument
  prometheus_simulation/         exists  — Tier-2 production (needs prometheus_io, WP9)
  tidmad_transformer/    exists  — Tier-1/3 subject model + training
  reconstruction_model/  exists  — DELight transformer (Tier-1 subject variant)
  oracle_diag/           NEW     — the diagnostics layer (WP1–WP7)
    hooks/               representation capture, sketches, cost benchmark
    metrics/             displacement family, angles, kNN, layer profile
    perturb/             output-null / aligned / random generators
    baselines/           alibi-detect wrappers + embedding/output/uncertainty
    abstain/             split conformal, risk–coverage
    consequence/         K per tier, C4 matrix, conditional AUROC
    protocol/            config schema, freeze/hash, mode enforcement, registry
    report/              bootstrap CIs (correct units), RESULT.md writer
    tests/
experiments/
  cov_B_dissociation/    each: config.yaml, run.py, analysis.py,
  cov_A_kappa_sweep/           PREREGISTRATION.md (per-experiment section),
  cov_C_nsu_waveform/          results/{dev,confirmatory}/ (gitignored),
  cov_D_power/                 RESULT.md (committed, generated)
  paired_E1_detection/ … paired_E5_geometry/
  tidmad_T1_sigma_contrast/
docs/
  PREREGISTRATION.md     the frozen protocol (WP0); hash recorded in configs
  IMPLEMENTATION_PLAN.md this file
  REVIEW_PROMPTS.md      reviewer instructions, before and after
```

---

## 2. Shared interfaces (write these first; everything plugs into them)

```python
# oracle_diag/protocol/types.py
@dataclass(frozen=True)
class Family:            # a declared or undeclared perturbation family
    name: str            # "N1_module_loss", "S4_cascade", "U2_throughgoing", "null", "aligned", ...
    contract: str        # "N" | "S" | "U" | "artifact" | "clean"
    declared: bool       # False => abstention target, never in training/calibration
    severity: float | None

@dataclass
class Window:            # the monitoring unit; the resampling unit is event_group
    ids: np.ndarray      # event ids in the window
    event_group: int     # groups that share latent events (paired geometries share a group)
    family: Family
    seed: int

@dataclass
class RepresentationBatch:
    stage: str           # named hook point
    Z: np.ndarray        # (n, d) or sketch
    outputs: np.ndarray  # model outputs for the same events
    meta: dict

class Metric(Protocol):
    name: str
    def fit(self, clean: list[RepresentationBatch]) -> "Metric": ...      # clean reference only
    def score(self, batch: RepresentationBatch) -> float: ...             # alarm magnitude A
    def needs_covariance(self) -> bool: ...                               # True only for pullback-whitened
    def is_analytic_for(self, family: Family) -> bool: ...                # True when the result is by construction (A-B3)

class Perturbation(Protocol):
    family: Family
    def apply(self, batch: RepresentationBatch, rng) -> RepresentationBatch: ...

class Consequence(Protocol):
    name: str
    def score(self, clean_outputs, perturbed_outputs, truth) -> float: ...  # K
```

Rules: `Metric.fit` sees clean data only; `score` never sees labels; every
`Perturbation` records its exact parameters into `RepresentationBatch.meta`.

---

## 3. Work packages

Each WP lists purpose, deliverables, interfaces, acceptance criteria (AC — all
must pass), and dependencies. Effort is in focused engineering days.

### WP0 — Protocol freeze (`docs/PREREGISTRATION.md`) · 2 d · no deps

Deliverables: the frozen protocol, in three hashed parts (`core`, `counts`,
`bridge`; §0.12). `core` contains:

- **One claim ladder** **[A-B8]** — the proposal's, adopted everywhere:
  **C1** detection · **C2** attribution *with abstention* (abstention is a C2
  endpoint family, not a separate claim) · **C3** cost · **C4** alarm ranks
  consequence · **C5** stage localization, tested by activation patching only
  (LoRA repair is out of scope, §7). `EXPERIMENT_DESIGN.md`'s table is
  reconciled to this ladder.
- **The claim → experiment → endpoint → threshold table**, both directions
  complete, including **E0** (the deterministic E-fault gate:
  sensitivity/specificity on planted unit/coordinate/metric faults) and
  numeric endpoints for E1–E4, T1 and cov_E, not only E5.
- **Equivalence margins for every comparison** **[A-B9]**: ΔF1 ±0.05; power
  5 points; for cov_A, "degrades" means Δρ_Euclid(κ=10³ vs κ=1) ≤ −0.2 with
  95 % CI excluding 0, and "does not degrade" means |Δρ_pullback| ≤ 0.10 with
  the CI inside ±0.10; anything else is "inconclusive".
- **The matching specification** **[A-B5]**: features (n_pulses, total
  charge, time spread), standardized by the clean pool; caliper 1.0 in
  standardized Euclidean distance; 1:1 without replacement; **unmatched N
  events are retained in a declared `N*_unmatched` cell and reported
  separately, never dropped**; realized match rate reported with the result.
- **Multiplicity** **[A-B11]**: the primary endpoint per claim; BH-FDR 0.05
  over secondaries within claim; the §5 table scored as one summary statistic.
- Declared N/S family lists per tier; the U list; κ_m per tier with
  justification; FAR budget 1 %; resampling units incl. model seed; severity
  ladders; positive-control definitions; **the baselines' tuning grids**
  **[A-B12]**; the "strong generic" arm definition.

`freeze <part>` writes `protocol/frozen/<part>.json` with the SHA-256 of the
corresponding file section.
AC: every claim C1–C5 maps to ≥ 1 experiment with a numeric primary endpoint
and threshold; every experiment maps back to a claim; the §A reviewer finds no
unresolved blocker.

### WP1 — `oracle_diag.hooks` · 3 d · deps: none (models exist)

Deliverables: `capture(model, batch, stages) -> list[RepresentationBatch]` for
the compact transformer (stages: input embedding, each encoder block, pooled,
head) and for DynEdge (the six D4 points + pooled); `Sketch` (10 % event
sampling; fixed-size random-projection + moment sketch); `benchmark()` for C3.
AC: hooked tensors equal a manual forward to 1e-6; stage names are stable and
listed in one registry; sketch power loss measured (C3 endpoint: ≤ 5 points
F1/power loss vs full capture, tested on Tier-1 dev data); benchmark reports
p50/p95 latency and bytes/event on a named machine over ≥ 30 runs.

### WP2 — `oracle_diag.metrics` · 4 d · deps: WP1

Deliverables: `StandardizedDisplacement` (normalized by clean within-stratum
dispersion); `JacobianProjectedDisplacement` (J = ∂output/∂Z at the stage via
autograd; displacement projected onto J's leading right-singular subspace);
**`PullbackWhitenedDisplacement(Σ)`** **[A-B2]** — the paper's own statistic:
displacement δZ measured in the metric G = Jᵀ Σ⁻¹ J, i.e. ‖Σ^{-1/2} J δZ‖, the
Σ⁻¹-whitened output-space displacement pulled back to the stage
(`needs_covariance=True`). A standalone Σ⁻¹ metric *at the representation* is
not well defined (Σ lives in output/input space) and is not delivered;
`PrincipalAngles`; `KNNRetention`; `LayerProfile`.
Each metric declares `is_analytic_for(family)` **[A-B3]**: `Jacobian
Projected` and `PullbackWhitened` return True for `null`/`aligned` families
built from the same Jacobian, and reports mark those cells "by construction".
AC (analytic): on a linear model `y = W Z` with output noise covariance Σ,
`PullbackWhitened` equals the Mahalanobis distance of `W δZ` under Σ to 1e-9;
`JacobianProjected` of a null-space perturbation is exactly 0 and of an aligned
one equals its norm times the singular value; kNN retention is 1.0 on identity
and → 1/k on random permutation. AC (control): each metric is monotone across
the positive-control ladder on Tier-1 dev data.

### WP3 — `oracle_diag.perturb` · 3 d · deps: WP1

Deliverables: `OutputNull(k)`, `OutputAligned(k)`, `NormMatchedRandom(k)` —
representation-level perturbations at stage k built from the local Jacobian
SVD, norm-matched to a declared displacement magnitude; input-level adapters
that wrap `noise_module` N families (waveforms) and `prometheus_simulation`
interventions (point clouds) behind the same `Perturbation` protocol.
AC: on the linear model, `OutputNull` changes outputs by < 1e-6 relative and
`OutputAligned` by the predicted amount; norms match to 1e-6; on a *synthetic
nonlinear* model with a known Jacobian, first-order agreement to the declared
tolerance. **No acceptance criterion touches the real subject model's K**
**[A-B7]** — that measurement is cov_B's confirmatory result and may not be
observed before cov_B's prediction is registered.

### WP4 — `oracle_diag.baselines` · 2 d · deps: WP1

Deliverables: wrappers with the `Metric` interface around `alibi-detect`
(RBF-MMD, classifier two-sample test, corrected univariate KS on inputs),
embedding mean/covariance distance, output-distribution and
uncertainty/entropy tests; a common `calibrate(clean_null, far=0.01)`.
AC: on held-out clean windows every baseline's realized FAR is within the
binomial 95 % CI of 1 %; the strong baseline arm ("all generic") is committed
to in `PREREGISTRATION.md` before any N/S result is looked at.

### WP5 — `oracle_diag.abstain` · 2 d · deps: WP2

Deliverables: split-conformal nonconformity over the declared-family
signature; unseen-family AUROC; selective risk and macro-F1 vs retained
coverage; risk–coverage AUC; macro-F1 at 80 % coverage.
AC: marginal coverage guarantee holds on exchangeable synthetic data (within
the finite-sample bound); U families are never present in calibration
(enforced by `Family.declared` in the loader).

### WP6 — `oracle_diag.consequence` · 3 d · deps: WP1, subject models

Deliverables: `AngularError` (Tier 2), `BrazilBandRel` wrapping TIDMAD's
upstream code on the frozen 20-file subset (Tier 3, D2), `WeightedResidual(Σ)`
— the likelihood-weighted reconstruction residual under the *true* Σ (Tier 1);
the C4 2×2 matrix with κ_m; conditional-on-alarm AUROC/AUPRC; within-severity
stratification; `ActivationPatch(stage_k)` — substitute the clean stage-k
representation into the perturbed forward pass and measure K recovery.
AC: K functions are pure and unit-tested against hand-computed values; the
conditional AUROC refuses to run pooled across severities (raises unless a
stratum is given); activation patching recovers ≥ 95 % of K on a perturbation
applied *at* stage k (self-consistency control).

### WP7 — `oracle_diag.protocol` + `report` · 3 d · deps: WP0

Deliverables: config schema (dataclasses, validated), `freeze` and hash check,
`mode` enforcement (confirmatory refuses without frozen hash and without
held-out seed lists), experiment registry, data-manifest hashing, bootstrap
over (event_group, perturbation_seed) with the correct nesting, `RESULT.md`
generator (endpoint, estimate, CI, n, config hash, commit), and the
**toy-import guard**: a test that fails if anything under `experiments/`
imports `prometheus_simulation.toy` or any `demo_*` symbol; and the **figure
provenance guard**: every figure under `latex/figures/` produced by the
pipeline carries a sidecar `.json` naming its confirmatory `RESULT.md` and
hashes, and a test fails on any figure whose sidecar points at `results/dev/`
or is missing.
AC: a confirmatory run with a stale hash exits non-zero with a clear message;
a synthetic experiment with known truth gets correct CI coverage (≈ 95 %) under
the nested bootstrap; the toy guard fails on a deliberately planted import.

### WP8 — Tier-1 experiments (ORACLE-Cov) · 6 d · deps: WP1–WP7

Order matters: **B first** (unblocked, a day), then D (sizes everything),
then A and C.

- **cov_B_dissociation.** Subject: compact transformer trained under Σ̂
  (inverse-PSD objective) on `noise_module` data. Arms at matched displacement
  norm at the pooled stage and one mid stage: `null`, `aligned`, `random`,
  `clean`. Monitors: all WP2 + WP4. **What is and is not a test here**
  **[A-B3]**: that `JacobianProjected` and `PullbackWhitened` rank `aligned` ≫
  `null` is *true by construction* (same Jacobian) and is reported as such,
  not as a finding. The falsifiable, pre-registered content is (i) **K**: the
  actual downstream physics loss is ≈ 0 on `null` and large on `aligned` at
  the same norm — i.e. the local linearization predicts the nonlinear
  consequence (threshold: K_null ≤ 0.1 K_aligned, CI excluding 0.5); (ii)
  every magnitude-only monitor (Euclidean displacement, embedding distance,
  MMD, C2ST) ranks `null` ≥ `aligned` — an empirical failure of real
  baselines, not a definition; (iii) `random` falls between. Seeds: 20
  perturbation × 5 model seeds, with model seed as a bootstrap level (§0.5).
  Output: the first citable figure. Depends on `core` only.
- **cov_D_power.** Simulate the null (clean vs clean windows) for every
  endpoint at candidate n; size windows-per-cell for 80 % power at 1 % FAR
  against the smallest severity in the ladder; write the numbers into the
  `counts` part of the pre-registration. **The production event count is an
  output of this step, not an input**; 120k in earlier documents is a
  placeholder. cov_D's null draws use a seed range disjoint from every later
  FAR-calibration seed range (listed in `core`).
- **cov_A_kappa_sweep.** Train under Σ̂; deploy under Σ with
  κ(Σ̂⁻¹Σ) ∈ {1, 3, 10, 30, 100, 300, 1000}, constructed by
  `MultiChannelNoiseGenerator` with declared correlation structure (implied and
  realized covariance both saved). Endpoint: Spearman rank agreement between
  Euclidean displacement and K, and between whitened displacement and K, as a
  function of κ, with CIs. Prediction: the former degrades with κ, the latter
  does not (control, not monotonicity — do not over-commit).
- **cov_C_nsu_waveform.** N = `noise_module` families (PSD change, drift,
  lines, glitches, non-Gaussian tails) at the ladder; S = clean windows from
  declared support tails (amplitude, rise time, pile-up spacing); U = held-out
  families. **The primary C2 contrast is N versus its content-matched clean
  twins** (the audit P1.1 design; WP0 matching spec) **[A-B4]**; the S
  physics families are the *secondary* N/S contrast, each S cell additionally
  matched to its assigned N cell on the same three features so that the
  secondary is not fingerprint recognition either. Endpoints: C1 (power at
  1 % FAR, delay in windows), C2 (macro-F1 on matched cells; ΔF1 vs the
  all-generic arm with CI and the three-way reading; abstention AUROC on U;
  risk–coverage AUC), C3 cost, C4 observational within severity strata.
- **cov_E_patching (C5).** Activation patching at each stage on cov_C's
  consequential N cells: fraction of K recovered when the clean stage-k
  representation is substituted; the localization claim is "stage k* recovers
  ≥ 80 % of K and no earlier stage recovers ≥ 50 %". No adaptation step.

AC: each experiment has `PREREGISTRATION.md` section, `RESULT.md` with CIs,
`mode: confirmatory` artifacts, and passes the toy guard; cov_B's sign
prediction is stated in the pre-registration *before* the confirmatory run.

### WP9 — Tier-2 production and experiments (ORACLE-Paired) · 8 d + cluster · deps: WP1–WP7, D5 for released-number claims only

Deliverables: `prometheus_simulation.prometheus_io` (Prometheus parquet →
`EventPhotons`, with schema test against a real Prometheus output file);
cluster submission from `prometheus_config_pairs` — one injecting run with
the **injection cylinder sized against ARCA, the larger geometry** (D1 caveat
b), one replaying run, both at common site; a **1k-event pilot** that (a)
re-profiles CPU cost and (b) measures the realized match rate of the WP0
matching spec against the overproduced clean pool *before* the full queue;
the full production at the `counts` numbers; strata + matching + export;
GraphNeT dataset conversion; DynEdge inference with WP1 hooks; **E0** (E-fault
gate), **E1** detection (power at 1 % FAR, delay), **E2** attribution on
matched cells + abstention on U1–U4, **E3** cost on DynEdge, **E4** C4
observational (conditional-on-alarm AUROC on angular error within severity
strata), **E5** the pre-registered prediction table tested once.
**Gate for the Tier-2 confirmatory run**: the D5 CPU-vs-GPU re-inference test
is run first; if the 0.944° offset persists it must be shown stable across
severity strata on dev data (the proposal's own gate condition), otherwise the
offset is a confound on K and E4 does not run. This is stricter than
`EXPERIMENT_DESIGN.md`'s earlier sequencing, which gated only quoted NuBench
numbers; angular error as K makes the inference path part of the experiment.
AC: pilot reproduces exactly on rerun (water/olympus); provenance columns
complete for 100 % of events; the pilot match rate is ≥ 90 % inside the
caliper, else the overproduction factor is raised and the change recorded
before production; E5 reports per (monitor, family, severity) the registered
prediction and the outcome, scored as one summary statistic, with no rows
added, removed or reworded.

### WP10 — Tier-3 (TIDMAD) · 3 d · deps: WP1–WP6

Deliverables: the two Σ̂ trainings (MSE, inverse-PSD) with identical seeds and
data; monitors on both; `K_dev` disclosed as non-scientific; `K_rel` on the
frozen 20-file subset, identical list both arms; 328-band truncation and
in-band file selection declared.
AC: the two arms differ *only* in the loss; `K_rel` file list is hash-frozen;
the monitor contrast between arms is reported with CIs.

### WP11 — Paper assembly · ongoing

Each figure/table in `latex/` names the experiment id and `RESULT.md` it comes
from (the WP7 sidecar); a script regenerates every figure from
`results/confirmatory/` and CI fails if any sidecar is missing or dev-sourced;
the proposal's E-matrix is replaced by the realized one; refuted predictions
appear in the paper text, not only in `RESULT.md`.

---

## 4. Order and gates

| Milestone | Contents | Gate to pass |
|---|---|---|
| **M0** | WP0 `core` written and **frozen**; WP7 skeleton | §A review: no blockers; `core` hash recorded |
| **M1** | WP1, WP2, WP3 with analytic tests | all AC green; positive-control ladders monotone |
| **M2** | cov_B confirmatory (depends on `core` only), WP4–WP6 | cov_B's K-prediction registered in `core` *before* the run; confirmed or refuted — either is reportable |
| **M3** | cov_D → freeze `counts`; cov_A, cov_C, cov_E; fill and freeze `bridge` | `counts` and `bridge` hashes recorded |
| **M4** | WP9 pilot (cost + match rate) + production, WP10; D5 test | pilot exact-replay check; D5 resolved or offset shown stable across strata |
| **M5** | E0–E5 run **once** in confirmatory mode | §B review: no blockers |
| **M6** | WP11 | every figure traces to a confirmatory `RESULT.md` via sidecar |

Nothing in M4–M5 starts before `bridge` is frozen. That ordering is the
scientific content of the design, not a project-management preference.

**Schedule reality.** WP0–WP10 sum to ~40 focused engineering days on a
near-linear dependency chain, with `oracle_diag`, `prometheus_io` and the
§3.2 reimplementation all from scratch. Plan on **4–6 calendar months**. If
it has to shrink, cut in this order: (1) the second geometry (keep ORCA only;
keep the bridge on N/S families — the layout claim moves to future work);
(2) cov_E / C5; (3) WP10's `K_rel`, keeping only the two-Σ̂ monitor contrast.

---

## 5. The pre-registration bridge (filled at M3, tested at M5)

For each monitor in the frozen set and each ORACLE-Paired family, one row,
written *before* any Tier-2 data is looked at, with a **decision rule that two
readers score identically** **[A-B10]**:

| Monitor | Family | Severity | Stratum | Endpoint | Threshold | Direction | Basis |
|---|---|---|---|---|---|---|---|
| Euclidean displacement (pooled) | N1 module loss | 25 % | all | power at 1 % FAR | ≥ 0.80 | fires | cov_C N ladder |
| Euclidean displacement (pooled) | S1 low-E, matched to N1@25 % | — | matched | power at 1 % FAR | ≥ 0.80 | fires (**false attribution**) | cov_C content confound |
| Jacobian-projected (pooled) | S1 low-E, matched to N1@25 % | — | matched | power at 1 % FAR | ≤ 0.10 | quiet | cov_B |
| … | … | … | … | … | … | … | … |

Rows whose outcome is analytic by construction (`is_analytic_for`) are marked
and excluded from the summary statistic. The E5 result is **one number**: the
fraction of non-analytic rows whose outcome matches the registered direction
at the registered threshold, with a binomial 95 % CI, plus the full table. No
row may be added, removed or reworded after `bridge` is frozen; the git
history of the file is part of the evidence. A monitor that predicts
out-of-sample failures in a testbed it never saw is the strongest form these
claims can take.

---

## 6. Definition of done for the whole study

- Every claim C1–C5 has a confirmatory `RESULT.md` with CI, config hash and
  commit; every figure traces to one.
- The toy guard, the freeze check and the mode enforcement all pass in CI.
- §B review reports no blocker.
- The pre-registration table has a recorded outcome for every row.
- Negative or equivocal results are in the paper with their equivalence
  margins, not in a drawer.

---

## 7. Scope decisions forced by the §A review — for the collaboration

1. **C5 is patching-only.** The claim ladder keeps C5 (stage localization) but
   tests it by activation patching alone (cov_E). LoRA repair — the "diagnosed
   stage adapts better than a wrong stage" experiment — is out of scope for
   this paper. Reason: it had no work package, Hase et al. make adaptation
   gains a weak validator anyway, and it is first on any cut list.
2. **D5 gates Tier-2 confirmatory, not just quoted numbers.** Because angular
   error is the consequence variable, an unexplained 0.9° offset in the
   inference path is a confound on K. The CPU-vs-GPU test runs before E4, and
   if the offset persists it must be shown stable across severity strata.
3. **The proposal's Phase 2 must change from SPINE to Panda** to match D3
   (`paper3_proposal.tex` still says "SPINE/Graph-SPICE transfer"). _[16 Sep:
   stale — the proposal reads "Panda scale transfer … SPINE remains the later
   domain-specific extension"; done.]_
4. **Deployability is claimed narrowly.** Tier 2 shows transfer to a testbed
   and model the mechanism tier never saw, on declared families; it does not
   show deployability against undeclared real shifts. The abstention endpoint
   on U1–U4 is what supports the weaker statement, and the paper says so.
5. **"Licence-clean" is conditional** on no NuBench artifact or released
   parameter set entering the production. Our §3.2 uses the paper's stated
   values and two declared assumptions (TTS window, noise rate); nothing is
   read from NuBench files. Record this in the dataset README, and still ask
   GraphNeT for a licence on their artifacts.

---

## 8. MLST revision plan — 16 September 2026

### 8.1 Purpose and current evidence

Turn the research review into a coherent manuscript and an executable test of
the incremental scientific value of representation monitoring. This section
extends the existing work packages; it does not replace the canonical design
or create a second experiment programme. Baseline checkout: `f24430a`.

Working thesis, to be tested rather than asserted as a result:

> Under declared acquisition and training-support interventions, determine
> when frozen representations add information beyond acquisition-quality and
> input/output monitors, and whether that information improves identification
> of scientifically consequential reconstruction failures.

The earlier conversational review read an older checkout. In this checkout,
`src/latent_monitor/`, the designed contrasts, linear estimators, subject
interfaces, HeST integration, and Tier-1 development results already exist.
Audit and extend them rather than recreating them. In particular:

- The 6 September report finds that noise-only statistics distinguish several
  acquisition and event interventions. These belong in the strong generic
  baseline; attributing its success to internal representations is invalid.
- The report records large covariance-mismatch alarms with little consequence,
  and flat activation-patching results on the linear subject. Preserve these
  findings and their limitations.
- The reported 13/14 table is development evidence, not confirmation of the
  paper's general claims. Trained-transformer and public-data results remain
  subject to their actual gates.
- The proposal, canonical design and TODO disagree about C4 conditioning and
  claim priority. Resolve these explicitly before writing new runners.

### 8.2 Claim hierarchy and scope

Keep existing claim IDs for traceability. Make **C2 incremental attribution**
and **C4 scientific consequence** the two central questions, each with one
declared primary endpoint. C1 is a detection prerequisite; C3 is supporting
cost evidence. C0 physical-variable organisation and C5 patching/localization
are supporting mechanistic analyses unless independently justified as core.
Do not infer physical axes from probe accuracy alone or stage localization
from uniform recovery at all patched stages.

Minimum evidence package: controlled waveform mechanism study, one feasible
realism validation, and a bounded public-data check if accessible. Choose the
realism arm using existing readiness and licence/parity gates; do not silently
replace the current HeST/LUCiD/Prometheus sequence with the older NuBench-only
recommendation. Document which conclusions are lost when an arm is deferred.
Additional architectures, geometry campaigns, SPINE/Panda transfer, and broad
efficiency optimization are outside this revision's critical path. Preserve
existing work as optional rather than deleting it.

### 8.3 Ordered revision packages

#### R0 — Reconcile the current state (extends WP0)

Read the live documents, implementation, existing reviews and actual result
artifacts. Produce a compact status/traceability table linking each claim to
its endpoint, code, evidence level, unresolved gate and next action. Record
contradictions and their resolutions in a dated note under `docs/reviews/`.

Explicitly map namespaces: proposal **E** means evaluation-contract fault,
whereas the controlled-variable table also uses **E** for event/physics
variation. Use unambiguous labels in new schemas and reporting; preserve old
artifact identifiers through a documented mapping. Likewise distinguish the
covariance condition number κ from the scientific harm threshold κ_m.

Acceptance: no implemented component is called missing without inspection;
no development result is upgraded to confirmatory; no unresolved collaborator
decision is represented as approved. Drafting and local development proceed
without treating a pending freeze decision as a blocker to all work.

#### R1 — Revise the scientific narrative and prior-art positioning (WP0)

Update `EXPERIMENT_DESIGN.md` Part I and the proposal together. Present:
scientific decision → competing explanations → added value of representations
→ controlled mechanism → independent consequence → transfer and limitations.
Replace unconditional language such as an alarm "can be made to rank"
consequence with a hypothesis and its conditions until evidence supports it.

Verify and discuss the following primary sources, using the existing local
paper collection first where available:

- MLST scope: https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-technology/
- Rabanser et al., *Failing Loudly*: https://arxiv.org/abs/1810.11953
- Zhang et al., *Why did the Model Fail?*: https://proceedings.mlr.press/v202/zhang23ai.html
- Guillory et al., *Predicting with Confidence on Unseen Distributions*: https://arxiv.org/abs/2107.03315
- Garg et al., *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance*: https://arxiv.org/abs/2201.04234
- Angelopoulos and Bates, conformal prediction: https://arxiv.org/abs/2107.07511

Integrate with the existing physics references and Paper-1 boundary. Novelty
must exceed observing that distribution distance and damage can disagree.
Do not claim a first or an exhaustive literature search without evidence.

Acceptance: a short contribution paragraph names what the experiment can
establish beyond these works and states what observation would refute it.

#### R2 — Freeze-ready attribution design and information contract (WP1–WP4)

Specify and implement, where absent:

1. A feature-availability manifest distinguishing reference fitting,
   development labels, alarm-time measurements, delayed-label monitoring,
   and evaluation-only truth. Noise-only records are operational features
   only when the acquisition actually supplies them; otherwise label the
   corresponding arm privileged/oracle and report a feasible alternative.
2. Identical data, calibration windows, tuning budgets and available side
   information for generic and layerwise comparisons. The generic arm must
   include available noise-only quality statistics. Add a noise-only ablation
   so attribution gains can be assigned to their actual source.
3. Difficult N/S contrasts with overlapping multiplicity, signal strength,
   or generic shift score, alongside the inclusive population. Declare
   matching variables, dev-chosen calipers, overlap failures and retained
   fraction. Do not hide unmatched cases or interpret matching as a universal
   causal identification result.
4. Group splits by underlying event/injection identity: all geometries,
   replays and corruption variants stay together. Separate reference fit,
   development/tuning, clean calibration and final evaluation. Hold out
   intervention types, severity ranges and seeds as declared; unknown-family
   examples do not enter tuning or calibration.
5. Separate origin (N/S/mixture/unknown) from harm (benign/harmful). Include
   clean rare harmful events and mild benign corruption where supported.
6. Define abstention through feature novelty and report risk–coverage curves,
   unknown-family detection and retained coverage. State the exchangeability
   assumptions; ordinary split conformal does not guarantee unknown-shift
   rejection. Retained coverage is a reporting point, not a threshold tuned
   on confirmatory labels.

Acceptance: tests fail on overlapping event groups and evaluation-only feature
leakage; a tiny deterministic fixture exercises all five comparison arms and
the noise-only ablation, with provenance. Distinguish declared conditional
attribution from identifying unknown real causes.

#### R3 — Make C4 test scientific harm across the full evaluation set (WP2–WP4)

Recommend **AUROC for K ≥ κ_m across all held-out intervention cells** as the
single primary ranking endpoint. Define the cell sampling/weighting scheme in
advance; report AUPRC with prevalence, missed-harm rate at the alert budget,
and false rejection of valid rare events as secondary endpoints. Conditional
strong-alarm analysis remains secondary alarm triage. Add family/severity
breakdowns and the designed dissociation to expose severity confounding;
pooled association alone does not establish the mechanism.

Define K, baseline subtraction or normalization, units, uncertainty and κ_m
per arm. Prefer an independent physical endpoint (amplitude bias/resolution,
signal recovery, angular error) alongside the diagnostic weighted residual.
Do not use the same weighted training objective as the sole demonstration of
scientific advantage. Harm probabilities, if implemented, require calibration
evaluation; a ranking score need not be described as a probability.

Operational metrics use the assumed/reference covariance; realized covariance
and truth belong to oracle controls or evaluation. Distinguish exact Gaussian
full-covariance likelihoods, diagonal inverse-PSD approximations and quadratic
diagnostics under non-Gaussian noise. Compare per-cell realized weighting with
a fixed reference or physical metric so a changing denominator cannot conceal
damage. Specify zero-baseline and undefined-class metric behavior.

Acceptance: a fixture with harmful low-alarm cells changes the primary result
and missed-harm rate; no conditioning silently drops those cells. Reports
retain all four alarm–harm quadrants, cell counts and independent resampling
units. Pick κ_m from scientific requirements or mark it pending; inventing a
number to make a config pass is not acceptable.

#### R4 — Strengthen the existing mechanism test (WP2/WP6)

Audit `latent_monitor/designed.py` and existing projector/Jacobian code first.
Use norm-matched task-aligned, task-null and random perturbations in the same
declared coordinate system. Specify which frozen head defines sensitivity,
the null-space/rank requirement, how scaling is chosen without evaluation
labels, and the local-linear regime. Check actual output/physical loss, not
only the Jacobian prediction; quantify nonlinear approximation failure.

Norm must be matched in the reference alarm metric being challenged. A
task-aware monitor built using the head is expected to separate constructed
directions; this is a positive control, not independent evidence of transfer.
Make physically supported interventions on a held-out arm the transfer test.

Acceptance: deterministic linear controls recover the analytic null/aligned
answers; unavailable null spaces produce explicit skips, not fake nulls;
nonlinear evidence is labelled smoke, development or trained evaluation
accurately. Existing covariance-alarm/low-consequence findings stay visible.

#### R5 — Statistical gates, reporting and reproducibility (WP3/WP7)

Use a development-only power/precision study to choose independent calibration
and evaluation counts. A hundred clean windows gives only 1%-step empirical
FAR resolution and is not by itself precise validation of a 1% FAR. State
per-window versus run-level error and the dependence assumptions for delay.
Account for monitor selection and multiple layer tests in calibration.

Resample the actual independent event groups, perturbation seeds and model
seeds, respecting crossed versus nested structure. Give each primary endpoint
an interval and a prespecified inference rule; distinguish equivalence,
superiority and inconclusive evidence. Do not promote metrics because they
outperformed baselines on the evaluation set. Retain existing versioned
protocol hashes, manifest hashes, model/data provenance and dev/confirmatory
separation. If preregistration is external, record the dependency explicitly
and provide an unfrozen local draft/interface; never fabricate its hash.

Acceptance: one bounded CPU development smoke generates a traceable report;
confirmatory execution fails closed on missing required freezes or gates.
External data/GPU/licence blockers have exact prerequisites and commands.

#### R6 — Synchronize the manuscript and hand off (WP0/WP7)

Synchronize the proposal, canonical design, this plan, TODO, README and reviewer
prompts. Keep current section and claim IDs usable or add explicit mappings.
Update review instructions that currently prohibit pooled C4 analysis without
distinguishing primary operational ranking from mechanism/stratified checks.

Proposed results structure: (1) protocol and information availability;
(2) generic/noise-only versus layerwise attribution; (3) norm-matched mechanism
control; (4) all-cell consequence results and alarm–harm matrix; (5) independent
transfer; (6) failures, abstention and bounded cost. Empty results are plans,
never fabricated plots or numerical claims.

Acceptance: relevant tests pass, links resolve, LaTeX builds with citations and
references resolved if the toolchain is available, and rendered pages are
visually inspected. Report unavailable checks honestly. Deliver a changed-file
summary, evidence/status table, test commands/results and remaining decisions.

### 8.4 Execution boundary and definition of done

**First implementation pass:** R0–R1, protocol amendments R2–R5, the smallest
missing reusable code/tests needed for one CPU development smoke, and R6.
Reuse completed components. Do not launch full training, large downloads,
confirmatory campaigns, publication, outreach or collaboration messages as
part of this handoff. This boundary makes the revision reviewable without
confusing implementation completion with scientific validation.

**Later evidence pass:** after the relevant gates and scientific decisions
are resolved, run the trained-subject comparison, selected realism transfer,
and public-data check. These remain outstanding even if the implementation
pass succeeds. Collaborator agreement required for a protocol freeze remains
a pending milestone, not a reason to stop reversible draft implementation.

Done means the revised claim → protocol → code → report chain is internally
consistent and locally executable, with every unrun scientific claim clearly
labelled. It does not mean MLST acceptance is established.

### 8.5 Status of the first implementation pass (16 September 2026)

| package | status | where |
|---|---|---|
| R0 reconcile | **done** — traceability table, EF/EV and κ_cond/κ_m namespace, 14 contradictions with resolutions | `reviews/2026-09-16_R0_reconciliation.md` |
| R1 narrative + prior art | **done** — six sources verified (four from `reference/papers/`, two on the web); contribution paragraph with refutation conditions; "can be made to rank" replaced by a conditional hypothesis | `EXPERIMENT_DESIGN.md` §I.1, Part V.1; `latex/paper3_proposal.tex` §1, §4, §6 |
| R2 information contract, splits, origin/harm, abstention | **done (protocol + code + tests)** — manifest and alarm-time contract; group splits with declared hold-outs; five arms + two ablations; origin/harm labels; abstention assumptions stated | `PREREGISTRATION.md` §2–§4; `latent_monitor/protocol/{availability,splits,arms,labels,features}.py`; `tests/test_protocol.py` |
| R3 all-cell C4 | **done (code + tests)**; κ_m **pending** on every arm (dev uses the proposal's provisional 10 %) | `protocol/consequence.py`; `PREREGISTRATION.md` §1, §4 |
| R4 designed control | **done** — norm matching in a declared metric, explicit `NullSpaceUnavailable`, `linearization_check`, reproducible seeds (the old seed stream depended on the process hash salt) | `latent_monitor/designed.py`; `tests/test_designed_and_smoke.py` |
| R5 gates, smoke | **done** — `dev`/`confirmatory` with fail-closed freeze/threshold checks; provenance; one CPU smoke (~7 s, 56 kB) | `protocol/mode.py`, `protocol/smoke.py`; `results/latent_monitor_smoke_dev/` |
| R6 synchronise | **done** — proposal rebuilt (7 pages, references resolved), design, plan, TODO, README, review prompts, self-review | this file; `REVISION_REPORT_2026-09-16.md`; `reviews/2026-09-16_mlst_revision_self_review.md` |

Not done and not started, by design (§8.4): trained-transformer comparison,
any realism-arm run, public-data check, full FAR/power study, freeze of any
part, collaborator messages.

### 8.6 Second pass — the two-claim revision (16 September 2026, same day)

The independent review of §8.5 (`TWO_CLAIM_REVISION_PLAN.md`) found the paper
still too broad and several statements unsupported. The second local pass
(`REVISION_REPORT_2026-09-16_pass2.md`) narrowed the manuscript to **two
claims**, corrected the mathematical spine (M1–M5) and repaired the protocol
implementation (I1–I13). Status is stated in the vocabulary *interface
implemented / unit-tested control / development smoke / trained-model evidence /
transfer evidence / confirmatory*: everything in this pass is at the first three
levels; nothing is at the last three. Items §8.5 had marked "done" that were
in fact declarative or absent — abstention, scientific resampling, difficult
contrasts, full information-contract enforcement — are now implemented as
interfaces with unit tests and one development smoke, not more. The next gate
is the Phase-B precision study, then the trained-subject run in development
mode.
