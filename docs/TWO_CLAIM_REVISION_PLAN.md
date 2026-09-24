# Two-claim revision and implementation-repair plan

_16 September 2026. This plan follows the independent review of the first MLST
revision pass. It narrows the paper to two scientific claims and records the
remaining mathematical, protocol, implementation, and evidence problems found
in that pass. It does not freeze a protocol, approve a threshold, or upgrade a
development result. Where this plan conflicts with the current proposal,
`EXPERIMENT_DESIGN.md` Part V, or the unfrozen `PREREGISTRATION.md`, this plan is
the proposed next revision; the conflict must be resolved in the source
documents before any freeze._

## 1. Decision

The present repository is a credible protocol prototype, not yet evidence for
the paper's scientific conclusions. The focused suite passes (58 passed, 1
skipped on 16 September), the new protocol layer fixes several genuine
problems, and the development smoke is correctly labelled non-citable. The
paper nevertheless remains too broad and several central mathematical and
statistical statements are still unsupported or incorrect.

The manuscript will make exactly two primary scientific claims. Detection,
cost, probes, Fisher diagnostics, designed perturbations, and patching become
controls or supporting analyses. They are reported when useful, but are not
numbered as independent claims.

## 2. The two claims

### Claim 1 — incremental attribution value

**Question.** On declared, alarm-time-observable interventions, do intermediate
representations improve acquisition-versus-valid-physics attribution beyond a
strong generic monitor that already receives every operational input-quality,
output, uncertainty, final-embedding, and noise-only feature?

**Primary estimand.** On a frozen hard evaluation set with overlapping generic
signatures, the paired difference

`ΔF1_attr = macro-F1_N,S(full intermediate-layer arm) − macro-F1_N,S(committed all-generic arm)`.

The arm called “intermediate-layer” must contain only information genuinely
absent from the generic arm. Input/whitening hooks, output hooks, final-embedding
duplicates, paired twins, truth, realised covariance, and intervention labels
cannot create the increment.

**Inference.** Keep the existing benefit/equivalence/inconclusive/detriment
reading provisionally: benefit requires a 95% interval above zero and point
estimate at least 0.10; equivalence requires the whole interval inside ±0.05.
These margins remain unfrozen until justified against achievable precision.

**Required supporting results.** Inclusive and hard-matched results; confusion
matrices; family and held-out-severity results; noise-only and feature-capacity
ablations; operational versus privileged variants; selective risk and retained
coverage for unknown-family abstention; an end-to-end table including clean,
known N/S, mixtures, and unknowns so a good conditional N/S classifier is not
mistaken for a useful deployed decision rule.

**Refutation.** Equivalence or detriment on the hard contrasts, or an apparent
gain attributable to extra input/output transformations, noise-only records,
feature count, event identity, or unavailable side information.

### Claim 2 — incremental scientific-harm ranking

**Question.** Does a predeclared task-sensitive representation score rank
scientific harm better than the committed generic monitor on held-out,
physically generated intervention cells?

**Primary estimand.** The paired difference

`ΔAUROC_harm = AUROC_K≥κm(task-sensitive representation score) − AUROC_K≥κm(committed generic score)`

over all held-out intervention cells under a frozen cell-weighting scheme. The
absolute AUROCs, AUPRC and prevalence, four alarm–harm quadrants, missed-harm
rate, and false rejection of benign valid-rare cells remain required supporting
results. Conditional strong-alarm triage remains secondary.

The generic score and task-sensitive score must be fixed without evaluation
labels. If either is learned from harm labels, it is fitted/tuned only on
development intervention families and evaluated on held-out families,
severities, perturbation seeds, and model seeds. Selecting the better of several
generic scores after evaluation is prohibited; either commit one score or
account for selection inside development.

**Refutation.** A paired interval for ΔAUROC that includes zero without meeting
a predeclared equivalence rule, systematic low-alarm/high-harm failures, or a
gain that disappears under held-out physical families. The output-null/aligned
construction is a positive control and cannot establish this claim.

## 3. Correct the mathematical spine before extending code

These are blockers because the present proposal uses them to motivate both
claims.

### M1 — separate measurement and task pullbacks

The proposal currently writes `J_hᵀ Σ⁻¹ J_h` for a physics-output map even
though Σ is a measurement-space covariance. The code already exposes two
different Jacobians and should govern the correction:

- reconstruction/measurement sensitivity:
  `M_recon = J_reconᵀ J_recon`, because `jac_recon` already returns
  `Σ_hat^{-1/2} ∂x_hat/∂r`;
- task sensitivity:
  `M_task = J_yᵀ W_y J_y`, where `W_y` is a declared metric in physics-output
  units, independent of measurement Σ.

State dimensions, maps, units, covariance assumptions, and locality explicitly.
Neither metric is automatically a measure of realised scientific harm; Claim 2
tests that connection empirically.

Acceptance: dimensional unit tests; no `J_outputᵀ Σ⁻¹ J_output` remains unless
the output is explicitly a reconstructed measurement with matching dimension.

### M2 — distinguish resolvability from training support

`P_exc/P_unexc` derived from `jac_recon` measure locally resolved versus weakly
resolved representation directions. They do not by themselves identify which
directions the training distribution excited. Rename them accordingly in new
code and reports, preserving a legacy mapping for old artifacts.

Estimate training support separately, for example with reference-latent density,
distance or a simulator-known physical tangent span. Validate the estimator on
constructed in-support and out-of-support controls before using it for
abstention. Fisher rank can support a local identifiability statement; it is not
proof that an event is outside training support.

Acceptance: the manuscript allows both in-span and out-of-span valid rare
physics, matching the existing `S_in_span` result; no universal “S lives in the
unexcited complement” statement remains.

### M3 — make acquisition/support signatures conditional hypotheses

The current claim that acquisition shift remains in the excited span is
contradicted by the development table, where gain drift, channel loss, and
jitter have out-of-span fractions near one. Replace the N/S geometric dichotomy
with a table of conditional signatures:

- covariance-only N may change noise-only residual statistics without a mean
  representation shift;
- structural N may create representation and residual shifts;
- S may be in-support-but-rare or outside estimated support;
- origin and harm remain separate.

The attribution classifier tests whether the collection of signatures is
useful; the signatures do not prove causal origin outside the declared
intervention design.

### M4 — restrict invariance claims

Singular subspaces are invariant to orthogonal basis changes, not arbitrary
invertible reparameterisations. State the exact invariance group for every
statistic. Add controlled orthogonal, scaling, and shear reparameterisation
tests. Either construct an invariant statistic under the required group or
report the dependence as a limitation.

### M5 — bound the companion-paper dependency

The cited Paper 1 is presently an unpublished self-reference. Copy the minimum
lemma and assumptions needed here or make this paper's claims conditional on a
clearly stated result. Verify proposition numbering against the actual build.
Do not let an unavailable companion proof carry the central argument.

## 4. Implementation review of the 16 September pass

### What is good and should be preserved

- Explicit alarm-time versus evaluation-only feature inventory.
- Event-group splitting and exclusion of held-out families from non-evaluation
  partitions.
- Strong generic arm containing operational noise-only features, plus the two
  noise-only ablations.
- All-cell C4 tables retaining low-alarm/high-harm cases.
- Pending/provisional/declared harm thresholds and fail-closed threshold gate.
- Reproducible integer seed stream, explicit null-space skip, and nonlinear
  linearisation diagnostics for the designed control.
- Development labelling, provenance, and preservation of the legacy table.

### Blocking implementation problems

#### I1 — abstention is specified but not implemented

The proposal promises conformal novelty, unknown-family AUROC, selective
risk–coverage AUC, and F1 at retained coverage. `protocol/` currently provides
labels and split exclusion only; it contains no abstention score, calibration,
unknown evaluation, or selective-risk implementation. Either implement and
test this chain or remove abstention from Claim 1 and the abstract for this
paper.

Acceptance: unknown families appear only in evaluation; calibration uses clean
calibration groups; scores and thresholds use no unknown examples; reports
include unknown AUROC, risk–coverage curve/AUC, coverage, retained-known F1,
and counts. Coverage assumptions are stated without claiming arbitrary-OOD
guarantees.

#### I2 — the feature contract is declarative, not data-flow enforcement

`AlarmTimeContract` checks feature names against manually assigned metadata. It
cannot detect an implementation that computes an `alarm_time` feature from
truth or a paired twin. Replace bare feature dictionaries with a feature batch
carrying source/provenance tags, or add builders with typed inputs that make
truth/twin/realised-Σ unavailable. Add an adversarial test in which a
misdeclared implementation attempts to smuggle truth into an allowed name.
Describe the residual limitation honestly: code review remains necessary.

#### I3 — the “layerwise” increment is contaminated

`full_layerwise` currently adds Mahalanobis scores from the whitened input, z,
pre-output, and output hooks as well as intermediate stages. Several duplicate
or non-representational information already present in the generic arm. A gain
could come from a richer nonlinear transformation of input/output rather than
intermediate representations.

Define:

- `all_generic`: current operational generic information;
- `generic_rich`: generic variables plus the same reference-distance transforms
  and comparable feature capacity;
- `intermediate_only`: strictly internal hooks, excluding whitened/raw input,
  final z, pre-output and output duplicates;
- `full_intermediate`: `generic_rich + intermediate_only`.

Claim 1 compares `full_intermediate` with `generic_rich`. Keep the old arms as
development diagnostics only. Add hook-drop and matched-dimension controls.

#### I4 — C4 does not yet test incremental value

The code reports absolute harm AUROC for final-embedding and an uncalibrated
mean of six layer Mahalanobis values. It has no committed all-generic harm score
and no paired ΔAUROC. Implement Claim 2's scores and paired comparison. Calibrate
the combined layer score on clean development/calibration data rather than
averaging heterogeneous Mahalanobis quantities solely because each was divided
by dimension.

#### I5 — C4 uncertainty uses the wrong generalisation unit

The primary estimand is across intervention cells, but the current CI resamples
event groups while retaining the same fixed cells. It measures event-sampling
uncertainty conditional on the chosen interventions and cannot support a claim
about held-out failure families. Use hierarchical resampling appropriate to the
claim: intervention family/cell or perturbation seed at the outer level, event
groups within cell, and model seeds outside or nested as declared. With too few
outer units, report descriptive intervals and state that confirmatory inference
is impossible rather than manufacturing precision.

#### I6 — classifier fitting reuses the `reference_fit` partition

The same group partition used to fit the subject/reference statistics supplies
supervised N/S examples to `fit_arm`. Introduce a distinct `attribution_train`
partition, or split `development` into classifier-train and hyperparameter-tune.
Reference-fit groups must remain clean/reference fitting only. Confirmatory
evaluation stays untouched once tuning begins.

#### I7 — no difficult contrasts or operational joint decision

The smoke evaluates inclusive N/S examples and excludes clean, geometry,
mixtures, constructed controls, and unknowns. Implement the hard matched
contrasts specified by Claim 1 and report overlap/retention. Add a joint
operational evaluation of detection → known-family attribution/abstention so
conditional macro-F1 is not presented as whole-system performance.

#### I8 — the nonlinear designed perturbation lacks a valid inverse

For the linear tied subject, `decode(delta)` produces the intended `delta z`.
For a nonlinear subject, adding `decode(delta)` is not generally a local inverse
of the encoder. Either keep the construction explicitly linear-only or solve a
constrained local inverse problem using the encoder Jacobian and verify input
validity. A small reported linearisation error is a diagnostic, not proof that
the generated input is physically meaningful.

The aligned direction must target the declared physical K. A random direction
in the whole output row space may change timing while leaving the primary
amplitude consequence nearly unchanged, as the smoke already shows. Construct
`target_aligned(K)` and `target_null(K)` using the gradient and metric of the
declared consequence.

### Important robustness problems

#### I9 — metric edge cases need tightening

- Macro-F1 currently silently omits a class absent from truth and prediction.
  Fix a policy: all declared classes contribute with zero division handled as
  declared, or the endpoint becomes undefined with a reason.
- Weighted AUPRC is order-dependent for tied scores because ties are processed
  one item at a time. Aggregate equal-score thresholds before integration and
  test permutation invariance.
- Validate weights for length, finiteness, non-negativity, and positive total
  mass in AUROC/AUPRC and quadrant metrics.
- Check logistic optimizer success, finite parameters, nonempty partitions,
  and presence of every required class; fail with an informative error.
- Bootstrap intervals must record failed/undefined replicates and implement the
  declared hierarchy rather than labelling a simple group bootstrap as the
  complete crossed/nested analysis.

#### I10 — high-dimensional reference distances are under-validated

Per-hook covariance inverses can be singular when hook dimension exceeds the
number of reference events. Replace the ad hoc pseudoinverse with a declared
shrinkage or dimension-reduction procedure fitted only on reference data.
Calibrate each hook score to a common clean-null scale before combination. Add
stability tests across reference sample size and representation rescaling.

#### I11 — confirmatory gates do not yet establish a reproducible run

The freeze record stores a commit but `require_gates` does not verify the
current commit, dirty state, environment lock, data manifest, counts freeze,
bridge freeze, model/checkpoint hash, or result destination. A run can be
labelled citable after only a matching protocol-file hash and declared κ_m.

Confirmatory mode must verify all dependencies required by that run, reject a
dirty or mismatched code state unless a full source-tree hash is frozen, record
data/model/environment hashes, and write only to a confirmatory result path.
Do not freeze anything while the scientific decisions remain pending.

#### I12 — terminology overstates cell-level quantities

`valid_rare_event_rejection` currently operates on cells, not events. Rename it
to `valid_rare_cell_rejection` or add the actual event-level estimator. State
whether K and the alarm are event-, window-, cell-, or run-level everywhere.
The C4 alarm threshold currently begins as a per-window 1% threshold and is then
applied to a median cell alarm; justify that decision rule or calibrate a
cell-level threshold separately.

#### I13 — the claimed 1% FAR remains unresolved

Run a development precision study before choosing calibration counts. Report a
binomial or dependence-aware interval for realised FAR. Define non-overlapping
windows, temporal dependence, multiplicity across monitors, and run-level false
alert. The current smoke's 18 clean calibration windows are only a plumbing
check.

## 5. Documentation repair

### D1 — replace the six-claim ladder

Rewrite the proposal's Claims section around Claim 1 and Claim 2 only. C0, C1,
C3, and C5 become named supporting analyses without success thresholds that
read like additional headline claims. Remove C5 drift between the proposal and
design.

### D2 — consolidate instead of layering amendments

`EXPERIMENT_DESIGN.md` currently preserves contradictory earlier text and says
Part V governs. Rewrite the live sections so there is one current definition.
Move superseded text to the archive or rely on git history. The reader should
not need precedence rules to know the protocol.

### D3 — align status language with actual code

Remove “done” for abstention, scientific resampling, difficult contrasts, and
full information-contract enforcement. Distinguish interface implemented,
unit-tested control, development smoke, trained-model evidence, transfer
evidence, and confirmatory result.

### D4 — narrow the evidence ladder

Minimum viable paper:

1. controlled Tier-1 study on a trained nonlinear subject;
2. one independent transfer arm selected by readiness and scientific validity;
3. the two primary paired comparisons and their failures/limitations.

TIDMAD is an external check if its physical K is defensible. Additional
geometries, C0 estimator classes, C3 optimisation, Panda/SPINE transfer, and
stage localisation are follow-up work unless they directly resolve one of the
two claims.

## 6. Ordered execution

### Phase A — theory and schema gate

Complete M1–M5, D1–D3, and update the Subject interface names without breaking
legacy artifacts. Add dimensional and invariance controls. No trained-model
campaign starts until this passes adversarial review.

### Phase B — protocol correctness

Complete I1–I13 with targeted tests. Produce a new development smoke containing
unknowns, clean windows, hard N/S contrasts, `generic_rich`,
`intermediate_only`, the task-specific harm score, the committed generic harm
score, and paired ΔF1/ΔAUROC. Preserve the existing smoke as dated evidence; do
not silently overwrite it.

### Phase C — precision and feasibility

Run development-only studies for FAR calibration size, outer-unit counts for
C4, model-seed variability, matched-contrast overlap, and nonlinear local
inverse validity. Use these results to decide whether the proposed margins and
confirmatory inference are feasible. Unachievable precision changes the claim
or design before freeze.

### Phase D — scientific evidence

Run the trained Tier-1 nonlinear subject in development mode. If the controls
and hard contrasts pass, select exactly one transfer arm whose licences,
constants, checkpoint parity, and physical K gates are satisfied. Freeze only
after thresholds, held-outs, cell weights, scores, margins, counts, and
dependency hashes are agreed.

### Phase E — manuscript

Report the result according to its sign. A negative outcome is valid if the
positive controls, precision, strong baseline, and transfer design pass. State
that intermediate hooks do not justify their cost when equivalence or detriment
is supported. Do not add architectures or claims to rescue a negative result.

## 7. Definition of done for the next implementation pass

The next local pass is complete when:

- the mathematical spine is dimensionally correct and conditional claims match
  the observed counterexamples;
- only two scientific claims remain in the live proposal and canonical design;
- abstention is either implemented end to end or removed from the claims;
- Claim 1 compares truly intermediate information with a capacity-matched
  generic baseline on hard contrasts;
- Claim 2 reports paired ΔAUROC against a committed generic score with an
  interval at the correct independent levels;
- reference fitting, attribution training, tuning, calibration, and evaluation
  use disjoint event groups;
- metric edge cases, optimiser failures, covariance estimation, gates, and
  terminology have targeted tests;
- a bounded development smoke passes and is explicitly non-citable;
- no threshold, freeze, trained-model result, transfer result, or scientific
  conclusion is invented.

Passing this definition makes the protocol credible enough for the next
scientific run. It does not make the paper validated or submission-ready.
