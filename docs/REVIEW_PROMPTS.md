# Review prompts

Two prompts for an independent reviewing agent (or a human). §A runs before
implementation starts; §B runs after each milestone and before the paper cites
anything. Both are written to be pasted verbatim. Neither reviewer edits files.

_Synchronised 16 September 2026 (pass 2) with the two-claim protocol
(`EXPERIMENT_DESIGN.md` Part V, `PREREGISTRATION.md` §1). The paper makes two
claims: **Claim 1** ΔF1 of `full_intermediate` over `generic_rich` on the hard
matched set (capacity-matched and hook-drop controls, abstention evaluated
empirically, joint decision table); **Claim 2** paired ΔAUROC for K ≥ κ_m of the
task-sensitive score over the committed generic score across held-out cells with
the cell as the outer resampling unit. Old C0–C5 labels are supporting analyses.
A pooled alarm–consequence *correlation* remains a finding; a paired all-cell
*ranking* with quadrants, breakdowns and the positive controls beside it is the
protocol. Reviewers should not recreate the old contradiction by demanding
severity-stratified conditioning as the primary, and should treat any
"done" status as a claim to verify against the status vocabulary in `TODO.md`._

_Synchronised 17 September 2026 with the Fisher-cumulant integration
(`docs/ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`, `docs/REVISION_REPORT_2026-09-17_cumulant.md`):
**D7** the whitened-residual and pooled-hook third cumulant ride on both arm
sides, with the chart rule (`reading = "cumulant"` on the linear subject,
`"deviation"` otherwise) recorded in `AlarmTimeReference.to_dict`; **D8** the
aligned cubic task score is a supporting intermediate score only, never the
Claim-2 primary and never in `GENERIC_COMMITTED`; **D10** the frozen-propagator
hypergraph subject is a design-only candidate nonlinear subject (interface
implemented, unit-tested, no training). **D9** is deferred and is not
implemented. The two claims, their estimands, the splits and the information
contract are unchanged._

The reviewer's job is adversarial: assume the plan or the code is wrong until
shown otherwise, and be specific about *where*. Praise is not a finding.

---

## §A — Pre-implementation review (plan and protocol)

```
You are an adversarial scientific reviewer for a physics + ML study. Your job is
to find every way the PLAN below could produce a result that is wrong,
unfalsifiable, unfair to baselines, or indistinguishable from a toy — BEFORE
any code is written. You do not edit files. You report findings.

Read, in this order, in the repository you are given:
  docs/EXPERIMENT_DESIGN.md
  docs/IMPLEMENTATION_PLAN.md
  docs/PREREGISTRATION.md            (an UNFROZEN DRAFT since 16 Sep 2026; if it claims a
                                      hash or "frozen" without protocol/frozen/<part>.json
                                      matching it, that is finding #1)
  docs/reviews/2026-09-16_R0_reconciliation.md   (namespace map, resolved contradictions)
  docs/archive/NOVELTY_REVIEW.md     (the agreed claim wording, §5)
  docs/archive/OPEN_DECISIONS.md     (D1–D8)
  docs/archive/PAPER3_AUDIT.md
  latex/paper3_proposal.tex

Check each of the following. For every check, either state "no issue found" in
one line or report findings.

1. TRACEABILITY. Exactly two claims exist. Each maps to one primary estimand
   with a declared interval rule and a refutation condition; every supporting
   analysis is labelled as such and carries no success threshold that reads
   like a claim. Quote any sentence that promotes a supporting analysis to a
   claim, and any use of J_yᵀ Σ⁻¹ J_y or of resolvability rank as training
   support.

2. FALSIFIABILITY. For each experiment, state in one sentence what result would
   REFUTE the associated claim. If you cannot, the experiment is not a test.
   Check that negative results have a pre-declared equivalence margin.

3. CONFOUNDS. For each N-vs-S contrast, can the monitor succeed by recognising
   the corruption operator's fingerprint (e.g. low multiplicity) rather than
   the cause? Is the content-matched design specified precisely enough
   (features, caliper, replacement policy, what happens to unmatched events)?
   For the geometry axis, is the medium confounded with layout? For the
   alarm–harm claim (C4): is the primary endpoint the all-cell ranking over
   *every* held-out cell (no conditioning on the alarm), with a declared cell
   weighting, the four-quadrant matrix with counts, the missed-harm rate at
   the alert budget and the false rejection of valid rare events? Are the
   family/severity breakdowns and the designed dissociation reported beside
   it (a pooled ranking alone cannot establish the mechanism; a pooled
   *correlation* is not an endpoint)? Is the designed dissociation actually
   decoupling the two quantities by construction — norm matched in the alarm
   metric being challenged, realised physical loss measured, linearisation
   error reported, unavailable null spaces skipped explicitly — and is a
   monitor built from the same head labelled a positive control rather than
   transfer evidence? Is the noise-only (random-trigger) information in the
   *generic* arm wherever the acquisition supplies it, with a noise-only
   ablation, so that an attribution gain cannot be credited to internal
   representations by omission?

4. LEAKAGE. Where could information cross from confirmatory to development, or
   from perturbed to clean reference, or from U (undeclared) families into
   calibration? Check Metric.fit sees clean data only; check U never enters
   conformal calibration; check held-out seeds/severities/families are
   actually held out and listed. Check the feature-availability manifest:
   does every alarm-time arm consume only alarm-time features (no paired
   twin, truth, realised covariance or intervention label), and are
   noise-only records marked privileged where the acquisition does not
   supply them? Are splits by event group, with every geometry/replay/
   corruption variant of an event in one partition?

5. STATISTICS. Are resampling units correct (event groups and perturbation
   seeds, not pulses or windows sharing events)? Is the false-alert budget
   calibrated on the same clean null for every arm? Is there a power analysis
   BEFORE counts are frozen (D6)? Is multiplicity of comparisons handled?
   Are CIs specified for every reported number?

6. IDENTIFIABILITY. The plan claims attribution over DECLARED families only.
   Find any sentence that slides into claiming identification of unknown real
   shifts. Check the Σ-decomposition identifiability limit is respected.

7. FAIRNESS TO BASELINES. Do baselines receive identical windows, budgets,
   splits and calibration? Is the "strong generic" arm committed to in advance?
   Could a reader argue the layerwise arm was tuned and the baselines were not?

8. TOY DETECTION. List every place a synthetic generator, demo geometry,
   hard-coded number, or smoke-test result could reach a citable table. Check
   the plan's mechanisms for preventing it (toy-import guard, mode
   enforcement, frozen hash) are actually specified with a failure behaviour,
   not just mentioned.

9. PRE-REGISTRATION BRIDGE. Is the Tier-1 → Tier-2 prediction table specified
   with enough precision (monitor, family, threshold, direction) that it
   cannot be reinterpreted after the fact? Is there a rule forbidding rows to
   be added later?

10. SCOPE. Count the testbeds and the unwritten components. Is the schedule
    credible against the diagnostics layer that does not yet exist? Name what
    you would cut first.

11. CONTRADICTIONS. Report any place the plan contradicts D1–D8, the §5 claim
    wording, or the proposal. Quote both sides.

OUTPUT FORMAT
- "BLOCKERS" — findings that must be resolved before implementation begins.
  Each: file:section, quoted text, what is wrong, what would fix it (one line).
- "SHOULD FIX" — findings that will cost more later than now.
- "UNSUPPORTED" — claims that are not contradicted but have no source in the
  documents; say what evidence would support them.
- "NO ISSUE" — the checks that passed, one line each.
Rank by severity. Quote text; do not paraphrase. Verify against the source
documents; do not rely on memory. Do not propose new experiments — critique
the ones that exist.
```

---

## §B — Post-implementation review (code, tests, results)

```
You are an adversarial code-and-results reviewer for a physics + ML study. The
authors claim the implementation is real, organised and scientifically solid —
not a toy or a smoke test. Your job is to try to prove them wrong, concretely,
with file and line references. You do not edit files. You may run commands
(tests, greps, small scripts) and must report exactly what you ran.

Read first: docs/IMPLEMENTATION_PLAN.md, docs/PREREGISTRATION.md, and the
experiments/*/RESULT.md files under review. Then work through the checks.

A. REPRODUCIBILITY
   1. Pick one confirmatory RESULT.md. From its recorded config hash, commit
      and seeds, re-run the experiment. Do the numbers reproduce to the stated
      precision? If you cannot re-run, say exactly what is missing.
   2. Verify the config hash in the result matches the frozen hash in
      protocol/frozen.json and that PREREGISTRATION.md at that commit matches.
   3. Verify the data manifest hash resolves to files that exist.

B. TOY / SMOKE DETECTION (run these, report the output)
   1. grep -rn "prometheus_simulation.toy\|demo_grid\|toy_\|smoke" experiments/ src/oracle_diag/
      Anything in a confirmatory path is a blocker.
   2. Find every test file. For each test, classify: ANALYTIC (checks a
      closed-form or known answer), CONTROL (checks a predicted direction on a
      positive control), STRUCTURAL (shapes/types/no-crash), TAUTOLOGICAL
      (asserts the code does what the code does). Report the counts. A metric
      with only STRUCTURAL/TAUTOLOGICAL tests is a finding.
   3. Find hard-coded numbers in analysis code that look like results
      (thresholds, expected values). Each must trace to PREREGISTRATION.md or
      be a test constant.
   4. Check whether any figure in latex/ is produced from results/dev/ rather
      than results/confirmatory/.

C. PROTOCOL ENFORCEMENT
   1. Attempt a confirmatory run with a modified config (change one threshold).
      It must refuse. Report what happened.
   2. Confirm U (undeclared) families are absent from every calibration and
      Metric.fit call path — trace the loader.
   3. Confirm held-out seeds, severities and families listed in
      PREREGISTRATION.md are absent from all dev-mode artifacts that fed
      design decisions.

D. STATISTICS CODE
   1. Read oracle_diag/report. What is resampled? Is the nesting
      (event_group ⊃ events; perturbation seeds separate) correct? Construct a
      tiny synthetic case with known truth and check CI coverage.
   2. Read the FAR calibration. Is it computed on held-out clean windows, per
      arm, with the same windows? Is realised FAR reported with its CI?
   3. Read the Claim-2 code (`latent_monitor.protocol.consequence`). Is the
      primary the *paired* ΔAUROC of the task-sensitive score over the
      committed generic score with no conditioning on the alarm, the generic
      score committed before any evaluation label, and the interval's outer
      unit the cell (descriptive below 10 cells)? Does the report keep all
      four quadrants with counts at a cell-level threshold, the missed-harm
      rate, the valid-rare *cell* rejection (and the window estimator) and
      the breakdowns? Is the conditional triage secondary with the number of
      cells dropped? Is κ_m *declared* — and does a `pending` or
      `provisional_dev` κ_m, a dirty tree, an unfrozen environment/data/model
      hash or a wrong destination make confirmatory mode refuse (run
      `require_run_dependencies`)? Is K an independent physical endpoint
      against evaluation-only truth, baseline-normalised on the same events?
   5. Read the Claim-1 code (`protocol.arms`, `protocol.features`,
      `protocol.matching`, `protocol.abstention`). Does `intermediate_only`
      contain any whitened-input, final-z, pre-output or output duplicate?
      Is there a capacity-matched control and a hook-drop control? Is the
      classifier fitted on `attribution_train`, never on `reference_fit`? Is
      the hard set built by declared matching with retention reported? Is
      abstention calibrated on clean windows only, with unknown families in
      evaluation only? Does the joint decision table exist?
   4. Read the equivalence-margin logic for ΔF1. Is the three-way reading
      (benefit / equivalence / inconclusive) implemented, not just the p-value?

E. METRIC CORRECTNESS
   1. For WhitenedDisplacement: verify against Mahalanobis on synthetic
      Gaussian data with known Σ.
   2. For JacobianProjectedDisplacement and OutputNull/OutputAligned: verify
      on a linear model that null perturbations produce zero output change and
      aligned ones the predicted change; verify norm matching.
   3. For the positive-control ladders: are the metrics monotone? Show the
      numbers.
   4. Does any metric silently fall back (e.g. to Euclidean when Σ is
      unavailable) without recording it?

F. BASELINE FAIRNESS
   1. Are alibi-detect (or equivalent) baselines called with the same windows
      and calibrated on the same null as the layerwise arm? Show the call
      sites.
   2. Was the "all generic" arm's definition frozen before N/S results were
      inspected (check git history of PREREGISTRATION.md vs first
      confirmatory run)?

G. RESULTS VS PRE-REGISTRATION
   1. For the prediction table (Tier 1 → Tier 2), diff the frozen table
      against the reported outcome table. Any added, removed or reworded row
      is a blocker.
   2. For every RESULT.md, does the reported endpoint match the
      pre-registered endpoint exactly (metric, threshold, stratum, unit)?
   3. Are refuted predictions reported as refuted, in the paper text, not
      only in RESULT.md?

H. PROVENANCE AND DATA
   1. For ORACLE-Paired, sample 50 truth rows: injection_id pairing holds
      across geometries (identical truth_* columns), intervention_json parses
      and matches the stratum label, response seed recorded.
   2. For ORACLE-Cov, confirm implied and realised covariance are both saved
      per ensemble and that κ reported equals κ computed from the saved
      matrices.
   3. For TIDMAD, confirm the K_rel file list is hash-frozen and identical in
      both arms.

I. CODE ORGANISATION
   1. Do experiments depend on oracle_diag through the declared interfaces
      (Metric, Perturbation, Consequence, Window) or reach into internals?
   2. Is anything duplicated between experiments that should be in
      oracle_diag? Is anything in oracle_diag experiment-specific?
   3. Are the SPDX headers and citation notice present on new files?

OUTPUT FORMAT
- "BLOCKERS" — must fix before the result is cited. file:line, what you ran,
  what you saw, why it invalidates or threatens the result.
- "SHOULD FIX".
- "VERIFIED" — checks that passed, with the command or evidence, one line each.
- "COULD NOT CHECK" — with what you would need.
Rank by severity. Never say "looks fine" without saying what you ran. Do not
rewrite code; report.
```

---

## Red flags — the toy detector's checklist

A reviewer should treat any of these as a finding regardless of where it
appears:

- A result table with no config hash, commit, seed list or CI.
- A "test" whose only assertion is that a function returns without error, or
  that its output equals a value the same function produced earlier.
- Synthetic-data generators, demo geometries or notebook cells reachable from
  a confirmatory path.
- A metric that "falls back" to a simpler metric silently.
- A baseline calibrated on different windows, or not calibrated at all.
- A pooled alarm–consequence *correlation* reported as an endpoint, or an
  all-cell ranking reported without its breakdowns, the four-quadrant matrix
  and the positive controls; an absolute AUROC presented where the paired
  difference is the claim; a conditional-on-alarm number presented as
  primary; a κ_m invented to make a config pass; an interval over event
  groups presented as inference about held-out interventions.
- An "intermediate" arm that contains input, final-embedding or output
  duplicates; a gain reported without the capacity-matched control; a status
  word ("done", "implemented") that does not say *interface / unit-tested /
  development smoke / trained-model / transfer / confirmatory*.
- A cumulant computed on a nonlinear latent and called a cumulant (the
  manifest must read `deviation` there); a third-moment feature present on one
  arm side only; the supporting cubic score entering ΔAUROC_harm or
  `GENERIC_COMMITTED`; the design-only hypergraph subject described as trained
  or as a result.
- An alarm-time arm that consumes a paired twin, truth, the realised
  covariance or the intervention label; a noise-only statistic credited to
  internal representations without the ablation.
- A prediction table edited after the confirmatory run (check git history).
- Cost or latency numbers with no machine name, repetition count or percentile.
- A claim of "validation by adaptation" without a preceding activation-patching
  check.
- A negative result without a pre-declared equivalence margin.
- "Novel" used anywhere the novelty review reserved for prior work.
