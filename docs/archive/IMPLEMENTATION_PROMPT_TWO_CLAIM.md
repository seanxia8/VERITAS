# Implementation prompt — two-claim revision

Copy the prompt below into an implementation agent working in this repository.
The authoritative task specification is `docs/TWO_CLAIM_REVISION_PLAN.md`.

```text
Implement the bounded local pass in
/Users/dowlingwong/Documents/VERITAS/docs/TWO_CLAIM_REVISION_PLAN.md.

Goal
Turn the paper and protocol into a coherent two-claim study:

1. Do intermediate representations add N-versus-S attribution information
   beyond a capacity-matched strong generic monitor with identical alarm-time
   side information?
2. Does a predeclared task-sensitive representation score rank scientific harm
   better than a committed generic score on held-out physical interventions?

Do the implementation and document edits. Do not return another plan.

Repository discipline
- Work in /Users/dowlingwong/Documents/VERITAS.
- Read any AGENTS.md first. Inspect git status and preserve every existing user
  or agent change. The worktree is intentionally dirty. Do not reset, clean,
  commit, push, or edit/remove `Claude outputs/` or unrelated scratch files.
- Read, in order: TWO_CLAIM_REVISION_PLAN.md; the current proposal;
  EXPERIMENT_DESIGN.md; PREREGISTRATION.md; REVISION_REPORT_2026-09-16.md;
  the two 16 September review notes; the complete `src/latent_monitor/` code and
  tests; the smoke artifact; the Tier-1 result report.
- Audit existing implementations before adding replacements. Preserve the old
  dated development artifacts and their legacy schema.

Required work

A. Correct the mathematical spine (M1–M5).
- Separate the measurement reconstruction metric
  M_recon = J_recon^T J_recon (jac_recon is already whitened) from the task
  metric M_task = J_y^T W_y J_y in declared physics-output units.
- Remove dimensionally invalid J_output^T Sigma^-1 J_output statements.
- Rename new-code P_exc/P_unexc semantics to resolved/weakly-resolved; preserve
  legacy reads. Implement a distinct training-support estimator/control before
  using “outside support” operationally.
- Rewrite N/S geometry as conditional signatures, including in-span S and
  structural N counterexamples already observed.
- Restrict invariance claims and add orthogonal/scale/shear controls.
- Make the paper self-contained enough that an unpublished companion paper does
  not carry an uncheckable central theorem.

B. Repair the protocol implementation (I1–I13).
- Implement abstention end to end (clean-only calibration, unknown AUROC,
  risk–coverage AUC/curve, retained-known F1 and counts) or remove abstention
  from the paper's claims and abstract. Prefer implementation if it can be done
  cleanly in this bounded pass.
- Strengthen the information contract beyond name metadata with provenance-
  carrying feature batches or typed builders; add an adversarial leakage test.
- Define `generic_rich`, `intermediate_only`, and `full_intermediate` so the
  primary increment cannot come from whitened input, final z, pre-output,
  output, or duplicated distance transforms. Add hook-drop and matched-capacity
  controls. Keep old arms as development diagnostics.
- Implement a committed generic harm score and a task-sensitive score; report
  paired delta-AUROC as Claim 2 primary. Fix task-specific aligned/null controls
  to target the declared K.
- Replace the C4 event-only bootstrap with a hierarchy whose outer unit matches
  intervention generalisation (family/cell or perturbation seed), with event
  groups inside and model seeds as declared. Refuse inferential claims when
  outer units are insufficient.
- Reserve `reference_fit` for clean subject/reference fitting. Add a distinct
  attribution-training partition or split development into train/tune.
- Add hard matched N/S contrasts, overlap/retention reporting, and a joint
  operational detection→attribution/abstention table containing clean, N, S,
  mixtures, and unknowns.
- Keep designed perturbations linear-only unless a constrained local inverse of
  the nonlinear encoder is implemented and input validity is checked.
- Fix macro-F1 missing-class policy, tied-score AUPRC, weight validation,
  optimiser/partition/class checks, bootstrap diagnostics, high-dimensional
  covariance shrinkage/calibration, confirmatory dependency verification,
  cell/event terminology, and FAR precision handling.

C. Consolidate the documents (D1–D4).
- Rewrite the live proposal around exactly two primary scientific claims.
  Detection, cost, probes, designed controls, Fisher analysis, and patching are
  supporting analyses, not C0–C5 headline claims.
- Consolidate EXPERIMENT_DESIGN.md; remove the “Part V overrides earlier text”
  pattern. Archive superseded prose only if the repository's archive rules
  allow it; git history is sufficient otherwise.
- Update PREREGISTRATION.md, TODO, README, review prompts, implementation plan,
  and the revision report to match actual code status. Do not mark abstention,
  difficult contrasts, scientific resampling, or enforcement “done” unless the
  acceptance tests truly pass.
- Keep the minimum evidence ladder: trained Tier 1 plus one gated transfer arm.
  Mark every additional arm and former claim as supporting or future work.

Testing and review
- Add meaningful tests for every repaired failure. Include analytic dimensional
  checks, orthogonal/scale/shear behavior, adversarial feature leakage,
  disjoint fit/train/tune/calibration/evaluation groups, unknown-only evaluation,
  tied-score AUPRC permutation invariance, missing classes, invalid weights,
  optimiser failure, outer-level C4 resampling, task-specific null/aligned
  controls, covariance stability, and confirmatory dependency gates.
- Run the full latent_monitor suite and a new bounded CPU development smoke.
  Write the new smoke to a new dated directory; do not overwrite the 16
  September artifact. Ensure JSON is strict and results say NOT CITABLE.
- Apply REVIEW_PROMPTS §A to the revised protocol and §B to the implementation.
  Record this as a transparent self-review, never an independent review.
- Build the proposal twice and visually inspect the rendered PDF if the local
  toolchain is available. Report unavailable checks honestly.

Boundaries
- Do not run full GPU training, download large datasets, send collaborator
  messages, freeze a protocol, invent kappa_m or margins, publish, commit, or
  push. Prepare explicit gated commands for later scientific runs.
- Do not reinterpret the current smoke as evidence. Its layerwise attribution
  delta is negative/inconclusive and its layerwise harm AUROC is weak; preserve
  that record.
- Do not add architectures or testbeds to rescue the story. A negative result
  remains reportable.

Deliverables
1. Corrected two-claim proposal and one internally consistent canonical design.
2. Hardened protocol code and targeted tests implementing the bounded fixes.
3. A new dated non-citable smoke and a dated implementation/review report.
4. A final summary listing exact files changed, tests and commands, which audit
   findings were fixed, which remain blocked, and the next scientific gate.

Completion means the local protocol is credible enough to run the trained
subject next. It does not mean the claims are validated or the paper is ready
for MLST submission.
```
