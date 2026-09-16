# Implementation prompt — MLST revision

Copy the following prompt into an implementation agent working in this repository.
The authoritative revision checklist is §8 of `docs/IMPLEMENTATION_PLAN.md`;
this file is a handoff instruction, not a second scientific design.

```text
Implement the MLST revision plan in /Users/dowlingwong/Documents/VERITAS.

Objective
Make the paper and executable development protocol answer: when do frozen
representations add diagnostic information beyond strong acquisition-quality
and input/output monitors, and does that improve identification of scientific
harm? Implement the bounded first pass in docs/IMPLEMENTATION_PLAN.md §8.4.
Do the work, rather than returning another proposed plan.

Start by reading
1. Any applicable AGENTS.md and git status; preserve unrelated user changes.
2. docs/IMPLEMENTATION_PLAN.md §8, then the existing work packages it extends.
3. docs/EXPERIMENT_DESIGN.md (canonical), docs/TESTBEDS.md, docs/TODO.md.
4. docs/RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md and its actual result artifacts.
5. latex/paper3_proposal.tex, README.md, docs/REVIEW_PROMPTS.md and relevant
   existing reviews.
6. src/latent_monitor/ and tests; relevant simulation and tidmad_transformer
   code, existing experiment runners, bibliography and reference manifest.

Reconcile before editing
The conversational review predates this checkout. Do not assume latent
monitoring is absent or rebuild its existing subjects, projectors, designed
contrasts and estimators. The current development results already show that
noise-only statistics separate several families, covariance alarms can have
little consequence, and linear patching does not localize a unique stage.
Preserve those findings and make the noise-only baseline explicit.

Execute R0–R6 in order, using the acceptance criteria in §8. In particular:
- Keep C2 and C4 central, with one primary endpoint each; demote C1/C3 to
  supporting and bound C0/C5. Keep claim IDs and historical result provenance.
- Verify primary prior-art sources and revise the contribution paragraph;
  distinguish the proposal from existing shift-malignancy, attribution and
  unlabeled performance-estimation work. Do not invent citations or novelty.
- Add an alarm-time information contract and feature availability enforcement.
  Give generic and layerwise monitors identical available side information,
  calibration and tuning budgets, including operational noise-only records.
- Enforce event-group splits across all paired variants; specify held-out
  families, severities and seeds, hard matched contrasts and overlap reporting.
- Separate acquisition/support origin from scientific harm. Reconcile E as
  evaluation fault versus E as event variation without breaking old artifacts.
- Evaluate C4 primarily across all held-out cells, with conditional alarm
  triage secondary. Include low-alarm harm, valid-rare-event rejection, family
  breakdowns, declared cell weighting and an independent physical endpoint.
- Distinguish assumed from realized covariance and exact likelihood from
  inverse-PSD approximation. Prevent truth/covariance leakage into monitors.
- Audit and extend the existing norm-matched null/aligned/random control;
  label its constructed nature and local-linear limitations honestly.
- State abstention assumptions and evaluate unknowns empirically; do not
  imply conformal calibration guarantees rejection of arbitrary unknowns.
- Preserve versioned freezes and confirmatory gates. Prepare unfrozen drafts
  for missing decisions, not fabricated thresholds, approvals or freeze hashes.

Implementation boundaries
Use existing modules and interfaces wherever possible. Add only the reusable
code needed to enforce the revised protocol, compute/report the endpoints and
run one bounded CPU development smoke. Write meaningful tests for leakage,
splitting, metric conditioning and analytic controls. Do not build unrelated
architectures or run a full GPU/data campaign. Follow the repo's CPU environment
instructions; do not force the Linux CUDA lockfile onto macOS.

Do not send messages to collaborators, publish, release data, push changes or
claim a protocol is approved. Keep external data, licences, collaborator
agreement and GPU work as explicit gated follow-ups. Do not stop all work
because one external arm is blocked; complete independent local changes.
If a scientific threshold cannot be justified, leave it explicitly pending
and let confirmatory mode fail closed while development mode remains usable.

Review and validation
Apply docs/REVIEW_PROMPTS.md §A to the amended protocol before dependent code
changes and §B after milestones. Record a transparent self-review in a dated
docs/reviews/ note; do not present it as independent review. Resolve local
blockers before proceeding with affected work. Synchronize those prompts with
the amended C4 endpoint so old wording does not recreate contradictions.

Run targeted tests and a deterministic CPU smoke with config/seed provenance.
Keep smoke artifacts small and explicitly developmental. Build the proposal
twice if needed for references and inspect rendered pages; use applicable
PDF skills for rendering/visual QA. Report missing tools honestly rather than
claiming the PDF was verified. Do not manually replace existing numerical
results with invented illustrative outcomes.

Deliver
1. Updated proposal, canonical design, implementation plan, TODO, README and
   review prompts, with consistent claims/endpoints/status.
2. Minimal code/tests/reporting changes supporting the amended protocol.
3. A dated revision report with the claim-to-evidence table, contradictions
   resolved, exact validation commands/results and any blocked checks.
4. A concise final summary: what changed, what was tested, what scientific
   evidence remains unrun, and the next gate. Do not say the study is validated
   or submission-ready merely because the implementation passes tests.
```
