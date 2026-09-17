# Implementation prompt — Fisher-cumulant integration (17 September 2026)

Copy the prompt below into an implementation agent working in this repository.
The authoritative task specification is `docs/ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`.
Before dispatch, the author ticks the decision items in that note's §5; Blocks C
and D below run only when ticked.

```text
Implement the bounded local pass specified in
/Users/dowlingwong/Documents/VERITAS/docs/ARITRA_CUMULANT_INTEGRATION_2026-09-17.md.

Goal
Add the first non-Gaussian rung of the Fisher-cumulant tower (Bal et al.,
arXiv:2605.03063v2) to the two-claim study as SUPPORTING machinery, and bring
the proposal, the canonical design and the unfrozen pre-registration into
agreement with it. The two claims and their primary estimands do not change.
Do the implementation and the document edits. Do not return another plan.

Author decisions in force for this run (edit before dispatch)
- [ ] D7  pooled third-moment transform enters the arms on BOTH sides (Block A)
- [ ] D8  cubic task score is the registered supporting intermediate score (Block B)
- [ ] D9  R(3) hook-selection diagnostic, reference-fit-only reporting (Block C)
- [ ] D10 frozen-propagator hypergraph subject as a candidate third subject —
          design and interface only, no training (Block D)
An unticked block is not implemented and not mentioned as implemented anywhere.

Repository discipline
- Work in /Users/dowlingwong/Documents/VERITAS on branch dev.
- Read any AGENTS.md first. Inspect git status and preserve every existing user
  or agent change; the worktree is intentionally dirty. Do not reset, clean,
  commit, push, or edit/remove `Claude outputs/`, `_to_delete/`, or unrelated
  scratch files.
- Read, in order: the integration note above; `Claude outputs/
  aritra_ideas_mapping_2026-09-17.md` §3; `latex/paper3_proposal.tex`;
  `docs/EXPERIMENT_DESIGN.md` (Parts III and V); `docs/PREREGISTRATION.md`;
  `docs/TWO_CLAIM_REVISION_PLAN.md`; `docs/REVISION_REPORT_2026-09-16_pass2.md`;
  the complete `src/latent_monitor/` package including `protocol/` and
  `tests/`; `results/latent_monitor_tier1/` and the two 16 September smokes.
- The two source papers are not in the repository. Cite them only through the
  bibliography entries given in the note's §8, checked field by field against
  arXiv before use. Do not paraphrase results from them that the note does not
  quote.
- Audit existing implementations before adding replacements. `input_quality_
  features` already computes a raw-window kurtosis (`in_kurtosis`); the new
  statistics are on the WHITENED RESIDUAL and on the POOLED HOOKS in the
  reference chart, which is a different object. Say so in the docstring.

Required work

A. The whitened-residual and pooled-hook third cumulant (D7).
- In `latent_monitor/statistics.py` (or a new `cumulant.py` beside it) add an
  estimator of the connected third-cumulant tensor of a set of vectors in a
  declared chart, with (i) the per-coordinate third central moments in the
  reference PCA basis (n_pcs coordinates) and (ii) the Frobenius norm of the
  full tensor minus a reference tensor. Add the fourth-order companion as a
  function but do not wire it into any arm. Add Bal et al.'s truncation-error
  ratio (quadratic vs quadratic-plus-cubic KL along a displacement direction)
  as a development-table quantity only.
- In `protocol/features.py`: extend `pooled_stage` to return the pooled third
  moment; add `hook_third_nulls` to `AlarmTimeReference` fitted on
  `reference_fit` exactly like `hook_second_nulls`; add
  `im_{hook}_third_maha` for the channel and token hooks; add the residual
  cumulant scalars to `generic_rich` (names prefixed `gr_resid_c3_`); extend
  `QUADRATIC_BASIS`/the capacity control so `generic_rich_matched` again has
  the feature count of `full_intermediate`. Every new feature carries the
  `{raw_window, reference_fit}` tags and no evaluation-only source.
- Do not touch `GENERIC_COMMITTED`. Do not add any new feature to it.
- Chart rule, enforced in code and docstrings: on the linear subject the
  statistic is a cumulant; on any other subject it is a deviation from the
  reference cell's third moment and is named as such in the manifest
  (`reading: "cumulant" | "deviation"` recorded in `to_dict`).
- Signature row: add the §2.1 conditional-signature row to
  `EXPERIMENT_DESIGN.md` §III.1 as a HYPOTHESIS with its Tier-1 test, and run
  that test on the linear subject under paired replay on the existing Tier-1
  families plus the sparse-burst family (`noise_module` non-Gaussian family).
  Report in the 13/1/0 match format in a new dated results directory. A
  negative outcome is reportable and is not to be explained away.

B. The cubic task score (D8).
- In `task_metric.py` add `TaskMetric.cubic_aligned(delta, I3)` returning
  Δ_a Δ_b Δ_c Î3_abc with Î3 estimated on `reference_fit` z in the M_task-
  whitened chart. In `protocol/features.py` add `raw_task_cubic`; in
  `protocol/consequence.py` calibrate it on `calibration` windows and report it
  ONLY as the supporting intermediate score beside the task length. It never
  enters ΔAUROC_harm's primary comparison and never enters the committed
  generic score.

C. R(3) hook-selection diagnostic (D9) — only if ticked.
- Implement greedy forward selection over candidate feature sets by
  redundancy-suppressed quadratic gain plus aligned cubic completion, scored by
  the retained local divergence fraction R(3) and aligned triplet coverage
  (Bal et al. Eqs. 6.30–6.34), with the direction family taken from the
  designed families (`designed.py`: output_aligned, task_aligned), which are
  linear-subject-only. It runs on `reference_fit` data only and REPORTS R(3)
  per candidate hook set. It does not select the arms in this pass. Raise
  `NonlinearSubjectUnsupported` for nonlinear subjects.

D. Frozen-propagator hypergraph subject (D10) — only if ticked; design and
   interface, no training.
- Write the design into `EXPERIMENT_DESIGN.md` §I.6 and `TESTBEDS.md` §2 as a
  candidate third subject for the nonlinear arm: vertices = channels; P2 from
  the measured noise covariance (the noise-Laplacian PE of the companion paper,
  cite Prop. stationary-pe by name only); P3 from the measured third noise
  cumulant with sign as an edge attribute and |w| in the Laplacian; P2, P3
  frozen from the reference cell; trainable readout only; the QUIVER
  zero-initialised residual gate as the optional trainable correction on top of
  the tied linear AE. State that P3 vanishes for Gaussian noise so the subject
  is non-trivial only on non-Gaussian cells.
- Add `latent_monitor/hypergraph_subject.py` implementing the `Subject`
  protocol (`represent`, `outputs`, `decode`, `jac_recon`, `jac_output`) with
  frozen propagators built from `noise_module` cumulants and an untrained
  readout, plus tests that the Laplacians are correctly normalised (Eq. 4.3)
  and that P3 is identically zero on a Gaussian reference. No training script,
  no GPU run, no result.
- Record in the design that this is one of two nonlinear subjects, both to be
  reported, neither chosen by result, and that it is gated on the trained
  Tier-1 run.

E. Documents.
- `latex/paper3_proposal.tex`: in the mechanism section, one paragraph after
  the measurement metric stating that M_recon is the quadratic rung of the KL
  expansion whose higher rungs are the connected cumulants of the whitened
  residual in the natural chart, exact only there (cite bal2026triality
  Thm 1, Cor 2, App. A), and that the study reports the first such rung as a
  supporting statistic; in the designed-controls text, one sentence citing
  bal2026quiver's paired-seed zero-initialised design as precedent for the
  capacity-matched control. No new claim. Keep the note at its current page
  budget; if a sentence must go to make room, say which and why.
- `docs/EXPERIMENT_DESIGN.md`: the §III.1 row (Block A); Part V.2 one
  paragraph on the cumulant rung and the chart rule; V.4 the new arm features;
  V.5 the supporting cubic score; §I.6/TESTBEDS §2 only under D10.
- `docs/PREREGISTRATION.md`: add decision items 7–10 to §8 with their ticked
  state; add the new features to the §3 information-contract list; add the
  supporting cubic score to §1's supporting results; keep "UNFROZEN" and every
  PENDING value exactly as they are. Do not declare κ_m, margins, severities or
  seed ranges.
- `docs/TODO.md`, `README.md`, `docs/REVIEW_PROMPTS.md`,
  `docs/IMPLEMENTATION_PLAN.md`: update to the actual code status using the
  repository's status vocabulary (interface implemented / unit-tested control
  / development smoke). Nothing is marked trained-model evidence.
- `references.bib`: the two entries from the note's §8, verified.

Testing and review
- Add the acceptance tests in the note's §6: Gaussian-zero with declared
  tolerance and n; orthogonal invariance and shear non-invariance recorded;
  arm-symmetry test that fails if a third-moment feature exists on one side
  only; tag/leakage coverage of every new feature; calibration of the cubic
  score; D10 Laplacian tests if ticked.
- Run the full `latent_monitor` suite (`PYTHONPATH=src python -m pytest
  src/latent_monitor/tests -q`) and a new bounded CPU development smoke into a
  NEW dated directory (`results/latent_monitor_smoke_dev_2026-09-17_cumulant/`).
  Do not overwrite any 16 September artifact. JSON strict; every result says
  NOT CITABLE.
- Apply `docs/REVIEW_PROMPTS.md` §A to the revised design text and §B to the
  implementation; record it as a transparent self-review, never an independent
  review.
- Build the proposal twice from `latex/` and inspect the rendered PDF if the
  toolchain is available; report unavailable checks honestly.

Boundaries
- Do not run GPU training, download data, message collaborators, freeze a
  protocol, invent κ_m or margins, publish, commit, or push.
- Do not change the two claims, their estimands, `GENERIC_COMMITTED`, the arm
  names, the splits, the hard-matching rule, or the information-contract phases.
- Do not compute a cumulant on a nonlinear latent and call it a cumulant.
- Do not add a VQC or any quantum machinery. Do not add any architecture beyond
  Block D, and Block D only as design and interface.
- Do not reinterpret the 16 September smokes or the 6 Sep Tier-1 table.
- If a block cannot be completed cleanly inside this pass, stop that block,
  leave the code in a tested state, and report the exact residual.

Deliverables
1. Code and tests for the ticked blocks, with the chart rule enforced.
2. Revised proposal, canonical design and pre-registration (still unfrozen),
   with decision items 7–10 recorded.
3. The Tier-1 signature-row result for the new hypothesis (development,
   non-citable) and a new dated smoke.
4. `docs/REVISION_REPORT_2026-09-17_cumulant.md`: exact files changed, tests
   and commands with exit codes, which blocks were ticked and completed, which
   remain blocked, and the next scientific gate.

Completion means the supporting cumulant machinery is implemented, tested and
documented consistently across code and text, and the pre-registration records
the decisions it now depends on. It does not mean either claim is validated,
that any signature is confirmed, or that the paper is ready for submission.
```
