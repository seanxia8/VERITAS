# Self-review of the Fisher-cumulant integration — 17 September 2026

_A **transparent self-review** by the implementation agent that made the
changes, applying `REVIEW_PROMPTS.md` §A to the revised design text and §B to
the implementation. It is **not** an independent review and must not be cited
as one. Status words follow `TODO.md`: interface implemented · unit-tested
control · development smoke · trained-model evidence · transfer evidence ·
confirmatory. Everything in this pass is at the first three levels._

Scope: blocks D7 (completed), D8 (completed), D10 (design and interface only).
D9 was deferred by the author and is not implemented.

## §A applied to the revised design text

Inputs: `ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`, `EXPERIMENT_DESIGN.md`
(§III.1 row, §V.2, §V.4, §V.5, §I.6, §V.6), `PREREGISTRATION.md` (§1, §3, §8),
`latex/paper3_proposal.tex` (mechanism, designed controls, bib), `TESTBEDS.md`
§2.6, `TODO.md`, `README.md`, `REVIEW_PROMPTS.md`, `IMPLEMENTATION_PLAN.md` §8.7,
`references.bib`.

**1 TRACEABILITY — no issue found.** No new claim is introduced. The proposal
says the cumulant rung is "not a third claim and changes no estimand"; the
design calls it a supporting statistic of the §III.1 table; the
pre-registration lists it under supporting results and records D7/D8/D10 as
decision items, not as estimands. `GENERIC_COMMITTED` and the Claim-2 primary
are untouched (verified in code, below).

**2 FALSIFIABILITY — no issue found, with one declared caveat.** The §III.1
residual-cumulant row has a refutable prediction (covariance N at reference
after re-whitening; non-Gaussian structure moves; in-span S at reference), and
the Tier-1 row is reported as a development negative at the tested size. The
supporting cubic score carries no threshold, so it is not a test — it is
labelled as such. Caveat: the row's power at `n_eval = 60, n_pcs = 3` is too low
to separate the glitch/sparse-burst families; the design does not claim
otherwise.

**3 CONFOUNDS.** Two are named rather than hidden. (i) The per-window
third-moment feature is noisy (a rank-1 object per window); the population
reading is used in the signature runner. (ii) The chart is the natural chart
only under its stated conditions; off it the manifest reads `deviation`. No
arm-symmetry confound: the transform is on both sides or neither, enforced by
`test_arm_symmetry_and_matched_capacity_after_the_third_moment_transform`.

**4 LEAKAGE — no issue found.** The new features are built in the typed builder
from `{raw_window, reference_fit}` only; the residual cumulant uses the observed
window's own residual; the cubic score uses `reference_fit` `z`. No
evaluation-only source. The adversarial builder test still holds; the new
features are covered by `test_new_features_are_tagged_alarm_time_and_never_evaluation_only`
and by the all-names tag audit.

**5 STATISTICS.** The Tier-1 signature row is a development test: a population
third-cumulant deviation with a 400-draw window bootstrap (q99 z = 2.33). It is
not a Claim-1 or Claim-2 statistic. The supporting cubic score is calibrated on
`calibration` windows like the task length and reported beside it; it never
enters `paired_delta_auroc`.

**6 IDENTIFIABILITY — no issue found.** Nothing here claims identification of
unknown real shifts.

**7 FAIRNESS TO BASELINES.** The capacity control is re-truncated to the new
intermediate count (46 / 46 in the 17 Sep smoke) and includes the residual
cumulant scalars. The QUIVER precedent is cited for the evidence design, not for
quantum machinery; no quantum content is added.

**8 TOY DETECTION — no issue found.** Both new artifacts write `NOT CITABLE`
in the first line and strict JSON; neither is under `results/confirmatory/`.
The 16 September artifacts are untouched.

**9 BRIDGE — unchanged**, out of this pass.

**10 SCOPE.** One paragraph, one sentence of text, one table row, one design
subsection and one design-only module (`hypergraph_subject.py`). Nothing else
was added.

**11 CONTRADICTIONS.** Checked `EXPERIMENT_DESIGN.md` and `PREREGISTRATION.md`
for the chart rule: the pooled-hook third moment reads `cumulant` on the linear
subject and `deviation` otherwise; the whitened-residual third cumulant reads
`cumulant` unconditionally under the Gaussian reference. The `to_dict` records
both, and the proposal paragraph states the same. No contradiction found.

## §B applied to the implementation

Commands (macOS, Python 3.14.6, numpy 2.4.6, scipy 1.18.0):

```
$ PYTHONPATH=src python3 -m pytest src/latent_monitor/tests -q
  94 passed, 1 skipped in 13.02s            # skipped: test_torch_subject (torch absent); baseline was 84 passed, 1 skipped

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_2026-09-17_cumulant
  DEVELOPMENT SMOKE — NOT CITABLE.
  Claim 1 ΔF1 (hard set): 0.0 [0.0, 0.0] → inconclusive (n_windows 20 < 30)
  Claim 2 ΔAUROC: 0.1667 (task 1.0 vs generic 0.8333)
  strict JSON verified (json.load); generic_rich_matched 46 == full_intermediate 46; reading "cumulant"

$ PYTHONPATH=src python3 -m latent_monitor.run_cumulant_signature --out results/latent_monitor_sig_c3_2026-09-17
  12 match, 0 documented, 2 mismatch        # strict JSON; NOT CITABLE

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --mode confirmatory
  ConfirmatoryGateError: freeze part 'core' is missing; κ_m 'provisional_dev' not usable
```

**B.2 test classification** (10 new tests in `tests/test_cumulant.py`):
ANALYTIC 4 (Gaussian-zero cumulant with declared tolerance and n; orthogonal vs
shear invariance; fourth companion and truncation ratio; Laplacian normalisation
Eq. 4.3); CONTROL 4 (chart rule cumulant/deviation; cubic aligned contraction
and whitening; P3 Gaussian-zero with declared tolerance; hypergraph subject
α = 0 equals the tied linear AE); STRUCTURAL 2 (arm symmetry + matched capacity
+ manifest/builder agreement; tag/leakage coverage; interface shapes). TAUTOLOGICAL 0.

**B.3 hard-coded numbers:** the cumulant Gaussian tolerance (`6·sqrt(15/n)`,
`6·C·sqrt(15/m)`, `8·C²·sqrt(15/m)`), the bootstrap count 400 and the q99 are
development constants local to the tests/runner and are declared in file; none
is a claim threshold.

**B.4 no silent fallbacks:** a residual in a different `(C·N)` chart returns
`NaN` with a reason (never another chart); a third-moment feature absent on one
side is a structural test failure.

**C.1 enforcement:** run; refuses. **C.2:** unknown families are untouched by
this pass. **D Claim 2:** `paired_delta_auroc` still consumes only
`score_task`; `raw_task_cubic` is a separate report key. **E metrics:** the
cumulant estimators are tested against closed forms (Gaussian, orthogonal,
shear).

**BLOCKERS:** none for a development pass. **Open items carried into Phase B:**
(i) size `n_ref` and `n_pcs` for the enlarged third-moment feature set; the 17
Sep signature row is the first evidence that a third moment is noisier than a
second; (ii) D9 remains deferred; (iii) the design-only hypergraph subject is
untested on a trained run.

**Signature observations (development, not evidence):** covariance cells 4/4 at
reference after re-whitening (no false positives); in-span and geometry rows as
predicted; gain drift at reference as its linear-Gaussian prediction requires
and a z of 430 for channel loss, both as predicted; the glitch and the
non-Gaussian sparse-burst family did not exceed the reference band (q99 z =
2.33, deviations 30.9–31.9) — a development negative for the separator at this
size, reported and not explained away. This self-review records the pass as it
was implemented; `docs/reviews/2026-09-17_cumulant_independent_audit.md` adds an
independent finding that the residual statistic was estimated on the wrong
population, which the D7-bis follow-up addresses.
