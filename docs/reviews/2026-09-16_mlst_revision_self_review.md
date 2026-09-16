# Self-review of the MLST revision — 16 September 2026

_This is a **transparent self-review** by the implementation agent that made
the changes, applying `REVIEW_PROMPTS.md` §A to the amended protocol before the
dependent code changes and §B after them. It is not an independent review and
must not be cited as one. Findings marked **open** are carried into
`PREREGISTRATION.md` §7 and `TODO.md`; nothing here is a blocker to reversible
local work, and nothing here upgrades any result._

## §A applied to the amended protocol (before code)

Inputs: `EXPERIMENT_DESIGN.md` (Parts I–IV + the drafted Part V), `IMPLEMENTATION_PLAN.md` §8,
the drafted `PREREGISTRATION.md`, `reviews/2026-09-16_R0_reconciliation.md`, `latex/paper3_proposal.tex`.

**BLOCKERS (resolved before code)**

| # | finding | resolution |
|---|---|---|
| A1 | `PREREGISTRATION.md` did not exist here (TODO pointed at an external repo). | Unfrozen draft written here; `protocol/frozen/` absent by design; `mode.require_gates` refuses confirmatory. Not a fabricated freeze. |
| A2 | C4 primary contradicted itself across proposal / design / TODO / review prompts (R0 §3.1). | All-cell AUROC primary; conditional triage secondary; prompts synchronised so they cannot re-create the contradiction. |
| A3 | Every Δz statistic in `statistics.py` uses the paired twin; noise-only statistics are cell-level — the 13/14 lookup is replay-side, yet the design read as if it were alarm-time attribution. | Feature manifest with phases; alarm-time features in `protocol.features`; the 6 Sep table re-labelled as development evidence of signatures (`EXPERIMENT_DESIGN.md` §V.5, results doc header). |
| A4 | Noise-only statistics credited to the lookup, not to the generic arm (fairness to baselines, §A.7). | Noise-only features are in `all_generic`; `noise_only` and `all_generic_no_noise` ablations; privileged status where the acquisition lacks random triggers. |
| A5 | `DesignedFamily` produced a zero vector when the null space had rank 0 and seeded from `hash(kind)` (process-salted → not reproducible). | `NullSpaceUnavailable`; fixed integer seed stream; `linearization_check`; declared norm-matching metric. |
| A6 | No event-group split: the same event ids served as reference-vs-reference null and as evaluation; corrupted copies of fit events could be scored. | `protocol.splits`: four partitions by group; held-outs declared; unknown families excluded from fit/tuning/calibration and scored only on evaluation groups. |
| A7 | κ_m "provisionally 10 %" written as if declared. | `HarmThreshold` with `pending` / `provisional_dev` / `declared`; confirmatory refuses unless `declared`. |

**SHOULD FIX (open)**

- **O1** FAR resolution: the thresholds in the 6 Sep artifacts are q99 of 60 events / 200 bootstrap draws; the smoke's clean calibration partition is 18 windows. A 1 % budget is unresolved in both; the calibration count is an output of a development precision study that has not been run.
- **O2** Difficult N/S contrasts (overlapping multiplicity, signal strength, generic shift score) are specified in `PREREGISTRATION.md` §3 but not constructed on Tier 1; the smoke uses the inclusive population.
- **O3** The C4 acceptance point estimate and the held-out severity/seed ranges are `PENDING`.
- **O4** The proposal's C3 still fails at 10 % sampling on Tier 1 (pre-existing TODO; not touched).
- **O5** The "uncertainty" arm on the linear subject is a declared proxy (raw-domain reconstruction residual energy); the linear subject has no predictive uncertainty. On the transformer this arm must use a real uncertainty output or be dropped and said so.
- **O6** `C0` and `C5` remain mechanistic; nothing in this pass makes them core, and nothing should until independently justified.

**UNSUPPORTED (stated as hypotheses now)**

- "an alarm can be made to rank scientific consequence" → conditional hypothesis with refutation conditions (`EXPERIMENT_DESIGN.md` §I.1, §V.1; proposal §1 contribution paragraph).
- "layerwise information helps where generic statistics are ambiguous" — a prediction, unrun.

**NO ISSUE**

Traceability (every claim has one primary endpoint and a code path or an explicit "none"); identifiability (attribution over declared families only, unknowns to abstention, no conformal guarantee claimed); pre-registration bridge unchanged (WP0/§5 rules retained); scope (nothing added to the arm list).

## §B applied after the code changes

Commands run (cloud CPU, Python 3.11, numpy 2.4.4; repeated on the Mac VM before hand-off — see `REVISION_REPORT_2026-09-16.md` §5):

```
PYTHONPATH=src python -m pytest src/latent_monitor/tests -q            → 58 passed, 1 skipped (torch absent), ~22 s
PYTHONPATH=src python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev
                                                                          → 7.4 s; smoke_report.{json,md}; gate warning: freeze 'core' missing
PYTHONPATH=src python -m latent_monitor.protocol.smoke --mode confirmatory
                                                                          → ConfirmatoryGateError: freeze part 'core' is missing; threshold 'kappa_m_tier1' has status 'provisional_dev'
grep -rn "toy\|demo_\|smoke" src/latent_monitor/protocol/*.py             → only docstrings and smoke.py itself; no toy generator on a citable path (there is no citable path)
cd latex && pdflatex ×2                                                   → 7 pages, references and citations resolved; 2 pre-existing overfull boxes (tikz figure, "E0b")
```

**B.2 test classification** (new tests only; `tests/test_protocol.py`, `tests/test_designed_and_smoke.py`):
ANALYTIC 5 (weighted AUROC closed form and ties, AUPRC, group bootstrap spread, designed families exact in both metrics, seed reproducibility); CONTROL 4 (harmful low-alarm cells change the primary but not the conditional endpoint; single-class strata reported undefined; separable logistic rule and the three-way reading; smoke designed sign); STRUCTURAL 10 (contract refuses evaluation-only/delayed/undeclared features; privileged status; shared feature sets; disjoint splits; overlap raises; unknown families excluded; pending κ_m and missing freeze refuse; stale hash detected; null space skip; confirmatory smoke fails closed). TAUTOLOGICAL 0 by intent — no test asserts a value the same function produced earlier.

**B.3 hard-coded numbers**: `MARGINS` (0.10 / ±0.05) trace to the proposal §4 C2 and `PREREGISTRATION.md` §1; `FAR_BUDGET` 0.01 to plan §0.4; `HarmThreshold(1.10, "provisional_dev")` to the proposal's provisional value, labelled dev-only and refused in confirmatory mode; `TUNING_GRID` to `PREREGISTRATION.md` §2. `FRACTIONS` (0.40/0.20/0.15/0.25) are the smoke's own declared split and are recorded in its provenance.

**C.1 protocol enforcement**: run (above); refuses with both reasons.
**C.2 U families**: `assign_partitions` sends undeclared records to `evaluation` or `excluded` only (`test_unknown_and_held_out_families_never_enter_fit_tuning_or_calibration`).
**D.1 resampling**: `bootstrap_groups` resamples event groups and (crossed) perturbation seeds; tested against the textbook spread with one unit per group and zero spread with one group.
**D.3 C4**: `all_cell_ranking` has no conditioning; `conditional_triage` reports the cells it dropped; κ_m status enforced.
**E.2 designed families**: exact Δz and Δy on the linear subject to 1e-10 in both metrics; null realised Δy < 1e-10.
**E.4 silent fallbacks**: none added; `weighted_auroc`, `auprc`, `missed_harm_rate_at_budget`, `valid_rare_event_rejection`, `breakdown` return `nan` with an `undefined_reason` and counts, never a substituted value; per-hook Mahalanobis returns `nan` when a geometry cell's hook dimension differs.
**F.1 baseline fairness**: every arm passes through `fit_arm` with the same partitions, standardiser, classifier, grid and manifest.

**BLOCKERS**: none for the development pass. **COULD NOT CHECK**: torch-dependent `TransformerSubject` test (skipped: torch not installed in either environment used here); the trained-transformer and realism-arm runs (not part of this pass); anything confirmatory (nothing is frozen).

**Smoke observations (development-scale, not evidence)**: on the linear subject at toy size, `full_layerwise` did not beat `all_generic` (ΔF1 −0.04 [−0.09, 0.00], inconclusive), the noise-only statistics carried a measurable share of the generic arm's accuracy (ablation ΔF1 +0.10 [+0.04, +0.15]), and the all-cell ranking by the final-embedding alarm was 0.84 [0.40, 0.94] with a missed-harm rate of 0.75 at the budget — the in-span S cells and the structural cells are harmful with weak alarms, exactly the quadrant the old conditional endpoint could not see. These are illustrations of the machinery, at n = 30 evaluation groups, one seed, one linear subject.
