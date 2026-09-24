# Two-claim revision — implementation and review report, 16 September 2026 (pass 2)

_Second local pass, implementing `TWO_CLAIM_REVISION_PLAN.md` (M1–M5, I1–I13,
D1–D4) on the dirty worktree at `f24430a` plus the pass-1 changes. **Nothing is
frozen, no threshold or margin was invented, no trained-model or transfer
result exists, and no scientific conclusion is drawn.** Status vocabulary:
interface implemented · unit-tested control · development smoke ·
trained-model evidence · transfer evidence · confirmatory — this pass reaches
the first three only. The transparent self-review is
`reviews/2026-09-16_two_claim_self_review.md`; the pass-1 report keeps its
validation record with a superseding status note._

## 1. Files changed

| file | change |
|---|---|
| `latex/paper3_proposal.tex`, `.pdf` | D1/M1–M5: abstract rewritten around two claims; §1 contribution paragraph; §3 "Two metrics" (M_recon vs M_task, resolvability ≠ support, conditional signatures with the observed counterexamples, exact invariance groups, bounded companion dependency, Kay 1993 cited for the Fisher information); §4 "Two claims" replaces C0–C5 (Claim 1 / Claim 2 with estimands, controls, refutation; supporting analyses; abstention with the exchangeability caveat); Table 1 = paired Claim-2 endpoints; §5 monitoring design (typed builder, five partitions, primary/control arms, shrinkage, hard matching, cell-outer resampling); experiment matrix (E0, A1, A2, S1–S4, T1); roadmap = evidence ladder (minimum viable paper, Phases A–C, future work); Ledoit–Wolf and Kay bib entries; the companion preprint marked unpublished and non-load-bearing. 8 pages, references resolved. |
| `docs/EXPERIMENT_DESIGN.md` | D2: consolidated — header without precedence rules; Contents; §I.1 two claims; §I.2 Tier-1 status; **§I.3 two-claim table** with the minimum viable paper; §I.4 phases; Part II arm tables restated in two-claim language; **§III.1 conditional-signature table** replaces the dichotomy; §III.3.1 `P_resolved/P_weak` and M_task; §III.3.2 replay-side labelling; §III.3.3 lookup = development diagnostic; §III.3.4 abstention as implemented; **Part V rewritten** as "The two-claim protocol" (V.1–V.7); Appendix A row and map. Pass-1 text archived as `archive/EXPERIMENT_DESIGN_2026-09-16_pass1.md` (README row added). |
| `docs/PREREGISTRATION.md` | rewritten around the two claims: estimands, committed scores, inference rules (Claim-2 acceptance PENDING), spine, information contract, five partitions, K/κ_m table, statistics and the run-dependency gate, designed controls, decisions before freeze. Still an **unfrozen draft**. |
| `docs/TODO.md`, `README.md`, `src/latent_monitor/README.md`, `docs/REVIEW_PROMPTS.md`, `docs/IMPLEMENTATION_PLAN.md` (§8.6), `docs/REVISION_REPORT_2026-09-16.md` (status note) | D3: status language in the six-level vocabulary; no "done" for abstention, resampling, hard contrasts or enforcement beyond *interface / unit-tested / development smoke*; review prompts synchronised (traceability = two claims; §B.D.3/D.5 Claim-2/Claim-1 checks; red flags). |
| `src/latent_monitor/reference.py` | M1/M2: `P_resolved`, `P_weak`, `resolved_rank`, `M_recon`, `resolved_basis` properties; legacy fields kept; docstring states dimensions. |
| `src/latent_monitor/task_metric.py` | **new**: `TaskMetric` (W_y validated: shape in output units, PSD), `one_hot`, `from_resolutions`, `length`, `aligned_direction`, `null_projector`; `measurement_metric`; `InvarianceReport`. |
| `src/latent_monitor/support.py` | **new**: Ledoit–Wolf shrinkage; `SupportEstimator` (Mahalanobis ∨ kNN, clean-quantile standardised); `validate_support_estimator`; `displaced_controls`. |
| `src/latent_monitor/designed.py` | I8: `task_aligned` / `task_null` from M_task; linear-only guard (`NonlinearSubjectUnsupported`); per-output realised Δy in the linearisation check. `linear_subject.py`: `is_linear` declaration. `tier1.py`: `mixture_cells`. |
| `src/latent_monitor/protocol/availability.py` | I2: `AlarmTimeInputs`, source-tagged `FeatureBatch` (refuses forbidden sources, locks), contract with data-flow check; manifest groups for `generic_rich`, `intermediate`, `generic_quadratic`, `layerwise_legacy`. |
| `src/latent_monitor/protocol/features.py` | I3/I10: `build_alarm_time_features` (the only builder); `HookNull` (Ledoit–Wolf + PCA rule); `AlarmTimeReference` with support estimator and task metric; `NullCalibrator`; committed score definitions. |
| `src/latent_monitor/protocol/arms.py` | I3/I6/I9: primary/control/diagnostic arms; hook-drop arms; `attribution_train` fit; class/partition/optimiser checks (`ArmFitError`); macro-F1 policy with `macro_f1_report`; confusion; `read_delta` minima and degenerate-interval rule. |
| `src/latent_monitor/protocol/consequence.py` | I4/I5/I9/I12/I13: weight validation; tie-aggregated AUPRC; `paired_delta_auroc`; `cell_aggregate`, `cell_alarm_threshold`; `valid_rare_cell_rejection` (+ legacy alias) and `valid_rare_window_rejection`; `far_precision`; `bootstrap_hierarchical` (outer cell/family/seed ⊃ event group; model seeds outermost; descriptive below 10 outer units; failed replicates counted); levels stated. |
| `src/latent_monitor/protocol/abstention.py` | **new** (I1): `ConformalNovelty` (clean-only), `unknown_auroc`, `risk_coverage`, `evaluate_abstention`. |
| `src/latent_monitor/protocol/matching.py` | **new** (I7): `hard_match` (1:1, caliper, retention, overlap failures), `joint_decision_table`. |
| `src/latent_monitor/protocol/splits.py` | I6: `attribution_train` partition. |
| `src/latent_monitor/protocol/mode.py` | I11: `RunDependencies`, `require_run_dependencies` (freezes core/counts/bridge, freeze commit an ancestor of HEAD, clean tree or frozen tree hash, environment-lock hash, data-manifest hash, model hash, destination under `results/confirmatory`), `source_tree_hash`, `is_ancestor`. |
| `src/latent_monitor/protocol/smoke.py` | rewritten for the two-claim chain; writes to a **new dated directory**; strict JSON. |
| `src/latent_monitor/protocol/labels.py` | `mixture` origin; `declared=False` → origin `unknown`. |
| `src/latent_monitor/tests/test_spine.py`, `test_protocol_pass2.py` (**new**), `test_protocol.py`, `test_designed_and_smoke.py` (updated) | 26 new tests; 84 total (see §5). |
| `results/latent_monitor_smoke_dev_2026-09-16_pass2/` | **new** development smoke (`smoke_report.{json,md}`, 104 kB), NOT CITABLE. `results/latent_monitor_smoke_dev/` (pass 1) and `results/latent_monitor_tier1/` untouched. |
| `docs/reviews/2026-09-16_two_claim_self_review.md` | new. |

Not touched: `noise_module`, `herald_simulation`, `prometheus_simulation`, `tidmad_transformer`, `reconstruction_model`, `statistics.py`, `lookup.py`, `adjust.py`, `torch_subject.py`, notebooks, `TESTBEDS.md`, `references.bib`, `Claude outputs/`, `docs/IMPLEMENTATION_PROMPT_*.md`, `docs/TWO_CLAIM_REVISION_PLAN.md`.

## 2. Audit findings: fixed, partially fixed, blocked

| id | finding | status | where |
|---|---|---|---|
| M1 | separate M_recon from M_task; remove J_yᵀ Σ⁻¹ J_y | **fixed** (unit-tested: dimensional refusal, unit scaling) | `task_metric.py`, proposal §3, `test_spine.py` |
| M2 | resolvability ≠ training support; rename; separate estimator with controls | **fixed** (unit-tested control) — legacy reads preserved | `reference.py`, `support.py` |
| M3 | conditional N/S signatures incl. observed counterexamples | **fixed** in text (design §III.1, proposal §3); tested only by the existing signature tests | `EXPERIMENT_DESIGN.md` §III.1 |
| M4 | exact invariance groups + orthogonal/scale/shear controls | **fixed** (unit-tested); shear dependence of shrunk distances *reported*, not removed | `test_spine.py`, proposal §3 |
| M5 | bound the companion-paper dependency | **fixed** in text (Kay 1993 for the Fisher information; preprint non-load-bearing; proposition numbers withdrawn) | proposal §3, bib |
| I1 | abstention end to end | **interface implemented + unit-tested + development smoke**; unknown AUROC 0.60 at toy size — no evidence it works | `abstention.py` |
| I2 | data-flow contract + adversarial test | **fixed** with a documented residual (a lying tag passes) | `availability.py`, `test_protocol_pass2.py` |
| I3 | `generic_rich` / `intermediate_only` / `full_intermediate`, hook-drop, matched capacity | **fixed** (development smoke); hook-drop is uninformative on the linear subject (token duplicates channel) | `arms.py`, `features.py` |
| I4 | committed generic score, task-sensitive score, paired ΔAUROC | **fixed** (development smoke) | `consequence.py`, `smoke.py` |
| I5 | hierarchy with cell/family/seed outer unit; refuse inference | **fixed** (unit-tested); reference/calibration refits are *not* resampled (open O5) | `consequence.py` |
| I6 | `attribution_train`; `reference_fit` never trained on | **fixed** (unit-tested) | `splits.py`, `arms.py` |
| I7 | hard contrasts with retention; joint table | **interface implemented + development smoke**; the toy hard set is 20 windows → no reading | `matching.py` |
| I8 | designed perturbations linear-only; task-specific families | **fixed** (unit-tested control); nonlinear local inverse **not implemented** (future) | `designed.py` |
| I9 | macro-F1 policy, tied AUPRC, weights, optimiser/partition/class checks, bootstrap diagnostics | **fixed** (unit-tested) | `arms.py`, `consequence.py` |
| I10 | shrinkage / dimension reduction / calibration / stability tests | **fixed** (unit-tested); stability test **records instability** at n_ref ≲ 100 (open O4) | `features.py`, `test_spine.py` |
| I11 | confirmatory dependency verification | **fixed** (unit-tested); nothing frozen | `mode.py` |
| I12 | cell/window terminology; cell-level threshold | **fixed** | `consequence.py` |
| I13 | FAR precision | interval + resolvability reported; the **precision study is not run** (blocked on Phase B) | `consequence.far_precision` |
| D1 | two-claim proposal | **fixed**; 8 pages | `paper3_proposal.tex` |
| D2 | consolidate the design | **fixed**; pass-1 text archived | `EXPERIMENT_DESIGN.md` |
| D3 | status language | **fixed** | TODO, README, plan, prompts |
| D4 | narrow the evidence ladder | **fixed** in text | design §I.3–I.4, proposal roadmap |

**Blocked / not attempted (by boundary):** Phase-B precision study; κ_m; the trained-subject run; any transfer arm; the nonlinear local inverse; wiring the alibi-detect baselines; nested resampling of reference refits; the external pre-registration.

## 3. What the smoke shows (development, toy size, NOT evidence)

`results/latent_monitor_smoke_dev_2026-09-16_pass2/smoke_report.md`, linear subject, C = 6, N = 128, 120 event groups, seed 0, 6.5 s:

- Contract: 13 arms pass the declarative and data-flow checks; all operational.
- Partitions: reference_fit 30 % / attribution_train 20 % / development 15 % / calibration 15 % / evaluation 20 %; 180 windows excluded (held-out/undeclared families on non-evaluation groups).
- FAR: 3/24 clean evaluation windows alerted at the 1 % threshold from 18 calibration windows (realised 0.125, CI95 [0.03, 0.32]) — not resolvable at this size.
- Claim 1: `generic_rich` F1 0.856 / `full_intermediate` 0.903 / `intermediate_only` 0.653 inclusive; hard set 20 windows → ΔF1 0.000 [0, 0], **no reading**; inclusive ΔF1 +0.047 [+0.017, +0.082] (inconclusive by the margins); capacity-matched control on the hard set +0.055 [0.00, +0.16] (no reading); hook-drop ΔF1 exactly 0 (token duplicates channel on this subject); noise-only ablation +0.119 (pass-1 diagnostic).
- Abstention: conformal threshold from 18 clean windows (α = 0.1; realised clean retention 0.875); support-estimator control AUROC 1.00; unknown-family AUROC **0.60**; risk–coverage AUC 0.11.
- Joint decision table: clean 0.875 correct (quiet); N 0.00; S 0.00; mixture 0.00; unknown 0.17 — the committed generic detector rarely fires at this budget and alarmed windows mostly abstain; whole-chain correct rate 0.074.
- Claim 2: ΔAUROC +0.167 (task 1.000 vs generic 0.833; 4 harmful / 9 benign cells); cell-outer interval [−0.45, +0.29]; held-out families only (2 cells): 0.000, descriptive; missed harm at the cell threshold: generic 0.25, task 0.75; benign valid-rare cell rejection: generic 1.00, task 0.00.
- Designed controls (linear, exact): output-null 1.00 on every target; task-aligned moves the amplitude endpoint (2.78 in the Euclidean metric, 1.17 in the null-Mahalanobis metric); task-null leaves it at 1.00; output-aligned moves amplitude less than task-aligned (1.76 / 1.06) — the pass-1 observation that output-aligned mostly moved timing is confirmed and now controlled.

## 4. What the pass-1 record says, preserved

`results/latent_monitor_smoke_dev/`: layerwise attribution ΔF1 −0.039 [−0.090, +0.003] (inconclusive) and layerwise harm AUROC 0.56 [0.21, 0.70]. Unchanged, not reinterpreted.

## 5. Validation — exact commands and results

Cloud CPU (Python 3.11.15, numpy 2.4.4, scipy 1.17.1; copy of the checkout):

```
$ PYTHONPATH=src python3 -m pytest src/latent_monitor/tests -q
84 passed, 1 skipped in 23.31s            # skipped: test_torch_subject (torch not installed)

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_2026-09-16_pass2
DEVELOPMENT SMOKE — NOT CITABLE. …
Claim 1 ΔF1 (hard set): 0.0 [0.0, 0.0] → inconclusive (n_windows 20 < 30)
Claim 2 ΔAUROC: 0.1667 (task 1.0 vs generic 0.8333); outer-unit bootstrap over 13 cells
$ python3 -c "import json; json.load(open('results/latent_monitor_smoke_dev_2026-09-16_pass2/smoke_report.json'))"   # strict

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --mode confirmatory
ConfirmatoryGateError: confirmatory mode refused:
  - freeze part 'core' is missing (protocol/frozen/core.json)
  - threshold 'kappa_m_tier1' has status 'provisional_dev'; not usable in confirmatory mode
  (the run-dependency check additionally lists: counts/bridge freezes missing, dirty tree, unfrozen environment lock,
   no data manifest, no model hash, destination not under results/confirmatory)

$ PYTHONPATH=src python3 -m latent_monitor.run_table --out /tmp/table_check
13 match, 1 documented, 0 mismatch       # 12 non-designed rows numerically identical to results/latent_monitor_tier1/table.json

$ cd latex && pdflatex … ×2
Output written on paper3_proposal.pdf (8 pages).  No undefined references or citations.  1 pre-existing overfull box (tikz).
Pages 3 and 4 rendered and inspected (two-metrics section, two-claims section, Table 1).
```

Mac (the linked computer's VM, `/Users/dowlingwong/Documents/VERITAS` mounted; Python 3.12.14, numpy 2.5.3, scipy 1.18.1, scratch venv):

```
$ PYTHONPATH=src ~/venv-lm/bin/python -m pytest src/latent_monitor/tests -q -p no:cacheprovider
84 passed, 1 skipped in 15.39s

$ PYTHONPATH=src ~/venv-lm/bin/python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_2026-09-16_pass2
Claim 1 ΔF1 (hard set): 0.0063 [-0.108, 0.156] → inconclusive (n_windows 20 < 30)
Claim 2 ΔAUROC: 0.1667 (task 1.0 vs generic 0.8333); outer-unit bootstrap over 13 cells
(4.4 s; config hash 43cf86fcc97849ba; commit f24430a-dirty)      # this run is the one checked in under results/
strict JSON verified; confirmatory mode refused with the same two reasons
```

**Cross-platform difference, recorded:** the config hash, the partitions, the Claim-2 numbers and the designed controls are identical on both machines; the Claim-1 hard-set ΔF1 is 0.000 [0, 0] in the cloud and +0.006 [−0.11, +0.16] on the Mac because the L-BFGS logistic fits differ across BLAS builds and, on a 20-window set, a single flipped prediction moves the interval from degenerate to wide. Both readings are "no reading" by the declared minimum. The seeds cover the data and the bootstrap draws, not the optimiser's floating-point path; a confirmatory run must pin the environment (the gate checks the lock hash).

## 6. Gated commands for later scientific runs (not run)

```
# Phase B — development precision study (CPU, hours): calibration count, reference-fit count, outer-unit counts, model seeds
PYTHONPATH=src python -m latent_monitor.protocol.smoke --out results/dev_precision/<date>_g<N> --n-groups <N> --seed <s>   # sweep N ∈ {240, 480, 960}, s ∈ 0..4
#   read: far.ci95 width, claim1.hard_matching.n_pairs ≥ 30 windows, claim2.primary_delta.ci_hierarchical_outer_cell.n_outer ≥ 10

# Phase C — trained nonlinear subject (GPU; needs .venv-cpu/uv sync with torch): not implemented as a runner yet —
#   requires a TransformerSubject-backed smoke (subject fitted on reference_fit windows, then the same chain), and the
#   designed controls stay linear-only. Gate: Phase B counts, κ_m declared, Junjie's agreement on the framing.

# Transfer arm B (HeST): after Phase C — herald_simulation.simulate.all_cells → the same chain with K declared per PREREGISTRATION §5.
```

## 7. Next scientific gate

Phase B: size `calibration`, `reference_fit`, the hard set and the Claim-2 outer units on the controlled simulator, and decide whether the declared margins (0.10 / ±0.05; ≥ 30 hard windows; ≥ 10 outer cells) are achievable. If they are not, the claims or the design change **before** any freeze. Then declare κ_m for Tier 1 from a scientific requirement and run the trained subject in development mode. Completion of this pass means the local protocol is credible enough to run that subject next; it does not mean either claim is supported or that the paper is ready for MLST.
