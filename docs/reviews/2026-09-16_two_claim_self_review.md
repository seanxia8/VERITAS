# Self-review of the two-claim revision — 16 September 2026 (pass 2)

_A **transparent self-review** by the implementation agent that made the
changes, applying `REVIEW_PROMPTS.md` §A to the revised protocol and §B to the
implementation. It is not an independent review and must not be cited as one.
Status words follow `TODO.md`: interface implemented · unit-tested control ·
development smoke · trained-model evidence · transfer evidence · confirmatory.
Everything in this pass is at the first three levels._

## §A applied to the revised protocol

Inputs: `latex/paper3_proposal.tex` (two-claim), `EXPERIMENT_DESIGN.md` (consolidated),
`PREREGISTRATION.md` (unfrozen), `TWO_CLAIM_REVISION_PLAN.md`, the pass-1 review notes.

**1 TRACEABILITY — no issue found.** Exactly two claims; each has one primary estimand, an interval rule (Claim 2's acceptance point PENDING), a refutation condition and a code path (`protocol.arms` / `protocol.consequence`). Supporting analyses carry no thresholds. `grep` of the proposal finds no `J_h^T Σ^{-1} J_h`, no "co-primary", no "gauge-safe" used as a claim, no C0–C5 headings.

**2 FALSIFIABILITY — no issue found**, with one caveat: the Claim-1 margins (0.10 / ±0.05) and the Claim-1 minimum set size (30 windows / 10 groups) are declared but unjustified against achievable precision; Phase B decides whether they are reachable (open, O1).

**3 CONFOUNDS.**
- Fixed: `intermediate_only` contains no input, final-z, pre-output or output duplicate (test); capacity-matched control exists (quadratic expansion); hook-drop controls exist.
- **Finding (open, O2):** on the linear subject the token hook is the channel hook concatenated with a constant geometry embedding, so the hook-drop arms are *identical* to the full arm (ΔF1 = 0.000 [0, 0] in the smoke). Hook-drop is uninformative on this subject by construction and only means something on the trained nonlinear subject. Recorded in the smoke report and here; not a bug.
- **Finding (open, O3):** the hard set at toy size has 20 windows (10 pairs from 48 S / 168 N; overlap failures 38). No reading is issued below 30 windows (declared). The Phase-B study must size the evaluation partition so that the hard set clears the minimum with margin, or the claim is untestable at this design.
- Claim 2: the paired comparison, quadrants, missed harm and valid-rare rejection are all cell-level with a cell-level threshold calibrated by pseudo-cell bootstrap; the previous per-window-threshold-on-median-alarm rule is gone.

**4 LEAKAGE.**
- Fixed: `AlarmTimeInputs` has no truth/twin/label/realised-Σ field; `FeatureBatch` refuses forbidden source tags; arms are scored only from locked, tag-checked batches; `reference_fit` is never a supervised partition; unknown families reach only `evaluation`.
- **Residual (stated in code, paper and here):** a builder that lies about a source tag passes both layers (`test_adversarial_builder_cannot_smuggle_truth` demonstrates it). Code review remains part of the contract.
- The K used for Claim 2 (evaluation-only truth) is computed in the smoke *outside* the builder from `cell.batch` truth; it never enters the batch (checked by the tag audit: no feature carries `truth`).

**5 STATISTICS.**
- Fixed: Claim 2's interval resamples the cell as outer unit with event groups nested; descriptive below 10 cells; failed replicates counted; AUPRC tie-aggregated and permutation-invariant; weights validated; macro-F1 missing-class policy declared; FAR with a binomial interval.
- **Finding (open, O4):** reference distances are *unstable* at the smoke's reference size (measured: ordering correlation 0.65–0.93 at n_ref = 60 for d = 6). The smoke's hook nulls are fitted on ~36 windows. Any smoke number involving reference distances is therefore noisy in a way the intervals do not capture (the reference is not resampled). Phase B must size `reference_fit`.
- **Finding (open, O5):** the C2 hierarchical interval resamples cells but *not* the reference/calibration fit; a full nested resampling including reference refits is not implemented.

**6 IDENTIFIABILITY — no issue found.** Attribution over declared families; unknowns to abstention; the abstention guarantee is stated as clean-window retention only; matching is declared conditional, not causal.

**7 FAIRNESS TO BASELINES — no issue found** in design: one classifier, one grid, identical partitions, identical side information; the generic arm holds the same distance transforms as the intermediate arm; the committed generic harm score is fixed before any label. **Open (O6):** the alibi-detect baselines named in the proposal (MMD, C2ST, KS) are still not wired into `protocol.arms`.

**8 TOY DETECTION — no issue found.** The smoke writes to a dated directory with `NOT CITABLE` in the first line and strict JSON; the confirmatory gate refuses on eight independent grounds; no `results/confirmatory/` exists.

**9 BRIDGE — unchanged**, and out of this pass.

**10 SCOPE — improved.** Minimum viable paper = trained Tier-1 subject + one transfer arm; every other arm is future work in `EXPERIMENT_DESIGN.md` §I.3–I.4 and the proposal's roadmap box.

**11 CONTRADICTIONS.** Part V no longer "governs"; the ▸-amendment markers and the six-claim tables in Parts I–II are rewritten; the pass-1 design is archived. `grep` finds no "conditional on alarm" primary, no "C4 primary" in the live design. **Residual (O7):** Part II keeps its 5 Sep prose about arms (LUCiD/HeST detail) and Part III.4–III.9 keep the adjustment/repair material as supporting; they are consistent with Part V but long, and a reader of Part II alone will still meet the old arm ambitions marked as future work.

## §B applied to the implementation

Commands (cloud CPU, Python 3.11.15, numpy 2.4.4, scipy 1.17.1; repeated on the Mac VM — see `REVISION_REPORT_2026-09-16_pass2.md` §5):

```
PYTHONPATH=src python -m pytest src/latent_monitor/tests -q                 → 84 passed, 1 skipped (torch absent), 23 s
PYTHONPATH=src python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_2026-09-16_pass2
                                                                              → 6.5 s; strict JSON verified by json.load; NOT CITABLE in line 1
PYTHONPATH=src python -m latent_monitor.protocol.smoke --mode confirmatory   → ConfirmatoryGateError (freeze 'core' missing; κ_m provisional_dev)
PYTHONPATH=src python -m latent_monitor.run_table --out /tmp/table_check     → 13 match / 1 documented / 0 mismatch; the 12 non-designed rows numerically
                                                                                identical to results/latent_monitor_tier1/table.json (NaN-equal on the two
                                                                                geometry rows' undefined channel-correlation entry)
cd latex && pdflatex ×2                                                       → 8 pages; references and citations resolved; one pre-existing overfull box (tikz figure)
grep -rn "toy\|demo_" src/latent_monitor/protocol/*.py                        → docstrings only
```

**B.2 test classification** (pass-2 additions: `test_spine.py` 8, `test_protocol_pass2.py` 16, updates in `test_protocol.py` / `test_designed_and_smoke.py` 2): ANALYTIC 11 (metric shapes/units; resolution-weighted W_y; LW shrinkage limits; invariance groups ×2; task-aligned/null exactness; macro-F1 policy; AUPRC tie invariance; weight validation; null calibrator; paired ΔAUROC/cell threshold); CONTROL 6 (support estimator in/out; stability vs reference size — records the instability; hierarchical vs event-only spread; conformal retention and unknown AUROC; hard matching; joint table); STRUCTURAL 9 (batch refusals; adversarial builder; locked-batch contract; five disjoint partitions; undeclared class refusal; optimiser/partition errors; nonlinear subject refused; run-dependency gate; FAR precision). TAUTOLOGICAL 0.

**B.3 hard-coded numbers:** `MARGINS` (proposal §4, unfrozen), `MIN_WINDOWS_FOR_READING = 30`, `MIN_GROUPS_FOR_READING = 10`, `MIN_OUTER_UNITS_FOR_INFERENCE = 10`, `ABSTENTION_ALPHA = 0.10`, `FRACTIONS`, caliper 1.0, support floor 0.9, PCA rule n_ref/5, `HarmThreshold(1.10, provisional_dev)` — all declared in `PREREGISTRATION.md` as unfrozen, none derived from an evaluation result.

**C.1 enforcement:** run; refuses. **C.2 U families:** `assign_partitions` + `fit_logistic` refuse (tests). **D.1 resampling:** `bootstrap_hierarchical` tested against an event-only bootstrap on a cell-effect model. **D.3 Claim 2:** `paired_delta_auroc`, cell-level threshold, quadrants (tests). **E.2 designed:** exact in both metrics; task-specific families exact; nonlinear refused (tests). **E.4 silent fallbacks:** none; every undefined quantity returns `nan` with a reason and counts; the report prints "—" and the reason.

**BLOCKERS for the development pass:** none. **Open items carried:** O1–O7 above, plus the eight run-dependency warnings the smoke prints (no freeze parts, dirty tree, unfrozen environment/data/model hashes, non-confirmatory destination) — all expected in development.

**Smoke observations (toy size, not evidence):** Claim 1 — no reading on the 20-window hard set; inclusive ΔF1 +0.05 [+0.02, +0.08] (inconclusive), of which the capacity-matched generic control leaves +0.05 [0.00, +0.16]; `intermediate_only` alone is below `generic_rich`; noise-only statistics still add +0.12 to the pass-1 generic arm. Abstention — unknown-family AUROC 0.60; the whole-chain decision rule is correct on 7 % of windows because the committed generic detector rarely fires at the toy-size 1 % budget and alarmed windows mostly abstain. Claim 2 — ΔAUROC +0.17 (task 1.00 vs generic 0.83; 4 harmful / 9 benign cells) with a cell-outer interval [−0.45, +0.29]; on the two held-out cells ΔAUROC = 0 (descriptive). All preserved in `results/latent_monitor_smoke_dev_2026-09-16_pass2/`.
