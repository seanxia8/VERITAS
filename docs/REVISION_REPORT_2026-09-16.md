# MLST revision — implementation report, 16 September 2026 (pass 1)

_**Status note added the same day (pass 2):** the independent review of this
pass (`TWO_CLAIM_REVISION_PLAN.md`) found that several items below described
as "done" were declarative or absent — abstention (no implementation),
the information contract (name metadata only), "difficult contrasts" (not
built), scientific resampling (event groups only, not the intervention cell),
the contaminated "layerwise" arm, the missing paired Claim-2 comparison, and a
dimensionally invalid J_hᵀ Σ⁻¹ J_h in the proposal. Those are addressed in
`REVISION_REPORT_2026-09-16_pass2.md`, which supersedes the status claims of
§1–§3 here; the validation record in §5 and the file list stand as the record
of what pass 1 did. The pass-1 smoke (`results/latent_monitor_smoke_dev/`) is
preserved unchanged; its layerwise attribution delta was negative/inconclusive
and its layerwise harm AUROC weak, and neither number is evidence._

_First implementation pass of `IMPLEMENTATION_PLAN.md` §8 (R0–R6, boundary §8.4)
at baseline `f24430a` on `dev` plus the uncommitted §8. Everything below is
development work: **no protocol is frozen, no result is confirmatory, and no
scientific claim is validated by this pass.** The reconciliation is
`reviews/2026-09-16_R0_reconciliation.md`; the transparent self-review is
`reviews/2026-09-16_mlst_revision_self_review.md`; the amended protocol is
`PREREGISTRATION.md` (unfrozen draft) and `EXPERIMENT_DESIGN.md` Part V._

## 1. What changed (files)

| file | change |
|---|---|
| `latex/paper3_proposal.tex`, `.pdf` | R1/R6: contribution paragraph with refutation conditions and prior-art distinctions (Rabanser; Zhang; Guillory; Garg; Angelopoulos & Bates); C2 one primary endpoint (ΔF1 vs an all-generic arm that holds operational noise-only statistics; noise-only ablation); abstention = feature novelty, exchangeability caveat, no unknown-shift guarantee; C4 = all-cell AUROC for K ≥ κ_m, κ_m pending (provisional 10 % dev only), four quadrants, missed harm, valid-rare rejection, conditional triage secondary; independent physical K beside the diagnostic residual; assumed vs realised Σ; designed families as positive controls with norm matched in the challenged metric, linearisation error, explicit skip; information contract and event-group splits in §5; FAR-resolution sentence; Tier-1 development findings folded in as shaping the arms; E4 row; two new bib entries; date. Builds to 7 pages with all references resolved. |
| `docs/EXPERIMENT_DESIGN.md` | §I.1 hypothesis wording; §I.3 C2/C4 rows; §III.3.2 K and replay-side labelling; §III.3.3 noise-only statistics belong to the generic arm; §III.3.4 abstention assumptions; **new Part V** (chain as hypothesis, information contract, splits/origin/harm, amended endpoints, what the 6 Sep table is, results structure); Appendix A row and document map. Earlier text left in place; Part V governs. |
| `docs/PREREGISTRATION.md` | **new, unfrozen draft** of the `core` part: claims and primary endpoints with inference rules; information contract; splits and hold-outs; origin/harm and K per arm with κ_m **PENDING**; designed control; statistics and gates; the decisions required before a freeze. |
| `docs/IMPLEMENTATION_PLAN.md` | header status; §1 layout annotation (`oracle_diag` → `latent_monitor`); §7.3 marked stale; **§8.5** status table. The user's uncommitted §8 text is preserved verbatim. |
| `docs/REVIEW_PROMPTS.md` | synchronised with the amended C4 endpoint (§A.3, §A.4, §B.D.3, red flags, read list). |
| `docs/TODO.md`, `README.md`, `src/latent_monitor/README.md` | status, layout, commands, evidence-level wording ("validated development pilot" → "development-mode pilots"). |
| `docs/RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md` | evidence-level header added; numbers untouched. |
| `src/latent_monitor/designed.py` | R4: `random` family; `match_metric` (`euclidean` \| `null_mahalanobis`); `NullSpaceUnavailable` on rank-0 subspaces; `linearization_check`; fixed integer seed stream (the old `hash(kind)` stream was process-salted, i.e. not reproducible across runs). |
| `src/latent_monitor/protocol/` | **new**: `labels`, `availability`, `splits`, `features`, `arms`, `consequence`, `mode`, `smoke` (see §3). |
| `src/latent_monitor/tests/test_protocol.py`, `test_designed_and_smoke.py` | **new**: 19 tests (5 analytic, 4 control, 10 structural; 0 tautological). |
| `src/latent_monitor/pyproject.toml` | registers the `protocol` subpackage. |
| `results/latent_monitor_smoke_dev/` | **new** development smoke output (`smoke_report.{json,md}`, 56 kB), labelled not citable. `results/latent_monitor_tier1/` untouched. |
| `docs/reviews/2026-09-16_R0_reconciliation.md`, `docs/reviews/2026-09-16_mlst_revision_self_review.md` | new. |

Not changed: `noise_module`, `herald_simulation`, `prometheus_simulation`, `tidmad_transformer`, `reconstruction_model`, `statistics.py`, `lookup.py`, `reference.py`, `adjust.py`, `tier1.py`, the notebooks, `TESTBEDS.md`, `references.bib`.

## 2. Claim → evidence table (after this pass)

| claim | primary endpoint | code | evidence level | gate / next |
|---|---|---|---|---|
| C2 incremental attribution (central) | ΔF1 full-layerwise − all-generic (incl. operational noise-only), margins 0.10 / ±0.05 | `protocol.arms`, `protocol.features`, `protocol.availability` | dev smoke only (linear subject, toy size): ΔF1 −0.04 [−0.09, 0.00] → inconclusive; noise-only ablation +0.10 [+0.04, +0.15] | trained transformer under the alarm-time contract; difficult contrasts; freeze |
| C4 scientific harm (central) | all-cell AUROC for K ≥ κ_m, uniform cell weighting; secondaries listed in `PREREGISTRATION.md` §1 | `protocol.consequence` | dev smoke only: final-embedding alarm 0.84 [0.40, 0.94], layerwise 0.56 [0.21, 0.70], 4 harmful / 8 benign cells, missed harm 0.75 at budget, under the *provisional* κ_m = 1.10 | **κ_m pending on every arm**; confirmatory refuses |
| C1 detection | power at 1 % FAR, delay | `statistics`, `lookup.calibrate`; the smoke calibrates the (1 − FAR) quantile on 18 clean windows and reports it unresolved | dev | cov_D precision study |
| C3 cost | latency, bytes, loss | none | none | out of scope of this pass |
| C0 organisation | probe R², axis participation | `estimators/identification.py` | none run | Junjie's agreement on C0 |
| C5 localisation | patch recovery on a nonlinear subject | `adjust.activation_patch` | dev, flat on the linear subject | transformer |
| designed control | realised K ≈ 0 on null / large on aligned at matched norm; linearisation error | `designed.py` | analytic on the linear subject (exact to 1e-15) in both metrics; the amplitude endpoint is barely moved by an output-aligned δ (ratio 1.13–1.35; t0 2.6–3.7) — the sign holds on the t0/τ targets, which is why K must be declared per target and per arm | positive control only |

## 3. What the new code enforces

- **Information contract** (`availability.py`): every feature carries a phase; `AlarmTimeContract.check` raises `LeakageError` on `evaluation_only`/`delayed_label`/undeclared features; noise-only features are `operational` only when the acquisition supplies them, else the arm is `privileged`.
- **Alarm-time features** (`features.py`): per-window input-quality summaries, outputs, a declared uncertainty proxy, z and its Mahalanobis, per-hook Mahalanobis against reference-fitted stage nulls, energy splits about the reference mean, out-of-span from the residual, noise-only statistics of the window's own random-trigger records — **no paired twin, no truth, no realised Σ**.
- **Splits** (`splits.py`): four partitions by event group; overlap raises; held-out families/severities/seeds and undeclared families score only on evaluation groups and are `excluded` elsewhere.
- **Arms** (`arms.py`): five arms + `noise_only` + `all_generic_no_noise`; one L2 multinomial logistic classifier; λ chosen on `development`; scored once on `evaluation`; paired ΔF1 with a bootstrap over event groups and the three-way reading.
- **C4** (`consequence.py`): `HarmThreshold` with `pending`/`provisional_dev`/`declared`; `all_cell_ranking` (no conditioning), `alarm_harm_matrix` (all four quadrants + undefined, with counts and weights), `missed_harm_rate_at_budget`, `valid_rare_event_rejection`, `conditional_triage` (reports dropped cells), `breakdown`, `bootstrap_groups` (event groups × perturbation seeds, crossed). Undefined classes return `nan` with a reason and counts — never a substituted value.
- **Modes** (`mode.py`): `require_gates` refuses confirmatory mode on a missing/stale freeze or a non-`declared` threshold; `freeze_part` exists and is called by nothing; `provenance` records config hash, commit, seeds, versions, machine, mode, `citable=False` in dev.
- **Designed control** (`designed.py`): as in §1.

## 4. Contradictions resolved

Fourteen, listed with both sides in `reviews/2026-09-16_R0_reconciliation.md` §3. The load-bearing ones: C4 conditioning (all-cell primary, conditional secondary); claim priority (C2/C4 central, one primary each); the two "E"s (EF/EV) and the two κs (κ_cond/κ_m); noise-only statistics belong to the generic arm; paired-twin Δz statistics are replay-side, so the 13/14 table is development evidence of signatures, not alarm-time attribution; K_phys vs K_diag; the pre-registration lives here as an unfrozen draft; `oracle_diag/` never existed — `latent_monitor/` is the layer.

## 5. Validation — exact commands and results

Cloud CPU (Python 3.11.15, numpy 2.4.4, scipy 1.17.1; `/home/claude/VERITAS`, a copy of the checkout without `.git`):

```
$ PYTHONPATH=src python3 -m pytest src/latent_monitor/tests -q
58 passed, 1 skipped in 21.74s            # skipped: test_torch_subject (torch not installed)

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev
DEVELOPMENT SMOKE — NOT CITABLE. …
C2 ΔF1 primary: -0.039 [-0.090, 0.003] → inconclusive
C4 all-cell AUROC (final_embedding): 0.84375 (harmful 4 / benign 8)
C4 all-cell AUROC (layerwise): 0.5625 (harmful 4 / benign 8)
(7.4 s; config hash 434eb52d874147e3; gate warning: freeze part 'core' is missing)

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --mode confirmatory
ConfirmatoryGateError: confirmatory mode refused:
  - freeze part 'core' is missing (protocol/frozen/core.json)
  - threshold 'kappa_m_tier1' has status 'provisional_dev'; not usable in confirmatory mode

$ PYTHONPATH=src python3 -m latent_monitor.run_table --out /tmp/table_check      # regression of the 6 Sep table
13 match, 1 documented, 0 mismatch                                              # 12 non-designed rows bit-identical to results/latent_monitor_tier1/table.json;
                                                                                # the two designed rows differ only in their random directions (seed-stream fix), same attribution and sign

$ cd latex && pdflatex … ×2
Output written on paper3_proposal.pdf (7 pages).  No undefined references or citations.
Overfull boxes: 2, both pre-existing (tikz figure at lines 340–347; "E0b" in the E-matrix).
Pages 1, 3, 4, 7 rendered to PNG and inspected: title/date, contribution paragraph, C2/C4 text, the widened C4 table, bibliography — all render.
```

Mac (the linked computer's local VM, `/Users/dowlingwong/Documents/VERITAS` mounted; Python 3.12.14, scipy 1.18.1, in a scratch venv — the repo's `.venv-cpu` recipe was not built because the tests need only numpy/scipy/pytest):

```
$ PYTHONPATH=src ~/venv-lm/bin/python -m pytest src/latent_monitor/tests -q -p no:cacheprovider
58 passed, 1 skipped in 15.91s

$ PYTHONPATH=src ~/venv-lm/bin/python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev
C2 ΔF1 primary: -0.044 [-0.110, 0.011] → inconclusive
C4 all-cell AUROC (final_embedding): 0.84375 (harmful 4 / benign 8)
C4 all-cell AUROC (layerwise): 0.5625 (harmful 4 / benign 8)
(4.7 s; config hash 434eb52d874147e3; commit f24430a-dirty)     # this run is the one checked in under results/

$ … --mode confirmatory  → refused with the same two reasons
```

**Observed cross-platform difference:** the C4 numbers and the config hash are identical on both machines; the C2 ΔF1 differs at the 0.005 level (−0.039 vs −0.044) because the L-BFGS logistic fits differ slightly across scipy/BLAS builds. The deterministic seeds cover the data and the bootstrap, not the optimiser's floating-point path. This is recorded here so that a future confirmatory run declares the platform and pins the environment; it is one more reason nothing in this pass is citable.

**Blocked / not checked:** `test_torch_subject.py` (torch absent in both environments used); the `uv sync` GPU environment; any trained-subject run; the realism arms; the public-data check; `reference/papers/` PDFs for Angelopoulos & Bates (verified on arXiv instead) and the MLST scope page (verified on the IOP site; the journal accepts either an applied-ML advance in a science or a methodological advance motivated by one).

## 6. What remains unrun, and the next gate

Unrun scientific evidence: everything. Specifically (a) the five-arm comparison and the all-cell C4 on a *trained* `TransformerSubject` under the alarm-time contract; (b) the same on arm B (`herald_simulation`, 14 cells) — the first transfer test; (c) TIDMAD (data + GPU); (d) the development precision study that sets the calibration count; (e) difficult N/S contrasts on Tier 1; (f) κ_m per arm.

Next gate: **declare κ_m for Tier 1 from a scientific requirement and settle `PREREGISTRATION.md` §7 items 2–3** (held-out severities/seeds, the C4 acceptance point) — then the trained-transformer development run under the contract. The `core` freeze waits for collaborator agreement (Junjie on title/C0 and the Tier-2 family list) and is not a reason to stop reversible development work.

## 7. Boundaries respected

No messages sent, nothing published or released, nothing pushed, no freeze written, no threshold invented, no existing numerical result replaced (the 6 Sep artifacts are untouched; the smoke writes to a new directory), no GPU or data campaign, no full training. The Linux CUDA lockfile was not forced onto macOS.
