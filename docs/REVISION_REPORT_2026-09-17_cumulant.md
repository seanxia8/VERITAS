# Fisher-cumulant integration — implementation report, 17 September 2026

_Third local pass, implementing `docs/ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`
blocks D7, D8 and D10 on the dirty `dev` worktree at `778eb83`. **Nothing is
frozen; no κ_m, margin, severity or seed range was invented; no trained-model or
transfer result exists; no scientific conclusion is drawn.** Status vocabulary:
interface implemented · unit-tested control · development smoke ·
trained-model evidence · transfer evidence · confirmatory — this pass reaches the
first three only. The transparent self-review is
`reviews/2026-09-17_cumulant_self_review.md`._

Author decisions in force for this run (recorded in `PREREGISTRATION.md` §8
items 7–10):

- **D7 — pooled third-moment transform on both sides: TICKED yes, completed.**
- **D8 — cubic task score as the supporting intermediate score: TICKED yes, completed.**
- **D9 — R(3) hook-selection diagnostic: NOT TICKED, deferred.** Not implemented
  and not mentioned as implemented anywhere.
- **D10 — frozen-propagator hypergraph subject: TICKED yes, design and interface
  only, completed.** No training.

## 1. Files changed

| file | change |
|---|---|
| `src/latent_monitor/cumulant.py` | **new.** Connected third/fourth cumulant estimators, per-window and pooled-channel third tensors, PCA projection helpers, the KL truncation-error ratio (development-table only). Docstring states the raw-window-kurtosis distinction and the chart caveat. |
| `src/latent_monitor/reference.py` | `pooled_stage` returns `(mean, second, third)`; the third is the pooled channel third central-moment tensor (flattened). Callers updated. |
| `src/latent_monitor/protocol/features.py` | `CumulantNull` (compression to `n_pcs` reference-PC coordinates + one Frobenius-deviation scalar); `AlarmTimeReference` gains `hook_third_nulls`, `resid_cumulant`, `reading`, `task_whitening`, `task_I3` and records the chart rule in `to_dict`; builder adds `gr_resid_c3_{pc_i,norm,maha}`, `im_{hook}_third_maha`, `raw_task_cubic`; `QUADRATIC_BASIS` extended and the control re-truncated. |
| `src/latent_monitor/protocol/availability.py` | Manifest declares the new `generic_rich` and `intermediate` features and re-truncates the `generic_quadratic` count to `2·(3 + n_pcs)` so `generic_rich_matched` is again count-matched. |
| `src/latent_monitor/task_metric.py` | `TaskMetric.whitening()` (M_task^{1/2}) and `TaskMetric.cubic_aligned(delta, I3)`. |
| `src/latent_monitor/protocol/consequence.py` | `supporting_intermediate_cubic` (supporting-only report of the cubic score). |
| `src/latent_monitor/protocol/__init__.py` | exports `supporting_intermediate_cubic`. |
| `src/latent_monitor/protocol/smoke.py` | calibrates `raw_task_cubic`, aggregates it at cell level, reports `claim2.supporting_intermediate_scores` and the cell table column; 17 Sep smoke directory. |
| `src/latent_monitor/hypergraph_subject.py` | **new** (D10). `FrozenPropagators`, normalised graph and order-3 Laplacians (Eq. 4.3), `HypergraphSubject` (`Subject` interface, frozen P2/P3, QUIVER zero-initialised gate). Untrained. |
| `src/latent_monitor/run_cumulant_signature.py` | **new** (D7). Tier-1 signature-row runner on the linear subject with paired replay and the non-Gaussian sparse-burst family; writes strict JSON + markdown, `NOT CITABLE`. |
| `src/latent_monitor/tests/test_cumulant.py` | **new.** 10 acceptance tests (see §5). |
| `results/latent_monitor_smoke_dev_2026-09-17_cumulant/` | **new** development smoke (`smoke_report.{json,md}`), not citable. 16 Sep artifacts untouched. |
| `results/latent_monitor_sig_c3_2026-09-17/` | **new** D7 Tier-1 signature row (`signature_c3.{json,md}`), not citable. |
| `latex/paper3_proposal.tex` | mechanism paragraph on the KL-cumulant rung (cite `bal2026triality`), one sentence citing `bal2026quiver`'s paired-seed zero-initialised design, two `\bibitem`s. No new claim; no estimand changed. |
| `docs/EXPERIMENT_DESIGN.md` | §I.6 candidate third subject + component row; §III.1 residual-cumulant row and paragraph; §V.2 cumulant-rung paragraph; §V.4 new arm features; §V.5 supporting cubic score; §V.6 the two 17 Sep artifacts. |
| `docs/PREREGISTRATION.md` | §1 supporting results (cubic companion); §3 information contract (third-order features and cubic score); §8 items 7–10 with ticked state. Still **UNFROZEN**; every **PENDING** value unchanged. |
| `docs/TESTBEDS.md` | §2.6 candidate third subject (design and interface only). |
| `docs/TODO.md`, `README.md`, `docs/REVIEW_PROMPTS.md`, `docs/IMPLEMENTATION_PLAN.md` | status updated in the repository vocabulary; review red flags extended; §8.7 added. |
| `references.bib` | `bal2026triality`, `bal2026quiver` added. |
| `docs/reviews/2026-09-17_cumulant_self_review.md` | **new** transparent self-review. |

Not touched: `GENERIC_COMMITTED`, the two claims, their estimands, the arm
names, the splits, the hard-matching rule, the information-contract phases,
`Claude outputs/`, `_to_delete/`, the 16 September results, `noise_module`,
`herald_simulation`, `prometheus_simulation`, `tidmad_transformer`,
`reconstruction_model`, notebooks.

## 2. Bibliography verified (field by field against arXiv)

- `bal2026triality` — arXiv:2605.03063, title, authors (Bal, Klute, Maier,
  Spannowsky), v2 revised 7 May 2026, hep-ph. Verified at
  <https://arxiv.org/abs/2605.03063> on 17 Sep 2026.
- `bal2026quiver` — arXiv:2606.02785, title, authors (Bal, Binder, Klute,
  Maier, Spannowsky), 1 June 2026, cs.LG. Verified at
  <https://arxiv.org/abs/2606.02785> on 17 Sep 2026. Cited only for the
  paired-seed zero-initialised-gate evidence design; the quantum content plays
  no role.

No result is paraphrased from either paper beyond what the integration note
quotes (the KL expansion / connected-cumulant triality with the natural-chart
caveat; the zero-initialised gate design).

## 3. What the blocks do

- **D7.** The connected third cumulant of the whitened residual and of the
  pooled channel hooks. Compression is `n_pcs` reference-PCA coordinates plus
  one Frobenius-deviation scalar. The transform is on **both** arm sides or
  neither: `gr_resid_c3_*` in `generic_rich` and `im_{channel,token}_third_maha`
  in `intermediate`, with `generic_rich_matched` re-truncated (46 / 46 at the
  smoke's size). The fourth-order companion exists (`cumulant.fourth_cumulant`)
  and is wired into no arm. The chart rule is enforced in code and recorded
  (`AlarmTimeReference.to_dict`): `reading = "cumulant"` on the linear subject
  and for the whitened residual under the Gaussian reference; `"deviation"` from
  the reference cell's own third moment otherwise.
- **D8.** `raw_task_cubic = Δ_a Δ_b Δ_c Î3_abc`, `Î3` on `reference_fit` `z` in
  the M_task-whitened chart, calibrated on `calibration` windows, reported beside
  the task length via `consequence.supporting_intermediate_cubic`. It cannot
  enter `paired_delta_auroc` or `GENERIC_COMMITTED` (code path inspected).
- **D10.** `hypergraph_subject.py`: `Subject` interface; P2 the noise-Laplacian
  PE from the measured covariance; P3 from the measured third noise cumulant
  with the sign as an edge attribute and `|w|` in the Laplacian; P2/P3 frozen
  from the reference cell; QUIVER's zero-initialised gate (exactly the tied
  linear AE at α = 0). No training script, no GPU run, no result.

## 4. Tier-1 signature row (development, non-citable)

`results/latent_monitor_sig_c3_2026-09-17/signature_c3.{json,md}`, linear
subject, C = 8, N = 256, k = 6, 200 fit / 60 eval, 100 noise records, seed 0.
Statistic: population connected third-cumulant tensor of the whitened residual
in the reference residual PCA chart (`n_pcs = 3`), with a 400-draw
window-bootstrap null (q99 z = 2.33). Bal et al.'s truncation-error ratio
`q/(q+c)` along each cell's mean projected displacement is reported beside it
(development-table quantity only, per the note; e.g. 1.00 for the reference and
0.03 for channel loss).

| group | outcome |
|---|---|
| reference (control) | at reference (z −0.07) |
| covariance N ×4 (re-whitened) | at reference (z −0.6…−0.1) — **no false positives** |
| in-span S ×2 | at reference (z −1.7, +0.2) |
| geometry ×2 | skipped (different `(C·N)` chart, declared) |
| gain drift | at reference (z 0.77) — match: linear-Gaussian, predicted at reference |
| channel loss | moves (z 430) — match: dead channels leave a deterministic residual |
| glitch event | at reference (z −2.28) — MISMATCH against the out-of-support prediction |
| non-Gaussian sparse burst | at reference (z 0.60) — MISMATCH against the non-Gaussian prediction |

**Result: 12 match / 0 documented / 2 mismatch.** This is a **development
negative** for the separator at `n_eval = 60, n_pcs = 3`: the glitch event and
the non-Gaussian sparse burst did not exceed the reference window-bootstrap
band, while the covariance cells showed no false positives and the
linear-Gaussian structural cells behaved as predicted. Reported as recorded and
not explained away; the estimator is a population tensor of per-window rank-1
third moments, which is noise-dominated at this size (consistent with the
recorded instability of reference distances at `n_ref ≲ 100`).

## 5. Acceptance tests (`src/latent_monitor/tests/test_cumulant.py`)

Gaussian-zero cumulant with declared tolerance and n · orthogonal invariance and
shear non-invariance recorded · fourth companion and truncation ratio · chart
rule (`cumulant` / `deviation`) · arm symmetry and matched capacity + manifest /
builder agreement · tag and leakage coverage of every new feature · cubic
aligned contraction, whitening and calibration · Laplacian normalisation
(Eq. 4.3) · P3 Gaussian-zero with declared tolerance and non-Gaussian
non-triviality · `Subject`-interface conformance and α = 0 equals the tied
linear AE. (D10 tests included; D9 has no test because it is not implemented.)

## 6. Validation — exact commands and exit codes

macOS, Python 3.14.6, numpy 2.4.6, scipy 1.18.0:

```
$ PYTHONPATH=src python3 -m pytest src/latent_monitor/tests -q -p no:cacheprovider     # exit 0
  94 passed, 1 skipped in 12.95s            # skipped: test_torch_subject (torch absent); baseline 84 passed, 1 skipped

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_2026-09-17_cumulant   # exit 0
  Claim 1 ΔF1 (hard set): 0.0 [0.0, 0.0] → inconclusive (n_windows 20 < 30)
  Claim 2 ΔAUROC: 0.1667 (task 1.0 vs generic 0.8333)

$ PYTHONPATH=src python3 -m latent_monitor.run_cumulant_signature --out results/latent_monitor_sig_c3_2026-09-17   # exit 0
  12 match, 0 documented, 2 mismatch

$ python3 -c "import json; json.load(...)"                                             # exit 0, strict JSON both files

$ PYTHONPATH=src python3 -m latent_monitor.protocol.smoke --mode confirmatory           # exit 1 (expected)
  ConfirmatoryGateError: freeze part 'core' is missing; κ_m 'provisional_dev' not usable

$ which pdflatex                                                                        # exit 1 — toolchain UNAVAILABLE
```

**Proposal build: not run.** `pdflatex` is not installed on this machine, so the
two-pass build and PDF inspection of `latex/paper3_proposal.tex` could **not** be
performed. The mechanism paragraph and the designed-controls sentence were kept
tight (≈9 added lines) to protect the 8-page budget, and the two `\bibitem`s
were added to the document's self-contained bibliography, but the page count and
citation resolution are **unverified**. This is the honest residual of the pass's "Testing and review" requirement; the next environment with TeX must run
`cd latex && pdflatex ×2` and check the page count before the note is circulated.
**If the build overflows the 8-page budget, the sentence to cut is the final
clause of the new mechanism paragraph — "it is not a third claim and changes no
estimand" — because the same statement is already made in the "Two claims"
section; every other added sentence carries content the integration note
requires.**

## 7. Blocks ticked / completed / blocked

- **D7 — completed.** Interface implemented, unit-tested control, development
  smoke; chart rule enforced. Arm symmetry and matched capacity verified.
- **D8 — completed.** Supporting cubic score implemented, calibrated and
  reported; blocked by design from the primary estimand.
- **D9 — deferred by the author.** Not implemented; not mentioned as
  implemented.
- **D10 — completed to the pass limit.** Interface and Laplacian tests only; no
  training, no GPU run, no result; gated on the trained Tier-1 run; one of two
  nonlinear subjects, both to be reported, neither chosen by result.

## 8. Next scientific gate

Unchanged in kind, with one addition: **Phase B** must now also size `n_ref` and
`n_pcs` for the enlarged third-moment feature set — the 17 Sep signature row is
the first, non-citable evidence that a third-moment tensor is noisier than a
second. After Phase B: declare κ_m for Tier 1 from a scientific requirement and
run the trained subject in development mode. D10 stays below Phase B; D9 is
revisited only when designed controls exist for the subject whose hooks it would
choose among. Completion of this pass means the supporting cumulant machinery
is implemented, tested and documented consistently and the pre-registration
records the decisions it depends on; it does not mean either claim is
validated, that any signature is confirmed, or that the paper is submission
ready.

## 9. Boundaries respected

No GPU training; no data download; no collaborator contact; no freeze; no κ_m,
margin, severity or seed range invented; no commit, push or publish. The two
claims, their estimands, `GENERIC_COMMITTED`, the arm names, the splits, the
hard-matching rule and the information-contract phases are unchanged. No VQC or
quantum machinery was added; no architecture beyond D10, and D10 only as design
and interface. No cumulant is computed on a nonlinear latent and called a
cumulant. The 16 September smokes and the 6 September Tier-1 table are preserved
and not reinterpreted.
