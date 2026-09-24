# Independent audit of the 17 September cumulant pass — can the round close?

*17 September 2026, evening. Audited against the worktree at `778eb83` + uncommitted changes, from a separate environment (Linux VM, Python 3.10.12, numpy 2.2.6, scipy 1.15.3 — not the macOS/3.14 environment the implementing agent used). Report under audit: `docs/REVISION_REPORT_2026-09-17_cumulant.md`. Verdict first, evidence after.*

## Verdict

**Close D8, D10 and the document set. Do not close D7 as "completed" — close it as "implemented, signature row not yet informative", and dispatch the small follow-up below before the freeze.** The code is sound, the tests reproduce, the boundaries were respected, and the proposal builds inside the page budget. But the D7 residual statistic was estimated on the wrong population, and that — not sample size — is why the signature row returned 11/0/3. Re-estimated on the population the integration note meant, the same cells give the predicted signature on every row. That is a good result hiding behind a recorded negative, and it should be fixed before the number is read by anyone.

## What I verified independently

| item | agent's claim | independent check | status |
|---|---|---|---|
| Test suite | 94 passed, 1 skipped (baseline 84) | `PYTHONPATH=src python3 -m pytest src/latent_monitor/tests -q` → **94 passed, 1 skipped**, exit 0, on a different OS/Python/numpy | ✓ reproduced |
| Arm symmetry | `generic_rich_matched` 46 / `full_intermediate` 46 | `smoke_report.json` `arm_contract_status`: generic_rich 34, intermediate_only 12, full_intermediate **46**, generic_rich_matched **46**, drop arms 40/40 | ✓ |
| `GENERIC_COMMITTED` untouched | yes | `features.py` l. 66: `("gr_input_maha", "gr_output_maha", "z_mahalanobis")`, unchanged in the diff | ✓ |
| Cubic score isolated from the primary | yes | `smoke.py`: `paired_delta_auroc(s_task, s_gen, …)` is the primary; `supporting_intermediate_cubic(s_cubic, …)` is reported under `claim2.supporting_intermediate_scores` with `primary_estimate: false` | ✓ |
| Tags | `{raw_window, reference_fit}`, no evaluation-only source | every `b.add` of a new feature carries `{RW, RF}`; `test_new_features_are_tagged_alarm_time_and_never_evaluation_only` exists and passes | ✓ |
| Chart rule | `reading = cumulant / deviation` | `features.py` l. 237 keys on `subject.is_linear`; `LinearSubject.is_linear = True` (l. 50), `HypergraphSubject.is_linear = False` (l. 197); smoke `alarm_time_reference.reading = "cumulant"` | ✓ |
| 16 Sep artifacts untouched | yes | the 7 tracked files under `results/latent_monitor_{smoke_dev,smoke_dev_2026-09-16_pass2,tier1}` show no modification; only the two new 17 Sep directories are untracked | ✓ |
| D9 not implemented, not claimed | yes | only mentions are "deferred / not implemented" in `TODO.md` l. 44 and `PREREGISTRATION.md` §8 item 9 | ✓ |
| Pre-registration still unfrozen, PENDING unchanged | yes | header "DRAFT — UNFROZEN" intact; `PENDING` count 9 before and 9 after; items 7–10 added with ticked state | ✓ |
| Proposal build | **not run** (no TeX on the Mac) | built twice here after installing `lmodern`: **exit 0, 0 errors, 0 undefined citations, 8 pages**, bibliography ends about one-third down p. 8, the new mechanism paragraph sits on p. 3; the two new `\bibitem`s render as [21] and [22] | ✓ resolved — **no sentence needs cutting**; the committed `paper3_proposal.pdf` (8 pp) is stale and should be rebuilt in a TeX environment before circulation |
| Strict JSON | yes | both new JSON files load | ✓ |

## Findings

### F1 (blocking for the *reading* of D7, not for its code): the residual cumulant is estimated across windows, not within them

`CumulantNull.fit_residual` (`features.py` l. 157) takes each window's flattened whitened residual — a (C·N) = 2048-vector — as **one sample**, projects it on 3 reference-PCA directions, and forms the third cumulant of that 3-vector **across windows** (`third_central_moment(coords)`, n = 60 at evaluation). The signature runner reads the same object (`signature_c3.json` `chart.statistic`: "population connected third cumulant tensor … in the reference residual PCA chart (n_pcs = 3)").

That is not the object the integration note asked for. §2.1 of the note and the proposal's new paragraph speak of "the connected third cumulant of the whitened residual" — the cumulant of the whitened residual *samples*, which under the Gaussian reference are i.i.d. N(0,1) and of which every window has 2048. The across-window version throws away the within-window distribution entirely, keeps 3 of 2048 directions, and estimates a rank-3 tensor from 60 draws; the report's own diagnosis ("noise-dominated at n_eval = 60") is correct about the symptom and wrong about the remedy, because no Phase-B increase of n_ref fixes a statistic that ignores the samples.

**Reproduction.** Same subject, same cells, same 60 evaluation windows, same `SparseBurstCell`; per window, the standardised third and fourth moments of the 2048 whitened-residual samples; z against the reference windows' own spread (development check, two-sided |z| > 2.33):

| cell | skew z (median) | windows flagged | excess-kurt z (median) | windows flagged | predicted |
|---|---:|---:|---:|---:|---|
| reference | −0.1 | 0.02 | 0.0 | 0.02 | at reference |
| sigma_cov ×4 (corr up/down, bandwidth, line) — **no re-whitening** | ±0.2 | 0.02–0.07 | ±0.1 | 0.00–0.05 | at reference |
| sigma_struct:timing_jitter | −0.2 | 0.12 | 1.1 | 0.25 | at reference (documented) |
| event_in_span ×2 (oscillation, double pulse) | −0.1 / 0.6 | 0.02 / 0.10 | 0.1 / 0.2 | 0.03 / 0.05 | at reference |
| sigma_struct:gain_drift | **5.0** | 0.87 | −0.5 | 0.18 | moves |
| event:glitch | **6.1** | 0.83 | **24.2** | 1.00 | moves |
| sigma_struct:channel_loss | **−21.6** | 1.00 | **−7.3** | 1.00 | moves |
| sigma_non_gaussian:sparse_burst | **41.0** | 1.00 | **158.3** | 1.00 | moves |

Every row lands where the §III.1 hypothesis put it, including the three the agent's estimator missed (gain drift 0.77 → 5.0; glitch −2.28 → 6.1 / 24.2; sparse burst 0.60 → 41.0). Two further properties matter for the protocol: the covariance cells sit at reference **without** the evaluation-only re-whitening the agent's statistic needed, so the within-window statistic is alarm-time-legal as it stands; and it costs two scalars per window (or 2C per channel), independent of n_ref, so it is not subject to the n_ref ≲ 100 instability at all.

This table is a development check made during the audit, not a result; it is not written to `results/`. It is the reason the follow-up below is worth one short pass.

### F2 (minor, prediction): gain drift

The runner predicts "moves" for gain drift with the reason "a linear-Gaussian gain operation on the residual"; the module docstring (l. 24) says the opposite ("Gaussian operations on the residual"). For a third-cumulant statistic a Gaussian-preserving operation predicts *at reference*, so the prediction and its reason contradict each other. Empirically the within-window skewness does move (z = 5.0, F1 table) — the drift modulates the signal residual over the window, which is not a pure noise operation — so the *prediction* happens to be right for a reason the text does not give. Fix the reason string; keep the prediction.

### F3 (minor, text): the QUIVER sentence over-claims

`paper3_proposal.tex` l. 480–484: "The capacity-matched control **follows** the paired-seed, zero-initialised-gate evidence design of [22]: an augmented arm must coincide with its baseline at initialisation …". `generic_rich_matched` is a feature-count match for a logistic-regression classifier; nothing in it is zero-initialised or coincides with a baseline at step 0. The integration note asked for QUIVER as *precedent* for the argument form. Replace "follows" with "is the same argument form as" and drop "must coincide with its baseline at initialisation and", leaving "an augmented arm must earn any gain under identical splits and matched capacity, so a difference is read as information rather than capacity."

### F4 (cosmetic): `n_pcs` not surfaced

`smoke_report.json` `alarm_time_reference.n_pcs` is `None` at top level (the value lives inside the nested null records). Surface it once, since the pre-registration now declares it.

## What closes now

- **D8** — closes. Implemented, calibrated on `calibration`, reported as supporting with `primary_estimate: false`; at toy size the cubic score's cell-level AUROC is 0.50 (task length 1.00), which is the expected uninformative reading for 13 cells and is correctly labelled descriptive.
- **D10** — closes at the pass limit. `Subject` conformance, Eq. 4.3 normalisation, P3 Gaussian-zero and heavy-tail non-triviality, α = 0 ≡ tied linear AE — all tested; untrained; gated below Phase B in `TODO.md`.
- **Documents** — close, with F3's one-sentence edit and F2's reason string.
- **D7** — code closes (estimator module, arm symmetry, tags, chart rule, tests); the **feature definition and the signature row do not**. Reclassify item 7 in `PREREGISTRATION.md` §8 as "TICKED yes; residual statistic definition under revision (F1)" and run the follow-up.

## Follow-up pass (D7-bis) — copy-ready

```text
Bounded follow-up to docs/REVISION_REPORT_2026-09-17_cumulant.md, finding F1 of
docs/reviews/2026-09-17_cumulant_independent_audit.md. Same repository
discipline and boundaries as docs/IMPLEMENTATION_PROMPT_CUMULANT.md.

1. In cumulant.py add within-window sample cumulants of the whitened residual:
   per window, the standardised third and fourth cumulants of its C·N samples
   (pooled) and per channel (2C values), about the reference cell's per-channel
   moments. These are the connected cumulants of the sufficient statistic in
   the natural chart; document that the existing across-window PCA tensor is a
   different object (the third cumulant of the window-level distribution) and
   keep it as a development diagnostic only.
2. In protocol/features.py make the within-window cumulants the generic_rich
   third-order features (gr_resid_c3_skew_pooled, gr_resid_c4_kurt_pooled,
   gr_resid_c3_skew_ch_{i}, gr_resid_c4_kurt_ch_{i}), tags {raw_window,
   reference_fit}; retire gr_resid_c3_{pc_i,norm,maha} from the arm (keep the
   code path for the diagnostic); re-truncate the capacity control so
   generic_rich_matched is count-matched again; keep im_{hook}_third_maha as is
   (the pooled-hook object is correct). Surface n_pcs in to_dict (F4).
3. Re-run run_cumulant_signature.py with the within-window statistic and a
   window-level reference null (bootstrap over reference windows), writing to a
   NEW dated directory; keep the 17 Sep row as recorded. Fix the gain-drift
   reason string (F2). Report the match table; a negative is reportable.
4. Apply F3 to latex/paper3_proposal.tex (one sentence). Rebuild in a TeX
   environment if available; the audit confirmed 8 pages with the current text.
5. Update PREREGISTRATION.md §8 item 7, EXPERIMENT_DESIGN.md §III.1/V.4, the
   revision report and TODO to the actual status; add tests: within-window
   cumulant Gaussian-zero at n = C·N with declared tolerance; arm symmetry after
   the swap; tags. Run the full suite and a new dated smoke. No freeze, no κ_m,
   no commit, no push.
```

Estimated cost: half a day. After it, D7 can be marked completed at the same level as D8 and D10, and the round closes.
