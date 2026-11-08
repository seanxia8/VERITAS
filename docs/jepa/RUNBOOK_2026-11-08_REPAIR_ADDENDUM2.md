# Addendum 2 — latent gain estimator, a gain concept that is linearly present, and the ledger total

**Status: v1 (8 Nov 2026). Not yet reviewed or accepted.** It extends
`docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` (`45e5ff98…d119b`) and
`docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md` (`abe0904a…0621a`), both unchanged. It uses their
extensions, `<ext>` roots, results directories, readings and thresholds. Where it is silent, they
apply.

Base: `origin/JEPA = 146eea5`, implementation `08e9569`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-08_REPAIR_ADDENDUM2.md` before any outcome-generating step.

**No training. No data generation.** One new concept definition (E3u, §3), authorised below. No
existing reading rule or threshold changes. GPU: evaluation only, ≤ 0.08 L40S h.

---

## 0. Authority

Quote both lines verbatim in `DECISIONS.md`.

1. The author, 8 Nov 2026, 15:45 CET, choosing option A:
   > "按A做，并更新周报"

   Option A was: fix the latent gain estimator and rerun its readings, replace the LEACE concept
   with the occupancy-weighted gain design, and correct the S1 ledger base. Adding the E3u concept
   after the 7 Nov outcomes is a specification change (S3); this line authorises it. E1 and E2r
   stay reported as they are.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

Work as in the combined runbook's front matter §F2: the Mac `JEPA` worktree
(`/private/tmp/veritas-jepa-20261107`, or a fresh worktree at `origin/JEPA`); never the main checkout;
data and GPU steps over ssh.

---

## 1. Defects and gaps

| ID | What is wrong | Effect |
|---|---|---|
| E1 | `latent_estimate` (`ext_repair.py:2704-2735`) solves on a column-standardised design but returns the solution in standardised units. The standardisation also centres the columns across embedding rows, an affine shift with no intercept | X2-d latent stream (Pearson r −0.14 to −0.33, RMS ≈ the gain spread), X2-e EST(latent), and latent repair with δ̂_latent are invalid |
| E2 | The LEACE concepts E1 (one weighted log gain per event) and E2r (the run's 105 module log gains) are not linearly present in event-mean embeddings (unerased R² ≤ 0.02 for every trained encoder) | the price of invariance for trained encoders (X3-1 to X3-4) is unmeasured |
| E3 | `v4_caps.E2_lucid_l40s_deepthought2.charged_gpu_hours` in `RESOURCE_LEDGER.json` still reads 129.7461; the booked jobs sum to about 133.65 | the addendum S1 check reported 130.0961 instead of about 134.0 (still ≤ 160) |

---

## 2. Fix E1 — the latent gain estimator

**Rule (replaces the "standardised design" wording of Part A §6 and Appendix C):**
- Scale only, never centre.
- Column scale s_m = the root mean square of column m of the training-fold designs W Â_raw diag(ū_r), pooled over training runs and embedding rows; s_m = 1 where it is 0.
- Solve the ridge on the scaled design for δ′, then return **δ_m = δ′_m / s_m**.
- λ ∈ `probes.ALPHAS`, by the same five-fold run-grouped CV on probe-train runs. Â, z̄ᴺ, W and s are refit inside each fold.

**Tests (must pass):**
- Noise-free synthetic data through the scaled path: δ recovered to 1e-8 relative with λ = 0, for column scales spanning 10⁻³ to 10³.
- With λ > 0, the estimate is invariant to multiplying any design column by a constant (up to the matching penalty change) and is returned in raw units.
- Every caller uses the same path (CV, final fit, X2-e).
- No caller reads `delta` from a scaled solve without the back-transform (a grep test).

**Reruns:**
- X2-d latent stream (CPU);
- X2-e: the EST(latent) probe-test sets re-encoded for the 13 core models (GPU), then ρ_EST(latent);
- latent repair with δ̂_latent (CPU).

The source and raw streams, X2-b, X2-c, X2-f and R-1 are unaffected and reused; a test asserts this.

---

## 3. Fix E2 — a gain concept that is linearly present

**New concept E3u** (added beside E1 and E2r, which stay reported as `not_applicable`). For each event e,
in its own environment (N or D):

c_e,m = ln G^env_r,m · n_e,m,

where G^env is the run's module gain in that environment and n_e,m the number of hit PMTs of module m
in the event. This is the X2-b occupancy design written per environment; its drift part has
equivariance R² up to 0.77.

- Hit counts come from a CPU `bincount` of `geometry.module_of_pmt` over the event's `pmt_id`, aligned on `(run_id, event_id)`, as in X2-b.
- **Rank:** the Part A §7.2 max-statistic rule, unchanged: 200 run-id permutations, N and D together, keyed `keyed_rng(20261106, "paper3/x3/leace_perm", f"{label}/{domain}/E3u")`.
- **Applicability:** unchanged. If the unerased pooled R² is ≤ 0.02, the variant is `not_applicable`; if the erased R² is > 0.02, it is `unavailable (erasure incomplete)`.
- **Controls:**
  - 20 Haar-random subspaces of the same rank (`…/leace_rand`, `f"{label}/{domain}/E3u/{i}"`);
  - **a hit-count control:** erase, at the same rank r, the concept n_e (the 105 hit counts alone).

  c_e mixes gains with hit counts, and hit counts carry physics. The hit-count control measures the cost of erasing hit-count information alone.
- **Gain-specific** (replaces the X3-1 flag for E3u only): all three must hold —
  - ρ_E(E3u) > the 95th percentile of the 20 random-control ratios;
  - the run-cluster CI of res_E3u − mean(res_random) excludes 0;
  - ρ_E(E3u) > ρ_E(hit-count control), with the run-cluster CI of res_E3u − res_hitcount excluding 0.
- **Models and data:** as Part A §7.2. Photon readings for MAE × 4, T-1 × 4 and frozen-target × 2; electron energy for the four MAE models. Cached X1 embeddings only; no GPU.
- **Readings:** X3-1 to X3-4 with the Part A §7.2 thresholds; X3-5 adds the E3u ratios beside the raw-feature value.

**Tests:**
- c_e matches a direct per-PMT computation on a synthetic run.
- The hit-count control has the stated rank.
- Permutation keys include `E3u`.
- The gain-specific rule on synthetic values.

---

## 4. Fix E3 — the ledger total

- `charged_gpu_hours` of `E2_lucid_l40s_deepthought2` is recomputed as the sum of all booked L40S
  jobs and campaigns, and written once with a `DECISIONS.md` line. The expected value is about
  133.65; record the exact sum.
- Every S1 check (`ext_repair.s1_projection`, `ext_shortcut.addendum_s1_check` and this
  addendum's) must fail closed if `charged_gpu_hours` differs from that sum by more than 1e-3 h.
- Test: an inconsistent ledger fails S1.
- **S1 for this addendum:** charged total + 0.08 ≤ 160.

---

## 5. Budget

- **GPU:** ≤ 0.08 L40S h, evaluation only. The EST(latent) sets are projected at 0.02–0.06 h (the
  7 Nov X2-e job took 0.0204 h). The guard refuses any job that would cross 0.08 h.
- **CPU:** ≤ 3 wall-hours.
- **Entries:** ≥ 30 000 free after the projected writes.
- The earlier caps are closed and not reused.

---

## 6. Review, acceptance and gates

- Focused tests plus the full `src/jepa_noise/tests` suite once on the final SHA (deselect only the
  three known `test_guard` tests); both dry runs.
- **One fresh-context focused review** of the delta from `08e9569`. PASS or BLOCK per item, with
  file:line evidence:
  - (i) E1: scale-only standardisation and the back-transform on every path; the unaffected items untouched;
  - (ii) E2: the c_e and n_e concepts, rank rule, keys, controls and the gain-specific rule;
  - (iii) E3: the ledger sum and the fail-closed S1 in all three checks;
  - (iv) the gates, the import closure, the 0.08 h guard and entries.
- **Acceptance:** re-issue `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json` with the new
  `impl_commit` and the fields `addendum2_path` and `addendum2_sha256`. Both parts' gates also
  require the committed addendum 2's sha256 to match.
- Nothing outcome-generating runs before this record exists.

## 7. Execution order

1. **CPU:** E3 ledger fix and S1 → E1 X2-d latent rerun → E2 E3u LEACE family.
2. **GPU:** EST(latent) sets.
3. **CPU:** X2-e (latent) and latent repair with δ̂_latent → reports.

## 8. STOP conditions (the only ones)

- S1;
- S3 (any change beyond §0's authorisation);
- S4;
- S9 (a reproduction or the E1 synthetic recovery still fails after one fix attempt; stops the
  dependent items only);
- an import-closure failure;
- an entries-preflight failure;
- the 0.08 h cap.

## 9. Deliverables

- `results/jepa/20261106_repair/ADDENDUM2_2026-11-08.json`, with the E1 reruns, the E3u family with
  its controls, and the ledger correction.
- `READINGS_SUMMARY.md`, `SUMMARY.json`, `OUTCOME.json` and figures updated. The "Superseded"
  table gains every replaced value with defect IDs E1–E3.
- A `STATE.md` section, a ledger line, and the `DECISIONS.md` lines (both §0 quotes, the S3
  authorisation, the ledger correction).
- **Record correction:** the 8 Nov status message quoted the position-free JEPA's full-input vertex
  as 0.1260 / 0.1246 m. 0.1246 is seed 9000's half-pass value. The full-input values are
  0.1260 / 0.1405 m (`results/jepa/20261107_shortcut/readings.json`). Note it in `STATE.md`.

## 10. Out of scope

- Training; data generation;
- any other new concept, variant, model or factor;
- changes to the other readings or thresholds;
- Paper 3 information-destroying drifts (a generation budget and a new runbook).
