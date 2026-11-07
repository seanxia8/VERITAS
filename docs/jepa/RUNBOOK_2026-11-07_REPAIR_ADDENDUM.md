# Addendum to the combined repair-and-shortcut runbook — fixes, completion and the R-2 evaluation

**Status: v1 (7 Nov 2026). Not yet reviewed or accepted.** It extends
`docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` (sha256 `45e5ff98…d119b`, unchanged) and
uses its extensions, `<ext>` roots, results directories, readings and thresholds. Where this
addendum is silent, the combined runbook applies.

Base: `origin/JEPA = 9f5579e`, implementation `0cf61ff`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md` before any outcome-generating step.

**No training. No data generation. No change to any reading rule or threshold.** The only GPU work
is evaluation (§6).

---

## 0. Authority

Quote both lines verbatim in `DECISIONS.md`.

1. The author, 7 Nov 2026, 22:32 CET, choosing option A (fix the defects, complete the summary,
   evaluate the position-free JEPA; evaluation only, no training, no data):
   > "写进weekly report，然后我选A，你写一个小runbook"
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

The option text said "about 0.2 L40S h". The cap here is 0.35 h because the drift-map fix (D1) also
needs the latent-calibrated input sets re-encoded, and the X3-4 fix (D2) may need the random-encoder
gain-blind embeddings re-encoded. Both are evaluation only.

Work exactly as in the combined runbook's front matter §F2: edit, commit and push in the Mac `JEPA`
worktree (`/private/tmp/veritas-jepa-20261107`, or a fresh worktree at `origin/JEPA`); never touch
the main checkout; run data and GPU steps over ssh on Deepthought and deepthought2.

---

## 1. Defects and gaps this addendum closes

| ID | What is wrong | Effect on the 7 Nov record |
|---|---|---|
| D1 | The fitted drift map Â is used in raw units, but its coefficients act on standardised inputs | X2-c, the X2-d latent stream, the X2-e latent items, X2-f and Part B R-1 are invalid |
| D2 | X3-4 compares a mean-error point estimate with an RMS bootstrap interval | the vertex point (0.1337 m) lies outside its interval (0.1147–0.1208 m) |
| D3 | Execution fixes 7–8 run through a runtime adapter outside `<P>` | the reviewed code cannot reproduce `x3.json` on its own |
| D4 | The committed summary omits readings the runbook requires; figures lack legends | X1-c, X1-d, X1-e, X1-f, X2-a, X2-d stream labels, X2-e, X2-f and the LEACE rows are not in the record |
| D5 | Part B R-2 was trained but not evaluated (cap); the empty reading is labelled "single-seed" | Q-R2 is open |

---

## 2. D1 — the drift-map scale

**The defect.**
- `probes.fit_ridge` standardises its inputs: `RidgeFit.predict(u) = ((u − mean) / scale) @ coefficients + intercept` (`probes.py:27-31, 42-52`).
- `ext_repair._stack_coefficients` (`ext_repair.py:2796-2810`) and `_fit_equivariance_occupancy` (`:2646-2664`) stack `RidgeFit.coefficients` as Â without dividing by `scale`.
- Every consumer then applies Â to raw u:
  - `_x2c` (`:2812-2830`), latent repair with true gains;
  - `_latent_repair_rows` (`:3094-3104`), latent repair with estimated gains (X2-e);
  - `latent_estimate` and `latent_cv` (`:2501-2595`), the X2-d latent stream;
  - `identifiability_spectrum` (`:2833`), X2-f.
- R²_eq is not affected: `equivariance_fit` scores with `probe_predictions`, which standardises correctly.

**The fix (in `<P>`).**
- One helper, `raw_map(fit)`, returns Â_raw with Â_raw[d, m] = coefficients_d[m] / scale_d[m], for each embedding dimension d's fit.
- Columns that `fit_ridge` marked constant (scale set to 1, `probes.py:44-46`) get Â_raw[d, m] = 0.
- Every consumer listed above uses `raw_map`. No other code path may read `RidgeFit.coefficients` directly (enforced by a test that greps the module).
- The repair stays as the runbook defines it, z̃ = z_D − u Â_rawᵀ, with no intercept.

**Tests (must pass):**
- **Identity:** u Â_rawᵀ equals `predict(u) − predict(0)` to 1e-10 relative on random data, including constant columns.
- **Recovery:** on synthetic data Δz = u Aᵀ (no noise), Â_raw recovers A to 1e-6 relative.
- **Consumers:** each consumer gives the same result as a reference implementation that calls `predict(u) − predict(0)`.

**Self-check on the fitting conditions (CPU; before any X2 rerun).** On probe-test, for each of the
13 core models:
- **Identity on real data:** u_S Â_rawᵀ equals `predict(u_S) − predict(0)` from the stored fits, to 1e-8 relative. A failure after one fix attempt is a STOP for the D1 reruns (S9); the other items continue.
- **Descriptive:**
  - the removed share 1 − E‖z̃_S − z_N‖² / E‖z_S − z_N‖², with z̃_S = z_S − u_S Â_rawᵀ (u from the N hits, as in the fit), reported beside R²_eq;
  - the recovered fraction on S, (v_S − v_S̃) / (v_S − v_N), with the frozen readout P_N.
- **Flag (no STOP):** a model with R²_eq ≥ 0.4 whose recovered fraction on S is < 0 is reported as "latent repair fails even on the fitting condition".

**Reruns after the fix.** All CPU from the cached, sha-verified embeddings, except the EST(latent)
sets:
- X2-c (latent repair with true gains);
- X2-d, latent stream only (the source and raw streams do not use Â and are reused unchanged; the test asserts this);
- X2-f;
- X2-e: the EST(latent) probe-test sets re-encoded for the 13 core models (GPU, §6), then ρ_EST(latent); latent repair with δ̂_latent. EST(source) and EST(raw) are reused;
- Part B R-1, with the corrected ρ_LR. Its families and rule are unchanged.

The X2 reading rules and classes are those of Part A §6; the R-1 rule is Part B §3.

---

## 3. D2 — the X3-4 interval statistic

**The defect.** `_compare_at_d` (`ext_repair.py:4394-4414`) takes the point difference of
`resolution`, which is the mean 3-D error for vertex and the RMS for log-yield and energy. It
always bootstraps with `statistic=rms`.

**The fix (in `<P>`).**
- For vertex, the point and the interval both use the mean error (`ef.paired_run_bootstrap`, as in X1).
- For log-yield and energy, both use the RMS.
- In every case the point is computed on the same aligned pairs as the interval.
- The reviewer checks every other comparison in `ext_repair` and `ext_shortcut` for the same mismatch (X1-c, X3-2, X3-3, `_gain_specific`).

**Tests:**
- On synthetic paired errors, the point lies inside its 95 % interval for each metric.
- The vertex statistic is the mean.

**Rerun.** X3-b readings (X3-1 to X3-5) for all families.
- The per-event errors are not in `x3.json`, because `_strip_internal` drops them. They are recomputed from cached embeddings where a sha-verified cache exists.
- The `init9000` gain-blind-token embeddings are re-encoded on GPU only if no verified cache exists (§6).
- Raw features are recomputed on CPU.

---

## 4. D3 — fold execution fixes 7–8 into `<P>`

- `_x3_five`: a `null` efficient CRLB read from `x3_fisher.json` means ∞, as the runbook defines (Part A Appendix E).
- `_strip_internal`: also drops `fitted` and `prediction` objects, so `x3.json` serialises.
- Tests for both. The runtime adapter `/work/dwong/jepa_tmp_20261107/adapter2/jepa_repair_adapter2.py` is retired, and its retirement is recorded.

---

## 5. D4 — a complete committed summary

`results/jepa/20261106_repair/READINGS_SUMMARY.md` and `SUMMARY.json` gain, per model and seed:
- **X1-c:** v_SWunit (frozen) − v_D (refit), and v_SWunit (refit) − v_D (refit), each with its CI and class;
- **X1-d:** the three ceilings;
- **X1-e:** ρ_NOISY for σ ∈ {0.01, 0.02, 0.05, 0.10}, and σ½;
- **X1-f:** log-yield and electron-energy resolutions for N, D, SWunit and HW;
- **X2-a:** the variance shares, the frozen-probe costs and the additivity check;
- **X2-d:** per stream, **labelled** source / raw / latent: median per-run RMS error, Pearson r and coverage;
- **X2-e:** ρ_EST(k) for the three streams beside ρ_SWmod;
- **X2-f:** the singular values of W Â_raw and the per-module CRLB;
- **X3-b, for every family including LEACE E1 and E2r:**
  - the selected rank r;
  - the unerased and erased concept R²;
  - ρ_E with the 20 control ratios' 95th percentile and the gain-specific flag;
  - X3-2, X3-3 and X3-4 (corrected).

A **"Superseded 7 Nov values"** table lists each replaced number (ρ_LR, X2-d latent, X2-e latent,
X2-f, R-1, the X3-4 intervals), the new value, and the defect ID.

All six figures get axis labels and legends; the identification figure names its three streams.

---

## 6. D5 — Part B R-2 evaluation, and the GPU budget

- **R-2 evaluation:** one job for the two run directories, exactly Part B §5.3: full input and half
  pass via `ext_h12.evaluate_run`, plus the label-pass `f_event` over the §5.2 events. The readings
  R2-a, R2-b and R2-c follow Part B §5.4, unchanged.
- **Label fix:** `ext_shortcut.r2_readings` reports `not_run` when the runs carry no evaluation,
  never "single-seed (not classified)". Add a test.

**GPU cap for this addendum: 0.35 L40S h**, booked separately. Part A and Part B caps are closed and
not reused.

| Item | Projection (L40S h) |
|---|---|
| R-2 evaluation | 0.14 |
| EST(latent) probe-test sets, 13 models × 13 s × 1.3 | 0.06 |
| `init9000` gain-blind tokens, only if not cached | ≤ 0.07 |
| margin | 0.08 |

- The job guard refuses any job that would cross 0.35 h. Remaining items are then `not_run (cap)`.
- **S1:** ledger (≈ 132.0) + 0.35 ≤ 160.
- **CPU:** ≤ 3 wall-hours.
- **Entries:** ≥ 30 000 free after the projected writes (44 286 at the last check).

---

## 7. Review, acceptance and gates

- Tests: the new focused tests plus the full `src/jepa_noise/tests` suite once on the final SHA
  (deselect only the three known `test_guard` tests); both dry runs.
- **One fresh-context focused review** of the delta from `0cf61ff`. PASS or BLOCK per item, with
  file:line evidence:
  - (i) `raw_map` is correct, including constant columns, and is the only route to Â;
  - (ii) every Â consumer is converted; the source and raw streams are unchanged;
  - (iii) the D1 self-check and its STOP rule;
  - (iv) D2 statistics in every comparison of both modules;
  - (v) D3 folded in and the adapter retired;
  - (vi) the D4 completeness list and the superseded table;
  - (vii) the R-2 evaluation path and the `not_run` label;
  - (viii) the gates, the import closure, the 0.35 h guard, S1 and entries.
- **Acceptance:** re-issue `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json` with the new
  `impl_commit` and two new fields, `addendum_path` and `addendum_sha256`. Both parts' gates also
  require the committed addendum's sha256 to match.
- No outcome-generating step runs before this record exists.

---

## 8. Execution order

1. **CPU:** D1 self-check → X2-c, X2-d latent and X2-f reruns → X3-b readings rerun (D2), CPU parts.
2. **GPU, at most two jobs at once:** EST(latent) sets; R-2 evaluation; `init9000` gain-blind tokens only if needed.
3. **CPU:** X2-e (latent) → Part B R-1 rerun → both reports, with the D4 summary and figures.

## 9. STOP conditions (the only ones)

- S1;
- S3 (any change to a reading rule or threshold);
- S4;
- S9 (the D1 identity on real data, or a reproduction, still fails after one fix attempt; stops the
  dependent items only);
- an import-closure failure (quarantine; the fix needs a re-review and a re-issued acceptance);
- an entries-preflight failure;
- the 0.35 h cap.

A STOP in one item never stops the independent ones: D2–D5 do not depend on D1, except R-1.

## 10. Deliverables

- Updated `results/jepa/20261106_repair/` and `results/jepa/20261107_shortcut/`:
  `READINGS_SUMMARY.md`, `SUMMARY.json`, `OUTCOME.json`, figures, and a new
  `ADDENDUM_2026-11-07.json` with the self-check, the reruns and the superseded table.
- The 7 Nov files are updated in place, and every replaced value appears in the superseded table.
- A `STATE.md` section, a ledger line (GPU and CPU), and the `DECISIONS.md` lines (both §0 quotes,
  the adapter retirement).

## 11. Out of scope

- Training of any kind; data generation;
- any new variant, model or factor;
- changes to readings or thresholds;
- the in-`<P>` fix of the 6 Nov label-row collector;
- Paper 3 information-destroying drifts (they need a generation budget and a new runbook).
