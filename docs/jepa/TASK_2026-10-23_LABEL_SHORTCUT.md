# TASK — grid pause, label-shortcut diagnostic, fix path, `anchor_only` β (23 Oct 2026)

_To the orchestrator of the JEPA study (branch `JEPA`, at or after `19fb7c5`). From the author.
Do the four steps **in order**, one at a time. Commit this file unchanged as
`docs/jepa/TASK_2026-10-23_LABEL_SHORTCUT.md` before starting._

**Authority.** The author, 23 Oct 2026, 23:17 CEST, approving the plan in this file:
> "我认可你的建议，把1 2 3 4步写成一份发给 orchestrator 的简短任务说明，让它一项一项做"

("I approve your recommendations; write steps 1–4 as a short task brief for the orchestrator and
have it do them one by one.")

Follow-up, 23 Oct 2026, 23:21 CEST, confirming two points: (i) `anchor_only` runs at the anchored
arms' β (option A, step 4), and (ii) these approvals are recorded verbatim:
> "是的，我确认这两点"

("Yes, I confirm these two points.") Quote both verbatim in `DECISIONS.md`. The approval covers
exactly the actions below, including the decision rules, and nothing else.

**Stop and report to the author (one page, lettered options) only at these points:**
- S1: step 2's result is `ambiguous`.
- S2: step 2's result is `not_confirmed` (step 3B).
- S3: a fixed-label run fails scientifically (NaN, divergence, collapse).
- S4: ledger + projection exceeds 160 L40S GPU-h at any re-projection.
- S5: anything would change a registered prediction, criterion, cell or dataset.

Everything else is engineering: one line in `DECISIONS.md`, then continue. All §6.1/§6.2 host
rules, the guard and the no-touch packages are unchanged.

---

## Background (why)

**1. Grid stop.** `run_grid` refused `anchor_only` because
`configs/decisions/a10_dev_step3_beta_decision_2026-10-23.json` has `selected: null`
(`none_qualified`). Under A9 the author's ruling was implemented as a separate grid decision file
(`results/jepa/20261020_wp5_dev_cohort/step3_beta_decision_grid.json`, `anchor_only` at β = 1000,
`beta_not_qualified`). No A10 equivalent was written.
- The refusal is correct behaviour. **Do not change `run_grid` to skip null β.**
- **Do not drop `anchor_only`**: it is a registered A6 arm.

**2. Label shortcut (suspected).** In `lucid_arms.py::_encode_views`, the prediction labels for
`sigreg_jepa` come from `self.jepa.encode(batch.features)`:
- this is the encoder's read-out on the **full, unmasked** event;
- it has shared weights and no stop-gradient.

The label at a target PMT can therefore attend to context PMTs in its window, and the encoder can
make targets predictable by copying context information.

This conflicts with v3 WP4, which v4 §2 keeps in force: "Target encoder sees target-only tokens,
with all other positions masked … target outputs [zero dependence] on raw context". `ijepa_ema`
already conforms (`hidden_mask=~batch.target`), and so does the bridge (`encode(x, "target")`).
Affected arms: `sigreg_jepa`, `sigreg_jepa_anchored`, `sigreg_jepa_anchored_dark` and
`oracle_nominal_twin`, which all run through `_forward_sigreg`.

Supporting evidence, `A10_AUDIT_2026-10-23.md`, photon domain:
- trained `sigreg_jepa` vertex resolution 0.2644 m is **worse than its random-init floor**
  (0.1609 m);
- its module-gain run-mean R² is −0.004, against a floor of 0.219;
- predictor loss 0.013, SIGReg 9.8.

This is consistent with the shortcut but does not prove it. Step 2 tests it.

---

## Step 1 — Pause the grid (no deletions)

1. **grid-0:** it has already stopped. Leave its 15 completed runs as they are.
2. **grid-1:**
   - Let the run in progress finish and write its results.
   - Then stop the job before the next run starts, using the normal `jepa_run.sh`/guard path. No
     SIGKILL.
   - If it reaches the `anchor_only` refusal first, that is also fine: the refusal happens before
     any training.
3. **Record:**
   - one line per job in `INFRA_LOG.md` and `DECISIONS.md`;
   - the completed-run list (run_id, arm, cell, seed, SHA);
   - update `RESOURCE_LEDGER.json`.

**Done when:** no JEPA training job runs on deepthought2, and every completed run is listed.

## Step 2 — Diagnose the label shortcut (budget ≤ 3 L40S GPU-h, deepthought2)

Everything in this step is development, labelled `registered: false`, under
`results/jepa/20261023_label_shortcut_diag/`.

**2a. No-training dependence check** (supporting evidence, no criterion).
- Use the A10 dev `sigreg_jepa` s9000 checkpoint, plus a random-init encoder of the same
  architecture (init seed 9000).
- On 32 dev events, `within` split, compute for the full-input read-out at target PMTs:
  - G_c = ‖∂z_target / ∂x_context‖_F;
  - G_t = ‖∂z_target / ∂x_target‖_F.
- Report the share G_c / (G_c + G_t) for the trained and the random-init encoder.
- Also show that target-only encoding gives G_c = 0 exactly.

**2b. Paired training test** (decides the branch).
- **Runs**, all on the A10 dev cell with its frozen config, 10 000 steps, L40S:
  - `sigreg_jepa` with **target-only labels**, seeds 9000 and 9001. This is the fix of step 3.1,
    applied on a development branch of the code; do not merge it yet.
  - `sigreg_jepa` with the **current labels**, seed 9001. Seed 9000 already exists.
- **Evaluate** each run with the frozen dev step-2 protocol (ridge, photon domain): vertex
  resolution, module-gain run-mean R², final SIGReg, final predictor loss, and effective rank of
  the mean tokens.
- **Decision rule.** Paired by seed, compare the vertex resolution r_fixed against r_old:
  - `confirmed`: r_fixed ≤ 0.90 × r_old in **both** seeds.
  - `not_confirmed`: |r_fixed − r_old| < 0.10 × r_old in **both** seeds.
  - `ambiguous`: anything else. Stop at S1.
  - NaN, divergence, or effective rank < 2 in a fixed-label run is S3.
- **Report** (≤ 1 page): the table, the branch taken, the floors beside every number, and the
  GPU-h used.

## Step 3 — Act on the branch

**3A. `confirmed`: fix, verify, re-select β.** This restores the registered v3 WP4 arm
definition, so it is a defect fix, not a specification change. Record it as such.

1. **Fix.** In `_encode_views`:
   - the labels come from a **target-only pass**,
     `self.jepa.encode(batch.features, hidden_mask=~batch.target)`: same encoder, shared weights,
     no stop-gradient (LeJEPA semantics kept);
   - the SIGReg target term uses these labels;
   - keep the full unmasked pass **only** where it is used now for the anchor
     (`full_readout` → `_module_tokens`) and for evaluation.

   Change nothing else.
2. **Tests.**
   - New reverse-leakage test: the label tokens have **exactly zero** gradient with respect to
     every context-PMT input, for `within`, `across` and the balanced-occupancy mode.
   - The existing leakage, amplitude and determinism tests pass, and the full suite passes.
   - Show by diff that the code paths of `mae_mse`, `mae_chi2`, `ijepa_ema` and `anchor_only`
     are untouched.
3. **Review.** A fresh reviewer checks the diff and the tests: one review, small scope.
4. **Re-run A10 dev step 3** at the fixed SHA:
   - plain `sigreg_jepa` reference, plus β ∈ {0.1, 1, 10, 100, 1000} for `sigreg_jepa_anchored`,
     `sigreg_jepa_anchored_dark` and `anchor_only`;
   - same seed, rule and config as the frozen step 3;
   - budget ≤ 15 L40S GPU-h.
   
   Freeze a new decision file. The old one is kept and labelled `superseded_by_label_fix`.
   **Step 2's ridge decision stands without re-running**, because `mae_mse` alone (0.1049 m <
   3 × 0.1421 m) keeps the rule at ridge. Record that arithmetic.
5. **Grid bookkeeping.**
   - Completed grid runs of the four affected arms: label them `superseded_by_label_fix`. Keep
     all files; the ledger keeps their hours.
   - Completed runs of `mae_mse`, `mae_chi2` and `ijepa_ema` stay valid. Record the SHA mapping
     that shows their path is unchanged.
6. **Re-project** the remaining grid with the measured walls. Continue only if ledger +
   projection ≤ 160 GPU-h; otherwise stop at S4.

**3B. `not_confirmed`.** No code change. Stop at S2 with a one-page gate:
- (A) keep the current implementation, and disclose the deviation from v3 WP4 and the 2a numbers
  in the report;
- (B) apply the 3A fix anyway, for conformance with the registered arm definition, and re-run the
  affected arms. **Recommended: B.**

## Step 4 — `anchor_only` β for the grid, then resume

1. Write `configs/decisions/a10_dev_step3_beta_decision_grid_<date>.json`, the A10 analogue of
   the A9 grid file:
   - built from the **current** frozen step-3 decision: the 3A re-selection if 3A ran, otherwise
     the 23 Oct file;
   - `anchor_only.selected` = the frozen β of `sigreg_jepa_anchored` (today 100);
   - `status`/`label` = `beta_not_qualified`, table preserved;
   - the anchored arms unchanged.

   This applies the author's A9 rule as worded: "Run it in the grid at … the anchored arms' β,
   labelled `beta_not_qualified`. It is a descriptive control." Any prediction clause that needs
   a qualified `anchor_only` β is `not_evaluable`.
2. Point the grid's `--beta-decision` at this file, and keep `run_grid`'s refusal on null β.
3. **Resume** grid-0 and grid-1 through the existing automatic resume:
   - completed valid runs are skipped;
   - superseded runs are re-run under new run IDs;
   - rung (1) and every §3.7.2 rule apply.
4. **Report** (≤ 1 page): what ran in steps 1–4, the branch, the new β table, the re-projection,
   and the expected grid end date against §9 (WP5 by 12 Dec).

No grid result is reported to anyone before the grid completes, as before.
