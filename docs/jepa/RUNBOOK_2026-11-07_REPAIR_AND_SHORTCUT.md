# Combined runbook — Part A: Paper 3 repair pilot (X0–X3); Part B: shortcut-collapse follow-up (R-1, R-2, R-3)

**Status: 7 Nov 2026. Part A: design review passed (v3, Part A Appendix G). Part B: design not
separately reviewed; the joint review of §F5 covers its design items. Implementation not started;
not yet accepted.** Both parts are `registered: false`.

This file is the single runbook for both parts. Commit it unchanged as
`docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` on branch `JEPA`, at or after `ac8e220`,
before any outcome-generating step. It depends on no document outside the repository. The current
state is in `docs/jepa/STATE.md`.

---

## F0. Authority

Quote all three lines verbatim in `DECISIONS.md`.

1. The author, 6 Nov 2026, 12:56 CET (Part A):
   > "选 A，把它写成 Paper 3 新版规划文档，并为这三个实验写 runbook"
2. The author, 7 Nov 2026, 00:12 CET (Part B; the explicit training line for Part B R-2 only):
   > "一二三我都批，写成一个runbook"
3. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

The author asked on 7 Nov 2026 (00:20 CET) for both runbooks to be combined into one, placed in
the local repository, for an agent that works locally and runs experiments over ssh.

**Training:** only Part B §5.3 (one variant, two seeds, 10 000 steps each, one 20-step smoke).
Part A trains nothing; Part A X3-T stays **not authorised**. **No data generation** anywhere.

## F1. Precedence and conventions

- Where this front matter and a part disagree, the front matter wins.
- Inside a part, every § reference points to that part's own sections ("§8.4" in Part B is Part
  B's §8.4).
- Each part keeps its own extension name, module, CLI, `<ext>` root, results directory, readings,
  caps and STOP list:

  | | Part A | Part B |
  |---|---|---|
  | extension | `paper3_repair_pilot_2026-11-06` | `shortcut_followup_2026-11-07` |
  | code | `src/jepa_noise/ext_repair.py`, `scripts/jepa_ext_repair.py`, `src/jepa_noise/tests/test_ext_repair.py` | `src/jepa_noise/ext_shortcut.py`, `scripts/jepa_ext_shortcut.py`, `src/jepa_noise/tests/test_ext_shortcut.py` |
  | `<ext>` | `/ceph/dwong/jepa/returns/ext_repair_20261106/` | `/ceph/dwong/jepa/returns/ext_shortcut_20261107/` |
  | results | `results/jepa/20261106_repair/` | `results/jepa/20261107_shortcut/` |
  | GPU cap (L40S h) | 4.0 | 1.5 |
  | CPU cap (wall h) | 6 | 2 |

- A STOP in one part never stops the other. The only dependency between the parts is Part B R-1,
  which reads Part A's X1/X2 outputs and is `not_run` until they exist.

## F2. Where the agent works

**Local (the author's Mac, `~/Documents/VERITAS`):** editing, git, commits and pushes, and focused
unit tests if the local Python environment supports them. Nothing local reads `/ceph`.
- The main checkout of that clone is on branch `pmt/r7-response-v3`, with uncommitted changes
  (deleted `reference/papers/*` files and `.pytest-tmp-*` directories). These are the author's.
  **Never switch, stash, reset, clean or commit in that checkout.**
- The local branch `JEPA` is checked out in the worktree `/private/tmp/veritas-jepa`, 133 commits
  behind `origin/JEPA` (`ac8e220`). Work in a `JEPA` worktree at `origin/JEPA`:
  - fast-forward that worktree if it is clean;
  - otherwise leave it alone, create a new worktree on a new local branch from `origin/JEPA`, and
    push that branch to `origin JEPA` (fast-forward only).
- This file was placed untracked at `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` in the
  main checkout. Copy it byte for byte into the `JEPA` worktree, check that the sha256 is unchanged,
  and commit it there unchanged.
- Pushing from this Mac satisfies both parts' rule "push to `JEPA` from the author's Mac only".

**Remote, over ssh:** every step that reads data or uses a GPU.
- Deepthought (CPU): the full test suite (once per SHA), dry runs, Part A X0-a, X3-a, fits, LEACE,
  bootstrap and reports; Part B manipulation check, R-1 and report.
- deepthought2 (L40S): every encoder forward pass and the Part B training, always through
  `scripts/jepa_run.sh` (no guard bypass).
- Host clones pull the pushed SHA before any step. Detach long jobs with `setsid`. Monitor resource
  use; never saturate the shared servers.
- If ssh to a host fails: retry, then use the other host for CPU steps (as both parts allow). Only
  S4 (no GitHub route after three documented attempts) is a STOP.

## F3. Combined budgets

- **GPU:** Part A ≤ 4.0 and Part B ≤ 1.5 L40S h, 5.5 h in total. Neither part may borrow from the
  other's cap.
- **S1 launch rule (replaces both parts' S1 lines):** L40S ledger + remaining caps of both parts ≤
  160. At `ac8e220`: 129.7461 + 5.5 = 135.25.
- **Concurrency:** at most two of the user's L40S jobs at once across both parts, each ≤ 45 % of GPU
  memory. Deepthought must not be fully loaded.
- **Entries:** ≥ 30 000 free after the projected writes of both parts together.

## F4. Order

1. Set up the `JEPA` worktree (§F2) and commit this runbook unchanged.
2. Implement both parts (one implementation commit or several; the acceptance record names the
   final `impl_commit`).
3. Run the focused tests of both parts, the full `src/jepa_noise/tests` suite once on the final SHA
   (deselecting only the three known `test_guard` launcher tests), and both dry runs.
4. One independent review (§F5), then one acceptance record.
5. Execute. Suggested schedule, at most two L40S jobs at a time:
   - CPU: Part A X0-a and Part B §5.2 manipulation check; Part A X3-a whenever free;
   - GPU: Part A X0-b and Part B reproduction; then Part B smoke and the two R-2 runs on one GPU,
     alongside Part A X1 → X2 on the other; then Part A X3-b; then Part B evaluation;
   - reports: Part A report; Part B R-1 once Part A X1/X2 exist; Part B report.

## F5. Review, acceptance and gates

- **One fresh-context reviewer** reads this file and the full diff. The blocking checklist is the
  union of Part A §9.4 (i)–(viii) and Part B §9.4 (i)–(vi), PASS or BLOCK per item, with
  file:line evidence, plus:
  - (F-i) the two parts share no mutable state, and neither writes into the other's `<ext>` or
    results directory;
  - (F-ii) the combined budgets and S1 rule of §F3 are enforced in both CLIs.
- **One acceptance record:** `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json` =
  `{runbook_sha256, impl_commit, review_path, review_verdict, lead, date}`, with
  `review_verdict == "PASS"`.
- Both parts' gates (their §9.2) read the sha256 of this committed file and this acceptance record.
  Each part checks its own `<P>` against `impl_commit`.
- A closure STOP in either part quarantines that mode's outputs. The fix needs a re-review and a new
  acceptance record, and it does not stop the other part unless the fix touches the other part's
  `<P>`.

## F6. Deliverables

Each part writes its own deliverables (Part A §9.6, Part B §9.6). `STATE.md` gets one section per
part, the ledger one line per part, and `DECISIONS.md` the three §F0 quotes once.

---

# Part A — Paper 3 repair pilot: X1 calibration control, X2 equivariance and identification, X3 price of invariance

Design review at `ac8e220`: v1 BLOCK, v2 BLOCK, v3 PASS; all findings addressed (Appendix G). Part A trains nothing, generates no data, and changes no registered prediction. Authority, the combined budgets, the review and the acceptance record are in the front matter (§F0–§F5).

---

## 0. Authority

Quote both lines verbatim in `DECISIONS.md`.

1. The author, 6 Nov 2026, 12:56 CET, in the design conversation:
   > "选 A，把它写成 Paper 3 新版规划文档，并为这三个实验写 runbook"

   This authorises the design of X1–X3. X1–X3 train nothing, so no training line is needed and
   none is implied. Item X3-T (§7.3) is listed for completeness and is **not authorised**.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

### 0.1 Fix it yourself, do not stop to ask

- Fix small problems yourself (bug, path, environment, guard HOLD/SHED, launcher timeout), write
  one `DECISIONS.md` line ("author standing approval, 30 Oct 2026"), and continue.
- Stop only for the conditions in §8.4.

---

## 1. Questions

Background (`docs/jepa/STATE.md`; `results/jepa/20261103_followup/readings.json`, item A-2):

- Under strong module-gain drift (law F1, s = 0.05 → 0.30, paired events), the frozen readout of
  an MAE trained at s = 0.05 loses 8.6–17.2 mm of mean vertex error. Refitting the readout at
  s = 0.30 recovers part of it. An MAE retrained at s = 0.30 (T-1) is not consistently better than
  the refit (G-2).
- In all eight A-2 models, the best s = 0.30 vertex error stays 5.1–7.1 mm above the same model's
  s = 0.05 vertex error (the "5–7 mm residual").
- The generator multiplies the charge by the injected gains (`lucid_bombs.py:2105`) before the
  0.25 PE discriminator (`:2106-2110`) and before the NHits trigger (`:2125`).
- In this configuration `threshold_pe = walk_calibration_pe = 0.25`
  (`src/jepa_noise/configs/lucid_bombs_v2_calibrated.json`), so the walk residual is exactly zero
  (`lucid_bombs.py:2052-2053, 2108-2111`) and digit times do not depend on the gain.
- A gain change therefore moves the charge, which a calibration can invert, and also the **hit set
  and the trigger** (including prompt-window membership), which no charge calibration can invert.

| ID | Question | Why it is open | Item |
|---|---|---|---|
| **Q-X1** | How much of the drift cost does a charge calibration remove, and how much is the non-invertible part (hit set and trigger)? What is the 5–7 mm residual made of? How does calibration compare with refitting and retraining? | A-2 compared only a frozen and a refit readout; no calibration condition exists | X1 |
| **Q-X2** | Does the drift act on each frozen encoder's embedding as a fixed transformation (equivariance)? Can it be undone in latent space without labels? Can the module gains be identified, and from which stream? | Never measured | X2 |
| **Q-X3** | What does invariance to module gains cost in vertex and energy information — as a Fisher bound on the WCTE geometry, and in practice for invariant inputs and gain-erased readouts? | Never measured | X3 |

---

## 2. Snapshot, data, checkpoints and side effects

At `ac8e220`: L40S ledger `E2_lucid_l40s_deepthought2` 129.7461/170 (`RESOURCE_LEDGER.json`);
generation CPU 18.425639/48; free entries 37 986 at the last committed preflight
(`results/jepa/20261103_followup/plan.json`).

### 2.1 Pinned data

Data root `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/` (`ext_followup.DATA_ROOT_DEFAULT`).
Hashes are the sha256 of each set's `manifest.json`. X0 re-verifies all four; a missing or
mismatched key is a STOP for every item that uses it.

| Key | sha256 |
|---|---|
| `dev_probe_train__F1__0.05__s20261017` | `093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3` |
| `dev_probe_test__F1__0.05__s20261017` | `c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a` |
| `dev_probe_train__F1__0.3__s20261017__n10000x50` | `6df84887b3469723da3ddb0646c292ab655b263d6063f5fd713432a53514bb46` |
| `dev_probe_test__F1__0.3__s20261017__n10000x50` | `c68d83d1ef8cb202f3937f7cd48e3f74bd94f6d44ae203591053fde0605d1be2` |

- Open them only through `ext_followup.open_probe_sets(config, gen, geometry, data_root,
  setting=…)` (`ext_followup.py:532-556`). It checks the pinned hash read-only before
  `lucid_run.ensure_dataset`, and opens the s = 0.30 sets through their `registered=False` keys.
  `ensure_dataset` also recomputes every run's `state_sha256` (`lucid_run.py:339-345`).
- **No data generation of any kind is allowed.**

Per-run files read (all read-only), in `<set>/runs/<run_dir>/`:

| File | Content used |
|---|---|
| `hits.npz` | measured hits: `charge_pe` (gain-scaled charge, PE), `time_ns` (digit − trigger), `pmt_id`, `offsets`, `event_ids` |
| `twin_hits.npz` | the nominal twin: the same draws read out with every injected gain set to 1, through the same threshold and trigger (`lucid_bombs.py:596-600, 2237-2238, 2258-2260`) |
| `events.npz` | `event_ids`, `vertex_m`, `yield_photons`, `expected_pe` (**one total per event**: the PMT sum of the nominal expected PE, fixed response included, no injected gain; `lucid_bombs.py:2219, 2228-2234, 2267`), `domain`, particle truth |
| `manifest.json` | the run's `state_sha256` and `twin_present` |

### 2.2 Detector state and expected charge per run

- **State.** For each run, build the `RunSpec` exactly as `lucid_run._run_truth` does
  (`lucid_run.py:346-355`) and call `.state(geometry)`. The state gives `gains` (per PMT: module ×
  per-PMT injected gain), `module_gain` (105) and `pmt_gain` (1 995). X0 recomputes
  `RunSpec.state_sha256(geometry)` for every run of the four sets and compares it with the run's
  `manifest.json`. This repeats `ensure_dataset`'s own check and is kept deliberately. A mismatch is
  a STOP for X1 and X2.
- **Notation.** For run r and PMT p, G⁰_p and G¹_p are `state.gains[p]` at s = 0.05 and s = 0.30;
  m⁰ and m¹ are the module gains; mod(p) = `geometry.module_of_pmt[p]`.
- **Why the module ratio works.** The run ids do not depend on the setting
  (`lucid_bombs.py:670-673`), and the per-PMT factor is the same keyed draw in both settings
  (`:491-503`). So G⁰_p / G¹_p = m⁰_mod(p) / m¹_mod(p).
- **Per-PMT expected charge (derived truth; used only by the X2-d source stream).** For a photon
  event e of run r:

  μ_p(e) = `expected_charge(geometry, vertex_m[e], yield_photons[e], qe, state.attenuation_length_m, cfg)`
  × `fixed_charge_gain(geometry)`,

  with `qe = sample_qe(geometry, keyed_rng(dataset_seed, lb.NS_DRAWS, run_id, int(event_id), "wavelength"), cfg)`,
  where `run_id` is the RunSpec `run_id` (not `run_dir`) and `dataset_seed` is
  `config["data"]["dataset_seed"]` (20261017).
  This is the keyed stream `draw_event` uses for the QE (`lucid_bombs.py:1985-1990`) and the
  expectation the generator sums into `expected_pe` (`:2228-2234`). On every event used,
  Σ_p μ_p(e) must equal the stored `expected_pe[e]` to float32 relative 1e-6; otherwise the source
  stream is `unavailable`. This recomputation uses no Poisson draw, no `draw_event` and no readout,
  so it is not data generation.

### 2.3 Checkpoints

- Every checkpoint is accepted only on a sha256 match, re-hashed on the host by X0.
- Rows come from `ext_followup.build_inventory(hash_files=True)` (read-only; it re-hashes all 54
  labels — restricting the re-hash to the labels below is equally acceptable) and
  `ext_followup.t1_inventory_rows()`, which does not re-hash; use labels `t1_*`, never a
  `t1smoke_*` row.
- Arms are built with `ext_followup.build_item_arm` inside `ext_followup.item_eval_context(row)`.

**Core (13):**

| Label | Arm | Path (under `/ceph/dwong/jepa/returns/`) | sha256 |
|---|---|---|---|
| `mae_within_trigger9000` | `mae_mse` | `ext_mae_dejitter_20261031/train/runs/ext_dev_s005_within_trigger__mae_mse__s9000/final.pt` | `a2b5384f68266caf08a2563e3dc5d7d2a037ab90da0bfa5e42f659f64bef1c49` |
| `mae_within_trigger9001` | `mae_mse` | `…/ext_dev_s005_within_trigger__mae_mse__s9001/final.pt` | `210845c4d40f94b1f518de2dbf10c4ab8a3b0d7237a4c97a1f5bf7eba86539f1` |
| `mae_across_trigger9000` | `mae_mse` | `…/ext_dev_s005_across__mae_mse__s9000/final.pt` | `30eae1fec959eb22fec5417d4d089dcfacd6d1042b5ebf17a71d85d50aa85a64` |
| `mae_across_trigger9001` | `mae_mse` | `…/ext_dev_s005_across__mae_mse__s9001/final.pt` | `77d36d84b033a9b973afdfefab9b5f0505abf30e5887d8df7c130f52b090986f` |
| `t1_s030_within_trigger9000` | `mae_mse` | `ext_followup_20261103/train/runs/ext_dev_s030_within_trigger__mae_mse__s9000/final.pt` | `683f3ef9474ed0be3febcdc005bd4bbb0ed8ecd793b6df580b655e7f25462dbb` |
| `t1_s030_within_trigger9001` | `mae_mse` | `…/ext_dev_s030_within_trigger__mae_mse__s9001/final.pt` | `415b38031182fe4d5dd5f444ff12febe84cdd6a2e9ef9ce3774ab5a0e7bd8611` |
| `t1_s030_across_trigger9000` | `mae_mse` | `…/ext_dev_s030_across_trigger__mae_mse__s9000/final.pt` | `accc9016b9b2270777033f76fc15636ecaca17da6c4531e0853816f78c30a475` |
| `t1_s030_across_trigger9001` | `mae_mse` | `…/ext_dev_s030_across_trigger__mae_mse__s9001/final.pt` | `a8752fde554f7715bc96a67c3f5b717131e313155ae84a6497216710b6d193f8` |
| `init9000` | `sigreg_jepa` | seed-9000 rebuild (inventory row with `init: true`) | — |
| `fix9000` | `sigreg_jepa` | `ext_readout_20261101/ckpt/fix9000.pt` | `2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714` |
| `fix9001` | `sigreg_jepa` | `ext_readout_20261101/ckpt/fix9001.pt` | `95cb5e3f4f158a719098a4b3dc573eebb8c71645c22290f6483223ccc04c484e` |
| `frozen_target9000` | `sigreg_jepa` | `ext_h12_20261101/train/frozen_target/runs/dev_s005_within__sigreg_jepa__s9000/final.pt` | `8c6341c75d1463d0dcf0582a980c4b6cc700c9d755a2904305e7bcb36be8138f` |
| `frozen_target9001` | `sigreg_jepa` | `…/dev_s005_within__sigreg_jepa__s9001/final.pt` | `54122fa57a0018828754c61f6675ae5a240fc785abfcaab62f394397f375ffb5` |

**Optional (6):** `ijepa9000/9001`, `static_target9000/9001`, `mae_within_median9000/9001` (the
median wrapper comes from `item_eval_context`). They run only if the X0 projection with them fits
the cap. Paths and hashes are the `results/jepa/20261103_followup/inventory.json` rows.

The core set spans the response types the plan needs: MAE keeps gain information (A-2), the
fixed-label JEPA is collapsed and event-invariant, the frozen-target JEPA is event-dependent, and
the random-init encoder is an untrained map.

### 2.4 Reusable code (read-only imports; never edit)

- **`ext_followup`, called:** `open_probe_sets`, `require_pinned`, `manifest_path`,
  `build_inventory`, `t1_inventory_rows`, `build_item_arm`, `item_eval_context`,
  `stream_event_rows`, `stream_particle_rows`, `load_particle_truth`, `raw_event_features`
  (particle events only), `pair_events`, `paired_run_bootstrap`, `_probe_dataset`,
  `_vertex_errors`.
- **`ext_followup`, constants:** `A2_BOOTSTRAP_RESAMPLES`, `A2_BOOTSTRAP_SEED`, `A3_MIN_CONTAINED`,
  `A1_S005`, `A1_S030`, `FORCED_IMPORT_EXCEPTIONS`.
- **`ext_followup._energy_task`:** its conventions only (electrons, contained, ≥ 500 test events).
  It fits and scores in one call and returns R² only, so the frozen energy probe is written with
  `fit_probe`/`probe_predictions` (§3.2).
- **Never call an `ext_followup` function that reaches its gate:** `readout_a1`, `readout_a1b`,
  `readout_a2`, `readout_a3`, `readout_a4`, `reproduction`, `assemble_report`, `build_plan`,
  `t1_prepare_launch` and `train_t1`, nor any `mode_*` in `scripts/jepa_ext_followup.py`. At
  `ac8e220` that gate would probably *grant*, so this rule is enforced by test (§9.3), not by the
  gate.
- **`lucid_arms`:** `fit_probe` (per-target five-fold run-grouped ridge over `probes.ALPHAS` =
  (1e-6, 1e-4, 1e-2, 1, 100), `probes.py:9`), `probe_predictions`, `ProbeDataset`.
- **`lucid_run`:** `_run_truth`, `_vertex_error`.
- **`lucid_bombs`:** `RunSpec`, `read_sparse_hits`, `SparseHitStore`, `expected_charge`,
  `fixed_charge_gain`, `sample_qe`, `fiducial_vertex`, `nominal_state`, `keyed_rng`, `NS_DRAWS`,
  `SIGMA_SPE`, `V_DARK_READOUT`, `F3_SPREAD`, `load_generator_config`, `load_wcte_geometry`.
- **`point_encoder`:** `tokenise`, `FeatureScaling`, `LOG_CHARGE_INDEX`.
- **Setup** (config, generator, arms, geometry, device): exactly as
  `scripts/jepa_ext_followup.py` `load_setup` does.

### 2.5 Patches

`ext_repair.hit_transform_context(condition)` is a context manager. While it is active it
replaces **both** `point_encoder.read_sparse_hits` and `lucid_bombs.read_sparse_hits` with a
wrapper that, for a path `<set>/runs/<run_dir>/hits.npz` of the set the condition names:

1. reads that store, or for the HW condition the `twin_hits.npz` of the same run directory;
2. returns a `SparseHitStore` whose `charge_pe` is multiplied hit by hit by the condition's factor
   f_r(pmt_id) (§3.1), with `event_ids`, `offsets`, `pmt_id` and `time_ns` untouched.

Further rules:
- The wrapper **fails closed**: any other path raises.
- Both attributes are restored in a `finally`. The precedent is
  `ext_dejitter.install_median_time_wrapper`.
- Every evaluation-path hit read goes through one of the two patched names. The third binding,
  `lucid_arms.read_sparse_hits`, only reads the dark bank, which these arms never reach.
- **Other wrappers allowed:**
  - those `item_eval_context` already installs (the median-time `point_encoder.tokenise` wrapper;
    `ext_arch.build_arm_patch`);
  - the X3 invariant-token wrapper on `point_encoder.tokenise`, for `init9000` X3 passes only
    (§7.2), built the same way.
- Nothing else may be patched.

### 2.6 Forbidden call paths (each covered by a test)

- Any path reaching `lucid_run.ensure_dataset` other than through
  `ext_followup.open_probe_sets` (hash check first).
- `lucid_run.ensure_dark_bank`, `lucid_run.run_one`, or any optimiser step.
- `lucid_bombs.draw_event`, `generate_run`, `nominal_twin`, `write_generated_run`, or any other
  hit-level generation or readout. Analytic expectations are allowed: the derived-truth μ_p of §2.2,
  and X3-a's synthetic expected-charge inputs (`fiducial_vertex`, `sample_qe` and `expected_charge`
  on the `paper3/x3/fisher` stream).
- `scripts/jepa_ext_dejitter.py report`, and every `ext_followup` gate caller listed in §2.4.
- **Write scope.** The extension's Python code writes only to `<ext>` (§8.3),
  `results/jepa/20261106_repair/` and `TMPDIR`. The launcher's status files, the ledger, the
  `DECISIONS.md`/`STATE.md` lines and the new source and test files follow the existing
  repository rules.
- **Oracle information.** Probe-test true gains and twin hits never enter a deployable repair, an
  identification estimator, or any fit or hyperparameter choice. Probe-train simulation truth may
  calibrate X2-b, the latent stream (Â and its λ cross-validation) and the LEACE concepts.
  - Twin hits enter only the HW condition (including its refit probes) and the E1 eraser (§7.2).
  - Probe-test true gains enter only the declared oracle conditions (SWtw, SWunit, SWmod, SWunitN,
    NOISY, S), X2-c, and the scoring of X2-b, X2-d and the LEACE sanity check.

These rules govern the extension's CLI modes. Unit tests may generate tiny synthetic runs in a
pytest temporary directory with the existing fixtures (§9.3).

---

## 3. Shared definitions

### 3.1 Input conditions

Each condition is applied to probe-set runs through §2.5. The factor f_r(p) multiplies the measured
charge of every hit of PMT p in run r.

| Condition | Source set | File | Factor f_r(p) | Meaning |
|---|---|---|---|---|
| N | s = 0.05 | `hits` | 1 | nominal reference |
| D | s = 0.30 | `hits` | 1 | strong drift, no repair |
| S | s = 0.05 | `hits` | G¹_p / G⁰_p | the charge-scale part of the drift alone (N's hit set and trigger) |
| SWtw | s = 0.30 | `hits` | G⁰_p / G¹_p | D calibrated to its s = 0.05 twin's gains; only the non-invertible part remains |
| SWunit | s = 0.30 | `hits` | 1 / G¹_p | software calibration to unit gain (module and per-PMT) |
| SWmod | s = 0.30 | `hits` | 1 / m¹_mod(p) | module-only calibration (the ceiling of the EST conditions) |
| SWunitN | s = 0.05 | `hits` | 1 / G⁰_p | control: unit-gain calibration of the nominal data |
| HW | s = 0.30 | `twin_hits` | 1 | hardware restoration: unit gains at digitisation (oracle) |
| NOISY(σ) | s = 0.30 | `hits` | 1 / (G¹_p · exp(σ ξ_r,mod(p))) | full per-PMT calibration with a module-level error σ ∈ {0.01, 0.02, 0.05, 0.10}; one ξ ~ N(0,1) per run and module, shared by all σ (Appendix A) |
| EST(k) | s = 0.30 | `hits` | 1 / exp(δ̂ᵏ_r,mod(p)) | module calibration from stream k ∈ {source, raw, latent} (§6, X2-d); modules without an estimate keep factor 1 |

- Charges are transformed before tokenisation. The tokeniser's `log(max(q, 1e-6 PE))`
  (`point_encoder.py:163-164`) then applies as usual, so for every kept hit the log-charge feature
  moves by log f_r(p), to float32 rounding.
- **Caveat.** S and SWtw can produce charges below the 0.25 PE firing point, which never occur in
  measured data. This is inherent to the decomposition and is stated next to X2-a/b. The X2-b map
  Â is fit on S − N and applied to D in X2-c.

### 3.2 Embeddings, probes and metrics

- **Photon embeddings:** `ext_followup.stream_event_rows` (photon events; event mean of the frozen
  encoder's tokens; `split` and `mask_seed` from the row and the config), exactly as A-2.
- **Particle embeddings:** `ext_followup.stream_particle_rows` (muon and electron events).
- **Frozen probe P_N:** `lucid_arms.fit_probe` on the N probe-train embeddings, applied unchanged
  to every condition's probe-test with `probe_predictions`. A **refit probe** is fit on the
  condition's own probe-train, only where listed.
- **Metrics (probe-test), all computed from `probe_predictions` (R² as 1 − SSE/SST from the
  predictions):**
  - vertex: mean 3-D error (`lucid_run._vertex_error`) and median, in m;
  - log-yield: resolution = RMS(ŷ − ln Y) and R², photons;
  - electron energy: relative resolution = RMS((Ê − E)/E) and R², over contained electrons with
    `deposited_MeV`, with a probe fit on N particle-train contained electrons; the
    `A3_MIN_CONTAINED` = 500 test rule applies.
- **Uncertainty:** `ext_followup.paired_run_bootstrap` (1 000 run-cluster resamples, seed
  20261103) on paired per-event errors, with pairs from `ext_followup.pair_events` on
  `(run_id, event_id)`. Resolutions use the same run-cluster resampling of the RMS; a difference
  keeps the pairs together. Each seed is reported separately; there is no seed-population claim.
- **Drift cost:** Δ_D = v_D − v_N (vertex mean error, frozen P_N).
- **Recovered fraction of a repair c:** ρ_c = (v_D − v_c) / (v_D − v_N). It is computed only for an
  eligible model and seed: Δ_D ≥ 3 mm with a bootstrap CI that excludes 0. Otherwise it is reported
  as `no drift cost to recover`.
- **Caching:** every embedding set is written once to `<ext>` as `.npz` with its sha256 and reused
  by later items. Nothing is recomputed for a second item.

### 3.3 Classification rule for seeded readings

- A class reading is made per seed.
- Both seeds in the same class: that class.
- Different classes: `seeds disagree`, with both values reported.
- One seed eligible and the other not: both reported per seed, `not classified`.
- Single-seed models (`init9000`): `single seed (not classified)`, with the value reported. This is
  intended, including for its X3-b readings.
- Unseeded comparisons (the raw-feature family in X3-b): classified directly from their single
  value.
- Flags inside a reading (for example X3-1's "gain-specific") are evaluated per seed and combined by
  this rule.
- Ineligible models (§3.2) are reported as such.

---

## 4. X0 — preflight and reproduction gate

**X0-a. Preflight (CPU, read-only).** Write `plan.json` with:

- the repository SHA and tree status, and this runbook's sha256;
- the four data hashes;
- per-run checks on all runs of the four sets:
  - `state_sha256` recomputed (§2.2);
  - `twin_present` true, and `twin_hits.npz` event ids equal to `hits.npz` event ids;
  - `events.npz` has `expected_pe` and `domain`;
- **content checks of the decomposition:**
  - the `twin_hits.npz` arrays of the same run are equal in the s = 0.05 and s = 0.30 sets;
  - SWtw charges equal N charges on the hits common to both, to float32 relative 2e-7.

  A content mismatch is a STOP for the readings that rely on shared draws or compare across the two
  sets (X1-b, X1-g, X1-d's v_HW − v_N, X2-a, X2-b, X2-c); the other items continue. File hashes are
  not used for this check;
- the μ_p reconstruction check (§2.2) on every photon event of the D probe-test runs;
- the checkpoint inventory (§2.3) with sha matches;
- hardware and software, the GPU projection (§8.1), CPU hours and free entries.

If `twin_hits.npz` is missing in any run of a set, the HW condition is `unavailable` for that set.
Nothing is generated.

**X0-b. Reproduction gate (GPU, one job, one process, one GPU).** With the transform context
inactive, reproduce within 1e-3 m:

| Model | N (refit = frozen) | D refit | D frozen |
|---|---|---|---|
| `mae_within_trigger9000` | 0.104916 | 0.110177 | 0.114908 |
| `mae_within_trigger9001` | 0.116542 | 0.123645 | 0.133124 |
| `mae_across_trigger9000` | 0.120958 | 0.127027 | 0.129525 |
| `mae_across_trigger9001` | 0.119626 | 0.126149 | 0.136818 |
| `t1_s030_within_trigger9000` | 0.105388 | 0.110750 | 0.116299 |
| `t1_s030_within_trigger9001` | 0.100512 | 0.105751 | 0.110363 |
| `t1_s030_across_trigger9000` | 0.125605 | 0.130727 | 0.134039 |
| `t1_s030_across_trigger9001` | 0.119031 | 0.124470 | 0.130892 |
| `init9000` | 0.160072 | 0.170301 | — |

Sources: `results/jepa/20261103_followup/readings.json` (A-2), `ext_followup.A1_S005` and
`A1_S030`.

**Identity check.** On all N and D sets of `mae_within_trigger9000`, with the identity condition
active (factor 1 everywhere through §2.5):
- the tokenised features must be bit-identical to the untransformed ones;
- the embeddings must agree to max |Δ| ≤ 1e-6, computed in the same process on the same GPU.

A failure after one fix attempt is S9 for X1 and X2. X3-a does not depend on X0-b. The N and D
embeddings of X0-b are cached and reused by X1, so X1-a is consistent with the A-2 differences to
the X0-b tolerance (each term within 1e-3 m).

---

## 5. X1 — what does calibration recover?

**Models:** the 13 core models (§2.3), plus the optional ones if the cap allows.

**Photon sets per model:** N (train, test), D (train, test), SWtw (train, test), SWunit (train,
test), SWmod (test), SWunitN (test), HW (train, test), NOISY(σ) (test, × 4).

**Particle sets** for 6 models (`mae_within_trigger9000/9001`, `mae_across_trigger9000/9001`,
`init9000`, `fix9000`): N (train, test), D (test), SWunit (test), HW (test). For the four MAE
models also D (train), which the X3-b particle eraser needs (§7.2).

**Readouts per model:**
- vertex mean error and median for every condition, with P_N;
- refit vertex for D, SWtw, SWunit and HW;
- log-yield resolution, frozen for N, D, SWtw, SWunit and HW, refit for D, SWunit and HW;
- electron-energy relative resolution, frozen from N, for N, D, SWunit and HW.

**Readings (fixed now; per model and seed; two-sided; classes by §3.3):**

- **X1-a, drift cost.** Δ_D with its CI, consistent with the A-2 frozen difference to the X0-b
  tolerance.
- **X1-b, invertible share of the frozen drift cost.** ρ_SWtw, for eligible models:
  - *charge-scale dominated*: ρ ≥ 0.7;
  - *non-invertible dominated*: ρ ≤ 0.3;
  - otherwise *mixed*.
- **X1-g, what the post-refit residual is made of (primary, referenced to N).** The 5–7 mm
  residual R = v_D(refit) − v_N splits into:
  - R_cs = v_D(refit) − v_SWtw(refit): the charge-scale part after a refit;
  - R_ni = v_SWtw(refit) − v_N: the hit set and trigger at the s = 0.30 gains, against the
    s = 0.05 data.

  The events are paired on `(run_id, event_id)` across the two sets. Eligible if R ≥ 3 mm with a
  CI excluding 0. Report share_cs = R_cs / R:
  - *residual charge-scale dominated*: share_cs ≥ 0.7;
  - *residual non-invertible dominated*: share_cs ≤ 0.3;
  - otherwise *mixed*.

  **Secondary (descriptive), referenced to unit gain:** v_D(refit) − v_HW(refit) =
  [v_D − v_SWunit](refit) + [v_SWunit − v_HW](refit), reported beside v_N − v_HW(refit). The unit-gain
  reference also lacks the nominal 5 % / 10 % gain spreads, so this split mixes their cost with the
  drift's.
- **X1-c, calibration against refit.** The paired difference v_SWunit (frozen P_N) − v_D (refit):
  - *calibration better*: the CI's upper bound is < 0;
  - *refit better*: the CI's lower bound is > 0;
  - otherwise *not resolved*.

  The same reading for v_SWunit (refit) − v_D (refit), i.e. calibration followed by a refit.
- **X1-d, ceilings (descriptive).** v_HW − v_N and v_SWunitN − v_N (what the nominal 5 % module and
  10 % per-PMT gain spreads cost), and v_SWunit − v_HW (the non-invertible part against unit gain,
  frozen probe).
- **X1-e, precision curve (descriptive).** ρ_NOISY(σ) for the four σ. σ½ is defined only if
  ρ_SWunit > 0.1:
  - σ½ is the first grid σ where ρ_NOISY(σ) ≤ ½ ρ_SWunit, interpolated in log σ from the previous
    grid point;
  - "≤ 0.01" if it already holds at 0.01; "> 0.10" if it never holds;
  - otherwise `undefined`.
- **X1-f, energy (descriptive).** Log-yield and electron-energy resolutions for N, D, SWunit and HW,
  with run-cluster CIs.

**Interpretation rule (fixed now):**
- **X1-b** reads the frozen drift cost. **X1-g (primary)** reads the 5–7 mm residual that survives
  a refit.
- **Residual charge-scale dominated:** the residual is an uncalibrated charge scale. A readout
  refit cannot remove it, because one probe cannot know each run's gains; a calibration can. It is
  then the least repair for this drift, and X1-c says whether calibration alone beats a labelled
  refit.
- **Residual non-invertible dominated:** the residual comes from the hit set and the trigger (gain
  moves threshold crossings). No charge calibration or readout refit removes it. The next factors
  to study are the threshold (hit-loss) family and the time reference.
- The readings compare repairs within a model. None of them ranks encoders.

---

## 6. X2 — equivariance, latent repair and identification

**Models:** the 13 core models. **Additional photon sets per model:** S (train, test), and EST(k)
(test, × 3) after X2-d.

**X2-a, decomposition (descriptive; caveat §3.1).**
- Embedding changes:
  - total, Δz_tot = z_D − z_N;
  - charge-scale part, Δz_cs = z_S − z_N;
  - non-invertible part, Δz_ni = z_D − z_S.
- Report:
  - the variance shares E‖Δz_cs‖² / E‖Δz_tot‖² and E‖Δz_ni‖² / E‖Δz_tot‖²;
  - the frozen-probe costs v_S − v_N and v_D − v_S;
  - the additivity check: v_D − v_N against (v_S − v_N) + (v_SWtw − v_N).

**X2-b, equivariance.** Fit on probe-train pairs and score on probe-test pairs; the two sets hold
disjoint runs.
- Models:
  - *global*: Δz_cs ≈ A δ_r, with δ_r = ln m¹_r − ln m⁰_r (105 values per run);
  - *occupancy*: Δz_cs ≈ A u_e, with u_e,m = δ_r,m · n_e,m, where n_e,m is the number of hit PMTs of
    module m in the event (N hits). `stream_event_rows` returns only the total hit count, so n_e,m
    is computed on CPU from the condition's hit store: a `bincount` of `geometry.module_of_pmt` over
    the event's `pmt_id`, aligned on `(run_id, event_id)`.
- Fit with `lucid_arms.fit_probe` (one run-grouped ridge per embedding dimension, α over
  `probes.ALPHAS`).
- R²_eq = 1 − Σ‖Δz − Â u‖² / Σ‖Δz − mean Δz‖², pooled over the 128 dimensions, on probe-test.
  Report it for Δz_cs and for Δz_tot.
- Class (occupancy model, on Δz_cs; §3.3):
  - *equivariant*: R² ≥ 0.8;
  - *partly equivariant*: 0.4 ≤ R² < 0.8;
  - *non-equivariant*: R² < 0.4.

**X2-c, latent repair with the true gains (oracle).** z̃ = z_D − Â u(δ_r), using the occupancy model
of X2-b with n_e,m from the D hits, then P_N. Compute ρ_LR for eligible models (§3.3):
- *latent repair works*: ρ_LR ≥ 0.5;
- *fails*: ρ_LR ≤ 0.2;
- otherwise *partial*.

Report ρ_LR next to ρ_SWtw and ρ_SWunit.

**X2-d, identification of the module gains.** The target is ln m¹_r,m on the D probe-test runs (17
photon events per run). Three streams:
- **source** (known light; the analogue of a deployed calibration source):
  δ̂ = ln(Σ q / Σ μ), summed over the run's photon events and the module's PMTs with q > 0 and
  μ ≥ 3 PE, where μ is the reconstructed per-PMT μ_p(e) of §2.2;
- **raw** (unlabelled physics data): the run mean of the module's hit-mean log charge, minus the
  mean of the same statistic over all N probe-train events;
- **latent** (unlabelled, calibrated in simulation):
  δ̂ = argmin_δ ‖W(z̄ᴰ_r − z̄ᴺ) − W Â ū_r(δ)‖² + λ‖δ‖². Here z̄ᴰ_r is the run mean of the D embeddings,
  z̄ᴺ the N probe-train mean, ū_r the run-mean occupancy design, and W the inverse square root of
  the within-run covariance of the N probe-train embeddings.
  - λ ∈ `probes.ALPHAS` is chosen by five-fold run-grouped CV on probe-train runs.
  - **Â, z̄ᴺ and W are refit inside each fold.**
  - The columns of the design W Â diag(ū) are standardised on the training folds before the ridge,
    so the grid applies to a standardised design.
  - Simulation truth is allowed in the CV; probe-test truth is never used.
  - This is new code (a generalised ridge), not `fit_probe`.

Report per stream: the median over runs of the per-run RMS error over modules, Pearson r over
(run, module), and coverage (modules with an estimate). Descriptive.

The source stream uses truth (it stands for a source of known light). The raw and latent streams
use no labels.

**X2-e, closing the loop (GPU).** EST(k) conditions with each stream's δ̂, frozen P_N. Report
ρ_EST(k) next to the ceiling ρ_SWmod. Also latent repair with δ̂^latent (CPU). Descriptive.

**X2-f, identifiability spectrum (descriptive).** The singular values of W Â (occupancy model at
the mean occupancy) and the implied CRLB per module at 17 events per run.

**Interpretation:**
- equivariant, and latent repair works: the drift can be undone in latent space from a gain
  estimate, without labels;
- partly or non-equivariant: input calibration (X1) is the repair;
- X2-d and X2-e say which stream can supply the gains, and at what cost in vertex error.

---

## 7. X3 — the price of invariance

### 7.1 X3-a, Fisher accounting on the WCTE geometry (CPU)

**Events.** 1 000 photon events. Vertex from `fiducial_vertex(geometry, rng, cfg.fiducial_fraction)`,
yield log-uniform in [`cfg.y_lo`, `cfg.y_hi`], QE from `sample_qe`. The rng is
`keyed_rng(20261106, "paper3/x3/fisher", "photon")`. μ_p = `expected_charge`(…) ×
`fixed_charge_gain(geometry)` at the attenuation length of `nominal_state`, as in the readout
(`lucid_bombs.py:2083-2087`). `mixed_detectability_inputs` omits the fixed gain, so it is not the
reference here.

**Parameters.** θ = (x, y, z, ln Y); nuisance δ = the 105 module log-gains, evaluated at g = 1.

**Three models, all reported:**
- **C-mean (primary):** charge only, Gaussian, mean g_mod(p) μ_p, variance
  V_p = μ_p(1 + σ_spe²) + v_dark + s_p² μ_p² held fixed (the `lucid_bombs.fisher_event` model, with
  F3 marginalised). Proposition 0 of the plan is exact in this model, and P0 checks its
  implementation.
- **C-var (Gaussian variance-term sensitivity):** the same charge model with the variance scaling
  as the generator's, V_p(g) = g² [μ_p(1 + σ_spe²) + v_dark + s_p² μ_p²], and the Gaussian variance
  term ½ ∂V ∂V / V² added (Appendix E).
  - Photo-statistics then carry some gain-free information.
  - This information comes mostly from low-light PMTs, where the Gaussian term is crude. So C-var is
    reported with the variance term applied to all PMTs, to μ_p ≥ 1 PE and to μ_p ≥ 3 PE.
  - The true low-light information (the SPE peak position, the zero fraction) is not represented.
- **C-mean + occupancy (idealisation):** C-mean plus a Bernoulli hit channel,
  P_p = 1 − exp(−μ_p), gain-independent. The hit flag is in fact a function of the charge, so this
  double-counts low-light information and overstates I_known. It is reported as an upper
  idealisation, never as the headline.

Timing is excluded. The vertex ratios are therefore charge (and occupancy) ratios, and they
overstate the vertex price for a timing-informed detector. Only ln Y is compared with experiment
(X3-5).

**Jacobians.** ∂μ/∂x_i by central difference, h = 1 mm; ∂μ/∂ln Y = μ; ∂(gμ)_p/∂δ_m = μ_p [mod(p) = m].

**Information about θ per event, in each model (formulas in Appendix E):**
- I_known (gains known);
- I_inv, the efficient information with all module gains free;
- I_cal(σ), with a Gaussian prior of width σ on each δ_m, for σ ∈ {0.003, 0.01, 0.03, 0.1, 0.3}.

**Report:** the median over events of the CRLB (diagonal of the 4 × 4 inverse) for x, y, z and ln Y,
per model and case; the ratios inv/known; and σ₉₀, the largest σ that recovers ≥ 90 % of the
known-gain information for each parameter, by interpolation in log σ. A parameter without
information has CRLB = ∞ and ratio = ∞, never the pseudo-inverse value (Appendix E). In C-mean,
ln Y with all module gains free is such a parameter, by P0.

**Proposition check P0 (must pass).** In the C-mean model, on every event, the charge efficient
information for ln Y with all module gains free, divided by the known-gain charge information for
ln Y, must be ≤ 1e-6. ∂μ/∂ln Y is the sum of the module-gain columns, so this is zero by
construction. A failure after one fix attempt is S9 for X3-a. P0 checks the C-mean implementation,
not the generator; C-var is a Gaussian sensitivity, not a faithful model of the generator.

### 7.2 X3-b, invariant variants against retain-and-calibrate (GPU and CPU)

**Families.** In every family, the *calibrated* variant is the standard variant's frozen probe
applied to SWunit D inputs.

| Family | Standard | Invariant | Calibrated |
|---|---|---|---|
| raw features (no encoder) | raw-std: `[ln Σq, NHits, Σq-weighted mean/SD of x, y, z, t]`; `ext_followup.raw_event_features` for muon/electron events, and a new photon version written to the same formula | raw-inv: `[NHits, hit-count-weighted mean/SD of x, y, z, t]` (no charge) | raw-std features on SWunit D charges, raw-std probe |
| random-init encoder (`init9000`) | standard tokens | each hit token's log charge minus the mean log charge of the hit tokens of its module in the same event — invariant to module gains for a fixed hit set, to float32 rounding; unhit tokens unchanged | X1's SWunit reading for `init9000` |
| trained encoders | unerased embeddings | LEACE-erased embeddings (concepts below; Appendix D) | X1's SWunit reading for the same model |

**Trained encoders: model lists and eraser fits.**
- **Photon readings (vertex, log-yield):** MAE × 4, T-1 × 4, frozen-target × 2. The eraser is fit on
  pooled N and D photon probe-train embeddings (both from X1).
- **Electron-energy readings:** the four MAE models only. The eraser is fit on pooled N and D
  particle probe-train embeddings (the D particle-train sets added in §5). T-1 and frozen-target
  LEACE readings are photon-only.

**Concepts (per event):**
- **E1**, the effective log gain: ḡ_e = Σ_p q^HW_p ln G_p / Σ_p q^HW_p, with twin charges as weights
  and the event's own environment's G. Rank 1.
- **E2r**, the run's module log-gain vector (105), erased at a reduced rank r chosen on probe-train
  only.
  - **Permutation unit: the run id.** Each run id is assigned another run id's concept vectors,
    and the run's N and D vectors move together, because they come from the same keyed draw
    (`lucid_bombs.py:497-498`) and share events.
  - 200 permutations, keyed by `keyed_rng(20261106, "paper3/x3/leace_perm", f"{label}/{domain}")`
    (domain `photon` or `particle`). The null is the distribution of the **largest** singular value
    of W Σ_XZ under permutation.
  - **Max-statistic rule:** r = #{j : s_j > q95(null s₁)}. It is conservative and does not
    over-select.
  - r is reported. With r = 0, E2r is `not applicable`.
- **Random controls**, for E1 (rank 1) and for E2r (rank r): erase a random subspace of the same
  rank in the whitened space, using the same formula, with 20 Haar-random bases from
  `keyed_rng(20261106, "paper3/x3/leace_rand", f"{label}/{domain}/{concept}/{i}")`, i = 0…19.

**Erasure sanity.** Fit `fit_probe` from probe-train embeddings to the concept and score on
probe-test, once unerased and once erased. Both R² values are reported, pooled over the concept's
outputs.
- If the unerased R² is ≤ 0.02, the concept is not linearly present, and that variant's readings
  are `not applicable`.
- Otherwise the erased R² must be ≤ 0.02; if not, the variant is `unavailable (erasure
  incomplete)`.

**Protocol.**
- For each standard and invariant variant, the probe is fit on that variant's own N probe-train.
- Metrics come from N probe-test (price) and from D probe-test with the same frozen probe
  (robustness).
- The calibrated variant is evaluated on SWunit D probe-test with the standard variant's frozen
  probe.
- For the invariant `init9000` tokens: photon N (train, test) and D (test), and particle N (train,
  test) and D (test).
- Metrics: vertex mean error, log-yield resolution, and electron-energy relative resolution (raw,
  `init9000`, and the four MAE models).

**Readings (fixed now; per variant pair; classes by §3.3):**
- **X3-1, price in energy.** ρ_E = res_inv(N) / res_std(N), for log-yield and for electron energy:
  - *price present*: ρ_E ≥ 1.2, and the run-cluster CI of res_inv − res_std excludes 0;
  - *no price*: ρ_E ≤ 1.05;
  - otherwise *unclear*.

  For LEACE variants, the ratio for each of the 20 random controls is reported beside it. A price
  counts as gain-specific only if two conditions hold:
  - the LEACE ρ_E exceeds the 95th percentile of the 20 control ratios;
  - the run-cluster CI of res_LEACE − mean(res_control) excludes 0.
- **X3-2, price in vertex (descriptive).** v_inv(N) − v_std(N), with its CI.
- **X3-3, robustness.** v_inv(D) − v_inv(N) and res_inv(D) − res_inv(N), next to the standard
  variant's. *Robust*: |Δv| ≤ 2 mm and a change in energy resolution of ≤ 5 % relative.
- **X3-4, dominance at D.** The calibrated variant against the invariant variant at D:
  - *calibration dominates*: vertex and energy resolution both better, with CIs excluding 0;
  - *invariance dominates*: both worse, with CIs excluding 0;
  - otherwise *mixed*.
- **X3-5, theory side by side (descriptive).** The empirical ρ_E for log-yield next to the X3-a ratio
  √(CRLB_inv / CRLB_known) for ln Y in four rows: C-mean + occupancy, and C-var with the variance
  term on all PMTs, on μ_p ≥ 1 PE and on μ_p ≥ 3 PE. C-mean alone gives ∞ by P0. It is listed as
  the charge-only limit, not as a comparison point.

**Interpretation:**
- price present (gain-specific) and calibration dominates: the plan's Claim 0 holds on LUCiD. An
  invariant representation pays in energy information; a retaining one plus calibration does not;
- no price: occupancy carries the energy at this light level. Claim 0 is then stated for the charge
  channel only, and the paper says so.

### 7.3 X3-T — not authorised

An MAE trained on invariant tokens, 2 seeds × 10 000 steps (about 1.2 L40S h). It needs an explicit
author line, quoted in a new §0 entry, and does not run under this runbook.

---

## 8. Resources and stops

### 8.1 Caps

- **GPU:** L40S only, ≤ 4.0 GPU-h for X0-b, X1, X2 and X3-b together.
  - Projection at X0: Σ sets × per-set wall × 1.3. The per-set walls are measured follow-up job
    walls, which already include model builds and overhead: photon set 13 s
    (`a2_incl_t1_20261106`: 0.1126 h for 32 photon sets); particle set ≤ 88 s, a conservative upper
    bound (`a32_20261106`: 0.5395 h, which also covered photon sets and CPU fits).
  - Expected about 2.7 h for the core and about 3.3 h with the optional models. This includes the
    SWtw (train) sets added in v3, about 0.06 h. The design reviewer's recount was 2.43 h and
    2.85 h from the v2 set lists, and 2.49 h and 2.94 h from the v3 lists.
  - If the projection exceeds 4.0, drop the optional models. If it still exceeds 4.0, STOP (S1)
    before any GPU work.
  - During execution, stop *before* a job that would cross the cap. Every remaining item is
    `not_run (cap)`.
- **CPU:** ≤ 6 wall-hours (tests, X0-a, X3-a, fits, LEACE with its permutations, bootstrap, report),
  booked under this extension in `RESOURCE_LEDGER.json`. The generation cap is untouched (no data).
- **S1 launch rule:** replaced by the combined rule of front matter §F3.

### 8.2 Where each part runs

`/ceph/dwong/jepa/` is shared by both hosts.

| Part | Host | Device |
|---|---|---|
| Implementation, focused tests, full suite (once per SHA), dry run | Deepthought | CPU (`nice -n 10`, `OMP_NUM_THREADS=4`); deepthought2 CPU if Deepthought drops SSH sessions |
| X0-a, X3-a, every standalone fit, LEACE, bootstrap, report | Deepthought | CPU, same rule |
| Every encoder forward pass: X0-b, X1, X2 (S and EST sets), X3-b encoder passes | deepthought2 | L40S, always through `scripts/jepa_run.sh` (no guard bypass) |
| Independent code review | any host with a read-only clone | — |
| Acceptance record (lead) | lead's clone; pushed to `JEPA` | — |

### 8.3 Host rules

- At most two of the user's L40S jobs, each ≤ 45 % of GPU memory.
- Guard HOLD or SHED: wait and retry; book failed attempts.
- `TMPDIR=/ceph/dwong/jepa/tmp/`; ≥ 30 000 free entries after the projected writes.
- `<ext>` = `/ceph/dwong/jepa/returns/ext_repair_20261106/`, with sub-roots `x0/`, `x1/<label>/`,
  `x2/<label>/` and `x3/`. Results go to `results/jepa/20261106_repair/`.
- Never overwrite or resume another root. Push to `JEPA` from the author's Mac only.

### 8.4 STOP conditions (the only ones)

- S1;
- S3 (a real specification change, including any change to a reading or threshold after outcomes
  are seen);
- S4 (no GitHub route after three documented attempts);
- S9 (the X0-b gate or the P0 check still fails after one fix attempt);
- an import-closure failure at the end of a mode (§9.2): that mode's outputs are quarantined (kept,
  marked, never used). Any fix changes code in `<P>` (breaking gate (c)) or the runbook, so the
  rerun needs a re-review and a new acceptance record first;
- the X0-a content check (for the readings it lists only);
- a missing or mismatched pinned dataset or checkpoint (for the affected items only);
- an entries-preflight failure;
- a cap that would be crossed.

On a STOP: preserve logs and partial outputs, record the STOP, and continue the independent items.
Dependencies: X1 and X2 need X0-b; X2-e needs X2-d; the X3-b LEACE variants need the X1
embeddings; X3-a depends on nothing.

---

## 9. Implementation, review, acceptance and execution

### 9.1 Implementation

- Commit the combined runbook unchanged (front matter, §F2).
- Add `src/jepa_noise/ext_repair.py`, `scripts/jepa_ext_repair.py` and
  `src/jepa_noise/tests/test_ext_repair.py`. These names are free at `ac8e220`.
- CLI modes: `dry-run`, `preflight` (X0-a), `reproduce` (X0-b), `x1`, `x2`, `x3-fisher`, `x3`,
  `report`. **No training mode exists.**

### 9.2 Gate

`ext_repair.assert_authorised(mode)` is called from the library entry point, not only from the CLI.
Any failure raises `SystemExit(2)`. It requires:

- (a) the sha256 of the committed `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` to equal the value
  in the acceptance record;
- (b) `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json` =
  `{runbook_sha256, impl_commit, review_path, review_verdict, lead, date}`, with
  `review_verdict == "PASS"`;
- (c) both `git diff --quiet <impl_commit> -- <P>` and an empty
  `git ls-files --others --exclude-standard -- <P>`, where `<P>` is:
  - `src/jepa_noise/{__init__,ext_repair,ext_followup,ext_readout,ext_h12,ext_dejitter,ext_arch,lucid_run,lucid_arms,lucid_bombs,lucid_views,point_encoder,bridge_run,train,crlb,jepa,backbone,linear,probes,metrics,sigreg}.py`;
  - `src/noise_module_lucid/` and `src/jepa_noise/configs/`;
  - `scripts/{__init__,jepa_ext_repair,jepa_ext_followup,jepa_pregrid_checks,jepa_ext_dejitter,jepa_ext_readout}.py`;
  - this runbook.

**Import-closure assertion.** The dry run executes the setup path of every CLI mode, and **every
mode re-checks at its end**, that every repository module in `sys.modules` (by `__file__`) lies
inside `<P>` or `ext_followup.FORCED_IMPORT_EXCEPTIONS`. In the dry run, a module outside is a
failure fixed before review. In a mode, it is the closure STOP of §8.4.

The acceptance record goes in a later commit, so HEAD equality is never required.

### 9.3 Tests

- **§2.5 context:**
  - the identity condition gives bit-identical features;
  - both attributes are restored, also after an exception;
  - the factor mapping is per run (a synthetic two-run set);
  - the HW redirect returns the twin store with identical event ids;
  - a path outside the condition's set raises.
- **Factors:** build a tiny run at **two settings with shared draws** in a pytest temporary
  directory. Use `lucid_bombs.generate_run(lb.from_synthetic(), GEN_CFG, run, with_twin=True)`
  twice with RunSpecs differing only in `setting`; `test_lucid_bombs.py:742` shows the
  single-setting call. Then check, to float32 tolerance:
  - SWtw charges equal the s = 0.05 charges on common hits;
  - SWunit charges equal the twin charges on common hits;
  - the two twins are equal.

  This test-only synthetic generation is allowed; §2.6 governs the extension's modes.
- **State check:** a recomputed `state_sha256` equals the manifest value, and a corrupted manifest
  fails.
- **μ_p reconstruction:** on the synthetic run, Σ_p μ_p(e) equals the stored `expected_pe[e]` for
  every photon event.
- **Invariant tokens:** invariant to arbitrary module scalings for a fixed hit set, to float32
  tolerance; unhit tokens unchanged.
- **Raw features:**
  - raw-std on muon/electron events reproduces `ext_followup.raw_event_features` bit-identically;
  - raw-inv is invariant to module scalings for a fixed hit set.
- **LEACE:**
  - on synthetic Gaussian data, the post-erasure cross-covariance with the concept is ≤ 1e-8
    relative;
  - the eraser is idempotent;
  - on synthetic data with shared N/D concept draws (200 runs × 17 events × 2 environments) and the
    fixed keyed seeds:
    - the max-statistic rule gives r ≤ the planted rank at every tested amplitude;
    - it gives r = the planted rank (5 of 105) when the planted signal's standard deviation per
      direction is 10 × the noise standard deviation;
    - on the null, r = 0 in at least 17 of 20 keyed seeds (expected about 19 of 20 at the 95th
      percentile);
  - permutation moves each run's N and D vectors together;
  - the random controls have the stated rank.
- **Equivariance fit:** recovers a known A on synthetic linear data; probe-test runs never enter a
  fit or a hyperparameter choice.
- **Identification estimators:**
  - source and raw recover δ exactly on noise-free synthetic data;
  - latent recovers δ with λ = 0 on a well-conditioned noise-free synthetic design, to 1e-8
    relative.
- **Fisher accounting:**
  - P0 holds on the synthetic geometry in the C-mean model;
  - the C-mean charge information for one parameter equals `lucid_bombs.fisher_event` for the same
    μ and derivative;
  - the C-var model reduces to C-mean when the variance term is switched off;
  - the μ cut of the C-var variance term applies only to the PMTs it names;
  - in C-mean with all module gains free, ln Y is reported as CRLB = ∞ and ratio = ∞, and the
    x, y, z CRLBs equal the pseudo-inverse values to 1e-9 relative (Appendix E).
- **Bookkeeping:** the eligibility rule; ρ undefined below threshold; the §3.3 classification;
  the run-cluster resolution bootstrap.
- **Forbidden paths:** sentinel patches show that no §2.6 path (including every listed
  `ext_followup` gate caller) is reached by any mode's setup path.
- **Gate:** the negative cases (a), (b), (c).
- **Hash mismatch and no overwrite.**

The focused tests plus the full `src/jepa_noise/tests` suite run once on the final SHA, deselecting
only the three known `test_guard` launcher tests. No GPU work is a test.

### 9.4 Independent review

A fresh-context reviewer reads the actual diff and this runbook. Blocking checklist:

- (i) condition factors and the §2.5 context (twin redirect, recomputed states, hash and content
  checks, failing closed);
- (ii) the X0-b reproduction and the identity check;
- (iii) probes, metrics, pairing and bootstrap;
- (iv) the X2 fits (run-disjoint train and test; no probe-test truth in any fit or hyperparameter
  choice);
- (v) the X3-a Fisher accounting, P0 and the C-var term;
- (vi) LEACE (reduced rank, permutation rule, random control) and the invariant transforms;
- (vii) the gate, the import closure and the forbidden call paths;
- (viii) the cost projection against the caps.

The reviewer records PASS or BLOCK per item, with file:line evidence. Blockers are fixed and
re-reviewed. Then the lead writes the acceptance record.

### 9.5 Execution

- Use the accepted SHA and the CLI-generated commands only. Recheck the guard, the ledger, the jobs
  and the data.
- **Order (fixed):** X0-a → X0-b → X1 (in parallel, X3-a on CPU) → X2 (S sets, X2-a/b/c, X2-d, then
  the X2-e EST sets) → X3-b → report.
- GPU steps go through the launcher:
  `JEPA_JOB_MODULE=scripts.jepa_ext_repair JEPA_JOB_KIND=evaluation scripts/jepa_run.sh <run_id> --host deepthought2 --duration-minutes <M> -- <mode args>`.
  CPU steps run directly with `nice -n 10` and 4 threads.

### 9.6 Deliverables

In `results/jepa/20261106_repair/`:
- `plan.json`, `inventory.json`, `reproduction.json`, `x1.json`, `x2.json`, `x3_fisher.json`,
  `x3.json`, `readings.json`;
- `README.md`, with the readings tables and these figures:
  - the drift ladder per model (N, D, refit, SWtw, SWunit, HW, NOISY);
  - the X1-g residual split;
  - ρ by repair and model;
  - R²_eq per model;
  - identification error by stream;
  - Fisher CRLB ratios against σ for the three models;
  - the price-of-invariance table with the random control;
- `LARGE_ARRAYS.json` (sha256 of every cached embedding file > 5 MB on `/ceph`);
- a `STATE.md` section, a ledger line and the `DECISIONS.md` lines.

---

## 10. Out of scope

Not authorised by this runbook:
- training of any kind, including X3-T;
- data generation, including attenuation drift (law F2) and new threshold, efficiency-loss or
  module-loss cells;
- new response families, or re-digitisation;
- any change to a registered prediction;
- the TIDMAD arm;
- the maintenance-policy study.

---

## Appendix A. Factor arrays

For run r, with `S0 = RunSpec(s=0.05).state(geometry)` and `S1 = RunSpec(s=0.30).state(geometry)`
built as in §2.2, and `mod = geometry.module_of_pmt`:

- `SWtw = S0.gains / S1.gains`; `S = S1.gains / S0.gains`;
- `SWunit = 1 / S1.gains`; `SWunitN = 1 / S0.gains`;
- `SWmod = 1 / S1.module_gain[mod]`;
- `NOISY(σ) = 1 / (S1.gains * exp(σ * xi[mod]))`, with
  `xi = keyed_rng(20261106, "paper3/x1/noisy", run_id).standard_normal(105)`, drawn once per run and
  shared by the four σ values `0.01, 0.02, 0.05, 0.1`;
- `EST(k) = 1 / exp(delta_hat_k[mod])`, with NaN replaced by 0.

## Appendix B. Equivariance fits

- Features: global, u = δ_r (105); occupancy, u_e,m = δ_r,m · n_e,m.
- Targets: Δz (128 per event), one `lucid_arms.fit_probe` target per dimension: run-grouped ridge,
  α ∈ `probes.ALPHAS`, intercept fitted.
- R²_eq pooled over the 128 dimensions on probe-test, as in §6.

## Appendix C. Identification estimators

- **Source:** per run and module, sum over the run's photon events and the module's PMTs with q > 0
  and μ_p(e) ≥ 3 (μ_p from §2.2): δ̂ = ln(Σ q / Σ μ). Modules with no such PMT get NaN.
- **Raw:** per event and module, the mean of ln q over hit PMTs; the run mean over events with a
  hit in the module; minus the mean of the same statistic over all N probe-train events with a hit
  in the module.
- **Latent:** as in §6.
  - W = (C + εI)^(−1/2), with C the pooled within-run covariance of the N probe-train embeddings
    and ε = 10⁻⁶ tr(C)/128.
  - λ ∈ `probes.ALPHAS`, by five-fold run-grouped CV on probe-train runs (error against simulation
    truth there).
  - Â, z̄ᴺ and W are refit inside each fold, and the design columns are standardised on the
    training folds. The final fit uses all probe-train runs.

## Appendix D. LEACE and its reduced-rank form

Belrose et al., *LEACE: Perfect linear concept erasure in closed form*, NeurIPS 2023
(arXiv 2306.03819).

1. Centre X (n × d) and Z (n × k).
2. Σ_XX = XᵀX/n, Σ_XZ = XᵀZ/n.
3. W = (Σ_XX + εI)^(−1/2), with ε = 10⁻⁶ tr(Σ_XX)/d.
4. SVD W Σ_XZ = U S Vᵀ. Full LEACE (E1, rank 1) uses all columns of U with S > 10⁻⁸ S_max. The
   reduced-rank form (E2r) uses the first r columns U_r.
5. P = U_r U_rᵀ; the eraser is r(x) = x − W⁻¹ P W (x − μ_X).

**Rank rule (E2r).**
- Permute run ids, with each run's N and D concept vectors moving together; 200 permutations.
- For each permutation, record the largest singular value of W Σ_XZ.
- r = #{j : s_j > q95 of those maxima} (§7.2).

**Random controls.** Replace U_r with a Haar-random orthonormal d × r basis (r = 1 for E1), with
20 draws.

Checks:
- on the fitting set, Cov(r(X), Z) ≈ 0 along the erased directions (full LEACE: relative
  ≤ 10⁻⁸);
- on probe-test, the pooled R² of the concept from the erased embeddings is ≤ 0.02 (§7.2).

## Appendix E. Fisher accounting

- **C-mean.** I = Σ_p j_p j_pᵀ / V_p, with j_p = (∂μ_p/∂θ, ∂μ_p/∂δ) at g = 1 and V_p fixed.
- **C-var (Gaussian variance-term sensitivity).** I = Σ_p [ j_p j_pᵀ / V_p + c_p ½ (∂V_p)(∂V_p)ᵀ / V_p² ],
  with V_p(g) = g² [μ_p(1 + σ_spe²) + v_dark + s_p² μ_p²].
  - The variance derivatives at g = 1 are ∂V_p/∂δ_m = 2 V_p [mod(p) = m] and
    ∂V_p/∂θ = (1 + σ_spe² + 2 s_p² μ_p) ∂μ_p/∂θ.
  - c_p = 1 for the PMTs in the cut (all; μ_p ≥ 1 PE; μ_p ≥ 3 PE — three reported rows) and 0
    otherwise.
- **Occupancy.** I^O_θθ = Σ_p (∂P_p/∂θ)(∂P_p/∂θ)ᵀ / (P_p(1 − P_p)), with
  ∂P_p/∂θ = e^(−μ_p) ∂μ_p/∂θ.
- **Cases**, with the blocks of I in (θ, δ):
  - I_known = I_θθ;
  - I_inv = I_θθ − I_θδ I_δδ⁺ I_δθ;
  - I_cal(σ) = I_θθ − I_θδ (I_δδ + σ⁻² I)⁻¹ I_δθ.

  Occupancy, when included, adds I^O_θθ to each.
- PMTs with μ_p ≤ 0 are skipped. Pseudo-inverse (rcond 10⁻¹²) for I_δδ when a module has no light.
- **CRLBs** are the diagonal of the inverse of the 4 × 4 θ-information, with one rule for a
  parameter that carries no information:
  - if its efficient information (its diagonal entry in I_inv or I_cal(σ)) is ≤ 10⁻⁶ of its entry
    in I_known, its CRLB and its ratio to the known-gain CRLB are reported as ∞. The pseudo-inverse
    value (about 0) and the `inv` value (large, possibly negative) are never reported;
  - its row and column are then dropped, and the other CRLBs are the diagonal of the inverse of the
    remaining block;
  - in C-mean this applies to ln Y with all module gains free, which is zero by P0. The null
    direction is then the pure ln Y axis, so the x, y, z CRLBs equal the pseudo-inverse values.
  - Medians over events treat ∞ as larger than any finite value. The number of events with
    CRLB = ∞ is reported per model, case and parameter; under the μ_p ≥ 1 PE and μ_p ≥ 3 PE cuts,
    C-var can reduce to C-mean on dim events.

## Appendix F. Reproduction targets

The X0-b table (§4). Sources: `results/jepa/20261103_followup/readings.json` → `A-2` →
`<label>.environments.{s005,s030}.{vertex_refit_m, vertex_frozen_m}`; `ext_followup.A1_S005["init9000"]`
and `A1_S030["init9000"]["vertex_resolution_m"]`.

## Appendix G. Review record

**Design review v1** (fresh reviewer, repository level at `ac8e220`, with local checks on the
synthetic fixture): **BLOCK**, 5 blocking and 20 nonblocking findings.

| Finding | Change in v2 |
|---|---|
| B1: `expected_pe` is a per-event total | §2.1 corrected. §2.2 reconstructs per-PMT μ_p from the keyed QE stream, checked against the stored total; §2.6 allows this derived truth; X0-a runs the check |
| B2: 105-dimensional LEACE concept is degenerate | E2 is replaced by E2r: reduced rank chosen by a run-grouped permutation rule on probe-train, a random same-rank control, a pinned sanity-R² protocol (§7.2, Appendix D) |
| B3: `metrics.py` missing from `<P>` | added; R² computed from predictions; the import closure is re-asserted at the end of every mode (§9.2) |
| B4: the X1 rule attributed the post-refit residual with a frozen quantity, and assumed a walk effect | new reading X1-g (refit split D / SWunit / HW); rule rewritten; "walk" removed (walk residual is identically zero at `threshold_pe = walk_calibration_pe`) (§1, §2.1, §5) |
| B5: X3-b particle and LEACE protocol inconsistent | model lists pinned; D particle-train sets added for the four MAE models; the calibrated variant defined as the standard frozen probe on SWunit D (§5, §7.2) |
| N1 / N2 / N3 / N4 | X3-a reports C-mean (primary, P0), C-var (generator-faithful sensitivity) and occupancy (labelled idealisation); timing caveat corrected; reference fixed (§7.1, Appendix E). [Historical wording. After NN2, C-var is a Gaussian variance-term sensitivity, not a generator-faithful model.] |
| N5 | twin identity by content, and SWtw-vs-N charges; a content mismatch stops the shared-draw readings (§4, §8.4) |
| N6 | out-of-range-charge caveat (§3.1) |
| N7 | §3.3 classification rule (seeds disagree, single seed); X3-4 symmetric CIs; σ½ rule; X1-a by construction |
| N8 | one ξ per run shared by all σ; label corrected (§3.1, Appendix A) |
| N9 | "no probe-test truth in any fit or hyperparameter choice"; Â refit inside the λ CV folds (§2.6, §6, §9.4) |
| N10 | SWtw log-yield refit dropped (§5) |
| N11 | frozen energy probe via `fit_probe`/`probe_predictions` with `_energy_task` conventions (§2.4, §3.2) |
| N12 | fits use `probes.ALPHAS` via `fit_probe`; the latent estimator is new code on the same grid (§6, Appendices B–C) |
| N13 | test tolerances; two-setting synthetic run spelled out (§9.3) |
| N14 | gate-caller names corrected and completed; enforced by sentinel test (§2.4, §2.6, §9.3) |
| N15 | write scope narrowed to the extension's code (§2.6) |
| N16 | wrapper fails closed; reused wrappers acknowledged (§2.5) |
| N17 | features bit-identical; embeddings ≤ 1e-6 in one process (§4) |
| N18 | line range `:532-556`; free entries 37 986; the particle-set wall is described as an upper bound |
| N19 | redundancy of the state check noted, check kept (§2.2) |
| N20 | inventory re-hash may be restricted to the needed labels (§2.3) |

**Focused re-review of v2** (same reviewer, same scope, with new local and simulation checks):
**BLOCK**. B1, B3, B4 and B5 resolved; B2 partly resolved; N1–N20 addressed, N9 partly. One new
blocking finding and 13 nonblocking findings.

| Finding | Change in v3 |
|---|---|
| NB1: the E2r rank rule (permutation unit unspecified; same-index comparison over-selects; its test cannot pass reliably) | permutation by run id with N and D together; max-statistic rule r = #{s_j > q95(null s₁)}; the test restated to r ≤ planted always, r = planted at a stated strong amplitude, r = 0 on the null (§7.2, Appendix D, §9.3) |
| NN1: X1-g referenced to HW, not N | X1-g primary is now N-referenced, D / SWtw(refit) / N, with SWtw (train) added (+0.06 h); the HW-referenced split and v_N − v_HW(refit) are secondary (§5) |
| NN2: C-var is a Gaussian sensitivity | renamed; reported for variance-term cuts all / μ ≥ 1 / μ ≥ 3 PE; low-light information stated as missing (§7.1, Appendix E) |
| NN3: §2.6 oracle sentence contradicted the runbook | rewritten as suggested (§2.6) |
| NN4: "only recomputation" forbade X3-a inputs | the prohibition is scoped to hit-level generation and readout (§2.6) |
| NN5: closure failure labelled S9 | separate closure STOP with quarantine (§8.4, §9.2) |
| NN6: gaps in §3.3 | one-seed-eligible, unseeded raw family, `init9000` X3-b and per-seed flags covered (§3.3) |
| NN7: one random-control draw | 20 Haar draws, including the rank-1 control for E1; gain-specific needs > q95 of the controls and a CI against their mean (§7.2) |
| NN8: vacuous sanity check | unerased R² reported; ≤ 0.02 unerased means `not applicable` (§7.2) |
| NN9: μ_p wording | RunSpec `run_id`, `int(event_id)`, the dataset seed named (§2.2) |
| NN10: latent CV optimism and grid scale | Â, z̄ᴺ and W refit per fold; standardised design; λ = 0 test (§6, Appendix C, §9.3) |
| NN11: config path | `src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` (§1) |
| NN12: "by construction" | "consistent to the X0-b tolerance" (§4, §5) |
| NN13: content-check STOP list | X1-g and X1-d's v_HW − v_N added (§4) |

**Confirmation check of v3** (same reviewer, same scope, with new simulation checks of the rank
rule and of the C-mean CRLB): **PASS**, no blocking findings.
- NB1 is resolved. Of NN1–NN13, 12 are resolved and NN3 is partly resolved (wording).
- Four nonblocking edits were recommended. All are applied in this version:

| Finding | Change |
|---|---|
| A1: in C-mean, ln Y with all module gains free carries no information (P0), and the pseudo-inverse would report its CRLB as about 0 | CRLB = ∞ and ratio = ∞ for a parameter whose efficient information is ≤ 10⁻⁶ of its known-gain value; medians treat ∞ as the largest value, and such events are counted; X3-5 compares with C-mean + occupancy and the three C-var rows (§7.1, §7.2, Appendix E) |
| A2: a fix after a closure STOP changes `<P>` or the runbook | the rerun needs a re-review and a new acceptance record first (§8.4) |
| A3: the LEACE rank test's amplitude and null handling | amplitude stated as signal standard deviation per direction = 10 × the noise standard deviation; the null passes if r = 0 in at least 17 of 20 keyed seeds (§9.3) |
| A4: permutation and control streams keyed by label only; historical wording in the v1 table | the domain is added to both keys (§7.2); bracketed note on the v1 table's N1–N4 row |
| NN3 (partly resolved) | the oracle rule now names probe-test true gains, and allows probe-train simulation truth to calibrate X2-b, the latent stream and the LEACE concepts (§2.6) |

The reviewer's cost recount is about 2.49 L40S h for the core lists and 2.94 h with the optional
models. The projection in §8.1 (about 2.7 h and 3.3 h) is conservative and under the 4.0 h cap.

---

# Part B — Shortcut-collapse follow-up: R-1 objective and repairability, R-2 position-free trainable target, R-3 positioning

Part B's design was not reviewed separately; the joint review (front matter §F5) covers its design items (§9.4 (i)–(ii)). R-1 reads Part A outputs; R-2 is the only training in this file; R-3 was done at design time.

What this part does:
- **R-1** (CPU, report only): after the repair pilot has produced its X1 and X2 readings, compare
  how repairable the generative encoders (MAE) are against the joint-embedding encoders (JEPA).
- **R-2** (GPU, **training, authorised**): train the fixed-label SIGReg-JEPA with one change — the
  target (label) pass no longer sees PMT position or direction — and test whether the
  event-invariant collapse disappears while the target stays trainable.
- **R-3** (desk, done at design time): the literature check and positioning. Its result is recorded
  in §7 and already written into the Paper 3 plan. The execution agent only cites it in the README.

---

## 0. Authority

Quote both lines verbatim in `DECISIONS.md`.

1. The author, 7 Nov 2026, 00:12 CET, in the design conversation, approving items 1, 2 and 3 of
   the proposal (1: the generative-versus-JEPA repairability expectation; 2: a target-side fix of
   the JEPA collapse, about 1 L40S h of training; 3: positioning and citations):
   > "一二三我都批，写成一个runbook"

   This is the explicit training line for R-2, and only for R-2 as specified in §5: one variant,
   two seeds, 10 000 steps each, one 20-step smoke, under the cap of §8.1. No other training is
   implied. X3-T of the repair pilot stays **not authorised**.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

### 0.1 Fix it yourself, do not stop to ask

- Fix small problems yourself (bug, path, environment, guard HOLD/SHED, launcher timeout), write
  one `DECISIONS.md` line ("author standing approval, 30 Oct 2026"), and continue.
- Stop only for the conditions in §8.4.

---

## 1. Background and questions

Committed results this runbook builds on (`docs/jepa/STATE.md`; `results/jepa/20261101_h12/`,
`20261102_static/`, `20261102_scale/`, `20261103_followup/`):

- The fixed-label SIGReg-JEPA (`fix9000`/`fix9001`, cell `dev_s005_within`) collapses to an
  **event-invariant** code: `f_layout` 0.9970, `f_event` 7.9e-5 (E0 priors). Its full-input photon
  vertex error is 0.219259 / 0.204281 m, worse than the random-init floor 0.160072 m.
- The H12 study found **H2 dominant**: the trainable target admits a trivial solution. Freezing the
  target at a random init (E3) restores learning: vertex 0.138 / 0.131 m.
- Zeroing the geometry columns everywhere (E2 `geomblind`) does not help (0.830 / 0.842 m). A frozen
  static target whose event columns are zeroed (`static_target`) does not recover (0.160374 /
  0.161148 m). Three times more data and steps (`scale`) does not help (`f_event` ≈ 1e-6).
- EMA target (`ijepa`), larger SIGReg weight (`lambda`) and stop-gradient (`stopgrad`) are
  **intermediate**: `f_event` between 1e-3 and 1e-2.
- The `fix9000` per-PMT invariant code is not identified by any single fixed PMT attribute
  (code-origin addendum). It is still event-invariant, and in the label pass (§2.2) the only
  event-invariant inputs a target token carries are its scaled position and direction (feature
  columns 0–5).
- Every MAE checkpoint is event-dependent, keeps module-gain information, and is the best encoder on
  every task.

An external result points the same way. Baade et al. (arXiv 2610.20533) find that LeJEPA, whose
regulariser SIGReg is ours, collapses to a shared watermark when one is available. They also find
that a generative target, which must explain all latent variables, avoids this (§7).

| ID | Question | Item |
|---|---|---|
| **Q-R1** | Are generative (MAE) encoders more repairable under gain drift than joint-embedding (JEPA) encoders: closer to equivariant, better repaired in latent space and by calibration? | R-1 |
| **Q-R2** | If the trainable target cannot see PMT identity (position and direction), does the event-invariant collapse disappear, with the target still trainable? | R-2 |

---

## 2. Snapshot, data, code and side effects

### 2.1 Snapshot

At `ac8e220`: L40S ledger `E2_lucid_l40s_deepthought2` 129.7461/170; generation CPU 18.425639/48;
free entries 37 725 (live check of 6 Nov). Part A (the repair pilot) may run before, after or alongside Part B. Its GPU cap (4.0 h) is
counted in the combined S1 rule (front matter §F3) whether or not it has run.

### 2.2 The label pass in the code (what R-2 changes)

In `lucid_arms.LucidArm._encode_views` (`lucid_arms.py:405-420`) the SIGReg arm encodes three
times with the one trainable encoder:
- context: `jepa.encode(features, hidden_mask=batch.hidden)`;
- full: `jepa.encode(features)` (anchor and evaluation only);
- **label**: `jepa.encode(features, hidden_mask=~batch.target)`, gathered at the hidden (target)
  indices. It is not detached, so the encoder is trained through it.

`_forward_sigreg` (`:464-474`) uses the label for the prediction MSE and for the SIGReg target term.
Tokens carry nine features (`point_encoder.FEATURES`): 0–2 scaled position, 3–5 direction, 6 ln q,
7 time, 8 hit flag. The predictor reads the context read-out plus position queries (the three
position columns of the geometry, `point_encoder.py:357-369`). For the `within` split the target
set depends on the event id and its hits (`lucid_views.within_mask`), so it changes from event to
event.

### 2.3 Pinned data and setup

- R-2 uses exactly the data, cell, configuration and trainer of the H12 E3 and `static_target` runs:
  - data root `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`, opened with the same read-only
    hash checks (`ext_h12.preflight_data`, `ext_h12.datasets_for`);
  - cell `dev_s005_within` (`lucid_run.dev_cell`); arm `sigreg_jepa`;
  - config `src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (the `jepa_ext_static.py`
    default) and the arms config `load_setup` builds there;
  - `lucid_run.run_one` with `evaluate=False`, as in `ext_h12.train_variant` /
    `ext_static.run_one_training`.
- Hashes (manifest sha256, as in the H12 runbook): dev_train `b1b3f4…`, probe-train `093ea894…`,
  probe-test `c11edc71…`. Copy the full values from `ext_h12.DATA_MANIFESTS`. A mismatch is a STOP
  for R-2.
- **No data generation of any kind.**

### 2.4 Reusable code (read-only imports; never edit)

- `ext_h12`: `preflight_data`, `datasets_for`, `load_probes`, `evaluate_run` (full-input and
  half-pass readout through `ext_readout.evaluate_item`), `REPRODUCTION`,
  `REPRODUCTION_TOLERANCE_M`, `attach_frozen_target` (precedent only; not called).
- `ext_static`: the structure of `build_arm_patch`, `prepare_run_dir`, `run_one_training`,
  `evaluate_run_dirs`, `summarize_run` and `static_target_r`, copied into the new module and
  renamed. Do not import `ext_static` functions that check its own runbook gate.
- `ext_readout`: `layout_fractions` and `effective_rank` (already called by `evaluate_item`).
- `ext_followup._classify_event` (the A-1 class rule; pure function).
- `lucid_run`: `run_one`, `build_arm`, `dev_cell`, `run_id_for`, `sha256_file`, `git_head`,
  `git_status`.

### 2.5 Forbidden paths (each covered by a test)

- Any data generation; `lucid_run.ensure_dark_bank`; any path to `ensure_dataset` other than the
  hash-checked `ext_h12.datasets_for` / `load_probes`.
- Any training other than the R-2 smoke and the two R-2 runs; resuming another root.
- Every gate-reaching function of other extensions (`ext_static.assert_authorised`, the
  `ext_followup` readout and report callers, `ext_repair` modes).
- Writing into the repair pilot's `<ext>` or results directory (R-1 reads them only).

**Write scope.** The extension writes only to `<ext>` = `/ceph/dwong/jepa/returns/ext_shortcut_20261107/`,
`results/jepa/20261107_shortcut/` and `TMPDIR`. Repository bookkeeping (ledger, `STATE.md`,
`DECISIONS.md`, new source and test files) follows the existing rules.

---

## 3. R-1 — are generative encoders more repairable? (CPU, report only)

**Runs only when** `results/jepa/20261106_repair/readings.json` exists with X1 and X2 completed
(the pilot's `report` mode has run). Otherwise R-1 is `not_run (pilot outputs missing)`. That is
not a STOP; the R-2 report goes out without R-1, and R-1 is filled in later with
`report --r1-only`.

**Inputs (read-only):**
- `results/jepa/20261106_repair/{x1.json,x2.json,readings.json}`;
- for the response size only: the pilot's cached N and D photon probe-test embeddings in
  `/ceph/dwong/jepa/returns/ext_repair_20261106/x1/<label>/`, each accepted only if its sha256
  matches the pilot's `LARGE_ARRAYS.json` or cache sidecar.

**Families (fixed now):**

| Family | Models (pilot core) | Optional (if the pilot ran them) |
|---|---|---|
| generative | `mae_within_trigger9000/9001`, `mae_across_trigger9000/9001`, `t1_s030_within_trigger9000/9001`, `t1_s030_across_trigger9000/9001` | `mae_within_median9000/9001` |
| joint-embedding, working | `frozen_target9000/9001` | `ijepa9000/9001` (intermediate) |
| joint-embedding, collapsed | `fix9000/9001` | `static_target9000/9001` |
| untrained reference | `init9000` | — |

The collapsed models are reported separately. They are the information-poor end of the family and
would confound "objective" with "collapse" if pooled.

**Table, one row per model:**
- **Response size:** S = E‖z_D − z_N‖² / E‖z_N − z̄_N‖², paired on `(run_id, event_id)`.
- From the pilot X2:
  - R²_eq (occupancy model, Δz_cs) and its class;
  - ρ_LR (latent repair with true gains).
- From the pilot X1:
  - ρ_SWtw, ρ_SWunit;
  - the X1-g share_cs;
  - the drift cost Δ_D and its eligibility.
- **Class in the plan's four-way taxonomy:**
  - *invariant* if S < 0.01 and Δ_D is not eligible;
  - otherwise the X2-b class: *equivariant*, *partly equivariant* or *non-equivariant*;
  - a model with Δ_D eligible and every repair ρ ≤ 0.2 is additionally flagged *information-losing
    (candidate)*.

**Reading R1 (descriptive; expectation stated before the pilot outcomes are seen).** Expectation:
generative encoders are more repairable than the working joint-embedding encoder. Compare the median
over the eight generative models with each `frozen_target` seed, for R²_eq and for ρ_LR:
- *supports*: the generative median exceeds both `frozen_target` seeds on both quantities;
- *contradicts*: the generative median is below both seeds on both quantities;
- otherwise *mixed*.

ρ_SWunit (calibration) and S are reported beside these and not combined into the reading. With 8
against 2 models this is a description, not a test. The README says so in one line.

---

## 4. Shared definitions for R-2

- **Event-dependence class** (the A-1 rule, `ext_followup._classify_event`), on full-input
  `f_event` from `ext_readout`:
  - *event-dependent*: `f_event` ≥ 0.01 in both seeds;
  - *event-invariant*: ≤ 1e-3 in both seeds;
  - otherwise *intermediate*.
- **Floors:** random-init full-input vertex 0.160072 m; half-pass 0.174821 m
  (`ext_static.INIT_FULL_VERTEX_M`, `INIT_HALF_VERTEX_M`).
- **Recovery rule** (the `static_target_r` / E3 rule, unchanged):
  - *recovered*: in both seeds, full-input vertex < 0.160072 m, full-input token ln q R² ≥ 0.5,
    half-pass vertex < 0.174821 m and half-pass token ln q R² ≥ 0.5;
  - *not recovered*: in both seeds, full-input vertex ≥ 0.160072 m or full-input token R² < 0.5;
  - otherwise *unclear*.
- **Reference values** (descriptive; never thresholds), full-input vertex in m: `fix` 0.219259 /
  0.204281, E3 frozen target 0.138 / 0.131, `static_target` 0.160374 / 0.161148, MAE `mae9000`
  0.104949.

---

## 5. R-2 — a position-free trainable target (GPU; training authorised)

### 5.1 The variant

Variant `target_posfree` (variant label only; `arm_name` stays `"sigreg_jepa"`). It differs from the
fixed-label SIGReg-JEPA in exactly one place: **the label pass receives a clone of
`batch.features` with columns 0–5 (scaled position and direction) set to zero.**
- Unchanged: the context pass, the full pass, the predictor and its position queries, the SIGReg
  terms, the loss weights, the optimiser, the schedule, the data, the cell and the seeds.
- The label is **not** detached, and the encoder stays shared and trainable. The H2
  trainable-target condition is kept on purpose. Only the raw material for an event-invariant
  per-PMT code is removed from the target side.
- Installed by rebinding `arm._encode_views` through a process-level `lucid_run.build_arm` wrapper,
  as in `ext_static.attach_static_target` / `build_arm_patch`. The rebound function is the
  `lucid_arms.py:405-420` body with `label_features = posfree_features(batch.features)` in the
  label pass only.

**Mechanism under test.** In the label pass, a target token's only event-invariant inputs are
columns 0–5. Without them, a target token sees its own ln q, time and hit flag, the learned mask
token elsewhere, and window membership in the fixed Hilbert order. The target set changes per event,
so window composition gives no stable per-PMT identity. An event-invariant per-PMT label code then
has no direct input to be built from.
- If R-2 recovers, the collapse needs the target's access to PMT identity, and a JEPA can be
  fixed without freezing its target.
- If it does not recover, the trivial solution survives through the remaining positional structure
  (token order and windows). The next manipulation is then the token order, not the features.

### 5.2 Manipulation check (CPU, before any training; must pass)

On one tokenised batch of the training cell (seed 9000 init):
- the label-pass input has columns 0–5 exactly zero and columns 6–8 equal to the original; the
  original `batch.features` is not mutated; the context and full passes receive the original;
- **identity-blindness:** permuting columns 0–5 across PMTs in the original features leaves the
  label tensor unchanged (max |Δ| ≤ 1e-6), while it changes the context read-out (max |Δ| > 1e-3);
- gradients reach the encoder through the label pass (a non-zero gradient on `input_proj.weight`
  from the label MSE alone);
- report, descriptively, the label `f_layout` and `f_event` over 64 events at init.

A failure after one fix attempt is a STOP for R-2 (S9).

### 5.3 Runs

- **Reproduction first** (evaluation only, GPU): `ext_h12.REPRODUCTION` — `fix9000` 0.219258,
  `init9000` 0.160072 full-input vertex within 1e-3 m (`REPRODUCTION_TOLERANCE_M`). A failure after
  one fix attempt is S9 for R-2.
- One 20-step smoke (seed 9000) in its own root; finite metrics; it is archived and never resumed.
- Two runs, seeds 9000 and 9001, 10 000 steps each, fresh start (`resumed_from_step` 0), each in
  its own root under `<ext>/train/runs/` with a `variant.json` sidecar. Per-run wall cap 1.0 h.
- One evaluation job for both run directories: `ext_h12.evaluate_run` (full input and half pass),
  as in `ext_static.evaluate_run_dirs`, plus the label-pass `f_event` of each final checkpoint over
  the same 64 events as §5.2.

### 5.4 Readings (fixed now)

- **R2-a, event dependence (primary).** The §4 class of `target_posfree` on full-input `f_event`.
- **R2-b, recovery.** The §4 recovery rule.
- **R2-c, descriptive.** Per seed:
  - full-input and half-pass vertex and token ln q R²;
  - effective rank, module-gain R² and `f_layout`;
  - the label-pass `f_event`;
  - all beside the §4 reference values.

**Interpretation (fixed now):**
- *event-dependent* and *recovered*: the trainable target's access to PMT identity is the channel
  of the shortcut. A JEPA with a trainable target learns once that channel is closed. This
  explanation is stronger than E3 (frozen target) and points to a design rule: the target view
  must not carry inputs that identify the token independently of the event.
- *event-invariant* or *not recovered*: identity reaches the target through the remaining positional
  structure. Report it and design the next variant (token-order randomisation in the label pass); do
  not train it here.
- *intermediate* or *unclear*: report both seeds, with no third seed here.

Vertex may stay above the MAE reference even if R-2 recovers. The readings ask whether the collapse
is removed, not whether JEPA beats MAE.

---

## 6. Interaction with Part A

R-2 checkpoints are not added to Part A (that would change Part A's reviewed model lists). If R-2
reads *event-dependent*, a later runbook may add `target_posfree9000/9001` to the X1/X2 model list.
This runbook does not do that.

---

## 7. R-3 — positioning (done at design time; recorded here)

Searches of 7 Nov 2026. The novelty claim is narrowed accordingly.

| Work | What it shows | Consequence for us |
|---|---|---|
| Baade et al., *Generative Representation Learning for Scalable Latent Tokenization*, arXiv 2610.20533 | Discriminative and LeJEPA objectives can be satisfied by a feature subset (feature suppression); on watermarked data LeJEPA and DINOv3 collapse to the shared watermark; a generative (PMI) target avoids this | external evidence that SIGReg prevents dimensional collapse but not shortcut collapse; supports the expectation of R-1 |
| Chen, Luo and Li, *Intriguing Properties of Contrastive Losses*, NeurIPS 2021 | introduces feature suppression in contrastive learning | the older name for the phenomenon |
| *When Graph-JEPA Learns the Wrong Thing*, arXiv 2608.20516 | a Graph-JEPA with healthy linear probe and effective rank carries zero instance information; the degenerate solution is a global optimum of the coupled predictor / EMA-target objective | "effective rank misses the collapse" is **already published**; we must not claim it as new |
| Yang et al., *Denoising Vision Transformers*, ECCV 2024 | ViT features split into an input-dependent term and an input-independent, position-related term | precedent for the `f_layout` / `f_event` decomposition; cite it as such |

**What stays ours:**
- the cause isolated by interventions on the target (trainable, frozen, static, EMA, stop-gradient,
  and position-free in R-2);
- a detector setting where the shortcut is the detector itself — geometry shared by every event —
  not a planted watermark;
- consequences in physical units (vertex mm, energy resolution);
- the link between the objective and repairability (R-1, and Paper 3).

The execution agent cites these four works in the README's interpretation section, in one or two
sentences each, and makes no other literature claim.

---

## 8. Resources and stops

### 8.1 Caps

- **GPU (R-2 only):** L40S, ≤ 1.5 GPU-h for reproduction, smoke, two runs and evaluation together.
  - Projection: two runs about 0.45–0.6 h each (E3 frozen-target runs took 0.39 / 0.49 h; the
    trainable label pass adds backward work), smoke about 0.06 h, evaluation about 0.14 h, so about
    1.2 h in total.
  - Stop *before* a job that would cross the cap. Remaining items are `not_run (cap)`.
- **CPU:** ≤ 2 wall-hours (tests, manipulation check, R-1, report).
- **S1 launch rule:** the combined rule of front matter §F3 (129.7461 + 5.5 = 135.25 at `ac8e220`).

### 8.2 Hosts

| Part | Host | Device |
|---|---|---|
| Implementation, tests, dry run, manipulation check, R-1, report | Deepthought | CPU (`nice -n 10`, `OMP_NUM_THREADS=4`); deepthought2 CPU if Deepthought drops SSH sessions |
| Reproduction, smoke, runs, evaluation | deepthought2 | L40S, always through `scripts/jepa_run.sh` (no guard bypass) |
| Independent review | any host with a read-only clone | — |

### 8.3 Host rules

- At most two of the user's L40S jobs at once (the repair pilot's included), each ≤ 45 % of GPU
  memory.
- Guard HOLD or SHED: wait and retry; book failed attempts.
- `TMPDIR=/ceph/dwong/jepa/tmp/`; ≥ 30 000 free entries after the projected writes.
- Detach long jobs with `setsid`, as in the 3 Nov follow-up.
- Never overwrite or resume another root. Push to `JEPA` from the author's Mac only.

### 8.4 STOP conditions (the only ones)

- S1;
- S3 (a real specification change, including any change to a reading or threshold after outcomes
  are seen);
- S4 (no GitHub route after three documented attempts);
- S9 (the reproduction or the manipulation check still fails after one fix attempt);
- an import-closure failure at the end of a mode (§9.2): its outputs are quarantined, and the fix
  needs a re-review and a new acceptance record;
- a data-hash mismatch (R-2 only);
- an entries-preflight failure;
- a cap that would be crossed.

On a STOP: preserve logs and partial outputs, record it, and continue the independent part (R-1
does not depend on R-2, and R-2 does not depend on R-1).

---

## 9. Implementation, review, acceptance and execution

### 9.1 Implementation

- Commit the combined runbook unchanged (front matter, §F2).
- Add `src/jepa_noise/ext_shortcut.py`, `scripts/jepa_ext_shortcut.py` and
  `src/jepa_noise/tests/test_ext_shortcut.py`; these names are free at `ac8e220`.
- CLI modes: `dry-run`, `preflight`, `manipulation` (§5.2), `reproduce`, `smoke`,
  `train --seed {9000,9001}`, `evaluate`, `r1`, `report [--r1-only]`. The `train` mode accepts only
  these two seeds and 10 000 steps; the smoke only 20 steps.

### 9.2 Gate

`ext_shortcut.assert_authorised(mode)` is called from the library entry point. Any failure raises
`SystemExit(2)`. It requires:
- (a) the sha256 of the committed `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` to equal the
  acceptance record's, and the §0 author line present verbatim in it;
- (b) `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json` =
  `{runbook_sha256, impl_commit, review_path, review_verdict, lead, date}`, with
  `review_verdict == "PASS"`;
- (c) `git diff --quiet <impl_commit> -- <P>` and no untracked files under `<P>`, where `<P>` is:
  - `src/jepa_noise/{__init__,ext_shortcut,ext_h12,ext_readout,ext_followup,lucid_run,lucid_arms,lucid_bombs,lucid_views,point_encoder,bridge_run,train,crlb,jepa,backbone,linear,probes,metrics,sigreg}.py`;
  - `src/noise_module_lucid/` and `src/jepa_noise/configs/`;
  - `scripts/{__init__,jepa_ext_shortcut}.py`;
  - this runbook.

**Import closure.** Every mode re-checks at its end that every repository module in `sys.modules`
lies inside `<P>` or `ext_followup.FORCED_IMPORT_EXCEPTIONS`. In the dry run, a module outside is
fixed before review. In a mode, it is the closure STOP of §8.4.

The acceptance record goes in a later commit, so HEAD equality is never required.

### 9.3 Tests

- **Variant:** `posfree_features` zeroes exactly columns 0–5 of a clone and never mutates its
  input; the rebound `_encode_views` equals the original `lucid_arms` body when the zeroing is
  disabled (bit-identical on a synthetic batch); the label is not detached.
- **Identity-blindness:** the §5.2 permutation property on a synthetic geometry.
- **Build patch:** restores `lucid_run.build_arm` in a `finally`, also after an exception; applies
  only to `sigreg_jepa`.
- **Training guard:** `train` refuses any seed other than 9000/9001, any step count other than
  10 000, and an existing run root; `smoke` refuses anything but 20 steps.
- **R-1:** the family table rejects an unknown label; the S definition on synthetic embeddings;
  the reading rule (supports / contradicts / mixed) on synthetic values; `not_run` when the pilot
  outputs are missing; a cache whose sha256 mismatches is refused.
- **R-2 readings:** the §4 class and recovery rules on synthetic values, including a seed
  disagreement.
- **Forbidden paths:** sentinel patches show that no §2.5 path is reached by any mode's setup.
- **Gate:** the negative cases (a), (b), (c).

The focused tests plus the full `src/jepa_noise/tests` suite run once on the final SHA, deselecting
only the three known `test_guard` launcher tests.

### 9.4 Independent review

A fresh-context reviewer reads this runbook and the diff. The checklist is blocking, PASS or BLOCK
per item, with file:line evidence:
- (i) **design:** R-2 changes only the label-pass input (§5.1), and the mechanism argument of §5.1
  matches the code (columns, target-set sampling, Hilbert windows);
- (ii) **design:** the R-1 families and reading are fixed and cannot be tuned after the pilot
  outcomes;
- (iii) the variant installation, the manipulation check and the training guard;
- (iv) the reproduction, the evaluation path and the readings;
- (v) the gate, the import closure, the forbidden paths and the write scope;
- (vi) the cost projection against the cap and the S1 rule.

Blockers are fixed and re-reviewed. Then the lead writes the acceptance record.

### 9.5 Execution

- Use the accepted SHA and the CLI-generated commands only. Recheck the guard, the ledger, the jobs
  and the data.
- **Order:** preflight → manipulation → reproduce → smoke → train 9000 → train 9001 → evaluate →
  report. R-1 runs whenever the pilot outputs exist.
- GPU steps go through the launcher:
  `JEPA_JOB_MODULE=scripts.jepa_ext_shortcut JEPA_JOB_KIND=<training|evaluation> scripts/jepa_run.sh <run_id> --host deepthought2 --duration-minutes <M> -- <mode args>`.

### 9.6 Deliverables

In `results/jepa/20261107_shortcut/`:
- `plan.json`, `manipulation.json`, `reproduction.json`, `runs.json`, `readings.json` (R2-a,
  R2-b, R2-c, and R1 or `not_run`);
- `README.md`: the readings, the R-1 table, the reference values, the R-3 citations, and per-job
  hours;
- `LARGE_ARRAYS.json` for any file > 5 MB on `/ceph`;
- a `STATE.md` section, a ledger line and the `DECISIONS.md` lines (both §0 quotes).

---

## 10. Out of scope

- Any training other than §5.3; any third seed; any other variant (token-order randomisation, slow
  target, relative-offset queries), including X3-T;
- data generation of any kind;
- changes to the repair pilot or to any registered prediction;
- the controlled-watermark stress test (needs generation and training; a later runbook).
