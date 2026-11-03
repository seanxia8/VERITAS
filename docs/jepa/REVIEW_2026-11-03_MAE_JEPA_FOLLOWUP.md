# Independent implementation review — MAE / JEPA follow-up extension

**Verdict: BLOCK.** The implementation is substantial and much of it is faithful, but the
§6.2(d) gate does not cover the whole `<P>` set (demonstrated), the §3 A1 "recomputed item"
reproduction rule is silently skipped for every item whose committed vertex lives in the D2 /
P-5 / M-f artifacts (demonstrated), the T-1 launch does not consult the A1 s = 0.30 gate, the
`--include-t1` readout path is a no-op, and the view-design inventory rows can never be
`available`. These are listed below as B1–B5 and must be fixed and re-reviewed before the lead
writes the acceptance record.

| Item | Subject | Verdict | Main evidence |
|---|---|---|---|
| (i) | pairing and truth leakage | **PASS** (N1) | `ext_followup.py:1120-1223, 2436-2464`; host pairing spot-check |
| (ii) | per-checkpoint inventory and evaluation configuration | **BLOCK** (B5) | `ext_followup.py:782-977`; view mapping at `:946-970` |
| (iii) | A-4 masked-pass reference and invariance test | **PASS** (N7) | `ext_followup.py:3085-3242, 3245-3374`; tests `test_ext_followup.py:463-531` |
| (iv) | A1 reproduction on s = 0.05 and s = 0.30 | **BLOCK** (B2, B3) | `ext_followup.py:1344-1419` correct; `:1538-1584, 1836-1841` incomplete |
| (v) | gate (a)–(e) | **BLOCK** (B1) | `ext_followup.py:337-469`, pathspec at `:391`; demo below |
| (vi) | forbidden call paths | **PASS** (N6) | code audit; only test is weak |
| (vii) | cost projection against the caps | **PASS** (S3, N4) | `ext_followup.py:1048-1117, 1264-1328`; A0 measured 3.9949 h |
| (viii) | A-1b collector / 0.1043 reproduction / read-only `dev_train` reader | **PASS** | `ext_followup.py:1872-1948, 2102-2200, 2203-2300`; independent 0.10429278172645076 |

Reviewed object: **`3191e690f0248ee614ee18d12dd17b41aca088dd`** on branch `JEPA`, base
`f5b7434`. The clone `/ceph/dwong/jepa/code/followup_20261103` is clean at that SHA. The three
implementation files are byte-identical between the clone and the local copy
(`sha256 9eace6ad…` `ext_followup.py`, `88002bbb…` `scripts/jepa_ext_followup.py`,
`b042ec37…` `test_ext_followup.py`). `/tmp/review_input/diff.patch`
(`sha256 95357c657288ffb5039a1b6c0f7e2d1e111f5444c055c264c9015885fb483b59`) is exactly
`git diff f5b7434 3191e69` (verified on deepthought2). The binding runbook
`docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md` has sha256
`063e908774ffe29b3d59ef40ab54ef5d02ff462b6aa1e49eef57c61c49e7af24` (local binding copy and
the committed file agree). The committed `DECISIONS.md` carries both authority lines.

This review was read-only except for temporary fixtures under
`/ceph/dwong/jepa/tmp/pytest_review_{gate,guard}/`, the temporary local copy of this document,
and the review file itself. Nothing was committed or pushed, and the implementation, registered
files, runbook, DECISIONS/STATE and ledger were not modified (the review file is the only new
file in the clone).

---

## Blocking findings

### B1 — §6.2(d) gate only diffs/checks `P_FILES`; the `<P>` directories are not protected (item v)

`ext_followup.py:117` defines `P_DIRS = ("src/noise_module_lucid/", "src/jepa_noise/configs/")`
and `in_p()` (`:280-285`) honours them for the import closure, but the git tree check uses

```python
paths = [name for name in P_FILES]          # ext_followup.py:391
```

for both `git diff --quiet <impl_commit> -- <paths>` (`:393`) and
`git ls-files --others --exclude-standard -- <paths>` (`:398-399`). `p_paths()` (`:274-277`)
is also `P_FILES`-only. Therefore modifications and untracked files under
`src/noise_module_lucid/` and `src/jepa_noise/configs/` — the executed PMT/readout package and
the pinned run/generator/arms configs — are invisible to gate (d).

Demonstrated on deepthought2 with a throw-away git repository at
`/ceph/dwong/jepa/tmp/pytest_review_gate/repo` (runbook + DECISIONS + acceptance record; all
checks otherwise passing):

* pristine tree: `checks["d"]["pass"] = True`, untracked `[]`;
* modify `src/jepa_noise/ext_followup.py` (in `P_FILES`): `checks["d"]["pass"] = False`
  (correct);
* modify the tracked `src/jepa_noise/configs/lucid_arms_v0.json`, modify the tracked
  `src/noise_module_lucid/readout.py`, add untracked `src/noise_module_lucid/extra.py`:
  **`checks["d"]["pass"] = True`**, `diff_clean: True`, `untracked: []`, while
  `git status --porcelain` reports all three changes.

This is a false-PASS of the gate that is supposed to pin the reviewed tree. Fix: include
`P_DIRS` in both pathspec arguments (and in `p_paths`) and add a negative test that touches a
file under each directory.

### B2 — §3 A1 "recomputed items must reproduce their own committed full-input vertex" is silently skipped for most recomputed A-1/A-3/A-4 items (item iv)

Runbook §3 A1 (`:239-241`): "**Recomputed items (A-1, A-3, A-4).** Before any of its metrics is
used, each recomputed item must also reproduce its own committed full-input vertex within
1e-3 m. A mismatch marks the row `unavailable`."

`committed_vertex_map()` (`ext_followup.py:1538-1572`) reads only
`20261101_readout/matrix.json`, `20261101_readout/nonlinear.json`, `20261101_h12/runs.json`,
`20261102_static/runs.json`, `20261102_arch/runs.json` and `20261102_scale/runs.json`. It does
**not** read the three committed sources that hold the vertices of the recomputed A-1/A-3/A-4
items:

* `results/jepa/20261029_d2_partial/METRICS.md` — `strong_drift` 9000 vertex **0.233209**,
  gain R² 0.097482 (A-1 evaluates `strong_drift` at s = 0.30, §2.1 `:195`);
* `results/jepa/20261030_dejitter/readings.json` P-5 — the six SIGReg de-jitter vertices
  (`within_trigger/s9000` 0.224385…, `within_median/s9000` 0.222653…, `across_median/s9000`
  0.347506…; both seeds present);
* `results/jepa/20261031_mae_dejitter/readings.json` M-f — the eight MAE de-jitter vertices
  (`within_trigger/s9000` 0.1049156, `within_median/s9000` 0.0852425, `across_trigger/s9000`
  0.1209577, `across_median/s9000` 0.0970344; both seeds present).

`check_item_reproduction()` (`:1575-1584`) returns `{"checked": False, "pass": None}` when the
label is absent from the map, and the call sites only reject when
`reproduction.get("checked") and not reproduction.get("pass")` — `readout_a1` `:1836-1841`,
`readout_a3` `:3025, :3037-3043, :3068-3075`, `readout_a4` `:3252, :3277-3281, :3324-3330`.
With `checked: False` the recomputed metrics are kept.

Demonstrated on the clone:

```
committed_vertex_map has strong_drift9000 : False
committed_vertex_map has sigreg_within_trigger9000 : False
committed_vertex_map has sigreg_within_median9000 : False
committed_vertex_map has mae_within_trigger9000 : False
committed_vertex_map has mae_across_median9001 : False
check_item_reproduction("strong_drift9000", 0.999, map)
  -> {'checked': False, 'pass': None, 'expected_m': None, 'got_m': 0.999}
check_item_reproduction("mae_within_trigger9000", 0.999, map)
  -> {'checked': False, 'pass': None, 'expected_m': None, 'got_m': 0.999}
```

Consequence: A-1/A-3/A-4 can report results for 2 `strong_drift`, 6 SIGReg de-jitter and 8 MAE
de-jitter items without the reproduction the runbook promises; an evaluation-configuration
regression (cell, wrapper, zero-geometry, split) would not be caught. Fix: add the D2 table,
P-5 and M-f to the map (or hard-code the values from the runbook), and record an explicit
`no committed target` status for the items of spec defect S1 instead of silently passing.

### B3 — the T-1 launch does not enforce "both A1 s = 0.30 rows pass" (item iv)

Runbook §5 (`:555`): "T-1 launches only after both A1 s = 0.30 rows pass." The implementation
never reads `stageA/A1/reproduction.json`:

* `mode_train_t1` (`scripts/jepa_ext_followup.py:302-339`) optionally checks `--plan` for the
  A0 entries/overall pass (`:310-315`) and the live entries preflight (`:316-322`), then calls
  `train_t1`;
* `train_t1`/`t1_prepare_launch` (`ext_followup.py:3412-3487`) enforce the (a)–(e) gate, whose
  (e) only whitelists `(mae_mse, {s030_within_trigger, s030_across_trigger}, {9000, 9001},
  10000)` plus the one smoke (`:411-431, :597-600`).

So a launch can be authorised after an A1 s = 0.30 failure (S9 is operator-procedural only).
Fix: have the launch path load `<ext>/stageA/A1/reproduction.json` and require
`s030_pass is True` (and `s005_pass` for Stage A readouts / report).

### B4 — `--include-t1` is a no-op for the A-1/A-2 readouts, so the §4 T-1 readouts cannot be produced through the CLI (item iv/report)

`mode_readout` merges the completed `t1_*` inventory rows when `--include-t1` is given
(`scripts/jepa_ext_followup.py:277-278`), but then passes `labels=None` to the body
(`:281-289`). `readout_a1` iterates `A1_COMPUTE_LABELS + A1_MLP_FILL_LABELS`
(`ext_followup.py:1816-1817`) and `readout_a2` iterates `A2_LABELS` (`:2544`); neither list
contains `t1_s030_*` labels, and neither body inspects the inventory's extra rows when
`labels is None`. Consequently the four T-1 checkpoints are never evaluated, and
`t1_readings` (G-1/G-2, `:3605-3639`) consumes only A-2 model rows whose label starts `t1_`, so
it will always be empty. This contradicts §4 "Readouts: the A-2 readouts on both environments
…; the A-1 metrics" and §6.5 "Use the accepted SHA and the CLI-generated commands only". Fix:
pass `labels=sorted(t1_inventory_rows(...))` for A-1/A-2 (and book to the T-1 cap) or add the
T-1 labels to the iterated sets when present.

### B5 — view-design inventory rows can never be `available` (item ii)

`build_inventory` (`ext_followup.py:946-970`) builds `wanted = {sha: str(key)}` from
`results/jepa/20261103_view/runs.json`, while the view writer emits keys
`f"{split}__{seed}"` (`scripts/jepa_ext_view.py:342`; README line 452: "keyed `<split>__<seed>`").
It then looks rows up as `view_paths.get(f"view_{split}{seed}")`
(`:958-969`, e.g. `view_within9000`), which never exists, and it never records the sha on the
row.

Demonstrated with a temporary root containing `runs.json` keyed `within__9000` /
`across__9001` and `find_final_pt` forced to match both shas:

```
view inventory row view_within9000 -> status unavailable source None sha None
view inventory row view_across9001 -> status unavailable source None sha None
```

Because the separately-authorised view design is expected to commit `runs.json` before A-1,
A0's inventory will misreport all 16 view labels as `unavailable` (and §2.1 requires one row
per label accepted only on a sha match). Fix: key `view_paths` by `view_{split}{seed}` and set
`sha256`.

---

## Per-item evidence

### (i) Pairing and truth leakage — PASS

* A0 pairing: `pairing_check` (`ext_followup.py:1120-1166`) joins the s = 0.05 and s = 0.30
  probe-train/probe-test sets on `(run_id, event_id)` and compares
  `vertex_m, yield_photons, particle, domain, contained, deposited_MeV, direction_m,
  energy_MeV` via `_join_truth` (`:1192-1223`, continuous columns with rtol 1e-9 / atol 1e-12,
  identity columns exactly, and a `non_bitwise` record). The dry run reports
  `setup.pairing.pass = true` (`/ceph/dwong/jepa/tmp/followup_verify/dryrun.json`).
* A-2 pairing: `pair_events` (`:2436-2464`) requires unique `(run_id, event_id)` keys on both
  sides, identical key sets and identical truth, and raises `ValueError` otherwise;
  `readout_a2` marks the model unavailable on a pairing failure (`:2562-2567`). The §6.3 test
  `test_pair_events_on_run_and_event_id_with_a_mismatch`
  (`test_ext_followup.py:386-408`) covers valid pairing, a deliberately mismatched key set,
  different truth and duplicate keys — genuine.
* Feasibility on real data (read-only spot-check, 2 probe-test runs): s = 0.05 vs s = 0.30
  `event_ids` and `vertex_m` are bit-identical; `yield_photons` differs only by ≤1.4e-9
  (allclose at rtol 1e-9), as the code's own comment anticipates, so the exact-vertex
  `pair_events` will not spuriously fail. The same holds for the first dev_train run.
* Truth is never an input feature: the label collector (A-1b) reads only tokenised hits, the
  A-3 loader aligns `events.npz` truth by `event_id` (`:2607-2633, :2636-2692`) and the A-4
  leak (prompt-window cut) is stated in the payload (`:3256-3257`) and runbook (`:402-404`).
* Nonblocking N1: the dev_train s = 0.05 / s = 0.30 pair (used by T-1 training and A-1b
  marginals) is not covered by `pairing_check` (spec defect S4).

### (ii) Per-checkpoint inventory and evaluation configuration — BLOCK

Positive evidence:

* one row per label with arm/split/seed/beta/source/absolute path/sha256/data_setting/eval
  flags (`ext_followup.py:708-722, 782-977`); rows are accepted only after
  `lr.sha256_file(...) == expected` (`:725-737`), and the dry run on the host yields
  54 labels / 46 available / 0 unavailable / 8 `pending (view design)`;
* matrix/SIGReg-variant/anc100/de-jitter/target-control/scale/arch sources follow §2.1
  (`:803-944`) — the host dry run confirms every non-view family is available;
* evaluation configuration: `item_eval_context` installs `ext_arch.build_arm_patch`
  (`:1446-1455`; the patch supplies the encoder signature, `ext_arch.py:128-155`), the median
  wrapper (`ext_dejitter.py:60-76`), E2 zero-geometry (`:1519`), anchor build through
  `pregrid.build_loaded` after the dark-bank hash check (`:1465-1474`,
  `jepa_pregrid_checks.py:247-270`), s = 0.30 rows carry `data_setting = 0.30` and are
  evaluated on the s = 0.30 probe sets (`:1373-1376`), and view rows are pinned to
  `split="within"` (`:958-970`);
* `item_cell` builds the s = 0.30 cell required by §2.1 `:197-198` (`:1423-1433`).

Blocking defect: B5 (view family). Nonblocking defects:

* N2 — in the non-anchor path the s = 0.30 cell is computed and discarded:
  `cell = item_cell(row, config)` (`:1463`) is not passed to `er.item_arm` (`:1475-1478`),
  which rebuilds the arm with its own hard-coded s = 0.05 cell (`ext_readout.py:688-700`).
  This contradicts §2.1 `:197-198`; it has no numerical effect today because the only arm
  that reads `settings.s` is `mae_chi2` (`lucid_arms.py:520`) and no evaluated Stage A/T-1 arm
  is `mae_chi2`, but the requirement is not implemented.
* N3 — inventory rows have no "data key" field (only `data_setting`), although §2.1 `:156-158`
  asks for it; paths/shas are present.
* N10 (see below) — view rows also would not record their sha.

### (iii) A-4 masked-pass reference and invariance test — PASS

* `context_time_shift` (`ext_followup.py:3085-3125`) implements both references: median of the
  visible hit times (0 if none), and the imported NHits rule
  (`lucid_bombs.pmt_time_rule()[0]`, `noise_module_lucid/readout.py:352-370`) over the visible
  times, returning 0 shift and counting `n_ctx_no_trigger` when there is no visible hit or no
  cluster. The stored token time is `digit − trigger`
  (`lucid_bombs.py:574-575, 2304`), and `nhits` is translation-equivariant, so the all-token
  branch reproduces the committed (trigger-subtracted or median-wrapped) convention and the
  context-only branch yields `digit − T_ctx`; this matches the runbook `:372-404`.
* `apply_time_shift` (`:3128-3144`) reduces **every hit** token's time feature and leaves
  unhit tokens unchanged; `masked_reference_readout` (`:3147-3242`) encodes once per reference
  with `hidden_mask=batch.hidden` (so hidden positions are overwritten by `mask_embed`,
  `point_encoder.py:308-309`), forms `context_only`/`propagated` from the same tokens, and
  records `n_ctx_no_trigger`/`n_ctx_no_hit`; `readout_a4` uses `ViewMasker(n_min=0)` through
  `_probe_batch` (`lucid_run.py:885-892`) and the registered within masker.
* Readings `a4_readings` (`:3333-3374`) implement the fixed three-valued rule exactly
  (material `|Δ| > max(0.01, |v9000−v9001|)` in both seeds; immaterial `|Δ| ≤ 0.01` in both;
  else unclear; Δ token charge R² descriptive), and `half_pass` is explicitly excluded from
  the reading.
* §6.3 tests `test_a4_context_reference_is_invariant_to_hidden_time_perturbations`,
  `…zero_visible_hit_and_none_trigger_rules`, `…median_differs_from_all_token_when_hidden_times_move`
  (`test_ext_followup.py:463-531`) exercise real encode calls and cover the masked pass,
  unhit-token invariance, the zero-visible-hit rule and the None-trigger rule.
* N7: the named §6.3 invariance test is at primitive level; it never calls
  `masked_reference_readout`. The end-to-end "with fixed masks" property is argued from the
  primitives, not tested; add an end-to-end test.

### (iv) A1 reproduction on s = 0.05 and s = 0.30 — BLOCK

Positive evidence: `reproduction()` (`ext_followup.py:1344-1419`) evaluates the five s = 0.05
rows on the s = 0.05 probe sets and the two s = 0.30 rows on the s = 0.30 sets, comparing the
committed values (`A1_S005 :170-176`, `A1_S030 :177-180`) at `|Δ| ≤ 1e-3 m` and
`|Δ| ≤ 1e-3` absolute R² (`A1_TOLERANCE :181`), records `s005_pass`/`s030_pass` and the stop
scopes (`:1409-1419`), and the CLI returns exit 2 on a failure
(`scripts/jepa_ext_followup.py:239-256`). The s = 0.05 `sigreg_within_median9000` row runs
inside the median wrapper (`:1446-1455`). This part matches the runbook `:219-241`.

Blocking defects: B2 (the A1 "recomputed items" bullet is not implemented for the D2/P-5/M-f
items) and B3 (the launch dependency is not enforced). Nonblocking: the s = 0.30 cell is
discarded (N2); the `stops` list is built with a convoluted double comprehension (`:1415-1418`)
but is correct; `reproduction()` opens the s = 0.30 probe sets even when only the s = 0.05 rows
are needed (`:1350-1351`), so a missing s = 0.30 pin would STOP the s = 0.05 gate instead of
only the s = 0.30 rows (conservative but broader than the stated scope).

### (v) gate (a)–(e) — BLOCK

* (a) `ext_followup.py:346-355`: the committed runbook sha must equal the acceptance record's
  `runbook_sha256`; before the record exists it must equal the pinned `063e9087…`.
* (b) `:357-367`: for `train-t1`, the exact line `我同意T-1训练` must be in §0 read through
  `section_zero` (`:249-257`) **and** in `DECISIONS.md`.
* (c) `:369-384`: `docs/jepa/ACCEPTANCE_2026-11-03_MAE_JEPA_FOLLOWUP.json` must contain
  `{runbook_sha256, impl_commit, review_path, review_verdict, lead, date}` and
  `review_verdict == "PASS"`.
* (d) `:386-409`: `git diff --quiet <impl_commit> -- <P>` and an empty
  `git ls-files --others --exclude-standard -- <P>`; the pathspec bug is B1.
* (e) `:411-431`: only `{mae_mse} × {s030_within_trigger, s030_across_trigger} × {9000, 9001}
  × {10000}` plus exactly the one 20-step smoke; the out root must be named `train`/`smoke`;
  an existing run directory is refused (`refuse_existing_run_dir :600-606`, used first in
  `t1_prepare_launch :3420`).
* `assert_authorised` raises `SystemExit(2)` on any problem (`:465-468`) and is called from the
  library entry points (`build_plan :1268`, `reproduction :1348`, `readout_a1 :1793`,
  `readout_a1b :2207`, `readout_a2 :2535`, `readout_a3 :2997`, `readout_a4 :3249`,
  `t1_prepare_launch :3416`, `assemble_report :3651`), not only from the CLI. The negative
  tests a–e are present and genuine (`test_ext_followup.py:152-229`).
* Spec defect S2: the executed-import closure cannot be satisfied by `<P>` alone; the
  implementation's forced-import exception (`:118-129`, dry-run: 27 forced files, 71 checked
  modules, 0 violations) is a deliberate deviation from the literal runbook text and must be
  resolved in the runbook/acceptance record.
* Spec defect S5 / nonblocking N8: the dry-run setup path is only a representative subset of
  the six CLI modes (see below), so the closure is asserted over fewer executed modules than
  the runbook describes.
* N3: the (e) out-root check is basename-only (`:417-418`); the CLI pins
  `<ext>/smoke|<ext>/train` (`scripts/jepa_ext_followup.py:307-309`), but the library gate
  would accept any directory named `smoke`/`train`.

### (vi) Forbidden call paths — PASS

Code audit (no call to the forbidden functions in any extension path):

* `lr.ensure_dataset` is reached only at `ext_followup.py:538` (probe sets) and `:555` (train
  cohort), each after `require_pinned` (`:531`, `:554`), and through `pregrid.datasets_for` in
  `t1_prepare_launch`, after `require_pinned` for all three s = 0.30 keys (`:3421-3425`);
  the s = 0.30 path passes `registered=False` (`:532, :558`; `jepa_pregrid_checks.py:220-226`)
  and the opened `dataset.key` is asserted against the pinned key (`:543-546, :561-563,
  :3426-3431`);
* `ext_h12.load_probes`, `ext_readout.load_probes`, `scripts.jepa_ext_readout.load_probes`,
  `preflight_data(regenerate=True)`, `_regenerate_probe_sets` and
  `scripts.jepa_ext_dejitter.mode_report` are never called (grep: only comments/tests);
* `lr.ensure_dark_bank` is reached only through `pregrid.build_loaded` for anchor arms, after
  `require_pinned(DARK_BANK_KEY)` in `build_item_arm` (`:1468-1472`); `require_dark_bank`
  itself checks the pin before `ensure_dark_bank` (`:566-577`);
* `lr.run_one` is called once (`:3460`) after `refuse_existing_run_dir` (`:3420`) and the
  gate (e); the T-1 host check reads `manifest.jsonl` only (`:3507-3571`).

Nonblocking N6: the §6.3 test `test_forbidden_call_paths_are_not_used`
(`test_ext_followup.py:342-377`) patches the forbidden functions to raise but only exercises
`read_dev_marginals` on an empty root (it stops in `require_pinned` before any data read) and
`assemble_report` (which touches none of them). It does not exercise the real opener, readout
or T-1 paths, so it would not catch a regression — the audit above, not the test, is the
evidence.

### (vii) Cost projection against the caps — PASS

* A0 projection: `stage_a_projection` (`ext_followup.py:1077-1095`) implements Σ items × 126 s
  × pass factor × 1.3 exactly; the item table (`:1048-1074`) has
  5 + 2 A1, 23 A-1, 16 A-1 MLP-fill, 13 A-1b ×0.6, 4 A-2 ×2, 11 A-3 ×2 and 8 A-4 ×0.5
  = 87.8 item-equivalents, i.e. **3.9949 L40S h** ≤ 5.5 h (host dry run: 3.9949, pass true);
  `build_plan` adds the projection STOP before any GPU item (`:1312-1314`), and the CLI exits 2
  on a failed plan (`scripts/jepa_ext_followup.py:214-235`);
* CPU projection 7.0 h ≤ 8 h (`:1098-1102`); S1 = 116.1507 (ledger, verified at
  `docs/jepa/RESOURCE_LEDGER.json` `v4_caps.E2_lucid_l40s_deepthought2.charged_gpu_hours`) + 8.0
  + 9.0 = 133.1507 ≤ 160 (`:1105-1117`); entries preflight is 5884 entries against the 30 000
  floor (`:638-660`).
* T-1: the fixed launch table is 4 runs + 1 smoke; the stated 2.7 h projection is below the
  3.5 h cap, but N4 below applies.
* Spec defect S3: the A0 pass-factor sentence is internally inconsistent (1 for a masked
  pass vs 0.5 for an A-4 masked-only reference); the implementation follows the 0.5 rule while
  `readout_a4` performs one full-input reproduction pass plus two masked reference passes per
  checkpoint (`:3275-3289, :3147-3242`). Even a conservative re-count keeps Stage A within
  the 5.5 h cap (≈4.5–4.9 h), but the projection rule and the implemented work disagree.
  §5's "about 12 items" for A-1b also disagrees with the 13 A-1b items in §3 (the code counts
  13; the total is still inside the cap).
* N4: `T1_CAP_H` (`:204`) and `T1_PROJECTED_GPU_H` (`:1045`) are dead constants — nothing
  records or checks T-1 GPU hours at launch or in the report. The fixed launch table bounds
  T-1 at ≈2.7 h projected, but the cap is procedural only.
* N (minor): there is no cumulative per-item accounting during Stage A execution, so the
  "stop before an item that would cross the cap" rule (`runbook :506-508`) is not implemented
  in code; the fixed A0 projection is the control.

### (viii) A-1b: collector reproduction, 0.1043 reproduction, read-only dev_train reader — PASS

* Collector: `collect_label_rows` (`ext_followup.py:2102-2178`) re-implements the committed
  E0/E1 core (`ext_h12.py:784-880`): `lr._probe_batch(split, mask_seed=views.mask_seed=0)`,
  `encode(features)` for layout, `encode(features, hidden_mask=~batch.target)` for labels,
  `hidden & hit`, 32 tokens per event with `er.SAMPLE_SEED`, event offsets, positions
  `geometry_features[:, :3]`; it does not touch `ext_h12._collect_label_rows`/`load_probes`.
  `verify_label_collector` (`:2181-2200`) compares `y/pmts/events` exactly against the committed
  cache through `_read_label_cache` (`ext_h12.py:882-905`; cache meta
  `fix_sha256 = 2d912a2d…`, `max_label_events = 2000`, verified on the host) and returns early
  before any other item is used (`readout_a1b :2221-2243`). The §6.3 test
  `test_label_collector_reproduces_the_committed_fix9000_rows`
  (`test_ext_followup.py:648-668`) ran and passed on deepthought2 (focused log: 39 passed).
* 0.1043 reproduction: `readout_a1b` computes the fix9000 C-group ridge through the follow-up's
  own probe code (`a1b_probes :2048-2075`, `_pmt_probe_split :2009-2016` — identical logic to
  `ext_h12._pmt_split :590-597`; `_probe_one_group :2019-2045` — identical to
  `ext_h12.code_origin_probes` linear/MLP/kNN calls) against
  `A1B_REPRODUCTION_R2 = 0.10429278172645076` with 1e-3 tolerance
  (`:184-186, :2244-2262`). I independently reproduced this on deepthought2 using the committed
  label cache and a fresh read-only probe-test C pass:
  `a1b_probes C linear r2 = 0.10429278172645076`, absolute diff **0.0**, pass true
  (M=20, C=120, All=140 features). The committed reference
  `results/jepa/20261101_h12/e0e1_code_origin.json` gives `all.linear = 0.10429278172645076`,
  `all.mlp = 0.4335279`, matching §3 A-1b's "0.1043/0.434".
* Read-only `dev_train` reader: `read_dev_marginals` (`:1872-1948`) pins the manifest first
  (`require_pinned`, `:1881-1882`), then reads `manifest.json` → `runs/<dir>/hits.npz`
  (`lb.read_sparse_hits`) and `events.npz` `domain` directly; it never calls
  `ensure_dataset`/`load_probes`. The test `test_marginal_reader_never_reaches_ensure_dataset`
  (`test_ext_followup.py:559-578`) monkey-patches `ensure_dataset` and asserts zero calls and a
  pre-read pin refusal — genuine and passed; `test_scale300k_marginals_use_the_pinned_300k_training_set`
  (`:691-700`) confirms the 300k key/pin. The `domain`-key A0 assertion is
  `domain_key_assertion` (`:1226-1261`) and is exercised on real data by
  `test_scale300k_manifest_hash_and_domain_keys` (`:624-644`).
* Remaining A-1b requirements are implemented: 13 labels (12 collapsed + init9000 reference,
  `:1019-1023`), M = 4 groups × 5 statistics = 20 features (`:1865-1867`, verified M=20),
  C = the committed six groups / 120 features (`ext_h12.py:299-331`; host value
  `n_features C=120`), split by PMT 80/20 seed 0 over present PMTs, MLP early stopping on 10 %
  of the training PMTs, kNN-5, both-seed agreement and the ordered class rules
  (`_a1b_readings :2303-2333`, `a1b_class :2078-2088`), the scale300k M on its own 300k set
  (`:2274-2277`), and the PMT count recorded per item (`:2287`).
* Nonblocking: N11 (the C reproduction unnecessarily reads the full 100 k-event dev_train
  marginals even though C does not use M; `readout_a1b_committed` ignores its `inventory`
  argument and uses module-level `ROOT`, `:2336-2347`). Spec defect S1 means the
  "own committed vertex" rule cannot be met for `init9000`'s sibling `ijepa/lambda/stopgrad/
  anc100` items as written.

---

## §6.3 test audit

All 39 focused tests passed on deepthought2 at this SHA (`FOCUSED_RC=0`, 221 s). The full suite
is discussed under Evidence caveats. Every bullet of §6.3 has at least one test; genuineness:

| §6.3 bullet | Test(s) | Assessment |
|---|---|---|
| run-grouped split disjointness | `:411-419` | genuine (each run in one fold; train/test run-disjoint) |
| A-2 pairing on `(run_id, event_id)`, mismatched case | `:386-408` | genuine (mismatched keys, different truth, duplicates) |
| A-4 invariance (masked pass, unhit, zero-visible-hit, None-trigger) | `:463-531` | genuine at primitive level; **not end-to-end through `masked_reference_readout`** (N7) |
| every §2 forbidden call path blocked | `:342-377` | **largely vacuous** (N6); only `read_dev_marginals` on an empty root and `assemble_report` are executed |
| A-1b marginal reader opens no `ensure_dataset` | `:559-578` | genuine (records zero calls; pre-read pin refusal) |
| within target set = odd parity | `:581-591` | genuine (uses the real 1 995-PMT layout, `lv.within_mask` + `er.within_target_mask`) |
| across partition: every module target `n_modules // 2` times | `:594-605` | genuine (exact enumeration over all offsets) |
| `across_mask` offsets χ² uniformity, 20 000 keyed draws | `:608-620` | genuine (105 bins, every bin populated, p > 1e-3) |
| scale300k manifest hash and A0 `domain`-key assertion | `:624-644`, `:1054-1071` | genuine (real data run loops every dev_train run; synthetic negative also) |
| general label collector reproduces committed fix9000 rows | `:648-668` | genuine; ran and passed |
| T-1 cell and keyword-argument unit test | `:704-721`, `:723-779` | genuine (compares against the actual `jepa_ext_deitter.mode_train` call, not a copy) |
| anc100 build path, dark bank before `ensure_dark_bank` | `:850-870`, `:873-894` | ordering genuine for `build_item_arm` (`require_pinned` before `build_loaded`); `require_dark_bank` itself is exercised but is not on the real anchor path (N5); `build_loaded` is mocked in the first test |
| gate negative cases (a, b, c, d, e) | `:152-229` | genuine; (d) does not cover `P_DIRS`, consistent with B1 |
| hash mismatch | `:266-285` | genuine (fails before any opener) |
| no overwrite | `:934-948` | genuine for the helpers; does not call `train_t1` through `run_one` |

Also present and genuine: `test_import_closure_flags_a_module_outside_p` (`:232-262`),
`test_a0_projection_is_inside_the_stage_a_cap` (`:898-910`),
`test_inventory_is_built_read_only_and_marks_unmatched_rows` (`:913-930`),
`test_t1_host_manifest_check` (`:807-846`), `test_cli_modes_and_defaults` (`:952-971`),
the A-1/A-1b/A-3 helper tests (`:973-1041`) and `test_entries_preflight…` (`:1044-1052`).

---

## Spec defects (runbook infeasible or inconsistent exactly as written)

### S1 — §3 A1 "each recomputed item must reproduce its own committed full-input vertex" is unsatisfiable for 7 A-1 items

`ijepa`, `lambda` and `stopgrad` (2 seeds each) are the s = 0.05 predictability checkpoints
(`results/jepa/20261029_predictability/probes/<label>900*.json` point at
`ext_dev_s005_within_*` runs), and `anc100` is also s = 0.05. A-1 evaluates them "under each
item's own evaluation configuration" (§2.1 moves only strong-drift items to s = 0.30), but the
only committed vertices for ijepa/lambda/stopgrad are the **s = 0.30 D2** values in
`results/jepa/20261029_d2_partial/METRICS.md`; the predictability probe JSONs carry only
linear-probe readings, and no committed full-input vertex exists for `anc100`. Under the literal
rule these items can never be used. Required runbook action: either define the reproduction
target (e.g. "where a committed full-input vertex exists") or exempt these items explicitly.

### S2 — §6.2(d) executed-import closure cannot be satisfied by a `<P>`-only import set

`<P>` (`runbook :582-587`) lists importers but not the tracked dependencies that a `<P>` module
force-imports at import time: `src/jepa_noise/backbone.py:62` imports
`reconstruction_model.model` (reached through `bridge_run`/`jepa`), and
`src/noise_module_lucid/dataset.py:32` (`from noise_module import to_jsonable`) and
`interventions.py:28` import `noise_module`. The dry run records 27 such forced repo files
(`/ceph/dwong/jepa/tmp/followup_verify/dryrun.json`
`closure_forced_imports`: 24 × `src/noise_module/*`, 3 × `src/reconstruction_model/*`;
71 modules checked, 0 violations after the exemption). The code documents the deviation
(`ext_followup.py:118-129`) but the runbook says "every repository file in `sys.modules` … is
inside `<P>`" (`:588-591`), which is impossible with the stated `<P>`. Required runbook action:
add these paths to `<P>`, or define the forced read-only dependency class explicitly (as the
implementation did).

### S3 — A0 pass-factor rule is internally inconsistent; A-4 cost understated

`runbook :212-214`: "Pass factor: 1 for a photon full or masked pass; 2 for A-3 (μ + e); 0.5
for an A-4 masked-only reference." A masked pass is priced at 1 in the same sentence in which
a masked-only reference is priced at 0.5, and `readout_a4` actually performs one full-input
reproduction pass **plus two** masked reference passes per checkpoint
(`ext_followup.py:3275-3289, 3147-3242`). The implementation's factor is 0.5
(`STAGE_A_PASS_FACTORS :1034`), so the 3.9949 h projection follows the runbook's arithmetic,
not the code's likely cost; a conservative re-count is ≈4.5–4.9 h, still under 5.5 h. §5's
"A-1b adds about 12 items" (`:504-505`) also disagrees with the 13 A-1b items in §3
(`:276-279`; code `A1B_LABELS` = 13). Required runbook action: state the A-4 factor
unambiguously (and re-check the projection).

### S4 — §2 "A0 re-verifies the pairing on the host" does not say which sets, and the dev_train pair is not checked

Runbook `:118-126` presents the pairing as a property of "the s = 0.05 and s = 0.30 sets" and
the pin table lists the dev_train pair as well as probe train/test; the implementation's
`pairing_check` covers only the two probe sets (`ext_followup.py:1129-1130`). The dev_train
pair is used by T-1 (s = 0.30 training) and A-1b (s = 0.05 marginals). A host spot-check of the
first dev_train run shows identical `event_ids` and bit-identical `vertex_m` (yield differs by
~1e-9 relative), so the property likely holds, but it is not verified by A0. Required runbook
action: name the checked sets (or add the dev_train pair to A0).

### S5 — §6.2(d) "the dry run executes the setup path of every CLI mode" is not what the dry run does

`mode_dry_run` (`scripts/jepa_ext_followup.py:91-210`) exercises the gate, config/generator/
arms/geometry load, `data_status`, `build_inventory`, `pairing_check`, both probe openers, one
rebuilt init arm, one marginal run, an 8-event label chunk, an 8-event readout sample, the T-1
cell/kwargs and the entries/projection — but not `build_plan` (inventory mode), `reproduction`
(reproduce mode), any complete `readout_*` body, `assemble_report`, or `t1_prepare_launch` /
`pregrid.datasets_for` / `lr.run_one`. The executed-import closure is therefore asserted over a
subset of the claimed setup paths. No violation is observed because the remaining imports are
stdlib/third-party, but the stated procedure is not implemented literally. Either implement the
full setup or narrow the runbook text.

---

## Independently reproduced checks (read-only)

1. File identity: `sha256sum` of the three implementation files in the clone equals the local
   copies (`9eace6ad…`, `88002bbb…`, `b042ec37…`); clone `git rev-parse HEAD` =
   `3191e690f0248ee614ee18d12dd17b41aca088dd`, working tree clean.
2. Diff identity: `git diff f5b7434 3191e69 | sha256sum` =
   `95357c657288ffb5039a1b6c0f7e2d1e111f5444c055c264c9015885fb483b59` = supplied
   `diff.patch`; 5 files, 6031 lines.
3. Runbook sha256 `063e908774ffe29b3d59ef40ab54ef5d02ff462b6aa1e49eef57c61c49e7af24` (binding
   local copy and committed file), matching `RUNBOOK_SHA256` (`ext_followup.py:93`).
4. Focused tests: `FOCUSED_RC=0`, 39 passed in 221 s (`focused.log`).
5. Dry run: `DRYRUN_RC=0`; `closure_pass: true`, 71 modules checked, 27 forced read-only
   exceptions, 0 violations, `setup_errors: []`; inventory 54 labels (46 available, 0
   unavailable, 8 pending view design); pairing pass true; projection 3.9949 h pass; entries
   pass (`dryrun.json`).
6. A-1b C-group reproduction via the implementation's own `a1b_probes` (committed label cache
   + fresh probe-test C features, on deepthought2): **R² = 0.10429278172645076**, diff 0.0
   against `A1B_REPRODUCTION_R2`, tolerance 1e-3 → pass; feature counts M=20, C=120, All=140.
7. Truth pairing spot-check (probe-test s = 0.05 vs s = 0.30, runs 0000/0001, and dev_train run
   0): identical `event_ids`; bit-identical `vertex_m`; `yield_photons` allclose at 1e-9
   (max diff 1.4e-9); `direction_m/energy_MeV/deposited_MeV/track_length_m` differ only at
   round-off (≤1e-12), exactly as `_join_truth`'s comment says.
8. Gate-(d) demonstration: see B1.
9. `committed_vertex_map` coverage: see B2.
10. View mapping demonstration: see B5.
11. T-1 reference manifest `/ceph/dwong/jepa/returns/ext_mae_dejitter_20261031/train/manifest.jsonl`
    has the s = 0.05 rows `ext_dev_s005_{within_trigger,across}__mae_mse__s{9000,9001}` with
    `outcome=complete`, `resumed_from_step=None`, `loader_dataset_manifest_sha256` /
    `probe_dataset_sha256` as pinned; the fields that will legitimately differ for the s = 0.30
    rows are in `T1_ALLOWED_DIFFERENCES` (`ext_followup.py:229-235`); `fallback`, `config_sha256`,
    `geometry_sha256`, `optimizer`, `readout_label`, etc. are config-derived and unchanged.
    `t1_host_manifest_check` (`:3507-3571`) therefore looks feasible (unverified on GPU).

## Evidence caveats

* **Full suite is not green on the supplied host.** `rc.txt` now contains
  `FOCUSED_RC=0`, `DRYRUN_RC=0`, **`FULL_RC=1`**; `full.log` ends
  `3 failed, 652 passed, 2 skipped in 647.55s`. The three failures are
  `test_guard.py::test_launcher_records_the_real_job_pid_and_it_is_never_foreign`,
  `…::test_launcher_enforces_the_timeout_and_writes_status` and
  `…::test_launcher_claim_blocks_a_second_start_on_the_same_gpu`. `test_guard.py` and
  `scripts/jepa_launch.py` are not modified by the diff; the tests hard-code
  `--host deepthought` (`test_guard.py:38, 1396-1397`) and the launcher refuses on this machine
  (`scripts/jepa_launch.py:299`):
  `"host mismatch: requested 'deepthought' but this machine is 'deepthought2'; pid.json and the
  guard status/claim paths would be wrong"`. Running the three tests in isolation on
  deepthought2 reproduces the same 3 failures. The runbook's host table (`:522`) assigns the
  full suite to **Deepthought**. So the failures are host-specific and not attributable to this
  implementation, but §6.3's "the full suite runs once on the final SHA" is not evidenced green
  as supplied: the Lead should either record the three known host-name failures or re-run the
  full suite on Deepthought before acceptance.
* The A1 reproduction itself, the A-2/A-3/A-4 encoder passes and the T-1 host check require GPU
  checkpoint forward passes and were not run here (no GPU per the review constraints); their
  correctness is assessed from the code and the committed values, and they remain to be
  exercised at execution time.
* Review-time clone status: the only untracked file after this review is this review document;
  the clone is otherwise clean at `3191e69`.

---

## Summary of required fixes before re-review

1. B1 — include `P_DIRS` in the gate-(d) pathspecs (and `p_paths`), with a negative test.
2. B2 — read the D2 METRICS table, the P-5 and M-f readings in `committed_vertex_map`; make
   "no committed target" explicit instead of `checked: False` silently passing (and resolve S1
   for ijepa/lambda/stopgrad/anc100).
3. B3 — require the A1 s = 0.30 reproduction before a T-1 launch, or record the dependency as
   an explicitly accepted procedural control.
4. B4 — make `--include-t1` actually pass the `t1_*` labels to A-1/A-2 and book the readouts to
   the T-1 cap.
5. B5 — fix the view-design runs.json-key → inventory-label mapping and record the sha.
6. Resolve spec defects S1–S5 in the runbook (or record the deviations in the acceptance
   record) before the acceptance record is written; add end-to-end A-4 and non-vacuous
   forbidden-path tests (N6, N7).

---

# Re-review (round 2) — 2026-11-03

**Reviewed SHA:** `59c0a0d5dd1f33c138c324824de6da11575d706b` (parent `3191e69`), branch `JEPA`,
base `f5b7434`. The clone `/ceph/dwong/jepa/code/followup_20261103` is at HEAD `23626fd`
(a DECISIONS-only commit on top); `git diff 59c0a0d HEAD -- <P>` is empty and there is no
untracked file under `<P>`, so the reviewed tree is the working tree at HEAD. The delta
`git diff 3191e69 59c0a0d` is byte-identical to
`/ceph/dwong/jepa/tmp/followup_final/delta_3191e69_59c0a0d.patch`
(sha256 `a7dfbccb4d821d01be5e10144f8faad7da3c17899a9455261c94a20101b0ac15`; 3 files,
+1004/−114). The three files are byte-identical to the local copies:
`720988de…158f93` (`ext_followup.py`, 4134 lines), `f3e6766d…51c7ab`
(`scripts/jepa_ext_followup.py`, 416), `b3b9b51e…8f2a61f` (`test_ext_followup.py`, 1591).
The runbook is unchanged (`063e9087…`).

**New overall verdict: PASS.** The five blocking findings B1–B5 are fixed and demonstrated;
the re-issued §6.4 checklist (i)–(viii) is all PASS, and my independent focused run is green
(53 passed, 295.67 s, exit 0). There are no remaining **code** blockers. What remains is (a) a
set of nonblocking reporting/documentation defects R1–R5 and (b) two acceptance-record/evidence
conditions E1–E2 (the §6.3 full-suite record and the live entries preflight); these do not
change any reading or gate and are not code defects. The original review above is retained
unchanged; the round-1 verdicts on (ii), (iv) and (v) are superseded by this re-review.

## B1–B5: FIXED (with demonstrations)

### B1 — gate (d) now protects the `<P>` directories: FIXED

`P_DIRS = ("src/noise_module_lucid/", "src/jepa_noise/configs/")` (`ext_followup.py:117`);
`p_paths` now returns files **and** directories (`:274-277`); the gate (d) pathspec is
`paths = [name for name in (*P_FILES, *P_DIRS)]` (`:392`) for both `git diff --quiet` and
`git ls-files --others`, and `checks["d"]["paths"]` reports both lists (`:403-410`).

Demonstrated on a throw-away git repo under
`/ceph/dwong/jepa/tmp/pytest_review_gate2/repo` (runbook + DECISIONS + acceptance record):

* pristine: `d.pass = True`, and `P_DIRS ⊆ checks.d.paths`;
* modify the tracked `src/jepa_noise/configs/lucid_arms_v0.json` → `d.pass = False`;
* modify the tracked `src/noise_module_lucid/readout.py` → `d.pass = False`;
* add untracked `src/noise_module_lucid/extra.py` → `d.pass = False`,
  `untracked = ['src/noise_module_lucid/extra.py']`.

This closes the round-1 false-PASS; the new test `test_gate_d_protects_the_p_dirs`
(`test_ext_followup.py:209-236`) is genuine (the `_fake_repo` fixture now tracks files in both
`P_DIRS`, `:97-103` in the delta) and it restores the modified files between cases.

### B2 — the A1 "recomputed item" reproduction rule is now implemented: FIXED

`committed_vertex_map` now adds (i) the six SIGReg de-jitter vertices from
`results/jepa/20261030_dejitter/readings.json` P-5 and the eight MAE de-jitter vertices from
`results/jepa/20261031_mae_dejitter/readings.json` M-f
(`_dejitter_committed_vertices`, `ext_followup.py:1655-1677`; applied `:1740-1748`), and
(ii) the two strong-drift s = 0.30 D2 vertices, hard-pinned and cross-checked against a parse of
`results/jepa/20261029_d2_partial/METRICS.md` (`STRONG_DRIFT_VERTEX = {9000: 0.233209,
9001: 0.217213}`, `strong_drift_committed_vertices` `:1619-1653`; a parsed table that disagrees
with the pins raises, so a changed METRICS.md cannot pass).

`check_item_reproduction` (`:1729-1744`) now returns, instead of a silent `checked=False`:

* `status="not_applicable"` + `NO_COMMITTED_VERTEX_REASON` (`:1619-1622`) when no committed
  full-input vertex exists at the item's setting (S1: `ijepa/lambda/stopgrad` and `anc100`, and
  any `t1_*` item);
* `status="unavailable", checked=True, pass=False` when the recomputed vertex is missing;
* `status="pass"|"unavailable"` on the tolerance comparison.

`readout_a1` surfaces a `reproduction` block with `checked` and `not_applicable` label lists
(`:2024-2042`); `readout_a3` and `readout_a4` attach `_reproduction_summary`
(`:3311, :3561, :3590-3606`).

Demonstrated on the clone:

```
committed_vertex_map: strong_drift9000=0.233209, strong_drift9001=0.217213,
  sigreg_within_trigger9000=0.2243854227, sigreg_within_median9000=0.2226532908,
  sigreg_across_median9001=0.3203138185, mae_within_trigger9000=0.1049156019,
  mae_across_median9001=0.1006316929, fix9000=0.2192585910
ijepa9000 / lambda9000 / stopgrad9000 / anc100 / t1_s030_within_trigger9000
  -> status "not_applicable", checked False, pass None, reason = NO_COMMITTED_VERTEX_REASON
strong_drift9000 with a wrong value -> "unavailable"
mae_within_trigger9000 with a wrong value -> "unavailable"
strong_drift_committed_vertices() on the real METRICS.md -> {9000: 0.233209, 9001: 0.217213}
```

The new tests `test_committed_vertex_map_covers_the_recomputed_items` (`:1188-1226`) and
`test_d2_strong_drift_table_must_agree_with_the_pins` (`:1228-1242`) are genuine and ran green.
S1 is therefore no longer silently reinterpreted: the rows are explicitly `not_applicable`
(with the written reason) and are not counted as reproduced.

### B3 — T-1 launches now require the A1 s = 0.30 gate: FIXED

`require_a1_s030_gate` (`ext_followup.py:3688-3720`) reads
`<ext>/stageA/A1/reproduction.json`; a missing record, `s030_pass != True`, or either required
row (`init9000`, `strong_drift9000`) failing its vertex or module-gain check prints the reason
to stderr and raises `SystemExit(2)`. `t1_prepare_launch` calls it **before**
`refuse_existing_run_dir`, `require_pinned` and `datasets_for`
(`:3731` vs `:3733-3741`), and `train_t1` passes `ext_root` through and returns `a1_gate`
(`:3747-3754, :3794-3803`); the CLI prints `a1_s030_gate`
(`scripts/jepa_ext_followup.py:341`) and fails the launch on a bad host/cost check.

Demonstrated: absent record → `SystemExit(2)`; `s030_pass=False` → `SystemExit(2)`;
valid record → `{'s030_pass': True, 'rows': ['init9000', 'strong_drift9000'],
's005_pass': True, 'required_rows': [...]}`. The new test
`test_t1_launch_requires_the_a1_s030_gate` (`:1244-1305`) is genuine: it covers missing/false/
valid, and with the gate removed it verifies **no** launch step runs
(`ordering == []`), while a valid gate gives `ordering == ["refuse", "pins", "datasets"]`.

### B4 — `--include-t1` now actually evaluates the T-1 checkpoints: FIXED

`readout_label_list` (`ext_followup.py:1028-1043`) returns the fixed item table plus the
`t1_*` labels from the inventory; the CLI passes that list to the body when `--include-t1` is
given (`scripts/jepa_ext_followup.py:278-289`); `readout_a2` uses it when `labels is None`
(`:2743`) and keeps the s = 0.30 environment rows for the s = 0.05-trained MAE and the T-1
checkpoints in `env_cache` (`:2744-2806`), computing G-2 event-by-event through
`pair_events` + the A-2 run-cluster bootstrap (`_t1_g2`, `:2812-2847`). `t1_readings` now fills
G-1, G-2 and the T-1 A-1 rows (`:3960-4000`), `assemble_report` writes the T-1 cost and
readings (`:4019-4030`), and the report README carries a "T-1 GPU cost (N4)" block
(`:4105-4116`).

Demonstrated: `readout_label_list("A-1", {t1_s030_within_trigger9000,
t1_s030_across_trigger9001})` = 39 base labels + both T-1 labels (no duplicates);
`readout_label_list("A-2", ...)` = 4 base + both. The new tests
`test_readout_label_list_includes_the_t1_rows` (`:1307`),
`test_t1_readings_are_populated_from_the_actual_a2_payload` (`:1359`),
`test_cli_include_t1_passes_the_explicit_t1_labels` (`:1532`) and
`test_assemble_report_populates_t1_readings_and_cost` (`:1558`) are genuine; the latter
verifies the written `readings.json` G-1/G-2/cost values, not just a helper.

### B5 — the view-design inventory mapping is fixed: FIXED

`build_inventory` parses the view `runs.json` keys `<split>__<seed>` and maps them to the
inventory labels `view_<split><seed>`, records the committed sha and `committed_vertex_m`, and
searches the view root for the matching `final.pt` (`ext_followup.py:960-1000`);
`view_root` is injectable for tests. Demonstrated with a synthetic committed view tree:
`view_within9000`, `view_across9001`, `view_module_parity9001` and `view_random_half9000` are
all `status="available"` with the correct sha, an existing source and
`committed_vertex_m = 0.15`. The new test `test_view_inventory_rows_are_available_from_committed_runs`
(`:1424-1434`) is genuine (it monkey-patches all external return roots and builds a real
committed view tree).

## N2, N3, N4, N6, N7: FIXED

* **N2** — `apply_item_cell` (`ext_followup.py:1544-1566`) installs the row's §2.1
  `ArmSettings` on an arm built by `er.item_arm` and records `_ext_item_cell`
  (`cell_id, setting, s_arm, s_p, split`); the anchor path records the same; `evaluate_item_row`
  now reports `evaluation_cell` and `evaluation_config` (`:1606-1607`). The s = 0.30 cell is
  actually used (the test `test_s030_cell_is_installed_on_the_built_arm`, `:1381-1398`, checks
  the settings differ from the s = 0.05 row). It remains numerically inert for the evaluated
  arms, as documented (only `mae_chi2` reads `settings.s`, `lucid_arms.py:520`).
* **N3** — gate (e) now requires `out_root` to be exactly `<ext>/{smoke,train}` (resolved), not
  merely a directory with the right basename; `assert_authorised` takes `ext_root`
  (`ext_followup.py:416-433`). Demonstrated: `<ext>/train` passes; `elsewhere/train` fails.
  The updated `test_gate_e_only_the_pinned_launches` (`:236-270`) covers the wrong-parent case.
* **N4** — `t1_cost` (`ext_followup.py:3911-3927`) sums completed-run `wall_seconds` into
  measured GPU hours and reports `projected_hours`, `cap_hours`, `cap_pass`,
  `projection_pass`; `train_t1` returns it, the CLI prints it and exits 2 if the cap is
  exceeded (`scripts/jepa_ext_followup.py:337-345`), and `assemble_report` writes it. The new
  test `test_t1_inventory_rows_and_the_t1_cost` (`:1320-1357`) checks the measured value, the
  pass and the over-cap failure.
* **N6** — `test_forbidden_call_paths_are_not_used` (`test_ext_followup.py:376+`) now executes
  the real openers against synthetic pinned data with recording fakes: a failed pin raises
  before any opener; the s = 0.30 opener is called only with `registered=False` and after the
  hash check; a bad dark-bank hash never reaches `ensure_dark_bank`; and an existing run dir is
  refused before `datasets_for` (`dataset_calls == []`). This is no longer vacuous.
* **N7** — `test_a4_end_to_end_masked_reference_invariance` (`:1477-1514`) runs the real
  `masked_reference_readout` twice on a synthetic dataset (base vs hidden-time-shifted) with
  `return_streams=True`, checks the `context_only`/`propagated` event streams are bit-identical
  under the context-only reference, that the all-token reference changes, and the
  zero-visible-hit / None-trigger counts. This closes the end-to-end gap.

## Re-issued §6.4 checklist on `59c0a0d`

| Item | Verdict | Evidence |
|---|---|---|
| (i) pairing and truth leakage | **PASS** | `pairing_check` now also joins the dev_train pair (`ext_followup.py:1175-1186`); dry run: probe_train/probe_test/dev_train all pass (4 runs each); `pair_events` unchanged and tested; host truth spot-check from round 1 still valid |
| (ii) inventory and evaluation configuration | **PASS** | view family fixed (B5); `data_key` added to every row (`_row`, `:718-731`); s = 0.30 cell installed and recorded (N2); E2/median/arch/anchor/s = 0.30 configs unchanged from round 1; host dry-run inventory 54 labels / 46 available / 0 unavailable / 8 pending view |
| (iii) A-4 masked-pass reference and invariance test | **PASS** | implementation unchanged from round 1 (`context_time_shift`/`apply_time_shift`/`masked_reference_readout` `:3335-3436`, `readout_a4` `:3500`); the §6.3 test is now end-to-end through `masked_reference_readout` (N7) |
| (iv) A1 reproduction on s = 0.05 and s = 0.30 | **PASS** | the two A1 tables unchanged and correct; own-reproduction now checked for D2/P-5/M-f targets and explicitly `not_applicable` elsewhere (B2); T-1 launch gated on both s = 0.30 rows (B3) |
| (v) gate (a)–(e) | **PASS** | (a)–(c) unchanged and tested; (d) now covers `P_DIRS` (B1, demonstrated); (e) exact out root (N3). The §6.2(d) forced-import deviation is recorded in DECISIONS and matches the dry run (71 checked, 27 forced, 0 violations) |
| (vi) forbidden call paths | **PASS** | code audit from round 1 still holds; the §6.3 test is now non-vacuous (N6) |
| (vii) cost projection against the caps | **PASS** | conservative A-4 factor 3.0 → 107.8 item-equivalents = 4.9049 h ≤ 5.5 h (dry run); CPU 7.0 ≤ 8; S1 133.1507 ≤ 160; T-1 measured/projection cost now recorded with a cap check (N4) |
| (viii) A-1b collector / 0.1043 / read-only `dev_train` reader | **PASS** | unchanged from round 1; the C-only probe path now avoids reading the 100k marginals (`a1b_probes(..., requested=("C",))`, `:2438-2442`), and the round-1 independent reproduction (C ridge = 0.10429278172645076, exact) still applies |

## Remaining nonblocking findings

* **R1 (reporting, A-3/A-4).** `_reproduction_summary` (`ext_followup.py:3590-3606`) reads
  `record["label"]`, but `_a3_photon_reproduction` (`:3318-3326`) and `_a4_reproduction`
  (`:3581-3589`) return `check_item_reproduction(...)` without a label. Demonstrated:
  `_reproduction_summary([check_item_reproduction("ijepa9000", 0.5, map)])` gives
  `not_applicable: [None]`, and several checked A-3 items would collide under the key `"?"`.
  The per-model records still carry the status/reason (no silent pass), but the summary is
  unusable for labelling. Fix: attach the label in `_a3_photon_reproduction`/`_a4_reproduction`
  or pass `(label, record)` pairs. This does not change any reading; fix before the report if
  possible.
* **R2 (accounting, T-1).** `t1_cost` sums only manifest rows with `outcome == "complete"`
  (`:3915-3917`); runbook §5 says failed attempts and smokes are also booked. The fixed launch
  table keeps the authorised work at ≈2.7 h projected, but a failed-but-recorded attempt's
  wall time would be omitted from the measured figure. Minor; document or sum all attempted
  rows.
* **R3 (DECISIONS vs code, S3).** The DECISIONS line records S3 correctly as "A-4 counted
  conservatively at 3.0 per checkpoint (107.8 item-equivalents = 4.9049 L40S h, ≤ 5.5 h)", but
  it also says the projection is "reported both per the runbook rule and with the conservative
  A-4 count". The code reports **only** the conservative projection:
  `stage_a_projection` (`:1080-1097`) returns a single `projected_l40s_hours` (dry run 4.9049)
  and nowhere reports the runbook-rule 87.8-equivalent / 3.9949 h figure. Correct the
  DECISIONS sentence, or add the runbook-rule projection to `plan.json` as a second field.
* **R4 (DECISIONS, S1 nuance).** The S1 record says the only committed vertices are the
  s = 0.30 D2 values; that is true for `ijepa/lambda/stopgrad`, but `anc100` has no committed
  full-input vertex at all. The disposition is unaffected (all are `not_applicable`), but the
  sentence should say so.
* **R5 (test weakness).** `test_reproduction_summary_surfaces_not_applicable`
  (`test_ext_followup.py:1522-1530`) includes `assert ... or True` (vacuous) and does not catch
  R1, because it passes records that already carry a `label`. Add a case with a label-less
  record (the real `_a3_photon_reproduction` shape) or fix R1 and assert labels.
* **E1 (full-suite evidence / record).** The §6.3 requirement "the full `src/jepa_noise/tests`
  suite run once on the final SHA" is not evidenced by a completed green run on the runbook's
  assigned host. The deepthought attempt was killed: `rc_deepthought.txt` has
  `FULL_DT_RC=143` (SIGTERM), and the 109-byte partial log shows one failure at ≈10 % (test
  #29 in collection order is an untouched `test_bridge_run` case) before termination; a second
  deepthought attempt was started at 14:47 UTC and had produced no result by the end of this
  review. On deepthought2 I verified: (a) the 3 `test_guard` launcher failures are
  host-name-specific — running those tests at the untouched `f5b7434` checkout on deepthought2
  gives the same `3 failed, 134 deselected`, and they hard-code `--host deepthought`
  (`test_guard.py:38, 1396-1397`; the launcher refuses with
  `scripts/jepa_launch.py:299`); (b) the full suite at `59c0a0d` on deepthought2 with exactly
  those 3 verified pre-existing host-specific tests deselected is **666 passed, 2 skipped,
  3 deselected in 537.61 s, exit 0**. The DECISIONS environment line claims the 3 tests "pass
  on deepthought"; that is plausible but is not supported by any supplied log (the only
  deepthought log was killed). For the acceptance record: either complete the deepthought run
  (or a targeted 3-test deepthought run) and cite it, or record the observed state exactly
  (deepthought2 full suite green modulo 3 host-specific tests that fail identically at
  `f5b7434`; deepthought attempt killed by the SSH/session limit). I do not regard E1 as a code
  blocker; it is an evidence/record condition.
* **E2 (entries preflight, transient).** The `dryrun.json` captured at 14:38 has
  `entries.pass = false` (35,388 → 29,504), and the live check at re-review time was also
  marginal; after both the Lead's and my own review basetemps were removed the live check
  passed (37,234 → 31,350 ≥ 30,000). This is a live execution-time STOP (A0 re-checks it), not
  a code defect; the acceptance record should cite the checked value at acceptance time.

## Evidence collected in this re-review

* Focused suite (my run, clone at HEAD, `PYTHONPATH=src`, `nice -n 10`,
  `pytest src/jepa_noise/tests/test_ext_followup.py`,
  basetemp `/ceph/dwong/jepa/tmp/pytest_review_final`): **53 passed in 295.67 s, exit 0**
  (Lead's run: 53 passed in 282.21 s). All 53 tests reviewed for genuineness; the 14 new ones
  are listed above.
* Full suite (my run on deepthought2 at `59c0a0d`, the 3 pre-existing host-specific launcher
  tests deselected): **666 passed, 2 skipped, 3 deselected in 537.61 s, exit 0**.
  The deselected names are
  `test_guard.py::test_launcher_records_the_real_job_pid_and_it_is_never_foreign`,
  `…::test_launcher_enforces_the_timeout_and_writes_status`,
  `…::test_launcher_claim_blocks_a_second_start_on_the_same_gpu`; they fail identically at the
  untouched `f5b7434` checkout on deepthought2 (3 failed, 134 deselected, 21 s).
* Dry run (`/ceph/dwong/jepa/tmp/followup_final/dryrun.json`, `DRYRUN_RC=0`): closure pass
  (71 modules checked, 27 forced read-only exceptions, 0 violations), `setup_errors: []`,
  inventory 54 labels / 46 available / 0 unavailable / 8 `pending (view design)`, pairing
  3 sets pass (`n_compared` 4 each, incl. dev_train), projection 4.9049 h (107.8
  item-equivalents, pass), S1 and the T-1 cells/kwargs recorded.
* Gate demonstrations: gate (d) P_DIRS (pristine/modified/untracked) and gate (e) exact root —
  see B1/N3 above; gate (b), (c) unchanged and tested.
* `committed_vertex_map`/`check_item_reproduction`/T-1-gate/include-t1 demonstrations — see
  B2/B3/B4 above.
* Tree identity: `git diff 59c0a0d HEAD -- <P>` empty; `git ls-files --others --exclude-standard
  -- <P>` empty; clone HEAD `23626fd`; the only untracked file in the clone is this review.
* Ledger unchanged: `v4_caps.E2_lucid_l40s_deepthought2.charged_gpu_hours = 116.1507`;
  S1 = 116.1507 + 8.0 + 9.0 = 133.1507 ≤ 160.

## Spec-defect record check (DECISIONS commit `23626fd`)

The appended lines (`docs/jepa/DECISIONS.md`, commit `23626fd`, 5 lines) were read against the
code:

* **S2** (forced-import infeasibility) — accurate: the code exempts exactly the two tracked
  packages (`ext_followup.py:129`, `FORCED_IMPORT_EXCEPTIONS = ("src/reconstruction_model/",
  "src/noise_module/")`) and the dry run records 27 forced read-only files with zero
  violations. The record's claim that the reviewer confirmed the infeasibility is correct
  (round-1 review).
* **S4** (dev_train pair) — accurate: `pairing_check` now includes the dev_train pair
  (`:1175-1186`), and the dry run shows all three sets passing.
* **S5** (representative dry-run setup) — accurate: `dryrun.json` has the setup keys claimed
  (gate/data/inventory/pairing/openers/arm/label rows/readout sample/T-1 cells/kwargs/
  entries/projection), and the closure is asserted over the executed modules.
* **S1** — accurate disposition (rows marked `not_applicable`, never counted as reproduced)
  apart from the `anc100` nuance in R4.
* **S3** — the disposition (A-4 factor 3.0, 107.8 equivalents, 4.9049 h ≤ 5.5 h) matches the
  code, but the "reported both per the runbook rule and with the conservative A-4 count"
  sentence does not (R3).
* **Environment deviations** — (ii) GitHub credentials and (i) deepthought SSH are external
  claims I cannot verify from the clone; (iii) the 3 `test_guard` failures being host-specific
  and pre-existing is **verified** (reproduced at untouched `f5b7434` on deepthought2); the
  sub-claim "they pass on deepthought" is **not supported by the supplied logs** (E1). The
  Phase-1 CPU booking and entries figures are point-in-time; the L40S ledger value is verified
  unchanged.

**Conclusion on the spec defects:** S2, S4, S5 and S1 are now adequately recorded rather than
silently reinterpreted, and the code behaves as recorded (S1 via explicit `not_applicable`,
S2 via the forced-import exception and zero-violation dry run, S4 via the dev_train pair, S5 via
the representative setup; the §5 "12 vs 13" arithmetic is recorded and the code counts 13). S3
is adequately *dispositioned* (conservative A-4 factor, still inside the cap), but the DECISIONS
sentence about dual reporting must be corrected (R3). The runbook itself remains unchanged, as
required; the acceptance record should cite these DECISIONS lines.

## Conditions for the acceptance record (no code change needed)

1. **E1** — cite a completed green full suite or the exact observed state; the deepthought
   attempt must be completed or its kill (143) and the verified pre-existing host-specific
   failures recorded (correct the unsupported "they pass on deepthought" wording accordingly).
   My deepthought2 evidence is: full suite minus the 3 verified pre-existing host tests =
   666 passed / 2 skipped / 3 deselected, exit 0.
2. **R3** — correct the DECISIONS S3 "reported both" sentence (or add the runbook-rule
   projection to `plan.json`).
3. **E2** — record the entries preflight value at acceptance time (re-check after basetemps /
   other jobs are cleared; currently live 37,234 → 31,350 passes, but the 14:38 dry-run value
   was marginal).
4. Optionally fix R1 (A-3/A-4 reproduction-summary labels), R2 (failed-attempt booking), R4
   (S1 anc100 wording) and R5 (vacuous test assertion); none affects any reading, gate or
   threshold, and none blocks acceptance.

**Re-review verdict: PASS** for the implementation and for checklist (i)–(viii). The five
blocking findings are fixed and demonstrated; the remaining items are the evidence/record
conditions E1–E2 and the nonblocking R1–R5 above, none of which is a code defect. Read-only
constraints were honoured: only this review document was written in the clone; the
implementation, runbook, registered files, DECISIONS/STATE and ledger were not modified; no
commit or push was made and no ACCEPTANCE record was written. All temporary review directories
under `/ceph/dwong/jepa/tmp/pytest_review_*` were removed.
