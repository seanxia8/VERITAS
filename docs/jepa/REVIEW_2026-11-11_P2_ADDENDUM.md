# Independent implementation review — P2 addendum (fair R3u ceiling, P2-f_u, η, F1/F2 fold, hygiene)

**Item:** `paper3_p2_addendum_2026-11-11`, implementation only (A1–A7). CPU only; no
outcome-generating step was run by this review.
**Runbook:** `docs/jepa/RUNBOOK_2026-11-11_P2_ADDENDUM.md`, sha256
`f97ed8101e5d0383a85ada70d974df4bb54bc70e63865d48fe77cb879cf59e0a` (re-hashed, matches).
**Commit range reviewed:** `a38e320..6f14af8` (`b5ad78a` = A1–A6 implementation, `6f14af8` = the
committed-payload loader fix). Branch `jepa-runbook-20261107`, HEAD
`6f14af865e15aaab0c03bfe20e06cb68260dd3b6`, worktree clean.
**Files reviewed:** `src/jepa_noise/ext_p2a.py` (new, 1109 lines),
`scripts/jepa_ext_p2a.py` (new, 239), `src/jepa_noise/tests/test_ext_p2a.py` (new, 287),
`src/jepa_noise/ext_p2.py` (F1/F2 folds), `scripts/jepa_cpu_launch.py` (allow-list),
`test_ext_repair.py`/`test_ext_shortcut.py`/`test_ext_followup.py` (A5).
**Reviewer:** fresh-context subagent, read-only clone; did not write this code.

## Verdict

**BLOCK.** Five blocking findings (B1–B5). The most serious is that the A1 ceiling check — the
primary item of this addendum — cannot detect a ceiling violation: the paired CI is computed for the
opposite contrast and the S4/pass plumbing that should stop the run is dead code. Separately, the A6
`shift` mode is executable without the acceptance gate, and `report` writes the READINGS before it
evaluates any stop condition.

## What I verified, and how

Host: `deepthought.etp.kit.edu`, clone `/ceph/dwong/jepa/code/jepa_work_20261107` at
`6f14af865e15aaab0c03bfe20e06cb68260dd3b6` (i.e. **at the implementation SHA**, not behind; only
five unrelated untracked `results/jepa/20261106_repair/*` files, pre-existing). Commands run with
`PYTHONPATH=src:.`, `OMP_NUM_THREADS=MKL_NUM_THREADS=4`, `TMPDIR=/work/dwong`.

| check | command | result |
|---|---|---|
| addendum tests | `pytest src/jepa_noise/tests/test_ext_p2a.py -q` | **11 passed** (28.9 s) |
| four ledger tests (A5) | `test_booked_gpu_hours_is_the_exact_ledger_sum`, `test_addendum_s1_uses_only_the_addendum2_cap_on_a_corrected_ledger`, `test_booked_gpu_hours_recomputes_the_v4_caps_recent_sum`, `test_addendum_s1_fails_closed_on_an_inconsistent_ledger` | **4 passed** |
| P2 module after the fold | `pytest src/jepa_noise/tests/test_ext_p2.py -q` | **26 passed** |
| A5 xfail | `pytest ...::test_label_collector_reproduces_the_committed_fix9000_rows -rxX` | **1 xfailed** (109 s) — the failure is real, not skipped |
| lint | `python3 -m pyflakes` on the five changed modules (pyflakes 3.2.0) | **clean** |
| runbook sha | `sha256sum docs/jepa/RUNBOOK_2026-11-11_P2_ADDENDUM.md` | `f97ed81…59e0a`, matches `RUNBOOK_SHA256` |
| P2-e primary numbers | recomputed the K15 secondary Spearman from the committed `p2.json` | 0.625, n = 15 (matches) |
| live gain table vs committed | `p2x.gain_r2_eq_table(".")` against `p2.json[*].gain_r2_eq` | all 15 identical today (see M13) |
| ceiling rule sign | direct call of `p2x._difference_reading` on a synthetic violation | see B1 |
| mirror | `ls -ld /dev/shm/p2_data_mirror` on deepthought | **absent** (nothing left to remove; still must be recorded — see M1) |

Not run (deliberately, review scope): `equivalence`, `p2a`, `shift`, `report` modes. They read
pinned caches, append to the shared CPU job ledger and would produce outcome files; that belongs to
execution, not review.

Code-level checks I performed: read all 1109 lines of `ext_p2a.py` and the CLI, the full
`b5ad78a`/`6f14af8` diffs, the runtime adapters `/work/dwong/p2_20261109/fix_x3p_energy.py` and
`fix_p2_exact.py`, the pre-fold `_x3p_blocks`, `r2_design`/`_align_subset`/`_difference_reading`/
`raw_map`/`fit_ridge`, the committed `p2.json`/`x3p.json` structure, `RESOURCE_LEDGER.json`, and
the P2 READINGS.

Note on the SHAs: the brief calls `6f14af8` “the implementation commit”, but it only fixes the
loader paths; the substantive implementation is `b5ad78a`. The acceptance record must therefore name
`6f14af8` (it includes the fold and the loader fix); both are inside the reviewed range.

## Blocking findings

### B1 — A1 ceiling check: the paired CI is computed for the opposite contrast, so S4 cannot fire

`src/jepa_noise/ext_p2a.py:314` computes `ceiling_difference = _diff(err_r2, err_r3u, runs_e)`.
`_diff` delegates to `ext_p2._difference_reading`, documented and implemented as **`right − left`**
(`src/jepa_noise/ext_p2.py:1686–1694`: `mean_difference_m` is `mean(right) − mean(left)`). So
`ceiling_difference` is the pair/CI of **v_R3u − v_R2**, while:

- `ext_p2a.py:359` reports `v_R2_minus_v_R3u_m = v_R2 − v_R3u` (the negated quantity);
- `ext_p2a.py:360` labels `ceiling_difference["ci95"]` as that gap's CI — it is the CI of its
  negative;
- `ext_p2a.py:362–363` flags a violation with
  `ceiling_gap < −CEILING_TOLERANCE_M and ceiling_difference["ci95"][1] < 0.0`.
  `ci95[1] < 0` means the CI of (R3u − R2) is wholly negative, i.e. **v_R2 > v_R3u**, which
  contradicts `gap < −5 mm`. The correct tail is `ci95[0] > 0`.

Demonstrated on the host with the production helper (constant arrays give a zero-variance bootstrap,
which is the sharpest case):

- v_R2 = 0.080, v_R3u = 0.090 (R2 *better* by 10 mm — the runbook's violation, A1 lines 89–91):
  `gap = −0.010`, `ci_m` as written `[+0.010, +0.010]`, `violation = False`.
- The only case that could satisfy the code's conjunction is R2 worse than R3u, which is the
  expected direction.

Consequence: `ceiling["violation"]` is effectively always `False`; `_ceiling_summary`'s
`n_violation_gt_5mm_ci` (used by READINGS table 3) is therefore always 0 and S4 (runbook lines
186, 89–91) can never trigger on the real data. The reported per-cell `ci_m` also has the wrong sign
relative to its own point estimate, so even a correct violation would be written into `p2a.json`
with a self-contradictory CI. A1 is the primary item of the addendum; its acceptance rule is not
implemented.

### B2 — the S4 / 1e-9 result is never propagated to `pass`, and two summary fields are fabricated

- `src/jepa_noise/ext_p2a.py:775` sets `"violations": recomputed_mismatch if False else []`. This is
  dead code: the list is unconditionally empty.
- `src/jepa_noise/ext_p2a.py:721` sets `payload["pass"] = bool(not payload["ceiling_check"]["violations"])`
  — i.e. always `True`.
- `src/jepa_noise/ext_p2a.py:750/773` reports `max_abs_recomputed_minus_committed_m = 0.0` from a
  local `max_abs_recomputed = 0.0` that is never updated; the actual per-cell
  `check_abs_lt_1e-9` failures are only collected in `recomputed_mismatch` (line 774) and are then
  discarded by line 775.

Consequence: `scripts/jepa_ext_p2a.py:80–82` only exits non-zero when `payload["pass"] is False`, so
`p2a` mode returns 0 even when a cell's recomputed `v_R2` disagrees with the committed `p2.json`
value by more than 1e-9, or when the (never-detected, see B1) ceiling violation occurs. S4 requires
the run to stop and write nothing to READINGS; nothing stops.

### B3 — the A6 `shift` mode runs with no acceptance gate

`shift_readings` (`src/jepa_noise/ext_p2a.py:595–662`) never calls `assert_authorised`, unlike
`run_p2a` (`:687`), `run_equivalence` (`:812`) and `assemble_report` (`:1077`).
`mode_shift` (`scripts/jepa_ext_p2a.py:182–186`) calls it directly and then `_finish`, which only
checks the import closure and `payload["pass"]`. With no `ACCEPTANCE_2026-11-11_P2_ADDENDUM.json`,
or with a record whose `review_verdict` is `BLOCK`, or with the runbook sha changed, or with the
`<P>` tree dirty versus `impl_commit`, **A6 still executes and writes its payload**. Runbook §4.3
("Commit the acceptance record … before any outcome-generating step") and §4.2 are therefore not
enforced for one of the five modes, and `test_gate_pending_denied_and_granted` does not cover it.

### B4 — `report` writes READINGS_SUMMARY.md and both figures before evaluating any stop

`assemble_report` writes `p2a.json`, `equivalence.json`, `x3p_shift.json`, `SUMMARY.json`,
`READINGS_SUMMARY.md` and both PNGs at `src/jepa_noise/ext_p2a.py:1094–1100`, and only afterwards
computes `payload["pass"]` at `:1105–1106`. Combined with B2 (`p2a.pass` is always `True`) this
means:

- a failed A4 equivalence (S3) or a failed A6 run still leaves a complete READINGS in
  `results/jepa/20261111_p2a/`; `_finish` returns 2 only after the files exist;
- a ceiling violation (S4) is neither detected (B1) nor consulted here, so READINGS is written
  although S4 says "Write nothing to READINGS".

This is the same class of defect the P2 review recorded as N5 ("report not stopping on
incomplete/cap overrun"). The pass condition must be evaluated before any write, and must include
`ceiling_check.n_violation_gt_5mm_ci == 0`, `n_cells_with_mismatch == 0`, `recomputed_mismatch == []`
and `complete`.

### B5 — A7 is unimplemented: the generated READINGS carries neither required correction

`_readme` (`src/jepa_noise/ext_p2a.py:953–1020`) is the only producer of `READINGS_SUMMARY.md`, and
it contains no statement that P2-c is a transform check (R1 = T1.0 by construction, not an empirical
finding) and no note that `fix9000` T0.5 is flagged information-destroying only because its refit
(3.18 mm) is worse than no repair (2.94 mm) and is recorded as refit noise. The committed P2
READINGS also lacks both statements (I checked `results/jepa/20261109_p2/READINGS_SUMMARY.md`), so
A7 (runbook lines 152–158) is unsatisfied anywhere. A manual append is not durable: a re-run of
`report` rewrites the file from `_readme`. A7 is an explicit §3 item and the review checklist asks
for A1–A7; the two statements must be emitted by the generator.

## Non-blocking findings and suggestions

1. **M1 (medium) — `EXECUTION.json` has no producer and §6 table 7 is fed by a file nobody writes.**
   Neither `ext_p2a.py` nor `scripts/jepa_ext_p2a.py` writes
   `results/jepa/20261111_p2a/EXECUTION.json`; `grep -rn EXECUTION.json` finds only the runbook and
   the A5 test reason. §6 lists it as a required output and §5 requires it to carry a stop. Table 7
   ("resources, tests, mirror removal") reads
   `results/jepa/20261111_p2a/execution_extra.json` (`scripts/jepa_ext_p2a.py:197`), which nothing
   produces either, so table 7 is empty by construction and the A5 mirror record (host, size freed,
   time — `/dev/shm/p2_data_mirror` is already absent on deepthought) has nowhere to go. If the
   runbook intends a hand-assembled record, say so in the runbook and fix the A5 reason (M12),
   which currently cites the unwritten file.
2. **M2 (medium) — A1's log-yield refit is missing.** Runbook line 80: "Also the log-yield probe
   (descriptive)." There is no `log_yield` call anywhere in `ext_p2a.py` (`grep` finds none);
   `ceiling_cell` (`:271–370`) computes only the vertex probe. A1 is therefore only partly
   implemented.
3. **M3 (medium) — READINGS table 2 omits ρ_R3.** Runbook line 204 requires per model at
   K5/K15/K30: ρ_R2, ρ_R3, ρ_R3u, η, C0. `p2a_summary` stores only `rho_R2`, `rho_R3u`, `eta`, `C0`
   (`ext_p2a.py:933–944`) and `_readme` prints those three (`:967–977`); ρ_R3 exists only in
   `p2a.json` (from `p2_cell["refit"]`).
4. **M4 (medium) — table 1's ρ_R3 is a ratio of means, not the mean of ρ_R3.**
   `ext_p2a.py:920–931` first assigns `R_c_m` (metres) to `rho_R3_mean`, then overwrites it with
   `mean(R_c)/mean(Δ_c)`. The runbook's "mean ρ_R2, ρ_R3, ρ_R3u" is a mean over the per-cell
   readings; with unequal per-model Δ_c the ratio of means differs from the mean of ratios. The
   fallback (line 921) is also wrong on its own (an mm value in a ρ field).
5. **M5 (medium) — the raw reference's `R_c_u_ci_m` is the wrong contrast.**
   `ext_p2a.py:333–337`: `R_c_u_m = v_raw_R3u − v_raw_N`, but
   `R_c_u_ci_m = _diff(raw_c_aligned, raw_r3u_errors)["ci95"]`, the CI of `v_raw_R3u − v_raw_c`.
   Reader-facing field name and number disagree. (The vertex cell's `R_c_u_ci_m` at
   `:346–347` is correct.)
6. **M6 (medium) — the raw refit attaches `u` to raw rows positionally.**
   `ext_p2a.py:320–321` does `augmented(data["raw_c_train"], u_train)`. The raw rows come from
   `_raw_rows(raw_photon_features(...))` over *all* probe events, while `u_train` comes from the
   embedding cache rows; nothing checks lengths or the `(run_id, event_id)` keys. A length mismatch
   raises, but a same-length reordering would silently pair each event's hit-count/dead-map design
   row with another event's raw features and corrupt the descriptive raw bound. Everywhere else in
   this code base, cross-row work goes through `_align_subset`/`pair_events`; use a keyed alignment
   (or at least assert the key order).
7. **M7 (medium) — the exact-subset ceilings are computed but never summarised or checked.**
   `_ceiling_summary` (`ext_p2a.py:747–777`) iterates only `entry["cells"][name]["ceiling"]`; the
   nested `exact_subset_reading["ceiling"]` is dropped. A1 says "for every cell evaluated in P2
   (full cell and exact subset)" and table 3 asks for the number of cells below R3u. As written the
   sweep covers 105 cells instead of 210, and an exact-subset violation would not enter S4.
8. **M8 (medium) — `compare_numbers` can pass vacuously.** `ext_p2a.py:781–805` only recurses into
   keys present on both sides and does nothing for strings; `run_equivalence` accepts a cell/model
   with `n_compared == 0` and no mismatch as equivalent (`:883–884`). An "unavailable" fresh cell
   (`{"status": "unavailable", "reason": …}`, as `_cell_for_model` returns on any `ValueError`) or a
   renamed key would compare to nothing and pass S3. Require `n_compared > 0` per cell/model and a
   minimum total, and fail when a fresh side is `unavailable`.
9. **M9 (medium) — the equivalence run mutates the live Stage-1 CPU job ledger.**
   `run_equivalence` patches `p2x.EXT_ROOT` and `p2x.assert_authorised` (`ext_p2a.py:860–870`) but
   not `p2x.JOBS_RECORD` (`ext_p2.py:61`, bound to
   `/ceph/dwong/jepa/returns/ext_p2_20261109/jobs.jsonl` at import). The inner `x3p` therefore calls
   `record_job("x3p", "cpu", …)` (`ext_p2.py:971–981`, `x3p` tail) and appends a line to the shared
   ledger, mislabelled as an `x3p` job. The "ledger unchanged (134.1505)" budget is a GPU number, so
   the pin itself survives, but the CPU ledger is silently changed and the A5 ledger tests read a
   different file. Redirect `JOBS_RECORD` to scratch too, or record the equivalence as a deliberate
   addendum job.
10. **M10 (medium) — the launcher allow-list is a prefix match, and one entry is dead.**
    `scripts/jepa_cpu_launch.py:69–71` accepts when `text.startswith(prefix)` for any of the three
    prefixes, so `scripts.jepa_ext_p2_anything` is allowed; and because `scripts.jepa_ext_p2` is
    checked before `scripts.jepa_ext_p2a`, the latter never reaches its own entry (it maps to
    `ext_p2_20261109` only through prefix shadowing). `test_cpu_launcher_allow_list`
    (`test_ext_p2a.py:239–247`) tests the happy paths and one rejection but not the prefix hole.
    Accept exactly the name or `prefix + "."`, and add a rejection case such as
    `scripts.jepa_ext_p2_evil`.
11. **M11 (medium) — the `shift` output name and `report` disagree.**
    `_finish`/`_out_json` write `<mode>.json`, i.e. `shift.json`, when `--out` is a directory
    (`scripts/jepa_ext_p2a.py:47–53, 70–82`), while `mode_report` reads
    `results/jepa/20261111_p2a/x3p_shift.json` (`:194`). The natural invocation
    `--out results/jepa/20261111_p2a` therefore leaves `report` failing with a confusing
    "report needs p2a.json, equivalence.json and x3p_shift.json". Either write the §6 name from the
    mode or make the mode take an explicit output file.
12. **M12 (medium) — A5's xfail reason is an unverified diagnosis that cites an unwritten file.**
    `test_ext_followup.py:754–760` claims "environment-level float/torch mismatch" and says it is
    "recorded in results/jepa/20261111_p2a/EXECUTION.json". The test genuinely fails (I reproduced
    the xfail), and the shapes match while values do not — but no evidence for the float/torch
    attribution is in the tree or in any record, and the cited file is not produced (M1). `strict=False`
    also means an XPASS would be silent. Record the actual evidence (e.g. the compared row, dtype,
    max abs difference, torch version) at execution time and prefer `strict=True` once the cause is
    known, so a fix cannot be missed.
13. **M13 (medium) — A3 takes its x-values from the live `/ceph` gain table, not the committed one.**
    `load_inputs` calls `p2x.gain_r2_eq_table(root)` (`ext_p2a.py:394`), which for the two new models
    reads the mutable `Path(EXT_ROOT)/x2b/<label>.json` (`ext_p2.py:2087–2090`), whereas the
    primary/secondary it is compared with were computed from `p2.json[*].gain_r2_eq`. The 15 values
    happen to match today (checked), so the secondary 0.625 will reproduce, but the provenance is not
    pinned. Read `gain` from the committed `p2.json` payload.
14. **M14 (minor) — `p2f_u`'s note and the "keep the P2 fields" requirement.**
    `ext_p2a.py:500` emits only `"beyond refit (R4/R5 territory)"`; runbook A2 (lines 99–100)
    prescribes "cheapest rung X reaches 80 % of the refit ceiling; R_c^u = … mm remains beyond refit
    (R4/R5 territory)", and the else branch "cheapest rung X". The numbers are all present in the
    block, but the required sentence is not produced. Also, the original P2-f fields are not carried
    beside `P2-f_u` in `p2a.json` (`:712`); they remain only in `p2.json`, so a reader of `p2a.json`
    cannot see both labels side by side.
15. **M15 (minor) — `fair_design` decides by shape, not by condition kind.**
    `ext_p2a.py:231–234` drops the second column when the design has exactly two columns and the
    second is all ones. With the real 105-module geometry that is unambiguous, but with a
    2-module geometry a K/P cell whose second module column happens to be all ones would lose a real
    module column. Use `condition.kind == "dead"` for the branch.
16. **M16 (minor) — A3's n = 13 sensitivity silently uses K5 only.**
    `ext_p2a.py:564` calls `_eligible_cells(..., "K5")`. The runbook (line 113) says "the primary
    Spearman without fix9000/fix9001" without naming a level; the primary exists at three levels.
    Emit all three or state in the output why K5 was chosen.
17. **M17 (minor) — `_readme` hard-codes the frozen primary numbers.**
    `ext_p2a.py:990–991` prints "rho_s K5/K15/K30 = 0.282/0.218/0.286 (p 0.156/0.208/0.147, n=15)".
    I verified these against the committed `p2.json` P2-e block (0.28214/0.21786/0.28571,
    p 0.1557/0.2081/0.1471, n = 15) — they are correct today, but a report generator that asserts a
    frozen number instead of reading the payload cannot detect drift. The A3 "post hoc / exploratory"
    label is also not carried into the READINGS section heading (runbook A3 says "in every output").
18. **M18 (minor) — missing-cache handling.** `model_cells` returns `{"status": "unavailable"}` for
    a missing condition cache (`ext_p2a.py:439–441`) but `run_p2a` indexes that dict unconditionally
    (`:705–712`), so S2 manifests as a `KeyError` traceback rather than a recorded stop. Also
    `payload["complete"]` (`:722–724`) treats "unavailable" as complete (`!= "no events"`).

## §4.2 / §4.3 checklist

- **A1 (fair R3u ceiling, ceiling check, R3h, raw refit, C0) — NOT satisfied.** Design construction
  is right: `fair_design` drops the T/R1 constant column and keeps all 105 dead-module columns for
  K/P (`ext_p2a.py:228–235`), it is built for both roles (`:279–281`), and R3u/R3h are fitted as
  specified (`:292–295`) on train and tested on test. `v_R3u`, `R_c^u` + CI, `information_destroying_u`
  and ρ_R3u match §3. The **ceiling check and S4 are not implemented** (B1, B2, M7), the
  **log-yield refit is missing** (M2), the raw reference's CI is the wrong contrast and its design
  alignment is positional (M5, M6). The recomputed R2 *is* comparable to the committed R2 (I traced
  `hbar`, the dead map, the alignment and the constant-column drop through `fit_ridge`/`raw_map`, so
  the 1e-9 comparison should be achievable), but its result never gates anything (B2).
- **A2 (P2-f labels) — satisfied with two text gaps.** Rungs R0<R1<R2<R3<R3u, target
  `0.8*max(ρ_R3, ρ_R3u)`, first rung reaching it, `beyond_refit = information_destroying_u`, and
  `p2.json` is never written (`assemble_report` reads it, `run_p2a` reads it). The prescribed note
  sentence and carrying the original P2-f fields are missing (M14).
- **A3 (η) — satisfied with provenance caveats.** η = ρ_R2/ρ_R3u per K level, cells with
  ρ_R3u ≤ 0.05 excluded and listed, 10 000 keyed permutations under `paper3/p2a/perm`, seed
  20261111, one-sided p with the `(exceed+1)/(n+1)` convention, the existing secondary recomputed
  from `latent.r2_eq` at K15 (reproduces 0.625), the n = 13 sensitivity, and the fixed
  two-of-three rule. Live-table x-values and the K5-only sensitivity are M13/M16.
- **A4 (F1/F2 fold and equivalence) — fold satisfied and genuinely equivalent; equivalence has a
  vacuity hole.** I diffed the fold against both runtime adapters: `_x3p_blocks` now skips energy in
  the generic loop and builds it from the particle rows exactly as `fix_x3p_energy.py` did
  (`ext_p2.py:2388–2404`), and all six `_mask_rows` call sites now pass `_trig_lookup(...)`
  (`ext_p2.py:1999–2001, 2124–2135`), exactly as `fix_p2_exact.py` did. Both unit tests are real:
  F1 compares the block against an independently fitted energy probe, F2 asserts the kept counts
  (120 vs 100) through a fake `cell_reading`, not a tautology. The equivalence compares against the
  **committed** `results/jepa/20261109_p2/{p2.json,x3p.json}` (that is what `6f14af8` fixed), writes
  through scratch `/work/dwong/p2a_equiv_ext`, and therefore does not overwrite the committed
  payloads; the `assert_authorised` bypass is scoped to the inner `x3p` call and restored in a
  `finally`, which is acceptable. Residual issues: vacuous comparison (M8) and the live-ledger write
  (M9).
- **A5 (hygiene) — partially satisfied.** The launcher accepts predictability/ext_p2/ext_p2a and
  rejects `jepa_ext_repair`, and the ext-root mapping is defensible (the addendum reads and writes
  nothing under a new returns root; it reuses `ext_p2_20261109`); the prefix hole is M10. The four
  ledger tests are genuinely derived from the committed ledger and pass — no pinned 133.6492/129.7461
  remains in any assertion (grep confirms); note the `base_hours` assertion is now self-referential
  and cannot catch a base-selection bug. The fix9000 test is a real failure, but the diagnosis is
  unrecorded (M12). The `/dev/shm` mirror is an operational step, not code — acceptable, but it must
  be **recorded**, and there is currently no record path (M1); the mirror is already absent on
  deepthought.
- **A6 (shift shares) — satisfied in substance.** Paired N/D test events via `ef.pair_events`,
  whitened by `er.whitening_matrix` on train N exactly as X3-P, SVD of `W Â`, share removed by the
  top-r left singular vectors, Frobenius share, and 20 Haar controls keyed
  `paper3/p2/x3p/<label>/<r>/<i>` — byte-identical RNG names to `ext_p2.x3p` (`X3P_NAMESPACE`);
  the control is drawn in the same whitened space, so the share is comparable. No class, no
  threshold. 12 models × r ∈ {1,3,10}. The only defect is that this mode is ungated (B3).
- **A7 (record corrections) — NOT satisfied** (B5).
- **§4.3 (review/acceptance workflow) — partially.** The gate itself is correct and complete: it
  requires the addendum runbook sha256 `f97ed81…` to match both the working tree and the acceptance
  record, an acceptance record with `review_verdict == "PASS"`, `review_path` exactly
  `docs/jepa/REVIEW_2026-11-11_P2_ADDENDUM.md` with the file present, and a clean `<P>` tree versus
  `impl_commit` including untracked files under `<P>` (`ext_p2a.py:137–192`); the executed-import
  closure covers the modules actually loaded and exempts the documented forced imports
  (`src/noise_module/`, `src/reconstruction_model/`). **But one mode bypasses the gate entirely**
  (B3), so "no execution mode can run without the acceptance" is false.

## P2 primary verdict

**Untouched.** `mixed` is not recomputed, rewritten or consulted by any code path: `run_p2a` and
`assemble_report` only read `results/jepa/20261109_p2/p2.json`; the READINGS line that prints the
primary is a literal (verified against the committed payload); no P2 reading rule or threshold
constant is modified — the `ext_p2.py` diff touches only the two `_trig_lookup` call sites and
`_x3p_blocks`, and `DRIFT_COST_MIN_M`, `REFIT_DESTROYED_M`, `P2F_SHARE`, `P2E_RHO_SUPPORT` and
`P2E_P_SUPPORT` are unchanged and imported by reference. No GPU job, no training, no data generation,
no pinned-data or `/ceph` data edit is present in the diff (`git diff --stat a38e320..6f14af8`:
eight code/test files only).

## Reproduce

```
ssh deepthought.etp.kit.edu
cd /ceph/dwong/jepa/code/jepa_work_20261107            # 6f14af8
git rev-parse HEAD
PYTHONPATH=src:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/work/dwong \
  python3 -m pytest src/jepa_noise/tests/test_ext_p2a.py -q            # 11 passed
PYTHONPATH=src:. ... python3 -m pytest -q \
  src/jepa_noise/tests/test_ext_repair.py::test_booked_gpu_hours_is_the_exact_ledger_sum \
  src/jepa_noise/tests/test_ext_shortcut.py::test_addendum_s1_uses_only_the_addendum2_cap_on_a_corrected_ledger \
  src/jepa_noise/tests/test_ext_shortcut.py::test_booked_gpu_hours_recomputes_the_v4_caps_recent_sum \
  src/jepa_noise/tests/test_ext_shortcut.py::test_addendum_s1_fails_closed_on_an_inconsistent_ledger  # 4 passed
python3 -m pyflakes src/jepa_noise/ext_p2a.py scripts/jepa_ext_p2a.py \
  src/jepa_noise/tests/test_ext_p2a.py src/jepa_noise/ext_p2.py scripts/jepa_cpu_launch.py          # clean
```
