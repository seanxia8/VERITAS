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

## Round 2 (delta review)

**Commit range reviewed:** `6f14af8..21a116a` (fix `21a116a`, parent `d635e7e`; the round-1
record is `d635e7e`). Fix-only diff `d635e7e..21a116a`: 4 files, +470/−150. The full range
including the round-1 record: 5 files, +812/−150. `src/jepa_noise/ext_p2.py` is not touched by
the fix; no P2 primary reading rule or threshold changed.

**Tested SHA:** `21a116a5eec2a1482a3a82067513e5fc358f4199`, pinned in a detached host worktree
`/work/dwong/p2a_review_21a116a` (the shared clone `/ceph/...` was already at `0b2edde` when I
first reached it; `/private/tmp/veritas-jepa-20261107` advanced to `0b2edde` at 00:16:29 during
the review — see “Moving HEAD” below). Runbook sha256 re-hashed: `f97ed81…59e0a`, matches.
Host `deepthought`, `PYTHONPATH=src:.`, `OMP_NUM_THREADS=MKL_NUM_THREADS=4`,
`TMPDIR=/work/dwong/p2a_tmp`, `nice -n 10`.

**Verdict: BLOCK.** Two *new* blocking defects introduced by the fix: (R2-B1) the A4 equivalence
can never pass because `_cell_equivalence` short-circuits on an equal `"status"`, and (R2-B2)
`p2a_summary` raises `KeyError: 'refit'` on a clean `p2a` payload. The round-1 blockers B1–B5
are fixed in mechanism, but B2/B4/B5 cannot be exercised on a successful `report` until these are
fixed. One round-1 medium (M6) is not genuinely fixed, and one (M8) is defeated by R2-B1; M12 is
still unresolved.

### What I re-verified, and how

| check | command | result |
|---|---|---|
| addendum tests | `pytest src/jepa_noise/tests/test_ext_p2a.py -q` | **16 passed** (28.73 s) |
| four ledger tests + A5 launcher | the four ledger node-ids + `test_ext_cpu_launch.py` | **8 passed** (5.16 s) |
| P2 module after the fold | `pytest src/jepa_noise/tests/test_ext_p2.py -q` | **26 passed** (8.05 s) |
| fix9000 xfail | `pytest -rxX ...::test_label_collector_reproduces_the_committed_fix9000_rows` | **1 xfailed** (112.90 s) — the failure is real |
| lint | `python3 -m pyflakes` on the five changed modules (pyflakes 3.2.0) | **clean** |
| tests are genuine | 21a116a test file run against 6f14af8 code | **7 failed, 9 passed** — the 5 new tests plus the strengthened η and launcher assertions fail on the pre-fix code |
| B1 sign (helper) | `ceiling_gap_ci` demo / test | see B1 below |
| summary crash | call `p2a_summary` on a cell shaped exactly like `ceiling_cell()` | **KeyError: 'refit'** at `ext_p2a.py:1067` (see R2-B2) |
| exact-subset coverage | count committed exact readings | 105 full + 90 exact = **195** cells; T1.0w exact is `unavailable` in committed P2, so ~210 is an over-count |

Code read in full: the `21a116a` versions of `ext_p2a.py` (1325 lines), the CLI, the test file,
the launcher, and the `d635e7e..21a116a` diff. Two candidate defects were checked and dismissed:
the R2 recompute with the fair design (constant dropped) is numerically identical to P2's R2
because `fit_ridge` zero-variance-centres the constant column and the probe has its own intercept;
and `run_equivalence`’s exact-subset minimum (1000) is safely below the committed ~4846 numeric
leaves. The `0b2edde` follow-up (see below) mirrors P2's R2 design exactly but is numerically a
no-op for the same reason.

### New blocking findings

**R2-B1 — A4 equivalence can never pass: `_cell_equivalence` skips all numbers when both sides
are `"evaluated"`.** `src/jepa_noise/ext_p2a.py:925-940` returns `n_compared=0` and
`mismatches=[]` whenever `fresh` and `committed` both carry the same `status`. Every X3-P model
entry carries `"status": "evaluated"` (`ext_p2.py:2513`; all 12 committed `x3p.json` models have
it), so in `run_equivalence` the X3-P comparison at `:1014-1015` compares nothing:
`x3p_energy.n_compared` (`:1021`) is 0, and the new minimum `x3p_min = 100` (`:1024-1032`) makes
`pass=False` even when the blocks are byte-identical. Demonstrated on the host: comparing the
committed `mae_within_trigger9000` entry with itself gives `n_compared=0`; the sum over the 12
committed models is 0 < 100. Consequence: A4 always reports S3; `assemble_report` stops at
`:1283-1284`; no READINGS. This defeats the very M8 non-vacuity guard it was added for. Fix: only
short-circuit for non-`evaluated` statuses; when `status == "evaluated"`, fall through to
`compare_numbers` (and treat `status` as one more compared field).

**R2-B2 — `p2a_summary` raises `KeyError: 'refit'` on a clean payload; table 1's R_c and
destroying-R3 are not produced.** `ceiling_cell`’s returned cell (`:391-420`) has no `refit` key;
`run_p2a` adds only `P2-committed` with `rho_R2`/`rho_R3`/`r2_eq` (`:807-808`). `p2a_summary`
does `float(cell["refit"]["R_c_m"])` at `:1067`, so every evaluated cell raises. `report` calls
`p2a_summary` at `:1298`, after the stop check; a clean run (once R2-B1 is fixed) therefore
crashes, `_guard` writes only `error.json`, and no `p2a.json`/`SUMMARY.json`/`READINGS_SUMMARY.md`/
figures/`EXECUTION.json` are produced. Demonstrated on the host with a cell shaped exactly like
`ceiling_cell()` output: `KeyError: 'refit'` at line 1067. Even with `.get`, the values would be
wrong: `n_information_destroying_R3` reads `cell.get("refit")` at `:1076` and would count 0, and
`R_c_m` would be `None`, although the committed `refit` block has both. Fix: carry
`refit.R_c_m`/`refit.information_destroying` into the addendum cell (or read them from `p2` in
`p2a_summary`).

### New non-blocking findings

1. **R2-N1 (M6) — not genuinely fixed: `_aligned_rows` applies the permutation in the wrong
   direction.** `ext_p2a.py:269-287` builds `order[i] = target_position(source_row_i)` and then
   indexes the *source* arrays with it (`:286`). For a genuine reorder this misaligns: source
   values `[10,20,30]` with target key order `C,A,B` returns `[20,30,10]`, not `[30,10,20]`
   (demonstrated on the host with `_aligned_rows`). It is only accidentally correct when source
   and target are in the same order (the current case), and can `IndexError` if lengths differ.
   The raw-std refit is descriptive, so this is non-blocking, but the M6 claim is not met. The
   correct construction is the inverse permutation (`inverse[order[i]] = i`, then `array[inverse]`).
2. **R2-N2 — both figures are empty.** `eta_plot` (`:1222`) still reads `row.get(level)`, but
   `p2a_summary` now stores `row["levels"][level]` (`:1095-1099`), so the η scatter has no
   points. `ceiling_plot` uses `summary["_cells"]`, whose builder skips every cell on
   `cell.get("v_R2_m") is None` (`:1303`) — no cell has a top-level `v_R2_m` (the value is
   `cell["ceiling"]["v_R2_recomputed_m"]`, `:1307-1308`) — and the fix removed the old
   per-condition fallback, so the ceiling figure is blank too. §6 requires both figures.
3. **R2-N3 — a stopped `report` can leave stale READINGS.** The stop path (`:1293-1297`) writes
   `EXECUTION.json` and returns before `p2a.json`/`SUMMARY.json`/`READINGS_SUMMARY.md`/figures,
   which satisfies “write nothing to READINGS” for *this* run. It does not remove or mark a
   previous run’s READINGS/figures, so a results directory can show a stop plus old readings.
   Also, `_finish` (`scripts/jepa_ext_p2a.py:81-84`) writes `report.json` whenever `--out` is
   passed, so a stop with `--out` writes two files, not only `EXECUTION.json`.
4. **R2-N4 (M14) — the note is present but not the prescribed sentence.** `p2f_u` (`:551-559`) emits
   “cheapest rung R3u reaches 0.180 of the 80% refit ceiling target 0.144; R_c^u = 0.045000 m
   remains beyond refit (R4/R5 territory)”. The runbook A2 wording asks for “R_c^u = … mm” and,
   in the else branch, only “cheapest rung X”; the numbers are present, the units/wording are
   not. `P2-f_original` is kept at `:800`, so the “keep the P2 fields” part is met.
5. **R2-N5 (M12) — still unresolved.** `test_ext_followup.py:754-760` still has `strict=False` and still
   asserts the unverified “environment-level float/torch mismatch”; the cited
   `EXECUTION.json` now has a producer (M1), but no diagnostic evidence (row, dtype, max abs
   difference, torch version) was added. The test does genuinely xfail (reproduced).
6. **R2-N6 — exact-subset completeness is not gated.** `run_p2a` (`:792-798`) leaves an
   `unavailable`/`no events` exact subset in the cell but does not append it to
   `payload["unavailable"]`, and `complete` (`:818-823`) requires only
   `n_cells_checked > 0`, not the expected 195. A partial exact sweep would pass. (The committed
   data gives 195: 105 full + 90 exact, so this is a latent gap, not a current failure.)
7. **R2-N7 — the new tests do not cover the new blockers.** No test calls `p2a_summary`, `_readme` or
   `render_figures`, and no test exercises an equal `"evaluated"` status through
   `_cell_equivalence`; `test_ceiling_summary_propagates_violation_and_mismatch` uses a 2-cell
   fixture, not the 195-cell sweep; `test_ceiling_gap_ci_sign_and_violation` tests the helper but
   never asserts `ceiling_cell`’s `violation` flag. This is why R2-B1/R2-B2 passed the suite.
8. **R2-N8 — the S4 record lacks the probe settings.** `_execution_payload` (`:1241-1262`) carries the
   `ceiling_check` with the offending cell/gap/CI, but the runbook §5 S4 also asks to “report the
   cell and the probe settings”; the probe settings are not recorded.
9. **R2-N9 — minor:** `_means` (`:1042-1049`) is now unused; `_readme` uses `n = row.get("n_cells") or 15`
   (`:1120`), so a zero-cell condition would print `/15`; `CEILING_CI_M` is unused.

### Status of the round-1 findings

- **B1 (ceiling sign) — resolved.** `ceiling_gap_ci` (`:260-266`) calls
  `_difference_reading(err_r3u, err_r2)`, i.e. `v_R2 − v_R3u`; `ceiling["v_R2_minus_v_R3u_m"]`
  is the same quantity (`:409`); the violation is `gap < −5 mm and ci95[1] < 0` (`:412-413`); the
  planted 10 mm violation fires and the expected direction does not (test passes).
- **B2 (S4/1e-9 propagation) — resolved in mechanism.** `_ceiling_summary` (`:855-893`) iterates
  full and exact ceilings, reports a real `max_abs_recomputed_minus_committed_m` and a real
  `recomputed_mismatch` list; `run_p2a` turns S4/mismatch/incompleteness into `stops` and
  `pass=False` (`:824-833`); `assemble_report` re-checks (`:1279-1286`). The S4 path itself is
  real (test passes), but the successful report is blocked by R2-B1/R2-B2.
- **B3 (A6 gate) — resolved.** `shift_readings` calls `assert_authorised("shift", root=root)`
  before opening probes (`:659`); `mode_shift` goes through it; `test_shift_readings_is_gated`
  fails on 6f14af8 and passes now.
- **B4 (stop before write) — resolved.** All stops are computed (`:1272-1286`) before any
  `p2a.json`/`equivalence.json`/`x3p_shift.json`/`SUMMARY.json`/READINGS/figures write; a stop
  writes `EXECUTION.json` only (`:1293-1297`); S4 is consulted (`:1279-1280`). Caveats R2-N3.
- **B5 (A7 corrections) — resolved.** `_readme` §8 (`:1189-1195`) is generated by the report and
  contains both required statements (P2-c transform check with R1 = T1.0 by construction;
  fix9000 T0.5 refit 3.18 mm worse than no-repair 2.94 mm, refit noise). It is unreachable until
  R2-B1/R2-B2 are fixed, but the generator is correct.

**M1–M18:** **M1 resolved** (`_execution_payload` `:1241-1262`, fed by `execution_extra.json` in
`mode_report` `scripts/jepa_ext_p2a.py:208`; stop path writes it — caveat: the runbook still does
not document `execution_extra.json`, so its table-7 fields are null unless hand-written). **M2
resolved** (log-yield block `:374-390`, correct `log_yield_metrics`/`fit_log_yield_probe` use).
**M3 resolved** (table 2 has ρ_R3, `:1131-1141`). **M4 partly resolved** (ρ_R3 is now the mean of
the per-cell committed values, `:1072-1084`; the same rewrite crashes on `refit`, R2-B2, and
would lose R_c/destroying-R3). **M5 resolved** (raw CI now
`_difference_reading(raw_n, raw_r3u)` `:365-370`). **M6 NOT genuinely resolved** (R2-N1).
**M7 resolved** (full + exact scopes, 195 cells; caveat R2-N6). **M8 NOT resolved** (R2-B1
defeats it for the X3-P models; the vacuity guard does work for the exact-subset cells). **M9
resolved** (`JOBS_RECORD` redirected `:1000-1010`; `record_job` uses the module global
`ext_p2.py:977`; restored in `finally`). **M10 resolved** (exact allow-list,
`scripts/jepa_cpu_launch.py:59-84`; seven predictability modules + ext_p2 + ext_p2a; prefix holes
rejected). **M11 resolved** (`DEFAULT_NAMES`/`_out_json`, `scripts/jepa_ext_p2a.py:47-64`; shift
writes `x3p_shift.json`, report reads it `:205`). **M12 NOT resolved** (R2-N5). **M13 resolved**
(committed gain `:756-763`; all 15 committed `gain_r2_eq` are present). **M14 partly resolved**
(note present, `P2-f_original` kept; wording/units differ, R2-N4). **M15 resolved**
(`fair_design` branches on `condition.kind != "dead"`, `:231`). **M16 resolved** (K5/K15/K30
sensitivity, `:622-632`). **M17 resolved** (primary verdict and numbers from committed `p2.json`,
`:766-778` and `:1153-1165`; no literal numbers). **M18 resolved for full cells** (`:785-789`
records `unavailable` and stops; exact-subset gap R2-N6).

### Moving HEAD

The shared worktree and `origin/JEPA` advanced to `0b2edde` (“mirror P2’s exact R2 repair design
and run alignment in the fair ceiling check”) 50 s after `21a116a`, during this review. All tests
and line citations are for the pinned `21a116a`. `0b2edde` changes only `ceiling_cell` (R2 on
`u2` plus aligned run ids); it does not touch `_cell_equivalence` or `p2a_summary`. I re-ran the
two blocker demos against `0b2edde`: both persist (`n_compared=0`; `KeyError: 'refit'`). The P2
primary is untouched: the fix range contains no `ext_p2.py` change and no changed P2 threshold.

### Reproduce (round 2)

```
ssh deepthought.etp.kit.edu
# pinned at 21a116a (detached worktree; the shared clone may be ahead at 0b2edde)
cd /work/dwong/p2a_review_21a116a
PYTHONPATH=src:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/work/dwong/p2a_tmp \
  nice -n 10 python3 -m pytest src/jepa_noise/tests/test_ext_p2a.py -q        # 16 passed
PYTHONPATH=src:. ... nice -n 10 python3 -m pytest -q <four ledger node-ids> \
  src/jepa_noise/tests/test_ext_cpu_launch.py                                # 8 passed
PYTHONPATH=src:. ... nice -n 10 python3 -m pytest src/jepa_noise/tests/test_ext_p2.py -q  # 26 passed
PYTHONPATH=src:. ... nice -n 10 python3 -m pytest -q -rxX \
  src/jepa_noise/tests/test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows  # 1 xfailed
python3 -m pyflakes src/jepa_noise/ext_p2a.py scripts/jepa_ext_p2a.py \
  src/jepa_noise/tests/test_ext_p2a.py src/jepa_noise/ext_p2.py scripts/jepa_cpu_launch.py  # clean
# the two blocker demos:
#   p2a._cell_equivalence(committed_x3p_entry, committed_x3p_entry, label) -> n_compared 0
#   p2a.p2a_summary(<cells shaped like ceiling_cell()>) -> KeyError 'refit' at :1067
```

## Round 2b (final R2-design fix review)

**Commit range reviewed:** `21a116a..0b2edde` — the fix `0b2edde` (parent `21a116a`), one file,
`src/jepa_noise/ext_p2a.py`, +9/−5, all inside `ceiling_cell`. Tested SHA
`0b2edde02d5fb509d1103080069725da5e9be1a8` on `deepthought.etp.kit.edu`
(`/ceph/dwong/jepa/code/jepa_work_20261107` at that SHA; `PYTHONPATH=src:.`,
`OMP_NUM_THREADS=MKL_NUM_THREADS=4`, `TMPDIR=/work/dwong/p2a_tmp`, `nice -n 10`). The local
worktree was clean when this review started; it advanced to `402c6c4` while the review ran, so all
line citations below are for `0b2edde` unless stated.

**Verdict: PASS for `0b2edde`.** The diff is correct, minimal and does what it claims: the R2
fit/repair of the A1 ceiling check now mirrors `ext_p2.cell_reading`, and the recomputed `v_R2`
equals the committed `p2.json` value exactly (0.0 on all 15 required cells and on 78 more
full/exact comparisons across 6 models). No reading rule, threshold or P2 primary is touched.
For the record: `0b2edde` still carried the two round-2 defects R2-B1 (`_cell_equivalence`
short-circuit, `ext_p2a.py:929-940`) and R2-B2 (`p2a_summary` `KeyError: 'refit'`,
`ext_p2a.py:1071`); this diff does not touch them and I reproduced both at this SHA. HEAD has
since advanced to `402c6c4` ("fix the ... blockers R2-B1/B2"), which is outside this review.

### What I verified

1. **The diff is exactly the R2-design change.** `u2_train`/`u2_test` are now
   `np.asarray(p2x.r2_design(...), float)` (`ext_p2a.py:312-316`); the fit uses
   `run_ids_train=runs_t`, `run_ids_test=runs_e` (`:332-341`); the repair is
   `z_c,test − u2_test Âᵀ` (`:343-346`). The A1 fair design `u_train/u_test` (`:310-311`) is
   still used by R3u (`:327-328`), the raw-std refit (`:354-357`) and the log-yield refit
   (`:381-382`); R3h uses `hit_design` (`:329-330`), which is what A1 specifies (`[z_c, h_e/h̄]`;
   the brief's "R3h uses the fair design" is imprecise — the code is right).
2. **Parity with `ext_p2.cell_reading` (the source of the committed value).** Same design
   (`r2_design`, `ext_p2.py:1829-1830`), same dead map (union of `c_train` and `c_test` runs,
   `ext_p2a.py:219-225` = `ext_p2.py:1824-1828`), same `equivariance_fit` kind
   (`ext_p2.py:1841-1845`), same `dz` alignment (`:1831-1836`), same repair
   (`:1851-1853`), same frozen-error `index_ce` (`:1854`). `runs_t`/`runs_e` are the same arrays
   as P2's `np.asarray(c_train["run_ids"])`/`np.asarray(c_test["run_ids"])`: `_align_subset`
   (`ext_p2.py:1637-1657`) raises if any `other` key is missing and returns the rows in `other`
   order, and both paths normalise to `astype(str)` inside `equivariance_fit`
   (`ext_repair.py:2908-2913`). I found no remaining difference that could break 1e-9.
3. **Numerical check (reproduced, `p2a.load_inputs` → `p2a.model_cells` → `p2a.ceiling_cell`).**
   Every cell reproduces the committed value exactly; `check_abs_lt_1e-9` is True for all 15 and
   the absolute difference is **0.0** for all 15 (recomputed = committed, columns collapsed):

   | model | condition | recomputed `v_R2` = committed `v_R2` | `check_abs_lt_1e-9` |
   |---|---|---|---|
   | mae_within_trigger9000 | T0.5 | 0.10723802782487844 | True |
   | mae_within_trigger9000 | T1.0 | 0.16839765335390564 | True |
   | mae_within_trigger9000 | T1.0w | 0.17160174096185654 | True |
   | mae_within_trigger9000 | K15 | 0.13738042010700283 | True |
   | mae_within_trigger9000 | P10 | 0.11675178852243973 | True |
   | fix9000 | T0.5 | 0.2222813513911528 | True |
   | fix9000 | T1.0 | 0.34541815631072226 | True |
   | fix9000 | T1.0w | 0.34529639227011816 | True |
   | fix9000 | K15 | 0.2648302770170208 | True |
   | fix9000 | P10 | 0.23865305421486155 | True |
   | target_posfree9000 | T0.5 | 0.1266343428530203 | True |
   | target_posfree9000 | T1.0 | 0.1758298242652871 | True |
   | target_posfree9000 | T1.0w | 0.17635694387309486 | True |
   | target_posfree9000 | K15 | 0.18180427127358906 | True |
   | target_posfree9000 | P10 | 0.14786437418201903 | True |

4. **Design separation (reproduced).** For `mae_within_trigger9000`/T1.0 the fair design is
   `(3400,1)` and P2's is `(3400,2)` (the second column is the constant 1; the first columns are
   identical); for K15 both are `(3400,105)` and element-wise identical (max |difference| 0).
5. **Old-path comparison (what the fix changed).** Re-running the parent's exact logic (fair
   design + raw `c_train`/`c_test` run ids) reproduces the committed `v_R2` to 0.0 on 14/15
   required cells and 5.83e-16 on `mae_within_trigger9000`/T1.0 — i.e. the 1e-9 equality also held
   at `21a116a`. Cause: `la.fit_probe` defaults to `fit_intercept=True`
   (`lucid_arms.py:1258-1265`, used at `ext_repair.py:2915`), and `probes.fit_ridge` pins a
   constant column by setting `mean=x0`, `scale=1` (`probes.py:44-46`), so its coefficient is
   exactly 0 and the column is redundant with the fitted intercept (`probes.py:50`). The fix is
   therefore a provenance/robustness alignment (it no longer relies on that redundancy), not the
   change that makes the check pass. The commit message's "aligned run ids" is also inert: the
   old ids were already aligned.
6. **Full-surface spot check.** A sweep over all 15 models × 7 conditions (full and exact) was
   started; the first 6 models completed before it was stopped for time: 78 compared scopes, every
   absolute difference exactly 0.0.
7. **No other behaviour changed.** `git diff 21a116a..0b2edde --stat` is one file, +9/−5, all in
   `ceiling_cell`; the R3u/R3h/raw/log-yield call sites are unchanged, so `v_R3u`, `rho_R3u`,
   `information_destroying_u`, `v_R3h` and the raw/log-yield readings are bit-identical. A2
   (`p2f_u`, `:536`) reads the committed `latent.rho_R2`; A3 (`eta_readings`, `:598-599`) reads the
   committed `rho_R2` and the unchanged `rho_R3u`; A6, the gate and `report` are untouched. No
   threshold, reading rule or `p2.json` value changed (the range contains no `ext_p2.py` or
   results change). The only fields that can move are the ceiling block's `v_R2`-derived ones, and
   they are numerically identical.
8. **Tests and lint at `0b2edde`.** `pytest src/jepa_noise/tests/test_ext_p2a.py -q` → **16
   passed in 26.49 s** (includes the one-cell A1 fixture
   `test_ceiling_cell_runs_with_a_dead_condition`); `pyflakes src/jepa_noise/ext_p2a.py` → clean
   (rc 0). I did not re-run the ledger/P2 suites: the diff touches neither.

### Non-blocking findings

1. **The fix is numerically a no-op** (see 5 above). Correct and worth keeping — it makes the
   equivalence structural instead of accidental — but it must not be credited with making the A1
   check pass: the check passed at `21a116a` too. The commit message overstates this.
2. **No regression test.** The 16 tests pass at both `21a116a` and `0b2edde` and do not exercise
   the design choice (`test_ceiling_cell_runs_with_a_dead_condition` only asserts that the
   function returns). A test that spies on `p2x.r2_design` vs `fair_design` in the R2 and R3u
   paths, or a fixture cell with a known committed `v_R2`, would pin the separation. The runtime
   1e-9 gate covers it on real data.
3. **Duplicate alignment.** `_align_subset(n_test, c_test)` is called twice (`:321` and `:335`);
   pre-existing, harmless, a few extra key scans.
4. **`check_abs_lt_1e-9` is `False`, not `None`, when the committed value is absent** (T1.0w
   exact: P2's exact `latent` is `unavailable` for all 15 models; `:418-421` uses
   `... or np.nan`). `_ceiling_summary` (`:880-890`) skips the pair, so there is no gate effect,
   but the per-cell field is misleading. Pre-existing; not introduced by this commit.
5. **Brief wording.** R3h uses `hit_design`, not the fair design, matching A1; only the review
   brief's phrasing is off. No code defect.

### Explicit statement

The A1 ceiling check's 1e-9 equality holds: every one of the 15 required `model × condition` cells
reproduces the committed `latent.v_R2_m` with an absolute difference of **0.0**
(`check_abs_lt_1e-9` True for all 15), and the 78 additional full/exact scopes checked also differ
by 0.0. The fair-design separation holds in code: R2's fit and repair use P2's `r2_design`
(constant column kept for T/R1) while R3u, the raw-std refit and the log-yield refit use
`fair_design` (constant dropped for T/R1), and R3h uses `hit_design`. At the fit level the two
designs coincide numerically because the constant column is redundant with the probe intercept;
the separation is therefore structural and does not itself move any reading.

### Reproduce

```
ssh deepthought.etp.kit.edu
cd /ceph/dwong/jepa/code/jepa_work_20261107            # 0b2edde
PYTHONPATH=src:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/work/dwong/p2a_tmp \
  nice -n 10 python3 -m pytest src/jepa_noise/tests/test_ext_p2a.py -q     # 16 passed in 26.49 s
python3 -m pyflakes src/jepa_noise/ext_p2a.py                              # clean
# 15-cell check: p2a.load_inputs(...) -> p2a.model_cells(...) -> p2a.ceiling_cell(...)
#   for {mae_within_trigger9000, fix9000, target_posfree9000}
#       × {T0.5, T1.0, T1.0w, K15, P10}
#   every ceiling["check_abs_lt_1e-9"] is True; abs(recomputed - committed) = 0.0 (table above)
# old path (parent's fair design + raw run ids) on the same five mae cells:
#   0.0 / 5.83e-16 / 0.0 / 0.0 / 0.0
# blocker demos still true at 0b2edde:
#   p2a._cell_equivalence(committed_x3p_entry, committed_x3p_entry, label) -> n_compared 0
#   p2a.p2a_summary(<ceiling_cell-shaped cells>) -> KeyError 'refit' at ext_p2a.py:1071
```

## Round 3 (final delta review)

**Commit range reviewed:** `0b2edde..798b39b` — the fix commits `402c6c4` (parent `0b2edde`) and
`798b39b` (parent `402c6c4`). Fix-only diff: 3 files, +141/−43 — `src/jepa_noise/ext_p2a.py`,
`src/jepa_noise/tests/test_ext_p2a.py`, `src/jepa_noise/tests/test_ext_followup.py`.
`src/jepa_noise/ext_p2.py`, the results payloads, the P2 primary rule/threshold and the ledger are
not touched. Tested SHA `798b39b1ee92fc565c4ea9efd437fa2de87b7e6f` in
`/private/tmp/veritas-jepa-20261107` and in the host clone
`/ceph/dwong/jepa/code/jepa_work_20261107` (same SHA; the host path resolves to
`followup_20261103`). Runbook sha256 re-hashed: `f97ed8101e5d0383a85ada70d974df4bb54bc70e63865d48fe77cb879cf59e0a`,
matches. Host `deepthought`, `PYTHONPATH=src:.`, `OMP_NUM_THREADS=MKL_NUM_THREADS=4`,
`TMPDIR=/work/dwong/p2a_tmp`, `nice -n 10`.

**Verdict: BLOCK.** One blocking defect remains: the R2-B1 fix now makes the *exact-subset* leg of
A4 fail on the legitimately unavailable `T1.0w` cells, so `run_equivalence` still always reports
S3. The x3p leg itself is fixed and clean (2040 numeric comparisons, 0 mismatches); R2-B2, M6,
M14, both figure bugs and the stale-READINGS stop path are resolved.

### Blocking finding

**R3-B1 — A4 can never pass: the exact-subset leg now fails on the 15 legitimately unavailable
`T1.0w` cells.** `_cell_equivalence` (`src/jepa_noise/ext_p2a.py:936-957`) appends a mismatch for
*any* fresh cell with `status == "unavailable"` (`:954-956`), even when the committed side is also
`"unavailable"`. `run_equivalence` compares all 15 models × all 7 conditions
(`ext_p2a.py:976-989`), including `T1.0w`; the committed `exact_subset_reading` for `T1.0w` is
`{"status": "unavailable"}` for all 15 models (verified from the committed `p2.json`). Every
`T1.0w` comparison therefore mismatches: if the fresh cell is `"unavailable"` the unconditional
block fires; if it is `"no events"` or evaluated, the status differs from the committed
`"unavailable"` and the status-differ branch fires. Consequently
`exact_subset["n_cells_with_mismatch"] >= 15` and `payload["pass"]` is `False`
(`ext_p2a.py:1047-1050`), so `assemble_report` stops at `:1303-1304` and no READINGS are written.
This is the same failure class as R2-B1, moved to the exact-subset leg; at `0b2edde` the
equal-status short-circuit let these 15 cells pass vacuously, so the R2-B1 fix (`402c6c4`)
regressed them.

Reproduced end-to-end by the shared rehearsal (gate monkeypatched in-process; scratch only, no
results dir). `/work/dwong/p2a_tmp/equiv_rehearsal2.log` finished with:

```
EQUIV_PASS False
EXACT {"cells": 105, "max_abs_difference": 0.0, "n_compared": 4846,
        "n_cells_with_mismatch": 15, "n_compared_min": 1000}
X3P {"models": 4, "max_abs_difference": 0.0, "n_compared": 2040,
      "n_compared_min": 100}
EXACT_MISMATCH [{"cell": "mae_within_trigger9000__T1.0w", ...,
                 "mismatches": [{"path": "mae_within_trigger9000/T1.0w",
                                 "reason": "fresh cell is 'unavailable'"}],
                 "status_match": "unavailable"}, ...]
X3P_MISMATCH []
```

A one-model direct check agrees: for `mae_within_trigger9000`, the six non-`T1.0w` exact cells
compare 53/54/54/54/54/54 numeric leaves with 0 mismatches; `T1.0w` gives `n_compared=0`,
`status_match="unavailable"` and the mismatch `"fresh cell is 'unavailable'"`. Minimal fix: allow
equal `"no events"`/`"unavailable"` to be legitimately vacuous (as the task spec says) and fail a
fresh `"unavailable"` only when the committed side is not unavailable — or skip committed-unavailable
exact scopes in `run_equivalence` — while keeping the fresh-unavailable-vs-evaluated failure. Add a
regression test for the both-unavailable case and for the exact loop; the current 20-test suite does
neither, which is why the suite is green.

### What I verified, and how

| check | command / probe | result |
|---|---|---|
| addendum tests | `pytest src/jepa_noise/tests/test_ext_p2a.py -q` | **20 passed** in 28.70 s |
| four ledger tests + A5 launcher | the four ledger node-ids + `test_ext_cpu_launch.py` | **8 passed** in 5.75 s |
| P2 module after the fold | `pytest src/jepa_noise/tests/test_ext_p2.py -q` | **26 passed** in 7.49 s |
| fix9000 xfail | `pytest -rxX ...::test_label_collector_reproduces_the_committed_fix9000_rows` | **1 xfailed** in 116.39 s |
| lint | `python3 -m pyflakes` on the five modules (pyflakes 3.2.0) | **clean** (rc 0) |
| x3p self-comparison | loop over `p2a.committed_x3p()["models"]` through `p2a._cell_equivalence(entry, entry, label)` | **4344** numeric leaves: 510 for each of the 4 MAE models, 288 for each of the other 8; 0 mismatches |
| x3p mutation | one numeric leaf (`x2b_r2_eq`) changed by +1.0 | **1 mismatch**, path `mae_within_trigger9000/x2b_r2_eq`, difference 1.0 |
| equivalence rehearsal | `/work/dwong/p2a_tmp/equiv_rehearsal2.log` | **EQUIV_PASS False**; exact 4846 compared / 15 mismatches (all `T1.0w`); x3p 2040 compared / 0 mismatches — the x3p closure is now passed |
| R2-B2 summary | minimal `run_p2a`-shaped payload, all 7 conditions, real `ceiling_cell` key set + `P2-committed` | `R_c_m = 0.03`, `n_information_destroying_R3 = 1`, `n_cells = 1`; also `status="evaluated"` cell variant gives the same; `run_p2a:812-817` copies `refit.R_c_m`/`v_R3_m`/`information_destroying` |
| M6 | `_aligned_rows` with source `A,B,C` = `[10,20,30]`, target `C,A,B` | `[30,10,20]`; a source missing a target key raises `ValueError` |
| M14 | `p2f_u` with `R_c_u_m=0.045`, beyond refit | `"cheapest rung R3u reaches 80% of the refit ceiling; R_c^u = 45.000 mm remains beyond refit (R4/R5 territory)"` |
| figures | synthesized `summary` with `per_model.levels` and `_cells` (`v_R3u_m`/nested `v_R2`), `p2a.render_figures` in a temp dir | both PNGs non-empty: `p2a_ceiling.png` 32 905 B, `p2a_eta_scatter.png` 21 999 B; `skipped=[]` |
| stop path | `p2a.assemble_report` stop fixture (`test_report_stop_clears_stale_readings`) | stale `READINGS_SUMMARY.md`/`p2a_ceiling.png` removed; only `EXECUTION.json` written by `assemble_report`; `pass=False`, `written=False` |
| completeness | synthetic 15×7 full cells + 90 exact ceilings (no `T1.0w` exact) through `_ceiling_summary` | `n_cells_checked = 195 = 15*7*2 − 15`, matches `run_p2a:827-830` |
| closure scoping | read `ext_p2a.py:1012-1030`, `scripts/jepa_ext_p2a.py:81-93`, `ext_p2a.import_closure:104-137` | `p2x.assert_closure` and `assert_authorised` are captured/restored in `finally`; `_finish` still runs the addendum's own `import_closure`; no unscoped bypass |
| tests are genuine | current test file at `6f14af8` | **11 failed, 9 passed**; the 4 new tests fail (the helpers do not exist at that SHA) |
| tests are genuine (precise pre-fix) | current test file at `0b2edde` | the 4 new tests fail with the real bugs: `n_compared=0`; aligned `[20.0, 30.0, 10.0]`; `KeyError 'refit'` at `ext_p2a.py:1071`; stale `READINGS_SUMMARY.md` still present |
| regressions | `git diff --stat 0b2edde..798b39b`; grep for GPU/training/ledger writes | 3 files only; no `ext_p2.py`, results, P2 threshold/rule or ledger change; x3p is `device="cpu"`; no training/data generation |
| budget feasibility | recorded committed wall times: `p2.json` 2472.7 s, `x3p.json` 516.0 s; rehearsal equivalence ~26 min under shared load | CPU ≤ 4 wall-h is plausible with wide margin (the five gated modes themselves were not run) |

### Non-blocking findings

1. **`_cell_equivalence`'s `statuses_match` expression has an operator-precedence bug**
   (`ext_p2a.py:941-942`): `fresh_status is not None or committed_status is not None and
   fresh_status == committed_status` parses as `fresh_status is not None or (...)`, so
   `statuses_match` is `True` whenever the fresh side has any status. It does not change the
   current pass/fail because a differing status already appends a mismatch, but it is misleading
   and should be parenthesised. The unconditional fresh-unavailable block (`:954-956`) is the
   actual R3-B1 cause and should be scoped.
2. **The CLI stop path still writes `report.json` in addition to `EXECUTION.json`.** `_finish`
   (`scripts/jepa_ext_p2a.py:81-84`) writes the mode payload whenever `--out` is given, before the
   closure/pass checks; a stop therefore leaves `EXECUTION.json` + `report.json` in the results
   dir. `assemble_report` itself writes only `EXECUTION.json` and removes the eight stale files
   (`ext_p2a.py:1314-1325`), so the stale-READINGS bug is fixed; this is the round-2 R2-N3
   second half, still open, and is not stale readings. Demonstrated with `_finish` on a synthetic
   stop: rc 2, `files ['report.json']`.
3. **The new tests miss the both-unavailable exact case.** `test_cell_equivalence_is_not_vacuous`
   tests `no events`/`no events` (allowed) and `unavailable`/`evaluated` (fails), but not
   `unavailable`/`unavailable`; `test_cell_equivalence_compares_evaluated_numbers` only tests
   `evaluated`/`evaluated`. No test drives `run_equivalence`'s exact loop, which is why R3-B1
   passes the 20-test suite. Add both.
4. **The completeness gate is count-only.** `run_p2a:827-830` requires
   `n_cells_checked >= 195`, but does not pin *which* 15 exact scopes are absent; a different set
   of 15 missing exact ceilings would still satisfy the count. The current committed data gives
   exactly 195 (105 full + 90 exact; `T1.0w` exact is unavailable for all 15). Low risk, but a
   per-scope check would be stronger.
5. **M12/xfail evidence citation is stale.** The `test_ext_followup.py` xfail reason now carries
   useful measurements (`max|dy| = 4.41e-6`, `mean|dy| = 4.97e-7`, torch 2.5.1+cu121) but cites
   `results/jepa/20261111_p2a/EXECUTION.json`, which does not exist in the worktree (nor anywhere
   under `/ceph/dwong/jepa/code` or `/work/dwong`); the test remains `strict=False`. The
   environment-level claim is still not backed by a produced evidence file.
6. **Cosmetic / pre-existing:** `_means` (`ext_p2a.py:1061`) and `CEILING_CI_M` (`:65`) are
   unused; `_readme` uses `n = row.get("n_cells") or 15`, so a zero-cell condition would print
   `/15`; `run_equivalence`'s inner x3p closure is forced to `{"pass": True}` rather than evaluated
   against the addendum closure, so a direct programmatic caller of `run_equivalence` (not via the
   CLI) does not run the addendum closure check (the CLI `_finish` does). None affects the current
   readings.

### Status of the requested fixes

- **R2-B1 — NOT fully resolved.** The x3p leg is genuinely fixed: the committed self-comparison
  gives 4344 numeric leaves, a mutated leaf fails, and the rehearsal's x3p leg compares 2040
  numbers with 0 mismatches (so the new `x3p_min = 100` is satisfiable). But the same function's
  unconditional fresh-`"unavailable"` mismatch now breaks the 15 legitimate `T1.0w` exact cells,
  so A4 still cannot pass. This is blocking finding **R3-B1**.
- **R2-B2 — resolved.** `ceiling_cell:423` carries the committed `refit` block, `run_p2a:812-817`
  copies `R_c_m`/`v_R3_m`/`information_destroying` into `P2-committed`, and `p2a_summary`
  (`:1085-1098`) reads them; the minimal payload returns numeric `R_c_m = 0.03` and
  `n_information_destroying_R3 = 1` instead of `KeyError: 'refit'`. The real code path was read.
- **M6 — resolved.** `_aligned_rows` (`:269-286`) emits source arrays in target key order:
  `[10,20,30]` with target `C,A,B` becomes `[30,10,20]`, and a missing target key raises
  `ValueError`.
- **M14 — resolved.** `p2f_u:548-560` emits the runbook wording with millimetres:
  `"cheapest rung R3u reaches 80% of the refit ceiling; R_c^u = 45.000 mm remains beyond refit
  (R4/R5 territory)"`.
- **Figure bugs — resolved.** `eta_plot` now reads `row["levels"][level]` (`:1243`) and the
  ceiling scatter is built from the nested `v_R2_recomputed_m` in `assemble_report:1331-1336`
  (`_cells` carries `v_R2_m`); both synthesized-summary PNGs are non-empty and contain points.
- **Stop-path stale READINGS — resolved in `assemble_report`.** On a stop, eight stale outputs
  (`READINGS_SUMMARY.md`, `p2a.json`, `equivalence.json`, `x3p_shift.json`, `SUMMARY.json`, both
  figures, `report.json`) are unlinked before `EXECUTION.json` is written (`:1314-1325`); the
  fixture test passes. The CLI wrapper's extra `report.json` is non-blocking finding 2 above.

No other blocking findings were found. The x3p closure change in `798b39b` is safe: the
`p2x.assert_closure`/`p2x.assert_authorised` overrides are restored in the `finally` block
(`ext_p2a.py:1022-1030`), and the addendum's own closure check still runs in `_finish`
(`scripts/jepa_ext_p2a.py:86`); the rehearsal confirms the inner x3p now gets past the old
`import-closure STOP` and produces 2040 clean comparisons.

### Reproduce (round 3)

```
ssh deepthought.etp.kit.edu
cd /ceph/dwong/jepa/code/jepa_work_20261107            # 798b39b
PYTHONPATH=src:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/work/dwong/p2a_tmp \
  nice -n 10 python3 -m pytest src/jepa_noise/tests/test_ext_p2a.py -q          # 20 passed in 28.70 s
PYTHONPATH=src:. ... nice -n 10 python3 -m pytest -q \
  src/jepa_noise/tests/test_ext_repair.py::test_booked_gpu_hours_is_the_exact_ledger_sum \
  src/jepa_noise/tests/test_ext_shortcut.py::test_addendum_s1_uses_only_the_addendum2_cap_on_a_corrected_ledger \
  src/jepa_noise/tests/test_ext_shortcut.py::test_booked_gpu_hours_recomputes_the_v4_caps_recent_sum \
  src/jepa_noise/tests/test_ext_shortcut.py::test_addendum_s1_fails_closed_on_an_inconsistent_ledger \
  src/jepa_noise/tests/test_ext_cpu_launch.py                                  # 8 passed in 5.75 s
PYTHONPATH=src:. ... nice -n 10 python3 -m pytest src/jepa_noise/tests/test_ext_p2.py -q  # 26 passed in 7.49 s
PYTHONPATH=src:. ... nice -n 10 python3 -m pytest -q -rxX \
  src/jepa_noise/tests/test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows  # 1 xfailed in 116.39 s
python3 -m pyflakes src/jepa_noise/ext_p2a.py scripts/jepa_ext_p2a.py \
  src/jepa_noise/tests/test_ext_p2a.py src/jepa_noise/ext_p2.py scripts/jepa_cpu_launch.py  # clean
# R3-B1 demo (rehearsal, 26 min; scratch only):
tail -20 /work/dwong/p2a_tmp/equiv_rehearsal2.log   # EQUIV_PASS False; EXACT 4846/15 mismatches; X3P 2040/0
# x3p self-comparison: 510*4 + 288*8 = 4344 numeric leaves; mutated leaf fails
# exact one-model check:  mae_within_trigger9000/T1.0w fresh unavailable, committed unavailable,
#   n_compared 0, mismatch "fresh cell is 'unavailable'"
# at 0b2edde the four new tests fail with n_compared=0 / [20,30,10] / KeyError 'refit' at :1071 /
#   stale READINGS; at 6f14af8 the current test file is 11 failed, 9 passed (helpers absent)
```

## Round 4 (final micro-review)

**Commit range reviewed:** `798b39b..f19106d` — the single commit `f19106d` (parent `798b39b`).
Fix-only diff: 2 files, +4/−1 — one condition in `_cell_equivalence`
(`src/jepa_noise/ext_p2a.py:954`) and one assertion added to
`test_cell_equivalence_is_not_vacuous` (`src/jepa_noise/tests/test_ext_p2a.py:291-293`).
`git diff --name-status 798b39b..f19106d` is exactly those two files; `ext_p2.py`,
`scripts/`, the acceptance/closure plumbing, the ledger and `results/` are untouched
(`git diff --stat 798b39b..f19106d -- src/jepa_noise/ext_p2.py scripts/ docs/ results/` is empty).
Tested SHA `f19106d6e724b318835a13c684f70590c7593a40` in
`/private/tmp/veritas-jepa-20261107` (`ext_p2a.py` blob `1909851`, test blob `4f257b4`), worktree
clean apart from this review file. `docs/jepa/RUNBOOK_2026-11-11_P2_ADDENDUM.md` sha256
re-hashed by Round 3 and unchanged since: `f97ed8101e5d0383a85ada70d974df4bb54bc70e63865d48fe77cb879cf59e0a`.
The host clone `/ceph/dwong/jepa/code/jepa_work_20261107` and the rehearsal log
`/work/dwong/p2a_tmp/equiv_rehearsal2.log` were **not mounted** on this reviewer's host, so the
log could not be re-read; every number below was regenerated independently from the committed
`p2.json`/`x3p.json`.

**Verdict: PASS.** R3-B1 is closed exactly as specified, with no other behavioural change.

### What I verified, and how

| check | command / probe | result |
|---|---|---|
| diff is exactly the described hunk | `git diff 798b39b..f19106d` | one condition at `ext_p2a.py:954` (`fresh_status == "unavailable"` → `... and committed_status != "unavailable"`) plus one test assertion; nothing else |
| addendum tests | `pytest src/jepa_noise/tests/test_ext_p2a.py -q` (Python 3.12.7, pytest 9.2.0, numpy 2.5.4) | **20 passed** in 11.76 s |
| targeted equivalence tests | `pytest ... -k cell_equivalence -v` | **2 passed** |
| lint | `python3 -m pyflakes ext_p2a.py tests/test_ext_p2a.py` (pyflakes **4.1.0**) | **clean** (rc 0) |
| both-unavailable, all 15 models | `p2a._cell_equivalence({"status":"unavailable"}, committed_T1.0w_exact, label)` over `p2a.CORE_MODELS` | **mismatches `[]` for all 15** |
| committed T1.0w exact status | read `p2.json[*].conditions["T1.0w"].exact_subset_reading` | **15/15 `status="unavailable"`** (reason `ValueError: five run-grouped folds need at least five runs`) |
| fresh-unavailable vs evaluated | `_cell_equivalence({"status":"unavailable"}, {"status":"evaluated"}, "x")` | mismatches: `fresh/committed status differ` **and** `fresh cell is 'unavailable'` (still fails) |
| genuinely vacuous | `_cell_equivalence({}, {}, "x")` | mismatch `no numeric leaves compared (vacuous)` (still fails) |
| equal "no events" | `_cell_equivalence({"status":"no events"}, {"status":"no events"}, "x")` | mismatches `[]` (correct) |
| cross-status no-events/unavailable | fresh `no events` vs committed `unavailable`, and the reverse | both mismatch (`status differ`); no silent pass |
| committed exact numeric leaves | `p2a.compare_numbers(cell, cell)` summed over 15×7 committed exact cells | **4846** (= 54×76 + 53×14 + 0×15) — exactly the rehearsal's `n_compared` |
| exact leg parity simulation at current code | `_cell_equivalence(committed, committed)` over 15×7 cells, aggregate as `run_equivalence:1001-1008` | `cells=105, n_compared=4846, max_abs_difference=0.0, n_cells_with_mismatch=**0**` |
| x3p leg parity simulation | `_cell_equivalence(committed_x3p, committed_x3p)` over `er.MAE_MODELS` | `models=4, n_compared=2040, max_abs_difference=0.0, mismatches=[]` |
| pass arithmetic | `run_equivalence:1047-1050` thresholds (exact ≥1000, x3p ≥100) | `payload["pass"] = **True**` |
| old-rule contrast | same parity simulation under the `798b39b` rule | **15** cells with a mismatch, all `reason="fresh cell is 'unavailable'"` — the R3-B1 symptom, now gone |
| regressions | diff scope, `grep` for `assert_authorised|import_closure|assert_closure|device|gpu|train|ledger` in the diff | none; gate/closure untouched; no GPU/training/ledger/data-generation path; no dead code (the edited branch is reachable and pyflakes-clean) |
| round-3 run unaffected | `git status --porcelain` before/after | only ` M docs/jepa/REVIEW_...md` (the Round 3 section); Round 4 appended below it, nothing rewritten |

### Is the A4 equivalence now passing?

**Yes.** Under the current code the exact leg is `n_compared=4846 ≥ 1000`, `max_abs_difference=0.0`,
`n_cells_with_mismatch=0` (the 15 `T1.0w` exact cells, previously the only failures, are now
equivalent), and the x3p leg is `n_compared=2040 ≥ 100`, `max_abs_difference=0.0`, `mismatches=[]`.
Every `payload["pass"]` conjunct at `ext_p2a.py:1047-1050` is satisfied, so `run_equivalence`
returns `pass=True` and `assemble_report` no longer stops at `:1303-1304`. The rehearsal's own
aggregate numbers (exact 4846 / 0.0, x3p 2040 / 0.0, quoted in Round 3) are reproduced exactly;
its 15 exact mismatches were all `reason="fresh cell is 'unavailable'"` on `T1.0w`, and the
committed `T1.0w` exact cell is `unavailable` for all 15 models, so the new guard converts
exactly those 15 to equivalent and leaves the other 90 cells unchanged.

### Is the both-unavailable rule correct?

**Yes.** Two `"unavailable"` cells carry no reading on either side, and the committed payload
records the unavailability as legitimate (`ValueError: five run-grouped folds need at least five
runs`, i.e. 3 train / 2 test events — under the five-fold minimum), so treating them as equal is
the correct semantics and is not vacuous-passing, because `status_match="unavailable"` is recorded.
The guard is narrowly scoped: a fresh `"unavailable"` against an **evaluated** committed dict still
produces two mismatches, `{} vs {}` still fails as vacuous, and any cross-status pair still fails
via the status-differ branch. The change therefore closes the last hole without weakening the check
that matters.

### Non-blocking findings

1. **Equivalence is status-only, not reason-aware** (`ext_p2a.py:954`). A fresh cell that is
   `unavailable` for a *different* reason — e.g. `{"status":"unavailable","reason":"condition cache
   missing"}` from `p2x._cell_for_model` (`ext_p2.py:2110`) — would be treated as equivalent to the
   committed `{"status":"unavailable","reason":"ValueError: five run-grouped folds..."}`. The
   committed cell stores its `reason`, so the check could compare reasons and still accept the
   legitimate pair. This is fail-open only for an already-unavailable committed cell; every
   committed *evaluated* cell still fails, so it cannot hide a numeric regression. Suggestion, not a
   blocker.
2. **Latent "no events" branch — fail-closed.** If the fresh exact `T1.0w` test subset were empty,
   `cell_reading` returns `{"status":"no events"}` (`ext_p2.py:1789`) rather than `unavailable`, and
   that mismatches the committed `unavailable` through the status-differ branch (verified above). So
   the residual risk is a loud A4 failure, never a false PASS. At the tested SHA the cell producer is
   byte-identical to `798b39b` (the diff touches only `_cell_equivalence`), where the rehearsal
   measured fresh = `unavailable` for all 15, so the risk is latent only.
3. **Pre-existing `statuses_match` operator-precedence quirk** (`ext_p2a.py:941-942`) remains, carried
   over from Round 3. It cannot produce a false pass today: whenever the two statuses differ, a
   mismatch is appended before the vacuous branch is consulted. Cosmetic/defensive only; unchanged by
   this commit.

### Reproduce (this host)

```
cd /private/tmp/veritas-jepa-20261107            # f19106d, clean
PYTHONPATH=src:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 -m pytest src/jepa_noise/tests/test_ext_p2a.py -q              # 20 passed
python3 -m pyflakes src/jepa_noise/ext_p2a.py src/jepa_noise/tests/test_ext_p2a.py  # clean
# both-unavailable (15 models) -> [] ; {} vs {} -> vacuous ; unavailable vs evaluated -> mismatch
# committed exact numeric leaves = 4846 ; parity simulation gives pass=True
```
