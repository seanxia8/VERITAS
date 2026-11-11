# Independent implementation review — Paper 3 P2, Stage 1

**Item:** `paper3_p2_infoloss_2026-11-09`, Stage 1 only (no training, no data generation).
**Runbook:** `docs/jepa/RUNBOOK_2026-11-09_P2_INFOLOSS.md` (sha256
`0dbabf1698f0deccaf1076c5a03a59c8bb9196a310cba2b1b0c741af5a56c57f`).
**Base:** `origin/JEPA = df588c6`; runbook committed unchanged at `d1442d6`.
**Final implementation commit reviewed:** `b3cc41192a8c953414237c966d48f63e2a268bf4`.
**Reviewers:** fresh-context subagents, read-only clones; the Lead did not review its own diff.

## Round history

| round | commit | verdict | blockers |
|---|---|---|---|
| 1 | `d5e12e1` | **BLOCK** | B1 R2 dead-channel design built `dead_by_run` from train runs only, so every K/P latent repair raised `KeyError` and P2-e lost all y-values; B2 X0-b asked `er.X0B_TARGETS["fix9000"]` which does not exist (the committed target is 0.219258); B3 `_metric_block` returned vertex `resolution` while the reused Part A X3 helper reads `mean_error_m`, so X3P-3 dominance never evaluated. Nine medium findings (X3P-2 CI, R1 run-id order, P2-c CI class, signed P2-a eligibility, X3P-4 Fisher rows, figure key, P2-e n handling, `assert_budget` projection factor, exactness-test non-vacuity). |
| 2 | `b4d2029` | **BLOCK** | The three blockers and all nine mediums were fixed, but the new pinned-data mirror was staged inside the GPU jobs: its Ceph I/O would be booked as GPU wall, the §7 projection omitted it, and the cumulative guard could refuse the primary `conditions` stage (N1). Mediums: mirror path/audit not recorded (N2), P2-e `n<3` still scored "mixed" (N3), mirror/R1/P10 test coverage thin (N4), report not stopping on incomplete/cap overrun (N5), concurrent mirror staging unlocked (N6). |
| 3 | `ef3b923` | **PASS** | N1-N6 resolved: only the CPU `preflight` stages (once, recorded in `plan.json` as `data_mirror` with base, per-set source manifest sha, directory and wall); `reproduce`/`new-models`/`conditions` require that base, verify it and fail closed; `gpu_set_projection` now includes the mirror verification and the §7 projection is 1.1761 h / 249 sets against the 1.3 h cap; a post-job `gpu_cap_check` STOPs on a cap overrun; `report` STOPs on incomplete P2/X0-b/cap; P2-e returns "not evaluable" for `n<3`; tests added for P10 generator exactness, the `cell_reading` R1 branch, mirror fail-closed/corruption and the inventory fail-closed paths. All §8.4 items (i)-(vi) PASS. |
| 4 | `b3cc411` | **PASS** | Delta-only check: `require_mirror_base` prefers an explicit `JEPA_P2_MIRROR` (node-local RAM) base over the recorded one, `mirror_audit` records `plan_base`/`env_base`/`used_base` in every GPU payload, and a unit test covers the preference and fail-closed behaviour. No regression of the round-3 PASS; staging remains a CPU/preflight step and the mirror content is still verified against the pinned source manifest. |

## §8.4 checklist (final)

- **(i) PASS — generator primitives and a genuine exactness test.** The §2.3
  transforms are built only from `lucid_bombs.leading_edge_time` and the imported
  `lucid_bombs.pmt_time_rule()` components (no reimplementation); the §8.3
  exactness tests generate a *second* run with the factor built into the readout
  (a `dataclasses.replace`d `GeneratorConfig` for thresholds, a zero-gain
  `DetectorState` for dead channels/PMTs) and compare the transform against it,
  never against itself. `test_threshold_transform_matches_the_generator`,
  `test_pure_hit_loss_transform_matches_the_generator`,
  `test_dead_channel_transform_matches_the_generator`,
  `test_p10_dead_pmt_transform_matches_the_generator`; the comparison is
  non-vacuous (`compared >= 1` asserted).
- **(ii) PASS — window-edge accounting and the exact-subset column.** Every
  condition reports the trigger efficiency, the number of events with
  `trig' > 0`, the `trig'` distribution and the exact-subset (`trig' <= 0`)
  reading; `p2.json` and `READINGS_SUMMARY.md` carry the exact-subset columns
  next to the full-cell readings. R1 uses the T1.0w stage's `trig'`. The T1.0w
  exact subset is small because the walk moves the trigger for almost every
  event (median `trig'` 0.05 ns, q95 0.87 ns) — reported, not hidden.
- **(iii) PASS — repairs and designs.** R1 is the label-free input repair on
  T1.0w (`q`-only re-application of the walk); R2 is label-free
  (`u_e,m = dead_m * h_e/hbar` for K/P, `h_e/hbar` + intercept for T; fitted on
  embedding deltas with the corrected `raw_map`, no truth); R3 is the labelled
  readout refit; R5 is the Mahalanobis abstention curve. No truth enters a
  label-free repair.
- **(iv) PASS — P2-e.** Spearman of the gain `R2_eq` against `rho_R2` per K
  level, one-sided p from 10 000 keyed permutations under `paper3/p2/perm`,
  eligibility (`Delta_c >= 3 mm` with a CI excluding 0) applied before the
  correlation, the n used and the excluded models reported, and the
  supported/refuted/mixed/not-evaluable class rule exactly as §5 states.
- **(v) PASS — X3-P.** Subspace from the SVD of `W * A_hat_raw` (the model's
  X2-b occupancy fit), top-r left singular vectors for r in {1,3,10};
  erasure `z' = z - W^-1 U_r U_r^T W (z - mu_N)` with W and mu_N from N train,
  applied to N/D/SWunit; 20 Haar-random rank-r controls keyed
  `paper3/p2/x3p/<label>/<r>/<i>`; X3P-1/2/3/4 with the fixed Part A thresholds
  (>=1.2 price, <=1.05 none, |Delta| <= 2 mm robust with the CI inside +-3 mm,
  Part A X3-4 dominance) and the Fisher/raw-feature descriptive row.
- **(vi) PASS — gate, closure, forbidden paths, write scope, cost projection.**
  §8.2 gate: the committed runbook sha256 and
  `docs/jepa/ACCEPTANCE_2026-11-09_P2_INFOLOSS.json` (`review_verdict == "PASS"`)
  must match; `<P>` is Part A's list plus `ext_p2`, `ext_repair`, `ext_shortcut`
  and `scripts/jepa_ext_p2.py`; the executed-import closure is re-checked at the
  end of every mode; there is no training or data-generation mode; writes stay
  under `<ext> = /ceph/dwong/jepa/returns/ext_p2_20261109`, `results/jepa/20261109_p2` and TMPDIR. The §7 GPU projection is 249 sets
  (225 condition + 10 X0-b + 14 new-model) x 13 s x 1.3 plus the mirror
  verification = **1.1761 h <= 1.3 h**; the optional six models are dropped
  (with-optional 1.5986 h); CPU projection 5.4 h <= 8 h; S1
  (booked 133.6708 + 1.3 = 134.9708 <= 160) is fail-closed on the ledger.

## Environment deviations recorded (non-blocking)

- **Pinned-data mirror for the GPU stages.** The runbook assumes the 7 Nov
  measured ~6 s per set. On the reference date deepthought2's CephFS client was
  ~0.7 MB/s with ~0.15-0.2 s per small file and its NFS client ~0.15 s per
  small file, so per-set reads alone would exceed the 1.3 h GPU cap. The
  implementation therefore stages a verified read-only copy of the four pinned
  probe sets once in the CPU `preflight` (recorded in `plan.json` as
  `data_mirror`) and the GPU jobs read a node-local RAM copy via
  `JEPA_P2_MIRROR` (both bases recorded in every GPU payload). The pinned data
  is never written; every mirrored run's `SHA256SUMS` is verified against the
  pinned source manifest; the mirror is a transient cache removed after
  execution.
- **Existing CPU launcher.** `scripts/jepa_cpu_launch.py` is hard-coded to the
  predictability extension and is outside `<P>`, so CPU modes run directly under
  `nice -n 10` with four threads and TMPDIR under `/work/dwong`, as the prompt
  specifies; the GPU modes still go through `scripts/jepa_run.sh` with no guard
  bypass.
- **Full suite.** The `src/jepa_noise/tests` suite has pre-existing,
  environment/ledger-state failures unrelated to this purely additive diff (the
  four ledger-value tests that hard-code 133.6492, the `fix9000` label-collector
  test, and a memory-dependent `test_lucid_run` test); the 26 P2 tests pass and
  the three known `test_guard` launcher tests are the only deselections.

## Sign-off

**PASS.** The implementation matches the runbook's Stage-1 scope: exact
generator-built transforms, the window-edge/exact-subset accounting, the
label-free R1/R2/R5 and labelled R3 repairs, the fixed P2 readings and the
primary P2-e test, and the X3-P projection price with its fixed thresholds and
controls. No reading rule or threshold was changed after outcomes.


## Validation evidence on the final SHA (`b3cc411`)

- Focused P2 tests: **26 passed** (`src/jepa_noise/tests/test_ext_p2.py`).
- `pyflakes` on `src/jepa_noise/ext_p2.py`, `scripts/jepa_ext_p2.py`,
  `src/jepa_noise/tests/test_ext_p2.py`: clean.
- Full `src/jepa_noise/tests` suite on Deepthought (three known `test_guard`
  launcher tests deselected): **5 failed, 869 passed, 2 skipped, 3 deselected**
  in 17:35. The five failures are exactly the pre-existing
  environment/ledger-state failures reproduced at the base commit with none of
  the P2 modules imported (`test_ext_followup.py::
  test_label_collector_reproduces_the_committed_fix9000_rows` and the four
  ledger-value tests in `test_ext_repair.py`/`test_ext_shortcut.py` that
  hard-code the superseded 133.6492 booked total). The P2 diff is additive
  only: `git diff --name-status d1442d6 b3cc411` adds three files and modifies
  no existing module or test.
- Dry runs: the `dry-run` mode exits 0 with the gate
  `gate_pending_acceptance` (pre-acceptance), closure pass, 0 setup errors,
    15/15 inventory available, GPU projection 1.1761 h / 249 sets, S1
  134.9708 <= 160 and entries 40007 -> 39407 >= 30000. The
  `jepa_run.sh --dry-run` launch preview was exercised separately; it refuses
  to start while the host guard is HOLD (by design) and starts nothing.
- Real-data CPU X0-a rehearsal on Deepthought (before acceptance, scratch ext
  root): pass=True, stops=[], data pins pass, identity 0 mismatches / 50
  samples / max time difference 0.0 ns, no no-trigger fallback events, 16
  transform bookkeeping files, trigger efficiencies 0.999-1.0 per condition,
  GPU projection 1.1761 h, S1/entries pass. The production X0-a is re-run at
  the accepted SHA.

## Sign-off

Both the round-3 review (`ef3b923`, all §8.4 items PASS) and the round-4
delta review (`b3cc411`, PASS) were performed by fresh-context reviewers with
read-only clones. The final implementation commit is
`b3cc41192a8c953414237c966d48f63e2a268bf4`.
