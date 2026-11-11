# Runbook — P2 addendum: fair refit ceiling, P2-f labels, F1/F2 fold, hygiene (CPU only)

Date: 11 Nov 2026. Base: `origin/JEPA = e473986` (P2 Stage-1 results committed).
Parent runbook: `docs/jepa/RUNBOOK_2026-11-09_P2_INFOLOSS.md` (sha256 `0dbabf16…c57f`).
Extension: `paper3_p2_addendum_2026-11-11`. Results: `results/jepa/20261111_p2a/`.

---

## 0. Authority

- The author, 11 Nov 2026 13:22 CET, quoted verbatim: "按A推进，按照你说的做".
  It approves option A of the 11 Nov review of the P2 Stage-1 results: a CPU addendum with the
  R3-plus-side-information control, the P2-f label fix, the launcher fix and the F1/F2 fold, before
  P2 is written into the weekly report and the Paper 3 plan.
- The standing rule, 30 Oct 2026: "别大事小事都问我，自己找解决方案，按最好的解决方案解决".
- **Authorised:** CPU computation on cached embeddings and on the pinned probe sets; linear probe
  fits (as in P2); code, test and record changes listed below.
- **Not authorised:** any GPU job, encoder training, data generation, P2 Stage 2, any change to a
  P2 primary reading rule or threshold. The P2-e primary verdict stays `mixed`.

## 0.1 Hard rules (unchanged from P2)

- Git: push to `JEPA` from the author's Mac only. Never switch, stash, reset, clean or commit in the
  Mac main checkout `~/Documents/VERITAS` (it is on `pmt/r7-response-v3` with the author's
  uncommitted changes). Work in a separate worktree or clone.
- Shared servers: never saturate or crash them. CPU jobs under `nice -n 10`, at most 2 at once,
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` ≤ 8 each, `TMPDIR=/work/dwong`.
- Never touch `/ceph/dwong/jepa/data/`, runbook-listed checkpoints or other projects' files.
- Commit this runbook unchanged before any implementation commit; record its sha256.

---

## 1. Why this addendum

The 11 Nov review of P2 Stage 1 found that the run is valid but one comparison is not fair:

1. **R2 and R3 see different inputs.** R2 subtracts `u_e Â_rawᵀ`, where the design `u_e` for the K
   and P conditions is `dead_m(run) · h_e / h̄` (the per-run dead-module map and the event's hit
   count after the transform), and `h_e / h̄` plus an intercept for the T conditions. R3 refits on
   `z_c` alone. R2 is therefore a linear readout of `[z_c, u_e]`; R3 is not given `u_e`.
   - Consequence: on `fix9000`/`fix9001` the label-free R2 beats the labelled R3 by 40–69 mm at
     K5/K15/K30; 20 of 75 cells have v_R2 < v_R3.
   - P2-b ("information-destroying") and P2-f ("cheapest rung") use R3 as the ceiling, so both are
     measured against the wrong ceiling. P2-f prints "cheapest rung R2" next to "beyond refit".
2. **P2-e is `mixed` and stays `mixed`.** The fix models are the largest outliers and are the cells
   affected by item 1. The secondary Spearman(gain R²_eq, R²_eq,K15) = 0.625 says equivariance
   itself carries across factors while ρ_R2 does not — consistent with information loss capping
   the repair. A ceiling-normalised repair share tests this directly (exploratory, §3 A3).
3. **P2-c is a transform check, not a finding.** R1 maps T1.0w to T1.0 by construction
   (|v_R1 − v_T1.0| ≈ 1e-6 m).
4. **Open records:** F1/F2 are runtime adapters outside `<P>`; the shipped CPU launcher is
   hard-coded to another extension; 5 pre-existing test failures; the `/dev/shm` pinned-data mirror.
5. **X3-P:** invariance was never achieved at r ≤ 10, so "no price" is not interpretable. A
   descriptive reading of how much of the drift shift the projection removes explains why.

---

## 2. Inputs

- P2 embedding caches (sha-verified, `ext_p2.embedding_cache` / `n_embedding_cache`) for the 15 core
  models × {N, T0.5, T1.0, T1.0w, K5, K15, K30, P10, R1} × {train, test}.
- P2 transform info (`load_transform_info`) and the pinned probe sets for `n_hit`, run ids and the
  raw-std features.
- Committed P2 outputs: `results/jepa/20261109_p2/{p2.json, x3p.json, SUMMARY.json}`.
- Part A caches for the X3-P D/N/SWunit embeddings (as used by `ext_p2.x3p`).

A missing cache entry or a sha mismatch is S2 (§5). Do not recompute embeddings.

---

## 3. Items

### A1 — Fair refit ceiling R3u (primary item of this addendum)

For every model × condition cell evaluated in P2 (full cell and exact subset):

- **R3u:** the same vertex probe as R3 (`er.fit_vertex_probe`, same regularisation selection, same
  train/test, same paired run-cluster bootstrap) fitted on `[z_c, u_c]`, where `u_c` is exactly
  `ext_p2.r2_design` for that cell with the constant column dropped (the probe has its own
  intercept). Test on `[z_c,test, u_c,test]`. Also the log-yield probe (descriptive).
- Readings: `v_R3u`, `R_c^u = v_R3u − v_N` with CI, `information_destroying_u` (R_c^u ≥ 3 mm with a
  CI excluding 0 — the P2-b rule, unchanged), and `ρ_R3u = (v_c − v_R3u)/(v_c − v_N)`.
- **R3h (descriptive):** the same with `[z_c, h_e/h̄]` only, to separate the hit count from the
  dead-module map.
- **Raw reference (descriptive):** raw-std refit on `[raw_c, u_c]`, the encoder-independent bound
  given the same detector-state information.
- **C0, does the embedding carry the hit count (descriptive):** on N, refit on `[z_N, h_e/h̄]` and
  report `v − v_N` with CI per model.
- **Ceiling check:** R2 is a linear readout of `[z_c, u_c]`, so `v_R2 ≥ v_R3u` up to estimation
  noise. Per cell report `v_R2 − v_R3u` with its paired CI. A cell with `v_R2 < v_R3u` by more than
  5 mm with a CI excluding 0 is a **ceiling violation** (S4).

### A2 — P2-f labels against the fair ceiling

Recompute P2-f per cell with rungs R0 < R1 < R2 < R3 < R3u:

- target = 0.8 × max(ρ_R3, ρ_R3u); cheapest rung = the first rung reaching the target.
- `beyond_refit` = `information_destroying_u` (not the R3 flag).
- The note reads: "cheapest rung X reaches 80 % of the refit ceiling; R_c^u = … mm remains beyond
  refit (R4/R5 territory)" when `beyond_refit`, else "cheapest rung X".
- Keep the P2 fields; add the new ones under `P2-f_u`. Do not overwrite `p2.json`.

### A3 — P2-e, ceiling-normalised secondary (post hoc, exploratory)

The primary P2-e verdict is unchanged (`mixed`). Add, labelled **post hoc / exploratory** in every
output:

- y = η = ρ_R2 / ρ_R3u per model at K5, K15, K30 (cells with ρ_R3u ≤ 0.05 are left out; report n).
- Spearman(gain R²_eq, η), one-sided permutation p from 10 000 keyed permutations, namespace
  `"paper3/p2a/perm"`, seed 20261111.
- The permutation p for the existing secondary Spearman(gain R²_eq, R²_eq,K15) = 0.625 (same
  namespace and seed).
- Sensitivity (descriptive): the primary Spearman without `fix9000`/`fix9001`, n = 13.
- Interpretation rule, fixed now: η-Spearman ≥ 0.5 with p ≤ 0.05 in at least two K levels reads
  "equivariance predicts the share of the *recoverable* cost that label-free repair recovers" —
  a hypothesis for replication, never a replacement of the primary verdict.

### A4 — Fold F1/F2 into `<P>`

- F1: `ext_p2._x3p_blocks` builds the energy block from the particle rows (as the adapter
  `/work/dwong/p2_20261109/fix_x3p_energy.py` did).
- F2: `_cell_for_model` passes `_trig_lookup(info)` to `_mask_rows` (as
  `/work/dwong/p2_20261109/fix_p2_exact.py` did).
- Add a unit test for each.
- **Equivalence check:** rerun the X3-P energy block (4 MAE models) and the exact-subset readings
  with the folded code; every number must equal the committed `x3p.json` / `p2.json` value within
  1e-9 (absolute). A mismatch is S3.

### A5 — Hygiene

- **CPU launcher:** `scripts/jepa_cpu_launch.py` takes the extension module from an explicit
  allow-list (predictability, ext_p2, this addendum) instead of the hard-coded one; test it.
- **Tests:** the four ledger tests in `test_ext_repair`/`test_ext_shortcut` read the expected booked
  total from the committed ledger (booked sum) instead of hard-coding 133.6492/129.7461. Diagnose
  `test_ext_followup::test_label_collector_reproduces_the_committed_fix9000_rows`: fix it if it is
  an environment or path issue; otherwise mark `xfail` with the recorded reason. Target: the full
  `src/jepa_noise/tests` suite with 0 unexplained failures.
- **Mirror:** check `/dev/shm/p2_data_mirror` on every host the P2 run used. It is our own copy.
  Remove it after the last read of this addendum and record host, size freed and time. Do not
  touch any other `/dev/shm` content.

### A6 — X3-P, why invariance fails (descriptive)

For the 12 X3-P models at r ∈ {1, 3, 10}, on paired N/D test embeddings in the whitened space W:

- shift share removed: `s_r = Σ‖U_rᵀ W (z_D − z_N)‖² / Σ‖W (z_D − z_N)‖²`, and the mean of the
  same for the 20 random controls;
- the linear share of the shift: R²_eq of the X2-b fit, and the share of `‖W Â_raw‖_F²` in the top r
  singular values.
- No class, no threshold. This explains, not tests.

### A7 — Record corrections

- READINGS: state that P2-c verifies the transform (R1 = T1.0 by construction) and is not an
  empirical finding.
- READINGS: `fix9000` T0.5 is flagged information-destroying because its refit is worse than no
  repair (R_c 3.18 mm > Δ_c 2.94 mm); record it as refit noise.
- DECISIONS/STATE: the `/dev/shm` mirror (A5) and the F1/F2 fold (A4).

---

## 4. Order and machines

1. **Mac:** commit this runbook unchanged on a branch from `origin/JEPA` in a separate worktree;
   record sha256; push from the Mac.
2. **Mac:** implement A1–A7 in `src/jepa_noise/ext_p2a.py`, `scripts/jepa_ext_p2a.py` and
   `src/jepa_noise/tests/test_ext_p2a.py` (A4/A5 edit the existing files). Unit tests on synthetic
   data for: R3u design and constant-column drop, the ceiling check, P2-f_u, the η Spearman and
   permutation, the shift share.
3. **Independent fresh-context review** (delta scope: this runbook's items only). BLOCK → fix →
   re-review. Commit the acceptance record (impl_commit, runbook sha256, verdict) before any
   outcome-generating step.
4. **Deepthought (CPU):** A4 equivalence check first, then A1/A2/A3, then A6. Full test suite.
5. **Deepthought:** A5 mirror removal after the last read.
6. **Mac:** commit results, DECISIONS, STATE; push.

Budget: CPU ≤ 4 wall-h; GPU 0; ledger unchanged (134.1505); entries floor 30 000.

---

## 5. Stops

- **S1:** any step needs a GPU, training or data generation.
- **S2:** a cache entry is missing or fails its sha check.
- **S3:** the A4 equivalence check differs by more than 1e-9.
- **S4:** a ceiling violation (A1). Write nothing to READINGS; report the cell and the probe settings.
- **S5:** two consecutive review BLOCKs on the same item.
- **S6:** a new test failure caused by this addendum.
- **S7:** CPU projection > 4 wall-h or entries < 30 000.

On a stop: finish what is safe, write `EXECUTION.json` with the stop, report, and wait.

---

## 6. Outputs

`results/jepa/20261111_p2a/`: `EXECUTION.json`, `p2a.json` (A1, A2, A3 per cell and per model),
`x3p_shift.json` (A6), `equivalence.json` (A4), `READINGS_SUMMARY.md`, and figures
`p2a_ceiling.png` (v_R2 against v_R3u per cell) and `p2a_eta_scatter.png`.

READINGS_SUMMARY tables:
1. Per condition (means over 15 models): Δ_c, R_c, R_c^u, # information-destroying under R3 and
   under R3u, mean ρ_R2, ρ_R3, ρ_R3u.
2. Per model at K5/K15/K30: ρ_R2, ρ_R3, ρ_R3u, η, C0.
3. Ceiling check: number of cells with v_R2 < v_R3u (any, and beyond 5 mm with CI).
4. P2-f_u: cheapest-rung counts per condition, and `beyond_refit` counts.
5. P2-e: primary (unchanged), η secondary, R²_eq,K15 secondary with p, the n = 13 sensitivity.
6. A6 shift shares.
7. Resources, tests, mirror removal.

## 7. Report to the author

English, ≤ 1 page: stops (if any), the seven tables' headline numbers, what changed relative to the
P2 Stage-1 READINGS, open items. No interpretation beyond the rules fixed in §3.
