# STATE — JEPA study (single source of truth)

Updated: 2026-10-29. Parts A–C complete; Pre-grid Parts A–E complete as nonregistered checks; report ready; grid paused.

## Pre-grid continuation (29 Oct 2026)

- Part C clean rerun passed; `complete:true`, anc100 beta 100, all expectations independently verified within 0.001 m. Evidence and complete table: `results/jepa/20261029_part_c/`; old failure retained; evaluation-mismatch reading not supported. Part C Titan charge 2.590016667 GPU-h.
- D0 smokes and both D1 floors passed; strong-drift cohorts match nominal event IDs and seeds. Evidence: `results/jepa/20261029_d0_smokes/`, `results/jepa/20261029_d1_partial/`. All ten D2 runs complete under `04f9edd`; `results/jepa/20261029_d2_partial/` includes the table, verified evidence and pre-declared readings. Both strong-drift seeds fall below their new gain floor, so drift-too-weak is not supported. No D2 jobs or reservations remain.
- Part E: fixed-delay smoke and all nine runs complete, 10000 steps, batch128; full cohort delay0.01. Complete CPU aggregate has 9/9 runs, no missing items, registered:false; evidence and comparison in `results/jepa/20261029_bridge_delay/`. Plain/anchored nuisance R² 0.432029/0.622117; descriptive B1 and R7 fail. Bridge total including smoke 3.318761 Titan GPU-h (<8). No jobs or reservations remain.
- The original frozen WP4a beta file is absent; its beta10/config/original SHA are independently corroborated by three registered anchored manifest rows and the development report. The nonregistered descriptor `configs/decisions/wp4a_beta10_delay_descriptor_2026-10-29.json` records those sources and explicitly is not the original; it was deployed via GitHub and used by the anchored extension. Author was asked for the original's path; no registered decision changed.
- A remote CPU test at `888596c` inadvertently wrote pytest temporary files in `/tmp/pytest-of-dwong/pytest-99` outside the authorized root. It was interrupted, its files preserved, and the constrained re-test uses `/ceph/dwong/jepa/tmp/`; see DECISIONS.
- All new outputs `registered:false`; no registered grid decision or grid launch. Live resource charges and reservations: `docs/jepa/RESOURCE_LEDGER.json`. Final report and comparison are recorded alongside the evidence.

## Pre-grid handoff (28 Oct 2026)

- Part C six-checkpoint queue failed on 26 Oct with partial output, exit 1: anchored arm construction requires beta; no six-item inspect.json exists. Case (e): preserve evidence and stop; no relaunch or D–F.
- Failure evidence and SHA256 manifest: `results/jepa/20261026_pregrid_resume/failure_20261028/`. Runbook Titan charge 1.412416667 GPU-h including the failed queue; grid remains paused, all extensions registered:false.

## Pre-grid checks (25 Oct 2026, historical record)

- Current execution is **RESOURCE_HOLD_QUOTA**, not a numerical S2 stop. Grid remains paused; no grid decision file or grid launch.
- Author patch SHA256 verified and applied at `2cadfd8`; final extension code `3ecdc5c` pushed through the explicitly authorized GitHub destination. S4 cleared; no-touch diff empty; runbook bytes unchanged.
- **Part A PASS:** full suite483/483 (492.68s), final deepthought2 CPU smoke12/12; both reproduction tables pass0.1-degree tolerance with max differences<1e-9 degrees, transferred hashes verified. Interrupted preflight attempts are preserved and excluded from acceptance.
- **Part B complete:** separate `registered:false` fixed-plain decision; both anchored arms beta1000; anchor_only none_qualified and descriptive option-A policy follows1000. Original decisions preserved.
- **Part C blocked by infrastructure:** init9000 evaluation hitOSError122 before saving probes; no numerical expectation verdict. Launcher and guard final-status writes also hit quota. Preserve the failed attempt; no retry or new job until quota recovery and fresh live guardOK.
- Read-only quota inventory:502724/500000 entries (2724 over); need>=32724 entries released to reach30000 free. Four transition clones total10407 entries, insufficient; no deletion performed. Author input requested on resource restoration or a scoped cleanup list.
- **Parts D/E pending.** No training, strong-drift generation or delayed bridge launched. New GPU charges:0.1125 Titan h (conservative upper-bound charge),0 L40S h. Global conservative ledger: Titan6.6353h, L40S84.409940h; L40S ledger+projections142.977123h<160, but resource HOLD blocks starts.
- Checkpoint/config hashes remain in `results/jepa/20261025_pregrid_checks/provenance.json`; report/evidence in that directory. All new outputs remain extensions (`registered:false`), with no registered verdict or dataset replaced.


## A10 status (23 Oct 2026)

- **A10 in force** at registration `ab8c916` despite the stale §3.8 "DRAFT" header.
- **Calibration:** Gate V passed on 10 held-out contained 500 MeV μ⁻ events (module pattern 0.902,
  log pattern 0.815, yield ratio 1.006, time 0.157 ns); calibrated config
  `lucid_bombs_v2_calibrated.json`.
- **Smoke/evaluation:** mixed 500-step smoke and mixed `smoke --evaluate` with per-domain floors
  passed; domains photon/muon/electron/particle.
- **Detectability:** v1 withdrawn. Photon A9 profiled re-run: F1 0.2574/0.6835/0.8898, F3 0.3226
  (R12 `not_evaluable`), F2 0.9190, T 0.9857, G6 false. Particle profiling not implemented;
  ceilings marked `lower_bound_unprofiled`; particle R6/R7/R12 not established.
- **Dev step 2:** ridge, run_mean; pooled reference 0.482004 m; frozen.
- **Dev step 3:** β=100 for anchored and anchored_dark; anchor_only none; frozen. Per-domain β table
  missing because `dev_cell` evaluated photon only.
- **Contained probe-test counts:** muon 452 → R13 `not_evaluable`; electron 2570 evaluable; photon
  not applicable. Energy ranges unchanged.
- **Grid:** 165 runs split 87/78; at 10:00Z grid-0 12/87 complete, grid-1 0/78; both running.
- **Ledger:** charged 54.413992 L40S GPU-h; projection 52.631150; total 107.045142 ≤160/170;
  no A10.6 fallback.
- **Paths:** `/work/dwong/jepa_a10_work/` used under the author's 22 Oct authorisation; recorded
  as a deviation from the `/ceph/dwong/jepa/`-only rule. `dcdf6ce` author-approved.
- **Next:** keep grid running; implement particle-domain profiling before reporting particle R6/R7;
  add per-domain dev β table if required; final report only after grid.

Updated: 2026-10-20 19:40Z by Lead. Registered at `151f26e`; WP2-v4 at `10693eb`.

## Where we are

- **WP0-v4: complete.** Registration `151f26e`; WP3 artifacts `dab95d8`; hygiene `6399a4d`;
  decisions/status/state `6f41b52`; CPAL dates verified (Recent Spotlight **18 Jan 2027 23:59
  AoE**); prediction status v4 written; pushed (`origin/JEPA` = registration chain).
- **Phase 0 preflight: complete** (`PREFLIGHT_A8.md` + the two host reports, `38c750f`). Both
  hosts reachable (FQDN `*.etp.kit.edu`), `/ceph` mount OK, GitHub fetch of the pushed SHA OK.
  Environments chosen read-only (§6.1 step 1): deepthought `/home/dwong/anaconda3`
  (torch 2.5.1+cu121, sm_50 cubin → Titan sm_52 OK); deepthought2 `/ceph/dwong/transformer/.venv`
  (torch 2.5.1+cu124, sm_89). No new env (quota). LUCiD generation → Deepthought CPUs (no JAX GPU).
- **Quota: G-QUOTA open** (`docs/jepa/GATE_G_QUOTA.md`, `786b6c7`). Shared `/ceph/dwong`: 500 000
  entries, 468 056 used, **31 944 free**, write-stop at 30 000 → 1 944 usable. Blocks only the
  storage volume of generation/E2 returns; E1, code, guard, probes continue. No deletion performed.
- **WP2-v4: committed** (`10693eb`): A4 law (base ρ = 0, A4 θ SD, 13-cell grid, channel primary,
  displacement D), population tables 13×4; D matches §A within 0.051°, γ_n exact, v3 law reproduces
  v3 §A. Fresh review (review-wp2v4) in progress.
- **WP3 E1a: full grid done** (630 rows, 5 seeds, n up to 1e5, 3 workers, ~4.4 min). Preliminary
  aggregate (unreviewed): R1 E1a clause passes (all 6 cells ≤5°); R2 E1a clause fails 2/6 cells
  (saturation of the angle at high γ_load); R3 rows not implemented yet; R3b fails (sample 32°
  vs population 1e-15 rad); R4a 32/42 cells ≤5°; R4b 33/42 at β=10. NOT final: R1/R2 need their
  E1b clauses; no result is reported until the light independent recompute after E1.
- **WP4a bridge: COMPLETE - B1 FAILS.** 12/12 registered runs (10 000 steps, batch 128, A4 g=0.02,
  channel split, nonlinear); beta = 10 selected on the independent development cohort. Combined
  read-only aggregate: plain nuisance R2 0.525128 vs anchored 0.515705, per-seed paired
  differences +0.08280/-0.01625/-0.03828, seed floor 0.0512676 -> B1 fail (seed 0 only,
  |dmean| 0.0094 << 0.1025); physics kept (anchored/plain MSE/CRLB 0.984/0.337/0.981); anchored
  CKA 0.99843 vs 0.94064. Amplitude-preservation criterion not met (post-time fails all 12; tokens
  0.90-0.91 only for sigreg/anchored). Results: docs/jepa/WP4A_BRIDGE_RESULTS.md; aggregate in
  results/jepa/wp4a_grid_combined/; 6.52 Titan GPU-h total. B1 = fail in PREDICTION_STATUS.json.
- **Guard: deployed and live on both hosts.** Two fresh reviews (final: all findings resolved),
  137 tests. Dry runs 12:18-12:29Z on both hosts passed (level OK, cephfs_xattr, unattributed
  <20 MiB, window warm, no status JSON, no SHED). Live guards in tmux `jepa_guard` since
  12:29-12:30Z. One <=10-min GPU probe per host ran through `jepa_run.sh` with no bypass:
  Titan 10.55 s (0.0029 GPU-h), L40S 126.07 s cold-cache (0.035 GPU-h); torch 2.5.1+cu121 /
  +cu124, capabilities (5,2) / (8,9), TF32 disabled, determinism 0.0.
- **E1b running locally** (attempt_001, W=2, 65 rows, ~5 h projected, started 12:37Z). E1c done
  (`not_evaluable`, frozen deep modes; dynamics init check requested). E1a final = attempt_007.
  Next after E1b: regenerate PREDICTIONS.md, then the E1 light independent recompute (fresh agent).
- **WP4b: scaffold in progress** (detector-state laws F1/F3/F2/F2'/R11b/T, nominal-twin contract,
  run/ID/storage structure, §3.5.3a Fisher/ceiling numerics), labelled
  `readout_synthetic_unrepaired`. **Two dependency findings**: the pinned LUCiD revision
  `eaf2e89` is not available locally (only `86a1871` in /ceph/dwong/ORACLE/external/LUCiD), and the
  PMT branch is still at synthetic 76-sensor fixture geometry, so the physical WCTE 105x19 map
  required by §3.5.3 does not exist yet. Physical binding waits for the author-named PMT SHA;
  G-PMT by 31 Oct if it has not arrived.

## Milestones reached (2026-10-17)

- **WP0-v4, preflight, guard, WP2-v4, WP3 E1 (E1a/E1b/E1c), WP4a bridge: complete.**
  E1 verdicts: R1 fail, R2 fail, R3 not_evaluable, R3b pass, R4a fail, R4b fail, R5 not_evaluable.
  B1 (bridge) fail. All recorded with evidence in PREDICTION_STATUS.json; `report E1` is
  byte-identical on repeated runs and E1 == all.
- Author answers executed: **G-QUOTA (A)** (+9 322 entries, 39.2k free, gate closed); **LUCiD
  pinned clone** at `/ceph/dwong/jepa/external/LUCiD` (eaf2e89f) with the WCTE geometry verified
  (1995 PMTs / 105 mPMT / 19 per module; 1-based pmt_id_in_mpmt; mpmt_id sparse with 45 absent).
- Author rules implemented: clean-tree gate for registered runs (E1 runner + host launcher),
  not_run resume fix, grid re-execution guard.
- Diagnostics committed: docs/jepa/E1_BRIDGE_FINDINGS.md (finite-sample floors, R3 conditioning,
  amplitude information ceiling).
- **E1 light independent recompute: ACCEPT WITH FIXES, fixes applied (`1d3753a`).** R2 now uses
  the registered seed-mean angles (E1a 5/6, failing cell theta0.1_eta3; R2 stays fail); R3b control
  wording corrected; `--rerun-rows` added for provenance repair. E1b provenance limitation recorded
  (8 rows from a pre-rule dirty attempt) and its repair is queued: re-run exactly the eight
  `g0_line0_s0..s4` + `g0p005_line0_s0..s2` rows under the clean SHA once the working tree is clean.
- **WP4b generator on the real WCTE geometry: committed (`fd14311`).** Path-configured geometry
  reader with hash verification, photon bombs, synthetic unrepaired readout, nominal twin,
  runs/storage/manifests, benchmark CLI; 34 tests, full suite 327 pass. Synthetic-fixture benchmark
  202-207 events/s, 972 hits/event, 1.07M-event volume = 1.4-1.5 h on one process (0.19% of the
  48 h cap). Real-geometry benchmark requested from the host operator.
- **E1b provenance repair COMPLETE** (`attempt_005` at clean `fd14311`, registered_run true,
  registered_sha recorded, `src_scripts_status` empty, no source change): 8/8 rows pass,
  0.556 worker-h; E1 cumulative 8.68 h of 24 h. Aggregate/PREDICTIONS draw the clean latest
  records; values unchanged; PREDICTIONS byte-identical twice.
- **WP4b §3.5.3a detectability check FROZEN on the real WCTE geometry.** Y_lo 6249.708098,
  Y_hi 541560.544764 (fallback step 1 from 320k), median yield 58177.275; F1 ceilings 0.2616 /
  0.6847 / 0.8914 at s = 0.02 / 0.05 / 0.10 (s = 0.02 excluded from R6/R7); F3 0.3243 even at
  s_p = 0.10 -> **R12 not_evaluable, no gate**; F2 0.9210 at a = 0.50 (**F2 selected**, F2' unused);
  T 0.9860; N_min fine at Y_lo (within 106.5, across proxy 97.5); **G6 not triggered**, R6/R7
  evaluable. Frozen into `configs/lucid_bombs_v0.json` (`frozen_by_detectability_check`) and
  DECISIONS. Engine extended to emit the F2-branch `f1_joint` row for R11 (`aa012d8`).
  Host confirmed both: detectability v2 with the F2-branch `f1_joint` (105 modules, median 0.6819,
  min 0.5360) and the frozen-yield benchmark (49.5 events/s, 1010.5 hits/event, 0.375 h on 16 cores
  = 0.78 % of the 48 h cap). Both JSONs committed under
  `results/jepa/20261017_wp4b_detectability/`.
- **WP4c committed (`3eb31ea`)**: `lucid_views.py` (Hilbert `across` cyclic contiguous half = 52
  target modules; `within` 9/10 with the 1-based field; N_min redraw/exclusion counters; balanced
  occupancy ablation) and `point_encoder.py` (per-PMT log-charge tokens, serialised window
  attention, 4 pre-norm d=128 blocks, read-out before any final norm, mask embedding, predictor).
  14 tests, leakage exactly 0.0, amplitude fixture R² 0.9990, 911 k parameters, batch-32 CPU peak
  ~2.8 GiB (4× inside the Titan cap). Tie-breaks and the training-only N_min scope in DECISIONS.
- **WP5 arm/probe layer committed (`1ecfeed`)**: seven §3.5.6 arms + dark variant, the anchor
  penalty (dark overlay + per-module rescale), the four probe forms (run-mean primary), floors and
  R11 leakage helpers; 14 tests; Monte-Carlo readout covariance max error 4.25 %. **v_dark
  corrected** from the scaffold 0.0 to the generator's dark-charge variance
  0.004578 PE² (λ_dark(1+σ_spe²)), so `mae_chi2`'s C_m is positive definite at μ=0 and no variance
  floor is applied; generator/detectability configs aligned; 63 related tests pass; the frozen
  detectability JSON will be re-run to match the committed config.
- **A9 follow-ups**: R5-ext executed (balanced init: 7/10 rows evaluable vs 0/10; γ=1.0 2/5 wins →
  fail, γ=0.5 2/5 evaluable → not_evaluable; labelled, no retune, R5 stays not_evaluable).
  B1-ext β development cohort running on the Titan (3 runs, β ∈ {10,100,1000}, commit `46c0adb`).
- **WP5 LUCiD runner committed (`66ae509`)**: §3.5.6 cell enumeration (main 162, secondary 39,
  development 17), per-cell training protocol, cached generation, four probe forms + floors,
  R6/R7/R7b/R11/R11b/R12 wiring, dev steps 1–4 (2–3 blocked), smoke/scaling/grid-chunk/aggregate
  modes; 9 tests (86 with related suites). Open questions recorded, not decided (s_p=0 cells,
  R12 cell, write-once control, dev step 4, probe-form rule, reference v_group, dark-window
  convention, twin probe wiring; R8/robustness scored but not emitted).
- **WP5 review: ACCEPT WITH FIXES, now fixed (`d645e5f`).** P0 defects (undefined name, unthreaded
  `physics_method`, missing metrics→fail instead of `not_evaluable`, three verdict-math errors,
  unenforced optimizer numerics) fixed and tested; P1 items largely closed (R8/law-T recorded
  `not_evaluable` with reasons, oracle-twin probes on the twin, dev step-3 decision flow, automatic
  resume, recorded 5k/3-seed fallbacks, complete floor-cache identity, registered reporting);
  `ijepa_ema` target encoder corrected to target-only tokens with the rest masked. Full package
  suite **411 passed**. LUCiD training numerics frozen (mirroring `bridge_v1.json`).
- **PMT branch merged** (`codex/junjie-pmt-response` at `0fba034`, merge commit `b6d95c9`): accepted
  by the author for development; fixed per-module response patterns (group_parity gain odd modules
  +3%, odd-module timing width +5%) are **fixed detector response** present in the readout and the
  nominal twin (the twin cancels only injected F1/F3 gains); dark counts are **off** in this
  response, so everything is labelled **`readout_dark_off`** and nothing from it is reportable;
  G-PMT is resolved for development, reported runs wait for the R-6 SHA (author will name it).
  Reference `v_group = 0.2172409115942029` is available in `configs/lucid_pmt/reference.json`.
- **Author instructions 2026-10-18 (verbatim in DECISIONS):** re-run the detectability check on the
  **merged readout with F3 marginalised** (variance += s_p²·μ_p²) before freezing [Y_lo, Y_hi],
  F2-vs-F2′ and s_p, recording old vs new ceilings; a **fresh R2 recompute** of the estimator change
  (`1d3753a`) against v3 §2.4/v4 §4, verdict recorded beside R2 in `PREDICTION_STATUS.json`;
  R5-ext after the freeze (project cost first; E1 8.7/24 h); B1-ext ≤ 4 Titan GPU-h in parallel.
- **B1-ext β development complete: β = 10 selected** (max ratio 0.952 ≤ 1.1; β=100 and 1000 fail);
  anchored 3-seed grid + beside-B1 aggregate running on the Titan (B1-ext ~1.56 of 4 GPU-h used).
- **Merged-readout integration committed (`28e24c8`)**: parity gain (odd dense modules +3 %) in
  the measured readout and the nominal twin (twin cancels only injected F1/F3 gains), odd-module
  timing-width factor, dark counts off with hard guards, **`readout_dark_off`** everywhere,
  `v_dark = 0` in `mae_chi2` (softplus keeps C_m positive definite; no floor), dark bank all-zero
  so **R7b is `not_evaluable`** on this readout, and the detectability variance is
  `μ(1+σ²) + v_dark + s_p²μ²` with in-force `s_p = 0.10`. Previous [Y_lo,Y_hi]/F2/F1 decisions are
  `superseded_pending_merged_readout`. Expected: 99 LUCiD tests, 425 full suite.
- **R2 recompute verdict recorded** (fresh reviewer, verdict (a): seed-mean is the registered
  estimator; R2 = 5/6 failing θ0.1/η3; 4/6 median is a non-registered post-hoc sensitivity) beside
  R2 in `PREDICTION_STATUS.json`, whose R1–R5/R3b rows are now the evaluated statuses.
- **PMT R-6 `b341037` merged (`2c932ee`)** — 1 kHz dark counts, 1 µs window [-200, 800] ns, ~2 dark
  hits/event; the readout is now **`readout_pmt_r6`** (dark on). The LUCiD stack is restored at
  `989773c`: `v_dark = 0.00109 PE²` in the ceilings and `mae_chi2`; the dark-window anchor draws real
  dark windows; **R7b evaluable again** with the dark-only draw's measured embedding change recorded
  beside it; **s_p restarts at the registered default 0.05** with the ladder order Y_hi → F2 → s_p.
  Full suite **432 passed**; four LUCiD suites 106 passed.
- **Detectability FROZEN on the R-6 readout** (decision JSON committed under
  `results/jepa/20261018_wp4b_detectability_r6/`, sha256 `1e4407ac…`): ceilings old→new recorded
  (max shift +2.2e-3, no pass/fail change); **Y_lo 6298.436828, Y_hi 539 433.964527**, median yield
  58 288.856; ladder 4 steps (Y_hi → F2 a=0.2 → F2 a=0.5 → s_p 0.05→0.10); F1 {0.05,0.10}; F2
  a=0.50; F3 fails at 0.325 → R12 `not_evaluable` no gate; G6 false; occupancy fine at Y_lo;
  readout `readout_pmt_r6`, v_dark 0.00109.
- **R5-ext re-run after the freeze**: 7 pass / 3 not_evaluable (identical, deterministic);
  γ=1.0 2/5 wins → fail, γ=0.5 2/5 evaluable → not_evaluable; E1 CPU 8.94 of 24 h. B1-ext closed as
  a development β trade-off (β=10; 2.4484 of 4 GPU-h).
- **Titan smoke + scaling passed** (`3e739b6`): 1.59 s/step, 2k/10k fit (764 MB / 3.69 GB peak).
  **L40S stage passed** with a read-only scipy/sklearn PYTHONPATH overlay (torch pin kept,
  equivalence: probe max diff 0.0, scipy path bit-identical): smoke **0.23014 s/step**
  (0.21388 steady), peak 5 365 MiB; 40k scaling **fits** (14.6 GB peak).
- **G2 = Option E** (L40S cap 170 GPU-h, registered protocol kept; shared-server rules unchanged;
  fallback to option A only if ledger+projection reaches 170 h). **TF32 rejected on speed**
  (1.224× < 1.5×); fp32 kept, 0.027 GPU-h. **L40S overlay** (read-only scipy/sklearn PYTHONPATH,
  torch 2.5.1+cu124 pin kept) with `PROVENANCE.json`, equivalence verified (probe diff 0.0, scipy
  path bit-identical), re-verified before each batch.
- **Two engineering defects found and fixed by host runs:** (i) `loader_cache_runs` defaulted to 12 →
  3.07 s/step thrash on 100-run cohorts; now 128 → steady **0.217 s/step**, 0.603 h per 10 000-step
  run; (ii) probe evaluation materialised 19 GiB float64 per form and was SHED at 66.1 GiB; now
  **streamed sufficient statistics** (`5f83806`), projected eval peak ~518 MiB, numerically
  equivalent (R² rel diff ≤2.5e-14).
- **s_p coherence (`615421c`) and detectability re-frozen at the in-force marginalisation**
  (`3dd4715`): ceilings shift ≤0.0035 (F1 0.02 0.2574, 0.05 0.6835, 0.10 0.8898; F3 0.3228; F2
  0.9189; T 0.9855), no pass/fail change, **yields unchanged** (Y_lo 6298.44, Y_hi 539 433.965,
  median 58 288.856), branches unchanged (F1 {0.05,0.10}, F2 a=0.50, F3 null → R12 not_evaluable,
  G6 false). Dataset cache regenerated at s_p=0.10 (old archived).
- **Dev step 2 = ridge** (superseded by the author's token-time ruling) and **dev step 3 stopped**
  cleanly at the author's word (2.5857 GPU-h, all 17 dev runs kept as superseded evidence, record
  sha256 `743b0c1d…`; partial anchored β vertex values 0.30604/0.28374/0.25750 m for β=0.1/1/10).
  L40S **5.4574/170** (3.8606 superseded); CPU generation 0.4452/48; entries 34,861 free.
- **A9 token-time implementation landed (`471909a`)**: token time = digit − NHits trigger from the
  **imported** readout rule (lazy import; clear error and **dataset generation refuses** until the
  author-named PMT SHA), emission offset uniform [−50,+50] ns as **truth only**, prompt window
  [−20,+60] ns with drop/count, exact-shift invariance test (dark off: bit-identical tokens and
  trigger shifted by exactly the constant; dark on: source-digit invariance + recorded
  dark-moved-trigger rate), ToF estimator unaffected. Full suite **445 passed, 10 skipped**
  (dataset-generating tests explicitly pending the merge). No datasets generated.
- **PMT SHA `5f4ecc0` merged (`4e07c0c`, author-named, supersedes b341037) and the A9 token time is
  live (`d073639`)**: the generator imports `readout.nhits_trigger_time` **and**
  `readout.prompt_window_mask` (never reimplemented), token time = digit − trigger after the
  `[−20,+60]` ns cut, emission offset U[−50,+50] ns truth-only, trigger-relative tokens with
  absolute times kept additively, counters recorded. Full suite **456 passed, 0 skipped**.
- **A9 development cohort COMPLETE and the β table frozen (20 Oct 2026).** Step 2 = **ridge**
  (sigreg_jepa 0.32363 m = 2.272× the reference 0.142446 m; mae_mse 0.11349 m; `fit_config` grid 5,
  v_group 0.2172409115942029). Step 3 (rule: largest β with vertex resolution ≤ 1.1 × plain
  0.32363 m): **`sigreg_jepa_anchored` → β = 1000** (0.20788 m), **`sigreg_jepa_anchored_dark` →
  β = 1000** (0.21010 m), **`anchor_only` → `none_qualified`** (best 0.44891 m). Both anchored arms
  suppress the module-gain nuisance to ≈0 from β = 0.1; `anchor_only` keeps it positive at low β and
  never reaches the physics threshold. Decisions committed under
  `results/jepa/20261020_wp5_dev_cohort/` (`8c6f5de3…`, `00b83993…`). The cohort job finished exit 0
  (6.9142 GPU-h) and the resume path is proved (10 000 / 500 / 7 917 checkpoints), with the CUDA-only
  R7b dark-arm crash fixed (`122de8f`) and tested.
- **Option A applied (author G2 standing rule):** ledger 32.8460 of 170 GPU-h + measured grid
  projection 134–154 GPU-h reaches the cap, so the grid will run at **3 seeds / 5 000 steps**
  (`fallback_active: true`, `registered: false`), via the named derived config
  `configs/lucid_run_v0_optionA.json` (`e74922b`; the registered default and its 457-test suite are
  unchanged). Expected 159 runs, grid ≈60 GPU-h, ledger + grid ≈93 of 170 h.
- **Author G2 amendment (20 Oct): grid runs at ladder rung (1)** — 10 000 steps, 3 seeds for every
  main arm, secondary retained (`lucid_run_v0_rung1.json`, sha `d5197c6d…`, 165 runs, `fallback_rung 1`);
  option A stays committed and unused; start only if ledger + measured projection ≤ 160 GPU-h (it is:
  **32.846 + 118.3 = 151.1 h**). β labels: anchored β = 1000 `edge_selected`, `anchor_only`
  β = 1000 `beta_not_qualified` (descriptive control).
- **Dev floors (20 Oct) — the §3 answer:** random-init **module-gain run-mean floor 0.39438**,
  propagated 6.4e-05, physics 0.16188 m. Trained plain `sigreg_jepa` run-mean **−0.00396**,
  `mae_mse` **0.31470**, anchored −0.00416 — **every trained module-gain R² is below the floor**, so
  no arm (including plain JEPA) carries F1 above its random-init floor at development scale: no F1
  trap for JEPA here (unlike the bridge's 0.525), and **R6/R7 may come out null**; the limitation is
  representation, not detectability (the §3.5.3a ceilings 0.68/0.89 are above the 0.6 requirement).
  The anchor improves vertex resolution monotonically with β (0.32363 → 0.20788 m at β = 1000).
- **A9 registered in the runbook** (`45a2303`, sha256 `9f20042c…` verified, committed unchanged);
  differences vs DECISIONS/code listed in DECISIONS (grid volume 165 vs the text's 159; detectability
  JSON citation/ceilings; ToF reference input wording; `fallback_rung` gap now fixed in code at
  `b4f76d5`). Cite §3.7 as **A9** from now on; companion design doc is v0.3.
- **GRID RUNNING at rung (1).** G1 job A launched at `368e1c2` (21 runs, `main_s002_across`, GPU 0,
  45 h budget, 17:23:57Z); 165 runs total; start decision ledger + projection 151.1 ≤ 160 GPU-h.
  Quota: reclaim 457 entries (`RECLAIM_2026-10-20_EVIDENCE.json` `68c99011…`) → free 33 976; guard
  `free_entries_start_min` temporarily 26 000 (`GUARD_PROFILE_RELAXATION.json`, SHED minimum 10 000
  unchanged; revert after the grid). Job B follows once the G1 datasets complete; first-cell report
  next.
- **Next**: freeze the WP5 numerics from the review, then the real-geometry 500-step Titan smoke,
  the scaling benchmark (2k/10k Titan, 40k L40S — deepthought2 rejoins), the development cohort,
  then the 201-run grid in guard-sized chunks.
- Standing rule: registered runs start from a git worktree pinned to the clean SHA (main tree free
  for development).
- In flight: A9 follow-ups (B1-ext <=4 Titan GPU-h; R5-ext CPU).

## Running jobs

- deepthought: none. Guard live (`jepa_guard`, from code/6468582), level OK, 0 SHED ever; GPU
  baseline restored. `code/work` = mutable working clone at `bbcded2` (clean); `code/89af5af`
  kept as the probe record.
- deepthought2: none. Guard live, OK, 0 SHED; jjanik's GPU1 unchanged.
- Local: E1b **detached** chain (`/tmp/e1b_chain.py` pid 56883, runner pid 56978, post companion
  pid 57774), attempt_003 at 41/58 rows (20:38 local, ~256 s/row, ETA ~21:45), peak RSS 1.93 GB,
  host free ~3.7 GB. Long local runs must be detached (two attempts were lost to agent-turn
  reaping). Do NOT commit src or run heavy suites while it runs (its finish-time source check
  compares src/jepa_noise/*.py hashes).
- **Pending commit**: a hardened read-only `aggregate` mode (strict manifest/probes validation,
  duplicate/unknown-arm refusal, `--complete-run` required for damaged status, `--allow-partial`
  forces not_evaluable, optional `--data-root`, streaming CKA, `--out` cannot be inside a source
  root) is in the working tree on top of `ad75100`; 21 tests claimed, to be verified + committed
  after E1b finishes (test RAM budget). The official B1 result stands from the `cd33ff1` run and
  re-running the hardened aggregate on the host will reproduce the same numbers with better
  provenance.
- Titan: **β development cohort complete**; **β = 10 selected** (scores 0.104152 / 0.104057 /
  0.097206); recorded in DECISIONS. The 12-run bridge grid is launching as four jobs of three runs
  (sigreg_jepa, anchored at β=10, mae_chi2 in parallel on GPUs 0/2/3; the parallel-attention
  ablation as job D when a slot frees), ~25-30 min/run, outputs under
  `returns/wp4a_grid_*`. Data root cache warm (~2 GB, 56 entries).
- **Preliminary WP4a finding**: on the benchmark run, the amplitude-preservation probe is
  representation-dependent — read-out tokens mean R² 0.912 (criterion met) but the post-time-block
  representation 0.808 (criterion < 0.9, random-init floor 0.738). Registered criterion therefore
  not met for sigreg_jepa seed 0 on the post-time representation; to be confirmed across the grid
  and reported as a finding, not retuned.
- **GPU evidence so far**: corrected 10-batch equivalence PASSED both hosts (worst rel 2.9e-6 /
  2.4e-6); 1-step CUDA bridge smoke PASSED on the Titan (4 arms, 14.4 s, anchored arm clean).
- **Quota floor**: free entries ~30,552 = 552 above the §6.2 write-stop. Generation blocked until
  G-QUOTA; beta cohort + bridge runs to be kept to a minimal file footprint.

## Queue

- deepthought: guard dry run (10 min) → live guard → ≤10-min probe job (`scripts/jepa_probe_env.py`)
  → 10-batch equivalence comparison vs the Mac.
- deepthought2: same, after deepthought proves the path.
- Then: bridge smoke on a Titan, scaling benchmark, LUCiD generator.

## Guard status at last check

- Not deployed. No `jobs/` dir yet. No HOLD/SHED events. `INFRA_LOG.md` not created.

## Next three actions

1. When E1b completes: commit the E1 runner/report/config/results, regenerate PREDICTIONS.md, and
   commission the E1 light independent recompute (§5).
2. When the bridge runner lands: commit, in-place update `code/work` to the new SHA, run the beta
   development cohort (3 x 10k steps) then the bridge grid (9 runs), harvest, and compute B1/R7.
3. G-QUOTA answer needed before LUCiD generation; meanwhile WP4b's blocked items stay blocked.

## Open gates

- **G-QUOTA** (storage volume; options A–D, A recommended: delete our own failed venvs).
  Blocks: LUCiD generation data volume and E2 returns when the write-stop is reached.
- Awaiting (not gates): PMT SHA (blocks reported LUCiD results); drift ranges from Junjie
  (defaults in DECISIONS.md).

## Budget

- E1 CPU: 0.995 h charged + 20261017_wp3_e1a (~0.06 h at 3 workers); cap 24 h.
- LUCiD/Titan: 0 of 300 GPU-h. LUCiD/L40S: 0 of 80 GPU-h. LUCiD generation CPU: 0 of 48 wall-h.
- GPU-hours used so far: 0.

## Last verified SHA on hosts

- Both hosts: `6f41b52` visible via `git ls-remote` at 11:43 UTC. No clone of the current chain on
  either host yet (only the old skeleton `72add60e` on deepthought).

## Environment / ledger

- Local: `/private/tmp/jepa-venv-cpu311/bin/python` (3.11.11, torch 2.5.1 CPU).
- Hosts: see above; guard and jobs must use `/home/dwong/anaconda3/bin/python3` (3.12.2).
- Ledger: `RESOURCE_LEDGER.json` (+ `v4_caps` block: Titan 300, L40S 80, LUCiD gen 48 CPU-h).
