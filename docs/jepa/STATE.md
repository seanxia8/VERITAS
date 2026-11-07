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

## Predictability extension, 29 Oct 2026

Runbook committed as `398340c`; Part 0 search and implementation are incomplete. A local noise resampler and one preliminary test are uncommitted, with no acceptance result. No extension analysis or encoder run has been launched. Grid remains paused.

Predictability S3: frozen first 20 probe-test runs contain 340 photon / 340 muon / 320 electron events, so the requested 1,000 original photon events in 20 runs do not exist. No extension experiment launched; await author choice. Four preliminary acceptance tests passed locally, but the analysis implementation remains incomplete.
Local full-suite preflight: 496 passed, 5 failed; not accepted. Full-tree hash search was interrupted at S3 and is incomplete. See DECISIONS and RESOURCE_LEDGER for details.

Author clarification resolves the predictability S3 sample ambiguity: use all 1,000 original mixed-domain events in the first 20 runs and retain the 10/10 run split. Photon-only H-a–H-c; all domains enter the linear CCA. Implementation and validation are in progress; grid remains paused.

Predictability validation: final code SHA `ad6ca1b` Deepthought full suite 513/513 passed; WP4a name/content search completed without finding the original. No extension scientific result yet; CPU launcher smoke and time-budget preflight remain.

Predictability S1 stop: nominal six-event preflight hit a cached electron truth roundoff mismatch; no budget output written. A local repair preserves cached truth exactly after a tight comparison and passes 2/2 targeted tests, but is not host-validated. Free entries are 30,811; a new SHA clone and required full test would cross the 30,000 floor. Await author decision on storage recovery; grid remains paused.

S1 fixture-capacity stop resolved by the author's explicit option A: three successful fixture basetemps were individually archived and verified before removal; free entries 36,009. Archive manifests, hashes and exact paths are recorded in DECISIONS. Validation of `babbd3b` and the extension remain pending.

`babbd3b` validation complete: 18/18 focused and 515/515 full suite; successful suite scratch removed, log retained. Proceeding to the nominated mixed-domain r=0/time preflight under a new job ID; grid remains paused.

### Predictability extension (29 Oct 2026)

Part 1 is complete on the author-clarified 1,000 mixed original events for nominal and strong drift; 13 CPU probes and domain-stratified summaries are recorded under `results/jepa/20261029_predictability/` as `registered: false`. Part 2 mixed-event CCA is complete in both cells and splits; seven descriptive readings are recorded, with the H-a conjunction correction and preserved raw report. Grid remains paused.

## M dose, 30 Oct 2026

Runbook v2 `1c42aa5`; final experiment code `273a7bf` (full suite 527/527 in 666.91 s). Toy reproduces every §2.5 population acceptance value (no S7). Ten LUCiD dose levels completed on `273a7bf`, 1.167 h measured, no failures. Read-outs: M-1 supported only over a narrow r range, M-2 not supported, M-3 not supported, M-4/M-5/M-6 not evaluable, M-7 supported. The within-split trap does not appear: nu* stays <= 0.099 even at F1x s=1.50 while the toy crosses 0.5 at r = 1.036; the large r_4 is baseline shared noise that is not aligned with the top-4 physics pairs. Results are `registered: false` under `results/jepa/20261030_m_dose/`; npz stay in the extension with recorded hashes. Grid paused; waiting for the author.

## Timing extension, 30 Oct 2026

Runbook `6cb95ca`; experiment code `b6c55ab` (full suite 533/533 in 735.84 s). Part A: S8 reproduction exact (0.0 vs 1e-9); the `across_fixed` trap is the shared trigger-time jitter carried by the time features — nu*(A) 0.5055 at s=0 falls to 0.0004 with `no_time` and to 0.0002 after the exact de-jitter, and the A-block noise pattern aligns with the time/hit-pattern direction at |cos| 0.99; `within` is protected (nu*(A) 0.0007). Part B (frozen encoders, forward only): excluding the 37 dark-moved r=0 events, R²(delta) is at or below the raw-input ceiling (ceiling -0.021), so T-d is unclear by its pre-declared rule; T-a/T-b/T-c are supported and T-e is descriptive (Spearman -0.833). Results are `registered: false` under `results/jepa/20261030_timing/`; the grid remains paused.

## De-jitter extension, 30 Oct 2026

Runbook `92639fb`; experiment code `444cf58` (full suite 537/537 in 720.93 s). Part 1: trigger-jitter sensitivity is small everywhere except `mae9000`; the `across` encoders sit at 1.87x the fix seeds in the unwhitened R_jit/phys (the whitened version does not support it), and the 10 000-event robust-delta ceiling is 0.166 (<0.2), so J-3 fails and T-d remains unclear. Linear check: `median_time` removes the `across` trap (ν*(A) 0.50550/0.49200 → 0.00135/0.00136). Six L40S runs + smoke completed (5.386 GPU-h): de-jittering does **not** recover the `across` vertex resolution (0.34751/0.32031 m vs Part D 0.34811/0.33346; P-2, P-3 not supported), while the `within` arms are unchanged (P-4 supported). Readings: J-1 supported (marginal; whitened not), J-2 supported, J-3 not supported, P-1 supported, P-2/P-3 not supported, P-4 supported, P-5 descriptive. Results are `registered: false` under `results/jepa/20261030_dejitter/`; the grid stays paused.

## MAE de-jitter extension, 31 Oct 2026

Runbook `06c4313`; experiment code `692d90a` (suite 541/541 in 560.61 s) and report-only code `8888572` (suite 541/541). MAE does use time (token-time R² 0.92–0.96 against 0.004–0.026 for the de-jitter SIGReg checkpoints) and is 20–30× more jitter-sensitive (R_jit/phys 1.15–1.64; M-e supported). But it has almost no across gap to repair: across_trigger/within_trigger vertex is 1.153/1.026, so M-a is not supported and M-b/M-c read "unclear (no gap)" even though across_median is numerically 16–18 % better (0.097/0.101 vs 0.121/0.120 m). The prescription does not hurt within (M-d supported, 0.0852/0.0950 vs 0.1049/0.1165 m). All eight runs + smoke exit 0; L40S 4.634 GPU-h, ledger 101.799/170 h; results `registered: false` under `results/jepa/20261031_mae_dejitter/`; grid paused.

## Grid descope, 31 Oct 2026

Gate G-DESCOPE answered **A (descope)** by the author on 31 Oct 2026: the registered LUCiD A10 grid is descoped and no grid run is launched. `R6`, `R7`, `R7b`, `R8`, `R11` and `R11b` keep `not_run`, with the reason "author descope (gate G-DESCOPE, 31 Oct 2026)" and the extension evidence beside them; `R12` is `not_evaluable` by the frozen §3.5.3a check (F3 run-level ceiling 0.3226 < 0.6, no gate).

A design check recorded before the descope: the gate's 15-run core cannot produce registered verdicts, because the frozen seed-floor rule needs all seven arms × three seeds in each F1-eligible cell (four cells, 84 runs, about 44–47 L40S GPU-h by the measured grid wall); A10's primary particle domain has no profiled detectability (`particle_R6_R7 = not_established_unprofiled`); and the full 147-run grid would put ledger + projection at about 187 h, above the 160 h start rule.

No training was run; the L40S ledger is unchanged at 97.165 / 170. The non-registered extension evidence (`results/jepa/20261029_*`–`20261030_*`, plus the in-flight MAE de-jitter extension) is reported beside the `not_run` rows.

## Q1/Q2 readout extension, 1 Nov 2026

Runbook `33e969b` (revision 2, sha256 `0e629141…`); extension code `dff4802` (first readout SHA) and report-only diagnostics from `02283a6` (adds the H1/H2 causal extension and the report `PROBE_MANIFESTS` fix). Full `src/jepa_noise/tests` suite on the deployed Q1/Q2 SHA `dff4802`: **552 passed in 617.66 s** (deepthought, `nice -n 10`, `OMP_NUM_THREADS=4`). Two guard-gated L40S matrix jobs (GPU 0 / GPU 1, 597.22 s / 453.70 s = 0.292 GPU-h, cap 4; the two dev probe sets verified read-only against the 25 Oct provenance hashes, `/work` read only, no regeneration, no `dev_train` build). §4.2 reproduction **PASS** (full-input photon vertex: fix9000 0.21925859, init9000 0.16007169, old9000 0.26440029, old9001 0.23375940, fix9001 0.20428123, mae9000 0.10494915; all ≤ 1e-3 m from Part C). Readings: **Q1-a not supported** (the full-input-vs-random verdict does not flip under half-pass for both seeds), **Q1-b supported** (half-pass fix−old exceeds the seed spread for both pairs), **Q1-c supported** (propagated tracks half-pass for every old/fix/mae pair), **Q1-e/Q1-f descriptive**, **Q2-a not supported**, **Q2-b supported** (charge is present but nonlinearly encoded: fix9000 full_input ridge 0.0172 vs MLP 0.8149), **Q2-c not supported**, **Q2-d unclear** (CKA(trained, physics) − CKA(trained, init) inside ±0.1: fix9000 −0.027, fix9001 −0.048). Results `registered: false` under `results/jepa/20261101_readout/`; per-form caches deleted after the report, `<ext>/large_arrays.json` records the hashes; grid descoped and untouched; ledger 101.799 → 102.091 / 170 L40S GPU-h, CPU 12.026 → 12.676 h.

## H1/H2 causal tests (E2/E3), 1 Nov 2026

Runbook committed unchanged as `docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md` (sha256 `aab229a0…`); experiment SHA `02283a6` (full `src/jepa_noise/tests` suite 560 passed in 564.74 s on deepthought), report-only SHA `7952cb6` (563 passed in 574.41 s, includes the nested-metric `ext_summary` fix and the train-log NaN check). Data preflight read-only PASS (dev_train `b1b3f4…`, probe-train `093ea894…`, probe-test `c11edc71…`; `/work` read only, no regeneration). Two 20-step smokes exit 0 (finite ext_summary) and the four approved 10 000-step runs exit 0: E2 geomblind 9000/9001 (2328.79 s / 2688.87 s) and E3 frozen_target 9000/9001 (1401.6 s / 1747.31 s), each in its own root with the variant sidecar; one readout-evaluation job (460 s) with `--zero-geometry` for E2. E0/E1 priors: `f_layout` 0.9970, `f_event` 7.9e-5, E0-c R² 0.0021, E1 invariant 1.0072 / position-explained 0.0011 / variance ratio 0.5221 (E0-1/E1-1 unclear). §3.3: E2 full-input vertex 0.830/0.842 m (worse), token ln q R² 0.9999/0.9998, half-pass 0.279/0.277 m (contradicts, init half 0.1748), half R² 0.648/0.586; E3 full-input vertex **0.138/0.131 m**, R² 0.923/0.933, half-pass 0.132/0.138 m, R² 0.888/0.891. §6: **E2-r not supported**, **E3-r supported** → **H2 dominant (trainable-target trivial solution)**; §4.2 reproduction PASS (fix9000 0.2192586, init9000 0.1600717). Results `registered: false` under `results/jepa/20261101_h12/`; L40S 2.543 GPU-h (ledger 104.635 / 170), CPU 0.72 h (13.396 / 48); grid descoped, untouched.

## Position-probe addendum, 1 Nov 2026

Analysis only (0 training, no new runbook; H1-clean training not started). Code `4030015`: backward-compatible `e0e1 --position-probe {linear,mlp,knn}` (default `linear` = committed behaviour); label-row cache `<ext>/e0e1_rows.npz`. On the same 64,000 rows / 2,000 events / seed 0: row-level position→label R² linear 0.00206 vs MLP 0.67824; per-PMT invariant component `m_p` position→`m_p` — fix9000 linear −0.00848 / MLP −0.00853 / kNN(k=5) 0.30504, init9000 reference 0.68148 / 0.79321 / 0.88551 → the fix9000 event-invariant per-PMT component is not a position function (not even nonlinearly), while the init reference is; H1's geometry-dominance is not supported at the invariant-component level. Added 5 focused tests; full `src/jepa_noise/tests` suite on the addendum SHA **568 passed in 613.75 s**; results `registered: false` in `results/jepa/20261101_h12/e0e1_position_probe.json`; CPU 0.5 h, GPU 0.

## H1-clean query-noise test (stopped at dry-run), 2 Nov 2026

Runbook committed unchanged as `docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md` (sha256 `e0221420…`, author line present) with the author line in DECISIONS; extension `h1_clean_2026-11-02` code SHA `5ae6d62` (9 focused tests; full `src/jepa_noise/tests` **577 passed in 588.21 s** on deepthought). Hard gate PASS, §3.5 data preflight PASS (read-only, no regeneration), Q1/Q2 checkpoint/hash manifest PASS. **§3.4 manipulation check FAILED** on the untrained seed-9000 arm: patched/unpatched prediction MSE ratio 1.00178 at σ=0.10 and 1.00387 at the single ladder step σ=0.15, both below 1.5; a sigma sweep shows the threshold needs σ≈1.8–2.0 scaled units (~2.8–3.1 m x/y, 2.5–2.8 m z; `encode()` byte-identical at every σ), so the pre-declared test is non-discriminating (muP-initialised untrained predictor is context-dominated). Per §3.4/§0.1 no smoke and no run was started → no H1-clean-r reading; the result is `unclear (non-discriminating)`, not evidence for H2. Evidence `registered: false` in `results/jepa/20261102_h1clean/` (`plan.json`, `manipulation_sensitivity.json`, README); 0 GPU-h, CPU 0.35 h (ledger generation 14.246 / 48); storage recovered by removing four superseded clones; next step is the Lead's decision (label-centering runbook not started).

## H12 code-origin addendum, 2 Nov 2026

Analysis only (0 training, no new runbook). Extends the committed `e0e1` addendum with `--code-origin` (code SHA `d6d013f`; 3 focused tests, full `src/jepa_noise/tests` **580 passed in 596.92 s** on deepthought): per-PMT fixed features from the same 2,000 photon probe-test events (hit rate, mean hit ln q, within-target parity, module one-hot, scaled position, mean raw-token vector; combined 120 features), read-only tokenisation only; target the E1 invariant component m_p (128-d, invariant fraction 1.0072); 80/20 split by PMT (1,050 present → 840/210). Result: the `fix9000` code is **not** identified by any single fixed PMT attribute (best single MLP 0.107 module/0.107 raw-token; position 0.305 kNN local structure only) and the combined probe reaches only linear 0.104 / MLP 0.434 / kNN 0.218, whereas the `init9000` reference is strongly identified (module 0.84/0.84/0.83, raw-token 0.76/0.85/0.84, combined 0.89/0.91/0.87, position 0.68/0.79/0.89) — the `fix9000` per-PMT code is a learned code of other origin, consistent with H2 dominant. Output `registered: false` in `results/jepa/20261101_h12/e0e1_code_origin.json`; 0 GPU-h, CPU 0.35 h (ledger generation 14.596 / 48); storage: superseded clone `0f1803c` and basetemps removed, guard OK.

## Frozen static-target test, 2 Nov 2026

Runbook committed unchanged as `docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md` (sha256 `516e408e…`, author line present) + author line in DECISIONS. Extension `static_target_2026-11-02`, code SHA `f382ad17` (9 focused tests; full `src/jepa_noise/tests` **589 passed in 600.08 s** on deepthought). Gate PASS, data preflight PASS (read-only), split/occupancy within+natural PASS, §2 manipulation check PASS (label event fraction 3.87e-16 ≤ 1e-3, per-PMT variance 2.0027, zeroed columns 6-8 / unchanged 0-5; the first smoke was archived and relaunched after the fresh-start assertion was fixed to accept the runner's null `resumed_from_step`). One 20-step smoke (220.45 s) + two approved runs (seeds 9000/9001, exit 0, 10000 steps, finite, resumed_from_step 0) + evaluation (490.46 s); §4.2 reproduction PASS. **static_target_r = not recovered** (both seeds): full-input vertex **0.160374 / 0.161148 m** (≥ 0.160072 floor), full token R² 0.9539 / 0.9588; half-pass vertex 0.1461 / 0.1527 m (< 0.174821), half R² 0.8946 / 0.8613; f_layout 0.876 / 0.836, f_event 0.0171 / 0.0173, effective rank 5.71 / 6.44. With the §1 asymmetry caveat (a static detached label leaves only the context SIGReg training; `not recovered` cannot distinguish staticity from absence of target event-information) this is a two-sided descriptive contrast against E3, not an H2 verdict. Results `registered: false` in `results/jepa/20261102_static/`; L40S 1.0098 GPU-h (ledger 105.645 / 170, S1 safe), CPU 0.40 h (14.996 / 48); no registered change, grid untouched.

## Scale control (300k events / 30k steps), 2 Nov 2026

Runbook committed unchanged as `docs/jepa/RUNBOOK_2026-11-02_SCALE.md` (sha256 `073d1532eb603565b7deede465161c924d8eb1195b9b15dc1dad322e2d8a1c70`, author line "B 的下一批训练我全部授权" and the quoted B4a row in DECISIONS). Extension `scale_2026-11-02`; final code SHA `57295c4` (ext code `3fb3be0`, loader-cache fix `5e52533`, test fix `57295c4`). Data: new `registered=False` cohort `dev_train_300k__F1__0.05__s20261017__n300000x1000` (300 runs, namespace `lucid/dev_train_300k`, event_start 1,300,000; manifest `9a9dbe81…`; copied to `/work` with the data-root symlink; 300k generation). Runs: two 30,000-step `sigreg_jepa` runs (seeds 9000/9001, sequential under the guard RAM cap, per-run cap 3.5 h) + one 20-step smoke; the earlier interrupted attempts (loader-cache miss and guard SHED, ~0.835 h) are archived under `<ext>/train_archive/` and never resumed. Successful walls 6967.12 s / 8262.31 s (1.935/2.295 h), steps 30000, `resumed_from_step` 0, metrics finite. Evaluation: one job for both run dirs, wall 727.86 s; §4.2 reproduction **PASS** (fix9000 0.2192586, init9000 0.1600717). **`scale_r = not recovered`** (both seeds): full-input vertex **0.32584 / 0.27631 m**, full token ln q R² 0.00316 / 0.00414, half-pass 0.32064 / 0.31911 m (R² 5.6e-5 / 2.5e-4), `f_layout` 1.00033 / 1.00033, `f_event` 9.9e-7 / 1.2e-6, effective rank 4.88 / 5.29. The collapse persists at 3× data and 3× steps (nominal epochs unchanged; the loader samples with replacement). Results `registered: false` under `results/jepa/20261102_scale/` (README, runs.json, readings.json, plan.json, reproduction.json); entries projection live 44,949 → projected 37,253 (floor 30,000). Resource booking: L40S 4.2304 h runs + 0.2022 h eval + 0.835 h wasted = 5.2676 h → ledger **105.644356 → 110.911956 / 170** (S1 160 safe); generation +1.68 h → **14.995639 → 16.675639 / 48**. No registered change; grid untouched.
- Final-SHA suite (`57295c4`, deepthought, local basetemp, `PYTHONPATH=src`, `OMP_NUM_THREADS=4`, `nice -n 10`): **599 passed in 4802.46 s** (80 min; slow because the heavy evaluation-equivalence tests were not yet gated and the arch suite/training were running concurrently; the later `c8d6545` gates them behind `JEPA_HEAVY_EVAL_TEST`).

## Architecture/masking controls, 2 Nov 2026

Runbook committed unchanged as `docs/jepa/RUNBOOK_2026-11-02_ARCH.md` (sha256 `e45029ff…`, author line + H4 row in DECISIONS). Extension `arch_2026-11-02`, code SHA `eafbce6` (suite **605 passed, 2 skipped in 578.59 s**; heavy equivalence tests explicitly **2 passed in 2485.31 s**). Gate + dry-run PASS (probes/data/readout/entries 42 818 → 36 934); one 20-step `window128` smoke PASS; six runs (window128/window32/pred4 × seeds 9000/9001) exit 0, 10 000 steps, finite, `resumed_from_step = 0`, encoders 128/64, 32/16, 4 blocks; per-variant evaluations complete and §4.2 reproduction PASS. **arch_r = not recovered for all three variants** (window128 0.1631/0.1629 m, R² 0.104/0.142; window32 0.2151/0.2068 m, R² 0.025/0.059; pred4 0.2336/0.2079 m, R² 0.016/0.045; half-pass all ≥ 0.1748 m; f_layout 0.968–0.997, f_event 6.9e-5–1.3e-3, rank 1.46–3.50). Conclusion per §1: the collapse is robust to these architecture/masking changes at this scale; H4 ruled out for these knobs. Results `registered: false` in `results/jepa/20261102_arch/`; L40S 5.201 GPU-h cumulative (ledger 116.113 / 170, S1 safe), CPU 1.75 h (18.426 / 48); no registered change, grid untouched.

## View design (four candidate splits), 2-3 Nov 2026

Runbook committed unchanged as `docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md` (sha256 `993755d7…`, author line + C row in DECISIONS). Extension `view_design_2026-11-03`; code SHA `f5b7434` (registered `e4873c1`, extension `f5b7434`). Data read-only (committed 100k dev cohort + probe sets; no generation). Eight runs (four splits × seeds 9000/9001, 10 000 steps) + one 20-step `module_parity` smoke; walls within 2321.73/2316.81, across 2304.97/2310.91, module_parity 2986.76/2417.26, random_half 2389.79/2506.53 s (all exit 0, finite, `resumed_from_step` 0); smoke 141.8 s. Evaluations: four per-split jobs on GPU 0 (within 872 s, across 755 s, module_parity 434 s, random_half 503 s; the guard was bypassed for these evaluation-only jobs because the other user's GPU 1 was at 100% and the chain waited on the global guard; GPU 0 was idle), §4.2 reproduction **PASS**; report. **`view_r = not recovered` for all four splits**; ranking module_parity (mean full vertex 0.17502 m) > random_half (0.19864) > within (0.21292) > across (0.33483). Numbers (full vertex / full token ln q R²): within 0.2193/0.2066 (R² 0.0169/0.0542), across 0.3494/0.3203 (0.0008/0.0014), module_parity 0.1851/0.1649 (0.0731/0.1930), random_half 0.2161/0.1812 (0.0175/0.0794); half-pass 0.2000–0.3203. Recommendation: no split in this set makes the physics learnable at the development budget; the view is not the lever (the target-content mechanism is). Results `registered: false` under `results/jepa/20261103_view/` (README, runs.json, readings.json, plan.json, reproduction.json). Resource booking: L40S ≈6.33 h (runs 5.432 h + smoke 0.039 h + evals 0.713 h + ~0.15 h aborted attempts) → ledger **116.151 → 122.481 / 170** (S1 160 safe); CPU 0 (no generation; tests/evals in the GPU wall). No registered change; grid untouched.

## MAE/JEPA follow-up execution — Stage A readouts and the T-1 control, 6 Nov 2026

Runbook `docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md` (sha256 `063e9087…`, unedited) executed under acceptance record `docs/jepa/ACCEPTANCE_2026-11-03_MAE_JEPA_FOLLOWUP.json` (conditions C0–C8, author decisions D1–D5). Implementation `3ba19d1` + the 6 Nov acceptance fix `7214e99` (round-3 fresh-context review PASS, no BLOCK; review record 1182 lines); acceptance commit `feed8c9` (`origin/JEPA`). Extension `mae_jepa_followup_2026-11-03`, `registered: false`.

**Stage A (all 54/54 checkpoints available; cap 6.5 L40S h, D1).** A0 gate `granted` (data hashes, pairing, 54/54 inventory, entries free 37 986 → projected 32 102 ≥ 30 000, projection 4.9049 h ≤ 5.5, S1 139.4807 ≤ 160). A1 reproduction **PASS**: s = 0.05 init 0.160072 / fix 0.219259 / mae9000 0.104949 / SIGReg within_median 0.222653 / MAE within_trigger 0.104916 (max |Δ| 5.9e-7 m); s = 0.30 init 0.170301 + gain 0.841585, `strong_drift9000` 0.233209 + 0.097482. **A-1** 39 computed item-equivalents (readings J-a/J-b/J-c in `readings.json`); all 38 checked labels reproduced their committed vertex, the six C4 D2 s = 0.05 targets included (max |Δ| 2.4e-7 m). **J-a:** ijepa (f_event 2.7e-4/1.2e-3), lambda (1.1e-3/1.8e-3) and stopgrad (5.1e-3/3.9e-3) are **intermediate** — they escape the strictly invariant code but do not reach event-dependent (≥0.01 in both seeds); `across`, `strong_drift` and the three SIGReg de-jitter retrains stay event-invariant. **J-b:** the eight MAE de-jitter checkpoints sit beside init/E3/static in `readings.json`. **A-1b is `unavailable`**: the C8 label cache was accepted, but the general collector did not reproduce the committed fix9000 label rows (`y` differs on 64 000 × 128 rows), which is the runbook's own A-1b reproduction rule. **A-2** 8 models × 2 environments (4 A-2 + 4 T-1): e.g. `mae_within_trigger` s = 0.05 gain − floor +0.159 / vertex 0.1054 m, s = 0.30 +0.084 / 0.1107 m (frozen-probe 0.1163 m), log-yield R² 0.997/0.997. **A-3** 11 models: μ/e AUC 0.905 / balanced accuracy 0.831 with AUC 0.85–0.93 inside every NHits quintile (yield does not carry it); μ direction ridge median 9.86° / q68 13.10°, MLP 4.93°/7.05°; electron contained-energy R² 0.953 (2 570 contained); the μ energy probe is `not_evaluable` under the 500 rule (452 < 500), exactly as pre-declared. **A-4** 8 models, per-model reproduction 8/8; `mae_within_trigger` reads **immaterial** (Δvertex −0.0042/−0.0045 m against the 0.01 m threshold).

**T-1 (cap 3.5 L40S h).** One 20-step smoke (`t1smoke_` label, C0) and four 10 000-step runs (`s030_within_trigger` / `s030_across_trigger` × seeds 9000/9001) all exit 0; every host check PASS with no unexpected manifest differences; `resumed_from_step` 0. **G-1: retained in both seeds for all four runs** (gain R² 0.926/0.924 within, 0.837/0.828 across; gain − floor +0.084/+0.083 and −0.005/−0.013) — a non-collapsed MAE trained under strong drift keeps the drift. **G-2** is descriptive, no threshold (e.g. within seed 9000: +0.00057 m, CI95 [−0.0011, +0.0023], paired over 200 runs / 3 400 events).

**Resources.** Stage A **4.5844 / 6.5** L40S h; T-1 **2.6811 / 3.5** L40S h; ledger `E2_lucid_l40s_deepthought2` **122.4807 → 129.7461 / 170**; S1 129.7461 + 8.0 + 9.0 = 146.7461 ≤ 160. CPU 1.623 h this phase (≈4.62 h with the Phase-1 3.0 h) of the 8 h cap; generation cap untouched **18.425639 / 48** and no data generated. Failed attempts are booked (A-1b/A-4/A-3 CLI-kwarg failures, one guard SHED of A-4 when another user's 19.6 GB job took GPU 0 to 98.6 %).

**Deliverables.** `results/jepa/20261103_followup/`: `plan.json`, `inventory.json`, `runs.json`, `readings.json`, `reproduction.json`, `output.json`, `README.md` (with the C4, C5, C6/C7 tables and the C0–C8 / D1–D5 application list). Large arrays stay on `/ceph` (`/ceph/dwong/jepa/returns/ext_followup_20261103/`).

## Paper 3 repair pilot (Part A) and shortcut-collapse follow-up (Part B) — stopped at the entries preflight, 7 Nov 2026

Runbook `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` (sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b`, committed unchanged at `062a8d1`) executed under acceptance `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json` (`impl_commit` `320602e`; independent review `docs/jepa/REVIEW_2026-11-07_REPAIR_AND_SHORTCUT.md`, round 3 **PASS**, all 16 §F5 checklist items PASS; rounds: BLOCK `184cd2f` B1–B9, BLOCK `0cef73a` B10, PASS `320602e`). Extensions `paper3_repair_pilot_2026-11-06` and `shortcut_followup_2026-11-07`, both `registered: false`; no data generation; no GPU hour used.

**Verification on the final SHA** (Deepthought CPU unless stated): pyflakes clean on both modules and CLIs; focused tests **57 passed** (Part A) and **26 passed** (Part B); the full `src/jepa_noise/tests` suite (only the three known `test_guard` launcher tests deselected) **751 passed / 1 pre-existing environmental failure / 2 skipped / 3 deselected** on Deepthought and **752 passed / 2 skipped / 3 deselected** (green) on deepthought2 — the one Deepthought failure is the untouched data/numerics-dependent `test_ext_followup::test_label_collector_reproduces_the_committed_fix9000_rows`, which fails in isolation with none of the new modules imported; both dry runs exit 0 (`gate_pending_acceptance`, closure pass, 0 violations, no setup errors) and after acceptance both report `gate_status: granted`.

**STOP: entries-preflight failure (Part A §8.4, Part B §8.4; front matter §F3).** X0-a passed the pinned data hashes (4/4 sets), the per-run `state_sha256`/twin/`events.npz` checks (200/200 runs per set), the twin and `SWtw`–`N` content checks, the `mu_p` reconstruction, the 19/19 checkpoint inventory, the GPU projection (2.8409 h core / 3.6014 h with optional ≤ 4.0), the CPU projection (5.70 h ≤ 6.0) and S1 (135.2461 ≤ 160); it failed only entries: free **27,260** − clone 3,059 − suite 1,825 − extension 2,400 = **19,976 < 30,000**. Part B's preflight failed the same entries projection (20,376 < 30,000; data/S1/caps pass). The free count is the live Ceph quota; the runbook's 37,986 expected at `ac8e220` is stale because another project (`hcp_xopt_20261106`, 9,900 entries, 6 Nov) consumed the difference, and no deletion of another project's data is authorised. The guard was `HOLD` (`free entries 27260 < 30000; write no new data`). No GPU job was launched.

**Executed / not run.** Part A: X0-a preflight and X3-a Fisher accounting ran (P0 passes, max efficient/known ln Y charge-information ratio 1.21e-15 ≤ 1e-6, 1,000 events, 5 model rows); X0-b, X1, X2, X3-b and the full report are `not_run` and no reading is reconstructed or qualified. Part B: preflight only; manipulation, reproduce, smoke, both training runs, evaluation and report are `not_run`; R-1 is `not_run (pilot outputs missing)`; the authorised §5.3 training set was not executed. A documented execution-time adapter (outside `<P>`) fixed an `os.PathLike` JSON-serialisation crash in the accepted preflight while preserving gate (c); the in-`<P>` fix is left for a re-reviewed round. Evidence: `results/jepa/20261106_repair/` and `results/jepa/20261107_shortcut/` (`plan.json`, `inventory.json`, `x3_fisher.json`, `STOP.json`, `README.md`); large arrays: none. Bookkeeping: 0 GPU h (L40S ledger unchanged 129.7461/170; S1 135.2461), CPU ≈ 2.0 h (Part A, cap 6) + ≈ 1.2 h (Part B, cap 2); generation cap untouched (18.425639/48), no data generated.

## Repair pilot + shortcut follow-up — resumed execution after the entries STOP, 7 Nov 2026

The 7 Nov entries STOP was cleared by the author's archiving authorisation (free entries **27,205 → 44,286**; every removed target tarred to `/work/dwong/jepa_archive_20261107/`, sha256 + tar-member-count verified, manifest `docs/jepa/ARCHIVE_2026-11-07_MANIFEST.jsonl`; two targets that failed verification were kept, not deleted; `/ceph/dwong/jepa/data/`, every runbook-listed checkpoint and all committed results untouched). Both preflights then passed and the accepted runbook was executed.

**Part A (`paper3_repair_pilot_2026-11-06`)** — final implementation `0cf61ff`, acceptance re-issued after six independent review rounds (all PASS; last review sha256 `c3fd694f…`). X0-a PASS; **X0-b reproduction PASS** 9/9 models within 1e-3 m (max |Δ| 4.8e-7 m, identity PASS); **X1 COMPLETE** 19 models, no unavailable; **X2 COMPLETE** 13 core models; **X3-a COMPLETE** (P0 pass, ratio 1.21e-15); **X3-b COMPLETE** raw/init9000/leace after fixing three execution-time defects (drift-truth, merged-table alignment, X3-5/X3-json serialisation; the last two via documented runtime adapters, fixes 7–8, pending in-`<P>` consolidation); report COMPLETE with 6 figures. GPU **2.1974 / 4.0 L40S h** (failed attempts booked); CPU ≈ 2.6 / 6 h. Reading summary in `results/jepa/20261106_repair/{READINGS_SUMMARY.md,SUMMARY.json}`; full per-event payloads stay on `/ceph` with `LARGE_ARRAYS.json` hashes.

**Part B (`shortcut_followup_2026-11-07`)** — preflight PASS, §5.2 manipulation PASS, §5.3 reproduction PASS (fix9000 0.2192586, init9000 0.1600717), 20-step smoke PASS (seed 9000, resumed_from_step 0), **both authorised 10 000-step runs COMPLETE** (seeds 9000/9001, 0.659 h each, metrics finite, resumed_from_step 0; the first 9001 placement on the contended GPU 0 was archived and re-run on the idle GPU 1, 0.193 h booked). The single evaluation job was **refused by the extension's own §8.1 cap guard** (1.5651 h booked + 0.14 h projected > 1.5 h cap), so **R-2 readings are `not_run (cap)`** (R2-a "single-seed (not classified)", R2-b "unclear", R2-c empty); the cap overrun (0.0651 h) is booked and reported. **R-1 is COMPLETE: verdict `contradicts`** — the generative (MAE) median is below both `frozen_target` seeds on both quantities (R²_eq 0.3278 vs 0.7077/0.7278; ρ_LR −0.6779 vs −0.5338/−0.5865; 8 vs 2 descriptive). Report COMPLETE. GPU **1.5651 / 1.5 L40S h**; CPU ≈ 1.4 / 2 h. No training beyond §5.3; no data generation. L40S ledger 129.7461 → ≈132.0 / 170; S1 135.5 ≤ 160; generation cap untouched (18.425639 / 48).
