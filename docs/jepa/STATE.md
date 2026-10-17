# STATE — JEPA study (single source of truth)

Updated: 2026-10-17 23:40Z by Lead. Registered at `151f26e`; WP2-v4 at `10693eb`.

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
- **Next**: LUCiD arm objectives + anchor penalty + the four probe forms (implementer running),
  then the WP5 trainer/cohort wiring, the 500-step Titan smoke, the scaling benchmark (2k/10k Titan,
  40k L40S — deepthought2 rejoins there), then the development cohort and main grid.
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
