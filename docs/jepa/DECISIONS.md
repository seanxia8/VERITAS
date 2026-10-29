# DECISIONS — JEPA study (append-only)

Runbook v4: `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`. One line per engineering decision
(v4 §5). Author approvals are quoted verbatim with date; nothing is inferred.

**Format.** `YYYY-MM-DD — decision — basis/authority.` Entries are never edited; a reversal is a
new entry that cites the old one.

---

## Author decisions and instructions received (verbatim, dated)

- **2026-10-17 — Launch prompt v4 issued.** Flags: `PUSH_JEPA=yes`, `GPU_ALLOWED=yes`,
  `DEEPTHOUGHT_GPUS=4`, `DEEPTHOUGHT2_GPUS=2`, `CONDOR_NEMO2=no`, `OPTIONAL_WPS=WP7`.
  Authority: "If this conflicts with those files, they win." (launch message to the orchestrator).
- **2026-10-17 — PMT readout, not yet named.** Author: "I have not yet named the PMT-branch SHA.
  Develop on the unrepaired readout, labelled `readout_synthetic_unrepaired`, and produce no
  reported LUCiD results until I add the SHA to `DECISIONS.md`." → §3.5.2 [A8] development path.
- **2026-10-17 — Drift ranges pending Junjie.** Author: "s, s_p and a are still to be confirmed
  with Junjie. Use the runbook defaults, logged in `DECISIONS.md`, until I change them, and change
  them only before training." → defaults listed below.
- **2026-10-17 — Ambiguity rule.** Author: "When unsure whether something is an engineering choice
  or a specification change, treat it as a specification change: write a gate with lettered
  options, and keep working on everything that does not depend on it."
- **2026-10-17 — Runbook/companion files are the author's.** "Commit them unchanged."
  "Leave `docs/LUCID_MODEL_LINEUP_2026-10-09.md` untouched; it is mine."

## Registered specification (WP0-v4)

- **2026-10-17 — Runbook v4 registered unchanged**, commit `151f26e`
  `jepa: register runbook v4 (author revision)`. SHA-256:
  - `RUNBOOK_JEPA_NOISE_v4_2026-10-16.md` = `3a489d305df6679db234e24f5b76ebb33a0e6740185e3d9e2a458c6830d8c428`
  - `AGENT_PROMPT_JEPA_v4_2026-10-16.md` = `ca68c61879b381feed36fa7f47725a9ea549372ea71df0a81584d2e0a2d0882f`
  - `check_v4_levels.py` = `4fa55821f7714c522bc3cace7a6eabfba1e51c9b7af6af3355556a325d93faf5`
  v3 stays byte-identical and its kept sections remain in force (v4 §2).
- **2026-10-17 — CPAL 2027 dates verified (WP0-v4 step 2).** Recent Spotlight Track submission
  deadline **18 January 2027, 23:59 AoE**; notification 1 February 2027; conference 23–26 March
  2027, Hitotsubashi Hall, Tokyo. Sources: <https://cpal.cc/call_for_papers/> (Important Dates) and
  <https://cpal.cc/deadlines/>, fetched 2026-10-17. This confirms runbook §9's planned
  "18 Jan 2027"; the plan's last milestone (8 Jan 2027) keeps 10 days of margin.
- **2026-10-17 — WP3 E1 budget projection recorded**, commit `dab95d8`. Hygiene (v4 §7): 56 KB
  JSON < 1 MB, 13 KB script, no raw arrays, provenance hashes match the registered v3 runbook.
  Status in the file itself: PROJECTION ONLY — it clears no gate.

## Drift-law parameter defaults in force (runbook §3.5.3; change only before training)

Until the author or Junjie changes them, these are the values every LUCiD cell uses:

- **F1 module gain:** s ∈ {0.02, 0.05, 0.10} (main cells); gain_m = 1 + s·u_m.
- **F3 independent per-PMT gain:** s_p = 0.05 in every LUCiD cell (may be raised to ≤ 0.10 only by
  §3.5.3a step 3 — that is the registered fallback, not a new choice).
- **F2 attenuation:** λ = λ0·exp(a·v), a from {0.10, 0.20, 0.50}; **the value is set by the
  §3.5.3a detectability check**, not chosen here. F2′ fallback b ∈ {0.05, 0.10}.
- **Law R11b (co-varying):** s = 0.05. **Law T (aging):** s_T = 0.10, R = 100 training runs.
- **Gain clipping:** each factor clipped to [0.5, 1.5] separately; clipped counts reported (A8).
- **Tier-1 bridge:** A4 law at g = 0.02, nominal SNR, channel split; θ0 = (4.5, 0.35, 0.08);
  E1b θ SD = (3, 0.1, 0.04)/√12 = (0.8660254, 0.0288675, 0.0115470).

## Engineering decisions (v4 §5 "decision default")

- **2026-10-17 — SSH targets.** The bare names `deepthought` / `deepthought2` do not resolve on this
  Mac (DNS). Use `dwong@deepthought.etp.kit.edu` and `dwong@deepthought2.etp.kit.edu` with
  `-o BatchMode=yes -o ConnectTimeout=20`. Verified 2026-10-17T11:39Z (deepthought: 40 cores;
  deepthought2: 128 cores). Same hosts as the `~/.ssh/config` entries.
- **2026-10-17 — Local interpreter.** `/private/tmp/jepa-venv-cpu311/bin/python` (3.11.11,
  numpy 1.26.4, scipy 1.17.1, torch 2.5.1 CPU, pytest 9.1.1, sklearn 1.9.1, jaxtyping present).
  This is the already-accepted local environment from v3; 75 existing tests pass in 5.9 s. No new
  local environment.
- **2026-10-17 — E1 worker count.** 3 = min(physical cores − 2, floor(free GB / 4)) at 10 cores /
  16 GB (v4 §6). Recorded per run; peak per worker < 4 GB.
- **2026-10-17 — Dirty-tree rule.** The v4 registration commit contains exactly the three v4 files,
  unchanged. The only file kept deliberately untracked is `docs/LUCID_MODEL_LINEUP_2026-10-09.md`
  (the author's). Every outcome-generating run records the code SHA and the working-tree status.
- **2026-10-17 — Budget ledger gains a `v4_caps` block.** Titan 300 GPU-h, L40S 80 GPU-h, LUCiD
  generation CPU 48 wall-h (§3.5.6/§6.1), with v3's E1 24-wall-hour cap kept. The v3 campaign rows
  are unchanged; the new caps live in a separate block so no v3 accounting is rewritten.

## Open questions (not gates yet)

- **PMT-branch SHA** not named (blocks reported LUCiD results, not development; G-PMT only if the
  repairs are not accepted by 31 Oct 2026).
- **Drift ranges s, s_p, a** pending Junjie; defaults above in force.
- **`docs/LUCID_MODEL_LINEUP_2026-10-09.md`** is untracked and untouched, per the author.

## Verification records

- **2026-10-17 — Appendix A reproduced at registration SHA.** Ran the registered script
  `PYTHONPATH=src python docs/jepa/check_v4_levels.py` at `151f26e` with the accepted v3 views
  (the script builds the A4 covariance directly from `base_rho = 0` and the E2-matched θ SD, so it
  does not need the WP2-v4 code). A4 channel-split angle k = 3: 0.0, 11.3, 22.3, 48.0, **76.9**,
  84.1, 87.5, 88.8, 89.4° for g = 0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9 — §A's 77.0°
  at g = 0.05 differs by 0.1°, within the 0.5° tolerance; γ_n matches §A exactly (0.000, 0.357,
  0.526, 0.690, 0.847, 0.917, 0.962, 0.980, 0.990); interleaved agrees within 0.5°. The v3 table
  (base ρ = 0.4, v3 θ SD, channel) reproduces exactly: 18.4, 74.0, 84.6, 89.2°. Output:
  `results/jepa/20261017_appendixA_check/check_v4_levels_151f26e.txt`.

## Phase 0 findings and rulings (2026-10-17)

- **2026-10-17 — Host GPUs are GTX TITAN X, compute capability 5.2, not sm_61.** The runbook and
  launch prompt say "GTX Titan 12 GB" and the prompt's sm_61 note is wrong for these cards. Recorded
  in `PREFLIGHT_A8.md`; the runbook is not edited (an erratum is the author's call). Consequence:
  the pinned torch build must contain sm_50 or sm_52 SASS/PTX; the installed
  `/home/dwong/anaconda3` torch 2.5.1+cu121 arch list contains **sm_50**, whose cubin is
  binary-compatible with sm_52, and `cuda.is_available()` reports capability (5, 2). Titans usable.
- **2026-10-17 — Environment per host (§6.1 step 1, existing env read-only).**
  - deepthought: `/home/dwong/anaconda3` — Python 3.12.2, **torch 2.5.1+cu121** (not the cu118/cu124
    pin; driver 525.147.05 = CUDA 12.0 is within CUDA 12 minor-version compatibility and
    `cuda.is_available()` is True), numpy 1.26.4, scipy 1.13.1, h5py 3.11.0, sklearn 1.5.1,
    jaxtyping 0.3.11, jax 0.4.30 CPU-only. Used read-only (`PYTHONDONTWRITEBYTECODE=1`); lives in
    `/home`, costs no quota entries. The build deviation from the pin is recorded here as §6.1
    requires, and the mandatory 10-batch equivalence test against the local environment precedes
    its first GPU job.
  - deepthought2: `/ceph/dwong/transformer/.venv` — Python 3.12.2, **torch 2.5.1+cu124** (the exact
    pin), capability (8, 9) = sm_89, numpy 1.26.4, h5py 3.14.0, jaxtyping 0.2.38; scipy, sklearn
    and jax absent. Read-only. Enough for LUCiD training (torch/numpy/h5py); probe fitting stays on
    Deepthought per §6.1.
  - The two hosts' CUDA builds differ, which is acceptable because every comparison set is
    single-GPU-type (§6.1). Equivalence tests are per host.
- **2026-10-17 — No new environment (not attempted).** §6.1 step 2 needs ≥ 100 000 free entries;
  31 944 free. G-ENV is not triggered because step 1 succeeded.
- **2026-10-17 — LUCiD generation on Deepthought CPUs.** JAX has no CUDA plugin in the installed
  env and sm_52 is below JAX's sm_70 minimum; deepthought2 has no jax. This is the §6.1 planned
  fallback: CPU generation, ≤16 cores, 48 wall-hour cap.
- **2026-10-17 — §6.2 "1 GB" means 1024 MiB.** The guard's start rule is
  `other_user_memory_mib_max = 1024.0` (binary interpretation, the standard for GPU memory). With
  the foreign idle Jupyter kernel at 1007–1011 MiB on deepthought's GPUs 0/2/3, those cards pass the
  start rule by 13–17 MiB and are re-checked before every start; GPU1 (4183 MiB) is blocked. This
  leaves exactly the 3 training-job GPUs of the §6.2 cap.
- **2026-10-17 — Quota: gate G-QUOTA written, no cleanup performed.** `docs/jepa/GATE_G_QUOTA.md`
  asks the author to choose among authorising deletion of our own failed installs (exact paths),
  a quota increase, a local-disk path, or a volume descope. Until answered we use sparse code
  checkouts and few-large-file data layout (§6.1/§6.2 rules, no deletion anywhere).

## Guard rulings (2026-10-17, from the fresh guard review)

- **2026-10-17 — Guard start rule reads §6.2 "1 GB"/"2 GB" as 1024 MiB** (binary), and the
  per-process GPU-memory SHED uses a documented CUDA-context tolerance of max(512 MiB, 5%) above
  the §6.2 cap, because `nvidia-smi` reports allocator + context memory (Titan batch-32 ≈ 9.6 GB
  against a 9.6 GB cap would otherwise SHED a compliant job). The nominal cap and the shed cap are
  both recorded in the guard status; this loosens the literal percentage only by the context size
  and is logged here as required by §5.
- **2026-10-17 — Live guard must warm ≥ 5 minutes before any start, and again after any restart.**
  The 5-minute mean GPU utilisation is the guard's own sample mean (nvidia-smi has no 5-minute
  mean); until the window is populated the guard HOLDs ("utilisation window not populated"). The
  operators start the guard, wait ≥ 5 minutes, then start jobs.
- **2026-10-17 — Per-process GPU utilisation is not attributable on the Titans.** `nvidia-smi pmon
  -c 1` runs but prints `-`/`0` for SM% for the compute PIDs on deepthought (Maxwell/driver 525)
  and `-`/`-` on deepthought2 (driver 570). The guard therefore uses memory attribution
  (`--query-compute-apps` memory + the full per-GPU process table for the GPU index) for the
  "another user started heavy work" HOLD, and logs `foreign_utilization_pct = null` when pmon
  cannot provide it. The GPU-level 5-minute mean utilisation is still enforced for starts.
- **2026-10-17 — Claim protocol.** One O_EXCL claim file per (host, GPU) under
  `<jobs>/.locks/`, published atomically with full content; unreadable fresh claims are treated as
  held; stale claims are reclaimed only when both launcher and job PIDs are dead. A stuck
  pre-pid.json launcher can block one GPU (availability only, never safety); if that happens the
  operator reports it and the guard's duplicate-GPU HOLD keeps the card safe.

## WP3 E1 execution decisions (2026-10-17)

- **2026-10-17 — E1 runner/attempt protocol.** One attempt directory per launch; completed rows are
  never overwritten; a re-run of an attempt whose implementation changed goes to a new attempt and
  the old attempt is archived (tar.xz + sha256) in the same results directory. `report E1` is
  regenerated only from the stored rows, and non-E1 rows are sourced solely from
  `PREDICTION_STATUS.json`.
- **2026-10-17 — R3 frozen transform.** The R3 per-view invariance diagnostic uses a fixed
  invertible transform (N(0,1) + 5I per seed/cell, independent per view for context and target),
  recorded in `configs/e1_grid_v4.json` before its outcome. It is an engineering choice inside the
  registered definition (v3 §2.4 R3, P2).
- **2026-10-17 — R3b implementation-defect correction.** Attempt 006 measured the oracle's angles on
  the whitened-coordinate functional; the registered definition requires row spaces mapped back to
  the original context coordinates (v3 §4; P2 "map directions back before comparing"). Corrected in
  attempt 007; both attempts are archived, no row overwritten. The corrected R3b outcome is
  prospective from that correction and is subject to the E1 independent recompute.
- **2026-10-17 — E1 worker count for E1b.** Registered formula W = min(physical cores − 2,
  floor(free GB / 4)), capped at 3, measured at each launch; W and peak RSS per worker recorded in
  the attempt manifest.
- **2026-10-17 — E1 attempts 001–006 are superseded** (solver-tolerance rows and the R3b
  implementation defect) and archived bit-exactly; only attempt 007's rows feed the report.

## Phase 0 guard deployment (2026-10-17)

- **2026-10-17 — Guard deployed after two fresh reviews.** First review ACCEPT WITH FIXES (F1 pid
  contract, F2 4th-training start, F3 claim race, F4 quota source), second review RESOLVED all
  except F3/R1, which were closed in a third delta round (atomic claim publication; utilisation
  warm-up HOLD) and verified by 137 tests. Commit `6468582`.
- **2026-10-17 — Host deployments.** Sparse blobless clones at `/ceph/dwong/jepa/code/<sha>`
  (`src scripts`, +334 entries); 10-minute dry run per host 12:18–12:29Z (no status JSON, no
  signals, level OK, `cephfs_xattr`, unattributed memory 3–7 / 19 MiB, window warm); live guards in
  tmux `jepa_guard` since 12:29/12:30Z; one ≤10-minute probe job per host through `jepa_run.sh`
  (Titan 10.55 s, L40S 126.07 s cold-cache), no bypass, 0 SHED.
- **2026-10-17 — Guard freshness race.** `jepa_run.sh` refuses a guard status not strictly younger
  than 120 s while the guard writes every 120 s, so a start must follow a guard cycle; a refusal
  (exit 3) is retried after the next cycle and never bypassed. Recorded by the host operator.
- **2026-10-17 — Probe fixture defect and correction.** The first probe version recorded the
  network's pooled heads, which the repository's muP initialisation zeroes, so the 10-batch
  equivalence fixture was identically zero (both hosts). The probe now captures the
  pre-`final_proj_norm` token grid with a hook and uses it for the fixture, the timing loss and
  determinism (`9cf53de`). The equivalence test will be re-run from the next deployed SHA; the
  environment facts from the first probes stand.
- **2026-10-17 — Shared jobs directory.** `/ceph/dwong/jepa/jobs/` is one directory visible from
  both hosts; guard status and job directories are per-host named, and the quota deltas between two
  samples include both operators' writes.

## WP4b dependencies: LUCiD revision and WCTE geometry (2026-10-17, findings)

- **2026-10-17 — The pinned LUCiD revision `eaf2e89f579d6679e90020c732ac2c846895d691` is not
  available locally.** The only LUCiD check-out on Deepthought is
  `/ceph/dwong/ORACLE/external/LUCiD` (git HEAD `86a1871ab4a090f9e6970935af86ca98f1d114b1`,
  installed as `lucid-sim 0.1.dev1+g86a1871ab`); the pinned commit is not in that repository
  (`git cat-file -e` fails). §3.5.2 pins `eaf2e89…`. Remote availability is being checked
  network-only; until it resolves, no LUCiD-generated artefact may be labelled as being at the
  pinned revision, and none is a reported result.
- **2026-10-17 — The PMT branch is still at synthetic-fixture geometry.** `codex/junjie-pmt-response`
  (worktree HEAD `56179eb`) carries `configs/lucid_pmt/reference.json` with
  `"status": "synthetic_dev"` and `"fixture_geometry": "synthetic 76-sensor cylinder, NOT WCTE
  geometry"`; its runbook §0 is a planning deliverable with implementation boxes unchecked.
  Runbook v4 §3.5.3 requires the **physical WCTE map (105 mPMT modules × 19 PMTs)** "from the PMT
  branch", which therefore does not exist yet. Consequence: WP4b's generator is built against an
  explicit geometry/readout interface and labelled `readout_synthetic_unrepaired`; it is bound to
  the physical geometry and the merged readout only after the author names the PMT SHA. If that has
  not happened by 31 Oct, G-PMT is written as §3.5.2 already provides.
- **2026-10-17 — WP4b scaffold scope.** The detector-state laws (F1, F3, F2, F2′, R11b, T), the
  nominal-twin contract, the run/ID/storage structure and the §3.5.3a Fisher-information/ceiling
  numerics are implemented and unit-tested without LUCiD and without claiming WCTE geometry; the
  event generator, readout binding and detectability run wait for the merged readout. All output is
  labelled development-only (`readout_synthetic_unrepaired`).
- **2026-10-17 — LUCiD pin is network-available; the local ORACLE check-out is stale, not divergent.**
  `git ls-remote` and the GitHub commits API (HTTP 200) on both `dowlingwong/LUCiD` and
  `CIDeR-ML/LUCiD` show `eaf2e89f579d6679e90020c732ac2c846895d691` as the current `main`/`HEAD`
  (tagged `v0.4.1`, 2026-10-06, "container: build PhotonSim v1.0.3"). The installed
  `/ceph/dwong/ORACLE/external/LUCiD` is `86a1871ab…` (2026-09-16), an ancestor **95 commits
  behind** the pin, which is why `git cat-file` could not see it. Ruling: generation will use the
  pinned revision from a read-only clone inside our own area
  (`/ceph/dwong/jepa/external/LUCiD`, added to `PYTHONPATH` ahead of the installed tree); the
  ORACLE check-out is never fetched into or modified. The clone is made when the generation
  environment is assembled (it needs the WCTE geometry/readout first), not now, because of the
  entry-quota pressure.

## WP4b generator scaffold (2026-10-17)

- **2026-10-17 — WP4b scaffold committed, development-only.** `lucid_bombs.py` implements the
  detector-state laws (F1, F3, F2, F2′, R11b, T) as geometry-agnostic keyed samplers with the A8
  [0.5, 1.5] per-factor clipping and clipped counts, the `DetectorState`/`state_for_run` contract,
  the nominal-twin contract (oracle-only, shared draws, `PermissionError` otherwise), the v3 §6
  cohort/run/ID structure with the `lucid/` prefix, manifest + CSR-like sparse storage with
  `SHA256SUMS`, the real N_min=20 exclusion/redraw accounting, and the §3.5.3a Fisher-information
  and R²-ceiling numerics. 24 tests; whole package 288 pass. Everything labelled
  `readout_synthetic_unrepaired`; nothing here is a reported result.
- **2026-10-17 — engineering choices inside the scaffold** (v4 §5 decision default, none changes a
  registered prediction, metric, cell or dataset role): the robustness-set event IDs
  `lucid/robustness` 220000–229999 (dev 1220000–1229999) are an allocation the v3 §6 table does
  not provide, chosen disjoint from every registered range; per-cell disjoint event blocks are
  passed explicitly via `event_id_start` and recorded in each manifest; namespaces are derived by
  SHA-256 from (seed, role, run/event/geometry), never integer offsets; sparse arrays use int32
  PMT ids and float32 charge/time.
- **2026-10-17 — open parameters that wait for the physical geometry/readout** (recorded, not
  guessed into the science): attenuation length `lambda0` (the PMT reference has absorption off;
  the placeholder 50 m is not a registered value), the F2/F2′ strength `a` for the R11/R11b cells
  (fixed only by the §3.5.3a detectability check), the R11b attenuation variant (F2 vs F2′ and its
  strength), the F2′ detector half-height H, and the WCTE yields Y_lo/Y_hi/N_min. These bind when
  the PMT SHA is named and the geometry lands; until then the generator's occupancy work,
  detectability check and 1000-event benchmark are blocked, and G-PMT applies at 31 Oct if the
  SHA has not arrived.

## Host code delivery and environment equivalence (2026-10-17)

- **2026-10-17 — §6.1 10-batch equivalence test PASSED for both hosts.** Host probe
  (`jepa_noise.probe_env`, pre-`final_proj_norm` grid fixture, fixed seed) vs the local Mac CPU
  probe at the same code SHA (`89af5af`): worst relative difference 2.9e-6 (deepthought, torch
  2.5.1+cu121/Titan X) and 2.4e-6 (deepthought2, torch 2.5.1+cu124/L40S), 10/10 batches, same
  fixture/model config, same code SHA; tolerances 1e-4 relative / 1e-3 absolute. Evidence:
  `results/jepa/20261017_env_equiv/`. The first probe (code `6468582`) was rerun because its
  fixture read the network heads, which the repository's muP initialisation zeroes; the corrected
  fixture reads the token grid.
- **2026-10-17 — code-delivery layout while G-QUOTA is open.** §6.1 asks for one clone per SHA
  under `code/<sha>/`; a sparse clone costs ~340 entries and only ~1 000 entries remain above the
  §6.2 write-stop, so from the next SHA on we keep **one working clone per host-pair** under
  `code/<sha>/`, created once and then updated **in place** by `git fetch origin JEPA` +
  `git checkout <sha>` only when no job is running, with the `git rev-parse HEAD` and clean-status
  recorded before and after every run and the SHA written into every run manifest. The existing
  `code/89af5af` clone stays untouched as the record of the two probe runs. This is a file-layout
  decision under §5, not a change to any registered prediction, metric, cell or dataset role; if
  G-QUOTA (B) is granted we return to one clone per SHA.
- **2026-10-17 — WP4a frozen-hash defect (found by the first Titan smoke).** `verify_bridge_products`
  compared a float64 `N_actual` marginal bit-exactly against a hash frozen on the Mac; LAPACK/BLAS
  last bits differ on the x86 hosts, so the 1-step CUDA smoke failed on deepthought before any CUDA
  code ran. Ruling: verification becomes tolerance-based (relative Frobenius ≤ 1e-8) with the
  computed sha recorded in the manifest, and the matrix is materialised once per host under a data
  root so later runs load identical bytes and verify the file's own sha. The smoke must be re-run
  after the fix; no bridge outcome may be produced before it passes.

## Phase 0/1 GPU evidence (2026-10-17)

- **2026-10-17 — WP4a 1-step CUDA smoke PASSED on the Titan (code `bbcded2`).** All four arms ran
  on `cuda:0`, exit 0, 14.36 s wall, no device/index_select error; the anchored arm (the one the
  device defect would have broken) ran clean; GPU back to baseline afterwards. The N_actual
  marginal was materialised once under `/ceph/dwong/jepa/data/bridge/<reference-sha>/` with
  `file_sha256 fa5d7d65…` and loaded by the run (tolerance + file-sha model). The overlay device
  path is now proven end to end.
- **2026-10-17 — GPU-hours so far** (ledger rows added): Phase-0 probes 0.0029 (Titan) and 0.035
  (L40S); corrected probes 0.0029/0.0010; WP4a smoke 0.0040 (Titan). Well inside the Titan 300 h /
  L40S 80 h caps.
- **2026-10-17 — quota floor reached for writes.** Free entries fell to ~30 552 after the working
  clone and the smoke, i.e. 552 above the §6.2 "no new data below 30 000" floor. Data generation
  is blocked until the author answers G-QUOTA; the β development cohort and the nine bridge runs
  are kept to a minimal file footprint (one npz + metrics JSON + hash per run) so they can still
  run inside the remaining margin. No cleanup of any kind is performed while the gate is open.

## WP4a bridge execution decisions (2026-10-17)

- **2026-10-17 — bridge batch = 128 (frozen config), not 32.** v4 §3.5.1 defines the bridge as the
  Tier-1 E2 of §3.2/§3.4, whose implementation spec (v3 §6 WP4) fixes batch 128; v4 §6.1's
  "batch 32 must fit ... same effective batch of 32" is the LUCiD encoder's registered batch
  (§3.5.5) and its gradient-accumulation fallback. The bridge runs at the config batch 128 (it fits
  far inside the 9.6 GiB Titan cap); any run at a different batch/steps is labelled `provisional`.
  The conflict and its resolution are recorded in each aggregate.
- **2026-10-17 — β is recorded in `beta_decision.json`, not in the frozen config.** The committed
  `configs/bridge_v1.json` has no beta value (only the "may not run until beta is selected" status);
  the driver writes/reads `returns/wp4a_beta_dev/beta_decision.json` with the v3 §1.3 rule, and the
  grid refuses anchored arms without a valid decision file. The selected β is copied into this log
  when the development cohort completes.
- **2026-10-17 — B1 uses the conservative 2× seed-floor reading** (v4 §3.5.8: a difference is
  claimed only if |Δmean| > 2 × floor; §3.5.9 inherits it). The 1× boolean is reported as
  supporting evidence only.
- **2026-10-17 — write footprint of the bridge runs** (quota discipline): per run one float16
  embeddings `.npz` (~666 MB), one rolling checkpoint (~9 MB), `probes.json`, `train_log.jsonl`,
  `run_status.json`, `SHA256SUMS` = 6 files; the 12-run grid ≈ 76 entries / ~8 GB and the β cohort
  ≈ 22 entries / ~2 GB, plus one cached event shard per cohort role (~18 files / ~2 GB). Bytes are
  unquotaed and far under the 2 TB data ceiling; the entry cost (~116) fits inside the remaining
  margin above the 30 000 write-stop.

## E1b interruption and cross-attempt row discovery (2026-10-17)

- **2026-10-17 — E1b attempt_002 stopped cleanly at 33/65 rows.** The runner's own resource guard
  stopped it (`resource_guard_host_free_ram`; the Mac's free RAM fell while the GPU-side bridge
  work was running) and it also recorded that scientific source files changed during the attempt
  (the WP4a commits landed mid-run). No row was overwritten; the attempt is preserved and the
  interruption is recorded. Accounting: 2.476 worker-hours charged for the stopped attempt.
- **2026-10-17 — known defect: rows are not carried across attempts.** attempt_003 restarted at
  W=1 and reported `completed_rows_found=0 pending=65`, so it recomputes the 33 rows attempt_002
  already completed (the write-once M diagnostic *is* found, so the discovery is inconsistent).
  The fix (make completed rows discoverable across attempts within a campaign, with provenance
  pointing at the originating attempt, never copying) is queued until attempt_003 finishes,
  because editing scientific source during a running attempt is exactly what the runner's
  source-change check stops on. Cumulative E1 stays well under the 24 h cap (~9 h projected
  including the stopped attempt), so the waste is time, not budget.
- **2026-10-17 — long local runs must be detached.** E1b attempt_003 was reaped when the
  implementer's agent session ended (8/65 rows, no traceback, no finish line). From now on, runs
  longer than a few minutes are launched detached (`nohup ... & disown`), with the PID recorded, so
  their lifetime is independent of any agent turn; the results are read from the attempt directory
  afterwards. Attempt_002 (33 rows) and attempt_003 (8 rows) are preserved; the cross-attempt
  resume fix lets the next attempt compute only the missing rows.

## WP4a β selection (2026-10-17)

- **2026-10-17 — β = 10 selected for `sigreg_jepa_anchored` on the bridge.** Development cohort
  (A4, g = 0.02, nonlinear, channel split, seed 100; evaluation-independent ID ranges), one run per
  β ∈ {0.1, 1, 10}, 10 000 steps each, all exit 0 (2014/2025/2024 s). v3 §1.3 rule — minimise the
  mean over physics parameters of development-validation MSE divided by the development
  probe-training variance: β = 0.1 : 0.104152, β = 1 : 0.104057, **β = 10 : 0.097206**. No tie
  (1e-8 relative rule not triggered). Record: config SHA 2f7b3c5e…, decision file
  `returns/wp4a_beta_dev/beta_decision.json`, grid values {0.1, 1, 10} as registered. The selected β
  is passed to the grid through the decision file; it is not written into the frozen config.

## WP4a bridge grid execution (2026-10-17)

- **2026-10-17 — 12/12 bridge runs complete on the Titan, all exit 0**, 10 000 steps, batch 128,
  `registered_run: true`, three jobs in parallel on GPUs 0/2/3 (GPU 1 never used, amensch's kernel
  untouched), plus the fourth job when a slot freed. Per-run wall seconds: sigreg_jepa
  1371.14/1361.98/1362.40; anchored (β=10) 1771.40/1766.67/1772.30; mae_chi2 619.49/615.34/615.15;
  parallel ablation 1307.60/1316.54/1300.83. Job walls 4114.72 / 5330.30 / 1871.46 / 3944.72 s
  (1.14 / 1.48 / 0.52 / 1.10 Titan GPU-h).
- **2026-10-17 — combined-aggregation incident.** The driver's `grid` mode has no aggregate-only
  path: re-invoking it over an out root **re-executes** runs. The host operator stopped it after
  11 min 06 s (CPU only, no GPU, no cohort-data writes) per instruction. Damage, through symlinks,
  in `wp4a_grid_sigreg`: s0 `run_status.json` clobbered and `checkpoint.pt`/`SHA256SUMS` overwritten
  by the aborted 282-step re-run (the final checkpoint artifact is lost, its sha survives in the
  manifest), s1 `run_status.json` reset; both runs' `embeddings.npz` and `probes.json` untouched,
  the other 10 run dirs untouched. The run outcomes stand: metrics and embeddings predate the
  damage, the job-level aggregate (17:25Z) and manifest record the completed runs, and a
  pre-damage compact snapshot is committed under `results/jepa/wp4a_grid_sigreg/`. Nothing is
  restored or rewritten; the damage is recorded in `PREFLIGHT_A8_deepthought.md` §14.
- **2026-10-17 — aggregate-only mode required.** B1/R7 cannot be read from any single job's
  aggregate (each holds one arm), and `grid` re-runs, so a new read-only `aggregate` mode is being
  added (plus a fix of the misleading `missing_required_nuisance_target_in_the_declared_set`
  reason string, which is emitted for the *missing comparison arm*, not for an undefined target).

## WP4a bridge outcome (2026-10-17)

- **2026-10-17 — B1 FAILS on the Tier-1 bridge.** Combined read-only aggregate over the 12 runs
  (`returns/wp4a_grid_combined/aggregate.json`, sha256 a5847744…): plain nuisance R² mean 0.525128
  vs anchored 0.515705, paired differences seed 0/1/2 = +0.08280 / −0.01625 / −0.03828, seed floor
  0.0512676 (2× = 0.10254). The criterion needs plain > anchored in every seed and |Δmean| > 2×
  floor; both fail (seed 0 only). Recorded as `fail`, reason `not_every_seed`, in
  PREDICTION_STATUS.json with the aggregate as evidence. Physics retention is kept (anchored/plain
  MSE/CRLB ratios: amplitude 0.984, t0 0.337, tau_t 0.981 — all ≤ 1.1), and the anchored encoder is
  the most seed-reproducible (CKA 0.99843 vs 0.94064 plain). This is a scientific outcome, not an
  implementation fault; no re-tuning, no re-run.
- **2026-10-17 — amplitude-preservation probe criterion not met (finding, v4 §3.5.7).** Post-time
  block fails in all 12 runs (0.774–0.817, random-init floor 0.7376); read-out tokens are
  arm-specific: `sigreg_jepa` 0.900–0.912 (pass), anchored 0.895–0.905 (2/3 pass), `mae_chi2` and
  the parallel ablation 0.75–0.80 (fail). Reported without repair.
- **2026-10-17 — status anomalies recorded, not repaired.** The two runs whose `run_status.json`
  was damaged by the aborted aggregation re-run are accepted on job-level manifest evidence and
  declared complete explicitly (`--complete-run`); the aggregate's `status_anomalies` names them.

## WP3 E1 provenance comparison (author-requested, 2026-10-17)

- **2026-10-17 — the committed E1 runner is NOT byte-identical to the code that produced the E1a
  and E1c rows.** Comparing the current file hashes with each attempt's recorded start/end source
  hashes:
  - E1a `attempt_007` (code_sha_start `9d59762`): run.py recorded start AND end `19ad201d82…`;
    report.py start `a58e62de35…` → end `2ec13782f0…` (`source_changed_during_run` recorded);
    neither current file matches either recorded hash.
  - E1c `attempt_001` (code_sha_start `ed3b199`): run.py start `09dc0a2114…` → end `0f5a8c4fa3…`;
    report.py start `bae158348d…` → end `a58e62de35…`; neither current file matches.
  - E1b `attempt_003_20261017T153221Z` (the finishing run): run.py `df4dd1c9d1…` and report.py
    `1760c30825…` **do** match the current files; `bridge_run.py`/`test_bridge_run.py` hashes differ
    only because the aggregate mode was added after it started (those files are not used by E1).
  Consequence: the E1a/E1c **row data** were produced by the code whose hashes are recorded in their
  `attempt.json` files (authoritative for those rows), while the status/report logic (R1/R2 clause
  combination, R3/R3b fixes) was changed afterwards and PREDICTIONS.md is regenerated with the
  current code. No row was recomputed for the report.
- **2026-10-17 — E1a and E1c ran from a dirty tree.** Their `code_sha_start` is a commit SHA but
  `run.py`/`report.py` were uncommitted working-tree files at the time (recorded in each
  `attempt.json` `git_status_start`), and both attempts record `source_changed_during_run`. This is
  recorded as a provenance limitation of the E1a/E1c rows; it is why the new rule below exists.
- **2026-10-17 — new rule (author directive):** a **registered run refuses to start if
  `git status --porcelain -- src scripts` is non-empty, and records the clean SHA.** Implemented in
  the E1 runner and in the host launcher (`jepa_run.sh`/`jepa_launch.py`); smokes/dev runs may
  override explicitly and are then marked non-registered.

## Author answers (2026-10-17 21:31, quoted verbatim)

**G-QUOTA — answered (A).**
> G-QUOTA: (A). I authorise deleting exactly these two paths, our own failed installs, whose
> evidence is already committed:
> - `/ceph/dwong/jepa/venv/cpu_py311_torch251`
> - `/ceph/dwong/jepa/venv/cpu_py311_torch251_retry1`
>
> Nothing else. I will also ask about a higher quota (B); continue as if it is not coming.

Recorded as the author's explicit authorisation for those two paths only (v4 §6 hard rule). Option
(B) is not relied on.

**WCTE geometry / LUCiD — author correction.**
> WCTE geometry is not blocked by the PMT branch.
> - The physical map is LUCiD's own `config/WCTE_geom_config.json` at the pinned revision. Its
>   SHA-256 is recorded in the PMT branch's `configs/lucid_pmt/upstream_manifest.json`
>   (`9943d602…`).
> - The PMT pilots `181b747` and `56179eb` already transported photons on the pinned WCTE geometry.
>   The 76-sensor cylinder is only the PMT contract-test fixture.
> - Clone LUCiD at `eaf2e89f` inside our area, read-only. Verify that hash, then build
>   `lucid_bombs.py` on the real WCTE geometry and `mpmt_id` map, with the readout labelled
>   `readout_synthetic_unrepaired`.
> - The PMT SHA itself is still pending; G-PMT on 31 Oct is unchanged.

**Approved follow-ups (A9), verbatim.**
> ## 5. Approved follow-ups (labelled, run beside the registered results, not replacing them)
>
> I approve these two follow-ups. Record this paragraph verbatim in `DECISIONS.md`.
> - **B1-ext.** Development cohort with β ∈ {10, 100, 1000} and the LUCiD selection rule (the largest
>   β whose physics stays within 1.1 × plain). Then `sigreg_jepa_anchored`, 3 seeds, at that β, on
>   the bridge cell. Report beside B1 under the same criterion. Budget ≤ 4 Titan GPU-hours.
> - **R5-ext.** E1c with balanced initialisation: per-layer SD = (1e-3)^(1/L), so the end-to-end
>   scale matches depth 1. Same learning rate, steps and eligibility rules. Freeze the config before
>   running; CPU only.
>
> Neither replaces a registered verdict. Both are reported as follow-ups. I will register them in
> the runbook as amendment A9.

## G-QUOTA option (A) executed; LUCiD pinned clone and WCTE geometry (2026-10-17)

- **2026-10-17 — G-QUOTA (A) executed by the host operator.** Exactly the two authorised paths were
  deleted; nothing else. Entries freed +9,322 (rfiles 417 783 → 408 753, rsubdirs 51 913 → 51 621),
  free entries 30 304 → ~39 600 (39 227 settled after the shared jobs/ writes); 52 MiB freed.
  `/ceph/dwong/jepa/venv/` left empty; `code/ data/ jobs/ returns/ logs/ raw/` intact. The gate is
  closed; (B) is not relied on.
- **2026-10-17 — LUCiD pinned read-only clone created** at
  `/ceph/dwong/jepa/external/LUCiD`, HEAD `eaf2e89f579d6679e90020c732ac2c846895d691` (the pin;
  "container: build PhotonSim v1.0.3"), clean, blobless+sparse (`lucid/`, `config/`), 460 entries /
  6.5 MB; `import lucid` works read-only with `PYTHONPATH` pointing there under the conda python.
  The ORACLE check-out was not fetched into or modified.
- **2026-10-17 — WCTE geometry confirmed from LUCiD's own config.** `config/WCTE_geom_config.json`
  sha256 `9943d6028e02bc7534a809b70e37893fedd1cd6f5e258babcbacc12613d5ed5c` (matches the `9943d602…`
  in the PMT branch's `configs/lucid_pmt/upstream_manifest.json`); it points to
  `config/wcte_geometry.npz` (sha256 `0a75b035…`), which carries the parallel sensor arrays:
  `positions_mm`/`directions` (1995, 3), `surfaces` (top 399 / barrel 1197 / bottom 399),
  `pmt_id` (1…1995), `mpmt_id` (int32), `pmt_id_in_mpmt` (int32), and the detector scalars
  (inner radius 1.5771308488 m, height 2.7769709938 m, sensor radius 0.04 m). **1995 PMTs in 105
  mPMT modules, 19 per module** (21 top + 63 barrel + 21 bottom).
- **2026-10-17 — two geometry facts that WP4b must honour:** `pmt_id_in_mpmt` is **1-based**
  (1…19), so §3.5.4's within-module even/odd partition must use the field as given and not assume
  0-based parity; and `mpmt_id` takes **105 distinct values over the range 0…105 with id 45
  absent**, so module indexing must not assume a dense 0…104 block. Both are recorded for the
  LUCiD generator/encoder implementation.

## E1 independent recompute — fixes applied (2026-10-17)

- **2026-10-17 — R2 estimator corrected to the registered definition.** v3 §2.4 registers
  "Spearman over seed-mean angles", i.e. per scale cell the mean across seeds at each γ_load level.
  `report.compute_r2` used the cross-seed median, which gives 4/6 passing cells with the failing
  cell θ=0.1/η=1; the registered seed-mean gives **5/6** (ρ = 0.8929, 0.750, 1.000, 1.000, 1.000,
  0.9643) with the failing cell **θ=0.1/η=3**. R2 remains **fail** either way; PREDICTIONS now
  reports the registered estimator and the identity of the failing cell. (The two estimators are
  both defensible a priori, so the choice is recorded here with the registered text as authority.)
- **2026-10-17 — E1b provenance limitation and repair.** 8 of the 65 E1b rows — the five
  `g0_line0` clean-cell rows and `g0p005` seeds 0–2 — were written by
  `attempt_003_20261017T141027Z_E1b` (code SHA `59efe0b8`, attempt status `running`, no
  `SHA256SUMS`, its `run.py` source hash not matching any git commit), and the named final attempt
  reused them (`skipped_completed_row`). R1/R2 E1b outcomes are unaffected in magnitude (clean-cell
  median 17.309° vs 10°; g-sweep 77.329° vs 45°), but the aggregate did not disclose the mix. Repair:
  a targeted `--rerun-rows` mode re-runs exactly those eight rows in a new attempt under the current
  clean SHA; the earlier records are preserved, the new records supersede them by row id, and the
  aggregate is regenerated afterwards. This is also why the new clean-tree rule exists.
- **2026-10-17 — R3b annihilation-control wording corrected** to the stored fixture's actual
  content (k=1, original target rank 1, conditioned residual rank 0), not "rank 1 < k 2".
- **2026-10-17 — E1a/E1c config-SHA provenance note.** Their rows record config SHA
  `0cc1688…`; the current `configs/e1_grid_v4.json` is `f2712b0c…` and `0cc1688…` was never
  committed (it differs only in the values the later fixes introduced; the grid, splits, laws and
  seeds are the registered ones). The row files retain the recorded hash; the config is not
  fabricated retroactively. E1b rows use the current committed config.

## WP4b generator on the real WCTE geometry (2026-10-17)

- **2026-10-17 — WP4b photon-bomb generator implemented on the physical WCTE layout.**
  `lucid_bombs.load_wcte_geometry` reads LUCiD's `config/WCTE_geom_config.json` / `wcte_geometry.npz`
  (path-configured; verifies the pointer sha256 `9943d602…` and the npz prefix `0a75b035…` by
  filename or explicit argument) and returns a validated 1995-PMT / 105-module / 19-per-module
  `DetectorGeometry` with a dense derived `module_index` over the sparse `mpmt_id` values (id 45
  absent) and the 1-based `pmt_id_in_mpmt`. Local tests use a schema-identical synthetic fixture;
  no LUCiD data is copied into the repository. The event generator computes
  `μ_p = Y·QE(λ)·CE·Ω_p/(4π)·exp(-d_p/λ)` for prompt, no-scattering, attenuation-only transport,
  with the disc solid angle (O((r/d)²) ≲ 2%), and a synthetic unrepaired readout
  (`readout_synthetic_unrepaired`): exact Poisson from shared uniforms, dark counts, SPE smearing,
  gains applied before the fixed-PE threshold, walk imported from
  `noise_module_lucid.response.leading_edge_time` (no fork). The nominal twin shares the same
  `SensorDraws` and is reachable only through the oracle-only contract.
- **2026-10-17 — engineering choices (one line each).** Geometry is path-configured and hashed by
  layout digest plus source/config file hashes (both recorded; the layout digest stays the verified
  manifest field). Sparse `mpmt_id` is carried raw with a dense derived index; surfaces are
  top/barrel/bottom labels with a `surface_map` override if the host npz stores integers. Twin is
  stored beside hits (`twin_hits.npz`) with oracle-only enforced by the API. "Same photon draws"
  means shared Poisson uniforms (counts identical for gain-only factors, correlated for F2). Dark
  counts are one keyed Poisson per event with the 4.2 kHz PMT-preset convention, config-driven.
  `events.npz` truth (vertex, yield, expected PE) is stored because §3.5.6 probes need it. Storage
  is CSR-like float32/int32 sparse.
- **2026-10-17 — synthetic-fixture benchmark (NOT WCTE), process peak RSS 270–274 MiB.**
  1000 events + twin: 4.84–4.96 s (202–207 events/s); 972 hits/event. Terciles (mean hit PMTs /
  module charge / hit fraction): low 370 / 4.7 PE / 0.186; mid 908 / 17.6 PE / 0.455; high 1642 /
  70.9 PE / 0.823. Placeholder yields: Y = 6 000 → 211 median hit PMTs (definition ~200);
  Y = 320 000 → 0.942 median hit fraction and 118 PE median module charge, so criterion (b)
  (0.95 hit fraction) binds before (a) (300 PE). Volume projection 1.07 M events = 1.44–1.47 h on
  one process, 0.09 h on 16 cores (0.19 % of the 48 h CPU cap). The real-geometry benchmark runs on
  the host and must be repeated after the yields are frozen by §3.5.3a.
- **2026-10-17 — pending, unchanged:** PMT SHA and the repaired readout (G-PMT 31 Oct), the
  detectability check fixing Y_lo/Y_hi/N_min, λ0 and the F2/F2′ strengths (§3.5.3a), real-geometry
  benchmark numbers, and the view masks (WP4c/encoder work).

## WP4b real-geometry benchmark (2026-10-17, host)

- **2026-10-17 — real WCTE geometry benchmark PASSED on deepthought** (CPU, code `fd14311`,
  geometry `/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`): the generator itself
  verified config sha `9943d602…` and npz sha `0a75b035…`, parsed layout digest `8b46bd3c…`,
  `synthetic: false`, 1995 sensors / 105 modules / 19 per module, `module_ids_contiguous: false`
  (independent confirmation of the missing id 45), surfaces barrel 1197 / bottom 399 / top 399
  (labels are strings `top|barrel|bottom`, so no `surface_map` override is needed), radius
  1.5771308488 m, height 2.7769709938 m, sensor radius 0.04 m.
- **2026-10-17 — benchmark numbers:** 1000 events + twin in 18.915 s = **52.867 events/s** on one
  process, **peak RSS 168.6 MB**, **897.5 hits/event**, sparse store 20.57 MiB per 1000 events
  (~21.5 GiB for the planned 1.07 M events). Volume projection **5.62 h on one process, 0.35 h on
  16 cores = 0.73 % of the 48 h CPU cap**. Terciles (mean hits / hit fraction / module PE / median
  charge): low 335.1 / 0.168 / 4.27 PE / 410.6 PE; mid 832.3 / 0.417 / 16.70 PE / 1 533.7 PE; high
  1 526.8 / 0.765 / 65.30 PE / 6 560.6 PE. Evidence:
  `/ceph/dwong/jepa/jobs/wp4b_benchmark_wcte_1000.json` and `PREFLIGHT_A8_deepthought.md` §17.
- **2026-10-17 — the projection uses the placeholder yield range** (6 000–320 000 photons,
  `placeholder_pending_detectability_check`); it must be re-benchmarked at the frozen [Y_lo, Y_hi]
  after §3.5.3a, as §3.5.6 requires.

## Standing execution rules (author, 2026-10-17)

- **2026-10-17 — every registered run starts from a dedicated git worktree pinned to the clean
  SHA.** Instead of serialising development on a clean main tree: `git worktree add
  ../VERITAS-run-<sha> <sha>`, run the registered campaign from that worktree (its `attempt.json`
  records the worktree's clean status and HEAD), and leave the main tree free for development. The
  main tree is never the working root of a registered run again. Development smokes may still run
  from the main tree with `--allow-dirty` and are marked non-registered.
- **2026-10-17 — `within` split accounting with the 1-based within-module ids.** `pmt_id_in_mpmt`
  is 1-based (1…19), so "even → context, odd → target" gives **9 context and 10 target PMTs per
  module** (evens 2,4,…,18 = 9; odds 1,3,…,19 = 10) — not 10/9 as a 0-based parity would. The mask
  code must use the field as given and record the per-module counts.
- **2026-10-17 — module identity is never positional.** `mpmt_id` has 105 distinct values over
  0…105 with **id 45 absent**; every translation between a module id and an array index goes through
  an explicit id→index map (`DetectorGeometry.module_index` / `module_ids`). Positional arithmetic
  on module ids is forbidden in the generator, the views/masks and the probes.

- **2026-10-17 — E1b provenance repair executed and closed.** `attempt_005_20261017T201351Z_E1b`
  re-ran exactly the eight affected rows (`g0_line0_s0..s4`, `g0p005_line0_s0..s2`) from the clean
  SHA `fd14311`, `registered_run: true`, `registered_sha fd14311…`, `src_scripts_status` empty, no
  source change during the run; 8/8 pass, 0.556 worker-hours. The eight superseded records from
  `attempt_003_20261017T141027Z` remain preserved; the aggregate and PREDICTIONS now draw the
  latest record per row id (the clean ones). R1/R2 E1b values are unchanged (17.309°, 77.329°);
  PREDICTIONS regenerated byte-identically twice (sha256 `78181b21…`).

## WP4b §3.5.3a detectability check — implementation rulings (2026-10-17)

- **2026-10-17 — the detectability engine is implemented and tested; the real-geometry numbers are
  pending.** `lucid_detectability.py` computes, per factor and setting, the profiled Fisher
  information from the generator's true expected charges (`μ_p = Y·QE·CE·Ω_p/(4π)·exp(−d_p/λ)·gains`
  before threshold; Poisson inflated by (1+σ_spe²)), the per-event and per-50-event-run ceilings
  `σ_f²/(σ_f²+1/I)`, the yield-range bounds (Y_lo at ~200 median hit PMTs; Y_hi = the lower of the
  300 PE median module charge and 0.95 median hit fraction), the N_min = 20 occupancy check, the
  full §3.5.3a fallback ladder and the G6 / R12-`not_evaluable` branches, and a frozen
  schema-versioned decision JSON with a `DECISIONS.md`-ready block and the re-benchmark input.
  14 tests (48 with the generator suite) pass.
- **2026-10-17 — ruling on the profiling block (implementer's question 1).** §3.5.3a names "the
  vertex, the yield, all other module gains and the other factors" as unknown/profiled. Per-PMT
  gains (F3's own signal) are **not** in that list, and profiling one free gain per PMT would be an
  incidental-parameter block that makes every other factor unidentifiable (all ceilings 0, a
  predetermined G6/R12). The implementation therefore keeps F3 in the true μ and out of the nuisance
  block. This is the faithful reading; no gate.
- **2026-10-17 — ruling on the F1/F2′ joint ceiling (implementer's question 2).** Main cells carry
  F1 but no F2/F2′, so F1's acceptance for R6/R7 uses the main-cell (F2-free) ceiling. In the R11
  cell F1 and F2/F2′ co-vary, and A8 requires their ceilings "computed with both profiled jointly":
  the joint row is recorded (`f1_joint`). If the real geometry selects F2′ and that joint F1 ceiling
  is below 0.6, then F1 as used by R11 cannot reach the required ceiling and **G6 is written for
  R11/R11b** (§3.5.3a: "If no F1 cell reaches 0.6, or F2/F2′ cannot, write G6 for the dependent
  predictions"). If F2 is selected, the degeneracy does not apply (attenuation is not a module-wise
  pattern) and the numbers will show it.
- **2026-10-17 — documented conventions accepted as engineering choices:** median yield is the
  population median `sqrt(Y_lo·Y_hi)` of the log-uniform yield law (not the realised cohort median;
  both recorded); the `across` N_min occupancy uses a random module half because `view_masks` is a
  WP4c stub (occupancy check only); F3's ceiling uses one representative PMT per module with
  min/max recorded (probes later use all hit PMTs); "median module charge" is the per-event mean
  over modules, median over events.

## WP4b §3.5.3a detectability check — FROZEN (real WCTE geometry, 2026-10-17)

Run on deepthought, CPU-only, code `9835ce5` (re-run at `80e9b57` for the joint-row evidence),
geometry self-verified (`9943d602…` config, `0a75b035…` npz, layout `8b46bd3c…`, `synthetic: false`,
1995/105/19), decision JSON `/ceph/dwong/jepa/jobs/wp4b_detectability_wcte.json` (9 970 B, sha256
`fbddb6fd…`, content sha `e3bcbd1a…`). Median yield 58 177.275 photons.

| factor | setting | R²_event | R²_run median | min | max | pass (≥0.6) |
|---|---|---|---|---|---|---|
| F1 | 0.02 | 0.004592 | 0.2616 | 0.1741 | 0.4023 | **no** |
| F1 | 0.05 | 0.027988 | 0.6847 | 0.5738 | 0.8053 | yes |
| F1 | 0.10 | 0.102428 | 0.8914 | 0.8323 | 0.9368 | yes |
| F3 | 0.10 | 0.005985 | 0.3243 | 0.1476 | 0.5061 | **no** |
| F2 | 0.50 | 0.001198 | 0.9210 | — | — | yes |
| T | 0.50 | 0.571469 | 0.9860 | — | — | yes |

- **Frozen yield range:** `Y_lo = 6 249.708098` photons (median 200.5 hit PMTs at Y_lo, hit
  fraction 0.1005) and `Y_hi = 541 560.544764` photons (hit-fraction-0.95 bound: median hit
  fraction 0.9499, 1 895 hit PMTs, module charge 191.57 PE; the 300 PE bound would have allowed
  846 767.8, the rule takes the lower). The initial 320 000 was raised by fallback step 1.
- **Fallback ladder applied (4 steps, recorded):** raise Y_hi 320 000 → 541 560.545; set F2 a
  0.10 → 0.20; set F2 a 0.20 → 0.50; raise F3 s_p 0.05 → 0.10 (F3 still 0.3243).
- **Decisions:** F1 eligible {0.05, 0.10}, the s = 0.02 cells stay in the grid but are excluded
  from R6/R7; **F2 chosen at a = 0.50** (F2′ not used, no b); F3 fails even at s_p = 0.10 →
  **R12 `not_evaluable`, no gate**; law T passes; **R6/R7 evaluable; G6 not triggered**.
- **N_min occupancy at Y_lo:** `within` target median 106.5 hit PMTs (min 84, exclusion 0.0);
  `across` proxy median 97.5 (min 64, below-N_min 0.0, exclusion after 3 redraws 0.0).
- **Caveats recorded:** F2 and T ceilings rest on a single run realisation (`min == median == max`)
  whereas F1 shows real spread across the 50-event draw; the `across` occupancy proxy uses a random
  module half because `view_masks` is a WP4c stub.
- **Re-run after the engine gained the F2-branch `f1_joint` row** (commit `80e9b57`) so R11's F1 leg
  has the joint F1/F2 ceiling evidence; the decisions above are expected identical and will be
  re-confirmed from that JSON.

## Detectability re-run, joint F1/F2 ceiling, frozen-yield benchmark (2026-10-17)

- **2026-10-17 — detectability re-run at `aa012d8` confirmed the frozen decisions and added the
  F2-branch joint evidence.** `wp4b_detectability_wcte_v2.json` (16 920 B, sha256 `540e9b2e…`,
  content sha `17bd075f…`, status frozen): every decision identical to v1 (Y_lo 6 249.708098,
  Y_hi 541 560.544764, median yield 58 177.275, F1 eligible {0.05, 0.10}, F2 a = 0.50, F3 fails,
  G6 false, R12 `not_evaluable`, R6/R7 evaluable, same four fallback steps, same occupancy);
  ceilings shifted by ≤0.0013 from the slightly different sampled events.
- **2026-10-17 — joint F1/F2 ceiling (R11 evidence, no criterion):** the F2 row now carries
  `f1_joint` with 105 per-module rows, `r2_run` median **0.6819**, min 0.5360, max 0.8127 — above
  the 0.6 ceiling requirement at the median, dipping below it for a minority of modules. No G6 is
  triggered by this (the frozen branch remains G6=false); the R11 criterion itself is the probe
  R² ≥ 0.5 plus the seed-floor rule, and this row is reported beside it.
- **2026-10-17 — module-index convention for `f1_joint`: dense position 0…104.** The joint rows are
  indexed by dense position, whereas `mpmt_id` skips 45. R11's comparison (and any probe/module
  mapping) must map dense position → `mpmt_id` through `DetectorGeometry.module_ids`; position ==
  mpmt_id must never be assumed. Recorded here and in the generator config's decision block.
- **2026-10-17 — generation re-benchmarked at the frozen yield range** (§3.5.6 requirement):
  1000 events in 20.208 s = **49.486 events/s** (was 52.867 at the placeholder range), peak RSS
  178.5 MB, **1 010.5 hits/event** (was 897.5), sparse 23.16 MiB per 1000 events (~24.8 GiB for the
  planned 1.07 M). Projection **6.006 h on one process, 0.37538 h on 16 cores = 0.78 % of the 48 h
  CPU cap**. `yield_range_status = frozen_by_detectability_check` echoed from the config; the
  geometry was re-verified in-run. Terciles (mean hits / hit fraction / module PE / median charge):
  low 370.5 / 0.186 / 4.89 PE / 461.2 PE; mid 970.3 / 0.486 / 22.56 PE / 2 030.6 PE; high
  1 692.7 / 0.849 / 104.2 PE / 10 338.4 PE. Evidence committed under
  `results/jepa/20261017_wp4b_detectability/`.

## WP4c views/masks and LUCiD point encoder (2026-10-17)

- **2026-10-17 — LUCiD views/masks and the serialised point encoder implemented (development only).**
  `lucid_views.py`: `across` orders modules on a 3D Hilbert curve of their centres and takes a
  per-event random **cyclic contiguous half** as the target; `within` uses the 1-based
  `pmt_id_in_mpmt` (even context / odd target = 9/10 per module, fixed); N_min = 20 hit PMTs in
  every target view with up to 3 redraws for `across`, exclusion counted for both splits; per-view
  hit fractions and counters; the occupancy ablation subsamples unhit **context** PMTs only with
  every target PMT still queried. Every mask carries a JSON key and `mask_from_key` reproduces it
  bit-identically. `point_encoder.py`: one token per PMT (all 1995) with fixed data-independent
  scaling (position / detector dimensions, direction unitless, ln(q/1 PE) for hits else 0, first-hit
  time /100 ns else 0, hit flag), serialised point transformer (Hilbert sort, 64-token windows
  shifted 32 on alternate layers, 4 pre-norm blocks d = 128, 4 heads, **no final norm** — the
  read-out is the block-4 output), one learned mask embedding for hidden tokens, predictor
  (2 blocks, d = 64, full attention over context outputs plus position queries at target PMTs).
  Tests: 14 pass, leakage **exactly 0.0** for all nine features and bit-exact invariance when the
  hidden inputs change, 10-step determinism, no-final-norm proof, amplitude **fixture** R² 0.9990
  (not the registered post-training criterion), 911 296 parameters, batch-32 CPU peak ~2.8 GiB
  (≈4× inside the 9.6 GB Titan per-process cap, so no gradient accumulation needed).
- **2026-10-17 — tie-break conventions fixed for the whole campaign (engineering, recorded):**
  `across` target = **floor(105/2) = 52 modules** (the other rounding, 53, is the alternative; the
  choice is frozen now so every `across` cell uses the same view sizes); window attention pads the
  serialised length to a multiple of 64 with masked pad tokens and uses a cyclic roll so the
  first/last boundary is attended; one mask vector shared across all positions; predictor attention
  is **full** over context+queries (the runbook's literal reading — at the 40k scaling size this is
  expected to be the memory bottleneck, and §3.5.5 explicitly allows recording "does not fit" as the
  benchmark result); predictor queries carry the three scaled position features only (no facing
  direction); direction divisor 1.0; charge floored at 1e-6 PE before the log.
- **2026-10-17 — N_min exclusion scope:** `tokenise` computes the N_min exclusion per batch; callers
  must apply it to **training events only** (v4 §3.5.3: the exclusion "applies to training events
  only"), so the WP5 probe-set loader must not exclude probe events on N_min. Registered-arm losses
  (mae_mse, mae_chi2, sigreg_jepa, anchor, ijepa_ema), the four §3.5.6 probe forms, the random-init
  floor, the real-geometry smoke and the 2k/10k/40k scaling benchmark are still pending.

## A9 follow-up code (labelled, non-registered) — 2026-10-17

- **2026-10-17 — B1-ext and R5-ext implemented as opt-in, labelled follow-ups.** No registered
  default changed (`git status --porcelain -- results/` empty; no runbook/DECISIONS/config edits by
  the implementer). 71 tests pass across `test_bridge_run`, `test_run_e1` and `test_dynamics`.
  - **B1-ext:** `bridge_run beta-dev --extension B1-ext --beta-grid 10,100,1000
    --selection-rule lucid_physics_tolerance --plain-reference <committed plain seed-0 run>` reads
    the plain `probes.json` read-only (records path + sha256), and selects the **largest β whose
    per-parameter physics MSE/CRLB ratio to plain is ≤ 1.1 on all three parameters**; none
    qualifying → `none_qualified`, no decision file, follow-up reported not evaluable. Decision JSON
    carries `extension: "B1-ext"`, `registered: false`, the full table and the 4.0 Titan GPU-h cap
    (recorded, not enforced by the tool). The extension grid marks every run/aggregate
    `registered: false`, `verdict_role: "follow_up"`; a normal grid refuses an extension decision
    and vice versa. Tie rule: a qualified β numerically equal to the largest's physics (rtol 1e-12)
    → the **smaller** β, except bit-identical physics → the **larger** β (an active anchor at
    identical physics); `tie_kind` records which applied.
  - **R5-ext:** `run E1c --e1c-init balanced` uses per-layer SD `(1e-3)^(1/L)` (end-to-end product
    1e-3, matching depth 1) with the registered lr/steps/eligibility unchanged; the default
    `uniform` init is array-identical to the registered one. Balanced runs are labelled
    `extension: "R5-ext"`, `registered: false`, are exempt from the dirty-tree registered refusal
    (but still record code SHA/git status), and **cannot** write into any registered E1a/E1b/E1c
    result directory (default `results/jepa/<date>_wp3_e1c_ext/`); the attempt freezes
    `config_effective.json` with the E1c init block before any row runs.
- **2026-10-17 — ambiguities recorded:** "within 1.1×" is applied per parameter (all three must
  pass), not to a mean; a metrics JSON without run metadata is accepted as a plain reference with
  unknown arm/seed provenance, while a `probes.json` naming a different arm/seed is refused; the
  4.0 GPU-hour cap is a recorded budget, not a meter.

## A9 follow-ups — first results (2026-10-17/18)

- **2026-10-17 — R5-ext (balanced init) executed, labelled follow-up, NOT a registered verdict.**
  `results/jepa/20261017_wp3_e1c_ext/attempt_001` (`--e1c-init balanced`, code `46c0adb`, registered
  false, 10 rows, 41 s/row): 7 pass / 3 not_evaluable (the registered uniform-init E1c had 10/10
  not_evaluable, so the balanced initialisation does restore evaluability for 7 of 10 rows). Against
  the R5 ordering rule (δ(L=3) < δ(L=1) in ≥ 4/5 evaluable seeds): **γ = 1.0: 2/5 wins → fail;
  γ = 0.5: only 2/5 seeds evaluable (depth 3 ineligible in three seeds) → not_evaluable**, so the
  follow-up does not demonstrate the ordering either. Reported beside R5, never replacing it; no
  retuning, and the uniform-init R5 remains `not_evaluable`.
- **2026-10-17 — R5-ext provenance:** extension rows are `registered: false`, `extension: "R5-ext"`,
  written outside the registered E1 result dirs, with `config_effective.json` freezing the balanced
  per-layer SDs ((1e-3)^(1/L), product 1e-3) before the rows ran. The attempt recorded
  `scientific source changed during the attempt` for `lucid_arms.py` (the concurrent LUCiD-arm
  implementer's new files, which no E1 code path imports); recorded, no row affected.
- **2026-10-17 — B1-ext running on the Titan** (`wp4a_b1ext_beta_dev`, 3 anchored runs at
  β ∈ {10, 100, 1000}, commit `46c0adb`, clean clone) with the LUCiD selection rule and the
  committed plain seed-0 reference (`probes.json` sha256 `8a03c2ae…`, identical on host and in the
  repo). Result to be reported when it finishes; ≤ 4 Titan GPU-hours recorded cap.

## WP5 LUCiD runner and its open questions (2026-10-17)

- **2026-10-17 — WP5 LUCiD runner implemented (development-only; no registered run executed).**
  `lucid_run.py`: the §3.5.6 cell enumeration (main 6 cells × arms × seeds = 162 runs; R11 15;
  R11b 9; dark-window anchor 3; law T 9; occupancy 3; development 17 — total 201+17), per-cell
  independent training (10 000 steps, batch 32, fp32/TF32-off, final-step freeze, rolling and
  30-minute checkpoints, SIGTERM-safe, resume validated on arm/cell/seed/β/config sha), cached
  generation keyed by (cohort, law, setting, seed) with hash-verified reuse, the four probe forms
  (run-mean primary) over six targets with random-init floors and the seed floor/CKA reporting, the
  six verdicts (R6, R7, R7b, R11, R11b, R12) only when the complete cell/seed set is present, the
  development-cohort steps 1–4 (step 1 runnable now; 2–3 blocked by the pending WP5 config review;
  step 4 refused pending its definition), `smoke`, `scaling` (with `does_not_fit` recorded) and
  guard-sized `grid` chunks, and a read-only `aggregate` that never edits
  `PREDICTION_STATUS.json`. 9 new tests pass (86 with the related suites).
- **2026-10-17 — open questions recorded, not decided** (from the runner's config/doc; the WP5
  config review will resolve them): the §3.5.6 cell list has no `s_p = 0` secondary cells, no
  write-once control, no development step 4 and no 1.2× CRLB / 1.1× SE probe-form rule — the prompt
  items that mention them are questions to the author, not implemented decisions; the dev step-2
  reference `v_group` resolution is currently `not_evaluable`; the dark-window writer convention
  (yield-0 windows from the generator readout, 512 records) needs confirmation; probe event-ID reuse
  across drift settings, the R6 within-excess floor, the R12 s-scope, and `mae_chi2`'s `s` on the
  law-T/R11b cells need readings; `oracle_nominal_twin` probes currently read real hits (twin probe
  wiring pending); R8/robustness are scored but not emitted among the six verdicts.
- **2026-10-17 — the registered optimizer/lr/schedule numerics remain `pending_review`** and the
  runner refuses grid/dev training until a WP5 config review freezes them (v4 §5 review
  requirement); a fresh reviewer is being commissioned.

## WP5 pre-outcome review and frozen LUCiD training numerics (2026-10-18)

- **2026-10-18 — WP5 independent review: ACCEPT WITH FIXES.** The cell/cohort plan is verified
  correct against §3.5.6 (main 6 cells × arm seeds = 162 runs; 39 secondary: R11 15, R11b 9,
  dark-window 3, law T 9, occupancy 3; development 17 = 2 + 15), and the implementer's claims about
  the runbook are confirmed: §3.5.6 contains **no** `s_p = 0` secondary cells, no R12 training
  cell, no write-once control, no development step 4 and no 1.2× CRLB / 1.1× SE probe-form rule.
  Those remain questions for the author, not implemented decisions. The review also confirmed the
  §3.5.3a fallback `s_p = 0.10` is used for every cell because DECISIONS is binding.
  **The runner is not yet executable end-to-end**: P0 defects — an undefined name in `_score`, an
  unthreaded `physics_method` parameter, missing metrics treated as `fail` instead of
  `not_evaluable`, three verdict-math errors (F2 floor, R11b arm set, R6 excess-based median SD),
  and the optimizer block not actually applying eps/fused/Muon momentum/nesterov or any schedule.
  P1 issues (robustness/R8 and law-T extrapolation never scored despite the doc; oracle-twin probes
  reading real hits; `ijepa_ema` target-encoder input; dev step-2 decision flow; auto-resume;
  fallback-run labelling; floor-cache identity; several missing registered reporting quantities) are
  queued before outcomes. A fixer is working; nothing trains until P0 lands.
- **2026-10-18 — frozen LUCiD training numerics (Lead decision).** Mirror the already-frozen WP4
  convention (`configs/bridge_v1.json:281-322`, itself the repository's conventions): AdamW
  **lr 1e-3**, betas (0.9, 0.999), weight decay 0, **eps 1e-8**, fused on CUDA with the documented
  CPU fallback; Muon **lr 1e-3**, momentum 0.95, nesterov, ns_steps 5; **constant learning rate, no
  warmup, no decay, grad clip null**; batch 32, 10 000 steps, fp32 with TF32 off. The runner must
  apply these (not merely record them) and refuse any other, unfrozen schedule. The runbook does
  not fix lr or schedule; this is the smallest defensible freeze (the repository's own values,
  already reviewed and frozen for WP4), recorded here so the WP5 review requirement is satisfied.
  The §3.5.6 5 000-step budget fallback remains a recorded, non-primary run if ever used.

## WP5 review — Lead rulings on the reviewer's questions (2026-10-18)

- **2026-10-18 — R8/robustness:** the R8 verdict and the aging-extrapolation ratio **are added** to the
  WP5 runner (R8 is a registered prediction, §3.5.9, and its robustness sets are already generated);
  the runner must not claim them without scoring them.
- **2026-10-18 — oracle-twin probes:** wired on the arm's **own input** (the nominal-twin readout);
  if that proves impossible for a given probe form, the oracle probes are `not_evaluable` with the
  reason recorded — never scored on real hits under an oracle label.
- **2026-10-18 — R6/R12 readings (literal §3.5.8/§3.5.9):** R6 uses the **within-cell excess over
  the floor** (R² − floor) > 2 × the seed floor; R12 is evaluated in **both splits at both
  R6-eligible s (0.05 and 0.10)**, and remains `not_evaluable` from the frozen F3 outcome anyway.
- **2026-10-18 — R11/R11b seed floor:** the median SD is taken over **all three arms** in the cell
  (§3.5.8), and the F2 condition uses the **F2 metric's own median**; the module-gain median is not
  the registered metric.
- **2026-10-18 — development step 2/3 wiring:** step 3 consumes the step-2 physics-probe decision;
  the physics reference defaults to the step-2 `sigreg_jepa` probes; if the PMT `reference.json`
  group is not available the step is recorded `not_evaluable` — **no silent ridge fallback**.
- **2026-10-18 — development seeds, dark bank, probe IDs:** keep the independent 9000-series
  development seeds (never reported), the 512 yield-0 dark-window records, and the reuse of the
  fixed 10k probe event IDs across drift settings (§3.5.3: one probe-train/probe-test set per drift
  setting serves both splits). All three recorded; change only on the author's word.
- **2026-10-18 — budget fallback:** the §3.5.6 5 000-step / 3-seed fallback must be **expressible as
  a registered run** (its own recorded, non-primary role) before the Titan projection approaches the
  cap; otherwise the only honest outcome would be G2. Implemented as a labelled fallback, not as a
  silent non-registered run.
- **2026-10-18 — no-action confirmations:** §3.5.6 contains no `s_p = 0` secondary cell, no R12
  training cell, no write-once control, no development step 4, and no 1.2× CRLB / 1.1× SE
  probe-form rule. None is added: adding one would be a new specification item (G5), not a runner
  bug.

## WP5 runner P0/P1 fixes applied (2026-10-18)

- **2026-10-18 — P0 defects fixed and tested; the runner is executable end-to-end.** `_score`'s
  undefined name; the unthreaded `physics_method` (grid now runs end-to-end on the synthetic
  fixture and the nested step-2 decision is read); missing metrics/floors now yield
  **`not_evaluable`** with explicit missing ids for R6, R7, R7b, R11, R11b and R12 (v3 §1.3);
  R11/R11b use the F2 metric's own median SD and all three arms, and R6's median SD is computed on
  the excess (per-arm SD is shift-invariant, so the observable change is that a missing non-test-arm
  floor now forces `not_evaluable`); the frozen optimizer numerics are actually applied (AdamW
  lr 1e-3, betas (0.9, 0.999), wd 0, eps 1e-8, fused on CUDA; Muon lr 1e-3, momentum 0.95,
  nesterov, ns_steps 5; explicit constant-lr no-op schedule, refusal of any other schedule, no
  warmup/decay, clip null) and the model/batches are moved to the requested device.
- **2026-10-18 — P1 items:** R8 and the law-T extrapolation ratio are recorded `not_evaluable` with
  precise reasons (the per-run `extra_sets` and aggregate `registered_reports` no longer claim they
  were scored — the doc is corrected); `oracle_nominal_twin` probes and floors now read
  `twin_hits.npz` under the same mask; the development step-3 flow consumes the nested step-2
  physics decision and the step-2 plain probes and refuses without them (no silent ridge fallback);
  automatic resume from a checkpoint in the run directory; 5 000-step / 3-seed fallbacks are
  accepted and recorded in every manifest (`fallback.*`) while never `registered:true`; the floor
  cache identity includes split, mask seed, `probe_max_events`, forms, twin flag and probe-set
  hashes; aggregate now records the R6 full-input R²−floor, R7b attenuation R², R12 full-input
  per-PMT R² and occupancy natural-vs-balanced quantities.
- **2026-10-18 — `ijepa_ema` target-encoder input corrected** (Lead fix, `lucid_arms.py`): the EMA
  target encoder now sees **target-only tokens with every other position masked**
  (`hidden_mask=~batch.target`), as v3 WP4 requires and v4 §2 keeps; previously it saw the full
  input. 50 tests pass across arms/runner/encoder.
- **2026-10-18 — readings confirmed:** (i) the R6 "excess median SD" is numerically identical to the
  raw per-arm SD because a per-arm floor shift cancels in an SD; the registered reading is kept and
  the completeness requirement is the behavioural part; (ii) a 3-seed fallback may produce verdicts
  but only as recorded non-registered fallback runs; (iii) R8's criterion **is** defined
  (§3.5.9 + §3.2.9: robustness ratio = vertex resolution under the common-mode amplitude ×2 shift
  divided by the s = 0 resolution, plain > anchored in ≥ 4/5 paired seeds, 3/3 on the aging-trend
  cell) and will be implemented before the main grid, with the shifted-set construction reading
  recorded then — until then R8 stays `not_evaluable` with that reason.

## Author instructions, 18 Oct 2026 (verbatim)

> ## 1. PMT SHA named
>
> "PMT branch `codex/junjie-pmt-response` at `0fba034` is accepted for the JEPA merge, for
> development. Reported LUCiD runs wait for the PMT R-6 SHA, which adds dark counts; I will name it."
>
> - **Merge `0fba034` into `JEPA`**, as the runbook §3.5.2 allows. Then re-run the detectability
>   check on the merged readout (see §3). Regenerate no datasets yet.
> - **The reference response has fixed per-module patterns:** gain (`group_parity`, odd modules +3%)
>   and timing width. Treat both as **fixed detector response**: they are present in the measured
>   readout and in the nominal twin. The twin removes only the injected F1/F3 gains. Record this in
>   `DECISIONS.md`.
> - **Dark counts are off in that response.** Until the R-6 SHA lands, the anchor's dark-window draw
>   and the private dark noise are not physical. Label everything `readout_dark_off`, and use none
>   of it for a reported result.
> - **G-PMT** is resolved for development now. For reported runs, it is resolved when R-6 is merged.
>
> ## 2. R2: independent check of the estimator change
>
> `1d3753a` changed the R2 estimator after the outcomes were known, moving the E1a clause from 4/6
> to 5/6. It may well be right, but it needs the v4 §5 light independent recompute now.
> - A fresh reviewer, without your notes, checks `1d3753a` against the registered R2 definition text
>   (v3 §2.4 and v4 §4).
> - The reviewer recomputes R1–R5 from the raw E1 tables.
> - Record the reviewer's verdict beside R2 in `PREDICTION_STATUS.json`.
>
> ## 3. Detectability check: treat F3 as random, not known
>
> Excluding the per-PMT gains from the profiled block is right, because free gains would push every
> ceiling to zero. But treating them as **known** overstates the F1 ceilings, and those ceilings set
> the yield range and the G6 branches.
> - Marginalise F3 instead: add s_p²·μ_p² to each PMT's variance, the same term as in the §3.5.6
>   `mae_chi2` covariance.
> - Re-run the check with this change on the merged readout before freezing `[Y_lo, Y_hi]`, F2 vs
>   F2′, and s_p.
> - Record the old and new ceilings side by side.
>
> ## 4. Approved follow-ups: schedule them
>
> B1-ext and R5-ext were approved on 17 Oct; I will register them in the runbook as A9.
> - **R5-ext** is CPU-only. Run it once the detectability check is frozen. E1 has used 8.7 of its
>   24 CPU-hours; project the R5-ext cost first.
> - **B1-ext** runs on Deepthought's Titans, at most 4 GPU-hours, in parallel with WP4b development.
> - Report both against their criteria, beside B1 and R5. Neither changes a registered verdict.

### Consequences recorded from §1 (18 Oct 2026)

- **The measured readout carries fixed per-module response patterns** — a `group_parity` gain
  (odd modules +3%) and a per-module timing width — and these are **fixed detector response**, not
  injected noise: they appear in the measured readout **and in the nominal twin**. The twin cancels
  only the injected F1/F3 gains. The generator's nominal twin and the probe targets therefore keep
  the parity gain and the module timing widths.
- **Dark counts are off in the merged response**: everything generated on that readout is labelled
  `readout_dark_off`, and **no result from it is reportable**. In particular the anchor's
  dark-window draw and the private dark noise are not physical until the R-6 SHA lands; the
  `v_dark` value used by `mae_chi2` is therefore a modelling constant for the dark-off readout, not
  a measured dark charge.
- **G-PMT is resolved for development now** (merge `0fba034` accepted); for reported runs it is
  resolved only when the PMT R-6 SHA (which adds dark counts) is merged. The author will name it.
- **Merge mechanics:** merge `0fba034` into `JEPA`, no dataset regeneration yet; the merged tree's
  PMT development artefacts are carried as-is (they are the branch's audit trail).

## B1-ext β development result (2026-10-17/18, Titan)

- **2026-10-18 — B1-ext β development cohort complete (code `46c0adb`, clean clone).** Three anchored
  runs at β ∈ {10, 100, 1000}, 10 000 steps each (1 868 / 1 859 / 1 874 s ≈ 1.56 GPU-h of the 4 h
  cap). Per-parameter physics MSE/CRLB ratio to the committed plain `sigreg_jepa` seed-0 reference
  (`probes.json` sha256 `8a03c2ae…`, identical on host and in the repo):
  β = 10 → amplitude 0.952, t0 0.317, tau_t 0.933, **max 0.952 ≤ 1.1 → qualifies**;
  β = 100 → amplitude 1.084, **t0 2.437**, tau_t 1.048 → does not qualify;
  β = 1000 → amplitude 1.763, **t0 7.673**, tau_t 1.683 → does not qualify.
  Rule selects the largest qualifying β = **10** (`tie_kind: null`), decision JSON written with
  `extension: "B1-ext"`, `registered: false`, tolerance 1.1, budget cap 4.0 GPU-h recorded.
  The anchored 3-seed grid at β = 10 and the beside-B1 aggregate are now running on the Titan.

## Merged PMT readout integrated into the generator and detectability (2026-10-18)

- **2026-10-18 — fixed detector response merged.** `lucid_bombs` now reads
  `configs/lucid_pmt/reference.json` (path + sha256 bound in the generator config and in every
  manifest) through the in-repo `noise_module_lucid.response_manifest` helpers and applies:
  - the `group_parity` **charge gain** `1 + 0.03·(dense module index % 2)` (odd modules +3 %) to
    the expected charge of the measured readout **and of the nominal twin** — the twin cancels only
    the injected F1/F3 gains;
  - the odd-module **timing-width factor** `1 + 0.05·(dense index % 2)` to the per-PMT digit
    Gaussian width;
  - the parity index is the **dense** module index (`geometry.module_of_pmt`, identical to
    `response_manifest.module_groups(mpmt_id)` over the sparse WCTE ids; dense index 45 is raw
    `mpmt_id` 46 — verified by test).
  The generator's own QE table, `ce = 0.95`, `tts_sigma_ns = 1.5` and the walk parameters are
  unchanged: only the two named fixed factors were merged. Absolute ceilings may shift when the
  other readout constants are aligned with the manifest later; the twin cancellation is unaffected.
- **2026-10-18 — dark counts are off in the merged response.** The readout no longer draws dark
  charge (config `dark_rate_hz = 0`, `dark_counts: off`, hard guards), every product is labelled
  **`readout_dark_off`** (configs, run manifests, sparse stores, benchmark and decision JSON), and
  **no result from this readout is reportable**. Consequences recorded: `v_dark = 0` in the
  `mae_chi2` covariance (valid because the head parameterises μ = softplus(·) > 0, so C_m stays
  positive definite; a true zero-mean PMT is the deterministic-zero limit, not a floored variance);
  the dark-window bank is an all-zero draw, so the anchor's dark overlay is the identity and
  **R7b is `not_evaluable` with reason `readout_dark_off_dark_overlay_is_identity`** on this
  readout (never a pass or a fail); the anchor's dark-window draw and the private dark noise are
  non-physical until the PMT R-6 SHA is merged.
- **2026-10-18 — F3 marginalised in the detectability check (author §3).** The per-PMT variance is
  **`μ_p(1 + σ_spe²) + v_dark + s_p²·μ_p²`** (the `mae_chi2` term) with the in-force
  **`s_p = 0.10`** (the frozen §3.5.3a fallback, not the 0.05 scan start), and μ_p includes the
  fixed parity gain. Exactly-zero-μ PMTs get weight 0 (dark off) — no NaN. The decision JSON records
  the label, response source+hash, `f3_marginalised: true`, the formula, and the **previous frozen
  ceilings side by side** with deltas; freezing is an explicit `--freeze`.
- **2026-10-18 — status of the previous decisions:** `[Y_lo, Y_hi]`, F2-vs-F2′, s_p and the F1
  eligibility are marked **`superseded_pending_merged_readout`** until the host re-run on the WCTE
  geometry with the F3-marginalised variance freezes the new values.
- **2026-10-18 — R2 recompute corrections.** The independent reviewer (verdict (a): the seed-mean
  estimator is the registered one) also found that the earlier DECISIONS line naming the median
  estimator's failing cell as θ0.1/η1 was wrong: the cross-seed median fails **θ0.1/η3 and
  θ0.4/η3** (4/6), while the registered seed-mean gives 5/6 with only θ0.1/η3 failing. The R1
  evidence text's misleading "(<=10)" comparator was corrected. R3's two solver exceptions are not
  scored as failures (the stored convention: an incomplete row is not a failure); recorded.

## Author corrections before freezing (18 Oct 2026)

### PMT R-6 SHA named (verbatim)

> "PMT SHA b341037 is accepted for JEPA reported runs; it supersedes 0fba034."

- **2026-10-18 — R-6 merged (merge commit `2c932ee`).** R-6 adds dark counts at **1 kHz per PMT in
  a 1 µs window [−200, 800] ns**, about **2 dark hits per event**, with times measured from the start
  of the readout window. Detectability is **not** frozen at `28e24c8`; the merged R-6 readout is the
  readout for reported runs.
- **2026-10-18 — `v_dark` restored** in the `mae_chi2` covariance and in the detectability ceilings as
  the R-6 dark-charge variance `λ_dark(1+σ_spe²) = 0.001 × 1.09 = 0.00109 PE²` (derived from the
  reference config, not a literal). The dark-window anchor draws **real** dark-only windows via
  `noise_module_lucid.readout.dark_noise_window`, and **R7b is evaluable again**.
- **2026-10-18 — R7b reporting requirement:** because ~2 dark hits sit against ~900 source hits, the
  **dark-only draw's measured embedding change** must be reported next to R7b, so a pass is not
  over-read.
- **2026-10-18 — new readout label `readout_pmt_r6`** (the merged R-6 response with dark counts on).
  `readout_dark_off` and `readout_synthetic_unrepaired` remain legacy constants, never written.

### s_p ladder restarted from the registered default

- **2026-10-18 — `s_p` restarts from 0.05.** The 0.10 value came from the superseded run that treated
  F3 as *known*; with F3 **marginalised**, the §3.5.3a adjustment order is applied from the defaults —
  Y_hi first, then the F2 `a` values, then (only if still short) `s_p` up to 0.10. Old and new values
  are recorded side by side, and the superseded 0.10 is retained only as history.

### B1-ext is a development β trade-off, not a second B1

- **2026-10-18 — B1-ext's 3-seed anchored grid duplicates the registered B1 anchored arm** (same
  bridge cell, arm, seeds 0,1,2, β = 10): it is stopped and labelled a replication if it ran at all.
  What the follow-up reports is the **development trade-off at β ∈ {10, 100, 1000}**: nuisance R²,
  physics MSE/CRLB per parameter, and which tolerance excluded β = 100 (t0 at 2.437×) and β = 1000
  (t0 at 7.673×, amplitude 1.763×, tau_t 1.683×). No registered verdict changes.

## B1-ext development β trade-off (18 Oct 2026) — reported beside B1, no verdict change

Source: the three β-development runs' `probes.json` under
`/ceph/dwong/jepa/returns/wp4a_b1ext_beta_dev/runs/` (each 10 000 steps, 1 868/1 859/1 874 s).

| β | nuisance R² (dev) | physics MSE/CRLB amplitude | t0 | tau_t | ratio to plain (amp / t0 / tau_t) | qualifies |
|---|---|---|---|---|---|---|
| 10 | 0.6094 | 1.0605 | 2.9816 | 1.0067 | 0.952 / 0.317 / 0.933 | **yes** |
| 100 | 0.6079 | 1.2069 | 22.944 | 1.1303 | 1.084 / **2.437** / 1.048 | no (t0) |
| 1000 | 0.4452 | 1.9641 | 72.237 | 1.8159 | 1.763 / **7.673** / 1.683 | no (t0, amplitude, tau_t) |

Plain seed-0 reference (`sigreg_jepa`, evaluation cohort, sha256 `8a03c2ae…`):
physics MSE/CRLB 1.1138 / 9.4142 / 1.0789.

**Reading:** from β = 10 to β = 100 the nuisance R² is flat (0.609 → 0.608) while the physics
degrades sharply (t0 ratio 0.32 → 2.44), and β = 1000 loses both. In the development cohort the
anchor at β = 10 costs no physics (all ratios ≤ 0.95) and is not weaker in nuisance protection than
much stronger anchors — so the failure of B1 is not "the anchor was too weak": stronger anchoring
buys no nuisance protection and costs physics. The dev-cohort nuisance R² (0.609) is not directly
comparable to the registered B1 evaluation-cohort value (0.516) because the cohorts differ; both are
reported. The 3-seed anchored grid would have replicated the registered B1 anchored arm and is
stopped/labelled a replication, not a second verdict.

- **2026-10-18 — B1-ext β table completed (dev cohort, plain seed-0 reference nuisance R²
  0.542285, physics mean MSE/CRLB 3.868947).** β = 10: nuisance R² 0.609399, physics MSE/CRLB
  1.0605 / 2.9816 / 1.0067, ratios 0.9522 / 0.3167 / 0.9331 → qualifies; β = 100: nuisance R²
  0.607889, physics 1.2069 / 22.9439 / 1.1303, ratios 1.0836 / **2.4372** / 1.0477 → excluded on
  **t0 only**; β = 1000: nuisance R² 0.445222, physics 1.9641 / 72.2368 / 1.8159, ratios
  **1.7634 / 7.6732 / 1.6831** → excluded on all three. β = 1000 also pushes nuisance R² **below
  the plain reference** (0.445 vs 0.542) — worse on B1's own metric as well as on physics — while
  β = 10 and β = 100 both improve it (0.609 / 0.608) and β = 10 keeps every physics ratio ≤ 0.952.
  Wall per β: 1 868.36 / 1 859.05 / 1 874.15 s (0.519 / 0.516 / 0.521 GPU-h). The stopped anchored
  grid: 3 191.6 s (s0 complete; s1, s2 interrupted), recorded as a replication, cumulative B1-ext
  **2.4484 GPU-h of the 4 h cap**, files kept.
- **2026-10-18 — host observation (not ours):** a `pmt_r4_data` session appeared on deepthought
  (CPU-only, JAX_PLATFORMS=cpu, outside `/ceph/dwong/jepa/`, not in the guard's `our_jobs`) —
  flagged as the PMT implementer's background work; the guard is unaffected.

## R-6 readout restored across the LUCiD stack (18 Oct 2026)

- **2026-10-18 — the merged R-6 response (dark counts on) is now the in-force readout.** Label
  **`readout_pmt_r6`**; dark rate **1 kHz/PMT**, readout window **[-200, 800] ns** (duration 1 µs,
  ~2 dark hits/event), dark PE uniform in the window measured from the window start, bypassing QE/CE
  and passing through the same SPE smearing, gains, fixed parity gain, discriminator and T-Q walk as
  source PE; the twin keeps the fixed response and cancels only the injected gains. `v_dark`
  restored as the derived R-6 dark-charge variance **0.00109 PE²** in the detectability ceilings and
  the `mae_chi2` covariance (floor remains 0; positive definite). The dark-window anchor draws real
  dark-only windows through `noise_module_lucid.readout.dark_noise_window`, so **R7b is evaluable
  again**; and the runner now records the **dark-only draw's measured embedding change** next to
  R7b (`r7b_dark_only_embedding_change`) so a pass cannot be over-read (~2 dark hits vs ~900 source
  hits).
- **2026-10-18 — s_p ladder restart implemented:** the detectability check starts F3 marginalisation
  at the registered default **s_p = 0.05** and applies the §3.5.3a order from the defaults (Y_hi
  first, then F2 `a`, then s_p up to 0.10 only if still short); the decision JSON records the old
  (0.10, known-F3) and new values side by side and compares against the last frozen v2 run, noting
  that the intermediate `28e24c8` run was never frozen.
- **2026-10-18 — values in the configs:** bombs/generator `dark_counts: "on"`, `dark_rate_hz 1000`,
  `readout_window_ns [-200, 800]`; detectability `v_dark 0.00109`, `f3_marginalised_spread 0.05`;
  arms `mae_chi2.v_dark 0.00109`; runner `readout_dark_pmt_r6` / dark on. 106 tests pass across the
  four LUCiD suites.

## Detectability check FROZEN on the R-6 readout (18 Oct 2026)

Run at code `0ab86e6`, real WCTE geometry re-verified (config `9943d602…`, npz `0a75b035…`, layout
`8b46bd3c…`, `synthetic: false`, 1995/105/19, `module_ids_contiguous: false`), readout
**`readout_pmt_r6`** (dark counts on, 1 kHz, [-200, 800] ns, `v_dark = 0.00109 PE²`), F3
**marginalised** at `s_p` starting from the registered default **0.05**. Decision JSON
`/ceph/dwong/jepa/jobs/wp4b_detectability_wcte_r6.json` (26 014 B, sha256 `1e4407ac…`, content sha
`307a0771…`, `status: frozen`, `decision_freeze_status: frozen_by_this_re_run`), wall 124 s, CPU only.
The intermediate `28e24c8` run was never frozen and is recorded as
`superseded_by_r6_dark_counts`; the dark-off v1/v2 decisions are superseded.

**Ceilings, old (v2, dark-off, F3 treated as known) → new (R-6, dark on, F3 marginalised, s_p 0.05 start):**

| factor | setting | old R²_run | new R²_run | Δ | pass |
|---|---|---|---|---|---|
| F1 | 0.02 | 0.260945 | **0.260901** | −4.4e−5 | false → false |
| F1 | 0.05 | 0.685515 | **0.686492** | +9.8e−4 | true → true |
| F1 | 0.10 | 0.890838 | **0.893069** | +2.2e−3 | true → true |
| F3 | 0.10 | 0.323420 | **0.324621** | +1.2e−3 | false → false |
| F2 | 0.50 | 0.920896 | **0.921345** | +4.5e−4 | true → true |
| T | 0.50 | 0.986021 | **0.986084** | +6.3e−5 | true → true |

No pass/fail changes; the largest shift is +2.2e−3.

**Frozen values:** `Y_lo = 6298.436828` (median 199.0 hit PMTs, hit fraction 0.0997), `Y_hi =
539 433.964527` (0.95 hit-fraction bound; the 300 PE bound would allow 834 189.16), median yield
**58 288.856**. Fallback steps in order: (1) raise Y_hi 320 000 → 539 433.96; (2) F2 a 0.1 → 0.2;
(3) F2 a 0.2 → 0.5; (4) raise F3 spread **0.05 → 0.10** (F3 still fails at 0.325). `s_p` restart
recorded: previous 0.10 → started at the registered 0.05 → raised to the 0.10 cap only at ladder
step 4; the 0.10 end value therefore survives, but now with the marginalised variance and the
correct order. Branches: F1 eligible {0.05, 0.10}, excluded {0.02}; **F2 chosen at a = 0.50**;
F2′ not used; F3 `f3_ok = false` → **R12 not_evaluable, no gate**; R6/R7 evaluable; law T ok;
**G6 false**. Occupancy at Y_lo (64 events): within target median 105.5 hit PMTs (min 84, exclusion
0.0), across proxy median 97.5 (min 61, below-N_min 0.0, exclusion after 3 redraws 0.0).
Both configs now point at this decision (yields, F2/f1/s_p, readout label, hashes); the generation
benchmark must be re-run at the new frozen yields (§3.5.6).

- **2026-10-18 — Titan memory shape for LUCiD training (engineering, no numeric change).** The first
  500-step smoke at `0ab86e6` failed with CUDA OOM before step 0: batch 32 over 1995 PMTs allocated
  9.19 GiB and requested a further 2.18 GiB against the 0.8 × 11.92 = 9.54 GiB per-process cap.
  The registered effective batch stays **32**; the micro-batch is **8 with gradient accumulation 4**
  (loss divided by 4, one optimizer step per 4 micro-batches; the manifest records both). Steps,
  optimizer, schedule and the effective batch are unchanged, so no registered numeric moved. The R-6
  smoke data cache is intact (`/ceph/dwong/jepa/data/lucid_r6`, readout `readout_pmt_r6`).

- **2026-10-18 — R5-ext re-run after the R-6 detectability freeze (labelled follow-up, no verdict
  change).** `results/jepa/20261018_wp3_e1c_ext_r5ext2/attempt_001` (balanced init, code `f6f9028`,
  0.137748 worker-hours): 7 pass / 3 not_evaluable (identical to the pre-freeze run — the protocol is
  deterministic). Against the R5 ordering rule: γ = 1.0 has 5/5 evaluable seeds with **2/5 wins**
  (fail); γ = 0.5 has only 2/5 evaluable seeds (depth 3 ineligible in three) → `not_evaluable`. So
  the balanced initialisation restores evaluability but does not demonstrate the depth ordering;
  reported beside R5, which remains `not_evaluable`. E1 cumulative CPU 8.94 h of 24.

- **2026-10-18 — stale smoke data cache refused (expected).** The first 500-step smoke attempt
  (before the freeze commit) wrote `/ceph/dwong/jepa/data/lucid_r6` under the pre-freeze generator
  config (digest `6779146f…`); after the freeze the config hashes to `61fdafa6…`, so the runner
  correctly refused to reuse it (no silent regeneration). Kept as evidence; the retry uses a fresh
  root `/ceph/dwong/jepa/data/lucid_r6_v2` (~31 MB, ~22 entries). The first attempt also OOM'd on the
  Titan at batch 32 before step 0, hence the micro-batch 8 × accumulation 4 shape (effective batch
  unchanged at 32).

## WP5 smoke and scaling on the Titan (18 Oct 2026) — and the cost question it raises

- **2026-10-18 — 500-step real-geometry smoke PASSED** (code `3e739b6`, readout `readout_pmt_r6`,
  dataset digest `61fdafa6…` matching the frozen generator config): 500 steps in 795.8 s on a Titan
  = **1.59 s/step (0.628 steps/s)**, micro-batch 8 × accumulation 4 (effective batch 32), allocator
  peak not recorded by the manifest; 1 s-sampler peak 5 589 MiB ≈ 58 % of the 9.54 GiB cap. The
  manifest recorded the frozen numerics exactly (`muon_adamw`, lr 1e-3, eps 1e-8, betas
  (0.9, 0.999), wd 0, constant schedule, warmup 0, clip null, fused true; Muon lr 1e-3, momentum
  0.95, nesterov, ns_steps 5; groups adamw 57 / muon 24). The smoke ran without `--evaluate`, so it
  produced no probe/floor values (a separate, longer run).
- **2026-10-18 — scaling on the Titan PASSED at both sizes:** 2 000 PMTs → 764 MB peak, 9.53
  events/s; 10 000 PMTs → 3 694 MB peak, 2.39 events/s; neither `does_not_fit` under the 9.6 GB cap.
  The 40k size belongs on the L40S and is measured there next.
- **2026-10-18 — cost projection flag (budget risk, not yet a gate):** 1.59 s/step on a Titan X
  implies **4.4 h per 10 000-step run**, i.e. ~888 GPU-h for the 201-run LUCiD grid — far beyond the
  Titan 300 GPU-h cap. The model-only scaling suggests ~0.30 s/step at 2 000 PMTs (9.5 events/s ÷ 32
  events/step), so roughly 5× of the smoke's step time is data-path overhead (tokenisation, mask/key
  construction, Hilbert sort, per-batch cache lookups). The L40S is ~15× the Titan's fp32 throughput,
  so an optimised L40S run should be far below 1.59 s/step — the decisive measurement is the same
  500-step smoke on deepthought2, which is requested now. Until it lands, the grid's cost against
  the 80 GPU-h L40S cap is unprojected and **no criterion run starts**; if the optimised L40S
  projection still exceeds the caps, the honest outcome is G2 with options, not a quiet run.

## L40S environment blocker and the overlay decision (18 Oct 2026)

- **2026-10-18 — the validated L40S venv cannot import the WP5 stack as-is.** The first L40S smoke
  attempt (code `3e739b6`, clean clone) failed at **import** in 1.87 s with
  `ModuleNotFoundError: No module named 'scipy'` (the venv `/ceph/dwong/transformer/.venv`, torch
  **2.5.1+cu124**, numpy 1.26.4, has no scipy/sklearn; seven WP5 modules import scipy). No CUDA
  context was created and **no GPU-hours were consumed**; the 40k scaling job is blocked by the same
  chain. (The path I gave the operator, `/home/dwong/transformer/.venv`, does not exist; the real
  venv is under `/ceph/dwong/`.)
- **2026-10-18 — decision: keep the cu124 torch pin and use a read-only PYTHONPATH overlay** for
  the CPU numeric libraries (`scipy`, `sklearn`, `joblib`, `threadpoolctl`) symlinked from the host's
  anaconda environment into `/ceph/dwong/jepa/env_overlay` (~10 entries). Nothing is installed into
  the venv, nothing is written by it, and the GPU stack stays exactly the pinned
  torch 2.5.1+cu124. The alternative — the host's cu121 anaconda interpreter — is rejected because
  the pinned build is a comparability requirement ("other torch/cuda builds are not comparable"),
  and a new environment is impossible on quota (≈38.4 k free entries against a ~100 k requirement),
  which would otherwise have been gate **G-ENV**. The overlay's provenance and an equivalence
  evidence set (import line, the committed environment probe, and a CPU numeric identity check of
  the scipy-dependent probe/CRLB path) must be recorded before any L40S training number is used.

## L40S environment overlay and the decisive WP5 numbers (18 Oct 2026)

- **2026-10-18 — read-only PYTHONPATH overlay works and is equivalent.** `/ceph/dwong/jepa/env_overlay`
  (5 symlinks: scipy, scipy.libs, sklearn, joblib, threadpoolctl.py, into the host anaconda env,
  Python 3.12.2; 1 CephFS entry). Import: torch **2.5.1+cu124** (the pin kept), CUDA 12.4, scipy
  1.13.1, sklearn 1.5.1, both resolving inside the overlay. Equivalence evidence: (i) the committed
  environment probe re-run under the overlay matches the committed host record on 50 common scalars
  with **max abs diff 0.0** (determinism 0.0); (ii) the scipy-dependent WP5 path (`linear.invsqrt`,
  `crlb.crlb`, `subspace_angles`) is **bit-identical** between the venv+overlay and the host
  anaconda interpreter on a fixed SPD input. The cu121 interpreter alternative is therefore not
  needed, and gate G-ENV is avoided.
- **2026-10-18 — L40S 500-step smoke PASSED:** 0.23014 s/step average (0.21388 s/step steady state,
  4.676 steps/s), peak memory 5 365 MiB, frozen numerics and `readout_pmt_r6` recorded, 0.0329 GPU-h.
- **2026-10-18 — L40S 40k scaling FITS:** 1.997 events/s, peak 14 577.9 MB (cap 21 000 MB), 0.0019 GPU-h.
- **2026-10-18 — projection: the 201-run grid does NOT fit the 80 L40S GPU-hour cap.**
  10 000 steps/run at the measured rate = **0.6393 h (average) / 0.5941 h (steady)**; 201 runs =
  **128.5 / 119.4 GPU-h = 160.6 % / 149.3 % of the cap** (training only, excluding probe/evaluation,
  so a lower bound). The runbook's LUCiD fallback order (§3.5.6) is: (1) 3 seeds for every main
  arm; (2) drop secondary runs from the end of the list; (3) reduce every LUCiD arm to 5 000 steps;
  (4) G2 — recording each step. Applying it: (1) main 162 → **126 runs** (80.5 h, still over);
  (2) dropping secondary runs from the end (occupancy 3, aging 9, dark-window 3, R11b 9, R11 15)
  reaches 79.7 h only by dropping **all** secondary runs, i.e. it makes R11, R11b, R7b, R8 and the
  aging-trend ratio `not_evaluable`, and still leaves no margin for probe/evaluation overhead;
  (3) 5 000 steps halves every run — with 3 seeds and the secondary runs retained the projection is
  ≈52.8 h training (≈63 h with evaluation, ~79 % of the cap). Because the ladder's step-2/step-3
  interaction changes which registered predictions are evaluable, this is a **G2 budget gate**
  (see `docs/jepa/GATE_G2_BUDGET.md`), not an engineering choice: no criterion run starts until the
  author rules.

## G2 ruling (author, 18 Oct 2026) — verbatim

> G2 ruling (author, 18 Oct 2026). Record this verbatim.
>
> Option E, not listed in the gate: keep the registered protocol (10 000 steps, 5/3 seeds, all
> secondary runs) and raise the L40S cap from 80 to 170 GPU-hours. That covers the 201 runs, the
> 17 development runs and the evaluation margin. The 80 h figure was a planning number, set before
> deepthought2 was chosen.
> - Shared-server rules unchanged: at most 2 jobs on deepthought2; the §6.2 guard limits and the
>   HOLD/SHED behaviour apply as before.
> - Fallback: if the ledger plus projected cost reaches 170 h, apply option A (3 seeds,
>   5 000 steps), recorded as a weaker protocol. Write no new gate for that.
>
> Optional speed-up, decided only on speed, because no criterion run exists yet:
> - benchmark TF32 on the L40S, ≤ 0.1 GPU-h;
> - if it gives ≥ 1.5× steps/s, enable TF32 for every LUCiD L40S run (all arms, development cohort
>   included) and record it; otherwise keep the current fp32 setting;
> - the bridge results stay as recorded.
>
> Environment overlay:
> - Record the version and file hashes of every overlaid package.
> - Re-verify them before each batch of runs. If anything changed, stop the batch and report.
>
> Then start the development cohort, followed by the main grid.

### Consequences recorded (18 Oct 2026)

- **L40S cap is now 170 GPU-hours** for the LUCiD work (development cohort, main grid, secondary
  runs, evaluation); the 80 h figure is superseded as a planning number. The Titan 300 h, CPU 48 h
  and quota caps are unchanged, as are the §6.2 memory/HOLD/SHED rules and the ≤2-concurrent-jobs
  limit on deepthought2.
- **Fallback without a new gate:** if the ledger plus projected cost reaches **170 h**, option A
  (3 seeds for every main arm, 5 000 steps for every LUCiD arm) is applied and recorded as a
  weaker protocol. No new gate is written for that.
- **TF32 is an experiment, decided on speed only:** benchmark on the L40S within 0.1 GPU-h; enable
  for every LUCiD L40S run only if it gives ≥ 1.5× steps/s, and record it; otherwise stay fp32.
  The bridge results keep their recorded precision.
- **Overlay provenance:** record every overlaid package's version and file hashes and re-verify
  them before each batch of runs; on any change, stop the batch and report.

## TF32 experiment verdict: keep fp32 (18 Oct 2026)

- **2026-10-18 — TF32 measured on the L40S and rejected on speed.** 500-step smoke with `--tf32`
  (code `c87b31b`, same warm root): **0.18808 s/step average / 0.18578 s/step steady** (5.317 /
  5.383 steps/s) versus fp32 **0.23014 / 0.21388** → ratio **1.224× average, 1.151× steady**, both
  **below the author's 1.5× threshold**. Peak memory unchanged (5 365 MiB). Per the G2 ruling, all
  LUCiD runs stay **fp32**; the experiment cost 0.027 GPU-h of the allowed 0.1. For the record only
  (not a decision): TF32 would have put the 201-run grid at 105.0 h versus 128.5 h.
- **2026-10-18 — overlay provenance recorded and re-verified.** `/ceph/dwong/jepa/env_overlay/
  PROVENANCE.json` (sha256 `0746e7ce…`) records the symlink targets, versions (torch 2.5.1+cu124 /
  CUDA 12.4; scipy 1.13.1; sklearn 1.5.1; joblib 1.4.2; threadpoolctl 3.5.0; numpy 1.26.4; Python
  3.12.2) and per-package file hashes (`scipy/__init__.py f09d8a34…`, `sklearn/__init__.py
  42723229…`, `joblib/__init__.py 1169bab9…`, `threadpoolctl.py f07b9a0e…`). All hashes re-verified
  unchanged before the TF32 batch; the operator re-verifies before every batch and stops on change.
- **2026-10-18 — L40S GPU-hours to date:** fp32 smoke 0.0329 + 40k scaling 0.0019 + equivalence
  probe ~0.0011 + failed import attempt 0.0005 + TF32 smoke 0.0270 ≈ **0.063 of the 170 h cap**.
  With fp32 kept: 17 dev runs ≈ 10.9 h + 201-run grid ≈ 128.5 h ≈ 139.4 h (~82 % of the cap before
  evaluation overhead).

- **2026-10-18 — dev step-2 reference fix.** The physics-probe reference must be fitted with the PMT
  reference JSON's own settings: `configs/lucid_pmt/reference.json` carries
  `fit_config = {v_group 0.2172409115942029, grid 5, scale_ns 3.0, t0_quantile 0.2, min_digits 6,
  max_iter 400}`, while the runner was using `FitConfig` defaults (**grid 9**). The resolver now
  reads the `fit_config` block (top-level keys still accepted), the config's
  `physics_reference_path` points at the PMT reference, and every run records the fit settings it
  actually used plus their source. Any dev step-2 result produced before this fix (`c87b31b`) is
  superseded and must be re-run (`f824075`).

## Loader-cache thrash on 100-run cohorts (18 Oct 2026) — stop-the-line and fix

- **2026-10-18 — the smoke-based projection was optimistic because the smoke had only 2 runs.** On the
  real development cohort (100 training runs) the L40S trained at **3.07 s/step** instead of
  0.23 s/step (GPU utilisation oscillating 0↔97 %, data-starved). Measured cause: the training
  loader's run cache defaults to `loader_cache_runs = 12` and the config had no such key, while every
  32-event batch draws 4 events from **8 runs chosen out of 100** — so ~88 % of chosen runs miss the
  cache and are re-tokenised from CephFS every step (one tokenised run ≈ **101.75 MB**). Unfixed,
  that is 8.55 h per 10 000-step run, ≈1 718 GPU-h for the 201-run grid — far past any cap.
- **2026-10-18 — fix: `loader_cache_runs: 128`** in the `data` block of `lucid_run_v0.json` (caches
  every run of a 100-run cohort; ~13 GB per job; deepthought2 has 1 006 GB RAM with ~664 GB free, so
  the two-job limit is unaffected). This is an engineering cache parameter, not frozen numerics; the
  sampler semantics are unchanged. **The projection must be re-measured on a dev-cohort (100-run)
  smoke before the dev cohort or the grid is quoted again**; the G2 ruling's 170 h cap and the
  fallback rule are unchanged.
- **2026-10-18 — two L40S attempts killed and recorded (not results):** wrong-SHA step 2
  (0.1544 GPU-h, `superseded_by_reference_fit_config_fix`) and the cache-thrash step 2
  (0.3949 GPU-h). L40S ledger **0.6127 of 170 GPU-h**. The dev datasets are complete and reusable
  (code-SHA-independent; geometry/generator digests only).

## Probe-evaluation memory blow-up and the streaming fix (18 Oct 2026)

- **2026-10-18 — a completed dev run was SHED by the guard during inline evaluation.** Run 1 of dev
  step 2 at `327c66d` trained its 10 000 steps at **0.2195 s/step** (2 195.2 s; the loader fix holds:
  steady 0.217 s/step, 12.56 GiB RSS for 100 cached runs), then the evaluation phase pushed the job to
  **66.1 GiB RAM against the 64 GB per-job cap** and the guard SHED it (exit 143, 0.6876 GPU-h booked
  as `killed_guard_shed`). Root cause: the per-event probe tables hold 10 000 events × 1 995 PMTs =
  19.95 M rows of 128-d tokens in **float64 ≈ 20.4 GB per split**, for four forms × six kinds with
  train+test and a concatenation copy for the propagated form → tens of GB. This affects **every
  registered run's evaluation**.
- **2026-10-18 — decision: fix the evaluation (option B), not the §6.2 cap.** Raising the RAM cap
  (option A) would need an author sign-off and would still be inefficient; the dev-only stopgap
  (`--probe-max-events`, option C) cannot serve registered criteria. The evaluation is being
  reworked to **stream sufficient statistics over event chunks in float32** (Gram matrices / Xᵀy,
  5 run-grouped folds, fold-internal standardisation — the v3 §1.3 protocol unchanged), with the
  chunk and measured peak recorded in `probes.json`. The 64 GB cap stands.
- **2026-10-18 — re-measured cost with the loader fix:** 0.217 s/step steady → **0.603 h per
  10 000-step run** (training only); 201 runs ≈121.3 h, dev cohort (17) ≈10.25 h ≈131.5 h before
  evaluation — inside the 170 h cap if evaluation is fixed and its cost stays modest. L40S charged
  **1.3490 / 170 GPU-h**; CPU generation cap charged **0.2620 / 48 wall-h** (dev datasets, single
  threaded, code-SHA-independent cache).

## Probe evaluation streaming fix (18 Oct 2026)

- **2026-10-18 — probe evaluation is now memory-bounded; the §6.2 cap is unchanged.** The
  per-event tables are no longer materialised: the probes stream event chunks and accumulate
  **per-run sufficient statistics** (Gram / Σxy / standardisation sums, float64, in 8192-row blocks),
  one form at a time, freeing each before the next. The ridge protocol is exactly v3 §1.3 (registered
  alpha grid validated against config, 5 run-grouped folds with the sorted round-robin rule, same
  constant-column rule, alpha tie-break and R² definition); `run_mean` is merged over the whole run so
  chunking cannot change a row. `evaluation.probe_chunk_events = 64`.
- **Memory:** one per-PMT per-event table was 19.0 GiB (10 000 × 1 995 × 128, float64; 38.1 GiB with
  the list + concatenate copy) and the measured job peak was 66.1 GiB. Now: 26.8 MiB of statistics
  per (form, kind) per split, worst form 314 MiB; a 64-event propagated chunk transient is 204 MiB;
  projected evaluation peak ~518 MiB (measured RSS 489 MiB); the unit regression case (2 000 events ×
  1 995 PMTs × 128-d) peaks at 87.6 MiB against the 512 MiB budget. `probes.json` records the chunk
  and the measured peak (additive `probe_memory`).
- **Equivalence:** identical alpha choices and R² relative differences ≤3.9e-15 (small case) and
  ≤2.5e-14 (runner-level, all forms, chunk splitting runs, multi-chunk run_mean); materialised tables
  bit-identical; the no-intercept variant also matches. Tolerance asserted at rtol 1e-6.
- **Other materialisation sites:** `ensure_floor` had the same blow-up and now streams too; the R7b
  dark-only batches are a lazy generator (was a list of all test runs' features); the remaining
  probes/CKA/leakage helpers were audited and are bounded per run/event. The legacy full-matrix path
  is retained debug/test-only. 29 runner + 21 arms tests pass.
- **Consequence:** the config sha changes with this fix, so the dev step-2 run checkpoint from
  `wp5_lucid_dev_v3` no longer matches and **run 1 retrains (~0.6 h)** in the step-2 rerun; the v3
  directory stays as evidence.

- **2026-10-18 — s_p coherence and the supersession of the 0.05-marginalised decision JSON.** The
  frozen R-6 decision marginalised F3 at 0.05 (the restart start), but the in-force cell spread after
  the §3.5.3a ladder is **0.10** — the same `s_p` the §3.5.6 `mae_chi2` covariance uses. The stack is
  now coherent: `F3_SPREAD = F3_SPREAD_MAX = 0.10` (in force, cells and variance term),
  `F3_SPREAD_DEFAULT = 0.05` (the detectability check's tested-signal start, ladder may raise it to
  0.10); configs (`lucid_bombs` detector-state and detectability block, arms cells, runner cells,
  detectability check) and tests aligned (119 tests across the four LUCiD suites). The committed
  decision JSON marginalised at 0.05 is therefore superseded and is being re-run at the in-force
  marginalisation; the committed copy will be replaced and the yields re-frozen from the re-run.

- **2026-10-18 — detectability re-frozen at the in-force F3 marginalisation (`615421c`).** New JSON
  `results/jepa/20261018_wp4b_detectability_r6/wp4b_detectability_wcte_r6_consistency.json` (file
  sha256 `411190bc…`, content sha `8c9b96d2…`, `frozen_by_this_re_run`, code `615421c`) replaces the
  0.05-marginalised JSON (`1e4407ac…`, recorded as superseded). Marginalisation now `s_p = 0.10`
  (was 0.05); the F3-signal ladder attempted [0.05, 0.10] and ends at 0.10; the s_p restart block
  records 0.1 → 0.1 with the registered default 0.05 and the step-3 cap 0.10. Fallback ladder
  unchanged: raise Y_hi 320 000 → 539 433.96; F2 a 0.1 → 0.2 → 0.5; F3 0.05 → 0.10.
  **Yields are unchanged** (Y_lo 6 298.44, Y_hi 539 433.965, median 58 288.856 — the bounds come from
  the hit-fraction/module-charge targets, not from the variance), and the branches are unchanged:
  F1 eligible {0.05, 0.10}, s = 0.02 excluded, F2 at a = 0.50, F3 `null` → R12 `not_evaluable` no
  gate, G6 false, R6/R7 evaluable; occupancy at Y_lo comfortable (within 106, across proxy 98).
  **Old → new ceilings** (Δ ≤ −0.0035, no pass/fail change): F1 0.02 0.2609 → **0.2574**;
  F1 0.05 0.6865 → **0.6835**; F1 0.10 0.8928 → **0.8898**; F3 0.3249 → **0.3228**; F2 0.9209 →
  **0.9189**; T 0.9857 → **0.9855**.
- **2026-10-18 — dataset-cache coherence finding.** The cached LUCiD datasets were generated when
  the generator's F3 spread was 0.05; the generator digest does not cover that parameter, but the
  per-run **state hashes** do, and `verify_manifest` correctly raises `state hash mismatch` on the
  cached runs. The old cache is being moved aside (kept as evidence, not deleted) and regenerated at
  s_p = 0.10 (~11–12 min CPU, ~1.9 GB, ~3 k entries; CPU-generation cap ≈0.45/48 h). No retained
  dataset from the 0.05 era is usable.

## Development step 2 — physics-probe form frozen: ridge (18 Oct 2026)

- **2026-10-18 — dev step 2 decision: `ridge`, not MLP** (`wp5_lucid_dev_v6/dev/step2_probe_decision.json`,
  sha256 `bc525bba…`, `registered: false`). The registered rule is "MLP iff BOTH linear vertex
  resolutions exceed 3× the reference fit": sigreg_jepa **0.31541 m** (ratio **2.216**) and mae_mse
  **0.09345 m** (ratio **0.657**), reference resolution **0.14231 m** (n = 10 000, failure rate
  3e-4) → not both above 3× → ridge. The reference was fitted with the PMT reference JSON's own
  `fit_config` — **grid 5, scale_ns 3.0, t0_quantile 0.2, v_group 0.2172409115942029**, source
  `configs/lucid_pmt/reference.json` — exactly as required by the 2026-10-18 fix. Primary form is
  `run_mean`. Runs: sigreg_jepa 10 000 steps / 2 463.51 s (vertex per-event R² 0.898, propagated
  0.969, pmt_gain ≈ 0), mae_mse 10 000 steps / 1 264.53 s (0.103 s/step); job 1.2749 GPU-h.
- **2026-10-18 — streaming-evaluation acceptance measured at production scale:** run 1's
  `probe_memory` records **chunk 64, n_rows_total 69 226 156, peak_delta 426 MiB, resident_delta
  244 MiB, sufficient_statistics true** (process peak 13.3 GiB is the loader cache, not the
  evaluation) — the 66.1 GiB blow-up is gone. The separate validation job measured peak_delta
  283.8 MiB / resident_delta 134.7 MiB with 3 455 670 rows.
- **2026-10-18 — data coherence:** the s_p = 0.05 cache (1.9 GB, ~3 k entries) is archived at
  `/ceph/dwong/jepa/data/lucid_r6_v2/lucid__sp005_superseded` and the datasets were regenerated at
  s_p = 0.10 (11.0 min CPU, 1 core). The run manifests record the new state hashes.
- **2026-10-18 — quota headroom flag:** entries are now 468 193 with **31 807 free**, against the
  30 000 write-stop. The full grid's dataset volume (~20 k entries by the operator's estimate) would
  cross the write-stop, so the grid will run with a **rolling per-cell data lifecycle** (generate a
  cell's datasets, run every arm/seed of that cell, record the dataset hashes in the run manifests,
  then remove that cell's datasets before the next cell) and the author will be told in the next
  report. No dataset is deleted without its hashes being recorded first.
- **2026-10-18 — ledger:** L40S **2.8717 / 170 GPU-h**; CPU generation **0.4452 / 48 wall-h**.

- **2026-10-18 — grid sizing and the reclaim plan (rolling per-cell lifecycle approved).** Measured
  at s_p = 0.10: one **main-cell dataset = 5 420 entries / 2.247 GB** (train 100 runs 604;
  probe_train 200 1 204; probe_test 200 1 204; robustness 1 204; baseline 1 204); secondary cells
  are smaller (r11/r11b/occupancy ≈3 012, dark_anchor ≈3 016, law_t ≈4 216). One completed run's
  returns cost ≈8 entries / 15 MB; the 201-run grid out ≈1 608 entries / 3.0 GB. With 31 792 free
  the rolling plan cannot start (a single main cell would cross the 30 000 write-stop), so the
  following **regenerable artefacts are released at grid start**, each recorded with its manifest
  sha256 before deletion and none of them a result: the archived s_p = 0.05 cache (3 157 entries;
  its manifests recorded), the dev datasets after their decisions are committed (3 012), the three
  old code clones `72add60e`/`6468582`/`89af5af` (1 551; re-clonable by SHA), and the stale smoke
  sets (144; regenerable in seconds) — **7 864 entries / ~4.1 GB**, taking free space to ≈39.6 k.
  Under the rolling plan (generate one cell → run every arm/seed of it → record the dataset hashes
  in the run manifests → remove that cell's datasets → next cell) the projection stays above 32 k
  for every cell (final after all 201 runs ≈33.0 k). `work`, returns, jobs, the evidence JSONs and
  the full ledger are kept. Nothing is deleted without its hashes being recorded first.

- **2026-10-18 — early (authorised) reclaim executed at the quota HOLD line.** Step-3 run outputs pushed
  free entries to 31 775, below the 32 000 HOLD line, so the operator executed **only the already
  authorised** part of the reclaim: the archived s_p = 0.05 cache was removed (3 157 entries /
  1 968 MB) after recording every dataset manifest sha256 and per-run state hash in
  `/ceph/dwong/jepa/data/lucid_r6_v2/ARCHIVED_s005_EVIDENCE.json` (200 910 B, sha256 `f7126f74…`,
  6 datasets / 522 runs) and keeping each archived `manifest.json` + `SHA256SUMS` (17 entries). Free
  entries returned to **34 880**. Nothing else was deleted; the dev datasets, old code clones and
  stale smoke sets remain untouched until grid start. Two transient guard HOLDs (another user's GPU
  usage 40 % then 32 % on a GPU we use) did not affect the running job and produced no SHED.

## Author rulings, 18 Oct 2026 (verbatim)

> Good work: the loader fix and streaming evaluation saved the campaign. Rulings (author, 18 Oct
> 2026; record verbatim).
>
> 1. Stop step 3 (β selection) now, and keep its partial runs as superseded development evidence.
>    Reason: the token time origin is changing. Trigger t = 0 is the true emission time, and the dev
>    result shows the effect: mae_mse's linear vertex (0.093 m) beats the time-of-flight fit
>    (0.142 m), which must estimate t0.
> 2. Adopt the PMT readout's final time rule from the SHA I will name: NHits with N = 4 in 20 ns,
>    and the prompt window [−20, +60] ns around the trigger. Digits outside it are dropped and
>    counted; with no trigger, the origin falls back to the window start. Import it; do not
>    reimplement it. This is part of A9.
> 3. After the SHA is merged, redo development steps 2 (probe form) and 3 (β) on the new input. The
>    ~14 GPU-h fits inside the 170 h cap. Detectability does not depend on time; do not re-run it.
> 4. In DECISIONS.md, cite the authorisation you relied on for deleting the archived s_p = 0.05
>    cache. If there is none, record it as an engineering decision inside our own area, with the
>    evidence file it left.
> 5. As planned: report the β table, then stop for my review before the 201-run grid.

### Lead records under §4 (authorisation for the reclaim)

- **2026-10-18 — the archived s_p = 0.05 cache reclaim was an engineering decision by the Lead, not
  covered by G-QUOTA (A).** G-QUOTA (A) authorised exactly two paths (the failed venvs); it does not
  cover datasets. The deletion was authorised by the Lead in-session as normal hygiene for a
  **regenerable, non-result** artefact inside our own area (`/ceph/dwong/jepa/data/lucid_r6_v2`), and
  was executed early only because free entries crossed the Lead's own 32 000 HOLD line. The evidence
  retained: `ARCHIVED_s005_EVIDENCE.json` (200 910 B, sha256 `f7126f74…`; 6 datasets, 522 runs, each
  with the dataset manifest sha256 and per-run manifest/state/event-id hashes) plus the kept
  `manifest.json` + `SHA256SUMS` copies (17 entries). No result, code or log was deleted; the
  reclaimed 3 157 entries / 1.97 GB returned free space to 34 880. Recorded here explicitly for the
  author's audit.

- **2026-10-18 — development step 3 stopped on the author ruling (clean SIGTERM, exit 0).** Wall
  9 308.48 s = **2.5857 GPU-h**; the runner checkpointed and exited 0 without a forced kill or SHED.
  All 17 dev runs (step 2 and step 3) are kept in place and marked
  `superseded_by_token_time_origin_change` (record
  `returns/wp5_lucid_dev_v6/SUPERSEDED_BY_TOKEN_TIME_ORIGIN.json`, sha256 `743b0c1d…`), with the step-2
  and step-3 decisions marked superseded (`bc525bba…`, `3435892f…` — the latter invalid, written from
  interrupted runs). Dev datasets kept for the merge decision. **Partial β values (superseded
  evidence only, anchored arm vertex resolution):** β = 0.1 → 0.30604 m, β = 1 → 0.28374 m,
  β = 10 → 0.25750 m (all `ridge`, n = 10 000, failure rate 0); the β selection will be redone on the
  new time-origin input. L40S charged **5.4574 / 170 GPU-h**, of which **3.8606 h superseded** by this
  ruling; CPU generation 0.4452/48; free entries 34 859; nothing running.

## Author corrections before the redo, 18 Oct 2026 (verbatim; part of A9)

> Thanks: a clean stop and an honest audit note. Two corrections before the redo (author, 18 Oct
> 2026; record verbatim; part of A9):
>
> 1. Token time is the digit time minus the NHits trigger time from the readout rule, after the
>    prompt-window cut. The true emission time must never enter any input or any token. It exists
>    only as a truth/diagnostic field. Your plan's phrase "true-emission-origin token times" is
>    wrong if it means anything else: confirm.
> 2. Leak guard:
>    - The generator gives each event an emission-time offset, uniform in [−50, +50] ns from the
>      keyed RNG and recorded as truth. The readout window stays as it is.
>    - Add an invariance test: shifting every photon time of an event by a constant leaves the
>      tokens unchanged (bit-identical apart from dark-hit draws, which use their own keys), and
>      leaves the trigger-relative times unchanged.
>    - The time-of-flight reference fit is unaffected; it estimates t0.
> 3. Wait for the PMT SHA I name. Its head 5f4ecc0 carries the N = 4 and prompt-window code, but its
>    validation is not yet reported, so do not merge it before I name it. Meanwhile, implement 1–2
>    against that interface with unit tests, but generate no datasets.
> 4. Then: merge, regenerate the development datasets, redo steps 2 and 3, report the β table, and
>    stop for my review before the 201-run grid.

### Lead confirmation and interpretation (§1)

- **Confirmed, and my earlier phrasing was wrong.** There is no "true-emission-origin token time" in
  the design. The token time is, exactly:
  `t_token = t_digit − t_NHits_trigger`, where `t_digit` is the readout digit time **after** the
  prompt-window cut and `t_NHits_trigger` is the trigger returned by the PMT readout's rule
  (N = 4 digits within 20 ns; `None` → the readout-window start, counted). The **true emission time
  (and the per-event emission-time offset) never enters any input, token, feature, mask or probe**;
  it is stored only as truth/diagnostic metadata. Any code path that would use it as an input is a
  defect. The invariance test in §2 is the guard that proves it.
- **The event emission-time offset** is drawn per event, uniform in [−50, +50] ns, from the keyed
  RNG and recorded as truth; the readout window is unchanged. Shifting every photon time of an event
  by a constant must leave the tokens bit-identical (apart from dark hits, which use their own keys)
  and leave the trigger-relative times unchanged.
- **The time-of-flight reference fit is unchanged**: it consumes digit times and estimates t0, so
  the trigger-relative convention does not affect its resolution.

- **2026-10-18 — invariance test made precise (author).** Recorded verbatim: "Invariance test, made
  precise: with dark counts off, a constant shift of all photon times must give bit-identical tokens
  and a trigger time shifted by exactly that constant. With dark counts on, assert the same for the
  source-derived digits only, and record how often a dark coincidence moves the trigger."
  Implementation consequence: the readout must expose the **source-derived** digit times separately
  from dark digits (or recompute the source-only trigger for the test), the dark-off test asserts
  token bit-identity **and** `trigger(shifted) − trigger(original) == shift`, the dark-on test asserts
  the same on the source-derived digits, and a recorded diagnostic counts the events in which a dark
  coincidence moves the trigger (full-digital trigger ≠ source-only trigger) — that rate is reported
  with the run/readout record, not hidden.

- **2026-10-18 — A9 token-time implementation landed in the generator (pending the named PMT SHA).**
  `lucid_bombs` now carries the A9 time convention: a `time_rule` record and a lazy `pmt_time_rule()`
  import of `nhits_trigger_time`/`NHITS_N`/`NHITS_WINDOW_NS`/prompt-window from
  `noise_module_lucid.readout` (a clear `RuntimeError` when the named SHA is absent — **no silent
  fallback**, and dataset generation refuses), a keyed per-event `emission_offset_ns` uniform in
  [−50, +50] ns added to source photon arrivals only and recorded as truth (`events.npz`
  `emission_offset_ns`, `GeneratedRun.emission_offsets_ns`, never a feature), `HitReadout.time_ns` =
  digit − trigger after the prompt-window cut, additive `absolute_time_ns` for the ToF estimator
  (which estimates t0 and is unaffected), `trigger_ns`/`source_trigger_ns` diagnostics, and
  `ReadoutRecord` counters `n_events`, `n_no_trigger`, `n_prompt_dropped`,
  **`n_dark_moved_trigger`/`dark_moved_trigger_rate`** for the author's refinement. The config
  records the rule; `pmt_time_rule_available()` lets dataset-generating tests skip with an explicit
  reason until the merge. Tests: the bombs suite is green (56+ tests: token = digit − trigger,
  prompt cut + counting, `None` → window-start + counter, missing-merge error, leak guard
  bit-identical tokens, emission-offset uniformity/keying/truth field, config/manifest check,
  constant-shift invariance with the exact trigger shift, source-digit invariance and the
  dark-moved-trigger rate); the 10 dataset-generating integration tests in the other suites are
  **skipped explicitly pending the author-named SHA** (they cannot generate coherent data before
  the merge). No datasets were generated.

- **2026-10-18 — invariance refinement finalised (two readings recorded).** (i) The dark-moved-trigger
  counterfactual is computed on the **finite source-derived digits only**
  (`index[n_photon[index] > 0]`): a dark-only PMT has no source time, and the imported rule rejects
  non-finite digit times, so feeding it an infinite time would crash on ordinary events; when no dark
  digit exists the counterfactual trigger is the trigger itself. `n_dark_moved_trigger` and
  `dark_moved_trigger_rate` are written to the run manifest and the benchmark payload. (ii) The
  **no-trigger case follows the imported prompt-window mask**: when the window is undefined (no
  trigger), every digit is kept and only `n_no_trigger` counts the fallback — our implementation does
  not invent a cut around the window-start origin. Evidence: the bombs suite is green (57 tests,
  including a dark-coincidence test that proves the trigger moves, is counted, and never feeds an
  infinite digit; a strict stub mirrors the interface's finite-time requirement) and the full package
  suite is **446 passed, 10 skipped** (the skips are the dataset-generating tests pending the
  author-named PMT SHA). No datasets were generated.

## PMT SHA named (author, 18 Oct 2026) — verbatim

> PMT SHA named (author, 18 Oct 2026; record verbatim): "5f4ecc0 on codex/junjie-pmt-response is
> accepted for JEPA reported runs; it supersedes b341037."
>
> It contains readout.nhits_trigger_time(n=4, window_ns=20), readout.prompt_window_mask(trigger,
> (−20, 60)) and readout.select_digits. It passed the PMT validation: dark-on vertex within 0.026 of
> dark-off, 0 fallbacks, and 32 source digits dropped, each correctly because an earlier dark
> photoelectron on the same PMT set its time. Commits after 5f4ecc0 change only results and docs, not
> src/.
>
> Proceed as agreed:
> 1. Merge 5f4ecc0.
> 2. Token time = digit time − NHits trigger time, after the prompt-window cut. The true emission time
>    and the per-event offset are truth/diagnostic fields only.
> 3. Emission-time offset U[−50, +50] ns per event (keyed RNG), with the invariance test made precise:
>    - dark counts off: a constant shift of all photon times gives bit-identical tokens, and a trigger
>      time shifted by exactly that constant;
>    - dark counts on: assert the same for the source-derived digits only, and record how often a dark
>      coincidence moves the trigger.
> 4. Regenerate the development datasets and redo development steps 2 (probe form) and 3 (β).
>    Detectability stays frozen.
> 5. Report the β table, then stop for my review before the 201-run grid.

- **2026-10-18 — `5f4ecc0` merged (merge commit `4e07c0c`) and the A9 time convention is live
  (`d073639`).** The generator now imports **both** `readout.nhits_trigger_time` and
  `readout.prompt_window_mask` (validated against the registered A9 record) and applies the imported
  mask for the prompt-window cut — never reimplemented; the temporary "pending the named SHA" error
  and the 10 test skips are gone. Token time = digit time − NHits trigger after the cut; emission
  offset U[−50, +50] ns keyed, truth-only; trigger-relative tokens in `hits.npz`, absolute times kept
  additively; counters (`n_no_trigger`, `n_prompt_dropped`, `n_dark_moved_trigger`, rate) recorded.
  The full package suite is **456 passed, 0 skipped** (the dataset-generating tests now exercise the
  real merged rule). The development datasets are invalid under the new convention and are being
  regenerated at s_p = 0.10; development steps 2 and 3 will be redone (~14 GPU-h), then the β table
  is reported and the grid waits for the author's review.

- **2026-10-18 — dev step 2 redone on the A9 token-time input at `d073639`: probe form = `ridge`.**
  Datasets regenerated under the new convention (old dev/smoke sets recorded in
  `/ceph/dwong/jepa/data/lucid_r6_v2/DEV_TIME_LEGACY_EVIDENCE.json`, sha256 `b4ceab6a…`, 7 datasets /
  522 runs with per-run manifests and state hashes, then deleted; new dev_train 100 runs + probe
  train/test 200+200, ~12.5 min CPU, 3 012 entries / ~1.9 GB). Decision JSON
  `wp5_lucid_dev_v7/dev/step2_probe_decision.json` sha256 `8c6f5de3…`, `registered: false`: the MLP
  rule requires **both** linear resolutions above 3× the reference — sigreg_jepa **0.32363 m**
  (ratio **2.272**), mae_mse **0.11349 m** (ratio **0.797**), reference **0.142446 m** (n = 10 000,
  failure 4e-4) fitted with the required `fit_config` (grid 5, scale_ns 3.0, t0_quantile 0.2,
  v_group 0.2172409115942029) — so **ridge**, primary form `run_mean`. Walls: sigreg_jepa 2 674 s,
  mae_mse 1 536 s; job 1.4189 GPU-h. Streaming probes at production scale: **69 182 980 rows,
  Δpeak 432.5 MiB** (sigreg) / 76.1 MiB (mae_mse). Ledger: L40S **6.8763 / 170 GPU-h**, CPU
  generation **0.6535 / 48 wall-h**. Note the new convention moved the linear resolutions
  (mae_mse 0.093 → 0.113 m; sigreg 0.315 → 0.324 m; reference 0.14231 → 0.14245 m) without changing
  the ridge decision. Dev step 3 (15 β runs) is running; the β table follows and the grid waits for
  the author's review.

- **2026-10-18 — shared-GPU contention during the dev step-3 redo (no budget impact).** Another user
  (`mbinder`) started `neo-1p1q/train.py` (~3.9 GiB) plus two evaluate processes on GPU 0 at ~21:00Z,
  taking the card to 100 % utilisation alongside our 5.9 GiB job; the guard moved to **HOLD**
  ("GPU 0: another user's processes hold 5174 MiB > 2048 MiB on a GPU we use") and our rate fell from
  0.215–0.29 s/step to ~0.906 s/step (≈0.85 → ≈2.5 h per run while it lasts). **HOLD blocks new
  starts only; the running job continues and no SHED occurred.** GPU-hours are charged as wall × 1
  GPU, so contention costs calendar time, not the 170 h cap. Contingency agreed: if the contention
  persists beyond ~1 h and GPU 1 is genuinely free, migrate after the next 30-minute checkpoint (kill
  cleanly, wait for the HOLD to clear, restart at the same `--out` on `cuda:1`; completed runs skip
  and the interrupted run auto-resumes, bounding the loss to one checkpoint interval). Default
  remains to let it run.

- **2026-10-19 — contention persisted >2 h (calendar risk only; no SHED).** `mbinder`
  (`/work/mbinder/src/neo-1p1q/train.py`, four processes ≈10.3 GiB on GPU 0, ~100 % util) has held
  GPU 0 since ~21:00Z; the guard has been in HOLD almost continuously since ~22:43Z (45 HOLD
  evaluations) and **never SHED this job**. Our step-3 rate is 0.92–0.98 s/step (baseline
  0.215–0.29), i.e. ≈2.9 h/anchored run → the remaining 14 runs are ≈35–40 h under contention versus
  ≈12 h if it clears. GPU 1 is free, but a host-level HOLD blocks new starts (the launcher refuses),
  so no migration is possible until the HOLD clears; killing would only lose in-flight progress.
  Decision: keep waiting; GPU-hours are unaffected (wall × 1 GPU) and the 170 h cap is not at risk.
  Nothing is interrupted, nothing is retuned, and the grid still waits for the author's review.

- **2026-10-19 — contention cleared; guard semantics and the grid plan verified (two corrections).**
  `mbinder`'s four processes (~10.3 GiB on GPU 0, ~100 % util) left; the guard returned OK and the
  step-3 rate recovered to **0.25 s/step** without any kill or migration. **Correction to my earlier
  reading: GPU-hours are charged as *job wall-clock × 1 GPU* (§6.1), so sustained contention inflates
  the charged hours as well as calendar time** — the operator is right, and the grid will only start
  cells with the guard OK and book actual walls.
- **2026-10-19 — start-gate semantics (read from the code): HOLD is host-wide, there is no per-GPU
  start path.** `jepa_guard.evaluate()` builds host-wide `hold_reasons` (any foreign process
  >2048 MiB on any GPU we use, unattributed memory, duplicate/mismatch/missing inputs) and sets
  `start_gpu=None` whenever the level is HOLD; `status_is_usable()` refuses non-OK; `jepa_run.sh`
  exits 3 on a non-OK/refused status. Migration to a free GPU is therefore impossible while HOLD
  stands, and unnecessary when it clears (the running job speeds up by itself).
- **2026-10-19 — grid dry-run plan accepted (no execution; waits for the author's review).**
  Data-sharing groups and order: **G1** main s=0.02 across+within (54 runs, 5 datasets ≈5 420
  entries), **G2** main s=0.05 across+within + occupancy (57), **G3** main s=0.10 across+within +
  dark anchor (57), **G4** r11 + r11b (24, ≈3 616 entries), **G5** law-T aging last (9, ≈4 216
  entries). Per group: pre-batch verification (HEAD == review SHA, clean tree, overlay hashes,
  config sha, guard ts+level, free entries) → launch job 1 alone to generate the group datasets
  (wait for every manifest+SHA256SUMS; `ensure_dataset` has no lock, so two jobs must never generate
  the same dataset concurrently) → launch job 2 on the other GPU (read-only reuse) → verify every
  run `complete` with `probes.json` → write the group's dataset evidence JSON (dir, manifest sha256,
  per-run state hashes) → entry checkpoint → delete only that group's dataset dirs and confirm the
  drop → next group. Gap-filling uses `--arms`/`--seeds` subsets. Stop rules: no start unless guard OK
  with status <2 min; HOLD → report; free entries <32 000 → HOLD flag; <30 000 → stop the batch;
  ledger+projection ≥170 h → option A fallback (no new gate). **Projection: ≈126–140 GPU-h**
  (main ≈100–108, secondary ≈26–27; mae_chi2/ijepa_ema/anchor_only/oracle estimates to be refined on
  the first cells), **calendar ≈75–80 h (~3.2 days) on 2 GPUs**, margin ≈30–44 h to the cap.

- **2026-10-19 — the dev step-3 launcher hit its own 18-hour timeout (calendar event, not a SHED).**
  The job started 2026-10-18T21:04:42Z with `planned_minutes 720` → `timeout_seconds 64800`; at
  15:04:42Z it was SIGTERMed and at 15:09:42Z recorded `outcome: timeout, timed_out: true, wterm 9,
  wall 65 100.78 s`. The current run (`sigreg_jepa_anchored_dark` β = 0.1) was interrupted at
  **step 8 339/10 000** with its checkpoint saved (`e30d8d06…`); four `anchored_dark` β runs never
  started and **no step-3 β decision was written**. 14 of the 17 dev runs are complete (step-2 ×2,
  anchored ×5, anchor_only ×5, one interrupted anchored_dark). Cause: the episodic contention on
  GPU 0 (mbinder / jkleck), not our code; the guard never SHED the job. Fix: relaunch `dev --step 3`
  at the same output root with a longer launcher duration (≥1.4× the remaining expected wall under
  contention, e.g. 1 800 min); completed runs are skipped and the interrupted run resumes from its
  checkpoint. The grid stays gated on the completed β table and the author's review.

- **2026-10-19 — step 3 relaunched with a 45-hour launcher budget; the timeout's cost and the budget
  warning are recorded.** `wp5_lucid_dev_step3_v7b` started 16:08:46Z on GPU 1 at code `d073639`
  (clean, same config sha `ce452513…`, overlay hashes re-verified), `--duration-minutes 1800` →
  launcher timeout 162 000 s, same output root. Exact unfinished state at relaunch: 9 of 15 step-3
  runs complete (`anchored` β0.1/1/10/100/1000 and `anchor_only` β0.1/1/10/1000), **three**
  interrupted with resumable checkpoints (`anchor_only` β100 at step 7 917, `anchored_dark` β0.1 at
  step 8 339, `anchored_dark` β1 at step 500), three never started (`anchored_dark` β10/100/1000);
  the 2 step-2 runs are complete; **no step-3 decision had been written**. Every interrupted
  checkpoint validates (config `ce452513…`, identity, steps_scheduled 10 000) so auto-resume is
  valid; the operator verifies `resumed_from_step > 0` and completion to 10 000.
- **2026-10-19 — timeout accounting and budget warning.** The timed-out job books **18.0836 GPU-h**
  (§6.1: wall × 1), taking L40S to **24.9599 / 170 GPU-h**; the relaunch is expected to add ~6–10,
  i.e. ~31–35 h after step 3. Against the grid projection of 126–140 GPU-h the total is
  **157–175 h — approaching or exceeding the 170 h cap** depending on contention and on the still
  estimated arms. Under the author's G2 ruling the recorded response is option A (3 seeds for every
  main arm, 5 000 steps for every LUCiD arm) as a weaker protocol, with no new gate; the decision will
  be taken at the β-table report with the measured per-arm rates, and the ledger is kept exact. This
  is the §6.1 wall × 1 effect: contention inflates charged hours as well as calendar time.

- **2026-10-19 — CUDA-only crash in the R7b dark-only diagnostic fixed (`122de8f`).** The dev cohort's
  first dark-arm evaluation raised `AttributeError: 'Tensor' object has no attribute 'features'` in
  `dark_only_embedding_change`: it called `_to_device(batch.features, device)` although `_to_device`
  takes a `LucidBatch` (it moves every tensor field). On CPU the function's early return hid the bug,
  which is why the 456-test suite passed while the L40S crashed. Fixed to
  `_to_device(batch, device).features`, with a **regression test that spies on the device move and
  asserts it receives a batch**, not a tensor; full suite **457 passed**. The same crash-free test
  cannot be re-validated on CUDA locally, so the host's next dark-arm run is the acceptance.
- **2026-10-19 — step-3 relaunch status:** the 45-hour-budget relaunch re-evaluated the 9 completed
  step-3 runs and proved the resume path (`anchor_only` β100 resumed at step 7 917 → completed; dark
  β0.1 resumed at step 8 339 → trained to 10 000) before crashing in that evaluation. Cohort state:
  13/15 step-3 runs complete (anchored ×5, anchor_only ×5, plus the two step-2 runs), with dark β0.1
  at its 10 000 checkpoint (evaluation only), dark β1 resumable at step 500, and dark β10/β100/β1000
  never started. Ledger: L40S **25.9319 / 170 GPU-h** (timeout 18.0836 + relaunch 0.9719 booked);
  CPU generation 0.6535/48. Free entries fell to **31 735** (below the 32 000 flag, above the 30 000
  write-stop, partly from other dwong activity), so the Lead authorised reclaiming the two inactive,
  re-clonable code clones `72add60e` and `89af5af` (1 111 entries; the active guard clone
  `6468582` is kept) as hygiene inside our own area.

## A9 development cohort complete — β table frozen (20 Oct 2026)

- **2026-10-20 — dev step-3 completed at `122de8f` (clean, config `ce452513…`, overlay verified).**
  Job `wp5_lucid_dev_step3_v7c`: exit 0, `timed_out false`, 17:30:26Z → 00:25:18Z, wall 24 891.27 s
  = **6.9142 GPU-h**; all 15 step-3 runs complete (17 runs in the root including step 2). The resume
  path is proved end to end: `sigreg_jepa_anchored_dark` β0.1 resumed at 10 000 (evaluation only,
  315 s — the R7b crash is gone), dark β1 resumed at 500 → 10 000 (6 031.65 s), `anchor_only` β100
  resumed at 7 917 → 10 000. Decisions committed under `results/jepa/20261020_wp5_dev_cohort/`:
  step-2 (`8c6f5de3…`, ridge) and **step-3 (`00b83993…`)**.
- **β selection (registered rule: largest β with vertex resolution ≤ 1.1 × plain `sigreg_jepa`
  0.3236265 m = 0.355989 m):**
  - `sigreg_jepa_anchored` → **β = 1000** (0.20788 m);
  - `sigreg_jepa_anchored_dark` → **β = 1000** (0.21010 m; β100 was marginally better at 0.19867 m,
    but the rule takes the largest qualifying β);
  - `anchor_only` → **`none_qualified`** (best 0.44891 m > 0.355989 m).
- **β table (physics vertex resolution m / nuisance R² = module_gain-propagated / run_mean /
  pmt_gain-propagated):**
  `sigreg_jepa_anchored`: β0.1 0.31082 / −0.00228 / −0.00416 / −1.20e-4; β1 0.29825 / −0.00251 /
  −0.00367 / −0.99e-4; β10 0.23610 / −0.00202 / −0.00413 / −0.78e-4; β100 0.20934 / −0.00241 /
  −0.00465 / −0.82e-4; **β1000 0.20788** / −0.00155 / −0.00471 / −1.24e-4.
  `sigreg_jepa_anchored_dark`: β0.1 0.31040 / −0.00233 / −0.00429 / −1.28e-4; β1 0.30487 / −0.00223 /
  −0.00339 / −1.04e-4; β10 0.24192 / −0.00201 / −0.00413 / −0.72e-4; β100 0.19867 / −0.00240 /
  −0.00428 / −0.90e-4; **β1000 0.21010** / −0.00218 / −0.00446 / −1.01e-4.
  `anchor_only`: β0.1 0.57043 / −0.00065 / **0.05575** / −0.40e-4; β1 0.54840 / −0.00083 /
  **0.03255** / −0.47e-4; β10 0.47131 / −0.00104 / 0.00038 / −0.43e-4; β100 0.46216 / −0.00085 /
  −0.00694 / −0.35e-4; β1000 0.44891 / −0.00072 / −0.00661 / −0.43e-4.
  Reference points: plain `sigreg_jepa` physics 0.32363 m, module-gain propagated −0.00182;
  `mae_mse` physics 0.11349 m, module-gain run_mean 0.31470. **Reading:** both anchored arms suppress
  the module-gain nuisance to ≈0 already at β0.1 and keep improving the vertex resolution with β;
  `anchor_only` leaves a positive module-gain run-mean R² at low β (0.0558, 0.0326) and never reaches
  the physics threshold (best 0.4489 m) — hence `none_qualified`.
- **2026-10-20 — the "no anchor / anchor too weak" question is settled for the development cohort:**
  the anchor at β ≥ 0.1 suppresses module-gain nuisance in both anchored arms, and β = 1000 is the
  largest qualifying strength (physics improves monotonically with β in the range tested); the
  anchor-only objective cannot pass the physics threshold at any β in the grid.

- **2026-10-20 — Option A is applied at grid start (author's G2 standing rule).** The L40S ledger is
  **32.8460 / 170 GPU-h** and the measured per-run grid projection is **134–154 GPU-h**, so
  ledger + projection = **167–187 h**, which reaches the 170 h cap. The G2 ruling's recorded response:
  apply **option A — 3 seeds for every main arm and 5 000 steps for every LUCiD arm**, recorded as a
  weaker protocol, with no new gate. The grid will therefore run at 5 000 steps with 3 seeds, every
  manifest carrying `fallback_active: true`.
- **2026-10-20 — dev datasets reclaimed (authorised).** `dev_train` + `dev_probe_train` +
  `dev_probe_test` (3 012 entries) are deleted after recording `DEV_DATASET_EVIDENCE.json` (dataset
  dirs, manifest sha256, per-run state hashes), because the dev decisions are committed in the
  repository; free entries were 30 603 (603 above the write-stop) before the reclaim. The same
  Lead-authorised engineering-reclaim basis as the archived-cache and clone reclaims applies (all
  regenerable dev artefacts inside `/ceph/dwong/jepa/`, no results or code). The grid's rolling
  per-cell lifecycle keeps every later cell's data on disk only while that cell runs.

- **2026-10-20 — option A committed as a separate recorded config (`lucid_run_v0_optionA.json`), not as
  the default.** The operator's prepared patch replaces the registered config; instead the fallback is
  a **derived config** used only by the grid (`--config`), because (i) the registered config must stay
  the default for development, smoke and the tests, and (ii) the fallback must be a visible, named
  artefact rather than a silent change of the registered file. The derived config carries
  `steps = 5 000`, **3 seeds for `sigreg_jepa`/`sigreg_jepa_anchored`/`mae_mse` (and R11)**, the
  `fallbacks` block active with an `option_A` record (authority, date, reason, manifest effect), and
  `derived_from` the registered config. Every grid run built from it records `fallback_active: true`
  and `registered: false` (the weaker, non-primary protocol the author's G2 standing rule prescribes);
  the aggregate still computes every verdict from those runs, with `registered: false` summarising the
  protocol rather than dropping them. Expected run count under option A: **159** (6 main cells × 21,
  R11 9, R11b 9, dark anchor 3, law-T 9, occupancy 3), grid ≈60 GPU-h ⇒ ledger + grid ≈93 of 170 h.
  The registered default config is unchanged and its full test suite stays green (30 runner tests, full
  suite re-run).

## G2 amendment — grid protocol rung (1) at 10 000 steps (author, 20 Oct 2026, verbatim)

> "G2 amendment (author, 20 Oct 2026). Record this verbatim.
> My 18 Oct fallback text jumped straight to option A (3 seeds and 5 000 steps). The registered
> ladder's first rung is 3 seeds alone. With the measured option-A projection (≈60 GPU-h at 5 000
> steps, ledger + grid ≈93 of 170 h), the same 3-seed grid at 10 000 steps comes to about
> 110–120 GPU-h, so ledger + grid is about 143–153 h. That fits the cap. Halving every run while
> leaving ~75 h of the cap unused is the wrong trade.
> - Run the grid at **10 000 steps and 3 seeds for every main arm, all secondary runs retained**.
>   Record it as ladder rung (1), `fallback_rung: 1`. That is weaker than the registered protocol only
>   in the seed count, and R8's '3 of 3' clause applies.
> - Before the first job, re-project with the measured per-arm walls including evaluation. Start
>   only if ledger + projection ≤ 160 GPU-h, which keeps 10 h for contention. If it exceeds 160 h,
>   run option A as already prepared, with no new gate.
> - Training length never changes inside the grid. If the budget tightens mid-grid, apply rung (2):
>   before each secondary group, if ledger + remaining projection exceeds 170 h, drop secondary
>   groups from the end of the registered list (occupancy, aging trend, dark-window, R11b, R11).
>   Record each drop and mark the affected predictions `not_evaluable`.
> - Schedule the secondary groups after the main arms wherever the per-cell data lifecycle allows.
>   The ordering is an engineering choice, one line in `DECISIONS.md`.
> - Keep `configs/lucid_run_v0_optionA.json` committed and unused. Add the rung-(1) derived config the
>   same way: `derived_from` the registered config, registered default and tests unchanged."

## β table accepted, and `anchor_only` (author, 20 Oct 2026, verbatim)

> "β table (author, 20 Oct 2026). Record this verbatim.
> - Accepted as frozen: `sigreg_jepa_anchored` β = 1000 and `sigreg_jepa_anchored_dark` β = 1000.
>   Both sit at the top edge of the grid, and that is what the registered rule ('largest qualifying
>   β') gives when every β qualifies. Record them as `edge_selected`. Do not extend the grid. The
>   dark arm's better value at β = 100 (0.19867 m) does not change the pick: no re-selection after
>   seeing results.
> - `anchor_only` has no qualifying β (best 0.44891 m > 0.355989 m). Run it in the grid at β = 1000,
>   the anchored arms' β, labelled `beta_not_qualified`. It is a descriptive control. Any prediction
>   clause that needs a qualified `anchor_only` β is reported `not_evaluable` from it. Report the dev
>   finding itself, that the anchor alone never reaches the physics threshold at any β, as a
>   development result."

### Lead records for the above (20 Oct 2026)

- **Grid protocol = ladder rung (1):** 10 000 steps, 3 seeds for every main arm, all secondary runs
  retained; `fallback_rung: 1` recorded in the config, every manifest and the ledger. Weaker than
  the registered protocol **only in the seed count** (R8's "3 of 3" clause applies).
- **Start rule:** re-project with the measured per-arm walls including evaluation before the first
  job; start only if ledger + projection **≤ 160 GPU-h**; if it exceeds 160 h, run the prepared
  option A with no new gate.
- **Mid-grid rung (2):** before each secondary group, if ledger + remaining projection exceeds
  170 h, drop secondary groups from the end of the registered list (occupancy, aging trend,
  dark-window, R11b, R11), record each drop, and mark the affected predictions `not_evaluable`.
  Training length never changes inside the grid.
- **Secondary-group ordering (engineering, one line):** the secondary groups run **after** the main
  arms, in the registered order, wherever the per-cell data lifecycle allows (the plan already runs
  G1–G3 main pairs first, then G4 R11+R11b, then G5 law-T; occupancy/dark-anchor ride with the main
  cell whose data they reuse, as the plan specifies).
- **Configs:** `lucid_run_v0_optionA.json` stays committed and unused; a rung-(1) derived config is
  added the same way (`derived_from` the registered config, registered default and tests unchanged).
- **β recording:** `sigreg_jepa_anchored` and `sigreg_jepa_anchored_dark` at β = 1000 are
  **`edge_selected`**; no grid extension and no re-selection after results. `anchor_only` runs at
  β = 1000 labelled **`beta_not_qualified`** as a descriptive control; any prediction clause needing a
  qualified `anchor_only` β is `not_evaluable`, and the dev finding itself (the anchor alone never
  reaches the physics threshold at any β) is reported as a development result.

- **2026-10-20 — the grid protocol and its β labels are staged as committed derived artefacts.** Three
  named derived files, none of them replacing the registered default (which stays at 10 000 steps,
  5/3 seeds, and keeps its green test suite): **`lucid_run_v0_rung1.json`** (sha `d5197c6d…`;
  10 000 steps, 3 seeds for every main arm, secondary retained, `fallback_rung: 1`, 165 runs) is the
  grid protocol per the author's G2 amendment; **`lucid_run_v0_optionA.json`** (sha `a1476d61…`)
  stays committed and **unused** unless the rung-(1) re-projection exceeds 160 GPU-h, in which case it
  is applied with no new gate; and **`step3_beta_decision_grid.json`** keeps the frozen β selections
  and adds the author's labels (anchored arms β = 1000 `edge_selected`, `anchor_only` β = 1000
  `beta_not_qualified` as a descriptive control; no re-selection, no grid extension). The frozen
  step-3 decision itself is unedited.
- **2026-10-20 — dev-cohort §3 readings (before the floors, to be completed).** Trained dev probes:
  `sigreg_jepa` physics 0.32363 m, module-gain **propagated −0.00182**, **run-mean −0.00396**,
  pmt-gain propagated −0.00011; `mae_mse` physics 0.11349 m, module-gain propagated −0.00062,
  **run-mean 0.31470**, pmt-gain propagated −0.00010; `sigreg_jepa_anchored` physics 0.31082 m,
  module-gain propagated −0.00228, run-mean −0.00416, pmt-gain propagated −0.00012. The random-init
  floors were not stored by the dev cohort (the `random_init_floor` block is empty and no floor
  artefacts exist), so the floor comparison the author asked for is being measured on the host by
  regenerating only the development probe datasets, computing `ensure_floor` for the three arms, and
  deleting the regenerated data afterwards. The anchor improves the vertex resolution monotonically
  with β (plain 0.32363 m → 0.31082 at β = 0.1 → **0.20788 at β = 1000**), which is recorded as a
  descriptive development finding and does not change the selection.

## Dev cohort floors measured — no arm carries F1 above its random-init floor (20 Oct 2026)

- **2026-10-20 — random-init floors computed on the dev cell** (CPU, untrained encoder with the shared
  architecture, init seed 0, the regenerated dev probe datasets; identical for all three arms because
  only the losses differ). Artefacts: `dev_floors_summary.json` sha256 `66930f71…`, per-arm floor
  JSONs under `/ceph/dwong/jepa/returns/wp5_dev_floors_<arm>/floors/`, dataset evidence
  `DEV_FLOORS_EVIDENCE.json` sha256 `e1e72890…`. **Floor:** physics vertex resolution **0.16188 m**
  (median 0.14201, n = 10 000, failure 0); module-gain **propagated R² 6.4e-05**, **run-mean
  R² 0.39438**; pmt-gain propagated −1.48e-04, run-mean 0.26281; module-gain per-event 0.01718;
  pmt-gain per-event 0.02329.
- **2026-10-20 — the author's §3 reading is confirmed: plain JEPA does NOT carry F1 above the floor,
  and neither does any other arm.** Trained dev values: plain `sigreg_jepa` physics 0.32363 m,
  module-gain propagated −0.00182, **run-mean −0.00396**; `mae_mse` physics 0.11349 m, propagated
  −0.00062, **run-mean 0.31470**; `sigreg_jepa_anchored` physics 0.31082 m, propagated −0.00228,
  run-mean −0.00416. The run-mean floor (**0.39438**) is above every trained module-gain R² — plain
  JEPA sits far below it, and `mae_mse`'s 0.31470 is **also below the floor**, so it is not evidence
  of above-floor F1 either. (Only `mae_mse`'s physics 0.11349 m is better than the 0.16188 m floor.)
  Plain JEPA does not carry F1 above its random-init floor at development scale, so the dev cohort
  shows **no F1 trap for JEPA here** (unlike the bridge, 0.525) and **R6/R7 may come out null** on
  the grid. This decouples the two explanations: the §3.5.3a detectability ceilings (F1 0.6835 at
  s = 0.05 and 0.8898 at s = 0.10, both above the 0.6 requirement) say the run-mean probe *can*
  measure F1 if the representation carries it — the limitation is the learned representation, not
  detectability. Recorded as a development finding before the grid completes, as the author asked.
- **2026-10-20 — descriptive anchor finding (unchanged selection):** the anchor improves the vertex
  resolution monotonically with β (plain 0.32363 m → 0.31082 at β = 0.1 → **0.20788 at β = 1000**).
- **2026-10-20 — quota headroom decision for the grid (Lead engineering, recorded).** One main-cell
  group's datasets need 5 420 entries; starting at free ≈33.5 k leaves a generation peak of ≈28.1 k,
  below the guard's **30 000 start minimum** — not because of the hard 500 k quota (≈28 k free is
  5.6 % of capacity) but because of our own conservative start threshold. Decision: (i) reclaim the
  remaining jepa-internal, non-result artefacts (superseded `wp5_lucid_dev_v6` root, old WP4a result
  roots, old job dirs; ≈1 100 entries, evidence already recorded), and (ii) during the grid period
  lower the guard profile's `free_entries_start_min` from 30 000 to **26 000**, recorded and reverted
  to 30 000 after the grid. The guard's SHED minimum (10 000) is unchanged, no other §6.2 limit is
  touched, and the hard quota cap stands. The alternative (raising `ceph.quota.max_files`) needs a
  filesystem administrator and is requested in the report but is not a blocker.

## A9 registered — differences between §3.7 and the DECISIONS/code record (20 Oct 2026)

The author's A9 amendment was committed unchanged as `45a2303` (`jepa: register runbook amendment A9
(author revision)`), sha256 `9f20042cd48ddb978385f227946d722f5697e4c6f4fa79e7eb4a6457ce4851bb`
verified before the commit. Per the author's instruction the runbook was **not edited**; the
following differences are listed for correction, with `DECISIONS.md` (and the code) remaining the
authority until then. A9 adds no new instruction; it registers the 17–20 Oct decisions.

1. **Grid volume in §3.7.2 vs the rung-(1) config:** §3.7.2 says "Expected volume: 159 runs
   (6 main cells × 21, R11 9, R11b 9, dark-window anchor 3, law T 9, occupancy 3)". The committed
   rung-(1) config (`lucid_run_v0_rung1.json`, sha `d5197c6d…`) derives from the author's own
   rung-(1) wording "**3 seeds for every main arm**, all secondary runs retained" and therefore keeps
   R11 at its registered **5 seeds → 15 runs**, giving **165 runs** (main 126 + R11 15 + R11b 9 +
   dark 3 + law T 9 + occupancy 3). The 159 figure is exactly the **option-A** volume (where R11 was
   cut to 3 seeds). The grid will run the 165-run set (DECISIONS authority); if the author intends
   R11 to be cut to 3 seeds under rung (1) as well, the rung-(1) sentence should say "3 seeds for
   every arm".
2. **Detectability JSON and ceilings in §3.7.4:** §3.7.4 cites `wp4b_detectability_wcte_r6.json`
   (sha256 `1e4407ac…`) with F1 ceilings 0.686 / 0.893 / 0.261 and "the largest shift was +2.2e−3".
   That JSON is the earlier 0.05-marginalised run, recorded here as **superseded**; the current
   frozen authority is the s_p-coherence re-freeze
   `results/jepa/20261018_wp4b_detectability_r6/wp4b_detectability_wcte_r6_consistency.json`
   (file sha256 `411190bc…`, content `8c9b96d2…`), whose ceilings are **F1 0.2574 / 0.6835 / 0.8898,
   F3 0.3228, F2 0.9189, T 0.9855**, with the largest shift **−3.5e−3**. The frozen yields, the
   ladder steps, the F2 a = 0.50 choice, R12 `not_evaluable` and G6 false are identical between the
   two; no pass/fail changed. All three configs point at the consistency JSON.
3. **Physics-reference input in §3.7.3:** the text says the time-of-flight fit "reads absolute digit
   times". The code's sparse store carries **trigger-relative token times** (A9 `t_token`), so the
   reference estimator currently reads those; `HitReadout.absolute_time_ns` carries the absolute
   times for a caller but the store has no absolute-time array. The fit estimates t0 itself, so a
   constant per-event offset is absorbed and no numeric effect is expected, but text and code differ
   in what is read. Not changed without the author's word.
4. **`fallback_rung` in manifests in §3.7.2:** the text requires `fallback_active` **and
   `fallback_rung`** in every manifest. The code recorded only `fallback_active`; this was a code
   gap, now fixed (`b4f76d5`): manifests record `fallback_rung` (1 for the seed cut alone, 3 for
   option A) plus `fallback_rungs` (`[1]` or `[1, 3]`). No runbook edit.
5. Other numbers in §3.7 checked and **consistent** with DECISIONS/code: B1-ext 2.45 of 4 GPU-h (our
   2.4484 h); the β table and labels (`edge_selected`, `beta_not_qualified`); the ridge step-2
   decision and the 0.142446 m reference; the L40S 170 h cap; TF32 1.22×/1.15×; the start rule
   (151.1 ≤ 160 h); rung (2) drop order; the readout `5f4ecc0` / merge `4e07c0c`, `v_dark = 0.00109
   PE²`, the token-time rule, the emission-offset leak guard and the R7b dark-only diagnostic.

- **2026-10-20 — the grid is running at rung (1); A9 verified, quota relaxations recorded.** G1 job A
  launched at `368e1c2` (clean; `lucid_run_v0_rung1.json` `d5197c6d…`, registered default
  `ce452513…`, β decision `a574429f…`, overlay hashes verified; precheck file
  `grid_G1_jobA_precheck.json`), 21 runs on `main_s002_across`, GPU 0, 45 h launcher budget,
  17:23:57Z. `enumerate_cells` confirmed **165 runs**; the operator independently confirms the §3.7.2
  "159" figure is the option-A volume and that R11 keeps its registered 5 seeds, so there is **no
  run-count disagreement** — the difference list stands as recorded.
- **2026-10-20 — reclaim and guard relaxation for the grid period (Lead-authorised engineering).**
  Reclaim (`RECLAIM_2026-10-20_EVIDENCE.json` sha `68c99011…`): 70 targets / 457 entries of
  superseded pre-A9 dev root, old WP4a result roots and old job/probe/smoke dirs; the guard clone,
  floors, plans and `returns/wp5_grid` kept; nothing outside `/ceph/dwong/jepa`. Free entries after
  reclaim **33 976**; with the G1 generation peak ≈28.5 k, above the temporary start minimum. The
  guard's `free_entries_start_min` was lowered **30 000 → 26 000** for the grid period
  (`GUARD_PROFILE_RELAXATION.json`, script `fc330571…` → `e0af3f91…`), the guard was restarted and
  verified (OK, `caps.free_entries_start_min = 26000`, **SHED minimum 10 000 unchanged**); it will be
  reverted to 30 000 and the guard restarted after the grid. The ledger stays at **32.8460 / 170
  GPU-h** with no speculative rows; G1's wall is booked when the cell completes. Start decision
  unchanged: ledger + projection 151.1 ≤ 160 GPU-h → rung (1) is the grid of record.

## Author decision, 22 October 2026 — A10 mixed event domains

- **2026-10-22 — A10 slots decided (author).** The author's message: "generator：hybrid；
  e⁻ 样本来源（analytic 先做；mixing 1:1:1；primary domain：particle primary + photon control".
  The earlier instruction "μ⁻ 和 e⁻ 都要" remains in force. Recorded decisions:
  generator **hybrid** (analytic Cherenkov production + a mandatory PhotonSim μ⁻ validation
  subset); electron source **analytic first**; mixing **1:1:1** photon:μ:e (333/333/334 per
  1000-event training run; 17/17/16 per 50-event probe run); primary domain **particle (μ+e)
  for R6/R7**, with the photon domain as the isotropic control and the pooled result secondary.
  A10 is drafted in `docs/jepa/GATE_A10_MIXED_DOMAINS_2026-10-22.md` and
  `docs/jepa/AMENDMENT_A10_MIXED_DOMAINS_2026-10-22.md`; it is **not in force** until the
  orchestrator commits the runbook §3.8 unchanged as
  `jepa: register runbook amendment A10 (author revision)`. No outcome-generating run may use A10
  before that registration. Drift values s, s_p and a remain pending from Junjie.

## A10 audit (author, 23 Oct 2026)

### §0 Ruling — verbatim

"A10 audit (author, 23 Oct 2026). Record this verbatim.
The A10 grid may continue only while ledger + remaining projection stays ≤ 170 L40S GPU-hours. No
result from it is reported until §2 and §3 below are recorded. Every A10 step since 22 Oct needs its
one-line `DECISIONS.md` entry and an updated `STATE.md`; that is part of the protocol, not optional."

### §1–§6 audit entries (23 Oct 2026)

- **2026-10-22 — A10 registered.** Runbook §3.8 committed at `ab8c916`; the §3.8 header still says
  "DRAFT, NOT IN FORCE" but registration put A10 in force. Header text is stale; author will clean it
  at the next runbook revision.
- **2026-10-22 — A10 generator/per-domain implementation.** Analytic Cherenkov generator and
  particle truth (`6d65dc6`), per-domain probe/floor/evaluation plumbing (`9c92469`), PhotonSim
  validation and mixed detectability helper (`31e7fb3`), and transport adapter (`b3d9713`); mixed
  500-step smoke passed after the particle first-arrival fix (`b0b984f`).
- **2026-10-22 — A10 sensor-mapping close-out and Gate V calibration.** LUCiD transport sensor
  indices are a permutation of the WCTE geometry order; mapping fix `0f6c1c4`. Calibrated analytic
  yield 2248.5396 per MeV and validated on 10 held-out contained 500 MeV μ⁻ events (module pattern
  0.902, log pattern 0.815, yield ratio 1.006, time 0.157 ns; config `lucid_bombs_v2_calibrated`,
  commit `0bbdd60`; evidence `docs/jepa/CALIBRATION_CHERENKOV_2026-10-22.md`).
- **2026-10-22 — Mixed smoke `--evaluate` + per-domain floors recorded.** Run `a10_smoke_eval_3`,
  500 steps, exit 0; domains photon/muon/electron/particle; probes and floors committed under
  `results/jepa/20261022_a10_smoke_eval/`; smoke floor path added in `e90b71b`.
- **2026-10-22 — Detectability v1 frozen (now superseded).** `a10_detectability_freeze_v1.json` used
  20 events, no profiling and gate 0.5; retained only as the old side of the §2 comparison.
- **2026-10-22 — Dev step 2 complete.** Mixed dev cohort, 100k events; decision **ridge**,
  **run_mean**; pooled reference 0.482004 m; frozen at
  `configs/decisions/a10_dev_step2_probe_decision_2026-10-22.json`.
- **2026-10-23 — Dev step 3 complete/frozen.** β audit exit 0 after 11.4153 GPU-h;
  `sigreg_jepa_anchored` β=100, `sigreg_jepa_anchored_dark` β=100, `anchor_only` none qualified;
  frozen at `configs/decisions/a10_dev_step3_beta_decision_2026-10-23.json`. Per-domain β table is
  not recorded: `dev_cell` evaluated photon only; recorded as a gap.
- **2026-10-23 — A10 grid launch.** 165 runs, split 87/78 across GPUs 0/1 with the guard; grid-0
  `a10_grid_gpu0_wait` and grid-1 retry `a10_grid_gpu1_retry`; decisions frozen before launch.
- **2026-10-23 — G1 predecessor closed.** G1 job A/B (1272.63 s + 318.89 s, exit 2 device mismatch
  fixed in `7e6cfa8`) booked and marked `superseded_by_A10`; no G1 result reused.
- **2026-10-23 — §1 ledger booked.** Charged total **54.413992**; remaining projection **52.631150**; ledger+projection **107.045142**
  ≤160 h start rule and ≤170 h cap; no A10.6 fallback triggered.
- **2026-10-23 — §2 detectability re-run.** Photon profiled A9 procedure (50 events, gate 0.6):
  F1 0.2574/0.6835/0.8898, F3 0.3226 → R12 not_evaluable, F2 0.9190, T 0.9857, G6 false; v1
  withdrawn. Particle domains lack profiling; naive F1/F3 marked **lower_bound_unprofiled**; their
  R6/R7/R12 verdicts are not established until profiling is implemented.
- **2026-10-23 — §3 readings recorded.** Per-domain reference: photon 0.1421 m, muon 0.6520 m,
  electron 0.6885 m, pooled 0.4820 m; ridge stands. Representation: plain `sigreg_jepa` effective
  rank 7.021, final SIGReg 9.814, predictor 0.0133; photon vertex floor 0.1609 vs trained 0.2644
  (**below floor**); module-gain floor 0.2185 vs trained −0.0044 (**below floor**).
- **2026-10-23 — §4 contained counts.** Probe-test contained: muon 452 (<500 → R13
  `not_evaluable`), electron 2570 (evaluable); photon containment not applicable. Grid energy ranges
  unchanged.
- **2026-10-23 — §5 paths/deviations.** Author authority for `/work`: 2026-10-22 message
  **"you can work from `/work/dwong`"**. A10 work root `/work/dwong/jepa_a10_work/` and code clone
  `/work/dwong/jepa_a10` are a **deviation** from the `/ceph/dwong/jepa/`-only hard rule; author to
  rule. Correction line (author, 21 Oct 2026): **commit `dcdf6ce` is author-approved.**

### Author task brief — 23 Oct 2026

- **2026-10-23 — Author approval quoted verbatim.** "我认可你的建议，把1 2 3 4步写成一份发给
  orchestrator 的简短任务说明，让它一项一项做" (23 Oct 2026, 23:17 CEST).
- **2026-10-23 — Follow-up confirmation quoted verbatim.** "是的，我确认这两点" (23 Oct 2026,
  23:21 CEST): (i) `anchor_only` runs at the anchored arms' β (option A, step 4); (ii) these
  approvals are recorded verbatim.
- **2026-10-23 — Task brief committed unchanged.** `docs/jepa/TASK_2026-10-23_LABEL_SHORTCUT.md`
  at `cef3dd5`, before starting.
- **2026-10-23 — Step 1 grid pause.** grid-0 stopped at 15 completed runs (exit 2, anchor_only
  null β); grid-1 paused after its in-progress run with SIGTERM (exit 143, no SIGKILL), 10
  completed runs. No JEPA training job remains on deepthought2; 25 completed runs listed in
  `results/jepa/20261023_label_shortcut_diag/grid_completed_runs.json`. Ledger charged
  **67.471034** L40S GPU-h; projection deliberately pending step 2/3 re-projection.

### Step 2 label-shortcut diagnostic — 23 Oct 2026

- **2026-10-23 — Step 2a dependence check.**
  - Trained `sigreg_jepa` s9000 full-input label context share **0.1273**; random-init share
    **0.2304**; target-only encoding gives `G_context = 0` exactly (32 dev events, within split,
    8 projections/event).
- **2026-10-23 — Step 2b paired result = confirmed.** Fixed target-only labels reduce photon
  vertex resolution from 0.264400 → 0.219259 m (seed 9000; ratio 0.8293) and 0.233759 →
  0.204281 m (seed 9001; ratio 0.8739). Both ≤0.90 → **confirmed**; no S1/S3. Effective ranks
  4.56/6.19; module-gain run-mean still below the 0.218548 floor. Step-2b GPU wall **2.8193 h**
  ≤3 h. Evidence `results/jepa/20261023_label_shortcut_diag/`.

### Step 3A — label fix and re-selection (24 Oct 2026)

- **2026-10-24 — Label defect fix committed.** `_encode_views` labels now come from
  `jepa.encode(features, hidden_mask=~target)`; full unmasked pass retained only for anchor and
  evaluation (`c2dce54`). Reverse-leakage tests for within/across/balanced added; 24 arm tests and
  full suite **471 passed** on deepthought. Three guard launcher tests fail on deepthought2 only
  because they hardcode `deepthought`; they pass on deepthought.
- **2026-10-24 — Fixed-label A10 dev step 3 launched.** `a10_dev_step3_fixed` at `c2dce54`, same
  seed/rule/config as frozen step 3; plain reference from the old step-2 probes; new decision will
  supersede `a10_dev_step3_beta_decision_2026-10-23.json` as `superseded_by_label_fix`.
- **2026-10-24 — Step 2b ledger.** Diagnostic GPU wall **2.8193 h including the existing old seed9000 run** (new runs 2.1218 h:
  old seed9001 2407.54 s, fixed seed9000 2575.51 s, fixed seed9001 2655.57 s). Ledger charged
  **69.592873** L40S GPU-h before the fixed step-3 run.
- **2026-10-24 — Grid bookkeeping at pause.** 25 completed runs listed; **affected arms**
  `sigreg_jepa`, `sigreg_jepa_anchored`, `sigreg_jepa_anchored_dark`, `oracle_nominal_twin` are
  labelled `superseded_by_label_fix`; `mae_mse`, `mae_chi2`, `ijepa_ema` remain valid. The only
  code diff is `lucid_arms._encode_views`; those three paths do not use the changed labels.
  Evidence `results/jepa/20261023_label_shortcut_diag/grid_bookkeeping.json`.
- **2026-10-24 — Step 3A review PASS.** Fresh reviewer verdict: production fix correct; target-only labels have exactly zero context gradient; full pass retained for anchor/evaluation; other arm numerics unchanged. Two non-blocking findings: the balanced regression test was degenerate, and `WP5_ARMS.md` still described the old full-input labels. Both corrected (balanced test now uses a sparse single-event view with `extra_hidden` non-empty and a positive target-gradient control; WP5_ARMS updated). 24 arm tests pass.


### Pre-grid extensions — 25 Oct 2026

- **2026-10-25 — Author authority (24 Oct 2026, 23:32 CEST), verbatim:**
> "我需要你改几个细节，重新做一遍实验，
> 1. 线性模型组，加两组实验，一组是纯噪音没物理，一组是纯物理没噪音，来验证猜想
> 2. 第二，在JEPAtraining 时候泄露是在lucid data上对吧，然后避免泄露不可以用打乱组别对吧。因为组的顺序也是物理信息的一环
> 3. 在toy model的traceset里面，我们要是加上各组arriving time 逐组变晚。加一个这样的causal关系呢？
> 4. 建议恢复 grid 前先做的三项检查，你准备好code和prompt，setup好后我让agent去跑
> 5. 在给俊杰的report里面加上结果table，以及各个量是如何算的，特别是物理量和噪音比
> 6. 我们对JEPA已经做了的修复，sound吗？用两个encoder会不会解决问题，还有别的解决办法吗？双保险"

- **2026-10-25 — Author authority (25 Oct 2026, 02:00 CEST), verbatim:**
> "全部都可以在deepthought上跑对吧？不如把所有写成一个长runbook，我直接交给deepthought agent"

- **2026-10-25 — Superseding execution instructions.** Archive the supplied runbook byte-for-byte as `RUNBOOK_2026-10-25_PREGRID_ALL.md` before any new experiment. It replaces the 24 Oct pre-grid task and supersedes the 23 Oct task step 4. The grid stays paused; no grid decision file or grid launch is authorized. All new results are extensions, `registered: false`, outside registered results; existing frozen decisions and verdicts remain intact.
- **2026-10-25 — Worktree choice.** Reuse the clean `/private/tmp/veritas-jepa` worktree on `JEPA` at `fecc1f7022e2ee90ef5f9d480d11ad7198bfa71b`; preserve the PMT checkout and its untracked results.
- **2026-10-25 — SSH configuration.** Local SSH config defines only the FQDN hosts `deepthought.etp.kit.edu` and `deepthought2.etp.kit.edu`; short aliases do not resolve. Use those configured FQDNs for the same authorized hosts. Both connections and the shared `/ceph/dwong/jepa/` path pass read-only checks; S5 is not triggered.
- **2026-10-25 — Missing execution input.** The supplied attachment contains the runbook only. `0001-jepa-extensions-2026-10-24.patch` is absent from the attachment directory and local attachment/download/worktree searches. Do not reconstruct the author's patch, substitute new code, or claim Part A acceptance. Dependent Parts A–F await that exact patch; independent read-only Part 1 status checks continue.

- **2026-10-25 — Part 1 completion received.** Remote `run_status.json` reports exit 0, 53,341.44 s = 14.817067 L40S GPU-h (within 15 h); local ledger provisional charge replaced by final measured charge. Remote decision preserved; not frozen locally before the S4 stop. Local booked L40S subtotal 84.148833 h; projection not refreshed, no budget clearance inferred.
- **2026-10-25 — S4 STOP.** Automatic approval review rejected `git push origin JEPA` to `https://github.com/seanxia8/VERITAS.git` before execution, citing unverified private-content egress and insufficient explicit payload/destination authorization. No indirect push or remote code transfer attempted. All execution stops pending author resolution. Missing patch is an additional input dependency. Report and lettered options: `results/jepa/20261025_pregrid_checks/README.md`.

- **2026-10-25 — Explicit destination/payload authorization received, verbatim:**
> 我授权把本地 `JEPA` 分支（包括 `0001-jepa-extensions-2026-10-24.patch` 的内容和你刚才的提交）推送到 [https://github.com/seanxia8/VERITAS](https://github.com/seanxia8/VERITAS) 的 `JEPA` 分支。这是我们项目自己的仓库（原名 seanxia8/oracle），代码按 runbook §6.1 只能经 GitHub 到服务器。patch 在 `~/Downloads/0001-jepa-extensions-2026-10-24.patch`。按 RUNBOOK_2026-10-25_PREGRID_ALL 从 Part A 继续。
- **2026-10-25 — Part A input recheck.** The named `/Users/wongdowling/Downloads/0001-jepa-extensions-2026-10-24.patch` returns `No such file or directory`; filename searches of Downloads, Desktop and Codex attachments find no patch. Push authorization is now explicit; continue the authorized documentation push while awaiting the exact patch. No substitute implementation, GPU job or grid launch.
- **2026-10-25 — S4 cleared by authorized push.** `git push origin JEPA` succeeded, advancing GitHub from `fecc1f7` to `0ef3df2`. Exact author patch remains unavailable at the supplied path; Part A application, tests and remote pinned clones remain pending. No experiments launched.

- **2026-10-25 — Author patch verified and applied.** Downloads patch SHA256 `1f17e823e8238efc369ce9c0527de3fe7927162329a18a2f28fd798104621817` matches the author value; `git am` produced `2cadfd899b3e46b01fc58bc1a13715497f63b330`, pushed through the authorized GitHub remote. The runbook was already committed unchanged before any experiment.
- **2026-10-25 — CPU validation environment repair.** The historical local CPU interpreter no longer has pytest; the main PMT interpreter has no torch. Full-suite verification therefore runs CPU-only in the existing Deepthought conda environment. Deepthought2's pinned cu124 environment lacks pytest, so its required CPU-only smoke uses its existing conda interpreter (9/9 passed); all L40S training still uses the unchanged cu124 environment plus its established read-only numeric overlay. No environment installed or modified. Patch was pushed before full-suite completion solely to make code available for remote CPU testing; no GPU starts before test acceptance.
- **2026-10-25 — Extension runner engineering corrections before GPU execution.** Stream event-statistic token encoding in 8-event chunks, retaining event/RNG order; respect the smoke's 200-event-per-probe-run cap in every evaluation stage. Reuse the fixed-label training cache's nonregistered key; mark every newly generated strong-drift cohort nonregistered. These affect only the extension script and its tests, never registered runner/config paths, targets or criteria.
- **2026-10-25 — Read-only dev-data alias.** Existing dev caches reside under `/work/dwong/jepa_a10_work/data_dev`. To enforce this runbook's `/ceph/dwong/jepa/` write boundary, use a new extension cache root under `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/` containing links to the existing frozen nominal caches (verified manifest hashes in `provenance.json`). Only new strong-drift keys are written under the new root; no old cache or external path is modified. This path remapping preserves the exact events and seeds and prevents dev-data regeneration.
- **2026-10-25 — Part 1 decision frozen locally.** Copy the completed fixed-label decision to `configs/decisions/a10_dev_step3_beta_decision_labelfix_2026-10-25.json`; preserve the original 23 Oct decision. Remote source SHA256 `1a3da325536736927b9896d21c67df39615c58c5ad2d027fab3219e78d41f5cf`. Both anchored arms select 1000 with the old plain; anchor_only is none_qualified. Part B re-selection is separate, nonregistered, with no grid decision file.
- **2026-10-25 — Exact dev inputs.** `--config src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` SHA256 `e6730ac14ddf475b857a0f8b07407f8b54e487e2e7af827fd00bcecc5268e034`; `--generator-config src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` SHA256 `90ac1601c40171eeb7c669a08df1fe6e6b5b332fee2a3cca689ae057a51dfe88`; `--geometry /ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`. These config bytes match the fixed-label run source. Original data root `/work/dwong/jepa_a10_work/data_dev`; write-safe alias as above.
- **2026-10-25 — Part C checkpoint old9000.** `/work/dwong/jepa_a10_work/dev_step2/runs/dev_s005_within__sigreg_jepa__s9000/final.pt`; SHA256 `b1632266f2f8ca9fd49f9a06383734fda77432f5247294e79ed65bc1936f7503`. Read-only; no training.
- **2026-10-25 — Part C checkpoint old9001.** `/work/dwong/jepa_a10_work/old_s9001/runs/dev_s005_within__sigreg_jepa__s9001/final.pt`; SHA256 `14e3590f9ed6649d86f3de751a56c3590bb748f0b4a15984bb056373bff97ceb`. Read-only; no training.
- **2026-10-25 — Part C checkpoint fix9000.** `/work/dwong/jepa_a10_work/fixed_s9000/runs/dev_s005_within__sigreg_jepa__s9000/final.pt`; SHA256 `2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714`. Read-only; no training.
- **2026-10-25 — Part C checkpoint fix9001.** `/work/dwong/jepa_a10_work/fixed_s9001/runs/dev_s005_within__sigreg_jepa__s9001/final.pt`; SHA256 `95cb5e3f4f158a719098a4b3dc573eebb8c71645c22290f6483223ccc04c484e`. Read-only; no training.
- **2026-10-25 — Part C checkpoint mae9000.** `/work/dwong/jepa_a10_work/dev_step2/runs/dev_s005_within__mae_mse__s9000/final.pt`; SHA256 `efd54c2586f8b5378dfe2febffe2ceb954c50185f6b07939c2b892c825270664`. Read-only; no training.
- **2026-10-25 — Part C checkpoint anc100.** `/work/dwong/jepa_a10_work/dev_step3_fixed/runs/dev_s005_within__sigreg_jepa_anchored__s9103__b100/final.pt`; SHA256 `04136cd290f9863a50059767dff4b19c3848f4ff0900843cc1b6a9b7082c1ba7`. Read-only; no training.
- **2026-10-25 — Delayed bridge extension metadata/aggregation correction.** Before Part E, ensure full-length delayed runs and their statuses/aggregates are explicitly `registered: false`; reject mixed/missing delay provenance. With `JEPA_TOY_DELAY` set, aggregate exactly the runbook's three arms × seeds 0–2, without requiring the unrequested parallel-attention arm or using `--allow-partial` for a complete delayed extension. Environment-unset registered behavior, training numerics and criteria are unchanged. Delay extension budget remains 8 Titan GPU-h.
- **2026-10-25 — Full-suite engineering interruption (no scientific run).** Buffered pytest output obscured the current test position. Source inspection of `test_beta_dev_extension_refuses_bad_invocations` established that `run_beta_dev` generates the full development cohort before rejecting missing/invalid `--beta-grid`. Preserve their logs and partial test data; interrupt only the two owned pytest processes and move unchanged beta-grid argument validation ahead of cohort generation. Valid invocations, data keys, training numerics and registered scientific behavior are unchanged. Re-run the complete final suite after this engineering fix.
- **2026-10-25 — Part A linear reproduction PASS.** Deepthought CPU, nice 10, OMP/BLAS <=4: 110 rows, 810 per-row/method maximum-angle comparisons; maximum absolute difference `3.9412384467141237e-10` degrees <=0.1 degrees; no missing-angle discrepancies. Compact acceptance plus received rows under `results/jepa/20261025_pregrid_checks/`.
- **2026-10-25 — Part B fixed-plain re-selection.** Frozen independent output `a10_dev_step3_beta_decision_fixedplain_2026-10-25.json`, `registered: false`: plain 0.26440029142206084 -> 0.21925859104673096 m; both anchored arms remain beta 1000; anchor_only remains none_qualified. Option A policy follows the anchored beta (1000) for descriptive anchor_only execution, without changing its qualification status or writing a grid decision.
- **2026-10-25 — Part C scheduling.** Check the specified `init9000=sigreg_jepa:init:9000 --expect init9000=0.160904` first as a cheap reproducibility gate, then the six immutable checkpoints if it passes. The existing floor utility defaults to seed 0; retain the runbook's seed 9000 and original expectation rather than silently substituting seeds or tolerances. Stop immediately if its observed full-input difference exceeds 1e-3 m.
- **2026-10-25 — Interrupted test attempts are not acceptance evidence.** SIGTERM did not immediately exit the two owned pytest processes; a subsequent SIGINT did, with no SIGKILL. Original-patch attempt ended 469 passed / KeyboardInterrupt; the intermediate attempt ended 4 failed, 54 passed / KeyboardInterrupt, including interrupted fixture training with missing probes. Preserve both logs and all partial test data; neither result qualifies. Accept only the fresh complete final suite without interruption.
- **2026-10-25 — Part A population reproduction PASS.** 81 angle entries; maximum absolute difference 9.80163983173e-10 degrees (tolerance 0.1 degrees). Both reproduction tables and the Part B decision match their remote SHA256. Reproductions ran 00:24:18Z–00:30:46Z (388 s CPU wall envelope; not LUCiD generation).
- **2026-10-25 — Exact final-SHA deepthought2 CPU smoke PASS.** At `3ecdc5cc20663d62f32ad52432643aa0a95db6b0`, the two prescribed test modules pass 12/12 in 10.77 s (the original 9 plus three engineering regressions). Existing conda interpreter used CPU-only; no environment mutation. Runbook byte hash and no-touch path diff also verified.
- **2026-10-25 — Part A final acceptance.** Full final suite at `3ecdc5c`: 483 passed in492.68 s, uninterrupted (480 original-plus-patch tests and three engineering regressions). Deepthought2 final CPU smoke12/12 passed. Both CPU reproductions pass and received hashes match. Part C exact init9000 gate launch requested through guard/front door, one Titan, no training; ledger booking pending final wall.
- **2026-10-25 — Resource HOLD.** At00:41:13Z Deepthought guard reports free entries11,468 <30,000; no further launches while HOLD. Existing `pregrid_C_init9000` evaluation continues under the running guard. No cleanup, quota/guard modification or assumption about ownership of the quota change. SHED at<10,000 remains in force.
- **2026-10-25 — Part C infrastructure failure, not S2.** `pregrid_C_init9000` exited after `OSError122 Disk quota exceeded` writing `init9000/full_input/probes.json.tmp`; its launcher also failed to create final `run_status.json`. No `inspect.json` or numerical reproduction verdict was produced. Preserve all directories/logs/checkpoints and do not retry while resource-HOLD persists. Conservative upper-bound charge405 s=0.1125 Titan GPU-h; exact launcher final wall unavailable.
- **2026-10-25 — Quota/guard evidence and bounded inventory.** `/ceph/dwong` max500000 entries, rfiles423702+rsubdirs79022=502724 (2724 over); byte quota0 (unlimited). Need at least32724 released entries to reach30000 free. Four unused transition clones created in this execution total10407 entries (~1.82GB allocation), insufficient to recover the start floor. Current3ecdc5c clone retained; no files deleted and no quota/guard settings changed. Guard atomic status writes also hit quota; require quota recovery and a fresh live OK guard before any launch. Read-only inventory and exact error logs are in `infra/`. Ask author to restore quota or supply a separately scoped cleanup list; this is an external resource HOLD, not an invented S1–S5 numerical gate.

- **2026-10-26 — Author resumes the runbook from Part C; grid stays paused.** Quota cleanup was separately authorized and completed. Resume request: "配额问题解决了，按 runbook 从 Part C 继续，保留之前失败的记录，grid 继续暂停。" Preserve `results/jepa/20261025_pregrid_checks/` as the failed-infrastructure checkpoint and all original partial outputs. Snapshot the failed job including its stale PID record before guard housekeeping. Restart only the Ceph-job guards from previously tested GitHub code `3ecdc5c`; existing `/work` guards and other users remain untouched.
- **2026-10-26 — Part C retry provenance and budget.** Six checkpoint SHA256 values and both config hashes reverified against the 25 Oct provenance; current HEAD differs from execution SHA only outside src/scripts. Reuse the five frozen nominal data aliases under `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`; no generation. Evaluation has no resumable training state, so use new run ID `pregrid_C_init9000_retry1` and output `inspect_dev/init9000_retry1`, linked to the retained failed attempt. Keep `init:9000`, expected `0.160904 m` and `1e-3 m` tolerance unchanged; any failed reproduction invokes S2, without tuning/retry. Budget before retry: runbook Titan 0.1125/30 h, conservative L40S ledger plus remaining grid plus extensions 142.977123<160 h, generation 0.6535/48 h. New starts require fresh live guard OK and warm utilization window.

- **2026-10-26 — Part C init9000 passes as specified.** `pregrid_C_init9000_retry1` completed exit0 in785.11 s (0.218086111 Titan GPU-h), full-input0.16007161647162788 m versus0.160904 m (absolute difference0.00083238352837212 m<1e-3), half-pass0.17482120319755615 m. Keep seed9000 and frozen expected value unchanged. This is an extension result, not a grid decision. The old quota failure remains separately recorded and charged0.1125 h. Remaining six items await fresh guardOK; current foreign-GPU HOLD prevents new starts.

- **2026-10-26 — Bounded Part C wait queue deployed.** GitHub SHA `2b24630906d327231f05defa6d7ada5751988d2a` adds only the Part C coordinator and six gate tests to src/scripts relative to the accepted3ecdc5c scientific source. Both hosts pass18 CPU-only tests (6.82s/11.48s), no environment change. Remaining six checkpoint items are one inspect job, planned100 min based on6×13.1 min plus margin, timeout150 min; reserve2.5 Titan GPU-h, total runbook ceiling2.830586111<30. Queue waits on the existing fresh guard and front door; any failed expectation writes S2 with options, and the queue never auto-advances to D/grid. This is orchestration, not new training or a registered change.

- **2026-10-26 — Persistent follow-up not created.** Automatic approval review rejected a proposed15-minute heartbeat that would monitor Part C and later advance D–F, citing lack of explicit authorization for persistent scheduling/future GPU actions. No automation was created and no indirect replacement is attempted. The already authorized Part-C-only queue remains active and stops after Part C. Explicit author approval is required before creating that persistent follow-up.

28 Oct: handoff to new agent, author's request.
- **28 Oct — Part C failure preserved.** Case (e): queue exited 1 while constructing anchored arm without required beta, after five partial probe pairs. Evidence copied and SHA256 verified; charge 3894.59 s / 3600. Stop without relaunch or Part D–F; grid paused.
- **28 Oct — Author decision on the anc100 failure: option A** (verbatim, 28 Oct 2026, 23:21 CET: "做选项a"). Engineering fix, not a protocol change: `jepa_pregrid_checks.inspect` now rebuilds a beta-selected arm with the beta stored in its own checkpoint identity (must name the same arm; anc100 must carry beta = 100), refuses a beta arm at random init, and rewrites `inspect.json` after every item with `complete: false` until the last. Beta weights only the training loss, so no registered prediction, criterion, cell, dataset or frozen decision changes (not S3). The Part C queue gains `--attempt`; attempt `20261029` re-runs **all six** items in one invocation into its own job/queue/output paths, carries the failed attempt's 3894.59 s charge, and rejects a report that is incomplete or whose anc100 beta is not 100. The 26 Oct attempt and its evidence stay untouched. The queue itself still ends after Part C; what follows is set by the 28 Oct handoff.
- **29 Oct — Part C clean rerun passed.** f29137c, 496 local tests and 25 Deepthought CPU smoke tests passed. All five expectations independently checked <=0.001 m, complete:true, anc100 beta 100; retained old failure. Evaluation mismatch not supported (half/full 1.484355, 1.436397). Launcher 4239.36 s = 1.1776 Titan GPU-h; evidence `results/jepa/20261029_part_c/`; registered:false, grid paused.
- **29 Oct — D0 lambda property repair.** The lambda smoke failed before training because `LucidArm.coefficient_lambda` is read-only; preserve and charge 8.32 s, replace only the extension arm's immutable settings with a copy carrying λ=0.2, verify real-arm loss by test, and retry the smoke in a new output and job path after tests and GitHub deployment. No registered configuration or grid decision changed.
