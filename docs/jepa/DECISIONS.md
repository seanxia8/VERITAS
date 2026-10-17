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
