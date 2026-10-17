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
