# Runbook — Regenerate All Data with LUCiD, Archive the Analytic Toy Data, Retrain and Re-evaluate (r2-lucid), 21 Nov 2026

Status: authorised for immediate unattended execution. Multi-night campaign. The night-1 milestone is defined in §9.
Extension: `r2_lucid_2026-11-21`, `registered: false`.
Branch: `lucid-mc` (created from `origin/dev`) in worktree `~/Documents/VERITAS-worktrees/lucid-mc`. Push only to `origin/lucid-mc`; the author merges.
Remote clone: `/ceph/dwong/jepa/r2_lucid_repo`, a clone of branch `lucid-mc` on the shared filesystem, usable from portal1 and Deepthought.

---

## 0. Why this runbook exists

On 21 Nov 2026 we found that **every round-1 and round-2 dataset (WCTE, SK, JUNO, HK) was produced by our own analytic photon-bomb generator** (`src/jepa_noise/lucid_bombs.py`, `r2_generator.py`), not by LUCiD.
- LUCiD supplied only geometry configs.
- LUCiD transport (`noise_module_lucid/lucid_transport.PromptTransport`, pinned `eaf2e89`, absorption off) was used only for the Gate V comparison and a side study.

The analytic generator has:
- an isotropic point source;
- straight-line prompt transport with exp(−d/λ);
- no scattering or reflection;
- a Gaussian first-hit time with fixed σ_t;
- Poisson photo-electrons with SPE smearing, a threshold, dark counts and an NHits trigger.

**Goal.** Regenerate the datasets with the current public LUCiD, run the same models through the same training and evaluation, and report how the conclusions change. The analytic data are **archived, not deleted**. They stay useful as an exact-likelihood "toy world".

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `lucid-mc`:
1. **Author, 21 Nov 2026, 02:14 CET:**
   > "没经过lucid模拟啊，我们需要重新在portal1上（和deephtought是同一个文件系统）上克隆lucid，重新生成数据（lucid的数据！，这一点很重要），然后重新训练。你能不能帮我写一个runbook，然后让一个claude agent（在cli里面，能连上portal1，通宵去跑lucid生成数据），我们需不需要改架构，把数据从我们的玩具数据改到lucid，能不能一个agent自己通宵去干，克隆，生成正经lucid数据，archive之前toy数据，重新做training，拿结果"

   (Translation: "It never went through LUCiD simulation. We need to clone LUCiD again on portal1 (same filesystem as Deepthought), regenerate the data — LUCiD data, this is very important — and retrain. Write a runbook for a Claude agent in the CLI that can reach portal1 and run LUCiD data generation overnight: clone, generate proper LUCiD data, archive the old toy data, retrain, get results.")
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".
   - Never stop to ask.
   - At every decision point take the recommended option, record one line in `DECISIONS.md` and `$CC_STATE/notes/r2-lucid.md`, and continue.

### 0.2 Binding principles

- **P1 — LUCiD does the physics.** Photon generation, transport (absorption, scattering, reflection, wavelength effects, as supported by the cloned release), sensor response and stochastic sampling all come from LUCiD. VERITAS code may only:
  - (a) define an event source if LUCiD lacks one (an isotropic point source);
  - (b) convert LUCiD output into the VERITAS round-2 HDF5 schema;
  - (c) apply the documented NHits trigger, prompt window and token-time convention, plus dark counts, **only where LUCiD provides none**;
  - (d) apply the PMT electronics response from VERITAS's own `src/noise_module_lucid/` (QE/CE where LUCiD does not apply them, TTS per photo-electron, SPE charge, discriminator and walk) **only if LUCiD's sensor response does not cover these**. This module was designed to sit after LUCiD photon hits; see `docs/PMT_LUCID_INTEGRATION_REPORT_2026-10-16.md`.

  Every VERITAS-side step is listed in `docs/DATA_LINEAGE.md`.
- **P2 — No silent substitution.** Never fall back to `lucid_bombs` or `r2_generator` physics for LUCiD datasets. The only permitted degradations are the labelled sampling tiers of §L2. If even tier B is impossible, stop that item, record why, and continue with the others.
- **P3 — Archive, never delete.** Toy datasets and their results are neither moved nor deleted. They are labelled and made read-only (§L5).
- **P4 — Architecture unchanged.** The encoder, MAE/JEPA objectives, training loop and probes stay the same. Only data roots, dataset-derived constants (token time scale, prompt window, yield range) and the new generator module change.
- **P5 — Automatic gates, no questions.** Validation gates (§L4) have numeric criteria. A failed gate stops that detector, not the campaign, and is reported.
- **P6 — Other phases.**
  - `r2-ablate` is deferred: do not start it, and do not use GPUs for it.
  - `r2-verify` (CPU only, its own branch) may run concurrently.
  - Never touch other phases' worktrees, branches or remote clones (`/work/dwong/r2_repo`, `/work/dwong/r2_verify_repo`).

### 0.3 Safety constraints (always hold)

- Main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Before every commit, check that `git rev-parse --show-toplevel` is the `lucid-mc` worktree.
- Never modify or delete round-1 or round-2 data, checkpoints, returns or committed results. Never force-push. Never push to `main` or `dev`.
- Do not touch the existing pinned clone `/ceph/dwong/jepa/external/LUCiD`.
- LUCiD's licence is not yet finalised. Use LUCiD for internal research only: no redistribution of LUCiD code or LUCiD-generated data outside the group's storage, and record the licence status.
- **Shared servers:**
  - never saturate CPUs (leave ≥ 4 cores free on any host);
  - on shared GPUs, follow the parent guard rules (deepthought2: ≤ 2 jobs, ≤ 45% VRAM each, launched through the existing guard `scripts/jepa_run.sh` on `dev`);
  - check GPU occupancy before every launch.
- **JAX memory (critical on shared GPUs).** By default JAX pre-allocates 75% of GPU memory, which would break the 45% rule and could crash other users' jobs. Every LUCiD/JAX process on a GPU must run with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` (on deepthought2; 0.80 only on a GPU verified to be idle and unshared). Verify peak memory with `nvidia-smi` during the first job on each host. On CPU-only hosts set `JAX_PLATFORMS=cpu` and cap threads (`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=<n>`, `OMP_NUM_THREADS=<n>`).

### 0.4 Hosts, paths, budget

- **portal1.** Host `portal1.etp.kit.edu`, user `dwong`, defined in the author's `~/.ssh/config`. Reach it with the `p1` wrapper (`p1 '<cmd>'`; default host `portal1.etp.kit.edu`, overridable via `$CC_STATE/hosts_p1.env`). It shares the filesystem with Deepthought (verify which mounts: `/ceph`, `/work`). Use portal1 for cloning, environments, CPU work and, if it has GPUs, LUCiD generation. **portal1 CPU policy (author decision, 21 Nov 2026, 02:47 CET: "Half, with backoff").** portal1's CPUs are much faster than Deepthought's.
- Use up to **half of portal1's cores** (`nproc` / 2) for LUCiD generation (JAX on CPU), conversion, readout and evaluation, at `nice -n 10`.
- Always leave at least 8 cores free.
- **Backoff:** every 10 minutes compute other users' load (1-minute load average minus our own running workers). If that exceeds a quarter of the cores, halve our worker count. Restore it step by step when their load drops.
- Record the core count, CPU model and the backoff events in `HOSTS.md` and the notes.
- Measure throughput on portal1 CPU against a Titan or L40S GPU in L1/L4, and put generation wherever it is faster per wall-clock hour. Both may run in parallel.
- This replaces the earlier 4-core cap. It also overrides the "≤ 24 cores across hosts" budget line for portal1: portal1 workers count separately, within this policy.
- **GPU options, in order of preference for LUCiD generation:**
  1. portal1 GPUs, if present and free;
  2. deepthought2 L40S, under the guard;
  3. Deepthought Titans, if free (12 GB);
  4. HTCondor batch, only if the LUCiD environment can run in the batch image (verify; otherwise use batch for training only, as in round 2).
- **Paths:**

  | What | Path |
  |---|---|
  | New LUCiD clone | `/ceph/dwong/jepa/external/LUCiD_main_<sha8>` |
  | Environment | `/ceph/dwong/envs/lucid_<sha8>` (or another shared path visible from all used hosts; record it) |
  | Data | `/ceph/dwong/jepa/data/lucid_v1/<detector>/` |
  | Returns | `/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/` |
  | Scratch | `/work/dwong/r2_lucid/` if visible, otherwise under the returns path |

- **Budget.**
  - GPU: ledger line `R2_lucid_gpu`, cap **120 GPU-h**, in `docs/round2/RESOURCE_LEDGER.json` on branch `lucid-mc`.
  - CPU: ≤ 24 cores at a time across hosts, `nice -n 10`.
  - Ceph: estimate the volume before mass generation, and stop if it would leave less than 20% free in the target filesystem.
  - If the GPU projection exceeds the cap, drop items from the bottom of §L7's priority list.

---

## Stage L0 — Preflight

1. Confirm the worktree is on `lucid-mc` (the launcher creates it). Copy this runbook into `docs/round2/`, record its SHA256 and the §0.1 quotes in `DECISIONS.md`, then commit and push.
2. Write `$CC_STATE/status/r2-lucid.json` with `{"phase": "r2-lucid", "state": "running", "started": <UTC>}`.
3. **Reach portal1.**
   - Run `p1 hostname`. The wrapper uses `portal1.etp.kit.edu` with the author's ssh config (user `dwong`, BatchMode). If it fails with a host-key or authentication error, write `P1_HOST=dwong@portal1.etp.kit.edu` to `$CC_STATE/hosts_p1.env` and retry once.
   - If portal1 is unreachable, do CPU work on `dt1` and GPU work on `dt2` or the Titans, and record the substitution.
   - Record whether portal1 is a login node (core count, load, GPUs, other users).
4. **Inventory**, recorded in `results/round2/20261121_lucid/HOSTS.md`:
   - per host: OS, Python and conda, mounted filesystems (`/ceph`, `/work`), free space;
   - GPUs (`nvidia-smi`): model, memory, current occupancy;
   - CUDA driver version.
5. Create the remote clone `/ceph/dwong/jepa/r2_lucid_repo` at `origin/lucid-mc`.

## Stage L1 — Clone, install and smoke-test LUCiD

1. Run `git clone https://github.com/CIDeR-ML/LUCiD /ceph/dwong/jepa/external/LUCiD_main_<sha8>` at the current `main`. Record the full SHA, commit date and the licence statement.
2. Create a fresh environment: Python as required by LUCiD, `pip install -e .[all]` or the minimal set. Install JAX with CUDA wheels matching the GPU host's driver; JAX on CPU is acceptable on CPU-only hosts.
3. Run `./scripts/download_data.sh` to fetch the SIREN emitter and PhotonSim tables into the clone's data directory (record the size).
4. Run, recording outputs and timings:
   - `examples/hello_simulate.py`;
   - the quickstart tutorial logic as a script;
   - LUCiD's own test suite, if present.
5. **Output:** `results/round2/20261121_lucid/LUCID_INSTALL.md`, covering SHA, environment and versions, smoke results, and throughput (photons per second, per GPU or CPU).

## Stage L2 — Capability audit and pre-decided defaults

Read LUCiD's README, `docs/reference/dataset-schema.md`, `lucid/production/`, `lucid/sources/`, `lucid/simulation/` (`setup_event_simulator`, `sensor_response`), `lucid/wavelength/`, `lucid/propagation/` and the `lucid-run-job` entry point. Write `results/round2/20261121_lucid/LUCID_CAPABILITIES.md` answering each question with file and line references:

| Question | Default decision |
|---|---|
| Detector configs for WCTE, SK, HK (cylinder; measured PMT file via `Cylinder.from_pmt_file` if shipped) and JUNO (sphere)? | Use LUCiD's own configs, preferring measured-PMT files. Record PMT counts and radii against the toy geometries. If a detector is missing, skip it and report. |
| Physics processes: absorption, scattering, reflection / multi-bounce, wavelength-dependent QE and optics? | Turn on everything the release supports, with LUCiD's default or calibrated water parameters. Record each flag. |
| Is there a stochastic, data-like sampling mode giving sampled per-PMT charge and hit time (or per-photon hits)? | Choose the highest available **sampling tier** (below) and record it in every file and in the report. |
| Isotropic point source / calibration emitter in `lucid/sources/`? | Use it for photon bombs. If absent, implement `IsotropicPointSource` with LUCiD's source API: uniform directions, LUCiD's default emission spectrum (otherwise a flat 300–600 nm spectrum, recorded), emission at t = 0 plus a per-event truth offset uniform in [−50, 50] ns. |
| Muon and electron tracks (SIREN / PhotonSim tables, production chain)? | Generate e⁻ and μ⁻ if the shipped tables support them. Use energy ranges matching the round-2 particle cohorts (read the ranges from `r2_geometry.py` / `r2_generator.py` configs). Vertices uniform in the fiducial volume, directions isotropic. If electrons are unsupported, produce μ⁻ only and record it. |
| Sensor response: TTS, SPE charge, threshold, dark noise, trigger? | Use LUCiD's sensor response. Add the VERITAS NHits trigger, prompt window and 1 kHz dark counts only where LUCiD has none (P1(c)), listing each in `DATA_LINEAGE.md`. |
| Output schema? | Convert into the VERITAS round-2 pack schema (§L3). Keep LUCiD's native files alongside for provenance. |

**Sampling tiers.** Use the highest tier available.

| Tier | Definition | `dataset_kind` |
|---|---|---|
| **A1** | LUCiD's native data-like / Monte Carlo mode (or the `lucid-run-job` production chain) emits sampled per-PMT charges and hit times. | `lucid_mc_native` |
| **A2** | LUCiD traces individual photons with stochastic decisions (absorption, scattering, reflection) sampled as hard choices, and exports per-photon (sensor, arrival time, wavelength). The PMT electronics response is then applied by LUCiD's `sensor_response`, or by `noise_module_lucid` per P1(d). The first-hit time is then the earliest photo-electron, so it sharpens with photon count. | `lucid_mc_photons` |
| **B** | Only expected per-PMT responses exist. Charge is Poisson-sampled on LUCiD's expected photo-electrons; hit time is drawn from LUCiD's own per-PMT time output and timing-spread model. Allowed only if LUCiD exposes a per-PMT time distribution or spread; a single mean time with a VERITAS-invented spread is not allowed. Tier B data and results must be labelled as such everywhere. | `lucid_expected_sampled` |
| **None of the above** | **Stop generation** (P2) and report what is missing. | — |

**Regime defaults.** Choose the photon-bomb yield range in a pilot so that total detected PE per event has 5th, 50th and 95th percentiles of roughly 200, 3,000 and 30,000 in every detector. This is deliberately lower than the toy data. Record the achieved quantiles and the hit-occupancy quantiles.

**Cohort structure.** Mirror the round-2 packs: the same cohort names (train / dev-train, probe-train, probe-test, particle sets), runs per cohort, events per run and seeds namespace. Read counts and structure only from `results/round2/20261112_data/DATA_MANIFEST.json` and the round-2 HDF5 attributes. For WCTE, mirror the round-1 training size (100,000 training events) and the round-2 WCTE probe cohorts.

## Stage L3 — Generator adapter (VERITAS side, conversion only)

0. **Geometry.**
   - Build the VERITAS geometry objects for each detector from the **new** LUCiD detector config, with the existing `r2_geometry` rules:
     - real `mpmt_id` on WCTE;
     - Hilbert-ordered pseudo-modules of the same size k on SK/HK/JUNO;
     - fiducial volume at 0.8 × each dimension;
     - position scaling.
   - Compare with the current round-2 geometry (PMT count, positions, directions, radii, height).
   - If the PMT positions are identical within 1 mm, reuse the frozen round-2 geometry so that the views and encoder bindings are unchanged. Otherwise freeze a new geometry file per detector (`configs/lucid_v1/<det>_geometry_<sha8>.npz`, SHA256 pinned) and record every difference.
   - The encoder and views need no code change in either case, because they consume the geometry object.
1. Implement `src/jepa_noise/lucid_mc_generator.py` and `scripts/lucid_generate.py`.
   - Inputs: detector, cohort, run index, seed and the L2 decisions.
   - Each run: build the LUCiD source and simulator, sample events, convert to the round-2 pack schema (identical dataset names, dtypes and attributes, so that `r2_store` and the trainers load them unchanged), and write provenance attributes:
     - LUCiD SHA, config hashes and physics flags;
     - seeds and the source definition;
     - `dataset_kind` according to the sampling tier (§L2);
     - the list of VERITAS-side steps.
   - Write **every truth field the round-2 packs carry**, under the same names and units, so that all probes and particle tasks work unchanged:
     - vertex;
     - source yield;
     - particle type, direction and energy;
     - emission-time offset;
     - containment flags;
     - trigger time;
     - anything else present.

     List the truth fields of a round-2 pack first and assert that each one is filled or explicitly marked not applicable.
   - **WCTE format.** The `r2-repair` WCTE launcher trains from the round-1 shard layout (100,000-event F1 s = 0.05 training shards) and probes from the round-2 WCTE probe pack. Write LUCiD WCTE data in exactly the formats those launchers consume. If they differ, write both, so that the WCTE training and probe code paths stay unchanged.
   - **Detector state.** Round-1/2 data carry a detector state:
     - per-PMT gain spread F1, s = 0.05 for training;
     - the fixed parity gain;
     - the timing-width response.

     For an apples-to-apples comparison, apply the same detector-state draws (same keyed namespaces and seeds) through the existing VERITAS detector-state and readout code. This is electronics and calibration state, not photon physics, so it is permitted under P1(d) and must be listed in `DATA_LINEAGE.md` and allowed explicitly in G6.
2. Recompute the dataset-derived constants from LUCiD pilot data with the same procedures as round 2, and store them in a new config `src/jepa_noise/configs/lucid_v1.json`:
   - per-detector token time scale;
   - prompt window;
   - NHits threshold.
3. Tests in `src/jepa_noise/tests/test_lucid_mc_generator.py`:
   - schema identity with a round-2 toy pack (names, dtypes, shapes per event);
   - `r2_store` load round-trip;
   - determinism for a fixed seed;
   - provenance attributes present;
   - no import of `lucid_bombs` physics functions in the generation path (assert on the module graph).
4. Run the full native regression suite, expecting no new failures, then commit.

## Stage L4 — Validation gates (per detector, pilot of 500 events)

| Gate | Criterion |
|---|---|
| G1 loading | `r2_store` loads the pilot pack. A two-step training dry run (MAE and JEPA) succeeds. |
| G2 photon-statistics sanity | Total detected PE is linear in source yield (fitted slope within ±10% of a straight line through the origin, R² ≥ 0.98). Hit-time residuals (t − d·n_g/c − t₀) peak within ±2 ns of 0, with a core width ≥ the LUCiD TTS. The fraction of late hits (> 10 ns) is recorded; it must be > 0 if scattering or reflection is enabled. |
| G3 LUCiD self-consistency | For 3 fixed source positions × 200 repeated events, mean sampled per-PMT charge against LUCiD's expected response: Pearson ≥ 0.95 and mean ratio within ±5%. For tier B this check is trivially satisfied; instead report the per-PMT hit-time distribution against LUCiD's time output. |
| G4 regime and trigger | Achieved detected-PE quantiles within a factor 2 of the §L2 targets. Record the trigger efficiency against yield. If less than 90% of events above the 5th-percentile yield trigger, adjust the yield range (not the trigger rule) and record the change. |
| G6 no toy leakage | The generation path's import graph contains no `lucid_bombs` / `r2_generator` physics functions (expected charge, solid angle, analytic arrival time). Geometry, trigger and readout helpers are allowed only as listed in `DATA_LINEAGE.md`. |
| G5 displays and ToF | Event displays for 6 random and 3 low-PE events per detector, in the style of `results/round2/20261117_repair/figures/event_displays/`. An independent simple ToF-only fit (Huber least squares on first-hit times) runs and gives finite errors. Report the value; no threshold. |

Write `results/round2/20261121_lucid/VALIDATION.md` with figures. A failed gate stops generation for that detector only.

## Stage L5 — Archive the analytic toy data (no moves, no deletions)

1. Inventory every toy dataset and record path, size, mtime, file count and SHA256. For files > 1 GB, record the SHA256 of the first and last 64 MB plus the size. Datasets:
   - round 1: the WCTE training and probe shards under `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/` and the related ext data;
   - round 2: `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`;
   - the WCTE regression probe pack under `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`.
2. Write `/ceph/dwong/jepa/data/ARCHIVE_ANALYTIC_TOY_2026-11-21.md` and the repository document `docs/DATA_LINEAGE.md`. The latter records:
   - what the analytic toy is (§0);
   - which results and documents used it (all round-1 and round-2 results);
   - that those results are "analytic-toy-world" results;
   - the new LUCiD lineage.
3. Make the round-2 toy packs read-only (`chmod a-w`, files only). Round-1 data are already protected by the `dt1` guard; do not change them.
4. On branch `lucid-mc`, switch the default data roots of the round-2 trainers and evaluators to `lucid_v1`. Keep the toy data selectable by an explicit flag (`--data-kind analytic_toy`).

## Stage L6 — Mass generation

- **Order:** WCTE → SK → JUNO → HK.
- **Units of work.** One run file per job.
  - Write to a temporary name and rename atomically only after the file is closed and validated (event count, schema).
  - Skip files that already exist with matching provenance and SHA, so resumes are safe.
- **GPU sharing.** LUCiD generation and torch training share the same guarded deepthought2 lanes (≤ 2 jobs in total). Prefer portal1 or Titan GPUs for generation, so that the L40S lanes stay free for WCTE training on night 1.
- **Throughput.** Use the throughput measured in L1 and L4 to choose hosts and parallelism within §0.3 and §0.4. Record GPU-h per detector in the ledger.
- **Per detector**, write `results/round2/20261121_lucid/DATA_MANIFEST_<det>.json` with counts, regime quantiles, provenance and SHA256 per file.

## Stage L6b — Pipelining and batch submission (author addendum, 21 Nov 2026 02:45 CET)

The author is concerned about wall time ("我怕时间不够"). These rules override any reading of L6 → L7 → L8 as strictly sequential.

1. **Pipeline per detector.**
   - Start a detector's L7 training as soon as its L6 manifest is written and verified; do not wait for the other detectors.
   - Generation of the next detector continues in parallel.
   - Start L8 evaluation of each model as soon as its training finishes.
2. **Training placement.**
   - **WCTE:** deepthought2 L40S through the guard (≤ 2 lanes) plus any free Titans. The runs are short.
   - **SK, JUNO, HK: HTCondor batch first**, with the round-2 tooling unchanged:
     - `scripts/r2_batch/stage_code.sh`: code tarball at the current `lucid-mc` commit, listed in `/ceph/srv/dwong/r2_lucid/SHA256SUMS`;
     - stage the verified LUCiD packs read-only to `/ceph/srv/dwong/r2_lucid/<det>/` with SHA256SUMS. This is the xrootd read path, as in `RUNBOOK_2026-11-13_R2_STAGING_DWONG.md` and `RUNBOOK_2026-11-13_R2_HTCONDOR_BATCH.md`;
     - `scripts/r2_batch/submit.py`: `--gpu a100` (TOpAS A100-40GB) or `--gpu l40s` (NEMO2), `accounting_group = cms`, ≤ 12 concurrent jobs;
     - returns via HTCondor file transfer and `scripts/r2_batch/sync.py` into `/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/batch/`.
   - **Smoke first.** Run a 20-step batch smoke per detector and arm before production, to project runtime as in round 2. The HK JEPA W128 fallback rule of L7 is unchanged.
   - **No-wait failover.** If a batch job sits idle for more than 30 minutes while a guarded deepthought2 lane or a Titan is free, run it natively and `condor_rm` the idle duplicate.
3. **Generation placement.** Generation stays on portal1, Titans or deepthought2 GPUs, unless the batch image is verified to run the LUCiD JAX environment. CPU-side conversion and readout may run on `dt1` or portal1 in parallel, within the core limits.
4. **Accounting.** Book native and batch GPU-h under `R2_lucid_gpu`.
5. **Expected wall time once a detector's data exist.**

   | Detector | Expected wall time |
   |---|---|
   | WCTE matrix | ≈ 2–4 h on two lanes |
   | SK runs | ≈ 6–8 h each, in parallel on batch |
   | HK runs | ≈ 12–16 h |

   Target: WCTE results by the morning of 21 Nov. SK, JUNO and HK within about 1.5–2 days, limited by generation throughput and HK training length.

## Stage L7 — Retraining (recipes unchanged)

Use the same code paths, recipes, 10,000 steps, windows and micro-batch rules as the round-2 and `r2-repair` runs. Only the data roots and the dataset-derived constants of §L3 change. Training jobs may use dt2 L40S (guard), Titans and HTCondor batch, as in round 2. Save nine snapshots for the WCTE and SK runs.

**Priority** — drop from the bottom if the cap is threatened:
1. **WCTE:** `mae_mse` s9000/s9001, `mae_repaired` s9000/s9001, `jepa_repaired` s9000/s9001, `sigreg_jepa` s9000; `init` s9000/s9001 (no training).
2. **SK:** `mae_mse` s9000, `mae_repaired` s9000, `jepa_repaired` s9000, `sigreg_jepa` s9000; `init` s9000/s9001.
3. **JUNO:** `mae_repaired` s9000, `jepa_repaired` s9000; `init` s9000/s9001.
4. **HK:** `mae_repaired` s9000, `jepa_repaired` s9000; `init` s9000/s9001. JEPA uses W128 only if a W640 smoke projects more than 14 h; record the choice.
5. **WCTE time-scale sweep** for `mae_repaired`: token time-loss scale ∈ {450, 50, 15, 5, 1.5} ns, seed 9000. Implement it through the existing time-loss weight or scale argument of the repaired MAE (an equivalent weight is (50 / scale)²); do not change any other recipe element. Optional; it seeds the main-paper gate experiment.

## Stage L8 — Evaluation

For every trained model and `init`:
- the standard round-2 suite and the repaired probe suite:
  - all-mean, hit-mean and weighted-mean Ridge;
  - spatial-moment Ridge and MLP;
  - raw-feature spatial baseline;
  - token controls;
- particle tasks where particle data exist;
- f_event and f_layout;
- worst-event displays.

Also compute an **approximate** ToF-only ceiling with the simple independent fit from L4/G5, labelled approximate. Make no CRLB claim for LUCiD data.

Write `results/round2/20261121_lucid/TOY_VS_LUCID.md`: for each detector and arm, toy value next to LUCiD value for the main metrics, with the spatial-moment readout as the primary vertex metric. Always report the gain over `init`.

## Stage L9 — Report and acceptance

1. Write `results/round2/20261121_lucid/LUCID_REGEN_REPORT.md`:
   1. what changed and why;
   2. LUCiD SHA, capabilities and the VERITAS-side steps;
   3. validation gates;
   4. the regime table;
   5. the toy-versus-LUCiD results;
   6. which claims in `docs/round2/CLAIMS_LEDGER_2026-11-21.md` survive, change or fail on LUCiD data, by claim ID, as a proposal (do not edit the ledger);
   7. open issues.
2. Write `docs/round2/ACCEPTANCE_2026-11-21_R2_LUCID.json` with SHA256 and PASS/FAIL per deliverable. Update `RESOURCE_LEDGER.json` and `DECISIONS.md` on `lucid-mc`.
3. Commit and push to `origin/lucid-mc`.
4. Write `$CC_STATE/reports/r2-lucid.md` and set the status to `done`.
5. Print the report in full.

## 9. Night-1 milestone and pacing

- **Night 1 target:**
  - L0–L5 complete;
  - L6 for WCTE and SK;
  - L7 and L8 for WCTE;
  - SK batch smokes and production training submitted as soon as the SK data pass L4 and the L6 manifest (§L6b);
  - an interim `results/round2/20261121_lucid/INTERIM_NIGHT1.md` with the WCTE toy-versus-LUCiD table, committed and pushed.
- **Later:** JUNO and HK generation, SK/JUNO/HK training and evaluation continue on the following days under the same runbook. The driver resumes automatically.
- If any stage blocks for more than 2 h on infrastructure (environment, GPU availability), record it, move to the next independent item (e.g. the archive, or another detector), and come back.
- **Early visibility.** Commit and push the following as soon as each exists, before mass generation:
  - `LUCID_INSTALL.md`;
  - `LUCID_CAPABILITIES.md`, including the chosen sampling tier;
  - `VALIDATION.md`, with gate results and event displays.

  The author and the collaborator can then check the physics in the morning, even if training is not finished.

## 10. Review log

**Round 1 (21 Nov 2026, 02:30 CET).** Changes made in this revision:
1. portal1 host set to `portal1.etp.kit.edu` (author's ssh config, user `dwong`). The `p1` wrapper default was updated, with a login-node load check.
2. JAX GPU pre-allocation disabled and the memory fraction capped. Without this, JAX takes 75% of a shared GPU, breaking the 45% guard.
3. Sampling tiers A1/A2/B added instead of a hard stop. Tier B is allowed only with LUCiD's own timing model and must be labelled everywhere. Tier A2 gives first-photon timing that sharpens with photon count, which removes one known flaw of the toy.
4. PMT electronics via `noise_module_lucid` permitted (P1(d)) only where LUCiD's sensor response does not cover it.
5. Geometry rebuild from the new LUCiD configs with the frozen `r2_geometry` rules. The round-2 geometry is reused if the positions agree within 1 mm.
6. All round-2 truth fields required in the LUCiD packs, so that probes and particle tasks run unchanged.
7. New gate G6 (no toy-physics imports in the generation path). G4 now records trigger efficiency.
8. Atomic file writes. Generation and training share the deepthought2 lane budget.
9. The time-scale sweep is tied to the existing weight argument.
10. Early commits of the install, capabilities and validation documents.
11. WCTE output written in the formats the WCTE launchers actually consume (round-1 shard layout for training, round-2 probe pack for probes).
12. Detector state (F1 gain spread, fixed response) reproduced through the existing VERITAS detector-state code, so that toy and LUCiD datasets differ only in photon physics.

**Round 2 (21 Nov 2026, 02:45 CET, after launch).** Added §L6b at the author's request: per-detector pipelining, HTCondor batch first for SK/JUNO/HK with the round-2 staging and submission tooling, no-wait failover, and wall-time expectations. If you are the running agent and see this section for the first time, commit the updated runbook with the message "Runbook: author addendum L6b (pipelining and batch)" and follow it from now on.

**Round 3 (21 Nov 2026, 02:47 CET, after launch).** portal1 CPU policy changed from a 4-core cap to "half of the cores, nice 10, keep ≥ 8 free, automatic backoff when other users' load rises" (§0.4, author decision). Generation goes wherever measured throughput per wall-clock hour is best: portal1 CPU or the Titan/L40S GPUs, in parallel. The running agent should apply this from the next job it launches on portal1, including scaling up the current environment build if that helps.

**Known open risks.**
- The public LUCiD may lack both a data-like mode and photon-level export. Generation then stops before mass production, with a report.
- Electron SIREN tables may be absent, which would give a μ-only particle set.
- portal1 may have no GPU. Generation then runs on Titans or deepthought2 and may not reach SK on night 1.
- The LUCiD licence is not finalised: internal use only.

## Launch (Mac Terminal)

First check that non-interactive ssh to portal1 works. It must print the host name without asking for a password:

```bash
ssh -o BatchMode=yes portal1.etp.kit.edu hostname
```

Then launch:

```bash
nohup bash ~/Documents/VERITAS/docs/round2/claude-code/run_lucid.sh >/dev/null 2>&1 &
```

Monitor with:
- `tail -f ~/Documents/VERITAS_offrepo/claude-code/logs/r2-lucid.log`
- `cat ~/Documents/VERITAS_offrepo/claude-code/notes/r2-lucid.md`
