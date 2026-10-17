# WP5 LUCiD runner — `jepa_noise.lucid_run`

**Status: development only.** The readout is `readout_synthetic_unrepaired` (§3.5.2);
no number produced by this code is a reported result. The registered optimizer/lr/schedule
are `pending_review`, so `grid` and `dev` steps 2–3 refuse to train until the WP5 config
review freezes them (gate to the Lead). `smoke` uses a labelled smoke-only optimizer.

Files: `src/jepa_noise/lucid_run.py` (runner), `src/jepa_noise/configs/lucid_run_v0.json`
(plan, frozen references and `unresolved_questions`), `src/jepa_noise/tests/test_lucid_run.py`.
Read-only imports: `lucid_bombs` (generator/cache/manifests), `lucid_views` (masks),
`point_encoder` (model/tokenise), `lucid_arms` (seven arms, anchor, four probe forms,
random-init floor), `bridge_run` (provenance/atomic/status helpers).

## What the runner fixes from §3.5.6

| Set | Cells | Arms × seeds | Trainings |
|---|---|---|---|
| main | `s ∈ {0.02, 0.05, 0.10} × split ∈ {across, within}`, F3 `s_p = 0.10` | `sigreg_jepa`, `sigreg_jepa_anchored`, `mae_mse` ×5; the other four ×3 | 162 |
| R11 | `within`, law F2 (F1 s = 0.05 + F2 a = 0.50 + F3 co-vary) | `sigreg_jepa`, `sigreg_jepa_anchored`, `mae_mse` ×5 | 15 |
| R11b | `within`, law R11b (co-varying scalar, s = 0.05, a = 0.50) + F3; probes on the independent R11 law | same three arms ×3 | 9 |
| dark anchor | `within`, F1 s = 0.10 | `sigreg_jepa_anchored_dark` ×3 | 3 |
| law T | `within`, law T (r/R = 0..0.99) + F3 | same three arms ×3 | 9 |
| occupancy | `within`, F1 s = 0.05, balanced context views | `sigreg_jepa` ×3 | 3 |
| **total** | | | **201** |
| development | `s = 0.05, within`, independent IDs/seeds | step 2: `sigreg_jepa`, `mae_mse`; step 3: three β arms × five β | **17** |

- s = 0.02 cells are trained but excluded from R6/R7 (frozen ceiling); F3 failed §3.5.3a,
  so R12 is `not_evaluable` (recorded, no gate); F2 is used at a = 0.50, F2′ is not used.
- Data: one 100k-event training set per drift setting (split is a view, so it serves both
  splits), 1000-event runs; probe sets are one probe-train and one probe-test per drift
  setting, 10 000 events in 50-event runs (200 states each), serving both splits; robustness at s = 0.20 and the s = 0 baseline;
  law T in-range train/test and the [1.9, 2.1] extrapolation set.
- `run_mean` is the primary probe form for every registered criterion; the four forms
  (`per_event`, `run_mean`, `propagated`, `raw_input`) are all produced, with the
  random-init floor run through the same probe code (one cache per arm × cell).
- Verdicts R6, R7, R7b, R11, R11b, R12 are emitted only when the full cell/seed set is
  present; otherwise `not_evaluable` with the explicit missing run ids.

## Cells the runbook does not fix — questions, not decisions

1. **No `s_p = 0` secondary cells.** §3.5.3 fixes F3 `s_p = 0.05`, adjusted only through
   §3.5.3a; the frozen check raised it to **0.10 for every LUCiD cell**, including R11,
   R11b and T. The runner creates no `s_p = 0` cell.
2. **R12 is not a training cell.** It is a control on the main cells, evaluable only if
   F3 passes §3.5.3a and R6 passes. F3 failed, so R12 is `not_evaluable`; no R12 cell is run.
3. **No "write-once control" exists in §3.5.6.** The five secondary items are R11, R11b,
   dark-window-only anchor, aging trend, occupancy ablation. A write-once cell/control is
   not defined and is not implemented.
4. **Development step 4 (robustness seed selection) is not in §3.5.6.** The runbook fixes
   three dev steps and 17 dev trainings (2 + 15). `dev --step 4` refuses and refers to the Lead.
5. **The per-event vs run-mean probe-form decision with a 1.2× CRLB ratio / 1.1× expected SE
   is not in §3.5.6.** The runbook fixes `run_mean` as primary and dev step 2 as the
   *physics-probe type* (ridge vs MLP at 3× the reference fit). Only the latter is implemented.
6. **R6's within-excess seed floor** is not spelled out. The runner uses the within-cell
   §3.5.8 floor (`max(SD_within, median SD over arms in that cell)`) and records it.
   **R12's "both splits"** does not name s; the runner reads both splits at every
   R6-eligible s (0.05, 0.10). Both readings need confirmation.
7. **Probe event-ID reuse.** The single registered 10k probe ranges are reused across drift
   settings (same event IDs and keyed draws, distinct dataset directories and state hashes).
   If disjoint ranges per setting are intended, the runbook does not name them.
8. **Dark-only windows.** The generator has no no-source writer. The runner generates
   noise-only windows from the generator's own readout at `yield_photons = 0` (nominal
   state, 512 records) — an engineering choice needing confirmation.
9. **Physics reference.** `reference.json` / v_group is not available locally (§3.5.2
   pending). Until `evaluation.physics_reference_path` is named, the reference fit
   resolution is `not_evaluable` and dev step 2 records `not_evaluable`.
10. **R11/R11b probe form and leakage basis.** The criterion does not name a form; the
    runner uses `run_mean` (the §3.5.9 preamble) and the seed-mean leakage, recording every seed.
11. **Occupancy ablation arm.** "LeJEPA" is read as `sigreg_jepa`; the natural baseline is
    the main s = 0.05 `within` cell.
12. **`mae_chi2` cell spread on law T / R11b.** A trend has no single s; the runner passes
    s = 0.10 (T_SPREAD) for law T and s = 0.05 for R11b. Needs confirmation.

## Data cache

One directory per `(cohort, law, setting, dataset seed)` under `$DATA/lucid/<key>/`, with the
generator's per-run `hits.npz`, `twin_hits.npz`, `events.npz`, `manifest.json`,
`SHA256SUMS`, plus a dataset-level `manifest.json` (geometry/config hashes, state hashes,
run rows). Existing datasets are verified by manifest/state hashes (and `SHA256SUMS` with
`--verify-hashes`); they are **never regenerated silently**. Interrupted generation is
resumed run by run. Dark/occupancy cells reuse the main s = 0.10 / 0.05 hits when the
key matches, so the registered 1.07 M-event projection is an upper bound.

At the frozen yield range the generator measures 49.486 events/s, 1 010.5 hits/event,
178.5 MB RSS; the full 1.07 M projection is 6.0 h on one process / 0.375 h on 16 cores
(0.78 % of the 48 h CPU cap). A 100k-event training set is ≈ 0.56 h single-process,
≈ 2.1 min on 16 cores; a 10k probe set ≈ 3.4 min single-process.

## Commands

Common environment: code from the clean worktree for the registered SHA; `PYTHONPATH=src`;
single-threaded BLAS (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`); one
job per GPU under the §6.2 guard; `CUDA_VISIBLE_DEVICES` from the guard; never more than
3 Titan training jobs (the 4th GPU is for ≤ 30 min jobs).

### (a) Titan 500-step smoke (real geometry, CPU generation + GPU training)

```bash
cd /ceph/dwong/jepa/code/<sha>
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/ceph/dwong/jepa/env/<env>/bin/python
GEOM=/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json
$PY -m jepa_noise.lucid_run smoke \
  --out /ceph/dwong/jepa/jobs/wp5_smoke_titan --data-root /ceph/dwong/jepa/data \
  --geometry "$GEOM" --device cuda:0 --arm sigreg_jepa --steps 500 \
  --runs 2 --run-events 1000
```

This generates 2 × 1000 events on the CPU host (cached), trains one `sigreg_jepa` run for
500 steps at batch 32 with the labelled smoke optimizer, and writes
`runs/smoke_s005_within__sigreg_jepa__s0000/{run_status.json,checkpoint.pt,final.pt,SHA256SUMS}`
plus `manifest.jsonl` (arm, cell, seed, config SHA, code SHA, clean-tree flag, device, wall,
steps, geometry/data hashes, readout label). Add `--evaluate` only for a probe smoke (it is
not the registered evaluation). Expected wall: generation ≈ 40 s, 500 steps ≈
0.5–2 min on a Titan depending on the environment (recorded, not promised).

### (b) Scaling benchmark (no training)

2k and 10k on a Titan; 40k on an L40S (≤ 1 GPU-hour each; the guard applies):

```bash
$PY -m jepa_noise.lucid_run scaling --out /ceph/dwong/jepa/jobs/wp5_scaling_titan \
  --data-root /ceph/dwong/jepa/data --device cuda:0 --sizes 2000,10000 \
  --batch 8 --hit-fraction 0.2
$PY -m jepa_noise.lucid_run scaling --out /ceph/dwong/jepa/jobs/wp5_scaling_l40s \
  --data-root /ceph/dwong/jepa/data --device cuda:0 --sizes 40000 \
  --batch 8 --hit-fraction 0.2
```

Writes `scaling.json` with wall, events/s, peak memory, parameter count, and
`status: does_not_fit` (never a crash) when the conservative feasibility estimate exceeds
the cap or the forward/backward raises OOM. `--memory-cap-mb` overrides the §6.2 cap
(9 600 Titan / 21 000 L40S by default). This was **not run in this task**.

### (c) Development cohort

Step 1 (detectability — frozen decision; no training):

```bash
$PY -m jepa_noise.lucid_run dev --step 1 --out /ceph/dwong/jepa/returns/wp5_dev \
  --data-root /ceph/dwong/jepa/data --geometry "$GEOM"
```

writes `dev/step1_detectability_decision.json` (Y_lo/Y_hi, F1 eligible {0.05, 0.10},
F2 a = 0.50, F3 failed, G6 false, R12 not_evaluable) with `registered: false` and the
source sha256. It defaults to the committed
`results/jepa/20261017_wp4b_detectability/wp4b_detectability_wcte_v2.json`.

Steps 2 and 3 (blocked by the WP5 config review):

```bash
$PY -m jepa_noise.lucid_run dev --step 2 --out ... --data-root ... --geometry "$GEOM"
# 2 dev trainings (sigreg_jepa, mae_mse); writes dev/step2_probe_decision.json:
# physics probe = ridge unless BOTH linear vertex resolutions > 3 x the reference fit.
$PY -m jepa_noise.lucid_run dev --step 3 --out ... --data-root ... --geometry "$GEOM" \
  --plain-reference /ceph/dwong/jepa/returns/wp5_dev/runs/dev_s005_within__sigreg_jepa__s9000/probes.json
# 15 trainings (3 arms x 5 beta); writes dev/step3_beta_decision.json:
# the largest beta with vertex resolution <= 1.1 x the plain sigreg_jepa.
$PY -m jepa_noise.lucid_run dev --step 4 ...   # refused: not in §3.5.6
```

`dev --dry-run --step 2|3` prints the run list and writes nothing. Step 1 is runnable now;
steps 2–3 return exit 2 with the pending-review gate until the config is frozen.

### (d) Main grid, split into guard-sized chunks

Plan first (writes nothing):

```bash
$PY -m jepa_noise.lucid_run grid --out /ceph/dwong/jepa/returns/wp5_main \
  --data-root /ceph/dwong/jepa/data --geometry "$GEOM" --dry-run
$PY -m jepa_noise.lucid_run grid ... --dry-run --cells main_s005_within --arms sigreg_jepa --seeds 0,1
```

Then one chunk per tmux job (guard-sized: one cell × one arm × ≤5 seeds), e.g.:

```bash
# A. main grid, one cell/arm chunk per job (run chunk list checked against the guard)
$PY -m jepa_noise.lucid_run grid --out ... --data-root ... --geometry "$GEOM" \
  --cells main_s002_across --arms mae_chi2 --seeds 0,1,2
$PY -m jepa_noise.lucid_run grid ... --cells main_s005_within --arms sigreg_jepa --seeds 0,1,2,3,4
$PY -m jepa_noise.lucid_run grid ... --cells main_s010_across --arms oracle_nominal_twin --seeds 0,1,2
# B. secondary, in the §3.5.6 order (R11 -> R11b -> dark -> T -> occupancy)
$PY -m jepa_noise.lucid_run grid ... --cells r11_f1f2f3_within --arms sigreg_jepa --seeds 0,1,2,3,4
$PY -m jepa_noise.lucid_run grid ... --cells r11b_law_within --arms mae_mse --seeds 0,1,2
$PY -m jepa_noise.lucid_run grid ... --cells dark_anchor_s010_within \
  --beta-decision /path/to/dev/step3_beta_decision.json --seeds 0,1,2
$PY -m jepa_noise.lucid_run grid ... --cells law_t_aging_within --arms sigreg_jepa --seeds 0,1,2
$PY -m jepa_noise.lucid_run grid ... --cells occupancy_s005_within_balanced --seeds 0,1,2
```

Anchored arms require `--beta-decision` (from `dev --step 3`); without it the runner refuses.
Each chunk generates the datasets it needs (cached) before training; the loader draws 4
events from each of 8 runs per batch (no batch from a single run; events uniform), with the
nominal-twin batch only for `oracle_nominal_twin`. Evaluation runs the four probe forms and
writes `probes.json` + float16 per-run embeddings; the random-init floor is cached once per
arm × cell under `floors/`. Resume is automatic when a checkpoint exists in the run dir, or
explicit with `--resume PATH` for a single-run chunk; `--stop-after` is the interruption test.

Aggregate (read-only over the run roots; never edits `PREDICTION_STATUS.json`):

```bash
$PY -m jepa_noise.lucid_run aggregate --out /ceph/dwong/jepa/returns/wp5_aggregate \
  --runs-root /ceph/dwong/jepa/returns/wp5_main /ceph/dwong/jepa/returns/wp5_secondary \
  --detectability results/jepa/20261017_wp4b_detectability/wp4b_detectability_wcte_v2.json
```

writes `aggregate.json`, `predictions_fragment.json` and `PREDICTIONS_UPDATE.md`
(R6, R7, R7b, R11, R11b, R12 with values/missing lists and seed-mean CKA).

## Memory, runtime and accounting

- batch 32 is ≈ 2.8 GiB peak per WP4c on CPU, ≈ 4× inside the 9.6 GB Titan cap, so no
  gradient accumulation is needed; if a host profile disagrees, `--micro-batch 8` keeps the
  effective batch at 32 and records `gradient_accumulation: 4`.
- fp32, TF32 off, deterministic, single BLAS thread; `torch.set_per_process_memory_fraction`
  is applied by `scripts/jepa_run.sh` at the §6.2 cap. One manifest row per run.
- 10 000 steps is the registered length; `smoke --steps` is the only override.
- Ledger: L40S criterion work ≤ 80 GPU-hours, Titan (smoke/scaling/generation/evaluation)
  ≤ 300 GPU-hours, CPU generation ≤ 48 wall-hours on ≤ 16 cores (§6.1, §3.5.6).

## What remains

- **WP5 config review** (optimizer family, lr, schedule, warmup, clip, Muon numerics) — the
  only blocker for `grid` and `dev` steps 2–3. The config keeps every field `pending_review`.
- Real-geometry runs: smoke, scaling, the development cohort and the 201-run grid have **not**
  been executed here (CPU-only local Mac; no ssh/GPU).
- `reference.json` v_group for the physics reference fit (PMT branch pending).
- Confirmation of the questions above, especially the dark-window convention, probe event-ID
  reuse, R6/R12 seed-floor readings, and the missing `s_p = 0` / write-once / dev-step-4 cells.
- Robustness sets (s = 0.20, s = 0, aging extrapolation) are generated and scored by the
  runner, but R8 is outside the six verdicts requested here and is not emitted.
- `oracle_nominal_twin` probes currently read the real hits, not the nominal twin (§3.5.6
  says probes use the arm's own input); the twin probe-set wiring is pending.
- `lucid_run.py` is 1518 lines and the test file 242; the pair is over the 1500-line budget
  requested for the task. Trimming is possible by dropping the robustness/CKA extras.
