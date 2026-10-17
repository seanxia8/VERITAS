# WP5 LUCiD runner — `jepa_noise.lucid_run`

**Status: development only.** The readout is `readout_synthetic_unrepaired` (§3.5.2);
no number produced by this code is a reported result. The optimizer/lr/schedule are
**frozen** by the 2026-10-18 Lead decision (`docs/jepa/DECISIONS.md`, mirroring
`configs/bridge_v1.json:281-322`), so `grid` and `dev` steps 2–3 train with those
numerics. A config that still says `pending_review` is refused.

## Frozen training numerics (Lead decision, 2026-10-18)

| Item | Value |
|---|---|
| AdamW | lr 1e-3, betas (0.9, 0.999), weight decay 0, eps 1e-8 |
| fused | `True` on CUDA, documented CPU fallback `False` (recorded per run) |
| Muon | lr 1e-3, momentum 0.95, nesterov, ns_steps 5 |
| schedule | `constant` — an implemented no-op: no warmup, no decay, no cosine |
| grad clip | null |
| batch / steps / precision | 32 / 10 000 / fp32, TF32 off |

`build_optimizers` applies these values (not merely records them), returns an explicit
constant-schedule object that is stepped every iteration and checks that no learning rate
changed, and **refuses** any other schedule value or any frozen-value edit. Each run
manifest records the family, lr, betas, eps, weight decay, fused flag/device, Muon
numerics, schedule and gradient clip actually used.

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
500 steps at batch 32 with the frozen optimizer numerics (smoke is non-registered because
of its synthetic geometry and reduced steps), and writes
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

The optimizer freeze removes the old pending-review gate. `dev --step 3` now consumes
the step-2 decision automatically: it reads `dev/step2_probe_decision.json` (the nested
`physics_probe.decision`, `ridge` or `mlp`) and the step-2 plain `sigreg_jepa` run's
`probes.json` as the β reference. It refuses if step 2 has not run; `--plain-reference`
is only an explicit override. There is no silent ridge fallback — the method and its
source are recorded in the step-3 decision.

Step 1 (detectability — frozen decision; no training):

```bash
$PY -m jepa_noise.lucid_run dev --step 1 --out /ceph/dwong/jepa/returns/wp5_dev \
  --data-root /ceph/dwong/jepa/data --geometry "$GEOM"
```

writes `dev/step1_detectability_decision.json` (Y_lo/Y_hi, F1 eligible {0.05, 0.10},
F2 a = 0.50, F3 failed, G6 false, R12 not_evaluable) with `registered: false` and the
source sha256. It defaults to the committed
`results/jepa/20261017_wp4b_detectability/wp4b_detectability_wcte_v2.json`.

Steps 2 and 3 (training with the frozen numerics):

```bash
$PY -m jepa_noise.lucid_run dev --step 2 --out ... --data-root ... --geometry "$GEOM"
# 2 dev trainings (sigreg_jepa, mae_mse); writes dev/step2_probe_decision.json:
# physics probe = ridge unless BOTH linear vertex resolutions > 3 x the reference fit.
# The decision is nested under "physics_probe".
$PY -m jepa_noise.lucid_run dev --step 3 --out ... --data-root ... --geometry "$GEOM"
# 15 trainings (3 arms x 5 beta); reads the step-2 decision and plain probes
# automatically; writes dev/step3_beta_decision.json: the largest beta with vertex
# resolution <= 1.1 x the plain sigreg_jepa, plus the physics method and its source.
# --plain-reference PATH is an explicit override, never a requirement.
$PY -m jepa_noise.lucid_run dev --step 4 ...   # refused: not in §3.5.6
```

`dev --dry-run --step 2|3` prints the run list and writes nothing.

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

## Fallbacks and resume

- **Auto-resume is automatic** when `runs/<run_id>/checkpoint.pt` exists (docs above);
  `--resume PATH` remains an explicit override and `--stop-after` is the interruption test.
  Resumed checkpoints validate arm/cell/seed/β/config sha and are bit-identical.
- **§3.5.6 budget fallbacks are accepted and recorded, never silent.** `load_run_config`
  accepts the 5 000-step fallback and the "3 seeds for every main arm" fallback; any other
  seed mix is rejected. A fallback config is recorded in each manifest
  (`fallback.active/steps/seed_fallback`) and can never set `registered: true`.
- The **random-init floor cache identity** includes the split, mask seed, `probe_max_events`,
  probe forms and the probe-set hashes, so a changed mask/cap cannot silently reuse a floor.

## What remains

- Real-geometry runs: smoke, scaling, the development cohort and the 201-run grid have **not**
  been executed here (CPU-only local Mac; no ssh/GPU).
- `reference.json` v_group for the physics reference fit (PMT branch pending).
- Confirmation of the questions above, especially the dark-window convention, probe event-ID
  reuse, R6/R12 seed-floor readings, and the missing `s_p = 0` / write-once / dev-step-4 cells.
- **R8 and the law-T extrapolation criterion are `not_evaluable`, not scored.** The
  robustness (s = 0.20 / s = 0) and [1.9, 2.1] extrapolation probe sets are generated and
  cached, but §3.5.9 names R8 only as a "robustness ratio" and fixes no law-T
  extrapolation criterion, so no registered quantity is defined. The aggregate records
  this explicitly (`registered_reports.R8_robustness`, `.law_T_extrapolation`) instead of
  claiming a score.
- The aggregate now also records the §3.5.9 reporting quantities that have no verdict:
  R6 full-input module-gain R² − floor, R7b attenuation (F2) R² of anchored vs plain in
  the R11 cell, R12 full-input per-PMT-gain R² per split, and the occupancy
  natural-vs-balanced comparison.
- **`oracle_nominal_twin` probes read the arm's own nominal-twin input**
  (`runs/<run>/twin_hits.npz`), not the real hits (§3.5.6), with the same view mask.
- **`ijepa_ema` target-encoder input (open, needs another file).** `lucid_arms.py` feeds the
  EMA target encoder the full input and gathers the masked target tokens; v3 WP4 line 233
  asks for the masked target-token input. The fix must change `_forward_ijepa_ema` inside
  `lucid_arms.py`, which is outside this runner fix's file scope; it is reported to the Lead
  and is not silently patched here. Recorded in `unresolved_questions`.
- `lucid_run.py` and the test file are over the 1500-line pair budget requested for the
  original task; trimming is still possible by dropping the robustness/CKA extras.
