# RUNBOOK — scale control: 300k events and 30k steps (2 Nov 2026, reviewed, AUTHORISED)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the design
agent, for the author. Branch `JEPA`, at or after `845c7d6`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-02_SCALE.md` before any run.
- Everything needed is in this file and the repository; it depends on no outside document (the
  author line and the B4a row are quoted below).
- The reusable helpers are the committed library modules `src/jepa_noise/ext_readout.py`,
  `ext_h12.py`, `ext_h1clean.py` (not the scripts); absent at the deployed SHA → stop.
- An independent review checked this file against `845c7d6`; its blocking findings are fixed
  here (per-run cap, generation mode, full `ensure_dataset` call, entries gate).

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 2 Nov 2026:
   > "B 的下一批训练我全部授权"
   The authorised B4a row of the batch plan (quoted verbatim):
   > "| B4a | scale：300k events / 30k steps，2 seeds | 2 | 3×0.6 + eval 0.15 | **≈3.9** |
   > 1.7（生成）+0.2 | 数据 6.9 GB；每个新数据集吃 entries |"
   This authorises **exactly** two `sigreg_jepa_scale300k` runs (seeds 9000, 9001), the 300k
   cohort, and **one** 20-step smoke. No other training.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Gate (implemented in `src/jepa_noise/ext_scale.py`, called from the library training/smoke
entry, not only the CLI).** `assert_authorised()` must:
1. read the committed runbook `docs/jepa/RUNBOOK_2026-11-02_SCALE.md` (its own `RUNBOOK_REL`)
   and require the **full** author line `"B 的下一批训练我全部授权"` (exact match — the substring
   `"我授权"` does not occur in it) **and** the B4a row text;
2. require the same line in `DECISIONS.md`;
3. exit 2 otherwise.
The negative test in §3.3 checks exit 2 when the line is absent.

**Status.** Extension `scale_2026-11-02`, `registered: false`. `<ext>` =
`/ceph/dwong/jepa/returns/ext_scale_20261102/`; results `results/jepa/20261102_scale/`. Never
overwrite an earlier root. Grid descoped. Full suite once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Fix small problems yourself (bug, path, env, guard HOLD, non-finite loss) with one
`DECISIONS.md` line ("author standing approval, 2 Nov 2026") and continue; writes only inside
`/ceph/dwong/jepa/`, no registered changes, within §5, nothing outside `/ceph`. Stop only for
S3/S4/S9/S1, a missing probe set, or a storage/entries preflight failure (§3.4).

## 1. Why this run

The collapse is target-driven (E3 frozen random target recovers; the trainable target collapses
to an event-invariant code; a frozen static target learns no physics), but all of that is at the
**development scale** (100k events, 10k steps, 2 seeds). The remaining H5 question: **is the
collapse a scale artifact?** This run increases the unique data (100k → 300k events) and the
training length (10k → 30k steps); the effective epochs are **nominally** unchanged
(300k × 3.2 = 960k = 30k × 32; 100k × 3.2 = 320k = 10k × 32 — the loader samples 8 runs × 4
events per step with replacement, so unique coverage per epoch is below 100%).

- **recovered**: the collapse does not persist at 3× data and 3× steps → it is (at least partly)
  a scale artifact. Pre-declared follow-up: **100k events at 30k steps** (2 seeds) separates the
  steps from the data.
- **not recovered**: the collapse persists at this scale. Pre-declared follow-up (optional,
  budget permitting): **300k events at 10k steps** to test the data-only axis.

## 2. Variant and reading

**`sigreg_jepa_scale300k`** (`arm_name` stays `"sigreg_jepa"`). The training cohort is a
`registered=False` extension cohort with its own namespace and an **unused event range**:
`{"namespace": "lucid/dev_train_300k", "event_start": 1300000, "n_events": 300000,
"run_length": 1000}` (the existing dev_train is 1,000,000–1,099,999; dev_probe_train 1,200,000;
dev_probe_test 1,210,000; dev_robustness 1,220,000 — no overlap). Dataset key expected:
`dev_train_300k__F1__0.05__s20261017__n300000x1000`; the manifest records namespace/event_start
via its run list. The probe sets are the unchanged dev probe sets (read-only). Training:
**30 000 steps** via the `run_one(steps=30000)` override; the config's `training.steps` stays
**10000** (`load_run_config` rejects other values, `lucid_run.py:78`) and every other
hyper-parameter is unchanged; seeds 9000/9001. Out root `<ext>/train/scale300k/`, per-variant
sidecars, run-dir refusal and `resumed_from_step == 0`.

**Reading `scale_r`** (two-sided, both seeds), implemented in `ext_scale` over `eh.recover`'s
**rows** (do **not** reuse its verdict: `eh.recover` uses `<=` and calls a vertex-fail/charge-pass
pair "partial", `ext_h12.py:1091-1130`). Floors from `ext_h12.INIT_FULL_VERTEX_M = 0.160072`
and `INIT_HALF_VERTEX_M = 0.174821` (`ext_h12.py:47-48`; the committed `decision.json` floors
are 0.16007169…/0.17482124…):
- `recovered`: both seeds full-input vertex **strictly < 0.160072 m**, full token ln q R² ≥ 0.5,
  half-pass vertex < 0.174821 m, half-pass token R² ≥ 0.5.
- `not recovered`: both seeds full-input vertex ≥ 0.160072 m **or** full token ln q R² < 0.5.
- `unclear`: anything else (partial, seed disagreement, half-pass contradiction).
Report `f_layout`, `f_event`, effective rank, wall, steps, `resumed_from_step`, and the cohort
key/manifest sha256.

## 3. Implementation

### 3.1 Modules, aliases, cohort call

`scripts/jepa_ext_scale.py` (modes `data`, `dry-run`, `smoke`, `train`, `evaluate`, `report`),
`src/jepa_noise/ext_scale.py` (gate, cohort spec, wrappers, `scale_r`), `src/jepa_noise/tests/
test_ext_scale.py`. Aliases: `lr = jepa_noise.lucid_run`, `er = ext_readout`, `eh = ext_h12`,
`ec = ext_h1clean`; `lr.ROOT`; `lr.sha256_file`, `lr.git_head`, `lr.git_status` (imported into
`lucid_run` from `bridge_run`). Reuse: `er.evaluate_item`, `er.layout_state`, `er.layout_add`,
`er.layout_fractions`, `eh.evaluate_run`, `eh.reproduction`, `eh.preflight_data`,
`eh.train_log_check`, `ec.assert_fresh_resume`; do not reuse `eh.train_variant`/
`prepare_run_dir`/`summarize_run` (variant-locked).

Exact cohort call (the `data` mode; `ensure_dataset` is idempotent and verifies an existing
manifest):

```
train_ds = lr.ensure_dataset(
    data_root=Path(data_root), geometry=geometry, generator_config=gen,
    cohort_spec={"namespace": "lucid/dev_train_300k", "event_start": 1300000,
                 "n_events": 300000, "run_length": 1000},
    cohort_name="dev_train_300k", law="F1", setting=0.05, seed=20261017,
    registered=False, code_sha=code_sha, verify_hashes=False)
```

Probes: `_, probes = eh.datasets_for(cell, config, gen, geometry, Path(data_root), code_sha=None)`
(this builds the 100k dev_train too — it exists; call after the data mode). `eh.preflight_data`
requires the 100k dev_train key (`ext_h12.py:93-114`); call it and stop on failure. Never let
`eh.load_probes` generate an absent probe set — check the probe directories exist first and stop
if not.

Exact `lr.run_one` call (as in `ext_h12.train_variant:230-239`):

```
lr.run_one(arm_name="sigreg_jepa", cell=lr.dev_cell(config), seed=SEED, beta=None,
           config=config, arms_config=arms_config, geometry=geometry,
           loader_dataset=train_ds, probe_datasets=probes, dark_bank=None,
           out_root=<ext>/train/scale300k, data_root=Path(data_root),
           config_sha=lr.sha256_file(config_path), code_sha=code_sha,
           status_text=lr.git_status(lr.ROOT), device=device, steps=30000,
           micro_batch=None, resume=None, stop_after=None, checkpoint_minutes=None,
           evaluate=False, allow_pending=False, step_callback=None, probe_max_events=None,
           save_embeddings=False, physics_method="ridge", tf32=False, log=print)
```

`run_one` auto-resumes when `checkpoint.pt` exists (`lucid_run.py:811-812`): before each run the
extension must use a `prepare_run_dir`-style refusal on `out_root/runs/<run_id>` and assert
`resumed_from_step == 0` with `ec.assert_fresh_resume` (`ext_h1clean.py:182-194`) after the
run; archive a failed run dir, never relaunch into it.

### 3.2 CLI (all modes; defaults where sensible)

```
python scripts/jepa_ext_scale.py data --out <ext>/cohort.json --data-root <DATA> \
  --config <CFG> --generator-config <GEN> --geometry <GEOM>
python scripts/jepa_ext_scale.py dry-run --out <ext>/plan.json --data-root <DATA> \
  --config <CFG> --generator-config <GEN> --geometry <GEOM> --device cpu
python scripts/jepa_ext_scale.py smoke --seed 9000 --out <ext>/smoke --steps 20 \
  --probe-max-events 200 --data-root <DATA> --config <CFG> --generator-config <GEN> \
  --geometry <GEOM> --device cuda
python scripts/jepa_ext_scale.py train --seed 9000 --out <ext>/train/scale300k \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM> --device cuda
python scripts/jepa_ext_scale.py evaluate --run <ext>/train/scale300k/runs/<run_id> \
  --out <ext>/eval --config <CFG> --generator-config <GEN> --geometry <GEOM> \
  --data-root <DATA> --readout-ext /ceph/dwong/jepa/returns/ext_readout_20261101 \
  --readout-items /ceph/dwong/jepa/returns/ext_readout_20261101/items.json --device cuda
python scripts/jepa_ext_scale.py report --out results/jepa/20261102_scale \
  --plan <ext>/plan.json --eval <ext>/eval --reproduction <ext>/eval/reproduction.json \
  --config <CFG> --geometry <GEOM>
```

Order: `data` (CPU) → `dry-run` → `smoke` → `train` ×2 → `evaluate` ×2 → `report`. `<run_id>` =
`dev_s005_within__sigreg_jepa__s9000|9001`. `<CFG>` =
`src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (sha256 `e6730ac1…`), `<GEN>` =
`src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` (sha256 `90ac1601…`), `<GEOM>` =
`/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`, `<DATA>` =
`/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`. `dry-run` exits 2 on failure; `smoke`/`train`
refuse if the 300k cohort manifest is absent.

### 3.3 Tests

(1) the full author line (not the `"我授权"` substring) authorises, and the gate exits 2 when
the line is absent; (2) the cohort spec/key/manifest (300 000 events, 300 runs, namespace +
event_start 1 300 000, `registered: false`); (3) `load_run_config` rejects a config with
`training.steps` outside {10000, 5000} while `run_one(steps=30000)` works; (4) `eh.evaluate_run`
on a final checkpoint equals `lr.evaluate_arm_run` on the same checkpoint; (5) run-dir refusal and
`resumed_from_step == 0`; (6) `scale_r` returns the three-valued verdict on synthetic rows; (7)
preflight/entries failure paths (missing probe set; projected free entries < 30 000); (8)
smoke-level training finite.

### 3.4 Data, storage, entries

The 300k cohort is **new data** in an unused event range: generation ≈1.68 h at 49.486 events/s
(book under `lucid_generation_cpu`; rate includes the twin), storage ≈4.8 GB (measured dev_train
604 entries / 1.592 GB per 100k, `PREFLIGHT_A8_deepthought2.md:254`). The `dry-run` preflight
must, **before any write**: read the live free-entry count from the guard/`find | wc -l` (not
`du` bytes), project the cohort (≈1 812 entries) + a fresh clone (≈3 059) + the suite (≈1 825) +
`<ext>` (≈1 000) ≈ 7 700 entries against the current free count (**40 959** at review →
≈33 300 after, above the 30 000 floor), and stop if the projection is < 30 000 or the guard is
HOLD. Record the live count and projection in `plan.json`. Probe sets are committed and
read-only (manifest `093ea894…`/`c11edc71…`); if absent, stop (never regenerate).

## 4. Acceptance and reproduction (S9)

1. Full suite once on the deployed SHA:
   `nice -n 10 env OMP_NUM_THREADS=4 python -m pytest -q --basetemp=<ext>/tmp/pytest
   -p no:cacheprovider` (as in `RUNBOOK_2026-11-01_H12_CAUSAL.md:224-226`).
2. Reproduction uses the committed hard-coded `eh.REPRODUCTION` (`ext_h12.py:51-52`; not
   `decision.json`) and the committed `results/jepa/20261101_h12/reproduction.json`: `fix9000`
   full-input vertex `0.219258` ± 1e-3 and `init9000` `0.160072` ± 1e-3; exit 2 on mismatch.
3. The cohort manifest records 300 000 events / 300 runs / namespace / event_start; the entries
   projection passes; one smoke finite; both runs exit 0, 30 000 steps, no NaN,
   `resumed_from_step == 0`.
4. **Per-run cap 3.5 h** (overrides the batch plan's 90-min default; expected 1.15–2.23 h, the
   3× scaling of the committed 10k walls 0.384–0.742 h). A run killed at the cap would
   auto-resume; archive the run dir before any relaunch and re-assert `resumed_from_step == 0`.
5. No file outside `/ceph/dwong/jepa/` on the host (the host clone is
   `/ceph/dwong/jepa/code/<sha>/`; the Mac clone is for editing/pushing); `git status` shows only
   this extension's files.

## 5. Budget

| Resource | Cap | Notes |
|---|---|---|
| L40S | ≤ 8 GPU-h; ≤ 2 **concurrent** jobs; **per-run cap 3.5 h** | 2 runs at 30k steps (expected 1.15–2.23 h each) + 1 smoke + evaluations ≈ 4–5.5 GPU-h |
| Titan | 0 | not needed |
| CPU | ≤ 4 wall-hours (Deepthought, `nice -n 10`, `OMP_NUM_THREADS=4`) | 300k generation ≈1.68 h, tests, evaluation, report |
| Storage | cohort ≈4.8 GB + `<ext>` ≤ 2 GB; projected free entries stay above 30 000 | see §3.4 |

Ledger at review **105.644356 / 170**; this runbook ≤ 8 GPU-h → projection ≈ 114 < 160 (S1);
generation `lucid_generation_cpu` 14.995639 + 1.68 ≈ 16.68 / 48. Book every job in
`RESOURCE_LEDGER.json` (`v4_caps.E2_lucid_l40s_deepthought2`).
`TMPDIR=/ceph/dwong/jepa/tmp/`; pytest `--basetemp` inside `<ext>/tmp/`, `-p no:cacheprovider`.

## 6. Outputs, schemas, commit, report

1. `<ext>/cohort.json`: `{"registered": false, "extension", "key", "manifest_sha256",
   "namespace", "event_start", "n_events", "run_length", "n_runs", "entries", "gb"}`.
   `<ext>/plan.json`: `{"registered": false, "extension", "cohort": {…}, "probes": {key,
   manifest_sha256}, "readout": {"ext", "items", "checkpoints_ok"}, "entries": {free_before,
   projected_after, floor}, "floors": {"full": 0.160072, "half": 0.174821},
   "criteria": "…"}`.
2. `results/jepa/20261102_scale/`: `README.md` (`registered: false`), `runs.json` (per seed:
   `seed, wall_seconds, steps, resumed_from_step, metrics_finite, cohort_manifest_sha256,
   full_input: {vertex_m, token_charge_r2}, half_pass: {…}, f_layout, f_event, eff_rank`),
   `readings.json` with `{"registered": false, "extension", "criteria": {"floors": {…},
   "rule": "…"}, "verdict": "recovered"|"not recovered"|"unclear", "per_seed": {"9000": {…},
   "9001": {…}}}`, `reproduction.json`, `plan.json`; each ≤ 1 MB.
3. `<ext>/`: checkpoints, sidecars, caches deleted after `report`, hashes in
   `<ext>/large_arrays.json`, `manifest.jsonl`; the cohort stays in the data root (record its
   manifest sha256 and size in `cohort.json`).
4. `DECISIONS.md` one line per engineering choice; a short `STATE.md` entry; book resources; do
   not touch `PREDICTIONS.md`/`PREDICTION_STATUS.json`.
5. Commit order: runbook + code + tests first, then the cohort manifest, then runs, then results;
   report the final SHA.
6. `report` requires **both** seeds; missing either, or a non-finite metric, exits 2. Report to
   the author in Chinese (≤ 1 page): the scale_r direction with numbers, the cohort size/entries,
   GPU/CPU hours, final SHA.
7. Do not resume the grid; the pre-declared separation follow-ups are in §1.
