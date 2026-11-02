# RUNBOOK — architecture/masking controls: window 128, window 32, 4-block predictor (2 Nov 2026, reviewed, AUTHORISED)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the design
agent, for the author. Branch `JEPA`, at or after `57295c4`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-02_ARCH.md` before any run.
- Everything needed is in this file and the repository; it depends on no outside document (the
  author line and the H4 row are quoted in full below).
- The reusable helpers are the committed library modules `src/jepa_noise/ext_readout.py`,
  `ext_h12.py`, `ext_h1clean.py` (not the scripts); absent at the deployed SHA → stop.
- An independent review checked this file against `57295c4`; its blocking findings (evaluate with
  the variant config, run-id collision, eval-output collision) are fixed here.

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 2 Nov 2026:
   > "B 的下一批训练我全部授权"
   The authorised H4 row of the batch plan (quoted as one line, verbatim):
   > "| H4 | architecture/masking：3 变体 × 2 seeds × 10k steps | 6 | 0.75/run | **≈4.5** | ~0（复用 dev 数据） | 低风险 |"
   This authorises **exactly** six `sigreg_jepa` runs — variants `window128`, `window32`,
   `pred4` × seeds 9000/9001 — on the committed 100k dev data, plus **one** 20-step smoke. No
   other training.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Gate (implemented in `src/jepa_noise/ext_arch.py`, called from the library training/smoke
entry, not only the CLI).** `assert_authorised()` reads the committed runbook
`docs/jepa/RUNBOOK_2026-11-02_ARCH.md` (`RUNBOOK_REL`), requires the full author line
`"B 的下一批训练我全部授权"` (exact; the substring `"我授权"` does not occur in it) **and** the H4
row literal above, requires the author line in `DECISIONS.md`, and raises `SystemExit(2)`
otherwise; the negative test is in §3.3. Do **not** reuse `ext_h1clean.assert_authorised`/
`ext_scale.assert_authorised`.

**Status.** Extension `arch_2026-11-02`, `registered: false`. `<ext>` =
`/ceph/dwong/jepa/returns/ext_arch_20261102/`; results `results/jepa/20261102_arch/`. Never
overwrite an earlier root. Grid descoped. Full suite once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Fix small problems yourself (bug, path, env, guard HOLD, non-finite loss) with one
`DECISIONS.md` line ("author standing approval, 2 Nov 2026") and continue; writes only inside
`/ceph/dwong/jepa/`, no registered changes, within §5, nothing outside `/ceph`. Stop only for
S3/S4/S9/S1 (defined in `RUNBOOK_2026-11-01_H12_CAUSAL.md:59-62`), a missing dev dataset/probe
set, or an entries preflight failure (§3.4).

## 1. Why this run

The collapse is target-driven: E3 (frozen random target) recovers (0.138/0.131 m); the trainable
target collapses to a learned event-invariant per-PMT code (0.219/0.204 m; `f_layout` 0.997,
`f_event` 7.9e-5); a frozen static target learns no physics (0.160/0.161 m, `845c7d6`); scaling
data and steps 3× does not rescue it (`results/jepa/20261102_scale/`, `scale_r = not recovered`,
0.326/0.276 m, `f_event` ≈1e-6). The remaining H4 question: **is the collapse an artefact of the
chosen architecture/masking?** This run varies the three knobs most likely to matter, one at a
time, at the same data and budget: the serialisation window (64 → 128 and 32) and the predictor
depth (2 → 4 blocks).

- **any variant recovered** (both seeds below the floor with charge): the collapse is not
  architecture-robust; report which knob and its numbers.
- **none recovered**: the collapse is robust to these architecture/masking changes at the
  development scale; H4 is ruled out for these knobs (a larger architecture sweep is a separate
  decision).

## 2. Variants and reading

All six runs are `arm_name="sigreg_jepa"`, cell `dev_s005_within`, 10 000 steps (the config's
`training.steps`), seeds 9000/9001; only the `EncoderConfig` differs, injected through the
process-level `lr.build_arm` wrapper (§3.1). The variant label is sidecar/eval only — never
`arm_name`. **Each variant uses its own out root `<ext>/train/arch/<variant>/`** (the run_id
`dev_s005_within__sigreg_jepa__s9000|9001` is identical across variants; only two distinct run_ids
exist, so the per-variant root is what disambiguates).

| variant | override (`dataclasses.replace(pe.EncoderConfig(), …)`) | out root |
|---|---|---|
| `window128` | `window=128, shift=64` | `<ext>/train/arch/window128/` |
| `window32` | `window=32, shift=16` | `<ext>/train/arch/window32/` |
| `pred4` | `n_pred_blocks=4` | `<ext>/train/arch/pred4/` |

Each run dir carries a `variant.json` sidecar (precedent `ext_h12.py:199-214`):
`{"registered": false, "extension": "arch_2026-11-02", "variant": <variant>, "seed": <int>,
"arm": "sigreg_jepa", "cell_id": "dev_s005_within", "out_root": <per-variant root>,
"run_id": <run_id>, "steps": 10000, "device": <str>}`.

**Reading `arch_r` per variant (two-sided, both seeds)**, implemented in `ext_arch` over
`eh.recover(...)["seeds"]` (do **not** reuse its verdict: it uses `<=` and calls a
vertex-fail/charge-pass pair "partial"). Floors: `ext_h12.INIT_FULL_VERTEX_M = 0.160072`
(rebuilt seed-9000; the separately *named* random-init floor `0.160904` is a different quantity,
`RUNBOOK_2026-11-01_H12_CAUSAL.md:228`), `ext_h12.INIT_HALF_VERTEX_M = 0.174821`:
- `recovered`: both seeds full-input vertex **strictly < 0.160072 m**, full token ln q R² ≥ 0.5,
  half-pass vertex < 0.174821 m, half-pass token R² ≥ 0.5.
- `not recovered`: **for each seed** (`full vertex ≥ 0.160072 m` **or** `full token ln q R² <
  0.5`), and both seeds satisfy it.
- `unclear`: anything else.
Report per variant: wall, steps, `resumed_from_step`, full/half vertex, token charge R²,
`f_layout`, `f_event`, effective rank.

## 3. Implementation

### 3.1 Modules, loaders, wrapper, exact `run_one` call

`scripts/jepa_ext_arch.py` (modes `dry-run`, `smoke`, `train`, `evaluate`, `report`),
`src/jepa_noise/ext_arch.py` (gate, `VARIANTS`, the `build_arm` wrapper, `arch_r`),
`src/jepa_noise/tests/test_ext_arch.py`. Aliases: `lr = lucid_run`, `la = lucid_arms`,
`pe = point_encoder`, `er = ext_readout`, `eh = ext_h12`, `ec = ext_h1clean`; `lr.ROOT`;
`lr.sha256_file`, `lr.git_head`, `lr.git_status`.

Setup (the `scripts/jepa_ext_h12.py:65-71` template):

```
config = lr.load_run_config(<CFG>)
gen = lb.load_generator_config(<GEN>)
arms_config = la.load_config()
geometry = lb.load_wcte_geometry(<GEOM>)
device = lr.configure_runtime(<device>)
cell = lr.dev_cell(config)
train_ds, probes = eh.datasets_for(cell, config, gen, geometry, Path(<DATA>), code_sha=None)
```

Wrapper (process-level, restored in a `finally`); `run_one` calls `lr.build_arm` without
`encoder_config`, so the wrapper injects the override:

```
import dataclasses
original = lr.build_arm
def build(arm_name, geometry_, arms_config, cell, seed, beta=None, dark_bank=None,
          encoder_config=None):
    if encoder_config is None:
        encoder_config = dataclasses.replace(pe.EncoderConfig(), **OVERRIDES[variant])
    return original(arm_name, geometry_, arms_config, cell, seed, beta=beta,
                    dark_bank=dark_bank, encoder_config=encoder_config)
lr.build_arm = build
```

Exact `lr.run_one` call (as in `ext_h12.train_variant:230-239`; `steps=None` = config 10 000):

```
lr.run_one(arm_name="sigreg_jepa", cell=cell, seed=SEED, beta=None, config=config,
           arms_config=arms_config, geometry=geometry, loader_dataset=train_ds,
           probe_datasets=probes, dark_bank=None, out_root=<ext>/train/arch/<variant>,
           data_root=Path(<DATA>), config_sha=lr.sha256_file(<CFG>), code_sha=code_sha,
           status_text=lr.git_status(lr.ROOT), device=device, steps=None,
           micro_batch=None, resume=None, stop_after=None, checkpoint_minutes=None,
           evaluate=False, allow_pending=False, step_callback=None, probe_max_events=None,
           save_embeddings=False, physics_method="ridge", tf32=False, log=print)
```

`run_one` auto-resumes when `checkpoint.pt` exists (`lucid_run.py:811-812`): refuse an existing
`out_root/runs/<run_id>` before launching and assert `resumed_from_step == 0` with
`ec.assert_fresh_resume(result["training"])` (`ext_h1clean.py:182-194`; the status dict is
returned at `lucid_run.py:866`); archive, never relaunch into a failed dir.

### 3.2 Evaluation (with the variant config) and CLI

**The evaluation must install the same wrapper before calling `eh.evaluate_run`**: `eh.evaluate_run`
→ `er.evaluate_item` → `er.item_arm` (`ext_readout.py:697-711`) rebuilds the arm via
`lr.build_arm` **without** `encoder_config`, so without the wrapper the readout would use
window=64 and would fail to strict-load `pred4` (extra predictor blocks). The evaluate mode
therefore reads the variant from the run's `variant.json` (or `--variant`), activates the wrapper,
and restores it afterwards. `--out` is **per variant** (`<ext>/eval/<variant>`), because the run
ids collide across variants.

```
python scripts/jepa_ext_arch.py dry-run --out <ext>/plan.json --data-root <DATA> \
  --config <CFG> --generator-config <GEN> --geometry <GEOM> \
  --readout-ext /ceph/dwong/jepa/returns/ext_readout_20261101 \
  --readout-items /ceph/dwong/jepa/returns/ext_readout_20261101/items.json --device cpu
python scripts/jepa_ext_arch.py smoke --variant window128 --seed 9000 --out <ext>/smoke \
  --steps 20 --probe-max-events 200 --data-root <DATA> --config <CFG> \
  --generator-config <GEN> --geometry <GEOM> --device cuda
python scripts/jepa_ext_arch.py train --variant window128 --seed 9000 \
  --out <ext>/train/arch/window128 --data-root <DATA> --config <CFG> \
  --generator-config <GEN> --geometry <GEOM> --device cuda
python scripts/jepa_ext_arch.py evaluate --run <ext>/train/arch/window128/runs/<run_id> \
  --out <ext>/eval/window128 --data-root <DATA> --config <CFG> \
  --generator-config <GEN> --geometry <GEOM> \
  --readout-ext /ceph/dwong/jepa/returns/ext_readout_20261101 \
  --readout-items /ceph/dwong/jepa/returns/ext_readout_20261101/items.json --device cuda
python scripts/jepa_ext_arch.py report --out results/jepa/20261102_arch \
  --eval <ext>/eval --plan <ext>/plan.json --reproduction <ext>/eval/reproduction.json \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM>
```

`--run` may be a run dir or its parent; evaluate writes `<out>/eval.json` with
`runs = {<seed>: {...}}` for that variant and a `reproduction` block. `<run_id>` =
`dev_s005_within__sigreg_jepa__s9000|9001`; with the per-variant roots, the six runs are
addressable. `<CFG>` = `src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (sha256
`e6730ac1…`), `<GEN>` = `src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` (sha256
`90ac1601…`), `<GEOM>` = `/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`,
`<DATA>` = `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`. The single authorised smoke is
`window128` seed 9000 (§0). `dry-run` exits 2 on failure; `train`/`smoke` refuse if the dev
cohort/probe sets are missing.

### 3.3 Tests

(1) the exact full author line authorises and the gate exits 2 when it is absent; (2) the wrapper
injects the override and leaves a plain build unchanged; (3) the wrapper is active during
evaluation: `pred4` strict-loads its 4-block predictor and a `window128` evaluation arm reports
window=128/shift=64 (and not the default 64/32) — compare against a default-config evaluation to
prove the difference; (4) `eh.evaluate_run` equals `lr.evaluate_arm_run` on the same final
checkpoint **with the same variant wrapper active**; (5) run-dir refusal and
`resumed_from_step == 0`; (6) `arch_r` three-valued on synthetic rows per variant; (7) dry-run
failure paths (missing readout items, probe set, entries projection); (8) smoke-level training
finite.

### 3.4 Data, storage, entries

Reuses the committed 100k dev cohort (`dev_train` `b1b3f4…`) and probe sets `093ea894…`/
`c11edc71…` (read-only; never regenerated). No generation. `dry-run` counts entries from the live
guard free count, projects a fresh clone (≈3 059) + the suite (≈1 825) + `<ext>` (≤ 1 000
entries), and stops if the projection is < 30 000 or the guard is HOLD. `<ext>` ≤ 2 GB and ≤
1 000 entries.

## 4. Acceptance and reproduction (S9)

1. Full suite once on the deployed SHA:
   `nice -n 10 env OMP_NUM_THREADS=4 python -m pytest -q --basetemp=<ext>/tmp/pytest
   -p no:cacheprovider` (as in `RUNBOOK_2026-11-01_H12_CAUSAL.md:224-226`).
2. Reproduction via the committed hard-coded `eh.REPRODUCTION` (`ext_h12.py:51-52`) and
   `results/jepa/20261101_h12/reproduction.json`: `fix9000` `0.219258` ± 1e-3, `init9000`
   `0.160072` ± 1e-3; exit 2 on mismatch.
3. One smoke (window128 seed 9000) finite; six runs exit 0, 10 000 steps, no NaN,
   `resumed_from_step == 0`, sidecars present; per-run cap 2.5 h (expected 0.38–0.75 h) and a
   cumulative cap of 6 GPU-h for the whole runbook; archive before any relaunch.
4. No file outside `/ceph/dwong/jepa/` on the host; `git status` shows only this extension's
   files.

## 5. Budget

| Resource | Cap | Notes |
|---|---|---|
| L40S | ≤ 6 GPU-h **cumulative** (stop at 6); ≤ 2 **concurrent** jobs; per-run cap 2.5 h | 6 runs at 10k steps (committed 10k walls 0.38–0.75 h) ≈ 4.5 GPU-h + 1 smoke + evaluations |
| Titan | 0 | not needed |
| CPU | ≤ 2 wall-hours (Deepthought, `nice -n 10`, `OMP_NUM_THREADS=4`) | tests, evaluations, report |
| Storage | `<ext>` ≤ 2 GB and ≤ 1 000 entries; projected free entries stay above 30 000 | no new data |

Ledger at review: HEAD `105.644356 / 170`; the scale booking (`+5.267644` → `110.911956`) was
being committed with `results/jepa/20261102_scale/` at review time. This runbook ≤ 6 GPU-h →
projection ≈ 117 < 160 (S1). Book every job in `RESOURCE_LEDGER.json`
(`v4_caps.E2_lucid_l40s_deepthought2`). `TMPDIR=/ceph/dwong/jepa/tmp/`.

## 6. Outputs, schemas, commit, report

1. `results/jepa/20261102_arch/`: `README.md` (`registered: false`), `runs.json` (keyed
   `<variant>__<seed>`: wall, steps, resumed_from_step, metrics_finite, full/half vertex, token
   charge R², f_layout, f_event, eff_rank), `readings.json` with
   `{"registered": false, "extension": "arch_2026-11-02", "criteria": {"floors": {"full":
   0.160072, "half": 0.174821, "charge_r2_min": 0.5}, "rule": "<the pinned §2 rule>"},
   "variants": {"window128": {"verdict": "recovered"|"not recovered"|"unclear", "per_seed":
   {"9000": {…}, "9001": {…}}}, "window32": {…}, "pred4": {…}}}`, `reproduction.json`,
   `plan.json`; each ≤ 1 MB.
2. `<ext>/`: checkpoints, sidecars, caches deleted after `report`, hashes in
   `<ext>/large_arrays.json`, `manifest.jsonl`.
3. `DECISIONS.md` one line per engineering choice; a short `STATE.md` entry; book resources; do
   not touch `PREDICTIONS.md`/`PREDICTION_STATUS.json`.
4. Commit order: runbook + code + tests first, then runs, then results; report the final SHA.
5. `report` requires all six runs (three variants × two seeds); missing any exits 2. Report to
   the author in Chinese (≤ 1 page): per-variant verdicts with numbers, resources, final SHA.
6. Do not resume the grid; follow-ups (other architecture knobs, view design) are separate
   runbooks.
