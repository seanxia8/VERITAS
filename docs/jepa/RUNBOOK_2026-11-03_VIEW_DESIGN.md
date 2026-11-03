# RUNBOOK — view design: four candidate splits (2 Nov 2026, reviewed, AUTHORISED)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the design
agent, for the author. Branch `JEPA`, at or after `1f6713c`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md` before any run.
- Everything needed is in this file and the repository; it depends on no outside document (the
  author line and the C row are quoted in full below).
- The reusable helpers are the committed library modules `src/jepa_noise/ext_readout.py`,
  `ext_h12.py`, `ext_h1clean.py` and the view code `src/jepa_noise/lucid_views.py` (read-only
  imports; the two new splits are registered by a process-level patch, never edited in place).
- An independent review checked this file against `e9ba1a0` (the only later changes to
  `1f6713c` are DECISIONS/ledger docs); its blocking findings (C-row literal, evaluate-item split)
  are fixed here.

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 2 Nov 2026:
   > "B 的下一批训练我全部授权"
   The authorised C row of the batch plan (quoted as one line, verbatim):
   > "| C | view design：4 个候选 split × 2 seeds × 10k steps | 8 | 0.75/run | **≈6.0** | ~0（复用 dev 数据，split 是 runtime mask） | 低风险 |"
   This authorises **exactly** eight `sigreg_jepa` runs — four candidate splits (`within`,
   `across`, `module_parity`, `random_half`) × seeds 9000/9001 — on the committed 100k dev
   data, plus **one** 20-step smoke. No other training.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Gate (implemented in `src/jepa_noise/ext_view.py`, called from the library training/smoke
entry, not only the CLI).** `assert_authorised()` reads the committed runbook
`docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md` (`RUNBOOK_REL`), requires the full author line
`"B 的下一批训练我全部授权"` (exact; the substring `"我授权"` does not occur in it) **and** the C
row literal above, requires the author line in `DECISIONS.md`, and raises `SystemExit(2)`
otherwise; negative test in §3.4. Do not reuse the H1-clean/scale/arch gates.

**Status.** Extension `view_design_2026-11-03`, `registered: false`. `<ext>` =
`/ceph/dwong/jepa/returns/ext_view_20261103/`; results `results/jepa/20261103_view/`. Never
overwrite an earlier root. Grid descoped. Full suite once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Fix small problems yourself (bug, path, env, guard HOLD, non-finite loss) with one
`DECISIONS.md` line ("author standing approval, 2 Nov 2026") and continue; writes only inside
`/ceph/dwong/jepa/`, no registered changes, within §5, nothing outside `/ceph`. Stop only for
S3/S4/S9/S1 (defined in `RUNBOOK_2026-11-01_H12_CAUSAL.md:59-62`: S1 = L40S ledger + projection
> 160 GPU-h; S3 = a real spec change; S4 = no GitHub route after three documented attempts; S9 =
an acceptance check fails after one fix attempt), a missing dev dataset/probe set, or an entries
preflight failure (§3.5).

## 1. Why this run

The collapse is target-driven (E3 frozen random target recovers; the trainable target collapses
to a learned event-invariant code; frozen static target learns nothing) and is robust to scale
(3× data + 3× steps, `scale_r = not recovered`, 0.326/0.276 m, `results/jepa/20261102_scale/`)
and to the architecture knobs (window 128/32, 4-block predictor, `arch_r = not recovered`,
0.163–0.234 m, `results/jepa/20261102_arch/`). The remaining H4/C question: **is the choice of
view split the lever?** The registered `within` split is parity-based; `across` (a per-event
random contiguous half of the Hilbert-ordered modules) carries the trigger-jitter trap
(ν* = 0.51 without drift). This run tests two further splits — whole-module parity and a seeded
random half — beside the two registered ones, at the same data and budget.

- **any candidate recovered** (both seeds below the floor with charge): that split is a candidate
  for the designed view; report which and its numbers.
- **none recovered**: no split in this set makes the physics learnable at the development
  budget; the view is not the lever (the target-content mechanism is).

## 2. Candidate splits and reading

All runs are `arm_name="sigreg_jepa"`, 10 000 steps, seeds 9000/9001; the only change is the
split recorded in the cell (and the masker). Out root per candidate
`<ext>/train/views/<split>/` (the run_id `dev_s005_within__sigreg_jepa__s9000|9001` is identical
across splits — only two distinct run_ids exist — so the per-candidate root disambiguates).
Each run dir carries a `variant.json` sidecar (precedent `ext_h12.py:199-214`):
`{"registered": false, "extension": "view_design_2026-11-03", "split": <split>, "seed": <int>,
"arm": "sigreg_jepa", "cell_id": <cell_id>, "out_root": <per-split root>, "run_id": <run_id>,
"steps": 10000, "device": <str>}`. The sidecar `cell_id` stays `dev_s005_within` for all four
splits (`dataclasses.replace` keeps it), so the report keys runs by `<split>__<seed>`, never by
`cell_id`.

| split | definition | status |
|---|---|---|
| `within` | registered parity split: even 1-based `pmt_id_in_mpmt` → context, odd → target | existing (`lucid_views.within_mask`) |
| `across` | Hilbert order of the modules, per-event random contiguous half with redraws until `target_hits ≥ n_min` (not a fixed offset) | existing (`lucid_views.across_mask`) |
| `module_parity` | **new**: whole modules by **dense** `geometry.module_index` parity (even → context, odd → target); raw `mpmt_id` parity would differ because raw id 45 is absent | extension |
| `random_half` | **new**: seeded permutation of the 1 995 PMTs (`lb.keyed_rng(dataset_seed, "lucid/views", "random_half", 20260101)`), first `n_pmt//2 = 997` → context, the remaining 998 → target, fixed across events; salt and seed recorded in the key | extension |

**Reading `view_r` per split (two-sided, both seeds)**, implemented in `ext_view` over
`eh.recover(...)["seeds"]` (do **not** reuse its verdict: it uses `<=` and calls a
vertex-fail/charge-pass pair "partial"). Floors: `ext_h12.INIT_FULL_VERTEX_M = 0.160072`
(the measured rebuild; `0.160904` is the older recorded expected value of the same init9000
full-input vertex from the 26 Oct pregrid resume), `INIT_HALF = 0.174821`:
- `recovered`: both seeds full-input vertex **strictly < 0.160072 m**, full token ln q R² ≥ 0.5,
  half-pass vertex < 0.174821 m, half-pass token R² ≥ 0.5.
- `not recovered`: **for each seed** (`full vertex ≥ 0.160072 m` **or** `full token ln q R² <
  0.5`), and both seeds satisfy it.
- `unclear`: anything else.
Report per split: wall, steps, `resumed_from_step`, full/half vertex, token charge R²,
`f_layout`, `f_event`, effective rank; rank the splits by full vertex and recommend the best.
Caveat to report: `half_pass` always uses the registered `within` target mask
(`ext_readout.py:66,743`), so for `module_parity`/`random_half` it is the same fixed probe as
every other run — comparable, but not "the pass the run saw during training".

## 3. Implementation

### 3.1 Modules, loaders, the split patch (exact code)

`scripts/jepa_ext_view.py` (modes `dry-run`, `smoke`, `train`, `evaluate`, `report`),
`src/jepa_noise/ext_view.py` (gate, `SPLIT_NAMES`, the masker patch, wrappers, `view_r`),
`src/jepa_noise/tests/test_ext_view.py`. Aliases: `lr = lucid_run`, `la = lucid_arms`,
`pe = point_encoder`, `lv = lucid_views`, `er = ext_readout`, `eh = ext_h12`, `ec = ext_h1clean`;
`lr.ROOT`; `lr.sha256_file`, `lr.git_head`, `lr.git_status`.

Setup (the `scripts/jepa_ext_h12.py:65-71` template):

```
config = lr.load_run_config(<CFG>); gen = lb.load_generator_config(<GEN>)
arms_config = la.load_config(); geometry = lb.load_wcte_geometry(<GEOM>)
device = lr.configure_runtime(<device>)
cell = dataclasses.replace(lr.dev_cell(config), split=SPLIT)
train_ds, probes = eh.datasets_for(cell, config, gen, geometry, Path(<DATA>), code_sha=None)
```

`lucid_run` imports `ViewMask, ViewMasker` at import time (`lucid_run.py:26`); `_masks_for`
(`:361-370`), `_tokenise_run` (`:372-374`) and the evaluation path (`:885-891`) construct
`ViewMasker(geometry, split, seed=…, occupancy=…, n_min=…)`. `point_encoder` has its **own**
binding (`point_encoder.py:40`, used at `:183`), so both must be patched.

```
import dataclasses, numpy as np
from . import lucid_bombs as lb, lucid_views as lv

SPLIT_NAMES = ("within", "across", "module_parity", "random_half")
RANDOM_HALF_SALT = 20260101

class SplitMasker(lv.ViewMasker):
    def __init__(self, geometry, split, *, seed=0, replicate=0, cell="", n_min=lv.N_MIN,
                 max_redraws=lv.MAX_REDRAWS, occupancy="natural", n_target_modules=None,
                 module_bits=lv.MODULE_HILBERT_BITS):
        if split not in SPLIT_NAMES:
            raise ValueError(f"unregistered split {split!r}")
        # bypass ViewMasker.__init__'s split-in-SPLITS check; copy its field setup
        self.geometry, self.split, self.seed = geometry, split, int(seed)
        self.replicate, self.cell = int(replicate), cell
        self.n_min, self.max_redraws = int(n_min), int(max_redraws)
        self.occupancy = occupancy; self.n_target_modules = n_target_modules
        self.module_bits = int(module_bits)
        self.counter = lv.ViewExclusionCounter(n_min=self.n_min, max_redraws=self.max_redraws)
        self.fraction_context = self.fraction_target = 0.0
        self.n_seen = self.n_balanced = 0

    def mask(self, event_id, hit_pmt):
        if self.split in ("within", "across"):
            return super().mask(event_id, hit_pmt)
        hit = lv._hit_array(self.geometry, hit_pmt)
        if self.split == "module_parity":
            in_context = (self.geometry.module_index % 2) == 0
        else:  # random_half
            rng = lb.keyed_rng(int(self.seed), "lucid/views", "random_half", RANDOM_HALF_SALT)
            order = rng.permutation(self.geometry.n_pmt)
            in_context = np.zeros(self.geometry.n_pmt, dtype=bool)
            in_context[order[: self.geometry.n_pmt // 2]] = True
        context = np.flatnonzero(in_context).astype(np.int64)
        target = np.flatnonzero(~in_context).astype(np.int64)
        context_hits, target_hits = int(hit[context].sum()), int(hit[target].sum())
        accepted = target_hits >= self.n_min
        key = {"split": self.split, "geometry_sha256": self.geometry.digest,
               "event_id": int(event_id), "seed": int(self.seed),
               "salt": RANDOM_HALF_SALT if self.split == "random_half" else None,
               "occupancy": "natural", "n_min": int(self.n_min),
               "context_hits": context_hits, "target_hits": target_hits,
               "module_index_parity": "even=context,odd=target" if self.split == "module_parity" else None}
        mask = lv.ViewMask(self.split, int(event_id), context, target, accepted, 0,
                           context_hits, target_hits, None, (), np.empty(0, np.int64), key)
        self.counter.record_within(target_hits); self.n_seen += 1
        if accepted:
            fr = mask.fractions(hit)
            self.fraction_context += fr["context"]; self.fraction_target += fr["target"]
        return mask

@contextlib.contextmanager
def masker_patch():
    old_lr, old_pe = lr.ViewMasker, pe.ViewMasker
    lr.ViewMasker = pe.ViewMasker = SplitMasker
    try:
        yield
    finally:
        lr.ViewMasker, pe.ViewMasker = old_lr, old_pe
```

Install `masker_patch()` around `train`/`smoke` only. The evaluation of the trained checkpoint
uses the registered probe split (`within`) and **must not** install the patch (see §3.3).

### 3.2 Exact `run_one` call

```
lr.run_one(arm_name="sigreg_jepa", cell=cell, seed=SEED, beta=None, config=config,
           arms_config=arms_config, geometry=geometry, loader_dataset=train_ds,
           probe_datasets=probes, dark_bank=None, out_root=<ext>/train/views/<split>,
           data_root=Path(<DATA>), config_sha=lr.sha256_file(<CFG>), code_sha=code_sha,
           status_text=lr.git_status(lr.ROOT), device=device, steps=None,
           micro_batch=None, resume=None, stop_after=None, checkpoint_minutes=None,
           evaluate=False, allow_pending=False, step_callback=None, probe_max_events=None,
           save_embeddings=False, physics_method="ridge", tf32=False, log=print)
```

`run_one` auto-resumes when `checkpoint.pt` exists (`lucid_run.py:811-812`): refuse an existing
`out_root/runs/<run_id>` before launching and assert `resumed_from_step == 0` with
`ec.assert_fresh_resume(result["training"])` (`ext_h1clean.py:182-194`); archive a failed run
dir, never relaunch into it.

### 3.3 Evaluation item (pinned) and CLI

**The evaluate mode must not derive the item split from the run** (`variant.json`/`run_status.json`
record `module_parity`/`random_half`, and `ext_readout.evaluate_item:723` uses `item["split"]`;
`lucid_run.py:885-891` builds a `ViewMasker` from the caller's split). Pin the item:

```
item = {"label": run_id, "arm": "sigreg_jepa", "split": "within", "seed": SEED,
        "source": str(run_dir / "final.pt")}
```

and do not install `masker_patch()` around evaluation. The evaluate mode writes
`<ext>/eval/<split>/eval.json` and `<ext>/eval/reproduction.json` (the `ext_arch` precedent,
`scripts/jepa_ext_arch.py:254-266`); `report` reads the four per-split eval files.

```
python scripts/jepa_ext_view.py dry-run --out <ext>/plan.json --data-root <DATA> \
  --config <CFG> --generator-config <GEN> --geometry <GEOM> \
  --readout-ext /ceph/dwong/jepa/returns/ext_readout_20261101 \
  --readout-items /ceph/dwong/jepa/returns/ext_readout_20261101/items.json --device cpu
python scripts/jepa_ext_view.py smoke --split module_parity --seed 9000 --out <ext>/smoke \
  --steps 20 --probe-max-events 200 --data-root <DATA> --config <CFG> \
  --generator-config <GEN> --geometry <GEOM> --device cuda
python scripts/jepa_ext_view.py train --split module_parity --seed 9000 \
  --out <ext>/train/views/module_parity --data-root <DATA> --config <CFG> \
  --generator-config <GEN> --geometry <GEOM> --device cuda
python scripts/jepa_ext_view.py evaluate --split module_parity \
  --run <ext>/train/views/module_parity/runs/<run_id> \
  --out <ext>/eval/module_parity --data-root <DATA> --config <CFG> \
  --generator-config <GEN> --geometry <GEOM> \
  --readout-ext /ceph/dwong/jepa/returns/ext_readout_20261101 \
  --readout-items /ceph/dwong/jepa/returns/ext_readout_20261101/items.json --device cuda
python scripts/jepa_ext_view.py report --out results/jepa/20261103_view --eval <ext>/eval \
  --plan <ext>/plan.json --reproduction <ext>/eval/reproduction.json \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM>
```

`<run_id>` = `dev_s005_within__sigreg_jepa__s9000|9001`. `<CFG>` = `…lucid_run_v0_mixed_calibrated.json`
(sha256 `e6730ac1…`), `<GEN>` = `…lucid_bombs_v2_calibrated.json` (`90ac1601…`), `<GEOM>` =
`/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`, `<DATA>` =
`/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`. The single authorised smoke is
`module_parity` seed 9000; `train`/`smoke` derive `<ext>` from `--out` (two levels up from the
per-split root) and re-check `<ext>/plan.json`. `dry-run` exits 2 on failure; `train`/`smoke`
refuse if the data preflight fails.

### 3.4 Tests

(1) the exact full author line authorises and the gate exits 2 when it is absent; (2) the masker
patch: `within`/`across` masks are bit-identical to the unpatched `ViewMasker`; `module_parity`
puts every module's 19 PMTs on one side with the **dense** `module_index` parity; `random_half`
gives a fixed **997/998** partition across events (seeded salt 20260101, key recorded); the
`report()` counters are non-zero after a batch for the new splits; (3) `pe.ViewMasker` is patched
too (a direct `pe.tokenise(split="module_parity")` works); (4) run-dir refusal and
`resumed_from_step == 0`; (5) `view_r` three-valued on synthetic rows per split; (6) dry-run
failure paths (missing readout items, probe set, entries projection); (7) smoke-level training
finite with the patched masker; (8) the evaluate item is pinned to `split="within"` even for a
`module_parity` run.

### 3.5 Data, storage, entries

Reuses the committed 100k dev cohort (`dev_train` `b1b3f4…`) and probe sets `093ea894…`/
`c11edc71…` (read-only; never regenerated). **`eh.datasets_for` calls `ensure_dataset`, which
regenerates a missing set**, so `train`/`smoke`/`dry-run` must first call `eh.preflight_data`
(or the `ext_arch.require_data` equivalent) and stop on a missing set or manifest-hash mismatch.
`dry-run` also reads the live free-entry count from the guard
(`/ceph/dwong/jepa/jobs/guard_deepthought2.json`, `caps.free_entries`; HOLD check), projects a
fresh clone (≈3 059) + the suite (≈1 825) + `<ext>` (≤ 1 000), and stops if the projection is
< 30 000 or the guard is HOLD. `<ext>` ≤ 2 GB and ≤ 1 000 entries.

## 4. Acceptance and reproduction (S9)

1. Full suite once on the deployed SHA:
   `nice -n 10 env OMP_NUM_THREADS=4 python -m pytest -q --basetemp=<ext>/tmp/pytest
   -p no:cacheprovider` (as in `RUNBOOK_2026-11-01_H12_CAUSAL.md:224-226`).
2. Reproduction via the committed hard-coded `eh.REPRODUCTION` (`ext_h12.py:51-52`) and
   `results/jepa/20261101_h12/reproduction.json`: `fix9000` `0.219258` ± 1e-3, `init9000`
   `0.160072` ± 1e-3; exit 2 on mismatch. The evaluate mode writes `<ext>/eval/reproduction.json`
   and `report` reads it.
3. One smoke (`module_parity` seed 9000) finite; eight runs exit 0, 10 000 steps, no NaN,
   `resumed_from_step == 0`, sidecars present; per-run cap 2.5 h (expected 0.38–0.75 h for the
   default arm; H12 walls 0.384–0.742 h) and a cumulative cap of 8 GPU-h; archive before any
   relaunch.
4. No file outside `/ceph/dwong/jepa/` on the host; `git status` shows only this extension's
   files — remove/ignore the pre-existing untracked `.pytest-tmp-predictability-full/` before
   the final check.

## 5. Budget

| Resource | Cap | Notes |
|---|---|---|
| L40S | ≤ 8 GPU-h **cumulative** (stop at 8); ≤ 2 **concurrent** jobs; per-run cap 2.5 h | 8 runs at 10k steps (H12 default-arm walls 0.384–0.742 h) ≈ 6 GPU-h + 1 smoke + evaluations ≈ 6.8 |
| Titan | 0 | not needed |
| CPU | ≤ 2 wall-hours (Deepthought, `nice -n 10`, `OMP_NUM_THREADS=4`) | tests, evaluations, report |
| Storage | `<ext>` ≤ 2 GB and ≤ 1 000 entries; projected free entries stay above 30 000 | no new data |

Ledger at release **116.151 / 170** (`1f6713c`; 116.113 at the reviewed `e9ba1a0`); this runbook
≤ 8 GPU-h → projection ≈ 124 < 160 (S1). Book every job in `RESOURCE_LEDGER.json`
(`v4_caps.E2_lucid_l40s_deepthought2`). `TMPDIR=/ceph/dwong/jepa/tmp/`.

## 6. Outputs, schemas, commit, report

1. `results/jepa/20261103_view/`: `README.md` (`registered: false`), `runs.json` (keyed
   `<split>__<seed>`: wall, steps, resumed_from_step, metrics_finite, full/half vertex, token
   charge R², f_layout, f_event, eff_rank), `readings.json` with
   `{"registered": false, "extension": "view_design_2026-11-03", "criteria": {"floors": {"full":
   0.160072, "half": 0.174821, "charge_r2_min": 0.5}, "rule": "<the pinned §2 rule>"},
   "splits": {"within": {"verdict": "recovered"|"not recovered"|"unclear", "per_seed": {…}},
   "across": {…}, "module_parity": {…}, "random_half": {…}}, "ranking": […],
   "recommendation": "…"}`, `reproduction.json`, `plan.json`; each ≤ 1 MB.
2. `<ext>/`: checkpoints, sidecars, caches deleted after `report`, hashes in
   `<ext>/large_arrays.json`, `manifest.jsonl`.
3. `DECISIONS.md` one line per engineering choice; a short `STATE.md` entry; book resources; do
   not touch `PREDICTIONS.md`/`PREDICTION_STATUS.json`.
4. Commit order: runbook + code + tests first, then runs, then results; report the final SHA.
5. `evaluate`/`report` take `--split`; `report` requires all eight runs (four splits × two
   seeds); missing any exits 2. Report to the author in Chinese (≤ 1 page): per-split verdicts
   with numbers, the ranking, the recommendation, resources, final SHA.
6. Do not resume the grid; follow-ups (split parameterizations, target-content controls) are
   separate runbooks.
