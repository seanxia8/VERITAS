# RUNBOOK — clean H1 causal test: predictor position-query noise (2 Nov 2026, reviewed, NOT YET AUTHORISED)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the design
agent, for the author. Branch `JEPA`, at or after `35c3faa`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md` before any run.
- Everything needed is in this file and the repository; it depends on no outside document.
- "§3.5.5"/"§3.5.6" refer to `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`.
- Reuses the committed Q1/Q2 and H1/H2 extensions (`scripts/jepa_ext_readout.py`,
  `scripts/jepa_ext_h12.py`); if they are absent at the deployed SHA, stop (dependency).
- An independent review checked this file against `35c3faa`; its blocking findings are fixed here._

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 1 Nov 2026, 23:42 CET:
   > "我授权，你就按这个顺序做：A1（你一句话）+ A2（论文 A 审阅）"
   Item A1 is the clean H1 query-noise test in this file. It authorises **exactly** two
   `sigreg_jepa_query_noise` runs (seeds 9000, 9001) and **one** 20-step smoke. No other
   training.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Hard gate (enforced by the extension).** The `train` and `smoke` modes must exit 2 unless
the committed runbook's §0 contains the author line (substring "我授权") **and**
`DECISIONS.md` contains the same line; no run may start otherwise. If the deployed runbook's
§0 is the earlier "TRAINING NOT AUTHORISED" banner, stop and report (wrong SHA).

**Status.** Extension `h1_clean_2026-11-02`, `registered: false`. `<ext>` =
`/ceph/dwong/jepa/returns/ext_h1clean_20261102/`; results `results/jepa/20261102_h1clean/`.
Never overwrite an earlier root. The LUCiD grid is descoped. Full suite once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Fix small problems yourself (bug, path, env, guard HOLD, non-finite loss, σ calibration within
the pre-declared ladder) with one `DECISIONS.md` line ("author standing approval, 2 Nov 2026")
and continue. Writes only inside `/ceph/dwong/jepa/`; no registered changes; within §5 budget;
nothing outside `/ceph`. The only self-fix that changes the treatment is the pre-declared σ
ladder in §3.4; nothing else may change the variant. Stop only for S3 (real spec change), S4
(no GitHub route after the three documented attempts), S9 (an acceptance check fails after one
fix attempt), S1 (L40S ledger + projection > 160 GPU-h), a missing dataset after the §3.5
preflight, or a failed manipulation check at σ = 0.15 (§3.4).

## 1. Why this run

The H1/H2 extension (`35c3faa`) gave: **E3 (frozen random target) supported → H2 dominant**
(trainable-target trivial solution); **E2 (geometry feature ablation) not supported formally**,
because zeroing the six geometry columns also removes the input the vertex readout needs
(full vertex 0.83 m), while the same run shows the layout-dominated/event-independent regime
disappearing (layout fraction 0.021, event fraction 0.35; the fix9000 label has layout 0.997,
event 7.9e-5). E2 is therefore "consistent with H1" but not a valid causal test.

This runbook keeps the **encoder input unchanged** (all geometry present) and removes only the
predictor's ability to use the *exact* target position: independent Gaussian noise is added to
the predictor's position-query inputs during training. The test is **one-sided**:
- a **recovery** (the model beats the random-init floor) would show that the exact-position
  query route is what keeps the model collapsed → the **query-position route** is supported;
- a **non-recovery** is **not** evidence for H2; it is `unclear (non-discriminating)` unless
  the manipulation check (§3.4) shows the noise actually removed the query information, in
  which case it is `not supported` for the *query route*.

**The label-content version of H1 is already answered (addendum, `b5a18a6`).** On held-out
PMTs the `fix9000` label's event-invariant per-PMT component is **not** position-explained:
linear −0.008, MLP −0.009, kNN 0.305 (local-neighbour structure only), against the `init9000`
reference 0.68/0.79/0.89. The row-level MLP R² 0.678 is an **in-sample per-PMT fit** (the row
split is by event, so the same PMTs appear in train and test), not a position function that
generalises to new PMTs. So the trained label is a fixed per-PMT code of *other* origin, not a
function of the PMT position. This runbook tests a different route: whether the predictor's
exact-position query is what lets the model use that code. A positive result therefore supports
the **query-position route**, not "the label is geometry". (A direct test of the code's content
is the label-centering variant noted in the H12 follow-up; it is a separate runbook.)

## 2. Variant

**`sigreg_jepa_query_noise`** (label only; `arm_name` stays `"sigreg_jepa"`, because
`la.build_arm` rejects names outside `ALL_ARMS`, `src/jepa_noise/lucid_arms.py:83,326-327`).
The predictor is `LucidPointJepa.predict` (`src/jepa_noise/point_encoder.py:365-369`):
`queries = self.query_features[target_index]` (`:368`), then
`return self.predictor(context, queries)` (`:369`). Add i.i.d. `N(0, σ²)` to each of the
three scaled query coordinates of each (event, target) row, before the query projection,
**during training only**. σ is in `FeatureScaling` units (positions divided by
`radial_scale_m` for x,y and `axial_scale_m` for z); with the WCTE scalars
(`docs/jepa/PREFLIGHT_A8_deepthought.md:984`) R = 1.57713 m, H/2 = 1.38849 m, **σ = 0.10**
corresponds to **0.157 m (x,y) / 0.139 m (z)** — about one PMT pitch (~0.15 m) and the vertex
floor (0.160 m). The noise is drawn from the global torch RNG (seeded from the run seed at
`src/jepa_noise/lucid_run.py:800`), so an uninterrupted run is reproducible. The encoder
input, label, SIGReg, predictor architecture and all other hyper-parameters are unchanged.
Probe evaluation uses `encode` only (`lucid_arms.py:699-716`; `ext_readout.py:717-759`), so it
cannot see the noise — provided no evaluation runs inside the patch context.

**Runs:** seeds 9000 and 9001, plus one 20-step smoke. Out root `<ext>/train/query_noise/`,
with `variant.json` and `variants.json` sidecars (the `run_one` manifest has no variant field).
The extension must **refuse an existing run directory** (reuse the H12 guard,
`src/jepa_noise/ext_h12.py:191-205`) and assert `resumed_from_step == 0`: `_checkpoint` does
not save the global torch RNG (`src/jepa_noise/lucid_run.py:646-653`), so a resume would
replay the noise stream from step 0.

## 3. Implementation

### 3.1 Code

`scripts/jepa_ext_h1clean.py` (modes `dry-run`, `smoke`, `train`, `evaluate`, `report`),
`src/jepa_noise/ext_h1clean.py` (query-noise patch, sidecars, thin variant wrappers),
`src/jepa_noise/tests/test_ext_h1clean.py`. Reuse (import read-only) from the committed
extensions: `er.evaluate_item`, `eh.evaluate_run`, `eh.reproduction`, `eh.recover`,
`eh.preflight_data`, `eh.load_probes`, `eh.datasets_for`, and the `build_arm` wrapper pattern
`eh.build_arm_patch` (`src/jepa_noise/ext_h12.py:162-179`). Do **not** reuse
`eh.train_variant`/`eh.prepare_run_dir`/`eh.discover_runs`/`eh.summarize_run` blindly: they
hardcode `eh.VARIANTS`/`eh.EXTENSION` (`ext_h12.py:31-37,191-205,244-270`) and reject the new
label; implement thin wrappers in `ext_h1clean` that call the same primitives.

**Patch.** Replace `arm.jepa.predict` on the built arm with a function that (i) gathers the
context exactly as the original does (`readout.gather(1, context_index.unsqueeze(-1).expand(...))`),
(ii) builds `queries = self.query_features[target_index]` and adds
`torch.randn_like(queries) * σ` (same device/dtype), (iii) calls `self.predictor(context, queries)`.
The original method is restored in a `finally`. The patch is installed by the process-level
`lucid_run.build_arm` wrapper and must never be active during `evaluate`/`report` (those run
`evaluate=False` training and a separate evaluation process).

**Tests.** (1) the encoder input tensor is byte-identical with the patch installed (noise
touches only the query tensor); (2) with a fixed reseed before each pass, two patched forward
passes give the same noise, and the patched prediction differs from the unpatched one
(max |Δ| > 0); (3) an evaluation run outside the patch context equals an unpatched evaluation;
(4) the run-dir refusal triggers on an existing directory and `resumed_from_step == 0`; (5)
smoke-level training with the patch gives a finite loss and finite metrics.

### 3.2 Exact `run_one` call

```
lr.run_one(arm_name="sigreg_jepa", cell=lr.dev_cell(config), seed=SEED, beta=None,
           config=config, arms_config=arms_config, geometry=geometry,
           loader_dataset=train_ds, probe_datasets=probe_ds, dark_bank=None,
           out_root=<ext>/train/query_noise, data_root=<DATA>,
           config_sha=sha256_file(config_path), code_sha=git_head(ROOT),
           status_text=git_status(ROOT), device=device, steps=None, micro_batch=None,
           resume=None, stop_after=None, checkpoint_minutes=None, evaluate=False,
           allow_pending=False, step_callback=None, probe_max_events=None,
           save_embeddings=False, physics_method="ridge", tf32=False, log=print)
```

The wrapper signature must match `build_arm(arm, geometry, arms_config, cell, seed, beta=None,
dark_bank=None, encoder_config=None)` (`ext_h12.py:167-170`).

### 3.3 CLI

```
python scripts/jepa_ext_h1clean.py dry-run --out <ext>/plan.json --data-root <DATA> \
  --config <CFG> --generator-config <GEN> --geometry <GEOM>
python scripts/jepa_ext_h1clean.py smoke --seed 9000 --out <ext>/smoke \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM> --steps 20
python scripts/jepa_ext_h1clean.py train --seed 9000 --out <ext>/train/query_noise \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM>
python scripts/jepa_ext_h1clean.py evaluate --run <ext>/train/query_noise/runs/<run_id> \
  --out <ext>/eval --readout-ext /ceph/dwong/jepa/returns/ext_readout_20261101 \
  --readout-items /ceph/dwong/jepa/returns/ext_readout_20261101/items.json
python scripts/jepa_ext_h1clean.py report --out results/jepa/20261102_h1clean \
  --eval <ext>/eval --reproduction <ext>/eval/reproduction.json
```

`<CFG>` = `src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (sha256 `e6730ac1…`),
`<GEN>` = `src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` (sha256 `90ac1601…`),
`<GEOM>` = `/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`,
`<DATA>` = `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`. `dry-run` validates the data
preflight and the checkpoint/hash manifest and writes `plan.json`; it must run before the first
`train`/`smoke`. `<run_id>` = `dev_s005_within__sigreg_jepa__s9000|9001`.

### 3.4 σ and the manipulation check (pre-declared)

σ = **0.10** scaled units. Before the first training run, `dry-run` performs a **manipulation
check** on one fixed saved batch with the untrained arm (seed 9000): the prediction MSE with
the query noise must be **≥ 1.5 ×** the MSE without it. If it is not, use σ = **0.15** (one
step, record it); if it still fails, **stop and report** (the noise does not remove the query
information; the test is non-discriminating). The check result is written into `plan.json` and
reported.

### 3.5 Data (read-only; no regeneration)

The three development keys and recorded manifest sha256 from
`RUNBOOK_2026-11-01_H12_CAUSAL.md` §3.4: `dev_train__F1__0.05__s20261017__n100000x1000`
`b1b3f4…`, `dev_probe_train__…` `093ea894…`, `dev_probe_test__…` `c11edc71…` (confirmed in
`results/jepa/20261025_pregrid_checks/provenance.json`). If any is absent or differs, stop and
report — never regenerate (the data root is a link farm into `/work`). The preflight runs in
`dry-run` and again at the start of `train`/`smoke`.

## 4. Acceptance and reproduction (S9)

1. Full suite once on the deployed SHA.
2. Reproduction: `fix9000` full-input vertex `0.219258` ± 1e-3 and `init9000` `0.160072`
   ± 1e-3 (`decision.json` floors; the *named* random-init floor `0.160904` is a different
   quantity — cite `DECISIONS.md:2394`), using the Q1/Q2 readout code; exit 2 on mismatch.
3. Noise isolation tests (§3.1) pass; the encoder input is untouched; no evaluation runs inside
   the patch context.
4. The manipulation check passes (§3.4); one smoke is finite; both runs exit 0, 10 000 steps,
   no NaN, `resumed_from_step == 0`, variant sidecars present.
5. No file outside `/ceph/dwong/jepa/` is written. The repo clone is
   `/ceph/dwong/jepa/code/<sha>/`, so `results/jepa/`, `DECISIONS.md` and `STATE.md` are inside
   it; the only non-`/ceph` write is the git push to GitHub. `git status` shows only this
   extension's files (the pre-existing untracked `.pytest-tmp-predictability-full/` is ignored).

## 5. Budget

| Resource | Cap | Notes |
|---|---|---|
| L40S | ≤ 6 GPU-h; ≤ 2 **concurrent** jobs; 90 min per run | 2 runs (existing JEPA dev runs 0.44–0.96 h) + 1 smoke + evaluations; expected ≈ 3 h |
| Titan | 0 | not needed |
| CPU | ≤ 3 wall-hours (Deepthought, `nice -n 10`, `OMP_NUM_THREADS=4`) | tests, evaluation fits, report |
| Storage | ≤ 2 GB and ≤ 1 000 entries under `<ext>` (the H12 `<ext>` was 86 MB for six checkpoints plus caches); free entries stay above 30 000 | two 7.5 MB checkpoints plus caches |

Current L40S ledger **104.635 / 170**; S1 start rule 160, so 104.635 + ≈ 3 ≈ 108 < 160. Book
every job in `RESOURCE_LEDGER.json` (`v4_caps.E2_lucid_l40s_deepthought2`).
`TMPDIR=/ceph/dwong/jepa/tmp/`; pytest `--basetemp` inside `<ext>/tmp/`, `-p no:cacheprovider`.

## 6. Pre-declared reading (one-sided)

**H1-clean-r**, evaluated with the Q1/Q2 readout protocol (full-input and half-pass photon
vertex resolution, `token_log_charge_r2`), using `eh.recover`'s criteria with `partial`
mapped to `unclear` and **strict** half-pass comparison (`< 0.174821`):

- **supported**: both seeds have full-input vertex ≤ `0.160072`, token ln q R² ≥ 0.5, and
  half-pass vertex < `0.174821` with half-pass charge ≥ 0.5.
- **not supported**: neither criterion holds for both seeds, or the half-pass contradicts.
- **unclear**: partial (one of the two criteria for both seeds) or the seeds disagree.

**Interpretation (one-sided; the label-content version of H1 is already not supported by the
`b5a18a6` addendum):**
- **supported** → the exact-position query route contributes to the collapse (the predictor
  uses the query position to select/combine the fixed per-PMT code); this is a **mechanism**
  result, not evidence that the label is a position function.
- **not supported** → the query route is not sufficient; this is **not** evidence for H2. If
  the manipulation check passed, record "query route not supported"; if it did not, record
  `unclear (non-discriminating)`.
- Also report the run's `f_layout`, `f_event` and effective rank (Q1/Q2 estimators). If
  `f_layout` stays ≈0.99 and `f_event` ≈10⁻⁵, the query noise did not change the label's layout
  dominance even if the vertex readout improves; say so.

## 7. Outputs, commit, report

1. `results/jepa/20261102_h1clean/`: `README.md` (`registered: false`), `runs.json` (per seed:
   wall, full/half metrics, `f_layout`, `f_event`, effective rank), `readings.json`
   (H1-clean-r verdict), `reproduction.json`, `plan.json` (manipulation check); each ≤ 1 MB.
2. `<ext>/`: checkpoints, `variants.json`, per-run `variant.json`, caches (deleted after
   `report`), hashes in `<ext>/large_arrays.json`, `manifest.jsonl` (written by `run_one`).
3. One `DECISIONS.md` line per engineering choice; a short `STATE.md` entry; book resources;
   do not touch `PREDICTIONS.md`/`PREDICTION_STATUS.json`.
4. Commit order: runbook + code + tests first (and the DECISIONS line), then `dry-run`/smoke/
   runs, then results; report the final SHA.
5. Report to the author in Chinese (≤ 1 page): the H1-clean-r verdict with numbers, the
   manipulation check, `f_layout`/`f_event`, GPU/CPU hours and the ledger, the final SHA.
6. Do not resume the grid; a follow-up (label-centering variant, third seed) is a separate
   runbook.
