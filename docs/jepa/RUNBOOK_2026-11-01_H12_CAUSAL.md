# RUNBOOK — H1/H2 causal tests: geometry feature ablation and frozen target (1 Nov 2026, reviewed)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the design
agent, for the author. Branch `JEPA`, at or after the Q1/Q2 extension's final SHA. Commit this
file unchanged as `docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md` before any run.
- Everything needed is in this file and the repository; it depends on no outside document.
- "§3.5.5"/"§3.5.6" refer to `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`.
- It reuses the Q1/Q2 extension (`docs/jepa/RUNBOOK_2026-11-01_READOUT.md`,
  `scripts/jepa_ext_readout.py`, `<ext_readout>` = `/ceph/dwong/jepa/returns/ext_readout_20261101/`).
  **Preflight: if the Q1/Q2 extension is not committed at the deployed SHA, stop (dependency).**
- An independent review checked this file against `28ff553`; all blocking findings are fixed here._

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 1 Nov 2026:
   > "H1/H2 的因果检验 我授权"
   and the decision order and conjunction the author fixed:
   > "E2（训练，2 seeds，~2×0.5 h）：几何消融…… H1 预测塌缩消失；H2 预测仍塌。"
   > "E3（训练，2 seeds，~2×0.5 h）：frozen target…… H2 预测塌缩消失；H1 预测帮助有限"
   > "合取：如果 E2 恢复、E3 也恢复 → H1+H2 都在起作用（最可能）；只 E2 恢复 → H1 主导；只 E3 恢复 → H2 主导；都不恢复 → 转 H4（architecture/masking）或 H5（scale）。"
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Training approved, and only this:**
- E2 (`sigreg_jepa_geomblind`): seeds 9000 and 9001;
- E3 (`sigreg_jepa_frozen_target`): seeds 9000 and 9001;
- one 20-step smoke per patched variant;
- the synthetic CPU training inside tests.

**No other training. No new data. No grid.** Phase 0 (E0/E1) is evaluation only; it reads the
Q1/Q2 cached representations where they exist (forward passes otherwise, ≤ 0.5 GPU-h).

**Status.**
- Extension: `registered: false`, name `h12_causal_2026-11-01`.
- `<ext>` = `/ceph/dwong/jepa/returns/ext_h12_20261101/`; results = `results/jepa/20261101_h12/`.
- Never overwrite an earlier `results/jepa/2026*` or `/ceph/dwong/jepa/returns/ext_*` root.
- The LUCiD grid is descoped; nothing here touches it. If the descope record is not yet committed,
  it does not change this runbook.
- Run the full suite once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Small fixes and ambiguities are yours: a bug in your own code, a failing test you can fix, a
blocked tool/path/route, an environment problem, a data-format mismatch, an unclear scope, a step
slower or larger than estimated, a guard HOLD. For each: choose the best conservative solution,
write one line in `DECISIONS.md` ("author standing approval, 1 Nov 2026"), continue, report once.

**The chosen option must meet all four conditions:**
1. Writes go only inside `/ceph/dwong/jepa/` and only to files this extension created.
2. It changes no registered prediction, criterion, cell, dataset, frozen decision or verdict, and
   does not edit registered code or frozen configs. The E2/E3 patches are **runtime** hooks
   installed inside the extension process (precedent: `patch_stopgrad`,
   `scripts/jepa_pregrid_checks.py:90-102`); they are not edits to registered code.
3. It stays within the §5 budget.
4. It writes nothing outside `/ceph/dwong/jepa/` (including `/tmp`) and touches nothing belonging
   to another user. The dev data root is a link farm into `/work`; see §3.4.

**Stop and report only for:** a real S3 change; an action failing conditions 1–4; **S4** (no
GitHub route after the three documented attempts; an integration 403 is not S4); **S9** (a §4
acceptance/reproduction check fails after one fix attempt, including a patched variant that cannot
be built or whose smoke fails twice); **S1** (L40S ledger + projections > 160 GPU-h).

## 1. Why this run

Q1/Q2 measure *what* the trained JEPA represents. This runbook tests *why* it collapsed:

| | H1: deterministic-geometry shortcut | H2: trivial solution of a trainable target |
|---|---|---|
| mechanism | the target label is dominated by fixed PMT geometry; a predictor with position queries fits it without physics | the label comes from the same trainable encoder, so the encoder can make it event-independent |
| causal test | E2: zero the geometry features → collapse should disappear | E3: frozen random target → collapse should disappear |

Phase 0 gives the cheap prior from the `fix9000` representation and label; Phase 1 is the causal
test. The decision table is §6.

## 2. Phase 0 — E0/E1 (evaluation only)

Inputs: the `fix9000` checkpoint (and the rebuilt `init9000`) from the Q1/Q2 manifest
(`<ext_readout>/items.json`); the Q1/Q2 embedding caches when present, forward passes otherwise.
Row sampling (fixed, seed 0): at most 2 000 probe-test events, at most 32 hidden (target-half) hit
PMTs per event for the label rows; token-level quantities use the Q1/Q2 sampling (4 000 events,
32 hit tokens/event, seed 0). Do **not** materialise all 1 995 tokens per event; accumulate
sufficient statistics in event batches. Caps: ≤ 2 GB in `<ext>`, ≤ 4 CPU-h.

- **E0-a `f_layout`**: on the cached `fix9000` `full_input` token outputs `z_{p,e}`, per-dimension
  average of `Var_p(m_p) / Var_{p,e}(z)` with `m_p = mean_e z_{p,e}`. If the denominator is 0 in a
  dimension, drop that dimension and report `n_dropped` (a fully constant representation reports
  `n/a`).
- **E0-b `f_event`**: same rows, per-dimension average of `Var_e(m_e) / Var_{p,e}(z)` with
  `m_e = mean_p z_{p,e}`. `f_layout` and `f_event` are not additive; report both.
- **E0-c position-only label ridge**: rows = the sampled label rows `(e, p)`; `x` = the scaled
  PMT position (3 features, from `geometry_features`), `y` = the target-label vector at that
  hidden PMT (`arm.jepa.encode(features, hidden_mask=~batch.target)[e, p]`, 128-d). Fit a
  multi-output ridge on 80 % of the **events** (split by event, so no event leaks across the
  split), score on the other 20 %; report `R² = 1 − mean_dim(SSE/SST)` and the row count. This
  asks how much of the label is a function of position alone.
- **E1 label decomposition**: on the same rows, group by PMT; `m_p` = mean label over the rows of
  PMT `p`; report (i) invariant fraction `Var_p(m_p)/Var(rows)`, (ii) position-explained fraction
  of `m_p` (multi-output ridge from position, per-PMT), (iii) total label variance and its ratio
  to the `init9000` label variance on the same rows.

Readings (descriptive priors, they do not override §6):
- E0-1 (H1 prior): `f_layout ≥ 0.5`, `f_event ≤ 0.5 f_layout`, E0-c R² ≥ 0.5.
- E0-2 (H2 prior): `f_layout ≤ 0.2`, `f_event ≤ 0.1`.
- E1-1 (H1 prior): invariant fraction ≥ 0.5 and position-explained ≥ 0.8.
- E1-2 (H2 prior): label-variance ratio ≤ 0.1 and position-explained ≤ 0.2.
- Otherwise `unclear`; report the numbers and continue.

**Correction to the shorthand in the author's rule:** "cross-event variance ≈ 0" alone does not
separate H2, because a layout-only representation is also event-invariant. The pair (`f_layout`,
`f_event`) is the discriminator: high layout + low event → H1; low layout + low event → H2.

## 3. Phase 1 — E2 and E3 (the approved training)

### 3.1 Base and variants

Base: development cell `dev_s005_within` (s = 0.05, within, mixed 1:1:1 domains), frozen config,
10 000 steps, batch 32, fp32/TF32 off, seeds 9000 and 9001 — the recipe of `fix9000/9001`.

**E2 `sigreg_jepa_geomblind`.** The six geometry feature columns (`pos_x, pos_y, pos_z, dir_x,
dir_y, dir_z`; `FEATURES[0:6]`, `src/jepa_noise/point_encoder.py:46-47`) are set to zero in a
**clone** of the `features` tensor inside a wrapper installed on `arm.jepa.encode`, so every
encoder pass — training and evaluation — sees zero geometry. This is a **feature ablation**, not a
full geometry removal: the Hilbert token order, the within-target parity mask and the predictor's
position queries remain geometry-defined; the verdict for E2 is therefore "consistent with H1",
not a proof.

**E3 `sigreg_jepa_frozen_target`.** The prediction label comes from a **frozen random** second
encoder: a second `LucidPointJepa` with the same `EncoderConfig`/scaling, built under
`torch.manual_seed(0)` (the class has no internal seeding; save and restore the RNG state around
it), `eval()`, `requires_grad_(False)`. Hold it with `object.__setattr__(arm,
"_ext_frozen_target", frozen)` so it is **not** registered as a submodule; `arm.state_dict()`
must be identical to a plain `sigreg_jepa` arm's (test §4.4). The label is its target-only
encoding `hidden_mask=~batch.target`, gathered at `hidden_index` (the current definition,
`src/jepa_noise/lucid_arms.py:405-419`). `arm._encode_views` is rebound on the built arm:
`context`/`prediction` from `arm.jepa`, `label_full` from the frozen encoder, detached.
**The frozen label feeds both loss terms** — the prediction MSE and the SIGReg target term
(`_forward_sigreg`, `lucid_arms.py:464-472`); the context-side SIGReg term still trains. State
this in the report: E3 also changes the SIGReg target term, not only the MSE.

### 3.2 Build and launch

Both variants call `lr.run_one(...)` with `arm_name="sigreg_jepa"`; the extension installs a
process-level wrapper on `jepa_noise.lucid_run.build_arm` (module global used at
`src/jepa_noise/lucid_run.py:801`), builds the registered arm, applies the E2 or E3 rebind, and
removes the wrapper afterwards. `run_one` itself is unchanged. The patched arm keeps the
registered `arm_name`, so both variants get the same `run_id`
(`dev_s005_within__sigreg_jepa__s9000|9001`); **each variant uses its own output root**
(`<ext>/train/geomblind/`, `<ext>/train/frozen_target/`) or the second variant would auto-resume
the first one's checkpoint (`run_one` resumes when `checkpoint.pt` exists,
`lucid_run.py:811-812`). Record the variant in `<ext>/variants.json` and in a per-run sidecar
`variant.json` next to `run_status.json` (the `run_one` manifest has no variant field; do not edit
it).

Exact call (all keywords; `evaluate=False`, readout evaluation is a separate step):

```
lr.run_one(arm_name="sigreg_jepa", cell=lr.dev_cell(config), seed=SEED, beta=None,
           config=config, arms_config=arms_config, geometry=geometry,
           loader_dataset=train_ds, probe_datasets=probe_ds, dark_bank=None,
           out_root=<ext>/train/<variant>, data_root=<DATA>, config_sha=config_sha,
           code_sha=code_sha, status_text=status_text, device=device, steps=None,
           micro_batch=None, resume=None, stop_after=None, checkpoint_minutes=None,
           evaluate=False, allow_pending=False, step_callback=None, probe_max_events=None,
           save_embeddings=False, physics_method="ridge", tf32=False, log=print)
```

`train_ds`/`probe_ds` come from the pregrid helper (`datasets_for`, which builds
`dev_train` + `dev_probe_train/dev_probe_test`); `config`/`geometry`/`arms_config` load exactly as
in `jepa_pregrid_checks.setup`.

### 3.3 Evaluation

After each run, evaluate its `final.pt` with the Q1/Q2 readout protocol (full_input and half_pass:
photon vertex resolution, token ln q R², effective rank, event SIGReg, and the gain rows), and
compare with the paired `fix9000/9001` rows and the `init9000` baseline. For **E2** runs the six
geometry columns are zeroed in the probe batches before the Q1/Q2 form functions are called
(`--zero-geometry`); otherwise the checkpoint is evaluated off-distribution. E3 evaluations are
unchanged. The "token ln q R²" used for the recovery threshold is exactly the Q1/Q2
`token_log_charge_r2` protocol (`ridge_r2`, same sampling and alpha grid), so the numbers are
comparable with the Q1/Q2 matrix.

**Smoke** per variant first: 20 steps, `--probe-max-events 200`, out `<ext>/smoke/<variant>/`,
no NaN, finite `ext_summary.json`. **Runs:** E2 9000, E2 9001, E3 9000, E3 9001, each ≤ 90 min,
≤ 2 jobs on deepthought2; guard front door:

```
JEPA_JOB_MODULE=scripts.jepa_ext_h12 JEPA_JOB_KIND=training \
  scripts/jepa_run.sh <run_id> --host deepthought2 --duration-minutes 90 -- \
  train --variant geomblind --seed 9000 --out /ceph/dwong/jepa/returns/ext_h12_20261101/train/geomblind
```

(same with `--variant frozen_target` and its own out root). `report`/`evaluate` jobs use
`JEPA_JOB_KIND=analysis` and `--cpu` where appropriate.

### 3.4 Data (read-only)

`--data-root /ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`,
`--config src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (sha256 `e6730ac1…`),
`--generator-config src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` (sha256 `90ac1601…`),
`--geometry /ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`.

Only the development sets are used. Their keys and recorded manifest sha256
(`results/jepa/20261025_pregrid_checks/provenance.json`, confirmed by
`results/jepa/20261026_pregrid_resume/data_hash_verification.json`):

| set | key | manifest sha256 |
|---|---|---|
| training | `dev_train__F1__0.05__s20261017__n100000x1000` (the `registered=False` key that `datasets_for` uses) | `b1b3f4f401bcee48c49c347b67d7f5cbd4410213f08ef8b9d2848edc9f2cd13a` |
| probe-train | `dev_probe_train__F1__0.05__s20261017` | `093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3` |
| probe-test | `dev_probe_test__F1__0.05__s20261017` | `c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a` |

The data root is a link farm into `/work`; `ensure_dataset` resolves it. **Do not regenerate.**
Preflight: `realpath` the root; if the three directories exist and their `manifest.json` sha256
equals the table (the original files, not regenerated), use them read-only. If a set is absent or
differs, **stop and report (data unavailable, S9-class)** — regenerating would write outside
`/ceph` through the link farm, and a regenerated manifest cannot match because it contains
`created_utc`. If the resolved root is outside `/ceph` and the data is not used for writing, that
is the only allowed exception; record it.

## 4. Acceptance and reproduction (S9)

1. Full suite passes once on the deployed SHA (the repository's documented suite, v4 §6.1:
   `nice -n 10 env OMP_NUM_THREADS=4 python -m pytest -q --basetemp=<ext>/tmp/pytest
   -p no:cacheprovider` from the clone root).
2. Reproduction: the Q1/Q2 readout code re-evaluates `fix9000` and gives full-input vertex
   `0.219258` ± 1e-3, and `init9000` gives `0.160072` ± 1e-3 (the rebuilt seed-9000 value; the
   named random-init floor is `0.160904`). The extension exits 2 on a mismatch.
3. E2 test: the wrapper's input clone is zero in columns 0–5 and the original tensor is unchanged
   (the wrapper never mutates its input); with the wrapper removed, the same batch gives the same
   output as before.
4. E3 test: after one synthetic CPU training step the frozen encoder's parameters are bit-identical
   and its gradients are `None`; `set(arm.state_dict())` equals a plain arm's; the frozen label
   enters both the prediction MSE and the SIGReg target term; the loss is finite.
5. Run-dir separation: the E2 and E3 run directories are different; a second launch of the same
   variant refuses an existing run dir (the extension adds that guard; `run_one` would auto-resume).
6. Each smoke/run writes a finite `ext_summary.json` with no NaN; the variant sidecar exists; no
   file outside `/ceph/dwong/jepa/` is written; `git status` shows only this extension's files
   (the pre-existing untracked `.pytest-tmp-predictability-full/` is ignored).

## 5. Budget

Current L40S ledger: **97.165 / 170** charged; S1 start rule 160. This runbook's projection
≈ 5 L40S GPU-h (4 runs at the measured 0.44–0.96 h + 2 smokes + 4 readout evaluations), so
97.165 + 5 ≈ 102 < 160. Cap **8 GPU-h**. Up to **2 jobs** on deepthought2, 90 min per run; CPU
≤ 4 h (`nice -n 10`, `OMP_NUM_THREADS=4`); Titan 0; storage ≤ 2 GB and ≤ 1 000 entries under
`<ext>`; free entries stay above 30 000. Book each job in `RESOURCE_LEDGER.json` under
`v4_caps.E2_lucid_l40s_deepthought2`.

`TMPDIR=/ceph/dwong/jepa/tmp/`; pytest `--basetemp` inside `<ext>/tmp/`, `-p no:cacheprovider`,
removed after success.

## 6. Pre-declared readings and the decision

Recovery, per variant, is three-valued over the two seeds, with `full_input` primary and
`half_pass` a required secondary check (the Q1/Q2 premise is that the verdict is readout-sensitive):

- **supported**: both seeds have `full_input` vertex ≤ `0.160072` **and** token ln q R² ≥ 0.5,
  **and** `half_pass` does not contradict (its vertex also beats the init `half_pass` value and its
  charge is ≥ 0.5 for both seeds).
- **partial**: one of the two criteria (vertex / charge) holds for both seeds, or the two seeds
  disagree.
- **not supported**: neither criterion holds for both seeds, or `half_pass` contradicts.

| ID | Reading | Value |
|---|---|---|
| E2-r | Geometry feature ablation | supported / partial / not supported |
| E3-r | Frozen random target | supported / partial / not supported |
| E0/E1 | priors | §2 readings, descriptive |

**Decision (the author's conjunction):**

| E2-r | E3-r | verdict |
|---|---|---|
| supported | supported | H1 + H2 both active (most likely) |
| supported | not supported | H1 dominant (deterministic-geometry shortcut) |
| not supported | supported | H2 dominant (trainable-target trivial solution) |
| not supported | not supported | neither H1 nor H2; move to H4 (architecture/masking) and H5 (scale) |
| any `partial` | any | `unclear`; report and design the next runbook (a third seed or the geometry-randomisation variant) — no extra training here |

## 7. Outputs, commit, report

1. `results/jepa/20261101_h12/`: `README.md` (`registered: false`), `e0e1.json`,
   `runs.json` (per run: variant, seed, wall, full_input/half_pass rows, comparison with
   fix/init), `decision.json` (§6 with the actual values), `reproduction.json`; each ≤ 1 MB.
2. `<ext>/`: checkpoints, `variants.json`, per-run `variant.json`, embedding caches (deleted after
   `report`), hashes in `<ext>/large_arrays.json`.
3. One `DECISIONS.md` line per engineering choice; a short `STATE.md` entry; book every resource;
   do not touch `PREDICTIONS.md`/`PREDICTION_STATUS.json` (no verdict changes).
4. Commit and push after the reproduction check and at the end.
5. **Report to the author (≤ 1 page, in Chinese):** E0/E1 priors, E2/E3 values, the §6 verdict,
   GPU/CPU hours and the ledger, the final SHA.
6. Do not resume the grid. The geometry-randomisation variant, a third seed, and the H4/H5 tests
   are separate runbooks.
