# RUNBOOK — readout-matched evaluation and nonlinear information probes of the LUCiD JEPA checkpoints (1 Nov 2026, reviewed)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the design
agent, for the author. Branch `JEPA`, at or after `28ff553`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-01_READOUT.md` before any run.
- Everything needed is in this file and in the repository; it depends on no outside document.
- "§3.5.5", "§3.5.6" and "§3.5.8" refer to `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`.
- An independent review checked this file against the code at `28ff553`; all blocking findings
  are fixed here. This is an evaluation-only runbook: no encoder training._

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 1 Nov 2026:
   > "我们也要做Q1+Q2 先做（0 训练，1–2 天，决定论文怎么写）→ Q3（新故事，~1 天 + 几小时 GPU"
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Approved, and only this:**
- forward passes of the frozen checkpoints listed in §3.3 on the existing development probe sets;
- ridge and small-MLP **probe fits** on the resulting embeddings (analysis, not model training);
- the synthetic CPU training inside tests;
- regenerating the two development probe sets **only if** their cached copies are missing (§3.4).

**No encoder is trained. No grid run. No development-cohort run. No new event generation except
the two probe sets in §3.4.** If a checkpoint is missing after the §3.3 fallbacks, report and
evaluate the available subset; do not retrain.

**Status.**
- Every result is an extension: `registered: false`, extension name `readout_diagnostic_2026-11-01`.
- Results go outside the registered result directories and never replace a verdict, a prediction
  status or a frozen decision.
- **Never overwrite** `results/jepa/20261029_*`, `20261030_*`, `20261031_*` or any earlier
  `/ceph/dwong/jepa/returns/ext_*` root.
- The LUCiD grid was descoped on 31 Oct 2026 (gate G-DESCOPE, option A). Nothing here touches it.
- Run the full suite once per deployed SHA (the suite is not an experiment).

### 0.1 Fix it yourself, do not stop to ask

Small fixes and ambiguities are yours: a bug in your own code, a failing test you can fix, a
blocked tool/path/route, an environment problem, a data-format mismatch, an unclear scope, a step
slower or larger than estimated, a guard HOLD. For each: choose the best conservative solution,
write one line in `DECISIONS.md` ("author standing approval, 1 Nov 2026"), continue, report once
at the end.

**The chosen option must meet all four conditions:**
1. **Writes** go only inside `/ceph/dwong/jepa/` and only to files this extension created.
2. It changes no registered prediction, criterion, cell, dataset, frozen decision or verdict, and
   does not edit registered code or frozen configs. New files are
   `scripts/jepa_ext_readout.py`, `src/jepa_noise/ext_readout.py`,
   `src/jepa_noise/tests/test_ext_readout.py`; they may import the existing modules read-only.
3. It stays within the §5 budget.
4. It writes nothing outside `/ceph/dwong/jepa/` (including `/tmp`) and touches nothing belonging
   to another user.

**Stop and report only for:**
- a real S3 change;
- an action that fails conditions 1–4;
- **S4**: no route to GitHub (try the Mac clone push, a Deepthought clone with credentials, then
  authenticated `gh`; an integration 403 is not S4);
- **S9**: a §4 acceptance/reproduction check fails after one fix attempt;
- **S1**: the L40S ledger plus projections would exceed 160 GPU-h.

## 1. Why this run

`results/jepa/20261029_part_c/METRICS.md` gives, for the same frozen checkpoints, two readouts:

| checkpoint | full-input vertex (m) | full-input ln q R² | half-pass vertex (m) | half-pass ln q R² |
|---|---:|---:|---:|---:|
| init9000 (rebuilt, not a checkpoint) | 0.160072 | 0.998318 | 0.174821 | 0.998395 |
| old9000 | 0.264401 | 0.00401509 | 0.140020 | 0.647803 |
| old9001 | 0.233759 | 0.00352886 | 0.130648 | 0.772495 |
| fix9000 | 0.219258 | 0.0175646 | 0.325457 | 0.000445807 |
| fix9001 | 0.204281 | 0.0542048 | 0.293429 | 0.00165443 |
| mae9000 | 0.104949 | 0.987893 | 0.0832159 | 0.983219 |

The paper's deep-model negative result therefore has to be stated per readout, and the 23 Oct
label fix is a covariate rather than a neutral background. Two questions decide how to write it:
- **Q1** — does "trained JEPA is worse than random init" hold under every readout form, or only
  under full input?
- **Q2** — is the lost charge/vertex information destroyed, or present but nonlinearly encoded?

Two cheap descriptive signals for the mechanism hypotheses (H1: deterministic-geometry shortcut;
H2: trivial solution of a trainable target) are also recorded here: the fraction of the
representation explained by the fixed PMT layout, and its event-varying fraction (§6, Q1-e/f).
The causal tests for H1/H2 are a separate runbook (S/Q3); this one only measures.

## 2. Readout forms and metrics

For each item (§3.3) and each form below, record the metrics in the table after it.

| form | definition |
|---|---|
| `full_input` | encoder on the complete event: `arm.jepa.encode(batch.features)` |
| `half_pass` | `HalfPassEncoder(arm.jepa, within_target_mask(geometry))` (signature `HalfPassEncoder(base, target_mask)`, `scripts/jepa_pregrid_checks.py:61-86`): context-half tokens from the context-only pass, target-half tokens from the target-only pass; `within` only; `across` is per-event random, so record `n/a` |
| `propagated` | encoder on `hidden_mask=batch.hidden` (the standard view mask); read the **masked/target positions**, i.e. what the context carries about the hidden view |
| `context_only` | the same pass as `propagated`; read the **context positions** only |
| `raw_input` | the raw token features (no encoder): event-level mean of the 9 features; the registered raw-input ceiling |

Metrics, per form:
- **vertex resolution (m)**: event-level representation = mean over the form's tokens; fit
  `la.fit_probe` on probe-train and predict with `la.probe_predictions`; score with
  `lr._vertex_error` (exactly the registered physics probe, `src/jepa_noise/lucid_run.py:1187-1200`).
  `raw_input` uses the mean raw feature vector.
- **token ln q R²**: `ridge_r2` (`scripts/jepa_pregrid_checks.py:137-158`) on
  - `full_input`: all hit tokens;
  - `half_pass`: hit tokens from the pass each token saw;
  - `propagated`: target-half hit tokens, masked-position outputs;
  - `context_only`: context-half hit tokens only;
  - `raw_input`: `n/a`.
  Rows: at most 4 000 probe-test events, at most 32 hit tokens per event, sampling seed 0, exactly
  as `event_level` does (`scripts/jepa_pregrid_checks.py:161-193`); report `n_train`, `n_test`,
  `alpha`.
- **effective rank / participation ratio / event SIGReg**: `effective_rank` and `epps_pulley`
  (`:114-133`) on the event-level representation of the form.
- **module gain**: `module_gain::run_mean` R² from the existing `evaluate_arm_run` payload for
  `full_input` and `half_pass`; `module_gain::propagated` R² for `propagated`; `n/a` for
  `context_only` and `raw_input`. Do **not** invent a per-form run-mean: run-mean pooling is not
  form-specific in the existing code.

**Q2 probes** (on `full_input`, and on `half_pass` for the within items):
- MLP probe, 2 hidden layers × 256, early stopping on 10 % of probe-train, fixed seed, on the
  **identical rows** used by the ridge fit: `.x`/`.y` from the same tables;
  (a) vertex from the event-level representation, (b) ln q from the sampled hit-token outputs.
  Reuse the ridge alpha grid `(1e-4, 1e-2, 1, 100)` of `ridge_r2` for the ridge baseline, and
  `lr._fit_mlp`-style settings (2 × 256, patience 10) for the MLP; report both on the same rows.
- CKA: `lr.linear_cka(x, y)` (`src/jepa_noise/lucid_run.py:1800-1804`; the only CKA in the repo)
  between event-level representations (mean-pooled, `full_input`): (i) trained vs `init9000`,
  (ii) trained vs the raw physics summary `[vertex x, y, z, log yield]` per event,
  (iii) `init9000` vs the raw physics summary. Two distinct objects for the trained row; name
  each representation in the output.

**Protocol note.** The `token ln q` and vertex numbers above are the Part-C-comparable protocol
(`ridge_r2` / `lr._vertex_resolution`, no run-grouped folds). The registered-fold numbers
(`evaluate_arm_run`) are reported beside them from the same pass, but are not mixed into the Q2
comparisons.

## 3. Implementation, checkpoints, data

### 3.1 New files, read-only imports

- `scripts/jepa_ext_readout.py` — modes `dry-run`, `matrix`, `nonlinear`, `report`.
- `src/jepa_noise/ext_readout.py` — event-level form representations, token sampling, MLP probe,
  CKA wrapper, reproduction block.
- `src/jepa_noise/tests/test_ext_readout.py` — §3.7 tests.

All existing modules are imported read-only. The registered config and the checkpoint files are
never written; checkpoints are copied into `<ext>/ckpt/` before use.

### 3.2 Paths

- `<ext>` = `/ceph/dwong/jepa/returns/ext_readout_20261101/`.
- Results = `results/jepa/20261101_readout/` (written by `report` only).
- Code: a fresh shallow clone at the deployed SHA under `/ceph/dwong/jepa/code/<sha>/` (§6.1 of v4), never a dirty tree.
- `TMPDIR=/ceph/dwong/jepa/tmp/`; pytest `--basetemp` inside `<ext>/tmp/`, `-p no:cacheprovider`,
  removed after success.

### 3.3 Checkpoint manifest (read-only)

Copy each verified checkpoint into `<ext>/ckpt/<label>.pt` and hash the copy; record both hashes.

| label | arm | beta | source path | expected sha256 |
|---|---|---|---|---|
| `old9000` | `sigreg_jepa` | — | `/work/dwong/jepa_a10_work/dev_step2/runs/dev_s005_within__sigreg_jepa__s9000/final.pt` | `b1632266f2f8ca9fd49f9a06383734fda77432f5247294e79ed65bc1936f7503` |
| `old9001` | `sigreg_jepa` | — | `/work/dwong/jepa_a10_work/old_s9001/runs/dev_s005_within__sigreg_jepa__s9001/final.pt` | `14e3590f9ed6649d86f3de751a56c3590bb748f0b4a15984bb056373bff97ceb` |
| `fix9000` | `sigreg_jepa` | — | `/work/dwong/jepa_a10_work/fixed_s9000/runs/dev_s005_within__sigreg_jepa__s9000/final.pt` | `2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714` |
| `fix9001` | `sigreg_jepa` | — | `/work/dwong/jepa_a10_work/fixed_s9001/runs/dev_s005_within__sigreg_jepa__s9001/final.pt` | `95cb5e3f4f158a719098a4b3dc573eebb8c71645c22290f6483223ccc04c484e` |
| `mae9000` | `mae_mse` | — | `/work/dwong/jepa_a10_work/dev_step2/runs/dev_s005_within__mae_mse__s9000/final.pt` | `efd54c2586f8b5378dfe2febffe2ceb954c50185f6b07939c2b892c825270664` |
| `across9000` | `sigreg_jepa` | — | `/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/across_s9000/runs/ext_dev_s005_across__sigreg_jepa__s9000/final.pt` | verify by its sibling `ext_summary.json` sha256 `708dc0092018fed09bdd29c7cf578e3586bd2277cafdacbbee005a88ad5c13cd` in `results/jepa/20261029_d2_partial/SHA256SUMS`, then hash the checkpoint |
| `across9001` | `sigreg_jepa` | — | `/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/across_s9001/runs/ext_dev_s005_across__sigreg_jepa__s9001/final.pt` | sibling `ext_summary.json` sha256 `e72809b6dc39b7c6cf6a84f8222afbf627a3a56bc656f3e144494ea7f8985aaa` in the same `SHA256SUMS`, then hash the checkpoint |
| `init9000` | `sigreg_jepa` | — | rebuilt with `lr.build_arm("sigreg_jepa", ..., seed=9000)` | no hash (rebuilt); §4 checks it |

The `old9000/old9001` hashes are the recorded ones (`docs/jepa/DECISIONS.md:2146-2147`); the
`fix/mae` hashes are from `results/jepa/20261029_predictability/probes/*.json`. If a `/work`
path is gone, use the `/ceph` copy in the same run family
(`returns/ext_pregrid_20261025/train/<label>_s<seed>/runs/.../final.pt`) and verify it by its
sibling `ext_summary.json` sha256 against `results/jepa/20261029_d2_partial/SHA256SUMS` (where
present) or by reproducing the §4 full-input vertex. If fewer than **four** of the six priority
items (`old9000/1`, `fix9000/1`, `mae9000`, `init9000`) are available, stop and report.

### 3.4 Data

Read-only: `--data-root /ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`,
`--config src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (sha256 `e6730ac1…`),
`--generator-config src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` (sha256 `90ac1601…`),
`--geometry /ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`.

Only the two `dev_probe_train` / `dev_probe_test` sets at `F1` s=0.05 are needed
(10 000 events each). **Do not build `dev_train`**: build the probe sets directly, never through
`jepa_pregrid_checks.datasets_for` (it also builds the 100 000-event training cohort).

The data root is a link farm into `/work/dwong/jepa_a10_work/data_dev`. Preflight: verify both
probe-set directories against `results/jepa/20261025_pregrid_checks/provenance.json`
(`dev_probe_train` sha256 `093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3`,
`dev_probe_test` sha256 `c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a`).
If a set is dangling or differs, regenerate it under the same root (deterministic: pinned
generator config and `dataset_seed`), require the regenerated directory sha256 to equal the
provenance value, record the CPU charge and new entries, and continue. If it does not match after
one attempt, stop (S9).

### 3.5 Commands (one job per item group)

```
python scripts/jepa_ext_readout.py dry-run --out <ext>/plan.json \
  --manifest <ext>/items.json --data-root <DATA> --config <CFG> \
  --generator-config <GEN> --geometry <GEOM>
python scripts/jepa_ext_readout.py matrix --items init9000,old9000,old9001,fix9000 \
  --out <ext>/matrix_a --forms full_input,half_pass,propagated,context_only,raw_input \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM>
python scripts/jepa_ext_readout.py matrix --items fix9001,mae9000,across9000,across9001 \
  --out <ext>/matrix_b --forms full_input,propagated,context_only,raw_input \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM>
python scripts/jepa_ext_readout.py nonlinear --matrix <ext>/matrix_a --matrix <ext>/matrix_b \
  --out <ext>/nonlinear --items init9000,old9000,old9001,fix9000,fix9001,mae9000
python scripts/jepa_ext_readout.py report --matrix <ext>/matrix_a --matrix <ext>/matrix_b \
  --nonlinear <ext>/nonlinear --out results/jepa/20261101_readout
```

`dry-run` validates the manifest, the checkpoint hashes, the probe-set preflight and the planned
items × forms; it writes no embeddings and starts no run. `matrix` and `nonlinear` write under
`<ext>` only; `report` writes only to `results/jepa/20261101_readout/`. Launch through the guard
front door (`scripts/jepa_run.sh <run_id> --host deepthought2 --duration-minutes 90 -- <args>`),
at most two jobs, one item group per job (the two `matrix` calls may run in parallel). Book each
job in `RESOURCE_LEDGER.json` (`v4_caps.E2_lucid_l40s_deepthought2`).

### 3.6 Code specification

In `src/jepa_noise/ext_readout.py`:
- `form_representations(arm, batch, form)` → `(event_repr, token_repr, token_mask, y_vertex)`
  per §2: `full_input` = `arm.jepa.encode(features)`; `half_pass` =
  `HalfPassEncoder(arm.jepa, within_target_mask(geometry))`
  (`scripts/jepa_pregrid_checks.py:61-81`); `propagated` = `encode(features, hidden_mask=batch.hidden)`
  with the masked-position mask `batch.hidden`; `context_only` = the same pass with the context
  positions; `raw_input` = `batch.features`. Event-level = mean over the form's tokens; token
  ln q rows = the §2 sampling (seed 0, 4 000 events, 32 tokens/event).
- `mlp_probe(x_train, y_train, x_test, y_test, seed)` → 2 × 256, patience 10, early stopping on
  10 % of train, standardised inputs; return `{r2|resolution, n_train, n_test, seed}`.
- `cka(x, y)` → `lr.linear_cka(x, y)`.
- `layout_and_event_fractions(event_repr, token_repr)` → the two descriptive fractions of §6
  (per-dimension `Var_p(m_p)/Var_{p,e}(z)` and `Var_e(m_e)/Var_{p,e}(z)` on the `full_input`
  token outputs).
- `reproduction_block(...)` → the §4.2 checks.

The `matrix` mode builds the probe tables once per (item, form), fits the ridge probes, records
the metrics, and writes `matrix.json` with `{items: {label: {form: {metric: value|"n/a", reason}}}}`.
The `nonlinear` mode reads the cached event/token representations (written by `matrix`), fits the
ridge and MLP probes on identical rows, and writes `nonlinear.json`. `report` copies both into
`results/jepa/20261101_readout/` with a README marked `registered: false`, ≤ 1 MB each, and writes
`reproduction.json`.

### 3.7 Tests to add

1. `form_representations` on a synthetic batch: `full_input`/`raw_input` shapes, `half_pass`
   context/target provenance (context tokens equal the context-only pass output), `propagated`
   and `context_only` complementary (their union is the context-pass output).
2. `mlp_probe` recovers a known nonlinear function on synthetic data (R² above a fixed floor) and
   is deterministic for a fixed seed.
3. `cka` returns 1.0 for `x` vs `x` and 0.0 for independent columns (fixed seed), and matches
   `lr.linear_cka` exactly (it is that function).
4. `matrix` refuses an item whose checkpoint sha256 differs from the manifest and a form unknown
   to the cell; `dry-run` writes no output outside `--out`.

## 4. Acceptance and reproduction (S9)

1. Full suite passes once on the deployed SHA.
2. Each `matrix` job writes `reproduction_<group>.json` for the items it holds and exits 2 on a
   mismatch (`reproduction_pass: false`); `report` merges them into `reproduction.json`. Full-input
   vertex resolution for `old9000`, `old9001`, `fix9000`, `fix9001`, `mae9000`, `init9000` equals
   the recorded Part C value within **1e-3 m** (0.264401, 0.233759, 0.219258, 0.204281, 0.104949,
   0.160072); any failure after one fix attempt is S9.
3. `matrix.json` and `nonlinear.json` are valid JSON, each ≤ 1 MB; every requested item × form is
   present with a value or an explicit `n/a` plus a reason; `plan.json` lists exactly the items ×
   forms of §3.5.
4. No file outside `/ceph/dwong/jepa/` is written; `git status` shows only this extension's files
   (the pre-existing untracked `.pytest-tmp-predictability-full/` is ignored).

## 5. Budget

| Resource | Cap | Notes |
|---|---|---|
| L40S | ≤ 4 GPU-h, ≤ 2 jobs, 90 min per job | two `matrix` jobs, ≤ 4 items each; Part C's measured rate is 1.1776 GPU-h for 6 items × 2 forms, so ≤ 2.1 h here, plus probe fitting |
| Titan | 0 | not needed |
| CPU | ≤ 3 wall-hours (Deepthought, `nice -n 10`, `OMP_NUM_THREADS=4`) | MLP probes, CKA, report; probe-set regeneration if needed (≤ 0.2 h) |
| Storage | ≤ 2 GB and ≤ 1 000 entries under `<ext>`; nothing below 30 000 free entries | 8 checkpoints × 7.5 MB; one embedding cache per item × form (≤ 200 MB) |

Book every job in `RESOURCE_LEDGER.json` under `v4_caps.E2_lucid_l40s_deepthought2` (L40S) and
`lucid_generation_cpu` (probe-set regeneration). The ledger stands at 97.165 / 170 charged; the
S1 start-rule projection includes the in-flight MAE extension and must be re-checked before the
first job.

## 6. Pre-declared readings (descriptive, not gates)

Definitions: the **seed spread** of a quantity is `max − min` over the matched seed pair
(9000/9001). A comparison "worse by more than the spread" means the difference of the means
exceeds the larger of the two quantities' seed spreads. `unclear` rules are stated per row.

| ID | Reading | Supported if |
|---|---|---|
| Q1-a | The trained-vs-random verdict is readout-dependent | for `fix9000` and `fix9001`: full-input vertex > 1.1 × full-input init floor **and** half-pass vertex ≤ 1.1 × half-pass init floor; `unclear` if the two seeds disagree |
| Q1-b | The half-pass difference between the label versions is not seed noise | for each seed pair, half-pass vertex(fix) − half-pass vertex(old) > the seed spread of that pair, both pairs; `unclear` if one pair fails by less than the spread |
| Q1-c | `propagated` carries the same information as `half_pass` | for every pair among {old, fix, mae} (within items), the sign of the vertex difference is the same in both forms and the mean difference exceeds 2 × the seed spread |
| Q1-e | Deterministic-layout signal (H1) | descriptive: per-dimension average of `Var_p(m_p) / Var_{p,e}(z)`, where `z_{p,e}` is a token output, `m_p` its mean over events; high for `fix*` vs `init` is the H1 signature |
| Q1-f | Event-dependence signal (H2) | descriptive: per-dimension average of `Var_e(m_e) / Var_{p,e}(z)`, where `m_e` is the mean over PMTs (the event embedding); near zero for `fix*` is the H2 signature; with Q1-e it separates a layout-only representation (high Q1-e, low Q1-f) from a constant/degenerate one (both low) |
| Q2-a | Charge is destroyed, not just unreadable | MLP token ln q R²(`fix9000`, full_input) ≤ 0.1 **and** the ridge value ≤ 0.05 **and** the init ridge ≥ 0.9 |
| Q2-b | Charge is present but nonlinearly encoded | MLP token ln q R²(`fix9000`, full_input) ≥ 0.5 **and** the ridge value ≤ 0.1; `unclear` if the MLP R² is between 0.1 and 0.5 |
| Q2-c | Vertex information is destroyed | MLP vertex resolution(`fix9000`) > 1.1 × the init MLP vertex resolution |
| Q2-d | The trained representation is closer to physics than to a random projection | `CKA(trained, raw physics) − CKA(trained, init) > 0.1`; `unclear` within ±0.1; else not supported |

Report every row for both seeds, with the numbers, and say which readout each number belongs to.

## 7. Outputs, commit, report

1. `results/jepa/20261101_readout/`: `README.md` (`registered: false`), `matrix.json`,
   `nonlinear.json`, `reproduction.json`, `PREDICTIONS_UPDATE` not needed (no verdict).
2. `<ext>/`: checkpoints, embedding caches (temporary; delete the per-form caches after `report`),
   `plan.json`, hashes in `<ext>/large_arrays.json`.
3. One `DECISIONS.md` line per engineering choice; a short `STATE.md` entry; book the resources.
4. Commit and push after the reproduction check and at the end.
5. **Report to the author (≤ 1 page, in Chinese):** Q1-a/b/c/e/f and Q2-a/b/c/d with numbers, the
   readout matrix, the checkpoint hashes, GPU/CPU hours and the ledger, the final SHA.
6. Do not resume the grid (descoped). The causal H1/H2 tests (geometry ablation, frozen target)
   are a separate runbook and need a separate author line.
