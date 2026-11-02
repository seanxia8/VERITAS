# RUNBOOK — target-content ablation: a frozen static (event-invariant) target (2 Nov 2026, reviewed, NOT YET AUTHORISED)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the design
agent, for the author. Branch `JEPA`, at or after `a348558`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md` before any run.
- Everything needed is in this file and the repository; it depends on no outside document.
- Reuses the committed Q1/Q2 and H1/H2 extensions (`scripts/jepa_ext_readout.py`,
  `scripts/jepa_ext_h12.py`); if they are absent at the deployed SHA, stop (dependency).
- An independent review checked this file against `a348558`; its blocking findings are fixed here.

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 2 Nov 2026:
   > "frozen static-target 变体训练 我授权"
   This authorises **exactly** two `sigreg_jepa_static_target` runs (seeds 9000, 9001) and **one**
   20-step smoke; no other training.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Hard gate (implemented in `src/jepa_noise/ext_static.py`, called from the library
training/smoke entry, not only the CLI).** `assert_authorised()` must:
1. read the committed runbook `docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md` (its own
   `RUNBOOK_REL`, not the H1-clean one) and require the author-line substring above present;
2. require the same line in `DECISIONS.md`;
3. exit 2 otherwise. `ext_static` defines its own `AUTHOR_LINE` constant; it must **not** reuse
   `ext_h1clean.assert_authorised` (which hard-codes the H1-clean runbook and would pass
   regardless). The negative test in §3.3 checks that the gate exits 2 when the line is absent.

**Status.** Extension `static_target_2026-11-02`, `registered: false`. `<ext>` =
`/ceph/dwong/jepa/returns/ext_static_20261102/`; results `results/jepa/20261102_static/`. Never
overwrite an earlier root. Grid descoped. Full suite once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Fix small problems yourself (bug, path, env, guard HOLD, non-finite loss) with one
`DECISIONS.md` line ("author standing approval, 2 Nov 2026") and continue; writes only inside
`/ceph/dwong/jepa/`, no registered changes, within §5, nothing outside `/ceph`. Stop only for
S3 (real spec change), S4 (no GitHub route after the three documented attempts), S9 (an
acceptance check fails after one fix attempt), S1 (L40S ledger + projection > 160 GPU-h), a
missing dataset after the §3.4 preflight, or a failed manipulation check (§2).

## 1. Why this run

Evidence at `a348558`: E3 (frozen **random** target, whose labels are event-dependent) removes
the collapse (full vertex 0.138/0.131 m, token charge 0.92/0.93); the fixed-label run `fix9000`
(target trainable) collapses (0.219/0.204 m; token charge 0.017/0.051 from
`results/jepa/20261101_readout/matrix.json`); the collapsed representation is dominated by a
event-invariant per-PMT part (E0 `f_layout` 0.997, `f_event` 7.89e-5 on the **full-input
tokens**; E1 label invariant fraction 1.007), and that per-PMT code is **not** explained by the
PMT's fixed attributes (combined 120-feature probe on held-out PMTs: MLP 0.434, linear 0.104,
against the `init9000` reference 0.89–0.91; `e0e1_code_origin.json`). So the trivial solution's
content is a *learned, event-invariant per-PMT code*.

This runbook asks the target-content question: **is an event-invariant (static) target alone
sufficient to prevent learning?** The target is a frozen random encoder fed **geometry-only**
features (columns 6–8 = ln q, time, hit flag zeroed in the target pass), so its target-only
label is a function of geometry only — exactly static across events for the `within` split with
natural occupancy (§2). The context encoder sees the full features; the predictor, losses and
hyper-parameters are unchanged.

- **not recovered**: a static target suffices to keep the model from learning the physics.
- **recovered**: a static frozen target does not prevent learning; the trainable target's
  ability to *change* the label (not mere staticity) is the key.

**Asymmetry, stated so the report does not over-read.** With a static (detached) label the MSE
optimum is a constant output, so the prediction loss gives the context encoder no event-specific
signal, and (because the label is detached) the SIGReg target term is a constant with zero
gradient — only the context half regularises. "Recovered" is therefore nearly unreachable and
would more likely signal a leak; "not recovered" cannot distinguish staticity from a mere
absence of target event-information. Report it as a two-sided descriptive contrast against E3
(which also detaches its label), not as a verdict about H2.

## 2. Variant and readings

**`sigreg_jepa_static_target`** (variant label only; `arm_name` stays `"sigreg_jepa"`). A second
`LucidPointJepa` with the same `EncoderConfig`/scaling is built under `torch.manual_seed(0)`
(save/restore the RNG state) — **the same seed-0 net as E3's frozen target**, so the only
difference from E3 is the zeroed event columns — `eval()`, `requires_grad_(False)`, held
**outside the arm's module tree** (`object.__setattr__`) so `set(arm.state_dict())` equals a
plain arm's. Freezing/device plumbing reuses `eh.attach_frozen_target` (`ext_h12.py:116-156`,
including its `_apply` override so the frozen net moves to the run device). The label is that
frozen encoder's target-only encoding `hidden_mask=~target` of a **clone** of `batch.features`
whose columns 6, 7, 8 are zeroed (the clone is never mutated; the context pass keeps the full
features); the label is detached and feeds **both** the prediction MSE and the SIGReg target
term (state the constant-target-term consequence from §1). Implementation by rebinding
`arm._encode_views` (the E3 precedent) through a process-level `build_arm` wrapper in
`scripts/jepa_ext_static.py`; `lr.run_one` is called with `arm_name="sigreg_jepa"`; the wrapper
is restored afterwards. Do **not** reuse `eh.build_arm_patch`/`eh.apply_variant` (they accept
only `geomblind`/`frozen_target` and would install E3's full-feature label).

**Split/occupancy pinned.** Staticity holds only for `within` + natural occupancy (the
development cell: `cells.development.split = within`, `lr.dev_cell`; natural is forced by
`lucid_arms.py:399-401`). The run must assert both; `across` (per-event target modules) or
balanced occupancy would silently break the static label → stop if the cell differs.

**Manipulation check (implementation guard, not a scientific gate).** On one fixed batch:
(i) assert the target-pass clone has columns 6–8 exactly zero and columns 0–5 equal to the
original; (ii) compute the label's event fraction with the committed estimators
(`er.layout_fractions`; handle its `None` when all per-token variances are zero) and assert
≤ 1e-3; (iii) compute the per-PMT variance with `er.layout_state` and assert > 0. For
within+natural the staticity is by construction, so this check only catches implementation
errors (wrong columns, `across`, balanced). `dry-run` exits 2 on failure.

**Runs:** seeds 9000/9001 + one 20-step smoke. Out root `<ext>/train/static_target/`, with
`variant.json`/`variants.json` sidecars; refuse an existing run dir (archive → record →
relaunch, never auto-resume) and assert `resumed_from_step == 0` (reuse
`ec.assert_fresh_resume`, `ext_h1clean.py:182-190`).

**Reading `static_target_r` (two-sided, both seeds), implemented in `ext_static` over
`eh.recover`'s rows with these pinned thresholds:**
- `recovered`: both seeds full-input vertex **strictly < 0.160072 m**, full token ln q R²
  ≥ 0.5, half-pass vertex < 0.174821 m, half-pass token R² ≥ 0.5.
- `not recovered`: both seeds full-input vertex ≥ 0.160072 m (an encoder that does not train
  stays at the floor) **or** full token ln q R² < 0.5.
- `unclear`: anything else (partial, seed disagreement, half-pass contradiction).
`eh.recover` uses `<=` and treats a vertex-pass/charge-fail pair as partial; do not reuse its
verdict, only its rows. Report the numbers and the direction; no H2 claim.

## 3. Implementation

### 3.1 Code and reuse

`scripts/jepa_ext_static.py` (modes `dry-run`, `smoke`, `train`, `evaluate`, `report`),
`src/jepa_noise/ext_static.py` (gate, frozen-static-target rebind, manipulation check, sidecars,
`static_target_r`), `src/jepa_noise/tests/test_ext_static.py`.

Reuse (read-only): `er.evaluate_item`, `er.layout_state`, `er.layout_add`, `er.layout_fractions`,
`eh.attach_frozen_target`, `eh.evaluate_run`, `eh.reproduction`, `eh.preflight_data`,
`eh.load_probes`, `eh.datasets_for`, `eh.train_log_check`, `ec.assert_fresh_resume`. Do **not**
reuse `eh.build_arm_patch`, `eh.apply_variant`, `eh.train_variant`, `eh.prepare_run_dir`,
`eh.summarize_run` (variant-locked) or `scripts/jepa_ext_h12.discover_runs`
(`scripts/jepa_ext_h12.py:102`, hard-codes `eh.VARIANTS`).

Exact `lr.run_one` call (mirrors `ext_h12.train_variant:230-239`):

```
lr.run_one(arm_name="sigreg_jepa", cell=lr.dev_cell(config), seed=SEED, beta=None,
           config=config, arms_config=arms_config, geometry=geometry,
           loader_dataset=train_ds, probe_datasets=probe_ds, dark_bank=None,
           out_root=<ext>/train/static_target, data_root=<DATA>,
           config_sha=sha256_file(config_path), code_sha=git_head(ROOT),
           status_text=git_status(ROOT), device=device, steps=None, micro_batch=None,
           resume=None, stop_after=None, checkpoint_minutes=None, evaluate=False,
           allow_pending=False, step_callback=None, probe_max_events=None,
           save_embeddings=False, physics_method="ridge", tf32=False, log=print)
```

### 3.2 CLI

```
python scripts/jepa_ext_static.py dry-run --out <ext>/plan.json --data-root <DATA> \
  --config <CFG> --generator-config <GEN> --geometry <GEOM> --device cpu
python scripts/jepa_ext_static.py smoke --seed 9000 --out <ext>/smoke --steps 20 \
  --probe-max-events 200 --data-root <DATA> --config <CFG> --generator-config <GEN> \
  --geometry <GEOM> --device cuda
python scripts/jepa_ext_static.py train --seed 9000 --out <ext>/train/static_target \
  --data-root <DATA> --config <CFG> --generator-config <GEN> --geometry <GEOM> --device cuda
python scripts/jepa_ext_static.py evaluate --run <ext>/train/static_target/runs/<run_id> \
  --out <ext>/eval --readout-ext /ceph/dwong/jepa/returns/ext_readout_20261101 \
  --readout-items /ceph/dwong/jepa/returns/ext_readout_20261101/items.json --device cuda
python scripts/jepa_ext_static.py report --out results/jepa/20261102_static \
  --plan <ext>/plan.json --eval <ext>/eval --reproduction <ext>/eval/reproduction.json
```

`<run_id>` = `dev_s005_within__sigreg_jepa__s9000|9001`. `<CFG>` =
`src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (sha256 `e6730ac1…`), `<GEN>` =
`src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` (sha256 `90ac1601…`), `<GEOM>` =
`/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`, `<DATA>` =
`/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`. `dry-run` runs the data preflight, the
split/occupancy assertion and the manipulation check, writes `plan.json`, and exits 2 on
failure; `evaluate` checks that the readout extension and `items.json` exist before it runs.
The full-suite command is the one in `RUNBOOK_2026-11-01_H12_CAUSAL.md:224`.

### 3.3 Data and tests

Data (read-only; no regeneration): the three development keys and manifest sha256 from
`RUNBOOK_2026-11-01_H12_CAUSAL.md` §3.4 (`dev_train__F1__0.05__s20261017__n100000x1000`
`b1b3f4…`, `dev_probe_train__…` `093ea894…`, `dev_probe_test__…` `c11edc71…`; confirmed in
`results/jepa/20261025_pregrid_checks/provenance.json`). Absent/different → stop.

Tests: (1) the authorisation gate exits 2 while the banner is present and passes once the line
is present in both the runbook §0 and `DECISIONS.md`; (2) the frozen encoder moves to the run
device (its `_apply` override) and its parameters are bit-identical to E3's seed-0 net; (3)
after one synthetic training step the frozen params are bit-identical and grads are None, and
`set(arm.state_dict())` equals a plain arm's; (4) the target-pass clone has columns 6–8 zero and
the original features unchanged, and the context pass still sees the full features; (5) the
manipulation check passes on a synthetic within+natural batch and fails on an `across`-like one;
(6) run-dir refusal + `resumed_from_step == 0`; (7) smoke-level training finite.

## 4. Acceptance and reproduction (S9)

1. Full suite once on the deployed SHA.
2. Reproduction: `fix9000` full-input vertex `0.219258` ± 1e-3 and `init9000` `0.160072` ± 1e-3
   (Q1/Q2 readout code; `decision.json` floors); exit 2 on mismatch.
3. Gate tests, device/isolation tests and the manipulation check pass; one smoke finite; both
   runs exit 0, 10 000 steps, no NaN, `resumed_from_step == 0`.
4. No file outside `/ceph/dwong/jepa/` is written on the host (the host clone is
   `/ceph/dwong/jepa/code/<sha>/`; the Mac clone is only for editing and pushing; only the git
   push is non-`/ceph`). `git status` shows only this extension's files (the pre-existing
   untracked `.pytest-tmp-predictability-full/` is ignored).

## 5. Budget

| Resource | Cap | Notes |
|---|---|---|
| L40S | ≤ 6 GPU-h; ≤ 2 **concurrent** jobs; 90 min per run | 2 runs (committed H12 dev walls 0.38–0.74 h) + 1 smoke + evaluations; expected ≈ 3 h |
| Titan | 0 | not needed |
| CPU | ≤ 3 wall-hours (Deepthought, `nice -n 10`, `OMP_NUM_THREADS=4`) | tests, evaluation fits, report |
| Storage | ≤ 2 GB and ≤ 1 000 entries under `<ext>`; free entries stay above 30 000 | two 7.5 MB checkpoints + caches |

Current L40S ledger **104.635 / 170**; S1 start rule 160 → projection ≈ 108, safe. Book every
job in `RESOURCE_LEDGER.json` (`v4_caps.E2_lucid_l40s_deepthought2`).
`TMPDIR=/ceph/dwong/jepa/tmp/`; pytest `--basetemp` inside `<ext>/tmp/`, `-p no:cacheprovider`.

## 6. Outputs, commit, report

1. `results/jepa/20261102_static/`: `README.md` (`registered: false`), `runs.json` (per seed:
   wall, full/half metrics, `f_layout`, `f_event`, effective rank, the frozen-label event
   fraction and per-PMT variance), `readings.json` with pinned keys
   `{"static_target_r": {"verdict": "recovered"|"not recovered"|"unclear", "seeds": {...}}}`,
   `reproduction.json`; each ≤ 1 MB.
2. `<ext>/`: checkpoints, sidecars, caches deleted after `report`, hashes in
   `<ext>/large_arrays.json`, `manifest.jsonl`.
3. One `DECISIONS.md` line per engineering choice; a short `STATE.md` entry; book resources; do
   not touch `PREDICTIONS.md`/`PREDICTION_STATUS.json`.
4. Commit order: runbook + code + tests first (plus the DECISIONS line), then smoke/runs, then
   results; report the final SHA.
5. Report to the author in Chinese (≤ 1 page): the static_target_r direction with numbers, the
   manipulation check, `f_layout`/`f_event`, resources and the final SHA; state the asymmetry
   caveat from §1.
6. Do not resume the grid; a follow-up (target-content ladder, scale, view design) is a separate
   runbook.
