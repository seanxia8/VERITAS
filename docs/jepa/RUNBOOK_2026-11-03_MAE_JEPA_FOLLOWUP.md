# LUCiD MAE / JEPA follow-up — five frozen-model readouts and one MAE training control

**Status: DRAFT v7 (2026-11-03).** Independent reviews at `f5b7434`:
- review of v1 (fresh reviewer): BLOCK, 8 blocking + 12 nonblocking findings;
- review of v2 (a second fresh reviewer): BLOCK, 2 blocking + 8 nonblocking findings;
- focused re-review of v3 (the second reviewer): BLOCK, 2 blocking (the gate-(d) path set; the
  T-1 host check);
- re-review of v4: **PASS** (checklist (i)–(vii) all PASS). The review records are kept with
  the design documents, outside the repository; this runbook does not depend on them.
- focused review of v5: BLOCK, 3 blocking items, all in A-1b (membership not identifiable;
  readings credited geometry to input statistics; reproduction target mismatched). All are
  fixed in v6 (Appendix E);
- re-review of v6: BLOCK, 1 blocking item (the across-membership test was too tight); fixed in
  v7 (Appendix F);
- re-review of v7: **PASS** (checklist (i)–(viii) all PASS).
- **v5 (author request, 3 Nov 2026, 11:22 CET).** Added A-1b, the origin of the per-PMT code;
  the Stage A cap moves to 5.5 and S1 to "+ 9.0"; a host table is added (§5). Appendix D
  lists the changes.

The runbook text is cleared for implementation. Execution still needs the implementation
review and the acceptance record (§6).

All findings are addressed here; the change tables are in Appendices A–C. This version is **not
accepted**: no outcome-generating work runs until the
implementation has passed independent review and the lead has written the acceptance record
(§6). Extension `mae_jepa_followup_2026-11-03`, `registered: false`. This runbook does not change
any registered prediction, does not restart the descoped grid, and does not reopen work that was
stopped before.

Branch `JEPA`, at or after `f5b7434`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md` before any outcome-generating step. It
depends on no document outside the repository. The current state is in `docs/jepa/STATE.md`.

---

## 0. Authority

Quote both lines verbatim in `DECISIONS.md`.

1. The author, 3 Nov 2026, 10:38 CET, in the design-agent conversation:
   > "我同意T-1训练"

   This authorises **exactly** the four T-1 training runs and the one 20-step smoke in §4, and
   nothing else. The B-batch authorisation does not cover T-1, and this line covers no other
   training. Stage A (§3) trains nothing and runs under the standing rule below.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

### 0.1 Fix it yourself, do not stop to ask

- Fix small problems yourself (bug, path, environment, guard HOLD, non-finite loss), write one
  `DECISIONS.md` line ("author standing approval, 30 Oct 2026"), and continue.
- Write only inside `/ceph/dwong/jepa/`, except for the repository deliverables listed in §6.5.
  Make no registered changes, stay within §5, and touch nothing else outside `/ceph`.
- Stop only for the conditions in §5.

---

## 1. Questions

Background (from `docs/jepa/STATE.md`):
- The trainable-target SIGReg-JEPA collapses to a learned, event-invariant per-PMT code (H2):
  - E3 (frozen random target) recovers;
  - the static target does not;
  - scale (`20261102_scale`) and architecture (`20261102_arch`) do not recover it.
- The finding that module-gain drift is not a top shared direction in `within` comes from a
  linear CCA on the data (`20261029_predictability/linear_cca.md`). No *non-collapsed*
  encoder has been tested at s = 0.30.

| ID | Question | Why it is open | Item |
|---|---|---|---|
| **Q-B1** | Which training variants escape the event-invariant code (ijepa, lambda, stopgrad, anc100, strong_drift, the de-jitter SIGReg retrains)? Does `f_event` track vertex across trained checkpoints? | `f_layout`/`f_event` exist only for init, old, fix, mae9000, across, E2, E3, static, scale and arch. ijepa's photon "predictable" log-charge R² is 0.5285/0.8083 (`20261029_predictability/probes/ijepa900{0,1}.json`), yet its vertex is worse than init; its event dependence is unmeasured | A-1 |
| **Q-B2** | Where do the MAE checkpoints sit on event dependence, beside init, E3 and static? | Only mae9000 (Part C) is measured: `f_layout` 0.8829, `f_event` 0.0455. The eight MAE de-jitter checkpoints (`20261031_mae_dejitter`) have not been measured | A-1 |
| **Q-B3** | What does the collapsed per-PMT code encode? Is it a function of the training distribution's per-PMT statistics? | The H12 code-origin check (`20261101_h12/e0e1_code_origin.json`) had three limits: (1) it covered fix9000 only; (2) it used per-PMT statistics from 2 000 *photon probe-test* events, whereas training used 100 000 mixed photon/μ/e events; (3) it had no time spread. Its result was combined MLP 0.434 (init 0.91). Target membership is not a per-PMT variable (A-1b) | A-1b |
| **Q-A1** | Does a non-collapsed model keep strong module-gain drift, and does that cost vertex? | MAE gain R² at s = 0.05 is 0.166–0.217, at its floor, which cannot discriminate. The s = 0.30 floor is 0.841585; SIGReg, which is collapsed, falls to 0.097/0.217 there | A-2, T-1 |
| **Q-D1** | Do the representations carry particle physics beyond photon vertex: μ/e type, direction, deposited energy? | All probes so far use photon vertex and yield. The dev probe sets are mixed photon/μ/e (17/17/16 per 50-event run), with truth in `events.npz` | A-3 |
| **Q-P1** | How much does the time reference leak target information into context tokens, and does it matter for the target-hidden readouts? | `ext_dejitter.median_hit_time` (`ext_dejitter.py:33-46`) runs over all hit tokens at tokenisation. The NHits trigger (`noise_module_lucid.readout.nhits_trigger_time`, `readout.py:352`) runs over all fired digits before the prompt cut. Both happen before the encoder's mask | A-4 |

**Already answered; do not repeat:**
- encoder jitter sensitivity (`20261030_dejitter/sensitivity.md`, MAE M-e);
- the MAE across gap (M-a, not supported);
- target dynamics E2/E3/static (`20261101_h12`, `20261102_static`);
- scale and architecture.

**View design** (`RUNBOOK_2026-11-03_VIEW_DESIGN.md`) is authorised separately. Do not launch,
retry, stop or alter it. `results/jepa/20261103_view/` does not exist at `f5b7434`. A-1 reuses
its committed `runs.json` if it exists when A-1 starts; otherwise those rows are
`pending (view design)`.

---

## 2. Snapshot, data and side effects

At `f5b7434`:
- L40S ledger 116.1507/170; generation CPU 18.425639/48 (`RESOURCE_LEDGER.json` `v4_caps`).
- The untracked `.pytest-tmp-predictability-full/` on the author's Mac: ignore it. Do not
  delete or commit it; exclude it from `git status` checks.

**Pinned data.**
- Data root `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/` (confirm on the host).
- Hashes are the sha256 of `manifest.json`. The s = 0.30 manifests are committed in
  `results/jepa/20261029_d1_partial/data/ext_pregrid_dev_20261025/lucid/`.

| Key | sha256 |
|---|---|
| `dev_probe_train__F1__0.05__s20261017` | `093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3` |
| `dev_probe_test__F1__0.05__s20261017` | `c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a` |
| `dev_train__F1__0.05__s20261017__n100000x1000` | `b1b3f4f401bcee48c49c347b67d7f5cbd4410213f08ef8b9d2848edc9f2cd13a` |
| `dev_probe_train__F1__0.3__s20261017__n10000x50` | `6df84887b3469723da3ddb0646c292ab655b263d6063f5fd713432a53514bb46` |
| `dev_probe_test__F1__0.3__s20261017__n10000x50` | `c68d83d1ef8cb202f3937f7cd48e3f74bd94f6d44ae203591053fde0605d1be2` |
| `dev_train__F1__0.3__s20261017__n100000x1000` | `37e727cc6fe402396c9091209a47e6dcfd3be6ed6f756bd38fb75ef10f88c75d` |
| `dev_train_300k__F1__0.05__s20261017__n300000x1000` (A-1b marginals for scale300k only) | `9a9dbe81ba63cea1c2ccad27e3d5cce7df9cb6e99faf0a9b340c6cec18c23a4a` |
| `dark_noise__s20261017__n512` (dark bank, used only by anc100) | `fff2f4b0d4a51ddef8229442933f7474c31d1c0953796ebf434ea5be664b016b` |

A0 re-verifies every hash on the host. A missing or mismatched key is a STOP for every item that uses it.
**No data generation of any kind is allowed in this extension.**

**Pairing (verified in the code).** The s = 0.05 and s = 0.30 sets share their event and run
keys:
- `run_id` does not depend on the setting (`lucid_bombs.py:670`);
- event draws, F3 and readout noise are keyed on (seed, run, event, role) (`lucid_bombs.py:491-499, 1798-1799, 1985-2011`);
- F1 rescales the same `u`;
- the PE count is shared, and the gain acts after it (`lucid_bombs.py:2101-2105`).

So truth and photon counts are identical. Only charge, threshold crossing, walk and trigger
differ. A0 re-verifies this on the host by joining on `(run_id, event_id)` and comparing truth.

**Forbidden call paths (each covered by a test).**
- Any path that reaches `lr.ensure_dataset` (`ext_h12.load_probes`/`datasets_for`,
  `scripts/jepa_ext_readout.py::load_probes`, `pregrid.datasets_for`), unless it is preceded by
  a read-only manifest-hash check of every key it will open.
  - The readout and h12 loaders hard-code `registered=True`. Called with s = 0.30, they would
    generate a **new** 20 000-event set. s = 0.30 sets are opened only through their
    `registered=False` keys above (`scripts/jepa_pregrid_checks.py:216-227` gets this right).
- `scripts/jepa_ext_readout.py::preflight_data(..., regenerate=True)`.
- `_regenerate_probe_sets`, which moves shared data directories.
- `scripts/jepa_ext_dejitter.py report`, which overwrites `results/jepa/20261030_dejitter/`.
- `lr.ensure_dark_bank` (`lucid_run.py:488-515`, reached through `pregrid.build_loaded` for
  anchor arms), unless it is preceded by a read-only manifest-hash check of the pinned dark bank.
- `lr.run_one` into an existing run directory (it auto-resumes).

**Reusable code (read-only imports; never edit or monkeypatch):**
- `src/jepa_noise/ext_readout.py`: forms `full_input`, `half_pass`, `context_only`,
  `propagated`; `layout_and_event_fractions`; ridge and MLP probes;
- `ext_h12.py`;
- `ext_dejitter.py` (`install_median_time_wrapper`);
- `ext_arch.py` (`build_arm_patch(variant)`, a context manager, `ext_arch.py:128`);
- `lucid_views.py`;
- `scripts/jepa_pregrid_checks.py` (`variant_cell`, `datasets_for`, `build_loaded`,
  `evaluate_forms`);
- `scripts/jepa_ext_dejitter.py`: read-only constants only. Its `mode_train` rejects unknown
  variants (`:60-61`, `:178-184`, `:749-751`), so T-1 re-implements its call sequence (§4).

### 2.1 Checkpoint inventory

A0 writes `inventory.json` with one row per label. Each row has: arm, cell, split, seed, beta,
encoder override, eval flags (zero_geometry, wrappers), time transform, data key, absolute path
and sha256.

Path sources:
- old/fix/mae9000/across: `scripts/jepa_ext_readout.py:66-105` `ITEMS` (path + sha; init9000
  is a seed-9000 rebuild);
- predictability families: the `checkpoint` field of the probe JSON;
- de-jitter families: the `large_arrays.json` key under the root named in the table;
- sha-only families: search only under `/ceph/dwong/jepa/returns/ext_h12_20261101/`
  (`ext_h12.EXT_ROOT`), `…/ext_static_20261102/`, `…/ext_scale_20261102/`,
  `…/ext_arch_20261102/` and `…/ext_view_20261103/`, for files named `final.pt`.

Every path is accepted only on a sha match. Hash sources:

| Family | Labels | Hash / path source |
|---|---|---|
| reference | init9000; strong-drift init floor (D1) | `20261101_readout/matrix.json` items; `20261029_d1_partial/README.md` |
| SIGReg plain | old9000/9001, fix9000/9001, across9000/9001 | `20261101_readout/matrix.json` items |
| SIGReg variants | ijepa (arm `ijepa_ema`), lambda (prediction + 0.2 × SIGReg), stopgrad, strong_drift, each 9000/9001 | `20261029_predictability/probes/<label>900{0,1}.json` (`checkpoint`, `checkpoint_sha256`) |
| anchored | anc100: arm `sigreg_jepa_anchored`, β = 100, **seed 9103, single seed** | `20261029_predictability/probes/anc100.json` |
| SIGReg de-jitter retrains | within_trigger, within_median, across_median, each 9000/9001 | `20261030_dejitter/large_arrays.json` (relative to `/ceph/dwong/jepa/returns/ext_dejitter_20261030/`) |
| MAE (Part C) | mae9000 (`efd54c25…`) | `20261101_readout/matrix.json` items |
| MAE de-jitter | within_trigger, within_median, across (trigger), across_median, each 9000/9001 (8 checkpoints, distinct from mae9000) | `20261031_mae_dejitter/large_arrays.json` (relative to `/ceph/dwong/jepa/returns/ext_mae_dejitter_20261031/`) |
| target controls | E2 geomblind, E3 frozen_target, static_target, each 9000/9001 | `20261101_h12/runs.json`, `20261102_static/runs.json` (sha only) |
| scale / arch | scale300k, window128, window32, pred4, each 9000/9001 | `20261102_scale/runs.json`, `20261102_arch/runs.json` (sha only) |
| view design | four splits × 9000/9001, **only if committed** | `results/jepa/20261103_view/runs.json` |

A row whose sha cannot be matched is `unavailable`. Never retrain or regenerate it.

**Evaluation configuration per family.** Each checkpoint is evaluated exactly as it was at
training time:
- E2: `zero_geometry=True`;
- arch: inside `ext_arch.build_arm_patch(variant)`, with the encoder signature from `runs.json`;
- anc100: built with `pregrid.build_loaded`, using the dark bank from the pinned key (after the
  hash check) and β from the checkpoint identity. `ext_readout.item_arm` cannot build anchor
  arms (`lucid_arms.py:335-336`);
- view design: `split="within"`, without the masker patch;
- median-time items: inside `install_median_time_wrapper(pe)`;
- strong-drift items: on the s = 0.30 probe sets.

`ext_readout.item_cell` hard-codes the s = 0.05 dev cell (`ext_readout.py:688-694`), so the
s = 0.30 path needs its own cell, built with `setting=0.30, s_arm=0.30`.

---

## 3. Stage A — frozen-model readouts (0 training)

**A0. Preflight (read-only).**
- Write `plan.json` with:
  - the repository SHA and tree status, and this runbook's sha256;
  - `inventory.json`;
  - the data manifests (full hashes), the config hashes and the pairing check;
  - hardware and software;
  - the projected Stage A L40S device-hours = Σ items × 126 s × pass factor × 1.3.
    - 126 s is the mean measured per-item wall of the `20261101_readout/matrix.json` items.
    - Pass factor: 1 for a photon full or masked pass; 2 for A-3 (μ + e); 0.5 for an A-4
      masked-only reference.
    - A-1b label passes use pass factor 0.6 (2 000 events, target-only).
    - Expected ≈ 3.5–4.9 h.
    - **If the projection exceeds the Stage A cap (5.5 h), A0 STOPs (S1) before any GPU work.**
  - CPU hours and free entries.

**A1. Reproduction gate.**
- Tolerance ≤ 1e-3 m for vertex and ≤ 1e-3 absolute for gain R².
- **s = 0.05 rows.** A failure stops all of Stage A.

  | Item | Convention | Committed value |
  |---|---|---|
  | `init9000` | trigger | 0.160072 |
  | `fix9000` | trigger | 0.219258 |
  | `mae9000` | trigger | 0.104949 |
  | SIGReg `within_median` 9000 | median wrapper | 0.222653 |
  | MAE `within_trigger` 9000 | trigger | 0.104916 |

- **s = 0.30 rows** (`20261029_d1_partial`, `20261029_d2_partial/METRICS.md`). A failure stops
  A-2, the strong-drift A-1 rows and **all of T-1, including its launch**.

  | Item | Vertex | Module-gain R² |
  |---|---|---|
  | init9000 floor | 0.170301 | 0.841585 |
  | `strong_drift9000` | 0.233209 | 0.097482 |

- **Recomputed items (A-1, A-3, A-4).** Before any of its metrics is used, each recomputed
  item must also reproduce its own committed full-input vertex within 1e-3 m. A mismatch
  marks the row `unavailable`.

**A-1. Event dependence (Q-B1, Q-B2).**
- **Reuse, do not recompute:** committed `f_layout`, `f_event`, full vertex, ridge token R² and
  effective rank for:
  - init, old, fix, across, mae9000;
  - E2, E3, static, scale, arch;
  - view design;
  - the eight MAE de-jitter token-R² and effective-rank values.
- **Compute:**
  - `f_layout`, `f_event`, full vertex, ridge and MLP token ln q R², MLP vertex and effective
    rank for: ijepa, lambda, stopgrad, strong_drift (each 9000/9001), anc100, the six SIGReg
    de-jitter retrains, and the eight MAE de-jitter checkpoints;
  - the missing MLP token R² and MLP vertex for: E2, E3, static, scale, arch and across.
- **Protocol:** the photon probe-test events with the exact `20261101_readout` protocol, under
  each item's own evaluation configuration (§2.1).
- **Classes (fixed now):**
  - *event-dependent*: `f_event` ≥ 0.01 in both seeds;
  - *event-invariant*: `f_event` ≤ 1e-3 in both seeds;
  - otherwise *intermediate*;
  - single-seed items (anc100, the init floors) are reported as `single-seed (not classified)`.
  - Reference values: fix 8.0e-5/1.65e-4, E3 0.0364/0.0381, static 0.0171/0.0173, E2 0.346/0.363,
    mae9000 0.0455, init 0.0790.
- **J-a:** report each SIGReg variant's class. E2 (`f_event` 0.35, vertex 0.83 m) and static
  (0.017, 0.160 m) already show that event dependence is **not sufficient** for good vertex. The
  new information is which training variants escape the invariant code at all.
- **J-b (descriptive):** report each MAE checkpoint's `f_event` and `f_layout` beside init,
  E3 and static. Draw no inference about target mechanism.
- **J-c (descriptive):** Spearman correlation between `f_event` and full-input vertex over
  s = 0.05, trigger-convention, full-geometry trained checkpoints only. Exclude strong_drift,
  median-time, E2 and anc100; list the members and n; no p-value.

**A-1b. Origin of the per-PMT code (Q-B3).** No training. The GPU is used only for target-only
forward passes; the statistics and probes run on CPU.

- **Items:**
  - collapsed: fix9000/9001, scale300k 9000/9001, window128/window32/pred4 9000/9001, across
    9000/9001 (12 items);
  - reference: init9000.
  - Each item uses its own evaluation configuration (§2.1).
- **Target (`m_p`).** Per item, the E1 invariant per-PMT component of its target embedding.
  - **Procedure:** the committed E0/E1 core: `lr._probe_batch(split=…, mask_seed=config["views"]["mask_seed"])` (= 0, the committed probe convention), then `arm.jepa.encode(features, hidden_mask=~batch.target)`, then 32 tokens from `hidden & hit` per event (sampling seed `er.SAMPLE_SEED`), over 2 000 photon probe-test events.
  - **Split:** `within` for fix, scale and arch; `across` for across. For across, `_probe_batch` builds `ViewMasker(split="across", n_min=0)`, so there are no redraws.
  - **Collector:** `ext_h12._collect_label_rows` is hard-wired to fix9000 and to `load_probes`, so `ext_followup` re-implements this core read-only, without editing `ext_h12`. The collector must reproduce the committed fix9000 rows before it is used on any other item.
  - **Cache:** the committed fix9000 label cache (`<h12 ext>/e0e1_rows.npz`) may be reused only if `_read_label_cache` accepts it (`meta.fix_sha256 == 2d912a2d…`, `max_label_events == 2000`).
  - Record the number of PMTs present per item. It can differ from 1 050, because across target sets cover every module.
- **Per-PMT features.**
  - **M (new): the training distribution's per-PMT marginals.** Read with a read-only reader.
    - **Data:**
      - for fix, arch, across and init, all events of the pinned
        `dev_train__F1__0.05__s20261017__n100000x1000`;
      - for scale300k, all events of its own training set
        `dev_train_300k__F1__0.05__s20261017__n300000x1000` (manifest sha256
        `9a9dbe81ba63cea1c2ccad27e3d5cce7df9cb6e99faf0a9b340c6cec18c23a4a`; disjoint events,
        `event_start` 1 300 000; `20261102_scale/runs.json`). Add it to the §2 pins.
    - **Reader:** after the manifest-hash check, read `manifest.json` → `runs/<run_dir>/hits.npz` (`offsets`, `pmt_id`, `charge_pe`, `time_ns`) and the per-event `domain` from `events.npz`. Never call `ext_h12.collect_per_pmt_features`, `load_probes` or anything else that reaches `ensure_dataset`.
    - **A0 assertion:** every `dev_train` run has `events.npz` with a `domain` key. A non-mixed run would silently give empty μ/e statistics.
    - **Statistics:** for each PMT, pooled and separately for photon, μ and e: hit rate; mean and SD of hit ln q; mean and SD of hit-token time (`time_100ns`, trigger convention). That is 20 features.
  - **C (committed, for continuity):** the committed six code-origin groups (`ext_h12.py:43, 299-329`): position, hit_rate, charge, parity, module one-hot, raw-token mean; 120 features. They are computed over the same 2 000 photon probe-test events. Because `collect_per_pmt_features` calls `load_probes`, C needs its own read-only probe-test pass.
  - **All** = M + C.
  - **Target membership is not a usable per-PMT feature**, so A-1b makes no membership claim:
    - under `within` it equals parity, which is already in C;
    - under `across`, every module's target frequency is `(n_modules // 2) / n_modules` by
      construction: a cyclic contiguous block from a uniform offset (`lucid_views.py:160-196`).
    - Both facts are asserted by test (§6.3).
- **Probes:** the committed code-origin protocol (`ext_h12.code_origin_probes` logic): ridge
  (`_multioutput_ridge`), MLP (early stopping on 10% of the training PMTs) and kNN-5. Split
  80/20 by PMT, seed 0, over the PMTs present for that item. Groups: M, C, All, and each
  single M statistic.
- **Reproduction:** on the fix9000 rows (2 000 photon probe-test events, 32 tokens, seed 0, PMT
  split 80/20 seed 0, n_pmt 1 050), the C set must give ridge R² 0.1043 within 1e-3
  (`e0e1_code_origin.json` `probes_by_label.fix9000.all.linear`; the ridge is deterministic).
  Otherwise all of A-1b is `unavailable`.
- **Readings (per item, MLP R², both seeds must agree; the classes are checked in the order listed, and the first match applies):**
  - *input-statistics code*: M ≥ 0.7;
  - *fixed-feature code (geometry or probe-test raw)*: M < 0.7 and All ≥ 0.7;
  - *not determined by fixed per-PMT features*: All ≤ 0.5;
  - otherwise *intermediate*.
  - Report M, C and All beside the init9000 reference and the committed fix9000 values (All
    MLP 0.434). The init value of 0.91 is mostly geometric (position MLP 0.79, module 0.84).
- **Interpretation rule (fixed now).** An invariant code is, by definition, the per-PMT mean of
  the target embedding, so "the predictor predicts each PMT's target mean" is not a finding.
  A-1b asks whether that mean is a function of the training data's per-PMT statistics. If an
  item reads *not determined by fixed per-PMT features*, the Paper B sentence is "a learned
  code, not determined by geometry or the training data's per-PMT input statistics".

**A-2. Frozen MAE under strong drift (Q-A1, part 1).**
- **Models:** MAE `within_trigger` 9000/9001 and MAE across (trigger) 9000/9001, trained at
  s = 0.05.
- **Evaluate** on the s = 0.05 and s = 0.30 photon probe sets, on paired events:
  - module-gain run-mean R² against the floor. The random-init full-input readout does not
    depend on the split, so one floor per environment serves both splits: s = 0.30 0.841585,
    s = 0.05 0.200498 (Part C init9000; the D2 "nominal" floor 0.218548 is not used);
  - full-input vertex with the probe refit on each environment's own probe-train;
  - secondary: vertex with the probe frozen from s = 0.05. This answers the shift question,
    not the representation question;
  - log-yield R².
- **Uncertainty:** paired event-level vertex differences, with a run-cluster bootstrap that
  resamples the 200 probe-test runs jointly across environments, with the probe fit held fixed
  (1 000 resamples, keyed seed 20261103). Each seed is reported separately; make no
  seed-population claim.

**A-3. Particle-task probes (Q-D1).**
- **Checkpoints:** init9000, fix9000/9001, E3 9000/9001, ijepa 9000/9001, MAE `within_trigger`
  9000/9001, MAE `within_median` 9000/9001.
- **Data:** the μ and e events of the s = 0.05 probe-train and probe-test.
  - A new read-only truth loader reads `events.npz` (`particle`, `direction_m`, `energy_MeV`,
    `deposited_MeV`, `contained`; written at `lucid_bombs.py:2262-2285`) and aligns it on
    `event_id`.
  - Event embeddings are the full-input pooled embeddings.
  - Splits are fixed by run. Hyper-parameters are chosen by run-grouped 5-fold CV on probe-train
    only.
- **Tasks:**
  - μ/e classification: L2-logistic on standardised features. Report AUC and balanced accuracy,
    overall and within NHits quintiles, so that yield alone cannot carry it.
  - Direction: ridge and MLP onto the unit vector. Report median and 68% angular error, per
    domain, over all μ or e events.
  - Deposited energy: contained events only.
    - **500 rule:** a per-domain energy probe is `not_evaluable` if that domain has fewer than
      500 contained events in probe-test, counted from `events.npz` `contained` on the pinned
      set.
    - The committed counts are μ 452 and e 2 570 (`STATE.md:44-45`; `A10_AUDIT_2026-10-23.md`
      §4), so the μ energy probe is expected to be `not_evaluable` and the e energy probe to
      run.
- **Baselines:**
  - the random-init floor;
  - a raw-feature baseline: `[ln Σq, NHits, Σq-weighted mean and SD of (x, y, z, t)]` over hit
    tokens, with the same probes.
- **Label:** this is the simplified A10 Cherenkov generator, not WCSim/PhotonSim; the results
  are simulator-internal. A-3 does not evaluate the registered R13.

**A-4. Time-reference leakage (Q-P1).**
- **Checkpoints:** MAE `within_trigger` 9000/9001, MAE `within_median` 9000/9001, fix9000/9001,
  E3 9000/9001.
- **Readout:** the `context_only` and `propagated` forms of the masked pass (`batch.hidden` =
  the registered `within` target set, `ViewMasker(n_min=0)`).
  - `half_pass` is **not** target-hidden: its target positions come from a sub-pass in which the
    target tokens are visible (`ext_readout.py:97-101`). It is therefore reported descriptively
    only, with a per-sub-pass reference (context sub-pass: hits with `hit & ~t`; target sub-pass:
    hits with `hit & t`).
- **References (two per model):**
  - *all-token*: the committed convention.
  - *context-only*:
    - median models: the median of the raw (unwrapped) hit-token times over `hit & ~hidden`,
      applied to every hit token; 0 if no hit is visible;
    - trigger models:
      `T_ctx = nhits_trigger_time(time_feature[hit & ~hidden] × scaling.time_scale_ns)`, in ns
      relative to the stored global trigger. Token time is `time_ns / time_scale_ns` (100 ns,
      `point_encoder.py:71, 165`), and the NHits window is 20 ns (`readout.py:344-366`).
      - Every hit token's time feature is reduced by `T_ctx / scaling.time_scale_ns`.
      - If `T_ctx` is `None`, apply shift 0 and count the event in `n_ctx_no_trigger`.
- **Invariance test (end to end, with fixed masks).** Perturb the stored times of hidden
  (target) hit tokens, including one perturbation that moves the all-token median (median
  models) or the all-token NHits proxy (trigger models).
  - Under the context-only reference, the context-position features `features[~hidden]` and
    the `context_only`/`propagated` outputs must be bit-identical. Hidden-position features
    are expected to differ; they are overwritten by `mask_embed` (`point_encoder.py:308-309`).
  - Under an all-token reference that is *recomputed from the perturbed tokens*, they must
    change.
  - Stored trigger-model tokens already have the global trigger subtracted. That is why the
    all-token side of the test recomputes the proxy rather than reusing the stored reference.
- **Residual leak, not removed:** the prompt-window cut ([−20, +60] ns around the global
  trigger) decides which context hits exist, and that depends on target-PMT digits. Removing it
  needs re-digitisation, which is out of scope.
- **Readings** (Δ = context-only minus all-token, `context_only` form, per seed):
  - *material*: |Δvertex| > max(0.01 m, |v₉₀₀₀ − v₉₀₀₁| of that model's all-token readout) in
    both seeds;
  - *immaterial*: |Δvertex| ≤ 0.01 m in both seeds;
  - otherwise *unclear*.
  - Δ token charge R² is reported descriptively.
- **Interpretation:** the models were trained with the all-token reference, so the context-only
  evaluation is out of distribution. It is a sensitivity bound and does not invalidate any
  full-input result.

---

## 4. T-1 — MAE trained under strong drift (Q-A1, part 2)

Authorised by §0 line 1.

- **Runs:**
  - Four runs: `mae_mse` × {`s030_within_trigger`, `s030_across_trigger`} × seeds {9000, 9001},
    10 000 steps each.
  - One smoke: `mae_mse`, `s030_within_trigger`, seed 9000, 20 steps.
  - Nothing else: no second smoke, and no relaunch into an existing run directory.
- **Code path.** `ext_followup.train_t1` re-implements the exact call sequence of
  `scripts/jepa_ext_dejitter.py:742-792`:
  1. `refuse_existing_run_dir`;
  2. the read-only manifest-hash check;
  3. `pregrid.datasets_for`;
  4. `lr.run_one(arm_name="mae_mse", beta=None, dark_bank=None, steps=10000, evaluate=True, physics_method="ridge", save_embeddings=False, …)`;
  5. `pregrid.build_loaded`;
  6. `pregrid.evaluate_forms`;
  7. `ext_summary.json` with the same schema and `extension="mae_jepa_followup_2026-11-03"`.

  `jepa_ext_dejitter.py` supplies only read-only constants; it is neither edited nor
  monkeypatched. Inputs:
  - config `src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json` (`e6730ac1…`);
  - generator config `lucid_bombs_v2_calibrated.json` (`90ac1601…`);
  - `lucid_arms_v0.json`;
  - the data root of §2.

  Full config hashes go in `plan.json`.
- **New cells:**
  - `s030_within_trigger` = `replace(lr.dev_cell(config), cell_id="ext_dev_s030_within_trigger", setting=0.30, s_arm=0.30)`;
  - `s030_across_trigger` = `replace(scripts.jepa_pregrid_checks.variant_cell(config, "across"), cell_id="ext_dev_s030_across_trigger", setting=0.30, s_arm=0.30)`.
  - `s_arm` reaches only `ArmSettings.s`, which only `mae_chi2` reads (`lucid_arms.py:520`), so
    it has no effect on `mae_mse`.
  - Both use transform `none`. Training data is `dev_train__F1__0.3__s20261017__n100000x1000`;
    probes are opened through the `registered=False` s = 0.30 keys (§2).
- **Required checks.**
  1. **Unit test (before launch).**
     - For the pairs (`s030_within_trigger`, `within_trigger`) and (`s030_across_trigger`,
       `across`), `dataclasses.asdict(new_cell)` and `dataclasses.asdict(old_cell)` differ in
       exactly `{cell_id, setting, s_arm}`.
     - The keyword arguments that `ext_followup` passes to `lr.run_one` equal those of
       `jepa_ext_dejitter.mode_train`, except `cell`, `out_root`, `loader_dataset` and
       `probe_datasets`. The fixed ones are: `arm_name="mae_mse"`, `beta=None`, `steps=10000`,
       `evaluate=True`, `physics_method="ridge"`, `save_embeddings=False`, `dark_bank=None`,
       and the same config, arms and geometry objects.
  2. **Host check (after the smoke and after each run; not a unit test).**
     - **What is compared.** `run_one` writes no per-run manifest file. It appends one row per
       run to `<out_root>/manifest.jsonl` (`lucid_run.py:756-758, 865`). Compare the new row
       (matched by `run_id`) in `<ext>/{smoke,train}/manifest.jsonl` with the s = 0.05 row
       `ext_dev_s005_{within_trigger,across}__mae_mse__s<seed>` in
       `/ceph/dwong/jepa/returns/ext_mae_dejitter_20261031/train/manifest.jsonl`.
     - **Allowed differences.** The rows may differ only in `{run_id, cell_id, setting, s_arm, loader_dataset, loader_dataset_manifest_sha256, probe_datasets, probe_dataset_sha256, code_sha, git_status, clean_tree, host, device, started_at_utc, finished_at_utc, wall_seconds, final_checkpoint_sha256, probes, physics, probe_files}`.
       - The smoke is compared with `within_trigger` seed 9000, and may also differ in
         `{steps, steps_completed, final_step}` (20 against 10 000).
     - **Required values.**
       - Both rows have `outcome == "complete"`.
       - The new row has `resumed_from_step` ∈ {0, null}.
       - `loader_dataset_manifest_sha256` is `37e727cc…`.
       - `probe_dataset_sha256` is {`6df84887…`, `c68d83d1…`}.
     - Any other difference is S9.
- **Readouts:**
  - the A-2 readouts on both environments. This gives a 2 × 2 of training environment ×
    evaluation environment, with the A-2 models as the s = 0.05 row;
  - the A-1 metrics.
- **Readings (two-sided, descriptive):**
  - **G-1:** gain R² minus the floor on the s = 0.30 evaluation set (floor 0.841585):
    - *retained*: ≥ −0.10 in both seeds;
    - *suppressed*: ≤ −0.30 in both seeds (SIGReg was −0.744/−0.624);
    - otherwise *intermediate*.
  - **G-2:** on the s = 0.30 evaluation set, the full-input vertex of the s = 0.30-trained MAE
    minus that of the s = 0.05-trained MAE, per split and seed. Use paired events and the A-2
    bootstrap. Report the difference; there is no threshold.
- **Interpretation rule (fixed now):**
  - The CCA diagnostic models the linear *predictive* objective. Gain-scaled charge is part of
    MAE's reconstruction target, so *retained* is expected MAE behaviour and is **not** a
    failure of the diagnostic.
  - T-1 establishes whether a non-collapsed model keeps strong drift, and whether that costs
    vertex.
- **Cost:** measured at 0.555 L40S h per MAE run, including in-run evaluation
  (`mae_dejitter_l40s_runs`: 15 993.89 s for 8 runs).
  - Projection: 4 runs + smoke + cross-evaluation ≈ 2.7 h.
  - **Cap: 3.5 L40S GPU-h.**

---

## 5. Resources and stops

- **Caps:**
  - Stage A: **L40S only, ≤ 5.5 GPU-h**, with the projection rule of A0. A-1b adds about
    12 items × 126 s × 0.6 × 1.3 ≈ 0.3 h.
    - During execution, stop *before* an item that would cross the cap, never mid-item.
    - If the cap is reached, every remaining item in the fixed order (§6.5) is
      `not_run (cap)`.
  - T-1: ≤ 3.5 L40S GPU-h. T-1 readouts (A-2 and A-1 metrics on the four T-1 checkpoints) are
    booked to the T-1 cap.
  - No data generation.
  - CPU (Deepthought): ≤ 8 wall-hours in total for tests, statistics, probes, bootstrap and
    the report. Book them as CPU wall-hours under this extension's campaign in
    `RESOURCE_LEDGER.json`, separately from the generation cap, which stays untouched because
    no data is generated.
  - **S1 launch rule:** L40S ledger + view-design cap (8.0) + 9.0 ≤ 160 (at `f5b7434`:
    116.1507 + 8.0 + 9.0 = 133.15).
- **Where each part runs.** `/ceph/dwong/jepa/` is shared by both hosts.

  | Part | Host | Device |
  |---|---|---|
  | Implementation, focused tests, full suite (once per SHA), dry-run | Deepthought | CPU (`nice -n 10`, `OMP_NUM_THREADS=4`); never a Deepthought GPU |
  | A0 preflight (hashes, inventory, pairing, projection) | Deepthought | CPU |
  | A-1b marginals (M) and the read-only probe-test pass for group C | Deepthought | CPU |
  | Every encoder forward pass: A1, A-1, A-1b labels, A-2, A-3 embeddings, A-4 | deepthought2 | L40S |
  | T-1 smoke, 4 runs and their evaluations | deepthought2 | L40S |
  | Probe fitting inside an evaluation job (the `ext_readout` path in A1, A-1 and A-2; the T-1 in-run evaluation) | deepthought2, in that same job | L40S job |
  | Standalone fits on cached rows (A-1b, A-3, MLP rows, bootstrap), readings, report | Deepthought | CPU; cache only the pooled embeddings and sampled rows these need |
  | Independent code review | any host with a read-only clone | — |
  | Acceptance record (lead) | lead's clone; pushed to `JEPA` | — |
- **Host rules:**
  - at most two of the user's L40S jobs in total, each ≤ 45% of GPU memory;
  - **view-design jobs are expected live jobs**: wait for a free slot, do not STOP;
  - `TMPDIR=/ceph/dwong/jepa/tmp/`;
  - ≥ 30 000 free entries after the projected writes;
  - `<ext>` = `/ceph/dwong/jepa/returns/ext_followup_20261103/`, with sub-roots:
    - the smoke: `--out <ext>/smoke`. The run id does not depend on the step count
      (`lucid_run.py:202-204`), so the smoke must not share a root with the seed-9000 run;
    - the four runs: `--out <ext>/train`;
    - Stage A outputs: `<ext>/stageA/<item>`;
  - results go to `results/jepa/20261103_followup/`;
  - never overwrite or resume another root.
- **STOP conditions** (the only ones):
  - S1;
  - S3 (a real specification change, including any change to a reading or threshold after outcomes are seen);
  - S4 (no GitHub route after three documented attempts);
  - S9 (an A1 gate row or a T-1 host check still fails after one fix attempt). A recomputed
    item that fails its own reproduction is marked `unavailable` without a STOP;
  - a missing or mismatched pinned dataset or checkpoint (for the affected items only);
  - an entries-preflight failure;
  - a cap that would be crossed.
- **On a STOP:**
  - preserve logs and partial outputs, and record the STOP;
  - items that do not depend on the stopped one continue. Dependencies:
    - T-1 launches only after both A1 s = 0.30 rows pass;
    - G-2 additionally needs the A-2 s = 0.30 rows;
    - A-1, A-1b, A-3 and A-4 are otherwise independent;
  - book failed attempts, smokes and evaluations.

---

## 6. Implementation, review, acceptance, execution

1. **Implementation.**
   - Commit this runbook unchanged (see the header).
   - Add `src/jepa_noise/ext_followup.py`, `scripts/jepa_ext_followup.py` and
     `src/jepa_noise/tests/test_ext_followup.py`. These names are free at `f5b7434`.
   - CLI modes: `dry-run`, `inventory`, `reproduce`, `readout --item {A-1,A-1b,A-2,A-3,A-4}`,
     `train-t1`, `report`.
   - Reuse the existing modules read-only.
2. **Gate.** `ext_followup.assert_authorised(mode)` is called from the library entry point,
   not only from the CLI. Any failure raises `SystemExit(2)`. It:
   - (a) reads the committed `docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md` and requires
     its sha256 to equal the value in the acceptance record;
   - (b) for `train-t1`, requires the exact string `我同意T-1训练` inside §0 (read through the
     `_section_zero` helper, as in `ext_view.py:75-108`) and in `DECISIONS.md`;
   - (c) reads `docs/jepa/ACCEPTANCE_2026-11-03_MAE_JEPA_FOLLOWUP.json` =
     `{runbook_sha256, impl_commit, review_path, review_verdict, lead, date}` and requires
     `review_verdict == "PASS"`;
   - (d) requires both `git diff --quiet <impl_commit> -- <P>` (working tree against
     `impl_commit`) and an empty `git ls-files --others --exclude-standard -- <P>`.
     - `<P>`:
       - `src/jepa_noise/{__init__,ext_followup,ext_readout,ext_h12,ext_dejitter,ext_arch,lucid_run,lucid_arms,lucid_bombs,lucid_views,point_encoder,bridge_run,train,crlb,jepa,backbone,linear,probes,sigreg}.py`;
       - `src/noise_module_lucid/` and `src/jepa_noise/configs/`;
       - `scripts/__init__.py`, `scripts/jepa_ext_followup.py`, `scripts/jepa_pregrid_checks.py`,
         `scripts/jepa_ext_dejitter.py` and `scripts/jepa_ext_readout.py`;
       - `docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md`.
     - The dry run executes the setup path of every CLI mode, then asserts that every
       repository file in `sys.modules` (by `__file__`) is inside `<P>`. The check is on
       executed modules, not on a static walk, because lazy imports that no Stage A or T-1 path
       executes are irrelevant.
     - A module outside `<P>` is a dry-run failure, which the implementer fixes before review,
       not at execution.
     - Because `<P>` is limited to this closure, unrelated commits (for example from the
       concurrent view-design work) do not trip the gate.
     - The acceptance record goes in a later commit, so HEAD equality is never required;
   - (e) for `train-t1`, accepts only
     `(arm, variant, seed, steps) ∈ {mae_mse} × {s030_within_trigger, s030_across_trigger} × {9000, 9001} × {10000}`,
     plus exactly one smoke `(mae_mse, s030_within_trigger, 9000, 20)`. It refuses a second
     smoke and any launch into an existing run directory.
3. **Tests:**
   - run-grouped split disjointness;
   - A-2 pairing on `(run_id, event_id)`, including a deliberately mismatched case;
   - the A-4 invariance test (masked pass, unhit tokens, zero-visible-hit rule, `None`-trigger rule);
   - every forbidden call path of §2 is blocked;
   - A-1b:
     - the dev_train marginal reader opens no path that reaches `ensure_dataset`;
     - membership:
       - for `within`, the target set equals the odd-parity set;
       - for `across`, enumerating `_across_partition` over all `n_modules` offsets puts every
         module in exactly `n_modules // 2` target sets;
       - offsets drawn by `across_mask` (20 000 keyed draws) pass a χ² uniformity test at
         p > 1e-3;
     - the scale300k manifest hash check, and the A0 `domain`-key assertion;
     - the general label-row collector reproduces the committed fix9000 rows (same seed and
       events) before it is used on other items;
   - the T-1 cell and keyword-argument unit test (§4);
   - the anc100 build path, with the dark-bank hash check before `ensure_dark_bank`;
   - the gate's negative cases (a, b, c, d, e);
   - hash mismatch;
   - no overwrite.
   - The focused tests plus the full `src/jepa_noise/tests` suite run once on the final SHA. No GPU training as a test.
4. **Independent review** (fresh context, actual diff). Blocking checklist:
   - (i) pairing and truth leakage;
   - (ii) per-checkpoint inventory and evaluation configuration;
   - (iii) the A-4 masked-pass reference and invariance test;
   - (iv) A1 reproduction on both s = 0.05 and s = 0.30;
   - (v) gate (a)–(e);
   - (vi) forbidden call paths;
   - (vii) cost projection against the caps;
   - (viii) A-1b: the general label collector reproduces the committed fix9000 rows; the 0.1043
     reproduction; the read-only `dev_train` reader.
   - The reviewer records PASS or BLOCK per item. Blockers are fixed and re-reviewed. Then the
     lead writes the acceptance record.
5. **Execution.**
   - Use the accepted SHA and the CLI-generated commands only.
   - Recheck the guard, ledger, jobs and data.
   - **Order (fixed):** A0 → A1 → A-1 → A-1b → A-2 → A-4 → A-3.
     - T-1 may start once both A1 s = 0.30 rows pass, and then runs in parallel.
     - If the Stage A cap is reached, every remaining item in this order is `not_run (cap)`.
   - Deliverables: `plan.json`, `inventory.json`, `runs.json`, `readings.json`,
     `reproduction.json`, `README.md`, and ledger and `STATE.md` lines.
   - Large arrays stay on `/ceph` with their hashes. No push or merge beyond the repository's
     rules.

---

## 7. Out of scope

Not authorised by this runbook:
- unsupervised adaptation (LoRA, full fine-tune);
- MAE objective ablations;
- further JEPA target-dynamics arms;
- new view or mask designs;
- a second dataset or mechanism;
- trigger re-digitisation;
- new dark-noise, TTS or dead-PMT conditions;
- any data generation.

---

## Appendix A. v1 → v2 changes (review of 2026-11-03 at `f5b7434`)

| Finding | Change |
|---|---|
| B1 inventory | §2.1 rebuilt. Hash sources are named per family. `lam10` → `lambda`; anc100 is single-seed 9103; ijepa is arm `ijepa_ema`; mae9000 is separate from the 8 MAE de-jitter checkpoints; SIGReg `within_trigger` is listed as a de-jitter retrain |
| B2 evaluation config | Per-family evaluation configuration added (E2 zero geometry, arch wrapper, view `within`, median wrapper, s = 0.30 cell). A-1 now separates reuse from compute. Every recomputed item must reproduce its committed vertex first |
| B3 A-4 | Readout moved to the masked-pass `context_only`/`propagated` forms; half-pass is descriptive only. Correct trigger function, `None` rule, prompt-cut residual leak stated, invariance test corrected |
| B4 A1 | Exact values added; s = 0.30 floor and `strong_drift9000` added to the gate; ambiguous `within_median` fixed (0.222653) |
| B5 side effects | All forbidden call paths listed, data pinned with hashes, generation fallback removed (paired sets already exist) |
| B6 T-1 config | Code path, configs and the two new cells specified; manifest-diff test added |
| B7 gate | §0 authority added (with time and channel); acceptance-record path and schema; gate checks (a)–(e) with an allowlist |
| B8 caps | Stage A ≤ 3.0 L40S GPU-h, L40S only, with a measured projection; S1 counts view design at its 8.0 cap |
| N1 | Q-B1 lists E2; J-a/J-b rewritten (E2 and static already show event dependence is not sufficient) |
| N2 | MAE count corrected (8 de-jitter checkpoints) |
| N3 | Containment from `STATE.md:45` (μ 452, e 2 570); truth loader, classifier and raw baseline defined |
| N4 | Single floor per environment for both splits (0.841585 at s = 0.30) |
| N5 | View-design jobs are expected; S1 uses its 8.0 cap |
| N6 | A-4 readings are three-valued; seed spread defined |
| N7 | J-c population fixed; single-seed items not classified |
| N8 | State reference → `docs/jepa/STATE.md` |
| N9 | §0.1 "fix it yourself" clause added |
| N10 | ijepa charge figure cited exactly |
| N11 | `.pytest-tmp…` directory: ignore, do not delete |
| N12 | Extension, `<ext>` and results names pinned |

## Appendix B. v2 → v3 changes (second review, 2026-11-03, at `f5b7434`)

| Finding | Change |
|---|---|
| NB-1 Stage A cap (blocking) | Cap raised to 5.0 L40S GPU-h. A0 projection formula fixed, and A0 STOPs if the projection exceeds the cap. Execution order fixed (A-1 → A-2 → A-4 → A-3); items left when the cap is reached are `not_run (cap)`. S1 becomes "+ 8.5" (132.65) |
| NB-2 T-1 manifest test (blocking) | Replaced by a cell/keyword-argument unit test before launch, plus a host-side manifest comparison with an explicit allowed-difference set (otherwise S9) |
| N-T1a | T-1 re-implements the `mode_train` call sequence; `jepa_ext_dejitter.py` is neither edited nor monkeypatched |
| N-T1b | Smoke, train and Stage A sub-roots pinned (`<ext>/smoke`, `<ext>/train`, `<ext>/stageA/<item>`) |
| N-anc | Dark bank pinned (`fff2f4b0…`); `ensure_dark_bank` added to the forbidden list; anc100 build path specified |
| N-B1 | Path sources and search roots named for every family |
| N-A4a | Invariance test checks only `features[~hidden]` and the outputs |
| N-A4b | NHits proxy units fixed (× `time_scale_ns`); `None` → shift 0 |
| N-500 | The 500 rule is defined in A-3 |
| N-gate | (d) compares the working tree with the commit, restricted to the import closure |
| N-stop | S9 limited to the A1 gate rows and the T-1 host checks; dependencies of T-1 and G-2 made explicit |
| Notes | Full s = 0.05 hashes; `build_arm_patch` named; §0.1 repository-deliverables exception; Q-A1 wording |

## Appendix C. v3 → v4 changes (re-review of v3, 2026-11-03)

| Finding | Change |
|---|---|
| V3-B1 gate (d) path set (blocking) | `<P>` now includes the full top-level import closure (`__init__`, `bridge_run`, `train`, `crlb`, `jepa`, `backbone`, `linear`, `probes`, `sigreg`, `scripts/__init__.py`). The dry-run assertion is defined on executed modules (`sys.modules`), not on a static walk. A failure is fixed before review |
| V3-B2 T-1 host check (blocking) | The check compares `manifest.jsonl` rows matched by `run_id`, not per-run files. The allowed differences add `final_checkpoint_sha256`, `probes`, `physics`, `probe_files`, plus `{steps, steps_completed, final_step}` for the smoke. Both rows must have `outcome == "complete"`, and the new row must have `resumed_from_step` ∈ {0, null} |
| V3-N1 | The header now points to Appendices A–C |

## Appendix D. v4 → v5 changes (author request, 3 Nov 2026)

| Change | Where |
|---|---|
| New question Q-B3 and item A-1b: what the collapsed per-PMT code encodes. The committed code-origin check is redone on 12 collapsed items plus init. Features: training-distribution marginals (all 100 000 `dev_train` events; pooled and per photon/μ/e; hit rate, ln q mean/SD, time mean/SD) and target-membership frequency under each item's training masker, plus the committed groups. Readings, a reproduction check and an interpretation rule are fixed in advance | §1, §3 |
| Order A0 → A1 → A-1 → A-1b → A-2 → A-4 → A-3 | §6.5 |
| Stage A cap 5.0 → 5.5 L40S GPU-h; S1 "+ 8.5" → "+ 9.0" (133.15); CPU cap ≤ 8 h | §3 A0, §5 |
| Host table (Deepthought CPU, deepthought2 L40S) | §5 |
| CLI `--item A-1b`; three A-1b tests | §6 |

## Appendix E. v5 → v6 changes (focused review of v5, 2026-11-03)

| Finding | Change |
|---|---|
| V5-B1 membership not identifiable (blocking) | Feature T and the membership reading removed. Under `within`, membership equals parity; under `across`, every module has the same target frequency by construction (`lucid_views.py:160-196`). Both are asserted by test, and the Paper B sentence no longer mentions membership |
| V5-B2 readings credited geometry to input statistics (blocking) | Classification is on the M group: *input-statistics code*, *fixed-feature code*, or *not determined by fixed per-PMT features* |
| V5-B3 reproduction target (blocking) | C is the committed six groups (120 features); the reproduction target is ridge R² 0.1043 |
| N1 | Label masker is `_probe_batch(..., mask_seed = views.mask_seed = 0)`; across uses `n_min = 0` |
| N2 | Label-cache reuse goes through `_read_label_cache` (`fix_sha256 2d912a2d…`) |
| N3 | scale300k's M is computed on its own 300k training set (pinned `9a9dbe81…`) |
| N4 | Host table split: in-job probe fits on deepthought2; standalone fits on Deepthought |
| N5 | C re-implemented read-only; n_pmt recorded per item |
| N6 | Checklist (viii) added for A-1b |
| N7 | CPU ledger named |
| N8 | Header no longer points to files outside the repository |

## Appendix F. v6 → v7 changes (re-review of v6, 2026-11-03)

| Finding | Change |
|---|---|
| V6-B1 membership test too tight (blocking) | The ±0.01 sampling test (per-module SD ≈ 0.0035 over 105 modules, so it would fail about 40% of the time) is replaced by an exact enumeration of `_across_partition` and a χ² uniformity test on the offsets (p > 1e-3) |
| Host table | "membership draws (T)" removed |
| Readings | The A-1b classes are checked in the listed order; the first match applies |
