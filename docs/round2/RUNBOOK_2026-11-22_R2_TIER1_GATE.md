# Runbook — Tier-1 decision campaign: un-bundle the MAE repair, time-unit dose-response, sealed evaluation, yardsticks (r2-tier1), 22 Nov 2026

**Phase** `r2-tier1`. **Branch** `tier1`, created from `origin/lucid-mc`. **Worktree** `~/Documents/VERITAS-worktrees/tier1`. **Extension** `r2_tier1_2026-11-22`. **Results** `results/round2/20261122_tier1/`.

A second agent, `r2-tier1v`, recomputes the headline numbers blind on branch `tier1-verify` (`RUNBOOK_2026-11-22_R2_TIER1_VERIFY.md`). The two phases talk only through the metric-free manifests of §T2.4 and §T7.3.

---

## 0. Why this runbook exists

The LUCiD regeneration (`r2-lucid`) and a steering review (22 Nov) leave one question to settle before any paper is written: **does the per-channel loss scale, by itself, change what a masked autoencoder learns from PMT data, or did the "repair" work for other reasons?** This campaign answers it by Fri 27 Nov with pre-registered criteria, and closes the defects found since 21 Nov. The same runs serve all three possible papers (§0.2).

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `tier1`:

1. **Author, 22 Nov 2026, 11:32 CET:**
   > "写runbook，把没修的，以及这个方案写进去，不要问junjie，他是共同作者但是没时间。我们继续推进，按照spotlight去进行"

   (Translation: "Write the runbook. Put the unfixed items and this plan into it. Do not ask Junjie; he is a co-author but has no time. We keep moving, aiming at the Spotlight track.")
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决". Never stop to ask. At every decision point take the recommended option, record one line in `DECISIONS.md` and `$CC_STATE/notes/r2-tier1.md`, and continue.

Consequences:
- **No step may depend on the collaborator.** Where LUCiD expertise would help (§T6), work from LUCiD's code and documentation, time-box it, and use the stated fallback.
- **Target venue:** CPAL 2027 Recent Spotlight Track, deadline 18 Jan 2027 (non-archival), plus a journal submission. The CPAL Proceedings deadline of 5 Dec is not a target.

### 0.2 What is being decided

| Outcome | Paper lead |
|---|---|
| **A′** | "Loss share versus information share": at a fixed training budget, the per-channel loss scale decides which channel a masked pretrainer learns; a noise-matched scale is the principled default. |
| **A-alt** | Masking geometry on fixed sensor arrays is what matters. |
| **P2** (fallback) | Three ways to misread a self-supervised study on detector data: readout, bundled repairs, simulation world. |

§2 registers the criteria that select among them. The agent reports the outcome; the author decides.

### 0.3 Facts established from the code on 22 Nov (confirm in T0, then rely on them)

1. **The MAE "repair" bundles four factors.** Arm `mae_repaired` differs from `mae_mse` in:
   - mask: `within` (fixed parity, the same PMTs hidden in every event) → `spatial_patch` (per-event Hilbert patches, half hidden);
   - time-loss unit: the detector's window scale (WCTE 100 ns, SK 450 ns) → 50 ns;
   - loader: run-contiguous micro-batches → `shuffle_events_within_batch`;
   - encoder attention window: default 64 → 64 / 384 / 320 / 640 (WCTE / SK / JUNO / HK), in `ext_r2.encoder_config_for`.
2. **Per-event random masking has never been run.** `lucid_views.SPLITS = ("across", "within", "spatial_patch")`.
3. **The time unit is a pure loss weight.** `build_r2_arm` sets `time_loss_weight = (scaling.time_scale_ns / time_loss_scale_ns)²` for `mae_repaired`; the input time feature keeps the window scale. The flag `--time-loss-scale-ns` exists for `mae_repaired` only.
4. **The committed "init" is not config-matched.** `make_model("init")` builds the default encoder (window 64). Repaired arms on SK / JUNO / HK use larger windows. The matched control is each run's own `snapshot_step_00000.pt`.
5. **Snapshots exist.** WCTE and SK runs saved steps 0, 50, 100, 200, 500, 1000, 2000, 5000, 10000; JUNO and HK saved 0, 500, 2000, 10000.
6. **Token time is trigger-relative.** It is the digitised first-hit time minus the NHits trigger time, inside the prompt window. Absolute and trigger times are stored. LUCiD's digitised hits are cached under `/ceph/dwong/jepa/data/lucid_v1/<det>/cache/`.
7. **Probe cohorts.** `dev_probe_train` and `dev_probe_test` hold 10,000 events each (200 runs × 50, 1:1:1 photon / μ / e, so ≈ 3,400 photon bombs). The `_s030` cohorts are gain-state twins of the same digits, not independent noise replicates.
8. **Table-builder bug.** `scripts/lucid_toy_vs_lucid.py::lucid_rows` keys runs by (detector, arm, seed); sweep runs overwrite the default run. The WCTE `mae_repaired` seed-9000 row of `TOY_VS_LUCID.md` is the 5 ns run.
9. **Measured costs (r2-lucid ledger).** MAE training: WCTE ≈ 0.5 GPU-h (L40S), SK ≈ 5 GPU-h (A100), JUNO ≈ 4.3, HK ≈ 14.
10. **Existing runs that count as cells of this campaign** (read-only, in the `r2_lucid_2026-11-21` returns): `sk__mae_repaired__s9000`; `wcte__mae_repaired__s9000`, `…__s9001`, and `wcte__mae_repaired__s9000__ts450`, `__ts15`, `__ts5`, `__ts1p5`.

### 0.4 Binding principles

- **P1 — LUCiD does the physics.** New data (§T2) come from the `r2-lucid` generator path, tier A1, with the same configuration. Never use `lucid_bombs` / `r2_generator` physics.
- **P2 — Pre-registration before launch.** Copy §2 verbatim to `docs/round2/PREREG_2026-11-22_R2_TIER1.md`, commit and push it **before the first production training job is submitted**. Later changes are dated amendments appended to that file, never edits.
- **P3 — One factor at a time.** All E1/E2 cells share the encoder configuration, loader, optimiser, steps, micro-batch, data and seeds of the existing `mae_repaired` run of that detector. Only the named factor changes. The default recipe must keep its hash (§T1.6).
- **P4 — Sealed test set.** §T2.3 is binding.
- **P5 — Frozen readouts.** Appendix A is frozen. No readout, feature, cut or probe hyper-parameter changes after the first read of sealed data.
- **P6 — Run identity.** Every table, cache and JSON is keyed by the full run id including suffixes. Never key by (detector, arm, seed).
- **P7 — No collaborator dependency** (§0.1).
- **P8 — Other phases.** `r2-tier1v` runs in parallel on its own branch and clone; never touch them. `r2-ablate` stays deferred. Never write into the `r2_lucid_2026-11-21` returns, the LUCiD packs, or any other phase's worktree, branch or clone.

### 0.5 Safety constraints (always hold)

- Main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Before every commit, check that `git rev-parse --show-toplevel` is the `tier1` worktree.
- Never modify or delete existing data, checkpoints, returns or committed results. Never force-push. Push only to `origin/tier1`, fast-forward only.
- LUCiD: internal research use only; no redistribution of LUCiD code or LUCiD-generated data outside the group's storage.
- **Shared servers.** Never saturate CPUs. deepthought2: ≤ 2 GPU jobs, ≤ 45 % VRAM each, through `scripts/jepa_run.sh`. Check occupancy before every launch.
- **portal1.** This phase may use up to one third of portal1's cores at `nice -n 10`, always leaving ≥ 8 free. Every 10 minutes, if other users' load exceeds a quarter of the cores, halve the worker count; restore step by step. (`r2-tier1v` has its own, smaller share.)
- **JAX memory.** Every LUCiD/JAX process on a GPU runs with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` on shared GPUs (0.80 only on a GPU verified idle and unshared).
- **HTCondor.** ≤ 12 concurrent jobs, `accounting_group = cms`.
- **Idle polling.** While only waiting for jobs, poll at most every 25 minutes in one foreground Bash call.

### 0.6 Hosts, paths, budget

- Hosts: `dt1` (Deepthought CPU), `dt2` (deepthought2 L40S), `p1` (portal1). Tooling and layout are those of `r2-lucid`: read `$CC_STATE/notes/r2-lucid.md`, `docs/round2/DECISIONS.md` and `results/round2/20261121_lucid/HOSTS.md` first.

| What | Path |
|---|---|
| Remote clone | `/ceph/dwong/jepa/r2_tier1_repo` |
| New returns | `/ceph/dwong/jepa/returns/r2_tier1_2026-11-22/` (set `R2_REPAIR_OUTPUT_ROOT=<this>/train` for every new training) |
| Existing returns (read-only) | `/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/` |
| Sealed cohorts | `/ceph/dwong/jepa/data/lucid_v1/<det>/sealed/` and `<det>_lucid_v1_sealed.h5` |
| Scratch | `/work/dwong/r2_tier1/` |
| Batch staging | as in `r2-lucid` (`/ceph/srv/dwong/r2/…`, `SHA256SUMS`) |

- **GPU budget.** Ledger line `R2_tier1_gpu`, cap **110 GPU-h**, in `docs/round2/RESOURCE_LEDGER.json` on branch `tier1`. Expected use ≈ 100: SK trainings 75, WCTE trainings 12, sealed generation ≤ 6, evaluation ≤ 6, yardsticks ≤ 10. If the projection exceeds the cap, drop items from the bottom of the priority list in §E.
- **CPU.** ≤ 12 cores on `dt1` at `nice -n 10`, plus the portal1 share.
- **Ceph.** Estimate volume before generating; stop if a write would leave < 20 % free.

---

## 1. Outstanding defects this campaign closes

| ID | Defect (found 21–22 Nov) | Closed in | Acceptance |
|---|---|---|---|
| D1 | `TOY_VS_LUCID` builder key collision (§0.3 item 8) | T1.5 | Regression test; rebuilt table shows WCTE `mae_repaired` s9000 spatial Ridge 0.0521 m |
| D2 | No LUCiD number has been recomputed blind | `r2-tier1v`, via T2.4 / T7.3 | Manifests A and B on `origin/tier1` |
| D3 | SK repaired MAE has one seed | E (seed 9001) | Cells SK-4, SK-5 complete |
| D4 | Probe test set is ≈ 3,400 events and the readout was designed on it | T2 | Sealed cohorts validated; access log exists |
| D5 | Repair bundle not un-bundled (four factors) | E, T7 | Registered contrasts of §2.3 reported |
| D6 | Standard per-event random masking never run | T1.1, E | `random` split tested; cells trained |
| D7 | No likelihood yardstick; the "ToF ceiling" line is a crude fit | T1.5, T6 | Line removed or relabelled; `YARDSTICKS.md` |
| D8 | Energy and PID scored with saturated metrics (R², AUC) | T3 | Resolution and mis-ID at fixed efficiency reported |
| D9 | Conclusions depend on the probe | T3, T7 | MLP × 3 seeds and attentive × 3 seeds reported |
| D10 | Training-budget axis unexamined although snapshots exist | T4 | `BUDGET_AXIS.md` |
| D11 | Token time is trigger-relative; a common per-event shift may be what the time loss buys | T5 | Share reported; conditional cell run or skipped by rule |
| D12 | "Gain over init" used a non-matched init (window 64) | T3 | Step-0 snapshots scored as matched init |
| D13 | WCTE energy not evaluable under LUCiD's containment flag | T3.5 | Counts and a proposal recorded; not used in the gate |
| D14 | One native test fails on `dev` and here (`test_rate_zero_is_identical_to_the_pre_r6_reference`) | T9 | Triage note |

**Deferred, listed so that nothing is lost** (not in this campaign): JEPA ablations (`r2-ablate`: EMA only, batch fix only, latent-only JEPA); second SK seeds for plain MAE, repaired JEPA and fixed-label JEPA; plain MAE and fixed-label JEPA on JUNO / HK; HK JEPA at the full window; re-reading Paper-3 costs with the spatial readout; noise-matched loss (E3), 20k-step extension (E2b), timing-smear knob (E4), supervised yardsticks (E5b), JUNO / HK transfer (E6).

---

## 2. Pre-registration (copy verbatim to `PREREG_2026-11-22_R2_TIER1.md`)

### 2.1 Cells and notation

- A cell is (detector, mask, time unit, seed), with everything else as in the existing `mae_repaired` run of that detector (encoder window, shuffled loader, 10,000 steps).
- Masks: `patch` = `spatial_patch`; `random` = per-event uniform half of the PMTs; `parity` = `within`.
- Time units in ns. Seeds A = 9000, B = 9001, C = 9002.
- e(m, u, s): mean 3-D vertex error in metres on the **sealed** photon bombs, frozen spatial-moment Ridge readout (Appendix A).
- Per-seed contrast T(u₁→u₂; m, s) = e(m, u₂, s) / e(m, u₁, s) − 1. T̄ is its mean over seeds.

### 2.2 Statistics

- SE_boot: paired bootstrap over sealed test events (2,000 resamples) of the per-seed contrast, combined over seeds.
- SE_seed: SD_pooled / √n_seeds, where SD_pooled is the RMS, over the registered contrasts of that detector, of the across-seed SD of the per-seed contrast.
- N_σ = |T̄| / max(SE_boot, SE_seed).
- A contrast is **clear** when N_σ ≥ 2 and it has the same sign in every seed.
- **Borderline rule.** If a registered SK contrast has 1.5 ≤ N_σ < 2.5, train seed C for its two cells (inside the cap, dropping from the bottom of the priority list) and recompute.

### 2.3 Registered contrasts

| ID | Detector | Contrast |
|---|---|---|
| CT-p50 | SK | T(450→50; patch) |
| CT-p5 | SK | T(450→5; patch) |
| CT-r50 | SK | T(450→50; random) |
| CM-450 | SK | e(patch, 450) / e(random, 450) − 1 |
| CM-50 | SK | e(patch, 50) / e(random, 50) − 1 |
| WT-p | WCTE | T(450→50), T(450→5), T(450→1.5) at patch |
| WT-r | WCTE | T(450→50), T(450→5), T(450→1.5) at random |
| WM | WCTE | e(patch, u) / e(random, u) − 1 at u = 450 and u = 5 |

Secondary outcomes, same contrasts: μ and e direction (median angle), electron and muon energy resolution (68 % half-width of (Ê − E)/E), μ mis-identification at 99 % e efficiency.

### 2.4 Predictions

- **P1 (sign).** CT-p50, CT-p5, CT-r50 and all WT contrasts are negative.
- **P2 (monotone).** On SK at patch, e falls along 450 → 150 → 50 → 15 → 5. Only a clear reversal of ≥ 3 % between adjacent units counts against it.
- **P3 (trade-off).** On SK at patch, electron energy resolution worsens from 450 to 5 (contrast positive and clear). Muon energy has the same sign.
- **P4 (knee).** Let u₅₀ be the unit at which the time term is half of the end-of-training loss of the default (50 ns) run of that detector, computed from its training log (§T5.2) and appended to this file as a dated amendment **before any E-cell is evaluated**. Let u_half be the largest tested unit at which the reduction from 450 reaches half of the total reduction to the finest tested unit. Prediction: u₅₀ / 3 ≤ u_half ≤ 3 u₅₀ on each detector, and u_half(SK) > u_half(WCTE).
- **P5 (specificity).** With the charge-loss weight × 100 at (patch, 50), the vertex error is not more than 3 % lower than at (patch, 50), and the electron energy resolution is not worse.
- **P6 (direction).** The direction contrasts have the sign of the vertex contrasts.

### 2.5 Falsification, kill criteria and outcomes

- **K1.** On SK, both CT-p50 and CT-p5 are smaller than 5 % in magnitude or not clear. Then A′ is dropped. If CM-450 or CM-50 is ≤ −10 % and clear, the outcome is **A-alt**; otherwise **P2**.
- **K2.** P2 fails by a clear reversal, or the total reduction 450 → 5 on SK at patch is below 5 %. Same consequence as K1.
- **F1.** CT-p5 ≤ −10 % with N_σ ≥ 3 and P2 holds: outcome **A′ strong**.
- **A′ weak.** −10 % < CT-p5 ≤ −5 %, clear, and P3 confirmed. The claim is then worded as allocation between channels, not as a gain.
- **F4 / K3 (probe robustness).** The sign of CT-p5 under the MLP probe (mean of 3 probe seeds) and under the attentive probe (mean of 3) must equal the Ridge sign. A clear opposite sign under the attentive probe gives outcome **P2** with "probe-dependent conclusions" as a fourth pitfall. Non-clear differences are reported only.
- **F5 (wording, registered both ways).** Compare CT-p5 on SK at 5,000 and at 10,000 steps (snapshots). If its magnitude at 10,000 is at most half of that at 5,000: "the loss scale sets the order and rate of learning". If it is at least as large: "at this budget the loss scale selects the solution". Otherwise: "undetermined at 10,000 steps".
- **K4 (data consistency).** (a) A headline LUCiD number fails blind recomputation beyond tolerance with no diagnosed definitional cause; or (b) the joint yardstick fit started at truth (§T6) has a mean bias above 5 standard errors in any coordinate on SK or WCTE. Then finish running jobs, start nothing new, and report.

| Outcome | Condition |
|---|---|
| A′ strong | F1 met; K2, K3, K4 not fired |
| A′ weak | as defined above; K2, K3, K4 not fired |
| A-alt | K1 or K2 fired and a mask contrast ≤ −10 % and clear |
| P2 | otherwise |

### 2.6 Not registered

Everything else (JUNO / HK, the budget axis beyond F5, yardsticks, the parity bridge) is exploratory and is labelled so.

---

## 3. Stages

**Order.** T0 → T1 → smokes → submit E (do not wait for anything else) → T2, T3, T5, T6 in parallel with training → T4, T7 as runs finish → T8. Commit early and often; push after every stage.

### T0 — Preflight (≤ 30 min)

1. Create `$CC_STATE/status/r2-tier1.json` (running) and the notes file. Commit this runbook and the copied `CLAIMS_LEDGER_2026-11-22.md` (the launcher places both in `docs/round2/`).
2. Clone or update `/ceph/dwong/jepa/r2_tier1_repo` at the branch head.
3. Confirm the ten facts of §0.3 against the code and the returns tree (one line each in the notes). If one is wrong, record it and adapt; do not stop.
4. Inventory: GPU occupancy on `dt2`, batch queue, portal1 cores, Ceph free space; existence of the cache pickles of §0.3 item 6; `micro_batch`, window and memory fraction from `recipe.json` of `sk__mae_repaired__s9000` and `wcte__mae_repaired__s9000`.
5. Commit `PREREG_2026-11-22_R2_TIER1.md` (§2 verbatim) and push. Record the commit in the notes.

### T1 — Code changes, each with a focused test

1. **`random` split** in `lucid_views.py`: hide `n_pmt // 2` PMTs chosen by `keyed_rng(seed, NS_VIEWS, "random", replicate, event_id).permutation(n_pmt)`; accept the view when the hidden hits ≥ `n_min`, as `spatial_patch` does. Add it to `SPLITS`, `mask_from_key` and `ViewMasker`.
   - Tests: deterministic per event id; different across events; exactly `n_pmt // 2` hidden; `within` and `spatial_patch` masks for 200 event ids identical to the base commit (golden file written before the change).
2. **`--mask {spatial_patch,random,within}`** for arm `mae_repaired` in `ext_r2_repair_train.py` (default `spatial_patch`). The encoder configuration and `shuffle=True` stay those of `mae_repaired` for every mask.
3. **`--charge-loss-weight`** (default 1.0) multiplying `loss_charge` in `lucid_arms._forward_mae_mse`; log it and the raw per-term losses.
4. **Run names and recipe.** Suffixes in this order, each omitted at its default: `__m<mask>`, `__ts<unit>` (existing rule), `__cw<weight>`. Add the new fields to the recipe only when non-default. Refuse to write under the `r2_lucid_2026-11-21` returns.
5. **Table builder (D1, D7).** In `scripts/lucid_toy_vs_lucid.py`, key by run id; main tables take default-recipe runs only; variants go to a separate sweep table. Remove the "Approximate ToF-only ceiling" lines, or relabel them "crude fit, not a bound". Add a regression test with a fake tree holding a default run and a sweep run. Rebuild `TOY_VS_LUCID.md/.json` into `results/round2/20261122_tier1/` (do not overwrite the `20261121_lucid` copy) and list the cells that changed.
6. **Default-recipe hash test.** The recipe hash for (sk, `mae_repaired`, 9000, defaults) equals `config_sha256` in the existing run's `run_record.json`; the same for WCTE.
7. **Evaluation entry point** `scripts/tier1_eval.py`: one run (final checkpoint or a snapshot step) → one JSON keyed by run id and step, with every metric of §T3.2 on the requested test cohort, bootstrap resamples saved, and a line appended to `SEALED_ACCESS_LOG.md` when the sealed file is read.
8. Run the focused tests and the native suite. Then one independent fresh-context review (subagent) of the diff with this checklist: defaults unchanged (item 6); one-factor property (P3); no analytic-generator physics; run-identity keys (P6); sealed access only through item 7. A BLOCK means fix and review again.

### Smokes

A 20-step batch smoke for SK with `--mask random` and one with `--time-loss-scale-ns 450`, to confirm memory and project runtime. WCTE: one 20-step smoke per new flag on `dt2`.

### E — Training matrix

All cells are arm `mae_repaired` with flags. Submit in priority order. SK goes to HTCondor A100 (walltime request 10 h); WCTE goes to the guarded `dt2` lanes, free Titans, and idle batch slots. No-wait failover as in `r2-lucid` §L6b.

**SK (≈ 5 GPU-h each).** Existing: (patch, 50, A).

| Priority | ID | Mask | Unit | Seed | Purpose |
|---|---|---|---|---|---|
| 1 | SK-1 | patch | 450 | A | CT-p50, CM-450 |
| 1 | SK-2 | random | 450 | A | CT-r50, CM-450 |
| 1 | SK-3 | random | 50 | A | CT-r50, CM-50 |
| 2 | SK-4 | patch | 450 | B | seed |
| 2 | SK-5 | patch | 50 | B | seed (D3) |
| 2 | SK-6 | random | 450 | B | seed |
| 2 | SK-7 | random | 50 | B | seed |
| 3 | SK-8 | patch | 5 | A | CT-p5 |
| 3 | SK-9 | patch | 5 | B | CT-p5 |
| 4 | SK-10 | patch | 15 | A | P2, P4 |
| 4 | SK-11 | patch | 150 | A | P2, P4 |
| 5 | SK-12 | parity | 450 | A | bridge to the legacy MAE |
| 5 | SK-13 | patch | 50, charge weight × 100 | A | P5 |
| 6 | SK-14 | patch | 15 | B | seed |
| 6 | SK-15 | patch | 150 | B | seed |
| 7 | SK-16 | patch | 1.5 | A | optional; only if ≥ 6 GPU-h remain at the end |

**WCTE (≈ 0.5 GPU-h each).** Existing: patch at 450 A, 50 A, 50 B, 15 A, 5 A, 1.5 A.

| Priority | Cells | New runs |
|---|---|---|
| 1 | patch × {450, 50, 5, 1.5} × {A, B, C}, missing ones | 7 |
| 1 | random × {450, 50, 5, 1.5} × {A, B, C} | 12 |
| 2 | patch × 15 × {B, C} | 2 |
| 3 | parity × {450, 5} × A; patch 50 A with charge weight × 100 | 3 |

**Rules.**
- Use the micro-batch, window and memory fraction of the existing run of that detector (T0 item 4).
- Book every job in the ledger with its full run id.
- A failed job is diagnosed, fixed and resubmitted once; a second failure of the same cell is recorded and skipped.
- Conditional cells: the borderline rule of §2.2, and the median-reference cell of §T5.1.

### T2 — Sealed cohorts, readout freeze, verification manifest A

1. **Generate** cohort `sealed_test` for SK and WCTE first, then JUNO and HK: 30,000 events per detector (600 runs × 50; 1:1:1 photon / μ / e), detector-state law and setting as `dev_probe_test` (s = 0.05), no `_s030` twin, same LUCiD commit, tier A1 and configuration. The cohort name must enter the key derivation so that event keys are disjoint from every existing cohort.
2. **Validate** each sealed cohort: timing-residual peak and late-light fraction (gate G2) and regime quantiles (G4) within 10 % of `dev_probe_test`; no analytic physics in the path (G6); event-id and run-key disjointness from all existing cohorts. Pack to `<det>_lucid_v1_sealed.h5` with manifest and SHA256SUMS. Never rewrite an existing pack. Add the cohort to `docs/DATA_LINEAGE.md`.
3. **Sealed rules (binding).**
   - Probes are fitted on `dev_probe_train`, selected on `dev_probe_test` (from now on the validation split), and applied to `sealed_test`.
   - No readout, feature, cut, hyper-parameter or bug-fix decision may use sealed results. A bug found later is fixed on validation evidence, and the re-read is logged.
   - Only `scripts/tier1_eval.py` reads sealed packs, and it logs every read in `results/round2/20261122_tier1/SEALED_ACCESS_LOG.md` (UTC, commit, run id, purpose).
4. **Verification manifest A** (metric-free), `results/round2/20261122_tier1/verify_inputs/VERIFY_INPUTS_A.json`, pushed as soon as SK and WCTE sealed packs exist (update it when JUNO and HK are added). For every existing LUCiD case of the `r2-lucid` headline table (four detectors: legacy init seeds 9000 and 9001; `mae_mse`, `mae_repaired`, `jepa_repaired`, `sigreg_jepa` where trained; WCTE seed 9001 runs) and for each run's step-0 snapshot (matched init), give: run id, detector, arm, seed, step, checkpoint path and SHA256, encoder configuration, scaling, how the model is constructed, pack paths and SHA256 (`dev_probe_train`, `dev_probe_test`, sealed), cohort names. **No metric values, no result paths.**
5. Write `docs/round2/READOUT_SPEC_v1.md` from Appendix A plus the SHA256 of `r2_probe_repair.py`, and commit it before the first sealed read.

### T3 — Re-score the existing LUCiD checkpoints (D8, D9, D12, D13)

1. Cases: every case of manifest A.
2. Metrics, each on `dev_probe_test` and on `sealed_test`, with bootstrap intervals:
   - **Vertex (photon bombs):** Ridge all-mean; Ridge spatial-moment (primary); raw-feature baseline; MLP on `spatial_moment_full` and on the latent-only features, 3 probe seeds each; attentive probe (Appendix A.4), 3 probe seeds.
   - **Direction (μ, e):** Ridge on the unit vector from the spatial-moment features, renormalised; median and 68 % angle. Also the committed MLP direction probe, for continuity.
   - **Energy (μ, e; LUCiD-contained):** Ridge on ln E from the spatial-moment features; report the 68 % half-width and RMS of (Ê − E)/E, and R² for continuity.
   - **PID:** logistic regression on the spatial-moment features; μ mis-identification at 99 % e efficiency, e mis-identification at 99 % μ efficiency, AUC.
   - **Collapse:** f_event, f_layout.
3. Document which features and probes the committed particle suite used (file:line), so that old and new numbers can be related.
4. **Matched init:** score every run's step-0 snapshot. Report gains over the matched init and, separately, over the legacy init.
5. **WCTE energy (D13):** report how many μ and e are contained under LUCiD's flag, and under one alternative that you define and record (for example, true track end inside the tank). Do not use WCTE energy in the gate.
6. Output: `RESCORE_LUCID.md` and `.json` (keyed by run id and step).

### T4 — Budget axis (D10)

1. Score snapshots 0, 500, 1000, 2000, 5000, 10000 with the Ridge spatial-moment readout on `dev_probe_test` (exploratory, not sealed) for: the five WCTE sweep runs (seed A) and the existing SK runs.
2. When SK-1, SK-4, SK-8, SK-9 and the two (patch, 50) runs are complete, score their 5,000- and 10,000-step snapshots on the sealed set for F5.
3. Output: `BUDGET_AXIS.md`, one figure (error against step, one line per time unit).

### T5 — Time reference and the registered knee

1. **Common-shift share (D11).** On `dev_probe_train` photon bombs, per detector: V_tot = variance of the time feature over all prompt hits of all events; V_shift = variance across events of the per-event median hit time. Report V_shift / V_tot, and the SD of (trigger time − earliest physics hit time) from the stored fields.
   - **Rule.** If V_shift / V_tot > 0.10 on SK or WCTE, implement `--time-reference median` (subtract the per-event median prompt-hit time before scaling, for inputs and targets) and train one extra cell per affected detector: (patch, 50, A, median). Otherwise record the value and skip.
2. **u₅₀ (P4).** From `train_log.jsonl` of the default (50 ns) `mae_repaired` run of each detector, average `time_raw`, `charge` and `hit` over the last 100 steps. u₅₀ = T_scale × √(time_raw / (charge + hit)), with T_scale the detector's `time_scale_ns`. Append the four values to the PREREG file as a dated amendment and push **before evaluating any E-cell**. Also report the same number from the plain-MAE logs, for comparison only.

### T6 — Yardsticks without an exact likelihood (D7). Time box: 8 agent-hours, ≤ 10 GPU-h

Events: photon bombs of `dev_probe_test`, SK and WCTE (300 for Y1, 200 for Y2). Parameters θ = (x, y, z, t₀, ln N_γ).

- **Y1 — surrogate Fisher split.**
  - Counts: expected PE per PMT, μ_p(θ), from LUCiD's expected (non-sampled) response for an isotropic source, the mode already used by validation gate G3, plus the dark expectation in the prompt window. F_counts = Σ_p (∂μ_p/∂θ)(∂μ_p/∂θ)ᵀ / μ_p, by autodiff or checked finite differences.
  - Time: use LUCiD's own first-photon time term as implemented in the clone (cite file:line; do not write a new likelihood). F_time = covariance of its score at θ_true over 32 independently sampled replicates of the same truth. These replicate events are diagnostics; they join no cohort.
  - Report σ_vertex = √trace of the vertex block of F⁻¹ for counts-only, time-only and joint, with nuisances kept in the matrix (drop a nuisance only where that channel cannot constrain it), and the time share σ_counts² / (σ_counts² + σ_time²). Give distributions, and medians by radius and yield tertile.
  - Label: "surrogate Fisher from LUCiD's expectation model; omits SPE smearing, charge-dependent jitter and threshold; an optimistic bound".
- **Y2 — achieved resolution.** Minimise LUCiD's reconstruction loss on the sampled events with counts only, time only and both. Start at truth (primary); for 50 events also start at the charge-weighted centroid. Report mean and median 3-D error, per-coordinate bias with standard error, and the fraction converged, next to the frozen Ridge readouts on the same events. The joint fit feeds K4(b).
- **Fallback** if LUCiD's loss cannot be evaluated for photon bombs inside the time box: (i) a time-only iterated prompt-window Huber fit with one fitted effective speed per detector and multi-start; (ii) a counts-only Poisson fit with solid angle × attenuation × cos acceptance, attenuation length fitted per detector. Label them "simple fits, not LUCiD's likelihood", never a bound, and record that the information share then waits for Tier 2.
- Output: `YARDSTICKS.md` and `.json`.

### T7 — Evaluate the E-cells and build the gate table

1. As each run completes: collect, verify the checkpoint, and score it with `scripts/tier1_eval.py` on `dev_probe_test` and `sealed_test` (all metrics of §T3.2; attentive and MLP probes for the SK cells of the registered contrasts and all WCTE cells).
2. As soon as SK-1 … SK-7 are scored, commit `INTERIM_E1.md` with the four un-bundling contrasts.
3. **Verification manifest B** (metric-free): `verify_inputs/VERIFY_INPUTS_B.json`, same fields as manifest A for every E-cell and its step-0 and 5,000-step snapshots, plus the list of registered contrasts by run id. Push it when the SK priority 1–3 cells and the WCTE priority 1 cells are complete; update it as later cells finish.
4. `GATE_TABLE.json` and `.md`: every registered contrast with T̄, per-seed values, SE_boot, SE_seed, N_σ, clear or not; the secondary outcomes; P1–P6, F1, F4, F5, K1–K4 verdicts; the outcome of §2.5.

### T8 — Gate report, ledger proposal, acceptance

1. `results/round2/20261122_tier1/GATE_G1.md` (≤ 3 pages plus tables):
   - the outcome and the registered evidence for it;
   - un-bundling (mask × unit, one point per seed), dose-response on SK and WCTE with u₅₀ marked, energy against unit, budget axis, yardstick ladder (figures under `figures/`);
   - the defect table of §1 with status;
   - the blind-recomputation status from `origin/tier1-verify` if its reports exist, otherwise "pending";
   - deviations from this runbook.
2. `CLAIMS_LEDGER_PROPOSAL_2026-11-27.md`: claim-by-claim changes to ledger v2. Do not edit the ledger itself.
3. `docs/round2/ACCEPTANCE_2026-11-22_R2_TIER1.json`: one PASS / FAIL / SKIPPED line per deliverable of §4 and per defect of §1.
4. Set the status file to `done`, write the phase report, and print it in full.

### T9 — Housekeeping (≤ 30 min, last)

Triage `test_rate_zero_is_identical_to_the_pre_r6_reference`: decide whether the reference hash is stale or the behaviour changed, and record the finding in `DECISIONS.md`. Change no behaviour.

---

## 4. Deliverables (all on `origin/tier1`)

`PREREG_2026-11-22_R2_TIER1.md` with the u₅₀ amendment · `READOUT_SPEC_v1.md` · sealed-cohort manifests and `SEALED_ACCESS_LOG.md` · `verify_inputs/VERIFY_INPUTS_A.json` and `_B.json` · rebuilt `TOY_VS_LUCID.md` · `RESCORE_LUCID.md` · `BUDGET_AXIS.md` · `YARDSTICKS.md` · `INTERIM_E1.md` · `GATE_TABLE.json/.md` · `GATE_G1.md` · `CLAIMS_LEDGER_PROPOSAL_2026-11-27.md` · `ACCEPTANCE_2026-11-22_R2_TIER1.json` · updated `RESOURCE_LEDGER.json`, `DECISIONS.md`, `DATA_LINEAGE.md`.

## 5. Targets and stops

| When (CET) | Target |
|---|---|
| Mon 23 Nov, 03:00 | PREREG pushed; T1 reviewed; first SK wave and WCTE lanes running |
| Mon 23 Nov, evening | Sealed SK and WCTE validated; manifest A pushed; u₅₀ amendment pushed |
| Tue 24 Nov | All trainings complete; `INTERIM_E1.md`; manifest B pushed |
| Wed 25 Nov | `RESCORE_LUCID.md`, `BUDGET_AXIS.md`, `YARDSTICKS.md`; draft gate table |
| Fri 27 Nov, 08:00 | `GATE_G1.md` and acceptance on `origin/tier1` |

A late item is reported, not waited for: commit what exists on Fri 27 Nov 08:00, then finish the rest.

**Stops.** K4 as registered. GPU cap reached: stop submitting, finish and report. A sealed-rule violation: record it in the access log and in the report; never hide it. Nothing else stops the phase.

## 6. After the gate (information only; separate runbooks follow)

| Dates | Work |
|---|---|
| 28 Nov – 5 Dec | Tier 2. If A′: noise-matched loss, WCTE below-resolution units, 20k-step extension, timing-smear knob (CPU re-pack of cached digits). If P2: JEPA ablations and seeds. |
| 6 – 19 Dec | Tier 3: supervised yardsticks, JUNO / HK transfer, theory appendix. |
| 20 Dec – 3 Jan | Full draft; number freeze with every abstract number at V. |
| 18 Jan 2027 | CPAL Recent Spotlight submission; journal submission the same week. |

---

## Appendix A — Frozen readout definitions

Token features x_p ∈ ℝ⁹ per PMT: scaled position r_p (indices 0–2), direction (3–5), ℓ_p = ln(max(q_p, 10⁻⁶ PE) / 1 PE) for hit PMTs (index 6), τ_p = t_p / T_scale (index 7), hit flag h_p (index 8). z_p ∈ ℝ¹²⁸ is the token embedding from `encode_tokens` of the full, unmasked event. N is the number of PMTs.

**A.1 All-mean.** z̄ = (1/N) Σ_p z_p.

**A.2 Spatial-moment latent (768 features), the primary readout.** Let H = {p : h_p = 1} (all tokens if H is empty) and P = {p ∈ H : τ_p ≥ 0}.
- m₁ = mean of z_p over H.
- m₂ = Σ_H w^q_p z_p with w^q_p ∝ exp(3 ℓ_p), normalised over H.
- m₃ = Σ_P w^e_p z_p with w^e_p ∝ exp(3 ℓ_p − 4 (τ_p − min_P τ)), normalised over P; if P is empty, m₃ = m₂.
- C = (1/|H|) Σ_H (r_p − r̄) z_pᵀ ∈ ℝ³ˣ¹²⁸, with r̄ the mean of r_p over H.
- Feature vector: [m₁, m₂, m₃, vec(C)].

**A.3 Raw-feature baseline (33 features).** The construction of A.2 with z_p replaced by (ℓ_p, τ_p, h_p), plus 15 summaries: |H|/N; ln max(|H|, 1); mean and SD of ℓ over H; mean and SD of τ over P; r̄; Σ w^q_p r_p; Σ w^e_p r_p. `spatial_moment_full` is A.2 plus the same 15 summaries.

**Ridge protocol.** Features standardised on the training split; one Ridge per target coordinate with intercept; α ∈ {10⁻⁶, 10⁻⁴, 10⁻², 1, 100} chosen by five-fold cross-validation with folds grouped by run; vertex error = mean Euclidean distance on the test split.

**A.4 Attentive probe (new; frozen here).** Per event, at most 512 hit tokens (keyed subsample by event id). Eight heads: score s_{h,p} = q_hᵀ W z_p / 8 with W: 128 → 64 and learned q_h; a_{h,p} = softmax over the event's tokens. Each head outputs [Σ_p a_{h,p} V z_p, Σ_p a_{h,p} r_p] with V: 128 → 32. The 8 × 35 values pass through LayerNorm and one linear layer to the standardised target. AdamW (lr 10⁻³, weight decay 10⁻²), batch 256, cosine schedule, ≤ 200 epochs, early stopping on the validation split with patience 25, 3 seeds.

**MLP probe.** As committed (`r2_probe_repair.fit_mlp`), unchanged, with 3 probe seeds.

---

## Review log

- **Round 1 (author's assistant, 22 Nov).** Checked against the code on `lucid-mc`: four-factor bundle and non-matched init (`ext_r2.py:134–177`, `ext_r2_repair_probes.py:81`); loss-weight-only time unit; fixed-parity `within` mask; snapshot schedule; cohort sizes; cached digits; builder key collision; measured GPU costs. SK runs cost ≈ 5 GPU-h, not the 3 assumed in the review, so the cap is 110 GPU-h with a priority list.
- **Open risks.** (i) Two seeds per SK cell: handled by the borderline rule. (ii) The yardstick may fall back to simple fits. (iii) The weekly Claude usage limit with two agents: the driver resumes by itself.

## Launch (Mac Terminal, one window per agent, in the foreground)

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_tier1.sh
```

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_tier1v.sh
```

Monitor with `tail -f ~/Documents/VERITAS_offrepo/claude-code/logs/r2-tier1.log` and `cat ~/Documents/VERITAS_offrepo/claude-code/notes/r2-tier1.md`.
