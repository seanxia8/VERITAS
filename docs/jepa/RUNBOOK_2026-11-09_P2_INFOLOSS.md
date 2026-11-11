# Paper 3 P2 — drifts that destroy information (threshold, dead channels) and the price of invariance by projection

**Status: v1 (9 Nov 2026). Design not yet independently reviewed; implementation not started; not
yet accepted.** Extension `paper3_p2_infoloss_2026-11-09`, `registered: false`.

**Stage 1** (this runbook, authorised):
- trains nothing and **generates no data**;
- every factor is applied by an exact transform of the stored hits (§2.3);
- the GPU is used only for encoder forward passes, ≤ 1.3 L40S h.

**Stage 2** (§9, **not authorised**): an efficiency-loss law in the generator plus data generation.
It needs a separate author line that names its budget.

Base: `origin/JEPA = df588c6`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-11-09_P2_INFOLOSS.md` before any outcome-generating step. The design context
is the Paper 3 plan (Project doc `claude/paper3-plan-v2-repairable-2026-11-06.md`, §3 Claim 2, §8).
This runbook depends on no document outside the repository.

---

## 0. Authority

Quote both lines verbatim in `DECISIONS.md`.

1. The author, 9 Nov 2026, 00:47 CET:
   > "下一步。 写A（推荐）：写 P2 的 runbook，做阈值 / 丢 hit、坏道这类会破坏信息的漂移，把"投影掉增益方向"顺带放进去；数据生成预算到时候单独请你批。"

   This authorises Stage 1: no training, no data generation. Stage 2 needs its own author line.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

Work as in the combined runbook's front matter §F2: edit, commit and push in the Mac `JEPA` worktree
(`/private/tmp/veritas-jepa-20261107`, or a fresh worktree at `origin/JEPA`); never the main checkout;
data and GPU steps over ssh (Deepthought CPU, deepthought2 L40S through `scripts/jepa_run.sh`).

---

## 1. Why, and the questions

The pilot (results in `results/jepa/20261106_repair/`) found three things:
- module-gain drift is invertible at the input: calibration fully repairs all 19 encoders;
- label-free latent repair recovers 16–57 %, and how much tracks the encoder's equivariance
  (Spearman 0.79);
- the price of invariance for trained encoders could not be measured, because neither concept was
  linearly readable from event-mean embeddings.

Claim 2 of the plan (the response class fixes the least repair) needs drifts that **destroy**
information, where no input calibration can help.

| ID | Question | Item |
|---|---|---|
| Q-P2a | How much vertex information do threshold raises and dead channels destroy, per encoder, after the best label-free and labelled repairs? | P2-a, P2-b, P2-f |
| Q-P2b | Is the response class a property of the representation? Does equivariance measured under gain drift predict label-free latent repair under a different, information-destroying drift? | P2-e (primary) |
| Q-P2c | When part of a drift is invertible (time walk) and part is not (hit loss), does the input repair recover exactly the invertible part? | P2-c |
| Q-X3P | What does invariance to gain cost a trained encoder, if the directions that gain drift moves are projected out? Does that projection make the encoder robust to the drift? | X3-P |

---

## 2. Data, models, transforms and side effects

### 2.1 Data (read-only, pinned)

- The four probe sets of the combined runbook (Part A §2.1, same hashes, opened only through
  `ext_followup.open_probe_sets`).
- The factors of §2.3 are applied to the **s = 0.05 (N) sets**, so no gain drift is mixed in. The
  s = 0.30 sets are used only by X3-P and by X2-b for the two new models.
- **No data generation of any kind in Stage 1.**

### 2.2 Models

- **Core (15):** the 13 core models of Part A §2.3 (same hashes), plus `target_posfree9000/9001`
  (the two Part B R-2 runs; hashes from `results/jepa/20261107_shortcut/`, re-verified by X0).
- **Optional (6):** as in Part A, only if the X0 projection fits the cap.
- The two new models first get the Part A sets that later readings need:
  - N, D and S (train and test) and SWunit (test);
  - their X2-b equivariance fit (gain R²_eq), with the corrected `raw_map`.

### 2.3 Exact transforms of the stored hits

All transforms act on one event's stored hits (`pmt_id`, `charge_pe` q, `time_ns` t relative to the
old trigger). They use only the generator's own primitives:
- `lucid_bombs.leading_edge_time`;
- the imported trigger rule `lucid_bombs.pmt_time_rule()`, i.e. `nhits_trigger_time`,
  `prompt_window_mask`, n, the trigger window and the prompt window (−20, 60) ns.

Nothing is reimplemented.

1. **Threshold raise T(τ, c)**, with τ the new discriminator threshold and c the walk-calibration
   point, both in PE:
   - **keep** hits with `isfinite(leading_edge_time(q, τ·mv_per_pe))`;
   - **time:** t′ = t + [leading_edge_time(q, τ·mv) − leading_edge_time(q, c·mv)]. The stored
     residual is zero, because the stored threshold and calibration are both 0.25 PE;
   - **re-trigger:**
     - trig′ = `nhits_trigger_time(t′_kept, …)`;
     - no trigger: the event is not triggered and is dropped;
     - t″ = t′ − trig′; apply `prompt_window_mask(t″)`.
2. **Dead channels K:** delete every hit of a dead PMT (its dark hits included), then re-trigger and
   cut as in 1. Times are otherwise unchanged.
3. **Window-edge incompleteness** (the only inexactness):
   - when trig′ > 0, hits that would enter the new prompt window from beyond the old upper edge
     (t ∈ (60, 60 + trig′] ns) were never stored;
   - the number of such events and the trig′ distribution are reported per condition;
   - every reading is also given on the **exact subset** (events with trig′ ≤ 0) as a robustness
     column.

**Conditions** (all on N):

| Condition | Definition |
|---|---|
| T0.5 | τ = c = 0.5 PE (pure hit loss) |
| T1.0 | τ = c = 1.0 PE (pure hit loss) |
| T1.0w | τ = 1.0, c = 0.25 PE (hit loss plus time walk; the walk part is invertible) |
| K5, K15, K30 | 5 %, 15 %, 30 % of the 105 modules dead. Nested sets per run: the first ⌈f·105⌉ entries of a permutation from `keyed_rng(20261109, "paper3/p2/dead_modules", run_id)` |
| P10 | 10 % of the 1 995 PMTs dead per run (`"paper3/p2/dead_pmts"`, first ⌈0.1·1995⌉ of a permutation) |

Events dropped by the trigger are dropped from both members of every pair. The trigger efficiency is
reported per condition.

### 2.4 Reuse and forbidden paths

- **Reuse:**
  - `ext_followup` (as in Part A §2.4, gate callers forbidden);
  - `ext_repair`: probes, metrics, bootstrap, `raw_map`, `equivariance_fit`, `_x2c`-style repair, the
    LEACE eraser maths, `paired_run_bootstrap`;
  - the transform context of Part A §2.5, extended with a hit-**set** transform (it may drop hits and
    rewrite `time_ns`; it fails closed and restores in `finally`).
- **Forbidden:**
  - any data generation (`draw_event`, `generate_run`, `write_generated_run`, `ensure_dataset`
    outside `open_probe_sets`);
  - any optimiser step;
  - every other extension's gate-reaching function.
- **Test-only synthetic generation is allowed** (§8.3).
- **Write scope:** `<ext>` = `/ceph/dwong/jepa/returns/ext_p2_20261109/`, results in
  `results/jepa/20261109_p2/`, and `TMPDIR`.
- **Oracle information:**
  - The dead-channel lists and the condition parameters (τ, c) count as known detector state. In a
    real detector they come from monitoring, so label-free repairs may use them.
  - Truth labels enter only the refit repairs and the scoring.

---

## 3. Shared definitions

- **Embeddings, probes and metrics:** as Part A §3.2. The frozen probe P_N is fitted on N train.
  Vertex is the mean 3-D error; log-yield is the RMS resolution.
- **Uncertainty:** run-cluster paired bootstrap (1 000, seed 20261103).
- **Classification of seeded readings:** Part A §3.3.
- **Drift cost** Δ_c = v_c − v_N (frozen P_N). It is eligible if ≥ 3 mm with a CI excluding 0.
- **Recovered fraction** of repair r: ρ_r = (v_c − v_r) / (v_c − v_N), only for eligible cells.
- **Repairs:**
  - **R0:** none.
  - **R1, input repair:** defined only for T1.0w. It re-applies the walk correction at c = τ, which
    turns T1.0w into T1.0 using q alone.
  - **R2, latent repair from the known detector state (label-free):** z̃ = z_c − v_e Â_rawᵀ.
    - Â is fitted on probe-train pairs (N, c) with `fit_probe` per embedding dimension and `raw_map`.
    - Design for K and P: v_e,m = dead_m · h_e / h̄, where dead_m is the fraction of module m's PMTs
      that are dead, h_e the event's hit count after the transform, and h̄ the N-train mean.
    - Design for T-type conditions: v_e = h_e / h̄ (one column) and an intercept.
    - The equivariance R²_eq,c is reported as in Part A X2-b.
  - **R3, readout refit:** a probe fitted on the condition's own train set (labels).
  - **R5, abstention (descriptive):** score = Mahalanobis distance of z to the N-train embeddings
    (label-free). Report vertex error at coverage 100 / 80 / 50 % next to N at the same coverage.

---

## 4. X0 — preflight and reproduction

- **X0-a:**
  - data and checkpoint hashes, including the two new models;
  - transform self-checks on 200 real events per condition: an identity transform T(0.25, 0.25)
    reproduces the stored hits exactly (same hits, same times to 1e-9 ns, same trigger);
  - trigger efficiency and window-edge counts per condition;
  - the GPU projection, entries, and S1 = ledger (133.6708) + 1.3 ≤ 160, with the fail-closed ledger
    consistency check of addendum 2.
- **X0-b:**
  - reproduce three N vertex values from Part A X0-b within 1e-3 m (`mae_within_trigger9000`,
    `fix9000`, `init9000`);
  - reproduce the two new models' full-input vertex from `results/jepa/20261107_shortcut/readings.json`
    (0.12600 / 0.14054 m) within 1e-3 m.

A failure after one fix attempt is S9 for everything except X3-P on the 13 old models.

---

## 5. P2 readings (fixed now)

Per model and seed, per condition; two-sided; classes by Part A §3.3.

- **P2-a, drift cost:** Δ_c and the trigger efficiency.
- **P2-b, information destroyed:** the refit residual R_c = v_c(R3) − v_N.
  - *Information-destroying* if R_c ≥ 3 mm with a CI excluding 0.
  - Otherwise *refit-recoverable*.
  - The same reading for the raw summary features (raw-std, frozen and refit), as an
    encoder-independent reference for how much the hits themselves lost.
- **P2-c, invertible part (T1.0w only):**
  - ρ_R1 is read against the walk-only cost (v_T1.0w − v_T1.0) / (v_T1.0w − v_N), the share of the
    cost that is walk;
  - *invertible part recovered* if R1 leaves v within 2 mm of v_T1.0 (CI);
  - otherwise *not recovered*.
- **P2-d, latent repair:** R²_eq,c and ρ_R2 per eligible cell. The classes are those of Part A X2-c:
  works ≥ 0.5, fails ≤ 0.2, partial between.
- **P2-e, the Claim 2 test (primary; across the 15 core models, one value per model):**
  - x = gain R²_eq (Part A X2-b occupancy, corrected; for the two new models, §2.2);
  - y = ρ_R2 under K5, K15 and K30 separately. Models whose cell is ineligible are left out, and the
    n used is reported.
  - Spearman ρ_s with a one-sided permutation p, from 10 000 keyed permutations
    (`"paper3/p2/perm"`).
  - **Supported:** ρ_s ≥ 0.5 and p ≤ 0.05 in at least two of the three K levels.
  - **Refuted:** ρ_s ≤ 0.1 in all three.
  - Otherwise **mixed**.
  - Secondary (descriptive): the same for T1.0, and Spearman between gain R²_eq and R²_eq,K15 (is
    equivariance a property of the representation across factors?).
- **P2-f, least repair per cell (descriptive):** the cheapest rung of R0 < R1 < R2 < R3 that reaches
  ≥ 80 % of R3's recovered fraction. A cell that is *information-destroying* under P2-b is flagged
  "beyond refit (R4/R5 territory)", with its abstention curve.
- **P2-g, abstention (descriptive):** for K30 and T1.0, error at 80 % and 50 % coverage against N
  at the same coverage.

**Interpretation (fixed now):**
- *P2-e supported:* the response class carries across factors. Equivariance measured once, under
  gain drift on paired simulation, predicts which encoders a label-free latent repair can rescue
  from a different, information-destroying drift. This is Claim 2's nonlinear prediction.
- *Refuted:* latent repairability is factor-specific. Claim 2 is reduced to "measure the class per
  factor".
- *P2-b information-destroying cells* bound what any frozen-encoder repair can do. They are where
  P5 (repairable pretraining) or abstention is needed.

---

## 6. X3-P — the price of invariance by projection (CPU, plus the two new models' sets)

- **Models:** MAE × 4, T-1 × 4, frozen-target × 2, `target_posfree` × 2. Electron energy for the four
  MAE models only (cached particle sets).
- **Subspace:** the whitened gain-response map W Â_raw from the model's X2-b occupancy fit (Â on
  probe-train). Take its SVD and the top r left singular vectors U_r, for r ∈ {1, 3, 10}.
- **Erase:** z′ = z − W⁻¹ U_r U_rᵀ W (z − μ_N), with W and μ_N from N train. Apply it to the N, D and
  SWunit embeddings.
- **Controls:** 20 Haar-random rank-r bases in the whitened space, keyed
  `f"paper3/p2/x3p/{label}/{r}/{i}"`.
- **Readings** (Part A §7.2 thresholds, unchanged), per r:
  - **X3P-1, price:** ρ_E = res_erased(N) / res_std(N) for log-yield (and electron energy).
    - Price present at ≥ 1.2 with the CI excluding 0; none at ≤ 1.05.
    - Gain-specific only if ρ_E > the 95th percentile of the 20 control ratios and the CI of
      res_erased − mean(res_control) excludes 0.
  - **X3P-2, robustness:** v_erased(D) − v_erased(N) with the erased frozen probe.
    - *Invariance achieved* if |Δ| ≤ 2 mm (CI within ±3 mm).
    - Reported next to the unerased drift cost.
  - **X3P-3, dominance at D:** the calibrated unerased model (SWunit, standard probe) against the
    erased model at D, for vertex and energy; Part A X3-4 rule.
  - **X3P-4:** the ρ_E values beside the Fisher rows and the raw-feature 3.56 (descriptive).

**Interpretation:** for the r at which invariance is achieved (X3P-2), X3P-1 is the measured price of
invariance for that trained encoder, and X3P-3 says whether keeping the information and calibrating
beats it.

---

## 7. Resources and stops

- **GPU (Stage 1):** ≤ 1.3 L40S h.
  - Projection: 15 models × 14 condition sets (7 conditions × train/test), plus 14 extra sets for
    the two new models, at the measured per-set wall × 1.3. The 7 Nov X2-e job measured about 6 s per
    set; plan with 13 s.
  - Optional models are dropped first if over the cap.
  - The guard refuses any job that would cross 1.3 h.
- **CPU:** ≤ 8 wall-hours (transforms, fits, permutations, bootstrap, reports).
- **S1:** ledger + 1.3 ≤ 160, with the fail-closed ledger consistency check.
- **Entries:** ≥ 30 000 free after the projected writes. Cache each condition's embeddings as one
  file per model, not per set.
- **Hosts:** as the combined runbook §F2–§F3: at most two L40S jobs at once, each ≤ 45 % of
  memory; `setsid` for long jobs; resource monitoring.
- **STOP conditions (the only ones):**
  - S1;
  - S3 (a real specification change; any change to a reading or threshold after outcomes);
  - S4;
  - S9 (X0-b, the identity transform or the synthetic exactness test of §8.3 still fails after one
    fix attempt);
  - an import-closure failure (quarantine; re-review and a new acceptance record);
  - a data or checkpoint hash mismatch (affected items only);
  - an entries-preflight failure;
  - the cap.
  
  A STOP never stops the independent items. X3-P on the 13 old models depends only on cached
  embeddings.

---

## 8. Implementation, review, acceptance and execution

### 8.1 Implementation
- Add `src/jepa_noise/ext_p2.py`, `scripts/jepa_ext_p2.py` and `src/jepa_noise/tests/test_ext_p2.py`.
- CLI modes: `dry-run`, `preflight`, `reproduce`, `new-models` (the §2.2 sets and X2-b), `conditions`
  (encode the condition sets), `p2`, `x3p`, `report`. **No training or generation mode exists.**

### 8.2 Gate
- As Part A §9.2. The runbook sha256 and `docs/jepa/ACCEPTANCE_2026-11-09_P2_INFOLOSS.json`
  (`review_verdict == "PASS"`) must match.
- `<P>` is Part A's list plus `ext_p2`, `ext_repair`, `ext_shortcut` and `scripts/jepa_ext_p2.py`.
- The import closure is re-checked at the end of every mode.

### 8.3 Tests
- **Exactness of the transforms against the generator (test-only synthetic generation):**
  - In a pytest temporary directory, generate a small synthetic run twice with shared keyed draws:
    once with the stored configuration, and once with the factor built into the readout.
    - For T: a `GeneratorConfig` with `threshold_pe = τ` and `walk_calibration_pe = c`, via
      `dataclasses.replace`. The manifest binding in the config loader is bypassed in the test only.
    - For K and P: a detector state whose dead PMTs have gain 0.
  - The §2.3 transform of the first run must equal the second run on every triggered event (same
    hits, times within 1e-9 ns, same trigger), except events flagged as window-edge cases.
  - Failure is S9.
- **Identity transform** reproduces the stored hits; the transforms never mutate their input; dead
  sets are nested across K levels.
- **R1** maps T1.0w to T1.0 exactly on synthetic data.
- **R2:** the design construction; `raw_map` used; the identity u Â_rawᵀ = predict(u) − predict(0).
- **P2-e:** Spearman, the permutation p and the class rule on synthetic values.
- **X3-P:** the erasure removes exactly the rank-r whitened subspace; the controls have rank r.
- **Bookkeeping, forbidden paths (sentinels), the gate negatives, no overwrite.**
- The full `src/jepa_noise/tests` suite once on the final SHA (deselect the three known
  `test_guard` tests).

### 8.4 Independent review
A fresh-context reviewer reads this runbook and the diff. The checklist is blocking, with file:line
evidence:
- (i) the transforms use only the generator's primitives, and the exactness test is genuine (it
  compares against the generator, not against the transform itself);
- (ii) the window-edge accounting and the exact-subset column;
- (iii) the repairs and designs (R1, R2, R3, R5), and no truth in label-free repairs;
- (iv) the P2-e statistic, n handling and permutation;
- (v) the X3-P subspace, erasure and controls;
- (vi) the gate, closure, forbidden paths, write scope and the cost projection.

Then the lead writes the acceptance record.

### 8.5 Execution order
X0-a → X0-b → `new-models` → `conditions` (GPU) → P2 readings (CPU) → X3-P (CPU) → report. X3-P on
the 13 old models may start right after X0-a.

### 8.6 Deliverables
- `results/jepa/20261109_p2/`: `plan.json`, `reproduction.json`, `p2.json`, `x3p.json`,
  `READINGS_SUMMARY.md`, `SUMMARY.json`, figures;
- figures:
  - drift cost and the repair ladder per condition;
  - the P2-e scatter (gain R²_eq against ρ_R2) per K level;
  - abstention curves;
  - the X3-P price against r with the random-control band;
- `LARGE_ARRAYS.json`;
- a `STATE.md` section, a ledger line and the `DECISIONS.md` lines.

---

## 9. Stage 2 — not authorised (stated so the author can approve it later)

**What:**
- An efficiency-loss law in `lucid_bombs`: per-module photon detection efficiency ε_m < 1, applied as
  binomial thinning of the photoelectron count before the charge smear. Unlike hit deletion, it
  cannot be emulated from stored hits.
- Generate N probe sets (train and test) at two levels, mean ε = 0.85 and 0.7 with module spread,
  with keyed draws shared with the existing sets.

**Why:** efficiency loss is the realistic middle case. The charge scale is partly invertible by
calibration, while lost hits are not. It tests Claim 2's prediction that input calibration recovers
only the invertible part.

**Cost to state in the approval:**
- generation CPU projected from the measured walls of the 6 Nov s = 0.30 probe-set generation
  (`RESOURCE_LEDGER.json`; the agent fills in the number at preflight). Generation cap used so far:
  18.43 / 48 h;
- about 0.4 L40S h of encoding;
- a generator code change with its own review.

**It runs only after an author line that names the CPU and GPU hours.**

---

## 10. Out of scope

- Training of any kind, including P5;
- any data generation (Stage 2 needs its own line);
- attenuation drift;
- the TIDMAD arm;
- changes to the earlier runbooks or to registered predictions.
