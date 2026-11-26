# Runbook — Sparse hit tokens on SK, then HK and JUNO: the loss fix where dilution is large, absence tokens for μ/e, the mask result on a third detector, and the window where late light matters (r2-sparse-sk), 26 Nov 2026

**Phase** `r2-sparse-sk`. **Branch** `sparse-sk`, created from `origin/sparse2`. **Worktree** `~/Documents/VERITAS-worktrees/sparse-sk`. **Results** `results/round2/20261126_sparse_sk/`.

No other phase is running. All earlier phases are done; their branches, clones, returns and data are read-only for this phase.

**Three stages in one phase:** Stage S (SK, §0–§3), then Stage H (HK, §4) and Stage J (JUNO, §5). Stage H's design depends on Stage S's decision; Stage J's CPU re-pack may run during Stage S.

---

## 0. Why this runbook exists

`r2-sparse2` (WCTE, 3 seeds, sealed, frozen spatial-moment Ridge) found, on equal components:

| Cell | Vertex error | vs C2 | Notes |
|---|---|---|---|
| C2 (dense, fixed loss) | 0.0549 m | | |
| S1L (sparse twin of C2) | 0.0533 m | −2.9 %, clear | μ dir −12 %, e dir −9 %, both clear; learned (step 0: 0.0742 m) |
| S2L (time-ordered, position prediction) | 0.0550 m | +0.2 % | μ dir −18 %, e dir −15 %, both clear; vertex parity already at step 0 |

The registered decision fired for S1L: scale the sparse family to SK. Three open points go with it:

1. **Dilution.** WCTE is the least diluted detector (≈ 56 % occupancy). On SK at least ≈ 78 % of PMTs are unhit in a median photon bomb, so both the loss fix (F2) and sparse tokens should matter more there. The loss fix has never been tested on SK.
2. **μ/e identification is the sparse family's weak point.** μ mis-ID at 99 % e efficiency: S1L +21 % (not clear), S2L +179 % (clear). The likely cause is that the sparse encoder has no explicit view of where PMTs stayed dark.
3. **Time order.** On WCTE a 32-hit block spans a median 0.40 ns (one LUCiD TDC tick), so time order carries little. SK is ≈ 10 times larger.

This phase tests all three on SK with 3 seeds, plus one cheap WCTE cell for point 2.

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `sparse-sk`:

1. **Author, 26 Nov 2026, 02:32 CET:**
   > "现在写这份 SK 的 runbook，后面都还有哪些步骤，只有sk吗，hk和juno呢？"

   (Translation: "Write this SK runbook now. What steps follow; only SK, what about HK and JUNO?") This approves the SK follow-up proposed at 02:31 CET, including absence context for μ/e.
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决". Never stop to ask; record decisions in `DECISIONS.md` and `$CC_STATE/notes/r2-sparse-sk.md`.
3. **Author, 22 Nov 2026:** do not depend on the collaborator.
4. **Author, 26 Nov 2026, 02:37 CET:**
   > "改runbook，把hk和juno append到sk后面"

   (Translation: "Revise the runbook: append HK and JUNO after SK.") This authorises Stages H and J (§4, §5).

### 0.2 Binding rules

- **Real LUCiD data only.**
  - SK: the `lucid_v1` SK packs (window [−20, +270] ns), the packs every earlier SK cell trained and was scored on, so all SK contrasts stay paired with Tier-1 and r2-mask. The SK window is not changed in this phase (§2.7).
  - WCTE: `lucid_v1_w2`, as r2-sparse2.
  - Never use the analytic generator or any analytic-world pack or checkpoint. Assert `dataset_kind` and LUCiD `e651e836` in every recipe.
- **Reuse, read-only:** SK `mae_repaired` default (patch 173, legacy loss) seeds 9000/9001/9002 from r2-lucid, r2-tier1 and r2-follow; WCTE C2 and S1L from r2-sparse and r2-sparse2.
- **Never overwrite** existing packs, caches, checkpoints or returns. New trainings write only under `/ceph/dwong/jepa/returns/r2_sparse_sk_2026-11-26/` (`R2_REPAIR_OUTPUT_ROOT=<this>/train`).
- **Pre-registration.** Copy §2 verbatim to `docs/round2/PREREG_2026-11-26_R2_SPARSE_SK.md`. Commit and push **before any new run is evaluated**. Later changes are dated amendments appended below, never edits.
- **Sealed set:** rules of `RUNBOOK_2026-11-22_R2_TIER1_GATE.md` §T2.3. Sealed packs only through `scripts/tier1_eval.py`; every read logged in `results/round2/20261126_sparse_sk/SEALED_ACCESS_LOG.md`.
- **Frozen readouts:** `READOUT_SPEC_v1.md` unchanged, with the r2-sparse readout adapter for sparse models (scatter into the full PMT array, active = all PMTs; absence tokens of §2.3 are never part of the readout).
- **Resources.**
  - HTCondor: at most 4 concurrent jobs from this phase.
  - deepthought2: at most one lane, only when a slot is free.
  - CPU: ≤ 6 cores on `dt1` at `nice -n 15`; portal1 up to one sixth of its cores, leaving ≥ 8 free.
  - JAX on shared GPUs: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`.
- **Safety:** main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Push only to `origin/sparse-sk`, fast-forward only. LUCiD: internal research use only.
- **Notes.** Re-read `$CC_STATE/notes/r2-sparse-sk.md`, including any AUTHOR ADDENDUM in it, at every poll. Poll at most every 25 minutes while waiting.

### 0.3 Paths and budget

| What | Path |
|---|---|
| Remote clone | `/ceph/dwong/jepa/r2_sparse_sk_repo` |
| Returns | `/ceph/dwong/jepa/returns/r2_sparse_sk_2026-11-26/` |
| Scratch | `/work/dwong/r2_sparse_sk/` |

GPU ledger line `R2_sparse_sk_gpu`, cap **60 GPU-h**:

| Item | ≈ GPU-h |
|---|---|
| SK dense C2, 3 runs (≈ 5.3 each, as the SK default) | 16 |
| SK sparse S1L, S1A, S2L, 9 runs (fewer tokens; smokes set the estimate, assume ≈ 3 each) | 27 |
| WCTE S1A, 3 runs | 2 |
| Scoring (SK on A100 batch; WCTE on CPU) | 9 |
| Margin | 6 |

If the projection after smokes exceeds the cap, drop in this order: SK-S2L seed 9002, then SK-S2L entirely. Never drop SK-C2 or SK-S1L.

Tooling, hosts and batch layout follow r2-mask (SK batch) and r2-sparse2 (sparse code): read `$CC_STATE/notes/r2-mask.md` and `$CC_STATE/notes/r2-sparse2.md` (read-only) first.

---

## 1. Setup

### T0 (≤ 20 min)

- Status file and notes; commit this runbook; create the remote clone at the branch head.
- Inventory: condor queue, dt2 slots, Ceph free space; staged SK training pack and sealed pack (paths and SHA256 from r2-mask).
- Locate and record, read-only, the reused run directories (SK default × 3 seeds; WCTE C2 and S1L × 3 seeds) with recipe hashes and code tarballs.
- Measure on SK `lucid_v1` `dev_train` photon bombs (CPU): hits per event (median, q10, q90), occupancy, and the time span of 32 consecutive hits in time order. Record them; they set SK-S2L's expectation and the sparse step-time estimate.

---

## 2. Pre-registration (copy verbatim to `PREREG_2026-11-26_R2_SPARSE_SK.md`)

### 2.1 Cells (seeds A = 9000, B = 9001, C = 9002; 10,000 steps; encoder width, depth, heads and attention window of the default `mae_repaired` of each detector: SK window 384, WCTE 64; patch mask of the default `mae_repaired`, same RNG key)

**SK (`lucid_v1`):**

| ID | Tokens | Recipe | Status |
|---|---|---|---|
| SK-D0 | all PMTs | default `mae_repaired` (legacy loss, patch 173) | reused |
| SK-C2 | all PMTs | SK-D0 plus `--loss fixed --placeholder sentinel` (as WCTE C2) | new |
| SK-S1L | hit PMTs | sparse twin of SK-C2 (as WCTE S1L: C2's linear embedding, legacy full-attention decoder, queries every PMT in the masked patches, no count head) | new |
| SK-S1A | hit PMTs + absence tokens | SK-S1L plus the absence tokens of §2.3 | new |
| SK-S2L | hit PMTs, time order | as WCTE S2L (C2's linear embedding, full-attention decoder, 32-hit blocks, many-to-one position assignment) | new |

**WCTE (`lucid_v1_w2`):**

| ID | Recipe | Status |
|---|---|---|
| C2, S1L | as r2-sparse2 | reused |
| W-S1A | S1L plus the absence tokens of §2.3 | new |

New runs: 4 SK cells × 3 + 1 WCTE cell × 3 = 15.

Sentinel time for SK dense tokens: (270 + 50) ns / 450 ns = 0.711.

### 2.2 Code reuse

All new cells use the r2-sparse2 code (`166e388` lineage) plus the absence tokens of §2.3. No other change. The recipe hash test must show all 14 earlier recipes unchanged.

### 2.3 Absence tokens (S1A cells)

- **One token per visible patch** (a Hilbert patch of the full PMT list that is not masked in this event).
- **Features:**
  - scaled xyz of the patch centroid (3);
  - the patch's hit fraction (hits / PMTs);
  - ln(1 + number of hits);
  - ln(1 + Σ q / 1 PE).
- **Embedding:** its own `Linear(6, 128)` plus a learned type embedding.
- **Placement:** in the encoder's serialised order at its patch's position, so that windowed attention sees it together with that patch's hits. Hit tokens keep their C2 embedding.
- **Masked patches get no absence token.** The encoder input must not depend on masked patches' hits.
- **Readout:** absence tokens are not part of the readout. Only hit tokens are scattered into the PMT array, so the readout is unchanged.

What it adds: an explicit, coarse view of where PMTs stayed dark in the visible part of the event, at one token per patch (SK ≈ 11,146 / 173 ≈ 64 patches, about half visible).

### 2.4 Leakage tests (blocking, before production)

- (a) S1L, S1A: the queried PMT set depends on the mask only.
- (b) S1A: absence tokens of an event are unchanged when hits in masked patches are added, removed or moved.
- (c) S2L: predictor inputs are unchanged when hidden hits' positions are permuted; full attention among queries exposes only hit times.
- (d) All new cells: the gradient of the loss with respect to masked hits' input features is exactly zero.
- (e) SK-C2's recipe equals SK-D0's except `--loss` and `--placeholder` (field-by-field diff recorded).

### 2.5 Outcomes and statistics

- **Primary:** mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1, A.2), step 10,000.
- **Secondary:**
  - MLP and attentive vertex probes (3 probe seeds each);
  - μ and e direction;
  - μ mis-ID at 99 % e efficiency;
  - energy resolution where the containment flag allows it (SK);
  - gain over matched init (step 0).
- **Efficiency (reported):** training step time and peak memory per cell, measured on the same GPU type (A100), median over steps 1,000–2,000.
- **Statistics** as in `PREREG_2026-11-22_R2_TIER1.md` §2.2: T per seed, T̄, SE_boot (2,000 paired resamples), SE_seed = SD_pooled / √3, N_σ, "clear" = N_σ ≥ 2 and one sign in every seed. SK SD_pooled over all registered SK contrasts with 3 seeds (r2-mask, r2-follow and this phase); WCTE SD_pooled as r2-sparse2 plus this phase.

### 2.6 Registered contrasts and predictions

| ID | Contrast | Prediction | Reading |
|---|---|---|---|
| K1 | SK-C2 vs SK-D0 | SK-C2 better, clear | the loss fix where dilution is large |
| K2 | **SK-S1L vs SK-C2** | SK-S1L not worse by more than 5 % | replication of WCTE P1 on SK |
| K3 | SK-S1L vs SK-D0 | no sign registered | sparse tokens and loss fix together, against the Tier-1 default |
| K4 | SK-S2L vs SK-C2 | vertex: no sign; μ and e direction: SK-S2L better, clear | replication of WCTE S2L's direction gains |
| K5 | SK-S1A vs SK-S1L | μ mis-ID lower, clear; vertex not worse by more than 2 % | absence tokens fix the μ/e weakness |
| K6 | W-S1A vs S1L (WCTE) | as K5 | the same on WCTE |

Margins as in r2-sparse §2.5: a "not worse by more than x %" prediction holds if T̄ ≤ +x %, fails if T̄ > +x % and the contrast is clear, and is inconclusive otherwise.

### 2.7 Decision and scope

- **Sparse tokens become the default tokenisation** for the next detector (HK) if K2 holds and either K5 holds or SK-S1L's μ mis-ID is not clearly worse than SK-C2's.
- **The loss fix becomes the default dense loss** if K1 holds.
- **Otherwise** report; the WCTE result stands as WCTE-only.
- **The Stage S decision sets the tokenisation T\* for Stages H and J** (§4.1): SK-S1A if K2 and K5 hold; else SK-S1L if K2 holds; else dense C2.
- **Not in this phase:**
  - a wider SK window (late light beyond +270 ns is 0.8 % of first-per-PMT signal digits; a re-pack from cache is possible later and would break pairing with all earlier SK cells);
  - sparse JEPA.

Everything else is exploratory and is labelled so.

---

## 3. Stages

Order: T0 → T1 code → leakage tests and review → PREREG push → smokes (SK A100: SK-C2, SK-S1L, SK-S1A, SK-S2L; WCTE dt2: W-S1A) → E → T7 → T8.

### T1 — Code

1. Absence tokens (§2.3) behind `--absence-tokens` for `sparse_mae`; refused for `sparse_time` and for dense arms.
2. SK support for the sparse arms on `lucid_v1`: dataset kind, readout adapter, evaluator path; a golden test that the adapter reproduces the dense spatial-moment features on SK for 50 events.
3. Step-time and memory logging (median over steps 1,000–2,000) in every new training.
4. Leakage tests (a)–(e); recipe hash test (14 earlier recipes unchanged); native suite; one fresh-context review with the checklist: real LUCiD data only; defaults unchanged; leakage tests; readout unchanged; sealed access only through the evaluator. A BLOCK means fix and review again.

### E — Trainings

- HTCondor A100 (≤ 4 concurrent), in this order: SK-C2 × 3, SK-S1L × 3, SK-S1A × 3, SK-S2L × 3.
- dt2 lane: W-S1A × 3.
- A failed run is diagnosed, fixed and resubmitted once; a second failure is recorded and the cell is skipped.

### T7 — Evaluation and tables

1. Score every new run at steps 0 and 10,000 on validation and sealed: the primary readout, the probes, the direction, PID and SK energy suite.
2. `SPARSE_SK_TABLE.md` and `.json`:
   - seed means for every cell;
   - K1–K6 with T̄, per seed, SE_boot, SE_seed, N_σ, clear and verdicts;
   - secondary outcomes; matched-init gains;
   - step time and memory;
   - the decision.
3. **Exploratory:**
   - the time span of SK-S2L's 32-hit blocks;
   - the own-block mass fraction against chance (as r2-sparse2's f_own);
   - μ mis-ID versus event brightness for SK-C2, SK-S1L, SK-S1A.

### T8 — Report

`results/round2/20261126_sparse_sk/SPARSE_SK_REPORT.md` (≤ 1.5 pages plus tables):
- whether sparse tokens hold on SK (K2, K3);
- whether the loss fix helps where dilution is large (K1);
- whether absence tokens fix μ/e (K5, K6);
- what time order adds on SK (K4);
- efficiency;
- the decision; deviations.

Commit and push the Stage S report and print it in full. Do not set the phase `done`: Stages H and J follow (§4–§6).

**Target:** Stage S report by **Sat 28 Nov, 12:00 CET**. Commit what exists then, and finish the rest.

**Stops:** none beyond the safety rules and the GPU cap. A failed leakage test blocks production of that cell until fixed.

---

## 4. Stage H — HK: the mask result on a third detector, and sparse tokens where they matter most

### 4.0 Why

- **The mask result.** On SK the vertex gain follows the post-interpolation residual U exactly (ρ = 1.0, 3 seeds); on WCTE it does not (the gain peaks at a moderate patch and falls). A third detector decides whether the lead claim is SK-specific. HK is a real planned detector; r2-mask's M1 already measured its shortcut budget (exploratory): 19,746 PMTs, median spacing 1.00 m, default patch 308; U = 0.577 (random) and 0.594 (patch 308).
- **Sparse tokens.** At least ≈ 87 % of HK PMTs are unhit in a median photon bomb, so sparse tokens save the most compute here. A dense HK training costs ≈ 14 GPU-h.

### 4.1 Start condition and tokenisation

Stage H starts once Stage S's decision (§2.7) is recorded in `DECISIONS.md`. T\* is SK-S1A, SK-S1L or dense C2, as §2.7 says. Record T\* in the HK PREREG appendix before any HK training.

### 4.2 Data

The `lucid_v1` HK packs (window [−20, +460] ns), paired with the Tier-1 HK cells. The window is not changed (late light beyond +460 ns is 1.3 % of first-per-PMT signal digits). Real LUCiD data only; same assertions as §0.2.

### 4.3 M1 on HK (CPU, before any HK training)

With the r2-mask M1 script and settings unchanged, compute U on HK photon bombs (and μ, e, exploratory) for random, patch 77, patch 308 and patch 1232. Record all four U values in the HK PREREG appendix.

### 4.4 Cells (3 seeds 9000/9001/9002; 10,000 steps; HK default encoder, window 640)

**If T\* is sparse (S1A or S1L):**

| ID | Tokens | Mask | Status |
|---|---|---|---|
| H-C2 | all PMTs, dense C2 recipe | patch 308 | new × 3 |
| H-T-rnd, H-T-77, H-T-308, H-T-1232 | T\* | random, patch 77, 308, 1232 | new × 3 each |

15 runs: 3 dense (≈ 43 GPU-h) and 12 sparse (estimate from HK smokes).

**If T\* is dense:**

| ID | Tokens | Mask | Status |
|---|---|---|---|
| H-C2 | dense C2 recipe | patch 308 | new × 3 |
| H-C2-rnd, H-C2-1232 | dense C2 recipe | random, patch 1232 | new × 2 seeds (9000, 9001) |

7 dense runs (≈ 100 GPU-h). The mask part is then exploratory (2 seeds, 3 configurations).

### 4.5 HK predictions (copy to `PREREG_2026-11-26_R2_SPARSE_SK.md` as appendix H, with T\* and the U values, before any HK run is evaluated)

| ID | Contrast | Prediction |
|---|---|---|
| H1 | H-T-308 vs H-C2 (sparse branch only) | not worse by more than 5 % |
| H2 | each patch vs random (T\*) | T̄ < 0 and clear, 3 seeds |
| H3 | Spearman ρ between U and the seed-mean vertex gain over matched init, 4 configurations | ρ ≥ 0.8 |
| H4 | step time and peak memory, H-T-308 vs H-C2 (A100) | reported |

**Reading H3.** If H3 holds, the lead claim may say "on SK and HK". If it fails while H2 holds, the claim stays "on SK; on WCTE and HK the gain saturates or turns over beyond a detector-dependent scale".

### 4.6 HK budget and order

- Separate ledger line `R2_hk_gpu`, cap **110 GPU-h**: dense 43 (or 100 in the dense branch), sparse ≈ 36 (12 runs), scoring 10, margin as remains.
- HTCondor A100, ≤ 4 concurrent jobs for the whole phase. Order: H-C2 seed 9000 and H-T-308 seed 9000 first (they give H1 and the efficiency number early), then the rest by seed.
- If the projection exceeds the cap, drop H-T-77 first, never H-C2 or H-T-308.

### 4.7 HK outputs

`HK_TABLE.md` / `.json` and `HK_REPORT.md` (≤ 1 page plus tables): H1–H4, the U–gain figure for WCTE, SK and HK together (`U_GAIN_3DET.png`), and the wording the lead claim may use.

---

## 5. Stage J — JUNO: the window where late light matters most, and sparse tokens on a third geometry

### 5.0 Why

JUNO, here a water sphere of JUNO's size, is the detector whose late light the current window cuts most: 3.6 % of first-per-PMT signal digits lie beyond +200 ns (W0 of r2-sparse). Its LUCiD cache is untruncated, like WCTE's, so a wider window can be re-packed from the cache. This stage is exploratory in geometry (not the real liquid-scintillator JUNO) but registered in its contrasts.

### 5.1 J0 — JUNO window re-pack (CPU; may run during Stage S)

- Apply r2-sparse §1 to JUNO unchanged: the W1 rule for P and A, the dark-noise extension with the first-light proxy and the calibrated offset (r2-sparse DECISIONS), validation as W2 §5.
- Write the new data kind `lucid_v1_w2` for JUNO (`dev_train`, `dev_probe_train`, `dev_probe_test`, `sealed_test`) under `/ceph/dwong/jepa/data/lucid_v1_w2/juno/`.
- A failed validation blocks Stage J trainings on the new packs.
- Record the window in the JUNO PREREG appendix.

### 5.2 Cells (3 seeds; 10,000 steps; JUNO default encoder, window 320, patch 156)

| ID | Data | Tokens | Status |
|---|---|---|---|
| J-C2v1 | JUNO `lucid_v1` | dense C2 recipe | new × 3 |
| J-C2v2 | JUNO `lucid_v1_w2` | dense C2 recipe | new × 3 |
| J-Tv2 | JUNO `lucid_v1_w2` | T\* (dense T\* means this cell is dropped) | new × 3 |

### 5.3 JUNO predictions (appendix J of the PREREG, with the window, before any JUNO run is evaluated)

| ID | Contrast | Prediction |
|---|---|---|
| J1 | J-C2v2 vs J-C2v1 | no sign registered (window effect where late light is largest) |
| J2 | J-Tv2 vs J-C2v2 | not worse by more than 5 % |

### 5.4 JUNO budget and order

- Ledger line `R2_juno_gpu`, cap **40 GPU-h**: 6 dense runs ≈ 26, 3 sparse ≈ 4, scoring 4, margin 6.
- Trainings use the dt2 lane or HTCondor after Stage S's trainings are submitted, sharing the ≤ 4 concurrent HTCondor jobs.

### 5.5 JUNO outputs

`JUNO_TABLE.md` / `.json` and `JUNO_REPORT.md` (≤ 1 page plus tables): the window study and data validation, J1 and J2.

---

## 6. Phase completion

- **Targets:**
  - Stage S report by Sat 28 Nov, 12:00 CET;
  - Stages H and J by **Thu 3 Dec, 20:00 CET**.
- **Final outputs:**
  - `SPARSE_SK_REPORT.md`, `HK_REPORT.md` and `JUNO_REPORT.md` under `results/round2/20261126_sparse_sk/`;
  - one acceptance file `docs/round2/ACCEPTANCE_2026-11-26_R2_SPARSE_SK.json` covering all three stages;
  - the status set to `done`;
  - all three reports printed in full.

---

## Launch (Mac Terminal, its own window, in the foreground)

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_sparse_sk.sh
```
