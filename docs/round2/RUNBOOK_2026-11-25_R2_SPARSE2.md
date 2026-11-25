# Runbook — Sparse hit tokens, fair rerun on WCTE: C2's embedding and full-attention decoder, and time-width blocks for S2 (r2-sparse2), 25 Nov 2026

**Phase** `r2-sparse2`. **Branch** `sparse2`, created from `origin/r2-follow` (which contains `sparse-tok` and `mask-scale`). **Worktree** `~/Documents/VERITAS-worktrees/sparse2`. **Results** `results/round2/20261125_sparse2/`.

No other phase is running. All earlier phases are done; their branches, clones, returns and data are read-only for this phase.

---

## 0. Why this runbook exists

`r2-follow` Part A (WCTE, 3 seeds, sealed, frozen spatial-moment Ridge) showed that two components of the r2-sparse design cost accuracy on their own:

| Change from C2 | Vertex error | vs C2 |
|---|---|---|
| none (C2) | 0.0549 m | |
| query-isolated decoder + count head (C2dc) | 0.0633 m | +15.3 %, clear |
| Fourier positional code + content MLP (C3e) | 0.0591 m | +7.8 %, clear |
| both (C3) | 0.0726 m | +32.3 %, clear |

Every sparse cell of r2-sparse (S1, S1n, S1p, S2) carried both. The registered contrast R3 (S1 vs C3) compared like with like, but no sparse cell was ever compared with the best dense cell on equal terms. The isolated decoder was introduced only to stop the query set leaking hit positions in S1n and S1p. S1 and S2 do not need it:
- S1 queries every PMT in the masked patches, a set fixed by the mask alone;
- S2 queries carry only hit times, which are given anyway.

Separately, r2-sparse found that S2 learns *where* the light is but not *when*: 8 % of its predicted mass lands on hits of the query's own block, close to chance. A 32-hit block spans about 1–2 ns on WCTE (≈ 1,000 hits arrive within ≈ 40 ns), comparable to the 1-PE timing jitter (σ ≈ 0.95 ns). Adjacent blocks cannot be told apart in time, so the task cannot reward learning "when".

This phase reruns S1 and S2 with C2's components, and adds S2 with time-width blocks. All on WCTE, 3 seeds, real LUCiD data.

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `sparse2`:

1. **Author, 25 Nov 2026, 12:21 CET:**
   > "现在就写这份 runbook"

   (Translation: "Write this runbook now.") This approves the proposal of 12:15 CET: rerun S1 and S2 with a full-attention decoder, and give S2 time-width blocks.
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决". Never stop to ask; record decisions in `DECISIONS.md` and `$CC_STATE/notes/r2-sparse2.md`.
3. **Author, 22 Nov 2026:** do not depend on the collaborator. The sparse, time-ordering and position-prediction ideas are his; nothing here waits for him.

### 0.2 Binding rules

- **Real LUCiD data only:** `lucid_v1_w2` (window [−110, +50] ns), exactly the packs r2-sparse and r2-follow trained on. Never use the analytic generator or any analytic-world pack or checkpoint. Assert `dataset_kind lucid_v1_w2` and LUCiD `e651e836` in every recipe.
- **Reuse, read-only:** C2 (×3) from r2-sparse, S1 and S2 (×3) from r2-sparse. Their checkpoints and evaluations are read in place.
- **Never overwrite** existing packs, caches, checkpoints or returns. New trainings write only under `/ceph/dwong/jepa/returns/r2_sparse2_2026-11-25/` (`R2_REPAIR_OUTPUT_ROOT=<this>/train`).
- **Pre-registration.** Copy §2 verbatim to `docs/round2/PREREG_2026-11-25_R2_SPARSE2.md`. Append the chosen block width Δ if W1 changes it (§2.3). Commit and push **before any new run is evaluated**. Later changes are dated amendments, never edits.
- **Sealed set:** rules of `RUNBOOK_2026-11-22_R2_TIER1_GATE.md` §T2.3. Sealed packs only through `scripts/tier1_eval.py`; every read logged in `results/round2/20261125_sparse2/SEALED_ACCESS_LOG.md`.
- **Frozen readouts:** `READOUT_SPEC_v1.md` unchanged, with the r2-sparse readout adapter for sparse models (scatter into the full PMT array, active = all PMTs).
- **Resources.**
  - deepthought2: at most one lane, only when a slot is free.
  - HTCondor: at most 4 concurrent jobs from this phase.
  - CPU: ≤ 6 cores on `dt1` at `nice -n 15`; portal1 up to one sixth of its cores, leaving ≥ 8 free.
  - JAX on shared GPUs: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`.
- **Safety:** main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Push only to `origin/sparse2`, fast-forward only. LUCiD: internal research use only. Poll at most every 25 minutes while waiting.

### 0.3 Paths and budget

| What | Path |
|---|---|
| Remote clone | `/ceph/dwong/jepa/r2_sparse2_repo` |
| Returns | `/ceph/dwong/jepa/returns/r2_sparse2_2026-11-25/` |
| Scratch | `/work/dwong/r2_sparse2/` |
| Reused runs (read-only) | `/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/train/runs/` (C2, S1, S2) |

GPU ledger line `R2_sparse2_gpu`, cap **15 GPU-h**:

| Item | ≈ GPU-h |
|---|---|
| 9 WCTE trainings (≈ 0.9 each on dt2, less on A100) | 8 |
| Scoring (mostly CPU; A100 batch if CPU queues back up) | 3 |
| Smokes | 0.5 |
| Margin | 3.5 |

If the projection exceeds the cap, drop S2Lt seed C first, then S2Lt entirely.

Tooling, hosts and batch layout follow r2-sparse and r2-follow: read `$CC_STATE/notes/r2-sparse.md` and `$CC_STATE/notes/r2-follow.md` (read-only) first.

---

## 1. Setup and measurement

### T0 (≤ 20 min)

- Status file and notes; commit this runbook; create the remote clone at the branch head.
- Inventory: dt2 slots, the batch queue, Ceph free space.
- Locate and record, read-only, the 9 reused run directories (C2, S1, S2 × 9000/9001/9002), with recipe hashes and code tarballs.

### W1 — Time-block width for S2Lt (CPU, `dev_train`, before any training)

- Measure on `lucid_v1_w2` `dev_train` photon bombs:
  - the time span covered by 32 consecutive hits in time order (median, q10, q90);
  - the number of non-empty Δ-wide bins per event for Δ ∈ {2, 5, 10} ns, and hits per bin (median, q10, q90).
- **Rule:** Δ = 5 ns (≈ 5 × the 1-PE jitter), unless the median event then has fewer than 6 non-empty bins; in that case use the largest Δ in {5, 2} that gives ≥ 6. Record the measurement and Δ in the PREREG appendix.
- Also record, for the r2-sparse S2 runs (read-only, `dev_probe_test`, seed 9000), the chance level of the "own-block" mass fraction: the mean over hidden blocks of (hits in the block) / (all candidates' hits), so that the 8 % can be stated against it.

---

## 2. Pre-registration (copy verbatim to `PREREG_2026-11-25_R2_SPARSE2.md`)

### 2.1 Cells (WCTE `lucid_v1_w2`; seeds A = 9000, B = 9001, C = 9002; 10,000 steps; encoder width, depth, heads and window of the default WCTE `mae_repaired`; patch mask as C2, same RNG key)

| ID | Tokens | Embedding | Decoder | Queries | Target / loss | Status |
|---|---|---|---|---|---|---|
| C2 | all PMTs | C2 linear (9 features) | legacy, full attention | linear map of xyz, every masked PMT | fixed loss (r2-sparse §2.2) | reused |
| S1 | hit PMTs | Fourier code + content MLP | isolated | positional code + query embedding, every masked PMT | fixed loss, count head | reused |
| S2 | hit PMTs, time order | Fourier code + content MLP | isolated | τ, block index, mask embedding; one per hidden hit | many-to-one assignment, 32-hit blocks | reused |
| **S1L** | hit PMTs | **C2's linear layer** over the same 9 features, hit flag = 1 | **legacy, full attention** (as C2) | **linear map of xyz** (as C2), every PMT in the masked patches | fixed loss, **no count head** | new |
| **S2L** | hit PMTs, time order | **C2's linear layer**, hit flag = 1 | **full attention** among context and queries | τ, block index, mask embedding; one per hidden hit | many-to-one assignment, **32-hit blocks** | new |
| **S2Lt** | hit PMTs, time order | as S2L | as S2L | τ, **bin index**, mask embedding; one per hidden hit | many-to-one assignment, **Δ-wide time bins** (§2.3) | new |

New runs: 3 cells × 3 seeds = 9.

**S1L is the sparse twin of C2.** It is built from the same modules with the same flags (`--loss fixed`, linear embedding, legacy decoder, no count head, the same input features including direction and ln(1 + q)). The one difference is the token set: unhit PMTs and masked hits are absent from the encoder (MAE style), instead of being present as placeholders and mask tokens. S1L vs C2 is therefore the clean dense-versus-sparse comparison.

**S2L and S2Lt:**
- The candidate scoring keeps the fixed positional code: score_p = ê · PE(r_p) / T, softmax over every PMT that is not a visible token, mass λ_p, loss −Σ_p n_p ln(λ_p + ε) per hidden block (or bin), n_p = 1 for that block's hidden hits.
- Only the encoder's token embedding changes, to C2's linear layer.
- No charge target and no count head, as in S2.

### 2.2 Leakage tests (blocking, before production)

- (a) **S1L:** the set of queried PMTs is a function of the mask only. Two events with the same mask but different hits produce identical query index sets.
- (b) **S1L:** a query's input carries only its PMT's scaled xyz, never charge, time or hit flag.
- (c) **S2L, S2Lt:** predictor inputs are unchanged when the positions of hidden hits are permuted. Queries carry no position, direction or charge.
- (d) **S2L, S2Lt:** full attention among queries exposes only the hidden hits' times, which the queries already carry. Test: the predictor output is unchanged when the hidden hits' PMT indices are permuted among themselves, holding their times.
- (e) **All new cells:** the encoder never sees a masked hit (gradient of the loss with respect to masked hits' input features is exactly zero).

### 2.3 S2Lt time bins

- Hits are assigned to bins of width Δ (from W1, default 5 ns) on the trigger-relative time axis, starting at −110 ns.
- Among non-empty bins, a fresh half is hidden per event (⌈n/2⌉ if odd), drawn with the r2-sparse time-block RNG namespace.
- A query carries its hit's τ, its bin index (learned embedding up to the maximum bin count) and the mask embedding.
- The loss is computed per hidden bin, divided by that bin's number of hidden hits, then averaged over bins.

### 2.4 Outcomes and statistics

- **Primary:** mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1, A.2), step 10,000.
- **Secondary:** MLP and attentive vertex probes (3 probe seeds each); μ and e direction; μ mis-ID at 99 % e efficiency; gain over matched init (step 0).
- **Statistics** as in `PREREG_2026-11-22_R2_TIER1.md` §2.2: T per seed, T̄, SE_boot (2,000 paired resamples), SE_seed = SD_pooled / √3, N_σ, "clear" = N_σ ≥ 2 and one sign in every seed. WCTE SD_pooled is recomputed over all registered WCTE contrasts of r2-sparse, r2-follow and this phase.
- **Diagnostic (registered, used by P6):** for S2, S2L and S2Lt, on `dev_probe_test`, the own-block mass fraction f_own (the share of a query's mass on hidden hits of its own block or bin), and its ratio to the chance level from W1.

### 2.5 Registered contrasts and predictions

| ID | Contrast | Prediction | Reading |
|---|---|---|---|
| P1 | **S1L vs C2** | S1L not worse by more than 5 % | the co-author's claim that sparse tokens can reach dense level, on equal components |
| P2 | S1L vs S1 | S1L better, clear | removing both handicaps from the sparse encoder |
| P3 | S2L vs S2 | S2L better, clear | the same for S2 |
| P4 | **S2L vs C2** | no sign registered | time-ordered position prediction against the best dense cell |
| P5 | S2Lt vs S2L | no sign registered | time-width blocks, effect on vertex |
| P6 | S2Lt vs S2L, on f_own / chance | S2Lt higher in every seed | whether wider time blocks make "when" learnable |

**Reading the margins** (as r2-sparse §2.5): a "not worse by more than x %" prediction holds if T̄ ≤ +x %, fails if T̄ > +x % and the contrast is clear, and is inconclusive otherwise.

### 2.6 Decision rule

- **Scale the sparse family to SK** (a follow-up runbook): at least one of S1L, S2L, S2Lt is not worse than C2 by more than 2 % on the primary outcome, or better than C2 clearly on vertex or on both directions; in either case with MLP and attentive probes of the same sign.
- **Otherwise:** the sparse family is a negative result on WCTE with fair components. Report how much of the r2-sparse gap the handicaps explained (P2, P3).

Everything else is exploratory and is labelled so.

### 2.7 Known limits (recorded, not fixed)

- WCTE is the least diluted detector (median photon-bomb occupancy ≈ 56 %), so sparse tokens save the least here. SK (≥ 78 % unhit) is where they would matter.
- S1L drops the hit flag's information only in form: every token is a hit. It also loses the encoder's explicit view of where PMTs stayed dark. If P1 fails, the next step is to give the hit tokens the fixed geometry of all PMTs as extra context.
- The 50 ns time unit of C0 and C2 is kept; Tier-1 found smaller units better on WCTE.

---

## 3. Stages

Order: T0 → W1 → T1 code → leakage tests and review → PREREG push → smokes → E → T7 → T8.

### T1 — Code (each part with a focused test)

1. **`--embed linear` for sparse arms:** the sparse encoder takes C2's input projection (`Linear(9, 128)`, the same init) applied to the hit tokens' 9 standard features with hit flag 1, ln(1 + q) charge as C2. Defaults and existing cells unchanged.
2. **`--decoder legacy` for `sparse_mae`:** C2's `TargetQueryPredictor` (full attention over context + queries, queries = linear map of scaled xyz). Allowed only with `--queries all`; refuse with `neg` or `pos` (their query sets depend on hits).
3. **`--decoder full` for `sparse_time`:** full attention over context and queries, queries as S2.
4. **`--blocks {count32,time}` and `--block-width-ns`** for `sparse_time`.
5. **f_own diagnostic** with chance level, for S2-type checkpoints.
6. **Recipe hash test:** every existing recipe (C2, C3, C3e, C2dc, S1, S1n, S1p, S2) hashes unchanged.
7. Leakage tests (a)–(e); native suite; one fresh-context review of the diff with the checklist: real LUCiD data only; defaults unchanged; leakage tests; readout unchanged; sealed access only through the evaluator. A BLOCK means fix and review again.

### Smokes

20 steps each for S1L, S2L and S2Lt: memory, step time, finite losses. Project runtimes before production.

### E — Trainings

dt2 lane for seed 9000 (priority S1L, S2L, S2Lt); HTCondor A100 for 9001 and 9002 (≤ 4 concurrent). A failed run is diagnosed, fixed and resubmitted once; a second failure is recorded and the cell is skipped.

### T7 — Evaluation and tables

1. Score each new run at steps 0 and 10,000 on validation and sealed: primary readout, MLP and attentive probes, direction and PID suite.
2. `SPARSE2_TABLE.md` and `.json`:
   - seed-mean errors for C2, S1, S2, S1L, S2L, S2Lt;
   - P1–P6 with T̄, per seed, SE_boot, SE_seed, N_σ, clear and verdicts;
   - secondary outcomes; matched-init gains;
   - f_own and chance for S2, S2L, S2Lt;
   - the decision-rule verdict.
3. **Exploratory:**
   - S2Lt predicted-mass maps for 6 events next to S2's (one figure);
   - f_own split by LUCiD direct/indirect if available in the cache.

### T8 — Report

`results/round2/20261125_sparse2/SPARSE2_REPORT.md` (≤ 1.5 pages plus tables):
- whether sparse tokens reach dense level on equal components (P1);
- how much of the r2-sparse gap the two handicaps explained (P2, P3);
- whether time-width blocks make "when" learnable (P5, P6);
- the decision; deviations.

Also `docs/round2/ACCEPTANCE_2026-11-25_R2_SPARSE2.json`. Set the status to `done` and print the report in full.

**Target:** report by **Thu 26 Nov, 14:00 CET**. Commit what exists then, and finish the rest.

**Stops:** none beyond the safety rules and the GPU cap. A failed leakage test blocks production of that cell until fixed.

---

## Launch (Mac Terminal, its own window, in the foreground)

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_sparse2.sh
```
