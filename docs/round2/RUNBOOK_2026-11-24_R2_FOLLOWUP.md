# Runbook — Follow-up: attribute the C3 loss on WCTE, and a third seed for the SK mask result (r2-follow), 24 Nov 2026

**Phase** `r2-follow`. **Branch** `r2-follow`, created from `origin/sparse-tok`. **Worktree** `~/Documents/VERITAS-worktrees/r2-follow`. **Results** `results/round2/20261124_follow/`.

No other phase is running. All earlier phases (`r2-tier1`, `r2-tier1v`, `r2-mask`, `r2-sparse`) are done; their branches, clones, returns and data are read-only for this phase.

---

## 0. Why this runbook exists

Two open points from the phases that finished on 24 Nov:

1. **r2-sparse, R2b.** C3 (fixed positional code + content MLP, query-isolated decoder, count head) is +32.3 % worse than C2 on WCTE sealed vertex (3 seeds, clear, all probes same sign), although its pretext losses equal C2's. C3 changed three things at once, so the loss cannot be attributed (`SPARSE_REPORT.md` §4). The embedding was the co-author's proposal; the author will not report it as harmful before the attribution is known.
2. **r2-mask, Q2.** On SK, the frozen-readout vertex gain follows the post-interpolation residual U exactly (ρ = 1.0 over random and patch 43 / 173 / 692; patch vs random up to −14.8 %, clear). This is the strongest candidate for the paper's lead claim, but it rests on 2 seeds (`MASK_REPORT.md` §5).

This phase adds the minimum runs that settle both: 6 WCTE trainings (two single-change cells × 3 seeds) and 4 SK trainings (seed C of the four M2 cells).

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `r2-follow`:

1. **Author, 24 Nov 2026, 15:59 CET:**
   > "开始，去写runbook，顺便帮我写一个简洁的report给俊杰"

   (Translation: "Go ahead, write the runbook, and also write a concise report for Junjie.") This approves the two follow-ups proposed at 15:58 CET: the C3e attribution runs and a third SK seed for the mask result.
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决". Never stop to ask; record decisions in `DECISIONS.md` and `$CC_STATE/notes/r2-follow.md`.
3. **Author, 22 Nov 2026:** do not depend on the collaborator.

### 0.2 Binding rules

- **Real LUCiD data only.** WCTE runs train on `lucid_v1_w2` (as all r2-sparse cells); SK runs train on `lucid_v1` (as all r2-mask SK cells). Never use the analytic generator or any analytic-world pack or checkpoint. Assert `dataset_kind` and LUCiD `e651e836` in every recipe.
- **Same code as the cells being compared.**
  - WCTE: the r2-sparse code tarball `code/sparse_9bfc364a.tar.gz` (commit `9bfc364`) and the same flags as C2, changing only what §2.1 names.
  - SK: the code and recipe of the existing seed-A/B cells, changing only `--seed`. Before submitting, rebuild each seed-C recipe and check that it equals the seed-A recipe of the same cell field by field except the seed (record the diff in the notes). Use the r2-mask tarball `code/mask_87628ef6.tar.gz` for all four SK cells if its recipe for the default patch and for random equals the Tier-1 recipe except the seed; otherwise use each cell's original tarball.
- **Never overwrite** existing packs, caches, checkpoints or returns. New trainings write only under `/ceph/dwong/jepa/returns/r2_follow_2026-11-24/` (`R2_REPAIR_OUTPUT_ROOT=<this>/train`). Existing runs are read in place, read-only.
- **Pre-registration.** Copy §2 verbatim to `docs/round2/PREREG_2026-11-24_R2_FOLLOW.md`, commit and push **before any new run is evaluated**. Later changes are dated amendments appended below, never edits.
- **Sealed set.** The rules of `RUNBOOK_2026-11-22_R2_TIER1_GATE.md` §T2.3 apply. Sealed packs are read only through `scripts/tier1_eval.py`; every read is logged in `results/round2/20261124_follow/SEALED_ACCESS_LOG.md`.
- **Frozen readouts.** `docs/round2/READOUT_SPEC_v1.md` unchanged, with the r2-sparse evaluator extension (data kind from the recipe).
- **Resources.**
  - deepthought2: at most one lane from this phase, only when a slot is free.
  - HTCondor: at most 4 concurrent jobs from this phase.
  - CPU: ≤ 6 cores on `dt1` at `nice -n 15`; portal1 up to one sixth of its cores, leaving ≥ 8 free.
  - JAX on shared GPUs: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`.
- **Safety.** Main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Push only to `origin/r2-follow`, fast-forward only. LUCiD: internal research use only. Poll at most every 25 minutes while waiting.

### 0.3 Paths and budget

| What | Path |
|---|---|
| Remote clone | `/ceph/dwong/jepa/r2_follow_repo` |
| Returns | `/ceph/dwong/jepa/returns/r2_follow_2026-11-24/` |
| Scratch | `/work/dwong/r2_follow/` |
| Existing WCTE cells (read-only) | `/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/train/runs/` (C2, C3) |
| Existing SK cells (read-only) | as listed in `results/round2/20261123_mask/` and `docs/round2/DECISIONS.md` of `origin/mask-scale` |

GPU ledger line `R2_follow_gpu`, cap **34 GPU-h**:

| Item | ≈ GPU-h |
|---|---|
| 4 SK trainings (A100, ≈ 5.3 each, as r2-mask's ps43 seed A) | 21.5 |
| SK scoring (step 0, 5,000, 10,000; validation and sealed; A100 batch) | 3 |
| 6 WCTE trainings (dt2 ≈ 0.9 each, or A100) | 5.5 |
| Margin | 4 |

If the projection exceeds the cap, drop the WCTE cell C2dc first (§2.1), never an SK cell.

Tooling, hosts and batch layout follow `r2-sparse` and `r2-mask`: read `$CC_STATE/notes/r2-sparse.md` and `$CC_STATE/notes/r2-mask.md` (read-only) first.

---

## 1. Setup

### T0 (≤ 20 min)

- Status file and notes; commit this runbook; create the remote clone at the branch head.
- Merge `origin/mask-scale` into `r2-follow` so that `scripts/mask_scale_analysis.py`, the SK M1/M2 results and the patch-size code are on the branch. Resolve conflicts in `DECISIONS.md`, `RESOURCE_LEDGER.json` and `STATE` files by keeping both sides' entries; resolve no code conflict by editing tested code (if one arises, take `mask-scale`'s version of mask code and `sparse-tok`'s version of sparse code, and run both phases' focused tests).
- Inventory: dt2 slots, the batch queue, Ceph free space.
- Locate and record, read-only: the 6 existing WCTE run directories (C2, C3 × 9000/9001/9002) and the 8 existing SK run directories (random, patch 43, patch 173, patch 692 × 9000/9001), with their recipe hashes and code tarballs.
- Rebuild the 6 new WCTE recipes and the 4 SK seed-C recipes; record the field-by-field diff to the reference recipe (§0.2). Any difference other than the intended one is fixed before submission.

---

## 2. Pre-registration (copy verbatim to `PREREG_2026-11-24_R2_FOLLOW.md`)

### 2.1 Cells

**Part A — WCTE attribution** (`lucid_v1_w2`; seeds 9000, 9001, 9002; 10,000 steps; r2-sparse code `9bfc364`; all with `--loss fixed --placeholder sentinel` as C2):

| ID | Flags beyond C2 | Status |
|---|---|---|
| C2 | none | existing (r2-sparse) |
| C3e | `--embed fourier` (legacy decoder, no count head) | **new** |
| C2dc | `--decoder isolated --count-head` (linear embedding) | **new** |
| C3 | `--embed fourier --decoder isolated --count-head` | existing (r2-sparse) |

**Part B — SK third seed** (`lucid_v1`; seed C = 9002; 10,000 steps; recipes identical to seeds A/B except the seed):

| ID | Mask | Status |
|---|---|---|
| SK-rnd-C | random | **new** |
| SK-p43-C | patch 43 | **new** |
| SK-p173-C | patch 173 (default) | **new** |
| SK-p692-C | patch 692 | **new** |

Seed-C step-0 weights must be bitwise identical across the four SK cells (as for seeds A and B); check by hash.

### 2.2 Outcomes and statistics

- **Primary:** mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1, A.2), at step 10,000.
- **Secondary:** MLP (3 probe seeds) and attentive (3 probe seeds) vertex probes; μ and e direction; gain over matched init (step 0). SK also at step 5,000.
- **Statistics** as in `PREREG_2026-11-22_R2_TIER1.md` §2.2 (T per seed, T̄, SE_boot from 2,000 paired resamples, SE_seed = SD_pooled / √n_seeds, N_σ, "clear" = N_σ ≥ 2 and one sign in every seed). WCTE SD_pooled is recomputed over this phase's and r2-sparse's registered WCTE contrasts; SK SD_pooled over the M2 contrasts with 3 seeds.

### 2.3 Part A contrasts and attribution rule

| ID | Contrast | What it measures |
|---|---|---|
| A1 | C3e vs C2 | the new embedding alone |
| A2 | C2dc vs C2 | the query-isolated decoder and count head alone |
| A3 | C3 vs C3e | decoder and count head, given the new embedding |
| A4 | C3 vs C2dc | the new embedding, given the new decoder |

**Attribution rule.** A component "accounts for" the C3 loss if its single-change contrast (A1 for the embedding, A2 for the decoder and count head) is clear and at least half of the registered R2b (+32.3 %, i.e. T̄ ≥ +16.2 %). If neither does but A3 or A4 does, the loss is an interaction and is reported as such. If none is clear, R2b is reported as not reproduced in parts.

No sign is predicted.

### 2.4 Part A diagnostics (registered as exploratory, all CPU)

For C2, C3e, C2dc, C3 (seed 9000, steps 0 and 10,000, `dev_probe_test`):
- **Token composition:** RMS per dimension of the positional code and of the content-embedding output at the encoder input; and the fraction of variance of the block-4 tokens explained by a linear map of the fixed positional code (geometry share) versus by (ln q, τ, hit) (content share).
- **Probe of geometry vs event:** R² of a linear probe from block-4 tokens to PMT position, and to the event vertex through the frozen readout, side by side.

These answer whether the positional code dominates the token (a scale problem) or whether the decoder change moves work out of the encoder.

### 2.5 Part B predictions (SK, 3 seeds)

| ID | Statement | Holds if |
|---|---|---|
| B1 | Every patch size beats random | each of p43, p173, p692 vs random: T̄ < 0 and clear, with 3 seeds |
| B2 | U predicts the gain (Q2 with 3 seeds) | Spearman ρ between U (M1, unchanged) and the seed-mean gain over matched init ≥ 0.8 over the 4 configurations; direction (μ and e) gains ρ ≥ 0.8 |
| B3 | The ordering is reproduced by seed C alone | seed C's gains alone give ρ ≥ 0.8 |

**Kill:** if B1 or B2 fails with 3 seeds, the SK mask claim is reported as "not reproduced with a third seed" and is not used as the paper's lead claim.

---

## 3. Stages

Order: T0 → recipes and diffs → PREREG push → submit SK (4 jobs, HTCondor A100, at once) → WCTE smokes (20 steps, C3e and C2dc) → WCTE trainings (dt2 lane, priority C3e then C2dc; A100 if dt2 is busy) → scoring → tables → report.

### T1 — SK

- Submit the 4 SK seed-C trainings at once (≤ 4 concurrent), snapshots at steps 0, 5,000 and 10,000.
- Score each at steps 0, 5,000, 10,000 on validation and sealed through the batch evaluator (A100), as r2-mask did.

### T2 — WCTE

- C3e and C2dc × 3 seeds. Score at steps 0 and 10,000, validation and sealed (CPU workers or A100 batch).
- Run the §2.4 diagnostics on CPU.

### T3 — Tables

- `FOLLOW_TABLE.md` and `.json`:
  - Part A: seed-mean errors of C2, C3e, C2dc, C3; A1–A4 with T̄, per seed, SE_boot, SE_seed, N_σ, clear; the attribution verdict; secondary probes; §2.4 diagnostics.
  - Part B: the M2 table with 3 seeds (gains over matched init, patch vs random contrasts, steps 5,000 and 10,000); B1–B3 verdicts; the updated `MASK_SCALE` figure (`MASK_SCALE_3seed.png`).
- Re-run `scripts/mask_scale_analysis.py` with seeds A, B, C; keep the 2-seed outputs unchanged and write the 3-seed outputs next to them.

### T4 — Report

`results/round2/20261124_follow/FOLLOW_REPORT.md` (≤ 1.5 pages plus tables):
- Part A: which component accounts for the C3 loss, with the diagnostics.
- Part B: whether the SK mask result holds with 3 seeds, and the exact wording the lead claim may use.
- Deviations.

Also `docs/round2/ACCEPTANCE_2026-11-24_R2_FOLLOW.json`. Set the status to `done` and print the report in full.

**Target:** WCTE Part A by Tue 24 Nov night; SK Part B and the report by **Wed 25 Nov, 14:00 CET**. Commit what exists then and finish the rest.

**Stops:** none beyond the safety rules and the GPU cap. A recipe diff other than the intended one stops submission of that cell until fixed.

---

## Launch (Mac Terminal, its own window, in the foreground)

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_follow.sh
```
