# PRE-REGISTRATION — r2-sparse-sk (26 Nov 2026)

Copied verbatim from `docs/round2/RUNBOOK_2026-11-26_R2_SPARSE_SK.md` §2 (branch `sparse-sk`), pushed before any new run of this phase was evaluated. Later changes are dated amendments below the amendments line, never edits of the text above it.

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

## Appendix A — operational readings (fixed 26 Nov 2026, before any new run is evaluated)

These readings implement §2 where its text leaves a choice; none changes a cell, a prediction or a margin.

- **A1. Absence patches.** The patches of §2.3 are the mask's patches: contiguous blocks of `absence_patch_size` PMTs of the full Hilbert order (SK 173, WCTE 32, the default patch of each detector). The default mask never hides the tail beyond 64 × 173 PMTs (SK: 24 PMTs; WCTE: 1995 − 62 × 32 = 11 PMTs); the tail is one more, always visible patch. An SK training event therefore has 33 absence tokens (32 visible patches + tail); at readout, with no mask, every patch is visible (65 tokens).
- **A2. Absence features** are computed over the visible PMTs of the patch (for a patch mask: the whole visible patch, as §2.3): mean scaled xyz (the encoder's position features), hits / PMTs, ln(1 + hits), ln(1 + Σ q / 1 PE) with q = exp(ln q feature). Embedding `Linear(6, 128)` (spectral μP init) plus a learned type vector (N(0, 0.02)), built after all S1L modules so that every shared weight equals S1L's at the same seed (recipe check: S1A's step-0 state equals S1L's on the shared keys).
- **A3. Placement and use.** In the Hilbert serialisation an absence token sits immediately before its patch's first PMT (key = start rank − 0.5). Absence tokens are context for the encoder and for S1L's legacy decoder (C2's full-attention predictor over all encoder outputs). The readout scatters hit tokens only (§2.3); at readout the encoder sees the absence tokens of all patches.
- **A4. Sparse encoder on SK.** All sparse cells use the detector's `mae_repaired` encoder (SK window 384, shift 192); for WCTE this equals the unchanged `EncoderConfig()` (window 64), so the WCTE sparse recipes keep their hashes (recipe check).
- **A5. SK-C2 sentinel** = (270 + 50) / 450 = 0.7111 from the SK `lucid_v1` window [−20, +270] ns; WCTE keeps (50 + 50) / 100 = 1.0.
- **A6. Contrast definition.** T per seed = e_test / e_base − 1 on the sealed photon bombs (μ mis-ID: misid_test / misid_base − 1; directions: mean angle; same paired event bootstrap), statistics as `PREREG_2026-11-22_R2_TIER1.md` §2.2 via the unchanged r2-sparse statistics code.
- **A7. Verdicts.** "Better, clear" holds if T̄ < 0 and clear, fails if T̄ > 0 and clear, else inconclusive. "Not worse by more than x %" as §2.6 (holds if T̄ ≤ +x %, fails if T̄ > +x % and clear, else inconclusive). K4 holds if both μ and e direction are better and clear (vertex reported). K5 / K6 hold if both parts hold (μ mis-ID lower and clear; vertex not worse by > 2 %); fail if either part fails; otherwise inconclusive.
- **A8. Decision §2.7.** "SK-S1L's μ mis-ID is not clearly worse than SK-C2's" = the SK-S1L vs SK-C2 μ mis-ID contrast is not (T̄ > 0 and clear). T\* = SK-S1A if K2 and K5 hold; else SK-S1L if K2 holds; else dense C2.
- **A9. SD_pooled.** SK: RMS of the across-seed SDs of the per-seed T of every registered SK contrast with 3 seeds (r2-mask / r2-follow 3-seed SK mask contrasts and K1–K5), per metric and cohort. WCTE: r2-sparse2's set (R1–R6, A1–A4, P1–P5) plus K6.
- **A10. Efficiency.** Median wall time per optimiser step over the 1,000 step intervals 1,001–2,000 and peak GPU memory up to step 2,000 (`efficiency.json` of each new run), A100 only. SK-D0 (reused, trained before the logger existed): the same median from its `train_log.jsonl` elapsed times (A100), memory = its recorded whole-run peak.
- **A11. SK scoring.** SK new runs are scored on HTCondor A100 (`tier1_eval_batch.py`); SK-D0's existing step-0 and step-10,000 evaluator outputs (r2-lucid / r2-tier1 / r2-mask / r2-follow eval trees) are reused read-only; WCTE C2 and S1L likewise from r2-sparse / r2-sparse2.

---

## Appendix J — Stage J (JUNO), runbook §5 (fixed 26 Nov 2026, before any JUNO run is evaluated)

- **J0 window (registered rule, r2-sparse §W1 unchanged):** P = max(50, q99.5(trigger − earliest signal digit) + 10 ns) rounded up to 10 ns = max(50, 97.6 + 10) → **110 ns**; A = smallest multiple of 50 ns with ≥ 99.5 % of first-per-PMT signal digits in [−P, A] → **350 ns** (99.71 %). JUNO `lucid_v1_w2` window **[−110, +350] ns**, time scale unchanged (333.3 ns). Data: `/ceph/dwong/jepa/data/lucid_v1_w2/juno/juno_lucid_v1_w2.h5` (SHA-256 2e759db1…) and `…_sealed.h5` (fa42a9bf…).
- **J0 validation (W2 §5):** passed for all four cohorts; one `dev_train` hit of 197 million changed by a dark-extension digit merged after its cached digit (registered merge rule; `DECISIONS.md`).
- **Cells (runbook §5.2, 3 seeds, 10,000 steps, JUNO default encoder, window 320, patch 156):** J-C2v1 (`lucid_v1`, dense C2 recipe: `--loss fixed --placeholder sentinel`, sentinel (200 + 50)/333.3 = 0.75), J-C2v2 (`lucid_v1_w2`, dense C2 recipe, sentinel (350 + 50)/333.3 = 1.2, the same "window end + 50 ns" rule), J-Tv2 (`lucid_v1_w2`, tokenisation T\* of the Stage S decision, recorded in appendix H; dropped if T\* is dense).
- **Predictions (runbook §5.3):** J1 J-C2v2 vs J-C2v1: no sign registered (window effect where late light is largest); J2 J-Tv2 vs J-C2v2: not worse by more than 5 % (appendix A7 margin reading).
- **Reading of J1 (operational):** J1 compares models trained and scored on different data (v1 and v2 packs of the same events); T per seed = e(J-C2v2 on v2 sealed) / e(J-C2v1 on v1 sealed) − 1 on the same sealed event ids (the bootstrap pairs events by id). Statistics as appendix A6; SD_pooled for JUNO = RMS of the across-seed SDs of J1 and J2 (no earlier 3-seed JUNO contrast exists).
