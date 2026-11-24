# FOLLOW_REPORT — r2-follow: attributing the C3 loss on WCTE, and a third SK seed for the mask result (24–25 Nov 2026)

Runbook `RUNBOOK_2026-11-24_R2_FOLLOWUP.md`. Pre-registration `PREREG_2026-11-24_R2_FOLLOW.md` (§2 verbatim, pushed ca1dfd0 at 15:24 UTC; amendment A1 = the code of the two new cells, A2 = a timestamp correction), pushed before any new run was evaluated (first evaluation 15:57 UTC). Real LUCiD data only: WCTE `lucid_v1_w2`, SK `lucid_v1`, LUCiD e651e836. Full numbers: `FOLLOW_TABLE.md` / `.json`; recipes: `RECIPE_DIFFS.md`.

**What ran.**
- WCTE: 6 new trainings, C3e and C2dc × seeds 9000/9001/9002, 10,000 steps, on the deepthought2 lane. C2 and C3 are the r2-sparse runs, read only.
- SK: 4 new trainings, seed C = 9002 for random, patch 43, 173 and 692, HTCondor A100.
- Scoring: all 10 runs through `scripts/tier1_eval.py`, validation and sealed (WCTE at steps 0 and 10,000; SK at 5,000 and 10,000, plus step 0 once).
- 21 sealed reads, all logged in `SEALED_ACCESS_LOG.md`.
- GPU: 27.5 of 34 GPU-h (`R2_follow_gpu`); C2dc did not have to be dropped.

## Part A — which component accounts for the C3 loss?

Sealed photon-bomb vertex error, frozen spatial-moment Ridge, seed means: **C2 0.0549 m, C3e 0.0591 m, C2dc 0.0633 m, C3 0.0726 m.**

| ID | Contrast | T̄ (A / B / C) | N_σ | clear |
|---|---|---|---|---|
| A1 | C3e vs C2: embedding alone | **+7.8 %** (+9.6 / +7.4 / +6.3) | 6.3 | yes |
| A2 | C2dc vs C2: isolated decoder + count head alone | **+15.3 %** (+14.5 / +16.4 / +15.0) | 12.5 | yes |
| A3 | C3 vs C3e: decoder + count head, given the embedding | **+22.8 %** (+23.2 / +25.1 / +20.0) | 18.5 | yes |
| A4 | C3 vs C2dc: embedding, given the decoder | **+14.8 %** (+17.9 / +15.4 / +10.9) | 12.0 | yes |

R2b (C3 vs C2) is reproduced: +32.3 %, N_σ 26.3. SD_pooled = 0.0213, over R1–R6 and A1–A4.

**Verdict by the registered rule (§2.3): an interaction.**
- Neither single change reaches half of R2b (+16.2 %). The decoder + count head comes close (+15.3 %); the embedding alone does not (+7.8 %).
- A3 does reach it (+22.8 %).
- The non-multiplicative part is I = (e_C3·e_C2)/(e_C3e·e_C2dc) − 1 = **+6.5 %** (N_σ 6.1, positive in every seed; exploratory).
- On a log scale the +32.3 % splits roughly into: decoder + count head ≈ ½, embedding ≈ ¼, their interaction ≈ ¼.

**Secondary probes.**
- A2 is clear on every probe: MLP +10.6 %, attentive +10.4 %, μ direction +11.8 %, e direction +7.7 %.
- A1 is smaller and only partly clear: attentive +4.6 % (clear), μ +5.8 % (clear), MLP +2.8 % and e +3.0 % (not clear).

**Diagnostics (§2.4, exploratory, seed 9000).**
1. *Not a scale problem.*
   - At the encoder input the fixed positional code does not dominate the hit tokens. Positional vs content RMS is 0.70 vs 0.76–1.04 in C3e and C3, against 0.50–0.57 vs 0.55–0.57 in the linear encoders.
   - The geometry share of block-4 variance is similar in all four models: 0.49–0.54 of all-token variance at step 10,000.
2. *The decoder change moves work out of the encoder.*
   - With the isolated decoder the encoder improves least from its initialisation. C2dc's matched-init gain is only 6 %, against 18 % for C2.
   - Content information in the block-4 hit tokens drops (content share 0.095 vs 0.128 for C2; 0.101 for C3 vs 0.143 for C3e).
   - The PMT-position probe R² falls during training in both isolated-decoder models (C2dc 0.997 → 0.994, C3 0.976 → 0.970) and rises in both legacy-decoder models.
3. *The embedding mostly costs through its starting point.*
   - The Fourier-code initialisation is already 32 % worse at step 0 (sealed 0.089 vs 0.067 m, seed means).
   - C3e recovers most of that (gain 33 % vs 18 % for C2) and ends +7.8 % behind. Whether a longer schedule would close the rest is not tested here.

**What this means for the co-author's proposal.**
- The new embedding is not the main cause of the C3 loss. Alone it costs +7.8 % on WCTE (clear), about a quarter of R2b.
- The query-isolated decoder with the count head costs about twice as much on its own.
- The two changes hurt more together than separately.
- It may be reported as: "the embedding alone is mildly worse (+8 %); the C3 loss comes mainly from the decoder change and its interaction with the embedding."
- It may not be reported as: "the embedding is harmful (+32 %)".

## Part B — does the SK mask result hold with three seeds?

Gain over matched init at step 10,000 (A / B / C):

| Config | U | Gain (A / B / C) | Mean gain |
|---|---|---|---|
| random | 0.549 | 21.9 / 24.6 / 25.7 % | 24.0 % |
| patch 43 | 0.552 | 31.4 / 28.6 / 31.1 % | 30.4 % |
| patch 173 | 0.565 | 32.8 / 35.0 / 34.1 % | 34.0 % |
| patch 692 | 0.600 | 35.0 / 34.2 / 35.2 % | 34.8 % |

- **B1 holds.** Patch vs random: −8.3 %, −13.1 %, −14.1 % (N_σ 5.6 / 8.8 / 9.5), each negative in every seed. At 5,000 steps: −7.6 / −14.3 / −16.4 %, all clear.
- **B2 holds.** ρ(U, vertex gain) = 1.0. μ and e direction gains give ρ = 0.8 each, exactly the threshold: both peak at patch 173 and are marginally lower at 692 (μ 43.1 vs 41.4 %, e 51.6 vs 51.4 %).
- **B3 holds.** Seed C alone gives ρ = 1.0.
- The kill rule does not fire. Figure: `results/round2/20261123_mask/MASK_SCALE_3seed.png` (the 2-seed outputs are unchanged).

**Exact wording the lead claim may use:**
> "On SK, the frozen-readout vertex gain over matched initialisation follows the post-interpolation residual U of the mask exactly: Spearman ρ = 1.0 over a random mask and patches of 43, 173 and 692 PMTs, with three seeds, and also with the third seed alone. Every patch size beats the random mask (−8 % to −14 % vertex error, clear in each case and in every seed). The contrast is already present at 5,000 steps. μ and e direction gains follow the same order up to patch 173 and saturate beyond it (ρ = 0.8)."

The restrictions of `MASK_REPORT.md` §5 still apply:
- not general: on WCTE the gain peaks at a moderate patch and then falls, while U keeps rising;
- not tied to the photon correlation length;
- one detector and four configurations.

## Deviations

1. **Code of C3e and C2dc (PREREG A1).**
   - `9bfc364` refuses both cells, so they run on `760c3d5` = `9bfc364` + `sparse_arms.FactorJepa`.
   - FactorJepa combines the exact C2 and C3 modules. "Decoder" = the whole C3 predictor (its positional-code queries and isolated attention) plus the count head.
   - `760c3d5` rebuilds C2's and C3's recipes and step-0 weights bit for bit.
   - Scoring check: C3e's and C2dc's step-0 scores equal C3's and C2's exactly.
2. **SK step 0 scored once, not per cell.** The seed-C initial weights are bitwise identical across the four cells (hash checked), so one step-0 score (the default-patch run) is the matched init, as in r2-mask.
3. **Analysis additions.**
   - The interaction contrast I is exploratory.
   - Diagnostic choices: block 4 is the encoder read-out (n_blocks = 4); the geometry regressor is the Fourier code for all four models; the linear encoders' positional part is W_geo·x_geo.
   - `mask_scale_analysis.py` gained `--sk-seeds` / `--out-stem`. With the defaults it reproduces the r2-mask outputs byte for byte.
4. **Operations.**
   - A poll-script pattern bug started a second CPU worker on Deepthought for about one hour. Both were pinned to cores 0–5, so the 6-core limit held, with oversubscribed threads.
   - The feeder polled every 2.5 min for about 30 min while waiting for the last SK scoring. The runbook allows at most one poll every 25 min.
   - The access log keeps `tier1_eval.py`'s default header "(r2-tier1)".
5. **Timing.** Everything finished on Tue 24 Nov, about 23:00 UTC, before the Wed 14:00 CET target.
