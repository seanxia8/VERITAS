# PREREG r2-sparse (24 Nov 2026) — sparse hit tokens on WCTE

Copied verbatim from `docs/round2/RUNBOOK_2026-11-24_R2_SPARSE_WCTE.md` §2 (lines 133–237 of the committed runbook f1bf214). The W1 window and the S1n ratio r are appended below the text. Later changes are dated amendments appended at the end, never edits.

---


### 2.1 Configurations (WCTE; seeds A = 9000, B = 9001, C = 9002; 10,000 steps; encoder width, depth, heads and attention window of the default WCTE `mae_repaired`)

| ID | Data | Tokens | Mask | Loss | What it isolates |
|---|---|---|---|---|---|
| C0 | `lucid_v1` | all PMTs | patch | current `mae_repaired` | reference; **existing** Tier-1 cells (patch, 50 ns) × {A, B, C}, reused |
| C1 | `lucid_v1_w2` | all PMTs | patch | current `mae_repaired` | the window (C1 vs C0) |
| C2 | `lucid_v1_w2` | all PMTs | patch | fixed loss (§2.2) and new placeholders | the loss fix (C2 vs C1) |
| C3 | `lucid_v1_w2` | all PMTs | patch | as C2, with the new embedding, the query-isolated decoder and the count head (§2.2a) | the new token and decoder design (C3 vs C2); the dense control for every S arm |
| S1 | `lucid_v1_w2` | hit PMTs only | patch | fixed loss; all masked PMTs queried; count head | sparse encoder (S1 vs C3) |
| S1n | `lucid_v1_w2` | hit PMTs only | patch | fixed loss; negatives subsampled; count head | the co-author's negative sampling (S1n vs S1) |
| S1p | `lucid_v1_w2` | hit PMTs only | patch | positives only: charge and time of masked hits; no hit classification; count head | whether absence information matters (S1p vs S1) |
| S2 | `lucid_v1_w2` | hit PMTs only, time order | time blocks | position prediction by soft many-to-one assignment (§2.3) | time-ordered "where is the light at time t" (S2 vs C3) |

New runs: 7 configurations × 3 seeds = 21. C1 and C2 keep the current embedding (one linear layer over the nine features); C3, S1, S1n, S1p and S2 use the new embedding. For a given seed and event, every arm uses the same patch mask (the RNG key of `mae_repaired`). The S arms' patches are the Hilbert patches of the full PMT list, so a sparse arm hides exactly the hits its dense counterpart hides.

### 2.2a New embedding (C3 and all S arms)

- **Positional code, fixed and not trained.** Scale x, y by the detector radius and z by the half-height, giving values in [−1, 1]. For each axis use 21 frequencies, log-spaced from a wavelength of twice the detector size down to the median nearest-neighbour PMT spacing, each in that axis's scaled units (record both). Take sin and cos of each: 3 × 21 × 2 = 126 dimensions, padded with 2 zeros to d_model = 128.
- **Content embedding:** a two-layer MLP (width 128, GELU) on the content features:
  - C3 (dense): ln(1 + q), τ and the hit flag;
  - S arms (hit tokens): ln q and τ.
- **Token** = content embedding + positional code. No direction feature.
- **Hidden tokens inside the dense encoder (C3):** a learned mask embedding + the positional code. Today they carry the mask embedding only, with no position.
- **Decoder (C3 and all S arms):** the existing predictor (2 blocks, d_pred = 64).
  - A query = a learned query embedding + a linear projection of the queried PMT's positional code, replacing the linear map of xyz. S2's queries are defined in §2.3.
  - Queries attend to the context tokens and to themselves only, never to other queries.
  - C3 also has the count head of §2.2.

  C3 is thus the dense control with every S-arm component. R3 isolates the sparse encoder input; R2b measures the new embedding, the query-isolated decoder and the count head together.
- **Scale at initialisation:** the positional code has an RMS of ≈ 0.71 per dimension. Initialise the content MLP's output layer so its RMS at step 0 is within a factor of 2 of that, and record both.
- **Rejected alternative:** a learned per-PMT embedding table keyed by PMT index. It is a pure lookup, it invites the static shortcut, and it does not transfer between geometries.

### 2.2 Fixed loss and placeholders (C2, C3, S1, S1n, S1p)

- **Hit:** binary cross-entropy on the queried PMTs, class-balanced.
  - Positives get weight w = clip(n_neg / n_pos, 0.05, 20), computed per batch over the queried set; negatives get weight 1.
  - The term is divided by the sum of the weights, so it stays on the scale of a plain BCE.
  - The same rule holds for every arm with a hit term, S1n included (on its sampled set).
  - On WCTE w is usually below 1: the median photon bomb hits ≈ 1,100 of the 1,995 PMTs.
- **Charge:** MSE of ln q on hit PMTs only.
- **Time:** 2 × Huber_δ(r) on hit PMTs only, with r = (τ̂ − τ) · T / 50 ns (WCTE: T = 100 ns) and δ = 0.2 (10 ns). The factor 2 makes it equal to C1's term r² for |r| ≤ 10 ns, so the time weight is unchanged where predictions are good; only the heavy tail (late reflected light, dark hits) becomes linear. Tier-1 showed that WCTE vertex responds clearly to the time weight, so this fix must not change it silently.
- Terms summed with weight 1. The count head is added with weight 0.1.
- **Placeholders (C2 and C3, dense tokens only):**
  - an unhit PMT gets time feature (A + 50 ns) / 100 ns, just after the window end ("no hit inside the window");
  - charge feature ln(1 + q / 1 PE), so no-hit = 0 and no real hit takes that value (every hit has q > 0; asserted in W2);
  - the hit flag is kept.

  C1 keeps the old placeholders on purpose.
- **Queries (S1, S1n, S1p):** decoder queries carry only a masked PMT's positional code plus the learned mask embedding (§2.2a). They attend to the context tokens and **never to each other**, so the set of queried positions cannot leak where the light is.
- **S1:** query every PMT inside the masked patches, hit or not.
- **S1n:** query all masked hit PMTs plus k = min(n_unhit_masked, ⌈r · n_hit_masked⌉) unhit PMTs drawn **only from inside the masked patches**. Half of them are drawn uniformly and half within two median PMT spacings of a masked hit ("hard" negatives).
  - The co-author's cap r = 2 was meant for SK-sized detectors, where ≥ 78 % of PMTs are unhit. On WCTE the median bomb hits more than half the PMTs, so r = 2 would bind only in dim events and S1n would mostly equal S1.
  - So r for WCTE is fixed in W2, before any training: from {0.25, 0.5, 1, 2}, the value whose median kept fraction of masked unhit PMTs on `dev_train` is closest to SK's median kept fraction at r = 2 (from SK `lucid_v1` `dev_probe_train` occupancy). Record r and both fractions in the PREREG file.
- **Count head (C3, S1, S1n, S1p):** predicts the number of masked hits per event from mean-pooled context tokens only, never from the queries. Loss is Poisson negative log-likelihood.

### 2.3 S2 — time-ordered position prediction

- **Tokens:** hit PMTs sorted by first-hit time (ties by PMT index); windowed attention in that order.
- **Mask:** cut the time-ordered sequence into blocks of 32 tokens and hide a fresh half of the blocks per event (⌈count/2⌉ if odd). Hidden tokens are removed from the encoder.
- **Queries:** one per hidden hit. Each carries **only** its time τ (through a small MLP), its block index and a learned mask embedding; no position, no direction, no charge. Each query outputs a vector ê in the space of the positional code. Queries attend to the visible tokens and to themselves only.
- **Loss, per hidden block k:**
  - The candidate set is every PMT that is not a visible token: hidden hits of all blocks plus unhit PMTs.
  - Each query spreads unit mass over the candidates with a softmax of ê · PE(r_p) / T, where PE is the fixed positional code of §2.2a and T a learned scalar temperature. Because PE is fixed, the target cannot collapse; because the scores come from coordinates, this is geometry, not an index lookup. A dot product of Fourier features behaves like a smooth distance kernel.
  - λ_p is the mass received by PMT p; n_p = 1 if p is a hidden hit of block k, else 0.
  - Loss = −Σ_p n_p · ln(λ_p + ε), divided by the number of hidden hits in the block.
  - Mass placed on unhit PMTs, or on hits of other time blocks, is wasted, so it is penalised. A PMT may receive mass from several queries (many-to-one, not Hungarian). This is the differentiable form of the co-author's "is the point on a hit PMT's surface" classification.
- **No count head.** A hidden block holds 32 hits by construction and there is one query per hidden hit, so the count is known and the head would be trivial.

### 2.4 Outcomes and statistics

- **Primary:** mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge readout (READOUT_SPEC_v1, A.2), each model on the sealed pack of its own data kind. The events are the same across data kinds, so contrasts are paired.
- **Secondary:** μ and e direction (median angle), μ mis-identification at 99 % e efficiency, MLP (3 probe seeds) and attentive (3 probe seeds) vertex probes, and gain over each run's matched init (step-0 snapshot). WCTE energy is not evaluated: LUCiD's containment flag leaves no contained electrons.
- **Statistics** as in `PREREG_2026-11-22_R2_TIER1.md` §2.2 (SE_boot, SE_seed from the 3 seeds, N_σ, "clear" = N_σ ≥ 2 and one sign in every seed).

### 2.5 Registered contrasts and predictions

| ID | Contrast | Prediction |
|---|---|---|
| R1 | C1 vs C0 | none (window effect, reported) |
| R2 | C2 vs C1 | C2 not worse by more than 2 % |
| R2b | C3 vs C2 | C3 not worse by more than 2 % |
| R3 | **S1 vs C3** | S1 not worse by more than 2 % |
| R4 | S1n vs S1 | within ± 3 % |
| R5 | **S1p vs S1** | S1p worse (absence information matters), clear |
| R6 | **S2 vs C3** | no sign registered |

**Reading the margins.**
- A "not worse by more than x %" prediction holds if T̄ ≤ +x %, fails if T̄ > +x % and the contrast is clear, and is inconclusive otherwise.
- R4 holds if |T̄| ≤ 3 %, fails if |T̄| > 3 % and the contrast is clear, and is inconclusive otherwise.
- With 3 WCTE seeds SE_seed ≈ 0.9 % (Tier-1 SD_pooled 0.0157), so "clear" needs |T̄| ≳ 1.8 %.

### 2.6 Decision rule

- **Scale to SK** (a follow-up runbook): S1, S1n or S2 is better than C3 by ≥ 5 % on vertex **or** direction, clear, with the MLP and attentive probes of the same sign, and no worse than the best of C1, C2 and C3 on the primary outcome (beating a weakened dense control does not count).
- **Keep the fixed loss as the new default** if R2 holds, and the new embedding if R2b holds, even if nothing else wins.
- **Otherwise:** report and stop; the sparse family is a negative result on WCTE.

Everything else is exploratory and is labelled so.

### 2.7 Known limits (recorded, not fixed)

- WCTE is the least diluted detector (median photon-bomb occupancy ≈ 56 %), so the loss fix (F2) and the sparse encoder should matter less here than on SK or HK. WCTE is the cheap safety check; the real test of F2 is the SK follow-up.
- All arms keep the 50 ns time unit of C0. Tier-1 found smaller units better on WCTE (WT-p, from 450 ns: −2.9 % at 50 ns, −7.0 % at 5 ns, −8.7 % at 1.5 ns), so no arm here is WCTE's best configuration. The contrasts are internal and unaffected.


---

## Appended before any training (24 Nov 2026): W1 window and S1n ratio

**WCTE window (W1 rule, `results/round2/20261124_sparse/WINDOW_STUDY.md`):** q99.5(trigger − earliest signal digit) = 97.2 ns → **P = 110 ns**; the smallest multiple of 50 ns with ≥ 99.5 % of first-per-PMT signal digits in [−110, A] is **A = 50 ns** (99.996 %). Window **[−110, +50] ns** around the NHits trigger (old [−20, +60] ns). Placeholder time of unhit PMTs (C2, C3): (A + 50 ns) / 100 ns = **1.0**. W0 finding (recorded, rule applied unchanged): the early "signal" digits on WCTE are dark photoelectrons that opened a LUCiD integration window on a PMT hit later (observed 0.309 per event, expected 0.317); no early direct light is lost, but v1 dropped those PMTs entirely.

**S1n ratio (`results/round2/20261124_sparse/S1N_RATIO.json`):** SK `lucid_v1` `dev_probe_train` median kept fraction of masked unhit PMTs at r = 2: 0.386. WCTE `lucid_v1_w2` `dev_train` (20 runs, training acceptance): r = 0.25 → 0.117, 0.5 → 0.232, 1 → 0.463, 2 → 0.926. **r = 1.**
