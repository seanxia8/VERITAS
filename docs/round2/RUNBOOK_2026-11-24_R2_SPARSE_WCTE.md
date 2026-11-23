# Runbook — Sparse hit tokens on WCTE: wider readout window, fixed loss, negative sampling, time-ordered position prediction (r2-sparse), 24 Nov 2026

**Phase** `r2-sparse`. **Branch** `sparse-tok`, created from `origin/tier1`. **Worktree** `~/Documents/VERITAS-worktrees/sparse-tok`. **Results** `results/round2/20261124_sparse/`.

Runs in parallel with `r2-mask` and `r2-tier1v`. It must not disturb either.

---

## 0. Why this runbook exists

On 23 Nov the author and co-author Junjie Xia reviewed how the current MAE/JEPA inputs are built. Six defects, or open questions, came out of that discussion. Each is checked against the code on `origin/tier1`.

| # | Finding | Evidence |
|---|---|---|
| F1 | Every PMT is a token. An unhit PMT gets time 0 and log-charge 0. Time 0 is the trigger time and log-charge 0 is ln(1 PE), so both collide with real values; only the hit flag separates them. | `point_encoder._event_features` and the `base` array (`point_encoder.py:163–184`) |
| F2 | The MAE charge and hit terms are averaged over **all** masked PMTs, while the time term is averaged over hit PMTs only. In a median LUCiD photon bomb at least ≈ 78 % (SK) and ≈ 87 % (HK) of PMTs are unhit, so easy zero targets dilute the charge and hit terms, by an amount that changes with brightness. | `lucid_arms._forward_mae_mse` |
| F3 | The readout keeps hits from trigger − 20 ns to trigger + 60 / 270 / 200 / 460 ns (WCTE / SK / JUNO / HK). The NHits trigger fires after the earliest signal digit by a median of 0 / 41 / 0 / 20 ns and a 95th percentile of 84 / 158 / 89 / 151 ns. Early direct light is therefore probably cut in many events, and late reflected light beyond the window is cut too. It is worse than a cut: `veritas_readout` picks each PMT's earliest digit before it applies the window, so a PMT whose earliest digit precedes trigger − 20 ns is dropped together with all its later, in-window light. | `results/round2/20261122_tier1/TIME_REFERENCE.json`; `configs/lucid_v1.json` `time_rule`; `lucid_mc_generator.veritas_readout` |
| F4 | Hit data are stored sparsely (one record per hit) and only densified at tokenisation. A hit-only tokeniser needs no new simulation. | `r2_store.tokenise_sparse` |
| F5 | The encoder embeds each token with a single linear layer over all nine features (position, direction, ln q, τ, hit), so position and content are mixed linearly. A linear function of xyz is a weak positional code. Masked tokens are replaced by one learned mask embedding; predictor queries are a linear map of the scaled xyz. For SK, HK and JUNO the direction is the inward wall normal, a function of position; for WCTE's multi-PMT modules it differs within a module but carries no response information in our LUCiD configuration. | `point_encoder.py:294–311`, `QUERY_FEATURES = 3` |
| F6 | For WCTE, which has no LUCiD trigger, the cached LUCiD digits keep every digit, and dark noise was generated over [first deposit − 100 ns, last deposit + 100 ns]. WCTE can therefore be re-packed with a different window from the cache, on CPU, without re-running LUCiD. SK and HK passed LUCiD's own trigger gate (200 ns plus 300 ns each side) before caching, so they cannot be widened this way. A WCTE pre-trigger longer than ≈ 100 ns reaches before the cached dark noise in most events, because the trigger sits at the first signal digit in the median event; §W2 step 2 fills that gap with LUCiD dark noise. | `lucid_mc_generator.py:418–470`; `dark_pad_no_trigger_ns = 100` |

The design proposals from the discussion are:
- a separate, fixed (not trained) positional code for PMT xyz, a small MLP for the content (charge, time, hit), and no direction feature;
- hit-only tokens;
- time ordering;
- predicting where masked hits are, with a many-to-one match to PMT surfaces;
- negative subsampling of unhit PMTs with a cap;
- a hit-count head.

This runbook tests them on WCTE, with 3 seeds, on real LUCiD data, in the order the author approved: first the minimal changes (window, fixed loss, new embedding, then the sparse encoder with spatial patch mask, negative sampling and count head), then the time-ordered position-prediction model.

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `sparse-tok`:

1. **Author, 23 Nov 2026, 23:40 CET:**
   > "根据我们的对话，写一个runbook，按照你上一条建议的顺序，在WCTE上3个seed试，记住要用真实LUCID数据，不要用玩具数据，那窗口又该怎么定呢？需要改吗？"

   (Translation: "Based on our discussion, write a runbook. Follow the order you proposed last, try it on WCTE with 3 seeds. Remember to use real LUCiD data, not toy data. How should the window be set? Does it need to change?")
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决". Never stop to ask; record decisions in `DECISIONS.md` and `$CC_STATE/notes/r2-sparse.md`.
3. **Author, 22 Nov 2026:** do not depend on the collaborator. The design ideas above are his; nothing in this runbook waits for him.

### 0.2 Binding rules

- **Real LUCiD data only.** Every dataset here is a re-packing of the cached LUCiD digits of `lucid_v1` (LUCiD `e651e836`, tier A1), or, if the cache is incomplete, a fresh LUCiD generation with the identical configuration. Never use the analytic generator (`lucid_bombs`, `r2_generator`) or any analytic-world pack or checkpoint. Assert `dataset_kind` and the LUCiD SHA in every pack manifest and in every training recipe.
- **Never overwrite existing packs, caches, checkpoints or returns.** The re-packed data get a new dataset kind, `lucid_v1_w2`, and new paths.
- **Pre-registration.** Copy §2 verbatim to `docs/round2/PREREG_2026-11-24_R2_SPARSE.md`. Append the window chosen in W1. Commit and push **before any training run is evaluated**. Later changes are dated amendments appended below the text, never edits.
- **Sealed set.** The rules of `RUNBOOK_2026-11-22_R2_TIER1_GATE.md` §T2.3 apply. The sealed cohort is re-packed with the new window (packing only, no metrics), and the repack is logged in `results/round2/20261124_sparse/SEALED_ACCESS_LOG.md`. Sealed packs are read only through `scripts/tier1_eval.py`, extended in T1.6.
- **Frozen readouts.** `docs/round2/READOUT_SPEC_v1.md` applies unchanged to every model (A.2 spatial-moment Ridge primary; MLP and attentive probes as specified). One known consequence is recorded, not fixed: the early-time-weighted mean uses prompt hits τ ≥ 0 only, so with a longer pre-trigger the earliest hits enter m₁, m₂ and C but not m₃. A variant with P = all hits in the window is reported as exploratory. The readout's raw features (ℓ = ln q, τ = t / 100 ns, hit flag) are always built from the pack in the standard encoding, independent of a model's input encoding (C2's ln(1 + q) and sentinel inputs, S1/S2's hit-only tokens). Only the token embeddings z come from the model.
- **Do not disturb other phases.**
  - Never touch other worktrees, branches, remote clones, returns or queues.
  - deepthought2 allows at most two GPU jobs across all phases. This phase uses at most one lane, and only when a slot is free; otherwise it uses HTCondor.
  - HTCondor: at most 4 concurrent jobs from this phase.
  - CPU: ≤ 6 cores on `dt1` at `nice -n 15`; portal1 up to one sixth of its cores, leaving ≥ 8 free.
- **Safety.**
  - Main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Push only to `origin/sparse-tok`, fast-forward only.
  - LUCiD: internal research use only.
  - JAX on shared GPUs: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`.
  - Poll at most every 25 minutes while waiting.

### 0.3 Paths and budget

| What | Path |
|---|---|
| Remote clone | `/ceph/dwong/jepa/r2_sparse_repo` |
| Re-packed data | `/ceph/dwong/jepa/data/lucid_v1_w2/wcte/` (shards, packs, sealed pack, manifests) |
| Returns | `/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/` (`R2_REPAIR_OUTPUT_ROOT=<this>/train`) |
| Scratch | `/work/dwong/r2_sparse/` |

GPU ledger line `R2_sparse_gpu`, cap **28 GPU-h**:

| Item | ≈ GPU-h |
|---|---|
| 21 WCTE trainings | 13 |
| Scoring | 4 |
| LUCiD regeneration of WCTE `dev_train`, only if its cache is missing (§W2) | ≤ 6 |
| Margin | 4 |

If the projection exceeds the cap, drop items from the bottom of §E.

Tooling, hosts and batch layout follow `r2-tier1`: read `$CC_STATE/notes/r2-tier1.md` (read-only) first.

---

## 1. The window: measure first, then fix it by rule

### W0 — Window study (CPU only, all four detectors, from the cached digits)

For 2,000 photon bombs, 500 muons and 500 electrons per detector, from the cached digits of `dev_probe_train`, with the NHits trigger recomputed exactly as the packs did:

1. **Lost early light:** the fraction of events, and of signal digits (physics flag), earlier than trigger − 20 ns; the distribution of (trigger − earliest signal digit).
2. **Late light:** the cumulative distribution of signal-digit times after the trigger, using first digits per PMT and all digits; the fraction later than the current window end.
3. **Dark noise:** dark digits per event inside candidate windows, as a function of pre- and post-trigger length; check that dark digits are uniform in time across each candidate window (they are generated only over [first deposit − 100 ns, last deposit + 100 ns]). For each candidate (P, A), report the fraction of events and the mean length of window not covered on each side, using the earliest and latest physics digit as proxies for the first and last deposit.
4. **Digits per PMT:** how often a PMT has more than one digit, and how late the later digits are.

For SK, JUNO and HK, report what the cache covers and mark SK and HK as truncated by LUCiD's trigger gate. Output `WINDOW_STUDY.md` and `.json`. This also answers the co-author's question about the true time range of the current samples.

### W1 — Choose the WCTE window by rule (record in the PREREG file)

- **Pre-trigger** P = max(50 ns, q99.5(trigger − earliest signal digit) + 10 ns), rounded up to 10 ns, at most 500 ns.
- **Post-trigger** A = the smallest multiple of 50 ns such that ≥ 99.5 % of first-per-PMT signal digits fall in [−P, A], at most 1,000 ns.
- **Coverage.** The window may reach outside the cached dark-noise span; this is expected on the pre-trigger side whenever P > ≈ 100 ns. §W2 step 2 fills the uncovered part with LUCiD dark noise. Do not shrink the window for this reason.
- **Reference.** Super-Kamiokande's vertex fitter uses 500 ns before and 1,000 ns after the trigger; the rule may not exceed that. WCTE is small (straight-line crossing ≈ 20 ns), so a much shorter A is expected.
- **Time scaling unchanged:** τ = t / 100 ns for WCTE, so the frozen readout's early-time weight keeps its meaning.

### W2 — Re-pack WCTE as `lucid_v1_w2`

1. Re-run the VERITAS readout on the cached LUCiD digits with the new window, through the cache path of `scripts/lucid_generate.py` (`run_from_cache`) with a modified `time_rule`, for `dev_train`, `dev_probe_train`, `dev_probe_test` and `sealed_test`. Keep the NHits trigger rule (N = 6, 20 ns) and the detector state unchanged. Mirror v1 exactly: the same runs, detector settings (s values), event order and pack layout.
2. **Dark-noise extension (WCTE).** The cache holds LUCiD dark noise only over [first deposit − 100 ns, last deposit + 100 ns], and deposit times are not cached. Use the event's earliest and latest physics digit as proxies for the first and last deposit. For every part of [trigger − P, trigger + A] outside [t_phys,min − 100 ns, t_phys,max + 100 ns]:
   - draw dark photoelectrons with LUCiD's `generate_dark_noise` (4.2 kHz, all PMTs);
   - pass them through the same LUCiD chain as `lucid_mc_generator.electronics`: `digitize_event`, `apply_readout_resolution`, the per-PMT width factor, TDC rounding and `apply_discriminator`;
   - key the random stream by `event_seed(seed, run_id, event_id, "dark_ext")`;
   - mark these digits dark (physics = False) and add them to the cached digits before `veritas_readout`;
   - where an added digit falls less than one integration window (200 ns) before a cached digit on the same PMT, LUCiD would have formed one digit: keep the earlier time, sum the charges, and count these merges (expected well under one per event).

   This is LUCiD's own noise model applied over a longer span, not new physics. If the extension fails validation, set P to the largest value that ≥ 99 % of events cover without it, and record that.
3. Store per hit PMT: the first digit's time and charge, plus the number of later digits and the total charge in the window. The extra fields are stored for later multi-hit work and not used here.
4. **If the `dev_train` cache is missing or incomplete,** regenerate WCTE `dev_train` with LUCiD (same configuration, same keyed seeds, so the same events) on HTCondor, booked under `R2_sparse_gpu`. Record it.
5. **Validate:**
   - event IDs and truths identical to `lucid_v1`;
   - every v1 hit is present in v2 with bit-identical time and charge (`veritas_readout` has no randomness), except on PMTs where an added dark digit now comes first; count those;
   - the trigger time equals v1's in ≥ 99.9 % of events;
   - signal-hit counts ≥ v1;
   - dark hits per event as predicted by W0, and uniform in time across the whole window after the extension;
   - every hit has q > 0 (record the minimum);
   - no analytic physics anywhere (static and run-time check, as gate G6 of `r2-lucid`).

   Write `DATA_MANIFEST_wcte_w2.json` and `SHA256SUMS`, and add `lucid_v1_w2` to `docs/DATA_LINEAGE.md`.

---

## 2. Pre-registration (copy verbatim to `PREREG_2026-11-24_R2_SPARSE.md`)

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

## 3. Stages

Order: T0 → W0 → W1 → W2 in parallel with T1 → PREREG push → smokes → E → T7 → T8.

### T0 — Setup (≤ 20 min)

- Status file and notes; commit this runbook; create the remote clone at the branch head.
- Inventory: `dt2` slots, the batch queue, which phases are running, and Ceph free space.
- Confirm F1–F6 of §0 against the code (one line each in the notes). A wrong one is recorded and the design adapted; never stop.
- Confirm the WCTE caches exist for `dev_train`, `dev_probe_train`, `dev_probe_test` and `sealed_test`, and record their coverage.

### T1 — Code, each part with a focused test

1. **Dataset kind** `lucid_v1_w2` in `data_roots`, with the window in its manifest. Loading a `lucid_v1_w2` pack while asking for `lucid_v1`, or the reverse, raises.
2. **Fixed loss and placeholders** behind flags of `mae_repaired`:
   - `--loss {legacy,fixed}`;
   - `--placeholder {legacy,sentinel}`;
   - `--embed {linear,fourier}` (§2.2a);
   - `--decoder {legacy,isolated}` and `--count-head` (C3 uses both; §2.2a).

   Defaults unchanged; the default recipe hash test of `r2-tier1` must still pass.
3. **Sparse tokeniser:** hit PMTs only, padded per batch with a key-padding mask; order `hilbert` or `time`; per token the position (for the positional code), ln q and τ; no direction, no hit flag.
4. **Sparse encoder path:**
   - same `EncoderConfig` width, depth, heads and window;
   - windowed attention over the per-event order with padding masks;
   - `encode_tokens` returns tokens aligned with the hit list.
5. **Arms** `sparse_mae` (S1, S1n, S1p via `--queries {all,neg,pos}`) and `sparse_time` (S2), with the count head, the S1n sampler and the S2 assignment loss. **Leakage tests (blocking):**
   - (a) S1 / S1n / S1p predictions for one query do not change when other queries are added or removed;
   - (b) S2 predictor inputs are unchanged when the positions of hidden hits are permuted;
   - (c) the count head output does not depend on the number of queries;
   - (d) S1n negatives all lie inside masked patches.
6. **Readout adapter** for sparse encoders:
   - scatter the hit tokens and the standard features (ℓ = ln q, τ = t / 100 ns, hit flag 1) into a (B, N_pmt) array, with zeros and hit flag 0 at unhit PMTs and active = all PMTs, then call the unchanged `r2_probe_repair.moment_features` and `pool_tokens`;
   - a padded hits-only array with active = hits would be wrong: the raw summary hit_count / active_count would be 1 in every event;
   - pooling modes that average over all tokens use the hit tokens for sparse models;
   - the attentive probe attends over the model's own tokens (all PMTs for dense models, hits for sparse ones); this difference is inherent and recorded.

   Extend `scripts/tier1_eval.py` to sparse models and to `lucid_v1_w2` packs; keep its sealed-access log. **Golden test:** on `lucid_v1`, a dense model's spatial-moment features computed through the adapter on its hit tokens equal those of the dense path for 200 events, for both outputs of `moment_features` (latent, and latent plus the 15 raw summaries).
7. Native suite. Then one fresh-context review of the diff with this checklist:
   - real LUCiD data only;
   - defaults unchanged;
   - leakage tests (a)–(d);
   - readout unchanged;
   - sealed access only through the evaluator.

   A BLOCK means fix and review again.

### Smokes

20 steps for each of C1, C2, C3, S1, S1n, S1p and S2 on `dt2` or batch: memory, step time, finite losses. Project runtimes before production.

### E — Training matrix (WCTE)

Submit in priority order. Each cell uses seeds {A, B, C}.

| Priority | Cells |
|---|---|
| 1 | C2, C3, S1, S2 |
| 2 | C1, S1p |
| 3 | S1n |

A failed run is diagnosed, fixed and resubmitted once; a second failure is recorded and the cell is skipped.

### T7 — Evaluation and tables

1. Score every run (and its step-0 snapshot) on validation and sealed: the primary readout, the MLP and attentive probes, the direction and PID suite, and the exploratory P = all-hits readout variant.
2. Also score: the C0 cells on `lucid_v1` sealed (from the Tier-1 returns, read-only, through the evaluator), and each data kind's raw-feature baseline.
3. `SPARSE_TABLE.md` and `.json`: every contrast R1–R6 with T̄, per-seed values, SE_boot, SE_seed, N_σ, clear; secondary outcomes; matched-init gains; the decision-rule verdict.
4. **Diagnostics (exploratory):**
   - S2's predicted-mass maps for 6 events (one figure);
   - S1 hit-probability maps against truth for the same events;
   - count-head calibration;
   - for S2, the fraction of predicted mass landing on unhit PMTs, split by LUCiD's direct/indirect flag of the hidden hits if available in the cache.

### T8 — Report

`results/round2/20261124_sparse/SPARSE_REPORT.md` (≤ 2 pages plus tables):
- the window study and the chosen window;
- the data validation;
- R1–R6 and the decision;
- what the result means for each of the co-author's proposals;
- deviations.

Also `docs/round2/ACCEPTANCE_2026-11-24_R2_SPARSE.json`. Set the status to `done` and print the report in full.

**Targets:** W0–W2 and T1 by Tue 24 Nov evening; trainings overnight; report by **Thu 26 Nov, 20:00 CET**. Commit what exists then, and finish the rest afterwards.

**Stops:** none beyond the safety rules and the GPU cap. A failed validation in W2 stops training on `lucid_v1_w2` until it is fixed.

---

## Launch (Mac Terminal, its own window, in the foreground)

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_sparse.sh
```
