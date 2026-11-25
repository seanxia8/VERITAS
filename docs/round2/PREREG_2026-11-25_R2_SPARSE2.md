# PRE-REGISTRATION — r2-sparse2 (fair sparse rerun on WCTE), 25 Nov 2026

Copied verbatim from `docs/round2/RUNBOOK_2026-11-25_R2_SPARSE2.md` §2 (runbook commit af69bb2), followed by the W1 appendix (§2.3, §W1) and the operational definition of the registered f_own diagnostic. Pushed before any new run is evaluated. Later changes are dated amendments, never edits.

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

## Appendix A — W1 measurement and the block width Δ (25 Nov 2026, before any training)

Measured by `scripts/sparse2_w1.py` (commit 1c178bb) on all 100,000 `lucid_v1_w2` WCTE `dev_train` photon bombs (LUCiD e651e836, dataset_kind lucid_v1_w2; hit times in [−110, +50] ns, one digit per PMT):

| Quantity | median | q10 | q90 |
|---|---|---|---|
| hits per event | 633 | 233 | 1,702 |
| time span of a 32-hit S2 block (ns) | 0.40 | 0.00 | 2.40 |
| non-empty 2 ns bins per event | 17 | 14 | 20 |
| non-empty 5 ns bins per event | 8 | 6 | 10 |
| non-empty 10 ns bins per event | 5 | 3 | 6 |
| hits per non-empty 2 ns bin | 19 | 1 | 125 |
| hits per non-empty 5 ns bin | 30 | 1 | 272 |
| hits per non-empty 10 ns bin | 30 | 1 | 479 |

**Rule (§W1):** Δ = 5 ns unless the median event then has fewer than 6 non-empty bins. The median event has 8 non-empty 5 ns bins, so **Δ = 5 ns** (the default; §2.3 is unchanged). Bins: floor((t + 110 ns) / 5 ns), 33 bin indices (0–32; index 32 only for a hit exactly at +50 ns).

**Chance level of the own-block fraction for S2** (r2-sparse S2 seed-9000 recipe, read-only; `dev_probe_test`, the r2-sparse diagnostic event set = the first 8 events of each run, 1,000 events; mask seed 20261017): mean over 17,661 hidden blocks of (hits in the block) / (all hidden hits of the event) = **5.66 %**. All S2 blocks hold 32 hits, so the mean over queries is the same 5.66 %.

## Appendix B — operational definition of f_own (25 Nov 2026, before any new run is evaluated)

On `dev_probe_test` (the same 1,000 events as Appendix A, the event set of the r2-sparse diagnostic), step 10,000, for S2, S2L and S2Lt, using each model's own training mask (keyed 32-hit blocks for S2/S2L, keyed 5 ns bins for S2Lt):

- For a query q (one hidden hit) with assignment mass m_q over the candidate PMTs: own_q = mass on the hidden hits of q's own block or bin; hid_q = mass on all hidden hits.
- **f_own** = mean over queries of own_q / hid_q (the share of the query's mass on hidden hits that lies on its own block or bin).
- **chance** = mean over queries of n_k(q) / n_hidden (hidden hits in q's block or bin over all hidden hits of the event): the f_own of a predictor that spreads its hidden-hit mass uniformly over the hidden hits. For S2/S2L (equal 32-hit blocks) this equals the §W1 "mean over hidden blocks"; for S2Lt's unequal bins it is the matched baseline (the mean over bins is 1/K regardless of bin sizes and would inflate S2Lt's ratio by construction).
- **P6** compares f_own / chance between S2Lt and S2L per seed.
- Also reported (not used by P6): the total-mass fraction own_q (the r2-sparse "8 %" statistic), the mass on hidden hits and on unhit PMTs, and the literal mean-over-blocks chance.
- Computed by `SparseTimeArm.block_mass_stats` (`src/jepa_noise/sparse_arms.py`).

## Appendix C — reading of the decision rule and verdicts (25 Nov 2026, before any new run is evaluated)

- §2.6 "with MLP and attentive probes of the same sign": the MLP-probe and attentive-probe vertex contrasts of the cell vs C2 (sealed, T̄) both have the same sign as the cell's primary contrast vs C2. "Not worse than C2 by more than 2 %": primary T̄ ≤ +2 %. "Better than C2 clearly on vertex": primary T̄ < 0 and clear; "on both directions": μ and e direction contrasts both T̄ < 0 and clear.
- P2, P3 ("better, clear"): hold if T̄ < 0 and clear; fail if T̄ > 0 and clear; inconclusive otherwise. P1 by the §2.5 margin reading (+5 %). P6 holds if S2Lt's f_own / chance exceeds S2L's in all three seeds, fails otherwise.
- SD_pooled (per metric and cohort) = RMS of the across-seed SDs of r2-sparse R1–R6, r2-follow A1–A4 and this phase's P1–P5, all recomputed from the evaluator outputs.
