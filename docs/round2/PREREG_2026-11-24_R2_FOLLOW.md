# PREREG r2-follow (24 Nov 2026)

Copied verbatim from `docs/round2/RUNBOOK_2026-11-24_R2_FOLLOWUP.md` §2 (runbook committed in this branch). Pushed before any new run was evaluated (no r2-follow run had been submitted at push time). Later changes are dated amendments below the amendments line, never edits.

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

## Amendments

(none)
