# r2-sparse2 report: fair sparse rerun on WCTE (25 Nov 2026)

**Data.** Real LUCiD data only: `lucid_v1_w2` WCTE (LUCiD e651e836, window [−110, +50] ns). Three seeds (9000/9001/9002), 10,000 steps.

**Cells.** Three new cells, 9 runs: S1L, S2L and S2Lt. C2, S1 and S2 were reused read-only from r2-sparse.

**Pre-registration.** `PREREG_2026-11-25_R2_SPARSE2.md` was pushed at 166e388 (12:12 UTC), before any new run was evaluated.

**Scoring.**
- Primary outcome: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge.
- All tables: `SPARSE2_TABLE.md` / `.json`. The f_own diagnostic: `FOWN.json`.

## Answers

**1. Do sparse tokens reach dense level on equal components? Yes (P1 holds).**
- S1L is the sparse twin of C2: C2's linear embedding and full-attention decoder, but only visible hits as tokens.
- On sealed vertex, S1L is 0.0533 m and C2 is 0.0549 m: T̄ = −2.9 % (−2.5 / −2.2 / −4.1 % by seed), N_σ 2.3, clear. S1L is slightly *better* than C2, not merely within the 5 % margin.
- S1L is also clearly better on μ direction (−12.4 %), e direction (−8.7 %) and the attentive probe (−7.0 %).
- The MLP probe is level (−1.1 %, not clear). μ mis-ID is worse (+20.6 %, not clear).

**2. How much of the r2-sparse gap did the two handicaps explain? All of it (P2 and P3 hold).**
- The two handicaps were the Fourier code with content MLP, and the query-isolated decoder with count head.
- S1 → S1L: −37.0 % (clear). This closes 105 % of S1's gap to C2 (0.0846 → 0.0533 m, C2 0.0549 m).
- S2 → S2L: −12.5 % (clear). This closes 98 % of S2's gap (0.0629 → 0.0550 m).
- The r2-sparse verdict that the sparse family is worse than dense was therefore an artefact of the components, not of sparse tokens.
- **P4 (S2L vs C2):** +0.2 %, level. S2L is clearly better on both directions (−18.0 %, −15.2 %) and on the attentive probe (−16.0 %). Its μ mis-ID is much worse (+179 %, clear).

**3. Do time-width blocks make "when" learnable? No (P5 reported, P6 fails).**
- W1 measured a 32-hit block spanning a median of only 0.40 ns (LUCiD's 0.4 ns TDC tick). Δ = 5 ns was chosen by the registered rule (median event has 8 non-empty bins).
- S2Lt is clearly worse than S2L on vertex (+15.3 %, clear), on both directions (+19.6 %, +17.4 %) and on μ mis-ID.
- f_own is the share of a query's hidden-hit mass that falls on its own block or bin. Ratio of f_own to chance:

  | Cell | f_own / chance | f_own | chance |
  |---|---|---|---|
  | S2 | 2.04 | 11.6 % | 5.7 % |
  | S2L | 2.49–2.54 | 14.0–14.4 % | 5.7 % |
  | S2Lt | 1.23–1.24 | 74 % | 60 % |

- S2Lt is lower than S2L in every seed, so P6 fails. With 5 ns bins one or two prompt bins hold most hits, so chance is already 60 % and the model adds little above it.
- S2 reproduces the r2-sparse "8 %" (own-block mass over total mass, 7.8 %).
- The figure `FOWN_MAPS.png` (exploratory) shows summed predicted mass for S2 and S2Lt on 6 photon bombs.

## Decision (§2.6 with PREREG appendix C)

**The rule fires for S1L, so the sparse family goes to SK in a follow-up runbook.**
- S1L clears both entry conditions: it is not worse than C2 by more than 2 % (−2.9 %), and it is clearly better on vertex and on both directions.
- Its MLP and attentive probes have the same (negative) sign.

S2L and S2Lt do not fire:
- **S2L:** primary +0.2 %, but both probes are negative, so they do not share the primary's sign under the appendix-C reading. S2L is level with C2 on vertex.
- **S2Lt:** +15.5 % worse than C2.

## Caveats (read before citing)

- **S2L's vertex parity is present at initialisation.** S2L/S2Lt at step 0 (C2's encoder init, time-ordered sparse windows) already score 0.0554 m. Training changes S2L's vertex by +0.7 % and worsens S2Lt by 14.5 %. S1L improves from 0.0742 to 0.0533 m.
  - So S1L's result is learned. S2L's vertex parity is not; its direction gains are learned.
  - S2L and S2Lt have identical step-0 scores, as expected, since their encoders start identically.
- **μ mis-ID is the sparse family's weak point.** It is worse for every sparse cell: S1L +21 % (not clear), S2L +179 %, S2Lt +443 % (both clear). The likely cause is that unhit PMTs are absent from the encoder, so it has no explicit view of where PMTs stayed dark (§2.7).
- **WCTE is the least diluted detector** (≈ 56 % occupancy), so sparse tokens save the least here. SK is where they matter (§2.7).
- **Margin of the P1 result.** P1's clearness rests on the pooled seed SE (N_σ 2.3; SD_pooled 0.0216 over R1–R6, A1–A4 and P1–P5). All three seeds favour S1L.

## Deviations and decisions (all in `docs/round2/DECISIONS.md`)

- **f_own definition (PREREG appendix B, before evaluation).** f_own = own-block mass over hidden-hit mass, against a query-weighted chance level.
  - This is identical to §W1's chance for 32-hit blocks.
  - It is needed for unequal 5 ns bins, where the literal mean over blocks (1/K) would inflate S2Lt's ratio.
  - The total-mass form is reported too.
- **Decision-rule reading (PREREG appendix C, before evaluation).** "Probes of the same sign" means the same sign as the cell's primary contrast vs C2.
  - Under the looser reading "MLP and attentive share a sign", S2L would also fire. The decision to go to SK is unchanged either way.
- **Ties in count-32 blocks (known limit).** LUCiD times are quantised to 0.4 ns. S2/S2L blocks break time ties by PMT index, so leakage test (d) holds exactly only when no tie straddles a block boundary.
  - This is S2's registered rule, kept for S2L by design. S2Lt's bins never split a tie.
- **Review.** The fresh-context review returned PASS with non-blocking findings. Finding 1 (full attention without `query_valid`) was fixed before production.
- **Tests.**
  - Recipe check: 14 existing recipes have equal hashes and bitwise-equal step-0 weights.
  - Native suite: 1,654 passed; the 7 failures are the same pre-existing ones r2-sparse recorded.
  - `test_leakage_a_query_isolation_c2dc` misses its 1e-6 tolerance in some thread settings. It fails the same way on r2-follow code, so it is pre-existing.
- **Resources.**
  - GPU: 7.05 of 15 GPU-h booked (`R2_sparse2_gpu`); no cell was dropped.
  - Sealed packs: 18 reads, all through `scripts/tier1_eval.py` (`SEALED_ACCESS_LOG.md`).
- **Not available.** The f_own split by LUCiD direct/indirect light: the cache keeps only a physics flag per digit.
