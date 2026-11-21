# r2-verify — Independent recomputation of the Round 2 load-bearing numbers

Runbook `docs/round2/RUNBOOK_2026-11-21_R2_VERIFY.md` (SHA256 `a7c9a48a…dfa07`), extension `r2_verify_2026-11-21`, `registered: false`. CPU only (Deepthought, ≤ 6 threads, nice 15). Every independent number was committed and pushed in the **blind seal `b755b23`** before any committed result was opened. Machine-readable outputs are in this directory; the comparison is in `v6_comparison.json`.

## 1. Summary: claim → check → recomputed → committed → tolerance → verdict

| Claim | Check | Recomputed (independent) | Committed | Tolerance result | Invariants | Verdict |
|---|---|---|---|---|---|---|
| C1: the original readout understates the content | K1 (a) all-PMT mean; (c) hit spatial moments | (a) e.g. SK MAE 0.9285 m, init 1.2747 m; (c) SK MAE 0.2444 m | (a) 0.9287 / 1.2755 m; (c) 0.1280 / 0.1830 m | (a) 30/30 pass; (c) 0/30 fail (definition, §4.1); own re-implementation of the committed form 11/11 pass (max 2.9 %) | I1 FAIL, I5 PASS | **Downgrade flag** (definitional, §4.1) |
| C2: the SK→HK degradation is a readout artefact | K1 (a), HK ÷ SK | s9000: 1.63 / 1.70 / 1.70 / 1.89; s9001: 1.62 / 1.52 / 1.39 (init/MAE/fixed/posfree) | identical values from the committed (a) | (a) 30/30 pass | I4 FAIL (2 of 7, both s9001) | **Downgrade flag** (§4.2) |
| C3: arm ordering under spatial moments | K1 (c) | MAE < init on 4/4 detectors × 2 seeds; fixed-label worst on 4/4 | same ordering | (c) values fail tolerance (definition) | I2 PASS, I3 PASS | **R (unchanged)** |
| C19: random init is a strong baseline | K1 (c) init vs (d) raw | SK 0.374 vs 0.864, JUNO 0.312 vs 0.569, HK 0.514 vs 1.513 m | 0.183 vs 0.427, 0.123 vs 0.212, 0.260 vs 0.837 m | (c), (d) fail (definition); own committed-style (d′) 4/4 pass | I6 PASS | **R (unchanged)** |
| C4: fixed-label JEPA collapses to static layout | K2 | static share 0.9967 / 0.9857 / 0.8966 / 0.9701 (WCTE/SK/JUNO/HK); f_event 7.8e-5 / 2.39e-4 / 2.08e-3 / 3.20e-4 | 0.9965 / 0.9849 / 0.8948 / 0.9684; f_event (s9000) — / 2.61e-4 / 2.32e-3 / 3.34e-4 | K2 16/16 pass; f_event 21/21 same order (sigreg within 4–10 %) | I7 FAIL, I8 PASS | **Downgrade flag** (§4.3) |
| C6: SIGReg per micro-batch scales with N | K3 | SK context EP 1.6085 / 3.2170 / 12.868 / 102.945 at N = 4/8/32/256 | 1.6085 / 3.2170 / 12.868 / 102.942 | N-ratios agree to 5e-6 | I9 PASS, I10 FAIL, I11 PASS | **Downgrade flag** (§4.4) |
| C8: the simulation is internally self-consistent | K4 + K1 | ToF mean 3.61 / 1.63 / 1.56 / 1.19 cm; timing CRLB 3.37 / 1.41 / 1.52 / 1.06 cm | ToF 3.64 / 1.52 / 1.64 / 1.13 cm; CRLB 1.86 / 1.43 / 1.51 / 1.09 cm | ToF 4/4; CRLB 3/4 (WCTE +82 %, model difference, §4.5) | I12 PASS, I13 PASS | **R (unchanged)** |
| C9: the best frozen readout is far from the ceiling | K4 + K1 | best (c) ÷ ToF = 14.9 / 12.6 / 28.8 (SK/JUNO/HK) | gap 8–16× (committed forms) | K1 (c) fails (definition) | I12, I13 PASS | **R (unchanged)** |
| C21: photon bombs are bright, high occupancy | K5 | HK median 2.6e4 PE, 9.6k PMTs hit (49 %) | display-only statement | descriptive | — | **V (descriptive)**, numbers revised (§3) |

Verdict rule (§V6.3): V = every tolerance and invariant passes; R = invariants pass but a tolerance fails; downgrade flag = an invariant fails. Every downgrade flag is diagnosed in §4. In each case the committed numbers themselves reproduce; the flag arises from how the claim or invariant is worded.

## 2. Invariants I1–I13

| Inv. | Statement (pre-registered) | Result | Evidence |
|---|---|---|---|
| I1 | (c) ≥ 3× better than (a) on SK/JUNO/HK, every case | **FAIL** | 18/21 ≥ 3; SK sigreg 2.52, JUNO sigreg 2.73, JUNO MAE s9001 2.52. With the committed construction: SK init 6.99, SK MAE 7.25 |
| I2 | (c) MAE < (c) init, 4 detectors × 2 seeds | **PASS** | e.g. SK 0.244 < 0.374, 0.242 < 0.379 m |
| I3 | (c) sigreg_jepa worse than (c) init, 4 detectors | **PASS** | WCTE 0.198 > 0.073; SK 0.868 > 0.374; JUNO 0.491 > 0.312; HK 1.045 > 0.514 m |
| I4 | (a) HK ÷ SK within ±20 % of R_fid ratio 1.912, every arm | **FAIL** | 5/7 inside [1.53, 2.29]; MAE s9001 1.52, posfree s9001 1.39 |
| I5 | shuffled-label error within 10 % of predict-the-mean | **PASS** | ratio 1.004–1.012 over 30 cases |
| I6 | (c) init < (d) raw on SK/JUNO/HK | **PASS** | both seeds, margin ≥ 1.8× |
| I7 | sigreg across-event fraction ≥ 50× smaller than MAE, static ≥ 0.95 (WCTE/SK/HK) | **FAIL** | fraction ratios 24.0 / 6.6 / 3.0 (committed values give 23.8 / 6.5 / 3.0); static 0.997 / 0.986 / 0.970 pass |
| I8 | JUNO sigreg less extreme than SK/HK | **PASS** | across-event fraction 0.103 vs 0.014 / 0.030; committed static 0.895 vs 0.985 / 0.968 |
| I9 | statistic ≈ linear in N | **PASS** | synthetic N(0, 4I) matches A·N + B within 0.6 %; real ratios 1 : 2.000 : 8.000 : 64.00 |
| I10 | 8×4 ≠ 1×32 (ratio far from 1) and gradient cosine < 0.9 | **FAIL** | ratio 0.1250 (pass); cosine 0.988 (fail); gradient-norm ratio 0.125 |
| I11 | independent vs official LeJEPA within 1e-4 | **PASS** | max relative difference 2.5e-8 (lejepa c293d29) |
| I12 | ToF converges ≥ 95 %, mean ≤ 5 cm (SK/JUNO/HK) | **PASS** | 100 %; 1.63 / 1.56 / 1.19 cm |
| I13 | ToF ≥ 5× better than the best (c) | **PASS** | 14.9× / 12.6× / 28.8× |

Other K1 results: (a) shrinks outer-quartile vertices inward by 0.5–1.0 m (SK MAE −0.51 m, HK init −1.03 m) while (c) is unbiased to within ±0.08 m in every quartile. The two-seed init spread is ≤ 0.046 m for (a) and ≤ 0.010 m for (c). Readout (b) passes tolerance in 30/30 cases.

## 3. K5 regime (photon probe-test cohort, 3400 events per detector; quantiles 5/25/50/75/95 %)

| Detector | Total PE | Hit PMTs | Hit fraction |
|---|---|---|---|
| WCTE (1995) | 302 / 809 / 2418 / 6865 / 16612 | 243 / 519 / 1039 / 1566 / 1852 | 0.12 / 0.26 / 0.52 / 0.79 / 0.93 |
| SK (11096) | 1680 / 4432 / 12619 / 35068 / 83616 | 1341 / 2783 / 5419 / 8440 / 10382 | 0.12 / 0.25 / 0.49 / 0.76 / 0.94 |
| JUNO (10000) | 1457 / 3669 / 9942 / 26277 / 60477 | 1211 / 2472 / 4900 / 7601 / 9301 | 0.12 / 0.25 / 0.49 / 0.76 / 0.93 |
| HK (19746) | 3200 / 8564 / 25994 / 75016 / 184747 | 2393 / 4972 / 9636 / 15003 / 18500 | 0.12 / 0.25 / 0.49 / 0.76 / 0.94 |

C21 holds qualitatively: the median event hits about half of all PMTs and the 5 % quantile still has about 12 %. The display-based numbers overstate the typical HK event. The cohort median is 2.6×10⁴ PE and 9.6k hit PMTs, below the quoted 11k–17k; that range lies between the cohort median and its 95 % quantile.

## 4. Discrepancies and diagnoses

**4.1 K1 (c), (d) fail tolerance 0/30 and 0/4.** My values are 1.3–2.6× larger than the committed ones. Diagnosis, from code read after the seal: `ext_r2_repair_probes.stream` with `r2_probe_repair.moment_features` builds `spatial_moment_latent` as [hit mean, charge-weighted mean (w ∝ q³), early-time-weighted mean (w ∝ q³·e^(−4Δt/T) over prompt hits), cross-moments with positions centred on the per-event hit centroid] (6×128). `raw_spatial_moment` uses the [ln q, t, hit] columns plus 15 summaries, which include the hit, charge-weighted and early-time-weighted position centroids. The runbook defines (c) as [z̄_hit, mean_hit(z·x̃/ỹ/z̃)] (4×128, uncentred). My own implementation of the committed construction reproduces the committed values in 11/11 encoder cases (max |Δ| 2.9 %) and 4/4 raw cases (max 2.0 %) (`v6_confirm/`). Centring alone gives 4/30. The committed numbers are therefore correct for their own definition; the runbook's (c)/(d) define a different, weaker readout. **Proposed fix:** the paper and ledger should name the readout explicitly ("weighted-mean + centred cross-moment readout"). C1's "5–10×" is confirmed only for that readout: 7.0–7.3× on the SK cases re-implemented, and 2.5–4.7× with the simpler runbook (c).

**4.2 I4 / C2.** The committed (a) values reproduce exactly (ratios agree to 0.01). The claim quotes only seed 9000 (1.63–1.89). Seed 9001 gives MAE 1.52 and position-free 1.39, outside ±20 % of 1.91. **Fix:** state the range over both seeds (1.39–1.89). The qualitative conclusion, a common degradation for all arms under (a), still holds.

**4.3 I7 / C4.** K2 reproduces the committed decomposition (16/16, layout within 0.01) and the C4 headline f_event values (sigreg within 4–10 %). I7 asked for a ≥ 50× ratio of the same-PMT across-event *fraction*. That fails on both my values and the committed ones (24 / 6.5 / 3.0×), because MAE tokens are themselves about 90 % static layout. The *absolute* across-event variance ratio is 312 / 99 / 48× (committed), and the event-mean ratio f_event(MAE)/f_event(sigreg) is about 140× on SK. **Fix:** state C4 with f_event or absolute variance, not the fraction ratio. Minor: for non-collapsed arms my f_event is 30–37 % below the 2026-11-12 readings (same order of magnitude), plausibly a different event sample.

**4.4 I10 / C6.** The N-scaling claim (1.61 at N = 4, 12.87 at N = 32) is reproduced to 5e-6. The independent statistic matches the official implementation to 2.5e-8. Accumulating 8×4 instead of 1×32 scales the value by 0.125 and the gradient norm by 0.125, but the gradient *direction* barely changes (cosine 0.988), because the evaluated embeddings are collapsed (per-dimension variance 1.6e-6). **Fix:** phrase C6 as "≈ 1/8 the intended regularisation *magnitude*". The cosine says nothing about pre-collapse training dynamics; on non-collapsed synthetic data it is 0.85.

**4.5 K4 CRLB on WCTE (+82 %).** By design my CRLB is timing-only; the committed CRLB adds Poisson-count information. On WCTE the charge information dominates: my simplified charge-only bound is 2.48 cm, and combining it with timing gives ≈ 2.0 cm vs the committed 1.86 cm. SK/JUNO/HK agree within 3 %, and ToF agrees within 8 % on every detector. n_g = 1.38 instead of the simulator's 1.33 would bias ToF by 0.4–0.8 m.

No repository code outside `verify/` was changed.

## 5. Independence statement

**Read before the seal `b755b23`:**
- This runbook.
- The tail of `docs/round2/DECISIONS.md` (r2-repair entries; these mention in prose that the repair CRLB uses Poisson-count plus time information, but contain no numbers used here).
- The output of the two §V0.4 extraction commands only (`INPUTS.txt`).
- Allowed repository code:
  - `point_encoder.py` (entire file; it contains fixture-only helpers `ridge_r2` / `amplitude_probe`, not used);
  - `lucid_arms.py` lines 1–135 (module docstring and config), 319–500 (`LucidArm`, `encode_tokens`), 758–775 (`build_arm`) and 1421–1430 (`random_init_floor`);
  - `r2_store.py` (`scaling_for`, `R2Cohort`, `open_cohort`, `tokenise_sparse`);
  - `r2_geometry.py` (`R2Geometry`, loaders);
  - import lines of `lucid_views.py` and `lucid_bombs.py`.
- Data and provenance: HDF5 packs and attributes; LUCiD geometry npz, physics and `materials/water.json`; `configs/lucid_pmt/reference.json`; and only the encoder/scaling config keys of each run's `run_record.json` / `recipe.json`.
- The official `lejepa` `EppsPulley` source, opened after my implementation was written.

Not imported: `lucid_arms` (so `sigreg.py` was never imported), and any `ext_r2*`, probe, ceiling or render code. `lucid_views.ViewMasker` was used as a black box for the K3 context view.

**Read only in V6 (after the seal):** the five committed JSON files of §V6.1; `20261112_eval/readings.json` (C4 f_event); `CLAIMS_LEDGER_2026-11-21.md`; `ext_r2_repair_probes.py` lines 60–240 and `r2_probe_repair.py` lines 1–100 (the diagnosis in §4.1).

**Code:** `verify/common.py`, `cases.py`, `k0_features.py`, `k1_dump.py`, `run_k1_dump.sh`, `k1_readout.py`, `k2_collapse.py`, `k3_sigreg.py`, `k4_tof.py`, `k5_regime.py` (all sealed at `b755b23`), and `v6_compare.py`, `v6_confirm.py`, `ridge_core.py`, `run_v6_confirm.sh` (V6). The encoder was used as a black box via `LucidPointJepa`. K0 shows the independent 9-feature rebuild equals the repository builder bit for bit on all four detectors.

**Limitation:** blindness was enforced by instruction, not by file permissions. This statement and the git history are the audit trail.
