# MASK_REPORT — r2-mask: shortcut budget, mask scale, readout-seeded LUCiD fit (23–24 Nov 2026)

Runbook `docs/round2/RUNBOOK_2026-11-23_R2_MASK_SCALE.md`; pre-registration `docs/round2/PREREG_2026-11-23_R2_MASK.md` (§1 verbatim, M1 numbers and the M3 expectation), pushed at 19:50 UTC on 23 Nov (2c0204b), before any M2 checkpoint was scored. The first sealed read came after the push. Details: `SHORTCUT_BUDGET.md`, `MASK_SCALE.md` (+ `.png`), `SEED_TEST.md`, `SEALED_ACCESS_LOG.md` (51 reads, all through `scripts/tier1_eval.py`). GPU: 36.0 of 40 GPU-h (`R2_mask_gpu`).

## 1. Shortcut budget (M1, no network; photon bombs, scoring events)

| Detector | Config | E_S (static) | E_L − E_S (local) | U = 1 − E_L | AvgDist (m) |
|---|---|---:|---:|---:|---:|
| WCTE | parity / random | 0.006 / 0.006 | 0.711 / 0.709 | 0.283 / 0.286 | 0.078 / 0.083 |
| WCTE | patch 8 / 32 / 128 / 256 | 0.006 | 0.703 / 0.689 / 0.659 / 0.630 | 0.291 / 0.305 / 0.335 / 0.364 | 0.16 / 0.29 / 0.47 / 0.60 |
| SK | parity / random | 0.017 / 0.017 | 0.434 / 0.434 | 0.549 / 0.549 | 0.71 / 0.72 |
| SK | patch 43 / 173 / 692 | 0.017–0.018 | 0.430 / 0.418 / 0.384 | 0.552 / 0.565 / 0.600 | 1.50 / 2.59 / 4.92 |

- **The identity shortcut is negligible:** E_S ≤ 2 % on every detector and particle type.
- **Interpolation from visible neighbours is the cheap route.** It explains 38–71 % of the target variance, and contiguous patches remove part of it.
- **The U range is much wider for μ and e than for photon bombs.** On WCTE, μ U runs from 0.25 to 0.62.
- **Correlation length ℓ_c (charge channel):** photon bombs 2.48 m (WCTE) and 13.9 m (SK); μ/e 0.66–0.68 m (WCTE) and 3.3–4.2 m (SK).
- **JUNO and HK (exploratory) show the same pattern** (`SHORTCUT_BUDGET.md`).

## 2. Patch-size sweep (M2): sealed vertex error, frozen spatial-moment Ridge, 10,000 steps

| Detector (seeds) | Config | G vs matched init | T vs random | N_σ | clear |
|---|---|---:|---:|---:|---|
| WCTE (A, B, C) | random | +17.4 % | – | – | – |
| | patch 8 / 32 (default) / 128 / 256 | +19.8 / +21.6 / +19.5 / +18.0 % | −2.9 / −5.0 / −2.5 / −0.7 % | 2.8 / 5.0 / 2.4 / 0.7 | yes / yes / no / no |
| SK (A, B) | random | +23.2 % | – | – | – |
| | patch 43 / 173 (default) / 692 | +30.0 / +33.9 / +34.6 % | −8.8 / −13.9 / −14.8 % | 3.8 / 6.1 / 6.4 | yes / yes / yes |

- **Secondary probes follow the vertex on SK.** MLP, attentive and μ/e direction are all clear for every patch size vs random; energy is unaffected.
- **Reproduction check.** The re-scored reused runs reproduce the committed r2-tier1 numbers: WCTE s9000 0.05330 m, and SK CM-50 −14.0 % / −13.8 %.

**Verdicts (as registered):**
- **Q1, U ordering:** holds on WCTE. On SK it fails on one link only: U(parity) = 0.5493 vs U(random) = 0.5486, a tie at the 10⁻³ level. The other links hold on SK: random < every patch, and U increases with s.
- **Q2, U predicts the gain (ρ ≥ 0.8, direction same sign):** holds on SK (ρ = 1.00 over 4 configurations; direction ρ 0.8 / 0.8). It does not hold on WCTE (ρ = 0.10). There, the gain peaks at the default patch 32 and falls for 128 and 256, although U keeps rising.
- **Q3, scale:** does not hold on either detector. The half-gain AvgDist is 0.15 m on WCTE and 1.38 m on SK, against photon-bomb ℓ_c of 2.48 m and 13.9 m (ratios 0.06 and 0.10). *Exploratory:* against the μ/e ℓ_c the ratios are 0.22–0.23 on WCTE and 0.33–0.42 on SK.
- **Q4, destination not rate:** not met as worded. The SK patch(default) vs random contrast is −13.9 % at 10k vs −14.5 % at 5k. *Exploratory:* the paired difference is +0.6 points (N_σ 1.7, not clear), so the contrast did not shrink materially.
- **KM:** does not fire. On SK, best (patch 692) vs worst (random) is −14.8 %, clear (N_σ 6.4), and SK's ρ is 1.0.

## 3. Readout-seeded LUCiD fit (M3; 200 Y2 photon bombs per detector)

Good-basin fraction is the share of events whose final error is ≤ 2 × the median truth-start final error.

| Detector, mode | (a) truth | (b) centroid | (c) readout s9000 | (d) readout, init | (e) raw baseline |
|---|---:|---:|---:|---:|---:|
| SK, counts-only (primary) | 88.5 % | 0.0 % | **47.5 %** | 27.5 % | 21.5 % |
| SK, joint | 87.5 % | 68.5 % | 75.5 % | 68.0 % | 73.0 % |
| WCTE, counts-only | 68.0 % | 11.0 % | 30.5 % | 20.0 % | 19.5 % |
| WCTE, joint | 84.0 % | 86.5 % | 95.5 % | 93.5 % | 95.5 % |

**Registered expectation (SK, counts-only):** (c) ≥ 80 % and (b) ≤ 30 %.
- The (b) part is met: 0.0 %.
- The (c) part is not met: 47.5 %. That is below 50 %, so **this piece is dropped** (§M3.5).
- **Caveat:** the yardstick's counts-only L-BFGS-B fits stop after about 5 iterations from every start. The final error therefore tracks the start error, and the test mostly ranks the starts.
- **Joint fits (exploratory):** all starts reach similar final errors, including the centroid.

## 4. Deviations (all in `docs/round2/DECISIONS.md`)

- **M1 photon-bomb counts:** 1,700 + 1,700 events (the whole `dev_probe_train` cohort) instead of 2,000 + 2,000.
- **M1 targets and masks:** targets follow the code's `mae_repaired` loss (ln q unscaled, not /3). Masks use n_min = 0, so every configuration sees the same events.
- **Reused runs:** the r2-tier1 and r2-lucid runs were re-scored from their checkpoints, read-only, into the r2-mask tree.
- **Step 5,000:** scored for every cell on both detectors (the four new SK cells were added after a first pass that covered only the Q4 cells; verdicts unchanged).
- **Matched init:** one step-0 score per seed. The weights are identical across masks (hash-verified).
- **M3 truth start:** re-fitted in the same job rather than read from the r2-tier1 Y2 output.
- **Batch tooling:** `tier1_eval_batch.py` now honours an r2-mask job tree and access log. The collector's `.prev_` copies are confined to the r2-mask returns.
- **CPU share:** portal1 briefly ran at 24 threads against a 21-core share (≥ 8 cores stayed free).
- **No runbook stop fired.** Everything finished before the Thursday 20:00 CET target.

## 5. What the lead claim may and may not say

**May say:**
- On a fixed sensor array, the PMT-identity shortcut is negligible (≤ 2 % of the target variance).
- The cheap route is interpolation from visible neighbours. It can be computed before training, and contiguous masks shrink it.
- On SK, larger contiguous patches leave more variance after interpolation (higher U), and the frozen-readout vertex gain follows this ordering exactly: ρ = 1.0 over random and three patch sizes, two seeds, patch vs random up to −15 %, clear.
- The SK mask contrast is already present at 5,000 steps and persists at 10,000.
- The kill criterion did not fire.

**May not say:**
- That U predicts the gain in general. On WCTE the gain peaks at a moderate patch size and falls beyond it while U keeps rising (Q2 fails). The relation holds only up to a detector-dependent scale.
- That this scale is set by the photon-bomb correlation length (Q3 failed as registered).
- That the gain is a "destination" effect beyond the SK default contrast (Q4 is borderline).
- That a frozen readout can replace the seed search of LUCiD's likelihood fit. The registered SK test is not met and the piece is dropped.

The SK evidence rests on two seeds and four configurations. The claim is exploratory beyond SK until it is reproduced with a third seed or on another detector.
