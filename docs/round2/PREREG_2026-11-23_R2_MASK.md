# PREREG — r2-mask (shortcut budget and mask scale), 23 Nov 2026

Section 1 of `docs/round2/RUNBOOK_2026-11-23_R2_MASK_SCALE.md`, copied verbatim (runbook §0.2), followed by the M1 numbers it asks for (WCTE and SK) and the M3 expectation (runbook §M3.5). Committed and pushed before any M2 run is evaluated (no M2 checkpoint has been scored at this commit).

---

## 1. Pre-registration (copy verbatim to `PREREG_2026-11-23_R2_MASK.md`)

### 1.1 Mask configurations

All hide about half of the PMTs.

| Name | Rule |
|---|---|
| `parity` | `within`: the same PMTs in every event |
| `random` | a fresh uniform half of the PMTs per event |
| `patch(s)` | PMTs ordered along the Hilbert curve and cut into contiguous patches of s PMTs; a fresh half of the patches per event. With an odd number of patches, hide ⌈count/2⌉. |

Sizes: WCTE s ∈ {8, 32, 128, 256}; SK s ∈ {43, 173, 692}. The defaults (32 and 173) are unchanged and their existing runs are reused. For every configuration record the realised hidden fraction and the fraction of PMTs that is never hidden.

### 1.2 Diagnostic quantities (M1; no neural network)

Targets and scalings are exactly those of the default training loss: hit flag; for hit PMTs, log-charge / 3 and time in units of 50 ns. All quantities are computed on `dev_probe_train`, with fitting and scoring on disjoint events.

- **Static predictor S:** each PMT's mean target over the fitting events.
- **Local predictor L:** one Ridge regression per detector and channel, shared across PMTs, from the same event's k = 8 nearest *visible* PMTs (their hit flags, log-charges, times and distances) plus the static mean.
- **Explained fractions** on scoring events: E_S and E_L (1 − MSE / variance), per channel and combined with the default loss weights.
- **Shortcut budget:** static = E_S; local = E_L − E_S; remainder **U = 1 − E_L**. U contains irreducible noise, which is the same for every mask on a detector, so only differences in U between masks are interpreted.
- **AvgDist:** mean distance from a hidden PMT to its nearest visible PMT, in metres and in units of the median nearest-neighbour spacing.
- **Correlation length ℓ_c:** the distance at which a single visible neighbour's explained fraction of log-charge falls to half of its nearest-neighbour value.

Report photon bombs (primary), muons and electrons separately.

### 1.3 Outcomes

Mean 3-D vertex error on sealed photon bombs with the frozen spatial-moment Ridge readout, as gain over the run's matched init (step-0 snapshot). Secondary: μ and e direction, energy resolution, and the MLP and attentive probes. Statistics (SE_boot, SE_seed, N_σ, "clear") as in `PREREG_2026-11-22_R2_TIER1.md` §2.2.

### 1.4 Predictions

- **Q1 (ordering of U).** U(parity) ≤ U(random) < U(patch), and U(patch(s)) increases with s.
- **Q2 (U predicts the gain).** Over the configurations {random, patch(s)} of a detector, the Spearman correlation between U and the vertex gain is ≥ 0.8. The direction gain has the same sign of correlation.
- **Q3 (scale).** On each detector, the configuration at which half of the gain from `random` to the best configuration is reached has AvgDist within a factor 3 of ℓ_c.
- **Q4 (destination, not rate).** The contrast patch(default) vs `random` on SK is at least as large at 10,000 steps as at 5,000 steps (unlike the time-unit contrast, which shrank).

### 1.5 Kill criterion

**KM.** On SK the best and worst configurations differ by less than 5 % or the difference is not clear, **and** the Spearman correlation of Q2 is below 0.5 on both detectors. Then the shortcut-budget claim is not the lead.

Everything else is exploratory and is labelled so.

---

## Appendix M1 — diagnostic numbers (appended before any M2 evaluation)

Source: `results/round2/20261123_mask/SHORTCUT_BUDGET.json` (code `1a2a0ab0`, `scripts/mask_shortcut_budget.py`). Targets as in the default `mae_repaired` loss (ln q unscaled, see DECISIONS 2026-11-23); photon bombs 1,700 fitting + 1,700 scoring events (all of `dev_probe_train`), μ and e 1,000 + 1,000.

### WCTE (N_PMT 1995, NN spacing 0.072 m, default patch 32)

| Config | hidden | never hidden | E_S | E_L − E_S | U (photon) | AvgDist m (spacings) | U (μ) | U (e) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parity | 0.526 | 0.474 | 0.0058 | 0.7111 | 0.2831 | 0.078 (1.09) | 0.2422 | 0.3285 |
| random | 0.500 | 0.000 | 0.0059 | 0.7086 | 0.2855 | 0.083 (1.16) | 0.2548 | 0.3374 |
| patch8 | 0.501 | 0.002 | 0.0062 | 0.7033 | 0.2905 | 0.160 (2.23) | 0.3026 | 0.3755 |
| patch32 | 0.497 | 0.006 | 0.0058 | 0.6888 | 0.3054 | 0.293 (4.08) | 0.3860 | 0.4462 |
| patch128 | 0.513 | 0.038 | 0.0056 | 0.6591 | 0.3352 | 0.474 (6.60) | 0.4961 | 0.5519 |
| patch256 | 0.513 | 0.102 | 0.0058 | 0.6303 | 0.3638 | 0.604 (8.40) | 0.6161 | 0.6210 |

ℓ_c (charge channel): photon 2.48 m (34.5 spacings), muon 0.66 m (9.2 spacings), electron 0.68 m (9.4 spacings). Q3 uses the photon-bomb ℓ_c (primary outcome is the photon-bomb vertex).

### SK (N_PMT 11096, NN spacing 0.698 m, default patch 173)

| Config | hidden | never hidden | E_S | E_L − E_S | U (photon) | AvgDist m (spacings) | U (μ) | U (e) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parity | 0.526 | 0.474 | 0.0172 | 0.4335 | 0.5493 | 0.705 (1.01) | 0.5181 | 0.5834 |
| random | 0.500 | 0.000 | 0.0174 | 0.4340 | 0.5486 | 0.723 (1.04) | 0.5193 | 0.5842 |
| patch43 | 0.500 | 0.000 | 0.0179 | 0.4302 | 0.5519 | 1.503 (2.15) | 0.5592 | 0.6012 |
| patch173 | 0.499 | 0.002 | 0.0168 | 0.4183 | 0.5650 | 2.592 (3.71) | 0.6023 | 0.6265 |
| patch692 | 0.499 | 0.002 | 0.0168 | 0.3837 | 0.5995 | 4.918 (7.05) | 0.6771 | 0.6826 |

ℓ_c (charge channel): photon 13.90 m (19.9 spacings), muon 3.26 m (4.7 spacings), electron 4.20 m (6.0 spacings). Q3 uses the photon-bomb ℓ_c (primary outcome is the photon-bomb vertex).

Q1 as computed on these numbers (photon bombs, before any M2 evaluation): see SHORTCUT_BUDGET.md; the U ordering is a property of the diagnostic only, and Q2–Q4 and KM are evaluated on M2 outcomes.

## Appendix M3 — registered expectation (appended before running M3)

Copied from runbook §M3: 5. Registered expectation, appended to the PREREG file before running: on SK the good-basin fraction is ≥ 80 % for start (c) and ≤ 30 % for start (b). **Drop this piece** if (c) is below 50 %.

Operational definitions (fixed here, before M3 runs): good basin = final 3-D vertex error ≤ 2 × the median final error of the truth start (a) for the same detector and fit mode; primary fit mode counts-only; start (c) = frozen spatial-moment Ridge readout (READOUT_SPEC_v1) of `<det>__mae_repaired__s9000` at step 10,000 fitted on `dev_probe_train`; start (b) = charge-weighted centroid of hit PMT positions.
