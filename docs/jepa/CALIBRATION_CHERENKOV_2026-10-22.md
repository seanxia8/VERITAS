# A10 analytic Cherenkov calibration — Gate V (22 October 2026)

**Status: development evidence. Gate V passes for the public 500 MeV μ⁻ sample; e⁻ and
1000/1500 MeV particle events remain unvalidated.** The calibrated generator is
`src/jepa_noise/configs/lucid_bombs_v2_calibrated.json`; the derived mixed run config is
`src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json`.

## What was wrong

1. **Sensor indexing.** The pinned LUCiD `PromptTransport` uses the LUCiD geometry sensor
   order, which is a permutation of the order in `wcte_geometry.npz`. Comparing transport
   hits directly against the analytic model scrambled the pattern (Pearson **−0.19**).
   The map is now built and validated (`cherenkov_validation.sensor_id_map`); the same
   event gives Pearson **0.71** and median |Δt| **0.17 ns**.
2. **Light yield.** With the mapping fixed, the observed/expected detected-light ratio was
   **4.50** (median over 10 contained events). The analytic yield was scaled by this factor.

## Calibration and holdout

- Source: public PhotonSim water μ⁻ 500 MeV, 20 events (`e500_000..019.npz`).
- Containment: isometry to a vertical, centred track (`cherenkov_validation.contain_event`);
  all photons are inside WCTE for these events.
- Fit: `e500_000..009`; holdout: `e500_010..019` (never used for the fit).
- Fitted `cherenkov_yield_per_MeV`: **2248.5396** (from 500.0 × 4.497079).
- Cone sigma kept at 0.08 rad. Broad/isotropic secondary components were tested and
  **worsened** the pattern; the primary-track model is retained, with the secondary-light
  limitation recorded.

## Holdout result (official)

| Metric | Median | Gate |
|---|---:|---:|
| yield ratio observed/expected | **1.006** | 0.5–2.0 |
| module-pooled pattern Pearson | **0.902** | ≥0.85 |
| PMT log-charge Pearson | **0.815** | ≥0.70 |
| median absolute time residual | **0.157 ns** | ≤5 ns |
| event-level min module Pearson | 0.877 | ≥0.85 per event |
| verdict | **PASS** | — |

The PMT-level **linear** Pearson is ~0.6–0.7 and is deliberately not the gate: it is
limited by Poisson and detector-geometry noise even for a correct light pattern. The
registered validation metrics are the module-pooled pattern and the PMT log-charge pattern;
this choice was fixed before the holdout was evaluated and is recorded in
`cherenkov_validation.py`.

## Evidence

- `results/jepa/20261022_a10_calibration/calibration_scan_10.json` — fit events.
- `results/jepa/20261022_a10_calibration/validation_holdout.json` — harness output, `passed=true`.
- Host copies: `/work/dwong/jepa_a10/returns/a10_validation_holdout_calibrated.json`,
  `/work/dwong/jepa_a10/returns/a10_calibration_scan_10.json`.

## Limitations

- Only 500 MeV μ⁻ is validated. 1000/1500 MeV μ⁻ cannot be made contained by isometry;
  e⁻ has no PhotonSim sample in the repository (analytic only).
- The model is a primary-track ring approximation; secondary light is not explicitly
  modelled. This is acceptable for the representation study only with the per-domain
  label and the module-level validation gate; it is not a detector calibration.
- The `lucid_bombs_v2_calibrated.json` config is development evidence, not a reported
  physics result.

## Next after Gate V

1. Mixed `smoke --evaluate` with the calibrated config (per-domain probes/floors).
2. Mixed detectability per domain; freeze ceilings.
3. Mixed development cohort (probe form + β audit).
4. Mixed grid on deepthought2 `/work`.
