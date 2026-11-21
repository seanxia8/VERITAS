# r2-lucid — LUCiD regeneration, retraining and re-evaluation (21 Nov 2026)

Extension `r2_lucid_2026-11-21`, `registered: false`. Branch `lucid-mc`. Runbook `docs/round2/RUNBOOK_2026-11-21_R2_LUCID_REGEN.md` (with author addendum §L6b).

## 1. What changed and why
Every round-1 and round-2 dataset had been produced by our analytic photon-bomb generator, not by LUCiD. All four detectors were regenerated with the current public LUCiD; the same encoders, objectives, recipes (10,000 steps), probes and evaluation code were rerun on the new data. The analytic data are archived read-only, not deleted (`/ceph/dwong/jepa/data/ARCHIVE_ANALYTIC_TOY_2026-11-21.md`, `docs/DATA_LINEAGE.md`); toy results are now labelled "analytic-toy world".

## 2. LUCiD, capabilities and VERITAS-side steps
LUCiD `e651e836` (main, 20 Nov 2026; Apache-2.0 code / CC BY 4.0 data, used internally). **Sampling tier A1 (`lucid_mc_native`)**: LUCiD's `lucid-run-job` kernel (sampled transport and detection, absorption, Rayleigh/Mie scattering, wall/sensor reflection, wavelength-dependent QE, K = 12) plus LUCiD's digitizer (SPE charge, charge-dependent PMT jitter, 0.4 ns TDC, 0.25 PE threshold, 4.2 kHz dark noise) and, for SK/HK, LUCiD's trigger. Photon bombs use LUCiD's isotropic source; μ⁻ (300–1500 MeV) and e⁻ (100–1000 MeV) use LUCiD's SIREN emitters (no Geant4 available). VERITAS adds only: keyed source draws, conversion to the round-2 schema, the round-2 detector state (F1 gains, fixed parity gain/timing width), one hit per PMT, and the NHits token-time convention (N re-derived at 4.2 kHz: 6/10/10/12). Details: [LUCID_CAPABILITIES.md](LUCID_CAPABILITIES.md), [LUCID_INSTALL.md](LUCID_INSTALL.md), [DATA_LINEAGE.md](../../../docs/DATA_LINEAGE.md).

## 3. Validation gates
All six gates pass on all four detectors ([VALIDATION.md](VALIDATION.md)): loading and MAE/JEPA dry runs; PE linear in yield (slope ratio 1.000–1.002, R² ≥ 0.995), residual peaks within ±0.9 ns, late light 1.5–21 %; LUCiD sampled vs expected totals within 0.7 % (centre Pearson noise-limited, decided on a 2000-event re-run); regime within ×2; displays and ToF fit finite; no toy physics in the generation path (static and run-time tests).

## 4. Regime (120,000 events per detector; detected PE q5/q50/q95)
| Detector | Photon bombs | μ⁻ | e⁻ | Photons simulated |
|---|---|---|---|---|
| WCTE | 188 / 2,296 / 28,138 | 1,272 / 3,406 / 6,991 | 1,272 / 3,099 / 8,941 | 9.5 × 10⁹ |
| SK | 205 / 2,478 / 30,503 | 2,905 / 8,046 / 19,095 | 1,568 / 4,405 / 12,413 | 1.4 × 10¹⁰ |
| JUNO | 202 / 2,447 / 30,021 | 1,512 / 4,205 / 10,079 | 811 / 2,294 / 6,461 | 1.8 × 10¹⁰ |
| HK | 204 / 2,468 / 30,452 | 2,222 / 6,166 / 14,712 | 1,198 / 3,382 / 9,508 | 1.5 × 10¹⁰ |

The photon-bomb regime is deliberately much dimmer than the toy's. No event lacked an NHits trigger; no event failed LUCiD's trigger.

## 5. Toy versus LUCiD (spatial-moment Ridge, mean 3-D error in m, seed 9000; toy in parentheses)
| Detector | init | MAE | repaired MAE | repaired JEPA | fixed-label SIGReg JEPA |
|---|---|---|---|---|---|
| WCTE | 0.066 (0.053) | 0.058 (0.040) | **0.052** (0.041) | 0.063 (0.047) | 0.153 (0.140) |
| SK | 0.527 (0.183) | 0.427 (0.128ᶠ) | **0.356** (0.119) | 0.449 (0.189) | 1.151 (0.724) |
| JUNO | 0.423 (0.123) | – | **0.288** (0.091) | 0.379 (0.149) | – |
| HK | 0.907 (0.260) | – | **0.625** (0.167) | 0.812 (0.256) | – |

ᶠ frozen round-2 checkpoint. WCTE seed 9001 agrees (init 0.066, MAE 0.056, repaired MAE 0.052, repaired JEPA 0.061). Full tables incl. MLP, all-mean, raw-spatial, token controls, f_event/f_layout and particle tasks: [TOY_VS_LUCID.md](TOY_VS_LUCID.md); night-1 WCTE detail: [INTERIM_NIGHT1.md](INTERIM_NIGHT1.md).
- Errors are 1.2–3.7× larger on LUCiD (dimmer events, scattering, PMT jitter); the ordering repaired MAE < MAE < repaired JEPA < init ≪ fixed-label JEPA holds wherever the arms were trained. Gains over init are **larger** on LUCiD (repaired MAE +0.014/+0.171/+0.136/+0.283 m; repaired JEPA +0.003/+0.078/+0.045/+0.095 m).
- Fixed-label SIGReg JEPA still collapses (f_event 0.0002 WCTE, 0.0001 SK). Repaired JEPA does not (f_event 0.17/0.12/0.13/0.07).
- The approximate ToF-only Huber fit (not a likelihood ceiling) has medians 0.139/0.217/0.271/0.277 m on LUCiD vs 0.034/0.014/0.015/0.011 m in the toy.
- WCTE time-scale sweep (repaired MAE, seed 9000): the spatial error falls monotonically as the time-loss unit shrinks, 0.0541 (450 ns) → 0.0521 (50) → 0.0512 (15) → 0.0501 (5) → 0.0485 m (1.5 ns), while f_event drops 0.198 → 0.134.
- Particle tasks: μ/e AUC ≥ 0.98 for every arm including init; SK/JUNO/HK energy R² 0.98–0.996; repaired MAE gives the best directions (SK μ 5.6°, e 4.7°). WCTE energy is not evaluable (LUCiD containment flag, see §7).

## 6. Claims ledger on LUCiD data (proposal; ledger not edited)
- **Survive:** C1 (all-mean 3–8× worse than spatial readout, every detector), C3 robust part (MAE < init; fixed-label worst), C4 (collapse), C12 (repaired MAE improves all-mean strongly — SK 2.06 → 0.84 m — and spatial too), C13 (repaired JEPA does not collapse), C19 (init beats the raw-feature spatial baseline on SK/JUNO/HK, 0.53 vs 0.67, 0.42 vs 0.54, 0.91 vs 1.19 m, and now also on WCTE, 0.066 vs 0.075), C5/C6/C20 (code-level, unaffected).
- **Change:** C2 (normalised repaired-MAE errors 2.6 / 2.1 / 2.4 % R_fid on SK/JUNO/HK instead of 0.9/0.7/0.7 %; the "common to all arms" reading still holds), C14 (repaired JEPA beats init on WCTE in both seeds, by 3–5 mm — upgrade candidate from P to S), C11 (original MAE time share 0.45 % WCTE / 2.0 % SK, still small; repaired MAE 61–90 % on large detectors), C16 (electron energy R² JEPA 0.993 vs MAE 0.982–0.986 on SK/HK; muon mixed — remains P), C18 (worst events are the dimmest, 110–200 hits, on every detector), C21 (LUCiD regime is deliberately dim, not bright).
- **Fail:** C15 (repaired JEPA is *better* than init on SK/JUNO/HK on LUCiD: 0.449 vs 0.527, 0.379 vs 0.423, 0.812 vs 0.907 m).
- **Not testable on LUCiD / superseded:** C8, C9, C10 (no exact likelihood or CRLB for LUCiD data; only an approximate ToF fit exists, whose median is only 1.1–2.3× below the best SSL mean error on SK/JUNO/HK and above it on WCTE), C17 (Gate V compared LUCiD with the toy; obsolete), C7, C22, C-P3, C-R1, C-P3b (round-1 or toy-specific; unchanged status).

## 7. Open issues
1. Single seed for SK/JUNO/HK trained arms; WCTE has two seeds. The sweep is single-seed.
2. Particle truth: the LUCiD containment flag uses the SIREN 99.99 % emission extent, so WCTE energy tasks have no contained electrons; `expected_pe` holds sampled detected PE (no expectation in MC).
3. Particles use LUCiD's SIREN surrogate (no PhotonSim/Geant4 here), thinned to unit photons (rays with weight above the thinning bound are clipped; counts per run in each shard manifest). WCTE uses LUCiD's own scalar optics config and the `hk` digitizer preset (no 3″ PMT model in LUCiD); JUNO is LUCiD's water sphere.
4. Plain MAE was not retrained on JUNO/HK and SIGReg JEPA not on JUNO/HK (outside the priority list); their toy comparisons use frozen checkpoints only.
5. Native suite on the final code: 1556 passed, 1 failure that fails identically on `origin/dev` (`test_rate_zero_is_identical_to_the_pre_r6_reference`).
6. Process deviations (DECISIONS.md): batch staging through thin wrappers instead of `submit.py`; up to 16 concurrent batch jobs before the §L6b addendum was seen; SK/JUNO/HK production submitted before smokes (the HK W640 smoke then forced W128, as the rule requires).

GPU: 93.1 of 120 GPU-h (`R2_lucid_gpu`): generation 18.9, training 66.4, evaluation 3.2, self-checks 0.5, failed or replaced jobs 4.1.
