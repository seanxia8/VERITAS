# Draft reply to Junjie, issue #5 (NOT POSTED)

_Draft for the author's review, 18 Oct 2026. Do not post until the author says so. Every number below is development evidence from synthetic response parameters; none of it is a calibrated WCTE result._

---

Hi Junjie,

Thanks again for the detailed corrections and for the thesis pointer. Here is where the PMT response work stands, followed by the decisions we need from you. Everything is on branch `codex/junjie-pmt-response`, not yet merged.

## What we fixed

1. **One response manifest.** Every simulation now builds its PMT response from a single versioned file, `configs/lucid_pmt/reference.json`. Every parameter there has a source, a status and a hash, and a test fails if any code builds a response from hard-coded numbers.
   - Per-channel gain and timing-width factors are applied per **mPMT module** (105 modules on WCTE, from the pinned geometry file).
   - Charge has one domain everywhere, [0.25, 1000] PE. Overflow is clamped and counted, never an error.
2. **Charge-dependent time-walk at a fixed threshold.** The old walk was a polynomial that was nearly flat over 0.25–20 PE. It is now `walk(q) = a + b/√q`, constant above q_sat = 50 PE: 4 ns at 0.25 PE, 0.5 ns at 16 PE.
   - The T-Q intervention changes the **slope** b (× 0.8, × 1.2). A pure offset change is kept as a separate control.
   - The conditional mean digit time follows the configured curve within 3σ at 0.25, 1, 2, 5 and 20 PE. Charges and hit membership are unchanged by a T-Q change.
   - A useful side result: on our prompt fixture, a pure timing offset has no effect on the vertex fit (K = 1.000, absorbed by the fit's t0). The slope change does have an effect (K = 1.03–1.14).
   - The width is still a Gaussian approximation, not your Eq. 6.3 kernel. We report the fraction of digits that land before the photon's arrival time.
3. **Honest trainability gates.** Gates are now stated relative to the constant predictor (the training mean), pooled and per energy:
   - vertex RMSE ≤ 0.8 × the constant, and ≤ 1.0 × in every energy;
   - direction error < 20°;
   - energy MAE ≤ 0.8 × the constant, and ≤ 1.0 × in every energy.

   The October multi-energy run measured against its constant predictor. Each entry is the model's error divided by the constant's; below 1 means better than the constant.

   | Endpoint (3 seeds) | Pooled | 500 MeV | 1000 MeV | 1500 MeV |
   |---|---|---|---|---|
   | Vertex RMSE / constant | 0.73–0.88 | 0.71–0.79 | 0.73–0.91 | 0.74–0.93 |
   | Initial-energy MAE / constant | 0.90–0.96 | 0.84–0.95 | **2.42–2.96** | 0.60–0.64 |

   Direction was learned well (7–9°). Initial energy was not: the model regressed to the mean, and at 1000 MeV it was worse than the constant. WCTE cannot contain 0.5–1.5 GeV muons, so initial energy is not observable for those events. This is why we need your call on the energy endpoint (D-E below).
4. **Dark counts.** The readout now has a dark-count stage: per-PMT Poisson dark hits, uniform in a 1 µs readout window. They get single-photon charge and the same T-Q walk and timing width. There are also dark-noise-only readout windows. The rate is a synthetic 1 kHz per PMT until we have yours.
5. **Energy endpoint and containment** (development defaults, pending your decision):
   - We use the primary muon's **deposited energy inside the volume**: kinetic energy at the vertex minus kinetic energy where it exits, 0 if it stops. It comes from PhotonSim's per-segment β.
   - A range-based check (PDG CSDA range in water, straight tracks, the same vertex and direction sampling) gives ≥ 50% containment in WCTE for μ⁻ at **100–350 MeV kinetic**: 100% up to 250 MeV, 94% at 300 MeV, 69% at 350 MeV, 35% at 400 MeV.
   - Ignoring multiple scattering makes these figures roughly a lower bound. The public PhotonSim samples start at 500 MeV, so contained tracks would need new generation.

## Development re-run with all of the above (R-4)

Setup: the same 60 PhotonSim events (500, 1000 and 1500 MeV μ⁻, 8 placements each), three seeds, and the energy target switched to deposited energy. Each entry below is model error ÷ constant-predictor error.

| Seed | Vertex, pooled | Direction | E_dep, pooled | E_dep, worst energy |
|---|---|---|---|---|
| 30 | 0.97 | 9.0° | 0.64 | 0.79 |
| 31 | 0.96 | 9.6° | 0.67 | 0.73 |
| 32 | 1.00 | 9.5° | 0.88 | 1.32 |

- **Direction** is fine.
- **Energy:** deposited energy is learnable, but total charge alone does as well (0.09 relative MAE). For escaping tracks it mostly measures path length inside the volume: about 340–370 MeV at every energy.
- **Vertex.** Vertex failed the gate in R-4. The cause was the time-feature origin, not dark counts or the time unit. We had moved the origin to the readout-window start, which puts every hit time about 200 ns (10 units at the 20 ns unit) away from the 0 given to unhit sensors. That is a conditioning problem: the timing information was unchanged, but training suffered. With the origin at t = 0, vertex reached 0.65–0.68 × constant on a validation split. In this simulation t = 0 is the true emission time, which is not available in data, so that is only an optimistic bound. The final rule uses a hit-based trigger: the earliest digit with at least 4 digits within 20 ns, falling back to the window start if there is none. Digits outside −20 to +60 ns of the trigger are dropped, and the unit is 20 ns. On the validation split this brings vertex to 0.79–0.84 × constant with dark counts on. On the test events (R-4c) it gives 0.81–0.88, still above the 0.8 gate in all three seeds. R-4, R-4b and R-4c are recorded as development fails on this 60-event pilot, and we will run nothing more on those events. The vertex gate will be re-evaluated in the next stage on fresh events, with the rule frozen.

  | Run (test events, vertex RMSE ÷ constant, pooled) | Seed 30 | Seed 31 | Seed 32 |
  |---|---|---|---|
  | R-4: window-start origin, 1 µs unit, dark counts on | 0.965 | 0.960 | 0.998 |
  | R-4b: window-start origin, 20 ns unit, dark counts on | 0.992 | 0.998 | 1.000 |
  | R-4c: hit trigger (N = 4) + prompt window, 20 ns unit, dark counts on | 0.812 | 0.877 | 0.883 |

  Direction and deposited energy pass in R-4c.
- All of this is development evidence on synthetic response values, not a WCTE calibration.

## Decisions we need from you

1. **D-E, the energy endpoint.** Our preference is **(A)**, with K's energy endpoint on **contained μ⁻ at 100–300 MeV kinetic**, where deposited energy equals initial energy. E_dep on escaping tracks (primary muon only) would stay as a diagnostic. Do you agree, or would you prefer:
   - (B) dropping energy, keeping vertex and direction only;
   - (C) something else?

   If (A), is primary-muon kinetic-energy loss the right truth for the diagnostic, or should it include the secondaries' deposits too?
2. **A measured T-Q curve.** Can you give us a measured or thesis-derived walk curve for the target PMT, in forward time and PE? An Eq. 5.13-type fit with its charge-coordinate conversion would do. Until then, a = 0, b = 2 ns and q_sat = 50 PE are synthetic placeholders.
3. **Target QE and CE for the WCTE 3-inch PMTs.** Could you send QE(λ) (≥ 3 wavelength points) and the collection efficiency? The current curve is synthetic.
4. **Crate and cable map.** Is there a channel-to-crate or channel-to-cable map for WCTE? The module response currently uses mPMT modules; crates would follow your original request more closely.
5. **Dark rate.** What dark rate per PMT should we use for the WCTE 3-inch PMTs, and does it depend on temperature or channel?
6. **Portfolio decisions:**
   - **D-T:** keep NuRadioMC or move to JAXTPC.
   - **D-L:** LUCiD as an attribution control and harm replication, or out of the paper (for Paper 3 / ORACLE; the JEPA study uses LUCiD regardless).
   - **D-M:** exclude simulation-model changes from the N/S cells.
   - **D-S:** the clean signal family and held-out definition for TIDMAD.
7. **Realistic drift sizes.** Are module gain drifts of 2/5/10% per run, an independent per-PMT gain spread of 5%, and a 10% attenuation-length change realistic for WCTE?

Thanks,
Dowling
