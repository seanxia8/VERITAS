# Memo — simulation testbeds for validating the physical latent and its diagnostics

**To:** ORACLE / Paper 3 working notes  **From:** Dow (drafted with Claude)  **Date:** 11 September 2026
**Re:** Which non-water-Cherenkov simulators could carry the *physical-latent* validation, and in what role
**Status:** proposal — nothing here is built; repo links and licences below were checked on 11 Sep except where marked "verify"

---

## 1. Why this is a different question from the testbed survey

`docs/TESTBEDS.md` ranks simulators against §0: physics controllable, geometry controllable, noise ours (a
clean sampled trace with no incumbent electronics model). That criterion serves the covariance claim — the
whitening lemma, κ(Σ̂⁻¹Σ), the N-vs-S dissociation.

The structured latent of TESTBEDS §2.2 asks for more. Its decoder is

    x̂_c(t) = a_c · s(t − τ_c ; z_shape) + r_c(t; z_free),    a_c = A(z_amp, z_pos, g_c),   τ_c = T(z_pos, g_c)

— a per-channel *share* times a *delayed template*, with a small free residual. For the §2.4 lookup to be a
test rather than a description, we need arms where that form is **physically true** (so a physics change must
land in `z_shape`/`z_pos`/`z_amp` with the residual at reference) and arms where it is **known to be false in a
predictable way** (so the residual-branch energy must rise and the abstain row must fire). We also need a
labelled shape/type axis richer than "ER vs NR at fixed energy", and a position that is read out *through*
`g_c` rather than learned as a channel pattern.

The two PMT arms (LUCiD, Prometheus) are one slot — same readout physics, same interface — and arm B (HeST)
has the fixed-energy ER/NR degeneracy. So the question is: which other solid simulations give a true-decoder
arm, a false-decoder arm, and, if wanted, a DELight-relevant crystal arm.

## 2. Candidates, by what each latent block maps to

| simulator | readout physics | z_amp / z_pos / z_shape / type | decoder form a_c·s(t−τ_c) | geometry axis | noise | licence · state | link | recommendation |
|---|---|---|---|---|---|---|---|---|
| **GW detector network** — bilby (+ PyCBC, LALSuite waveforms) | laser interferometers, 2–5 channels, 4–16 kHz strain | luminosity distance / sky position via inter-detector delays / masses & spins (waveform family) / BBH, BNS, NSBH, glitch | **exactly true**: antenna patterns F₊, F× are the share, light-travel delays are τ_c | detector network as config: H1L1, +V1, +K1, LIGO-India, hypothetical sites | published PSDs — Σ̂ is public; coloured Gaussian generation; **real noise with labelled glitches** (Gravity Spy) | bilby MIT ✅; PyCBC GPL-3, LALSuite GPL-2+ (verify); Bilby posteriors = ground truth for the latent | [bilby](https://git.ligo.org/lscsoft/bilby) · [PyCBC](https://github.com/gwastro/pycbc) · [LALSuite](https://git.ligo.org/lscsoft/lalsuite) | **★★★ true-decoder arm** — strongest ground truth (Bilby posteriors), published Σ̂, labelled glitches; ★★ only if the paper must stay in particle physics |
| **NuRadioMC / NuRadioReco** (TESTBEDS §3.1 #1) | in-ice radio antennas, ~10–25 ch per station, GHz | shower energy / vertex & direction from channel timing / EM vs hadronic shower (LPM elongation) / flavour | true: antenna response ⊗ Askaryan pulse with per-channel delays | JSON station/channel description | none by default; `channelGalacticNoiseAdder` is a native *coherent* foil | GPL-3 ✅, pure Python, active ✅ | [NuRadioMC](https://github.com/nu-radio/NuRadioMC) | **★★★ true-decoder arm** (particle-astro option) — best functional match on every axis, pure Python, native coherent-noise foil |
| **SolidStateDetectors.jl + LegendGeSim.jl** (§3.1 #2) | segmented HPGe, ≤ tens of contacts, 1 ns | energy / interaction position / **drift-path-dependent pulse shape** / single- vs multi-site, surface α | **known false**: shape and position are entangled through the drift path; the template decoder must fail in a predictable direction | contact segmentation (YAML/JSON CSG) | none in SSD; LegendGeSim adds preamp + `NoiseFromData` | both MIT ✅; Julia; Majorana release is the real twin | [SolidStateDetectors.jl](https://github.com/JuliaPhysics/SolidStateDetectors.jl) · [LegendGeSim.jl](https://github.com/legend-exp/LegendGeSim.jl) | **★★★ false-decoder arm** — the one control the diagnostics need; MIT; real twin (Majorana) exists |
| **Garfield++** (Heed + drift/diffusion + Ramo signals) | gas detectors — TPC, GEM, wire chambers; induced current per electrode | dE/dx / drift time & diffusion width → position / cluster train → shape / **e, μ, α, ions via ionisation statistics** | mostly true (weighting field × delayed cluster train) with a *physical* stochastic residual (cluster fluctuations) | electrode layout: analytic for wire chambers, field maps (ANSYS/Elmer/neBEM) otherwise | white `Sensor::AddNoise` only, switchable | GPL-3 per docs (verify from licence file); C++/ROOT; active (3 756 commits) | [Garfield++](https://gitlab.cern.ch/garfield/garfieldpp) | ★★ reserve — best particle-type axis (dE/dx) and a fourth readout physics, but C++/ROOT and the false/true status is mixed, so it settles nothing the pair above does not |
| **G4CMP → `qp_simulator` → `noise_module` (Al₂O₃ athermal budget)** | athermal phonons in Si / Ge / sapphire → TES | energy / position from phonon share & arrival / phonon-mode timing / ER vs NR, surface events | true — the HeST construction: arrival times → single-quantum template | sensor patterning; geometry in C++ (the catch) | none; `noise_module/al2o3_athermal.py` already is its noise model | GPL-3 ✅; **Geant4 10.4–10.7 only** ✅; the crystal counterpart of arm B | [G4CMP](https://github.com/kelseymh/G4CMP) | ★★ conditional — ★★★ if DELight relevance ranks first (crystal twin of arm B, Al₂O₃ budget ready); otherwise the Geant4-10 pin and C++ geometry cost more than it adds |
| **Allpix²** (§3.1 #3) | pixel / strip, thousands of channels | charge / cluster centroid / cluster shape / MIP vs ion | true at pixel level | `.conf` matrix, pitch, thickness | white per bin, switchable | MIT ✅, active ✅; pulses too short for κ on native records | [Allpix²](https://gitlab.cern.ch/allpix-squared/allpix-squared) | ★ not for this — true-decoder at pixel level but shape/type axis is weak and records too short for κ; keep for scale/architecture transfer only |

_Recommendation scale: ★★★ build it for the physical-latent paper · ★★ hold in reserve / conditional · ★ not for this purpose. The pair to build is one ★★★ true-decoder arm plus the ★★★ false-decoder arm (§3)._

Repositories:

- bilby — https://git.ligo.org/lscsoft/bilby (mirror https://github.com/bilby-dev/bilby); PyCBC — https://github.com/gwastro/pycbc; LALSuite — https://git.ligo.org/lscsoft/lalsuite; Gravity Spy labels — https://doi.org/10.5281/zenodo.5649212; GWOSC — https://gwosc.org
- NuRadioMC — https://github.com/nu-radio/NuRadioMC
- SolidStateDetectors.jl — https://github.com/JuliaPhysics/SolidStateDetectors.jl; LegendGeSim.jl — https://github.com/legend-exp/LegendGeSim.jl; Majorana AI/ML release — https://doi.org/10.5281/zenodo.8257027
- Garfield++ — https://gitlab.cern.ch/garfield/garfieldpp (docs https://garfieldpp.docs.cern.ch)
- G4CMP — https://github.com/kelseymh/G4CMP (paper: NIM A, 2023, doi 10.1016/j.nima.2023.168473)
- Allpix² — https://gitlab.cern.ch/allpix-squared/allpix-squared
- For reference, the current arms: LUCiD — https://github.com/CIDeR-ML/LUCiD (no licence, gate A0); HeST — https://github.com/spice-herald/HeST (gate B0); Prometheus — https://github.com/Harvard-Neutrino/prometheus; TIDMAD — https://doi.org/10.5281/zenodo.11458076

## 3. What follows

**A true-decoder arm and a false-decoder arm, not a longer list.** The physical-latent diagnostics are
validated by a pair: one arm where `a_c·s(t−τ_c)` is exactly the physics, so a physics change must show as a
block displacement with the residual at reference; one where it is not, so the residual branch and the abstain
row must fire. The GW network is the cleanest true case anyone has — a_c is literally an antenna pattern, τ_c a
light-travel time, and Bilby posteriors give an independent ground truth for `z_pos` and `z_shape`. NuRadioMC is
the true case if the paper should stay inside particle astrophysics. Segmented HPGe is the cleanest false case
and comes with a real labelled dataset already in TESTBEDS §4.1 (Majorana).

**G4CMP deserves a second look.** TESTBEDS §3.2 rejected it as "hits only, geometry in C++". Hits-only is
exactly what HeST gives and `qp_simulator` already consumes; the real cost is the C++ geometry and the Geant4-10
pin. The payoff is that it is the crystal twin of the helium arm, and `noise_module` already carries the sapphire
athermal budget — a DELight reader would see their own material class. If DELight relevance ranks above
methodological purity, this is the arm to add first.

**None of this displaces the three arms.** The independence argument stands: arm B is the second readout
physics, TIDMAD the noise nobody chose. Everything above is a fourth arm for the physical-latent paper, chosen by
what the diagnostics must be shown to do. Two PMT simulators would agree with each other by construction and add
no independence; that is why neither HeST nor TIDMAD should be traded for one.

## 4. Caveats to carry into any plan

- The GW arm is off-domain for a particle-detector readership and costs 2–3 weeks of domain entry; its
  non-stationary real noise is a feature for Σ̂ ≠ Σ claims and a burden for everything else.
- Garfield++ and G4CMP are C++ toolchains (ROOT; Geant4 ≤ 10.7). Budget the build before the physics.
- GPL (NuRadioMC, Garfield++, G4CMP, PyCBC) matters only if an adapter is *distributed* as a derivative;
  produced datasets are not covered. Same posture as the Prometheus arm.
- "verify" entries above were not read from a licence file on 11 Sep; do that before a plan depends on them.

## 5. Proposed next step

Add a §3.4 to `docs/TESTBEDS.md` with this table after the same verification pass the other rows received
(licence file, last commit, output format read upstream), and pick the pair — recommended: **NuRadioMC (true)
+ SolidStateDetectors.jl (false)** for a particle-physics paper, **bilby (true) + SolidStateDetectors.jl
(false)** if the claim-bearing real-data check with a published Σ̂ is wanted, **G4CMP** if the next arm must
speak to DELight.
