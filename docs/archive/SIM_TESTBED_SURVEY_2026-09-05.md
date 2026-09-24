# Two more simulation arms? LUCiD, ldmx-sw, and what the evidence says

_5 September 2026. Answers the four questions put on 2026-09-05: can LUCiD and
ldmx-sw vary geometry at fixed physics and physics at fixed geometry; can
`src/noise_module/` be reused in both; is the layout only a config file, and can
the two be given the same layout; and are there better candidates. Evidence is
from direct reads of both trees at HEAD (LUCiD `bdc6195`, 2026-08-21; ldmx-sw
`42b4a55`, 2026-09-01) plus a survey of fifteen other packages. Everything
marked **unverified** was read, not run._

---

## 0. The three findings that change decisions

**(a) `src/noise_module/`'s core construction is no longer unpublished.** Two
packages now do per-frequency Cholesky colouring of multichannel noise in the
open:

| | what | licence | first public |
|---|---|---|---|
| `spice-herald/pytessim` | `core/noise/Noise_Factory.py` — interpolate CSD → `np.linalg.cholesky` per frequency bin → `einsum('ijn,jn->in', L, z)` → ifft. Commit message says "Noise (sampled from either a PSD or CSD)". TESSERACT authors. | MIT | 2026-05-18 |
| `WireCell/wire-cell-toolkit` | `gen/src/CorrelatedAddNoise.cxx` — per-band colorer matrices `A_band[b]` with `A·Aᵀ ≈ correlation matrix`, loaded from a **user-supplied JSON file**, `U = A_b Z`. Also `CoherentAddNoise` + `GroupNoiseModel` = shared-private block structure. | LGPL | 2026-02-04 |

Neither returns the **realized** covariance, and neither is an ML benchmark, so
the §6.3 claim survives — but "no public package provides a known assumed *and*
realized covariance" must be rewritten as "no public package *reports the
realized* covariance", and both must be cited. A referee from the LArTPC world
finds `CorrelatedAddNoise` in ten minutes. Cite it first.

**(b) ldmx-sw is the wrong detector class for the headline claim.** Its noise is
`Gaus(0, NOISE·gain)` drawn independently *per time sample, per channel*
(`Tools/include/Tools/HgcrocEmulator.h:188-191`, called inside the `i_adc` loop
at `Tools/src/Tools/HgcrocEmulator.cxx:199` and `:252`). Ten samples at 25 ns is
a 250 ns record → a **six-bin** one-sided spectrum with one degree of freedom
each. κ(Σ̂⁻¹Σ) is not estimable at that record length, and there is no
cross-channel term anywhere in the tree (`grep -i 'correlat|covarianc|crosstalk'`
returns only ACTS track-fit covariances). §4 below gives the detail.

**(c) There is a dark-matter arm that snaps onto code you already own.**
`spice-herald/HeST` (MIT) is a superfluid-helium detector simulation that emits
per-sensor quasiparticle **arrival times** and has *no noise model at all*.
`src/qp_simulator/QPSimulator.py` already consumes exactly that
(`sim.generate(arrival_times_ns)`), and its own docstring already says
`trace + my_noise_module.generate(...)`. HeST supplies the one thing
`qp_simulator` lacks — a *geometry* to vary. See §5.

---

## 1. Can the geometry change while the physical event is held fixed?

### LUCiD — yes, and the pairing is exact, not approximate

Geometry is a standalone JSON file and nothing else. The whole of
`config/SK_like_geom_config.json`:

```json
{"material": "water", "detector_type": "cylinder",
 "geometry_definitions": {"radius": 16.9, "height": 36.2,
                          "n_sensors": 11000, "sensor_radius": 0.25}}
```

`detector_type ∈ {cylinder, sphere, box, string}`, registered by decorator
(`lucid/geometry/registry.py`), built by `generate_detector(config_path)`
(`lucid/geometry/detector.py:101`). Real PMT layouts load from `.npz` via
`Cylinder.from_pmt_file`. Sixteen detector configs ship: SK, SK_like,
SK_like_wbls, HK, BigHK, WCTE, WCTE_like, IWCD, JUNO, JUNO_wbls, TAO,
IceCube86_full/simple, MidBox, EOS, nuSCOPE.

The pairing is **in-tree and demonstrated**. `analysis/paper/fig_charge_displays.py`
reads one PhotonSim muon event once (line 179), then loops three geometry
*families* over the same photon dict and the same PRNG key:

```python
DETECTORS = [('Cylinder','SK_like_geom_config.json', ...),
             ('Sphere',  'JUNO_geom_config.json',    ...),
             ('Box',     'MidBox_geom_config.json',  ...)]
for label, geom, phys in sel:
    data_sim = setup_event_simulator(geom_p, n_photons, is_data=True, **common)
    cd, td = data_sim(trk, key, pdi)      # same trk, same key, same pdi
```

`analysis/paper/utils/make_geometries.py` does the same for an `n_sensors` scan
(2000…20000, step 1000) against one ROOT file. Because the event is a photon
list handed in as an argument, this is *exactly* paired — stronger than
Prometheus's injection-file replay, which re-runs photon propagation.

**One trap.** In the production path `lucid_options.apply_translation` defaults
`true` and draws the vertex from *the detector's own* fiducial volume
(`lucid/sources/writer.py:323-370`). Two differently-sized geometries get
different vertices at the same seed. Set `apply_translation: false` or supply
the translation explicitly. Same-size scans are unaffected. Also `n_sensors` is
a *target*: SK_like asks 11000 and places 10764; MidBox asks 9000, places 8932.

### ldmx-sw — yes, by two mechanisms, with one caveat that bites

**(i) `ReSimulator` seed replay.** `SimCore/src/SimCore/Simulator.cxx:103` saves
the full CLHEP/G4 RNG state per event into the header as `"eventSeed"`;
`SimCore/src/SimCore/ReSimulator.cxx:41-45` restores it and re-processes. The
Python docstring (`SimCore/python/simulator.py:223-228`) says the intended use
outright: *"If you require any changes to the simulation configuration, such as
loading a modified geometry, you can make those changes after creating the
resimulator."*

Pairing depth is sharper than "primaries only": the G4 RNG is one sequential
global stream, so everything reproduces bit-for-bit **up to the first step that
sees different material**. Change only the ECal/HCal and the beam transport
*and the target dark-brem/photonuclear vertex* also reproduce exactly. Change
the target or trackers and you are paired at the primary vertex only.

**Caveat, and it is a real one.** The dark-brem MadGraph library index is *not*
part of the restored RNG state. `G4DarkBreMModel.cxx:653-664` picks a starting
offset with `G4UniformRand()` at initialisation, and `sample()` then walks the
library with a persistent job-global counter (`currentDataPoints_[Z][E]++`,
`:698`). **Re-simulating a subset (`which_events=[...]`) desyncs the library and
gives a different A′ four-momentum.** Re-simulate whole files in order.
*Unverified — inferred from source, not run.*

**(ii) Scoring-plane staging — better, and I would use this instead.**
`FromScoringPlane` (`SimCore/python/generators.py:226-286`) reads scoring-plane
hits from a previous file and turns them into primaries. Worked example ships:
`SimCore/exampleConfigs/stage-one-sim-no-cal.py` (geometry
`ldmx-det-v15-8gev-no-cals`, writes plane hits) →
`stage-two-sim-cal.py` (`sim.generators = [FromScoringPlane.hcal()]`, full
calorimeter). The entire upstream event is frozen *as data*, so stage two
against N calorimeter geometries is exactly paired at the calorimeter face with
no RNG fragility. This is the same trick as Prometheus's injection-file replay,
one layer further downstream.

Seeds otherwise: default mode is `run`, master seed = run number
(`Framework/src/Framework/RandomNumberSeedService.cxx:35-53`), per-consumer seed
= `hash(name) + master`. Same run number + same config ⇒ bit-identical job.
⚠️ The `external` (fixed-master-seed) path looks broken on trunk:
`Framework/python/_rnss.py:24-25` declares `seed_node`/`seed` while
`.run()/.external()/.time()` assign `self.seed_mode`, and `_parameter_set.py:91`
raises `KeyError` on unknown attributes; separately C++ reads `master_seed`
while Python only ever sets `seed`. Set `p.run` instead. *Unverified — not run.*

---

## 2. Can the physics event type change at fixed geometry?

**LUCiD — yes, on three axes, one of which you get free.**

- *Particle gun (differentiable):* `ParticleParams(energy, position, theta, phi, t0)`
  is a JAX pytree, geometry-independent. Energy/vertex/direction vary freely and
  differentiably. **But particle type is a setup-time string** selecting a SIREN
  emitter, and only `muon` and `electron` emitters ship
  (`data/{water,ice,wbls}/{muon,electron}/`). No pi0, no CC/NC in this path.
- *External truth (`is_data=True`):* hand it any photon list. Production configs
  drive PhotonSim/GEANT4 macros: `GeV/01_mu`, `02_pi_plus`, `03_e`, `04_pi_minus`,
  `05_pi0`, `06_pbomb` (multi-particle), `Solar/01_e_low_energy`, plus a supernova
  path (sntools/SNEWPY) and a GENIE path (`lucid/production/run_genie.py` runs
  `gevgen` + `gntpc -f rootracker`; water target `1000080160[0.888],1000010010[0.112]`).
  ⚠️ The docs reference `GeV/13_genie_numu_nue.json`; **that file is not in the
  tree** at this commit. GENIE support is real in code but has no shipped example.
- *Free third axis:* **material at fixed geometry** — `SK_like` vs `SK_like_wbls`,
  `JUNO` vs `JUNO_wbls`, and water/ice/WbLS physics configs. That is a
  supported-but-rare-physics (S) family generator you were not counting.

**ldmx-sw — yes, and it is one line, because the CI is built that way.** Every
`.github/validation_samples/*/config.py` pins `det = "ldmx-det-v15-8gev"` and
differs only in scenario:

```python
# signal/config.py
my_sim = target.dark_brem(ap_mass=10.0, lhe=".../mA_0.01_run_1.csv", detector=det, ...)
# ecal_pn/config.py
my_sim = ecal.photo_nuclear(det, gen.single_8gev_e_upstream_tagger())
```

Scenario factories: `Biasing/python/target.py` (`electro_nuclear`, `photo_nuclear`,
`gamma_mumu`, `dark_brem`, `aprime_to_fcp`, `gamma_to_fcp`), `Biasing/python/ecal.py`
(`photo_nuclear`, `nonfiducial_photo_nuclear`, `gamma_mumu`, `deep_photo_nuclear`),
`Biasing/python/eat.py`. Seven primary generators (`Gun`, `Multi`, `Lhe`, `HepMC`,
`Gps`, `Genie`, `FromScoringPlane`), eleven filters, five photonuclear final-state
models. A′ mass is a scalar argument. **Fifteen CI-maintained worked configs.**
This is the best physics-event-type axis of anything surveyed, LUCiD included.

---

## 3. Is the layout just a config file, and can the two share a layout?

### "Only the config file?"

**LUCiD: yes, genuinely.** One JSON, four keys, meters, no code. Caveats are the
two above (`n_sensors` is a target; `apply_translation`) plus
`Cylinder.configure_grid(n_cap, n_angular, n_height, max_candidates_per_ray=4)`,
which is a correctness knob — too coarse a grid **silently drops sensors**.

**ldmx-sw: no. Three coupled places, and the GDML is the easy one.**

1. `Detectors/data/ldmx-det-v14/constants.gdml` is genuinely parametric —
   `num_bilayers=17` drives `<loop>`s in `ecal.gdml:369,388,548,820`;
   `back_hcal_numLayers=96`, `hcal_scintThick=20`, `tagger_layer_delta`,
   `target_thickness` are all constants. So far so good.
2. **`bilayer_absorber_cumulative` (`constants.gdml:237-256`) is hand-precomputed**,
   with the comment *"one cannot use loops when defining GDML variables, so this
   needs to be precomputed."* Change an absorber thickness and you regenerate it
   with `Detectors/util/ecal_layer_stack.py`.
3. **The readout geometry is a separate Python object that must be kept in sync.**
   `DetDescr/python/ecal_geometry.py` hard-codes `layer_z_positions`,
   `ecal_front_z`, `si_thickness`, `n_cell_r_height` per version, selected by a
   POSIX regex against the run header's detector name in
   `Ecal/src/Ecal/EcalGeometryProvider.cxx:91-133`, which **throws** if nothing
   matches. A new geometry needs a new `EcalGeometry.vXX()` entry *and* a
   `detectors_valid` regex. Same pattern in `Hcal/python/hcal_geometry.py`.
   Reconstruction `layer_weights` and `second_order_energy_correction`
   (`Ecal/python/digi.py:125-441`) are per-geometry constants too.

Live released geometries: `ldmx-det-v14`, `-v14-8gev`, `-v14-8gev-no-cals`,
`-v15-8gev`, `-v15-8gev-no-cals`, `ldmx-al-v15-8gev`, `ldmx-ti-v15-8gev`,
`ldmx-lyso-r4-v15-8gev`, `ldmx-vertTS-v14-8gev`, `ldmx-reduced-v3`,
`ldmx-hcal-prototype-v2.0`, plus six archived tarballs. **Recommendation: vary
across the shipped set, not by editing GDML.** Target material (W/Ti/Al/LYSO) and
beam energy (4/8 GeV) are already free axes; `-no-cals` vs full is a third.

### "Can the two experiments have the same layout?"

**No, and it is not worth trying.** A 34 m water cylinder with ~11k 20-inch PMTs
sampled at 1 ns and a 40-layer Si-W sampling calorimeter with hexagonal 8 mm pads
sampled at 25 ns are not the same object under any reparameterisation. Both
frameworks nominally offer a "box" (`MidBox` in LUCiD, `ldmx-reduced-v3`), but
matching a bounding box while channel count, channel meaning, sampling rate and
noise physics all differ buys nothing a referee will credit.

**What should be identical is the intervention grammar, not the geometry** — and
you already have that abstraction in `prometheus_simulation`. Make every arm
emit the same schema and the cross-testbed claim is about the *protocol*, which
is what the paper actually argues:

| shared object | already exists as |
|---|---|
| paired-event key | `export.py`'s `injection_id` |
| N1–N5 acquisition interventions | `interventions.py` |
| S / U families | `strata.py` |
| content-matched clean twins | `matching.py` |
| per-arm provenance record | `physics.py` |

Then define the abstract geometry axis once — *"detector granularity at fixed
active volume"* — and instantiate it per arm: `n_sensors` 2000→20000 in LUCiD,
`num_bilayers`/target material across ldmx's released set, geofile across
Prometheus's eleven. Same axis, three realisations, one table in the paper. That
is a defensible cross-domain claim; "we made them the same shape" is not.

---

## 4. Can `src/noise_module/` be reused? Honest answer per arm

### LUCiD — yes, and this is the strongest reason to want it

`lucid/simulation/sensor_response.py:389`
`build_make_hits_waveform(n_photons, window_ns=500.0, bin_width_ns=1.0,
tts_sigma_ns=1.0, smear_time=True, smear_charge=True)` returns a dense
`(num_detectors, n_time_bins)` array — TTS and gain smearing applied per photon,
then `segment_sum` into 1 ns bins, described in-code as the "1 GHz FADC
convention". Reached by `hit_mode='waveform'` or `'waveform_expected'`. **This is
the only per-channel uniformly-sampled time series in the water-Cherenkov world**
— 500 samples ⇒ a 250-bin one-sided spectrum, which is enough to estimate a PSD
and to make κ(Σ̂⁻¹Σ) measurable.

Existing noise is per-channel independent by construction: Poisson dark noise
(`digitizer.py:546-572`), independent SPE draws (Bellamy-94 mixture,
`:283-307`), independent Gaussian TTS, independent TDC jitter. There is **no
covariance concept anywhere** in the package (the only `covariance` is over fit
parameters, `lucid/fitting/fisher.py`, and its weight matrix is literally
`diag`). So your module does not compete with anything — it fills a hole.

**Adaptation required — three pieces, all small:**

1. **Units bridge.** LUCiD's waveform bin holds *summed photoelectron charge*,
   not volts. Add a per-channel SPE pulse template convolution to get mV, then
   add noise in mV. `templates.py` already does templates; this is ~40 lines.
   Without it, "PSD" has no physical meaning on that array.
2. **Swap the spectral model, keep the covariance machinery.** Johnson + SQUID
   1/f + thermal pole + paramagnetic spin is athermal-calorimeter physics and
   does not describe a PMT front end. The PMT-appropriate ASD is: white
   electronics floor + 1/f baseline drift + **mains pickup lines at 50 Hz and
   harmonics** + digitizer coherent pickup. Note that `artifact_injector.py`'s
   spectral-line injection *is already exactly the mains-pickup case* — the
   physics story writes itself. Add a `spectral_models.pmt_frontend()` alongside
   `al2o3_athermal`; `reference_budget.py` gives you the pattern for a
   closed-form, independently citable budget.
3. **Map the covariance structure onto real hardware.** In a water-Cherenkov
   detector, correlated noise is not a mathematical convenience — it is HV supply
   ripple, ground loops and shared front-end crates. That is *precisely*
   `multichannel_noise.py`'s **shared-private** mode with the sharing group set
   to the crate/string, and its **low-rank-latent** mode for a small number of
   global pickup sources. `string_id` is already in LUCiD's output, so the
   grouping key is free. This is the single most defensible sentence you can
   write about why the module belongs here.

What still does not apply: `psd_resampling.py`'s alias-fold path is only relevant
if you resample away from 1 GHz, and `temporal_noise.py`'s drift/piecewise
stationarity needs records longer than 500 ns — extend `window_ns` (it is an
argument) to get them.

### ldmx-sw — essentially no, and I would not force it

Against a 10 × 25 ns record:

| module | applicable? | why |
|---|---|---|
| `NoiseGenerator.py` (PSD synthesis) | **no** | 6 DFT bins, 1 dof each. Noise is white by construction — there is no spectrum to specify. |
| `multichannel_noise.py` | technically yes | You *can* draw a correlated `(N_chan, 10)` block. But κ(Σ̂⁻¹Σ) is not *estimable* from 10 samples, so the headline result cannot be measured back out. |
| `temporal_noise.py` | **no** | drift and piecewise stationarity over 250 ns is meaningless. |
| `artifact_injector.py` | **no** | spectral lines need many cycles in-window. |
| `psd_resampling.py` | **no** | nothing to resample. |
| `validation.py` | **no** | CSD-ensemble validation needs record length. |

Where you *would* inject, if you did it anyway (ranked):

1. **Conditions CSV, no code change.** `Conditions/python/SimpleCSVTableProvider.py`
   + `Ecal/python/ecal_hardcoded_conditions.py` — a per-DetID CSV with your
   `NOISE`/`PEDESTAL` columns gives channel-dependent *amplitude*, still white.
   (The shipped ECal value carries the comment `# 0.6 - ADC, almost certainly too
   optimistic`.)
2. **Make `HgcrocEmulator::noise` virtual.** `Tools/include/Tools/HgcrocEmulator.h:188-191`
   is the single choke point for ECal *and* HCal. Change it to
   `noise(channelID, i_adc)` returning a Cholesky-coloured length-10 vector;
   cover call sites `Tools/src/Tools/HgcrocEmulator.cxx:199` (real hits) and
   `:252,:255` (`noiseDigi`). For *cross-channel* correlation you must go one
   level up to `Ecal/src/Ecal/EcalDigiProducer.cxx::produce` (`:73-210`) and
   pre-generate the whole event's noise field, because `digitize()` is per-channel.
3. **Post-hoc on the analog pulse — the only route that gives a real PSD.**
   `Recon/include/Recon/Event/CompositePulse.h:86-91` exposes
   `double at(double time) const` (documented: *"does not put any noise into the
   measurement"*) and `HgcrocPulseTruth` persists the whole pulse when
   `save_pulse_truth_info=True`. You could resample at, say, 40 MS/s over 10 µs
   and build your own readout. **But then you are simulating a hypothetical
   LDMX-with-a-different-DAQ, and a referee will ask why you did not simulate the
   detector you named.** (Also: pulse-truth saving is HCal-only today —
   `Hcal/python/digi.py:184`; wiring it into the ECal producer is ~5 lines since
   `save_pulse_truth_info_` is public.)

**Two apparent defects found while reading, worth reporting upstream either way:**

- *Double-counted threshold offset for pure-noise calorimeter digis.*
  `Tools/src/Tools/NoiseGenerator.cxx:30-64` returns an amplitude already measured
  above pedestal (tail probability computed on `N(pedestal, σ)`, quantile taken on
  `N(0, σ)`), and `Ecal/src/Ecal/EcalDigiProducer.cxx:182-184` then adds
  `gain·(readoutThreshold − pedestal)` again. With the shipped constants
  (gain 0.015625 mV/ADC, ped 50, thr 53, σ 0.6): minimum output is exactly
  3.000 ADC, and after the extra addition the digi sits at 6.0 ADC above
  pedestal. Pure-noise ECal digis are biased ~3 ADC high and never populate the
  3–6 ADC band. Same pattern at `Hcal/src/Hcal/HcalDigiProducer.cxx:455-458`.
  *High confidence, arithmetic reproduced independently; not run against ROOT.*
- *Tracker noise affects the threshold but not the recorded waveform.*
  `Tracking/src/Tracking/Digitization/SiStripDigitizer.cxx:256-266` adds noise and
  thresholds on it, but `Tracking/src/Tracking/Reco/DigitizationProcessor.cxx:552-566`
  builds the 3 ADC samples for signal strips from the **noise-free** charges.
- Also: `timing_jitter = 0.25 ns` is configured but not implemented
  (`HgcrocEmulator.cxx:80-81`, `// TODO step 2: add timing jitter`), and TOT mode
  has `// @TODO NO NOISE` at `:134`.

---

## 5. Better candidates — and the dark-matter arm you should actually build

### Dark matter: **HeST → `qp_simulator` → `noise_module`**

`github.com/spice-herald/HeST` · MIT · 89 commits · last commit 2026-03-08 ·
pure Python + numba · `pip install .`

| criterion | verdict |
|---|---|
| **paired geometry** | ✅ `HeST/core/Geometry.py` ships `HeRALD_v1` (28 sensors, 6×6 mask, 1 cm CPDs at 1.1 cm pitch), `HeRALD_v1_monolithic` (**same cell, one sensor**), `HeRALD_LBNL`, `HeRALD_UMass_splitCPD`, `HeRALD_UMass_monolithic`. Entry point is `GetEvaporationSignal(detector, QPs, X, Y, Z, ...)` (`core/Detection.py:2078`) — the detector is an *argument*. Same quasiparticle population, two `VDetector`s, two calls. Exact pairing, no seed replay, no file format in between. |
| **physics type at fixed geometry** | ✅ `GetEnergyChannelFractions(energy, interaction)` with `interaction ∈ {"ER","NR"}` (`core/HeST_Core.py:257`); `core/WIMP_Generation.py` gives `WIMP_spectrum(mass_MeV, ...)`. Yields split four ways (singlet UV / triplet / IR / quasiparticle) — four physically distinct channels, i.e. free S-families. |
| **noise** | ✅ **none exists** — output is per-sensor `energies` + `arrivalTimes_us`. Its one waveform routine (`analysis/analysis_functions.py:69`) is a stub with a hardcoded absolute path and 2 channels. |

Why this is the right arm and not merely an available one:

1. **It stops exactly where your code starts.** HeST → per-sensor arrival times →
   `QPSimulator.generate(arrival_times_ns)` (already written, already documented
   with `trace + my_noise_module.generate(len(trace))`) → `multichannel_noise` →
   trace. **You own two of the three links.** Σ̂ and Σ are yours by construction
   because you wrote every line between the quasiparticle list and the sampled
   trace — which is precisely the condition §6.3 requires and which no
   off-the-shelf digitizer can give you.
2. **Superfluid helium is DELight's own target material.** This converts the arm
   from "a benchmark" into a DELight-relevant result, which matters for the
   authorship position in `claude/noise-module-ip-and-nubench-reproduction.md`.
3. `HeRALD_v1` vs `HeRALD_v1_monolithic` is a *28 → 1 channel* change on the
   identical cell. That is the cleanest possible instantiation of the granularity
   axis, and it makes the multichannel covariance claim non-trivial in one arm and
   vacuous in the other — a designed contrast, not an accident.

Units: HeST is µs, `QPSimulator` is ns at 2.5e5 Hz × 16384 samples = 65.5 ms;
HeST's window is 0–5000 µs. Compatible, one conversion.

Risks: 0 stars, effectively one maintainer (Greg Rischbieter), README says "early
developmental version", `.DS_Store` committed. LCE/QPE maps are precomputed and
shipped for the five designs; a *new* geometry needs a NERSC-scale map job, so
stay on the shipped five. Estimate to first paired dataset: **2–4 days**, mostly
rewriting `generate_waveform` for N sensors.

### The rest of the dark-matter field

| package | licence | waveform | Σ injectable | geometry axis | verdict |
|---|---|---|---|---|---|
| `spice-herald/pytessim` | MIT | yes (TES datastreams) | ✅ via `detprocess.FilterData.set_csd`, arbitrary (N,N,N_f) CSD | readout topology YAML, not shape | **Read it before writing the novelty section.** 4 commits, 2-channel hardcoded (`np.zeros((2,2,nsamples))`), ships no filter files, absolute paths in examples. Good second arm / validation target, bad primary. |
| `XENONnT/fuse` | BSD-3 | ✅ real `raw_records` | ✅ trivially — `add_noise()` indexes a per-channel stream by absolute sample position; swap the `.npz` | ❌ XENONnT's fixed 494-PMT TPC | Best waveform fidelity in the survey; **fails the geometry axis outright.** |
| `fewagner/cait` (CRESST/NUCLEUS/COSINUS) | GPL-3 | injects onto **recorded** baselines | ❌ Σ is unknown by construction | ❌ | Wrong direction for §6.3. |
| `G4CMP/G4CMP` | GPL-3 | ❌ no readout at all | — | Geant4 | Real phonon/charge transport, but six weeks of work to reach where HeST is on day one. |
| SuperCDMS `SuperSim` | — | — | — | — | **Not public.** Only a config repo (`kelseymh/SLACB33_G115`) references it. |
| LZ `BACCARAT` | — | — | — | — | No public repo found. |
| `NESTCollaboration/nest`, `flamedisx` | — | ❌ | — | ❌ | Yields / likelihood only. |
| DELight `SignalHelium` (Zenodo 10.5281/zenodo.13735332) | CC-BY-4.0 | ❌ | — | ❌ | Energy partitioning, one notebook. Cite; not a testbed. |
| `ldmx-sw` | GPL-3 | 10 × 25 ns | white by construction | released set | §4. Keep only as a scale/architecture-transfer arm where noise is not the subject — the role D3 assigned to Panda/SPINE. |

### Water Cherenkov: LUCiD if the licence lands, otherwise you already have Prometheus

**⚠️ LUCiD has no licence.** Verified: no `LICENSE*` in the tree; `pyproject.toml`
has classifiers but no `license` field; `README.md` says *"The license is being
finalized and will be added shortly"* and `CONTRIBUTING.md:67-68` says
contributions will be licensed "once it is finalized". **All rights reserved
today.** You cannot redistribute derived code or release a benchmark built on it.
Given the release posture in `EXPERIMENT_DESIGN.md` (ORACLE-Paired to Hugging
Face, licence-clean), this is a hard blocker, not a footnote.

**Action: email Terao / Alterkait this week asking them to add MIT or BSD-3.**
They have a `CITATION.cff` and a `CONTRIBUTING.md`, so the omission is almost
certainly an oversight — but budget 1–3 weeks and have Prometheus as the fallback.

| package | licence | waveform | geometry | verdict |
|---|---|---|---|---|
| **LUCiD** | **none** | ✅ `(n_sensors, n_bins)` @1 ns | ✅ 16 configs, JSON | Only WC package with a sampled trace. JAX autodiff is a bonus for §6.2's output-null construction. Blocked on licence. |
| **Prometheus** (in hand) | LGPL-2.1 | ❌ photon arrival times | ✅ **11** geofiles, not 6 — `arca, orca, gvd, icecube, icecube_gen2, deepcore, upgrade, trident, pone_triangle, demo_ice, demo_water` | Already built, already debugged, licence-clean, and you own §3.2 so you own the whole noise path. Weakness: telescopes, not water-Cherenkov detectors — it does not deliver SK/HK. |
| `WCSim/WCSim` | MIT | ❌ | ✅ **~35** configs (`SuperK`, `HyperK*`, `nuPRISM*`, `IWCD_mPMT`, generic cylinders at 14%/40% coverage) — the best geometry axis in existence | A digit is `map<int,double> pe/time`. One digitizer (`SKI`). Dark noise = extra Poisson PEs. Open issue #13 *"Simulate SK-IV Electronics"* confirms the electronics layer was never written. **You would build the readout stage from nothing and have nowhere in the data model to put it.** |
| `WireCell/wire-cell-toolkit` | LGPL | ✅ best-in-class correlated noise | ❌ channel sets are incommensurable across detectors | **Do not build an arm on it. Do cite it** — §0(a). Also a cheap cross-validation target for `multichannel_noise.py`, since it implements the same mathematics. |
| Chroma / olympus / ppc / icetray / JUNO offline / KM3NeT | — | ❌ / not public | — | No readout layer, or collaboration-internal. |

**Note the +5 geometries.** `docs/archive/OPEN_DECISIONS.md` D1 says six geofiles;
there are eleven. `demo_ice` and `demo_water` are ideal for a cheap A/B pilot.

---

## 6. Recommendation

**Do not add two arms. `EXPERIMENT_DESIGN.md` says "Three is the number" and
names ten testbeds as the project's largest schedule risk. That judgement was
right and nothing here overturns it.** What follows swaps one arm and sharpens
another.

| tier | now | proposed | why |
|---|---|---|---|
| 1 — mechanism | ORACLE-Cov (`noise_module`, synthetic) | **unchanged** | Still the only place Σ̂ and Σ are both known and the whitening lemma is provable. |
| 2 — paired realism | Prometheus / ORACLE-Paired | **unchanged; ship it** | Built, debugged, licence-clean, frozen public model (DynEdge), physics consequence variable. Do not restart this for LUCiD. |
| 3 — real data | TIDMAD | **unchanged** | One paragraph; blocks "it only works in your simulator". |
| **2b — dark matter** | *(none; LDMX proposed)* | **HeST → `qp_simulator` → `noise_module`** | §5. Two of three links already written, DELight-relevant, exact pairing, no competing noise model. 2–4 days. |

**LUCiD:** send the licence email now; it costs one message. If a permissive
licence lands, it becomes the strongest candidate to *merge* Tiers 1 and 2 — a
testbed with a real detector geometry *and* a sampled trace whose Σ you control,
which no other package in this survey offers. That is a bigger prize than a
fourth arm, and it is worth waiting for. Until then it is unusable.

**ldmx-sw:** do not build the dark-matter arm on it. Its geometry and
physics-event axes are excellent (§1, §2) and its CI configs are the best
documented of anything surveyed — so keep it on the shelf as a *scale and
architecture transfer* arm where noise is not the subject, the same role D3
assigned to Panda/SPINE. Building the covariance-geometry claim on a 250 ns
white-noise record would hand a referee the paper's weakest point for free.

### Immediate actions

1. **Rewrite the §6.3 novelty sentence** and add `pytessim` + Wire-Cell
   `CorrelatedAddNoise` to `reference/papers/papers.tsv`. Highest value, lowest
   cost, and it is a claim currently in the proposal that is no longer true as
   written.
2. **Email the LUCiD authors about the licence.** One paragraph.
3. **Pilot HeST** — `pip install`, run `HeRALD_v1` and `HeRALD_v1_monolithic` on
   one NR event, confirm the arrival-time arrays feed `QPSimulator` unchanged.
   Half a day, and it settles the whole dark-matter question.
4. **Correct D1's geofile count** to eleven in `docs/archive/OPEN_DECISIONS.md`.
5. Report the two ldmx-sw digitizer defects (§4) upstream regardless of the
   decision — they are real and cheap to write up.

---

## Appendix — HeST, run and measured (added 5 September, same day)

Everything below was executed, not read. Clone at `spice-herald/HeST` HEAD,
`pip install qetpy numba` then `sys.path.insert(0, 'HeST')`.

### What HeST actually simulates

Not a photon or shower simulation. It is a **superfluid-⁴He calorimeter response
model in two stages**, and it stops one step before an electronics trace.

**Stage 1 — yields.** `HeST/core/HeST_Core.py:GetQuanta(energy_eV, interaction)`
partitions a deposit into four channels. Measured, at 1 keV:

| | quasiparticles | IR photons | singlet UV | triplet molecules |
|---|---:|---:|---:|---:|
| `"NR"` | 900,238 | 7 | 20 | 1 |
| `"ER"` | 419,925 | 116 | 18 | 11 |

That 2.1× quasiparticle ratio at equal deposited energy *is* the ER/NR
discrimination handle, and it is a one-argument switch — the (B) axis, free.
`core/WIMP_Generation.py` adds `WIMP_dRate` / `WIMP_spectrum(mass_MeV, ...)` for a
recoil spectrum rather than a fixed energy.

**Stage 2 — quasiparticle transport and evaporation.**
`core/Detection.py:GetEvaporationSignal(detector, QPs, X, Y, Z, useMap=False, T=2.0)`
launches `QPs` quasiparticles isotropically from `(X, Y, Z)` in cm, propagates them
by ray-tracing against the cell surfaces with reflection / diffuse / Andreev
probabilities per surface, and records the ones that reach the liquid surface with
enough momentum to evaporate a helium atom onto a sensor.

**Return value.** `HestSignal`, and it has exactly two fields — verified with
`vars()`:

```
fields: ['energies', 'arrivalTimes']     # each a list of per-sensor lists
```

That is the entire output. **There is no noise model, no digitizer, no trace, no
electronics.** The one waveform helper (`analysis/analysis_functions.py:69`) is a
stub with a hardcoded absolute path into a former collaborator's home directory and
2 channels assumed. This is the reason to use HeST, not a defect to work around:
per-sensor arrival times are precisely `QPSimulator.generate(arrival_times_ns)`'s
input, and `QPSimulator`'s own docstring already ends
`trace + my_noise_module.generate(len(trace))`.

### Geometry is Python, not a config file — and that is better here

A `VDetector` (`core/Detection.py:122`) is constructed from five boolean
implicit-surface functions — `top`, `bottom`, `wall`, `liquid_surface`,
`liquid_conditions` — plus a list of `VSensor`s (each its own boolean condition) and
the surface-interaction probabilities `QP_wall_reflection_prob`,
`QP_wall_diffuse_prob`, `QP_wall_Andreev_prob` and their sensor equivalents. The
five shipped builders in `core/Geometry.py` are ~60-line functions, each returning
one `VDetector`. `HeRALD_v1` builds its sensor array from a literal:

```python
sqcm_width = 1;  sqcm_pitch = 1.1;  cell_rad = 3.5   # cm
array_map = np.array([[0,0,1,1,0,0],
                      [0,1,1,1,1,0],
                      [1,1,1,1,1,1],
                      [1,1,1,1,1,1],
                      [0,1,1,1,1,0],
                      [0,0,1,1,0,0]])
```

**24 sensors, not 28** — `array_map.sum() == 24`, confirmed by
`detector.get_nsensors()`. So a geometry sweep is a factory function over
`(cell_rad, fill_height, sqcm_pitch, array_map)`, with no file format, no parser and
no GDML. Nothing needs to be added to the package to vary geometry.

Verified inventory:

| builder | `get_nsensors()` |
|---|---:|
| `HeRALD_v1` | 24 |
| `HeRALD_v1_monolithic` | 1 |
| `HeRALD_UMass_splitCPD` | 2 |
| `HeRALD_UMass_monolithic` | 1 |
| `HeRALD_LBNL` | (top + bottom arrays) |

### ⭐ Event pairing is exact, and I measured it

`QP_propagation` (`core/Detection.py:1171`) samples the **entire** initial
population up front and vectorised — `dx, dy, dz = generate_random_direction(nQPs)`
at `:1244`, `momentum = Random_QPmomentum(nQPs, T=T)` at `:1258` — *before* any
geometry is touched. Geometry enters only in the propagation loop afterwards.

Therefore `np.random.seed(S)` with the same `(nQPs, X, Y, Z, T)` gives a
**bit-identical initial quasiparticle population in every detector**. Measured, 20,000
QPs from `(0, 0, 2.0)` cm, seed 42:

```
v1_28        nsensors=24  detected= 131
mono         nsensors= 1  detected= 174
umass_split  nsensors= 2  detected= 281

initial QP momenta identical across all three geometries: True   n = 20000
```

Same event, three geometries, three different observed signals. No seed-replay
machinery, no intermediate file, no scoring plane — which makes this a *stronger*
pairing guarantee than either ldmx-sw's `ReSimulator` or Prometheus's injection
replay, both of which re-run a stochastic propagation.

Repeat-run determinism also confirmed: two identical calls give element-wise equal
`arrivalTimes`.

### Cost, and the maps

`useMap=True` uses precomputed light-collection (LCE) and QP-evaporation (QPE)
maps. **The maps are not shipped** (`get_QPEmap()` returns `0.0` on a fresh detector)
and the `map_generation/` directory the README describes is not in the tree. This
does not block anything: `useMap=False` does the full first-principles propagation and
needs no maps, and `VDetector` exposes `create_LCEmap` / `create_QPEmap` /
`load_LCEmap` / `load_QPEmap` if you later want the speedup.

Measured throughput, `useMap=False`, one CPU core, after numba JIT:

```
  2000 QP -> 0.03 s  (69,553 QP/s)   15 detected   eff 0.750 %
 10000 QP -> 0.19 s  (53,485 QP/s)   84 detected   eff 0.840 %
```

≈55k QP/s ⇒ a 1 keV NR event (900k QPs) is **~17 s single-core**, yielding ~7,000
detected quasiparticles spread over the sensors. Sub-keV events — the actual
dark-matter regime — scale down linearly. Embarrassingly parallel. Hundreds of
paired events per geometry is an afternoon on a laptop, not a cluster job.

### Practical notes and risks

- **Install.** `setup.py` requires `detprocess`, which pulls `annoy` and `aplus`;
  `aplus` is Python-2-era and fails to build wheels on modern Python. `pip install
  qetpy numba` plus a `sys.path` insert is enough — `HeST_Core.py:5` only imports
  `qetpy.utils.{fft, ifft, fftfreq, rfftfreq}`. Worth a two-line note upstream.
- **Licence.** MIT *text*, but the copyright line is the unedited PyPA sample:
  `Copyright (c) 2018 The Python Packaging Authority`. The intent is clearly MIT and
  the org is consistent, but ask Rischbieter to name the real holder before building a
  released dataset on it — the same hygiene the noise module just went through.
- **API sharp edges.** `HestSignal` has no accessors (use `.energies`,
  `.arrivalTimes`); `VDetector.get_QPEmap()` returns `self.LCEmap_positions` — a
  copy-paste bug at `core/Detection.py:243`. `.DS_Store` and a stray `.npy` are
  committed.
- **Bus factor.** 89 commits, effectively one maintainer (Greg Rischbieter,
  rischbie@umich.edu), README says "early developmental version", zero stars. The
  package is small enough (3,800 lines total) that vendoring a pinned copy under
  `src/` is a reasonable hedge; the MIT licence permits it once the holder is named.
- **What HeST is not.** It is not a Geant4 simulation and has no external-background
  model, no cryostat, no material budget. Energy deposits are an input, not an output.
  If the paper needs a realistic background spectrum, that comes from elsewhere.

### The resulting chain

```
GetQuanta(E, "NR"|"ER")          -> n quasiparticles          [HeST, exists]
GetEvaporationSignal(detector,…) -> per-sensor arrival times  [HeST, exists]
QPSimulator.generate(times_ns)   -> clean per-channel trace   [ours, exists]
MultiChannelNoiseGenerator(Σ)    -> Σ̂ / Σ, correlated noise   [ours, exists]
```

Three of the four links already exist; the missing piece is a ~40-line loop that
calls `QPSimulator` once per sensor and stacks the result into `(n_sensors, n_samples)`.
Units: HeST reports µs, `QPSimulator` takes ns at 2.5e5 Hz × 16384 samples = 65.5 ms,
against a HeST window of order 5 ms — one conversion, no resampling.

---

## What was not verified

- Nothing in ldmx-sw was compiled or run. Every behavioural claim is from source;
  the two defect claims are analytic (the first with an independent numeric
  reimplementation of the ROOT/Boost calls, not against ROOT itself).
- `ReSimulator` bit-for-bit reproduction across two geometries — the claim you
  most need — needs a ~30 minute empirical test: 100 events, resim all with an
  altered ECal GDML, compare `SimParticles` with `trackID==1` and the A′
  four-momentum. The dark-brem library-index issue is what could bite.
- LUCiD's SIREN emitter path (needs the ~2 GB `scripts/download_data.sh` pull) and
  anything requiring PhotonSim / GEANT4 / GENIE / ROOT. Geometry construction,
  calibration-source transport and the digitizer/trigger *were* run: six configs
  instantiated, and one isotropic source pushed through `WCTE_like` (2444 sensors)
  and `MidBox` (8932) in ~6 s each on 2 CPU cores.
- Two LUCiD install traps found by running it: `IPython` is an **undeclared**
  dependency (`lucid/siren/training/monitor.py:16`, imported at module scope via
  `simulator.py`), and the soft-overlap lookup table (`lucid/overlap.py:255-365`,
  defaults `n_theta=n_rho=2000`) attempts a **9.5 GB** allocation on first use per
  sensor radius. Budget ≥12 GB RAM or run `temperature=None` (hard-step mode).
- HeST's LCE/QPE map regeneration cost for a new geometry — `map_generation/` is
  referenced in the README but not in the current tree.
- SuperSim, LZ BACCARAT, JUNO offline, KM3NeT: no public repository found. That
  is not proof they are closed, only that no public copy surfaced.
