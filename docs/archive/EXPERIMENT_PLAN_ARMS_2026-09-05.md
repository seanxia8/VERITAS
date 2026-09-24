# Experiment plan — the three arms, and a LUCiD-specific noise module

_5 September 2026. Written after the testbed survey
(`SIM_TESTBED_SURVEY_2026-09-05.md`). This is the arm-level plan: what each arm
is for, how it is driven, what it is allowed to prove, and what has to be built.
It **refines** `EXPERIMENT_DESIGN.md` (which stays canonical for the tier logic
and the release posture) and feeds work packages into `IMPLEMENTATION_PLAN.md`.
Everything marked ✅ was executed on this machine and the output is quoted._

---

## 0. What changed, and what did not

`EXPERIMENT_DESIGN.md` names three testbeds — ORACLE-Cov, Prometheus/ORACLE-Paired,
TIDMAD — and says "Three is the number." That judgement stands. This plan does not
add arms; it **retargets two of them**:

| | was | is | why |
|---|---|---|---|
| mechanism | ORACLE-Cov (synthetic waveforms) | unchanged | still the only place the whitening lemma is provable |
| realism | Prometheus (6 geometries, hit-level) | **LUCiD** (water Cherenkov, waveform-level), *conditional on a licence*; Prometheus stays as the fallback and keeps the frozen-public-model role | Prometheus emits photon arrival times, not a sampled trace, so the covariance claim cannot be tested there at all |
| dark matter | *(none)* | **HeST → `qp_simulator` → `noise_module`** | §5 of the survey; two of three links already written |
| real data | TIDMAD | unchanged | §5 below — adding HeST makes it *more* load-bearing, not less |

**Prometheus is not cancelled.** It is built, debugged and licence-clean, it is the
only arm with a frozen model nobody in this project trained (DynEdge) and a physics
consequence variable (angular error), and it is the fallback if LUCiD's licence does
not land. What it cannot do is carry the Σ̂-versus-Σ claim. Read the table above as
"LUCiD is added to the neutrino side for the covariance claim", not "Prometheus is
removed".

---

## 1. The three arms at a glance

| | **A — LUCiD** | **B — HeST** | **C — TIDMAD** |
|---|---|---|---|
| domain | water Cherenkov ν | superfluid-⁴He DM | axion haloscope, real |
| substrate | simulated PMT waveforms | simulated TES traces | recorded electronics noise |
| what varies | geometry (16 configs), material, particle | geometry (5 builders), ER/NR/WIMP | Σ̂ only (the loss) |
| Σ̂ known | yes (ours) | yes (ours) | yes (the loss) |
| Σ known | yes (ours) | yes (ours) | **no — and that is the point** |
| event pairing | exact (photon dict is an argument) | exact ✅ measured | n/a |
| noise source | `noise_module_lucid` (new preset) | `noise_module` + `qp_simulator` | nobody's — it is real |
| upstream physics author | Tufts/SLAC | SPICE/HeRALD | the ABRACADABRA-style DAQ |
| headline claim it carries | C1, C2, C4 at waveform level in a real geometry | C4 dissociation under a granularity change | external validity of C4 |
| blocker | **no licence** | licence copyright line | `K_rel` must survive |
| cost | 8–12 d after licence | 2–4 d | 3 d (WP10, exists) |

---

## 2. Why three, and why these three

The arms are not three helpings of the same evidence. Each answers a question the
other two cannot.

**The independence argument, which is the reason TIDMAD stays.** Tier 1 (ORACLE-Cov)
and arm B (HeST) both draw their noise from `src/noise_module/`. If that generator
has a systematic quirk — a wrong alias fold in `psd_resampling`, a conditioning
artifact in the Cholesky path at high κ, a spectral preset that happens to flatter
the whitened statistic — it contaminates both arms *identically*. Three synthetic
arms of which two share the instrument under test is not three observations. Arm C
is the only place the noise is real electronics noise that nobody in this project
chose, and therefore the only arm that can catch that class of error. Everywhere
else we set both Σ̂ and Σ; in arm C, Σ̂ varies against a fixed, unchosen Σ. That is
the exact shape of the external-validity claim.

**The domain argument.** A referee's first question about a covariance-geometry
result is "does this depend on your detector?" Arm A is a photon-counting PMT array
at 1 GHz; arm B is a phonon calorimeter at 250 kHz with 1–24 channels; arm C is a
single-channel SQUID readout on real data. Three readout physics, three sampling
rates, three channel counts, one predicted sign.

**The granularity argument.** The abstract axis common to A and B — *detector
granularity at fixed active volume* — is instantiated as `n_sensors` 2000→20000 in
LUCiD and as 24→1 sensors on the identical helium cell in HeST. Same axis, two
domains, opposite ends of the channel-count range. This is what replaces the
(abandoned) idea of giving two experiments the same layout.

---

## 3. Arm A — LUCiD (water Cherenkov)

### 3.1 Why this arm

It is the **only** package in the water-Cherenkov world that emits a dense
per-channel uniformly-sampled time series:
`lucid/simulation/sensor_response.py:389 build_make_hits_waveform(n_photons,
window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=1.0, smear_time=True,
smear_charge=True)` returns `(num_detectors, n_time_bins)` by `segment_sum` of
per-photon smeared charge into 1 ns bins — described in-code as the "1 GHz FADC
convention". WCSim, the standard, has no electronics layer at all (a digit is
`map<int,double> pe/time`; open issue #13 confirms it was never written), and
Prometheus stops at photon arrival times.

It also has no covariance concept anywhere — the only `covariance` in the package is
over fit parameters (`lucid/fitting/fisher.py`, whose weight matrix is literally
`diag`). So `noise_module` fills a hole rather than competing with an incumbent
model whose assumptions we would have to defend.

Two further properties earn their keep: geometry is a four-key JSON file and the
event is a **photon dict passed as an argument** (so pairing is exact, not replayed —
`analysis/paper/fig_charge_displays.py` already loops three geometry families over
one event and one PRNG key), and the whole forward model is JAX, so a simulator-side
Jacobian is available by autodiff for the §6.2 output-null construction.

### 3.2 The blocker, and the fallback

**LUCiD has no licence.** No `LICENSE*` in the tree, no `license` field in
`pyproject.toml`; `README.md` says "The license is being finalized and will be added
shortly" and `CONTRIBUTING.md:67-68` says contributions will be licensed "once it is
finalized". That is all-rights-reserved today, and `EXPERIMENT_DESIGN.md`'s release
posture (ORACLE-Paired to Hugging Face, licence-clean) makes it a hard gate, not a
footnote.

**Action, this week:** one email to Kensuke Terao and Omar Alterkait asking them to
add MIT or BSD-3. They maintain a `CITATION.cff` and a `CONTRIBUTING.md`, so it is
almost certainly an oversight. Budget 1–3 weeks.

**Gate A0.** No LUCiD work starts before a permissive licence is in the repository.
If it has not landed by M3, arm A reverts to Prometheus at hit level, the
covariance-geometry claim is carried by arms B and C alone, and the paper says so.

### 3.3 How it is driven

```
config/<NAME>_geom_config.json           <- geometry, 4 keys, metres
config/<NAME>_physics_config.json        <- optics, QE, TTS, gain, t0, per-PMT arrays
PhotonSim ROOT file (one, fixed)         <- the physical event
        |
setup_event_simulator(geom, n_photons, is_data=True,
                      hit_mode='waveform', window_ns=..., bin_width_ns=...)
        |
        v
charge_waveform : (n_sensors, n_bins)   [photoelectron charge per bin]
        |
   noise_module_lucid  (§7)
        v
voltage_waveform : (n_sensors, n_bins)  [mV, with Sigma_hat / Sigma reported]
```

**Geometry axis.** `analysis/paper/utils/make_geometries.py` already deep-copies
`SK_like_geom_config.json` and overrides only `geometry_definitions.n_sensors` for
2000…20000 in steps of 1000. Reuse it verbatim. Two named cells for the paired claim
(`SK_like` at 11000 and at 4000, say) plus the scan for the trend.

**Physics axis (free, and we were not counting it).** `SK_like` vs `SK_like_wbls`,
`JUNO` vs `JUNO_wbls` — same geometry, different *material* physics config. That is
an S-family generator that costs nothing. Particle type is a setup-time SIREN string
and only `muon` and `electron` emitters ship, so particle-type S-families come from
the PhotonSim path (`GeV/01_mu`, `03_e`, `05_pi0`, `06_pbomb`), not the
differentiable one.

**Traps, all verified:**

- `apply_translation` defaults `true` and draws the vertex from *the detector's own*
  fiducial volume (`lucid/sources/writer.py:323-370`). **Set it `false`** or supply
  the translation, or differently-sized geometries get different vertices at the same
  seed. Same-size scans are unaffected.
- `n_sensors` is a target, not a guarantee: SK_like asks 11000 and places 10764.
  Record the placed count, never the requested one.
- `Cylinder.configure_grid(..., max_candidates_per_ray=4)` **silently drops sensors**
  if the grid is too coarse. Assert placed-sensor count against a reference run.
- `IPython` is an undeclared dependency (`lucid/siren/training/monitor.py:16`,
  imported at module scope via `simulator.py`). `pip install ipython`.
- The soft-overlap lookup table (`lucid/overlap.py:255-365`, defaults
  `n_theta=n_rho=2000`) attempts a **9.5 GB** allocation on first use per sensor
  radius. Budget ≥12 GB RAM, or run `temperature=None` (hard-step mode).
- The GENIE example config the docs reference (`GeV/13_genie_numu_nue.json`) is not
  in the tree. GENIE support is real in code, but unexercised by any shipped example.

### 3.4 What arm A is allowed to prove

| claim | arm A's contribution | why it is arm A and not another |
|---|---|---|
| **C1** detection @1% FAR | on waveforms, in a real detector geometry, across a granularity change | the only arm with both a real geometry and a sampled trace |
| **C2** attribution N vs S, abstention on U | N from `noise_module_lucid` knobs + LUCiD's own acquisition knobs (QE, dark rate, TTS); S from the material axis and PhotonSim particle types; U held out | the material axis gives an S-family that is unambiguously *physics*, not acquisition |
| **C4** alarm ranks consequence | **the transfer test**: the κ-sweep prediction registered on Tier 1 is re-run here, in a geometry, with a different readout physics | this is the whole point of the arm — Tier 1 proves it, arm A shows it is not an artifact of the synthetic substrate |
| **C5** stage localization | replication only, by activation patching | no new mechanism |

Arm A does **not** carry: the whitening lemma itself (Tier 1), the frozen-public-model
claim (Prometheus/DynEdge), or a physics consequence variable in the NuBench sense —
its consequence variable is reconstruction error from LUCiD's own fitting layer
(`lucid/fitting/recon.py`), which we would be defining, so it must be declared as
such and not dressed up as an external metric.

---

## 4. Arm B — HeST (dark matter)

### 4.1 Why this arm

Because **it stops exactly where our code starts.** ✅ Verified: `HestSignal` has two
fields and nothing else —

```
fields: ['energies', 'arrivalTimes']     # each a list of per-sensor lists
```

No digitizer, no trace, no electronics, no noise model. `src/qp_simulator/QPSimulator.py`
already consumes per-sensor arrival times, and its docstring already ends
`trace + my_noise_module.generate(len(trace))`. Σ̂ and Σ are ours by construction
because nothing else sits between the quasiparticle list and the sampled trace.

And because the physics upstream is **not ours**. Tier 1 is already entirely
self-authored; if the dark-matter arm were too, a referee can say we built the world
we tested in. HeST puts independently written, published detector physics
(PRD; arXiv:2307.11877) upstream of our noise. That is what makes arm B a different
kind of evidence from Tier 1 rather than a second helping of it.

Third, superfluid ⁴He is DELight's own target material, which converts the arm from a
benchmark into a DELight-relevant result — worth something to the authorship position
recorded in `claude/noise-module-ip-and-nubench-reproduction.md`.

### 4.2 What HeST simulates

Two stages. Not Geant4, no photon transport, no background model — the energy
deposit is an **input**, not an output.

**Yields.** `HeST/core/HeST_Core.py:GetQuanta(energy_eV, interaction)` partitions a
deposit four ways. ✅ Measured at 1 keV:

| | quasiparticles | IR photons | singlet UV | triplet |
|---|---:|---:|---:|---:|
| `"NR"` | 900,238 | 7 | 20 | 1 |
| `"ER"` | 419,925 | 116 | 18 | 11 |

The 2.1× quasiparticle ratio at equal deposit is the ER/NR handle — a one-argument
switch. `core/WIMP_Generation.py` gives `WIMP_spectrum(mass_MeV, ...)` for a recoil
spectrum instead of a fixed energy.

**Transport.** `core/Detection.py:2078
GetEvaporationSignal(detector, QPs, X, Y, Z, useMap=False, T=2.0)` launches `QPs`
quasiparticles isotropically from a vertex in cm, ray-traces them against the cell
surfaces with per-surface reflection / diffuse / Andreev probabilities, and records
those that evaporate a helium atom onto a sensor.

### 4.3 Geometry: Python, not a config file — and that is better here

A `VDetector` is five boolean implicit-surface functions (`top`, `bottom`, `wall`,
`liquid_surface`, `liquid_conditions`) plus a list of `VSensor`s, each its own boolean
condition, plus the surface-interaction probabilities. The five shipped builders in
`core/Geometry.py` are ~60-line functions. `HeRALD_v1`'s array is a literal:

```python
sqcm_width = 1;  sqcm_pitch = 1.1;  cell_rad = 3.5   # cm
array_map = np.array([[0,0,1,1,0,0],[0,1,1,1,1,0],[1,1,1,1,1,1],
                      [1,1,1,1,1,1],[0,1,1,1,1,0],[0,0,1,1,0,0]])
```

So a geometry sweep is a factory over `(cell_rad, fill_height, sqcm_pitch, array_map)`.
No parser, no GDML, nothing to add to the package. ✅ `array_map.sum() == 24`, so
`HeRALD_v1` has **24** sensors.

✅ Verified inventory: `HeRALD_v1` 24 · `HeRALD_v1_monolithic` 1 ·
`HeRALD_UMass_splitCPD` 2 · `HeRALD_UMass_monolithic` 1 · `HeRALD_LBNL` top+bottom.

### 4.4 ⭐ Pairing is exact — measured, not inferred

`QP_propagation` (`core/Detection.py:1171`) samples the **entire** initial population
up front and vectorised — `generate_random_direction(nQPs)` at `:1244`,
`Random_QPmomentum(nQPs, T=T)` at `:1258` — *before any geometry is touched*.
Geometry enters only in the propagation loop afterwards. ✅ 20,000 QP from
(0, 0, 2.0) cm, seed 42:

```
v1_28        nsensors=24  detected= 131
mono         nsensors= 1  detected= 174
umass_split  nsensors= 2  detected= 281

initial QP momenta identical across all three geometries: True   n = 20000
```

Same event, three geometries, three observed signals, no replay machinery and no
intermediate file. A stronger pairing guarantee than ldmx-sw's `ReSimulator` or
Prometheus's injection replay, both of which re-run a stochastic propagation.

### 4.5 Cost and hazards

✅ Throughput `useMap=False`, one core, post-JIT: 2000 QP → 0.03 s (69.6k QP/s);
10000 QP → 0.19 s (53.5k QP/s), ~0.8 % evaporation efficiency. A 1 keV NR event
(900k QP) is **~17 s single-core** yielding ~7000 detected quasiparticles. Sub-keV
scales down linearly. Embarrassingly parallel.

- **Maps are not shipped** (`get_QPEmap()` returns `0.0`; `map_generation/` absent).
  Irrelevant: `useMap=False` is full first-principles propagation. `create_LCEmap` /
  `create_QPEmap` exist if the speedup is wanted later.
- **Install:** `setup.py` requires `detprocess` → `annoy` → `aplus`, which is
  Python-2-era and fails to build wheels. `pip install qetpy numba` is enough;
  `HeST_Core.py:5` only imports four names from `qetpy.utils`.
- **Licence:** MIT *text*, but the copyright line is the unedited PyPA sample,
  `Copyright (c) 2018 The Python Packaging Authority`. **Gate B0:** ask Greg
  Rischbieter (rischbie@umich.edu) to name the real holder before a released dataset
  depends on it — the same hygiene `src/noise_module/` just went through.
- **Bus factor:** 89 commits, one maintainer, "early developmental version", 0 stars,
  3800 lines. Vendor a pinned copy under `src/` once the copyright line is fixed.
- Minor bug to report: `VDetector.get_QPEmap()` returns `self.LCEmap_positions`
  (`core/Detection.py:243`).

### 4.6 The chain, and the one thing to build

```
GetQuanta(E, "NR"|"ER")          -> n quasiparticles          [HeST, exists]
GetEvaporationSignal(detector,…) -> per-sensor arrival times  [HeST, exists]
QPSimulator.generate(times_ns)   -> clean per-channel trace   [ours, exists]
MultiChannelNoiseGenerator       -> Sigma_hat / Sigma          [ours, exists]
```

**WP-B1 (~40 lines).** A loop calling `QPSimulator` once per sensor and stacking to
`(n_sensors, n_samples)`. Units: HeST reports µs; `QPSimulator` takes ns at
2.5e5 Hz × 16384 samples = 65.5 ms, against a HeST window of order 5 ms — one
conversion, no resampling.

### 4.7 What arm B is allowed to prove

| claim | contribution |
|---|---|
| **C4 (headline for this arm)** | the designed dissociation under a *granularity* change: 24 channels → 1 channel on the identical helium cell. With C=1 the multichannel covariance claim is vacuous by construction; with C=24 it is not. That is a designed contrast with a predicted sign, not an observation. |
| **C1, C2** | replication in a second readout physics; N families from `noise_module` knobs and HeST's surface probabilities, S families from ER/NR and the four yield channels, U from a WIMP spectrum held out |
| **C5** | replication only |

Arm B does **not** carry a physics consequence variable of external provenance. Its
consequence is whitened reconstruction error on the trace — the same variable the
2026-09-02 Tier-1 review recommended over the amplitude readout. Declare it.

---

## 5. Arm C — TIDMAD (real data)

### 5.1 Why it survives, and why arm B makes it *more* necessary

See §2. In one line: arms A and B are both simulated and arm B shares its noise
generator with Tier 1, so arm C is the only evidence that the effect is not a
property of `src/noise_module/`.

Cost is also decisive: WP10 is **3 days**, depends only on WP1–WP6 which are needed
anyway, and `src/tidmad_transformer/` already exists with the int8-saturation and
uninitialised-tail bugs (C1, C2) fixed on `dev`.

### 5.2 How it is driven

Unchanged from `IMPLEMENTATION_PLAN.md` WP10: the same ~0.6M-parameter two-stage
transformer trained twice with **identical seeds and data**, differing *only* in the
loss — MSE versus inverse-PSD-weighted. Two Σ̂ on one fixed, unchosen Σ. Monitors on
both; `K_dev` disclosed as non-scientific per D2; `K_rel` on the hash-frozen 20-file
subset with the 328-band truncation and in-band file selection declared.

### 5.3 What it is allowed to prove, and the exit condition

Arm C proves exactly one thing: **the monitor contrast between the two Σ̂ trainings
has the sign predicted on Tier 1, on noise nobody in this project generated.** It is
one paragraph. It is the paragraph that blocks "it only works in your simulator".

**Exit condition.** D2 records that TIDMAD's own authors disclaim the denoising
score, so the consequence variable rests entirely on the Brazil-band `K_rel`. If
`K_rel` does not survive scrutiny, arm C degrades to "two Σ̂ trainings, monitor
contrast, no consequence variable" — still worth the paragraph as an external check
on the monitor, but no longer claim-bearing. `IMPLEMENTATION_PLAN.md` already lists
exactly this descope. The replacement, if a claim-bearing real-data arm is required,
is LIGO/Gravity Spy, where the assumed covariance is literally the published ASD and
glitches are labelled Σ̂ ≠ Σ with ground truth — budget 2–3 weeks of domain entry
before treating it as available.

---

## 6. Claims × arms

| claim | Tier 1 ORACLE-Cov | A · LUCiD | B · HeST | C · TIDMAD | Prometheus |
|---|---|---|---|---|---|
| C1 detection @1% FAR | controlled families, power sizing | waveform-level, in geometry | second readout physics | — | frozen DynEdge |
| C2 attribution + abstention | full control, held-out U | material-axis S; U held out | ER/NR S; WIMP U | — | matched cells, U1–U4 |
| C3 cost | compact transformer | — | — | — | DynEdge |
| **C4 alarm ranks consequence** | **designed dissociation + κ sweep** | **transfer test of the registered κ prediction** | **designed granularity dissociation, 24→1** | **external validity** | observational, angular error |
| C5 stage localization | activation patching | replication | replication | — | replication |

The bolded row is the paper. Read left to right it is: proved, transferred to a
geometry, transferred to a second readout physics, survived real noise.

---

## 7. `noise_module_lucid` — the customised noise module

### 7.1 The finding that makes this cheap

**Almost nothing has to be written.** ✅ Verified by running it: the spectral layer is
already composable and physics-agnostic, driven from a registry
(`spectral_models._COMPONENT_TYPES`) via `NoiseConfig.components`, so a PMT front-end
ASD is **pure configuration** — no new `SpectralComponent` subclass, no new module:

```
record 512 ns | df = 1.953 MHz | Nyquist = 500 MHz
PSD grid: (257,) | integral = 1.0000 (target noise_power=1.0)
noise block: (64, 512) | metadata cov keys: ['implied_correlation',
  'implied_covariance', 'realized_correlation', 'realized_covariance']
kappa(Sigma_hat^-1 Sigma) = 8.252 | mean offdiag corr = 0.344
```

So `noise_module_lucid` is a **thin adapter package plus a preset**, not a fork. Four
things change; the other ~5000 lines do not.

### 7.2 ⚠️ Correction: 50 Hz mains is not representable

An earlier note suggested `artifact_injector`'s spectral lines were "already exactly
the mains-pickup case". ✅ That is wrong at LUCiD's native grid:

```
record 512 ns | df = 1.953 MHz
-> 50 Hz mains sits 2.56e-05 of one bin above DC: NOT representable
```

Over a 512 ns window, 50 Hz is a DC offset. The in-band physics between 2 MHz and
500 MHz is different and, fortunately, richer:

| component | `type` | physical origin | in band? |
|---|---|---|---|
| amplifier white floor | `white` | front-end thermal + shot | dominant |
| front-end bandwidth | `rolloff` (lowpass, corner ~250 MHz, order 2) | preamp / cable | yes |
| clock & switching pickup | `line` (e.g. 62.5 MHz, width 4 MHz) | ADC clock, DC-DC converters and harmonics | **yes — this is the real coherent line source** |
| cable-reflection ringing | `resonance` (centre ~150 MHz, half-width 20 MHz) | impedance mismatch | yes |
| flicker | `powerlaw` (−1, ref 10 MHz) | front-end 1/f | weak, ~2 decades only |

**1/f, drift and mains need a longer record.** `window_ns` is an argument, so raise
it; to reach the kHz decade at 1 ns bins would need ~10⁶ samples, so the practical
route is **decimation** — and that is where `psd_resampling.alias_fold_psd_density`
earns its place, because decimating without an anti-alias filter folds the
>Nyquist/2 content down. That fold is itself a declarable acquisition-contract term,
i.e. a free and physically honest **N-family**.

### 7.3 ⚠️ Two measured constraints that would otherwise poison the κ sweep

**(a) The realized-covariance estimator has a floor set by N/C.** ✅ Measured, single
`generate()` call, `shared_private`, `normalize_channel_variance=False`, seed 3:

| C | N=512 | N=4096 | N=32768 | N=262144 |
|---:|---:|---:|---:|---:|
| 16 | 1.968 | 1.297 | 1.105 | 1.037 |
| 64 | 8.252 | 2.174 | 1.346 | 1.116 |

These are **matched** cells — Σ̂ = Σ by construction — so every value above 1.0 is
estimator noise, not mismatch. Rule of thumb: **N/C ≳ 500 for a κ floor below ~1.1.**

Consequences for the design, and they are not optional:

- At LUCiD's default `window_ns=500`, a 64-channel cell has a κ floor of ~8. Any
  injected mismatch below that is invisible. **Use `window_ns` ≥ 16–32 µs**
  (16k–32k bins) for the covariance cells.
- **Do not use all ~10,764 PMTs as one covariance unit.** Take a *channel group* —
  a crate, a string, or a 16–64 PMT sub-array — as the covariance unit. This is also
  the physically right choice, because coherent pickup is per-crate, and `string_id`
  is already in LUCiD's output so the grouping key is free.
- Report the matched-cell κ floor alongside every swept κ. A κ sweep without its
  own null is not interpretable.

**(b) Channel gains are redrawn on every `generate()` call.** ✅ Verified:

```
gains identical across two generate() calls on the same object: False
```

`generate_shared_private` draws `gains` and `private_strengths` inside the method, so
each call implies a *different* covariance. Pooling records across calls therefore
does not estimate one Σ — it mixes several, and κ plateaus instead of converging
(measured: pooling 512 records at C=64 gave 5.26, versus 1.12 for a single long call).

**WP-N1:** add a `freeze_channel_gains=True` option (or a `gains=`/`private_strengths=`
override) to `MultiChannelNoiseGenerator`, so one covariance can span many records.
Small change, in `src/noise_module/`, and it benefits Tier 1 and arm B too.

### 7.4 The units bridge — the one piece of real physics to write

LUCiD's waveform bin holds **summed photoelectron charge**, not volts. A PSD on that
array has no physical meaning. So:

```
charge_waveform (n_sensors, n_bins)   [pe per 1 ns bin]
        |  convolve with a per-channel SPE voltage template
        v
signal_mV (n_sensors, n_bins)
        +  noise_mV from MultiChannelNoiseGenerator, PSD in mV^2/Hz
        v
trace_mV
```

The template is a two-pole pulse; `templates.pulse_template_2` already implements the
Probst/`cait` form and is directly reusable with PMT time constants (rise ~1–3 ns,
decay ~5–10 ns). Calibration constant: mV per photoelectron, taken from LUCiD's own
`gain` in the physics config so the two layers cannot silently disagree. ~40 lines.

### 7.5 Proposed layout

```
src/noise_module_lucid/           # new package, MIT, depends on noise_module
  __init__.py
  presets.py        PMT_FRONTEND_V1 : NoiseConfig components + provenance record
                    (the table in §7.2, every constant named and cited)
  units.py          spe_template(fs, tau_rise, tau_decay), charge_to_mV(...)
  adapter.py        add_readout_noise(charge_waveform, detector, cfg, rng)
                       -> (trace_mV, metadata)   metadata carries Sigma_hat, Sigma,
                          kappa, the matched-cell kappa floor, and the grouping key
  grouping.py       channel_groups_from_string_id(detector) -> list[np.ndarray]
  interventions.py  N-families: gain drift, dark-rate change, clock-line amplitude,
                    decimation/alias fold, group-coherence change
  tests/
```

Nothing in `src/noise_module/` is forked. The only upstream change is WP-N1 (§7.3b).

### 7.6 Acceptance criteria

1. `presets.PMT_FRONTEND_V1` builds a one-sided PSD whose discrete integral equals
   `noise_power` to 1e-12 on the LUCiD grid, and every component carries a named
   physical origin in the provenance record. *(Pattern: `reference_budget.py`.)*
2. On a **matched** cell (Σ̂ = Σ), the reported κ equals the measured estimator floor
   for that (C, N) to within its bootstrap interval — i.e. the null is calibrated
   before any mismatch is injected.
3. A swept κ cell reproduces its requested κ to within the matched-cell floor, over
   at least one decade.
4. `validate_csd_ensemble` passes on the generated multichannel block at the LUCiD
   sampling rate.
5. Round-trip: `charge_to_mV` applied to a unit-charge impulse returns a pulse whose
   integral equals the configured mV·ns per photoelectron.
6. The decimation/alias-fold N-family changes the realized in-band PSD in the
   direction `alias_fold_psd_density` predicts, checked against the closed form.
7. Every generated dataset writes a provenance record naming: LUCiD commit, geom and
   physics config hashes, placed sensor count (not requested), `apply_translation`,
   `window_ns`, `bin_width_ns`, the preset version, the grouping, Σ̂, Σ, κ, and the
   matched-cell κ floor.

### 7.7 Effort

| | days |
|---|---:|
| WP-N1 `freeze_channel_gains` in `noise_module` | 0.5 |
| `units.py` + SPE template + calibration | 1 |
| `presets.py` + provenance record | 1 |
| `adapter.py` + `grouping.py` | 1.5 |
| `interventions.py` (5 N-families) | 2 |
| tests + the matched-cell κ-floor calibration | 2 |
| **total** | **~8** |

---

## 8. Sequencing, gates, descope

**Gates.** A0 — LUCiD licence in the repository. B0 — HeST copyright line names a real
holder. Both are emails; send both this week. Neither blocks Tier 1.

**Order.** Tier 1 (unchanged) → arm B pilot (half a day; §4.6) → arm B build (2–4 d)
→ `noise_module_lucid` (8 d, starts only after gate A0) → arm A build → arm C (WP10,
3 d, any time after WP6) → pre-registration bridge → confirmatory runs.

Arm B goes first among the new work because it is cheap, ungated by anyone else's
licence decision, and it exercises `noise_module` against a *non-synthetic* upstream
for the first time — which is the risk most worth retiring early.

**Descope order, replacing the one in `IMPLEMENTATION_PLAN.md` §4.** The existing
list (second geometry → cov_E/C5 → WP10's `K_rel`) predates arm B. Revised:

1. The LUCiD geometry *scan* (keep two named geometries; the trend becomes future work).
2. Arm B's WIMP-spectrum U family (keep ER/NR).
3. cov_E / C5.
4. Prometheus's second geometry — ORCA only.
5. WP10's `K_rel`, keeping the two-Σ̂ monitor contrast.

Arm B and arm C are the last things to cut, for opposite reasons: arm B is the
cheapest claim-bearing arm, and arm C is the only unshared substrate. If the whole of
arm A has to go, the paper still has proved → transferred → survived-real-noise; it
loses "in a real detector geometry", and it must say so.

---

## 9. Open questions

- **Does the κ prediction transfer across sampling rate?** Tier 1 runs at the
  athermal-calorimeter rate, arm A at 1 GHz, arm B at 250 kHz. The lemma is
  rate-free, but the *estimator floor* is not — §7.3a's N/C rule has to be
  re-measured per arm and reported. Do this before the pre-registration bridge is
  frozen, not after.
- **What is arm A's consequence variable?** LUCiD's own `recon.py` is ours to define,
  which weakens it relative to Prometheus's angular error. Options: use LUCiD's
  Fisher/CRB machinery (`lucid/fitting/fisher.py`, remembering its documented √12
  "honesty factor"), or keep the physics consequence variable on the Prometheus arm
  and let arm A carry only the waveform-level claims. **Decide before building.**
- **Does arm B need a background model?** HeST has none. If the paper wants a
  realistic rate rather than a designed family, that comes from elsewhere and is
  probably not worth it.
- **Should `noise_module_lucid` live in ORACLE or beside `noise_module`?** Release
  posture says `noise_module` does not ship before the preprint. A separate package
  that *depends* on it can be released independently later; a fork cannot.
