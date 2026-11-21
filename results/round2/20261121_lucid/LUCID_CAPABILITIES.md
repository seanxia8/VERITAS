# LUCiD capability audit and pre-decided defaults (r2-lucid L2)

LUCiD `e651e836`. Paths are relative to the clone `/ceph/dwong/jepa/external/LUCiD_main_e651e836`.

## Chosen sampling tier: **A1 — `dataset_kind = lucid_mc_native`**

The generator runs LUCiD's own dataset-production kernel and electronics, i.e. the `lucid-run-job` chain minus PhotonSim:
- Kernel: `setup_event_simulator(geom, B, source='photons', K=12, temperature=0.0, charge_resolution=None, physics_config=..., default_detector_params=True, hits='segment_truth')` — exactly `lucid/production/dataprod/run_job.py:379-399` (`_simulator`). For `source='photons'` transport and detection default to `'sampled'` (`docs/reference/simulator.md`, "transport"/"detection" rows; `lucid/simulation/simulator.py:979-987`): every absorption, Rayleigh/Mie scattering, wall/sensor reflection and QE decision is a hard Monte Carlo draw (`lucid/simulation/photon_step.py:114-157`), and `segment_truth_sampled` (`lucid/simulation/hits.py:289-355`) returns the per-photoelectron deposits (sensor, true and smeared time, weight).
- Variable photon counts use LUCiD's `N` mask (`lucid/simulation/simulator.py:691-700`), the same mechanism as LUCiD's ray buckets (`lucid/sources/interaction.py:57-95`).
- Electronics: LUCiD's digitizer (`lucid/simulation/digitizer.py`): dark noise `generate_dark_noise` (:679), integration windows `digitize_event` (:211), sampled SPE charge and charge-dependent PMT jitter `apply_readout_resolution` (:417, `_sample_time_jitter` :366), TDC 0.4 ns, discriminator `apply_discriminator` (:402), late-light cap `_MAX_DIGIT_TIME_NS` (:137); then LUCiD's sliding-window trigger `find_trigger_gates` / `hits_in_gates` (`lucid/simulation/trigger.py:63,120`) where the detector config has one.

Tier A1 was available, so tiers A2 and B were not needed.

## Questions and decisions

| Question | Finding (file:line) | Decision |
|---|---|---|
| Detector configs | Measured-PMT cylinders `WCTE` (1995), `SK`/`SK_WAND` (11096), `HK`/`HK_WAND` (19746) via `Cylinder.from_pmt_file` (`lucid/geometry/cylinder.py:163`); `JUNO` = generated water sphere, R 17.5 m, 10000 sensors (`config/JUNO_geom_config.json`). `docs/getting-started/detectors.md`. | WCTE → `WCTE`; SK → `SK_WAND`; HK → `HK_WAND` (the frozen production descriptions with a readout block; same geometry and optics as `SK`/`HK`); JUNO → `JUNO`. PMT counts and radii equal the toy geometries (below). |
| Sensor order / positions | `from_pmt_file` reorders sensors barrel/top/bottom (`cylinder.py:229-242`, `_reorder_indices`) and snaps them onto the wall (`:246-247`, `_snap_to_wall` :292). | Index map frozen (`configs/lucid_v1/<det>_geometry_e651e836.npz`, `lucid_to_r2 = _reorder_indices`); raw LUCiD positions equal the round-2 positions exactly. After snapping: HK max 0.0018 mm, JUNO 0.001 mm → round-2 geometry reused; WCTE max 57.7 mm (median 5.8 mm, mPMT domes), SK max 28.7 mm → new frozen geometry with LUCiD's positions (round-2 order, modules and directions kept). |
| Physics processes | `config/materials/water.json`: Rayleigh + symmetric Mie and forward Mie (SK calibration, arXiv:1307.0162), absorption (SK + Pope & Fry); wall/sensor reflection with specular/diffuse split, `reflection_model='scalar_mix'` default (`lucid/simulation/reflection.py:93-110`); wavelength-dependent QE curves `config/pmt/SK_QE.json`, `HK_QE.json`; Cherenkov wavelength sampling `lucid/wavelength/spectrum.py:13`. | Everything on, LUCiD defaults: SK/HK/JUNO `wavelength_mode=True` with `water.json`; reflections from each physics config (wall 0.05/0.20, sensor 0.25/0.20). **WCTE**: its shipped physics config has no medium model and is scalar-only (`scalar_mode`: scattering 50 m, absorption 50 m, no Mie, QE 0.2), so WCTE runs `wavelength_mode=False` with LUCiD's own WCTE optics (recorded, not edited). K = 12 scattering/reflection iterations (production value). |
| Data-like sampling mode | Yes: sampled transport + detection + `segment_truth` hits (above) plus the digitizer. | Tier A1 (`lucid_mc_native`), recorded in every manifest. |
| Isotropic point source | `IsotropicSourceRandom` / `get_isotropic_rays_random` (`lucid/sources/calibration_sources.py:62,170,239`); the lattice version ignores its key, the random one is "for truth generation". | Photon bombs: directions from `get_isotropic_rays_random`, emission at t = 0, wavelengths sampled by LUCiD from the Cherenkov spectrum (LUCiD's default when a source gives none; `simulator.py:339-347`), per-event truth offset uniform in [−50, 50] ns added before digitizing. No VERITAS source was needed. |
| Muon and electron tracks | SIREN Cherenkov emitters for μ⁻ (domain 150–100,000 MeV) and e⁻ (1–100,000 MeV) in water (`data/water/{muon,electron}/cherenkov/emitter.json`); ray generator `make_cherenkov_surrogate_fn` (`lucid/sources/siren_rays.py:128`), emission time `predict_t0` (:440). The production chain needs PhotonSim/Geant4 (container), which is not available here. | μ⁻ 300–1500 MeV and e⁻ 100–1000 MeV, log-uniform (round-2 ranges, `lucid_bombs_v2_calibrated.json`), vertices uniform in the fiducial volume, isotropic directions, 1:1:1 domain plan as round 2. SIREN rays (weights = PMF × n_photons(E)) become unit photons by thinning with p = weight / n_chunks (n_chunks raised until the largest weight is ≤ 1); rays emitted outside the tank are dropped; the photons then go through the same production kernel. |
| Sensor response | `ResponseParams.tts` defaults to 0 (`lucid/detector_params.py:109-131`) and none of the four physics configs sets it; the PMT timing response lives in the digitizer (WCSim-derived charge-dependent jitter). Presets `basic`/`ski`/`hk` (`digitizer.py:93-135`); only `SK_WAND`/`HK_WAND` carry `digitizer`/`trigger` blocks (`docs/reference/digitizer-and-trigger.md`). | SK `ski` (SK 20″ SPE, Gaussian jitter); HK `hk` (R12860 B&L SPE, EMG jitter); JUNO `hk` (JUNO's 20″ Hamamatsu PMTs are the R12860); WCTE `hk` (closest LUCiD preset to a fast 3″ PMT; LUCiD has no 3″ model). Dark noise 4.2 kHz per PMT, threshold 0.25 PE, TDC 0.4 ns from the presets. LUCiD trigger on SK (200 ns, 25 hits) and HK (200 ns, 45 hits); none shipped for WCTE/JUNO. VERITAS adds only the NHits token-time trigger and prompt window (P1(c)); no VERITAS dark counts (LUCiD has them), no `noise_module_lucid` electronics (LUCiD covers SPE charge, jitter, threshold). |
| Output schema | `docs/reference/dataset-schema.md` (four-file production batch). | Converted into the round-2 per-run shards / `r2_store` pack (L3); LUCiD's native per-event digits are kept alongside as cache pickles (`/ceph/dwong/jepa/data/lucid_v1/<det>/cache/`). |

## Geometry against the toy

| Detector | PMTs LUCiD / round 2 | Radius, height (m) LUCiD / round 2 | PMT radius (m) | Position change |
|---|---|---|---|---|
| WCTE | 1995 / 1995 | 1.577, 2.777 / same | 0.040 | wall snap, max 57.7 mm |
| SK | 11096 / 11096 | 16.962, 36.378 / same | 0.254 | wall snap, max 28.7 mm |
| HK | 19746 / 19746 | 32.430, 65.929 / same | 0.254 | < 0.002 mm |
| JUNO | 10000 / 10000 | sphere 17.5 / same | 0.250 | < 0.001 mm |

## Regime (photon-bomb yield range, L2 pilot)

Detection efficiency (detected photoelectrons per emitted photon) at 60 random fiducial vertices × 20,000 photons, LUCiD kernel: WCTE 0.0659, SK 0.0652, JUNO 0.0345, HK 0.0503 (means). Log-uniform yield ranges solved for detected-PE quantiles 200 / 3000 / 30,000 (predicted 205 / 2500 / 30,800):

| Detector | Yield range (photons) | Toy range (round 2) |
|---|---|---|
| WCTE | 2,369 – 621,416 | 6,298 – 539,434 |
| SK | 2,383 – 623,687 | 24,904 – 1,881,272 |
| JUNO | 4,546 – 1,181,190 | 16,629 – 1,027,996 |
| HK | 3,097 – 810,763 | 120,970 – 10,974,019 |

Achieved quantiles are in `VALIDATION.md` (G4).

## Dataset-derived constants (`src/jepa_noise/configs/lucid_v1.json`)
Round-2 procedures re-run on the LUCiD data: the prompt window and token time scale depend only on geometry and are unchanged (WCTE (−20, 60) ns / 100 ns; SK (−20, 270) / 450; JUNO (−20, 200) / 333.3; HK (−20, 460) / 766.7). The NHits rule (`r2_geometry.nhits_n_rule`, 50 × P(Poisson(λ₂₀) ≥ N) ≤ 10⁻⁵) with LUCiD's 4.2 kHz dark rate gives N = 6 (WCTE), 10 (SK), 10 (JUNO), 12 (HK) instead of 4/6/6/8 at the toy's 1 kHz.
