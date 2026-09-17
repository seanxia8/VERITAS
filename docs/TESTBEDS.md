# Testbeds — the canonical list

_10 September 2026, branch `dev`. This is the one document that says which
simulations and which real datasets the verification programme uses, what each
one lets us control, what is already implemented in this repository, and what
the autoencoder has to look like for the controlled-variable diagnostics to be
readable on all of them. It supersedes the candidate lists scattered across
`docs/archive/SIM_TESTBED_SURVEY_2026-09-05.md` and `docs/archive/EXPERIMENT_PLAN_ARMS_2026-09-05.md`
(which remain the evidence files and are cited, not restated). Every "✅" below
was checked on this machine or on the upstream repository today._

---

## 0. The requirement, in one paragraph

A testbed serves the programme if three things are true at once. **Physics is
controllable**: the event type, energy, vertex and signal shape can be set and
held fixed. **Geometry is controllable**: sensor count, layout and cell size
can be changed while the physics is held fixed — ideally so that the *same
event* is observed by two detectors (exact pairing). **Noise is ours**: the
simulator emits a clean, per-channel, uniformly sampled trace and has no
electronics model of its own, so `src/noise_module/` supplies the noise with a
known assumed covariance Σ̂ and a recorded realized covariance Σ. Real datasets
cannot satisfy the third condition — that is what makes them the external
check — but they must at least ship sampled traces, some noise-only records
or a published spectral model, labels for signal type, and some
configuration variation.

---

## 1. First stage: the two simulations

| | **Arm A — LUCiD + `noise_module`** | **Arm B — HeST → `qp_simulator` → `noise_module`** |
|---|---|---|
| domain | water-Cherenkov PMT array (neutrino) | superfluid-⁴He calorimeter with TES/CPD sensors (dark matter; DELight's target material) |
| upstream physics (not ours) | LUCiD, JAX photon transport, `hit_mode='waveform'` → `(n_sensors, n_bins)` photoelectrons at 1 ns | HeST: `GetQuanta(E, "ER"\|"NR")` yields → `GetEvaporationSignal(detector, …)` per-sensor quasiparticle arrival times |
| trace (ours) | SPE voltage template ⊗ pe histogram → mV | `QPSimulator`: single-QP template (50 µs rise, 3 ms decay), 250 kHz × 16 384 samples |
| noise (ours) | `PMT_FRONTEND_V2` (`notebooks/pmt_frontend_v2.py`, 11 Sep): private = (amplifier white floor + weak 1/f) × \|H_fe\|² (250 MHz low-pass) × \|H_ring\|² (150 MHz bump) via `filtered`; shared per 64-PMT crate = 62.5 MHz clock + harmonics; `spectral_shared_private`, so ρ_ij(f) ≈ 0.9 at the clock and ≈ 0 on the floor. V1 (additive roll-off, flat 0.3 coherence) retired — `docs/reviews/LUCID_NOISE_REVIEW_2026-09-11.md` | `noise_module.tes_budget.HERALD_V1_PLACEHOLDER`: TFN, TES + shunt Johnson, SQUID white + 1/f, 50 Hz mains + harmonics, vibration lines; shared-private per cell |
| physics axis | source type/intensity (built-in isotropic flasher; PhotonSim μ/e/π⁰ files; SIREN μ/e gun), material at fixed geometry (water / WbLS) | ER vs NR at fixed energy; energy; vertex; WIMP recoil spectrum (held-out U) |
| geometry axis | one JSON file: `detector_type ∈ {cylinder, sphere, box, string}`, radius/height, `n_sensors` (target, not guarantee); 16 shipped configs | `make_cell(cell_radius_cm, fill_height_cm, sensor_pitch_cm, array_map)` from HeST primitives; shipped `HeRALD_v1` (24), `_monolithic` (1), `UMass_splitCPD` (2), `UMass_monolithic`, `LBNL` |
| pairing | exact — the photon source and PRNG key are arguments; set `apply_translation=False` for differently sized detectors | exact and **tested** — `QP_propagation` draws the whole initial population before geometry is touched; `event_id` seeds it (`tests/test_pairing.py`) |
| Σ̂ / Σ / κ reported | yes, per crate (`MultiChannelNoiseGenerator` metadata) | yes, per cell (`add_noise` → `kappa_floor` in `truth.parquet`) |
| **implemented?** | **notebook only** — Part B of `notebooks/one_event_herald_lucid.ipynb` and `notebooks/noise_models_herald_lucid.ipynb`; deliberately no `src/lucid_simulation` package | **yes** — `src/herald_simulation/` (14 cells, `simulate.py`, `tests/`, provenance), Part A of the same two notebooks |
| gate | **A0 — LUCiD has no licence.** ✅ Re-checked today: no `LICENSE` file, no `license` field in `pyproject.toml`, README still "under construction"; last upstream commit 2026-08-21 | **B0 — HeST's `LICENSE` is MIT text whose copyright line is still the unedited PyPA sample.** ✅ Re-checked today; last upstream commit 2026-03-09 |

### 1.1 Arm A status in detail — what exists, what does not

**Exists (executed, outputs stored; ✅ 14/14 code cells, 0 errors, 11 figures
in `one_event_herald_lucid.ipynb`; 9/9, 8 figures in
`noise_models_herald_lucid.ipynb`):**

- LUCiD imported from an external clone (`external/LUCiD`, or `LUCID_PATH`),
  `WCTE_like` cylinder (r = 2 m, H = 4 m, ~2 444 PMTs), isotropic flasher at
  the centre, 50 000 photons, `temperature=None` (hard-step overlap, avoids the
  9.5 GB lookup table), 512 ns window at 1 ns.
- The three moves, one at a time: **event** (intensity ×3), **geometry** (same
  flasher through the cylinder, a box, and a sparser cylinder: 2 444 → ~800
  PMTs), **noise** (clock line ×5; crate coherence 0.3 → 0.7; alias fold from
  decimating 1 GHz → 250 MHz without an anti-alias filter).
- The units bridge (`spe_template`, `to_mv`), the preset `PMT_FRONTEND_V2` and
  the crate-wise `add_pmt_noise` live in `notebooks/pmt_frontend_v2.py` (a helper next
  to the notebooks, not a package). The executed `.ipynb` outputs still show V1
  until the notebooks are re-run against a LUCiD clone; the builders
  (`_build_nb1.py`, `_build_nb2.py`) are on V2.
- The measured constraint that shapes the arm: on a 512-sample, 64-channel crate
  the matched-cell κ floor is ≈ 5 (N/C = 8), so covariance cells need
  `window_ns` ≥ 16–32 µs, and a crate — not the whole tank — is the covariance
  unit.

**Does not exist (and must not, until A0 lands):** the `noise_module_lucid`
package planned in `docs/EXPERIMENT_DESIGN.md` §II.7.5 (`presets.py`,
`units.py`, `adapter.py`, `grouping.py`, `interventions.py`, tests, provenance
record); the N-family interventions as code; a `Cell` list mirroring
`herald_simulation.simulate.all_cells`; the geometry *scan* (`n_sensors`
2000…20000). Effort once gated: ~8 days (arms plan §7.7).

**To re-run the notebook on this machine:** `git clone
https://github.com/CIDeR-ML/LUCiD external/LUCiD` (there is no `external/` in
the working tree today) and `pip install jax jaxlib flax optax h5py ipython`.
The notebook was executed elsewhere and carries its outputs; nothing from
LUCiD is vendored.

### 1.2 Arm B status in detail — what exists, what is placeholder

**Exists:** the full chain and its harness. `herald_simulation.simulate`
builds the reference cell (HeRALD_v1, 24 CPDs, ER 1 keV, `TES_HERALD_V1`) and
thirteen one-factor cells — geometry (24 → 1, 2-sensor split CPD),
Σ-covariance (bath correlation ↑, low-rank pickup modes, SQUID 1/f knee ×10,
mains ×8), Σ-structural (sensor loss, gain drift, timing jitter applied to
signal *and* noise), event (NR at the same energy; ER at ½× and 2×), and one
undeclared family (WIMP spectrum, 500 MeV). Every cell reuses the same
`event_id` list; output is `truth.parquet` + `traces.npy (n, C, N)` +
`provenance.json` (HeST commit, geometry hash and positions, budget with
provenance states, trace config). Cost: a 1 keV NR is ~9×10⁵ quasiparticles ≈
17 s single-core; `--qp-fraction` thins for development and records the
rescaling.

**Placeholder:** every constant in `HERALD_V1_PLACEHOLDER` except the two time
constants carries provenance state `placeholder`; they must be read from
arXiv:2307.11877 (`from_paper`) before any dataset is released.

**Two physics limitations to state honestly in the paper:**

- **ER/NR is degenerate with energy in a quasiparticle-only readout.** At 1 keV
  HeST gives 900 k QP for NR and 420 k for ER; the trace sees only the yield
  and the arrival-time distribution. So "signal type" in arm B is a
  *fixed-energy* contrast unless the photon channels (IR, singlet UV, triplet)
  are also transported to a trace. `events.quanta` records all four yields;
  `qp_simulator` transports only the quasiparticles. Adding a photon channel is
  the one physics extension that would make type recognition non-degenerate.
- **No background model.** HeST simulates a deposit, not a rate; families are
  designed, not sampled from a background.

---

## 2. What the autoencoder should look like

The question is not which encoder is most accurate; it is which structure makes
**"which physical factor moved"** a readable quantity. The linear results of
6 September (`RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md`) settle two things
the design must respect: an acquisition change shows in noise-only records and
a physics change cannot; and for the linear subject a geometry change repairs
through the head, so the geometry-invariance of the pooled latent is a
*prediction about nonlinear subjects*, not a fact. Everything below is built
on those.

### 2.1 Design principle: put the factors in different places by construction

Do not ask an unstructured latent to disentangle noise, position, amplitude and
shape on its own; unsupervised disentanglement is not identifiable (Locatello
et al. 2019). What *is* identifiable is disentanglement from **pairs that
differ in exactly one factor** (Locatello et al. 2020; Shu et al. 2020) — and
that is precisely what the one-factor cells and the exact event pairing
produce. The 控制变量 discipline is the identifiability condition. So the
architecture should (i) reserve a slot per factor, (ii) feed the factors we
*know* (geometry, Σ̂) as conditioning rather than asking the encoder to infer
them, and (iii) train with the paired swaps the simulators give for free.

### 2.2 The recommended shape

```
inputs   x  (C × N) traces      g_c  per-channel geometry (position, normal, area, group/crate id)      Σ̂
                                                                                                         │
[W]   x̃ = Σ̂^{-1/2} x            Kronecker Σ̂ = Σ_c ⊗ Circ(S(f)); a NAMED, replaceable parameter  ◄───────┘
 │
[S1]  h_c = f(x̃_c)              per-channel 1-D conv / small transformer, weights shared over channels
 │                               (the same f is run on random-trigger records → the noise branch, below)
[P]   tok_c = [h_c ‖ e(g_c)]     geometry enters HERE, as a feature — never as a channel index
 │
[S2]  z_sig = pool({tok_c})      permutation- and channel-count-invariant attention pooling with a
 │                               geometry-relative bias; trained with random channel masking
 │       z_sig = [ z_amp | z_pos | z_shape ]        three sub-blocks with their own probe heads
 │
[g]   x̂_c(t) = a_c · s(t − τ_c ; z_shape)          STRUCTURED decoder: a_c = A(z_amp, z_pos, g_c),
 │            + r_c(t; z_free)                      τ_c = T(z_pos, g_c), template s from z_shape;
 │                                                  small free residual branch r, its energy monitored
[q]   z_noise = q( PSD/CSD summary of Σ̂^{-1/2}(x − x̂) , random-trigger records )
 │
[o]   y = o(z_sig)              physics targets (E, vertex, ER/NR or particle)
      σ = o_Σ(z_noise)          Σ-state targets (line amplitudes, correlation, knee)
```

Why each piece is where it is:

- **Geometry is conditioning, not content.** If `g_c` enters at [P] and again
  in the decoder [g], then a paired event seen by two detectors should map to
  the *same* `z_sig`. That gives the single most powerful training signal the
  simulators offer: **cross-geometry reconstruction** — encode under detector
  a, decode under detector b's `g_c`, compare with the paired truth under b.
  It forces geometry out of `z_sig` and into the conditioning path, which is
  exactly the state the G-row of the lookup assumes. On the linear subject
  this is what a geometry-weighted pooling cannot do and the head has to; on
  the transformer it is the prediction under test.
- **Noise is a residual, not a latent of the signal.** Whitening in front and
  reconstruction in the whitened domain make Σ̂ a property of the loss. The
  noise state lives in a *separate* branch `q` fed by whitened residuals and
  by random-trigger records — the two things the 6 September table showed
  are the N-vs-S discriminator. Under Σ̂ = Σ the residual is white and `z_noise`
  sits at its reference; a covariance change moves `z_noise` and nothing in
  `z_sig`; a physics change moves `z_sig` and leaves `z_noise` at reference.
  The diagnostic is then a block-wise displacement, not a projector fitted
  after the fact.
- **Position is read out geometrically.** `τ_c = T(z_pos, g_c)` and
  `a_c = A(z_amp, z_pos, g_c)` are the analysis-by-synthesis form of
  triangulation and solid-angle sharing. A vertex encoded this way transfers
  across sensor counts because the decoder knows where each sensor *is*; a
  vertex encoded as a learned pattern over channel indices does not. The
  geometry-relative attention bias in [S2] (attend by distance/angle between
  sensor and the current vertex estimate) is the encoder-side twin of the
  same idea.
- **Shape is a template, amplitude a scalar.** The physical trace is
  `Σ_k A_k h_c s_k(t − τ_c)` plus noise; making the decoder that shape ties
  `z_shape` to *what* happened (arrival-time distribution ⊗ sensor response
  in HeRALD; ring timing/charge pattern in LUCiD) and `z_amp` to *how much*.
  A small free residual branch `r` catches what the structure cannot, and its
  energy fraction is itself a diagnostic: it rises only for supported-but-rare
  or undeclared physics (the `event_in_span` and abstain rows).
- **The analytic baseline is the same diagram with linear pieces.** Paper 1's
  tied linear autoencoder (OF / EMPCA) is [W]+[S1]+[S2]+[g] with `f, g`
  linear and `A, T` fixed by the template; `src/latent_monitor/linear_subject.py`
  already is that. Every signature is checked there first; the compact
  transformer (`src/reconstruction_model/`, `torch_subject.py`) is the
  nonlinear instance, and `models/pairwise_channel_masking.py` is the
  masked-channel training already in the tree.

### 2.3 Training objective (reference cell only, then freeze)

`L = ‖Σ̂^{-1/2}(x − x̂)‖²` (whitened reconstruction) `+ λ_G ‖Σ̂^{-1/2}(x_b − ĝ(z_sig(x_a), g_b))‖²`
(cross-geometry paired reconstruction, a/b two detectors of the same event)
`+ λ_p Σ_blocks L_probe` (block-wise supervision: energy → `z_amp`, vertex →
`z_pos`, type → `z_shape`, Σ-parameters → `z_noise`; small weights, one head
per block, no head sees another block) `+ λ_s L_swap` (swap `z_noise` between
two events: the decoded *clean* signal must not change; swap `z_shape` at
fixed energy: the decoded amplitude must not change) `+` random channel
masking on every batch. The model is trained on the reference cell only and
frozen; every other cell is evaluated against its paired clean twin, per
event.

### 2.4 What each factor does to this latent, and what to do about it

| factor moved | where it shows | statistic that names it | repair |
|---|---|---|---|
| Σ — covariance-type (correlation, line, knee, alias fold) | `z_noise` only; residual variance along excited directions ≠ 1 | `‖Δz_noise‖` off null, `‖Δz_sig‖` at null, noise-only PSD/CSD deviation | set Σ̂ ← Σ in [W] and **re-derive** [S1]/[S2] as the GLS projection (the whitening lemma, not a layer swap — 6 Sept result) |
| Σ — structural (gain drift, channel loss, jitter) | `z_noise` *and* a mean shift in `z_amp`/`z_pos`; layer profile peaks at [S1] | noise-only records move **and** `Δz_sig` has a consistent direction | activation-patch [S1]; if consequence recovers, LoRA on [S1] only |
| G — geometry | should be *small* in `z_sig` (cross-geometry loss is what makes it so); large at [P] tokens | cross-geometry reconstruction error; energy split at [P] vs `z_sig` | LoRA on `e(·)` and [S2] only; on a linear subject, refit the head |
| E — in-span rare physics (double pulse, oscillation) | `z_shape` moves a lot; residual branch energy up; noise branch at reference | out-of-span fraction low, `r` energy high, consequence up | recalibrate/extend the output head |
| E — out-of-span / undeclared (WIMP spectrum, glitch) | displacement in the unexcited complement; Fisher-rank drop | out-of-span fraction high, Mahalanobis novelty in the reference-cell metric | **abstain**; extend training support |
| designed output-null | large `‖Δz‖`, ≈ 0 output change | `P_null` energy ≈ 1 | none — the control |

The lookup in `src/latent_monitor/lookup.py` already encodes this order; the
structured latent makes the first column a block index instead of a fitted
projector, which is both cheaper and pre-registrable.

### 2.5 Per-arm meaning of "position", "shape", "type"

| | LUCiD | HeRALD |
|---|---|---|
| position | vertex / flasher position from first-light timing and charge pattern over PMTs | vertex (x, y, z) from the per-CPD share and the arrival-time distribution — weak along z, strong in the array plane |
| shape | ring vs blob timing structure; time-of-first-light per PMT | arrival-time distribution ⊗ single-QP template; nearly type-independent |
| type | particle species (μ / e / π⁰ via PhotonSim), material (water / WbLS) | ER vs NR **at fixed energy only** (§1.2); photon channels needed otherwise |
| amplitude | photons / pe per event | quasiparticle yield |

### 2.6 Candidate third subject — frozen-propagator hypergraph (D10, design and interface only)

Alongside the transformer, the study records a second **nonlinear** subject
(`latent_monitor/hypergraph_subject.py`) whose latent stays readable because
every propagator weight has a provenance:

- **vertices = channels** (TES/QP channels on arm B; band-frames on TIDMAD);
- **P2** from the measured noise covariance — the noise-Laplacian PE of the
  companion paper (cited by the name *Prop. stationary-pe* only); adjacency is
  the partial-correlation (precision) matrix and the propagator is the
  symmetric normalised Laplacian `L = I − D^{−1/2} A D^{−1/2}` (Eq. 4.3);
- **P3** from the measured third noise cumulant, with the sign carried as an
  **edge attribute** and `|w|` entering the order-3 Laplacian;
- **P2 and P3 are frozen from the reference cell**; only a small readout is
  trainable. The optional trainable correction is QUIVER's zero-initialised
  residual multiplicative gate (`arXiv:2606.02785` Eq. 8: `x → (1 + αΘ)x`,
  α = 0 at init), so the subject is exactly the tied linear AE at step 0.

P3 vanishes identically for Gaussian noise, so the subject is non-trivial only
on non-Gaussian cells — which is where the whitened-residual cumulant statistic
of `EXPERIMENT_DESIGN.md` §V.2 is also non-trivial. This is **one of two
nonlinear subjects; both are to be reported and neither is chosen by result**.
It is **gated on the trained Tier-1 run**: this pass builds the `Subject`
interface and the Eq. 4.3 / Gaussian-zero Laplacian tests, with no training run,
no GPU run and no result.

---

## 3. Other simulations that would serve — and those that would not

✅ Licences, geometry mechanisms, outputs and activity verified upstream today.
"Fit" is against §0: physics controllable / geometry controllable / clean
sampled trace with no incumbent noise model / exact pairing / licence.

### 3.1 Would serve (ranked)

| # | simulator | domain | licence | geometry | trace | own noise | pairing | why it would serve | cost / catch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **NuRadioMC / NuRadioReco** `nu-radio/NuRadioMC` | in-ice / in-air radio neutrino detection (RNO-G, ARIANNA, ARA) | GPL-3.0 | **JSON detector description**: stations/channels with position, orientation, antenna type, amplifier, cable delay, per-channel sampling rate | per-channel **voltage traces** at a configurable rate | `channelGenericNoiseAdder` (independent) **and** `channelGalacticNoiseAdder` (coherent across channels) — both switchable off | event list pre-generated to HDF5 independent of the detector; `config['seed']` seeds everything downstream → exact pairing across any detector JSON | the best functional match on every axis: physics (flavour, energy, vertex, direction, Askaryan model), geometry by file, dense traces, seeds; a third readout physics (GHz antennas) | pure Python, `pip install`; GPL matters only if our adapter is distributed as a derivative — outputs (datasets) are not covered; 2026-09-02 last commit |
| 2 | **SolidStateDetectors.jl + LegendGeSim.jl** `JuliaPhysics/SolidStateDetectors.jl` | HPGe (LEGEND) | MIT | **YAML/JSON CSG** with arbitrary contact segmentation | one waveform per contact at Δt = 1 ns | none in SSD; LegendGeSim adds preamp + `NoiseFromData` per channel, optional | Geant4.jl hits table saved and replayed into another detector config → exact | permissive; clean physics-driven pulse *shapes* (single- vs multi-site) — the strongest "signal shape/type" axis of any candidate; DELight-adjacent (cryogenic crystal, pulse-shape discrimination) | Julia, not Python; channel count is contacts of one crystal (≤ tens), so "geometry" means segmentation, not array size |
| 3 | **Allpix²** `allpix-squared/allpix-squared` | semiconductor pixel / strip | MIT | `.conf` geometry + detector-model files (pitch, matrix, thickness, any number of detectors) | per-pixel time-binned induced-charge pulses (`TransientPropagation` + `CSADigitizer`) | Gaussian per bin, independent, switchable | global `random_seed` | permissive, actively maintained (2026-09-09), many channels, config-only geometry | Geant4 build; pulses are short (integration ~500 ns), so the covariance record-length rule (N/C ≳ 500) bites |
| 4 | **larnd-sim** `DUNE/larnd-sim` (light channel) | pixel LArTPC (DUNE ND) | Apache-2.0 | YAML pixel layouts + detector properties; light geometry needs a new Geant4 LUT | light: `(n_triggers, n_channels, n_samples)` SiPM waveforms at 100 MS/s; charge: sparse packets, not traces | per-channel **measured noise spectrum**, independent (replaceable) | `rand_seed` + same edep-sim input | permissive, real collaboration code, per-SiPM sampled traces with a spectral noise model we can swap for a correlated one | **CUDA GPU mandatory**; light-geometry change is not config-only |
| 5 | **Wire-Cell toolkit** `WireCell/wire-cell-toolkit` | wire LArTPC (MicroBooNE, ProtoDUNE, SBND) | LGPL-3.0 | JSON wires schema + Jsonnet graph; parametric generation in `wire-cell-python` | per-channel ADC waveforms (tick ~0.5 µs, `nticks` configurable) | `EmpiricalNoiseModel` (independent, spectral) + `GroupNoiseModel`/`CoherentAddNoise` (group-coherent) — an **incumbent** correlated model, already cited as prior art | `Gen::Random` seeds + depo files | functionally excellent and the only candidate whose noise model is itself a published, measured coherent model — useful as the *foil* for ours | C++17/Jsonnet stack; LGPL; the incumbent noise model must be disabled, and then we are competing with it |
| 6 | **ratpac-two** `rat-pac/ratpac-two` | Geant4 PMT arrays (scintillator / water / WbLS; Eos, Theia) | GPL-3.0 | RATDB `.geo`/`.ratdb` tables incl. `PMTINFO` position arrays — no C++ for new layouts | per-PMT digitised waveforms (V1730: 0.5 GS/s 512 samples; V1742: 5 GS/s) | white Gaussian per sample, independent + dark rate; switchable | Geant4 seeds | the closest "PMT array with real waveforms and text geometry" analogue to LUCiD, with a licence | Geant4 + ROOT build (Docker exists); GPL |

**Recommendation for a third simulated arm, if one is ever needed:**
NuRadioMC if the goal is *another readout physics with many channels and a
native coherent-noise foil*; SolidStateDetectors.jl if the goal is *a
permissive, cheap, pulse-shape-rich arm* that speaks to cryogenic crystal
detectors. Neither is needed for the first stage.

### 3.2 Would not serve (kept for the record)

| simulator | why not |
|---|---|
| **Prometheus** `Harvard-Neutrino/prometheus` (LGPL-2.1) | geometry is a flat `.geo` text file (the easiest of all) and seeds are clean, but output is per-module photon hits; the new `fadc_digitization.py` returns *sparse non-zero* FADC bins, not a dense trace, and no electronic noise. Keeps its role as the frozen-public-model (DynEdge) arm; cannot carry the covariance claim **as is** — with a waveform bridge it can (§3.3), which makes it the fallback for arm A. |
| **WCSim** (MIT since `develop`) | digitised hits (time, charge) only; no waveform layer; new layouts need C++. LUCiD already covers the domain with traces. |
| **XENONnT fuse + NEST** (BSD-3 / Apache-2) | emits real per-PMT `raw_records`, but geometry is baked into collaboration pattern-map resources; a different geometry means new Geant4 optical maps. |
| **G4CMP** (GPL-3) | phonon/charge transport in Ge, Si, sapphire, CaWO₄ — the most DELight-relevant physics on the list — but hits only, geometry in C++, Geant4 10.4–10.7 pinned. HeST + `qp_simulator` already do the job for helium; revisit only if a *crystal* arm is wanted. |
| **pytessim** `spice-herald/pytessim` (MIT, © 2026 TESSERACT) | not a physics simulator: PSD/CSD "salting" generator, ≤ 2 channels, no geometry. Duplicates our noise generator; useful only as an independent cross-check of Σ sampling. |
| **ldmx-sw** | 10 samples × 25 ns records — a six-bin spectrum; κ not estimable (survey §(b)). Architecture-transfer arm at most. |
| **differentiable larnd-sim** `ynashed/larnd-sim` | charge chain only, no traces, dormant since 2024-11. |

### 3.3 Prometheus as a replacement — feasibility per recognised factor

_11 September 2026. Answers "if we replace HeST + our noise module with Prometheus, can
the model still recognise detector geometry, noise type and signal/physics type?"
Prometheus is read at the vendored commit `8c19938` (28 Aug 2026); "bridge" means the
`qp_simulator` pattern applied to Prometheus — histogram per-PMT photon times on a fixed
window, convolve with an SPE template, add `noise_module` per crate. ✅ = feasible as the
programme defines it; ⚠ = feasible with a stated limitation; ✘ = not feasible._

| recognised factor | **Prometheus alone** (hits / sparse FADC) | **Prometheus + bridge + `noise_module`** | **LUCiD + `noise_module` V2** (arm A) | **HeST → `qp_simulator` → `noise_module`** (arm B) |
|---|---|---|---|---|
| **geometry** | ✅ `.geo` text file, 6 shipped (ORCA, ARCA, IceCube, GVD, P-ONE, TRIDENT); exact pairing by replaying one injection file | ✅ same | ✅ one JSON file, 16 configs; exact pairing by PRNG key (`apply_translation=False`) | ✅ `make_cell(...)`, 24 → 1 on the identical cell; pairing tested |
| **physics / signal type** | ✅ **strongest**: ν flavour, CC/NC, track vs cascade, E, direction, vertex from LeptonInjector; angular error is a real consequence variable | ✅ same labels, now on a trace | ⚠ flasher / PhotonSim μ, e, π⁰ / SIREN μ, e gun; material (water ↔ WbLS); no neutrino kinematics, consequence variable still ours to define | ⚠ ER vs NR **degenerate with energy** at fixed E (quasiparticle-only readout); WIMP spectrum as held-out U |
| **noise type — covariance Σ** (κ, whitening lemma, `z_noise` branch) | ✘ no electronic noise; only rate/nuisance families (dark rate, QE, TTS, jitter, dead modules, correlated ⁴⁰K bursts); no Σ̂, no realised Σ | ✅ Σ̂ and Σ recorded per crate by `noise_module`; the digitiser is then ours (declare it) | ✅ V2: `filtered` front end, `spectral_shared_private` crate, ρ_ij(f) reported | ✅ per cell, `kappa_floor` in `truth.parquet`; TES budget constants placeholder |
| **noise type — structural** (gain drift, channel loss, jitter, alias fold) | ⚠ hit-level only: N1–N5 in `prometheus_simulation.interventions` | ✅ trace-level families incl. alias fold and quantisation | ✅ | ✅ |
| **readout physics** | PMT array, hit level | PMT array, 1–3.3 ns trace (hypothetical: real DOMs read ATWD/FADC at 300/40 MS/s or ToT) | PMT array, 1 GHz trace (LUCiD's convention) | TES calorimeter, 250 kHz — the second readout physics and the DELight link |
| **granularity axis** (same volume, different sensor count) | ⚠ six real geometries differ in volume too; no dense ↔ sparse of the *same* volume shipped | ⚠ same | ✅ `n_sensors` 2 000 → 20 000 (planned) | ✅ 24 → 1 |
| **licence / release** | ✅ LGPL-2.1; outputs not covered | ✅ | ✘ **gate A0**: no licence as of today | ⚠ **gate B0**: MIT text, PyPA sample copyright line |
| **cost to reach the trace** | — | ~ the LUCiD units bridge + a window convention (days) | notebook exists; `noise_module_lucid` ~8 days once A0 lands | built |
| **status today** | built, licence-clean, DynEdge frozen-public-model arm | not started | notebook only | 14 cells, tests green |

**Reading the table.** Prometheus *alone* keeps geometry and physics-type recognition and
loses the one the paper is about — there is no covariance to recognise. Prometheus *with the
bridge* keeps all three and is licence-clean, so it is the natural **replacement for arm A
(LUCiD) if gate A0 never lands**, with better physics labels than LUCiD. It is not a
replacement for arm B: swapping HeST for Prometheus would leave two PMT arrays, and drop the
24 → 1 granularity dissociation, the second readout physics, and the DELight relevance. The
caveat to state in either case: a dense 1–3.3 ns trace from a KM3NeT/IceCube-style module is a
hypothetical readout, so the arm would carry waveform-level claims under a declared digitiser
contract, exactly as arm A does.

---

## 4. Real-experiment datasets that would serve

✅ Release pages, DOIs, licences and contents verified today. Properties: **T**
sampled per-channel traces · **N** noise-only records or a published
PSD/ASD · **L** labels for signal type · **V** configuration/geometry
variation inside the release · **D** permissive licence + DOI.

### 4.1 Would serve (ranked)

| # | dataset | what it is | T | N | L | V | D | why it serves | catch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **CRESST-II/III pulse-shape data** (ORIGINS Dark Matter Data Center; arXiv:2508.03078) | 68 TES detectors, runs 33–35, 979 k train + 78 k test records, raw voltage traces downsampled to 512 samples (from 8 192 / 16 384 at 25 kHz), **random-trigger noise traces included and flagged**, `detectors.csv` per channel, four released models | ✔ | ✔ | ~ binary clean/artifact | ✔ 68 detectors, 3 runs, 2 record lengths | ? licence and DOI not stated | **the closest real analogue of arm B**: TES traces, per-detector noise records (Σ per detector, unchosen), detector-to-detector variation is a real "geometry" axis; DELight-relevant | 16–32× downsampling limits the spectral band; no particle-type label; confirm licence/DOI with the DMDC before use |
| 2 | **LIGO/Virgo GWOSC strain + auxiliary channels + Gravity Spy + GWTC PSDs** | strain O1–O4b (4/16 kHz, CC BY 4.0, DOIs e.g. O4b 10.7935/8emv-ag54); **auxiliary multi-channel releases** (O3: 40 H1 + 46 L1 channels, 13 TB; GW170814: ~500 channels/site); Gravity Spy glitch classes (22/24) with GPS times (10.5281/zenodo.5649212); per-event PSDs inside GWTC PE releases | ✔ | ✔ published ASD/PSD — the assumed covariance is literally public | ✔ glitch classes by GPS lookup | ✔ H1/L1/V1, hardware epochs O1→O4b | ✔ | the only real dataset where Σ̂ is *published* and Σ̂ ≠ Σ events are *labelled* (glitches) with ground truth; multi-channel aux data gives real cross-channel coherence | domain entry 2–3 weeks; glitch traces must be cut from bulk strain; volunteer-label file is CC BY-NC-ND |
| 3 | **Majorana Demonstrator AI/ML release** (10.5281/zenodo.8257027; arXiv:2308.10856) | 3.19 M ²²⁸Th calibration events, raw HPGe waveforms (3 800 samples, hybrid-sampled), 56 PPC detectors in two modules, labels: energy, AvsE (single- vs multi-site), DCR (surface α), LQ; 75/20/5 splits | ✔ | ~ pre-trigger baseline only | ✔ **pulse-shape types** | ~ 56 detectors, 2 modules | ~ DOI yes, licence informal | best labelled signal-shape/type set; pairs naturally with SolidStateDetectors.jl (§3.1 #2) as its simulation twin | no noise-only runs; one data set (DS6) |
| 4 | **Pierre Auger Open Data 2024** (10.5281/zenodo.10488964, CC BY-SA 4.0) | 81 k showers; per-PMT **FADC traces, 768 bins × 25 ns**, for SD-1500 and SD-750 stations; station positions and operational periods; FD pixels; weather; scalers | ✔ | ~ in-trace baselines only | ~ reconstructed E, X_max, zenith | ✔ two array spacings, 2004–2018 | ✔ (share-alike) | real multi-station traces with two documented station configurations | no random-trigger records; SSD/RD not included |
| 5 | **MicroBooNE open samples** (Zenodo 7262009 / 7262140 / 8370883 / 7261921, CC BY 4.0) | simulated ν **overlaid on real off-beam data**: per-wire ADC, 6 400 ticks, PMT OpHits, Geant4 truth, Pandora reco | ✔ | ~ noise is real (cosmic overlay) but no noise-only files; noise paper arXiv:1705.07341 | ✔ truth | ✘ one detector, one period | ✔ | the real-noise counterpart of the Wire-Cell simulator; the measured coherent-noise model is published | whether the wire table is raw or noise-filtered is not stated in the docs |
| 6 | **SuperCDMS R76 (NEXUS-DM / NSDF)** (arXiv:2507.13297) | above-ground Ge detector, Mar–Dec 2022, raw per-channel phonon traces (6 channels), metadata: trigger type, source (²²Na, PuBe, ²⁴¹Am), shielding configuration | ✔ | ~ "trigger type" suggests randoms | ✔ source/config | ~ shielding configs | ? no DOI/licence located | raw phonon traces with configuration labels | provenance gaps; confirm before depending on it |
| — | **TIDMAD** (arXiv:2406.04378 v3; CC BY 4.0; 10.5281/zenodo.11458076) | already arm C: 20 train + 20 val files with injected sinusoids (ch0 reference, ch1–2 SQUID), **208 science files with no injection = noise-only**, 10 MS/s, 8-bit | ✔ | ✔ | ~ injected sinusoids only | ✘ | ✔ | unchanged role: two Σ̂ trainings on one unchosen Σ | denoising score disclaimed by its authors (D2); `K_rel` must survive |

**Recommendation for the real-data side.** Keep TIDMAD as arm C. If a second
real arm is added, **CRESST first** — it is the real twin of arm B (TES traces,
random-trigger noise per detector, 68 detectors as the configuration axis) and
it is the dataset a DELight reader will recognise — with LIGO/Gravity Spy as
the claim-bearing alternative if a *published* Σ̂ with labelled Σ̂ ≠ Σ events is
required (the descope note in the arms plan §5.3 already names it). Majorana
is the dataset to use the day "signal type" becomes the headline.

### 4.2 Would not serve (checked, for the record)

| dataset | why not |
|---|---|
| IceCube public releases; Kaggle "Neutrinos in Deep Ice" | event/reconstruction level, or simulated pulse series; no waveforms. NuBench already covers the hit-level role. |
| KM3NeT Open Data Centre | one-week ORCA-4DU sample and supplementary event lists; no PMT waveform release; licences not stated. |
| XENON / LZ / PandaX | event lists, efficiencies, likelihoods; no waveforms (confirmed). |
| LEGEND-200, GERDA | no public waveform release found (LEGEND-200 first results arXiv:2509.21166 has none). |
| CUORE, EDELWEISS, NUCLEUS, COSINUS, DELight, QROCODILE, BREAD, Ricochet, MINER | no raw-trace releases located (absence not proven); EXCESS repository is spectra only. |
| RNO-G, ARA, ARIANNA, LOFAR-CR, AERA, GRAND | no real-waveform release; GRAND's ML set (10.5281/zenodo.18233878) is simulated with synthetic Galactic-scaled noise. |

---

## 5. The programme, as it stands

| tier / arm | substrate | status today | carries |
|---|---|---|---|
| Tier 1 — ORACLE-Cov | `noise_module` synthetic | linear-subject table done (13/14) | the whitening lemma; κ sweep; designed dissociation |
| **Arm B — HeST → `qp_simulator` → `noise_module`** | simulated TES, 1–24 ch, 250 kHz | **built**, 14 cells, tests green; constants placeholder; gate B0 open | granularity dissociation 24 → 1; second readout physics |
| **Arm A — LUCiD + `noise_module`** | simulated PMT, ~2 400 ch, 1 GHz | **notebook only**; gate A0 open (no licence as of today) | transfer of the κ prediction into a real geometry; waveform-level C1/C2 |
| Prometheus / DynEdge | hit-level | built, licence-clean | frozen public model; angular-error consequence; fallback for A |
| Arm C — TIDMAD | real SQUID | code exists, needs data + GPU | external validity |
| candidates | NuRadioMC · SSD.jl · CRESST · GWOSC/Gravity Spy · Majorana | surveyed, not started | held in reserve (§3.1, §4.1) |

### Actions this document implies

1. Send the two gate emails (A0 to Terao/Alterkait; B0 to Rischbieter) — both
   re-verified open today; nothing else on arm A moves before A0.
2. Arm B: replace `HERALD_V1_PLACEHOLDER` with values from arXiv:2307.11877;
   decide whether to transport a photon channel so ER/NR is not
   energy-degenerate (§1.2).
3. Architecture: implement §2.2 as the transformer `Subject` — cross-geometry
   paired loss, the `z_noise` branch on residuals and random triggers, the
   structured decoder — and run the §2.4 table on it; the linear subject
   already passes.
4. Write to the ORIGINS DMDC about the CRESST licence/DOI before any plan
   depends on it.
5. Keep this file canonical: when an arm changes status, edit the tables here
   and cite the evidence file, rather than adding another survey.
