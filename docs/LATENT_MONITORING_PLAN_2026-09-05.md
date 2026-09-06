# Controlled-variable latent monitoring — architecture, diagnosis, adjustment, and the per-arm build

> **Implemented 6 September 2026** — `src/latent_monitor/` (§§2–4), `src/herald_simulation/`
> (§7), `noise_module.tes_budget`, WP-N1. Results and the corrections the table forced
> are in `RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md`; read §1 of this plan with that note.

_5 September 2026. The 控制变量 plan: hold everything fixed, move **one** of
three factors — geometry **G**, noise **Σ**, event type **E** — and watch the
frozen model's latent space. This document answers four questions: which
architecture, how to drive it, how to tell *which part* of the latent space
moved, and what to adjust once you know. Then it says how each arm is used
under that protocol, how `noise_module` is integrated into LUCiD, and how the
HeST fork is built and its noise made HeRALD-shaped. It sits on
`EXPERIMENT_DESIGN.md` (tiers), `EXPERIMENT_PLAN_ARMS_2026-09-05.md` (arms) and
the proposal's §3 (mechanism); it does not restate them._

---

## 1. The design in one table

The proposal's §3 says a frozen model's representation is fixed by three
determinants — the **metric** Σ⁻¹ the measurement fixes, the **class** F the
architecture fixes, and the **support** T_S the training excites. The
controlled-variable design is that table read backwards: each factor we move
is one determinant, each determinant has a predicted latent signature, and
each signature names its own repair.

| factor moved | determinant | where it shows in the latent space (prediction) | the adjustment that follows |
|---|---|---|---|
| **Σ** — noise type, Σ̂ ≠ Σ, covariance-type | metric Σ⁻¹ | **no mean shift**; the residual variance along the excited directions departs from 1, scaled by the eigenvalues of Σ̂⁻¹Σ; magnitude tracks κ(Σ̂⁻¹Σ) | re-estimate Σ, replace Σ̂ in the whitening layer; **no gradient step**, decoder untouched |
| **Σ** — structural N (timing jitter, channel loss, gain drift) | metric, but seen through the channel stage | a **mean** shift in the output-aligned subspace — looks like E in the mean; distinguished by the variance signature and by a layer profile peaking at the per-channel stage | activation-patch stage *k* first; if consequence recovers, LoRA on the per-channel encoder only |
| **G** — geometry | class F, through the geometry embedding | mean shift concentrated at the **geometry-embedding stage**, *small* at the pooled z if the aggregation is geometry-invariant — that smallness is itself the prediction under test | LoRA on the geometry embedding + pooling only; everything else frozen |
| **E** — event type outside training support | support T_S | displacement in the **unexcited complement** T_S^⊥; Fisher-rank drop | **no weight adjustment is valid** — abstain, then extend the training support with data |
| designed output-null family | — | large Euclidean ‖Δz‖, ≈0 output change | none needed; this is the control that separates alarm from consequence |

Two nuances the Tier-1 dev runs (`DEV_UPDATE_2026-09-03.md`) already forced:
N is *not* one signature — covariance-type N is a variance story and
structural N is a mean story; and only t₀ and non-separable shifts land in
the complement. The table above carries both.

---

## 2. Architecture

### 2.1 The constraint the design imposes — and it is the whole answer

If geometry changes and the model must not be retrained, then the latent
space has to be **the same space** for every geometry. That rules out any
encoder whose input layer is tied to a channel count — a flat MLP or 2-D CNN
over `(n_sensors × n_bins)` gives a *different* z-space per geometry, and
"which part of the latent space moved" is then not a question. The
architecture must be a **set encoder over channels, with geometry as an
explicit per-channel feature, pooled to a fixed-dimension z**.

That is what the project already has. DynEdge (Prometheus) is a graph
encoder over hits with positions as features. The compact two-stage
transformer (`src/reconstruction_model/`, `src/tidmad_transformer/`) is a
per-channel stage followed by an aggregation stage. **The recommendation is
not a new architecture; it is to make the four latent-facing pieces explicit
and identical across arms**, so that every diagnostic in §3 is the same code
on every arm.

### 2.2 The recommended shape

```
x  (C channels × N samples)                       raw traces
 │
 ├─ [W]  whitening layer  x̃ = Σ̂^{-1/2} x           Σ̂ is a NAMED PARAMETER, not baked in
 │
 ├─ [S1] per-channel encoder  h_c = f(x̃_c)          shared weights across channels;
 │        1-D conv or small transformer over time    this is where structural N shows
 │
 ├─ [P]  channel token = [h_c ‖ e(pos_c, group_c)]  geometry embedding e(·): position,
 │                                                    orientation, crate/string id
 ├─ [S2] geometry-aware aggregation                 attention pooling or EdgeConv on
 │        z = pool({token_c})   z ∈ R^d, d fixed     positions; permutation-invariant,
 │                                                    channel-count-invariant
 ├─ [g]  decoder  x̂ = g(z)                          reconstruction head, whitened domain
 │                                                    — makes the inverse-PSD loss literal
 └─ [y]  output head  y = o(z)                      physics targets (E, vertex, direction /
                                                     E, position, ER-vs-NR)
```

Training objective: `‖Σ̂^{-1/2}(x − g(z))‖² + λ·L_phys(y)`. The first term is
what makes Σ̂ a property of the loss rather than of the data.

**Hook points** (six, matching D4): `x̃`, `h_c`, post-embedding tokens,
`z`, decoder pre-output, `y`. Everything in §3 is computed at these.

**Exposed Jacobians**: `J_g = ∂g/∂z` (recon), `J_o = ∂y/∂z` (output). These
are the `Subject` interface (`represent`, `outputs`, `jac_recon`,
`jac_output`) already defined in `oracle_cov/subjects.py`; each arm's model is
a `Subject`.

### 2.3 Analytic baseline first

Paper 1's tied-linear autoencoder is the same diagram with linear f, g, o.
Every statement in §1 is provable there, and the excited span T_S is exact,
not local. Run the full §3 protocol on the linear subject before the
transformer, on every arm. If a signature fails on the linear subject, it is
wrong; if it holds there and fails on the transformer, that is a finding.

### 2.4 Training protocol — the controlled-variable discipline

Train on the **reference cell only**: (G₀, Σ̂₀ = Σ₀, E₀ ∈ S). Freeze. The
model never sees a perturbed cell. Every perturbed cell is evaluated against
its **paired clean twin** — the same event through the reference cell — so Δz
is a **per-event vector**, not a distributional distance. This is what the
pairing machinery buys and why it was worth the effort: per-event Δz can be
projected, per-event; a distributional MMD cannot.

---

## 3. Recognising which part of the latent space changed

### 3.1 Four projectors, fitted once on the reference cell

At the reference cell, on held-out clean data:

| projector | built from | separates |
|---|---|---|
| **P_out / P_null** | row space of `J_o` (SVD, top-k right singular vectors) vs its orthogonal complement | output-aligned (consequential) vs output-null (harmless) directions of z |
| **P_exc / P_unexc** | eigenvectors of the pullback Fisher `I(z) = J_gᵀ Σ̂⁻¹ J_g` above vs below a rank threshold | excited support T_S vs its complement |
| **Π_ℓ** | per hook ℓ | the layer profile |

`k` and the rank threshold are fixed on the reference cell and pre-registered;
they are not tuned per family.

### 3.2 The statistics, per cell

For each perturbed cell with paired twins, per event: Δz = z(perturbed) −
z(twin). Then:

- **energy split**: `‖P Δz‖² / ‖Δz‖²` for each projector — *where* the
  displacement lives;
- **whitened residual variance** along the excited directions,
  `Var[eᵢᵀ Σ̂^{-1/2}(x − g(z))]` — departs from 1 under covariance-type N,
  stays at 1 under G and E;
- **Fisher rank** at the perturbed cell vs reference — drops under E;
- **layer profile** `‖Π_ℓ Δh‖` across the six hooks — peaks at S1 for
  structural N, at P for G, flat-then-late for E;
- **consequence** K: output error on the physics targets, and its
  conditional-on-alarm AUROC against the alarm magnitude (C4).

### 3.3 Attribution is a lookup, not a classifier

The pre-registration bridge (`IMPLEMENTATION_PLAN.md` §5) is the table in §1
with a decision rule per row that two readers score identically:

```
if  Fisher-rank drop  and  energy mostly in P_unexc          -> E  (support)   -> abstain
elif variance ratio departs from 1  and  no mean shift        -> Σ  covariance  -> re-whiten
elif mean shift in P_out  and  layer profile peaks at S1      -> Σ  structural  -> patch S1
elif mean shift  and  layer profile peaks at P, small at z    -> G  geometry    -> LoRA P/S2
elif large ‖Δz‖  and  ≈0 consequence                          -> output-null    -> no action
else                                                           -> abstain (undeclared)
```

The five-arm attribution classifier and the alibi-detect baselines (MMD,
C2ST, KS, embedding-mean distance) are run **beside** this, not instead of
it: they answer "did something change"; the lookup answers "which
determinant". That contrast is C2.

### 3.4 Abstention

Conformal on classifier margin does not detect undeclared families (Tier-1
dev: AUROC 0.25); a feature-space Mahalanobis nonconformity in the
reference-cell z-metric does (0.71). Use the Mahalanobis rule, and add the
Fisher-rank criterion as a second, mechanism-derived abstention trigger.

---

## 4. Adjustment — what to change once you know which part

The point of the table in §1 is that **the repair follows from the
diagnosis, and three of the four repairs are not gradient steps**.

| diagnosis | adjustment | what must be true afterwards (the test) | what must NOT be done |
|---|---|---|---|
| Σ, covariance-type | estimate Σ from noise-only records or residuals at the perturbed cell; set Σ̂ ← Σ in layer W | variance ratio returns to 1; K recovers; z mean unchanged | retraining anything — the decoder was never wrong |
| Σ, structural | activation-patch: substitute the *clean* S1 output into the perturbed forward pass. If K recovers, S1 is causal → LoRA on S1 with the twin pairs | patched K ≈ clean K before any LoRA is trained | LoRA on S2/g "because it helps" — that is the wrong-stage control, and it must repair *less* (C5) |
| G | LoRA on P + S2 only, on a small paired sample from the new geometry; W, S1, g, o frozen | layer profile flattens at P; z displacement shrinks; K recovers | touching S1 — the per-channel physics did not change |
| E, support | **none on weights.** Abstain; extend training support with data from the new family; retrain | Fisher rank restored after retraining | fine-tuning on the perturbed cell: outside T_S the representation is fixed by architecture and initialisation, not data (Paper 1 §7.4b), so weights fitted there fit noise |

Two honesty constraints, both from the audit: Hase et al. (2301.04213) show
localisation does not predict where editing works, so every stage-localised
repair is validated by **activation patching before LoRA**, never by LoRA
success alone; and "diagnosed-stage LoRA beats wrong-stage LoRA" is
Surgical Fine-Tuning's result (Lee et al.) — cite, and claim only the
instantiation.

---

## 5. How each arm is used under this protocol

One reference cell per arm; every other cell moves **one** factor and is
paired to the reference. The single permitted two-factor cell is G × Σ, as a
pre-registered robustness check, run last.

| arm | reference cell | G cells | Σ cells | E cells | U (abstain) |
|---|---|---|---|---|---|
| **Tier 1 ORACLE-Cov** | synthetic, Σ̂ = Σ | channel count 8 / 32 / 64 at fixed group structure | κ sweep; shared-private corr; low-rank latent; drift; artifacts | planted-factor families | held-out families |
| **LUCiD** | `SK_like` @ 11 000 placed, `PMT_FRONTEND_V1`, muon | `n_sensors` 4 000 and 20 000 (same cylinder); `MidBox` (box, same volume) | clock-line ×k; group coherence 0 → 0.6; alias fold 1 GHz → 250 MHz; per-channel gain drift; LUCiD dark rate | water → WbLS at fixed geometry; e⁻ and π⁰ via PhotonSim | supernova burst, pile-up |
| **HeST** | `HeRALD_v1` (24 ch), ER at fixed E, `TES_HERALD_V1` | `HeRALD_v1_monolithic` (1 ch), `HeRALD_UMass_splitCPD` (2 ch); `fill_height` | bath-fluctuation corr; vibration modes; SQUID 1/f knee; mains line | NR at the same E; the four yield channels | WIMP spectrum |
| **TIDMAD** | real noise, MSE loss | — | the *other* loss (inverse-PSD) — Σ̂ only | — | — |
| **Prometheus** | ORCA, DynEdge frozen | ARCA, +4 geofiles | NuBench §3.2 knobs (hit level) | S1–S5 | U1–U4 |

**What each arm proves in §1's table.** Tier 1 proves the table. LUCiD shows
the same rows hold when the channel stage is a photon-counting PMT array in a
real geometry, and adds the G row at scale. HeST shows the G row at the other
extreme — 24 → 1 channel makes the multichannel covariance claim vacuous by
construction, a designed contrast — and the E row on ER/NR, which is physics
nobody here wrote. TIDMAD shows the Σ row on noise nobody here generated.
Prometheus shows the G row on a model nobody here trained.

---

## 6. LUCiD — integrating and adjusting `noise_module`

The build is specified in `EXPERIMENT_PLAN_ARMS_2026-09-05.md` §7 (preset is
pure configuration; the units bridge; the N/C ≳ 500 rule; `freeze_channel_gains`).
What follows is the **integration into the protocol above**, which that
document does not cover.

### 6.1 Where it sits in the pipeline

```
PhotonSim ROOT (fixed event)
   └─ setup_event_simulator(geom, n_photons, is_data=True,
                            hit_mode='waveform', window_ns=32768, bin_width_ns=1.0)
        └─ charge_waveform (n_sensors, 32768)        [pe per ns bin]
             └─ noise_module_lucid.adapter.add_readout_noise(
                     charge_waveform, detector, preset, groups, rng)
                  └─ trace_mV, meta{Sigma_hat, Sigma, kappa, kappa_floor, groups}
                       └─ export (parquet; event_id, geometry_hash, preset_version)
```

`window_ns=32768` is not a default; it is the N/C rule for a 64-channel
group. Record it in provenance every time.

### 6.2 Σ̂ enters the model, not only the data

Layer **W** of the subject (§2.2) is initialised from `PMT_FRONTEND_V1`'s
one-sided PSD **and** its group covariance structure: Σ̂ = Σ_group ⊗ S(f) in
the block-diagonal-by-group approximation. That is the literal statement
"the encoder was trained under Σ̂". A Σ cell then realises a *different* Σ via
`MultiChannelNoiseGenerator` while W keeps Σ̂, and κ(Σ̂⁻¹Σ) is computed from
the two matrices the generator returns.

### 6.3 The Σ families, and what each does to the table

| family | generator knob | type | predicted row |
|---|---|---|---|
| clock-line amplitude ×k | `Line.scale` in the preset | covariance | variance ratio ≠ 1 at the line bins only — a *narrow-band* κ |
| group coherence | `corr_strength` 0 → 0.6 | covariance | off-diagonal Σ mismatch; κ grows with C |
| alias fold | decimate 1 GHz → 250 MHz, no anti-alias | covariance | broadband, predicted by `alias_fold_psd_density` in closed form |
| per-channel gain drift | `channel_gain_jitter` | structural | mean shift at S1 |
| dark rate | LUCiD's own `dark_rate_khz` | structural | mean shift at S1, sparse |
| channel loss | mask PMTs | structural | mean shift at P (tokens vanish) |

The first three are Σ̂ ≠ Σ in the strict sense and carry the C4 κ prediction;
the last three are N by contract but mean-shift in the latent, and the table
must say so.

### 6.4 Channel groups

`string_id` is already in LUCiD's output for string telescopes; for
cylinders, group by angular sector × height band (16–64 PMTs each) — this is a
stand-in for a crate. Declare it. The **group is the covariance unit**;
cross-group covariance is set to zero in Σ̂ and may be non-zero in Σ (a
low-rank global pickup) — that is a clean Σ family in its own right.

### 6.5 Acceptance for the integration (beyond the module's own)

- The subject trained on the reference cell reproduces the matched-cell κ
  floor (AC-2 of the arms plan) in its own residuals, i.e. layer W and the
  generator agree on Σ̂ to within the estimator floor.
- On the alias-fold cell, the whitened residual variance departs from 1 in
  the bins the closed form predicts, and nowhere else.

---

## 7. HeST — the fork, and HeRALD-shaped noise

### 7.1 What is confirmed about HeRALD

From arXiv:2307.11877: a **"transition-edge sensor based calorimeter"**
detects both the atomic (quantum-evaporation) signal and helium quasiparticle
excitations; energy threshold **145 eV at 5σ**; evaporation gain 0.15 ± 0.01.
TES readout implies a SQUID amplifier chain, but the specific TES parameters
(T_c, R_n, τ_eff), the SQUID noise floor and the baseline resolution are
**not** in the abstract and must be read from the paper before any constant
in the preset is fixed. Until then they are named placeholders.

### 7.2 Fork structure — mirror `prometheus_simulation`, leave HeST untouched

```
src/herald_simulation/
  external/HeST/            git submodule pinned to a commit (vendor once gate B0
                            resolves; keep LICENSE + a NOTICE naming the holder)
  README.md
  geometry.py               the five shipped builders + make_cell(cell_rad, fill_height,
                            sqcm_pitch, array_map) -> VDetector; geometry_hash()
  events.py                 quanta(E, "ER"|"NR"|WIMP) -> n_QP; evaporate(detector, n_QP,
                            vertex, seed) -> per-sensor arrival times; event_id = f(seed,
                            vertex, E, interaction) is the PAIRING KEY across geometries
  traces.py                 QPSimulator per sensor -> (C, N) clean ADC traces
  noise.py                  TES_HERALD_V1 preset (7.3) + MultiChannelNoiseGenerator;
                            returns Sigma_hat, Sigma, kappa, kappa_floor
  interventions.py          the Σ cells of §5 + structural N (sensor loss, gain drift)
  strata.py                 E cells: ER/NR at fixed E; the four yield channels; WIMP U
  matching.py               re-export prometheus_simulation.matching (content-matched twins)
  export.py                 parquet with event_id, geometry_hash, preset_version
  tests/
```

Rules for the fork: **no edits inside `external/HeST`**. Bugs found
(`get_QPEmap` returning `LCEmap_positions`, `Detection.py:243`; `setup.py`
pulling the unbuildable `aplus`; the PyPA copyright line) go upstream as
PRs. Everything of ours is in the adapter. That keeps "the physics is theirs"
true in the paper.

Pairing contract, verified: `np.random.seed(s)` before `GetEvaporationSignal`
gives a bit-identical initial quasiparticle population in every `VDetector`
(20 000 QP, three geometries, momenta identical). `events.evaporate` sets the
seed from `event_id` so the contract is enforced by construction, and a test
asserts it.

### 7.3 Noise made HeRALD-shaped

The existing `reference_budget.AthermalNoiseBudget` is a **magnetic**
calorimeter (Al₂O₃:Er, SQUID) — Johnson + SQUID + thermal pole + paramagnetic
spin. HeRALD is a **TES** on Si. The forms change; the machinery does not.
Add `TESNoiseBudget` beside it, closed-form and citable (Irwin & Hilton,
*Transition-Edge Sensors*, 2005), built from the existing components:

| term | form | component | note |
|---|---|---|---|
| TES Johnson | white, electrothermally suppressed below the loop roll-off | `white` + `rolloff(highpass, corner 1/2πτ_el)` | the suppression factor is a placeholder until R₀, L, β are read |
| thermal-fluctuation noise (TFN) | white current noise shaped by the responsivity: `S_TFN / (1 + (2πfτ_eff)²)` | `white` + `rolloff(lowpass, corner 1/2πτ_eff)` | dominant in-band term for a TES |
| load/shunt Johnson | white | `white` | |
| SQUID amplifier | white floor + 1/f, knee f_k | `white` + `powerlaw(−1)` | same two-parameter form already used in `AthermalNoiseBudget` |
| vibration / microphonics | narrow lines, 5–200 Hz | `line` ×n | **in band** at 250 kHz × 16 384 samples: df = 15.3 Hz — the opposite of LUCiD |
| mains pickup | 50 Hz + harmonics | `line` | in band; representable |
| paramagnetic spin | — | — | **remove**; not a TES term |

Sampling is inherited from `QPSimulator`: 2.5 × 10⁵ Hz, 16 384 samples,
65.5 ms — so a 24-channel group has N/C = 683, comfortably above the ≳ 500
floor rule. Pulse template: `QPSimulator`'s single-QP response (τ_rise 50 µs,
τ_decay 3 ms) is TES-shaped already; validate its time constants against the
published HeRALD pulse, and declare them.

**Multichannel structure for `HeRALD_v1`** — 24 CPDs on one cold stage:

| structure | physics | generator mode |
|---|---|---|
| shared, low-frequency | bath-temperature fluctuation seen by every TES | `shared_private`, shared spectrum = TFN-shaped low-pass |
| shared, lines | SQUID-array / wiring pickup — mains, vibration | `lowrank`, 1–3 latent modes carrying the lines |
| private, white | each TES's own Johnson + its SQUID floor | the private term |
| non-Gaussian, sparse | low-energy-excess bursts | `ArtifactInjector` bursts; **singles vs shared** (the distinction `pytessim` draws) is an N-vs-U family |

`corr_strength` and the number of latent modes are placeholders until the
HeRALD group's measured CSD, if any, is available; the *structure* is the
claim, the constants are declared.

### 7.4 Pilot, then acceptance

**Pilot (half a day, gate for everything else):** `HeRALD_v1` and
`HeRALD_v1_monolithic`, one 1 keV NR event, one seed; confirm arrival-time
lists feed `QPSimulator` unchanged; confirm the paired initial population;
one noise draw with `TES_HERALD_V1`; write the parquet.

**Acceptance:**

1. `events.evaporate` is bit-reproducible across geometries from `event_id`
   (test asserts identical `initial_momentum`).
2. `TES_HERALD_V1` integrates to `noise_power` on the 250 kHz grid; every
   term names its physical origin and its citation; placeholder constants are
   flagged as such in the provenance record.
3. Matched-cell κ floor at C = 24, N = 16 384 is measured and reported; the
   subject's layer-W residuals agree with it.
4. The 24 → 1 contrast reproduces the designed prediction: variance-ratio
   signature present at C = 24, absent (vacuous) at C = 1, on the same events.
5. ER vs NR at fixed energy lands in P_unexc with a Fisher-rank drop when NR
   is outside training support, and does not when it is inside.

---

## 8. Work packages and order

| WP | what | days | gate |
|---|---|---|---|
| N1 | `freeze_channel_gains` in `noise_module` | 0.5 | — |
| L0 | linear subject + §3 projectors + lookup on Tier 1 | 3 | — |
| H0 | HeST pilot (7.4) | 0.5 | B0 email sent |
| H1 | `herald_simulation` fork + `TESNoiseBudget` | 4 | H0 |
| S1 | make the transformer subject expose W, P, z, `jac_*` | 2 | — |
| A1 | `noise_module_lucid` (arms plan §7) | 8 | **A0 licence** |
| A2 | LUCiD cells + integration (§6) | 3 | A1 |
| C1 | TIDMAD WP10 | 3 | — |
| B | pre-registration bridge frozen | — | L0, S1 |
| R | confirmatory runs, all arms | — | B |

Order: N1 → L0 → H0 → S1 → H1 → C1 → (A1 → A2 when A0 lands) → B → R. HeST
before LUCiD because it is ungated and cheap; the linear subject before the
transformer because it is where the table is provable.

---

## 9. Risks specific to this plan

- **Geometry invariance is an assumption of the G row, not a given.** If S2
  is not channel-count-invariant in practice, G shows up at z and the G row's
  prediction fails — which is a finding about the architecture, not a failure
  of the protocol. Say so in the bridge.
- **Structural N and E share a mean-shift signature.** The lookup separates
  them by the variance ratio and the layer profile; if those are weak at
  realistic SNR, attribution between them degrades to "N-or-E, not covariance
  Σ". Report the confusion, do not hide it.
- **Estimator floors differ per arm** (1 GHz vs 250 kHz vs real). Re-measure
  the N/C rule on each arm before the bridge is frozen; a κ prediction made
  against Tier 1's floor is not transferable as a number, only as a sign.
- **HeRALD constants are placeholders.** The TES budget's *structure* is
  standard; its *numbers* are not ours to guess. Two states in the provenance
  record — `placeholder` and `from_paper` — and the paper reports which.
