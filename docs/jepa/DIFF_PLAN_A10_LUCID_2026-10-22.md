# A10 implementation diff plan — `lucid_bombs.py` / `lucid_run.py`

**Status: plan only. No code is changed in this pass.** The authoritative design is
[GATE_A10_MIXED_DOMAINS_2026-10-22.md](GATE_A10_MIXED_DOMAINS_2026-10-22.md) and the draft runbook
text [AMENDMENT_A10_MIXED_DOMAINS_2026-10-22.md](AMENDMENT_A10_MIXED_DOMAINS_2026-10-22.md). Work
starts only after A10 is registered.

Core invariants for every change below:

1. **One method, three data domains.** No new encoder/objective/anchor. Domain is metadata, never a
   model input.
2. **Photon-only path is bit-identical.** A config with `domains = ("photon",)` must reproduce the
   current datasets, probe tables and floors exactly; this is a regression test.
3. **One run = one detector state.** Every domain in a run shares F1/F3/F2/T and the dark bank.
4. **Per-domain probes and floors.** No pooled-only criterion; pooled result is secondary.
5. **No frozen config edited in place.** New versioned configs + hashes.

---

## 1. `src/jepa_noise/lucid_bombs.py`

### 1.1 Constants and labels

Add near the existing `READOUT_LABEL` / `ORACLE_ARM` constants:

```python
DOMAIN_PHOTON = "photon"
DOMAIN_MUON = "muon"
DOMAIN_ELECTRON = "electron"
DOMAINS = (DOMAIN_PHOTON, DOMAIN_MUON, DOMAIN_ELECTRON)
GENERATOR_ANALYTIC = "analytic_cherenkov"
GENERATOR_PHOTONSIM = "photonsim"
```

### 1.2 `GeneratorConfig`

New fields (all with defaults so old configs still load):

| field | default | meaning |
|---|---|---|
| `domains` | `(DOMAIN_PHOTON,)` | enabled domains |
| `domain_fractions` | `{"photon": 0.333, "muon": 0.333, "electron": 0.334}` | run composition; exact integer counts for 1000/50 |
| `generator_mode` | `"analytic_cherenkov"` | `analytic_cherenkov` or `photonsim` |
| `muon_energy_MeV` | `(300.0, 1500.0)` | log-uniform kinetic-energy range |
| `electron_energy_MeV` | `(100.0, 1000.0)` | log-uniform total-energy range |
| `particle_containment` | `"record"` | record containment; energy probes use contained subset |
| `cherenkov_n_phase` | existing `n_group` / phase index | `cos θ_C = 1/(β n)` |
| `shower_model` | `"simplified_axis"` | simplified e⁻ shower parameterisation |

`__post_init__` additions: fractions sum to 1, every enabled domain has a fraction, energies ordered
and positive, generator mode known, particle domains require the Cherenkov fields; keep the
existing frozen-response hash checks. `digest` automatically covers the new fields (it hashes all
dataclass fields), so datasets automatically bind the mixed config.

### 1.3 Truth containers

Add frozen dataclasses:

```python
@dataclass(frozen=True)
class ParticleTruth:
    domain: str
    vertex_m: np.ndarray             # (3,)
    direction: np.ndarray            # (3,) unit
    energy_MeV: float                # initial/total
    deposited_MeV: float | None      # in-volume, for contained events
    contained: bool
    segments: tuple | None           # (start_m, end_m, beta, charge) for muon
    shower_length_m: float | None    # electron
```

Extend `SensorDraws` (append fields with defaults to preserve positional construction in tests):

```python
domain: str = DOMAIN_PHOTON
truth: ParticleTruth | None = None
expected_pe: np.ndarray | None = None       # per-PMT mean charge, computed by the domain model
expected_time_ns: np.ndarray | None = None  # per-PMT first-arrival time
```

`key`, `run_id`, `event_id`, `vertex_m`, `yield_photons`, `qe_eff`, `poisson_u`, `charge_z`,
`transit_z`, `dark_counts`, `dark_times`, `window_ns`, `capacity`, `emission_offset_ns` stay.

### 1.4 Domain dispatch

Refactor `draw_event` into a dispatcher plus two private functions, keeping the existing RNG roles:

```python
def draw_event(...):
    domain = _domain_for_event(run_id, event_id, cfg, seed)
    if domain == DOMAIN_PHOTON:
        return _draw_photon_bomb(...)     # current body, bit-identical
    return _draw_particle_bomb(..., domain=domain)

def _domain_for_event(run_id, event_id, cfg, seed) -> str:
    # deterministic per-run composition from cfg.domain_fractions and the event index;
    # keyed on (seed, run_id, "domain") so it is reproducible and independent of event order
```

`_draw_photon_bomb` is the current body moved verbatim (no new draws, no changed role names) so the
photon-only regression holds.

`_draw_particle_bomb`:
1. draw species-specific truth: vertex (same fiducial sampler), direction (unit sphere), energy
   (log-uniform), containment/deposited energy from the segment/shower model;
2. build the emission segments: μ⁻ straight track with `β(E)`; e⁻ simplified axis + shower profile;
3. per segment, photon emission points and Cherenkov angle `cos θ_C = 1/(β n)`; propagate to each
   PMT with the pinned geometry/attenuation and the same solid-angle form as `expected_charge`;
4. aggregate `expected_pe` and first-arrival `expected_time_ns` per PMT;
5. draw the same Poisson/charge/transit/dark/emission-offset streams as the photon path, keyed by
   the same namespaces (`poisson`, `charge`, `transit`, `dark`, `dark_time`, `emission_offset`);
6. return `SensorDraws` with `domain`, `truth`, `expected_pe`, `expected_time_ns`.

New functions (pure, testable):

```python
def expected_charge_particle(geometry, truth, cfg) -> tuple[np.ndarray, np.ndarray]
def propagation_time_particle(geometry, truth, cfg) -> np.ndarray
def track_segments(truth, cfg) -> tuple         # μ
def shower_profile(truth, cfg) -> tuple         # e
def containment(geometry, truth) -> tuple       # contained flag + deposited energy
```

### 1.5 Readout

`readout_synthetic_unrepaired` currently recomputes
`mu = expected_charge(draws.vertex_m, draws.yield_photons, ...)` and `propagation_time_ns(...)`.
Change to:

```python
if draws.domain == DOMAIN_PHOTON:
    mu = expected_charge(...)               # unchanged, bit-identical
    t0 = propagation_time_ns(...)
else:
    mu = draws.expected_pe                  # from the particle model
    t0 = draws.expected_time_ns
```

The rest of the chain (QE/CE already folded into `mu` for particles; Poisson join with dark, SPE
smear, gain, fixed parity gain, threshold, walk, prompt window) is unchanged. The nominal twin
calls the same readout on `nominal_state`, so it works for particles without a second code path.
`fixed_charge_gain` and the parity response are still applied in the same place.

### 1.6 Generated run, manifests, storage

`GeneratedRun` additions (ordered after existing fields with defaults):

```python
domains: np.ndarray | None = None            # (n_events,) string
particle_truth: tuple | None = None          # ParticleTruth or None per event
expected_pe: np.ndarray | None = None        # (n_events, n_pmt) if storage budget allows, else None
```

`generate_run`:
- chooses the domain per event via `_domain_for_event`;
- stores `domains`, `particle_truth`; for photons the particle truth is `None`;
- keeps `vertices_m`, `yields_photons`, `emission_offsets_ns` for compatibility; for particles
  `yields_photons` is `NaN`/0 and the truth is in `particle_truth` (do not fake a yield);
- writes `domains` and flattened particle truth into `events.npz`; manifest gets per-domain counts,
  generator mode, energy ranges, mixing ratio and containment counts;
- `verify_manifest` accepts the new manifest schema and checks per-domain counts + hashes; old
  photon-only manifests remain readable (schema version check accepts both).

`write_generated_run`: schema constant bumped to a new dataset schema (e.g. `lucid-bomb-v2`), with a
`domains` field; the old schema remains accepted for replay/regression. `benchmark_generation` gains
per-domain events/s and reports the mixed cost.

`detectability`: add per-domain `mu_events`/`dmu_events` (photon: vertex/yield; muon/e: direction,
energy, containment); return per-domain ceilings. `r2_max`/`fisher_event` unchanged.

### 1.7 Photon-only regression test (hard requirement)

Take a frozen photon-only config and seed; assert `events.npz` and hits are byte-identical to the
current code, and that `SensorDraws` for photon events has no new draws (RNG role list unchanged).
This is the guard that the mixed code did not perturb the existing domain.

---

## 2. `src/jepa_noise/lucid_run.py`

### 2.1 Config and cells

New config keys (versioned file `configs/lucid_run_v0_mixed.json`):

```json
"data": {
  "domains": ["photon", "muon", "electron"],
  "domain_fractions": {"photon": 0.333, "muon": 0.333, "electron": 0.334},
  "generator_mode": "analytic_cherenkov",
  "particle_energy_MeV": {"muon": [300, 1500], "electron": [100, 1000]},
  "class_balanced_batches": true
},
"probe_sets": {"...": {"domain": "photon|muon|electron|all"}},
"evaluation": {"primary_domain": "particle", "domains": ["photon", "muon", "electron"]}
```

`Cell` additions (defaults keep old configs loadable):

```python
domains: tuple = (DOMAIN_PHOTON,)
primary_domain: str | None = None
```

`ProbeSet` gains `domain: str = "all"`; `_probe_sets` reads it. `enumerate_cells` passes the config
domains and primary domain; `_validate` checks every enabled domain has a probe set and a floor.

### 2.2 Dataset and loader

`ensure_dataset` / `_verify_dataset`: accept `lucid-bomb-v2`, verify per-domain counts and the
generator mode; reject a mixed manifest when the config asks photon-only and vice versa.

`TrainingLoader`:
- add `domains`, `domain_fractions`, `class_balanced` to `__init__`;
- `batch_at` keeps the 4-events-per-8-runs structure but, when `class_balanced`, chooses the
  per-domain counts in the batch (e.g. 16 photon + 8 μ + 8 e) using the run manifest's domain
  arrays; if a run lacks enough events of a domain, fall back to the next run and record the
  substitution rate;
- the domain label is **not** added to features; it is only used for selection and for the truth
  table.

`_tokenise_run` and `_to_device` are unchanged; `_run_truth` adds `domain`, `direction`,
`energy_MeV`, `deposited_MeV`, `contained`, `particle` per event.

### 2.3 Probes and evaluation

`probe_tables` / `_ingest_probe_chunks` / `probe_scores`:
- add `domain` argument; filter probe events by domain in `_probe_batch`/`_probe_source` (the
  domain lives in `events.npz`);
- result keys become `domain::kind::form`; pooled keys remain `kind::form` (secondary);
- per-domain `random_init_floor` and floor cache identity include the domain.

`evaluate_arm_run`:
- loop over `cell.domains`, run the four probe forms per domain, then the R6/R7/R11/R12 logic;
- R6/R7 per domain; primary domain from `cell.primary_domain`;
- domain-classifier diagnostic: fit a logistic/ridge classifier on frozen embeddings and report AUC
  per cell (diagnostic only, no gate).

`ensure_floor`:
- **fix the known device bug**: `model = la.random_init_floor(...).to(device)` before
  `probe_scores` (or pass `device` into `random_init_floor`);
- add the domain to the floor identity/cache key.

`_score`, `_r11_block`, `_reference_vertex_resolution`, `_r2`/`_res`/`_values`:
- accept domain-qualified tables; keep photon-only behaviour when `domains == ("photon",)`.

### 2.4 Grid and aggregation

`run_grid`:
- pass domains/fractions from config to `TrainingLoader` and `evaluate_arm_run`;
- the run manifest records the mixed composition and the domain counts;
- the aggregate emits per-domain R6/R7 + the pooled secondary row;
- do not change the run count formula: mixed replaces photon-only; the 165 rung-1 runs stand.

`report.py` / `PREDICTIONS.md`:
- add per-domain rows for R6/R7/R13; keep the registered rows for the photon domain;
- byte-identical regeneration for the photon-only config (regression).

### 2.5 Budget and projection

`estimate`/ledger helpers: generation cost is per domain; add the measured per-domain events/s from
`benchmark_generation`; re-project ledger + grid ≤ 160 h start rule, 170 h cap. The fallback order is
the one in A10.6; each step writes a manifest field.

---

## 3. New / changed configs (no in-place edits)

| file | status |
|---|---|
| `configs/lucid_bombs_v1_mixed.json` | new; domains, fractions, generator mode, energy ranges, hash |
| `configs/lucid_run_v0_mixed.json` | new; cells, probe sets per domain, primary domain, fallback flags |
| `configs/lucid_run_v0_rung1.json` | unchanged (photon-only, registered) |
| `configs/lucid_bombs_v0.json` | unchanged (photon-only, registered) |

Every config carries `derived_from`, a `sha256`, and a status (`registered` or `derived_non_registered`).
The frozen photon-only configs remain the reference for the regression test.

---

## 4. Tests to add

1. `test_domain_composition`: per-run counts match `domain_fractions`; every event has exactly one
   domain; class label absent from features.
2. `test_photon_bit_identical`: photon-only dataset/hits equal to the current code at the same
   seed/config.
3. `test_particle_truth`: μ track length/containment, e shower profile, Cherenkov angle, energy
   ranges, finite expected charge/time.
4. `test_same_state_across_domains`: one run's state hash is identical for all domains.
5. `test_per_domain_probe_floor`: floor cache keyed by domain; a mixed dataset produces separate
   floors.
6. `test_mixed_manifest_roundtrip`: write/verify manifest with domains, counts, generator mode.
7. `test_loader_class_balance`: batch domain counts match the registered mix within tolerance;
   substitution rate reported.
8. `test_device_floor_fix`: a CPU/GPU-agnostic unit test that the floor model and probe batch move
   to the same device (skip when CUDA is unavailable; assert the `.to(device)` call path).
9. `test_photon_only_report_byte_identical`: `report` regeneration for the photon-only config.

---

## 5. Order of implementation

1. Photon-only regression guard + `_draw_photon_bomb` extraction (no behaviour change).
2. `ParticleTruth`, `SensorDraws` extensions, particle generator, readout dispatch; unit tests.
3. Manifest/storage v2 + `lucid_run` dataset verification; mixed configs.
4. `TrainingLoader` class balance; per-domain probes/floors; `evaluate_arm_run` loop.
5. Detectability per domain; `benchmark_generation` mixed; budget projection.
6. 500-step mixed smoke on the Titan, then the deepthought2 check; then the 2-seed dev comparison
   (mixed vs photon-only vs particle-only) before A10 registration.
7. A10 registration; detectability freeze; grid relaunch.

**Do not run the 165-run photon-only grid before step 6 finishes**; the dataset change would
supersede it.

---

## 6. Risks and mitigations

| risk | mitigation |
|---|---|
| Photon path not bit-identical | step 1 regression test; no new RNG roles for photon events |
| Analytic Cherenkov too crude | mandatory PhotonSim μ⁻ validation subset with pre-declared tolerances; label `simplified_cherenkov`; E3-independent |
| Electron generator exists only as analytic | label; Junjie samples replace it later without changing the schema |
| Domain-class shortcut | no label input; domain AUC diagnostic; per-domain probes; optional class-balanced batches |
| Generation cost overruns | per-domain benchmark; 2:1:1 fallback; electron deferral; G2 |
| Mixed dataset changes R6/R7 | per-domain criteria registered in A10; pooled secondary; null reported |
| Code churn touches frozen paths | new configs, new schema; frozen photon configs untouched; guard checks |
