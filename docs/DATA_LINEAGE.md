# Data lineage

Last updated 21 Nov 2026 (r2-lucid, runbook `docs/round2/RUNBOOK_2026-11-21_R2_LUCID_REGEN.md`).

## 1. The analytic toy (all round-1 and round-2 data)

Every round-1 and round-2 dataset (WCTE, SK, JUNO, HK) was produced by our own analytic photon-bomb
generator, `src/jepa_noise/lucid_bombs.py` and `src/jepa_noise/r2_generator.py`, **not by LUCiD**. LUCiD
supplied only geometry configs; LUCiD transport (`noise_module_lucid/lucid_transport.PromptTransport`,
pinned `eaf2e89`, absorption off) was used only for the Gate V comparison and a side study.

The analytic generator has an isotropic point source (and an analytic Cherenkov ring model for muons
and electrons), straight-line prompt transport with exp(−d/λ), no scattering or reflection, a Gaussian
first-hit time with fixed σ_t, and Poisson photoelectrons with SPE smearing, a threshold, dark counts
and an NHits trigger.

**Every result and document that used these data is an "analytic-toy-world" result.** That covers all
round-1 results (`results/jepa/`, `results/pmt_*`, `results/lucid_response_toy_dev_2026-10-12/`, the
round-1 extension returns under `/ceph/dwong/jepa/returns/`) and all round-2 results
(`results/round2/20261112_*`, `20261113_titan`, `20261114_batch`, `20261117_repair`,
`results/round2/R2_RUN_REPORT.md`), and the documents built on them (the round-2 reports, the readings
summaries, the repair report and `docs/round2/CLAIMS_LEDGER_2026-11-21.md`). The toy remains useful as
an exact-likelihood toy world: its CRLB and ML ceilings are exact there, and only there.

The toy datasets are archived, not deleted: `/ceph/dwong/jepa/data/ARCHIVE_ANALYTIC_TOY_2026-11-21.md`
(+ `.json`, per-file SHA256, sha256 `8c641e3f…` / md `1e0c73f9…`). The round-2 packs are read-only.
On branch `lucid-mc` the toy stays selectable with `--data-kind analytic_toy` (or
`JEPA_DATA_KIND=analytic_toy`); see `src/jepa_noise/data_roots.py`.

## 2. The LUCiD lineage (`lucid_v1`, `dataset_kind = lucid_mc_native`, tier A1)

Data: `/ceph/dwong/jepa/data/lucid_v1/<detector>/` (`shards/<cohort>/<run>/`, `cache/`,
`<det>_lucid_v1.h5`, WCTE also `wcte_train/`). Manifests: `results/round2/20261121_lucid/DATA_MANIFEST_<det>.json`.

**LUCiD does the physics** (LUCiD `e651e836`, `/ceph/dwong/jepa/external/LUCiD_main_e651e836`):
photon generation (isotropic source rays; SIREN Cherenkov emitters for μ⁻/e⁻ with their emission-time
fit), wavelength sampling, transport with absorption, Rayleigh and Mie scattering and wall/sensor
reflection (K = 12 iterations, sampled, the `lucid-run-job` kernel settings), wavelength-dependent QE
and its Bernoulli detection, and the electronics: dark noise (4.2 kHz), per-sensor integration windows,
sampled SPE charge, charge-dependent PMT jitter, 0.4 ns TDC, 0.25 PE discriminator, and the sliding-window
trigger where the detector config has one (SK, HK). Details: `results/round2/20261121_lucid/LUCID_CAPABILITIES.md`.

**VERITAS-side steps** (each listed in every shard manifest as `veritas_steps`; code
`src/jepa_noise/lucid_mc_generator.py`):

| # | Step | Kind (runbook P1) |
|---|---|---|
| 1 | Source draws with the round-2 keyed RNG namespaces: vertex (`r2_geometry.fiducial_vertex`), photon-bomb yield (log-uniform, range from the L2 pilot), domain plan 1:1:1 (`lucid_bombs.domain_plan`), particle direction and energy, emission offset U[−50, 50] ns (truth only) | (a) event source definition |
| 2 | Photon bombs: LUCiD `get_isotropic_rays_random` directions, emission at t = 0, chunked into the kernel buffer with LUCiD's `N` mask | (a) |
| 3 | Particles: LUCiD SIREN rays (weights = PMF × n_photons(E)) thinned to unit photons with p = weight / n_chunks; rays emitted outside the tank dropped | (a) |
| 4 | Sensor index map LUCiD → round-2 PMT index (LUCiD reorders barrel/top/bottom) | (b) conversion |
| 5 | Emission offset added to deposit times before digitizing (as LUCiD's local-frame production digitize adds t0) | (b) |
| 6 | Round-2 fixed per-PMT timing-width factor (odd modules +5 %) multiplies LUCiD's sampled PMT jitter before LUCiD's TDC rounding | (d) detector state |
| 7 | One hit per PMT: the earliest surviving LUCiD digit | (b) |
| 8 | Charge × round-2 fixed parity gain (odd modules +3 %) × per-run F1 gain (`RunSpec.state`, s = 0.05 / 0.30); the twin uses the nominal gains | (d) detector state |
| 9 | VERITAS NHits trigger time and prompt window (`noise_module_lucid.readout.nhits_trigger_time`, `prompt_window_mask`) for the token time digit − trigger; window-start fallback (−200 ns) without an NHits trigger. N from the round-2 rule at LUCiD's dark rate (6/10/10/12) | (c) token-time convention |

Not applied on the VERITAS side: no VERITAS dark counts, no `noise_module_lucid` SPE/threshold/walk
(LUCiD covers them), no expected charge, no solid angle, no analytic arrival time. Gate G6 checks that
the generation path never imports or calls `lucid_bombs` / `r2_generator` physics (static and run-time
tests in `src/jepa_noise/tests/test_lucid_mc_generator.py`); the allowed `lucid_bombs` helpers are
`lucid_mc_generator.ALLOWED_LB_NAMES` (geometry, keyed RNG, cohort/run specs, detector state, sparse
storage).

Truth fields: all round-2 fields are written; `expected_pe` holds LUCiD-sampled detected photoelectrons
before electronics (MC data have no expectation) and `expected_pe_pmt` is not applicable
(`lucid_mc_generator.TRUTH_FIELD_STATUS`).

Geometry: HK and JUNO reuse the round-2 geometry (LUCiD positions agree within 0.002 mm); WCTE and SK
use frozen geometries with LUCiD's wall-snapped positions (`src/jepa_noise/configs/lucid_v1/`).

### 2.1 Sealed test cohort (`sealed_test`, r2-tier1, 22 Nov 2026)

Cohort `sealed_test`: 30,000 events per detector (600 runs × 50; photon / μ / e 1:1:1), same LUCiD
commit, tier A1, configuration and detector-state law as `dev_probe_test` (s = 0.05), no `_s030` twin.
The cohort name enters the keyed-RNG derivation (`lucid_mc_generator.COHORTS`), so event IDs and run
keys are disjoint from every other cohort (checked). Generated on HTCondor A100 (code `b1f500c`),
packed to `/ceph/dwong/jepa/data/lucid_v1/<det>/<det>_lucid_v1_sealed.h5` (read-only; never rewritten).
Manifests: `results/round2/20261122_tier1/DATA_MANIFEST_<det>_sealed.json`; validation against
`dev_probe_test` (G2, G4 within 10 %, disjointness, completeness): `SEALED_VALIDATION_<det>.json`;
G6 is the generator's static/run-time test in the native suite. **Only `scripts/tier1_eval.py` reads
these packs**; every read is logged in `results/round2/20261122_tier1/SEALED_ACCESS_LOG.md`.
SK, WCTE and HK validated 22 Nov on the default sample (60 reference runs); JUNO passed with the full reference cohort (200 runs): its photon-bomb median PE differs by −9.95 % (−10.3 % on 60 runs, a 1.5σ sampling fluctuation); see `SEALED_VALIDATION_juno_n200.json`.

### 2.2 WCTE wide-window re-pack (`lucid_v1_w2`, r2-sparse, 24 Nov 2026)

Dataset kind `lucid_v1_w2` (pack attr / manifest `dataset_kind = "lucid_v1_w2"`; `data_roots.LUCID_V1_W2`).
WCTE only. A re-readout of the cached `lucid_v1` LUCiD digits (`/ceph/dwong/jepa/data/lucid_v1/wcte/cache`,
LUCiD e651e836, tier A1, read-only) for `dev_train`, `dev_probe_train`, `dev_probe_test` and `sealed_test`,
with the readout window [−110, +50] ns around the unchanged NHits trigger (N = 6, 20 ns; r2-sparse W1 rule).
Same runs, events, truths, detector states and pack layout as `lucid_v1`; no LUCiD regeneration.
LUCiD dark noise (`generate_dark_noise`, 4.2 kHz, through LUCiD's digitizer chain, keyed
`event_seed(seed, run_id, event_id, "dark_ext")`) fills the parts of the window outside the cached dark span
(first-light proxy − 100 ns + 1.5 ns calibrated offset, 20 ns generation margin; `src/jepa_noise/lucid_w2.py`);
extension digits within one integration window of a cached digit are merged (earlier time, summed charge).
Per hit the pack also stores `extra_n_later`, `extra_q_total_window`, `extra_physics_first`, `extra_dark_ext_first`
(unused in r2-sparse). Data of record: `/ceph/dwong/jepa/data/lucid_v1_w2/wcte/` (`shards_w2b/`,
`wcte_lucid_v1_w2.h5`, `wcte_lucid_v1_w2_sealed.h5`, `wcte_train/`, `SHA256SUMS`); `shards/`, `shards_final/`,
`shards_w2/` are superseded validation passes (`SUPERSEDED.md`, never read). Manifest
`results/round2/20261124_sparse/DATA_MANIFEST_wcte_w2.json`; validation `results/round2/20261124_sparse/w2_validation/`.
Sealed packs are read only through `scripts/tier1_eval.py`; the re-pack, packing and structural validation of the
sealed cohort are logged in `results/round2/20261124_sparse/SEALED_ACCESS_LOG.md`.
