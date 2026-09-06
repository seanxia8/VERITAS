# `herald_simulation` — HeST → `qp_simulator` → `noise_module`

The paired superfluid-helium dark-matter arm of
`docs/LATENT_MONITORING_PLAN_2026-09-05.md` §7. Three of the four links already
existed; this package is the glue, the cells, and the provenance.

```
GetQuanta(E, "ER"|"NR")            -> n quasiparticles          [HeST]
GetEvaporationSignal(detector, …)  -> per-sensor arrival times  [HeST]
QPSimulator.generate(times_ns)     -> clean per-channel trace   [qp_simulator]
MultiChannelNoiseGenerator(TES)    -> Σ̂ / Σ, correlated noise   [noise_module.TESNoiseBudget]
```

HeST is used as a **pinned, unpatched library** (`fetch_hest.sh`, commit
`8ffd23e8`, 2026-03-08). Defects go upstream as PRs — that is what keeps
"the physics is theirs" true in the paper. Import needs `qetpy` and `numba`;
HeST's `setup.py` also lists `detprocess`, whose `aplus` dependency does not
build on modern Python — skip it.

## Quick start

```bash
bash src/herald_simulation/fetch_hest.sh
pip install qetpy numba pyarrow
PYTHONPATH=src python -m herald_simulation.simulate --out runs/herald_pilot --n-events 8 --qp-fraction 0.02
PYTHONPATH=src python -m pytest src/herald_simulation/tests -q
```

`--qp-fraction` thins the quasiparticle population for cost (a 1 keV NR is
~9×10⁵ QP ≈ 17 s single-core at ~55k QP/s); the arrival-time distribution is
kept and the trace amplitude is rescaled by the known factor, recorded in
`provenance.json`. Use 1.0 for production.

## The pairing contract (verified, tested)

`QP_propagation` samples the entire initial population — directions, then
momenta — before geometry is touched. `events.evaporate` seeds NumPy's global
generator from `event_id`, so **one `event_id` gives a bit-identical initial
quasiparticle population in every detector**. `tests/test_pairing.py` asserts
it on the 24-sensor / 1-sensor pair.

## Cells (`simulate.all_cells`)

| moved | cells |
|---|---|
| reference | `HeRALD_v1` (24 CPDs), ER at 1 keV, `TES_HERALD_V1` |
| geometry | `HeRALD_v1_monolithic` (1 sensor, identical cell) — the designed 24→1 contrast; `HeRALD_UMass_splitCPD` (2) |
| Σ, covariance | bath correlation ↑, low-rank pickup modes, SQUID 1/f knee ×10, mains ×8 |
| Σ, structural | sensor loss, gain drift, timing jitter (applied to signal *and* noise) |
| event | NR at the same energy; ER at ½ and 2× |
| undeclared | WIMP recoil spectrum (500 MeV) |

Every cell reuses the same `event_id` list. `make_cell(cell_radius_cm,
fill_height_cm, sensor_pitch_cm, array_map)` builds any HeRALD-shaped array
from HeST's own primitives and hashes its parameters; `shipped(name)` wraps
the five HeST builders.

## Noise: HeRALD-shaped, placeholder constants

`noise_module.tes_budget.HERALD_V1_PLACEHOLDER` — TFN with the responsivity
roll-off, TES Johnson with loop-gain suppression, shunt Johnson, SQUID white +
1/f, mains (50 Hz + harmonics) and vibration lines, no paramagnetic-spin term.
At 2.5 × 10⁵ Hz × 16 384 samples the resolution is 15.3 Hz, so the lines ARE
in band — the opposite of a 1 GHz PMT digitiser. Every constant carries a
provenance state (`placeholder` / `design` / `from_paper`); the shipped
budget is placeholder except the two time constants, which match
`QPSimulator`'s template. Replace with values read from arXiv:2307.11877
before any dataset is released.

`add_noise` returns Σ̂ (implied) and Σ (realized) and their κ; on a matched
cell at N/C = 683 that κ is the estimator floor (≈1.2), not a mismatch — the
cross-cell κ(Σ̂_ref⁻¹ Σ_cell) is what `latent_monitor` computes when a subject
is applied.

## Output

`<out>/<cell>/truth.parquet` (one row per event: `event_id`, geometry hash,
truth, yields, `kappa_floor`, structural intervention), `traces.npy`
`(n, C, N)` float32, `provenance.json` (HeST commit, geometry parameters and
positions, budget with provenance states, trace config, event ids).

## Gates

- **B0** — HeST's `LICENSE` is MIT text with the unedited PyPA copyright line.
  Ask Greg Rischbieter (rischbie@umich.edu) to name the holder before a
  released dataset depends on it.
- Upstream PRs to open: `VDetector.get_QPEmap()` returns `LCEmap_positions`
  (`core/Detection.py:243`); `setup.py`'s `detprocess` → `aplus` dependency.
