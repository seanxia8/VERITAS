# WP4b LUCiD generator — development contract (no reported results)

Owner: WP4b implementer. Scope: `src/jepa_noise/lucid_bombs.py`,
`src/jepa_noise/tests/test_lucid_bombs.py`, `src/jepa_noise/configs/lucid_bombs_v0.json`.
Binding: runbook v4 §3.5.2--§3.5.4, §3.5.3a, §7 and v3 §6 WP5.

**Status: `readout_synthetic_unrepaired`, development-only.** The physical WCTE geometry
(105 mPMT × 19 PMTs, physical `mpmt_id` map) and the pinned LUCiD revision
`eaf2e89f579d6679e90020c732ac2c846895d691` are not available. The only built-in geometry is an
8 × 7 synthetic cylinder used by tests; it is **not** WCTE and must not be labelled as such.
Nothing produced with this module may be entered in `PREDICTIONS.md` or a registered cell.

## Implemented (§3.5.3, §3.5.3a, §3.5.4, v3 §6 WP5)

- **Laws**, pure and geometry-agnostic, with keyed RNG namespaces:
  F1 `gain_m = 1 + s·u_m`, F3 `gain_p = 1 + s_p·w_p`, F2 `λ = λ0·exp(a·v)`,
  F2′ `gain_p ← gain_p·(1 + b·v·z_p/H)`, R11b (one `c` drives module gain and `v`),
  and T (`r/R`, `R = 100`). F1/F3/F2′ factors are clipped to [0.5, 1.5] **separately** and
  clipping counts are stored in `DetectorState.clipped`. `k_R11b` and `k_T` are persistent
  vectors from `lucid/k_R11b` and `lucid/k_T`; `state_for_run(law, setting, run_id, geometry)`
  returns per-PMT total gains `gain_m·gain_p`, λ, law/settings, run id, seed, namespace and a
  SHA-256 state digest. F3 is included in every cell; F2/F2′ also carry F1 at `s = 0.05`, so
  the R11 design has all three factors varying independently.
- **Nominal twin**: `nominal_twin(draws, geometry, readout, arm=...)` reuses the identical
  `Draws` object and the nominal state (gains 1, λ = λ0). It raises `PermissionError` unless
  `arm == "oracle_nominal_twin"`, encoding "only the oracle arm may read the twin". The
  `readout(state, draws) -> HitReadout` contract is the binding point for the merged PMT chain;
  the synthetic stand-in lives only in the test file.
- **Runs and IDs**: `COHORTS` reproduces the v3 §6 ranges under `lucid/` (train 0–199999,
  probe-train 200000–209999, probe-test 210000–219999, development 1000000–1219999) with
  training/dev runs of 1000 events and 100 runs per 100k, and probe/robustness runs of 50
  events and 200 distinct states per 10k. Every set has its own run ids and event blocks;
  `make_runs(..., settings=[...])` gives per-run settings for law T (`law_T_training_ratios`,
  `law_T_ratios(kind="in_range"|"extrapolation")`).
- **Manifest and storage**: `write_manifest` records run id, event span and event-id hash,
  law/setting, per-run state hash, geometry hash, dataset seed, `code_sha` (`git rev-parse
  HEAD`), readout label and `reported_results=False`; `verify_manifest` recomputes every hash.
  `SparseHitStore` is CSR-like (`event_ids`, `offsets`, `pmt_id`, `charge_pe`, `time_ns`);
  `write_sparse_hits`/`read_sparse_hits` round-trip it and `write_sha256sums`/
  `verify_sha256sums` provide the `SHA256SUMS` contract.
- **N_min**: `ViewExclusionCounter` implements the real §3.5.4 accounting: initial draw plus at
  most 3 `across` redraws, exclusion after the third failure, and `within` exclusion; per-cell
  redraw/exclusion rates are returned. `view_masks` is a documented `NotImplementedError`
  stub because both views need the physical geometry (Hilbert module order / module map).
- **Detectability (§3.5.3a)**: `fisher_event` computes
  `Σ_p (∂μ_p/∂f)² / (μ_p(1+σ_spe²) + v_dark)`; `r2_max` and `detectability` return the per-event
  and per-run (`I_run = Σ I_event`) ceilings and accept a caller-supplied
  `profiled_information` for the registered profiling fallback. The naive sum, which treats
  the vertex/yield/other factors as known, is not the registered profiling and is labelled by
  `Detectability.profiled = False`.

## Blocked on inputs not yet available

- **Physical geometry**: replace `synthetic_cylinder_geometry` with a loader for the PMT branch's
  `mpmt_id` map; supply `module_of_pmt`, `pmt_id_in_module`, `z_pmt_m`, `half_height_m`. This
  unblocks `view_masks` (Hilbert order of module centres and even/odd within-module masks).
- **Pinned LUCiD revision / merged PMT readout**: the real `readout` callable (QE·CE detection,
  SPE charge, TTS, dark counts, fixed threshold, walk) and the photon source that fills `Draws`.
  The nominal twin then uses the *merged* chain, and the detectability check is re-run.
- **Y_lo, Y_hi, N_min occupancy and the §3.5.3a ceilings** need development data on the merged
  readout; this module only provides the machinery. No value is frozen here.
- **Real storage volume/benchmark** (1000-event benchmark and the 48 h projection) waits for
  those inputs; the manifest and sparse layout are ready.

## Binding the real readout

1. Build `Draws` from the LUCiD photon/spe/transit/dark draws using
   `jepa_noise.lucid_bombs.keyed_rng(seed, ...)` so both arms reuse one object.
2. Implement `readout(state: DetectorState, draws: Draws) -> HitReadout` (or return a richer hit
   object and feed `sparse_from_readouts`). Gains must act **before** the threshold.
3. Call `nominal_twin(draws, geometry, readout, arm="oracle_nominal_twin")` only in the oracle
   arm; all other arms use `readout(state, draws)` with the drifted state.
4. Keep the `readout_synthetic_unrepaired` label until the merge is accepted, then re-label the
   manifest and re-run the detectability check. The real stand-in is never used in results.

## Commands

```sh
export PYTHONPATH=src
PY=/private/tmp/jepa-venv-cpu311/bin/python

$PY -m pytest src/jepa_noise/tests/test_lucid_bombs.py -q   # 24 passed
$PY -c "from jepa_noise import lucid_bombs as lb; \
g=lb.synthetic_cylinder_geometry(); \
print(lb.state_for_run('F1', 0.05, 'lucid/train/0000', g).digest)"
```

## Ambiguities in §3.5.3 (for the lead)

1. **λ0 is unspecified.** The PMT `reference.json` has absorption off and the runbook fixes only
   F2's form. `LAMBDA0_M = 50.0` is an explicit placeholder; it changes F2's scale, not its form.
2. **No robustness ID range in v3 §6.** `lucid/robustness` 220000–229999 (and
   `lucid/dev_robustness` 1220000–1229999) is a WP4b extension; the lead should register or
   replace it.
3. **R11b's `setting`.** The runbook says "the F2 (or F2′) variable `v = c`" but does not name
   which attenuation strength `a`/`b` goes with the co-varying training law. We interpret
   `setting` as F2 `a` and use the exponential λ form (the F2′ variant is not selected).
4. **Per-cell ID blocks.** v3 §6 gives one range per role; with several drift settings, each
   setting needs its own 10k probe sets. `make_runs` accepts `event_id_start` so the caller can
   allocate disjoint blocks; the allocation itself is not registered.
5. **F2′ `H`** is taken as the inner half-height (the runbook's definition); the physical value
   is pending the geometry.

## Engineering decisions (one line each)

- `Geometry` is an explicit dataclass with a canonical digest; no WCTE numbers are embedded.
- Keyed namespaces are SHA-256-derived (`seed, role, run/event/geometry`), never integer offsets.
- The test-only stand-in reads a shared `Draws` object so the twin cannot accidentally re-draw.
- Clipping is applied to each factor separately and counted as `module`/`pmt`/`gradient`.
- `all_runs` was omitted; `make_runs` plus `COHORTS` composes the disjoint sets explicitly.
- Sparse storage is CSR-like with float32 charge/time and int32 PMT ids, hashed in `SHA256SUMS`.
