# WP4c — LUCiD views, masks and the point encoder (development status)

Owner: WP4c implementer. Scope (new files only): `src/jepa_noise/lucid_views.py`,
`src/jepa_noise/point_encoder.py`, `src/jepa_noise/tests/test_point_encoder.py`,
`src/jepa_noise/configs/point_encoder_v0.json`, this file.
Binding: runbook v4 §3.5.4 (views and masks), §3.5.5 (LUCiD encoder WP), §3.5.3
(geometry, events, storage), §3.5.6 (probes), §6.2 (memory caps), and the
2026-10-17 DECISIONS entries on the WCTE geometry (1-based `pmt_id_in_mpmt`,
module id 45 absent, surfaces as strings).

**Status: development only.** The readout is `readout_synthetic_unrepaired`; no
number here may enter `PREDICTIONS.md`, a registered cell or a gate. Local tests
use the schema-identical synthetic geometry (`lucid_bombs.from_synthetic`, 1995
PMTs / 105 modules × 19, NOT WCTE) and tiny synthetic runs. `views.py` (the
Tier-1 law) and every no-touch package were left untouched.

## What is implemented

1. **Views and masks (`lucid_views.py`)**, exactly §3.5.4.
   - `across`: modules ordered along a 3D Hilbert curve of their centres; per
     event a random contiguous half of that order is the target, the rest the
     context. Target = `n_modules // 2` = 52 modules (988 PMTs), context = 53
     modules (1007 PMTs) on the real layout.
   - `within`: context = PMTs with **even** 1-based `pmt_id_in_mpmt` (2…18,
     9 per module, 945 PMTs), target = **odd** (1…19, 10 per module, 1050 PMTs);
     the partition is fixed for every event.
   - `N_min = 20` target hits. `across` redraws the random half up to 3 times
     (4 draws); an event that still fails is excluded and counted. `within`
     excludes and counts (A8).
   - Per-view hit fractions and per-cell counters from the generator's own
     `ViewExclusionCounter` (reused, not duplicated): accepted/excluded,
     `across_redraws`, redraw rate, both exclusion rates.
   - Occupancy ablation (§3.5.4, A8): `occupancy="balanced"` subsamples the
     *unhit* PMTs of the **context** view only, to match that view's hit count;
     every target PMT stays queried. Recorded, `natural` is the default.
   - Masks depend on (geometry, event id, seed, replicate) only; every mask
     carries a JSON key (`split`, `geometry_sha256`, `event_id`, `seed`,
     `replicate`, `offset`, `n_target_modules`, `n_min`, `max_redraws`,
     `module_hilbert_bits`, `occupancy`, plus the recorded attempts and hit
     counts) and `mask_from_key` recomputes the identical partition, checking
     the geometry digest and the recorded counts.
2. **Point encoder (`point_encoder.py`)**
   - `tokenise(run, geometry, ...)`: reads `hits.npz` + geometry into
     `(B, 1995, 9)` float32, the target/hidden masks, the context/target index
     arrays, per-event keys and the §3.5.4 report; excluded events are dropped.
   - Features, fixed data-independent scaling: position / (R, R, H/2); unit
     facing direction / 1.0; ln(q / 1 PE) for hit PMTs and 0 for unhit;
     first-hit time / 100 ns and 0 for unhit; hit flag.
   - Serialised point transformer: Hilbert 3D sort of the PMT tokens; attention
     inside 64-token windows, offset shifted by 32 on alternate layers (cyclic);
     4 pre-norm blocks, d = 128, 4 heads, no final norm. The read-out is the
     block-4 output; `PointEncoder.final_norm` is `None` and a test asserts that
     exactly `2 × n_blocks` norms run.
   - Masking: `encode(features, hidden_mask)` replaces hidden tokens with one
     learned `mask_embed` vector after the input projection, before block 1.
   - Predictor: 2 pre-norm blocks, d = 64, full attention, reading the context
     read-out plus fixed position queries at the target PMTs; output one
     d = 128 prediction per target PMT.
   - `smoke_loss` (MSE, stop-grad target read-out) exists only for the smoke and
     determinism tests; it is **not** a registered arm.
3. **Tests (`tests/test_point_encoder.py`, CPU): 14 tests, 3.0-3.7 s.**
4. **`configs/point_encoder_v0.json`**: frozen architecture and numeric settings;
   optimiser/learning-rate entries are `pending_review`.

## Exact commands

```sh
cd /Users/wongdowling/Documents/VERITAS
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 14 tests, ~3.5 s
/private/tmp/jepa-venv-cpu311/bin/python -m pytest src/jepa_noise/tests/test_point_encoder.py -q
# amplitude fixture R² and parameter count
/private/tmp/jepa-venv-cpu311/bin/python -m pytest src/jepa_noise/tests/test_point_encoder.py -q -s \
  -k "amplitude or parameter_count"
```

Memory/throughput measurement (no GPU in this session; CPU peak RSS is the local
proxy, and the recorded verdict is about the §6.2 *Titan* cap):

```sh
cd /Users/wongdowling/Documents/VERITAS
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/jepa-venv-cpu311/bin/python - <<'EOF'
import tempfile, pathlib, time, resource, numpy as np, torch
from jepa_noise import lucid_bombs as lb, point_encoder as pe
geom = lb.from_synthetic(); model = pe.LucidPointJepa(pe.EncoderConfig(), geom)
rng = np.random.default_rng(1); reads, ids = [], []
for e in range(32):
    pmt = np.sort(rng.choice(geom.n_pmt, 300, replace=False)).astype(np.int32)
    reads.append(lb.HitReadout(pmt, (rng.exponential(3., 300) + .3).astype(np.float32),
                               rng.normal(50., 5., 300).astype(np.float32))); ids.append(e)
d = pathlib.Path(tempfile.mkdtemp())
lb.write_sparse_hits(d / "hits.npz", lb.sparse_from_readouts(np.array(ids), reads))
batch = pe.tokenise(d, geom, split="within", seed=0)
for n in (2, 8, 32):
    sub = pe.LucidBatch(**{**batch.__dict__, "features": batch.features[:n],
                           "target": batch.target[:n], "hidden": batch.hidden[:n],
                           "context_index": batch.context_index[:n],
                           "target_index": batch.target_index[:n]})
    t0 = time.time(); pe.smoke_loss(model, sub).backward()
    print(n, round(time.time() - t0, 2), "s", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1048576, "MiB")
EOF
```

## Hilbert order construction

Skilling's *axes → transpose* algorithm (the formulation of the MIT-licensed
`hilbertcurve` package), re-implemented in vectorised numpy in
`lucid_views.hilbert_index_3d`:

1. the points are mapped to an integer grid of the **fixed detector box**
   `[-R, R] × [-R, R] × [-H/2, H/2]` with `2**bits` cells per axis
   (`bits = 12` for the PMT tokens, 10 for the module centres); the grid is a
   function of the geometry only, so the order never depends on hits or charges;
2. the inverse "undo excess work" pass, then the Gray encode, then the XOR flip
   (`t`), as in Skilling's published C code;
3. the transpose is packed into an integer with **bit `b` of coordinate `i` at
   position `3b + (2 − i)`** — coordinate 0 is the most significant of each row;
   this packing is the one point where the naive interleaving breaks curve
   continuity, and the continuous version is what is implemented;
4. ties (two points in one grid cell) are broken by the original index
   (`np.lexsort`), so the order is canonical.

The tests check the defining property on a full `2**3` grid: a bijection onto
0…511 in which consecutive points differ by Manhattan distance exactly 1.
`module_hilbert_order` and `token_hilbert_order` cache by geometry digest.

The `across` target is `order[(offset + arange(52)) % 105]` — a cyclic contiguous
half. The offset is drawn per attempt from
`lucid/views/across/<replicate>/<event_id>/attempt/<k>`.

## Mask definitions and counters

| split | context | target | per event | redraw | exclusion |
|---|---|---|---|---|---|
| `across` | 53 modules (1007 PMTs) | 52 modules (988 PMTs), contiguous in Hilbert order | random offset | up to 3 | counted |
| `within` | even 1-based `pmt_id_in_mpmt` (945 PMTs, 9/module) | odd (1050 PMTs, 10/module) | fixed | none | counted |

- `N_min = 20` target hits for both splits; the counters come from
  `lucid_bombs.ViewExclusionCounter` (`n_min`, `max_redraws`, `n_accepted`,
  `n_excluded`, `across_redraws`, `across_redraw_rate`, both exclusion rates).
- `ViewMasker.report()` adds the cell label, geometry digest, and the mean
  context and target hit fractions over accepted events.
- The occupancy ablation adds `extra_hidden` context positions; `hidden` is
  `target ∪ extra_hidden` in the batch, and the leaked-through set is empty by
  construction.
- Reported hit fractions are over the **natural** views. In the balanced mode
  the visible context is `H` hit + `H` unhit PMTs by construction (fraction 0.5);
  the report also counts `n_balanced_events`.
- **Per-cell**: one `ViewMasker` per (split × drift setting × seed/replicate);
  the report is the cell's counters. The registered `within` accounting is
  9 context / 10 target per module, from the 1-based field, and the tests assert
  it per module.

## Leakage test and outcome

`test_leakage_zero_gradient_and_exact_invariance` runs the context encoder for
both splits with a random scalar projection of the read-out:

- gradient of the context read-out with respect to **every** raw feature at
  every hidden/target PMT: exactly `0.0` (all 9 columns, including log charge,
  time and the hit flag), while the context-position gradients are non-zero;
- flipping the hidden positions' log charge, time and hit flag leaves the
  context read-out bit-identical (`torch.equal`).

The ablation test repeats the gradient check with the balanced context mask.
Outcome: exact zero, no tolerance used.

## Amplitude test and outcome

`test_amplitude_fixture_probe`: a standardised ridge probe (alpha 1.0) recovers
per-PMT log charge from the **unmasked** read-out tokens on hit PMTs, train/test
split by event (4/4 of eight 300-hit events).

**Fixture R² = 0.9990** at random init. This is a fixture, not the criterion:
the registered R² ≥ 0.9 for `sigreg_jepa` and `mae_mse` is measured after the
smoke and again on the final encoders (§3.5.5). The probe helper
(`point_encoder.ridge_r2` / `amplitude_probe`) is deliberately not the
registered probe (that is the v3 §1.3 ridge with run-grouped CV, WP5).

## Memory and parameter count

- trainable parameters: **911,296** (encoder 794,496 + predictor 116,800;
  the Hilbert order and query features are non-persistent buffers, not
  parameters).
- CPU, single-threaded BLAS, torch 2.5.1, 1995 PMTs, one forward+backward step
  (smoke objective, context + full read-out passes):

| batch | step (s) | process peak RSS (MiB) |
|---|---|---|
| 2 | 0.57 | 406 |
| 8 | 2.90 | 1036 |
| 32 | 10.12 | 2844 |

Peak RSS is a high-water mark; a second run of the same script gave batch 32 at
2482–2844 MiB, i.e. about 2.4–2.8 GiB. Baseline process RSS after imports and
model construction is 178.6 MiB.
Batch 32 therefore peaks at about **2.5 GiB**, roughly 4× inside the §6.2 Titan
per-process cap (9.6 GB); **gradient accumulation is not needed** and is
recorded as 1. GPU peak memory and events/s are pending (no GPU in this
session); the CPU numbers are a proxy, not a Titan measurement.

## Determinstic behaviour

`test_ten_step_determinism`: two models built and trained from the same seed
(10 AdamW steps, lr 1e-3 in the test only) give **identical** loss sequences;
the test-only optimiser settings are not a registered convention.

## Pending

- **Real-geometry smoke.** The local tests use `from_synthetic`; the real WCTE
  `config/WCTE_geom_config.json` path (pinned clone) is available only on the
  host. A 32-event real-geometry tokenise + forward/backward smoke is the next
  step, labelled development.
- **WP5 training config review.** Optimiser family, learning rate, betas, weight
  decay, schedule, warm-up and clipping are `pending_review` in
  `configs/point_encoder_v0.json`; no value is invented here. Training length
  (10,000 steps) and batch (32) are the registered §3.5.5 values.
- **The four probe forms** (§3.5.6: per event, run-mean primary, propagated at
  masked positions, raw-input ceiling) are not implemented; only the fixture
  amplitude probe is. The propagated form needs the trained encoder.
- **GPU memory/throughput** and the separate §3.5.5 scaling benchmark
  (2k / 10k / 40k synthetic PMTs) are not run.
- **Registered arm losses** (`mae_mse`, `mae_chi2`, `sigreg_jepa`, the anchor,
  `ijepa_ema`) and the `ijepa_ema` target normalisation are WP5.

## Engineering choices (one line each)

- `across` target = `n_modules // 2` = 52 modules (a cyclic half; 53 would be
  the other rounding).
- Window attention pads the serialised sequence to a multiple of 64 with masked
  pad tokens; valid tokens are unaffected, pad outputs are discarded.
- The window offset is a cyclic roll, so the boundary between the first and last
  window is attended as well.
- One learned mask vector for all hidden positions (the registered "a learned
  mask embedding").
- Predictor attention is full over context + queries; windowing it would cut the
  predictor off from most of the context.
- Predictor queries are the 3 scaled position features only (the registered
  "position queries"), not the facing direction.
- The direction divisor is 1.0 (unitless); positions use R and H/2.
- Charge is floored at 1e-6 PE before the log, so a zero-charge hit is finite.
- The ablation *masks* the subsampled unhit context PMTs rather than removing
  them from the sequence (same information content, simpler accounting).
- muP linear init repeats the `reconstruction_model.model` rule locally because
  that package is no-touch; the mask embedding is N(0, 0.02²).
- The read-out is returned in PMT index order (not serialised order), so probes
  can index tokens by PMT.
- The residual stream carries the raw scaled features, which is why the
  fixture amplitude probe is near 1 at random init; it will drop once the
  encoder is trained and is not evidence for the registered criterion.

## Open questions for the author (not decided here)

1. `across` half: 52 modules (implemented) or 53? The runbook says "half"; this
   is a one-line change and affects the view sizes of every `across` cell.
2. Should the predictor see the full context read-out or a bounded window (cost
   at 40k tokens in the scaling benchmark)?
3. Should the predictor queries include the facing direction as well as position?
4. Are the four `within`/`across` cells expected to use the same `N_min`
   exclusion accounting for probe sets (the runbook excludes training events
   only; `tokenise` counts and drops per batch, so callers must use it only on
   training events until WP5 decides).
