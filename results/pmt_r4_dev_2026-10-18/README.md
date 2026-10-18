# PMT R-4 development re-run (18 October 2026)

Runbook v2 §4 R-4, under C2 (`PMT_REMOTE=yes`, `PMT_GPU=yes`, `PMT_DATA_CAP_GB=5`) and the §2 shared-server rules, following the author's instructions of 18 Oct 2026 (`docs/PMT_DECISIONS_LOG.md`). **Development evidence, not citable.** The 60 physical events and 480 placements are those of `56179eb`, whose scores were already inspected, so this is not a confirmation.

## Steps

1. **Containment, range-based** (`containment/`). μ− at 100–350 MeV kinetic reach ≥ 50% containment in WCTE; 400 MeV does not. Details are in `containment/containment.json` and the decisions log.
2. **Primary segments** (`segments_01/`).
   - The extraction read only metadata branches (no photons) from CERNBox: **91.2 MB** for all 60 events, in 56 s on CPU.
   - All primary tracks are continuous and recorded down to β = 0.
   - Median Geant4 path lengths are 2.31 m (500 MeV), 4.74 m (1000) and 7.04 m (1500). The CSDA ranges are 2.30 m and 4.73 m.
   - The per-event segment `.npz` files stay on the host.
3. **Data generation** (`track_closure_01/`), 14 min 26 s on CPU (4 threads, `nice 10`).
   - Inputs: the existing photon sources, the code at commit `72985f2`, the manifest response (with 1 kHz dark counts), and the D-E target.
   - The transport gate passes. Readout: no overflow, and 21.1% of digits come before photon arrival.
   - E_dep median (min–max): 342 (213–500) MeV at 500 MeV, 345 (225–485) at 1000 MeV, 368 (220–553) at 1500 MeV.
   - Escaping muons are close to minimum-ionising, so E_dep mostly measures path length inside the volume and hardly changes with initial energy.
   - Only 3 of 480 placements stop inside, all at 500 MeV.
   - PhotonSim's `Segment_Edep`, summed inside the volume, is about 0.7 × the β-based E_dep (secondaries carry the rest). It is a cross-check only.
4. **Training** (`training_01/`): 3 seeds (30, 31, 32), 100 epochs each, one GTX TITAN X (GPU 3, chosen by the §2 rules; see `gpu_sample_pre_training.csv`), 25% memory fraction, peak 237 MB. **Wall time 55 s; 0.015 GPU-hours.** Checkpoints and `digits.npz` stay on the host, with hashes in `../PMT_CHECKPOINTS.json`.

## Gate table against the constant predictor (R-3, energy per subgroup)

See `gate_table.md`, produced by `gate_table.py`. Summary, as model ÷ constant:

| Seed | Vertex, pooled | Vertex, worst subgroup | Direction, pooled | E_dep, pooled | E_dep, worst subgroup | Verdict |
|---|---|---|---|---|---|---|
| 30 | 0.965 (FAIL, limit 0.8) | 0.973 (1500 MeV) | 9.0° | 0.644 | 0.785 (500 MeV) | FAIL (vertex) |
| 31 | 0.960 (FAIL) | 0.974 (1000 MeV) | 9.6° | 0.672 | 0.731 (1500 MeV) | FAIL (vertex) |
| 32 | 0.998 (FAIL) | 1.033 (1500 MeV, FAIL) | 9.5° | 0.877 (FAIL) | 1.315 (500 MeV, FAIL) | FAIL (vertex, energy) |

- **Direction:** passes in every seed and subgroup (8.4–11.2°).
- **E_dep:** passes pooled and per subgroup in two of three seeds.
- **Charge calibration:** fitted on the response-fit split only, a straight charge-to-E_dep fit reaches 0.091 relative MAE, as good as the model (0.082–0.112). The model adds nothing beyond total charge on this endpoint.
- **Vertex:** fails in every seed, barely better than the constant. That is worse than `56179eb`, which reached 0.73–0.88 on the same placements.
- The failure is retained, and nothing was retuned.

## Interpretation, not tested

The vertex regression lines up with two input changes made after `56179eb`:
- **E33:** the time feature went from `(t − first digit) / 20 ns` to `(t − window start) / 1000 ns`, which squeezes the prompt timing structure (about 30 ns) into roughly 0.03 of the feature range, a fiftyfold compression.
- **R-6:** about 2 dark hits per event.

Which one is responsible is untested. Testing it against these test events would mean tuning on inspected scores. The proposal for the author is an ablation on the validation split only: time scale 20 ns versus window length, dark counts on versus off.
