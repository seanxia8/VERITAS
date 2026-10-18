# PMT R-4b, vertex diagnostic and κ_m precision study (18 October 2026)

This follows the author's decisions of 18 Oct 2026, third message, in `docs/PMT_DECISIONS_LOG.md`. The time-feature rule, `(t − window start) / 20 ns`, was recorded and pushed at `fca0af9` before anything ran. All runs use code `e0a9e61`. **Development evidence, not citable.** R-4 (`../pmt_r4_dev_2026-10-18`) stays as recorded.

The full tables are in `summary.md`, produced by `summarise.py`.

## 2×2 diagnostic: validation split only, the test split is never predicted

| Arm | Vertex RMSE ÷ constant, pooled | E_dep MAE ÷ constant, pooled |
|---|---|---|
| 20 ns, dark on | 0.996 (0.991–1.004) | 0.661 |
| 20 ns, dark off | 0.948 (0.936–0.955) | 0.480 |
| 1 µs, dark on (the R-4 data set; same settings) | 0.964 (0.949–0.981) | 0.633 |
| 1 µs, dark off | 0.969 (0.957–0.978) | 0.526 |

- **The time unit does not explain the vertex regression,** and dark counts explain only a little: about 0.05 at 20 ns. Every arm stays at 0.95–1.00, against 0.73–0.88 in `56179eb`.
- **Input check (no training, no scores).** With dark counts off and the 20 ns unit, the inputs match `56179eb`'s bit for bit except the time feature:
  - charge, hit mask and positions are identical;
  - the targets are identical apart from the energy column (initial energy → E_dep);
  - every hit time is shifted by a per-event constant of +9.90 to +10.26 units (spread within an event < 2e-6), while unhit sensors stay at 0.
- **What is left:**
  - the time origin, which puts hits at about +10 units against an unhit 0. The timing information is the same, so this would be a conditioning effect;
  - the energy-target switch;
  - the new model seeds.

  None of these has been tested.
- **Caveat:** in this diagnostic the validation split is used for both checkpoint selection and evaluation, which flatters the numbers equally in every arm.

## Data limitation

Training uses only **480 placements in total** (60 physical events × 8), of which 192 (24 events) are for training.

| Training events per energy | Training placements | Vertex ÷ constant | E_dep ÷ constant |
|---|---|---|---|
| 4 | 96 | 1.023 | 0.888 |
| 6 | 144 | 1.016 | 0.810 |
| 8 | 192 | 0.996 | 0.661 |

- **E_dep looks data-limited:** it improves steadily as training data is added.
- **Vertex does not:** it stays at the constant at every size, so more data at this size would not rescue it. `56179eb`, with the same placements and its different time origin, did learn vertex.

## R-4b on the test events (same gates and seeds 30–32)

| Seed | Vertex, pooled | Vertex, worst energy | Direction | E_dep, pooled | E_dep, worst energy | Verdict |
|---|---|---|---|---|---|---|
| 30 | 0.992 | 1.017 | 10.0° | 0.943 | 1.125 | FAIL |
| 31 | 0.998 | 1.005 | 10.1° | 0.673 | 0.988 | FAIL (vertex) |
| 32 | 1.000 | 1.019 | 10.3° | 0.687 | 0.937 | FAIL (vertex) |

R-4 scored 0.96–1.00 on vertex. R-4b is no better: vertex fails in all seeds, and E_dep also fails in seed 30.

## κ_m precision study (`precision_01/`)

- **Set-up:** 32 independent keyed readout re-draws of the 96 validation placements (12 physical events), with transport and placement fixed, through the frozen R-4b checkpoints. That gives 48 disjoint pairs.
- **Null K:**

  | | Mean | SD |
  |---|---|---|
  | K_vertex | 0.999 | 0.028 |
  | K_energy (diagnostic, pending D-E) | 1.000 | 0.064 |
  | K = max | 1.025 | 0.040 |

- **Result:** 1 + 2 × SD_K = 1.080, so **κ_m = max(1.10, 1.080) = 1.10**, frozen in `configs/lucid_pmt/wp6_contract_v2.json`.
- **Caveats:**
  - SD_K shrinks roughly as 1/√(events per cell), so it should be re-checked if WP7 cells hold fewer than 12 physical events.
  - The energy component alone would give 1 + 2 × 0.064 = 1.128.

## Compute

| Job | Wall time | GPU-hours |
|---|---|---|
| Generation, 4 data sets (CPU, 4 threads, `nice 10`, two in parallel) | 16:52, 17:02, 15:10 and 6:45 | 0 |
| GPU sequence: 7 runs on GPU 3, chosen by the §2 sample in `gpu_sample_pre_r4b.csv` | 338 s in total | 0.094 |
| Precision inference (CPU) | 30 s | 0 |

No new source reads.
