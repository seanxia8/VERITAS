# Vertex diagnostic E40 (18 October 2026)

This follows the author's fourth message of 18 Oct 2026 in `docs/PMT_DECISIONS_LOG.md`. The design was recorded and pushed at `65fba28`, before any arm ran. Generation and training use code `91ef3f4`, which differs from `65fba28` only by the variable rename below. **Development evidence, not citable.** R-4 and R-4b stay as recorded.

- **Validation split only.** The test split was never predicted.
- **Caveat:** the validation split also selects the checkpoint, which flatters every arm equally.
- All arms: dark counts off, 20 ns unit, seeds 20–22, and the `56179eb` loss and checkpoint rule.
- Full tables: `summary.md`, made by `summarise.py`.

## Result

| Arm | Time origin | Energy target | Vertex RMSE ÷ constant | Direction (°) | Energy MAE ÷ constant |
|---|---|---|---|---|---|
| 0 | first digit (`56179eb`) | initial | 0.782 (0.725–0.850) | 7.5 | 0.823 |
| 1 | trigger time | initial | **0.681** (0.654–0.715) | 6.8 | 0.857 |
| 2 | trigger time | deposited | **0.652** (0.620–0.707) | 6.8 | 0.441 |
| 3 | window start | initial | 0.961 (0.949–0.975) | 9.8 | 0.831 |
| 4 | window start | deposited | 0.936 (0.933–0.938) | 10.2 | 0.496 |

All values are means over seeds 20–22, with the range in brackets. Arms 0, 1 and 3 fail the energy gate by construction, because the gate requires the D-E endpoint. Arm 2 passes every gate in all three seeds, but that is on the validation split, which also selected the checkpoints.

### Arm 0 reproduces `56179eb` exactly
- Its inputs and targets are bit-identical to the `56179eb` data set (`inputs_arm0_vs_56179eb.json`).
- All three checkpoints are byte-identical to the recorded `56179eb` hashes (`checkpoints.sha256`).
- Scored on validation, the `56179eb` checkpoints give the same 0.725–0.850 (`reference_56179eb.json`). That is within the author's 0.73–0.88 range.
- **So there is no code regression, and no bisection was needed.** It also shows that the new loss logging leaves the training trajectory unchanged.

### The cause is the time origin
- Moving the origin from the trigger time to the window start (+10 units on every hit, with unhit sensors at 0) raises the vertex ratio:
  - initial-energy target: from 0.68 to 0.96;
  - deposited target: from 0.65 to 0.94.
- Direction also worsens, from 6.8° to 9.8–10.2°.
- **The timing information is identical:** arms 1 and 3 differ only in the time column, by a constant 10.000 on hit sensors (`inputs_arm1_vs_arm3.json`). The effect is input conditioning.
- At initialisation, the window-start origin multiplies the vertex head's gradient on the shared encoder by about 5 (2.61 against 0.54), and the vertex loss starts at 1.43 × constant instead of 1.20 ×.

### The energy target does not starve the vertex head
- **Scale:** the deposited target has a smaller spread. Its log SD is 0.19, against 0.45 for initial energy, so energy's share of the constant loss is 5% instead of 24%.
- **At initialisation:** the network outputs are about 0, far from the deposited target's mean. So energy dominates the initial loss (54% against 17%) and the encoder gradient (1.76 against 0.20).
- **At the selected checkpoint:** energy takes 4–14% of the training loss, and vertex 81–92%.
- **Effect on vertex:** vertex is slightly *better* with the deposited target (0.65 against 0.68 with trigger origin; 0.94 against 0.96 with window start).
- So the early imbalance does not persist, and vertex is not starved.

### Seeds and the earlier arms
- Arm 4 repeats R-4b's `diag_u20_nodark` with seeds 20–22 instead of 30–32: 0.936 against 0.948. The seed effect is small.
- Arm 3's inputs equal R-4b's `data_u20_nodark` bit for bit.

### Why trigger beats first digit (0.68 against 0.78)
In this simulation the trigger time t = 0 is the true interaction time. Absolute hit times then carry the vertex-to-sensor distance, which a first-digit origin removes. A real trigger time is derived from hits and has jitter, so arm 1 is an optimistic bound for any operational trigger.

## Bug found and fixed during the run
- The first arm 2 generation (code `65fba28`) failed on its second placement. In the deposited-energy branch, `origin=raw['vertex_m']` shadowed the new time-origin variable.
- Fixed at `91ef3f4` by renaming the variable to `time_origin`.
- Arms 0, 1 and 3 never enter that branch, so their data were unaffected. All arms were then trained with `91ef3f4`.
- The failed log is kept in `data/generation_arm2_failed.log`.

## Compute

| Job | Wall time | GPU-hours |
|---|---|---|
| Generation: 4 data sets plus the arm 2 retry (CPU, 4 threads, `nice 10`, at most two at a time) | 14:34, 14:45, 13:15, 0:56 (failed), 13:17 (retry) | 0 |
| GPU: 5 arms × 3 seeds on GPU 0, chosen by the §2 sample in `gpu_sample_pre_r4c.csv` | 280 s in total | 0.078 |
| `56179eb` validation reference and input comparisons (CPU) | < 2 min | 0 |

No new source reads.
