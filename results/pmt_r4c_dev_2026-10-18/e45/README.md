# E45: NHits N = 4 + prompt window, validation (18 October 2026)

This follows the author's sixth message of 18 Oct 2026 in `docs/PMT_DECISIONS_LOG.md`. The rule, the arms and the pass mark were committed at `5f4ecc0`, before any run. **Development evidence, not citable.**

- Rule: `readout.nhits_trigger_time` with N = 4 in a 20 ns window, and `readout.prompt_window_mask` with a [−20, +60] ns window. Out-of-time digits are dropped from the model input. With no trigger, nothing is dropped.
- Setup: deposited-energy target, 20 ns unit, seeds 20–22.
- **Validation split only;** the test split was never predicted.
- Caveat: the validation split also selects the checkpoint.

The tables are in `summary.md`, made by `summarise.py`.

## Result: PASS
| Arm | Vertex ÷ constant, pooled (20 / 21 / 22) | Mean |
|---|---|---|
| (a) N = 4 + prompt window, dark on | 0.840 / 0.789 / 0.807 | 0.812 |
| (b) the same, dark off | 0.797 / 0.748 / 0.814 | 0.786 |
| (c) dark on, truth-dark digits removed (upper bound; never a fix) | 0.775 / 0.755 / 0.824 | 0.785 |

- **Pass mark:**
  - every seed of (a) is ≤ 0.88 (maximum 0.840);
  - mean(a) − mean(b) = +0.026, within ≤ 0.05.
- Fallback: 0 of 480 in every arm.

## Dropped digits, arm (a)
- 710 of 197 455 digits in all roles, and 147 of 39 557 in validation.
- **Source digits:** 32 were dropped. Every one also held a dark photoelectron that set its time before the trigger. **Pure-source digits dropped: 0.**
- Arm (c) removed 736 truth-dark digits. Its window then dropped only the same 32 dark-timed source digits.
- Arm (b)'s checkpoints are byte-identical to E44's dark-off checkpoints. Without dark counts, N = 4 gives the same trigger as N = 3, and the window drops nothing.

## Compute
- Generation (CPU, 4 threads, `nice 10`, at most two at a time): 14:32, 14:43 and 13:22.
- GPU: 3 arms × 3 seeds on GPU 0, 169 s in total, 0.047 GPU-hours. GPU 0 was chosen by the §2 sample in `gpu_sample_pre_e45.csv`.
- No new source reads.
