# NHits time-origin rule: validation (18 October 2026)

This follows the author's fifth message of 18 Oct 2026 and E43 in `docs/PMT_DECISIONS_LOG.md`. The design and the criterion were recorded at `1d9845f`, before the run.

- Rule: N = 3, a 20 ns window, origin `"nhits"`, 20 ns unit.
- Deposited-energy target, seeds 20–22.
- **Validation split only;** the test split was never predicted.
- Caveat: the validation split also selects the checkpoint.

| Arm | Vertex ÷ constant, pooled (seeds 20 / 21 / 22) | Worst energy | Direction (°) | E_dep ÷ constant | Fallback |
|---|---|---|---|---|---|
| Dark off | 0.797 / 0.748 / 0.814 | 0.82–0.86 | 7.2–7.5 | 0.48–0.52 | 0 of 480 |
| Dark on | 0.886 / 0.921 / 0.924 | 0.98–1.03 | 8.3–9.1 | 0.46–0.61 | 0 of 480 |

## Criterion (E43): FAIL, so stop
- **Dark off meets it:** every seed is ≤ 0.88, and the fallback rate is 0%.
- **Dark on does not:** every seed is above 0.88.
- So R-4c was not run on the test events.

## What it shows
- **The rule fixes the conditioning problem without dark counts.** Dark off gives 0.75–0.81, close to arm 0 (first digit, 0.78) and below the window-start origin (0.94).
- **With dark counts on, vertex loses about 0.12.** For comparison, the R-4b 2×2 lost about 0.05 at the window-start origin. With dark on, NHits gives 0.91 against 0.996 for the window start (seeds 30–32), so it is better but not enough.
- **The trigger itself works with dark on:** no fallbacks, and a dark false trigger is rare (about 1.2×10⁻³ per window; E43).
- **Hypothesis, untested:** under the NHits origin, the roughly 2 dark digits per event carry time features from about −15 to +35 units, against prompt hits near 0–2. The subject model's masked max pooling can let a single such outlier set a feature. Under the window-start origin every time sits in 0–50, so outliers stand out less.

## Compute
- GPU: 2 arms × 3 seeds on GPU 0, 111 s in total, 0.031 GPU-hours. GPU 0 was chosen by the §2 sample in `gpu_sample_pre_nhits.csv`.
- Generation: 14:46 and 14:18 (CPU, 4 threads, `nice 10`, two in parallel).
- No new source reads.
