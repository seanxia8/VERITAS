# R-4c: test events, once (18 October 2026)

This was authorised by the author's sixth message of 18 Oct 2026, conditional on E45 arm (a) passing on validation, which it did (E46). The design was recorded at `5f4ecc0` before any run. **Development evidence, not citable.**

- Rule: NHits trigger with N = 4 in a 20 ns window, plus the prompt window [−20, +60] ns.
- Setup: dark counts on, deposited-energy target, 20 ns unit.
- Seeds 30–32, with the same gates and checkpoint rule as R-4 and R-4b.
- Data: the E45 arm (a) data set (`e45_data_a`), whose test events had never been predicted.

| Seed | Vertex, pooled | Vertex, worst energy | Direction | E_dep, pooled | E_dep, worst energy | Verdict |
|---|---|---|---|---|---|---|
| 30 | 0.812 | 0.88 (500 MeV) | 7.8° | 0.642 | 0.67 | FAIL (vertex pooled) |
| 31 | 0.877 | 0.93 (1000 MeV) | 9.2° | 0.570 | 0.72 | FAIL (vertex pooled) |
| 32 | 0.883 | 1.02 (500 MeV) | 7.8° | 0.640 | 0.69 | FAIL (vertex pooled and 500 MeV) |

The gates are vertex ≤ 0.8 × constant pooled and ≤ 1.0 × in each energy; direction < 20°; E_dep ≤ 0.8 × pooled and ≤ 1.0 × in each energy.

**R-4c fails the vertex gate in all seeds.** Direction and E_dep pass everywhere.
- Vertex moves from 0.96–1.00 (R-4) and 0.99–1.00 (R-4b) to 0.81–0.88.
- The verdict is retained: nothing is retuned, and there is no further test-event run.

Compute: 56 s on GPU 0 (0.016 GPU-hours). GPU 0 was chosen by the 5-minute §2 sample in `gpu_sample_pre_r4c_test.csv`.
