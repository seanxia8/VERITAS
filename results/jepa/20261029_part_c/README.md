# Part C checkpoint evaluation — 29 Oct 2026

`registered: false`. Grid remains paused. One clean invocation of all six checkpoints at `f29137c18bbb740aabc44a1abdadd28dde6edade`; prior failed attempt retained. Full local CPU suite 496 passed in 153.32 s; Deepthought CPU smoke 25 passed in 12.98 s. Guard was fresh OK and quota 68551 free at launch.

The copied files and their host-generated SHA256SUMS verified 3/3. `inspect.json` has `complete: true`, exactly six items, and `anc100.beta = 100.0`; launcher exit 0. Independent absolute errors for old9000, old9001, fix9000, fix9001, mae9000 respectively: 0.00000105742, 0.000000294462, 0.000000776079, 0.000000394961, 0.000000187649 m; all <= 0.001 m. The earlier init9000 retry separately passed (see `../20261026_pregrid_resume/README.md`).

Evaluation mismatch reading is **not supported**: fixed seed 9000 half/full 0.325456977/0.219258224 = 1.484355; seed 9001 0.293429148/0.204281395 = 1.436397, both above 0.9.

This attempt consumed 4239.36 s = 1.1776 Titan GPU-h, bringing the runbook Titan total including failures to 2.590016667 GPU-h. Part D and E remain pending. No grid decision or grid run.

Subsequent note: D and E have since completed; their evidence and current status are in `../20261029_pregrid_report.md`. The original Part C evidence files are unchanged.
