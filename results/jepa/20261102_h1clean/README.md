# H1-clean causal test (predictor position-query noise) — 2 Nov 2026 — STOPPED at dry-run

`registered: false`. Extension `h1_clean_2026-11-02`, variant `sigreg_jepa_query_noise`
(registered `arm_name` stays `sigreg_jepa`).
Runbook: `docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md` (revision 3, sha256
`e02214204b7a3965665faa459adc718d2f9c061860ef66a1cb84a3d1940812c6`), committed unchanged.

**Status: stopped at the pre-declared §3.4 manipulation check. No smoke and no training run was
started** (the author line is present and the hard gate passed; the data preflight and the
checkpoint/hash manifest passed). This is a runbook §3.4 / §0.1 stop condition, not a code
failure: the query-noise patch is verified by the §3.1 tests and the encoder input is
byte-identical at every sigma.

- code SHA: `5ae6d6299fe6c7ce40aceb197a1a124f8e0db172` (deployed); full `src/jepa_noise/tests`
  suite on that SHA: **577 passed in 588.21 s** (deepthought)
- hard gate (§0): PASS (committed runbook §0 + `DECISIONS.md` carry the author line)
- §3.5 data preflight: PASS (dev_train `b1b3f4…`, probe-train `093ea894…`, probe-test
  `c11edc71…`; read-only, nothing regenerated)
- checkpoint/hash manifest (Q1/Q2 items used by evaluation/reproduction): PASS
- §3.4 manipulation check: **FAIL at both sigma = 0.10 and the single ladder step 0.15**

## §3.4 manipulation check (untrained seed-9000 arm, fixed saved batch, step 0, 32 events)

| sigma | prediction MSE without noise | with noise | ratio | pass (>= 1.5) |
|---:|---:|---:|---:|---|
| 0.10 | 4.801727 | 4.810290 | 1.001783 | False |
| 0.15 | 4.801727 | 4.820297 | 1.003867 | False |

## Diagnostic: the criterion needs sigma ~1.8–2.0 (pre-declared ladder is 0.10 → 0.15)

Same untrained arm and batch, sigma sweep (ad hoc analysis only; the variant is unchanged):

| sigma | ratio | max abs prediction change | relative prediction change | `encode()` identical |
|---:|---:|---:|---:|---|
| 0.10 | 1.001783 | 1.3626 | 0.1162 | True |
| 0.15 | 1.003867 | 2.0609 | 0.1740 | True |
| 0.30 | 1.015003 | 4.2086 | 0.3444 | True |
| 0.50 | 1.041663 | 5.5429 | 0.5582 | True |
| 0.75 | 1.093906 | 7.7592 | 0.7998 | True |
| 1.00 | 1.167137 | 9.9359 | 1.0243 | True |
| 1.50 | 1.377245 | 14.1869 | 1.4559 | True |
| 2.00 | 1.673103 | 18.3979 | 1.8834 | True |

The untrained predictor is dominated by the context path (its muP-initialised output is small), so
the pre-declared threshold `ratio >= 1.5` is only reached at sigma ≈ 1.8–2.0 scaled units
(≈ 2.8–3.1 m in x/y and 2.5–2.8 m in z — 13–18× the pre-declared sigma, far above one PMT pitch
and the 0.160 m vertex floor). With sigma = 0.10/0.15 the noise removes almost none of the query
information of the untrained arm, so the test as pre-declared is non-discriminating; per §3.4 the
runs are not launched and the result is **`unclear (non-discriminating)`** — not evidence for H2,
and no H1-clean-r reading exists for either seed.

## Outputs

- `plan.json` — hard gate, data/checkpoint preflight and the §3.4 manipulation check (with the
  full ladder record); `sigma_used: null`, `manipulation_pass: false`.
- `manipulation_sensitivity.json` — the sigma-sweep diagnostic above.
- `README.md` — this file. No `runs.json`/`readings.json`/`reproduction.json`: no run was made.

## Resources

0 L40S GPU-h (no smoke, no training, no evaluation). CPU (deepthought/deepthought2): dry-run +
diagnostics + the full suite, conservative 0.35 h charged to `lucid_generation_cpu`.
No registered file, config, prediction or verdict was touched; nothing outside `/ceph/dwong/jepa/`
was written by the hosts. Next step is the Lead's decision (the label-centering variant is a
separate, not-started runbook).
