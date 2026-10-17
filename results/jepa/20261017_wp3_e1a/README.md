# WP3 E1a results

Campaign `20261017_wp3_e1a`, runbook v4 §3.1/§3.2/§8 WP3. Cells/rows: d=40, k=3, θ scale
{.1,.4,1} × η scale {1,3} × γ\_load {0,.1,.25,.5,.75,.9,1} × n {1e3,1e4,1e5} × seeds 0–4
= 630 rows, ten registered method rows each.

## Final table

`attempt_007_20261017T122356Z_E1a/` — 630/630 rows `pass`, written by the compact deterministic
writer. `manifest.jsonl` has one provenance row per executed row; `attempt.json` records run id,
code SHA (`9d59762efb6ee1e4b9517d768ada335cde62378f`), config SHA-256, worker count (1), wall
time, consumed worker-seconds, per-status counts and start/end source hashes. `SHA256SUMS` binds
every row file, the manifest and `attempt.json`.

## Superseded attempts (archived, not deleted)

`SUPERSEDED_ATTEMPTS_001_006.tar.xz`
SHA-256 `2c98e2bc36961807c7bbaab61a116374c53cc62688586c3c1b66c7e399aea1cc` (2 594 160 bytes).

It contains the six earlier write-once attempt directories unchanged (1 266 row JSON files plus
their manifests/attempt records): smoke, the first full grid, two retry attempts, and the
corrected-code attempt. They are superseded because the first implementation reported the
conditional-oracle row using the whitened-coordinate functional instead of the raw pulled-back
context map; that implementation defect was found while verifying R3b, fixed, and the whole grid
re-executed. Per the write-once rule no row was ever overwritten; the superseded rows are preserved
bit-exactly in this archive so that a reviewer can extract and re-check them.

## Provenance notes

- The R3 transform is frozen in `src/jepa_noise/configs/e1_grid_v4.json` (`r3_transform`).
- Two of 210 R3 transformed-CCA diagnostics raise the reviewed solver's fixed
  isotropy/objective residual check; the rows and the failure are retained (no solver setting was
  changed). `aggregate_E1a.json` lists the failing row keys.
- Compact JSON is used to keep this directory inside the v4 §7 size limit (17.6 MB of file bytes).

## Regeneration

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/private/tmp/jepa-venv-cpu311/bin/python -m jepa_noise.report E1
```
