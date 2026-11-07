# Part A — Paper 3 repair pilot (`paper3_repair_pilot_2026-11-06`): STOP record

`registered: false`. Runbook `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md`, sha256
`45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b`, committed unchanged at `062a8d1`.
Implementation `320602ee49eed3339d3611c53d2de35d30dd5af6`; acceptance
`docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json`; independent review
`docs/jepa/REVIEW_2026-11-07_REPAIR_AND_SHORTCUT.md` verdict **PASS** (round 1 BLOCK, round 2 BLOCK,
round 3 PASS).

## Outcome: STOP — entries-preflight failure (Part A §8.4; front matter §F3)

X0-a passed every check except the entries projection:

| check | result |
|---|---|
| pinned data manifests (4 sets) | PASS |
| run `state_sha256` / twin / `events.npz` fields | PASS (200/200 runs on each of the four sets) |
| twin content and `SWtw`–`N` content checks | PASS |
| `mu_p` reconstruction vs stored `expected_pe` | PASS |
| checkpoint inventory (13 core + 6 optional) | PASS, 19/19 hashes verified |
| GPU projection | PASS (2.8409 h core; 3.6014 h with optional; cap 4.0) |
| CPU projection | PASS (5.70 h; cap 6.0) |
| S1 (front matter §F3) | PASS (129.7461 + 4.0 + 1.5 = 135.2461 ≤ 160) |
| **entries** | **FAIL**: free 27,260 − clone 3,059 − suite 1,825 − extension 2,400 = **19,976 < 30,000 floor** |
| closure | PASS |

The free-entry count is the live Ceph quota on `/ceph/dwong` (guard snapshot 2026-11-07T03:58:55Z,
level `HOLD`, reason `free entries 27260 < 30000; write no new data`). The runbook's 37,986 free at
`ac8e220` is stale: another project (`hcp_xopt_20261106`, 9,900 entries, created 6 Nov 13:34) has
consumed the difference. No deletion of another project's data is authorised, so the STOP stands.
No GPU job was launched. The live guard was also `HOLD` for the GPU start condition.

## What was executed (independent item)

- **X3-a — Fisher accounting** (`x3_fisher.json`): 1,000 photon events, 5 model rows (C-mean,
  C-mean+occupancy, C-var_all, C-var_ge1, C-var_ge3); **P0 passes** (max efficient/known ln Y charge-
  information ratio 1.21e-15 ≤ 1e-6); closure pass. X3-a depends on nothing (§8.4), so it ran as the
  independent item after the STOP, CPU only.

## Not run (blocked by the STOP)

X0-b, X1, X2, X3-b and the full report. There are no X1/X2/X3-b readings; their report sections do
not exist and nothing is reconstructed, estimated or qualified.

## Execution-time adapter (documented in DECISIONS.md)

The accepted preflight crashed writing `inventory.json`/`plan.json` because a nested `os.PathLike`
reached `json.dumps(default=float)`. A runtime adapter outside `<P>`
(`/ceph/dwong/jepa/tmp/jepa_rb_20261107/fix/wrap_repair.py`) stringifies `os.PathLike` in
`ext_repair._jsonable` only and then runs the unmodified CLI, gate and mode body — same outputs, no
reading/threshold/path/logic change, gate (c) unaffected. A proper in-`<P>` fix needs a re-review and
a new acceptance record.

## Files

`plan.json` (X0-a, full §4 payload), `inventory.json` (§2.3 rows and sha256s), `x3_fisher.json`
(X3-a), `STOP.json` (this record). No cached embeddings, checkpoints or figures exist. Large arrays:
none (the runbook's `LARGE_ARRAYS.json` is not applicable because no embedding cache was written).

## Resources

0.0 of the 4.0 L40S h cap (no GPU job); CPU ≈ 2.0 h of the 6 h cap (implementation/fix rounds,
focused tests, the full suite on the final SHA, both dry runs, the review rounds, X0-a and X3-a).
L40S ledger unchanged at 129.7461/170; S1 135.2461 ≤ 160.
