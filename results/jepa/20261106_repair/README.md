# Part A — Paper 3 repair pilot (`paper3_repair_pilot_2026-11-06`): executed

`registered: false`. Runbook `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md`, sha256
`45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b`, committed unchanged at `062a8d1`.
Final implementation `0cf61ff476bc8f3463b4595816ba40508cba239c` (acceptance
`docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json`, independent review
`docs/jepa/REVIEW_2026-11-07_REPAIR_AND_SHORTCUT.md`, **PASS**; six reviewed fix rounds).

This run first stopped at the entries preflight (see `STOP.json`). After the author's 7 Nov
archiving authorisation (free entries 27,205 → 44,286, all archived targets tar+sha256+member-count
verified) both preflights passed and the runbook was executed to completion.

## Executed items

| item | result |
|---|---|
| X0-a preflight | PASS (data hashes 4/4, run states/twins/events 200/200 per set, twin+SWtw content, mu_p reconstruction, inventory 19/19, GPU 2.8409 h, CPU 5.70 h, S1 135.2461, entries 44,286 → projected 36,901 ≥ 30,000) |
| X0-b reproduction | PASS 9/9 models within 1e-3 m (max |Δ| 4.8e-7 m), identity check PASS (54 s, cached embeddings) |
| X1 | COMPLETE 19 models (13 core + 6 optional), no unavailable; 1.0622 GPU-h |
| X2 | COMPLETE 13 core models, no unavailable; 0.4972 GPU-h |
| X3-a Fisher | COMPLETE; P0 PASS (max efficient/known ln Y information ratio 1.21e-15) |
| X3-b | COMPLETE raw / init9000 / leace families, no unavailable; 0.1483 GPU-h (three earlier attempts failed on execution defects, all booked) |
| report | COMPLETE (6 figures, closure pass) |

## Readings

`READINGS_SUMMARY.md` holds the X1/X2/X3-b reading tables and the X3-5 theory rows; `SUMMARY.json` the
machine-readable extract. The full per-event payloads (`x1.json` 101 MB, `x2.json` 34 MB, `x3.json`
84 MB, `readings.json` 15 MB, the auto-generated README 10.6 MB) stay on
`/ceph/dwong/jepa/returns/ext_repair_20261106/{x0,x1,x2,x3}` with sha256 in `LARGE_ARRAYS.json`.

## Resources

2.1974 of the 4.0 L40S h cap (X0-b 0.1849 incl. two failed attempts, X1 1.0622, X2 0.4972, X3-b
0.3729 incl. three failed attempts); CPU ≈ 2.6 of 6 h. L40S ledger 129.7461 → ≈132.0 / 170.
No data generation; no training.

## Execution-time adapters

Two X3-b serialisation fixes were applied through the documented runtime adapter outside `<P>`
(`/work/dwong/jepa_tmp_20261107/adapter2/jepa_repair_adapter2.py`, execution fixes 7–8 in
`DECISIONS.md`): `_x3_five` treats a null (JSON) efficient CRLB as infinity, and `_strip_internal`
also drops probe-fit/prediction objects so `x3.json` can be written. Folding them into
`ext_repair.py` needs a re-review and a new acceptance record (same process as fixes 1–6).
