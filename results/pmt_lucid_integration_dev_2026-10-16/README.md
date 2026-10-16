# PMT LUCiD integration development evidence

See `../../docs/PMT_LUCID_INTEGRATION_REPORT_2026-10-16.md` for interpretation, commands and limitations. All outputs are development/non-citable.

- `closure_01`: retained failed first transport closure; no training used it.
- `closure_02`: passing corrected prompt transport and cone check.
- `controls_01`: real-photon paired invariants.
- `training_01`: three baseline-only direction seeds, saved predictions and histories.
- `source_audit.json`, `manifest.json`: source identity/count checks and per-event hashes.
- `runtime_receipt.json`, `local_source_manifest.json`, `receiving_checks.json`: executed source binding and independent metric receiving.
- `checkpoint_verification_02.json`: actual remote checkpoint hashes verified. The empty first-attempt receipt is preserved; that verification script had a quoting syntax error and ran no model.
- `execution_scripts/`: actual extraction and diagnostic scripts. The packaged extraction CLI adds streaming bounds and inline source audit.
- `local_tests.txt`: 94 pass, one optional skip; `remote_pmt_tests.log`: 95 pass.

The 5.4 MiB digit dataset, source photon subset and checkpoints remain under `/work/dwong/pmt_lucid_20261016` on Deepthought. This local evidence stores hashes and small evaluation artifacts rather than duplicating the source photon dataset. No task jobs remain running.
