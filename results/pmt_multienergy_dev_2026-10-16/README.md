# Multi-energy prompt-track development evidence

Development/non-citable. See [report](../../docs/PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md), [author decisions](../../docs/PMT_CALIBRATION_AND_PROTOCOL_DECISIONS_2026-10-16.md) and [final configuration](../../configs/lucid_pmt/multienergy_pilot.json).

- `planned_24_event_config.json`: original cohort plan before the source-cap STOP.
- `source*/`: progress, resume accounting and balanced source manifest. Array files remain remote; hashes identify every retained event.
- `extraction_*.log`, `extractor_attempt02.py`, `execution_scripts/extractor_attempt01.py`: retained I/O attempts. The current extractor is in the task package. The source-cap STOP is a STOP; this run did not expand its cap to complete 24 events at 1,500 MeV.
- `track_closure_01/`: failed 16-candidate transport run; one independent sensor-assignment mismatch. No model trained on these digits.
- `diagnosis_02.json`: isolated comparison of 16 and 64 candidates. The correct first sensor was absent from the 16-candidate list.
- `track_closure_02/`: full rerun with 64 candidates, unchanged source events, placements, targets, splits and tolerances.
- `training_01/`: all three fixed model seeds, histories, per-energy/containment metrics, held-out predictions and disjoint response-fit charge calibration. Negative gate outcomes remain in the report; they are not rescued by changing thresholds.
- `runtime_receipt.json`, `receiving_checks.json`: source/code/config/dataset/checkpoint hashes, split audit, finite-value checks and independent endpoint recomputation.
- `execution_scripts/`: audit/receiving/bundling commands and exact earlier transport/data snapshots.
- `local_tests.txt`, `remote_tests_repaired_02.log`: final regression receipts. Earlier syntax-failure receipts remain visible; the duplicate provenance keyword was fixed before the repaired generation.

Remote root: `/work/dwong/pmt_multienergy_20261016`. Large source/digit arrays and checkpoint weights stay there; neither this folder nor the small validation sample establishes a target-PMT calibration or a scientific harm result. The single contained test placement does not support a contained-track performance estimate.
