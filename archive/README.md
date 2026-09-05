# archive/

Code, scripts and notes that are no longer on the active path. Kept for the
record, not maintained, not imported by anything. Superseded *documents* live
in `docs/archive/` instead.

| what | was | why it is here |
|---|---|---|
| `code/reconstruction_model_legacy/` | `src/reconstruction_model/legacy/` | scripts preserved from the old reconstruction-model copy; nothing imports them (2026-09-05) |
| `code/legacy_trace_scripts/` | `reference/legacy_scripts/` | pre-`qp_simulator` trace-generation helpers; referenced only from an archived doc |
| `scripts/delight_training_2026-08/` | `scripts/{train_triangular_*, submit_small_range_experiment.sh, submit_transfer_method_tests.sh, condor_method_probe.sh, submissions/, submit_ref/}` | the August DELight training experiments (triangular-pairwise, small-range, transfer-method probes) and their Condor submit files; last touched 2026-08-15, not referenced by the live `train_compact_l40s` path |
| `notes/finetuning/` | `reference/finetuning/` | LLM fine-tuning notebooks (verl, FSDP, DDP); unrelated to ORACLE |

Left in place on purpose, because they may still be the Tier-1 subject's
training path: `scripts/train_full_a100/`, `scripts/train_full_l40s/`,
`scripts/full_training_probe.sh`, `scripts/run.sh`, `scripts/train_local_l40s.*`.
Decide those when the compact subject's training is final.
