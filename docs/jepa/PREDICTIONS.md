# JEPA v3 prediction status

Development output; no prospective grid evidence yet. Analytic fixtures and the disclosed toy are reported separately.

| Prediction | Campaign | Status | Reason | Evaluation seeds | Numeric evidence |
|---|---|---|---|---|---|
| R1 | E1 | fail | evaluated_from_stored_tables | 0, 1, 2, 3, 4 | results/jepa/20261017_wp3_e1a/aggregate_E1a.json; cells 6/6 pass; max cell median angle 4.175 deg; E1b channel clean-cell median 17.309 deg (threshold 10 deg, fails above it) |
| R2 | E1 | fail | evaluated_from_stored_tables | 0, 1, 2, 3, 4 | results/jepa/20261017_wp3_e1a/aggregate_E1a.json; scale cells with Spearman>=0.8: 5/6; required cell median 85.984 deg; E1b channel Spearman 1.000 (>=0.8), seed-mean angle at g=.05 77.329 deg (>=45) |
| R3 | E1 | not_evaluable | r3_solver_tolerance_failure_2_of_210_rows | 0, 1, 2, 3, 4 | none |
| R3b | E1 | pass | evaluated_from_stored_tables | 0, 1, 2, 3, 4 | results/jepa/20261017_wp3_e1a/aggregate_E1a.json; eligible cells 42; pass 42; annihilation control rank-ineligible (outside primary grid); corrected-implementation outcome (oracle pullback fix; superseded attempts archived) |
| R4a | E1 | fail | evaluated_from_stored_tables | 0, 1, 2, 3, 4 | results/jepa/20261017_wp3_e1a/aggregate_E1a.json; cells 32/42 pass |
| R4b | E1 | fail | evaluated_from_stored_tables | 0, 1, 2, 3, 4 | results/jepa/20261017_wp3_e1a/aggregate_E1a.json; cells 33/42 pass at beta=10; beta=0.1/1 in table; E1b diagnostics results/jepa/20261017_wp3_e1b/aggregate_E1b.json |
| R5 | E1 | not_evaluable | seed_evaluable_lt_4 | 0, 1, 2, 3, 4 | none |
| B1 | E2-bridge | fail | not_every_seed_seed_0_only; abs(delta mean)=0.00942 below 2x seed floor 0.10254 (plain 0.525128 vs anchored 0.515705 nuisance R2) | 0, 1, 2 | results/jepa/wp4a_grid_combined/aggregate.json; docs/jepa/WP4A_BRIDGE_RESULTS.md |
| R6 | E2 | not_run | v4 A4/A7 redesign | none | none |
| R7 | E2 | not_run | v4 A4/A7 redesign | none | none |
| R7b | E2 | not_run | v4 A7/A8 | none | none |
| R8 | E2 | not_run | v4 A4/A7 redesign | none | none |
| R11 | E2 | not_run | v4 A7/A8 | none | none |
| R11b | E2 | not_run | v4 A7/A8 | none | none |
| R12 | E2 | not_run | v4 A7/A8 | none | none |
| R9 | E3 | not_run | G6_manifest_and_E2_beta_transfer_pending | none | none |
| R10 | E4 | not_run | WP5_acceptance_pending | none | none |
