# Shortcut-collapse follow-up (Part B) — 7 Nov 2026

`registered: false`. Extension `shortcut_followup_2026-11-07`, variant `target_posfree` (`arm_name` stays `sigreg_jepa`).
Runbook: `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` (sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b`).
Author-approved training: exactly two 10 000-step runs (seeds 9000/9001) and one 20-step smoke; no other training, no data generation.

- code SHA: `d84d0578d008ca9be97eeabf7dc7b3cff89e0648`
- gate: `granted`
- §5.2 manipulation check: PASS
- §5.3 reproduction: PASS

## R-2 variant

`target_posfree`: identical to the fixed-label SIGReg-JEPA except that the label pass receives a clone of `batch.features` with columns 0-5 (scaled position and direction) set to zero.  Context pass, full pass, predictor and position queries, SIGReg terms, loss weights, optimiser, schedule, data, cell and seeds unchanged; the label is not detached and the encoder stays shared and trainable.

## §5.2 manipulation check

| check | value |
|---|---|
| label_position_columns_zero | true |
| label_event_columns_equal | true |
| original_features_unchanged | true |
| context_receives_original | true |
| full_pass_receives_original | true |
| label max abs Δ under a position permutation | 0 (max 1e-06) |
| context max abs Δ under a position permutation | 8.97306 (min 0.001) |
| input_proj.weight grad (label MSE alone) | 0.116614 |
| label f_layout (init) | 0.0913237 |
| label f_event (init) | 0.305432 |
| label fraction events | 64 (target 64) |
| pass | true |

## R2-a event dependence (primary)

- class: **event-dependent**
- criteria: 

## R2-b recovery

- verdict: **recovered**
- criteria: recovered: both seeds full vertex < 0.160072, full token R2 >= 0.5, half vertex < 0.174821, half token R2 >= 0.5; not recovered: both seeds full vertex >= 0.160072 or full token R2 < 0.5

## R2-c descriptive (per seed)

| seed | full vertex m | full token R² | half vertex m | half token R² | eff. rank | module-gain R² | f_layout | label f_event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9000 | 0.126005 | 0.884185 | 0.124601 | 0.659746 | 1.54404 | 0.0560073 | 0.811059 | 0.0104079 |
| 9001 | 0.140544 | 0.920046 | 0.123853 | 0.704844 | 1.55524 | 0.0608046 | 0.879572 | 0.020311 |

Reference values (descriptive; never thresholds): fix_full_vertex_m = [0.219259, 0.204281]; frozen_target_full_vertex_m = [0.138, 0.131]; static_target_full_vertex_m = [0.160374, 0.161148]; mae9000_full_vertex_m = 0.104949; init_full_vertex_m = 0.160072; init_half_vertex_m = 0.174821

| seed | run_id | steps | resumed_from_step | checkpoint sha256 | wall h |
|---:|---|---:|---:|---|---:|
| 9000 | dev_s005_within__sigreg_jepa__s9000 | 10000 | 0 | `3bb04c30dd51a90602e632f20f553b635c211a73f3bb048778ef5b2ad45b610c` | 0.649371 |
| 9001 | dev_s005_within__sigreg_jepa__s9001 | 10000 | 0 | `e36a437d77714985e59009528b6e73ba092e213b7b209c630b970f275ddc71a2` | 0.65036 |

## R-1 — are generative encoders more repairable?

| model | family | S | R²_eq | R²_eq class | ρ_LR | ρ_SWtw | ρ_SWunit | share_cs | Δ_D (m) | eligible | class |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| fix9000 | joint_embedding_collapsed | 0.00804103 | 0.251413 | partial | 0.202655 | 1.00288 | 1.04299 | 0.989356 | 0.015205 | true | partial |
| fix9001 | joint_embedding_collapsed | 0.00724096 | 0.288037 | partial | 0.244098 | 1.01142 | 1.05291 | 0.992439 | 0.0165864 | true | partial |
| frozen_target9000 | joint_embedding_working | 0.00483479 | 0.707703 | latent repair works | 0.557711 | 1.00329 | 1.04096 | 1.00223 | 0.0218109 | true | latent repair works |
| frozen_target9001 | joint_embedding_working | 0.00397908 | 0.727779 | latent repair works | 0.568581 | 0.997833 | 1.04524 | 0.995783 | 0.0179163 | true | latent repair works |
| ijepa9000 | joint_embedding_working | 0.00210223 | n/a |  | n/a | 1.01847 | 1.06181 | 1.02196 | 0.0183297 | true | not classified |
| ijepa9001 | joint_embedding_working | 0.00303184 | n/a |  | n/a | 1.0065 | 1.04797 | 0.992472 | 0.0216543 | true | not classified |
| init9000 | untrained_reference | 0.00278904 | 0.771108 | partial | 0.411 | 0.994479 | 1.03769 | 0.989962 | 0.0165716 | true | partial |
| mae_across_trigger9000 | generative | 0.00416013 | 0.190141 | partial | 0.301889 | 0.994802 | 1.05046 | 1.00208 | 0.00856763 | true | partial |
| mae_across_trigger9001 | generative | 0.00625702 | 0.22329 | partial | 0.334298 | 0.998605 | 1.02349 | 1.01842 | 0.0171922 | true | partial |
| mae_within_median9000 | generative | 0.00195101 | n/a |  | n/a | 1.00188 | 1.05778 | 1.0119 | 0.010295 | true | not classified |
| mae_within_median9001 | generative | 0.00227455 | n/a |  | n/a | 0.990523 | 1.05773 | 0.982052 | 0.0122275 | true | not classified |
| mae_within_trigger9000 | generative | 0.0021716 | 0.66339 | latent repair works | 0.541838 | 0.987465 | 1.02132 | 0.965522 | 0.00999246 | true | latent repair works |
| mae_within_trigger9001 | generative | 0.00249217 | 0.432848 | partial | 0.310533 | 0.99605 | 1.03282 | 0.988753 | 0.016582 | true | partial |
| static_target9000 | joint_embedding_collapsed | 0.00632227 | n/a |  | n/a | 0.999387 | 1.03234 | 0.9899 | 0.0252594 | true | not classified |
| static_target9001 | joint_embedding_collapsed | 0.00816829 | n/a |  | n/a | 1.00413 | 1.07551 | 1.00115 | 0.0172932 | true | not classified |
| t1_s030_across_trigger9000 | generative | 0.00316969 | 0.192304 | fails | 0.157113 | 0.99562 | 1.03189 | 0.989551 | 0.00843364 | true | fails |
| t1_s030_across_trigger9001 | generative | 0.00488135 | 0.204914 | partial | 0.373063 | 1.00183 | 1.05401 | 1.00946 | 0.0118607 | true | partial |
| t1_s030_within_trigger9000 | generative | 0.0031924 | 0.466856 | partial | 0.421621 | 0.987255 | 1.02686 | 0.965458 | 0.0109112 | true | partial |
| t1_s030_within_trigger9001 | generative | 0.00353334 | 0.432351 | partial | 0.380603 | 1.00047 | 1.04857 | 1.01147 | 0.00985075 | true | partial |

R1 reading: **contradicts** — the generative median is below both frozen_target seeds on both quantities
Generative median: R²_eq = 0.32782, ρ_LR = 0.353681; frozen_target: R²_eq/ρ_LR = {'9000': 0.7077025052837376, '9001': 0.727779335545238} / {'9000': 0.5577105207232913, '9001': 0.5685811334651208}.
With 8 generative against 2 frozen_target models this is a description, not a test.

## R-3 positioning (citations only; §7)

- Baade et al., *Generative Representation Learning for Scalable Latent Tokenization*, arXiv 2610.20533: Discriminative and LeJEPA objectives can be satisfied by a feature subset (feature suppression); on watermarked data LeJEPA and DINOv3 collapse to the shared watermark while a generative target avoids this -- external evidence that SIGReg prevents dimensional collapse but not shortcut collapse, and support for the R-1 expectation.
- Chen, Luo and Li, *Intriguing Properties of Contrastive Losses*, NeurIPS 2021: Introduces feature suppression in contrastive learning, the older name for the phenomenon reported here.
- *When Graph-JEPA Learns the Wrong Thing*, arXiv 2608.20516: A Graph-JEPA with a healthy linear probe and effective rank carries zero instance information; the degenerate solution is a global optimum of the coupled predictor / EMA-target objective.  "Effective rank misses the collapse" is already published and is not claimed as new here.
- Yang et al., *Denoising Vision Transformers*, ECCV 2024: ViT features split into an input-dependent term and an input-independent, position-related term -- the precedent for the f_layout / f_event decomposition.

## Per-job hours

| mode | kind | wall h |
|---|---|---:|
| preflight | cpu | 0.000118333 |
| preflight | cpu | 3.19444e-05 |
| manipulation | cpu | 0.0230883 |
| reproduce | gpu | 0.0314292 |
| smoke | gpu | 0.0195542 |
| train | gpu | 0.658109 |
| train | gpu | 0.192181 |
| train | gpu | 0.657952 |
| report | cpu | 0.00226583 |

GPU 1.5592 h of 1.5 h; CPU 0.0255 h of 2.0 h.

## Outputs

- `plan.json` — gate, preflight, S1, caps, entries.
- `manipulation.json` — §5.2 check.
- `reproduction.json` — §5.3 reproduction gate.
- `runs.json` — per-run readouts and the R2 readings.
- `readings.json` — R2-a/R2-b/R2-c and R1 or `not_run`.
- `LARGE_ARRAYS.json` — sha256 of every file > 5 MB under `<ext>`.
- `README.md` — this file.


## Addendum (7 Nov 2026, v1) — D5 R-2 evaluation and corrected R-1

Addendum: `docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md` (sha256 `abe0904af0ed0b5a6586cee459c507ae0fa2a7c60c65ffb58192c71b7180621a`), committed unchanged; §0 author line: > "写进weekly report，然后我选A，你写一个小runbook".
This section updates the 7 Nov record in place (addendum §5 D4 / §10): the R2 readings were empty in the 7 Nov record because the closed Part B cap refused the evaluation, and the R-1 `ρ_LR` values were invalid through the D1 drift-map scale defect.

### D5 R-2 evaluation (evaluation only; no training)

- protocol: one job for the two run directories, exactly Part B §5.3: ext_h12.evaluate_run full input and half pass, plus the label-pass f_event over the §5.2 64 events
- status: **complete**
- evaluation source: `/ceph/dwong/jepa/returns/ext_shortcut_20261107/eval/addendum_eval.json`
- run directories: `/ceph/dwong/jepa/returns/ext_shortcut_20261107/train/runs/dev_s005_within__sigreg_jepa__s9000`, `/ceph/dwong/jepa/returns/ext_shortcut_20261107/train/runs/dev_s005_within__sigreg_jepa__s9001`

| seed | run dir | full vertex m | half vertex m | full token R² | half token R² | eff. rank | f_layout | label f_event |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 9000 | `/ceph/dwong/jepa/returns/ext_shortcut_20261107/train/runs/dev_s005_within__sigreg_jepa__s9000` | 0.126005 | 0.124601 | 0.884185 | 0.659746 | 1.54404 | 0.811059 | 0.0104079 |
| 9001 | `/ceph/dwong/jepa/returns/ext_shortcut_20261107/train/runs/dev_s005_within__sigreg_jepa__s9001` | 0.140544 | 0.123853 | 0.920046 | 0.704844 | 1.55524 | 0.879572 | 0.020311 |

- R2-a event dependence (primary): **event-dependent**
- R2-b recovery: **recovered**
- R2-c: 2 seed row(s), descriptive; reference values are never thresholds.

### Readings completeness (Part B)

| reading | status | value |
|---|---|---|
| R2-a | complete | event-dependent |
| R2-b | complete | recovered |
| R2-c | complete | ['9000', '9001'] |
| R-1 | complete | contradicts |

`complete` is true only when R2-a/R2-b/R2-c and R-1 are all present (addendum §5 D4 readings completeness for Part B).

### R-1 hand-off (addendum A2, review round 7)

- R-1 x2 source / readings basis: **d1_overlay** (overlay: true; corrected-results marker: true); the order-independent source order is d1_overlay -> corrected_results -> 7nov_results.

- stable results-dir hand-off: `/ceph/dwong/jepa/code/followup_20261103/results/jepa/20261107_shortcut/r1_addendum.json` (sha256 `d9cc559de4552060c95dcad4390e0bef98fb141f8659a707b1dc75c7e3169ec5`), with the reading at top-level `reading`/`r1`/`R-1` for Part A's superseded-table picker.
- `ADDENDUM_2026-11-07.json` carries the rerun under `items.R1_rerun` (status `complete`).
- 7 Nov readings preserved write-once: `/ceph/dwong/jepa/returns/ext_shortcut_20261107/addendum/preserved_7nov/readings.json` (sha256 `b675d20fe0b2fb1b2dc5477acf2aae7c958b7f0df004fa4ade7d2659186b4d2a`, written now: true) before the in-place rewrite; the superseded old values are read from that copy.

### Superseded 7 Nov values (Part B)

| quantity | superseded 7 Nov value | new value | defect | evidence |
|---|---|---|---|---|
| R-1 generative-median ρ_LR | -0.67786 | 0.353681 | D1 | /ceph/dwong/jepa/code/followup_20261103/results/jepa/20261106_repair/x2.json (corrected ρ_LR, read-only) |
| R-1 frozen_target9000 ρ_LR | -0.53384 | 0.557711 | D1 | /ceph/dwong/jepa/code/followup_20261103/results/jepa/20261106_repair/x2.json (corrected ρ_LR, read-only) |
| R-1 frozen_target9001 ρ_LR | -0.586497 | 0.568581 | D1 | /ceph/dwong/jepa/code/followup_20261103/results/jepa/20261106_repair/x2.json (corrected ρ_LR, read-only) |
| R2-a class | single-seed (not classified) | event-dependent | D5 | the addendum R-2 evaluation (the 7 Nov reading had no evaluation) |
| R2-b verdict | unclear | recovered | D5 | the addendum R-2 evaluation (the 7 Nov reading had no evaluation) |
| R2-c per-seed rows | empty | ['9000', '9001'] | D5 | the addendum R-2 evaluation |

### D2 statistic audit (Part B side)

- pass: true; vertex point statistic: mean (`ext_readout.vertex_resolution -> lucid_run._vertex_error: errors.mean()`)
- no comparison in `ext_shortcut` pairs a point estimate with a differently-defined interval; if a payload carries an interval, its declared statistic is checked against the point it belongs to.

### Figures

Part B produces **no figures**.  The addendum's six figures (with the D4 axis labels and legends, and the three-stream identification figure) are Part A's, under `results/jepa/20261106_repair/`; this README therefore has no figure legends to list (addendum §5 D4, Part B).

### Addendum budget (shared ledger, addendum §6)

- GPU 0.1406 h of 0.35 h; CPU 0.7456 h of 3.0 h; ledger `/ceph/dwong/jepa/returns/addendum_20261107_jobs.jsonl`.
- the closed Part A (4.0 h) and Part B (1.5 h) caps are not consulted by the addendum modes.

