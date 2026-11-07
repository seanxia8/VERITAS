# Shortcut-collapse follow-up (Part B) — 7 Nov 2026

`registered: false`. Extension `shortcut_followup_2026-11-07`, variant `target_posfree` (`arm_name` stays `sigreg_jepa`).
Runbook: `docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` (sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b`).
Author-approved training: exactly two 10 000-step runs (seeds 9000/9001) and one 20-step smoke; no other training, no data generation.

- code SHA: `72a1ba25e5013e74f14d2f3ccfd85f834554e3f6`
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

- class: **single-seed (not classified)**
- criteria: 

## R2-b recovery

- verdict: **unclear**
- criteria: recovered: both seeds full vertex < 0.160072, full token R2 >= 0.5, half vertex < 0.174821, half token R2 >= 0.5; not recovered: both seeds full vertex >= 0.160072 or full token R2 < 0.5

## R2-c descriptive (per seed)

| seed | full vertex m | full token R² | half vertex m | half token R² | eff. rank | module-gain R² | f_layout | label f_event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Reference values (descriptive; never thresholds): fix_full_vertex_m = [0.219259, 0.204281]; frozen_target_full_vertex_m = [0.138, 0.131]; static_target_full_vertex_m = [0.160374, 0.161148]; mae9000_full_vertex_m = 0.104949; init_full_vertex_m = 0.160072; init_half_vertex_m = 0.174821

## R-1 — are generative encoders more repairable?

| model | family | S | R²_eq | R²_eq class | ρ_LR | ρ_SWtw | ρ_SWunit | share_cs | Δ_D (m) | eligible | class |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| fix9000 | joint_embedding_collapsed | 0.00804103 | 0.251413 | non-equivariant | -0.888782 | 1.00288 | 1.04299 | 0.989356 | 0.015205 | true | non-equivariant |
| fix9001 | joint_embedding_collapsed | 0.00724096 | 0.288037 | non-equivariant | -0.835456 | 1.01142 | 1.05291 | 0.992439 | 0.0165864 | true | non-equivariant |
| frozen_target9000 | joint_embedding_working | 0.00483479 | 0.707703 | partly equivariant | -0.53384 | 1.00329 | 1.04096 | 1.00223 | 0.0218109 | true | partly equivariant |
| frozen_target9001 | joint_embedding_working | 0.00397908 | 0.727779 | partly equivariant | -0.586497 | 0.997833 | 1.04524 | 0.995783 | 0.0179163 | true | partly equivariant |
| ijepa9000 | joint_embedding_working | 0.00210223 | n/a |  | n/a | 1.01847 | 1.06181 | 1.02196 | 0.0183297 | true | not classified |
| ijepa9001 | joint_embedding_working | 0.00303184 | n/a |  | n/a | 1.0065 | 1.04797 | 0.992472 | 0.0216543 | true | not classified |
| init9000 | untrained_reference | 0.00278904 | 0.771108 | partly equivariant | -0.704688 | 0.994479 | 1.03769 | 0.989962 | 0.0165716 | true | partly equivariant |
| mae_across_trigger9000 | generative | 0.00416013 | 0.190141 | non-equivariant | -0.719296 | 0.994802 | 1.05046 | 1.00208 | 0.00856763 | true | non-equivariant |
| mae_across_trigger9001 | generative | 0.00625702 | 0.22329 | non-equivariant | -1.02284 | 0.998605 | 1.02349 | 1.01842 | 0.0171922 | true | non-equivariant |
| mae_within_median9000 | generative | 0.00195101 | n/a |  | n/a | 1.00188 | 1.05778 | 1.0119 | 0.010295 | true | not classified |
| mae_within_median9001 | generative | 0.00227455 | n/a |  | n/a | 0.990523 | 1.05773 | 0.982052 | 0.0122275 | true | not classified |
| mae_within_trigger9000 | generative | 0.0021716 | 0.66339 | partly equivariant | -0.636423 | 0.987465 | 1.02132 | 0.965522 | 0.00999246 | true | partly equivariant |
| mae_within_trigger9001 | generative | 0.00249217 | 0.432848 | partly equivariant | -0.630439 | 0.99605 | 1.03282 | 0.988753 | 0.016582 | true | partly equivariant |
| static_target9000 | joint_embedding_collapsed | 0.00632227 | n/a |  | n/a | 0.999387 | 1.03234 | 0.9899 | 0.0252594 | true | not classified |
| static_target9001 | joint_embedding_collapsed | 0.00816829 | n/a |  | n/a | 1.00413 | 1.07551 | 1.00115 | 0.0172932 | true | not classified |
| t1_s030_across_trigger9000 | generative | 0.00316969 | 0.192304 | non-equivariant | -1.22903 | 0.99562 | 1.03189 | 0.989551 | 0.00843364 | true | non-equivariant |
| t1_s030_across_trigger9001 | generative | 0.00488135 | 0.204914 | non-equivariant | -0.548677 | 1.00183 | 1.05401 | 1.00946 | 0.0118607 | true | non-equivariant |
| t1_s030_within_trigger9000 | generative | 0.0031924 | 0.466856 | partly equivariant | -0.50095 | 0.987255 | 1.02686 | 0.965458 | 0.0109112 | true | partly equivariant |
| t1_s030_within_trigger9001 | generative | 0.00353334 | 0.432351 | partly equivariant | -0.788511 | 1.00047 | 1.04857 | 1.01147 | 0.00985075 | true | partly equivariant |

R1 reading: **contradicts** — the generative median is below both frozen_target seeds on both quantities
Generative median: R²_eq = 0.32782, ρ_LR = -0.67786; frozen_target: R²_eq/ρ_LR = {'9000': 0.7077025052837376, '9001': 0.727779335545238} / {'9000': -0.533840048787172, '9001': -0.5864968707636384}.
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

GPU 1.5592 h of 1.5 h; CPU 0.0232 h of 2.0 h.

## Outputs

- `plan.json` — gate, preflight, S1, caps, entries.
- `manipulation.json` — §5.2 check.
- `reproduction.json` — §5.3 reproduction gate.
- `runs.json` — per-run readouts and the R2 readings.
- `readings.json` — R2-a/R2-b/R2-c and R1 or `not_run`.
- `LARGE_ARRAYS.json` — sha256 of every file > 5 MB under `<ext>`.
- `README.md` — this file.

