# Two-claim protocol smoke — DEVELOPMENT OUTPUT, NOT CITABLE

_DEVELOPMENT SMOKE — NOT CITABLE. Exercises the two-claim protocol chain on the linear subject at toy size; no scientific conclusion._

latent_monitor 0.1.0 · mode `dev` · config hash `43cf86fcc97849ba` · commit `f24430a-dirty` · seeds [0] · 4.4 s on `claude`

**Gate warnings (dev mode continues; confirmatory would refuse):**

- freeze part 'core' is missing (protocol/frozen/core.json)
- freeze part 'counts' is missing (protocol/frozen/counts.json)
- freeze part 'bridge' is missing (protocol/frozen/bridge.json)
- working tree is dirty and no matching frozen source-tree hash is declared
- environment lock hash is not frozen
- no data manifest declared
- no model / checkpoint hash declared
- confirmatory results must be written under 'results/confirmatory', not results/latent_monitor_smoke_dev_2026-09-16_pass2

## Information contract

| arm | features | status | data-flow checked |
|---|---:|---|---|
| generic_rich | 29 | operational | True |
| intermediate_only | 10 | operational | True |
| full_intermediate | 39 | operational | True |
| generic_rich_matched | 39 | operational | True |
| input_only | 6 | operational | True |
| output_uncertainty | 4 | operational | True |
| final_embedding | 6 | operational | True |
| all_generic | 20 | operational | True |
| all_generic_no_noise | 16 | operational | True |
| noise_only | 4 | operational | True |
| full_layerwise_legacy | 26 | operational | True |
| full_intermediate_drop_channel | 34 | operational | True |
| full_intermediate_drop_token | 34 | operational | True |

Partitions (windows): {'attribution_train': 264, 'calibration': 198, 'development': 198, 'evaluation': 336, 'excluded': 288, 'reference_fit': 396}. Held-out families: ['line_pickup', 'double_pulse']; undeclared: ['glitch'].
FAR budget 0.01: 3/24 clean evaluation windows alerted (realised 0.125, CI95 [0.02655931498624894, 0.3236113581888329]); resolvable in one step: False.

## Claim 1 — incremental attribution over {N, S}

| arm | role | status | F1 inclusive | F1 hard | F1 held-out fam. | features | λ |
|---|---|---|---:|---:|---:|---:|---:|
| generic_rich | primary | operational | 0.850 | 0.792 | 0.829 | 29 | 0.001 |
| intermediate_only | primary | operational | 0.653 | 0.744 | 0.681 | 10 | 0.01 |
| full_intermediate | primary | operational | 0.880 | 0.798 | 0.958 | 39 | 0.01 |
| generic_rich_matched | control | operational | 0.789 | 0.744 | 0.681 | 39 | 0.001 |
| full_intermediate_drop_channel | control | operational | 0.879 | 0.792 | 0.895 | 34 | 0.001 |
| full_intermediate_drop_token | control | operational | 0.887 | 0.847 | 0.895 | 34 | 0.001 |
| input_only | diagnostic | operational | 0.709 | 0.596 | 0.743 | 6 | 0.001 |
| output_uncertainty | diagnostic | operational | 0.700 | 0.540 | 0.873 | 4 | 0.001 |
| final_embedding | diagnostic | operational | 0.675 | 0.520 | 0.851 | 6 | 10 |
| all_generic | diagnostic | operational | 0.886 | 0.792 | 0.979 | 20 | 0.001 |
| all_generic_no_noise | diagnostic | operational | 0.778 | 0.744 | 0.853 | 16 | 0.001 |
| noise_only | diagnostic | operational | 0.818 | 0.950 | 1.000 | 4 | 0.001 |
| full_layerwise_legacy | diagnostic | operational | 0.883 | 0.792 | 0.937 | 26 | 0.001 |

**primary_hard** — full_intermediate − generic_rich [hard matched evaluation set (Claim-1 primary estimand), n=20]: ΔF1 = 0.006 [-0.108, 0.156] over 15 event groups → **inconclusive (n_windows 20 < 30)** (failed replicates 0)

**primary_inclusive** — full_intermediate − generic_rich [inclusive evaluation set, n=216]: ΔF1 = 0.030 [-0.012, 0.076] over 24 event groups → **inconclusive** (failed replicates 0)

**primary_heldout_families** — full_intermediate − generic_rich [held-out families, n=48]: ΔF1 = 0.130 [0.053, 0.234] over 24 event groups → **benefit** (failed replicates 0)

**control_matched_capacity** — full_intermediate − generic_rich_matched [hard set; generic arm with matched feature count, n=20]: ΔF1 = 0.054 [0.000, 0.164] over 15 event groups → **inconclusive (n_windows 20 < 30)** (failed replicates 0)

**control_drop_channel** — full_intermediate − full_intermediate_drop_channel [hard set; hook drop, n=20]: ΔF1 = 0.006 [-0.108, 0.156] over 15 event groups → **inconclusive (n_windows 20 < 30)** (failed replicates 0)

**control_drop_token** — full_intermediate − full_intermediate_drop_token [hard set; hook drop, n=20]: ΔF1 = -0.049 [-0.130, 0.000] over 15 event groups → **inconclusive (n_windows 20 < 30)** (failed replicates 0)

**control_intermediate_alone** — intermediate_only − generic_rich [hard set; internal hooks alone vs generic_rich, n=20]: ΔF1 = -0.047 [-0.287, 0.227] over 15 event groups → **inconclusive (n_windows 20 < 30)** (failed replicates 0)

**diagnostic_noise_only_ablation** — all_generic − all_generic_no_noise [pass-1 diagnostic, n=216]: ΔF1 = 0.108 [0.076, 0.153] over 24 event groups → **benefit** (failed replicates 0)

**diagnostic_legacy_layerwise** — full_layerwise_legacy − all_generic [pass-1 diagnostic (contaminated arm), n=216]: ΔF1 = -0.003 [-0.027, 0.021] over 24 event groups → **equivalence** (failed replicates 0)

Hard matching: 10 pairs from 48 S / 168 N windows (retained S 0.208, N 0.060; overlap failures 38; caliper 1.0; median pair distance 0.617).
Family-outer hierarchical interval on the hard set: [-0.200, 0.142] — descriptive only — 8 outer units < 10; confirmatory inference about held-out interventions is impossible at this size.

_macro-F1 over {N, S}; arms fitted on attribution_train, tuned on development, scored once on evaluation; reference_fit groups are never supervised examples; held-out and undeclared families entered no fit, tuning or calibration; the Claim-1 primary estimand is on the hard matched set_

## Abstention (support novelty, conformal on clean calibration windows)

α = 0.1, threshold 1.938 from 18 clean windows; finite-sample clean-retention bound 0.847; realised clean retention 0.875.
Support-estimator control AUROC (held-out clean vs displaced): 1.000 (usable: True).
Unknown-family AUROC: 0.604 (24 unknown windows; 4 abstained, 48 known abstained). Risk–coverage AUC 0.143.

| coverage | risk | retained known | retained unknown | retained-known macro-F1 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.158 | 111 | 9 | 0.782 |
| 0.8 | 0.182 | 172 | 20 | 0.827 |
| 0.9 | 0.181 | 194 | 22 | 0.847 |
| 1.0 | 0.179 | 216 | 24 | 0.880 |

## Joint operational table (detect → abstain → attribute)

| category | n | quiet | abstain | N | S | correct |
|---|---:|---:|---:|---:|---:|---:|
| clean | 24 | 21 | 3 | 0 | 0 | 0.875 (quiet) |
| N | 168 | 152 | 16 | 0 | 0 | 0.000 (N) |
| S | 48 | 21 | 27 | 0 | 0 | 0.000 (S) |
| mixture | 24 | 20 | 4 | 0 | 0 | 0.000 (N/S) |
| unknown | 24 | 20 | 4 | 0 | 0 | 0.167 (abstain) |

Overall correct-decision rate 0.074; 48 geometry windows not tabulated. _whole-chain decision rule; conditional N/S macro-F1 is not this table_

## Claim 2 — harm ranking: task-sensitive score vs committed generic score

κ_m: 1.1 (provisional_dev) — proposal (16 Sep): 'provisional 10% amplitude degradation' — development only; PREREGISTRATION §4 PENDING

**Primary ΔAUROC** = 0.167 (task 1.000 − generic 0.833; 4 harmful / 9 benign cells) — hierarchical interval [-0.453, 0.289] over 13 cells: **outer-unit bootstrap; supports a statement about interventions drawn like these**.
Held-out families only: ΔAUROC 0.000 on 2 cells — the Claim-2 estimand set proper — 2 held-out cells: descriptive only; below MIN_OUTER_UNITS_FOR_INFERENCE.
Supporting: intermediate score ΔAUROC 0.042; conditional triage (generic) AUROC 0.667, dropped 7 cells (1 harmful).
Cell threshold 0.636 (pseudo-cell bootstrap of 18 clean windows, cell size 24); missed harm at budget: generic 0.250, task 0.750; valid-rare cell rejection: generic 1.000, task 0.000; valid-rare window rejection (generic) 0.167.

| generic_committed | K < κ_m | K ≥ κ_m |
|---|---:|---:|
| low alarm | 6 | 1 |
| strong alarm | 3 | 3 |

| task_sensitive | K < κ_m | K ≥ κ_m |
|---|---:|---:|
| low alarm | 9 | 3 |
| strong alarm | 0 | 1 |

| cell | origin | K_phys | harm | generic | task | intermediate | held-out |
|---|---|---:|---|---:|---:|---:|---|
| event:glitch | unknown | 1.05 | benign | 0.98 | -0.27 | 0.55 | False |
| event_in_span:double_pulse | S_in_span | 4.31 | harmful | 3.62 | 3.08 | 3.68 | True |
| event_in_span:oscillation | S_in_span | 1.08 | benign | 1.12 | 0.13 | 0.81 | False |
| geometry:12ch | G | 1.33 | harmful | 0.73 | 0.55 | 0.64 | False |
| geometry:3ch | G | 1.47 | harmful | 1.41 | 0.41 | 1.27 | False |
| mixture:oscillation+gain_drift | mixture | 1.05 | benign | 0.81 | 0.20 | 0.64 | False |
| sigma_cov:bandwidth_down | N_cov | 1.00 | benign | 0.27 | 0.20 | -0.27 | False |
| sigma_cov:corr_down | N_cov | 0.99 | benign | 0.27 | 0.20 | -0.34 | False |
| sigma_cov:corr_up | N_cov | 1.01 | benign | 0.27 | 0.20 | -0.13 | False |
| sigma_cov:line_pickup | N_cov | 1.01 | benign | 0.27 | 0.13 | -0.34 | True |
| sigma_struct:channel_loss | N_struct | 1.16 | harmful | 0.55 | 0.55 | 0.34 | False |
| sigma_struct:gain_drift | N_struct | 0.98 | benign | 0.27 | 0.20 | -0.27 | False |
| sigma_struct:timing_jitter | N_struct | 0.99 | benign | 0.13 | 0.13 | -1.12 | False |

## Designed and task-specific controls (linear subject; positive controls)

| match metric | family | consequence ratio (amp, t0, τ) | declared-K ratio | lin. err Δy |
|---|---|---|---:|---:|
| euclidean | output_null | 1.00, 1.00, 1.00 | 1.00 | 1.7e-15 |
| euclidean | output_aligned | 0.97, 1.84, 1.05 | 0.97 | 1.9e-15 |
| euclidean | random | 0.95, 2.06, 1.10 | 0.95 | 1.4e-15 |
| euclidean | task_aligned | 0.97, 1.16, 0.99 | 0.97 | 1.8e-15 |
| euclidean | task_null | 1.00, 1.81, 1.06 | 1.00 | 2.0e-15 |
| null_mahalanobis | output_null | 1.00, 1.00, 1.00 | 1.00 | 3.4e-15 |
| null_mahalanobis | output_aligned | 1.01, 1.03, 1.02 | 1.01 | 6.0e-15 |
| null_mahalanobis | random | 0.99, 1.21, 1.02 | 0.99 | 4.2e-15 |
| null_mahalanobis | task_aligned | 0.98, 1.00, 1.00 | 0.98 | 1.7e-14 |
| null_mahalanobis | task_null | 1.00, 1.08, 1.01 | 1.00 | 3.7e-15 |

_constructed from the frozen head; positive controls, linear subject only (NonlinearSubjectUnsupported otherwise); realised consequence measured per output; linearisation error reported_
