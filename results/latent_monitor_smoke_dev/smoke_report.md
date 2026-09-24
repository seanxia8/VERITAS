# Protocol smoke — DEVELOPMENT OUTPUT, NOT CITABLE

_DEVELOPMENT SMOKE — NOT CITABLE. Exercises the amended protocol chain on the linear subject at toy size._

latent_monitor 0.1.0 · mode `dev` · config hash `434eb52d874147e3` · commit `f24430a-dirty` · seeds [0] · 4.7 s on `claude`

**Gate warnings (dev mode continues; confirmatory would refuse):**

- freeze part 'core' is missing (protocol/frozen/core.json)

## Information contract

| arm | features | status |
|---|---:|---|
| input_only | 6 | operational |
| output_uncertainty | 4 | operational |
| final_embedding | 6 | operational |
| all_generic | 20 | operational |
| full_layerwise | 31 | operational |
| noise_only | 4 | operational |
| all_generic_no_noise | 16 | operational |

Partitions (windows): {'calibration': 252, 'development': 336, 'evaluation': 480, 'excluded': 180, 'reference_fit': 672}. Held-out families: ['line_pickup', 'double_pulse'].
FAR budget 0.01 on 18 clean calibration windows — resolution 0.056, resolvable: False.

## C2 — attribution over {N, S} (alarm-time features only)

| arm | status | macro-F1 eval | held-out families | in-family held-out events | λ |
|---|---|---:|---:|---:|---:|
| input_only | operational | 0.773 | 0.525 | 0.844 | 1 |
| output_uncertainty | operational | 0.789 | 0.899 | 0.747 | 0.01 |
| final_embedding | operational | 0.725 | 0.900 | 0.644 | 0.001 |
| all_generic | operational | 0.935 | 1.000 | 0.912 | 0.001 |
| full_layerwise | operational | 0.892 | 0.983 | 0.857 | 1 |
| noise_only | operational | 0.846 | 0.967 | 0.807 | 0.01 |
| all_generic_no_noise | operational | 0.840 | 0.917 | 0.810 | 0.01 |

**delta_primary** — full_layerwise − all_generic (incl. operational noise-only): ΔF1 = -0.044 [-0.110, 0.011] over 30 event groups → **inconclusive**

**delta_noise_only_ablation** — all_generic − all_generic_no_noise: what the noise-only statistics add: ΔF1 = 0.095 [0.047, 0.162] over 30 event groups → **inconclusive**

**delta_layerwise_vs_no_noise** — full_layerwise − all_generic_no_noise: an unfair comparison shown only to expose the source of any gain: ΔF1 = 0.052 [0.013, 0.100] over 30 event groups → **inconclusive**

_macro-F1 over {N, S}; G, constructed and clean windows are excluded from the classifier; the arms were fitted on reference_fit groups, tuned on development groups, scored once on evaluation groups; held-out families entered no fit, tuning or calibration_

## C4 — all-cell scientific-harm ranking

κ_m: 1.1 (provisional_dev) — proposal §4 C4: 'provisionally a 10% degradation for the waveform amplitude task' — development only

### alarm = final_embedding

Primary all-cell AUROC(K ≥ κ_m) = 0.844 [0.395, 0.944] (4 harmful / 8 benign cells; AUPRC 0.793 at prevalence 0.333)
Secondary conditional triage (strong-alarm cells only): AUROC —, dropped 11 cells of which 3 harmful
Missed harm at budget: 0.750; valid-rare-event rejection: 0.000

| | K < κ_m | K ≥ κ_m |
|---|---:|---:|
| low alarm | 8 | 3 |
| strong alarm | 0 | 1 |

| cell | origin | K_phys | alarm/thr | harm |
|---|---|---:|---:|---|
| event:glitch | S_support | 0.98 | 0.91 | benign |
| event_in_span:double_pulse | S_in_span | 4.30 | 2.05 | harmful |
| event_in_span:oscillation | S_in_span | 1.01 | 0.83 | benign |
| geometry:12ch | G | 1.40 | 0.78 | harmful |
| geometry:3ch | G | 1.43 | 0.93 | harmful |
| sigma_cov:bandwidth_down | N_cov | 0.97 | 0.67 | benign |
| sigma_cov:corr_down | N_cov | 0.96 | 0.61 | benign |
| sigma_cov:corr_up | N_cov | 0.98 | 0.76 | benign |
| sigma_cov:line_pickup | N_cov | 0.97 | 0.65 | benign |
| sigma_struct:channel_loss | N_struct | 1.22 | 0.76 | harmful |
| sigma_struct:gain_drift | N_struct | 0.99 | 0.67 | benign |
| sigma_struct:timing_jitter | N_struct | 1.00 | 0.44 | benign |

### alarm = layerwise

Primary all-cell AUROC(K ≥ κ_m) = 0.562 [0.290, 0.823] (4 harmful / 8 benign cells; AUPRC 0.475 at prevalence 0.333)
Secondary conditional triage (strong-alarm cells only): AUROC 0.667, dropped 7 cells of which 2 harmful
Missed harm at budget: 0.500; valid-rare-event rejection: 1.000

| | K < κ_m | K ≥ κ_m |
|---|---:|---:|
| low alarm | 5 | 2 |
| strong alarm | 3 | 2 |

| cell | origin | K_phys | alarm/thr | harm |
|---|---|---:|---:|---|
| event:glitch | S_support | 0.98 | 1.53 | benign |
| event_in_span:double_pulse | S_in_span | 4.30 | 1.52 | harmful |
| event_in_span:oscillation | S_in_span | 1.01 | 1.02 | benign |
| geometry:12ch | G | 1.40 | 0.77 | harmful |
| geometry:3ch | G | 1.43 | 1.33 | harmful |
| sigma_cov:bandwidth_down | N_cov | 0.97 | 0.93 | benign |
| sigma_cov:corr_down | N_cov | 0.96 | 0.65 | benign |
| sigma_cov:corr_up | N_cov | 0.98 | 1.14 | benign |
| sigma_cov:line_pickup | N_cov | 0.97 | 0.88 | benign |
| sigma_struct:channel_loss | N_struct | 1.22 | 0.82 | harmful |
| sigma_struct:gain_drift | N_struct | 0.99 | 0.92 | benign |
| sigma_struct:timing_jitter | N_struct | 1.00 | 0.54 | benign |

## Designed dissociation (positive control, constructed from the frozen head)

| match metric | family | consequence ratio (amp, t0, τ) | Euclid alarm | null-Mahalanobis alarm | lin. err Δy |
|---|---|---|---:|---:|---:|
| euclidean | output_null | 1.00, 1.00, 1.00 | 3.00 | 7.38 | 1.8e-15 |
| euclidean | output_aligned | 1.04, 1.94, 0.98 | 3.00 | 31.24 | 1.6e-15 |
| euclidean | random | 0.98, 1.87, 1.03 | 3.00 | 20.85 | 1.2e-15 |
| null_mahalanobis | output_null | 1.00, 1.00, 1.00 | 1.42 | 3.00 | 3.0e-15 |
| null_mahalanobis | output_aligned | 0.99, 1.11, 0.98 | 0.49 | 3.00 | 5.3e-15 |
| null_mahalanobis | random | 1.00, 1.01, 1.00 | 0.66 | 3.00 | 5.3e-15 |

_constructed from the frozen head; a positive control, not transfer evidence. Realised consequence is measured, not predicted; linearisation error is reported._
