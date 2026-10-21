## Arms (validation split only; seeds 20–22; model ÷ constant, pooled)

| Arm | Setting | Vertex (20 / 21 / 22) | Mean | Worst energy cell, vertex | Direction (°) | E_dep | Fallback |
|---|---|---|---|---|---|---|---|
| (a) | prompt window, dark on | 0.840 / 0.789 / 0.807 | 0.812 | 0.90 | 7.6–8.0 | 0.47–0.54 | 0 of 480 |
| (b) | prompt window, dark off | 0.797 / 0.748 / 0.814 | 0.786 | 0.86 | 7.2–7.5 | 0.48–0.52 | 0 of 480 |
| (c) | dark on, truth-dark removed (upper bound) | 0.775 / 0.755 / 0.824 | 0.785 | 0.87 | 7.1–8.4 | 0.47–0.53 | 0 of 480 |

**Pass mark (E45):** arm (a) ≤ 0.88 in every seed: True (max 0.840); |mean(a) − mean(b)| ≤ 0.05: True (+0.026). **PASS**

## Dropped digits (all roles; validation in brackets)

| Arm | Digits read | Into the model | Prompt-window drops | of which source | of which source, no dark PE | Truth-dark removed |
|---|---|---|---|---|---|---|
| (a) | 197455 (39557) | 196745 (39410) | 710 | 32 | 0 | 0 |
| (b) | 196719 (39409) | 196719 (39409) | 0 | 0 | 0 | 0 |
| (c) | 197455 (39557) | 196687 (39402) | 32 | 32 | 0 | 736 |

Per (role, energy) cell counts: `summary.json`.
