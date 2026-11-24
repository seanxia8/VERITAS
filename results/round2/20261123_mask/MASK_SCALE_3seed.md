# MASK_SCALE — r2-mask M2 (patch-size sweep)

Runbook `RUNBOOK_2026-11-23_R2_MASK_SCALE.md` §M2, PREREG `PREREG_2026-11-23_R2_MASK.md` §1.3–1.5 (pushed 2c0204b before any M2 evaluation). Primary: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge readout (READOUT_SPEC_v1), gain G = 1 − e/e_init over the matched init (step-0 snapshot of the seed). Contrasts T = e(config)/e(random) − 1 with the statistics of PREREG_2026-11-22 §2.2. Every cell is `mae_repaired`, default recipe, 50 ns, 10,000 steps; only the mask differs.

## WCTE (seeds A, B, C; ℓ_c photon 2.48 m)

| Config | U (M1) | AvgDist (m) | e_init (m) | e @10k per seed (m) | e @10k mean | G @10k per seed | G mean | G @5k mean | T vs random | N_σ | clear |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| random | 0.2855 | 0.083 | 0.0675 | A 0.0561, B 0.0555, C 0.0555 | 0.0557 | A +15.6 %, B +17.4 %, C +19.2 % | +17.4 % | +13.8 % | – | – | – |
| patch8 | 0.2905 | 0.160 | 0.0675 | A 0.0536, B 0.0542, C 0.0545 | 0.0541 | A +19.3 %, B +19.4 %, C +20.6 % | +19.8 % | +15.7 % | -2.9 % | 2.83 | yes |
| patch32 (default) | 0.3054 | 0.293 | 0.0675 | A 0.0533, B 0.0525, C 0.0529 | 0.0529 | A +19.8 %, B +22.0 %, C +22.9 % | +21.6 % | +18.0 % | -5.0 % | 4.97 | yes |
| patch128 | 0.3352 | 0.474 | 0.0675 | A 0.0534, B 0.0559, C 0.0536 | 0.0543 | A +19.6 %, B +16.8 %, C +21.9 % | +19.5 % | +18.1 % | -2.5 % | 2.44 | no |
| patch256 | 0.3638 | 0.604 | 0.0675 | A 0.0560, B 0.0558, C 0.0542 | 0.0553 | A +15.8 %, B +17.1 %, C +21.1 % | +18.0 % | +14.6 % | -0.7 % | 0.70 | no |

Secondary (T vs random at 10k, sealed; N_σ):

| Config | vertex_mlp | vertex_mlp_latent | vertex_attentive | dir_mu | dir_e | energy_e | energy_mu |
|---|---:|---:|---:|---:|---:|---:|---:|
| patch8 | -3.0 % (0.8) | -2.9 % (1.2) | -5.3 % (4.4*) | -6.2 % (4.2*) | -5.6 % (3.9*) | – | -1.1 % (0.2) |
| patch32 | -7.0 % (1.8) | -5.8 % (2.4*) | -8.6 % (7.2*) | -9.1 % (6.2*) | -8.0 % (5.5*) | – | +2.3 % (0.4) |
| patch128 | -5.0 % (1.3) | +2.1 % (0.8) | -7.6 % (6.3*) | -7.5 % (5.2*) | -9.0 % (6.2*) | – | +11.4 % (2.2*) |
| patch256 | -1.1 % (0.3) | -2.1 % (0.8) | -4.3 % (3.6*) | -4.9 % (3.3*) | -7.5 % (5.1*) | – | +23.0 % (4.3*) |

## SK (seeds A, B, C; ℓ_c photon 13.90 m)

| Config | U (M1) | AvgDist (m) | e_init (m) | e @10k per seed (m) | e @10k mean | G @10k per seed | G mean | G @5k mean | T vs random | N_σ | clear |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| random | 0.5486 | 0.723 | 0.5484 | A 0.4279, B 0.4101, C 0.4116 | 0.4165 | A +21.9 %, B +24.6 %, C +25.7 % | +24.0 % | +17.2 % | – | – | – |
| patch43 | 0.5519 | 1.503 | 0.5484 | A 0.3758, B 0.3881, C 0.3814 | 0.3818 | A +31.4 %, B +28.6 %, C +31.1 % | +30.4 % | +23.5 % | -8.3 % | 5.57 | yes |
| patch173 (default) | 0.5650 | 2.592 | 0.5484 | A 0.3681, B 0.3534, C 0.3649 | 0.3621 | A +32.8 %, B +35.0 %, C +34.1 % | +34.0 % | +29.1 % | -13.1 % | 8.76 | yes |
| patch692 | 0.5995 | 4.918 | 0.5484 | A 0.3559, B 0.3581, C 0.3589 | 0.3576 | A +35.0 %, B +34.2 %, C +35.2 % | +34.8 % | +30.8 % | -14.1 % | 9.47 | yes |

Secondary (T vs random at 10k, sealed; N_σ):

| Config | vertex_mlp | vertex_mlp_latent | vertex_attentive | dir_mu | dir_e | energy_e | energy_mu |
|---|---:|---:|---:|---:|---:|---:|---:|
| patch43 | -10.0 % (6.5*) | -9.3 % (9.7*) | -10.2 % (10.5*) | -10.8 % (6.1*) | -13.2 % (6.5*) | -1.0 % (0.5) | -3.5 % (1.7) |
| patch173 | -15.6 % (10.0*) | -12.6 % (13.2*) | -18.1 % (18.7*) | -17.6 % (10.0*) | -22.3 % (10.9*) | -2.2 % (1.1) | -2.1 % (1.0) |
| patch692 | -12.2 % (7.8*) | -11.7 % (12.2*) | -20.5 % (21.1*) | -15.2 % (8.6*) | -21.9 % (10.7*) | +3.7 % (1.8) | +1.9 % (0.9) |

## Verdicts

- **Q1 WCTE**: holds — U parity 0.2831, random 0.2855, patches patch8 0.2905, patch32 0.3054, patch128 0.3352, patch256 0.3638 (parity ≤ random: True; random < every patch: True; increasing in s: True).
- **Q1 SK**: does not hold — U parity 0.5493, random 0.5486, patches patch43 0.5519, patch173 0.5650, patch692 0.5995 (parity ≤ random: False; random < every patch: True; increasing in s: True).
- **Q2 WCTE**: does not hold — Spearman(U, vertex gain) = 0.10 over 5 configurations; direction gain dir_mu 0.30, dir_e 0.70 (same sign: True).
- **Q2 SK**: holds — Spearman(U, vertex gain) = 1.00 over 4 configurations; direction gain dir_mu 0.80, dir_e 0.80 (same sign: True).
- **Q3 WCTE**: does not hold — best patch32; half-gain AvgDist 0.151 m vs ℓ_c 2.48 m (ratio 0.06).
- **Q3 SK**: does not hold — best patch692; half-gain AvgDist 1.384 m vs ℓ_c 13.90 m (ratio 0.10).
- **Q4 SK**: does not hold — T(patch default vs random) -13.1 % at 10k (A -14.0 %, B -13.8 %, C -11.4 %) vs -14.3 % at 5k (A -14.5 %, B -14.6 %, C -14.0 %). Exploratory: D = T(10k) − T(5k) = +1.27 points (SE_boot 0.30, SE_seed 0.67, N_σ 1.9).
- **KM**: does not fire — SK best patch692 vs worst random: T -14.1 %, N_σ 9.47, clear True; Spearman < 0.5 on both detectors: False.

SD_pooled (RMS over each detector's config-vs-random contrasts of the across-seed SD): wcte 0.0175, sk 0.0258.

Figure: `MASK_SCALE_3seed.png` (top: vertex gain vs U; bottom: sealed vertex error vs patch size, per seed).
