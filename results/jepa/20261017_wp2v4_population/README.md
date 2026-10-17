# WP2-v4 population tables — A4 design (13 cells × 4 splits)

_Generated 2026-10-17. Evidence level: **development output, not reviewed**._

Implementer: impl-wp2v4 (Agents-Team). Runbook v4 §3.2 (A4), Appendix A; work plan §8 WP2-v4.
Source: branch `JEPA`, working tree of 2026-10-17 (WP2-v4 uncommitted at generation time).
This directory contains only compact population summaries; no sampling, no tuning, no mask selection.

## Command

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /private/tmp/jepa-venv-cpu311/bin/python \
  results/jepa/20261017_wp2v4_population/make_population.py
```

The script is deterministic (seed 0; no event samples) and takes about 2–4 minutes.
It exits 0 on success; exits 2 with `status: "bug_check"` if any registered value
differs from runbook Appendix A by more than 1° (the runbook's STOP condition).

## Environment

| item | value |
|---|---|
| host | Dowlings-MacBook-Pro-2.local, macOS-26.6.2-arm64 |
| interpreter | `/private/tmp/jepa-venv-cpu311/bin/python` |
| Python / numpy / scipy | 3.11.11 / 1.26.4 / 1.17.1 |
| BLAS threads | `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` |
| source | `src/jepa_noise/views.py` (WP2-v4), config `src/jepa_noise/configs/waveform_v2_a4.json` |
| law | `v4-A4-waveform-v1` (base ρ = 0, E2-matched θ SD) |
| runtime | 125.9 s (52 A4 rows + 4 v3 reproduction rows) |

## Files

| file | contents |
|---|---|
| `make_population.py` | deterministic generator for this table |
| `population_v4.json` | 33 824 bytes; compact JSON with all rows, the §A checks, provenance and hashes |
| `README.md` | this file |
| `SHA256SUMS` | hashes of the three files above |

## Definition

**D (A4.5 displacement)** = population maximum principal angle (degrees) between the top-3 context
canonical directions of the linearised fixture and the Paper 1 reference `R_c` = top-3 GEVP of
`(A_c Σ_θ A_cᵀ, N_cc)`. Computed by `jepa_noise.views.displacement(population, k=3)` from the
blocks of `Tier1Views.population()`; the covariance code is not duplicated.

**γ_n** = population top noise canonical correlation of the same split and cell (reported, not used
for levels). Both are computed from exact covariances of the generating law.

The 13 cells are the A4 one-factor sweeps (`jepa_noise.views.A4_CELL_GRID`, validated by
`validate_a4_cells`, mirrored in `configs/waveform_v2_a4.json`): g ∈
{0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9} at line 0, and line ∈ {0.75, 1.5, 2.25, 3} at
g = 0; (g = 0, line = 0) is the clean reference cell. `channel` is the primary split,
`interleaved` M's alternative, `time`/`block` are diagnostic only (rank-deficient target).

## Results: D° / γ_n, A4 law (default)

| g | line | channel (primary) | interleaved (alt.) | time (diag.) | block (diag.) |
|---|---|---|---|---|---|
| 0 | 0 | 0.00 / 0.0000 | 0.00 / 0.0000 | 90.00 / 0.6037 | 3.56 / 0.0000 |
| 0.005 | 0 | 11.27 / 0.3571 | 11.29 / 0.3571 | 90.00 / 0.7186 | 86.84 / 0.2689 |
| 0.01 | 0 | 22.33 / 0.5262 | 22.35 / 0.5262 | 90.00 / 0.7881 | 87.94 / 0.4209 |
| 0.02 | 0 | 47.96 / 0.6895 | 47.98 / 0.6896 | 90.00 / 0.8587 | 88.59 / 0.5889 |
| 0.05 | 0 | 76.95 / 0.8474 | 76.96 / 0.8474 | 89.88 / 0.9294 | 88.90 / 0.7785 |
| 0.1 | 0 | 84.13 / 0.9174 | 84.14 / 0.9174 | 89.89 / 0.9615 | 88.95 / 0.8743 |
| 0.225 | 0 | 87.53 / 0.9615 | 87.53 / 0.9615 | 89.90 / 0.9820 | 88.94 / 0.9395 |
| 0.45 | 0 | 88.79 / 0.9804 | 88.79 / 0.9804 | 89.90 / 0.9908 | 88.93 / 0.9687 |
| 0.9 | 0 | 89.41 / 0.9901 | 89.41 / 0.9901 | 89.90 / 0.9953 | 88.93 / 0.9841 |
| 0 | 0.75 | 89.28 / 0.9881 | 89.28 / 0.9881 | 89.90 / 0.9944 | 88.93 / 0.9810 |
| 0 | 1.5 | 89.65 / 0.9940 | 89.65 / 0.9940 | 89.90 / 0.9972 | 88.93 / 0.9904 |
| 0 | 2.25 | 89.77 / 0.9960 | 89.77 / 0.9960 | 89.90 / 0.9981 | 88.92 / 0.9936 |
| 0 | 3 | 89.82 / 0.9970 | 89.82 / 0.9970 | 89.91 / 0.9986 | 88.92 / 0.9952 |

## Checks against runbook Appendix A (channel split, line 0)

| g | D (this run) | §A D | Δ | γ_n (this run) | §A γ_n | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.0 | +0.00 | 0.0000 | 0.000 | 0.0000 |
| 0.005 | 11.27 | 11.3 | −0.03 | 0.3571 | 0.357 | +0.0001 |
| 0.01 | 22.33 | 22.3 | +0.03 | 0.5262 | 0.526 | +0.0002 |
| 0.02 | 47.96 | 48.0 | −0.04 | 0.6895 | 0.690 | −0.0005 |
| 0.05 | 76.95 | 77.0 | −0.05 | 0.8474 | 0.847 | +0.0004 |
| 0.1 | 84.13 | 84.1 | +0.03 | 0.9174 | 0.917 | +0.0004 |
| 0.225 | 87.53 | 87.5 | +0.03 | 0.9615 | 0.962 | −0.0005 |
| 0.45 | 88.79 | 88.8 | −0.01 | 0.9804 | 0.980 | +0.0004 |
| 0.9 | 89.41 | 89.4 | +0.01 | 0.9901 | 0.990 | +0.0001 |

- maximum |ΔD| = **0.051°** (registered tolerance 0.5°; runbook STOP threshold 1°).
- maximum |Δγ_n| = **0.00050**.
- interleaved vs channel: maximum |ΔD| = **0.019°** (A4.5 expects agreement within 0.5°).
- k = 1 / k = 2 diagnostics (channel): g = 0 → 2.08° / 0.01°, g = 0.9 → 2.60° / 1.81°
  (§A: ≤ 2.1° / 0.0° and ≤ 2.6° / 1.8°).

Residual differences come from the A4 ε recomputation (§3.2.1): ε_dc = 0.0101072 from the A4 base
vs 0.0101959 from the v3 base used by `docs/jepa/check_v4_levels.py`. The A1 normalisation
`v_white = 0.4461113` is unchanged (reference-cell channel factor).

## v3 law reproduction (base ρ = 0.4, v3 θ SD; channel split, line 0)

| g | D (this run) | v3 §A | Δ |
|---|---|---|---|
| 0 | 18.36 | 18.4 | −0.04 |
| 0.005 | 74.00 | 74.0 | +0.00 |
| 0.05 | 84.63 | 84.6 | +0.03 |
| 0.45 | 89.20 | 89.2 | −0.00 |

Maximum |ΔD| = 0.043°: the v3 path is unchanged and reproduces its own Appendix A table
(`Tier1Views(base_rho=0.4, theta_sd=V3_THETA_SD)`, law `v3-A1-waveform-v1`).

## Notes

- θ SD default: `(3, 0.1, 0.04)/√12 = (0.8660254, 0.0288675, 0.0115470)`; the v3 values
  `(0.15, 0.005, 0.002)` remain selectable.
- ε_dc is recomputed from the A4 base (§3.2.1); the temporal PSD, line/common bases, loadings and
  the A1 `v_white` normalisation are unchanged.
- `time` and `block` rows are diagnostic only (target signal rank 2); they are reported, never used
  for a registered prediction.
- The population tables are exact model values, not sample estimates; sampling, E1b grids, M
  selection and E2 training are separate later steps.
