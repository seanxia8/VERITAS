# Label-shortcut diagnostic — A10 dev (23 Oct 2026)

Task: `docs/jepa/TASK_2026-10-23_LABEL_SHORTCUT.md`. Budget ≤3 L40S GPU-h.

## 2a — no-training dependence check

32 dev events, `within` split, full-input read-out at target PMTs; 8 random target-output
projections per event; gradient norms with respect to context and target input rows:

| Encoder | G_context | G_target | context share G_c/(G_c+G_t) |
|---|---:|---:|---:|
| trained `sigreg_jepa` s9000 | 289.443 | 1984.412 | **0.1273** |
| random-init, same architecture/seed | 141.755 | 473.635 | **0.2304** |
| target-only encoding | — | — | **0 exactly** (`G_context = 0.0`) |

The trained full-input label has **less** context dependence than random initialization; this
supports the shortcut concern only weakly. The target-only pass is exactly context-free.

## 2b — paired training test

All runs: A10 dev cell, frozen config, 10,000 steps, L40S; evaluation is the frozen dev step-2
protocol (ridge, photon domain). Fixed labels use a development branch with
`label_full = jepa.encode(features, hidden_mask=~target)`; the current-label runs use the old path.

| Seed | label | vertex resolution | module-gain run-mean | final SIGReg | final predictor | effective rank |
|---:|---|---:|---:|---:|---:|---:|
| 9000 | current | 0.264400 | −0.004433 | 9.814 | 0.01329 | 7.021 |
| 9000 | fixed | **0.219259** | −0.002087 | 10.517 | 0.01379 | 4.560 |
| 9001 | current | 0.233759 | −0.004246 | 9.386 | 0.01167 | 8.290 |
| 9001 | fixed | **0.204281** | +0.005503 | 10.281 | 0.01326 | 6.186 |

Random-init floors: vertex **0.160904 m**, module-gain run-mean **0.218548**. Both fixed seeds
remain worse than the vertex floor and far below the module-gain floor.

Paired rule: `confirmed` iff r_fixed ≤ 0.90 × r_old in both seeds.

- seed 9000 ratio: **0.8293**
- seed 9001 ratio: **0.8739**
- branch: **confirmed**

No NaN, divergence or collapse; fixed effective ranks 4.56 / 6.19 > 2. GPU-h used by 2b runs
(including data generation/eval wall): **2.8193 h** (≤3 h). The two fixed runs are non-registered
and were run with `JEPA_ALLOW_DIRTY=1` on a development branch before committing the fix.

## Branch taken

**3A — confirmed.** Apply the target-only label fix, add reverse-leakage tests, re-run A10 dev
step 3 at the fixed SHA (≤15 GPU-h), freeze the new decision, then re-project the grid and resume
with the A10 `anchor_only` grid β file.
