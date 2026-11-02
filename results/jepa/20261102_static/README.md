# Frozen static-target test (target-content ablation) — 2 Nov 2026

`registered: false`. Extension `static_target_2026-11-02`, variant `sigreg_jepa_static_target` (`arm_name` stays `sigreg_jepa`).
Runbook: `docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md` (revision 2, sha256 `516e408e8ba35d4f1de475afefb4420e726a3386d0824a7462c20414ddb5b456`).
Author-approved training: exactly two runs (seeds 9000/9001) and one 20-step smoke; no other training. The frozen seed-0 random encoder (the same net as E3's frozen target) produces the label from geometry-only features (event columns 6-8 zeroed in the target pass); the context encoder keeps the full features.

- code SHA: `f382ad173b2d9900e86b121b92f91e9f4fa4ec77`
- §4.2 reproduction: PASS

## §2 manipulation check (implementation guard; within + natural occupancy)

| check | value |
|---|---|
| event_columns_zero | True |
| geometry_columns_unchanged | True |
| original_features_unchanged | True |
| label event fraction | 3.87482e-16 (max 0.001) |
| per-PMT label variance | 2.00268 |
| static deviation (max abs) | 0 |
| pass | True |

## §2 reading (two-sided, descriptive contrast against E3)

- **static_target_r = not recovered**
- not recovered (cannot distinguish staticity from a mere absence of target event-information; two-sided descriptive contrast against E3)

**Asymmetry caveat (runbook §1).** With a static detached label the MSE optimum is a constant output, so the prediction loss gives the context encoder no event-specific signal and the SIGReg target term is a constant with zero gradient (only the context half regularises). `recovered` is therefore nearly unreachable and would more likely signal a leak; `not recovered` cannot distinguish staticity from a mere absence of target event-information. This is not a verdict about H2.

| seed | full vertex m | full token R2 | half vertex m | half token R2 | f_layout | f_event | eff. rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9000 | 0.160374 | 0.953932 | 0.146063 | 0.894629 | 0.876094 | 0.0170764 | 5.70634 |
| 9001 | 0.161148 | 0.958827 | 0.152657 | 0.861323 | 0.836027 | 0.0172651 | 6.44362 |

Floors/criteria: recovered: both seeds full vertex < 0.160072, full token R2 >= 0.5, half vertex < 0.174821, half token R2 >= 0.5; not recovered: both seeds full vertex >= 0.160072 or full token R2 < 0.5

## §4.2 reproduction

| item | expected m | recomputed m | |diff| m | pass |
|---|---:|---:|---:|---|
| fix9000 | 0.219258 | 0.219259 | 5.91047e-07 | True |
| init9000 | 0.160072 | 0.160072 | 3.09078e-07 | True |

## Outputs

- `runs.json` — per seed: wall, full/half readout, f_layout, f_event, effective rank, resumed_from_step; the manipulation check and split pin.
- `readings.json` — the pinned `static_target_r` reading.
- `reproduction.json` — §4.2 check.
- `plan.json` — gate, data preflight, split/occupancy, manipulation check.
- `README.md` — this file.

