# Gate G2 — LUCiD compute budget

Status: **OPEN — awaiting the author**. Date: 2026-10-18. Lead: `lead`.

## The projection

Measured on deepthought2 (L40S, code `3e739b6`, clean clone, readout `readout_pmt_r6`,
`/ceph/dwong/jepa/env_overlay`), 500-step real-geometry smoke: **0.23014 s/step** average
(0.21388 s/step steady, 4.676 steps/s), peak 5 365 MiB. A registered run is 10 000 steps at
effective batch 32, so **0.6393 h/run** (average) or 0.5941 h (steady).

Registered LUCiD volume (§3.5.6): 201 training runs (main 162, R11 15, R11b 9, dark-window 3,
aging-trend 9, occupancy 3) + their evaluation.

| scope | training GPU-h | fraction of the 80 h L40S cap |
|---|---|---|
| 201 runs × 10 000 steps (as registered) | **128.5** (average) / 119.4 (steady) | **160.6 % / 149.3 %** |
| 3-seed fallback, 10 000 steps (126 main + 39 secondary = 165 runs) | 105.5 | 131.9 % |
| 3-seed fallback + all secondary dropped (126 runs) | 80.5 | 100.6 % (no eval margin) |
| 3 seeds + 5 000 steps, secondary retained (165 runs) | 52.8 (≈63 with evaluation) | ~79 % |

Training only — evaluation adds on the order of 10–20 %, so the first two rows are lower bounds.

## Options

**A. Apply the registered ladder as (1)+(3): 3 seeds for every main arm and 5 000 steps for every
LUCiD arm, secondary runs retained.** Projection ≈52.8 h training, ≈63 h with evaluation (~79 % of
the 80 h cap, margin for guard overhead). Cost: the 5 000-step grid is explicitly a **weaker
protocol** under §3.5.6 and must be recorded as such; the 3-seed fallback changes the seed floor
counts (R8's "3 of 3" clause then applies). R6/R7/R7b/R8/R11/R11b/T remain evaluable, R12 stays
`not_evaluable`.

**B. Apply the ladder literally, (1) then (2).** 3 seeds main; drop secondary runs from the end of
the list (occupancy, aging trend, dark-window anchor, R11b, R11) until the projection fits at 10 000
steps. This fits only after dropping **all** secondary runs (80.5 h before evaluation, i.e. no
margin), and it makes **R11, R11b, R7b, R8 and the aging-trend ratio `not_evaluable`**; R6/R7 stay
evaluable at 10 000 steps and 3 seeds.

**C. Split by host.** Main grid on the L40S within the 80 h cap (3 seeds + 5 000 steps as in A) and
put the secondary runs on Deepthought's Titan cap (300 h; the Titan runs the same code at
1.59 s/step, so 39 runs × 10 000 steps ≈172 h, within 300 h). This deviates from §6.1's
"criterion work on the L40S" comparability rule and would need an explicit author exception; the
probes would then come from two GPU types, which the comparability rule forbids in one comparison
set unless every comparison is within a single host.

**D. Optimise first, then re-project.** The L40S smoke is already close to the model-only scaling
estimate, so a large speed-up is unlikely without changing registered numerics (TF32 off, fp32,
frozen architecture); not recommended as the primary path, but a 1.3× data-path win would bring the
as-registered grid to ~99 h — still over the cap.

## Lead recommendation

**A** — it is the registered ladder, it keeps every prediction that can be measured evaluable, and
it fits with margin. The weaker 5 000-step protocol and the 3-seed counts must be recorded in
`DECISIONS.md` and in every run manifest, and the seed-floor rules re-read accordingly.

No criterion run starts until the author chooses.
