# Phase r2-run — round 2 multi-detector report (15 Nov 2026)

Extension `r2_multidetector_2026-11-11`, `registered: false`. Authority: runbook `RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` plus the author addenda (Gate V ceiling, JUNO Titan v2, caps, SK order and co-tenancy, HTCondor batch, staging, batch reopen, A100 equal), each committed unchanged.

## What ran
- **WCTE regression** passed on the L40S, on a GTX Titan and on a TOpAS A100.
- **X0 for SK, JUNO and HK:** the §2.2 tables match, trigger checks pass and yield ranges converged. Gate V failed on all three with the round-1 absolute thresholds. It is informational under the addendum, so every number below carries that caveat. Ceiling diagnostic: WCTE, SK and JUNO done (1.30 of 3 CPU wall-h); HK skipped (needs > 15 CPU-h). Preflight report: `results/round2/20261112_x0/R2_PREFLIGHT.md`.
- **Data:** 900 runs per detector, packed and verified byte for byte (`sk_r2.h5`, `juno_r2.h5`, `hk_r2.h5`), with entries in `DATA_MANIFEST.json`.
- **Training (all three detectors):** six main runs (mae_mse, target_posfree, sigreg_jepa × seeds 9000/9001) plus `mae_mse_rf` (seed 9000), and every model evaluated with the five `init` encoders.
  - SK trained on the deepthought2 L40S, with part of it on TOpAS A100.
  - JUNO trained on TOpAS A100. The Titans were NO-GO under T-S1: 316 Titan-h needed against a 160 h cap.
  - HK trained on TOpAS A100 (§8 fallback step 0).
- **§8 seed-9002 rule:** every trigger was trained and evaluated. SK `target_posfree` (R2-b unclear), JUNO `sigreg_jepa` (R2-a intermediate), HK `target_posfree` (R2-b unclear). No trigger was skipped.

## Readings (two-seed; the three-seed versions give the same classes)
| | R2-a fixed-label | R2-b position-free | R2-c MAE > floor | R2-d ranking (τ vs WCTE) | R2-e RF − MAE vertex |
|---|---|---|---|---|---|
| SK | event-invariant | unclear | yes (−0.35/−0.43 m) | MAE < init < posfree < fixed (0.67) | −0.160 m [−0.175, −0.146], confounded |
| JUNO | intermediate | not recovered | yes (−0.26/−0.47 m) | same (0.67) | −0.165 m [−0.177, −0.154], confounded |
| HK | event-invariant | unclear | yes (−0.50/−0.77 m) | same (0.67) | −0.330 m [−0.355, −0.305], confounded |

On all three detectors, MAE beats the random-init floor and fixed-label JEPA is the worst arm. Position-free JEPA does not separate clearly from `init`. MAE's lead over `init` is partly receptive-field driven: the RF-matched MAE gains a further 0.16–0.33 m. Seed-for-seed (two seeds), the ranking puts `init` ahead of position-free JEPA, while WCTE puts position-free JEPA ahead (τ = 0.67). Details: `results/round2/20261112_eval/READINGS_SUMMARY.md`.

## Stops
No detector ended stopped. Stops that fired and were later lifted by author addenda: S2 (SK, JUNO) by the Gate V addendum; S1 for JUNO and HK on the L40S line, plus JUNO's Titan T-S1, cleared by the batch GO (HTCondor batch addendum). SK `mae_mse_rf` ran on A100 after its L40S S1. NEMO2 batch was unusable (xrootd unreachable); TOpAS A100 was used under the A100-equal addendum.

## Resources against the caps
- **GPU:** L40S 53.51 / 72 h; Titan 0.81 / 160 h (feasibility only); batch 190.23 / 200 GPU-h (SK 22.85, JUNO 42.05, HK 125.13, probes 0.20).
- **CPU generation:** 4.25 wall-h (SK 2.01, JUNO 1.09, HK 1.16), within the remaining round-1 cap (22.68 / 48 h in total).
- **CPU evaluation and reports:** well under 12 wall-h, because the probes ran inside the GPU evaluation jobs.

## Commits (origin/dev)
79 commits in this phase: from 928bf7f (start, 12 Nov) to 260bf89 (HK results). Detector results are in 7e52b6b (SK), bab2e0a (JUNO) and 260bf89 (HK).
