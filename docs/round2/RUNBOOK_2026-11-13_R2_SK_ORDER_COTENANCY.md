# Runbook addendum — Round 2: SK run order and GPU sharing on deepthought2

Date: 13 Nov 2026, 23:20. Parents:
- `RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` (sha256 `366f0680…`);
- `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md` (v2, `d237ec2b…`);
- `RUNBOOK_2026-11-12_R2_JUNO_TITAN.md` (v2, `0a8d76e9…`);
- `RUNBOOK_2026-11-13_R2_CAPS.md` (`d715eee6…`).

All parent rules hold unless changed here.

## 0. Authority

**Author, 13 Nov 2026, 23:17:** "你的建议A+B都采纳" (adopt both suggestions A and B).

**The suggestions** (23:14, translated from the Chinese):

> **A. Change the SK order (most important):** the next lane that frees up runs `sigreg_jepa` seed 9000 first, so that the seed-9000 set has all three arms, then the second posfree and `sigreg_jepa` seed 9001.
>
> **B. Relax the deepthought2 rule:** allow sharing a card with other users, with memory still ≤ 45 %, and raise the per-run wall cap from 1.5× to 2.5× the smoke time so that runs are not cut off halfway. The cost is that both we and the other users get slower.

**Standing rule (30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".

## 1. A — SK run order

From now on, a lane that is ready to launch a training job takes the highest-priority SK run that is not finished:
1. `sigreg_jepa` seed 9000 (not started);
2. `target_posfree` seed 9000 (resume from its checkpoint, step ≈ 3,168);
3. `target_posfree` seed 9001 (resume, step ≈ 3,548);
4. `sigreg_jepa` seed 9001;
5. `mae_mse_rf` (only if the 72 h cap holds it);
6. the parent's §8 seed-9002 rule.

**Rules for the reorder:**
- Re-queue both lanes so that `sigreg_jepa` seed 9000 is the next training job to launch.
- Do not kill a job that is actively training. Paused or held runs simply wait behind the higher-priority run.
- Evaluation of a finished model keeps going as soon as a lane is free, as before.

## 2. B — GPU sharing on deepthought2

1. **Starting next to other users.** Our job may start and run on an L40S that also has other users' processes, as long as all of these hold:
   - our job's peak memory is ≤ 45 % of the GPU (unchanged);
   - the GPU's total memory use, ours plus others, stays ≤ 90 % when ours starts, using our job's measured peak;
   - at most one of our jobs per GPU and at most two of our jobs in total (unchanged).

   Remove only the start and hold condition that refuses a GPU because another user has a process on it (the "other user > 2 GB on a GPU we use" HOLD).

   **Keep every protection for other users:**
   - shedding on GPU memory pressure, host RAM pressure, host load and storage entries;
   - the deepthought2 profile limits.

   **How to change it:**
   - Change the guard (`scripts/jepa_guard.py`, deepthought2 profile) and the lane runner only as far as this needs.
   - Add tests for the new start condition and for unchanged shedding.
   - Get a short fresh-context review of that diff before the new rule is used.
   - Record the before and after values in `DECISIONS.md`.
2. **Per-run wall cap: 2.5 × the smoke projection** for SK runs (was 1.5 ×), with all invocations still counted.
   - Update the watchdog to the new cap.
   - Runs the watchdog paused only because of the old cap become eligible again.
3. **Budget is unchanged.** The 72 L40S h ledger still applies, with shared-GPU wall hours booked as before. The priority order in §1 decides what the cap cuts.

## 3. Records and report

- `DECISIONS.md` gets this §0, the new guard values and the new cap.
- `STATE.md` and both ledger mirrors are updated.
- The r2-report states that SK runs after 13 Nov 23:20 trained on shared GPUs.
