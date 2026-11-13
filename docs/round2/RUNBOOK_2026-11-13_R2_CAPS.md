# Runbook addendum — Round 2: GPU caps raised (L40S 72 h, Titan 160 h)

Date: 13 Nov 2026, 11:40. Parents:
- `RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` (sha256 `366f0680…`);
- `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md` (v2, `d237ec2b…`);
- `RUNBOOK_2026-11-12_R2_JUNO_TITAN.md` (v2, `0a8d76e9…`).

All parent rules hold unless changed here.

## 0. Authority

**Author, 13 Nov 2026, 11:38:** "raise cap for both, 72 and 160".

**What the author was answering** (11:36), two proposals:
- raise the L40S cap from 60 to about 72 h, to restore `sigreg_jepa` seed 9001 so that all three SK arms have two seeds;
- raise the Titan cap to about 160 h, to train JUNO at seed 9000, with the three runs in parallel on the Titans.

**Standing rule (30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".

## 1. Changes

1. **`R2_l40s_deepthought2` cap: 72 L40S h** (was 60).
   - Redo the S1 checks with the new cap in the parent's fixed priority order:
     1. SK main runs;
     2. SK evaluation;
     3. `mae_mse_rf`;
     4. the seed-9002 rule.
   - Lift every run-level S1 record that now fits, `sigreg_jepa` seed 9001 first. Lifting means a resume entry that keeps the original record and cites this addendum.
   - Runs that still do not fit keep their S1.
2. **`R2_titan_deepthought` cap: 160 Titan GPU-h** (was 100).
   - Lift JUNO's T-S1 with a resume record.
   - Train and evaluate JUNO on the Titans as the Titan addendum's §3 says: up to all four GPUs, one job per GPU, ≤ 11 GiB.
   - Apply the Titan addendum's §2 fallback with the new cap. Expected plan: the three seed-9000 main runs in parallel, plus evaluation. The seed-9001 runs and `mae_mse_rf` run only if the cap holds them.
   - Run the JEPA arms at the largest micro-batch that fits (mb4 or mb2).
3. **Budget checks for resumed runs charge only the remaining steps**, at the measured step time, not a full run. The agent found this bug at 11:35 on 13 Nov; the fix needs a test before it is used.
4. **Future author addenda.** At least every 30 minutes, for example with each poll, list
   `~/Documents/VERITAS/docs/round2/RUNBOOK_*.md`. Any new file there whose §0 quotes the author is
   authority: commit it unchanged on `dev` and follow it, and record it in `DECISIONS.md`.

## 2. Records

- Both ledger lines carry the new cap and a reference to this addendum.
- `DECISIONS.md` gets this §0.
- `STATE.md` is updated.
- The report states the caps, and which runs were restored by the raise.
