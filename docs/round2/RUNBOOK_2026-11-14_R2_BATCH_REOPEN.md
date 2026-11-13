# Round 2 addendum: reopen the batch feasibility test (14 Nov 2026, 00:45)

Parents:
- `RUNBOOK_2026-11-13_R2_HTCONDOR_BATCH.md`
- `RUNBOOK_2026-11-13_R2_STAGING_DWONG.md`. Read it now if you have not.

This addendum reverses the B-S2 NO-GO recorded in commit `b751224`.

## 0. Why

1. **The B-S2 reason no longer holds.** The NO-GO said `/ceph/srv/dwong` does not exist. That was true at your 23:38 check, but the author created it at 23:55 (see `RUNBOOK_2026-11-13_R2_STAGING_DWONG.md`). So B-S2 is lifted.
2. **The real open problem is the no-match, and it was never diagnosed.** Tiny job 759618 matched no slot in 30 minutes, and the cause was not found.
   - `better-analyze` said all 60 slots were rejected by *the job's* requirements.
   - The last failed match came from `c03-001-140.gridka.de`, which is the TOpAS negotiator.
   - The NEMO2 slots you inspected advertise `Requirements = false`.

Keep `b751224` in history. Write a new feasibility record that supersedes it.

## 1. Facts from the ETP wiki (`docs/round2/reference/ETP_HTCondor_wiki_2026-11-13.pdf`, pp. 4–7)

- **NEMO2 GPUs:** L40S (9 nodes × 4) and H200 (2 nodes × 8). Jobs run on them "automatically as long as `+RemoteJob = True` and `RequestGPUs = 1`". `TARGET.CloudSite =?= "BwForCluster-Nemo2"` targets NEMO2 specifically.
- **NEMO account:** "When you use the NEMO cluster, please register your KIT account for NEMO." This may be why NEMO2 refuses us. Only the author can register.
- **TOpAS GPUs:** A100-40GB (`NVIDIA A100-PCIE-40GB`, 3 nodes × 8) and V100/V100S. TOpAS has `/ceph/srv`, which NEMO2 lacks. Use `TARGET.ProvidesEtpCeph =?= True` when you need it.
- **Free GPU overview:**
  ```
  condor_status -pool c03-001-140.gridka.de -constraint 'PartitionableSlot && TotalSlotGPUs>0 && State =?= "Unclaimed"' -af:t Name CPUs MEMORY DISK GPUs
  ```
- **Tutorial reference:** `GPU_batch/docker/submit.jdl` uses `accounting_group = cms.higgs`, image `tvoigtlaender/slc7-condocker-cuda-11.6-cudnn8-devel:base`, and no CloudSite requirement.

## 2. What to do

1. **Diagnose job 759618.** Record each item in the new feasibility record.
   - The job's full `Requirements` expression: `condor_history -l 759618 | grep -i requirements`.
   - `condor_q -better-analyze` per clause, against one NEMO2 L40S slot and one TOpAS A100 slot.
   - The state of each NEMO2 slot: `condor_status -const 'CloudSite =?= "BwForCluster-Nemo2"' -af Name State Activity Start`.
   - The free GPU overview above.
2. **Submit tiny variants.** Each is a 15-minute `nvidia-smi` job. Have at most 3 idle at once.
   - (a) The tutorial form unchanged: `+RemoteJob`, `request_GPUs = 1`, the tutorial's CUDA image, no CloudSite requirement. Start with `accounting_group = cms`; if refused, use `cms.higgs`.
   - (b) NEMO2 by CloudSite only.
   - (c) TOpAS A100: `TARGET.GPUs_DeviceName =?= "NVIDIA A100-PCIE-40GB"`.
   - Leave idle variants up to **3 h** before calling them no-match. Remote slots can take a while to appear. Re-check `better-analyze` every 30 min, and keep SK going meanwhile.
3. **Hardware allowed on batch.** Use whichever of these matches:
   - **NEMO2 L40S** (preferred, same hardware as deepthought2);
   - **TOpAS A100-40GB**;
   - **NEMO2 H200**.
   - Each type needs its own WCTE regression and smoke **before** any real run on it. Record the hardware for every run.
   - Do not use V100/V100S.
4. **Ledger.** Rename the batch ledger to `R2_batch_gpu`. The 200 GPU-h cap covers all batch hardware together.
   - Everything else in the parent stands: priorities, ≤ 12 jobs, ≤ 24 h walltime with chained resumes, checkpoints every ≤ 30 min, wall cap 2.5×, B-S1.
5. **Staging and checkpoints.**
   - **On TOpAS:** `/ceph/srv/dwong/r2/` may be readable directly when `ProvidesEtpCeph` is required. Test reading from it and writing back to it.
   - **On NEMO2:** follow `RUNBOOK_2026-11-13_R2_STAGING_DWONG.md` §1.3 (xrootd read and write tests, then the HTCondor self-checkpointing fallback).
6. **Stop rules.**
   - **If NEMO2 never matches but TOpAS does:** go ahead on TOpAS. Put "register KIT account for NEMO" in the author's to-do list in the report, as a possible cause.
   - **If no variant matches after 3 h:** B-S2. Record the per-clause analyses and remove all idle tiny jobs. The author will ask on Mattermost `ietp-htcondor` with that evidence.
7. **Small fix.** `status.sh` prints `DONE` for runs that are not done; for example, posfree9001 is still running and posfree9000 is paused at 3168. Fix the label so it shows running, paused or done.
