# Runbook addendum — Round 2: train on ETP HTCondor batch GPUs (JUNO, HK and the rest of SK)

Date: 13 Nov 2026, 23:40. Parents:
- `RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` (`366f0680…`);
- `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md` (v2, `d237ec2b…`);
- `RUNBOOK_2026-11-12_R2_JUNO_TITAN.md` (v2, `0a8d76e9…`);
- `RUNBOOK_2026-11-13_R2_CAPS.md` (`d715eee6…`);
- `RUNBOOK_2026-11-13_R2_SK_ORDER_COTENANCY.md` (`2e9869ea…`).

All parent rules hold unless changed here.

## 0. Authority

**Author, 13 Nov 2026, 23:35:**

> 1. accounting group是cms, 你让他先用 `condor_userprio` 和小作业去试
> 2. 200小时
> 3. 也搬过去

In English: (1) the accounting group is cms; check it first with `condor_userprio` and a small job. (2) 200 hours. (3) Move the rest of SK as well.

**The proposal the author was answering** (23:31, translated from the Chinese):
- First, a feasibility test: condor access; the data staging area and the environment; the WCTE regression and smokes on a batch L40S.
- If it passes: run all of JUNO and HK in parallel on NEMO2's L40S, and move SK's remaining runs there too (same GPU model). Runs already training on deepthought2 stay where they are.
- New ledger `R2_batch_l40s` with a cap of 150 L40S-h. The author set **200**.
- If it fails: record why, and round 2 continues as now.

**Source on the batch system:**
- the ETP wiki page "ETP HTCondor Batch System" (export of 13 Nov 2026, copy at
  `~/Documents/VERITAS/docs/round2/reference/ETP_HTCondor_wiki_2026-11-13.pdf`);
- the tutorial `htcondor-gpu-tutorial` on the author's Mac at `~/Downloads/htcondor-gpu-tutorial-main/` (see
  `GPU_batch/{docker,conda,cvmfs,LCG}/submit.jdl` and `executable.sh`).

Facts used from them:
- Jobs are submitted from portal machines, Deepthought among them.
- GPU jobs need `request_GPUs`, `+RemoteJob = True` and `+RequestWalltime`; they run in docker universe containers.
- NEMO2 offers L40S nodes (`TARGET.CloudSite =?= "BwForCluster-Nemo2"`, `TARGET.GPUs_DeviceName =?= "NVIDIA L40S"`).
- Remote jobs have no ETP `/ceph`. Large inputs move with `xrdcp` from `/ceph/srv/<user>` (`root://ceph-node-j.etp.kit.edu`), never with HTCondor file transfer over 100 MB.
- Fair share applies.

**Standing rule (30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".

## 1. What changes

1. **New ledger line `R2_batch_l40s`, cap 200 L40S GPU-h** (batch GPU wall hours, booked per job). `R2_l40s_deepthought2` (72 h) and `R2_titan_deepthought` (160 h) stay for work on those machines.
2. **Where things run after a GO** (§2):
   - **SK:** a run that is actively training on deepthought2 at GO time finishes there. Every SK run that has not started, or is paused, moves to batch L40S, resuming from its checkpoint where one exists. SK evaluations may also run on batch L40S (same GPU model).
   - **JUNO:** its stops (S1, T-S1) are lifted. All JUNO GPU work runs on batch L40S, not on the Titans.
   - **HK:** its S1 is lifted. HK first completes what generation needs on Deepthought's CPU (X0 yield range, trigger check and benchmark; its Gate V result is informational), then generation and pack, then all HK GPU work on batch L40S. The parent's §8 HK fallback applies with the batch cap.
3. **One GPU type per detector for training:** L40S throughout, so deepthought2 for SK so far and NEMO2 for everything moved. Every record carries the GPU model and the site.
4. **Micro-batch:**
   - For resumed SK runs, keep each run's micro-batch.
   - For new runs, use the largest of 8/4/2/1 whose peak memory is ≤ 90 % of the GPU; a batch GPU is exclusive, so the 45 % shared-server cap does not apply there.

## 2. Feasibility test (cap 3 h of wall time; GPU time books to the batch line)

1. **Access.** From Deepthought, run `condor_status` and `condor_q`, then `condor_userprio`, to see our user and the `cms` group.
2. **Tiny GPU job.**
   - Settings: docker universe; `accounting_group = cms`; `request_GPUs = 1`; `+RemoteJob = True`; `+RequestWalltime = 900`; NEMO2 L40S requirement.
   - The job runs `nvidia-smi` and a short torch CUDA check, and must run and report the GPU model.
   - If `cms` is refused, try once without a subgroup suffix, then once with the tutorial's `cms.higgs`, and record which one works. If none works, NO-GO (B-S2).
3. **Staging.**
   - `/ceph/srv/dwong` must exist and be writable. If it is missing, NO-GO (B-S2): the author has to ask the ETP admins.
   - Stage the following under `/ceph/srv/dwong/r2/`, each with sha256:
     - the code at the current `dev` commit (tarball);
     - a packed environment pinned to the exact torch and library versions used on deepthought2;
     - the detector HDF5 files and any checkpoints needed.
   - Jobs copy their inputs with `xrdcp`, check the sha256, run, and copy outputs back the same way: checkpoints every ≤ 30 min, run records, logs.
4. **Equivalence.** Run the WCTE regression (`init9000`, `mae_within_trigger9000`) on a batch L40S. It must match the committed values within 1e-3 m.
5. **Smokes.** 20 steps for each JUNO arm, and for SK `sigreg_jepa` at its micro-batch, on batch L40S. Record step time, peak memory, GPU model and node.
6. **Projection.** Project everything in §3 against 200 h.

**GO** if steps 2–5 pass. **NO-GO (B-S2):** record the reason in `results/round2/20261114_batch/feasibility.json` and `DECISIONS.md`. Round 2 then continues as it is now, and JUNO and HK keep their stops.

## 3. Execution under the 200 h cap

**Priority** when the cap cannot hold everything:
1. SK's remaining main runs;
2. SK evaluation;
3. JUNO's six main runs and evaluation;
4. HK main runs (with the parent's §8 fallback) and evaluation;
5. `mae_mse_rf` for SK, then JUNO, then HK;
6. the seed-9002 rule.

A detector's evaluation is reserved together with its main runs, so it is never cut after training.

**Jobs:**
- `accounting_group` as found in §2.
- `+RemoteJob = True`; NEMO2 L40S requirement; one GPU per job.
- `RequestCPUs` 4–8 and memory sized from the smoke.
- `+RequestWalltime` = 1.5 × the projected run time, at most 24 h. A run longer than that continues as a chain of resumed jobs from its checkpoints.
- At most **12** of our GPU jobs at once.
- Per-run wall cap: 2.5 × the batch smoke projection, all invocations counted.

**Monitoring:** `condor_q` and the job logs from Deepthought. A held or evicted job is resubmitted from its last checkpoint; resubmissions count toward the run's wall cap.

**Data on Ceph:**
- Results come back to the returns directory on `/ceph/dwong` (not `/ceph/srv`) and are committed as before.
- Free entries stay ≥ 30,000 on both areas.
- Remove staged copies from `/ceph/srv/dwong/r2/` after the round, keeping the checksum list.

**deepthought2:** keep running only the SK run(s) that were training at GO time, under the existing guard and sharing rules. Release the other lanes.

## 4. Stops

- **B-S1:** the projection exceeds the 200 h cap after the priority cut in §3. Runs below the cut are recorded as stopped (S1); runs above it go ahead.
- **B-S2:** NO-GO in the feasibility test.
- All parent stops (S3–S9) apply per detector. A stopped detector does not stop the others.

## 5. Records and report

- `DECISIONS.md` gets this §0, the accounting group used, the image and environment versions, and the staging paths.
- The three ledger lines are mirrored in `docs/round2/RESOURCE_LEDGER.json`.
- `STATE.md` is updated.
- The r2-report gives, per detector, where it trained (machine, GPU model) and the batch GPU hours.
