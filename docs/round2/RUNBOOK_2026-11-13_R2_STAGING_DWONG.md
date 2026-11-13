# Round 2 addendum: batch staging goes to /ceph/srv/dwong (13 Nov 2026, 23:58)

Supersedes `RUNBOOK_2026-11-13_R2_BATCH_STAGING.md`, which is withdrawn. Parent: `RUNBOOK_2026-11-13_R2_HTCONDOR_BATCH.md`.

## 0. Author decision

**Author, 13 Nov 2026, 23:55**, from a Deepthought shell:
- `mkdir dwong` inside `/ceph/srv/dowling` failed with "Permission denied".
- `mkdir /ceph/srv/dwong` succeeded.
- The author's instruction: "dowling写不进去，你让他写dwong里面吧" ("dowling is not writable, have it write into dwong").

## 1. What changes

1. **Staging root: `/ceph/srv/dwong/r2/`.** This is the path the parent already uses in §2.3 and §3.
   - Over xrootd the path is `root://ceph-node-j.etp.kit.edu://dwong/r2/...`.
   - Set `USE_USER=dwong` explicitly in every submit file and executable. Do not source the tutorial's `setup.sh`: it falls back to another user's directory when it thinks `/ceph/srv/$USER` is missing.
2. **`/ceph/srv/dowling` is off limits.** `dwong` cannot write there. Never write, create or delete anything in it, and do not retry.
3. **Checks before staging.** Record the result of each in `results/round2/20261114_batch/feasibility.json`.
   - **Ownership and access.** As `dwong` on Deepthought, run `ls -ld /ceph/srv/dwong`. Then create `r2/`, write a test file into it and remove the test file.
     - Leave the permissions of `/ceph/srv/dwong` itself as the author created them.
     - Never make anything world-writable.
   - **xrootd read.** Read a small test file back through `root://ceph-node-j.etp.kit.edu://dwong/r2/`. Do this from Deepthought and also from inside one batch job on a NEMO2 L40S.
     - The directory is new, so this proves xrootd can see it.
   - **xrootd write from a batch job.** Write a small test file back from a batch job. The tutorial only reads through xrootd: its outputs return through HTCondor file transfer, so this write is not known to work.
     - **If xrootd write is refused** but reads work: keep checkpoints with HTCondor instead. Use self-checkpointing (`checkpoint_exit_code` + `transfer_checkpoint_files`), or output transfer at the end of each chained job. This is allowed only if one checkpoint set is under 100 MB.
     - **If xrootd write is refused and checkpoints are 100 MB or larger:** stop with B-S2.
     - **If reads of the new directory fail:** stop with B-S2.
     - In both stop cases, report the exact error message. The author will ask the ETP admins (Mattermost `ietp-htcondor`). The tutorial README notes that `/ceph/srv` user directories are normally set up by the admins.
4. **Do not touch other directories.** Nothing outside `/ceph/srv/dwong/r2/` is created, changed or removed. After the round, remove the staged copies as the parent §3 says and keep the checksum list.

Everything else in the parent stands: ledger `R2_batch_l40s`, cap 200 L40S-h, priorities, at most 12 jobs, stops B-S1 and B-S2.
