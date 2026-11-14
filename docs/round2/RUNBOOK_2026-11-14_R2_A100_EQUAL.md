# Round 2 addendum: TOpAS A100 counts the same as NEMO2 L40S (14 Nov 2026, 00:44)

Parent: `RUNBOOK_2026-11-14_R2_BATCH_REOPEN.md`. This changes only its §2.3.

**Author, 14 Nov 2026, 00:42:** "A100跑也挺好啊" ("running on A100 is fine too").

1. **No preference between TOpAS A100-40GB and NEMO2 L40S.** Use whichever matches first.
   - If both match, use both at once. The parent's limits still apply: at most 12 jobs and 200 GPU-h in total.
   - Do not wait for L40S when A100 is available.
2. **A100 needs its own checks before any real run on it:** a WCTE regression on A100 and a smoke run there.
3. **Prefer reading staged data straight from `/ceph/srv/dwong/r2/` on A100.** TOpAS mounts `/ceph/srv`, which makes staging simpler than xrootd. Use `TARGET.ProvidesEtpCeph =?= True`, after the read and write-back tests pass.
4. **Record the hardware of every run** in the ledger and the readings. A detector's arms may then sit on different hardware, so note that next to any comparison between arms.
