> **WITHDRAWN 13 Nov 2026, 23:58.** `/ceph/srv/dowling` is not writable for `dwong`. Superseded by `RUNBOOK_2026-11-13_R2_STAGING_DWONG.md`: stage under `/ceph/srv/dwong/r2/` and never write to `/ceph/srv/dowling`.

# Runbook addendum — Round 2: batch staging directory

Date: 13 Nov 2026, 23:50. Parent: `RUNBOOK_2026-11-13_R2_HTCONDOR_BATCH.md` (`878f5969…`). All of its rules
hold except the staging path.

## 0. Authority

**Author, 13 Nov 2026, 23:46**, after listing `/ceph/srv`: "dowling is mine, and how to create a /ceph/srv/dwong?"
The answer given (23:48): use the existing `/ceph/srv/dowling`; there is no need to create `/ceph/srv/dwong`.

## 1. Change

1. **Staging root: `/ceph/srv/dowling/r2/`** (replaces `/ceph/srv/dwong/r2/` in the parent §2.3 and §3).
   Jobs read and write it over xrootd: `root://ceph-node-j.etp.kit.edu://dowling/r2/...`, following the
   tutorial's `//${USE_USER}/...` form.
2. **Check before staging.**
   - From Deepthought, as `dwong`, check that `/ceph/srv/dowling` is writable (`touch` and remove a test
     file).
   - Check that an `xrdcp` round trip of a small file through `root://ceph-node-j.etp.kit.edu://dowling/`
     works.
3. **If `/ceph/srv/dowling` is not writable for `dwong`:** create `/ceph/srv/dwong` (the parent is
   world-writable, mode 777), use it instead, and record that in `DECISIONS.md`.
4. **Do not touch** anything else under `/ceph/srv/dowling` (or any other user's directory). Only
   `r2/` is ours to create and remove.
