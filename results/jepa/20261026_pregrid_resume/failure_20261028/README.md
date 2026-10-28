# Part C queue failure — handoff evidence, 28 Oct 2026

`registered: false`. Handoff case (e): partial output and engineering failure; stop without relaunch. Grid remains paused.

The queue ran code `2b24630906d327231f05defa6d7ada5751988d2a`, 26 Oct 00:04:58–01:09:52 UTC on Deepthought Titan 0. Exit 1 after 3894.59 s = 1.081830556 GPU-h, charged in the ledger. The runbook Titan total is 1.412416667 GPU-h including the previous attempts. No new L40S job is evidenced by this queue.

`job.log` records five completed full/half probe pairs, then `ValueError` on construction of `sigreg_jepa_anchored`: the arm requires beta from the registered grid. `anc100` did not complete and there is no six-item `inspect.json`; this is an engineering failure, not an S2 verdict or Part C pass. Partial probe files alone do not establish acceptance of the complete check. No retry, overwrite, or Part D–F launch was performed.

The copied files reproduce the indicated paths beneath `/ceph/dwong/jepa/`. `SHA256SUMS` was generated on the host and all 17 files verified locally. The empty coordinator lock is evidence only. The original remote files remain in place.

At the read-only check, both hosts were reachable and `/ceph/dwong/jepa` existed on deepthought2. Ceph max entries 500000, used 428702, free 71298. The Deepthought guard file was stale (27 Oct 12:12:41 UTC); its OK cannot authorize a launch. GPU snapshots and Deepthought2's guard do not change the stop.
