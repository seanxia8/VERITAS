# 2026-10-16 storage recovery — audit vault (bulk files removed in WP0-v4)

This directory records the Deepthought quota `STOP` of 2026-10-16 and the recovery that followed.
The bulky infrastructure logs were removed from the tree in WP0-v4 (runbook v4 §7) and are
retained in git history; this README, `SHA256SUMS` and `recovery_provenance.json` are kept.

## What was removed, and where it still lives

Removed in commit `<hygiene commit, jepa: v4 hygiene (history retained)>`. Retrieve any file with the
blob id: `git cat-file -p <blob>` (or `git show f021d3c:<path>`).

| File | Bytes | Blob |
|---|---|---|
| `cleanup_additional_audit.jsonl` | 874 674 | `0fd1dd9afc4fcb600805732b079cd4d5ad409af1` |
| `cleanup_audit.jsonl` | 796 257 | `754db0953372d207379d9d251f2a8e9231f1a606` |
| `pycache_scandir_counts.jsonl` | 537 888 | `990ac31f260737886bb51016368d1359fab65853` |
| `approved_cache_directories.json` | 321 937 | `2186cb309ca7e7d8db2e08ec73177b7568049371` |
| `pycache_bounded_counts.jsonl` | 288 645 | `2c6e19020085fbe088f8c6a414098f512c5a40bc` |
| `approved_additional_cache_directories.json` | 280 342 | `195df50a1cd98b8e3307b292d26c4a8835f5e815` |
| `pycache_directory_counts.jsonl` | 12 505 | `d672df673d73d286006be0dd49fa047e95d326bf` |

Removed: the seven infrastructure audit/inventory files above, 3.11 MB of the directory's 3.18 MB.
Kept: `SHA256SUMS` (original hashes of all files, including the removed ones),
`recovery_provenance.json`, the quota inventories and progress notes (`quota_inventory.txt`,
`quota_progress.txt`, `quota_after_cleanup_progress.txt`, `directory_quota_stats.txt`,
`active_use_before_additional.txt`, `before_retry_state.txt`, `candidate_children.txt`,
`retry_*`), and the small `remote_env_retry1/` records.

## What the event was

A quota STOP on the Deepthought filesystem (`/ceph/dwong`), compounded by about 22k `.pyc` files in
other campaigns' environments. The v3 execution resolved it; v4 §0 F4 records that the cleanup
touched paths outside the JEPA area and that v4 §6 makes that a hard rule violation going forward.

The scientific result of the event is the WP0 remote-environment acceptance failure recorded in
`results/jepa/20261016_wp0_remote_env/`. Nothing here is evidence for any prediction.
