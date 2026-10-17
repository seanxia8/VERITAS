# GATE G-QUOTA — the shared `/ceph/dwong` entry quota cannot carry the campaign

Date: 2026-10-17. Written by the orchestrator. Blocks: nothing that is running now; blocks the
storage volume of WP4b generation and WP5 returns/checkpoints when the write-stop is reached.
Runbook v4 §6 ("a quota problem is a gate with options, not a cleanup"); §6.1 step 2; §6.2.

## Evidence

`/ceph/dwong` is a CephFS directory quota read from virtual xattrs (`getfattr -n ceph.quota.max_files
--only-values /ceph/dwong`): **max_files = 500 000**, max_bytes = 0 (no byte limit). Measured
2026-10-17 11:57 UTC from both hosts:

| | entries |
|---|---|
| used (files 416 387 + subdirectories 51 669) | 468 056 |
| free | **31 944** (6.4 %) |
| §6.2 write-stop (no new data below) | 30 000 free |
| available before the write-stop | **1 944** |

Our own JEPA area is 10 018 entries / 377 MB; the rest belongs to other campaigns of the same user
directory. One full code clone costs 657–658 entries (308 MB).

Planned campaign writing need, with the most compact layout we can defend:

| item | entries |
|---|---|
| code, one sparse checkout per SHA (~10 SHAs over the campaign, sparse to `src/ scripts/ configs`) | ~2 000–2 500 |
| job directories, 201 LUCiD runs × ~4 files (log, pid, metrics, config) | ~800 |
| returns, 201 runs × ~4 files (metrics JSON, compact embeddings, hash, README) | ~800 |
| LUCiD data shards, ≤100 MB, few large files | ~50–100 |
| development cohort, probes, robustness, E3/E4 | ~500 |
| **total** | **~4 200–4 800** |

So the campaign needs roughly **2.5× the remaining headroom**, and the margin is 1 944 entries
today. Without a decision we will either stop at the write-stop mid-campaign or let the guard SHED
jobs for storage; neither is acceptable silently.

Two failed python 3.11 CPU environment installs of ours are preserved under
`/ceph/dwong/jepa/venv/`: `cpu_py311_torch251` (24 entries) and `cpu_py311_torch251_retry1`
(9 317 entries, 55 MB). They are ours and inside the working root. Their installation evidence
(logs, requirements, exit status, verify script, provenance JSON) is already committed in
`results/jepa/20261016_wp0_remote_env/` and `results/jepa/20261016_storage_recovery/`, so deleting
the on-host trees loses no recorded evidence. §6's rule says a quota problem is a gate with
options, so this is proposed rather than done.

## Options

**(A) Author authorises deleting exactly these two paths** — `/ceph/dwong/jepa/venv/cpu_py311_torch251`
and `/ceph/dwong/jepa/venv/cpu_py311_torch251_retry1` — and confirms that this one-time deletion of
our own failed installs is the intended reading of §6 inside `/ceph/dwong/jepa/`. Frees 9 341
entries (→ 41 285 free, ~11 300 usable). **Recommended**: it is inside our working root, the
evidence is already in the repository, and no other user's data is touched. It buys roughly the
whole-campaign need but not a second full campaign.

**(B) Author asks the cluster operators to raise the `/ceph/dwong` directory quota** (for example
500 000 → 700 000 entries), or tells us which higher quota the directory already has. This is the
only option that removes the risk for the whole campaign and for re-runs.

**(C) Author names a local-disk path outside `/ceph`** for LUCiD data and returns (for example a
scratch filesystem on each host). §6.1's G-ENV (A) precedent applies. We would keep the working
root on `/ceph` for code and logs and write data where named.

**(D) Descope volume, keep the science grid:** all registered cells and seeds are kept, but we
write one HDF5 per dataset, prune checkpoints after harvest, keep returns to metrics + compact
embeddings + hashes, and accept that no infrastructure retry may write a second returns directory.
If this still cannot fit, the affected registered runs would have to be dropped, which is a §5 G5
change and would be proposed separately rather than assumed.

No option changes a registered prediction, metric, cell or dataset role on its own; (D) is the only
one that risks touching the registered grid, and only if it still does not fit.

## What continues while this is open

E1 (local, numpy), WP2-v4, WP4a code and tests, the guard, the 10-minute dry runs, the ≤10-minute
GPU probes, WP4b code and unit tests, and the scaling benchmark (small, few files). Generation and
E2 returns volume wait for the answer.

Answer with a letter (A/B/C/D). Until then no deletion is performed anywhere.
