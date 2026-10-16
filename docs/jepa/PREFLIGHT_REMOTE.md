# JEPA remote WP0 preflight — 16 October 2026

Status: partial; no experiments or GPU jobs launched. Registration `f36c179ad5efca3305379ed3143ddd545b02c760` was independently visible from Deepthought through GitHub HTTPS. Runbook v3 and both final independent dispositions were read. Skeleton SHA was subsequently published and cloned; package environment/import acceptance remains blocked after the preserved quota failure and sole retry timeout.

## Initial read-only observations

At 15:05 UTC, `deepthought.etp.kit.edu` had 40 CPUs, 15-minute load 2.10, 403 GiB available RAM and 307 TiB free on Ceph. The formula gives `min(32,floor(40-2.10-4))=32`, before concurrency/RAM allocation. This is not a reservation. No dwong Condor jobs were returned. Existing other-study processes were inspected read-only and not altered; no PMT checkout was touched. Python 3.11 and uv are available. No torch or repository imports have yet been asserted.

## Condor mechanism

The compact repository pattern in `scripts/train_compact_l40s/submit.generated.jdl` uses explicit file transfer, returned artifacts, `accounting_group=delight`, `BwForCluster-Nemo2`, NVIDIA L40S, and `use_x509userproxy=False`. The full H5/XRootD pattern additionally requires a proxy, but that is not a universal requirement. Live proxy check found no valid proxy; compact transfer remains a viable pattern subject to a JEPA-specific pinned job/config, resource projection and gates. Condor advertised one Unclaimed Idle Nemo2 L40S slot, plus occupied/draining slots; scheduler visibility is not guaranteed job capacity. No submission was attempted.

## Data and remaining checks

Metadata-only path searches at depth four under `/ceph/dwong/{datasets,transformer,work}` found no TIDMAD-named path. A second bounded search timed out after 15 seconds and is retained with exit124. This is not proof that data is absent elsewhere. No data content, features, CH2 labels or outcomes were read, and no downloads occurred. E3's actual candidate inventory, published metadata, role manifest, hashes, channel schema and target eligibility remain unverified (G6 scope).

At the end of the initial read-only stage, the dedicated namespace and one-clone-per-SHA import check awaited published package code. No remote files were changed by that initial stage. The later setup attempt below did create the dedicated clone and failed environment. Raw preflight stdout is already local under `results/jepa/20261016_wp0_remote/`, with SHA256SUMS; no large raw scientific evidence exists. A later executor must use the registered remote return/rsync contract for actual jobs.

## Resource accounting

Successful metadata observations span 15:05:02–15:06:11 UTC (69 seconds); conservatively charge the entire span as one-worker preparation, not as measured process CPU. Initial sandbox DNS failure occurred before host contact. No campaign benchmark, training, GPU allocation or outcome-generating grid ran. Detailed timestamps, failure and completion statuses are in provenance.json and logs. Remaining environment checks must be appended as a distinct immutable attempt.

## Dedicated environment attempt — infrastructure STOP

The orchestrator published skeleton SHA `72add60e0e92ad4ec344bb309adbfc023b486455`. It was cloned from GitHub and detached at that exact SHA in `deepthought.etp.kit.edu:/ceph/dwong/jepa/code/72add60e0e92ad4ec344bb309adbfc023b486455`; subsequent Git status was clean. Python 3.11.15 created the dedicated CPU environment. Torch2.5.1 CPU downloaded, but installation exited2 with `Disk quota exceeded (os error 122)` at `/ceph/dwong/jepa/venv/cpu_py311_torch251/lib/python3.11/site-packages/.tmpCQLJED`. No imports or environment-equivalence pass is claimed.

Setup started15:07:09 UTC; failure confirmed15:08:03; bounded diagnostics ended15:08:38. Conservatively charge89 seconds preparation (includes transfer/diagnostics), in addition to the earlier69 seconds; total158 seconds, zero GPU hours. `quota -s` returned1 without detail; dedicated paths used308M code,41K venv,1.5K returns. Global free capacity does not establish user quota. At that initial stage byte versus file-count quota was unresolved, and no cleanup/retry had been attempted; later authorized recovery is recorded below. This is an infrastructure blocker, not a G2 campaign-cap exhaustion.

Remote compact evidence lives at `deepthought.etp.kit.edu:/ceph/dwong/jepa/returns/20261016_wp0_remote_env/`. The remote attempt to write SHA256SUMS also failed quota, but the remote stdout SHA256 of setup.log was captured as `a417da442dacfbd0603c436e71d8fcb53741176d653620f6816392a424578e4c`. `rsync -av --checksum` retrieved the compact log to `results/jepa/20261016_wp0_remote_env/`; its locally recomputed hash matches exactly. Local provenance explicitly records the remote checksum-file failure rather than asserting complete remote manifest creation.

Required resolution: restore writable capacity/quota under the mandated dedicated namespace, preserve this failed attempt, and allocate any authorized single infrastructure retry to a fresh environment/return directory. Remote WP0 acceptance remains incomplete until required imports and frozen resolved environment pass. No training/GPU submission or new outcomes occurred.

## Authorized recovery and terminal retry disposition

STORAGE_RECOVERY.md documents the500000 file-plus-directory quota, source-backed bytecode-only cleanup, and restored31953-entry headroom at15:32:40UTC. The sole fresh-directory retry ran15:22:24–15:32:24UTC and timed out with outer exit124. Its trap artifact records0 but is explicitly nonauthoritative; required imports never ran. Six remote inventory hashes were verified on receipt, and the orchestrator rechecked the complete recovery inventory. Original and retry environments/logs remain preserved. No further retry, pinned-source execution or GPU launch occurred. A new bounded recovery decision is required; available headroom alone does not establish environment acceptance.
