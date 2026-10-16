# JEPA storage recovery — 16 October 2026

Status: cleanup complete; sole environment retry timed out, remote WP0 incomplete. The author explicitly authorized storage cleanup. This record separates filesystem recovery from scientific execution.

## Diagnosis

Ceph reports `/ceph/dwong` quota `max_bytes=0`, `max_files=500000`, recursive files444387 and directories55692, totaling500079 entries. The prior environment STOP therefore reflects file-count quota pressure despite ample global free bytes. Home uv/pip caches are on a separate filesystem and are not useful to this quota. Data archives and scientific datasets were preserved.

## Authorized targets and safeguards

The orchestrator approved exact manifest-listed `__pycache__` directories in `/ceph/dwong/venvs/dpo_batch3`, then additional observed caches in `/ceph/dwong/{paper1_env,meg_env,x3_env}` and deeper dpo_batch3 caches. Only regular non-symlink `*.cpython-*.pyc` with a regular adjacent `.py` source are deleted; ancestor symlinks are rejected. Cache directories are removed only if empty. Sources, sourceless bytecode, package metadata, raw arrays, checkpoints, prior failed JEPA environment and returns are preserved. No active candidate environment process was observed before cleanup.

Exact approved paths and counts are under `results/jepa/20261016_storage_recovery/approved*_cache_directories.json`; every deletion is recorded in `cleanup*_audit.jsonl`. Bounded inventory timeouts are retained. An unrelated `rm -rf models` process appeared in the active-use snapshot; it was neither started nor modified by this executor. Consequently aggregate quota changes cannot be attributed solely to this cleanup; audit counts establish our own deletion scope.

## Historical retry plan, before allocation

Preserve failed `/ceph/dwong/jepa/venv/cpu_py311_torch251` and its return directory. Use the existing clean GitHub checkout pinned at72add60e0e92ad4ec344bb309adbfc023b486455 for WP0 imports only. A single infrastructure retry requires measured headroom of at least25000 entries and orchestrator allocation. Use a fresh environment and return directory, Python3.11, torch2.5.1 CPU from the CPU wheel index, followed by exact remaining versions from the local frozen requirements (copied as retry_requirements.txt). Record any platform wheel difference. Set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1, RECONSTRUCTION_DISABLE_TORCH_COMPILE=1, TORCHDYNAMO_DISABLE=1 and PYTHONDONTWRITEBYTECODE=1. No GPU, training, benchmarks or scientific outcomes are part of this recovery.

## Cleanup disposition and allocated retry

Both cleanup processes ended. First pass exited124 at its180-second bound: 10280 bytecode files and 1376 empty directories are fully audited; any partial final-directory deletion is additional and not assigned an invented count. Additional pass finished15:19:34.545117 UTC and deleted12168 bytecode files (250148660 bytes), removing1352 empty cache directories. No additional cleanup was attempted. Exact summaries are in recovery_provenance.json.

At15:22:06 UTC no cleanup or environment retry processes remained, and the original failed JEPA paths were intact. Root counters407353 files+51376 directories give41271 free entries. The orchestrator allocated one600-second retry including installation/imports, one CPU thread, no GPU/outcomes. It was launched in fresh `/ceph/dwong/jepa/venv/cpu_py311_torch251_retry1` and `/ceph/dwong/jepa/returns/20261016_wp0_remote_env_retry1`. Original failed paths were preserved. This allocation was for infrastructure readiness only; its terminal timeout is recorded below.

Storage inspection through cleanup completion spanned438.545117 seconds (15:12:16–15:19:34.545117UTC), charged conservatively in full including timeouts. The subsequent resumed-state snapshot and retry are separately timestamped. Two disjoint cleanup subprocesses overlapped; no scientific CPU workers or GPU devices were allocated.

## Final disposition: storage recovered, environment incomplete

The single allocated retry ran15:22:24–15:32:24 UTC (exactly600 seconds) and outer `timeout` exited124 during the torch CPU installation step. Required imports were never reached. The shell EXIT trap recorded0 when externally terminated; that misleading low-level artifact is preserved and explicitly superseded by the outer124 terminal status and missing import evidence. No success is inferred from it. No second retry, scientific work or GPU job was launched.

At15:32:40UTC no retry/installation processes remained. Root quota counters416379 files+51668 directories leave31953 entries; storage capacity was restored, but the dedicated CPU environment is incomplete due to the fixed setup-time bound. Both the original quota STOP and retry1 remain intact. A further attempt requires explicit resolution and a new bounded allocation; none is inferred.

Remote compact evidence at `deepthought.etp.kit.edu:/ceph/dwong/jepa/returns/20261016_wp0_remote_env_retry1/` was copied with `rsync -av --checksum` into `results/jepa/20261016_storage_recovery/remote_env_retry1/`. All six entries in its remote SHA256SUMS passed on Deepthought and again locally. `retry_disposition.json` records the authoritative timeout, timestamp, version-request, import and resource status. Environment resolved versions are not asserted frozen because installation never completed.

Accounting:600 one-worker seconds for this retry, zero GPU hours; storage inspection/cleanup phase438.545117 seconds plus separately timestamped final receiving checks. These are infrastructure preparation charges to E1, not outcome-generating grids. Remote WP0 acceptance remains incomplete.
