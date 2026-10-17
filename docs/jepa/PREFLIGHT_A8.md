# PREFLIGHT (WP0-v4 / Phase 0 step 2–3) — both GPU hosts

Status: **complete, read-only**. Measurement window 2026-10-17 11:42–11:58 UTC. Detailed
measurements: `PREFLIGHT_A8_deepthought.md` (352 lines) and `PREFLIGHT_A8_deepthought2.md`
(151 lines). No job was started, no environment was built or modified, no remote write beyond a
`touch`+`rm` writability probe and the dmon sample.

## Access

| Check | Result |
|---|---|
| `ssh -o BatchMode=yes -o ConnectTimeout=20` | OK on both (`dwong@deepthought.etp.kit.edu`, `dwong@deepthought2.etp.kit.edu`; the short names do not resolve) |
| `git push` of branch `JEPA` | OK; `refs/heads/JEPA` = `6f41b52` visible from **both** hosts at 11:43 UTC |
| GitHub fetch of the pushed SHA | OK on both hosts (was `f021d3c` at 11:42:50, `6f41b52` after the push) |

## Hosts

| | **deepthought** | **deepthought2** |
|---|---|---|
| hostname | deepthought.etp.kit.edu | deepthought2.etp.kit.edu |
| cores | 40 (Xeon E5-2630 v4) | 128 |
| RAM | 503 GiB, 111.8 GiB free | 1006 GB, 694 GB free |
| load 1/5/15 min | 2.84 / 2.89 / 2.75 | 3.80 / 3.85 / 4.41 |
| GPUs | 4 × GeForce **GTX TITAN X**, 12 288 MiB | 2 × **L40S**, 46 068 MiB |
| compute capability | **5.2 (Maxwell)** — not sm_61 | 8.9 (Ada) |
| driver / CUDA | 525.147.05 / 12.0 → **cu118 required** | 570.211.01 / 12.8 → cu124 allowed |
| foreign GPU use | one idle 1-day-old Jupyter kernel (`amonsch`): 1011 / **4183** / 1007 / 1007 MiB, 0 % util | GPU0 0 MiB; GPU1 1705 MiB (`jjanik`, 3 procs); 5-min dmon **0.0 %** SM util both cards |
| usable now (§6.2 start rules) | GPUs 0, 2, 3 (GPU1 blocked: foreign > 1 GiB); = the 3-training-job cap | both cards |
| `/ceph/dwong/jepa/` | present, writable; `code/<old 72add60e>`, `logs/`, `raw/`, `returns/`, `venv/`; **no** `jobs/`, `data/`, `env/`, guard/PID files, tmux, our processes | same shared filesystem; same contents |
| quota | CephFS **directory** quota via virtual xattr: `max_files = 500 000`, `max_bytes = 0` | same (shared `/ceph/dwong`) |
| entries | 416 387 files + 51 669 dirs = **468 056 used, 31 944 free (6.4 %)** | same |
| bytes | no byte quota; `/ceph` 307 T free; JEPA area 377 MB | same |
| entry cost of one full clone | 658 entries / 308 MB | 657 entries / 308 MB |

The `/ceph` mount on deepthought2 **passes** — no gate, no author-named path needed (§6.1).

## Environment decision (§6.1 order)

- **Step 1 (existing GPU torch env, read-only) — taken on both hosts.**
  - **deepthought: `/home/dwong/anaconda3`** — Python 3.12.2, **torch 2.5.1+cu121**,
    `cuda.is_available() == True`, device capability (5, 2), arch list
    `sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90` (sm_50 cubin is binary-compatible with
    sm_52, so the Titans run); numpy 1.26.4, scipy 1.13.1, h5py 3.11.0, scikit-learn 1.5.1,
    jaxtyping 0.3.11, jax/jaxlib 0.4.30 **CPU-only**. It is writable, so by policy it is used
    read-only (`PYTHONDONTWRITEBYTECODE=1`). It lives in `/home` → 0 quota entries.
  - **deepthought2: `/ceph/dwong/transformer/.venv`** — Python 3.12.2, **torch 2.5.1+cu124**
    (the exact pin), `cuda.is_available() == True`, capability (8, 9); numpy 1.26.4, h5py 3.14.0,
    jaxtyping 0.2.38; **scipy, scikit-learn and jax absent**.
  - Builds differ by host (cu121 vs cu124), which is acceptable because every comparison set is
    single-GPU-type (§6.1): L40S criterion work on deepthought2, Titan work on deepthought. Each
    host's environment is still subject to the mandatory 10-batch equivalence test against the
    local environment before its first GPU job.
- **Step 2 (one new environment under `/ceph/dwong/jepa/env/`) — not permitted.** §6.1 requires
  ≥ 100 000 free entries; 31 944 are free (short by 68 056). Not attempted.
- **Step 3 (G-ENV) — not triggered**, because step 1 is viable on both hosts.
- **JAX / LUCiD generation.** JAX cannot use either GPU type: on deepthought the installed jaxlib
  has no CUDA plugin and sm_52 is below JAX's sm_70 minimum; on deepthought2 jax is absent and
  CPU-only in the fallback env. **LUCiD generation therefore runs on Deepthought's CPUs**, as
  §6.1 anticipates. CPU implementation on ≤16 cores, 48 wall-hour cap (§3.5.6).
- **Environment freeze**: each probe job records `python -m pip freeze` (or the conda
  `--export` equivalent) into its returns directory.

## Entry-quota plan (the one real constraint)

Only 31 944 entries are free and §6.2 forbids new data below 30 000, so the campaign has ~1 944
entries of headroom before any action. Ruling adopted (logged in `DECISIONS.md`):

1. Delete **our own two failed python-3.11 CPU venvs** under `/ceph/dwong/jepa/venv/`
   (+9 343 entries, 54.9 MB). They are inside our area, their setup evidence is preserved in
   `results/jepa/20261016_wp0_remote_env/` and `20261016_storage_recovery/`; nothing outside
   `/ceph/dwong/jepa/` is touched. → ~11 300 usable entries before the write-stop.
2. Code delivery stays one directory per SHA under `/ceph/dwong/jepa/code/<sha>/` but with a
   **sparse checkout** of the paths a run needs (`src/`, `scripts/`, `configs`, `pyproject`),
   ≈ 200–250 entries instead of 658, and clones whose runs are harvested are pruned after their
   manifests and hashes are recorded. Every run still records its SHA.
3. **Few large files**: data in shards ≤ 100 MB with `SHA256SUMS` (tens of files, not thousands);
   returns keep metrics JSON and the compact probe embeddings; checkpoints stay on `/ceph` only
   for the runs that need resume, pruned after harvest.
4. The guard and I track free entries; the guard SHEDs/reports below 30 000 and refuses starts
   below it. If the campaign cannot fit even after these measures, the response is **gate
   G-QUOTA** with lettered options (author requests a quota increase / author names a local-disk
   path / descope data volume), not a cleanup outside our area.

## Gate status

- `/ceph` on deepthought2: **PASS** (no gate).
- Environment: **PASS via existing read-only envs** (no G-ENV).
- Quota: **watch item, not a gate** at this point (see plan above).
- No item in Phase 0 step 2 failed.
