# Preflight A8 — deepthought2 (read-only)

- **Operator:** host-operator subagent `op-deepthought2`, for the Lead. Date **2026-10-17**.
- **All timestamps UTC.** The host clock is local CEST − 2 h (checked: `11:42:08Z` = `13:42:08` local).
- **Scope:** launch-prompt Phase 0 step 2 (read-only preflight) + runbook v4 §6.1 (environment discovery). No job started, nothing installed, no environment created, no other user's process touched, no sudo, no interactive ssh.
- **SSH used for every call:** `ssh -o BatchMode=yes -o ConnectTimeout=20 dwong@deepthought2.etp.kit.edu '<cmd>'`.
- **Writes made (only under `/ceph/dwong/jepa/`, both allowed):** one `touch`/read/`rm` test file at `11:42Z`; the dmon sample log `/ceph/dwong/jepa/logs/preflight_dmon_dt2_20261017T1145Z.txt`.
- Everything below is a measured value or says explicitly why it is unavailable.

## GATE CONDITION — `/ceph` and `/ceph/dwong/jepa/`

**PASS — no mount gate on deepthought2.**
`/ceph` is a CephFS mount (`[2a00:1398:5:fe01:3eec:efff:fe38:7926],[129.13.101.127]:/ceph_data` → `/ceph`, `rw`). `/ceph/dwong/jepa` exists (`drwxr-xr-x dwong cms`, 5 dirs) and is **writable**: at `11:42:08Z` created `/ceph/dwong/jepa/.preflight_write_test_1236494`, wrote and read back `hello`, removed it — all three steps OK. No work path needs to be named by the author for this host.

## 1. Host (measured 11:42:08Z; load 11:51:11Z, 11:56:47Z)

| Fact | Value |
|---|---|
| hostname / FQDN | `deepthought2` / `deepthought2.etp.kit.edu` |
| kernel | `Linux 5.14.0-570.62.1.el9_6.x86_64` (RHEL 9.6) |
| nproc | **128** |
| RAM | 1006 GB total; 694 GB free / 974 GB available; 32 GB used, ~285 GB buff/cache; no swap |
| load average (1/5/15) | 3.80 / 3.85 / 4.41 at 11:51Z (3.13/3.80/4.90 at 11:42Z; 3.92/3.78/4.19 at 11:56Z) |
| uptime / users | up 15 days, 19:46; 11 users logged in |
| user | `dwong`, uid 12100, gid 1200(cms) |

## 2. GPU state (full `nvidia-smi` at 11:42:15Z; CSV 11:42:15Z; final 11:57:21Z)

| GPU | Model | Driver | CUDA (sms) | Mem used/total | Util now | **5-min mean SM util** | CC | Temp | Power | Other users' processes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | NVIDIA L40S | 570.211.01 | 12.8 | **0 / 46068 MiB** | 0 % | **0.0 %** (60/60 samples) | 8.9 (sm_89) | 34 °C | 32 W / 350 W | none |
| 1 | NVIDIA L40S | 570.211.01 | 12.8 | **1705 / 46068 MiB** | 0 % | **0.0 %** (60/60 samples) | 8.9 (sm_89) | 49–50 °C | 90 W / 350 W | jjanik: 3 pythons, 562+548+576 MiB |

- **5-minute sample:** `nvidia-smi dmon -s u -d 5 -c 60`, window **11:43:19Z → 11:48:19Z** (60 samples/card, 5 s apart). Log: `/ceph/dwong/jepa/logs/preflight_dmon_dt2_20261017T1145Z.txt`. Per-GPU mean of the `sm` column: **0.0 % on both cards, all 60 samples**.
- **Other users' processes on GPU 1 (never touched):**

| PID | Owner | Elapsed | CPU % | RSS | GPU mem | Command |
|---|---|---|---|---|---|---|
| 345663 | jjanik | 7 h 37 m | 94.9 | 8.2 GB | 562 MiB | `python -u scripts/run_train_jet_tagger.py --config btag_scnms4_bquarktruth_protopt_16const_vtxz_optimized/part_10k` |
| 1083481 | jjanik | 4 h 16 m | 90.8 | 8.3 GB | 548 MiB | `… /deepsets_10k` |
| 1209350 | jjanik | 47 m | 85.6 | 8.4 GB | 576 MiB | `… /pnet_2p5k` |

- MIG: not enabled (`MIG M.` = N/A). ECC: 0. Persistence-M: Off. GPU 0 holds **0 bytes and 0 %** — fully free. GPU 1 holds only 3.7 % memory.
- **§6.2 start test, as measured now:** GPU 0 passes (free, 0 % < 60 %). GPU 1 passes: other-user memory 1705 MiB + our 45 % cap 20 730 MiB = 22 435 MiB = **48.7 % ≤ 85 %**, and 0.0 % < 60 %. Both L40S are therefore eligible for a first smoke **right now**; the limit is 2 JEPA jobs, ≤45 % (20 730 MiB) per process, ≤16 CPU cores, ≤64 GB RAM.
- After all environment probes the cards were unchanged at 11:57:21Z (GPU 0 back to 0 MiB — probe CUDA contexts released; GPU 1 still 1705 MiB, still jjanik only).

## 3. Free entries and bytes under `/ceph/dwong`; quota mechanism (measured 11:42Z, re-checked 11:57:21Z)

| Quantity | Value |
|---|---|
| quota mechanism | CephFS per-directory quota on `/ceph/dwong` via xattr: `ceph.quota.max_files=500000`, `ceph.quota.max_bytes=0` (0 = no byte limit) |
| entries used | `ceph.dir.rentries=468056` (stable: 468056 at both 11:42Z and 11:57:21Z) |
| files / dirs used | `ceph.dir.rfiles=416387` (→ ~51 669 dirs) |
| **entries free** | **500 000 − 468 056 = 31 944 (6.4 % headroom)** |
| bytes used by dwong | `ceph.dir.rbytes=1 391 743 462 157` ≈ **1.39 TB** (bytes are *not* quota-limited per user) |
| `/ceph` global | 1.3 P size, 994 T used, **307 T available**, 77 % — capacity is not the constraint |
| `df -i` | Inodes 211 479 114, no used/free columns (CephFS does not report inode use; use `ceph.dir.rentries`) |
| home FS (`/home/dwong`) | `home:/export/home/dwong` 50 G, 31 G used, 20 G avail — **separate from /ceph**, does not count against the entry quota |

**What already exists under `/ceph/dwong/jepa/`** (rentries 10 018, 377 MB, `ceph.dir.rbytes=377423855`):

| Path | State |
|---|---|
| `code/72add60e0e92ad4ec344bb309adbfc023b486455/` | one GitHub clone, HEAD `72add60e…`, all files present |
| `logs/` | only the dmon preflight log created by this preflight |
| `raw/` | empty |
| `returns/20261016_wp0_remote_env/`, `…_retry1/` | the two failed v3 WP0 environment attempts (setup.log etc.) |
| `venv/cpu_py311_torch251/` | py3.11 venv, **no torch at all** (import fails) |
| `venv/cpu_py311_torch251_retry1/` | py3.11, torch dir present but **broken** (`module 'torch' has no attribute '__version__'`), no numpy |
| `jobs/`, `env/`, `data/` | **do not exist**; no `guard_*.json`/`guard_*.log`, no PID files |
| tmux | **no tmux server/sessions for dwong** (verified before, after, and after the sampler exited); no `jepa_*` sessions, no JEPA processes |

## 4. GitHub reachable / branch `JEPA` fetchable (from deepthought2)

`git` present (`/usr/bin/git`); GitHub reachable without credentials (`GIT_TERMINAL_PROMPT=0`).

| Time (UTC) | Command | Result |
|---|---|---|
| **11:42:50Z** | `git ls-remote https://github.com/seanxia8/VERITAS.git refs/heads/JEPA` | `f021d3c19e22acf485d97318e7b0f104730e6c57  refs/heads/JEPA` (pre-registration state) |
| **11:43:35Z** | same | **`6f41b524044a580410e68a0acb398cf14e062edc  refs/heads/JEPA`** — matches the Lead's push and local Mac `HEAD` (verified locally at 11:50:43Z) |

No clone/checkout was made. Note for the orchestrator: the local Mac worktree had uncommitted modifications (`src/jepa_noise/{backbone,jepa,run,sigreg,train}.py`) at 11:50:43Z — the fetchable SHA is the pushed commit `6f41b52`, not that worktree.

## 5. Existing GPU torch environments, usable READ-ONLY (§6.1 step 1)

All probes ran with `PYTHONDONTWRITEBYTECODE=1`, from `/tmp`, one GPU only (`CUDA_VISIBLE_DEVICES=0`), imports + `cuda.is_available()` only; no training, no installs, no pyc writes into any env.

| Path | Python | torch | torch.version.cuda | `cuda.is_available()` | `get_arch_list()` | numpy / scipy / h5py / sklearn / jaxtyping / jax | Writable |
|---|---|---|---|---|---|---|---|
| **`/ceph/dwong/transformer/.venv`** | 3.12.2 | **2.5.1+cu124** | 12.4 | **True**, device L40S, cc **(8,9)** | `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` | 1.26.4 / **MISSING** / 3.14.0 / **MISSING** / 0.2.38 / **MISSING** | yes (treat read-only) |
| `/home/dwong/anaconda3` (base conda) | 3.12.2 | **2.5.1+cu121** | 12.1 | **True** | not probed | 1.26.4 / 1.13.1 / 3.11.0 / 1.5.1 / 0.3.11 / jax+jaxlib **0.4.30 but CPU-only** (`default_backend=cpu`, "CUDA-enabled jaxlib is not installed") | yes (treat read-only) |
| `/ceph/dwong/x3_env` | 3.12.2 | 2.9.1+**cpu** | None | False | `[]` | 2.5.3 / 1.18.1 / MISSING / MISSING / MISSING / MISSING | yes (read-only) |
| `/ceph/dwong/venvs/dpo_batch3` | 3.12.15 | 2.13.0+cu130 | 13.0 | **False** — *"NVIDIA driver … too old (found version 12080)"* (driver 12.8 < cu130) | `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — **no sm_50/sm_52** | 2.5.1 / MISSING / MISSING / MISSING / MISSING / MISSING | yes (read-only) |
| `/ceph/dwong/paper1_env` | 3.12.2 | — (no torch) | — | — | — | 2.5.3 / 1.18.1 / 3.16.0 / MISSING / MISSING / MISSING | yes (read-only) |
| `/ceph/dwong/meg_env` | 3.12.2 | — (no torch) | — | — | — | 2.2.6 / 1.15.3 / 3.16.0 / 1.7.2 / MISSING / MISSING | yes (read-only) |
| `/ceph/dwong/ots_env` | 3.12.2 | — | — | — | — | all probes MISSING (bare venv) | yes (read-only) |
| `/ceph/dwong/y19_venv`, `zc_venv` | 3.12.2 | — (no torch) | — | — | — | numpy/scipy/h5py only | yes (read-only) |
| `/ceph/dwong/jepa/venv/cpu_py311_torch251[_retry1]` | 3.11.15 | broken/absent | — | — | — | all MISSING | yes (mine) |
| `/home/dwong/anaconda3/envs/rl-course` | 3.10.20 | 2.8.0+cu126 | 12.6 | True | not probed | 2.2.6 / 1.15.3 / MISSING / 1.7.2 / MISSING / MISSING | yes (read-only) |
| `/home/dwong/.venv` | 3.12 | 2.13.0+cu130 (from `version.py`, no import) | 13.0 | not probed | not probed | not probed | yes (read-only) |

- **Best candidate = `/ceph/dwong/transformer/.venv`**: torch **2.5.1+cu124** is exactly the repository pin, cu124 is the build the driver permits (570 ≥ 550), and `cuda.is_available()==True` on the L40S with `get_device_capability(0)==(8,9)`. It **lacks scipy, sklearn and jax** — if the LUCiD code needs them, `meg_env`/`paper1_env` hold them as separate read-only CPU envs, or this becomes a gate/decision; no install is permitted by this preflight.
- **Arm/`sm_50` note (Lead's question):** `transformer/.venv`'s cu124 build **does contain `sm_50`** (so sm_52 Titans are covered by within-major cubin compatibility), but `x3_env` is a **CPU-only** build (`get_arch_list() == []`, `torch.version.cuda is None`) and `dpo_batch3` is **cu130 with `['sm_75',…,'sm_120']`, i.e. no sm_50/sm_52**, plus it cannot even use driver 12.8. Neither of those two can run torch on a Titan. The deepthought side is being checked separately by the other operator. Caveat on the wording: CUDA arch lists never contain `sm_52`; `sm_50` cubins run on `sm_52` (Titan X-class) devices under within-major compatibility, but **not** on `sm_35` (original GTX Titan) — the Deepthought operator should read the actual `torch.cuda.get_device_capability()` on a Titan to settle it.
- `/ceph/dwong/transformer/.venv` exists on deepthought2's view of `/ceph` (25 492 entries); whether it runs on Deepthought is **not tested here** (different driver; other operator's check).
- Cold-cache caveat: first `import torch` from an unwarmed Ceph page cache took **>150 s** (dpo_batch3 and the first transformer probe timed out); the same focused import succeeded in seconds once warm (transformer retry 11:56:44Z). Expect slow first-process startup on this host, not a broken env.

## 6. Driver vs the repo pin; compute capability

- Driver **570.211.01**; `nvidia-smi` reports CUDA 12.8. Runbook rule: torch 2.5.1 **cu124** if driver ≥ 550, else cu118 → **cu124 build indicated on deepthought2**; `transformer/.venv` is already 2.5.1+cu124.
- L40S compute capability **8.9 = sm_89** (both cards, `compute_cap` CSV column and `torch.cuda.get_device_capability(0)` both give `(8, 9)`); the cu124 wheel has no sm_89 cubin but its `sm_86` cubin runs on sm_89 (minor-version compatibility), proven by the successful CUDA init/device query on GPU 0.
- JAX: the only JAX present is **0.4.30 with a CPU-only jaxlib** in `/home/dwong/anaconda3` (`jax.default_backend() == 'cpu'`), so JAX does **not** use the L40S in any existing env. Per §6.1 the LUCiD generation therefore stays on CPU (Deepthought) as planned.

## 7. Our processes / guard / tmux (11:57:21Z)

- No JEPA processes, no `jepa_*` tmux sessions, no tmux server at all for `dwong`.
- No PID files, no `jobs/` dir, no guard (`guard_<host>.json`/`.log`) — none exists yet; the guard has never run on this host.
- The only tmux session this preflight created was `preflight_gpu` (the 5-minute `nvidia-smi dmon` sampler); it exited by itself at 11:48:19Z and no server remains.
- Other users' three `jjanik` processes (GPU 1) were only read; never signalled or reniced.

## 8. Entry headroom arithmetic (§6.1/§6.2)

| Requirement | Threshold | Measured | Verdict |
|---|---|---|---|
| §6.1 step 2 — new environment under `/ceph/dwong/jepa/env/` | ≥ **100 000** free entries | **31 944** free | **NOT PERMITTED** — short by **68 056** entries. Use §6.1 step 1 (existing env, read-only) or gate G-ENV. |
| §6.2 — stop writing new data | free entries **< 30 000** | 31 944 free | **1 944 entries above the write-stop line** (3.9 % of remaining free). |
| cost of one code clone (per SHA) | — | **657 entries** (`find`: 549 files + 108 dirs; `ceph.dir.rentries=658`), **308 MB** (`= 516 tracked files`; `.git` packfile 138 MB) | at most **2** more clones before the 30 000 line is crossed (2×657 = 1 314 → 30 630; the 3rd → 29 973 < 30 000). |
| our JEPA area | — | 10 018 entries, 377 MB (2.1 % of used entries) | — |
| §6.2 — JEPA data cap | > 2 TB | 0.377 GB now; dwong total 1.39 TB (no byte quota) | not a constraint; **entry count is the binding constraint**. |

Consequence for the orchestrator: on deepthought2, **do not plan a new environment**, and budget code delivery as ~657 entries per SHA clone; with the guard's 30 000-entry write-stop, clone headroom is effectively **2 additional SHAs** unless entries are freed outside our area (which the hard rule forbids us to do without the author naming exact paths).

## Facts for the orchestrator

1. `/ceph/dwong/jepa` is mounted, present and writable on deepthought2 — **the §6.1 mount gate does not apply**; no author-named path is needed.
2. Two L40S (46 068 MiB each), driver 570.211.01, CUDA 12.8, sm_89. GPU 0 idle/empty; GPU 1 carries only 1705 MiB from `jjanik` (3 pythons).
3. 5-minute mean SM utilisation 11:43:19–11:48:19Z: **0.0 % on both cards**; both pass the §6.2 start test, with ≤45 %/20 730 MiB per our process and ≤2 JEPA jobs.
4. Host is quiet: 128 cores, 15-min load 4.41–4.90 (≪ 0.7×128 = 89.6), ~694 GB RAM free.
5. Entry quota: 500 000 max_files on `/ceph/dwong`, **31 944 free** → a **new env (needs 100 000) is not allowed**, and the §6.2 write-stop (30 000) is only **1 944 entries away**.
6. Bytes fine: no per-user byte quota, 307 T free on `/ceph`, our area 377 MB.
7. `git ls-remote` from the host reaches GitHub; `refs/heads/JEPA` = **`6f41b524044a580410e68a0acb398cf14e062edc`** at 11:43:35Z (was `f021d3c…` at 11:42:50Z).
8. Only existing-GPU-torch candidate that matches the pin: **`/ceph/dwong/transformer/.venv`** — py3.12.2, torch **2.5.1+cu124**, `cuda.is_available()==True` on L40S, `get_arch_list()` contains `sm_50` (Titan-compatible in principle), but **no scipy/sklearn/jax**. `/home/dwong/anaconda3` has the fuller scientific stack (scipy/sklearn/jaxtyping) with torch 2.5.1+cu121 and CPU-only JAX.
9. `x3_env` is CPU-only torch (arch list empty) and `dpo_batch3` is cu130/`sm_75…sm_120` with CUDA unusable under driver 12.8 — **neither is a GPU candidate**, and neither can run on a Titan.
10. No JEPA jobs, no tmux sessions, no PID/guard files on this host; nothing of ours is running. Cheap first step when authorised: one ≤10-minute GPU smoke on GPU 0 with `transformer/.venv` (that and a guard dry run are all that is needed before real work).

## Unavailable / not measured (explicit)

- **Deepthought side** (its `/ceph` view, its driver 525, Titan `sm_52` runnability): not tested here — other operator's item.
- `anaconda3`/`rl-course` `get_arch_list()`: not probed (not requested; both are on `/home`, not `/ceph`).
- `/home/dwong/.venv` details: version read from `torch/version.py` only (2.13.0+cu130); no import/CUDA probe.
- First cold-cache `import torch` from `/ceph` was abandoned after 150 s twice; it is a cache-warmup effect, not an env failure (focused retry on `transformer/.venv` succeeded at 11:56:44Z).
- True 5-minute mean was measured with `dmon -s u -d 5 -c 60`; utilisation is instantaneous per 5 s sample — 60 samples × 5 s = 300 s, which is the 5-minute window.

## Guard deployment prep (Lead request, measured 12:09:30–12:10:09Z)

1. **`/ceph/dwong/jepa/jobs`** — empty, `drwxr-xr-x`; `mkdir -p` returned 0. It **already existed when I checked** (mtime 12:09Z; it may have been created concurrently): the 11:57Z preflight had `/ceph/dwong` rentries 468 056 and `jepa` 10 018, the 12:09Z before/after counts are 468 057 / 10 019 — a net delta of exactly **1 entry, the directory itself**. Before and after my call were identical: `rfiles=416387`, `rsubdirs=51670`, `rentries=468057`, `rbytes=1391743462157`; `jobs` rentries = 1. Nothing else was created.
2. **Interpreter** — `/home/dwong/anaconda3/bin/python3` → `python3.12`, **Python 3.12.2**; `import torch, numpy, jaxtyping` → `2.5.1+cu121 12.1 True 1.26.4 0.3.11` (rc=0, `PYTHONDONTWRITEBYTECODE=1`). `cuda.is_available()` did open a CUDA context for the check only (freed on exit; GPU 0 back to 0 MiB at 12:10:09Z, no tensors, no compute). Caveat: this env lives on the **host-local** NFS home (`home:/export/home/dwong`), so its path is not automatically the same on Deepthought — the other operator must confirm there. The L40S job env remains `/ceph/dwong/transformer/.venv`.
3. **`nvidia-smi`** — `/usr/bin/nvidia-smi`, 570.211.01 (NVML 570.211, CUDA 12.8). **`pmon -c 1` runs (rc=0) but per-process SM%/mem are not populated** (`-`/`-` for two PIDs, `0`/`0` for one) → per-process utilisation is unusable on this driver; the guard must use global `utilization.gpu` plus `--query-compute-apps` memory.
4. **`getfattr`** — `/usr/bin/getfattr` 2.6.0. xattrs on `/ceph/dwong`: `ceph.quota.max_files=500000`, `ceph.dir.rfiles=416387`, `ceph.dir.rsubdirs=51670` (**supported**, unlike `ceph.dir.rdirs`), `ceph.dir.rentries=468057`.
5. **`tmux`** — `/usr/bin/tmux` 3.2a; `tmux ls` → `no server running on /tmp/tmux-12100/default` (rc=1) — **empty**, as required.
6. **`git`** — `/usr/bin/git` 2.47.3. Sparse-clone arithmetic from the measured existing clone (SHA `72add60e…`, itself a **blob:none partial clone**, not shallow, full checkout): full checkout = **657 entries / 308 MB** (549 files + 108 dirs; `.git` = 49 entries / 138 MB, of which 143.5 MB blob pack + 123 KB tree/commit pack); sparse checkout of code+docs (`src` 274 files/3.9 MB, `scripts` 28/0.13 MB, `docs` 78/1.3 MB, ~10 top-level files) ≈ **490 entries / 12–15 MB** (5.3 MB measured checkout + 49 .git entries + ~123 KB measured metadata pack + ~5.3 MB fetched blobs, estimate); `--filter=tree:0 --no-checkout` ≈ 49 entries but not runnable. Against the 1 944-entry margin above the §6.2 write-stop, that is ~**4 sparse clones** vs ~**2.9 full clones**.

## Phase 0 step 4 execution on deepthought2 (authorised by Lead, 2026-10-17 12:18–12:41Z)

Executed in the Lead-mandated order. Guard SHA `6468582`, clone at `/ceph/dwong/jepa/code/6468582` (HEAD `64685829ec7e4230b843a914785038cc7914f7d4`, sparse `src scripts`).

1. **Interpreter** — `/ceph/dwong/transformer/.venv/bin/python` is a symlink to `/home/dwong/anaconda3/bin/python3` (py3.12.2); in the venv it reports **torch 2.5.1+cu124, `torch.version.cuda` 12.4, `cuda.is_available()` True**. It was used as `JEPA_PYTHON` for the probe.
2. **Sparse clone** — note: `/ceph` is shared, and `/ceph/dwong/jepa/code/6468582` was created at 12:18Z by the Deepthought-side operator over the shared filesystem (`.git/logs/HEAD`: `clone` at 12:18:05Z, `checkout: moving from main to 6468582` at 12:18:09Z). My own `git clone` therefore returned `fatal: destination path 'code/6468582' already exists` (rc=128); my `git sparse-checkout set src scripts` and `git checkout 6468582` succeeded (rc=0), HEAD verified. Clone cost: **441 entries** (`ceph.dir.rentries`), 439 filesystem entries (378 files + 61 dirs), **6.7 MB** total — smaller than the 490-entry / 12–15 MB estimate. `/ceph/dwong` rentries 468 392 (12:18:15Z, clone mid-flight) → 468 499 (12:18:29Z).
3. **10-min dry run** — tmux `jepa_guard_dryrun`, 12:18:59Z → killed 12:30:08Z. 6 evaluations in `jobs/guard_deepthought2.log`: HOLD 12:18:59, HOLD 12:20:59 (util window warming), then **OK** at 12:23:00, 12:25:00, 12:27:00, 12:29:00Z. No `guard_deepthought2.json`, no SHED, no signals. Last evaluation: level OK, reasons [], start GPU 0, util window warm (3 samples / 240.5 s span), `entry_source.source=cephfs_xattr` (max_files 500000, rfiles 416769, rsubdirs 51732, free 31 499), unattributed memory GPU0 0.0 MiB / GPU1 19.0 MiB, foreign memory GPU0 0 / GPU1 1705 MiB (jjanik's 3 trainings), host free RAM 96.7 %, loadavg 15 m 3.46.
4. **Live guard** — tmux `jepa_guard`, started 12:30:09Z (PID 1261061, `/home/dwong/anaconda3/bin/python3`), stdout `jobs/guard_deepthought2_live.log`. First status 12:30:10Z HOLD (warming), level **OK** from 12:34:11Z onward. Immediately before the probe: status 12:36:11Z, age 21.7 s, level OK, start GPU 0, `dry_run=False`. Still running after the probe and is the only JEPA process on the host.
5. **Probe** (the single authorised GPU job) — launcher `--dry-run` inspection first (rc=0; `python=/ceph/dwong/transformer/.venv/bin/python`, `start_gpu=0`, memory fraction 0.45, timeout 900 s, kill-after 300 s, `PYTHONPATH=src`, module `jepa_noise.probe_env`). Real run in tmux `jepa_probe_deepthought2_6468582` at 12:36:38Z; `jepa_launch` PID 1263737, GPU 0; finished **12:38:45Z, exit 0, outcome exited, wall 126.07 s, timed_out false**, `run_status.json` written, `pid.json` removed. Probe facts (`returns/probe_deepthought2_6468582/probe.json`, sha256 `48237a04…c822c41`): `status: ok`; host deepthought2; python 3.12.2; numpy 1.26.4; `repo_head 6468582…`; `device_used cuda:0`; `torch.version 2.5.1+cu124`, `version_cuda 12.4`, `cuda_available true`, `device_count 1`, `device_name NVIDIA L40S`, `device_capability [8, 9]`, `memory_total_mib 45467.5`, `arch_list ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']`, cudnn 90100; `tf32_cudnn_after false`, `tf32_matmul_after false` (both explicit false); **10 fixture batches** (batch_size 4 = 10×4×8×256); timing 5 steps, mean **3.11 ms/step**, **1285.8 events/s**, peak 31.7 MiB; determinism max |diff| 0.0; model 595 460 params. `freeze.txt` written (75 lines, incl. `torch==2.5.1+cu124`), 1490 B, sha256 `937522b2…f7036`. GPU 0 peak during the job 433 MiB (424 MiB charged), back to **0 MiB** after.
6. **Guard after** — status 12:40:12Z: level OK, `our_jobs []` (the probe disappeared), start GPU 0, age 28.4 s at 12:40:40Z. Entry counts 12:40:40Z: `/ceph/dwong` rfiles 416784, rsubdirs 51737, rentries 468521 → **31 479 free**; `/ceph/dwong/jepa` 10 483 entries / 384 479 228 B (384 MB).
7. Nothing else was started; jjanik's three GPU-1 trainings were never touched; the only tmux session remaining is `jepa_guard`.

## Probe 89af5af — corrected 10-batch equivalence (12:57:23–12:58:36Z)

- **Clone**: `/ceph/dwong/jepa/code/89af5af` was created by the Deepthought operator over the shared `/ceph` (not cloned by me, per instruction); HEAD `89af5af6575522c68c8f58d45039b5fd8df12635`, sparse `src scripts`, worktree clean. The corrected `probe_env.py` contains `readout: 'pre_final_proj_norm_token_grid'`, `grid_sum`, `grid_abs_sum`, `grid_shape`.
- **Guard before**: status 12:56:15Z read at 12:57:23Z = age 68.6 s, level **OK**, start GPU 0, `our_jobs []`. Launcher `--dry-run` inspection rc=0 (same interpreter/GPU/mem-fraction/timeout as before).
- **Job**: tmux `jepa_probe_deepthought2_89af5af_v2`, `jepa_launch` PID 1273171, GPU 0; started 12:57:23Z, finished **12:57:27Z, exit 0, outcome exited, wall 3.74 s, timed_out false** (`run_status.json` written, `pid.json` removed). The 3.74 s vs 126.07 s earlier is a **warm Ceph page cache**, not a code difference — the full fixture/timing/freeze outputs are present.
- **probe.json** (`returns/probe_deepthought2_89af5af_v2/`, 6541 B, sha256 `b82c05c6…f4f30`): `status: ok`; `device_used cuda:0`; **`fixture.readout = pre_final_proj_norm_token_grid`**; **10 batches**, `per_batch` length 10, every `grid_shape` **[32, 8, 128]**, **no zero grid** (`grid_sum` e.g. 39.596 / −341.750 / −14.277, `grid_abs_sum` ≈ 26 000 for all; `zero_grid_batches = 0`); torch **2.5.1+cu124** / `version_cuda 12.4` / `cuda_available true` / L40S / `device_capability [8, 9]` / arch_list `sm_50…sm_90` / cudnn 90100; `tf32_matmul_after false`, `tf32_cudnn_after false`; timing mean **2.998 ms/step**, **1334.4 events/s**, peak 31.9 MiB, 5 timed steps; determinism 0.0; `freeze.txt` 75 lines incl. `torch==2.5.1+cu124`, sha256 `937522b2…f7036` (identical to the 6468582 run, as the interpreter is the same).
- **Guard after**: 12:58:16Z status, age 20.8 s at 12:58:36Z, level **OK**, start GPU 0, `our_jobs []`, reasons []; only tmux `jepa_guard` remains; GPU 0 back to **0 MiB**; jjanik untouched (GPU 1 1705 MiB foreign).
- Entry counts 12:58:36Z: `/ceph/dwong` rentries 468993, rfiles 417189, rsubdirs 51804 → **31 007 free** — the shared quota is now being consumed by the Deepthought operator's clone and runs too, so headroom above the 30 000 write-stop is ~1 000 entries.

## WP5 LUCiD L40S smoke — attempted 2026-10-18 00:56Z, BLOCKED on interpreter

- **Repo**: `/ceph/dwong/jepa/code/work` (shared clone) already at **`3e739b656548e63760f1a460ece1c90fc9a917f9`**, clean tree; `git fetch origin JEPA` OK (origin/JEPA then `044414b`), `git checkout 3e739b6` rc=0. Sparse: `configs`, `results/jepa/20261017_wp4b_detectability`, `scripts`, `src`.
- **Provenance pair (failed job)**: code `3e739b6…` + dataset manifest sha256 `a4ce3efbaf72e7fca1a259c0f6d728a777aec62e59d4e392e9b0e2ef3081088e`; `readout_pmt_r6`; generator config digest `61fdafa6…` (from the manifest); values config `lucid_run_v0.json` sha256 `c55f36d3…`; geometry `8b46bd3c…`.
- **Environment gap**: `/ceph/dwong/transformer/.venv` (the validated cu124 pin env) has torch 2.5.1+cu124, numpy 1.26.4, h5py 3.14.0, jaxtyping 0.2.38 but **no scipy and no sklearn** (`pyvenv.cfg`: uv venv, `include-system-site-packages = false`). `jepa_noise.lucid_run` imports `bridge_run` → `crlb` → `linear` → `scipy.linalg`, and 7 WP5 modules import scipy, so the runner cannot even import. The Lead's command path `/home/dwong/transformer/.venv` does not exist; the real path is `/ceph/dwong/transformer/.venv`.
- **First launch**: guard OK (status 00:55:00Z, age 90 s at launch), launcher gate passed, GPU 0, `wp5_lucid_smoke_l40s`; `jepa_launch` exit **1**, wall **1.87 s**, `run_status.json` written, `pid.json` removed, `timed_out false`; `job.log` = `ModuleNotFoundError: No module named 'scipy'` at `lucid_run.py:24 → bridge_run.py:101 → crlb.py:3 → linear.py:8`. **0 GPU-hours** (no CUDA context was created).
- **Fallback env measured read-only**: `/home/dwong/anaconda3/bin/python3` = torch **2.5.1+cu121** (cuda 12.1, `cuda.is_available() True`, L40S capability (8,9), arch_list `sm_50…sm_90`), numpy 1.26.4, **scipy 1.13.1**, h5py 3.11.0, sklearn 1.5.1, jaxtyping 0.3.11; `lucid_run smoke --dry-run --device cpu` rc=0. It deviates from the **cu124** build pin (§6.1), so using it for a registered comparison needs the Lead/author decision plus a recorded build and equivalence check; no new env is possible on quota (≈38.4 k free entries < 100 k, §6.1 step 2 → G-ENV).
- **Cleanup**: memory sampler tmux killed; only `jepa_guard` remains; GPU 0 0 MiB, GPU 1 1 116 MiB (two `jjanik` trainings, untouched). Guard after: 00:57:01Z, age 19.8 s, level OK, start GPU 0, `our_jobs []`. Entries 00:57:19Z: rentries 461 602 → **38 398 free**.
- The 40k scaling job is blocked by the same import chain (it imports `lucid_run` as well), so no scaling number exists yet either.

## WP5 LUCiD L40S execution with the read-only scipy overlay (Lead decision B; 2026-10-18 00:58–01:04Z)

**Overlay** — `/ceph/dwong/jepa/env_overlay/`, 5 symlinks into `/home/dwong/anaconda3/lib/python3.12/site-packages` (the Lead's `python3.11` path does not exist; the env is 3.12.2): `scipy`, `scipy.libs`, `sklearn`, `joblib`, `threadpoolctl.py`. Provenance: built for Python **3.12.2**; scipy **1.13.1**, sklearn **1.5.1**, numpy **1.26.4**. Hashes: `scipy/__init__.py` `f09d8a342acb52bb982073fbeb2594d9363ff55433d3dd710066142db7c0bee9`, `sklearn/__init__.py` `427232298b8a94e7f39c3bc3a2260e207c9b66903e71499b6f01bc8190c6d890`, `threadpoolctl.py` `f07b9a0efe634d128c829b57ae5a8ccf14ac4b236203bbbd6dc6fce2c99b55a3`. **Entry cost: 1 entry** (the directory) — CephFS `rfiles`/`rentries` count only regular files and subdirs, so the symlinks are free; overlay rentries = 6 (dir + 5 links).

**Equivalence evidence (all three required):**
1. Import under venv + `PYTHONPATH=/ceph/dwong/jepa/env_overlay:src`: `torch 2.5.1+cu124`, `torch.version.cuda 12.4`, `cuda_avail True`; `scipy 1.13.1`, `sklearn 1.5.1`; `scipy.__file__ = /ceph/dwong/jepa/env_overlay/scipy/__init__.py`, `sklearn.__file__` under the overlay; `scipy.linalg/optimize/stats`, `sklearn.linear_model`, `joblib`, `threadpoolctl 3.5.0` all import. `lucid_run smoke --dry-run --device cuda:0` rc=0.
2. `jepa_noise.probe_env` re-run under the overlay (GPU 0, 01:00Z, `returns/probe_overlay_l40s/probe.json`): status ok, `readout pre_final_proj_norm_token_grid`, grid shape [32, 8, 128]; compared with the committed host probe record (`returns/probe_deepthought2_89af5af_v2/probe.json`, same cu124 venv without overlay): **50 common scalar fixture values, max abs diff 0.0**, same torch 2.5.1+cu124, determinism 0.0, timing 2.86 ms vs 3.00 ms probe step (noise).
3. CPU numeric identity of the scipy-dependent WP5 path on a fixed 8×8 SPD input: `linear.invsqrt` (scipy.linalg.eigh), `crlb.crlb`, `scipy.linalg.subspace_angles` give **bit-identical sha256 and sums** under venv+overlay and under the anaconda interpreter: `invsqrt dcb3c0ea…d81cf`, `crlb 3895d820…72101`, `subspace ae778302…91002`; sums 2.1681470083940173 / 1.1306838386260498 / 2.02393225490001.

**500-step real-geometry smoke** (through guard + `jepa_run.sh`, `PYTHONPATH=/ceph/dwong/jepa/env_overlay:src`, `JEPA_PYTHON=/ceph/dwong/transformer/.venv/bin/python`, planned 15 min): `wp5_lucid_smoke_l40s`, GPU 0, **exit 0, outcome complete**, `jepa_launch` wall **118.6 s** (run `wall_seconds 115.7 s`, 00:59:54→01:01:49Z), `steps_completed 500`, no interruption. Throughput: train-log **500 steps in 115.07 s → 0.23014 s/step, 4.345 steps/s**; steady state (step 10→500) **0.21388 s/step, 4.676 steps/s**. Peak GPU memory (external 2 s sampler): **5 365 MiB** on GPU 0 (well inside the 20 730 MiB 45 % cap); the runner does not record a torch-allocator peak for training (only `run_scaling` does). Manifest numerics: `batch 32`, **`micro_batch 8`**, **`gradient_accumulation 4`** (effective 32), `steps 500` (smoke; registered runs 10 000), `optimizer muon_adamw`, lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight_decay 0, **fused true on cuda**, muon lr 1e-3/momentum 0.95/nesterov/ns_steps 5, constant schedule no-op, grad clip null; `readout_label readout_pmt_r6`; `registered false`; cell `smoke_s005_within`, split within, s_arm 0.05, s_p 0.05, seed 0, occupancy natural; code_sha `3e739b6…` clean, config sha `c55f36d3…`, loader manifest sha `a4ce3efb…`, geometry `8b46bd3c…`; final checkpoint sha `21136c65…` (7.5 MB). GPU-hours **0.0329**.

**40k scaling** (`wp5_scaling_l40s_40k`, batch 8, hit fraction 0.2, synthetic layout): **status `fit`, does_not_fit false** — wall **6.66 s**, 4.007 s measured, **1.997 events/s**, `peak_memory_mb 14 577.9`, `estimated_activation_mb 8 847.4` (cap 21 000), parameter_count 911 296, geometry_sha256 `ec50fbab…`; `scaling.json` sha256 `dfc7e6a7…14230`. GPU-hours **0.0019** (0.02 % of a 12 h allocation).

**Projection** (201-run grid, 10 000 steps/run, training only; evaluation/probing not included, so this is a lower bound):
- per run: **0.6393 h** (avg 0.23014 s/step) or **0.5941 h** (steady 0.21388 s/step);
- 201 runs: **128.5 GPU-h** (avg) or **119.4 GPU-h** (steady) — **160.6 % / 149.3 % of the 80 GPU-h L40S cap** (shortfall 48.5 / 39.4 GPU-h). The grid as specified does **not** fit the cap on L40S either; a G2 budget decision or the §3.5.6 fallbacks are needed (e.g. 5 000-step grid ≈ 64.2 h = 80.3 % at the measured average rate, before the 3-seed fallback).
- GPU-hours consumed this pass: smoke 0.0329 + scaling 0.0019 + equivalence probe ≈0.0008 + failed import attempt 0.0005 ≈ **0.036 GPU-h total**.

**Guard**: before smoke status 00:59:01Z age 50.4 s OK start GPU 0; during smoke the guard correctly showed `our_jobs ['wp5_lucid_smoke_l40s']` and moved the would-be start to GPU 1; before scaling 01:03:02Z age 12.3 s OK start GPU 0; after 01:03:02Z age 49.8 s OK start GPU 0 `our_jobs []`, reasons []. Only tmux `jepa_guard` remains; GPU 0 back to 0 MiB; GPU 1 1 116 MiB from two jjanik trainings (untouched). Entries 01:03:51Z: `/ceph/dwong` rentries 461 627 → **38 373 free**.

## G2 Option E: new SHA, overlay provenance, TF32 experiment (2026-10-18 09:46–09:50Z)

- **SHA**: `/ceph/dwong/jepa/code/work` fetched and checked out **`c87b31b84a3bebeb103aecabbbaf2df908f56573`**; `git log --oneline -1` = "jepa: G2 answered with Option E (L40S cap 80 -> 170 GPU-h), TF32 experiment plumbing (--tf32, recorded per run), overlay provenance requirement"; clean tree; sparse unchanged.
- **Overlay provenance** written to `/ceph/dwong/jepa/env_overlay/PROVENANCE.json` (sha256 `0746e7ceb85f751274b2e58b3e191e4b3e8fedf07ac6fe20f5c957246556b851`, created 09:46:50Z): symlink targets resolve to `/home/dwong/anaconda3/lib/python3.12/site-packages/{scipy,scipy.libs,sklearn,joblib,threadpoolctl.py}`; versions **scipy 1.13.1, sklearn 1.5.1, joblib 1.4.2, threadpoolctl 3.5.0**, on torch 2.5.1+cu124 / numpy 1.26.4 / Python 3.12.2. Hashes: `scipy/__init__.py f09d8a34…7c0bee9`, `sklearn/__init__.py 42723229…c6d890`, `joblib/__init__.py 1169bab9…867b82`, `threadpoolctl.py f07b9a0e…99b55a3` (the JSON also holds per-package sample `*.py` hashes). Re-verified unchanged before the TF32 batch. Rule adopted: re-verify these hashes before every batch; any change → stop and report.
- **TF32 experiment** (`wp5_lucid_smoke_l40s_tf32`, same 500-step real-geometry smoke, `--tf32`, guard-gated, overlay PYTHONPATH): exit 0, outcome complete, 500 steps, `jepa_launch` wall **97.24 s**, manifest **`tf32: true`**, `wall_seconds 94.68`, train loop 94.04 s → **0.18808 s/step, 5.3169 steps/s** (steady 0.18578 s/step, 5.3827 steps/s); peak memory **5 365 MiB** (unchanged). Against the fp32 baseline (0.23014 avg / 0.21388 steady, 4.3452 / 4.6755 steps/s): **speed ratio 1.224× average / 1.151× steady** — **below the 1.5× threshold**, so per the ruling we keep **fp32** and record that the measured gain did not justify the authorised speed experiment (which itself cost 0.027 GPU-h, inside the ≤0.1 GPU-h allowance).
- **GPU-hours this pass**: fp32 smoke 0.0329 + scaling 0.0019 + equivalence probe ≈0.0011 + failed import 0.0005 + TF32 smoke 0.0270 ≈ **0.063 GPU-h cumulative**.
- **State**: guard after 09:49:03Z age 45.1 s OK start GPU 0 `our_jobs []`; only `jepa_guard` in tmux; GPU 0 and GPU 1 both **0 MiB** (the two `jjanik` trainings finished overnight, untouched by us); `/ceph/dwong` rentries 461 677 → **38 323 free**. No host-side ledger file exists; the ledger lives in the repo (Lead).

## WP5 dev cohort — reference-fit SHA change and the loader-cache blocker (2026-10-18 09:51–10:25Z)

- **Wrong-SHA step 2**: started 09:51:39Z at `c87b31b`; killed 10:00:55Z on the Lead's instruction once the shared `dev_train` cache was complete (100 runs, manifest+SHA256SUMS). `outcome signalled`, exit 143, wall 555.67 s = 0.1544 GPU-h, note `superseded_by_reference_fit_config_fix`; invalid, not reused. It generated `dev_train` (1.5 GB) and completed 80/200 `dev_probe_train` runs, all complete per-run (manifest written last).
- **Valid step 2 at `f824075…`**: checkout verified (`git log --oneline -1` = the `fit_config` fix commit), worktree clean, overlay hashes re-verified unchanged. Launched 10:01:13Z with fresh path `returns/wp5_lucid_dev_v2`. It reused the 80 partial probe runs, generated the rest, and completed both dev probe datasets (`dev_probe_train` 200 runs/156 MB, `dev_probe_test` 200 runs/157 MB, both with manifests) — the **dev data cache is now complete and code-sha-independent** (`_verify_dataset` checks geometry/generator digests only).
- **BLOCKER found in training**: training ran at **3.07 s/step** (steps 100/200/300/400 at 307.2/614.2/921.1/1233.3 s), GPU util oscillating 0↔97 % (data-starved), versus 0.23 s/step in the smoke. Cause: `TrainingLoader` (lucid_run.py:781) takes `cache_runs = config["data"].get("loader_cache_runs", 12)` and the config has **no `loader_cache_runs` key**, so only **12** tokenised runs are cached while each 32-event batch draws 4 events from **8 runs chosen from 100**; measured tokenised run = **101.75 MB** (features [1000,1995,9] 71.8 MB, target/log_charge/hidden ~22 MB), so ~88 % of the chosen runs miss and are re-tokenised from CephFS every step. The 500-step smoke had only 2 runs, hence everything was cached — the smoke-based projection (0.23 s/step) does not hold for the real 100-run cohort.
- **Killed** 10:24:55Z (`outcome signalled`, exit 143, wall 1421.47 s = **0.3949 GPU-h**; run 1 reached step 400, run 2 `mae_mse` started and was interrupted; both invalid). Killed via the process group after the first SIGTERM was absorbed by the run-1 signal handler.
- **Projection without a fix**: 10 000 steps = 8.55 h/run → dev step 2 (2 runs) 17.1 h, dev step 3 (15 runs) 128 h, 201-run grid **≈1 718 GPU-h ≫ 170 h cap**.
- **Proposed fix (needs Lead commit+push; I must not edit the shared worktree or runs become non-registered)**: add `"loader_cache_runs": 128,` to the `"data"` block of `src/jepa_noise/configs/lucid_run_v0.json` → caches all 100 runs (~**12.8–13.6 GB** RAM per job, inside the 64 GB cap and host free RAM), restoring the ~0.23 s/step rate. Then re-measure with a 500-step smoke **on the dev cohort** (not `smoke_train`) before restarting step 2.
- **Ledger** (repo `RESOURCE_LEDGER.json`, v4_jobs now 27 rows; L40S charged **0.6127 / 170 GPU-h**): both kills recorded (`wp5_lucid_dev_step2` 0.1544 h with `superseded_by_reference_fit_config_fix`; `wp5_lucid_dev_step2_v2` 0.3949 h with the cache-thrash note).
- **State at 10:25:36Z**: nothing running but `jepa_guard`; both GPUs 0 MiB; guard OK; `/ceph/dwong` rentries 464 719 → **35 281 free**; data cache 1.5 G + 156 M + 157 M + 31 M.

## Loader-cache fix validated, then a second blocker: evaluation exceeds the 64 GiB RAM cap (2026-10-18 10:27–11:14Z)

- **Rate check** at `327c66d…` (`loader_cache_runs = 128`): a 500-step `dev --step 2` measurement on the **dev cohort** gave steps 100/200/300/400/500 at 60.96/82.58/104.27/125.96/147.66 s → **steady 0.217 s/step** after a ~61 s cache-fill warm-up (first 100 steps 0.61 s/step; average incl. warm-up 0.295 s/step). Job RSS **12.56 GiB** (100 tokenised runs). Fix confirmed: 3.07 → 0.217 s/step. (`--stop-after` is not honoured by `dev` mode, so the check ran the full 10k run until killed after 500 steps; killed 10:29:59Z, wall 175.21 s = 0.0487 GPU-h, recorded.)
- **Valid step 2 v3** launched 10:30:13Z (`wp5_lucid_dev_v3`, GPU 1, `ready 327c66d`). Run 1 (`sigreg_jepa`, seed 9000) **completed 10 000 steps in 2 195.2 s = 0.2195 s/step** (final.pt + checkpoint.pt saved at step 10 000). Then, during inline probe evaluation at 11:11:25Z, the **guard SHED** the job: `our processes use 66.1 GiB RAM > cap 64 GiB` → SIGTERM to our newest job; exit 143 (wtermsig 15), `timed_out false`, wall **2 475.24 s = 0.6876 GPU-h**; run 2 (`mae_mse`) never started. Guard then returned to **OK at 11:13:27Z**; no operator action was involved and no other user's process was touched.
- **Root cause (estimated from the code)**: `evaluate_arm_run` builds per-event probe tables for all four forms and six kinds. The `pmt_gain` per-event table is 10 000 events × 1 995 PMTs = 19.95 M rows × tokens (d=128, **float64**) = **20.4 GB per split**; train+test = 40.9 GB, and the `propagated` form adds ~40.9 GB more before `np.concatenate` copies — consistent with the observed 66.1 GiB peak. This will hit **every registered run's evaluation**, not just dev.
- **Options sent to the Lead**: (A) author-approved raise of the §6.2 all-JEPA RAM cap for deepthought2 (64 → e.g. 128 GB; host 1006 GB total / 664 GB free), then resume (auto-resume from the step-10 000 checkpoint and re-run evaluation); (B) code fix — shard the per-event probe evaluation in float32/chunks and free per kind (needed for the 201-run grid in any case); (C) dev-only `--probe-max-events` cap as a stopgap (not sufficient for registered runs).
- **Ledger**: L40S charged **1.3490 / 170 GPU-h** (29 v4_jobs rows; this run 0.6876 h under `killed_guard_shed`); `lucid_generation_cpu` charged **0.2620 / 48 h** (dev dataset generation 0.1906 h, single-threaded, 1 core — booked explicitly).
- **Re-measured cost**: 0.217 s/step steady → **0.603 h per 10k-step run** (training only); 201 runs ≈ 121.3 h, dev cohort (17 runs) ≈ 10.25 h, i.e. ~131.5 h of training before evaluation — inside 170 h if the eval blocker is solved; evaluation cost still unmeasured.
- **State 11:13:57Z**: nothing running but `jepa_guard`; both GPUs 0 MiB; guard OK; `/ceph/dwong` rentries 464 756 → **35 244 free**; invalid `wp5_lucid_dev_v3` retained as evidence.

## Standing by for the streaming evaluation fix (2026-10-18 11:15Z)

- **Lead decision B**: fix the probe evaluation (stream sufficient statistics in float32, chunked, 5 run-grouped folds; chunk size + measured peak recorded in `probes.json`); the 64 GB per-job cap and all other §6.2 rules stand; options A (cap raise) and C (`--probe-max-events` for registered criteria) rejected.
- **Resume path verified read-only**: `run_one` auto-resumes from `checkpoint.pt` when present; `wp5_lucid_dev_v3/runs/dev_s005_within__sigreg_jepa__s9000/{checkpoint,final}.pt` are at **step 10 000** with identity matching the dev step-2 run and **config_sha256 `eb38d994…` equal to the current config sha** → a rerun skips run-1 training entirely and only re-runs evaluation. `run_status.json` "running" is not read as a refusal. Caveat: if the fix changes `lucid_run_v0.json`, the checkpoint sha no longer matches and run 1 would need retraining (~0.6 h); nothing will be cleared without the Lead.
- **Planned fix re-run**: fresh `--out wp5_lucid_dev_v4` with only the v3 step-10 000 `checkpoint.pt` copied in → run 1 auto-resumes + evaluates, run 2 trains from scratch; expected ≈0.65 h; v3 kept intact as evidence.
- **Planned fix validation**: a 500-step `smoke --evaluate` (same `evaluate_arm_run`, small probe sets) with a 2 s RSS sampler and the guard, reporting measured peak RSS vs baseline, the chunk size/peak in `probes.json`, and eval wall time; booked as a separate non-registered L40S job.
- **Ledger**: L40S **1.3490 / 170 GPU-h** (29 rows); CPU generation **0.2620 / 48 wall-h**; nothing new until the new SHA arrives.

## Coherence fix, detectability re-freeze, dataset regeneration, streaming validation, dev step 2 (2026-10-18 11:28–13:05Z)

- **SHA chain**: `5f83806` (memory-bounded probe evaluation) → `615421c` (F3 spread coherence: `F3_SPREAD = F3_SPREAD_MAX = 0.10`, `F3_SPREAD_DEFAULT = 0.05`) → **`3dd4715`** (refreeze committed; configs point at the new detectability JSON). The old-SHA jobs (validation v2, dev step 2 v5) were stopped when superseded: step2_v5 wall 608.19 s = 0.1689 GPU-h; validation v2 29.87 s = 0.0083 GPU-h.
- **Detectability re-freeze at `615421c`, `--freeze`** → `/ceph/dwong/jepa/jobs/wp4b_detectability_wcte_r6_consistency.json`, file sha256 `411190bc2cfac55a6009e6e329485fe2905cc0c2bc9bae6262277433f1a807a7`, content_sha256 `8c9b96d25606eec8d7a973c9377da73352150fbe66d4d8fe694efa7abf6b35ae`, frozen; **F3 marginalised at s_p = 0.10** (old 0.05); yields **Y_lo 6298.44 / Y_hi 539434** photons; ladder = 4 steps (raise_y_hi 320000→539433.96, set_F2_a 0.1→0.2→0.5, raise_F3_spread 0.05→0.10); F1 eligible [0.05, 0.1], excluded [0.02]; F2 choice F2 @ 0.5; F3 cannot reach 0.6 → **R12 not_evaluable, no gate**; G6 false; law T ≥ 0.6 true; readout `readout_pmt_r6`, v_dark 0.00109. Old-vs-new ceilings shifted ≤0.0035 with **no pass/fail change**. Reported and committed by the Lead.
- **Dataset incoherence found and fixed**: the generator draws F3 gains with `F3_SPREAD` but the dataset digest does not cover it, so the s_p=0.05-era cache passed digest checks but failed per-run **state hash** verification (`ValueError: state hash mismatch`). Old cache moved (not deleted) to `/ceph/dwong/jepa/data/lucid_r6_v2/lucid__sp005_superseded` (3 157 entries / 1.97 GB) and regenerated at s_p=0.10 inside dev step 2: dev_train 604 entries/1.592 GB, dev_probe_train 1 204/0.163 GB, dev_probe_test 1 204/0.164 GB; ~11.0 min CPU single-threaded (CPU cap +0.1832 h). Archived evidence recorded in `/ceph/dwong/jepa/data/lucid_r6_v2/ARCHIVED_s005_EVIDENCE.json` (sha256 `f7126f74…203df0`; 6 datasets, 522 runs, each with manifest sha256, per-run state and event-id hashes).
- **Streaming-evaluation acceptance at `3dd4715`** (500-step `smoke --evaluate`, with 2 s RSS sampler): `probe_memory` chunk 64, `peak_mb 1853.1`, `peak_delta_mb 283.8`, `resident_delta_mb 134.7`, `n_rows_total 3 455 670`, `sufficient_statistics true`; wall 123.35 s, training 108.42 s → eval+setup 14.9 s. The former 66.1 GiB blow-up is gone.
- **Dev step 2 at `3dd4715`** (`wp5_lucid_dev_v6`, GPU 0, guard-gated, plan 240 min): valid, **exit 0**, wall **4 589.51 s = 1.2749 GPU-h** (includes 11 min data regeneration). `sigreg_jepa` 10 000 steps, 2 463.51 s, physics resolution 0.31541 m; `mae_mse` 10 000 steps, 1 264.53 s, resolution 0.09345 m. Probe memory at production scale: chunk 64, peak_mb 13 279.1 (process peak incl. 12.85 GiB loader cache), **peak_delta 426.0 MiB**, resident_delta 244.2 MiB, **n_rows_total 69 226 156**.
- **Step-2 decision** (`dev/step2_probe_decision.json`, sha256 `bc525bba554f8b7d1b354f4b2b60c795925e397a6a41b277c4eecec69c9f2d39`): **physics probe = ridge** (rule: MLP iff both linear resolutions > 3× reference; ratios sigreg_jepa 2.216, mae_mse 0.657 vs reference **0.14231 m**), reference **fit_config used {grid 5, scale_ns 3.0, t0_quantile 0.2, v_group 0.2172409115942029}** with `v_group_source configs/lucid_pmt/reference.json`; primary form run_mean; registered false. Accepted by the Lead.
- **Grid sizing (approved reclaim)**: one main-cell dataset = train 604 + probe_train 1 204 + probe_test 1 204 + robustness 1 204 + baseline 1 204 = **5 420 entries / ~2.25 GB**; one run's returns = 8 entries / 15.05 MB; 201-run grid ≈ 1 608 entries / 3.0 GB. Current free 31 792 → a single main-cell dataset would cross the 30 000 write-stop, so the approved reclaim at grid start is: archived cache 3 157 + dev datasets 3 012 (after decisions committed) + old clones 72add60e/6468582/89af5af 1 551 + stale smoke 144 = **7 864 entries**, giving ≈39.6 k free and rolling per-cell peaks ≈33–34 k. Lead authorised this reclaim, keeping `work`, returns, jobs, evidence JSONs, ledger; record every deleted path + entry count.
- **Dev step 3 running** (started 13:05:00Z, GPU 1, plan 720 min): 15 runs (3 anchored arms × 5 β); first `sigreg_jepa_anchored` run at ~0.29–0.30 s/step. β table and selected β to follow.

## A9 token-time-origin ruling — development results superseded (2026-10-18 15:39–15:41Z)

- **Author ruling (via Lead, act immediately)**: STOP dev step 3. The token time origin becomes the **true emission time (trigger t = 0)**, and the PMT readout gains the final-time rule **NHits with N = 4 within 20 ns, prompt window [−20, +60] ns around the trigger, outside digits dropped and counted, no-trigger fallback to the window start**, to be **imported from a PMT SHA the author will name (A9, do not reimplement)**. Detectability is not re-run (time-independent), so the frozen R-6 decisions stand. The step-2 ridge decision is also superseded and will be redone on the new input (~14 GPU-h, inside the 170 h cap).
- **Stop**: SIGTERM to the job process group at 15:39:21Z; the runner checkpointed each run and exited **0** at 15:40:08Z, wall **9 308.48 s = 2.5857 GPU-h** (started 13:05:00Z). No SHED, no write-stop, no forced kill.
- **Runs kept as superseded evidence (nothing deleted, nothing reused)** — marker `superseded_by_token_time_origin_change`; full record in `/ceph/dwong/jepa/returns/wp5_lucid_dev_v6/SUPERSEDED_BY_TOKEN_TIME_ORIGIN.json` (sha256 `743b0c1db709951986def9605424bd55092c80d94cf56de227b9df344cacb779`, 17 runs with walls/status/files):
  - complete: `mae_mse` 1264.53 s; `sigreg_jepa` 2463.51 s (step 2); `sigreg_jepa_anchored` β0.1 3053.59 s, β1 3010.50 s, β10 2828.34 s (step 3);
  - partial: `sigreg_jepa_anchored` β100 interrupted at step 1200 (362.99 s); β1000 and all five `anchor_only` and all five `anchored_dark` runs interrupted at step 0 (3.4–6.0 s each).
- **Decisions marked superseded**: `dev/step2_probe_decision.json` sha256 `bc525bba…c9f2d39`; `dev/step3_beta_decision.json` sha256 `3435892f…da796a7` (written from interrupted runs — invalid).
- **Dev datasets kept** (their fate is decided when the new SHA is merged); the archived-evidence JSON and the 17 kept manifest entries are intact; no deletion this pass.
- **Ledger**: L40S charged **5.4574 / 170 GPU-h** (36 rows; step 2 1.2749 h and step 3 2.5857 h both marked superseded); CPU generation **0.4452 / 48 wall-h**.
- **State 15:40:23Z**: nothing running but `jepa_guard`; both GPUs 0 MiB; guard OK (last status 15:38:25Z OK); free entries **34 861**; background waiter killed; the earlier archived-cache reclaim (authorised by the Lead, 3 157 entries deleted, evidence kept) stands and is recorded as a Lead engineering decision in DECISIONS.

## A9 redo at d073639: step 2 accepted, step 3 timed out, then a code crash (2026-10-18 19:35Z – 2026-10-19 17:25Z)

- **Input**: PMT SHA `5f4ecc0` merged (`4e07c0c`); repo `d073639…` ("use the merged PMT readout's prompt_window_mask directly…"). Generator imports `nhits_trigger_time` and `prompt_window_mask` (verified importable); token time = digit time − NHits trigger after the [−20, +60] ns prompt cut; emission offset U[−50, +50] ns truth-only; counters `n_no_trigger`/`n_prompt_dropped`/`n_dark_moved_trigger`. Config sha `ce452513…`; generator digest unchanged `61fdafa6…`; yields unchanged; detectability not re-run (time-independent).
- **Dataset regeneration**: old dev+smoke sets recorded in `/ceph/dwong/jepa/data/lucid_r6_v2/DEV_TIME_LEGACY_EVIDENCE.json` (sha256 `b4ceab6a…35cb2f`; 7 datasets/522 runs with manifests + state hashes) then deleted; fresh datasets at s_p=0.10 in the A9 convention (dev_train 100 runs, dev_probe_train/test 200+200), ~12.5 min CPU, 1 core.
- **dev step 2 at d073639** (`wp5_lucid_dev_v7`, exit 0, 5 108.21 s = **1.4189 GPU-h**): decision **ridge** — sigreg_jepa 0.32363 m (2.2719× reference), mae_mse 0.11349 m (0.7968×), **reference 0.142446 m** with the required `fit_config {grid 5, scale_ns 3.0, t0_quantile 0.2, v_group 0.2172409115942029}`; primary form run_mean; streaming probes chunk 64, 69 182 980 rows, peak_delta 432.5/76.1 MiB. Decision sha256 `8c6f5de3…121d9fc`. Accepted by the Lead.
- **dev step 3** (`wp5_lucid_dev_step3_v7`, started 21:04:42Z, planned 720 min): contention waves from `mbinder` (`/work/mbinder/src/neo-1p1q/train.py`, up to ~11.5 GiB on GPU 0, 100 % util; guard HOLD/OK flapping, **no SHED**) stretched it; the launcher's own **18 h calendar timeout** fired at 15:04:42Z (SIGTERM) and SIGKILL at 15:09:42Z — outcome `timeout`, exit 137, wtermsig 9, wall **65 100.78 s = 18.0836 GPU-h**. Completed 5/5 `sigreg_jepa_anchored` + 4/5 `anchor_only`; interrupted `anchor_only` β100 at step 7,917, `sigreg_jepa_anchored_dark` β0.1 at 8,339, dark β1 at 500; dark β10/β100/β1000 never started; **no step3 decision**. Recorded as a calendar event, not a SHED. (Three runs were unfinished, not one.)
- **Relaunch** `wp5_lucid_dev_step3_v7b` (same `--out`, `--duration-minutes 1800` → 45 h timeout, 16:08:46Z, GPU 1): re-evaluated the 9 completed runs, **verified resume** (`anchor_only` β100 `resumed_from_step 7917` → 10 000, 713.45 s; dark β0.1 resumed at 8 339 → trained to 10 000). Then it **crashed** in dark-arm evaluation: `AttributeError: 'Tensor' object has no attribute 'features'` at `lucid_run.py:1537`, `features = _to_device(batch.features, device)` inside `dark_only_embedding_change` — `_to_device()` expects a `LucidBatch` (it itself reads `batch.features`), so the call must be `_to_device(batch, device).features` (line 1261 is the correct pattern). Only occurrence; only the dark arm triggers it. Job exit 1, wall 3 498.79 s = 0.9719 GPU-h; stopped and reported; not worked around.
- **State**: 13/15 step-3 runs complete (5 anchored + 5 anchor_only); dark β0.1 trained to 10 000 but eval failed (cheap resume), dark β1 checkpoint 500, dark β10/β100/β1000 not started; no β decision. Guard OK (17:22:05Z); both GPUs free of our jobs; free entries **31 735** (below the 32 000 HOLD flag, above the 30 000 write-stop) — flagged to the Lead, with the option to reclaim the two inactive clones 72add60e+89af5af (1 111 entries) while keeping the active guard clone 6468582.
- **Ledger**: L40S **25.9319 / 170 GPU-h** (39 rows; timeout 18.0836 h + relaunch 0.9719 h; dev step-3 wall to date 19.06 h), CPU generation 0.4452/48. Awaiting the fixed SHA (one-line evaluation fix) to resume: dark β0.1 at eval, β1 from 500, β10/β100/β1000 from scratch → ~4–6 h, then the β table.

## A9 completion: dev step 2 and step 3 redone at d073639/122de8f (2026-10-20 00:26Z)

- **dev step 3 final**: `wp5_lucid_dev_v7c` at fix SHA `122de8f…`, exit **0, not timed out**, 17:30:26Z → 00:25:18Z, wall **24,891.27 s = 6.9142 GPU-h**, planned 1800 min (45 h). All 15 step-3 runs complete; decision written: `/ceph/dwong/jepa/returns/wp5_lucid_dev_v7/dev/step3_beta_decision.json` sha256 `00b83993…7b2a4c1f`.
- **Resume proof** (fix SHA, config sha unchanged `ce452513…`): `anchor_only` b100 resumed_from_step 7917 (completed earlier), `sigreg_jepa_anchored_dark` b0.1 resumed_from_step **10000** (evaluation only, 315.16 s), `anchored_dark` b1 resumed_from_step **500** → 10 000 (6 031.65 s); the dark crash (`AttributeError` at lucid_run.py:1537) is gone under the fixed line `_to_device(batch, device).features`.
- **β selection** (rule: largest β with vertex resolution ≤ 1.1 × plain sigreg_jepa 0.3236265 m = 0.355989 m): `sigreg_jepa_anchored` → **1000** (0.20788 m), `sigreg_jepa_anchored_dark` → **1000** (0.21010 m; β100 was 0.19867 but the rule takes the largest qualifying β), `anchor_only` → **none_qualified** (best 0.44891 m > 0.355989). Physics method ridge.
- **β table (physics m; nuisance R² = module_gain propagated / module_gain run_mean / pmt_gain propagated)**:
  anchored: 0.1 0.31082 / -0.00228 / -0.00416 / -1.20e-4; 1 0.29825 / -0.00251 / -0.00367 / -0.99e-4; 10 0.23610 / -0.00202 / -0.00413 / -0.78e-4; 100 0.20934 / -0.00241 / -0.00465 / -0.82e-4; 1000 0.20788 / -0.00155 / -0.00471 / -1.24e-4.
  dark: 0.1 0.31040 / -0.00233 / -0.00429 / -1.28e-4; 1 0.30487 / -0.00223 / -0.00339 / -1.04e-4; 10 0.24192 / -0.00201 / -0.00413 / -0.72e-4; 100 0.19867 / -0.00240 / -0.00428 / -0.90e-4; 1000 0.21010 / -0.00218 / -0.00446 / -1.01e-4.
  anchor_only: 0.1 0.57043 / -0.00065 / **0.05575** / -0.40e-4; 1 0.54840 / -0.00083 / **0.03255** / -0.47e-4; 10 0.47131 / -0.00104 / 0.00038 / -0.43e-4; 100 0.46216 / -0.00085 / -0.00694 / -0.35e-4; 1000 0.44891 / -0.00072 / -0.00661 / -0.43e-4. (anchor_only leaves module-gain nuisance unsuppressed at low β; anchored/dark arms keep it ≈0 from β0.1.)
- **Ledger**: L40S **32.8460 / 170 GPU-h** (40 rows; dev step 3 total 18.0836 timeout + 0.9719 crash + 6.9142 final = 25.97 h), CPU generation 0.6535/48.
- **Quota alarm**: free entries **30 603** — only 603 above the 30 000 write-stop. The dev datasets (3 012 entries: dev_train 604, dev_probe_train 1 204, dev_probe_test 1 204) can now be reclaimed once the step-3 decision is committed (the step-2 decision is committed; the step-3 JSON sits in the returns root, outside the data root). No job may start until that reclaim or another release; reclaiming the dev datasets restores ≈33.6 k free.
- **Grid budget conclusion**: refined projection ≈134–154 GPU-h (measured sigreg_jepa 0.743 h/run incl. eval; mae_mse 0.427; anchored/dark 0.8–1.1; mae_chi2/ijepa_ema/anchor_only/oracle still estimates) ⇒ ledger 32.85 + grid ≈ 167–187 h, i.e. at or above the 170 h cap; **option A (3 seeds / 5 000 steps) should be applied at grid start** (expected grid ≈55–60 GPU-h, total ≈88–93 h).

## Grid preparation at the author-review freeze (2026-10-20 00:27–00:30Z)

- **Dev datasets reclaimed** (Lead-authorised): `DEV_DATASET_EVIDENCE.json` written first in `/ceph/dwong/jepa/data/lucid_r6_v2/` — sha256 `f99f691988a2cebdbbaf19215b46d504f37e588a4c790d59f8b7089da259a37c`, 3 datasets / 500 runs with dataset manifest sha256 + per-run state/event-id hashes. Deleted `dev_train` (604 entries), `dev_probe_train` (1 204), `dev_probe_test` (1 204) = 3 012 entries; free entries 30 603 → 33 450 at deletion, 33 611 at 00:29Z. Data root now holds only the dark bank (4 entries) and the evidence JSONs.
- **Option A prepared** (not committed by the operator; Lead applies the patch): `/ceph/dwong/jepa/jobs/OPTION_A_lucid_run_v0.json` sha256 `736b32e1c503a8041ce0754f3c8ae5f481b01ae8931792e781cf0e44fff9b41b`, minimal patch `/ceph/dwong/jepa/jobs/OPTION_A_config.patch` sha256 `37041c9e493946a34a6c8218bc67a3ec481d5efccb8c661810cb188757c2740a` (steps 10000→5000; sigreg_jepa/sigreg_jepa_anchored/mae_mse seeds 5→3; r11 seeds 5→3; fallbacks active + option_A block). Validated: `load_run_config` OK, `fallback_active` = active (steps 5000, steps_fallback true, seed_fallback true); every manifest will record fallback active and `registered=false`. Expected **159 runs** (6 main cells × 21 + r11 9 + r11b 9 + dark_anchor 3 + law_t 9 + occupancy 3).
- **Dry runs**: `grid --cells main_s002_across --dry-run` with the option-A config prints the 21-run list; launcher dry-run through `jepa_run.sh` passed the guard (start GPU 0, nothing started).
- **Plan file**: `/ceph/dwong/jepa/jobs/GRID_PLAN_OPTIONA.md` sha256 `afd0d1aab70c41ffe5f24ab9ec70d128580aa5d61c62b304dfe49bb353aa0ae9` — groups G1–G5, per-job commands, rolling lifecycle, entry checkpoints, stop rules, projection.
- **Grid projection (option A)**: ~60 GPU-h (main ~47.5, secondary ~12.5) ⇒ ledger 32.8460 + ~60 ≈ **93 h of 170**; calendar ~33 h baseline with 2 GPUs (~1.4 days) plus contention margin. The grid will run under the recorded weaker protocol (5 000 steps, 3 seeds).

## Grid config corrected to a separate derived config + plan update (2026-10-20 00:34–00:36Z)

- Checked out grid SHA **e74922b1a92d64246047cf52bef4ca73e9cae95b** ("commit option A as a separate recorded config…"); worktree clean; overlay PROVENANCE hashes re-verified unchanged.
- **Registered default untouched**: `src/jepa_noise/configs/lucid_run_v0.json` sha256 `ce452513…` (steps 10000, 5 seeds, fallback inactive, 201 runs) — stays the default for development/smoke/tests.
- **Grid config**: `src/jepa_noise/configs/lucid_run_v0_optionA.json` sha256 **a1476d61eae21badb980b3e9615d245c11c6b993a34a8ed708e325f5975d44d2** (matches the Lead's); steps 5000, 3 seeds for sigreg_jepa/sigreg_jepa_anchored/mae_mse and R11, fallbacks active with the option_A record, `name`/`derived_from` present. Validated: `load_run_config` OK, `fallback_active` active (steps+seed), **159 runs**.
- **Plan updated**: `/ceph/dwong/jepa/jobs/GRID_PLAN_OPTIONA.md` sha256 `c9f31cb7a47b6670b56d3254a5f5837e91a6838488b8504980cf626b14b357d0`; every grid command now passes `--config src/jepa_noise/configs/lucid_run_v0_optionA.json`, the precheck file records config sha `a1476d61…` and HEAD `e74922b1…`, and the launcher budgets are 1800 min (45 h) for G1–G4 and 900 min for G5.
- **Dry runs**: `grid --cells main_s002_across --config …optionA.json --dry-run` prints its 21-run list; `grid --cells r11_f1f2f3_within … --dry-run` passes. The **launcher** dry-run is currently refused by the guard in HOLD ("no GPU satisfies the start condition" — both GPUs carry foreign memory: 4 570 / 3 621 MiB; no job of ours running), and will be repeated when the guard returns OK.
- **State**: no grid cell started; free entries 33,602 (after the dev-dataset reclaim 30 603 → 33 450); guard HOLD from other users, not our processes; awaiting the author review and the Lead's explicit go-ahead.
