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
