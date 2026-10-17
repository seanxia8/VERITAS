# PREFLIGHT A8 — `deepthought` (read-only, Phase 0 step 2)

Host operator: `op-deepthought`. Remote user `dwong`, host `deepthought.etp.kit.edu`.
All measurements 17 Oct 2026, UTC times given per section. **Read-only**: the only write performed on
the host was the required `touch`+`rm` writability probe of `/ceph/dwong/jepa/`; no job, clone,
environment, install, mkdir, or signal was issued. No process of another user was touched.
Nothing in this file is a reservation of resources.

Command form for every remote call:
`ssh -o BatchMode=yes -o ConnectTimeout=20 dwong@deepthought.etp.kit.edu '<cmd>'` (BatchMode, no
interactive session, no sudo, no credentials stored). Long-risk probes were wrapped in
`timeout N`. SSH prints a post-quantum-key-exchange warning on stderr on every connection; it is
cosmetic and was filtered from quoted output.

---

## 1. Host

Measured 11:42:08 UTC (host local `13:42:08 CEST`).

| Fact | Value |
|---|---|
| hostname | `deepthought`, FQDN `deepthought.etp.kit.edu` |
| nproc | **40** |
| CPU | Intel Xeon E5-2630 v4 @ 2.20 GHz |
| uptime | up 59 days, 6:01, 6 users |
| load average | **2.84, 2.89, 2.75** (1/5/15 min) |
| RAM total | 527,721,272 kB = **503 GiB** |
| MemFree / MemAvailable | 117,243,932 kB = **111.8 GiB** / 395,547,988 kB = **377.2 GiB** |
| swap | 3 GiB, 0 used |
| `/ceph` mount | `[2a00:1398:5:fe01:...],[129.13.101.127]:/ceph_data on /ceph type ceph (rw,relatime,...)` |
| `/ceph` capacity | 1.3 PiB total, 994 TiB used, **307 TiB avail**, 77% |
| `/ceph` inodes (`df -i`) | 211,479,115, IUsed `-` (not reported; not the quota mechanism) |
| local disk | `lsblk`: sda 447.1G, `rhel_deepthought-root` 442.1G xfs on `/`, 4G swap |

`lsblk` shows no block device for `/ceph`: it is a CephFS network mount, so `lsblk` is not the
place to look for it (item 1 asked for both; both are quoted above).

Guard check: 15-min load 2.75 vs §6.2 limit `0.7 × 40 = 28` → **OK**. Host free RAM 111.8 GiB =
22.2% of total (MemAvailable 75.0%) vs the §6.2 floor "never below 20%" → **currently above, but
only ~2 GiB above the floor** by the strict MemFree reading; by MemAvailable it is comfortable.

---

## 2. GPU state

Measured 11:42:21 UTC (full `nvidia-smi`), compute capability and clocks 11:42:34 UTC, utilisation
samples 11:42:34–11:42:49 UTC (3 samples, 5 s apart).

Driver **525.147.05**, `CUDA Version: 12.0` (i.e. the driver's maximum runtime).
`nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv`:

```
index, name, driver_version, memory.total [MiB], memory.used [MiB], utilization.gpu [%], utilization.memory [%]
0, NVIDIA GeForce GTX TITAN X, 525.147.05, 12288 MiB, 1011 MiB, 0 %, 0 %
1, NVIDIA GeForce GTX TITAN X, 525.147.05, 12288 MiB, 4183 MiB, 0 %, 0 %
2, NVIDIA GeForce GTX TITAN X, 525.147.05, 12288 MiB, 1007 MiB, 0 %, 0 %
3, NVIDIA GeForce GTX TITAN X, 525.147.05, 12288 MiB, 1007 MiB, 0 %, 0 %
```

| GPU | Model | Compute cap | Mem used / total | Util (3 samples) | Owner / process | §6.2 start condition |
|---|---|---|---|---|---|---|
| 0 | GTX TITAN X | **5.2** | 1011 / 12288 MiB | 0 / 0 / 0 % | `amonsch`, PID 3103118 | mem 1011 MiB < 1024 MiB → *marginally* passes |
| 1 | GTX TITAN X | **5.2** | 4183 / 12288 MiB | 0 / 0 / 0 % | `amonsch`, PID 3103118 | 4183 MiB > 1 GiB → **fails / HOLD** |
| 2 | GTX TITAN X | **5.2** | 1007 / 12288 MiB | 0 / 0 / 0 % | `amonsch`, PID 3103118 | 1007 MiB < 1024 MiB → *marginally* passes |
| 3 | GTX TITAN X | **5.2** | 1007 / 12288 MiB | 0 / 0 / 0 % | `amonsch`, PID 3103118 | 1007 MiB < 1024 MiB → *marginally* passes |

Process table (all four GPUs, one foreign process):

```
|    0   N/A  N/A   3103118      C   ...envs/SANNT.dev/bin/python     1004MiB |
|    1   N/A  N/A   3103118      C   ...envs/SANNT.dev/bin/python     4176MiB |
|    2   N/A  N/A   3103118      C   ...envs/SANNT.dev/bin/python     1004MiB |
|    3   N/A  N/A   3103118      C   ...envs/SANNT.dev/bin/python     1004MiB |
```

Owner: `amonsch 3103118 3100461 1-04:38:27 1.4 6.5 34458168
/work/amonsch/venvs/miniforge3/envs/SANNT.dev/bin/python -m ipykernel_launcher --f=/run/user/12278/jupyter/runtime/kernel-v3a1ee17c32322983a2fc689eb12e52ec84b6085f9.json`

So: an idle Jupyter kernel of another user has held ~1 GB (4.2 GB on GPU 1) for >1 day, at 0%
utilisation. **No GPU is fully free in the sense of "no foreign allocation"**, but GPUs 0/2/3 are
just under the 1 GiB line and GPU 1 is over it. Because that kernel's allocations are idle and
stable, the ORCHESTRATOR should expect the guard to report **OK on 0/2/3, HOLD on GPU 1**, with the
caveat that "1 GB" in §6.2 is ambiguous between 1000 MB (then 1007–1011 MiB fail) and 1024 MiB
(then they pass) — a one-line decision is needed in `DECISIONS.md`. Nothing was signalled or
reniced; this process is not ours and was only inspected.

**Two findings that override the brief/runbook wording (recorded, runbook not edited):**

1. The cards are **GTX TITAN X (Maxwell, GM200), compute capability 5.2 — not sm_61.** The launch
   prompt's "GTX Titan = sm_61, CUDA cap 6.1" is wrong for this host. Consequences: no TF32 (the
   §6.1 "TF32 off" lines are no-ops on Maxwell), no tensor cores, no native bf16, and any wheel
   must contain an **sm_50** cubin (binary-compatible with sm_52 within the same major revision)
   or PTX ≤ 5.2 — see §5.
2. Driver 525.147.05 → **cu124 is excluded by §6.1's own rule** (driver < 550). Quoting it:
   "Check `nvidia-smi`: a driver ≥ 550 allows the cu124 build; otherwise use cu118."

---

## 3. Storage and quota

Measured 11:43:04 UTC (quota), 11:43:07 UTC (tree), re-checked 11:57:10 UTC.

`/ceph/dwong/jepa/` is **present, mounted and writable**:

```
TOUCH_OK /ceph/dwong/jepa/.preflight_touch_1792237343
REMOVE_OK
```

Contents (depth 3, 11:43:07 UTC):

| Path | files | subdirs | entries | bytes |
|---|---|---|---|---|
| `/ceph/dwong/jepa` | 9,604 | 414 | 10,018 | 377,423,855 (360 MiB) |
| `…/jepa/code` | 549 | 110 | 659 | 322,517,564 |
| `…/jepa/code/72add60e0e92ad4ec344bb309adbfc023b486455` | 549 | 109 | 658 | 322,517,564 |
| `…/jepa/venv` | 9,046 | 298 | 9,344 | 54,895,434 |
| `…/jepa/venv/cpu_py311_torch251` (failed, stub) | 20 | 5 | 25 | 35,207 |
| `…/jepa/venv/cpu_py311_torch251_retry1` (failed, timed out) | 9,026 | 292 | **9,318** | 54,860,227 |
| `…/jepa/returns` | 8 | 3 | 11 | 4,683 |
| `…/jepa/logs` | 1 | 1 | 2 | 6,174 |
| `…/jepa/raw` | 0 | 1 | 1 | 0 |

Present: `code/`, `logs/`, `raw/`, `returns/`, `venv/`. **Absent: `jobs/`, `data/`, `env/`** — so
`/ceph/dwong/jepa/jobs/` (guard status files, PID files) does not exist yet and must be created in
Phase 0 step 4 (allowed: directories under `/ceph/dwong/jepa/`). No guard file, no `.pid` file, no
`data/` exists. The clone present is the **old skeleton SHA `72add60e…`**, not the v4 SHA.

**Quota mechanism (this was the object of item 3):** a **CephFS directory quota (inode/entry quota)
on the `/ceph/dwong` subtree** — not a filesystem inode limit, not a POSIX user/project quota. The
working command is the Ceph *virtual* xattr, which `listxattr` does **not** enumerate; it must be
queried by name with `getfattr`. `getfattr -d /ceph/dwong` returns nothing (verified); per-name
queries work, exactly as the 16 Oct evidence file did:

```
ceph.quota.max_files     500000
ceph.quota.max_bytes     0
ceph.dir.rfiles          416387
ceph.dir.rsubdirs        51669
ceph.dir.rbytes          1391743462157     (1.27 TiB)
```

(= `getfattr --absolute-names --only-values -n ceph.dir.rfiles /ceph/dwong` etc.)

- `max_bytes = 0` → **no byte quota**; `/ceph` has 307 TiB free. Bytes are not the constraint.
- `max_files = 500000` counts **inodes, i.e. files + directories** (this is Ceph's documented
  meaning of `max_files`, and it is confirmed empirically by the 16 Oct quota STOP: 444,387 files +
  55,692 dirs = 500,079 entries tripped a 500,000 limit while files alone did not reach it).
- Therefore used = **468,056**, free = **31,944 entries** (11:57:10 UTC; 31,953 in the 16 Oct doc).
- Files-only headroom would be 83,613 — do **not** use that figure; the 16 Oct stop proves the
  entry (files+dirs) reading is the binding one, and it is the project's working figure.

---

## 4. GitHub reachability and the pushed SHA

Measured 11:42:55 UTC and again **11:43:25 UTC** from the host.

```
$ timeout 60 git ls-remote https://github.com/seanxia8/VERITAS.git refs/heads/JEPA
6f41b524044a580410e68a0acb398cf14e062edc	refs/heads/JEPA
EXIT=0
$ timeout 60 git ls-remote https://github.com/seanxia8/VERITAS.git refs/heads/main
f071ee7321ae47fa47be3ab86dac13780080c30c	refs/heads/main
```

**GitHub is reachable over HTTPS; branch `JEPA` is fetchable; pushed SHA =
`6f41b524044a580410e68a0acb398cf14e062edc` at 11:43:25 UTC.** This matches the Lead's target
exactly. (`f021d3c` was still the tip at 11:42:55 UTC; the registration commit `6f41b52` appeared
between the two calls.) Registry state is therefore one clone per SHA, 658 entries per clone —
see §8.

---

## 5. Existing GPU torch environments (read-only candidates, §6.1 order step 1)

Measured 11:43:32 UTC (static scan), **11:56:17–11:56:33 UTC** (versions, arch lists, CUDA
availability). Probes ran with `PYTHONDONTWRITEBYTECODE=1` (no `__pycache__` written into any
environment) and `CUDA_VISIBLE_DEVICES=3`; only driver-level queries were made, so **no CUDA context
and no GPU memory were allocated** (per-GPU `memory.used` identical before and after: 1011/4183/1007/1007 MiB).

| Env path | python | torch | torch.version.cuda | `get_arch_list()` | `cuda.is_available()` | Verdict |
|---|---|---|---|---|---|---|
| **`/home/dwong/anaconda3`** | 3.12.2 | **2.5.1+cu121** | 12.1 | `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` | **True**, `device_count` 1 (GPU 3 visible), `capability(0) = (5, 2)` | **usable read-only — the only GPU-capable env found** |
| `/ceph/dwong/x3_env` | 3.12.2 | 2.9.1**+cpu** | None | `[]` | False (CPU build: "Torch not compiled with CUDA enabled") | CPU only |
| `/ceph/dwong/venvs/dpo_batch3` | 3.12.15 | 2.13.0**+cu130** | 13.0 | `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` | False — *"The NVIDIA driver on your system is too old (found version 12000)"* | **unusable**: needs driver ≥ 13.0 and has no sm_50/52 |
| `/ceph/dwong/jepa/venv/cpu_py311_torch251{,_retry1}` | 3.11 (pyvenv.cfg) | incomplete | — | — | — | ours, failed; not an environment |
| `/ceph/dwong/{paper1_env,meg_env,ots_env,zc_venv,y19_venv}`, `/.conda/envs/remote-phase1-tools` | 3.12 | **torch ABSENT** | — | — | — | not torch envs |

**The critical architecture result:** the anaconda3 wheel's arch list contains **`sm_50`**, and a
cubin built for sm_50 is binary-compatible with sm_52 devices (same major revision, higher minor),
so the 5.2 Titans are runnable with this build. `is_available()` is True and the driver reports
capability `(5, 2)` through torch, so CUDA runtime init works on driver 525 even though the wheel is
cu121 (CUDA minor-version compatibility). This is a *strong indication*, not a kernel-launch proof:
the §6.1 10-batch equivalence test and the ≤10-minute smoke probe must still confirm real kernels.

Package inventory of the candidate (`anaconda3`, read via `importlib.metadata`):

```
python 3.12.2   torch 2.5.1+cu121   numpy 1.26.4   scipy 1.13.1   h5py 3.11.0
scikit-learn 1.5.1   jaxtyping 0.3.11   jax 0.4.30   jaxlib 0.4.30
cudnn 92000 (via torch.backends.cudnn)
```

That covers every "packages relevant to us" item (numpy, scipy, h5py, sklearn, jaxtyping, jax) in
one environment. The `x3_env`/`dpo_batch3` envs have no h5py/sklearn/jaxtyping/jax at all
(`x3_env`: numpy 2.5.3, scipy 1.18.1, the rest absent; `dpo_batch3`: numpy 2.5.1, the rest absent).

Writability (item 5 asks for it explicitly, because a writable env must **not** be modified):

```
/ceph/dwong/x3_env            owner=dwong dir_w=YES sp_w=YES
/ceph/dwong/venvs/dpo_batch3  owner=dwong dir_w=YES sp_w=YES
/home/dwong/anaconda3         owner=dwong dir_w=YES sp_w=YES
```

All three are writable by us, so all three are **write-forbidden by §6.1 step 1**: they are used
read-only, never modified, never `pip install`ed into. `anaconda3` lives in `/home` on the local
filesystem, so it consumes **zero** Ceph quota entries — a second advantage.

Two decision points for `DECISIONS.md` (not gates):
- the existing env is **cu121**, while §6.1 names "cu124, or cu118 if the driver is older than 550".
  A cu121 wheel is a third case that the driver accepts; using it read-only is the §6.1 step-1
  preference over installing anything. Record the build actually used.
- torch 2.5.1 matches the repository pin (`torch==2.5.1`) — only the CUDA build differs.

---

## 6. JAX

Measured 11:56:54–11:57:07 UTC.

- Present **only** in `/home/dwong/anaconda3`: `jax 0.4.30`, `jaxlib 0.4.30`. Nothing under `/ceph`
  has jax (`x3_env`, `dpo_batch3`, the other venvs, the JEPA venvs: absent).
- **No CUDA-enabled jaxlib is installed.** `site-packages/jax_plugins` is **absent**, `jaxlib/cuda/`
  is empty, and a bounded GPU init probe (GPU 3, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.01`, `timeout 60`) returned:

```
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
default_backend: cpu
devices: [CpuDevice(id=0)]
```

(NVIDIA runtime packages are present in that env — `nvidia/{cublas,cuda_runtime,cudnn,cufft,...,nccl}`
— but they are consumed by torch, not by jaxlib.)
- Even if a CUDA jaxlib were installed, the Titans are **sm_52**, far below JAX's minimum (current
  jaxlib CUDA builds carry sm_70+ cubins; a 0.4.30-era CUDA build would not help either).

**Finding: LUCiD generation must run on Deepthought's CPUs** (§6.1: "If not, LUCiD generation runs
on Deepthought's CPUs"). This is the expected outcome and matches §3.5.3 ("Generation runs on the
CPU host"). The §6.1 CPU cap applies: 48 wall-hours, ≤16 cores.

---

## 7. Our processes / existing JEPA state on the host

Measured 11:43:07 UTC.

- `tmux ls` → `no server running on /tmp/tmux-12100/default`. **No tmux sessions at all**, hence no
  `jepa_*` session, no `jepa_guard` session.
- `ps -u dwong` filtered for `jepa|python|git` → **empty**. No JEPA job, no guard, no python of ours
  running on this host.
- `/ceph/dwong/jepa/jobs` → `No such file or directory`; therefore no `guard_<host>.json`,
  no `guard_<host>.log`, no PID files. The only `*guard*`/`*.lock` matches found were the failed
  venvs' `.lock` files and `code/…/uv.lock`.
- No other user's process was touched, signalled, or reniced (item 7's constraint).

Conclusion: a clean slate — nothing of ours to reconcile, no stale job to reap.

---

## 8. Entry headroom arithmetic

All counts 11:57:10 UTC, from `getfattr … -n ceph.dir.rfiles|rsubdirs` on `/ceph/dwong`.

| Quantity | Entries |
|---|---|
| Quota `ceph.quota.max_files` | 500,000 |
| Used: files 416,387 + subdirs 51,669 | **468,056** |
| **Free now** | **31,944** |
| §6.1 step 2 requirement for a **new environment** ("≥100 000 free entries after the planned data files") | 100,000 → **shortfall ≈ 68,056 — NOT met** |
| §6.2 hard floor: "write no new data if free entries < 30 000" | 30,000 → only **1,944 entries of writing headroom** |
| Recoverable inside **our own** area: the two failed JEPA venvs | **9,343** (9,318 + 25) |
| Cost of one clone at this SHA (516 tracked files → 549 files + 109 dirs) | **658** |
| Current `/ceph/dwong/jepa` total | 10,018 |

Planned data, estimated (not measured — no generator exists yet; stated as an estimate with its
basis): sparse hits of ~300–800 hit PMTs/event at ~6 B/hit is ~5 kB/event, so 100k events ≈ 0.5 GB
≈ 5 shards at the §6.1 100 MB shard cap; ~10–15 dataset IDs (six main cells + development + probe
sets + robustness) at ~8 entries each (shards + manifest + `SHA256SUMS`) ⇒ **~10² entries for data**.
Returns are the larger term: ~210 runs (7 arms × 6 cells × 5 seeds) × (metrics JSON + ~5 checkpoint
files + ~3 float16 embedding files) ⇒ **~2,000 entries**, unless checkpoints are pruned by policy.
Total campaign ≈ **2,000–3,500 entries**.

**Verdicts:**
- §6.1 step 2 (new environment): **fails outright** — 31,944 free < 100,000, before any data. So the
  environment must be §6.1 step 1 (existing read-only env) — which §5 shows is viable — or G-ENV.
- §6.2 30,000 floor: met **today** (31,944) with **1,944 entries to spare**, i.e. the campaign is
  ~1–2k entries short of the estimated returns footprint. This is the one real capacity constraint
  on the host and needs one recorded decision, not a gate: either (a) checkpoint-retention policy +
  packing data into few large files (each ≤100 MB shard; a handful per dataset), or (b) delete the
  two failed JEPA venvs — **inside our own area**, so permitted by §6's hard rule, but they were
  deliberately preserved on 16 Oct, so it must be recorded in `DECISIONS.md` first (+9,343 entries,
  +54.9 MB). Deleting them would still not reach 100,000.

---

## Facts for the orchestrator

1. `deepthought` = 40 cores, load 2.84/2.89/2.75 (OK vs §6.2's 28), RAM 503 GiB total, 111.8 GiB
   free / 377.2 GiB available, up 59 d. 15-min-load start condition passes comfortably.
2. Driver **525.147.05 / CUDA 12.0** → §6.1 forces **cu118 (cu124 excluded)**; the one usable env is
   cu121 — record the build actually used.
3. Four **GTX TITAN X, 12288 MiB, compute capability 5.2 (Maxwell), not sm_61**; no TF32/tensor
   cores/bf16. This corrects the prompt's sm_61 line; the runbook was not edited.
4. All four GPUs carry a foreign, idle 1-day-old Jupyter kernel (`amonsch`, PID 3103118) holding
   1011/4183/1007/1007 MiB at 0% util: expect **HOLD on GPU 1**, OK (barely) on 0/2/3; the "1 GB"
   in §6.2 needs a 1000 MB vs 1024 MiB ruling.
5. GPU torch work **is possible read-only**: `/home/dwong/anaconda3` = python 3.12.2, torch
   **2.5.1+cu121**, arch list includes **sm_50** (binary-compatible with sm_52), `cuda.is_available()
   = True`, capability `(5, 2)`; numpy 1.26.4, scipy 1.13.1, h5py 3.11.0, sklearn 1.5.1, jaxtyping
   0.3.11. Writable → must stay read-only; lives in `/home` so costs 0 quota entries. The other two
   candidate envs are dead ends (x3_env is CPU-only 2.9.1+cpu; dpo_batch3 is 2.13.0+cu130, needs
   driver ≥13.0, no sm_50).
6. **JAX cannot use the GPUs**: jax/jaxlib 0.4.30 have no CUDA plugin (probe: "Falling back to
   cpu"), and sm_52 is below JAX's sm_70 minimum. ⇒ **LUCiD generation on Deepthought CPUs** (§6.1).
7. `/ceph/dwong/jepa/` mounted and writable; holds `code/<old sha 72add60e>`, `logs/`, `raw/`,
   `returns/`, `venv/` (two failed envs). **No `jobs/`, `data/`, `env/`** — create `jobs/` in
   Phase 0 step 4. No guard file, no PID file, no tmux session, no process of ours: clean slate.
8. Quota is a **CephFS directory (inode) quota** via virtual xattrs, `max_files=500000`,
   `max_bytes=0` (unlimited bytes, 307 TiB free on `/ceph`): used 468,056, **free 31,944 entries**.
9. §6.1 step 2 new env = **impossible** (needs 100k, has 31.9k, shortfall 68k). §6.2 30k floor: only
   **1,944 entries** of writing headroom; estimated campaign need 2,000–3,500 → one recorded
   decision (checkpoint-retention / few-large-files, and/or delete our two failed venvs for +9,343).
10. **GitHub reachable**, `refs/heads/JEPA` = **`6f41b524044a580410e68a0acb398cf14e062edc`** at
    11:43:25 UTC (matches the Lead's target; `main` = `f071ee7`). Per-SHA clone cost = **658 entries /
    308 MB**; the old clone present is `72add60e`, not the v4 SHA.

### Blockers / limits of this preflight

- **No hard blocker** for read-only GPU work on Deepthought; the existing `anaconda3` env plus the
  sm_50-in-arch-list finding means §6.1 step 1 is viable, subject to the mandatory 10-batch
  equivalence test and the ≤10-minute smoke probe (not run here: that is a job, out of scope).
- **Capacity constraint, not a blocker:** 31,944 free entries vs the 30,000 floor and the excluded
  100,000 new-env path (§8) — needs the Lead's `DECISIONS.md` line before data generation starts.
- Not measured, by design: 5-minute *mean* GPU utilisation (only three instantaneous samples — the
  guard computes the real mean), GPU throughput, and anything requiring a job.
- One probe hung (an env-version script over the `bash -s` stdin heredoc, killed after ~10 min); it
  was re-run as three bounded calls with `timeout`, which produced all the values above. Charge the
  whole preflight as ~15 minutes of one-worker read-only preparation, **0 GPU-hours**.

---

## 9. Guard-deployment preparation (measured 12:09:23–12:09:53 UTC)

Authorised write: **`mkdir -p /ceph/dwong/jepa/jobs`** (that path only, nothing else). Entry counts
before/after: `rfiles` 416,387 → **416,387** (unchanged), `rsubdirs` 51,669 → **51,670** (+1, the new
directory); `max_files` 500,000; `rbytes` unchanged. Used 468,057 → **free 31,943 entries**.
`jobs/` is currently empty; it is where `guard_<host>.json`, `guard_<host>.log` and the PID files go.

| Check | Result |
|---|---|
| Interpreter | `/home/dwong/anaconda3/bin/python3` = **Python 3.12.2**; `import torch, numpy, jaxtyping` → `2.5.1+cu121 12.1 True 1.26.4 0.3.11` (exit 0, no CUDA context: GPU memory unchanged before/after) |
| `nvidia-smi` | `/usr/bin/nvidia-smi`; **`pmon -c 1` works (exit 0)** but per-process **SM% is unusable on this Maxwell card**: `sm` column `-` on GPUs 0 and 3, `0` on 1 and 2; `mem`/`enc`/`dec` all `-`. Only 6 lines (2 header + 4 rows) |
| Per-process memory attribution | `nvidia-smi --query-compute-apps=pid,process_name,used_memory` works (exit 0) and returns all four rows (1004/4176/1004/1004 MiB for PID 3103118) — **but the CSV has no GPU index column**, so the guard must take the index from the full `nvidia-smi` process table (which lists GPU idx + PID + memory correctly) rather than from this query alone |
| `getfattr` | `/usr/bin/getfattr`; `max_files` **500000**, `dir.rfiles` **416387**, `dir.rsubdirs` **51670** |
| `tmux` | `/usr/bin/tmux`, **tmux 2.7**; `tmux ls` → `no server running on /tmp/tmux-12100/default`, **exit 1** (the guard must treat exit 1 + "no server running" as *empty*, not as an error) |
| `git` | `/usr/bin/git`, **2.43.7** → supports `--filter=blob:none` and `sparse-checkout` |
| Driver id | 525.147.05, VBIOS 84.00.1F.00.01 (4 GPUs) |

**Sparse-clone arithmetic** (basis measured on the existing clone; nothing cloned):
tracked files 516 total, of which `scripts/` 28, `src/` 274 (`src/jepa_noise` 15), `docs/jepa` 16;
full checkout = 549 files + 109 dirs = **658 entries / 308 MB**; `.git` alone = 33 files + 17 dirs =
50 entries / 138 MB. A `git clone --filter=blob:none --no-checkout` costs ≈ **50 entries, single-digit
MB** (no blobs fetched), then `git sparse-checkout set scripts` adds 28 files + 5 dirs = **33 entries**
→ **≈ 83 entries total**; `set scripts src` adds ≈ 274 files + dirs → **≈ 400 entries**. So the
sparse route saves ~58% (§6.1's "one clone per SHA" rule is satisfied without paying 658 each time).

---

## 10. Phase 0 step 4 execution on `deepthought` (17 Oct 2026, 12:18–12:38 UTC)

Sequence authorised by the Lead for this host only; every start went through the guard. Timeline is
UTC; host local time is UTC+2.

**Sparse clone (12:18:03–12:18:09).** `git clone --filter=blob:none --sparse`, then
`git sparse-checkout set src scripts`, then `git checkout 6468582` in
`/ceph/dwong/jepa/code/6468582`. `git rev-parse HEAD` = **`64685829ec7e4230b843a914785038cc7914f7d4`**
(exact match); `scripts/jepa_guard.py` (97,546 B), `scripts/jepa_launch.py`, `scripts/jepa_run.sh` all
present; `src/jepa_noise/` present (incl. `guard_config.json`, `probe_env.py`). Entry cost measured:
`rfiles` 416,387 → 416,668 (+281), `rsubdirs` 51,670 → 51,723 (+53) → **+334 entries, +1,805,616 B
(1.7 MB)**, better than the 658 of a full checkout and inside the 370–400 estimate.

**Guard dry run (12:18:28–12:29:25, tmux `jepa_guard_dryrun`, `--dry-run --interval 120`).** No
`guard_deepthought.json` was written (verified absent). Per-cycle JSON lines went to
`jobs/guard_deepthought.log` (the stdout log holds only the banner). Cycle at 12:22:29: `level OK`,
`start.gpu 0`, `dry_run true`, `entry_source.source "cephfs_xattr"` with `quota_max_files 500000`,
`rfiles 416769`, `rsubdirs 51732` → **free 31,499**, `unattributed_memory_mib` 3–7 MiB per card,
`utilization_window` warm (n=3, span 240.8 s), amonsch's kernel attributed as foreign at 0.0 %,
GPU 1 blocked ("other user's process PID 3103118 holds 4176 MiB > 1024 MiB limit"), **no SHED lines**.
Session killed by exact name; `tmux ls` then empty. Note: the event log grows ~29 KB/6 min because
each cycle appends a full status line — irrelevant for bytes, but each *run* also adds entries (§8).

**Live guard (12:29:26 →, tmux `jepa_guard`, stdout `jobs/guard_deepthought_stdout.log`).** First
status written within one 120 s cycle; `utilisation_window` warm after ~4 min, as intended (the guard
HOLDs until then). State at 12:37:29 (age 69.1 s): `level OK`, `dry_run false`, `start.gpu 0`,
`our_jobs []`, no PID-file errors, `free_entries 31,480`, `source cephfs_xattr`, util window warm
(n=3, span 241.6 s), per-GPU `start_ok` = `[0:True, 1:False, 2:True, 3:True]`, unattributed 3–7 MiB,
0 SHED events.

**Probe (12:35:37–12:35:48, the only job run).** Launched through `scripts/jepa_run.sh` after a
`--dry-run` inspection (which printed GPU 0, memory fraction 0.8, timeout 900 s, and started nothing).
Launcher: `started pid=3437908 gpu=0 timeout=900s`; `finished outcome=exited exit=0`.
`jobs/probe_deepthought_6468582/run_status.json`: `exit_status 0`, `timed_out false`,
`started_at 12:35:37Z`, `finished_at 12:35:48Z`, **`wall_seconds 10.55`**.
`returns/probe_deepthought_6468582/probe.json` acceptance facts:

| Criterion | Value |
|---|---|
| `status` | **`ok`** |
| `torch.cuda_available` | **true**; `device_count` 1; `device_used cuda:0` |
| device name / capability | **`NVIDIA GeForce GTX TITAN X` / `[5, 2]`** |
| `tf32_matmul_after` / `tf32_cudnn_after` | **false / false** (cudnn was true before the probe set it) |
| fixture batches | **10** (batch 4, 8 ch, 256 samples, seed 0) |
| `timing.mean_seconds_per_step` | **0.00884 s** (5 timed steps; 452 events/s; peak 31.7 MiB) |
| `freeze.txt` | **written**, 9,217 B (torch 2.5.1+cu121, numpy 1.26.4, jax 0.4.30, …) |
| `repo_head` | `64685829ec7e4230b843a914785038cc7914f7d4` |
| determinism | `determinism_same_device_max_abs_diff = 0.0` |

GPU state returned to baseline afterwards (1011/4183/1007/1007 MiB, 0 %). **Ledger: run_id
`probe_deepthought_6468582`, 1 GPU (Titan), wall 10.55 s → 0.0029 GPU-h, rounds to 0.0 GPU-h.**
`tmux ls` afterwards shows only `jepa_guard`; `our_jobs` is empty.

Notes for the orchestrator:
- **Freshness race:** the guard writes every 120 s and `jepa_run.sh` refuses a status not strictly
  younger than 120 s. A launch must follow a cycle closely; a refusal (exit 3) is to be retried after
  the next cycle, never bypassed. The probe launch happened 9 s after a cycle, hence accepted.
- **Shared `jobs/`:** `/ceph/dwong/jepa/jobs/` is shared through `/ceph`; the deepthought2 operator's
  `guard_deepthought2*` and `probe_deepthought2_6468582*` artifacts are visible here. They are not
  ours to touch. Quota deltas measured between two points therefore include that agent's writes.
- Cosmetic: `probe.json` records `cudnn 90100` (matching `nvidia-cudnn-cu12==9.1.0.70` in
  `freeze.txt`) while a driver-level query of the same interpreter earlier printed 92000 — lazy
  library resolution; `freeze.txt` is the authoritative record.

---

## 11. LUCiD pin provenance and guard housekeeping (read-only, 12:43:59–12:44:17 UTC)

### LUCiD pinned revision `eaf2e89f579d6679e90020c732ac2c846895d691` (runbook v4 §3.5.2)

Runbook text, quoted: "**LUCiD.** Pinned revision `eaf2e89f579d6679e90020c732ac2c846895d691`."

The pin is **not missing from the network** — it is the current tip of both candidate remotes:

```
$ git ls-remote https://github.com/dowlingwong/LUCiD | grep -i eaf2e89
eaf2e89f579d6679e90020c732ac2c846895d691	HEAD
eaf2e89f579d6679e90020c732ac2c846895d691	refs/heads/main
$ git ls-remote https://github.com/CIDeR-ML/LUCiD | grep -i eaf2e89
eaf2e89f579d6679e90020c732ac2c846895d691	HEAD
eaf2e89f579d6679e90020c732ac2c846895d691	refs/heads/main
eaf2e89f579d6679e90020c732ac2c846895d691	refs/tags/v0.4.1^{}
```

`ls-remote` alone only proves refs, so the existence was confirmed network-only via the commits API
(HTTP GET, no clone, no writes): `GET /repos/{dowlingwong,CIDeR-ML}/LUCiD/commits/eaf2e89…` → **HTTP
200 on both**. **Answer: both remotes contain the pinned commit; `dowlingwong/LUCiD` is the local
ORACLE clone's own `origin` and its `main` is exactly the pin**, and CIDeR-ML tags it `v0.4.1`.

Pinned commit metadata (API): date **2026-10-06T08:33:15Z**, author *Cesar Jesus Valls*, subject
"container: build PhotonSim v1.0.3".

Local installed LUCiD is simply **older**, not divergent: `/ceph/dwong/ORACLE/external/LUCiD` HEAD =
`86a1871ab4a090f9e6970935af86ca98f1d114b1` (2026-09-16T13:56:48+02:00), origin
`https://github.com/dowlingwong/LUCiD`; `git cat-file -e eaf2e89…` fails **in that clone** only
because it has not fetched since. API compare `86a1871…eaf2e89` → status **`ahead`, ahead_by 95,
behind_by 0**: the installed tree is an ancestor and the pin is **95 commits ahead** of it. Hence the
installed `lucid-sim 0.1.dev1+g86a1871ab` is 95 commits behind the runbook pin. Using the pin means
cloning/fetching it under `/ceph/dwong/jepa/` (never fetching into `/ceph/dwong/ORACLE`, which is
outside our write area) — a provenance decision for the orchestrator, not a blocker.

### Guard housekeeping (live guard, no restart)

`tmux` still has only `jepa_guard`. Status at 12:43:31Z (age 46.5 s): **level OK**, `dry_run false`,
`start.gpu 0`, `our_jobs []`, `free_entries 31,479` (`cephfs_xattr`), window warm (n=3, 241.0 s),
per-GPU `start_ok [0:T, 1:F, 2:T, 3:T]`. All five cycles since the probe
(12:35:28, 12:37:29, 12:39:30, 12:41:30, 12:43:31) were `OK` with `our_jobs` empty; **0 SHED lines**
in the live log. GPU memory back at baseline 1011/4183/1007/1007 MiB, 0 % each. Free entries:
500,000 − (416,784 + 51,737) = **31,479**.

---

## 12. Deployment at SHA `89af5af`: one clone, corrected probe (pass), bridge smoke (FAIL)

Measured 12:55:59–12:59:07 UTC on `deepthought`. One clone plus exactly two guard-gated jobs, run
one at a time; nothing else started.

### Clone

Destination `code/89af5af` was free (nothing forced). `git clone --filter=blob:none --sparse` +
`git sparse-checkout set src scripts` + `git checkout 89af5af`. `git rev-parse HEAD` =
**`89af5af6575522c68c8f58d45039b5fd8df12635`**, subject "jepa: WP4b generator scaffold (laws, twin,
runs/storage, Fisher ceilings)". 340 files checked out, `scripts/` 23 files, `src/jepa_noise/` 21
files. **Entry delta: `rfiles` 416,784 → 417,071 (+287), `rsubdirs` 51,737 → 51,790 (+53) → +340
entries, +7,402,522 B (7.1 MB)** — consistent with the previous sparse clone's +334.

### Job 1 — `probe_deepthought_89af5af_v2` (PASS)

Guard before launch: OK, age 38.5 s, `start.gpu 0`. Launch 12:56:19Z; launcher started 12:56:20Z,
finished 12:56:31Z, **`exit_status 0`**, `timed_out false`, `wall_seconds 10.52` (shell wall 12 s).
Guard after: OK, `our_jobs []`, GPU back at baseline.

| Acceptance item | Value |
|---|---|
| `status` | **`ok`** |
| `fixture.readout` | **`pre_final_proj_norm_token_grid`** |
| batches | **10** (batch 4, 8 ch, 256 samples, seed 0), 10 `per_batch` rows |
| `grid_shape` | **`[32, 8, 128]`** on every batch |
| `grid_sum` / `grid_abs_sum` | non-zero: sums 39.60, −341.75, −14.28, 31.97, −299.25, 30.77, −139.47, −173.98, −199.33, −121.15; abs sums 25,923–26,257 |
| timing | `mean_seconds_per_step` **0.00689 s**, peak 31.9 MiB |
| `freeze.txt` | written (9,217 B) |
| torch | 2.5.1+cu121, `cuda_available true`, GTX TITAN X `[5,2]`, tf32 matmul/cudnn after = false/false |
| `repo_head` | `89af5af6575522c68c8f58d45039b5fd8df12635` |

The degenerate fixture of the first probe is closed: the grid is non-trivial and the read-out label is
the reviewed one.

### Job 2 — `smoke_bridge_deepthought_89af5af` (FAIL, deterministic hash mismatch)

Guard before launch: OK, age 3 s (deliberately waited for the 12:57:35Z cycle; nothing forced).
Launch 12:57:38Z; launcher started 12:57:39Z, **finished 12:57:49Z, `outcome exited`, `exit_status 1`**,
`timed_out false`, `wall_seconds 10.68` (shell wall 12 s). Guard after: OK, `our_jobs []`, 0 SHED,
GPU back at baseline 1011/4183/1007/1007 MiB.

Traceback (`jobs/smoke_bridge_deepthought_89af5af/job.log`), after `calibration noise 4096/4096`:

```
File ".../src/jepa_noise/train.py", line 609, in <module>  raise SystemExit(main())
File ".../src/jepa_noise/train.py", line 590, in main  a4_bridge_products(progress=True, artifact_root=args.data_root)
File ".../src/jepa_noise/train.py", line 311, in a4_bridge_products  verify_bridge_products(frozen, products)
File ".../src/jepa_noise/train.py", line 179, in verify_bridge_products
RuntimeError: computed N_actual target marginal sha256 16955029c6a13117d3a19f21e1944f935bacde1e8b24e1c11103d9dea0f0fb6e
does not match the frozen config sha256 d31351ad8771a26a8f1fbbefee6efcad90978175f9bd346775fbdc788683fb28
```

**This is not the CUDA/device path and not the `index_select` error.** The failure is upstream of all
device code: it is the pre-outcome config verification inside `a4_bridge_products`, which runs on
numpy. Nothing was moved to CUDA, no arm ran, so the WP4a overlay device fix remains **unproven**.

Evidence gathered read-only:
- The frozen value is `src/jepa_noise/configs/bridge_v1.json` → `data.target_noise_covariance.sha256`
  = `d31351ad…`; the same file also freezes `generator_manifest_sha256 1cd3195b…`,
  `generator_noise_covariance_sha256 970b7818…`, `n_target 1024`.
- Config **and** `train.py` were last modified in the *same* commit `7f98b86` ("WP4a review fixes
  (pre-outcome config freeze)", 2026-10-17 14:35), so the constant is not stale relative to the code
  in this clone.
- The checked quantity is exact integer-free indexing: `marginal = noise_covariance[np.ix_(target,
  target)]` then `_array_sha256(marginal, "<f8")`. `np.ix_` is a pure copy, so a mismatch means the
  **covariance matrix itself differs in its last bits**.
- That matrix is built from `np.kron(channel_cov, temporal_covariance)` plus line/common terms, where
  `temporal_covariance` comes from the PSD through FFT and `leading_temporal_basis` (LAPACK
  `eigh` in `linear.py`), plus BLAS matrix products. It is a ULP-sensitive construction, and the hash
  is therefore a **platform fingerprint, not a portable invariant**: the frozen value was produced on
  the author's Mac; deepthought's numpy 1.26.4 is linked to **openblas64 0.3.23.dev**.
- Consequence: the failure is reproducible-by-construction on this host and cannot be fixed by device
  selection. There is also **no CLI switch to skip verification**: `--smoke` calls
  `a4_bridge_products(progress=True, artifact_root=...)` with the default `verify=True`.

Options for the orchestrator (decision, not mine): (A) re-freeze the two data-derived hashes on
deepthought with `python -m jepa_noise.train --freeze-config` and commit the updated
`bridge_v1.json`, recording in `DECISIONS.md` that this is a data-derived, pre-outcome re-freeze;
(B) replace exact-hash verification with a tolerance/platform-aware check (code change, needs review);
(C) write the marginal once under `/ceph/dwong/jepa/data/bridge/<sha256>/` and verify the stored file
instead of recomputing bit-for-bit. Until one is chosen, the 1-step CUDA bridge smoke cannot pass.

Quota after both jobs: `rfiles` 417,189 + `rsubdirs` 51,804 = 468,993 used → **free 31,007** (the
guard's own 12:57:35Z reading was 31,019; the small drift is the shared `jobs/` directory being
written concurrently by the deepthought2 operator).

---

## 13. Fix verification at `bbcded2e`: mutable `work` clone + bridge smoke PASS

Measured 13:04:34–13:06:09 UTC on `deepthought`. One clone (the last full clone; `code/work` is the
mutable clone from now on) plus exactly one guard-gated job.

### Clone `code/work`

Destination free (nothing forced). Same recipe, then `git checkout bbcded2e…`. `git rev-parse HEAD`
= **`bbcded2e0e823ed6e12b5f14837df8c4a3805df8`** ("jepa: fix WP4a platform-brittle frozen-hash
verification (found by first Titan smoke)"), detached HEAD, **`git status --porcelain` empty** (and
empty with `--ignored`). **Entry delta: `rfiles` 417,189 → 417,470 (+281), `rsubdirs` 51,804 →
51,857 (+53) → +334 entries.** `code/89af5af` untouched as the probe record; `code/6468582` still the
reviewed guard source that the live guard runs.

The fix is present in this checkout: `relative_frobenius_error()` with
`MATRIX_RELATIVE_TOLERANCE = 1e-8`, a tolerance-based reference object, and the exact sha retained as
an informational manifest record (`train.py` lines 154–288).

### Job — `smoke_bridge_deepthought_bbcded2` (PASS)

Guard before: OK, age 2 s (waited for the cycle; nothing forced). Launched 13:05:39Z; launcher
started 13:05:40Z, finished 13:05:54Z, **`exit_status 0`**, `outcome exited`, `timed_out false`,
**`wall_seconds 14.36`** (shell wall 16 s), `gpu_index 0`, `planned_minutes 10.0`, timeout 900 s.
Guard after: OK, age 31.5 s, `our_jobs []`, 0 SHED, GPU back to baseline 1011/4183/1007/1007 MiB.

Four arm summaries, all **`"device": "cuda:0"`**, `n_batches 1`, no exception anywhere in the log:

| Arm | seconds | notes |
|---|---|---|
| `mae_chi2` | 0.213 | `chi2` 799.64, loss 821.84 |
| `sigreg_jepa` | 2.830 | first arm, includes CUDA warm-up; loss in log |
| `sigreg_jepa_anchored` | 0.111 | anchor 0.5939, anchor_trace 2.8598, β placeholder 1.0, 942,528 params |
| `sigreg_jepa_parallel` | 0.079 | 975,424 params |

Error scan of `job.log`: `index_select`, `Traceback`, `RuntimeError`, `CUDA error`, device-side
asserts → **none**. The `index_select` device concern is therefore cleared on the Titan.

**Peak memory: not emitted by this path** — `train.py --smoke` records no peak-memory field (only
`probe_env` does). Bounded implicitly: the largest single allocation is the 1024×1024 float64 marginal
(8,388,736 B file, loaded not recomputed), models are 0.94–0.98 M parameters at batch 2×8×256, the
process ran under the launcher's 0.8 memory fraction (9.6 GiB Titan cap), and the GPU returned to
baseline the second the job exited.

### Cache manifest (materialised once on this host)

`/ceph/dwong/jepa/data/bridge/30c63a6e35c5c575416246440c360bab5d8cedd35a17846f82d79d8c80d11db7/`
containing `N_actual_target_marginal.npy` (8,388,736 B) and `manifest.json`:

```
code_sha                 : bbcded2e0e823ed6e12b5f14837df8c4a3805df8
computed_on              : deepthought
shape / dtype            : [1024, 1024] / <f8
file_sha256              : fa5d7d65cf7eb0bff8535c983f5fef625a6803fcfdb14e60fb9deb61ca808516
reference_file_sha256    : 30c63a6e35c5c575416246440c360bab5d8cedd35a17846f82d79d8c80d11db7
relative_frobenius_tolerance : 1e-08
views_constructor        : base_rho 0.0, common_strength 0.02, line_multiplier 0.0,
                           law v4-A4-waveform-v1, split channel, seed 0, cell bridge, signal_mode nonlinear
note                     : materialised once on this host from the pinned generator; later runs load
                           these bytes after verifying file_sha256
```

That is exactly the (C) behaviour asked for: the directory is keyed by the platform-independent
reference sha, the file's own sha is recorded and verified on load, and the exact computed sha is
informational only. Note the file sha (`fa5d7d65…`) differs from the frozen Mac constant
(`d31351ad…`) and from the earlier Titan recomputation (`16955029…`) — as expected for a bit-exact
fingerprint of a float64 LAPACK product; only the tolerance and the file sha are gates now.

**Quota after this job: `rfiles` 417,578 + `rsubdirs` 51,870 = 469,448 used → free 30,552**
(guard's 13:05:37Z reading 30,561). The clone cost +334, my job/cache artifacts ≈ 8 entries; the
remainder of the drift is the deepthought2 operator writing the shared `jobs/` directory.
**Live warning: the margin to the §6.2 "write no new data below 30,000 free entries" floor is now
552 entries** — dataset generation cannot start until G-QUOTA resolves.
