# r2-lucid host inventory (L0, 21 Nov 2026, ~01:40 UTC)

| Host | Role | OS | Cores / RAM | Load at check | GPUs (driver, CUDA) | Python / conda | Mounts |
|---|---|---|---|---|---|---|---|
| portal1.etp.kit.edu | login node: clone, env, wheel download (≤ 4 cores) | AlmaLinux 9.7 | 128 / 1007 GiB | 57 (26 users) | none (`nvidia-smi` absent) | anaconda3 Python 3.12.2 | `/ceph` (1.4 PB, 337 TB free, 75 % used), `/work` (1.6 TB, 1.2 TB free) |
| deepthought (`dt1`) | CPU work, tests, packing, HTCondor submit | RHEL 8.10 | 40 / 503 GiB | 6.6 | 4 × GTX TITAN X 12 GB (525.147.05, CUDA 12.0); GPU 2 held 3.8 GB by another user | anaconda3 Python 3.12 | same `/ceph`, `/work` |
| deepthought2 (`dt2`) | guarded L40S lanes | AlmaLinux 9.6 | 128 / 1006 GiB | 92 (18 users) | 2 × L40S 46 GB (570.211.01, CUDA 12.8); 14.3 GB and 9.6 GB used by other users | anaconda3 Python 3.12 | same `/ceph`, `/work` |
| HTCondor batch (`accounting_group = cms`) | LUCiD generation, training | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (Python 3.11) | per job | – | A100-PCIE-40GB (TOpAS/GridKa), L40S (NEMO2) | image Python 3.11 + staged wheels | no `/ceph`; inputs via xrootd from `/ceph/srv/dwong/r2` |

Ceph quota `/ceph/dwong`: 1,500,000 files, 522,601 used (977,399 free entries); no byte quota.

Findings that shaped the plan:
- portal1 is a shared login node without GPUs; it was used only for cloning, the environment and the wheelhouse download.
- The Titans cannot run the LUCiD JAX stack: JAX 0.6.2 with its CUDA 12.9 runtime on driver 525 (Maxwell, sm_52) fails with `CUDA_ERROR_ILLEGAL_ADDRESS` in `hello_simulate.py`.
- The deepthought2 guard reported `HOLD` (15-minute load > 0.70 × 128 cores) from the start of the phase; no job was started there while it held.
- LUCiD generation therefore runs on HTCondor batch GPUs (runbook §0.4 option 4), after verifying that the LUCiD environment runs in the batch image (smoke cluster 759831, A100, `jax.devices() = [CudaDevice(id=0)]`).

## portal1 CPU policy (author, 21 Nov 2026 02:47 CET; seen 06:35 UTC)
- portal1: 2 × AMD EPYC 7452 (64 cores, 128 threads), 1007 GiB RAM. Allowance: up to half of the threads (64) at `nice -n 10`, ≥ 8 free, halving when other users' 1-minute load exceeds 32.
- At 06:41 UTC other users' load was 11.8, below the backoff threshold. By then all LUCiD generation had finished on GPUs (deepthought2 L40S lanes: WCTE ≈ 12 events/s; TOpAS A100 batch: SK/JUNO/HK ≈ 7–8 events/s per job), far faster than the measured CPU rate (3.5–6 × 10⁴ photons/s on 8 Deepthought cores ≈ 0.3 events/s), so portal1 CPU was used only for evaluation offload. Backoff events are logged in the notes.
