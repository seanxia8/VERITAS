#!/usr/bin/env python3
"""Submit one r2-lucid generation job to HTCondor from Deepthought (see lucid_job.sh).

Usage: lucid_submit.py JOB_ID --gpu a100|l40s --walltime-h H --code code/lucidmc_<sha8>.tar.gz -- <lucid_generate.py run args>
The returns tarball is unpacked by lucid_collect.py into the lucid_v1 shard tree.
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

JOBS = Path("/work/dwong/r2_lucid/batch_jobs")
REPO = Path("/ceph/dwong/jepa/r2_lucid_repo")
IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
GPU_REQ = {"a100": '(TARGET.GPUs_DeviceName =?= "NVIDIA A100-PCIE-40GB")',
           "l40s": '(TARGET.CloudSite =?= "BwForCluster-Nemo2") && (TARGET.GPUs_DeviceName =?= "NVIDIA L40S")',
           "h200": '(TARGET.CloudSite =?= "BwForCluster-Nemo2") && (TARGET.GPUs_DeviceName =?= "NVIDIA H200")',
           "container": '((TARGET.GPUs_DeviceName =?= "NVIDIA L40S") || (TARGET.GPUs_DeviceName =?= "NVIDIA H200"))'}


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("separate the lucid_generate.py arguments with --")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id"); ap.add_argument("--gpu", default="a100", choices=sorted(GPU_REQ))
    ap.add_argument("--walltime-h", type=float, required=True); ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem-mb", type=int, default=32000); ap.add_argument("--disk-kb", type=int, default=20_000_000)
    ap.add_argument("--code", required=True)
    a = ap.parse_args(argv[:cut]); args = argv[cut + 1:]
    sums = Path("/ceph/srv/dwong/r2/SHA256SUMS").read_text()
    for f in (a.code, "lucid/wheels_lucid_cp311.tar", "lucid/LUCiD_e651e836.tar", "lucid/lucid_config.tar"):
        if f" {f}\n" not in sums + "\n":
            raise SystemExit(f"{f} is not staged")
    if a.walltime_h > 24:
        raise SystemExit("walltime above 24 h")
    d = JOBS / a.job_id / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d.mkdir(parents=True)
    shutil.copy(REPO / "scripts/r2_batch/lucid_job.sh", d / "lucid_job.sh")
    wheel = sorted(Path("/work/dwong/r2_batch/xw").glob("xrootd-*-cp311-*.whl"))[-1]
    shutil.copy(wheel, d / wheel.name)
    wall_s = int(a.walltime_h * 3600)
    sha = a.code.split("_")[-1].split(".")[0]
    env = [f"R2_CODE='{a.code}'", f"R2_TIME_LIMIT_S={max(600, wall_s - 900)}", "USE_USER=dwong",
           f"JEPA_CODE_SHA={sha}"]
    universe = ("Universe = docker\ndocker_image = " + IMAGE) if a.gpu == "a100" else \
               ("Universe = container\ncontainer_image = docker://" + IMAGE)
    jdl = f"""Executable = lucid_job.sh
Arguments = {' '.join(args)}
Output = out.txt
Error = err.txt
Log = log.txt
transfer_input_files = {wheel.name}
transfer_output_files = returns_out.tar
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
+RequestWalltime = {wall_s}
RequestCPUs = {a.cpus}
RequestMemory = {a.mem_mb}
request_disk = {a.disk_kb}
request_GPUs = 1
+RemoteJob = True
requirements = {GPU_REQ[a.gpu]}
accounting_group = cms
environment = "{' '.join(env)}"
{universe}
Queue
"""
    (d / "job.jdl").write_text(jdl)
    out = subprocess.run(["condor_submit", "job.jdl"], cwd=d, capture_output=True, text=True)
    cluster = out.stdout.strip().split()[-1].rstrip(".") if out.returncode == 0 else None
    rec = {"job_id": a.job_id, "dir": str(d), "cluster": cluster, "gpu": a.gpu, "walltime_h": a.walltime_h,
           "args": args, "code": a.code, "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stderr": out.stderr[-500:]}
    (d / "submit.json").write_text(json.dumps(rec, indent=1))
    with open(JOBS / "submissions.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return out.returncode


if __name__ == "__main__":
    raise SystemExit(main())
