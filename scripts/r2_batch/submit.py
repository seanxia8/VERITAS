#!/usr/bin/env python3
"""Submit one round-2 HTCondor batch job from Deepthought (see job.sh).

Usage: submit.py RUN_ID --gpu a100 --walltime-h H [--cpus 8] [--mem-mb 64000] [--files a b ...]
                 [--resume-run sk__mae_mse__s9000] [--smokes-for juno] [--extra-env K=V ...] -- ext_r2 args
Job state (batch ledger copy, smoke records, run directory to resume) is shipped as returns_in.tar.
"""
import argparse, json, shutil, subprocess, tarfile, time
from pathlib import Path

R = Path("/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11")
B = R / "batch"
JOBS = Path("/work/dwong/r2_batch/jobs")
REPO = Path("/work/dwong/r2_repo")
IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
GPU_REQ = {"a100": '(TARGET.GPUs_DeviceName =?= "NVIDIA A100-PCIE-40GB")',
           "l40s": '(TARGET.CloudSite =?= "BwForCluster-Nemo2") && (TARGET.GPUs_DeviceName =?= "NVIDIA L40S")',
           "h200": '(TARGET.CloudSite =?= "BwForCluster-Nemo2") && (TARGET.GPUs_DeviceName =?= "NVIDIA H200")'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id"); ap.add_argument("--gpu", default="a100", choices=sorted(GPU_REQ))
    ap.add_argument("--walltime-h", type=float, required=True); ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem-mb", type=int, default=64000); ap.add_argument("--disk-kb", type=int, default=40_000_000)
    ap.add_argument("--files", nargs="*", default=[]); ap.add_argument("--code", required=True)
    ap.add_argument("--resume-run", default=None); ap.add_argument("--smokes-for", default=None)
    ap.add_argument("--extra-env", nargs="*", default=[])
    import sys
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("separate the ext_r2 arguments with --")
    cut = argv.index("--")
    a = ap.parse_args(argv[:cut])
    args = argv[cut + 1:]
    if a.walltime_h > 24:
        raise SystemExit("walltime above 24 h: chain resumed jobs instead (addendum §3)")
    d = JOBS / a.run_id / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d.mkdir(parents=True)
    shutil.copy(REPO / "scripts/r2_batch/job.sh", d / "job.sh")
    with tarfile.open(d / "returns_in.tar", "w") as t:
        led = R / "ledger" / "RESOURCE_LEDGER_R2_BATCH.json"
        if led.exists():
            t.add(led, arcname="returns/ledger/RESOURCE_LEDGER_R2_BATCH.json")
        if a.smokes_for:
            for p in sorted((B / "smokes").glob(f"{a.smokes_for}__*/run_record.json")):
                t.add(p, arcname=f"returns/smokes/{p.parent.name}/run_record.json")
        if a.resume_run:
            src = B / "runs" / a.resume_run
            if not src.exists():
                src = R / "runs" / a.resume_run  # a deepthought2 run moving to batch
            for name in ("checkpoint.pt", "train_log.jsonl", "run_record.json"):
                if (src / name).exists():
                    t.add(src / name, arcname=f"returns/runs/{a.resume_run}/{name}")
    wall_s = int(a.walltime_h * 3600)
    env = [f"R2_FILES={' '.join(a.files)}", f"R2_CODE={a.code}", f"R2_TIME_LIMIT_S={max(600, wall_s - 1200)}",
           f"R2_EXTRA_ENV={' '.join(a.extra_env)}", "USE_USER=dwong"]
    universe = ("Universe = docker\ndocker_image = " + IMAGE) if a.gpu == "a100" else \
               ("Universe = container\ncontainer_image = docker://" + IMAGE)
    jdl = f"""Executable = job.sh
Arguments = {' '.join(args)}
Output = out.txt
Error = err.txt
Log = log.txt
transfer_input_files = returns_in.tar
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
    rec = {"run_id": a.run_id, "dir": str(d), "cluster": cluster, "gpu": a.gpu, "walltime_h": a.walltime_h,
           "args": args, "files": a.files, "code": a.code, "resume_run": a.resume_run,
           "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stderr": out.stderr[-500:]}
    (d / "submit.json").write_text(json.dumps(rec, indent=1))
    with open(JOBS / "submissions.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
