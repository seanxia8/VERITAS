#!/usr/bin/env python3
"""Bring a finished batch job back: unpack returns_out.tar into the returns tree under batch/,
and merge its new ledger entries into the returns ledger (R2_batch_gpu). Usage: sync.py JOB_DIR"""
import fcntl, json, sys, tarfile, tempfile, shutil
from pathlib import Path

R = Path("/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11")
B = R / "batch"
LED = R / "ledger" / "RESOURCE_LEDGER_R2_BATCH.json"


def key(e):
    return json.dumps(e, sort_keys=True)


def main(job_dir):
    d = Path(job_dir)
    if (d / "SYNCED").exists():
        print("already synced"); return
    sub = json.loads((d / "submit.json").read_text())
    out = d / "returns_out.tar"
    if not out.exists():
        raise SystemExit(f"{out} missing")
    tmp = Path(tempfile.mkdtemp(dir="/work/dwong"))
    with tarfile.open(out) as t:
        t.extractall(tmp)
    ret = tmp / "returns"
    shipped = set()
    with tarfile.open(d / "returns_in.tar") as t:
        try:
            shipped = {key(e) for e in json.load(t.extractfile("returns/ledger/RESOURCE_LEDGER_R2_BATCH.json"))["entries"]}
        except KeyError:
            pass
    new = []
    jl = ret / "ledger" / "RESOURCE_LEDGER_R2_BATCH.json"
    if jl.exists():
        new = [e for e in json.loads(jl.read_text())["entries"] if key(e) not in shipped]
    for e in new:
        e.setdefault("batch_cluster", sub.get("cluster")); e.setdefault("batch_job_dir", str(d))
    # the batch line books the job's full GPU wall time (download, setup, run): add the overhead
    import subprocess
    wall = None
    if sub.get("cluster"):
        q = subprocess.run(["condor_history", str(sub["cluster"]), "-limit", "1", "-af", "RemoteWallClockTime"],
                           capture_output=True, text=True).stdout.strip()
        try:
            wall = float(q.split()[0])
        except (ValueError, IndexError):
            wall = None
    booked = sum(float(e.get("gpu_h", 0)) for e in new)
    if wall is not None and wall / 3600.0 > booked:
        new.append({"kind": "batch_overhead", "gpu_h": wall / 3600.0 - booked, "job_wall_s": wall,
                    "batch_cluster": sub.get("cluster"), "batch_job_dir": str(d), "gpu": sub.get("gpu"),
                    "run_id": sub.get("run_id")})
    LED.parent.mkdir(parents=True, exist_ok=True)
    with open(LED.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        led = json.loads(LED.read_text()) if LED.exists() else {"line": "R2_batch_gpu", "cap_h": 200.0, "entries": [], "cpu_entries": []}
        led["entries"].extend(new)
        LED.write_text(json.dumps(led, indent=1, sort_keys=True) + "\n")
    for sub_dir in ("runs", "smokes", "eval", "wcte_regression"):
        src = ret / sub_dir
        if src.exists():
            shutil.copytree(src, B / sub_dir, dirs_exist_ok=True)
    for f in ret.glob("*.json"):
        shutil.copy(f, B / f.name)
    code = (ret / "JOB_EXIT_CODE").read_text().strip() if (ret / "JOB_EXIT_CODE").exists() else None
    (d / "SYNCED").write_text(json.dumps({"new_ledger_entries": len(new), "ext_r2_exit": code}))
    shutil.rmtree(tmp)
    print(json.dumps({"job": str(d), "new_ledger_entries": len(new), "gpu_h": sum(float(e.get("gpu_h", 0)) for e in new), "ext_r2_exit": code}))


if __name__ == "__main__":
    main(sys.argv[1])
