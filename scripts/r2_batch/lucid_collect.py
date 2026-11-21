#!/usr/bin/env python3
"""Collect finished r2-lucid batch jobs into the lucid_v1 data tree (run on Deepthought).

For each job directory with a ``returns_out.tar`` that is not yet collected: unpack it under
``/ceph/dwong/jepa/data/lucid_v1/<det>/.incoming/<job>`` (same filesystem as the target), move
every validated shard directory into ``shards/<cohort>/<run_dir>`` (an existing directory is never
overwritten), cache pickles into ``cache/``, other outputs into ``returns/``, and append a GPU
receipt (HTCondor RemoteWallClockTime of the whole job) to ``receipts.jsonl``.
"""
import json, os, subprocess, sys, tarfile, time
from pathlib import Path

JOBS = Path("/work/dwong/r2_lucid/batch_jobs")
DATA = Path("/ceph/dwong/jepa/data/lucid_v1")
RECEIPTS = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/ledger/receipts.jsonl")


def wallclock(cluster):
    out = subprocess.run(["condor_history", str(cluster), "-af", "RemoteWallClockTime",
                          "GPUs_DeviceName", "LastRemoteHost", "ExitCode"],
                         capture_output=True, text=True).stdout.split()
    return out


def collect(job_dir: Path):
    sub = json.loads((job_dir / "submit.json").read_text())
    tar = job_dir / "returns_out.tar"
    done = job_dir / "COLLECTED.json"
    if done.exists() or not tar.exists():
        return None
    args = sub["args"]
    det = args[args.index("--detector") + 1] if "--detector" in args else "misc"
    inc = DATA / det / ".incoming" / f"{sub['job_id']}_{sub['cluster']}"
    inc.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(inc, filter="data")
    ret = inc / "returns"
    moved, skipped = [], []
    for cohort_dir in sorted((ret / "shards").glob("*")) if (ret / "shards").exists() else []:
        for run_dir in sorted(cohort_dir.iterdir()):
            if run_dir.name.endswith(".tmp") or not (run_dir / "manifest.json").exists():
                skipped.append(str(run_dir)); continue
            dest = DATA / det / "shards" / cohort_dir.name / run_dir.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                skipped.append(str(dest)); continue
            os.rename(run_dir, dest)
            moved.append(str(dest))
    for pkl in sorted((ret / "cache").rglob("*.pkl")) if (ret / "cache").exists() else []:
        dest = DATA / det / "cache" / pkl.relative_to(ret / "cache")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.rename(pkl, dest)
    for extra in sorted(ret.glob("*.json")) + sorted(ret.glob("*.txt")) + [ret / "JOB_EXIT_CODE"]:
        if extra.exists():
            dest = DATA / det / "returns" / f"{sub['job_id']}_{sub['cluster']}_{extra.name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(extra, dest)
    wc = wallclock(sub["cluster"])
    wall = float(wc[0]) if wc else None
    receipt = {"line": "R2_lucid_gpu", "job_id": sub["job_id"], "cluster": sub["cluster"],
               "kind": "generation" if args[0] == "run" else args[0], "detector": det,
               "gpu": " ".join(wc[1:-2]) if len(wc) > 3 else sub["gpu"],
               "host": wc[-2] if len(wc) > 2 else None, "wall_s": wall,
               "gpu_h": (wall / 3600.0) if wall else None, "args": args,
               "shards_moved": len(moved), "shards_skipped": len(skipped),
               "exit_code": (ret / "JOB_EXIT_CODE").read_text().strip()
               if (ret / "JOB_EXIT_CODE").exists() else
               (DATA / det / "returns" / f"{sub['job_id']}_{sub['cluster']}_JOB_EXIT_CODE").read_text().strip()
               if (DATA / det / "returns" / f"{sub['job_id']}_{sub['cluster']}_JOB_EXIT_CODE").exists() else None,
               "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RECEIPTS, "a") as f:
        f.write(json.dumps(receipt) + "\n")
    done.write_text(json.dumps({"moved": moved, "skipped": skipped, "receipt": receipt}, indent=1))
    return receipt


def main():
    out = []
    for sub in sorted(JOBS.glob("*/*/submit.json")):
        r = collect(sub.parent)
        if r:
            out.append(r)
            print(json.dumps({k: r[k] for k in ("job_id", "cluster", "gpu_h", "shards_moved",
                                                "shards_skipped", "exit_code")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
