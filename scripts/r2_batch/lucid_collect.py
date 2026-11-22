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

# r2-tier1: R2_BATCH_JOBS / R2_BATCH_RECEIPTS / R2_BATCH_TRAIN / R2_BATCH_LINE select another phase's
# job tree, receipts, training returns and ledger line (defaults: r2-lucid, unchanged);
# R2_BATCH_EVAL / R2_BATCH_YARD: where tier1_eval and yardstick outputs go (unset: not collected).
JOBS = Path(os.environ.get("R2_BATCH_JOBS", "/work/dwong/r2_lucid/batch_jobs"))
DATA = Path("/ceph/dwong/jepa/data/lucid_v1")
RECEIPTS = Path(os.environ.get("R2_BATCH_RECEIPTS",
                               "/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/ledger/receipts.jsonl"))
TRAIN = Path(os.environ.get("R2_BATCH_TRAIN", "/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train"))
LINE = os.environ.get("R2_BATCH_LINE", "R2_lucid_gpu")
EVAL = Path(os.environ["R2_BATCH_EVAL"]) if os.environ.get("R2_BATCH_EVAL") else None
YARD = Path(os.environ["R2_BATCH_YARD"]) if os.environ.get("R2_BATCH_YARD") else None


def collect_torch(job_dir: Path, sub: dict):
    """Training/evaluation outputs -> TRAIN/{runs,retrain/by_run}/<tag>; older copies kept aside."""
    tar = job_dir / "returns_out.tar"
    inc = TRAIN / ".incoming" / f"{sub['job_id']}_{sub['cluster']}"
    inc.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(inc, filter="data")
    ret = inc / "returns"
    moved = []
    for kind in ("runs", "retrain/by_run", "smokes"):
        for d in sorted((ret / kind).glob("*")) if (ret / kind).exists() else []:
            dest = TRAIN / kind / d.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                os.rename(dest, dest.with_name(f"{d.name}.prev_{sub['cluster']}"))
            os.rename(d, dest)
            moved.append(str(dest))
    # r2-tier1: tier1_eval outputs eval/<run_id>/step_<NNNNN>/<files> -> EVAL/...; never overwritten
    if EVAL is not None and (ret / "eval").exists():
        for f in sorted(p for p in (ret / "eval").rglob("*") if p.is_file()):
            dest = EVAL / f.relative_to(ret / "eval")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                os.rename(dest, dest.with_name(f"{dest.name}.prev_{sub['cluster']}"))
            os.rename(f, dest)
            moved.append(str(dest))
    for log in sorted(ret.glob("SEALED_ACCESS_LOG*.md")):
        dest = (EVAL.parent if EVAL is not None else TRAIN) / "batch_access_logs" / f"{sub['job_id']}_{sub['cluster']}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(log, dest)
        moved.append(str(dest))
    code = (ret / "JOB_EXIT_CODE").read_text().strip() if (ret / "JOB_EXIT_CODE").exists() else None
    return moved, code


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
    if sub.get("torch_module"):
        moved, code = collect_torch(job_dir, sub)
        wc = wallclock(sub["cluster"])
        wall = float(wc[0]) if wc else None
        receipt = {"line": LINE, "job_id": sub["job_id"], "cluster": sub["cluster"],
                   "kind": "train" if "train" in sub["torch_module"] else "eval",
                   "run_id": next((a for a in args if "__" in a), None),
                   "module": sub["torch_module"], "args": args, "wall_s": wall,
                   "gpu_h": (wall / 3600.0) if wall else None,
                   "gpu": " ".join(wc[1:-2]) if len(wc) > 3 else sub["gpu"],
                   "host": wc[-2] if len(wc) > 2 else None, "moved": moved, "exit_code": code,
                   "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
        with open(RECEIPTS, "a") as f:
            f.write(json.dumps(receipt) + "\n")
        done.write_text(json.dumps({"receipt": receipt}, indent=1))
        return dict(receipt, shards_moved=len(moved), shards_skipped=0)
    if args and args[0] == "yardstick":  # r2-tier1 T6: outputs -> YARD/<job>_<cluster>/
        if YARD is None:
            return None
        dest = YARD / f"{sub['job_id']}_{sub['cluster']}"
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            t.extractall(dest, filter="data")
        wc = wallclock(sub["cluster"])
        wall = float(wc[0]) if wc else None
        code_file = dest / "returns" / "JOB_EXIT_CODE"
        receipt = {"line": LINE, "job_id": sub["job_id"], "cluster": sub["cluster"], "kind": "yardstick",
                   "args": args, "wall_s": wall, "gpu_h": (wall / 3600.0) if wall else None,
                   "gpu": " ".join(wc[1:-2]) if len(wc) > 3 else sub["gpu"],
                   "host": wc[-2] if len(wc) > 2 else None, "moved": [str(dest)],
                   "exit_code": code_file.read_text().strip() if code_file.exists() else None,
                   "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
        with open(RECEIPTS, "a") as f:
            f.write(json.dumps(receipt) + "\n")
        done.write_text(json.dumps({"receipt": receipt}, indent=1))
        return dict(receipt, shards_moved=0, shards_skipped=0)
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
    receipt = {"line": LINE, "job_id": sub["job_id"], "cluster": sub["cluster"],
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
