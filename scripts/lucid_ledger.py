#!/usr/bin/env python3
"""Aggregate the R2_lucid_gpu ledger line (runbook §0.4, cap 120 GPU-h) on Deepthought.

Sources: guarded deepthought2 jobs (``/ceph/dwong/jepa/jobs/r2lucid_*/run_status.json``, wall seconds
of the job process) and HTCondor batch receipts (``ledger/receipts.jsonl``, RemoteWallClockTime of
the whole job including setup), plus HTCondor history for batch jobs that were not collected
(failed NEMO2 attempts, the pre-calibration smoke). Writes ``ledger/R2_lucid_gpu.json``.
"""
import json, subprocess, sys, time
from pathlib import Path

JOBS = Path("/ceph/dwong/jepa/jobs")
RET = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/ledger")
BATCH = Path("/work/dwong/r2_lucid/batch_jobs")
CAP = 120.0


def main(out=RET / "R2_lucid_gpu.json"):
    entries = []
    for st in sorted(JOBS.glob("r2lucid_*/run_status.json")):
        r = json.loads(st.read_text())
        if r.get("wall_seconds") is None:
            continue
        entries.append({"source": "deepthought2_guard", "run_id": r["run_id"], "gpu": "NVIDIA L40S",
                        "host": r.get("host"), "gpu_index": r.get("gpu_index"),
                        "wall_s": r["wall_seconds"], "gpu_h": r["wall_seconds"] / 3600.0,
                        "outcome": r.get("outcome"), "exit_status": r.get("exit_status"),
                        "utc": r.get("finished_at_utc"),
                        "kind": "train" if "train_" in r["run_id"] and "gen_" not in r["run_id"]
                        else "eval" if "eval" in r["run_id"] else "generation"})
    seen = set()
    rec = RET / "receipts.jsonl"
    if rec.exists():
        for line in rec.read_text().splitlines():
            r = json.loads(line)
            seen.add(str(r["cluster"]))
            entries.append({"source": "htcondor", "run_id": r["job_id"], "cluster": r["cluster"],
                            "gpu": r.get("gpu"), "host": r.get("host"), "wall_s": r.get("wall_s"),
                            "gpu_h": r.get("gpu_h") or 0.0, "kind": r.get("kind"),
                            "exit_status": r.get("exit_code"), "utc": r.get("collected_utc")})
    for sub in sorted(BATCH.glob("*/*/submit.json")):
        s = json.loads(sub.read_text())
        c = str(s.get("cluster"))
        if c in seen or not c or c == "None":
            continue
        h = subprocess.run(["condor_history", c, "-af", "RemoteWallClockTime", "JobStatus"],
                           capture_output=True, text=True).stdout.split()
        wall = float(h[0]) if h else 0.0
        if wall <= 0:
            continue
        note = json.loads((sub.parent / "COLLECTED.json").read_text()).get("not_collected") \
            if (sub.parent / "COLLECTED.json").exists() else "running or not yet collected"
        entries.append({"source": "htcondor_uncollected", "run_id": s["job_id"], "cluster": c,
                        "gpu": s.get("gpu"), "wall_s": wall, "gpu_h": wall / 3600.0,
                        "kind": "overhead", "note": note})
    total = sum(e["gpu_h"] or 0.0 for e in entries)
    line = {"authority": "RUNBOOK_2026-11-21_R2_LUCID_REGEN.md §0.4", "cap_h": CAP,
            "booked_gpu_h": total, "remaining_h": CAP - total, "n_entries": len(entries),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "entries": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(line, indent=1) + "\n")
    print(json.dumps({k: line[k] for k in ("booked_gpu_h", "remaining_h", "n_entries")}))


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
