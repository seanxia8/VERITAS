#!/usr/bin/env python3
"""r2-sparse: add the ledger line R2_sparse_gpu (cap 28 GPU-h) to docs/round2/RESOURCE_LEDGER.json.

Inputs (JSON produced on Deepthought): batch receipts (receipts.jsonl) and the dt2 guard run_status files.
    sparse_ledger.py --receipts receipts.jsonl --dt2 dt2_status.json
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--dt2", required=True, help="JSON list of guard run_status records (deepthought2)")
    a = ap.parse_args()
    entries = []
    for line in Path(a.receipts).read_text().splitlines():
        r = json.loads(line)
        entries.append({"job": f"{r['job_id']} ({r.get('cluster')})",
                        "kind": "train" if r["job_id"].startswith("sparse_train") else "eval",
                        "gpu_h": float(r.get("gpu_h", 0.0)), "exit_code": str(r.get("exit_code")),
                        "where": "HTCondor A100 batch"})
    for r in json.loads(Path(a.dt2).read_text()):
        entries.append({"job": r["run_id"], "kind": "smoke" if "smoke" in r["run_id"] else "train",
                        "gpu_h": float(r["wall_seconds"]) / 3600.0, "exit_code": str(r["exit_status"]),
                        "where": "deepthought2 L40S (guarded lane)", "utc": r.get("finished_at_utc")})
    total = sum(e["gpu_h"] for e in entries)
    path = REPO / "docs/round2/RESOURCE_LEDGER.json"
    led = json.loads(path.read_text())
    line = {"authority": "RUNBOOK_2026-11-24_R2_SPARSE_WCTE.md §0.3", "cap_h": 28.0, "cpu_entries": [],
            "entries": entries, "booked_gpu_h": total,
            "by_kind": {k: sum(e["gpu_h"] for e in entries if e["kind"] == k) for k in ("train", "eval", "smoke")}}
    lines = led["lines"]
    if isinstance(lines, list):
        lines[:] = [x for x in lines if x[0] != "R2_sparse_gpu"] + [["R2_sparse_gpu", line]]
    else:
        lines["R2_sparse_gpu"] = line
    led["totals"]["R2_sparse_gpu"] = {"booked_gpu_h": round(total, 4), "cap_gpu_h": 28.0}
    path.write_text(json.dumps(led, indent=1) + "\n")
    print(json.dumps({"booked_gpu_h": round(total, 3), **{k: round(v, 3) for k, v in line["by_kind"].items()}}))


if __name__ == "__main__":
    main()
