#!/usr/bin/env python3
"""L5: inventory the analytic-toy datasets (no moves, no deletions, no writes inside them).

Per file: path, size, mtime and SHA256 (files > 1 GB: SHA256 of the first and last 64 MB plus the
size). Per dataset: total size, file count, newest mtime. Writes
``/ceph/dwong/jepa/data/ARCHIVE_ANALYTIC_TOY_2026-11-21.json`` (the markdown is written separately).
"""
import hashlib, json, os, sys, time
from pathlib import Path

DATA = Path("/ceph/dwong/jepa/data")
DATASETS = {
    "round1_ext_pregrid_dev_20261025": (DATA / "ext_pregrid_dev_20261025", "round 1: WCTE training and probe shards (analytic toy, R-6 readout)"),
    "round1_lucid_r6": (DATA / "lucid_r6", "round 1: WCTE analytic-toy data (R-6)"),
    "round1_lucid_r6_v2": (DATA / "lucid_r6_v2", "round 1: WCTE analytic-toy dev datasets and evidence (R-6 v2)"),
    "round1_bridge": (DATA / "bridge", "round 1: bridge datasets (analytic toy)"),
    "round1_bridge_delay_fixed001": (DATA / "bridge_delay_fixed001", "round 1: bridge delay datasets (analytic toy)"),
    "round2_sk": (DATA / "round2/sk", "round 2: SK pack (analytic toy)"),
    "round2_juno": (DATA / "round2/juno", "round 2: JUNO pack (analytic toy)"),
    "round2_hk": (DATA / "round2/hk", "round 2: HK pack (analytic toy)"),
    "round2_wcte_probe_pack": (Path("/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/wcte_regression"),
                               "round 2: WCTE regression probe pack (analytic toy)"),
}
BIG, CHUNK = 1 << 30, 64 << 20


def digest(path, size):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= BIG:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
            return h.hexdigest(), "full"
        h.update(f.read(CHUNK)); f.seek(size - CHUNK); h.update(f.read(CHUNK))
        h.update(str(size).encode())
        return h.hexdigest(), "first64MB+last64MB+size"


def main(out):
    record = {"schema": "jepa-archive-analytic-toy-v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "runbook": "docs/round2/RUNBOOK_2026-11-21_R2_LUCID_REGEN.md §L5", "datasets": {}}
    for key, (root, desc) in DATASETS.items():
        files = []
        for p in sorted(x for x in root.rglob("*") if x.is_file() and not x.is_symlink()):
            st = p.stat()
            sha, mode = digest(p, st.st_size)
            files.append({"path": str(p), "bytes": st.st_size, "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                          "sha256": sha, "sha256_mode": mode, "writable": bool(st.st_mode & 0o222)})
        record["datasets"][key] = {"root": str(root), "description": desc, "n_files": len(files),
                                   "bytes": sum(f["bytes"] for f in files),
                                   "newest_mtime_utc": max((f["mtime_utc"] for f in files), default=None),
                                   "files": files}
        print(key, len(files), record["datasets"][key]["bytes"], flush=True)
    Path(out).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DATA / "ARCHIVE_ANALYTIC_TOY_2026-11-21.json"))
