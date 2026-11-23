#!/usr/bin/env python3
"""r2-tier1: submit one ``scripts/tier1_eval.py`` scoring job to HTCondor A100 (run on Deepthought).

For a sealed read, the canonical access-log line is written first, here on Deepthought, by
``tier1_eval.py --log-only`` (the job also logs its read in a local copy that the collector files).
The run directory is shipped; the main pack and (if requested) the sealed pack are staged inputs.

  tier1_eval_batch.py --run-dir <train/runs/run_id> --steps 10000 0 --cohorts validation sealed \
      --purpose "..." [--probes ridge mlp attentive] [--photons-only] [--walltime-h 3]
  tier1_eval_batch.py --legacy-init --detector sk --seed 9000 --cohorts validation sealed --purpose "..."
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# r2-mask: an existing R2_BATCH_JOBS / R2_BATCH_REPO in the environment wins (defaults: r2-tier1)
ENV = {"R2_BATCH_JOBS": os.environ.get("R2_BATCH_JOBS", "/work/dwong/r2_tier1/batch_jobs"),
       "R2_BATCH_REPO": os.environ.get("R2_BATCH_REPO", str(REPO))}
TIER1_ACCESS_LOG = "/ceph/dwong/jepa/returns/r2_tier1_2026-11-22/SEALED_ACCESS_LOG.md"


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path)
    src.add_argument("--legacy-init", action="store_true")
    ap.add_argument("--detector")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", nargs="+", type=int, default=[10000])
    ap.add_argument("--cohorts", nargs="+", default=["validation", "sealed"])
    ap.add_argument("--probes", nargs="+", default=["ridge", "mlp", "attentive"])
    ap.add_argument("--photons-only", action="store_true")
    ap.add_argument("--budget", action="store_true",
                    help="T4 budget axis: validation, photons only, written to eval_budget/ (never mixed with eval/)")
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--code", required=True, help="staged code tarball, e.g. code/tier1_<sha8>.tar.gz")
    ap.add_argument("--walltime-h", type=float, default=3.0)
    ap.add_argument("--job-id")
    ap.add_argument("--access-log", default=TIER1_ACCESS_LOG,
                    help="canonical sealed-access log written on Deepthought (r2-mask: its own log)")
    a = ap.parse_args()
    if a.budget:
        if "sealed" in a.cohorts:
            ap.error("--budget is validation-only")
        a.photons_only = True
    if a.legacy_init:
        det, rid = a.detector, f"{a.detector}__init__s{a.seed}"
        src_args = ["--legacy-init", "--detector", det, "--seed", str(a.seed)]
        ship = []
    else:
        rec = json.loads((a.run_dir / "run_record.json").read_text())
        rid = rec["recipe"]["identity"]["run_id"]
        det = rid.split("__")[0]
        src_args = ["--run-dir", f"returns/runs/{rid}", "--step", *map(str, a.steps)]
        ship = [str(a.run_dir)]
    files = [f"data/lucid_v1/{det}_lucid_v1.h5"]
    pack_args = ["--validation-pack", f"staged/data/lucid_v1/{det}_lucid_v1.h5"]
    if "sealed" in a.cohorts:
        files.append(f"data/lucid_v1/{det}_lucid_v1_sealed.h5")
        pack_args += ["--sealed-pack", f"staged/data/lucid_v1/{det}_lucid_v1_sealed.h5"]
        log = [sys.executable, str(REPO / "scripts/tier1_eval.py"), "--cohorts", "sealed", "--log-only",
               "--purpose", a.purpose, "--access-log", a.access_log]
        log += (["--legacy-init", "--detector", det, "--seed", str(a.seed)] if a.legacy_init
                else ["--run-dir", str(a.run_dir), "--step", *map(str, a.steps)])
        subprocess.run(log, check=True, env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    job_args = [*src_args, "--cohorts", *a.cohorts, "--probes", *a.probes, *pack_args,
                "--out-root", "returns/eval_budget" if a.budget else "returns/eval", "--access-log", "returns/SEALED_ACCESS_LOG_batch.md",
                "--device", "cuda", "--memory-fraction", "0.9", "--purpose", a.purpose.replace(" ", "_")]
    if a.photons_only:
        job_args.append("--photons-only")
    job = a.job_id or f"eval_{rid}_" + "_".join(map(str, a.steps))
    cmd = [sys.executable, str(REPO / "scripts/r2_batch/lucid_submit.py"), job, "--gpu", "a100",
           "--walltime-h", str(a.walltime_h), "--code", a.code, "--torch-module", "scripts.tier1_eval",
           "--files", *files]
    if ship:
        cmd += ["--ship", *ship]
    cmd += ["--", *job_args]
    out = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **ENV))
    print(out.stdout.strip()[-400:], out.stderr.strip()[-400:])
    return out.returncode


if __name__ == "__main__":
    raise SystemExit(main())
