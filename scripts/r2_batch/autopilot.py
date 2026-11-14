#!/usr/bin/env python3
"""One pass of round-2 batch bookkeeping (run on Deepthought at each poll):
1. sync every submitted job that left the queue (idempotent);
2. a training job whose run is complete -> its evaluation (JUNO/HK: batch A100; SK: deepthought2 lane E);
3. a training job whose run is incomplete and not capped -> resubmit it to resume from its checkpoint.
Prints one line per action."""
import json, subprocess, sys
from pathlib import Path

R = Path("/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11")
B = R / "batch"
JOBS = Path("/work/dwong/r2_batch/jobs")
REPO = Path("/work/dwong/r2_repo")
PY = "/home/dwong/anaconda3/bin/python"
LANE_E = Path("/work/dwong/r2_ops/laneE.queue")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))


def in_queue():
    out = sh(["condor_q", "dwong", "-af", "ClusterId"]).stdout.split()
    return set(out)


def latest_by_run():
    subs = [json.loads(l) for l in (JOBS / "submissions.jsonl").read_text().splitlines() if l.strip()]
    last = {}
    for s in subs:
        if s.get("cluster"):
            last[s["run_id"]] = s
    return last


def arg(args, name):
    return args[args.index(name) + 1] if name in args else None


def main():
    q = in_queue()
    code = sh(["bash", str(REPO / "scripts/r2_batch/stage_code.sh")]).stdout.strip()
    for run_id, s in sorted(latest_by_run().items()):
        d = Path(s["dir"])
        if s["cluster"] in q:
            continue
        if not (d / "SYNCED").exists():
            if not (d / "returns_out.tar").exists():
                print(f"NO-OUTPUT {run_id} cluster {s['cluster']} (held/removed?)"); continue
            r = sh([PY, str(REPO / "scripts/r2_batch/sync.py"), str(d)])
            print(f"SYNC {run_id}: {r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]}")
        a = s["args"]
        if not a or a[0] != "train":
            continue
        det, arm, seed, mb = arg(a, "--detector"), arg(a, "--arm"), arg(a, "--seed"), arg(a, "--micro-batch")
        run = f"{det}__{arm}__s{seed}"
        rec_p = B / "runs" / run / "run_record.json"
        if not rec_p.exists():
            print(f"NO-RECORD {run} (job failed before training; see {d}/err.txt)"); continue
        rec = json.loads(rec_p.read_text())
        mark = d / "FOLLOWUP"
        if mark.exists():
            continue
        if rec.get("complete"):
            ckpt = f"returns/runs/{run}/checkpoint.pt"
            if det == "sk":
                line = (f"r2_eval_sk_{arm}_s{seed}|evaluation|40|evaluate --detector sk --items "
                        f"{arm}:{seed}:{B}/runs/{run}/checkpoint.pt --device cuda")
                with open(LANE_E, "a") as f:
                    f.write(line + "\n")
                print(f"EVAL-QUEUED-DT2 {run}")
            else:
                r = sh([PY, str(REPO / "scripts/r2_batch/submit.py"), f"r2_batch_eval_{run}", "--gpu", "a100",
                        "--walltime-h", "2", "--cpus", "8", "--mem-mb", "96000", "--code", code,
                        "--files", f"data/{det}_r2.h5", "--resume-run", run, "--",
                        "evaluate", "--detector", det, "--items", f"{arm}:{seed}:{ckpt}",
                        "--h5", f"staged/data/{det}_r2.h5", "--out", "returns", "--device", "cuda"])
                print(f"EVAL-SUBMIT {run}: {r.stdout.strip()[-160:]} {r.stderr.strip()[-200:]}")
            mark.write_text("eval")
        elif rec.get("wall_cap_hit"):
            print(f"CAPPED {run} at step {rec.get('steps')}"); mark.write_text("capped")
        else:
            wall = s.get("walltime_h", 10)
            r = sh([PY, str(REPO / "scripts/r2_batch/submit.py"), run_id, "--gpu", s.get("gpu", "a100"),
                    "--walltime-h", str(wall), "--cpus", "8", "--mem-mb", "96000", "--code", code,
                    "--files", *s["files"], "--smokes-for", det, "--resume-run", run, "--", *a])
            print(f"RESUME {run} from step {rec.get('steps')}: {r.stdout.strip()[-160:]} {r.stderr.strip()[-200:]}")
            mark.write_text("resumed")


if __name__ == "__main__":
    main()
