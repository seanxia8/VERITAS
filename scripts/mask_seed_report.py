#!/usr/bin/env python3
"""r2-mask M3.4-6: SEED_TEST.md / .json from the seedtest job outputs (``<det>_seed_NNNN.json``).

Per detector, start and fit mode: mean and median final 3-D vertex error, the good-basin fraction
(final error <= 2 x the median final error of the truth start, same detector and mode), mean
iterations and wall time, and the fraction of starts projected into the tank. Registered
expectation (PREREG appendix M3): on SK, counts-only (primary), good-basin fraction >= 80 % for the
readout start (c) and <= 30 % for the centroid start (b); the piece is dropped if (c) < 50 %.

Usage: mask_seed_report.py --inputs <dir with sk_seed_*.json> <dir with wcte_seed_*.json> --out-dir results/...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = {"truth": "(a) truth", "centroid": "(b) charge centroid", "readout": "(c) frozen readout, s9000 @10k",
          "init": "(d) frozen readout, matched init", "raw": "(e) raw-feature baseline"}
MODES = ("counts", "joint")


def load(dirs):
    rows = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*_seed_*.json")):
            r = json.loads(p.read_text())
            det = p.name.split("_seed_")[0]
            rows.setdefault(det, {})[r["index"]] = r
    return rows


def summarise(rows):
    out = {}
    for det, ev in rows.items():
        events = [ev[i] for i in sorted(ev)]
        starts = list(events[0]["starts"])
        d = {"n_events": len(events), "config": events[0]["config"], "modes": {}}
        for m in MODES:
            if m not in events[0]["starts"]["truth"]["fits"]:
                continue
            truth_err = np.array([e["starts"]["truth"]["fits"][m]["vertex_error_m"] for e in events])
            thr = 2.0 * float(np.median(truth_err))
            dm = {"good_basin_threshold_m": thr, "starts": {}}
            for s in starts:
                err = np.array([e["starts"][s]["fits"][m]["vertex_error_m"] for e in events])
                err0 = np.array([e["starts"][s]["start_error_m"] for e in events])
                nit = np.array([e["starts"][s]["fits"][m]["nit"] for e in events])
                wall = np.array([e["starts"][s]["fits"][m]["wall_s"] for e in events])
                clip = np.array([e["starts"][s]["clipped"] for e in events])
                good = err <= thr
                k = int(good.sum())
                se = float(np.sqrt(good.mean() * (1 - good.mean()) / len(good)))
                dm["starts"][s] = {"label": LABELS.get(s, s), "start_mean_m": float(err0.mean()),
                                   "start_median_m": float(np.median(err0)),
                                   "final_mean_m": float(err.mean()), "final_median_m": float(np.median(err)),
                                   "good_basin_fraction": float(good.mean()), "good_basin_n": k,
                                   "good_basin_se": se, "iterations_mean": float(nit.mean()),
                                   "wall_s_mean": float(wall.mean()), "clipped_fraction": float(clip.mean())}
            d["modes"][m] = dm
        out[det] = d
    return out


def verdict(s):
    sk = s.get("sk", {}).get("modes", {}).get("counts")
    if not sk:
        return {"evaluable": False}
    c = sk["starts"]["readout"]["good_basin_fraction"]
    b = sk["starts"]["centroid"]["good_basin_fraction"]
    return {"evaluable": True, "sk_counts_readout_good_basin": c, "sk_counts_centroid_good_basin": b,
            "readout_ge_80": bool(c >= 0.80), "centroid_le_30": bool(b <= 0.30),
            "expectation_met": bool(c >= 0.80 and b <= 0.30), "dropped": bool(c < 0.50)}


def render(s, v):
    md = ["# SEED_TEST — r2-mask M3 (readout-seeded LUCiD fit)", "",
          "Runbook `RUNBOOK_2026-11-23_R2_MASK_SCALE.md` §M3; expectation registered in "
          "`PREREG_2026-11-23_R2_MASK.md` (appendix M3) before running. Events: the 200 Y2 `dev_probe_test` "
          "photon bombs per detector of the r2-tier1 yardstick input. Fits: LUCiD's expected-response model "
          "and losses as in the yardstick (`tier1_yardstick.Model`, L-BFGS-B, maxiter 40 for every start); "
          "counts-only is primary (the joint time term has an unresolved t₀ offset). Starts (x, y, z) as listed; "
          "ln N from a linear fit of ln N on ln(total charge) over `dev_probe_train`; t₀ from the median time "
          "residual at the starting vertex (truth start: full true θ). Good basin = final error ≤ 2 × the median "
          "final error of the truth start (same detector and mode).", ""]
    for det in ("sk", "wcte"):
        if det not in s:
            continue
        d = s[det]
        md += [f"## {det.upper()} ({d['n_events']} events)", ""]
        for m, dm in d["modes"].items():
            md += [f"### {m}{' (primary)' if m == 'counts' else ''} — good-basin threshold "
                   f"{dm['good_basin_threshold_m']:.3f} m", "",
                   "| Start | start error mean / median (m) | final mean (m) | final median (m) | good basin | "
                   "iterations | wall (s/fit) | projected into tank |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, r in dm["starts"].items():
                md.append(f"| {r['label']} | {r['start_mean_m']:.3f} / {r['start_median_m']:.3f} | "
                          f"{r['final_mean_m']:.3f} | {r['final_median_m']:.3f} | "
                          f"{100 * r['good_basin_fraction']:.1f} % ± {100 * r['good_basin_se']:.1f} | "
                          f"{r['iterations_mean']:.1f} | {r['wall_s_mean']:.2f} | {100 * r['clipped_fraction']:.1f} % |")
            md.append("")
    md += ["## Registered expectation (SK, counts-only)", ""]
    if v.get("evaluable"):
        md.append(f"Readout start (c): {100 * v['sk_counts_readout_good_basin']:.1f} % in the good basin "
                  f"(expected ≥ 80 %: {'met' if v['readout_ge_80'] else 'not met'}); centroid start (b): "
                  f"{100 * v['sk_counts_centroid_good_basin']:.1f} % (expected ≤ 30 %: "
                  f"{'met' if v['centroid_le_30'] else 'not met'}). Expectation "
                  f"**{'met' if v['expectation_met'] else 'not met'}**."
                  + (" (c) is below 50 %: **this piece is dropped** (runbook §M3.5)." if v["dropped"] else ""))
    else:
        md.append("Not evaluable (SK counts-only results missing).")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    s = summarise(load(a.inputs))
    v = verdict(s)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "SEED_TEST.json").write_text(json.dumps({"phase": "r2-mask", "stage": "M3", "summary": s,
                                                          "verdict": v}, indent=1))
    (a.out_dir / "SEED_TEST.md").write_text(render(s, v))
    print(render(s, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
