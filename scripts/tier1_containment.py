#!/usr/bin/env python3
"""r2-tier1 T3.5 (D13): mu/e containment counts under LUCiD's flag and under an alternative.

LUCiD flag: ``contained`` as generated (SIREN 99.99 % emission extent inside the tank).
Alternative (defined here): the true track end, vertex + direction * track_length_m, lies inside
the detector volume (|z| < H/2 and rho < R of the generation geometry). Development cohorts only.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import data_roots as dr, lucid_bombs as lb, r2_store as rs  # noqa: E402


def counts(det, cohort_name):
    g = dr.geometry(det, "lucid_v1")
    c = rs.open_cohort(dr.pack(det, "lucid_v1"), cohort_name)
    out = {d: {"n": 0, "lucid_flag": 0, "track_end_inside": 0, "both": 0} for d in lb.PARTICLE_DOMAINS}
    for i in range(c.n_runs):
        t = c.run_truth(i)
        dom = np.asarray(t["domain"]).astype(str)
        end = np.asarray(t["vertex_m"], float) + np.asarray(t["direction_m"], float) * np.asarray(t["track_length_m"], float)[:, None]
        inside = (np.hypot(end[:, 0], end[:, 1]) < g.radius_m) & (np.abs(end[:, 2]) < g.half_height_m)
        flag = np.asarray(t["contained"], bool)
        for d in lb.PARTICLE_DOMAINS:
            m = dom == d
            out[d]["n"] += int(m.sum()); out[d]["lucid_flag"] += int((flag & m).sum())
            out[d]["track_end_inside"] += int((inside & m & np.isfinite(end).all(1)).sum())
            out[d]["both"] += int((flag & inside & m).sum())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {det: {c: counts(det, c) for c in ("dev_probe_train", "dev_probe_test")} for det in ("wcte", "sk", "juno", "hk")}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["wcte"]))


if __name__ == "__main__":
    main()
