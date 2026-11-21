#!/usr/bin/env python3
"""r2-lucid L8: worst-event displays (6 worst + 3 near-median photon bombs, spatial-moment readout)."""
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402

BY_RUN = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train/retrain/by_run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=9000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = f"{a.detector}__{a.arm}__s{a.seed}"
    p = np.load(BY_RUN / tag / "predictions.npz", allow_pickle=True)
    pred, truth = p["ridge_spatial_moment_latent"], p["vertex"]
    err = np.linalg.norm(pred - truth, axis=1)
    order = np.argsort(err)
    pick = list(order[::-1][:6]) + list(order[np.searchsorted(np.sort(err), np.median(err)) + np.arange(-1, 2)])
    geometry = dr.geometry(a.detector)
    cohort = rs.open_cohort(dr.pack(a.detector), "dev_probe_test")
    index = {r: i for i, r in enumerate(cohort.run_ids)}
    pos = geometry.positions_m
    if geometry.shape == "sphere":
        X = np.arctan2(pos[:, 1], pos[:, 0]); Y = np.arcsin(pos[:, 2] / np.linalg.norm(pos, axis=1))
        proj = lambda v: (np.arctan2(v[1], v[0]), np.arcsin(v[2] / max(np.linalg.norm(v), 1e-9)))
    else:
        R, H = geometry.radius_m, geometry.half_height_m
        X = np.where(geometry.surfaces == "barrel", np.arctan2(pos[:, 1], pos[:, 0]) * R, pos[:, 0])
        Y = np.where(geometry.surfaces == "barrel", pos[:, 2],
                     np.where(geometry.surfaces == "top", H + R + pos[:, 1], -H - R + pos[:, 1]))
        proj = lambda v: (np.arctan2(v[1], v[0]) * R, v[2])
    fig, axes = plt.subplots(len(pick), 2, figsize=(10, 2.3 * len(pick)))
    rows = []
    for row, j in zip(axes, pick):
        run, eid = str(p["run_ids"][j]), int(p["event_ids"][j])
        i = index[run]; store = cohort.run_store(i); k = int(np.flatnonzero(store.event_ids == eid)[0])
        sl = slice(store.offsets[k], store.offsets[k + 1])
        pm, q, t = store.pmt_id[sl], store.charge_pe[sl], store.time_ns[sl]
        for ax, val, lab in ((row[0], np.log10(np.maximum(q, 1e-2)), "log10 charge"), (row[1], t, "token time (ns)")):
            ax.scatter(X, Y, s=1, c="0.85"); sc = ax.scatter(X[pm], Y[pm], s=3, c=val, cmap="viridis")
            plt.colorbar(sc, ax=ax, label=lab); ax.set_xticks([]); ax.set_yticks([])
            for v, m, c in ((truth[j], "x", "red"), (pred[j], "+", "magenta")):
                ax.plot(*proj(v), marker=m, color=c, ms=8, mew=2)
        row[0].set_title(f"{a.detector.upper()} {a.arm} s{a.seed} {run} ev {eid}: error {err[j]:.3f} m, {pm.size} hits "
                         f"(x truth, + spatial readout)", fontsize=7)
        rows.append({"run_id": run, "event_id": eid, "error_m": float(err[j]), "n_hits": int(pm.size),
                     "truth_m": truth[j].tolist(), "pred_m": pred[j].tolist()})
    fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100); plt.close(fig)
    Path(a.out).with_suffix(".json").write_text(json.dumps({"tag": tag, "median_error_m": float(np.median(err)),
                                                            "events": rows}, indent=1) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
