#!/usr/bin/env python3
"""r2-lucid L8: toy-versus-LUCiD tables (TOY_VS_LUCID.md / INTERIM_NIGHT1.md + JSON).

Toy values: the committed repair-campaign results (``results/round2/20261117_repair``: retrained
s9000 runs in ``retrain/retrain_readings.json``; frozen checkpoints and init, both seeds, in
``probes/probe_repair_results.json``). LUCiD values: ``final.json`` records written by
``ext_r2_repair_evolution evaluate`` under ``/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train``.
The primary vertex metric is the latent spatial-moment Ridge readout; every arm is also shown as
its gain over ``init`` of the same seed. The approximate ToF-only ceiling is the same simple Huber
fit on both datasets (no CRLB claim for LUCiD data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
TOY = REPO / "results/round2/20261117_repair"
LUCID = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train/retrain/by_run")
DETS = ("wcte", "sk", "juno", "hk")
ARMS = ("init", "mae_mse", "mae_repaired", "jepa_repaired", "sigreg_jepa")


def toy_rows():
    out = {}
    r = json.loads((TOY / "retrain/retrain_readings.json").read_text())
    for tag, x in r["runs"].items():
        out[(x["detector"], x["arm"], int(x["seed"]))] = {"source": "retrained (repair)",
                                                            "standard": x["standard"], "repaired": x["repaired"]}
    p = json.loads((TOY / "probes/probe_repair_results.json").read_text())
    for x in p["runs"]:
        key = (x["detector"], x["arm"], int(x["seed"]))
        if key not in out:
            out[key] = {"source": "frozen checkpoint" if x["arm"] != "init" else "init",
                        "standard": None, "repaired": {"vertex": x["vertex"], "token": x["token"]}}
    return out


def lucid_rows():
    out = {}
    for f in sorted(LUCID.glob("*/final.json")):
        x = json.loads(f.read_text())
        out[(x["detector"], x["arm"], int(x["seed"]))] = {"source": "LUCiD", "standard": x["standard"],
                                                            "repaired": x["repaired"], "path": str(f)}
    return out


def metrics(row):
    if row is None:
        return {}
    v = row["repaired"]["vertex"]
    m = {"spatial_latent_m": v["ridge_spatial_moment_latent"]["mean_m"],
         "spatial_latent_rfid": v["ridge_spatial_moment_latent"]["mean_over_r_fid"],
         "spatial_mlp_m": v["mlp_spatial_moment"]["mean_m"],
         "all_mean_m": v["ridge_all_mean"]["mean_m"],
         "hit_mean_m": v["ridge_hit_mean"]["mean_m"],
         "weighted_means_m": v["ridge_weighted_means"]["mean_m"],
         "raw_spatial_m": v["ridge_raw_spatial_moment"]["mean_m"],
         "token_time_r2": row["repaired"]["token"]["post_ln_unmasked_r2"]["token_time_r2"],
         "token_charge_r2": row["repaired"]["token"]["post_ln_unmasked_r2"]["token_log_charge_r2"],
         "masked_pred_time_r2": row["repaired"]["token"]["masked_target_pred_r2"]["token_time_r2"]}
    s = row.get("standard")
    if s:
        m.update(f_event=s.get("f_event"), f_layout=s.get("f_layout"),
                 effective_rank=s.get("effective_rank"), light_yield_rms=s.get("light_yield_rms"),
                 mue_auc=s.get("particle", {}).get("mue_auc"),
                 muon_energy_r2=s.get("particle", {}).get("muon_energy_r2"),
                 electron_energy_r2=s.get("particle", {}).get("electron_energy_r2"),
                 muon_dir_deg=s.get("particle", {}).get("muon_direction_mlp_median_deg"),
                 electron_dir_deg=s.get("particle", {}).get("electron_direction_mlp_median_deg"))
    return m


def fmt(x, nd=4):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def tof(det, kind, n_max=1000):
    """Approximate ToF-only ceiling: Huber least squares on token first-hit times (probe-test bombs)."""
    from jepa_noise import data_roots as dr
    from jepa_noise import r2_store as rs
    from scipy.optimize import least_squares
    geometry = dr.geometry(det, kind)
    cohort = rs.open_cohort(dr.pack(det, kind), "dev_probe_test")
    errs = []
    for i in range(cohort.n_runs):
        store, truth = cohort.run_store(i), cohort.run_truth(i)
        for k in np.flatnonzero(truth["domain"].astype(str) == "photon"):
            sl = slice(store.offsets[k], store.offsets[k + 1])
            p, t = store.pmt_id[sl], store.time_ns[sl].astype(float)
            if p.size < 5:
                continue
            pos = geometry.positions_m[p]
            f = lambda q: t - q[3] - np.linalg.norm(pos - q[:3], axis=1) * 1.33 / 0.299792458
            r = least_squares(f, np.zeros(4), loss="huber", f_scale=2.0)
            errs.append(float(np.linalg.norm(r.x[:3] - truth["vertex_m"][k])))
            if len(errs) >= n_max:
                break
        if len(errs) >= n_max:
            break
    e = np.array(errs)
    return {"n": int(e.size), "mean_m": float(e.mean()), "median_m": float(np.median(e))}


def build(dets, out_md, title, with_tof):
    toy, luc = toy_rows(), lucid_rows()
    rec = {"toy_sources": ["results/round2/20261117_repair/retrain/retrain_readings.json",
                           "results/round2/20261117_repair/probes/probe_repair_results.json"],
           "lucid_source": str(LUCID), "rows": [], "tof": {}}
    L = [f"# {title}", "",
         "Mean 3-D vertex error (m) on photon-bomb probe-test events (run-grouped probes, 3,400 train / 3,400 test "
         "in the toy; LUCiD counts in the JSON). **Primary metric: latent spatial-moment Ridge.** Toy = analytic-toy "
         "world (round 2 / repair); LUCiD = `lucid_v1` (tier A1). Gain over init = init error − arm error (positive "
         "is better), same seed and same data. Toy rows marked *frozen* are the original round-1/2 checkpoints "
         "(their s9001 repaired arms were never trained in the toy).", ""]
    for det in dets:
        L += [f"## {det.upper()}", "",
              "| Arm | Seed | Spatial Ridge toy | Spatial Ridge LUCiD | Gain/init toy | Gain/init LUCiD | Spatial MLP toy | Spatial MLP LUCiD | All-mean toy | All-mean LUCiD | Raw spatial LUCiD | Token time R² toy / LUCiD | f_event toy / LUCiD | f_layout toy / LUCiD |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for seed in (9000, 9001):
            ti = metrics(toy.get((det, "init", seed))); li = metrics(luc.get((det, "init", seed)))
            for arm in ARMS:
                t, l = toy.get((det, arm, seed)), luc.get((det, arm, seed))
                if t is None and l is None:
                    continue
                mt, ml = metrics(t), metrics(l)
                gt = (ti.get("spatial_latent_m") - mt["spatial_latent_m"]) if mt and ti else None
                gl = (li.get("spatial_latent_m") - ml["spatial_latent_m"]) if ml and li else None
                tag = arm + (" *frozen*" if t and t["source"] == "frozen checkpoint" else "")
                L.append(f"| {tag} | {seed} | {fmt(mt.get('spatial_latent_m'))} | {fmt(ml.get('spatial_latent_m'))} | "
                         f"{fmt(gt) if arm != 'init' else '–'} | {fmt(gl) if arm != 'init' else '–'} | "
                         f"{fmt(mt.get('spatial_mlp_m'))} | {fmt(ml.get('spatial_mlp_m'))} | {fmt(mt.get('all_mean_m'))} | "
                         f"{fmt(ml.get('all_mean_m'))} | {fmt(ml.get('raw_spatial_m'))} | "
                         f"{fmt(mt.get('token_time_r2'), 3)} / {fmt(ml.get('token_time_r2'), 3)} | "
                         f"{fmt(mt.get('f_event'), 4)} / {fmt(ml.get('f_event'), 4)} | {fmt(mt.get('f_layout'), 3)} / {fmt(ml.get('f_layout'), 3)} |")
                rec["rows"].append({"detector": det, "arm": arm, "seed": seed, "toy_source": t and t["source"],
                                    "toy": mt, "lucid": ml, "gain_toy": gt, "gain_lucid": gl,
                                    "lucid_path": l and l.get("path")})
        part = [r for r in rec["rows"] if r["detector"] == det and r["lucid"].get("mue_auc") is not None]
        if part:
            L += ["", "Particle tasks (standard suite, LUCiD; toy where available):", "",
                  "| Arm | Seed | μ/e AUC toy / LUCiD | μ energy R² toy / LUCiD | e energy R² toy / LUCiD | μ dir. (°) toy / LUCiD | e dir. (°) toy / LUCiD |",
                  "|---|---:|---|---|---|---|---|"]
            for r in part:
                t, l = r["toy"], r["lucid"]
                L.append(f"| {r['arm']} | {r['seed']} | {fmt(t.get('mue_auc'), 3)} / {fmt(l.get('mue_auc'), 3)} | "
                         f"{fmt(t.get('muon_energy_r2'), 3)} / {fmt(l.get('muon_energy_r2'), 3)} | "
                         f"{fmt(t.get('electron_energy_r2'), 3)} / {fmt(l.get('electron_energy_r2'), 3)} | "
                         f"{fmt(t.get('muon_dir_deg'), 2)} / {fmt(l.get('muon_dir_deg'), 2)} | "
                         f"{fmt(t.get('electron_dir_deg'), 2)} / {fmt(l.get('electron_dir_deg'), 2)} |")
        if with_tof:
            rec["tof"][det] = {"toy": tof(det, "analytic_toy"), "lucid": tof(det, "lucid_v1")}
            tt, tl = rec["tof"][det]["toy"], rec["tof"][det]["lucid"]
            L += ["", f"Approximate ToF-only ceiling (simple Huber fit on token first-hit times, {tl['n']} probe-test bombs; "
                  f"not a likelihood ceiling): toy {tt['mean_m']:.3f} m mean / {tt['median_m']:.3f} m median; "
                  f"LUCiD {tl['mean_m']:.3f} / {tl['median_m']:.3f} m."]
        L.append("")
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_md).with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print("wrote", out_md, len(rec["rows"]), "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=list(DETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Toy versus LUCiD (r2-lucid L8)")
    ap.add_argument("--tof", action="store_true")
    a = ap.parse_args()
    build(a.detectors, a.out, a.title, a.tof)


if __name__ == "__main__":
    main()
