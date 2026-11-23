#!/usr/bin/env python3
"""r2-tier1 T6: YARDSTICKS.md / .json from the per-event yardstick outputs (tier1_yardstick.py).

Y1: surrogate Fisher split (counts-only, time-only, joint sigma_vertex; time share), distributions and
medians by radius and yield tertile. Y2: achieved resolution of LUCiD's reconstruction loss started at
truth (primary) and at the charge-weighted centroid (first 50), per-coordinate bias with SE, converged
fraction; next to the frozen Ridge readouts on the same `dev_probe_test` photon bombs (validation split).
Diagnostic re-fits with another time-term resolution are reported separately and never feed K4(b).
Run on Deepthought: tier1_yardstick_report.py --out results/round2/20261122_tier1/YARDSTICKS
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

T1 = Path("/ceph/dwong/jepa/returns/r2_tier1_2026-11-22")
YARD = T1 / "yardstick"
INPUTS = Path("/ceph/srv/dwong/r2/data/tier1")
RIDGE_RUNS = {"sk": ["sk__mae_repaired__s9000", "sk__init__s9000"],
              "wcte": ["wcte__mae_repaired__s9000", "wcte__init__s9000"]}
PAR = ("x", "y", "z", "t0", "lnN")
LABEL_Y1 = ("surrogate Fisher from LUCiD's expectation model; omits SPE smearing, charge-dependent "
            "jitter and threshold; an optimistic bound")


def load_rows(det, pattern):
    rows = {}
    for f in sorted(glob.glob(str(YARD / pattern / "returns/yardstick" / f"{det}_event_*.json"))):
        d = json.loads(Path(f).read_text())
        rows[d["index"]] = d
    return rows


def y2_summary(rows, part):
    th, tt = [], []
    for i in sorted(rows):
        y = rows[i].get(part) or {}
        if y:
            th.append(y)
            tt.append(rows[i]["theta_true"])
    out = {}
    for m in ("counts", "time", "joint"):
        fits = [(np.array(y[m]["theta"], float), np.array(t, float), bool(y[m]["success"]))
                for y, t in zip(th, tt) if m in y]
        if not fits:
            continue
        d = np.array([f[0] - f[1] for f in fits])
        err = np.linalg.norm(d[:, :3], axis=1)
        n = len(d)
        bias = d.mean(0)
        se = d.std(0, ddof=1) / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, bias / se, np.nan)
        out[m] = {"n": n, "mean_3d_m": float(err.mean()), "median_3d_m": float(np.median(err)),
                  "converged_fraction": float(np.mean([f[2] for f in fits])),
                  "bias": {p: float(b) for p, b in zip(PAR, bias)},
                  "bias_se": {p: float(s) for p, s in zip(PAR, se)},
                  "bias_over_se": {p: (None if not np.isfinite(v) else float(v)) for p, v in zip(PAR, z)},
                  "max_abs_vertex_bias_over_se": float(np.nanmax(np.abs(z[:3])))}
    return out


def ridge_on_events(det, event_ids):
    out = {}
    for rid in RIDGE_RUNS[det]:
        step = "step_00000" if "__init__" in rid else "step_10000"
        p = T1 / "eval" / rid / step / "validation.npz"
        if not p.is_file():
            out[rid] = None
            continue
        z = np.load(p, allow_pickle=True)
        dom = np.asarray(z["domain"]).astype(str)
        ph_ids = np.asarray(z["event_ids"])[np.char.find(np.char.lower(dom), "photon") >= 0]
        pos = {int(e): k for k, e in enumerate(ph_ids)}
        idx = [pos.get(int(e)) for e in event_ids]
        keep = [k for k in idx if k is not None]
        row = {"n_matched": len(keep)}
        for key in ("err_ridge_spatial_moment_latent", "err_ridge_raw_spatial_moment", "err_attentive_mean3"):
            if key in z.files and keep and len(z[key]) == len(ph_ids):
                e = np.asarray(z[key])[keep]
                row[key.replace("err_", "")] = {"mean_m": float(e.mean()), "median_m": float(np.median(e))}
        out[rid] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"label_y1": LABEL_Y1, "detectors": {}}
    for det in ("sk", "wcte"):
        rows = load_rows(det, f"ystick_{det}_[0-9]*")
        if not rows:
            continue
        ev = np.load(INPUTS / f"yardstick_events_{det}.npz", allow_pickle=False)
        meta = json.loads(str(ev["meta"]))
        idx = sorted(rows)
        y1 = np.array([[rows[i]["y1"][k] for k in ("sigma_vertex_counts_m", "sigma_vertex_time_m",
                                                   "sigma_vertex_joint_m", "time_share")] for i in idx], float)
        vert = np.asarray(ev["vertex"])[idx]
        r = np.hypot(vert[:, 0], vert[:, 1])
        ylds = np.asarray(ev["yield"])[idx]
        y1d = {"n": len(idx)}
        for j, k in enumerate(("counts", "time", "joint", "time_share")):
            col = y1[:, j]
            y1d[k] = {"median": float(np.nanmedian(col)), "q16": float(np.nanquantile(col, 0.16)),
                      "q84": float(np.nanquantile(col, 0.84))}
        for name, var in (("radius", r), ("yield", ylds)):
            edges = np.quantile(var, [0, 1 / 3, 2 / 3, 1])
            tb = []
            for t in range(3):
                m = (var >= edges[t]) & ((var <= edges[t + 1]) if t == 2 else (var < edges[t + 1]))
                tb.append({"range": [float(edges[t]), float(edges[t + 1])], "n": int(m.sum()),
                           "median_joint_m": float(np.nanmedian(y1[m, 2])),
                           "median_counts_m": float(np.nanmedian(y1[m, 0])),
                           "median_time_m": float(np.nanmedian(y1[m, 1])),
                           "median_time_share": float(np.nanmedian(y1[m, 3]))})
            y1d[f"by_{name}_tertile"] = tb
        # author addendum (22 Nov): LUCiD lucid/fitting/fisher.py "honesty factor" sqrt(12) on the counts Fisher
        hon = []
        for i in idx:
            y = rows[i]["y1"]
            Fc, Ft = np.array(y["F_counts"], float), np.array(y["F_time"], float)
            row = [np.nan, np.nan]
            if y.get("sigma_vertex_counts_m"):
                row[0] = y["sigma_vertex_counts_m"] * np.sqrt(12.0)
            try:
                cov = np.linalg.inv(Fc / 12.0 + Ft)
                row[1] = float(np.sqrt(np.trace(cov[:3, :3])))
            except np.linalg.LinAlgError:
                pass
            hon.append(row)
        hon = np.array(hon, float)
        st = y1[:, 1]
        share_h = hon[:, 0] ** 2 / (hon[:, 0] ** 2 + st ** 2)
        y1d["honesty_sqrt12"] = {"counts_median": float(np.nanmedian(hon[:, 0])),
                                 "joint_median": float(np.nanmedian(hon[:, 1])),
                                 "time_share_median": float(np.nanmedian(share_h))}
        eff = {}
        for ch, j in (("counts", 0), ("time", 1), ("joint", 2)):
            r = [rows[i]["y2"][ch]["vertex_error_m"] / y1[k, j] for k, i in enumerate(idx)
                 if rows[i].get("y2") and y1[k, j] and np.isfinite(y1[k, j])]
            eff[ch] = {"n": len(r), "median_err_over_sigma": float(np.median(r)),
                       "q16": float(np.quantile(r, 0.16)), "q84": float(np.quantile(r, 0.84))}
        y1d["achieved_over_surrogate"] = eff
        mc = load_rows(det, f"ystick_mcnoise_{det}_*")
        mcd = None
        if mc:
            common = sorted(set(mc) & set(rows))
            mcd = {"config": mc[common[0]]["config"], "n": len(common), "channels": {}}
            for ch in ("counts", "time", "joint"):
                e_def = np.array([rows[i]["y2"][ch]["vertex_error_m"] for i in common])
                e_mc = np.array([mc[i]["y2"][ch]["vertex_error_m"] for i in common])
                mcd["channels"][ch] = {"default_mean_m": float(e_def.mean()), "more_rays_mean_m": float(e_mc.mean()),
                                       "default_median_m": float(np.median(e_def)),
                                       "more_rays_median_m": float(np.median(e_mc))}
        y2_rows = {i: rows[i] for i in idx if rows[i].get("y2")}
        y2_ids = np.asarray(ev["event_id"])[sorted(y2_rows)]
        diag = load_rows(det, f"ystick_diag_{det}_*")
        d = {"meta": meta, "config": rows[idx[0]]["config"], "Y1": y1d,
             "Y2_truth_start": y2_summary(y2_rows, "y2"),
             "Y2_centroid_start": y2_summary({i: rows[i] for i in idx if rows[i].get("y2_centroid")}, "y2_centroid"),
             "ridge_same_events": ridge_on_events(det, y2_ids),
             "mc_noise_check": mcd,
             "diagnostic_sigma_refit": ({"config": diag[min(diag)]["config"],
                                         "Y2_truth_start": y2_summary(diag, "y2")} if diag else None)}
        d["K4b"] = {"joint_truth_start_max_abs_vertex_bias_over_se":
                    d["Y2_truth_start"].get("joint", {}).get("max_abs_vertex_bias_over_se"),
                    "joint_truth_start_bias_over_se_all": d["Y2_truth_start"].get("joint", {}).get("bias_over_se")}
        res["detectors"][det] = d
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    f = lambda x, n=4: "–" if x is None else f"{x:.{n}f}"
    L = ["# YARDSTICKS — what LUCiD's own model says about the vertex (r2-tier1 T6, exploratory)", "",
         "Events: `dev_probe_test` photon bombs (Y1: 300 per detector; Y2: the first 200), θ = (x, y, z, t₀, ln N_γ). "
         "Model: LUCiD's expected (non-sampled) isotropic-source response (`setup_event_simulator(..., "
         "transport='expected', detection='expected')`) plus the dark expectation; time term = LUCiD's "
         "`lucid/losses.py` `first_arrival_window_nll` (σ = 2.1 ns, LUCiD's reconstruction working point). "
         "Code `src/jepa_noise/tier1_yardstick.py`.", "",
         f"**Y1 label:** {LABEL_Y1}.", ""]
    for det, d in res["detectors"].items():
        y1 = d["Y1"]
        L += [f"## {det.upper()}", "", "### Y1 — surrogate Fisher split (σ_vertex = √trace of the vertex block of F⁻¹)", "",
              "| Channel | median σ_vertex (m) | 16–84 % |", "|---|---:|---|"]
        for k in ("counts", "time", "joint"):
            L.append(f"| {k} | {f(y1[k]['median'])} | {f(y1[k]['q16'])}–{f(y1[k]['q84'])} |")
        L += [f"| time share σ_c²/(σ_c²+σ_t²) | {f(y1['time_share']['median'], 3)} | "
              f"{f(y1['time_share']['q16'], 3)}–{f(y1['time_share']['q84'], 3)} |", "",
              "| Tertile | radius range (m) | joint | counts | time | time share | yield range | joint | time share |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|"]
        for t in range(3):
            a_, b_ = y1["by_radius_tertile"][t], y1["by_yield_tertile"][t]
            L.append(f"| {t + 1} | {a_['range'][0]:.2f}–{a_['range'][1]:.2f} | {f(a_['median_joint_m'])} | "
                     f"{f(a_['median_counts_m'])} | {f(a_['median_time_m'])} | {f(a_['median_time_share'], 3)} | "
                     f"{b_['range'][0]:.0f}–{b_['range'][1]:.0f} | {f(b_['median_joint_m'])} | {f(b_['median_time_share'], 3)} |")
        h = y1["honesty_sqrt12"]
        L += ["", "With LUCiD's honesty factor √12 on the counts Fisher (`lucid/fitting/fisher.py`; joint = (F_counts/12 + F_time)⁻¹): "
              f"median σ_vertex counts {f(h['counts_median'])} m, joint {f(h['joint_median'])} m, time share {f(h['time_share_median'], 3)}.", "",
              "Achieved Y2 error (truth start) over the raw surrogate σ_vertex of the same event (an efficient unbiased fit gives a median ≈ 0.89; "
              "with the √12 factor applied it would be ≈ 3.1):", "",
              "| Channel | n | median err/σ | 16–84 % |", "|---|---:|---:|---|"]
        for ch, e in y1["achieved_over_surrogate"].items():
            L.append(f"| {ch} | {e['n']} | {e['median_err_over_sigma']:.2f} | {e['q16']:.2f}–{e['q84']:.2f} |")
        L += ["", "### Y2 — achieved resolution (minimise LUCiD's loss on the sampled events)", "",
              "| Start | Channel | n | mean 3-D (m) | median 3-D (m) | converged | bias x / y / z (mm) ± SE | z-scores x / y / z / t₀ / ln N |",
              "|---|---|---:|---:|---:|---:|---|---|"]
        for start, key in (("truth", "Y2_truth_start"), ("centroid", "Y2_centroid_start")):
            for m, s in d[key].items():
                b, se, zz = s["bias"], s["bias_se"], s["bias_over_se"]
                L.append(f"| {start} | {m} | {s['n']} | {f(s['mean_3d_m'])} | {f(s['median_3d_m'])} | "
                         f"{s['converged_fraction']:.3f} | " +
                         " / ".join(f"{1e3 * b[p]:+.1f}±{1e3 * se[p]:.1f}" for p in PAR[:3]) + " | " +
                         " / ".join("–" if zz[p] is None else f"{zz[p]:+.1f}" for p in PAR) + " |")
        L += ["", "Frozen readouts on the same 200 events (validation split):", "",
              "| Run | matched | Ridge spatial mean / median (m) | raw-feature mean (m) | attentive mean (m) |",
              "|---|---:|---|---:|---:|"]
        for rid, row in d["ridge_same_events"].items():
            if not row:
                L.append(f"| `{rid}` | – | pending | – | – |")
                continue
            rs = row.get("ridge_spatial_moment_latent", {})
            L.append(f"| `{rid}` | {row['n_matched']} | {f(rs.get('mean_m'))} / {f(rs.get('median_m'))} | "
                     f"{f(row.get('ridge_raw_spatial_moment', {}).get('mean_m'))} | "
                     f"{f(row.get('attentive_mean3', {}).get('mean_m'))} |")
        mcd = d.get("mc_noise_check")
        if mcd:
            L += ["", f"Monte-Carlo-noise check (author addendum): Y2 truth start on the first {mcd['n']} events with "
                      f"n_rays = {mcd['config']['n_rays']:,} (default 65,536), everything else unchanged:", "",
                  "| Channel | mean 3-D default (m) | mean 3-D more rays (m) | median default | median more rays |",
                  "|---|---:|---:|---:|---:|"]
            for ch, c in mcd["channels"].items():
                L.append(f"| {ch} | {f(c['default_mean_m'])} | {f(c['more_rays_mean_m'])} | {f(c['default_median_m'])} | {f(c['more_rays_median_m'])} |")
        dg = d["diagnostic_sigma_refit"]
        if dg:
            L += ["", f"Diagnostic (not a gate input): Y2 truth start re-fitted with the time-term σ = "
                      f"{dg['config']['sigma_tts_ns']} ns (the detector's 1-PE jitter):", "",
                  "| Channel | n | mean 3-D (m) | t₀ bias (ns) ± SE | z-scores x / y / z / t₀ / ln N |", "|---|---:|---:|---|---|"]
            for m, s in dg["Y2_truth_start"].items():
                zz = s["bias_over_se"]
                L.append(f"| {m} | {s['n']} | {f(s['mean_3d_m'])} | {s['bias']['t0']:+.3f}±{s['bias_se']['t0']:.3f} | " +
                         " / ".join("–" if zz[p] is None else f"{zz[p]:+.1f}" for p in PAR) + " |")
        L.append("")
    L += ["## Settings against LUCiD's published recipe (`docs/guides/reconstruction.md`)", "",
          "| Setting | LUCiD MODEL_RECIPE / FIT_RECIPE | This yardstick |", "|---|---|---|",
          "| rays per evaluation | 250,000 | 65,536 (250,000 in the MC-noise check) |",
          "| reflection/scatter iterations K | 8 | 12 (the `lucid-run-job` kernel setting used to generate the data) |",
          "| temperature | 0.1 | 0.1 |",
          "| first-arrival time σ (TTS) | 2.1 ns | 2.1 ns (diagnostic: detector 1-PE jitter) |",
          "| first-arrival window δ | 1.0 ns | 1.0 ns |",
          "| time weight | 1.0 | 1.0 (counts + time NLL summed) |",
          "| random keys averaged | 8, fixed through the fit | 4, fixed |",
          "| source model | `ReconModel` track (vec9: E, vertex, direction, t₀) | isotropic point source (x, y, z, t₀, ln N_γ) — photon bombs |",
          "| energy/intensity scale | `energy_scale='nphot'` | ln N_γ free (counts, joint); fixed at truth in time-only |",
          "| optimiser | Fisher–Gauss–Newton, 150 steps, lr 4.0→1.5, λ 0.01, μ 0.1, refresh 8, Polyak mean of last 40 | L-BFGS-B, ≤ 40 iterations, steps scaled by (0.2 m, 0.5 ns, 0.05) |",
          "| starts | data-built seeds (energy scan, first-light grid, multilateration, fused) | truth (primary); charge-weighted centroid (first 50) |",
          "| t₀ in counts-only fit | — | fixed at truth (counts carry no t₀ information) |",
          "| dark expectation | in the data model | 4.2 kHz × prompt window added to μ |", "",
          "The label stays **surrogate** throughout; none of these numbers is a Cramér–Rao bound of the data.", ""]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("wrote", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
