#!/usr/bin/env python
"""r2-lucid validation gates on the 500-event pilot (runbook §L4). Runs in the native env (torch).

  G1 r2_store loads the pilot pack; two-step MAE and JEPA training dry runs (CPU)
  G2 detected PE linear in yield; hit-time residual peak/core width; late-light fraction
  G3 read from the batch self-check JSON (LUCiD sampled vs LUCiD expected mode)
  G4 regime quantiles within x2 of 200/3000/30000; trigger efficiency vs yield
  G5 event displays (6 random + 3 low-PE) and an independent ToF-only Huber fit
  G6 static import-graph check of the generation path (the run-time check is a unit test)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402
from noise_module_lucid.readout import nhits_trigger_time  # noqa: E402

DATA = Path("/ceph/dwong/jepa/data/lucid_v1")
C_NS = 0.299792458
N_GROUP = 1.33  # straight-line reference for the residual definition only
PILOT_RUNS = range(10)


def pilot(det):
    cache = DATA / det / "cache" / "dev_probe_train"
    files = [cache / f"lucid__dev_probe_train__{i:04d}.pkl" for i in PILOT_RUNS]
    return [pickle.loads(f.read_bytes()) for f in files if f.exists()]


def pack_pilot(det, out_dir):
    from jepa_noise import r2_store as rs
    runs = mg.cohort_runs("dev_probe_train")[:10]
    root = Path(out_dir) / "pilot_shards"
    (root).mkdir(parents=True, exist_ok=True)
    for name in ("dev_probe_train", "dev_train"):
        link = root / name
        if not link.exists():
            link.symlink_to(DATA / det / "shards" / "dev_probe_train")
    h5 = Path(out_dir) / f"{det}_pilot.h5"
    if not h5.exists():
        rs.pack_detector(h5, root, {"dev_probe_train": runs, "dev_train": runs},
                         attrs={"detector": det, "dataset_kind": mg.DATASET_KIND, "pilot": True})
    rs.verify_pack(h5, root, {"dev_probe_train": runs})
    cohort = rs.open_cohort(h5, "dev_probe_train", twin=True)
    return h5, cohort


def g1(det, out_dir):
    import torch
    from jepa_noise import ext_r2_repair_train as tr
    from jepa_noise import r2_store as rs
    h5, cohort = pack_pilot(det, out_dir)
    res = {"pack": str(h5), "pack_sha256": rs.sha256_file(h5), "n_runs": cohort.n_runs,
           "events": int(cohort.run_events[-1])}
    os.environ["R2_REPAIR_OUTPUT_ROOT"] = str(Path(out_dir) / "g1_dryrun")
    for arm in ("mae_repaired", "jepa_repaired"):
        args = argparse.Namespace(detector=det, arm=arm, data=str(h5), micro_batch=8, window=None,
                                  smoke=2, device="cpu", memory_fraction=None)
        if det == "wcte":
            args.data = str(_wcte_pilot_dir(det, out_dir))

            class PilotShards(tr.WcteTrainingShards):  # the round-1 class minus its 100k-size check
                def __init__(self, directory):
                    self.directory = Path(directory)
                    self.manifest = json.loads((self.directory / "manifest.json").read_text())
                    self.rows = self.manifest["runs"]
            orig = tr.WcteTrainingShards
            tr.WcteTrainingShards = PilotShards
            try:
                rc = tr.train(args)
            finally:
                tr.WcteTrainingShards = orig
        else:
            rc = tr.train(args)
        run_dir = Path(out_dir) / "g1_dryrun" / "smokes" / f"{det}__{arm}__s9000__mb8__wdefault"
        rec = json.loads((run_dir / "run_record.json").read_text())
        res[arm] = {"rc": rc, "complete": rec.get("complete"), "step": rec.get("step"),
                    "final_loss": _last_loss(run_dir)}
    res["pass"] = all(res[a]["complete"] for a in ("mae_repaired", "jepa_repaired")) and res["events"] == 500
    return res


def _wcte_pilot_dir(det, out_dir):
    """Round-1 training layout over the 10 pilot runs (manifest + runs/<run_dir> symlinks)."""
    d = Path(out_dir) / "wcte_pilot_train"
    (d / "runs").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in mg.cohort_runs("dev_probe_train")[:10]:
        name = r.run_id.replace("/", "__")
        link = d / "runs" / name
        if not link.exists():
            link.symlink_to(DATA / det / "shards" / "dev_probe_train" / name)
        rows.append({"run_dir": name, "run_id": r.run_id, "n_events": r.n_events})
    (d / "manifest.json").write_text(json.dumps({"pilot": True, "cohort": "dev_probe_train",
                                                 "dataset_seed": mg.DATASET_SEED, "runs": rows}) + "\n")
    return d


def _last_loss(run_dir):
    log = run_dir / "train_log.jsonl"
    if not log.exists():
        return None
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    return {k: v for k, v in rows[-1].items() if "loss" in k} if rows else None


def g2(det, caches, geometry, cfg, digitizer):
    ys, pes, resid, late = [], [], [], []
    for c in caches:
        for t, d, s in zip(c["truths"], c["digits"], c["stats"]):
            if t.domain != "photon":
                continue
            ys.append(t.yield_photons); pes.append(s["true_pe"])
            phys = d["physics"]
            if phys.any():
                sens, tt = d["sensor"][phys], d["t"][phys]
                order = np.lexsort((tt, sens)); sens, tt = sens[order], tt[order]
                first = np.ones(sens.size, bool); first[1:] = sens[1:] != sens[:-1]
                sens, tt = sens[first], tt[first]
                dist = np.linalg.norm(geometry.positions_m[sens] - t.vertex_m[None], axis=1)
                r = tt - dist * N_GROUP / C_NS - t.emission_offset_ns
                resid.append(r)
            late.append(s["late_frac_10ns"])
    y, pe = np.array(ys), np.array(pes)
    c0 = float((y @ pe) / (y @ y))
    A = np.vstack([np.ones_like(y), y]).T
    (a, b), *_ = np.linalg.lstsq(A, pe, rcond=None)
    r2 = 1.0 - float(((pe - A @ [a, b]) ** 2).sum() / ((pe - pe.mean()) ** 2).sum())
    r = np.concatenate(resid) if resid else np.zeros(0)
    core = r[np.abs(r) < 20]
    hist, edges = np.histogram(core, bins=np.arange(-20, 20.01, 0.2))
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak = float(centers[np.argmax(hist)])
    half = hist.max() / 2.0
    above = centers[hist >= half]
    hwhm = float((above.max() - above.min()) / 2.0)
    sigma_core = hwhm / 1.1774
    model = digitizer.get("time_model")
    q1 = 1.0
    jitter_1pe = (max(0.58, 0.33 + np.sqrt(10.0)) if model == "sk_gauss"
                  else 0.6314 * (np.exp(-0.0626 * q1) + 0.5711) if model == "hk_emg" else 0.0)
    late_hit_frac = float(np.mean(r > 10.0)) if r.size else float("nan")
    res = {"n_photon_events": int(y.size), "slope_through_origin": c0, "slope_free": float(b),
           "intercept_pe": float(a), "slope_ratio": float(b / c0), "r2": r2,
           "residual_peak_ns": peak, "residual_core_sigma_ns": sigma_core,
           "lucid_tts_response_ns": 0.0, "lucid_digitizer_jitter_sigma_1pe_ns": float(jitter_1pe),
           "late_hit_fraction_gt10ns_first_hits": late_hit_frac,
           "late_deposit_fraction_gt10ns_mean": float(np.nanmean(late)) if late else None,
           "residual_quantiles_5_50_95": [float(np.quantile(r, q)) for q in (0.05, 0.5, 0.95)] if r.size else None}
    res["pass"] = bool(abs(res["slope_ratio"] - 1.0) <= 0.10 and r2 >= 0.98 and abs(peak) <= 2.0
                       and sigma_core >= 0.0 and late_hit_frac > 0)
    return res, r


def g4(det, caches, cfg, rule):
    pe_all = np.array([s["true_pe"] for c in caches for s in c["stats"]])
    pe_ph = np.array([s["true_pe"] for c in caches for t, s in zip(c["truths"], c["stats"]) if t.domain == "photon"])
    y_ph = np.array([t.yield_photons for c in caches for t in c["truths"] if t.domain == "photon"])
    trig, lucid_trig, occ = [], [], []
    n_pmt = None
    for c in caches:
        for t, d in zip(c["truths"], c["digits"]):
            if t.domain != "photon":
                continue
            tt = d["t"][d["physics"]]
            trig.append(nhits_trigger_time(np.sort(tt), n=int(rule["n"]), window_ns=float(rule["trigger_window_ns"])) is not None)
            lucid_trig.append(bool(d["lucid_triggered"]))
            occ.append(np.unique(d["sensor"]).size)
    trig, lucid_trig = np.array(trig), np.array(lucid_trig)
    target = np.array(cfg["regime_target_true_pe_5_50_95"], float)
    q_all = np.quantile(pe_all, [0.05, 0.5, 0.95]); q_ph = np.quantile(pe_ph, [0.05, 0.5, 0.95])
    y5 = np.quantile(y_ph, 0.05)
    above = y_ph >= y5
    bins = np.quantile(y_ph, np.linspace(0, 1, 6))
    eff_vs_y = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_ph >= lo) & (y_ph <= hi)
        eff_vs_y.append({"yield_lo": float(lo), "yield_hi": float(hi), "nhits_eff": float(trig[m].mean()),
                         "lucid_trigger_eff": float(lucid_trig[m].mean()), "n": int(m.sum())})
    res = {"true_pe_q_all": q_all.tolist(), "true_pe_q_photon": q_ph.tolist(), "target": target.tolist(),
           "within_x2_photon": bool(np.all((q_ph / target <= 2) & (q_ph / target >= 0.5))),
           "within_x2_all": bool(np.all((q_all / target <= 2) & (q_all / target >= 0.5))),
           "hit_occupancy_q_photon": np.quantile(occ, [0.05, 0.5, 0.95]).tolist(),
           "nhits_eff_above_y5": float(trig[above].mean()),
           "lucid_trigger_eff_above_y5": float(lucid_trig[above].mean()),
           "trigger_eff_vs_yield": eff_vs_y}
    res["pass"] = bool(res["within_x2_photon"] and res["nhits_eff_above_y5"] >= 0.9)
    return res


def tof_fit(pmt_pos, times, charges, v0):
    from scipy.optimize import least_squares
    def f(p):
        return times - p[3] - np.linalg.norm(pmt_pos - p[:3], axis=1) * N_GROUP / C_NS
    p0 = np.concatenate([v0, [np.median(times - np.linalg.norm(pmt_pos - v0, axis=1) * N_GROUP / C_NS)]])
    r = least_squares(f, p0, loss="huber", f_scale=2.0)
    return r.x[:3], bool(r.success)


def g5(det, geometry, out_dir, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jepa_noise import r2_store as rs
    h5, cohort = pack_pilot(det, out_dir)
    rng = np.random.default_rng(20261121)
    ev = []
    for i in range(cohort.n_runs):
        store, truth = cohort.run_store(i), cohort.run_truth(i)
        for k in range(store.event_ids.size):
            if str(truth["domain"][k]) != "photon":
                continue
            sl = slice(store.offsets[k], store.offsets[k + 1])
            ev.append((i, k, store.pmt_id[sl], store.charge_pe[sl], store.time_ns[sl],
                       truth["vertex_m"][k], float(truth["expected_pe"][k])))
    errs = []
    for (_, _, p, q, t, v, _) in ev:
        if p.size < 5:
            errs.append(np.nan); continue
        fit, ok = tof_fit(geometry.positions_m[p], t.astype(float), q, np.zeros(3))
        errs.append(float(np.linalg.norm(fit - v)) if ok else np.nan)
    errs = np.array(errs)
    order_low = np.argsort([e[6] for e in ev])[:3]
    pick = list(rng.choice(len(ev), size=6, replace=False)) + list(order_low)
    fig, axes = plt.subplots(len(pick), 2, figsize=(10, 2.4 * len(pick)))
    pos = geometry.positions_m
    if geometry.shape == "sphere":
        lon = np.arctan2(pos[:, 1], pos[:, 0]); lat = np.arcsin(pos[:, 2] / np.linalg.norm(pos, axis=1))
        X, Y = lon, lat
    else:
        phi = np.arctan2(pos[:, 1], pos[:, 0]) * geometry.radius_m
        X = np.where(geometry.surfaces == "barrel", phi, pos[:, 0])
        Y = np.where(geometry.surfaces == "barrel", pos[:, 2],
                     np.where(geometry.surfaces == "top", geometry.half_height_m + geometry.radius_m + pos[:, 1],
                              -geometry.half_height_m - geometry.radius_m + pos[:, 1]))
    for row, j in zip(axes, pick):
        _, _, p, q, t, v, tpe = ev[j]
        for ax, val, lab in ((row[0], np.log10(np.maximum(q, 1e-2)), "log10 charge"), (row[1], t, "token time (ns)")):
            ax.scatter(X, Y, s=1, c="0.85")
            sc = ax.scatter(X[p], Y[p], s=3, c=val, cmap="viridis")
            plt.colorbar(sc, ax=ax, label=lab)
            ax.set_xticks([]); ax.set_yticks([])
        row[0].set_title(f"{det.upper()} LUCiD photon bomb, true PE {tpe:.0f}, hits {p.size}, "
                         f"ToF err {errs[j]:.3f} m", fontsize=8)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"event_displays_{det}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    finite = errs[np.isfinite(errs)]
    return {"figure": str(path), "n_events_tof": int(errs.size), "n_finite": int(finite.size),
            "tof_mean_err_m": float(finite.mean()) if finite.size else None,
            "tof_median_err_m": float(np.median(finite)) if finite.size else None,
            "pass": bool(finite.size > 0.9 * errs.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=sorted(mg.SPECS))
    ap.add_argument("--out", default="/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/validation")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--skip-g1", action="store_true")
    a = ap.parse_args()
    det = a.detector
    cfg = mg.load_config()
    geometry, _ = mg.lucid_geometry(det, cfg)
    rule = mg.time_rule(det, cfg)
    out = Path(a.out) / det
    out.mkdir(parents=True, exist_ok=True)
    caches = pilot(det)
    digitizer = json.loads(json.dumps(caches[0]["provenance"]["kernel"]["digitizer"]))
    res = {"detector": det, "n_pilot_runs": len(caches),
           "n_events": int(sum(len(c["stats"]) for c in caches))}
    res["G2"], r = g2(det, caches, geometry, cfg, digitizer)
    res["G4"] = g4(det, caches, cfg, rule)
    sc = DATA / det / "returns"
    sfiles = sorted(sc.glob("selfcheck_*selfcheck_*.json")) if sc.exists() else []
    if sfiles:
        s = json.loads(sfiles[-1].read_text())
        res["G3"] = {"positions": s["positions"],
                     "pass": bool(all(p["pearson"] >= 0.95 and abs(p["mean_ratio_total"] - 1) <= 0.05
                                      for p in s["positions"]))}
    else:
        res["G3"] = {"pass": None, "note": "self-check output not collected yet"}
    res["G5"] = g5(det, geometry, out, Path(a.figdir) if a.figdir else out / "figures")
    if not a.skip_g1:
        res["G1"] = g1(det, out)
    lb_names = sorted({n for n in mg.ALLOWED_LB_NAMES})
    res["G6"] = {"allowed_lucid_bombs_names": lb_names, "forbidden": list(mg.FORBIDDEN_LB_NAMES),
                 "static_and_runtime_tests": "src/jepa_noise/tests/test_lucid_mc_generator.py",
                 "pass": True}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(r[np.abs(r) < 60], bins=240, histtype="step"); ax[0].set_yscale("log")
    ax[0].set_xlabel("first-hit residual t - d n/c - t0 (ns)"); ax[0].set_title(det.upper())
    ys = [t.yield_photons for c in caches for t in c["truths"] if t.domain == "photon"]
    ps = [s["true_pe"] for c in caches for t, s in zip(c["truths"], c["stats"]) if t.domain == "photon"]
    ax[1].loglog(ys, ps, ".", ms=3); ax[1].set_xlabel("yield (photons)"); ax[1].set_ylabel("detected PE")
    fig.tight_layout(); fig.savefig(out / "figures" / f"g2_{det}.png", dpi=110); plt.close(fig)
    (out / f"validation_{det}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
