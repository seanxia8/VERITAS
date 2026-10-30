#!/usr/bin/env python3
"""Unregistered CPU-only M-dose experiment."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from jepa_noise import lucid_bombs as lb, point_encoder as pe, views
from jepa_noise.ext_predictability import resample_noise
from jepa_noise.ext_m_dose import f1x_state, m_terms, pair_diagnostics, project
from jepa_noise.lucid_views import ViewMasker, module_hilbert_order
from scripts.jepa_ext_predictability_linear import pool
from scripts.jepa_pregrid_checks import ridge_r2

ROOT = Path("/ceph/dwong/jepa/returns/ext_m_dose_20261030")
SEED = 20261017
LEVELS = (("F1", 0.), ("F1", .02), ("F1", .05), ("F1", .10),
          ("F1", .20), ("F1", .30), ("F1x", .50), ("F1x", .75),
          ("F1x", 1.), ("F1x", 1.5))


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def write_json(path, value):
    raw = (json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 1000000:
        raise ValueError("summary exceeds 1 MB")
    with path.open("xb") as f:
        f.write(raw)


def state(geo, law, spread, run, seed=SEED):
    if law == "F1":
        return lb.state_for_run("F1", spread, run, geo, seed=seed)
    return f1x_state(geo, spread, run, seed=seed)


def tokenise_hits(geometry, hits):
    # Supply an in-memory SparseHitStore to the registered tokeniser; raw events
    # never touch disk. Restore the original reader even if tokenisation fails.
    store = lb.sparse_from_readouts(np.arange(len(hits), dtype=np.int64), hits)
    original = pe.read_sparse_hits
    pe.read_sparse_hits = lambda path: store
    try:
        batch = pe.tokenise("in_memory", geometry, split="within",
                            masker=ViewMasker(geometry, "within", n_min=0))
    finally:
        pe.read_sparse_hits = original
    if len(batch.event_ids) != len(hits):
        raise RuntimeError("tokeniser unexpectedly excluded event")
    return batch.features.numpy(), batch.target_index[0].numpy()


def groups(geo):
    order = module_hilbert_order(geo)
    module = geo.module_of_pmt
    parity = geo.pmt_id_in_mpmt % 2
    return {
        "within": ([np.flatnonzero((module == m) & (parity == 0)) for m in range(geo.n_modules)],
                   [np.flatnonzero((module == m) & (parity == 1)) for m in range(geo.n_modules)]),
        "across_fixed": ([np.flatnonzero(module == m) for m in order[:len(order)//2]],
                         [np.flatnonzero(module == m) for m in order[len(order)//2:]])}


def pooled(features, group):
    return np.concatenate([pool(features, group[0]), pool(features, group[1])], axis=1).astype(np.float64)


def toy():
    rows = []
    for g in (0., .005, .01, .02, .05, .1, .225, .45, .9):
        fixture = views.Tier1Views(split="channel", line_multiplier=0., common_strength=g)
        p = fixture.population()
        blocks = tuple(p[n] for n in ("scc","stt","sct","ncc","ntt","nct"))
        r = pair_diagnostics(*blocks, 3)
        r["M_reg"] = m_terms(*project(blocks, "channel"), 3)[0]
        r["D"] = fixture.displacement(3)["angle_deg"]
        r.update(testbed="toy", law="toy", level=g, split="channel")
        r.pop("context"); r.pop("target")
        rows.append(jsonable(r))
    return rows


def collect(law, spread, geometry, cfg):
    readout = lb.make_readout(geometry, cfg)
    g = groups(geometry)
    sums = {name: {"sum": np.zeros(630 if name == "within" else 315),
                    "outer": np.zeros((630 if name == "within" else 315,)*2),
                    "A": np.zeros((630 if name == "within" else 315,)*2),
                    "B": np.zeros((630 if name == "within" else 315,)*2),
                    "null": np.zeros((630 if name == "within" else 315,)*2),
                    "means": []} for name in g}
    gains, clips, below = [], 0, 0
    for run in range(1000):
        rid = f"ext_m_dose/evt/{run:04d}"
        st = state(geometry, law, spread, rid)
        gains.append(np.log(st.module_gain))
        clips += st.clipped["module"]
        hits = []
        for e in range(10):
            draw = lb.draw_event(geometry, cfg, rid, run*10+e, seed=SEED, domain="photon")
            hits.append(readout(st, draw))
        feat, target = tokenise_hits(geometry, hits)
        below += int(np.sum((feat[:, target, 8] > 0).sum(axis=1) < lb.N_MIN))
        for name in g:
            x = pooled(feat, g[name])
            sums[name]["sum"] += x.sum(axis=0)
            sums[name]["outer"] += x.T @ x
            sums[name]["means"].append(x.mean(axis=0))
    prev = {}; first = {}
    for event in range(1000):
        rid = f"ext_m_dose/cal/{event:04d}"
        st = state(geometry, law, spread, rid)
        draw = lb.draw_event(geometry, cfg, rid, event, seed=SEED, domain="photon")
        for scheme in ("A", "B"):
            hits = []
            for r in range(8):
                sample = resample_noise(draw, geometry, cfg, r if scheme == "A" else r+1, seed=SEED)
                if scheme == "A":
                    sr = st
                else:
                    sd = int(lb.keyed_rng(SEED, "ext_predictability_state", rid, event, r).integers(0, 2**64, dtype=np.uint64))
                    sr = state(geometry, law, spread, rid, seed=sd)
                hits.append(readout(sr, sample))
            feat, _ = tokenise_hits(geometry, hits)
            for name in g:
                x = pooled(feat, g[name])
                residual = (x-x.mean(axis=0))*np.sqrt(8/7)
                sums[name][scheme] += residual.T @ residual / 8
                if scheme == "B":
                    dc = len(g[name][0])*3
                    if event == 0:
                        first[name] = residual.copy()
                    if event:
                        sums[name]["null"][:dc,dc:] += prev[name][:,:dc].T @ residual[:,dc:] / 8
                    prev[name] = residual.copy()
        if event % 50 == 0:
            print(f"cal {law} {spread} {event}/1000", flush=True)
    arrays = {"gain": np.asarray(gains), "clipped_fraction": np.asarray(clips/(1000*geometry.n_modules)),
              "below_n_min": np.asarray(below)}
    for name in g:
        v = sums[name]; dc = len(g[name][0])*3
        v["null"][:dc,dc:] += prev[name][:,:dc].T @ first[name][:,dc:] / 8
        arrays[name+"_S"] = (v["outer"] - np.outer(v["sum"],v["sum"])/10000)/9999
        arrays[name+"_A"] = v["A"]/1000
        arrays[name+"_B"] = v["B"]/1000
        arrays[name+"_null"] = v["null"]/1000
        arrays[name+"_means"] = np.stack(v["means"])
    return arrays


def analyse(data, law, spread, geometry):
    rows = []
    for name in ("within", "across_fixed"):
        S, A, B, Z = (data[name+"_"+x] for x in ("S","A","B","null"))
        dc = 315 if name == "within" else 156
        def blocks(N):
            return S[:dc,:dc], S[dc:,dc:], S[:dc,dc:], N[:dc,:dc], N[dc:,dc:], N[:dc,dc:]
        b = blocks(B)
        r = pair_diagnostics(*b, 4, A[:dc,dc:], (B-A)[:dc,dc:], Z[:dc,dc:])
        reg = m_terms(*project(b, name), 4)[0]
        nul = m_terms(b[0],b[1],b[2],b[3],b[4],Z[:dc,dc:],4)[2][0]
        context = r.pop("context"); r.pop("target")
        gain = data["gain"]; mean = data[name+"_means"][:,:dc] @ context
        r2 = [ridge_r2(gain[:500], mean[:500,j], gain[500:], mean[500:,j])["r2"] for j in range(4)]
        r.update(testbed="LUCiD", law=law if law=="F1" else "F1x_stress_unphysical",
                 level=spread, split=name, M_reg=reg, sigma_noise_null=float(nul),
                 ratio_null=float(nul/r["sigma_phys"]), at_floor=bool(r["sigma_noise"] <= 1.2*nul),
                 r2_gain=r2, clipped_fraction=float(data["clipped_fraction"]),
                 below_n_min=int(data["below_n_min"]))
        rows.append(jsonable(r))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode", choices=("toy","level"))
    ap.add_argument("--law", choices=("F1","F1x"))
    ap.add_argument("--spread", type=float)
    ap.add_argument("--geometry", default="/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json")
    ap.add_argument("--config", default="src/jepa_noise/configs/lucid_bombs_v2_calibrated.json")
    args=ap.parse_args()
    if not ROOT.is_dir(): raise ValueError("extension root must exist")
    if args.mode == "toy":
        write_json(ROOT/"toy.json", toy())
        return
    if (args.law,args.spread) not in LEVELS: raise ValueError("unknown level")
    digest=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    provenance=json.loads(Path("results/jepa/20261025_pregrid_checks/provenance.json").read_text())
    if digest != provenance["generator_config"]["sha256"]: raise ValueError("config hash mismatch")
    geometry=lb.load_wcte_geometry(args.geometry)
    cfg=lb.load_generator_config(args.config)
    start=time.monotonic()
    data=collect(args.law,args.spread,geometry,cfg)
    stem=f"{args.law}_{args.spread:.2f}"
    path=ROOT/(stem+".npz")
    with path.open("xb") as f: np.savez_compressed(f,**data)
    rows=analyse(data,args.law,args.spread,geometry)
    write_json(ROOT/(stem+".json"), {"registered":False, "extension":"m_dose_2026-10-30",
               "rows":rows, "config_sha256":digest, "npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
               "wall_seconds":time.monotonic()-start})


if __name__=="__main__":
    main()
