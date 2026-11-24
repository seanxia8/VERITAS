#!/usr/bin/env python3
"""r2-sparse T7.3: SPARSE_TABLE.{json,md} from the evaluator outputs (PREREG_2026-11-24_R2_SPARSE.md).

Contrasts R1-R6 (T = e_test / e_base - 1, seed-paired, sealed photon bombs, frozen spatial-moment Ridge),
per-seed values, SE_boot (paired bootstrap of the seed-mean contrast over sealed events, 2,000 resamples,
the r2-tier1 resampling), SE_seed = SD_pooled / sqrt(n_seeds) (SD_pooled = RMS over R1-R6 of the across-seed
SD, per metric), N_sigma = |T| / max(SE_boot, SE_seed), clear = N_sigma >= 2 and one sign in every seed.
Secondary outcomes, matched-init gains, raw-feature baselines, prediction verdicts (§2.5) and the decision
rule (§2.6). Reads only evaluator outputs (no pack is opened here).

    sparse_table.py --eval-root /ceph/dwong/jepa/returns/r2_sparse_2026-11-24/eval --out-dir results/round2/20261124_sparse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from tier1_gate import boot_idx, boot, point  # noqa: E402  (the r2-tier1 statistics, unchanged)

SEEDS = {"A": 9000, "B": 9001, "C": 9002}
CELLS = {
    "C0": "wcte__mae_repaired__s{s}",
    "C1": "wcte__mae_repaired__s{s}__dw2",
    "C2": "wcte__mae_repaired__s{s}__dw2__lfixed__psent",
    "C3": "wcte__mae_repaired__s{s}__dw2__lfixed__psent__efourier__disol__cnt",
    "S1": "wcte__sparse_mae__s{s}__dw2__qall",
    "S1n": "wcte__sparse_mae__s{s}__dw2__qneg",
    "S1p": "wcte__sparse_mae__s{s}__dw2__qpos",
    "S2": "wcte__sparse_time__s{s}__dw2",
}
REGISTERED = {"R1": ("C0", "C1"), "R2": ("C1", "C2"), "R2b": ("C2", "C3"), "R3": ("C3", "S1"),
              "R4": ("S1", "S1n"), "R5": ("S1", "S1p"), "R6": ("C3", "S2")}
EXTRA = {"S1n_vs_C3": ("C3", "S1n"), "S1p_vs_C3": ("C3", "S1p"), "C3_vs_C1": ("C1", "C3"),
         "S1_vs_C1": ("C1", "S1"), "S2_vs_S1": ("S1", "S2")}
METRICS = {
    "vertex": ("err_ridge_spatial_moment_latent", "mean", "photon"),
    "vertex_mlp": ("err_mlp_spatial_moment_full_mean3", "mean", "photon"),
    "vertex_attentive": ("err_attentive_mean3", "mean", "photon"),
    "dir_mu": ("ang_muon_ridge_spatial", "median", "muon"),
    "dir_e": ("ang_electron_ridge_spatial", "median", "electron"),
    "misid_mu": ("pid", "misid", "pid"),
    "vertex_allhits_exploratory": ("err_ridge_spatial_moment_latent_allhits", "mean", "photon"),
    "vertex_raw_baseline": ("err_ridge_raw_spatial_moment", "mean", "photon"),
}
COHORT_TAG = {"sealed": "sealed_test", "validation": "dev_probe_test"}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def get(self, rid, step, cohort):
        key = (rid, step, cohort)
        if key not in self.cache:
            d = self.root / rid / f"step_{step:05d}"
            if not (d / f"{cohort}.json").is_file():
                self.cache[key] = None
            else:
                js = json.loads((d / f"{cohort}.json").read_text())
                with np.load(d / f"{cohort}.npz") as z:
                    arr = {k: z[k] for k in z.files if not k.startswith("boot_")}
                self.cache[key] = (js, arr)
        return self.cache[key]

    def values(self, rid, step, cohort, metric):
        got = self.get(rid, step, cohort)
        if got is None:
            return None
        js, arr = got
        key, kind, tag = METRICS[metric]
        if kind == "misid":
            if "pid_p_mu" not in arr:
                return None
            return {"kind": kind, "p": arr["pid_p_mu"], "is_mu": arr["pid_is_mu"].astype(bool), "tag": tag,
                    "ids": arr["event_ids"][np.isin(arr["domain"], ("muon", "electron"))]}
        if key not in arr:
            return None
        dom = {"photon": "photon", "muon": "muon", "electron": "electron"}[tag]
        return {"kind": kind, "x": arr[key], "tag": tag, "ids": arr["event_ids"][arr["domain"] == dom]}


def contrast(store, base, test, metric, cohort, step=10000):
    per_seed, boots, missing = {}, [], []
    for label, s in SEEDS.items():
        r1, r2 = CELLS[base].format(s=s), CELLS[test].format(s=s)
        v1, v2 = store.values(r1, step, cohort, metric), store.values(r2, step, cohort, metric)
        if v1 is None or v2 is None:
            missing.append(r1 if v1 is None else r2)
            continue
        if not np.array_equal(v1["ids"], v2["ids"]):
            raise ValueError(f"events not aligned: {r1} vs {r2} ({metric}, {cohort})")
        n = len(v1["p"] if v1["kind"] == "misid" else v1["x"])
        idx = boot_idx(n, f"{COHORT_TAG[cohort]}/{v1['tag']}")
        p1, p2 = point(v1), point(v2)
        if p1 == 0.0:
            continue
        per_seed[label] = p2 / p1 - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = np.asarray(boot(v1, idx), float)
            boots.append(np.asarray(boot(v2, idx), float) / np.where(b1 == 0, np.nan, b1) - 1.0)
    out = {"base": base, "test": test, "metric": metric, "cohort": cohort, "per_seed": per_seed,
           "missing": sorted(set(missing))}
    if per_seed:
        v = np.array(list(per_seed.values()))
        out.update(T_bar=float(v.mean()), seed_sd=float(v.std(ddof=1)) if v.size > 1 else None,
                   SE_boot=float(np.nanstd(np.nanmean(boots, axis=0), ddof=1)),
                   same_sign=bool(np.all(v < 0) or np.all(v > 0)), n_seeds=int(v.size))
    return out


def finish(rows, registered_ids):
    pooled = {}
    for (cid, metric, cohort), c in rows.items():
        if cid in registered_ids and c.get("seed_sd") is not None:
            pooled.setdefault((metric, cohort), []).append(c["seed_sd"])
    pooled = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in pooled.items()}
    for (cid, metric, cohort), c in rows.items():
        if "T_bar" not in c:
            c.update(N_sigma=None, clear=False)
            continue
        sd = pooled.get((metric, cohort))
        c["SD_pooled"] = sd
        c["SE_seed"] = sd / np.sqrt(c["n_seeds"]) if sd is not None else None
        den = max(c["SE_boot"], c["SE_seed"] or 0.0)
        c["N_sigma"] = float(abs(c["T_bar"]) / den) if den > 0 else None
        c["clear"] = bool(c["N_sigma"] is not None and c["N_sigma"] >= 2 and c["same_sign"] and c["n_seeds"] == 3)
    return {f"{m}/{c}": v for (m, c), v in pooled.items()}


def verdict(cid, c):
    if c is None or "T_bar" not in c:
        return "not evaluable"
    t, clear = c["T_bar"], c["clear"]
    if cid in ("R2", "R2b", "R3"):
        return "holds" if t <= 0.02 else ("fails" if clear else "inconclusive")
    if cid == "R4":
        return "holds" if abs(t) <= 0.03 else ("fails" if clear else "inconclusive")
    if cid == "R5":
        return "holds" if (t > 0 and clear) else ("fails" if (t < 0 and clear) else "inconclusive")
    return "no prediction (reported)"


def seed_mean(store, cell, metric="vertex", cohort="sealed", step=10000):
    vals = []
    for s in SEEDS.values():
        v = store.values(CELLS[cell].format(s=s), step, cohort, metric)
        if v is not None:
            vals.append(point(v))
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def build(args):
    store = Store(args.eval_root)
    rows = {}
    for cohort in ("sealed", "validation"):
        for cid, (b, t) in {**REGISTERED, **EXTRA}.items():
            for metric in METRICS:
                if metric == "vertex_raw_baseline":
                    continue
                rows[(cid, metric, cohort)] = contrast(store, b, t, metric, cohort)
    pooled = finish(rows, set(REGISTERED))
    S = lambda cid, metric="vertex", cohort="sealed": rows.get((cid, metric, cohort))
    out = {"preregistration": "docs/round2/PREREG_2026-11-24_R2_SPARSE.md", "eval_root": str(args.eval_root),
           "SD_pooled": pooled, "contrasts": {}, "cells": {}, "matched_init_gain": {}, "raw_baseline": {}}
    for (cid, metric, cohort), c in rows.items():
        out["contrasts"].setdefault(cid, {}).setdefault(cohort, {})[metric] = c
    for cid in REGISTERED:
        out["contrasts"][cid]["prediction"] = verdict(cid, S(cid))
        out["contrasts"][cid]["registered"] = True
    for cell in CELLS:
        out["cells"][cell] = {}
        for cohort in ("sealed", "validation"):
            out["cells"][cell][cohort] = {m: dict(zip(("seed_mean", "n_seeds"), seed_mean(store, cell, m, cohort)))
                                          for m in METRICS if m != "vertex_raw_baseline"}
            per = {}
            for lab, s in SEEDS.items():
                v = store.values(CELLS[cell].format(s=s), 10000, cohort, "vertex")
                per[lab] = point(v) if v is not None else None
            out["cells"][cell][cohort]["vertex_per_seed"] = per
        # matched init: e(10000) / e(step 0) - 1 per seed (sealed, primary readout)
        g = {}
        for lab, s in SEEDS.items():
            rid = CELLS[cell].format(s=s)
            v0, v1 = store.values(rid, 0, "sealed", "vertex"), store.values(rid, 10000, "sealed", "vertex")
            if v0 is not None and v1 is not None:
                g[lab] = {"init_m": point(v0), "final_m": point(v1), "gain": point(v1) / point(v0) - 1.0}
        out["matched_init_gain"][cell] = g
    for kind, cell in (("lucid_v1", "C0"), ("lucid_v1_w2", "C1")):
        for s in SEEDS.values():
            v = store.values(CELLS[cell].format(s=s), 10000, "sealed", "vertex_raw_baseline")
            if v is not None:
                out["raw_baseline"][kind] = {"vertex_mean_m": point(v), "from_run": CELLS[cell].format(s=s)}
                break
    # decision rule (§2.6)
    dense = {c: seed_mean(store, c)[0] for c in ("C1", "C2", "C3")}
    best_dense = min((v for v in dense.values() if v is not None), default=None)
    scale = {}
    for cell, cid in (("S1", "R3"), ("S1n", "S1n_vs_C3"), ("S2", "R6")):
        crit = {}
        for metric in ("vertex", "dir_mu", "dir_e"):
            c = S(cid, metric)
            crit[metric] = bool(c and "T_bar" in c and c["T_bar"] <= -0.05 and c["clear"])
        probes_same = all(S(cid, m) and "T_bar" in S(cid, m) and S(cid, m)["T_bar"] < 0
                          for m in ("vertex_mlp", "vertex_attentive"))
        sm = seed_mean(store, cell)[0]
        not_worse = bool(sm is not None and best_dense is not None and sm <= best_dense)
        scale[cell] = {"better_by_5pct_clear": crit, "mlp_and_attentive_same_sign": probes_same,
                       "seed_mean_vertex_m": sm, "best_dense_vertex_m": best_dense,
                       "not_worse_than_best_dense": not_worse,
                       "fires": bool(any(crit.values()) and probes_same and not_worse)}
    fires = [c for c, v in scale.items() if v["fires"]]
    out["decision"] = {
        "scale_to_SK": scale, "scale_to_SK_fires": fires,
        "keep_fixed_loss_default": out["contrasts"]["R2"]["prediction"] == "holds",
        "keep_new_embedding": out["contrasts"]["R2b"]["prediction"] == "holds",
        "verdict": ("scale to SK with " + ", ".join(fires)) if fires else
                   "report and stop: the sparse family is a negative result on WCTE (no S arm passes the rule)",
        "rules": "§2.6: S1, S1n or S2 better than C3 by >= 5 % on vertex or direction, clear, MLP and attentive "
                 "probes of the same sign, and no worse than the best of C1, C2, C3 on the primary outcome "
                 "(seed-mean sealed vertex error <= the best dense seed mean)."}
    return out


def fmt(x, pct=False, nd=4):
    if x is None:
        return "—"
    return f"{100 * x:+.1f} %" if pct else f"{x:.{nd}f}"


def render(out):
    L = ["# r2-sparse SPARSE_TABLE (WCTE, seeds A/B/C = 9000/9001/9002)", "",
         "Primary: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1 A.2), "
         "each model on the sealed pack of its own data kind. T = e_test / e_base − 1 (negative = test better). "
         "SE_boot: paired bootstrap of the seed-mean contrast (2,000 resamples of sealed events, shared by all runs); "
         "SE_seed = SD_pooled/√3 (SD_pooled per metric over R1–R6); N_σ = |T|/max(SE_boot, SE_seed); clear = N_σ ≥ 2 "
         "and one sign in all three seeds.", "",
         "## Registered contrasts (primary, sealed)", "",
         "| ID | Contrast | T̄ | A / B / C | SE_boot | SE_seed | N_σ | clear | prediction | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    pred = {"R1": "none", "R2": "≤ +2 %", "R2b": "≤ +2 %", "R3": "≤ +2 %", "R4": "within ±3 %",
            "R5": "S1p worse, clear", "R6": "no sign"}
    for cid in REGISTERED:
        c = out["contrasts"][cid]["sealed"]["vertex"]
        b, t = REGISTERED[cid]
        ps = " / ".join(fmt(c["per_seed"].get(k), True) for k in SEEDS)
        L.append(f"| {cid} | {t} vs {b} | {fmt(c.get('T_bar'), True)} | {ps} | {fmt(c.get('SE_boot'), True)} | "
                 f"{fmt(c.get('SE_seed'), True)} | {fmt(c.get('N_sigma'), nd=2)} | {'yes' if c.get('clear') else 'no'} | "
                 f"{pred[cid]} | {out['contrasts'][cid]['prediction']} |")
    L += ["", "## Cells (seed means, sealed)", "",
          "| Cell | vertex [m] | per seed A / B / C | MLP vertex | attentive vertex | μ dir [°] | e dir [°] | μ mis-ID @99 % e | vertex, P = all hits (expl.) | gain vs own init |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for cell, d in out["cells"].items():
        s = d["sealed"]
        g = [v["gain"] for v in out["matched_init_gain"][cell].values()]
        L.append(f"| {cell} | {fmt(s['vertex']['seed_mean'])} | " +
                 " / ".join(fmt(s["vertex_per_seed"].get(k)) for k in SEEDS) +
                 f" | {fmt(s['vertex_mlp']['seed_mean'])} | {fmt(s['vertex_attentive']['seed_mean'])} | "
                 f"{fmt(s['dir_mu']['seed_mean'], nd=2)} | {fmt(s['dir_e']['seed_mean'], nd=2)} | "
                 f"{fmt(s['misid_mu']['seed_mean'])} | {fmt(s['vertex_allhits_exploratory']['seed_mean'])} | "
                 f"{fmt(float(np.mean(g)), True) if g else '—'} |")
    L += ["", "Raw-feature baselines (spatial-moment Ridge on the 3 raw features, sealed): " +
          "; ".join(f"{k} {fmt(v['vertex_mean_m'])} m" for k, v in out["raw_baseline"].items()), "",
          "## Secondary contrasts (sealed; T̄, N_σ, clear)", "",
          "| ID | MLP vertex | attentive vertex | μ dir | e dir | μ mis-ID | vertex P = all hits (expl.) | vertex (validation) |",
          "|---|---|---|---|---|---|---|---|"]
    for cid in list(REGISTERED) + list(EXTRA):
        cs = out["contrasts"][cid]

        def cell(m, cohort="sealed"):
            c = cs[cohort][m]
            if "T_bar" not in c:
                return "—"
            return f"{fmt(c['T_bar'], True)} ({fmt(c['N_sigma'], nd=1)}{', clear' if c['clear'] else ''})"
        L.append(f"| {cid}{'' if cid in REGISTERED else ' (expl.)'} | {cell('vertex_mlp')} | {cell('vertex_attentive')} | "
                 f"{cell('dir_mu')} | {cell('dir_e')} | {cell('misid_mu')} | {cell('vertex_allhits_exploratory')} | "
                 f"{cell('vertex', 'validation')} |")
    d = out["decision"]
    L += ["", "## Decision rule (§2.6)", ""]
    for cell, v in d["scale_to_SK"].items():
        L.append(f"- {cell}: better than C3 by ≥ 5 % and clear on vertex / μ dir / e dir = "
                 f"{v['better_by_5pct_clear']['vertex']} / {v['better_by_5pct_clear']['dir_mu']} / "
                 f"{v['better_by_5pct_clear']['dir_e']}; MLP and attentive same sign: {v['mlp_and_attentive_same_sign']}; "
                 f"seed-mean vertex {fmt(v['seed_mean_vertex_m'])} m vs best dense {fmt(v['best_dense_vertex_m'])} m → "
                 f"fires: **{v['fires']}**")
    L += [f"- Keep the fixed loss as default (R2 holds): **{d['keep_fixed_loss_default']}**; keep the new embedding "
          f"(R2b holds): **{d['keep_new_embedding']}**.", f"- **Verdict: {d['verdict']}.**", "",
          "Everything outside R1–R6 and the decision rule is exploratory."]
    missing = sorted({m for cid in out["contrasts"] for c in out["contrasts"][cid].get("sealed", {}).values()
                      if isinstance(c, dict) for m in c.get("missing", [])})
    if missing:
        L += ["", "Missing runs: " + ", ".join(missing)]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    out = build(a)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "SPARSE_TABLE.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    (a.out_dir / "SPARSE_TABLE.md").write_text(render(out))
    print(out["decision"]["verdict"])


if __name__ == "__main__":
    main()
