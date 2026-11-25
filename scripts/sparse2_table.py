#!/usr/bin/env python3
"""r2-sparse2 T7: SPARSE2_TABLE.{json,md} from the evaluator outputs (PREREG_2026-11-25_R2_SPARSE2.md).

Registered contrasts P1-P5 (T = e_test / e_base - 1, seed-paired, sealed photon bombs, frozen spatial-moment Ridge)
with the r2-tier1 / r2-sparse statistics unchanged (``sparse_table.contrast``: per seed, SE_boot = paired bootstrap of
the seed-mean contrast, 2,000 resamples of sealed events; SE_seed = SD_pooled / sqrt(3); N_sigma; clear = N_sigma >= 2
and one sign in all seeds). SD_pooled per metric and cohort = RMS of the across-seed SDs of r2-sparse R1-R6, r2-follow
A1-A4 and P1-P5, all recomputed here from the evaluator outputs (three eval trees). P6 on the f_own diagnostic
(``sparse2_fown.py``). Secondary outcomes, matched-init gains, verdicts (§2.5, appendix C) and the decision rule
(§2.6). Reads only evaluator outputs and the f_own JSON (no pack is opened here).

    sparse2_table.py --fown FOWN.json --out-dir results/round2/20261125_sparse2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sparse_table as st  # noqa: E402  (r2-sparse statistics, unchanged)
from follow_table import FOLLOW_EVAL, PART_A, SPARSE_EVAL, MultiStore  # noqa: E402  (adds C3e, C2dc to st.CELLS)

SPARSE2_EVAL = Path("/ceph/dwong/jepa/returns/r2_sparse2_2026-11-25/eval")
st.CELLS["S1L"] = "wcte__sparse_mae__s{s}__dw2__elin__dleg__qall"
st.CELLS["S2L"] = "wcte__sparse_time__s{s}__dw2__elin__dfull"
st.CELLS["S2Lt"] = "wcte__sparse_time__s{s}__dw2__elin__dfull__bt5"
P = {"P1": ("C2", "S1L"), "P2": ("S1", "S1L"), "P3": ("S2", "S2L"), "P4": ("C2", "S2L"), "P5": ("S2L", "S2Lt")}
EXTRA = {"S2Lt_vs_C2": ("C2", "S2Lt"), "S2Lt_vs_S2": ("S2", "S2Lt"), "S2L_vs_S1L": ("S1L", "S2L"),
         "S1_vs_C2": ("C2", "S1"), "S2_vs_C2": ("C2", "S2")}
PRED = {"P1": "S1L not worse by > 5 %", "P2": "S1L better, clear", "P3": "S2L better, clear", "P4": "no sign",
        "P5": "no sign", "P6": "S2Lt f_own/chance higher in every seed"}
READING = {"P1": "sparse tokens reach dense level on equal components", "P2": "removing both handicaps from S1",
           "P3": "removing both handicaps from S2", "P4": "time-ordered position prediction vs the best dense cell",
           "P5": "time-width blocks, effect on vertex", "P6": "wider time blocks make 'when' learnable"}
METRICS = ("vertex", "vertex_mlp", "vertex_attentive", "dir_mu", "dir_e", "misid_mu")
CELLS = ("C2", "S1", "S2", "S1L", "S2L", "S2Lt")


def verdict(cid, c):
    if c is None or "T_bar" not in c:
        return "not evaluable"
    t, clear = c["T_bar"], c["clear"]
    if cid == "P1":
        return "holds" if t <= 0.05 else ("fails" if clear else "inconclusive")
    if cid in ("P2", "P3"):
        return "holds" if (t < 0 and clear) else ("fails" if (t > 0 and clear) else "inconclusive")
    return "no sign registered (reported)"


def p6(fown):
    per = {}
    for lab in st.SEEDS:
        a, b = fown["cells"].get("S2Lt", {}).get(lab), fown["cells"].get("S2L", {}).get(lab)
        if a and b:
            per[lab] = {"S2Lt_ratio": a["ratio"], "S2L_ratio": b["ratio"], "S2Lt_higher": a["ratio"] > b["ratio"]}
    if len(per) < 3:
        v = "not evaluable" if not per else "inconclusive (fewer than 3 seeds)"
    else:
        v = "holds" if all(x["S2Lt_higher"] for x in per.values()) else "fails"
    return {"per_seed": per, "prediction": v}


def build(args):
    store = MultiStore([SPARSE_EVAL, FOLLOW_EVAL, SPARSE2_EVAL])
    reg = {**st.REGISTERED, **PART_A, **P}
    rows = {}
    for cohort in ("sealed", "validation"):
        for cid, (b, t) in {**reg, **EXTRA}.items():
            for metric in METRICS:
                rows[(cid, metric, cohort)] = st.contrast(store, b, t, metric, cohort)
    pooled = st.finish(rows, set(reg))
    S = lambda cid, metric="vertex", cohort="sealed": rows.get((cid, metric, cohort))  # noqa: E731
    fown = json.loads(Path(args.fown).read_text()) if args.fown else {"cells": {}}
    out = {"preregistration": "docs/round2/PREREG_2026-11-25_R2_SPARSE2.md",
           "eval_roots": [str(SPARSE_EVAL), str(FOLLOW_EVAL), str(SPARSE2_EVAL)],
           "SD_pooled": pooled, "SD_pooled_over": sorted(reg), "contrasts": {}, "cells": {}, "matched_init_gain": {},
           "f_own": fown}
    for (cid, metric, cohort), c in rows.items():
        if cid in P or cid in EXTRA:
            out["contrasts"].setdefault(cid, {"base": {**P, **EXTRA}[cid][0], "test": {**P, **EXTRA}[cid][1],
                                              "registered": cid in P}).setdefault(cohort, {})[metric] = c
    for cid in P:
        out["contrasts"][cid].update(prediction=PRED[cid], reading=READING[cid], verdict=verdict(cid, S(cid)))
    out["contrasts"]["P6"] = dict(p6(fown), registered=True, prediction_text=PRED["P6"], reading=READING["P6"])
    out["contrasts"]["P6"]["verdict"] = out["contrasts"]["P6"]["prediction"]
    out["r2_sparse_and_follow_recomputed"] = {cid: rows[(cid, "vertex", "sealed")] for cid in {**st.REGISTERED, **PART_A}}
    for cell in CELLS:
        d = {}
        for cohort in ("sealed", "validation"):
            d[cohort] = {m: dict(zip(("seed_mean", "n_seeds"), st.seed_mean(store, cell, m, cohort))) for m in METRICS}
            d[cohort]["vertex_per_seed"] = {}
            for lab, s in st.SEEDS.items():
                v = store.values(st.CELLS[cell].format(s=s), 10000, cohort, "vertex")
                d[cohort]["vertex_per_seed"][lab] = st.point(v) if v is not None else None
        out["cells"][cell] = d
        g = {}
        for lab, s in st.SEEDS.items():
            rid = st.CELLS[cell].format(s=s)
            v0, v1 = store.values(rid, 0, "sealed", "vertex"), store.values(rid, 10000, "sealed", "vertex")
            if v0 is not None and v1 is not None:
                g[lab] = {"init_m": st.point(v0), "final_m": st.point(v1), "gain": 1.0 - st.point(v1) / st.point(v0)}
        out["matched_init_gain"][cell] = g
    # decision rule §2.6 (reading: PREREG appendix C)
    vs_c2 = {"S1L": "P1", "S2L": "P4", "S2Lt": "S2Lt_vs_C2"}
    scale = {}
    for cell, cid in vs_c2.items():
        prim = S(cid)
        if prim is None or "T_bar" not in prim:
            scale[cell] = {"evaluable": False, "fires": False}
            continue
        t = prim["T_bar"]
        not_worse_2 = t <= 0.02
        better_vertex = t < 0 and prim["clear"]
        dirs = [S(cid, m) for m in ("dir_mu", "dir_e")]
        better_dirs = all(c and "T_bar" in c and c["T_bar"] < 0 and c["clear"] for c in dirs)
        probes = [S(cid, m) for m in ("vertex_mlp", "vertex_attentive")]
        same = all(c and "T_bar" in c and np.sign(c["T_bar"]) == np.sign(t) and t != 0 for c in probes)
        scale[cell] = {"evaluable": True, "primary_T_bar": t, "primary_clear": prim["clear"],
                       "not_worse_than_C2_by_2pct": bool(not_worse_2), "better_clear_vertex": bool(better_vertex),
                       "better_clear_both_directions": bool(better_dirs),
                       "probe_T_bar": {m: (c or {}).get("T_bar") for m, c in zip(("mlp", "attentive"), probes)},
                       "probes_same_sign_as_primary": bool(same),
                       "fires": bool((not_worse_2 or better_vertex or better_dirs) and same)}
    fires = [c for c, v in scale.items() if v.get("fires")]
    gap = {}
    for cid, base_cell in (("P2", "S1"), ("P3", "S2")):
        c = S(cid)
        # share of the r2-sparse gap to C2 that the fair rerun closes: (e_old - e_new) / (e_old - e_C2), seed means
        e_old, e_new, e_c2 = (st.seed_mean(store, x)[0] for x in (base_cell, P[cid][1], "C2"))
        if None not in (e_old, e_new, e_c2) and e_old != e_c2:
            gap[cid] = {"old_cell": base_cell, "new_cell": P[cid][1], "e_old_m": e_old, "e_new_m": e_new, "e_C2_m": e_c2,
                        "gap_closed_fraction": (e_old - e_new) / (e_old - e_c2),
                        "T_bar": (c or {}).get("T_bar")}
    out["gap_explained"] = gap
    out["decision"] = {
        "scale_to_SK": scale, "fires_for": fires,
        "verdict": ("scale the sparse family to SK (follow-up runbook) with " + ", ".join(fires)) if fires else
                   "negative result on WCTE with fair components: no sparse cell meets the §2.6 rule",
        "rule": "§2.6 with PREREG appendix C: a sparse cell (S1L, S2L, S2Lt) fires if [primary T̄ vs C2 ≤ +2 %, or "
                "T̄ < 0 and clear on vertex, or T̄ < 0 and clear on both μ and e direction] and the MLP and attentive "
                "probe contrasts vs C2 have the sign of its primary contrast."}
    return out


def fmt(x, pct=False, nd=4):
    if x is None:
        return "—"
    return f"{100 * x:+.1f} %" if pct else f"{x:.{nd}f}"


def render(out):
    L = ["# r2-sparse2 SPARSE2_TABLE (WCTE lucid_v1_w2, seeds A/B/C = 9000/9001/9002)", "",
         "Primary: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1 A.2), "
         "step 10,000. T = e_test / e_base − 1 (negative = test better). SE_boot: paired bootstrap of the seed-mean "
         "contrast (2,000 resamples of sealed events); SE_seed = SD_pooled/√3, SD_pooled over r2-sparse R1–R6, r2-follow "
         "A1–A4 and P1–P5 (vertex, sealed: " + fmt(out["SD_pooled"].get("vertex/sealed")) + "); N_σ = |T̄|/max(SE_boot, "
         "SE_seed); clear = N_σ ≥ 2 and one sign in all three seeds.", "",
         "## Registered contrasts (primary, sealed)", "",
         "| ID | Contrast | T̄ | A / B / C | SE_boot | SE_seed | N_σ | clear | prediction | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for cid in P:
        c = out["contrasts"][cid]["sealed"]["vertex"]
        b, t = P[cid]
        ps = " / ".join(fmt(c["per_seed"].get(k), True) for k in st.SEEDS)
        L.append(f"| {cid} | {t} vs {b} | {fmt(c.get('T_bar'), True)} | {ps} | {fmt(c.get('SE_boot'), True)} | "
                 f"{fmt(c.get('SE_seed'), True)} | {fmt(c.get('N_sigma'), nd=2)} | {'yes' if c.get('clear') else 'no'} | "
                 f"{PRED[cid]} | {out['contrasts'][cid]['verdict']} |")
    p6v = out["contrasts"]["P6"]
    ps = " / ".join((f"{v['S2Lt_ratio']:.2f} vs {v['S2L_ratio']:.2f}" if (v := p6v["per_seed"].get(k)) else "—")
                    for k in st.SEEDS)
    L.append(f"| P6 | S2Lt vs S2L, f_own/chance | — | {ps} | — | — | — | — | {PRED['P6']} | {p6v['verdict']} |")
    L += ["", "## Cells (seed means, sealed, step 10,000)", "",
          "| Cell | vertex [m] | per seed A / B / C | MLP vertex | attentive vertex | μ dir [°] | e dir [°] | μ mis-ID @99 % e | gain vs own init |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cell, d in out["cells"].items():
        s = d["sealed"]
        g = [v["gain"] for v in out["matched_init_gain"][cell].values()]
        L.append(f"| {cell} | {fmt(s['vertex']['seed_mean'])} | " +
                 " / ".join(fmt(s["vertex_per_seed"].get(k)) for k in st.SEEDS) +
                 f" | {fmt(s['vertex_mlp']['seed_mean'])} | {fmt(s['vertex_attentive']['seed_mean'])} | "
                 f"{fmt(s['dir_mu']['seed_mean'], nd=2)} | {fmt(s['dir_e']['seed_mean'], nd=2)} | "
                 f"{fmt(s['misid_mu']['seed_mean'])} | {fmt(float(np.mean(g)), True) if g else '—'} |")
    L += ["", "Gain vs own init = 1 − e(10,000)/e(0), sealed primary readout, seed mean.", "",
          "## Secondary contrasts (sealed; T̄ (N_σ, clear))", "",
          "| ID | MLP vertex | attentive vertex | μ dir | e dir | μ mis-ID | vertex (validation) |", "|---|---|---|---|---|---|---|"]
    for cid in list(P) + list(EXTRA):
        cs = out["contrasts"][cid]

        def cell(m, cohort="sealed"):
            c = cs[cohort][m]
            if "T_bar" not in c:
                return "—"
            return f"{fmt(c['T_bar'], True)} ({fmt(c['N_sigma'], nd=1)}{', clear' if c['clear'] else ''})"
        lab = cid if cid in P else f"{cs['test']} vs {cs['base']} (expl.)"
        L.append(f"| {lab} | {cell('vertex_mlp')} | {cell('vertex_attentive')} | {cell('dir_mu')} | {cell('dir_e')} | "
                 f"{cell('misid_mu')} | {cell('vertex', 'validation')} |")
    L += ["", "## f_own diagnostic (dev_probe_test, step 10,000; PREREG appendix B)", "",
          "| Cell | seed | f_own (own / hidden-hit mass) | chance | f_own / chance | own / total mass | mass on hidden hits | mass on unhit |",
          "|---|---|---|---|---|---|---|---|"]
    for cell, per in out["f_own"].get("cells", {}).items():
        for lab, r in per.items():
            L.append(f"| {cell} | {lab} | {100 * r['f_own']:.1f} % | {100 * r['chance']:.1f} % | {r['ratio']:.2f} | "
                     f"{100 * r['own_over_total']:.1f} % | {100 * r['mass_on_hidden_hits']:.1f} % | {100 * r['mass_on_unhit']:.1f} % |")
    g = out["gap_explained"]
    L += ["", "## How much of the r2-sparse gap to C2 the handicaps explained (seed means, sealed)", ""]
    for cid, v in g.items():
        L.append(f"- {cid}: {v['old_cell']} {fmt(v['e_old_m'])} m → {v['new_cell']} {fmt(v['e_new_m'])} m (C2 "
                 f"{fmt(v['e_C2_m'])} m): {100 * v['gap_closed_fraction']:.0f} % of the gap closed.")
    d = out["decision"]
    L += ["", "## Decision rule (§2.6, appendix C)", ""]
    for cell, v in d["scale_to_SK"].items():
        if not v.get("evaluable"):
            L.append(f"- {cell}: not evaluable")
            continue
        L.append(f"- {cell}: primary T̄ vs C2 {fmt(v['primary_T_bar'], True)} (clear {v['primary_clear']}); ≤ +2 %: "
                 f"{v['not_worse_than_C2_by_2pct']}; better clear (vertex / both directions): {v['better_clear_vertex']} / "
                 f"{v['better_clear_both_directions']}; MLP / attentive T̄ {fmt(v['probe_T_bar']['mlp'], True)} / "
                 f"{fmt(v['probe_T_bar']['attentive'], True)}, same sign: {v['probes_same_sign_as_primary']} → "
                 f"fires: **{v['fires']}**")
    L += [f"- **Verdict: {d['verdict']}.**", "", "Everything outside P1–P6 and the decision rule is exploratory."]
    missing = sorted({m for cid in out["contrasts"] for c in out["contrasts"][cid].get("sealed", {}).values()
                      if isinstance(c, dict) for m in c.get("missing", [])})
    if missing:
        L += ["", "Missing runs: " + ", ".join(missing)]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fown", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    out = build(a)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "SPARSE2_TABLE.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    (a.out_dir / "SPARSE2_TABLE.md").write_text(render(out))
    print(out["decision"]["verdict"])


if __name__ == "__main__":
    main()
