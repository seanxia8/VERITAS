#!/usr/bin/env python3
"""r2-sparse-sk T7: SPARSE_SK_TABLE.{json,md} from the evaluator outputs (PREREG_2026-11-26_R2_SPARSE_SK.md).

Registered contrasts K1-K6 (T = e_test / e_base - 1 per seed, sealed, frozen spatial-moment Ridge; secondary
metrics likewise) with the unchanged r2-tier1 / r2-sparse statistics (``sparse_table.contrast`` / ``finish``:
SE_boot = paired bootstrap of the seed-mean contrast, 2,000 resamples; SE_seed = SD_pooled / sqrt(3);
N_sigma = |T| / max(SE_boot, SE_seed); clear = N_sigma >= 2 and one sign in all seeds). SD_pooled (appendix A9):
SK = RMS of the across-seed SDs of the SK 3-seed mask contrasts (patch 43 / 173 / 692 vs random, r2-mask and
r2-follow) and K1-K5; WCTE = r2-sparse2's set (R1-R6, A1-A4, P1-P5) plus K6. Verdicts (appendix A7), the
decision (§2.7, appendix A8), matched-init gains, efficiency (appendix A10). Reads evaluator outputs, run
records and train logs only (no pack is opened).

    sparse_sk_table.py --out-dir results/round2/20261126_sparse_sk [--extra-json exploratory.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sparse_table as st  # noqa: E402  (r2-sparse statistics, unchanged)
from follow_table import FOLLOW_EVAL, PART_A, SPARSE_EVAL, MultiStore  # noqa: E402
from sparse2_table import P as P_SPARSE2, SPARSE2_EVAL  # noqa: E402  (adds S1L, S2L, S2Lt to st.CELLS)

RET = Path("/ceph/dwong/jepa/returns")
SKSK = RET / "r2_sparse_sk_2026-11-26"
SKSK_EVAL = SKSK / "eval"
MASK_EVAL = RET / "r2_mask_2026-11-23" / "eval"
TIER1_EVAL = RET / "r2_tier1_2026-11-22" / "eval"
ROOTS = [SKSK_EVAL, FOLLOW_EVAL, MASK_EVAL, TIER1_EVAL, SPARSE_EVAL, SPARSE2_EVAL]
TRAIN = {"SK-D0": {9000: RET / "r2_lucid_2026-11-21/train/runs/sk__mae_repaired__s9000",
                   9001: RET / "r2_tier1_2026-11-22/train/runs/sk__mae_repaired__s9001",
                   9002: RET / "r2_follow_2026-11-24/train/runs/sk__mae_repaired__s9002"}}

st.CELLS.update({
    "SK-D0": "sk__mae_repaired__s{s}",
    "SK-C2": "sk__mae_repaired__s{s}__lfixed__psent",
    "SK-S1L": "sk__sparse_mae__s{s}__elin__dleg__qall",
    "SK-S1A": "sk__sparse_mae__s{s}__elin__dleg__abs__qall",
    "SK-S2L": "sk__sparse_time__s{s}__elin__dfull",
    "SK-rnd": "sk__mae_repaired__s{s}__mrandom",
    "SK-p43": "sk__mae_repaired__s{s}__ps43",
    "SK-p692": "sk__mae_repaired__s{s}__ps692",
    "W-S1A": "wcte__sparse_mae__s{s}__dw2__elin__dleg__abs__qall",
})
st.METRICS.update({"energy_mu": ("rel_energy_muon", "hw68", "energy_muon"),
                   "energy_e": ("rel_energy_electron", "hw68", "energy_electron")})

K = {"K1": ("SK-D0", "SK-C2"), "K2": ("SK-C2", "SK-S1L"), "K3": ("SK-D0", "SK-S1L"), "K4": ("SK-C2", "SK-S2L"),
     "K5": ("SK-S1L", "SK-S1A"), "K6": ("S1L", "W-S1A")}
SK_MASK = {"M43": ("SK-rnd", "SK-p43"), "M173": ("SK-rnd", "SK-D0"), "M692": ("SK-rnd", "SK-p692")}
EXTRA = {"S1A_vs_C2": ("SK-C2", "SK-S1A"), "S2L_vs_S1L": ("SK-S1L", "SK-S2L"), "S1A_vs_D0": ("SK-D0", "SK-S1A"),
         "W-S1A_vs_C2": ("C2", "W-S1A")}
PRED = {"K1": "SK-C2 better, clear", "K2": "SK-S1L not worse by > 5 %", "K3": "no sign registered",
        "K4": "vertex: no sign; mu and e direction: SK-S2L better, clear",
        "K5": "mu mis-ID lower, clear; vertex not worse by > 2 %", "K6": "as K5 (WCTE)"}
READING = {"K1": "the loss fix where dilution is large", "K2": "replication of WCTE P1 on SK",
           "K3": "sparse tokens and loss fix together, against the Tier-1 default",
           "K4": "replication of WCTE S2L's direction gains", "K5": "absence tokens fix the mu/e weakness",
           "K6": "the same on WCTE"}
SK_METRICS = ("vertex", "vertex_mlp", "vertex_attentive", "dir_mu", "dir_e", "misid_mu", "energy_mu", "energy_e")
W_METRICS = ("vertex", "vertex_mlp", "vertex_attentive", "dir_mu", "dir_e", "misid_mu")
SK_CELLS = ("SK-D0", "SK-C2", "SK-S1L", "SK-S1A", "SK-S2L")
W_CELLS = ("C2", "S1L", "W-S1A")


class Store(MultiStore):
    """MultiStore plus the energy metrics (event ids of the contained, evaluable events)."""

    def values(self, rid, step, cohort, metric):
        if metric not in ("energy_mu", "energy_e"):
            return super().values(rid, step, cohort, metric)
        got = self.get(rid, step, cohort)
        if got is None:
            return None
        _, arr = got
        key, kind, tag = st.METRICS[metric]
        dom = "muon" if metric == "energy_mu" else "electron"
        if key not in arr or f"energy_event_ids_{dom}" not in arr:
            return None
        return {"kind": kind, "x": arr[key], "tag": tag, "ids": arr[f"energy_event_ids_{dom}"]}


def better_clear(c):
    if c is None or "T_bar" not in c:
        return "not evaluable"
    return "holds" if (c["T_bar"] < 0 and c["clear"]) else ("fails" if (c["T_bar"] > 0 and c["clear"]) else
                                                            "inconclusive")


def not_worse(c, x):
    if c is None or "T_bar" not in c:
        return "not evaluable"
    return "holds" if c["T_bar"] <= x else ("fails" if c["clear"] else "inconclusive")


def both(a, b):
    if "not evaluable" in (a, b):
        return "not evaluable"
    if a == "holds" and b == "holds":
        return "holds"
    if "fails" in (a, b):
        return "fails"
    return "inconclusive"


def verdicts(S):
    v = {"K1": better_clear(S("K1")), "K2": not_worse(S("K2"), 0.05), "K3": "no sign registered (reported)"}
    d = [better_clear(S("K4", m)) for m in ("dir_mu", "dir_e")]
    v["K4"] = both(*d)
    for k in ("K5", "K6"):
        v[k] = both(better_clear(S(k, "misid_mu")), not_worse(S(k), 0.02))
    v["_parts"] = {"K4_dir_mu": d[0], "K4_dir_e": d[1],
                   "K5_misid": better_clear(S("K5", "misid_mu")), "K5_vertex": not_worse(S("K5"), 0.02),
                   "K6_misid": better_clear(S("K6", "misid_mu")), "K6_vertex": not_worse(S("K6"), 0.02)}
    return v


def step_time_from_log(run_dir):
    """Median wall time per step over steps 1,001-2,000 from train_log.jsonl (appendix A10, SK-D0)."""
    p = Path(run_dir) / "train_log.jsonl"
    if not p.is_file():
        return None
    el = {}
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            if 1000 <= r["step"] <= 2000:
                el[r["step"]] = r["elapsed_seconds"]
    dt = [el[k] - el[k - 1] for k in range(1001, 2001) if k in el and k - 1 in el]
    return {"median_step_seconds": statistics.median(dt) if dt else None, "n_intervals": len(dt),
            "source": "train_log.jsonl elapsed_seconds"}


def efficiency(cell, seed, store_rid):
    if cell in TRAIN:
        d = TRAIN[cell][seed]
    else:
        root = SKSK / "train" / "runs"
        if cell in ("C2", "S1L"):
            root = (RET / "r2_sparse_2026-11-24/train/runs") if cell == "C2" else (RET / "r2_sparse2_2026-11-25/train/runs")
        d = root / store_rid
    rec_p = d / "run_record.json"
    if not rec_p.is_file():
        return None
    rec = json.loads(rec_p.read_text())
    eff = rec.get("efficiency")
    out = {"gpu_model": rec.get("gpu_model"), "peak_memory_gib_run": rec.get("peak_memory_gib"),
           "wall_seconds": rec.get("wall_seconds")}
    if eff:
        out.update(median_step_seconds=eff.get("median_step_seconds"), n_intervals=eff.get("n_intervals"),
                   peak_allocated_gib_2000=eff.get("peak_allocated_gib"), peak_reserved_gib_2000=eff.get("peak_reserved_gib"),
                   source="efficiency.json")
    else:
        out.update(step_time_from_log(d) or {})
    return out


def build(args):
    store = Store(ROOTS)
    reg_wcte = {**st.REGISTERED, **PART_A, **P_SPARSE2, "K6": K["K6"]}
    reg_sk = {**SK_MASK, **{k: K[k] for k in ("K1", "K2", "K3", "K4", "K5")}}
    rows_sk, rows_w = {}, {}
    for cohort in ("sealed", "validation"):
        for cid, (b, t) in {**reg_sk, **{k: v for k, v in EXTRA.items() if v[0].startswith("SK")}}.items():
            for m in SK_METRICS:
                rows_sk[(cid, m, cohort)] = st.contrast(store, b, t, m, cohort)
        for cid, (b, t) in {**reg_wcte, "W-S1A_vs_C2": EXTRA["W-S1A_vs_C2"]}.items():
            for m in W_METRICS:
                rows_w[(cid, m, cohort)] = st.contrast(store, b, t, m, cohort)
    pooled_sk = st.finish(rows_sk, set(reg_sk))
    pooled_w = st.finish(rows_w, set(reg_wcte))
    rows = {**rows_sk, **rows_w}
    S = lambda cid, metric="vertex", cohort="sealed": rows.get((cid, metric, cohort))  # noqa: E731
    out = {"preregistration": "docs/round2/PREREG_2026-11-26_R2_SPARSE_SK.md", "eval_roots": [str(r) for r in ROOTS],
           "SD_pooled": {"sk": pooled_sk, "wcte": pooled_w},
           "SD_pooled_over": {"sk": sorted(reg_sk), "wcte": sorted(reg_wcte)}, "contrasts": {}, "cells": {},
           "matched_init_gain": {}, "efficiency": {}}
    for (cid, metric, cohort), c in rows.items():
        if cid in K or cid in EXTRA or cid in SK_MASK:
            base, test = {**K, **EXTRA, **SK_MASK}[cid]
            out["contrasts"].setdefault(cid, {"base": base, "test": test, "registered": cid in K})\
                .setdefault(cohort, {})[metric] = c
    v = verdicts(S)
    for cid in K:
        out["contrasts"][cid].update(prediction=PRED[cid], reading=READING[cid], verdict=v[cid])
    out["verdict_parts"] = v["_parts"]
    for cell in SK_CELLS + W_CELLS:
        metrics = SK_METRICS if cell.startswith("SK") else W_METRICS
        d = {}
        for cohort in ("sealed", "validation"):
            d[cohort] = {m: dict(zip(("seed_mean", "n_seeds"), st.seed_mean(store, cell, m, cohort))) for m in metrics}
            d[cohort]["vertex_per_seed"] = {}
            for lab, s in st.SEEDS.items():
                val = store.values(st.CELLS[cell].format(s=s), 10000, cohort, "vertex")
                d[cohort]["vertex_per_seed"][lab] = st.point(val) if val is not None else None
        out["cells"][cell] = d
        g, e = {}, {}
        for lab, s in st.SEEDS.items():
            rid = st.CELLS[cell].format(s=s)
            v0, v1 = store.values(rid, 0, "sealed", "vertex"), store.values(rid, 10000, "sealed", "vertex")
            if v0 is not None and v1 is not None:
                g[lab] = {"init_m": st.point(v0), "final_m": st.point(v1), "gain": 1.0 - st.point(v1) / st.point(v0)}
            e[lab] = efficiency(cell, s, rid)
        if g:
            g["mean_gain"] = float(np.mean([x["gain"] for x in g.values()]))
        out["matched_init_gain"][cell] = g
        out["efficiency"][cell] = e
    # decision §2.7 (appendix A8)
    k2, k5, k1 = v["K2"] == "holds", v["K5"] == "holds", v["K1"] == "holds"
    s1l_misid = S("K2", "misid_mu")
    s1l_misid_clearly_worse = bool(s1l_misid and "T_bar" in s1l_misid and s1l_misid["T_bar"] > 0 and s1l_misid["clear"])
    sparse_default = k2 and (k5 or not s1l_misid_clearly_worse)
    t_star = "SK-S1A" if (k2 and k5) else ("SK-S1L" if k2 else "dense C2")
    out["decision"] = {
        "K1_holds": k1, "K2_holds": k2, "K5_holds": k5,
        "SK_S1L_misid_vs_C2": {k: (s1l_misid or {}).get(k) for k in ("T_bar", "N_sigma", "clear", "per_seed")},
        "SK_S1L_misid_clearly_worse_than_C2": s1l_misid_clearly_worse,
        "sparse_tokens_default_for_HK": bool(sparse_default),
        "loss_fix_default_dense_loss": bool(k1),
        "T_star": t_star,
        "rule": "§2.7 / A8: sparse tokens become the default tokenisation if K2 holds and (K5 holds or SK-S1L's mu "
                "mis-ID is not clearly worse than SK-C2's); the loss fix becomes the default dense loss if K1 holds; "
                "T* = SK-S1A if K2 and K5 hold, else SK-S1L if K2 holds, else dense C2."}
    if args.extra_json and Path(args.extra_json).is_file():
        out["exploratory"] = json.loads(Path(args.extra_json).read_text())
    return out


def pct(x, nd=1):
    return "–" if x is None else f"{100 * x:+.{nd}f} %"


def num(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def render(out):
    L = ["# SPARSE_SK_TABLE — r2-sparse-sk Stage S (sealed photon bombs unless noted)", "",
         "Statistics as `PREREG_2026-11-22_R2_TIER1.md` §2.2 (T = e_test/e_base − 1 per seed; SE_boot = paired bootstrap, "
         "2,000 resamples; SE_seed = SD_pooled/√3; N_σ = |T̄|/max(SE_boot, SE_seed); clear = N_σ ≥ 2 and one sign in "
         f"all seeds). SD_pooled vertex/sealed: SK {num(out['SD_pooled']['sk'].get('vertex/sealed'))} "
         f"(over {', '.join(out['SD_pooled_over']['sk'])}), WCTE {num(out['SD_pooled']['wcte'].get('vertex/sealed'))}.", "",
         "## Registered contrasts", "",
         "| ID | base → test | metric | T̄ | per seed (A/B/C) | SE_boot | SE_seed | N_σ | clear | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    show = {"K1": ("vertex", "misid_mu"), "K2": ("vertex", "misid_mu"), "K3": ("vertex",),
            "K4": ("vertex", "dir_mu", "dir_e"), "K5": ("vertex", "misid_mu"), "K6": ("vertex", "misid_mu")}
    for cid, ms in show.items():
        c0 = out["contrasts"][cid]
        for m in ms:
            c = c0["sealed"][m]
            ps = "/".join(pct(c.get("per_seed", {}).get(lab)) for lab in "ABC")
            L.append(f"| {cid} | {c0['base']} → {c0['test']} | {m} | {pct(c.get('T_bar'))} | {ps} | {pct(c.get('SE_boot'))} | "
                     f"{pct(c.get('SE_seed'))} | {num(c.get('N_sigma'), 1)} | {c.get('clear')} | "
                     f"{c0['verdict'] if m == ms[0] else ''} |")
    L += ["", f"Verdict parts: {json.dumps(out['verdict_parts'])}", "",
          "## Secondary outcomes (all registered contrasts, sealed)", "",
          "| ID | " + " | ".join(SK_METRICS) + " |", "|---|" + "---|" * len(SK_METRICS)]
    for cid in list(K) + list(EXTRA):
        c0 = out["contrasts"].get(cid)
        if not c0:
            continue
        cells = []
        for m in SK_METRICS:
            c = c0["sealed"].get(m)
            cells.append("–" if not c or "T_bar" not in c else f"{pct(c['T_bar'])}{' ✓' if c['clear'] else ''}")
        L.append(f"| {cid} ({c0['base']}→{c0['test']}) | " + " | ".join(cells) + " |")
    L += ["", "✓ = clear.", "", "## Cell seed means (sealed)", "",
          "| Cell | vertex [m] | MLP | attentive | μ dir [°] | e dir [°] | μ mis-ID | E res μ | E res e | gain over init |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for cell, d in out["cells"].items():
        s = d["sealed"]
        g = out["matched_init_gain"][cell].get("mean_gain")
        L.append(f"| {cell} | {num(s['vertex']['seed_mean'])} | {num(s['vertex_mlp']['seed_mean'])} | "
                 f"{num(s['vertex_attentive']['seed_mean'])} | {num(s['dir_mu']['seed_mean'], 2)} | "
                 f"{num(s['dir_e']['seed_mean'], 2)} | {num(s['misid_mu']['seed_mean'])} | "
                 f"{num((s.get('energy_mu') or {}).get('seed_mean'))} | {num((s.get('energy_e') or {}).get('seed_mean'))} | "
                 f"{pct(g)} |")
    L += ["", "## Efficiency (appendix A10)", "", "| Cell | seed | GPU | median s/step (1,001–2,000) | peak alloc GiB (≤ 2,000) | run peak reserved GiB |",
          "|---|---|---|---|---|---|"]
    for cell, e in out["efficiency"].items():
        for lab, x in e.items():
            if x:
                L.append(f"| {cell} | {lab} | {x.get('gpu_model')} | {num(x.get('median_step_seconds'), 3)} | "
                         f"{num(x.get('peak_allocated_gib_2000'), 2)} | {num(x.get('peak_memory_gib_run'), 2)} |")
    dcs = out["decision"]
    L += ["", "## Decision (§2.7)", "", f"- Sparse tokens default for HK: **{dcs['sparse_tokens_default_for_HK']}**; "
          f"loss fix default dense loss: **{dcs['loss_fix_default_dense_loss']}**; **T\\* = {dcs['T_star']}**.",
          f"- Rule: {dcs['rule']}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-json")
    a = ap.parse_args()
    out = build(a)
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / "SPARSE_SK_TABLE.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    (o / "SPARSE_SK_TABLE.md").write_text(render(out))
    print(json.dumps({"verdicts": {k: out["contrasts"][k]["verdict"] for k in K}, "decision": out["decision"]["T_star"]}))


if __name__ == "__main__":
    main()
