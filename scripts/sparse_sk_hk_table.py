#!/usr/bin/env python3
"""r2-sparse-sk Stage H / Stage J tables (PREREG_2026-11-26_R2_SPARSE_SK.md appendices H and J).

``hk``: HK_TABLE.{json,md} and U_GAIN_3DET.png. Cells H-C2 (dense C2 recipe, patch 308) and H-T-<mask> (the
Stage S tokenisation T*, masks random / patch 77 / 308 / 1232; dense-branch cells H-C2-rnd / H-C2-1232 when
T* is dense). H1 H-T-308 vs H-C2 (not worse by > 5 %); H2 each patch vs random under T* (T < 0 and clear);
H3 Spearman rho between M1's U (photon bombs) and the seed-mean vertex gain over matched init (step 0) over the
four configurations (rho >= 0.8); H4 step time and peak memory, H-T-308 vs H-C2. SD_pooled (HK) = RMS of the
across-seed SDs of H1 and H2 (no earlier 3-seed HK contrast exists). The figure shows U against the seed-mean
vertex gain for WCTE and SK (r2-mask 3-seed analysis) and HK (this phase).

``juno``: JUNO_TABLE.{json,md}: J1 J-C2v2 vs J-C2v1 (no sign), J2 J-Tv2 vs J-C2v2 (not worse by > 5 %), SD_pooled
over J1 and J2. Statistics are the unchanged r2-sparse ones (``sparse_table.contrast`` / ``finish``).

    sparse_sk_hk_table.py hk --tstar SK-S1A --m1 M1_HK.json --mask3 MASK_SCALE_3seed.json --out-dir <results>
    sparse_sk_hk_table.py juno --tstar SK-S1A --out-dir <results>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sparse_table as st  # noqa: E402
from sparse_sk_table import SKSK, SKSK_EVAL, Store, efficiency, not_worse, better_clear, pct, num  # noqa: E402

TOK = {"SK-S1A": "sparse_mae__s{s}{m}__elin__dleg__abs__qall", "SK-S1L": "sparse_mae__s{s}{m}__elin__dleg__qall",
       "dense C2": "mae_repaired__s{s}{m}__lfixed__psent"}
HK_MASKS = {"rnd": "__mrandom", "77": "__ps77", "308": "", "1232": "__ps1232"}
U_KEY = {"rnd": "random", "77": "patch77", "308": "patch308", "1232": "patch1232"}
METRICS = ("vertex", "vertex_mlp", "vertex_attentive", "dir_mu", "dir_e", "misid_mu", "energy_mu", "energy_e")
RHO_MIN = 0.8 - 1e-9


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or any(v is None for v in list(x) + list(y)):
        return None
    return float(spearmanr(x, y).correlation)


def contrast_rows(store, reg, metrics):
    rows = {}
    for cohort in ("sealed", "validation"):
        for cid, (b, t) in reg.items():
            for m in metrics:
                rows[(cid, m, cohort)] = st.contrast(store, b, t, m, cohort)
    return rows


def gains(store, cell):
    g = {}
    for lab, s in st.SEEDS.items():
        rid = st.CELLS[cell].format(s=s)
        v0, v1 = store.values(rid, 0, "sealed", "vertex"), store.values(rid, 10000, "sealed", "vertex")
        if v0 is not None and v1 is not None:
            g[lab] = {"init_m": st.point(v0), "final_m": st.point(v1), "gain": 1.0 - st.point(v1) / st.point(v0)}
    if g:
        g["mean_gain"] = float(np.mean([x["gain"] for x in g.values() if isinstance(x, dict)]))
    return g


def seed_means(store, cell, metrics):
    return {m: dict(zip(("seed_mean", "n_seeds"), st.seed_mean(store, cell, m, "sealed"))) for m in metrics}


def hk(args):
    tstar = args.tstar
    st.CELLS["H-C2"] = "hk__" + TOK["dense C2"].format(s="{s}", m="")
    sparse = tstar != "dense C2"
    cells = ["H-C2"]
    if sparse:
        for k, m in HK_MASKS.items():
            st.CELLS[f"H-T-{k}"] = "hk__" + TOK[tstar].format(s="{s}", m=m)
            cells.append(f"H-T-{k}")
        cfg = {k: f"H-T-{k}" for k in HK_MASKS}
        reg = {"H1": ("H-C2", "H-T-308"), **{f"H2-{k}": ("H-T-rnd", f"H-T-{k}") for k in ("77", "308", "1232")}}
    else:
        st.CELLS["H-C2-rnd"] = "hk__" + TOK["dense C2"].format(s="{s}", m="__mrandom")
        st.CELLS["H-C2-1232"] = "hk__" + TOK["dense C2"].format(s="{s}", m="__ps1232")
        cells += ["H-C2-rnd", "H-C2-1232"]
        cfg = {"rnd": "H-C2-rnd", "308": "H-C2", "1232": "H-C2-1232"}
        reg = {f"H2-{k}": ("H-C2-rnd", cfg[k]) for k in ("308", "1232")}
    store = Store([SKSK_EVAL])
    rows = contrast_rows(store, reg, METRICS)
    pooled = st.finish(rows, set(reg))
    S = lambda cid, m="vertex", c="sealed": rows.get((cid, m, c))  # noqa: E731
    m1 = json.loads(Path(args.m1).read_text())
    U = {k: m1["domains"]["photon"]["configs"][U_KEY[k]]["U"] for k in HK_MASKS}
    g = {c: gains(store, c) for c in cells}
    ks = [k for k in cfg if g[cfg[k]].get("mean_gain") is not None]
    rho = spearman([U[k] for k in ks], [g[cfg[k]]["mean_gain"] for k in ks]) if len(ks) == len(cfg) else None
    out = {"preregistration": "docs/round2/PREREG_2026-11-26_R2_SPARSE_SK.md (appendix H)", "T_star": tstar,
           "branch": "sparse" if sparse else "dense", "cells": {c: st.CELLS[c] for c in cells},
           "U_photon": U, "U_all_domains": {d: {k: m1["domains"][d]["configs"][U_KEY[k]]["U"] for k in HK_MASKS}
                                             for d in m1["domains"]},
           "SD_pooled": pooled, "SD_pooled_over": sorted(reg), "contrasts": {}, "matched_init_gain": g,
           "seed_means": {c: seed_means(store, c, METRICS) for c in cells}, "efficiency": {}}
    for (cid, m, c), r in rows.items():
        out["contrasts"].setdefault(cid, {"base": reg[cid][0], "test": reg[cid][1]}).setdefault(c, {})[m] = r
    if sparse:
        out["H1"] = {"prediction": "H-T-308 not worse than H-C2 by more than 5 %", "verdict": not_worse(S("H1"), 0.05)}
    else:
        out["H1"] = {"verdict": "not applicable (dense branch)"}
    h2 = {k: better_clear(S(f"H2-{k}")) for k in (("77", "308", "1232") if sparse else ("308", "1232"))}
    out["H2"] = {"prediction": "each patch vs random (T*): T < 0 and clear, 3 seeds", "per_patch": h2,
                 "verdict": ("not evaluable" if any(v == "not evaluable" for v in h2.values()) else
                             "holds" if all(v == "holds" for v in h2.values()) else
                             "fails" if any(v == "fails" for v in h2.values()) else "inconclusive")}
    out["H3"] = {"prediction": "Spearman rho(U, seed-mean vertex gain over matched init) >= 0.8",
                 "configs": ks, "U": [U[k] for k in ks], "gain": [g[cfg[k]].get("mean_gain") for k in ks],
                 "rho": rho, "verdict": ("not evaluable" if rho is None else "holds" if rho >= RHO_MIN else "fails")}
    for c in cells:
        out["efficiency"][c] = {lab: efficiency_hk(c, s) for lab, s in st.SEEDS.items()}
    e1, e2 = out["efficiency"]["H-C2"], out["efficiency"].get(cfg["308"] if sparse else "H-C2")
    out["H4"] = {"H-C2": e1, "H-T-308": e2 if sparse else None}
    if out["H3"]["verdict"] == "holds":
        out["claim_wording"] = "on SK and HK the vertex gain follows the post-interpolation residual U"
    elif out["H2"]["verdict"] == "holds":
        out["claim_wording"] = ("on SK; on WCTE and HK the gain saturates or turns over beyond a detector-dependent "
                                "scale")
    else:
        out["claim_wording"] = "on SK only; HK does not reproduce the patch-over-random gain (H2 does not hold)"
    mask3 = json.loads(Path(args.mask3).read_text())
    out["three_detectors"] = three_det(mask3, U, {k: g[cfg[k]].get("mean_gain") for k in cfg})
    figure(out["three_detectors"], Path(args.out_dir) / "U_GAIN_3DET.png")
    return out


def efficiency_hk(cell, seed):
    rid = st.CELLS[cell].format(s=seed)
    return efficiency("H", seed, rid)


def three_det(mask3, u_hk, g_hk):
    out = {}
    for det in ("wcte", "sk"):
        d = mask3["detectors"][det]
        out[det] = {"U": d["U"], "gain": {c: v.get("mean") for c, v in d["gain_vertex"].items()},
                    "seeds": d.get("seeds")}
    out["hk"] = {"U": {U_KEY[k]: v for k, v in u_hk.items()}, "gain": {U_KEY[k]: v for k, v in g_hk.items()}}
    for det, d in out.items():
        ks = [c for c in d["U"] if d["gain"].get(c) is not None]
        d["spearman"] = spearman([d["U"][c] for c in ks], [d["gain"][c] for c in ks])
    return out


def figure(td, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"wcte": "#4C72B0", "sk": "#DD8452", "hk": "#55A868"}
    for det, d in td.items():
        ks = sorted([c for c in d["U"] if d["gain"].get(c) is not None], key=lambda c: d["U"][c])
        if not ks:
            continue
        x = [d["U"][c] for c in ks]
        y = [100 * d["gain"][c] for c in ks]
        rho = d.get("spearman")
        ax.plot(x, y, "o-", color=colors[det], label=f"{det.upper()} (ρ = {rho:.2f})" if rho is not None else det.upper())
        for c, xi, yi in zip(ks, x, y):
            ax.annotate(c.replace("patch", "p").replace("random", "rnd"), (xi, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=colors[det])
    ax.set_xlabel("post-interpolation residual U (M1, photon bombs)")
    ax.set_ylabel("vertex gain over matched init [%] (seed mean, sealed)")
    ax.set_title("Mask scale: U vs vertex gain on three detectors")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def juno(args):
    tstar = args.tstar
    st.CELLS["J-C2v1"] = "juno__" + TOK["dense C2"].format(s="{s}", m="")
    st.CELLS["J-C2v2"] = "juno__mae_repaired__s{s}__dw2__lfixed__psent"
    cells, reg = ["J-C2v1", "J-C2v2"], {"J1": ("J-C2v1", "J-C2v2")}
    if tstar != "dense C2":
        st.CELLS["J-Tv2"] = "juno__" + TOK[tstar].format(s="{s}", m="").replace("__elin", "__dw2__elin")
        cells.append("J-Tv2")
        reg["J2"] = ("J-C2v2", "J-Tv2")
    store = Store([SKSK_EVAL])
    rows = contrast_rows(store, reg, METRICS)
    pooled = st.finish(rows, set(reg))
    S = lambda cid, m="vertex", c="sealed": rows.get((cid, m, c))  # noqa: E731
    out = {"preregistration": "docs/round2/PREREG_2026-11-26_R2_SPARSE_SK.md (appendix J)", "T_star": tstar,
           "cells": {c: st.CELLS[c] for c in cells}, "SD_pooled": pooled, "SD_pooled_over": sorted(reg),
           "contrasts": {}, "matched_init_gain": {c: gains(store, c) for c in cells},
           "seed_means": {c: seed_means(store, c, METRICS) for c in cells},
           "efficiency": {c: {lab: efficiency("J", s, st.CELLS[c].format(s=s)) for lab, s in st.SEEDS.items()}
                          for c in cells}}
    for (cid, m, c), r in rows.items():
        out["contrasts"].setdefault(cid, {"base": reg[cid][0], "test": reg[cid][1]}).setdefault(c, {})[m] = r
    out["J1"] = {"prediction": "no sign registered (window effect)", "verdict": "reported"}
    out["J2"] = ({"prediction": "J-Tv2 not worse than J-C2v2 by more than 5 %", "verdict": not_worse(S("J2"), 0.05)}
                 if "J2" in reg else {"verdict": "dropped (T* dense)"})
    return out


def render(out, title, ids):
    L = [f"# {title}", "", f"T\\* = {out['T_star']}. Statistics as `PREREG_2026-11-22_R2_TIER1.md` §2.2; "
         f"SD_pooled over {', '.join(out['SD_pooled_over'])}: vertex/sealed {num(out['SD_pooled'].get('vertex/sealed'))}.",
         "", "## Contrasts (sealed)", "", "| ID | base → test | metric | T̄ | per seed (A/B/C) | SE_boot | SE_seed | N_σ | clear |",
         "|---|---|---|---|---|---|---|---|---|"]
    for cid, c0 in out["contrasts"].items():
        for m in ("vertex", "vertex_mlp", "vertex_attentive", "dir_mu", "dir_e", "misid_mu"):
            c = c0["sealed"].get(m) or {}
            ps = "/".join(pct(c.get("per_seed", {}).get(lab)) for lab in "ABC")
            L.append(f"| {cid} | {c0['base']} → {c0['test']} | {m} | {pct(c.get('T_bar'))} | {ps} | {pct(c.get('SE_boot'))} | "
                     f"{pct(c.get('SE_seed'))} | {num(c.get('N_sigma'), 1)} | {c.get('clear')} |")
    L += ["", "## Verdicts", ""] + [f"- **{k}**: {json.dumps(out[k], default=str)}" for k in ids if k in out]
    L += ["", "## Cells (sealed seed means; gain over matched init)", "",
          "| Cell | run id | vertex [m] | MLP | attentive | μ dir | e dir | μ mis-ID | gain |", "|---|---|---|---|---|---|---|---|---|"]
    for c, rid in out["cells"].items():
        s = out["seed_means"][c]
        L.append(f"| {c} | `{rid}` | {num(s['vertex']['seed_mean'])} | {num(s['vertex_mlp']['seed_mean'])} | "
                 f"{num(s['vertex_attentive']['seed_mean'])} | {num(s['dir_mu']['seed_mean'], 2)} | {num(s['dir_e']['seed_mean'], 2)} | "
                 f"{num(s['misid_mu']['seed_mean'])} | {pct(out['matched_init_gain'][c].get('mean_gain'))} |")
    L += ["", "## Efficiency", "", "| Cell | seed | GPU | median s/step | peak alloc GiB (≤ 2,000) |", "|---|---|---|---|---|"]
    for c, e in out["efficiency"].items():
        for lab, x in e.items():
            if x:
                L.append(f"| {c} | {lab} | {x.get('gpu_model')} | {num(x.get('median_step_seconds'), 3)} | "
                         f"{num(x.get('peak_allocated_gib_2000'), 2)} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("hk", "juno"))
    ap.add_argument("--tstar", required=True, choices=tuple(TOK))
    ap.add_argument("--m1")
    ap.add_argument("--mask3")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    if a.stage == "hk":
        out = hk(a)
        (o / "HK_TABLE.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
        (o / "HK_TABLE.md").write_text(render(out, "HK_TABLE — r2-sparse-sk Stage H", ("H1", "H2", "H3", "H4", "claim_wording")))
        print(json.dumps({k: out[k].get("verdict") if isinstance(out[k], dict) else out[k] for k in ("H1", "H2", "H3")}))
    else:
        out = juno(a)
        (o / "JUNO_TABLE.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
        (o / "JUNO_TABLE.md").write_text(render(out, "JUNO_TABLE — r2-sparse-sk Stage J", ("J1", "J2")))
        print(json.dumps({k: out[k]["verdict"] for k in ("J1", "J2")}))


if __name__ == "__main__":
    main()
