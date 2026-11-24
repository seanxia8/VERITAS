#!/usr/bin/env python3
"""r2-follow T3: FOLLOW_TABLE.{json,md} (PREREG_2026-11-24_R2_FOLLOW.md §2.2-2.5).

Part A (WCTE, lucid_v1_w2, seeds 9000/9001/9002): seed-mean errors of C2, C3e, C2dc, C3; contrasts A1-A4 with the
r2-sparse statistics unchanged (``sparse_table.contrast`` / ``finish``: T = e_test / e_base - 1 per seed, SE_boot
from 2,000 paired resamples of sealed events, SE_seed = SD_pooled / sqrt(3), N_sigma, clear = N_sigma >= 2 and one
sign in all three seeds), SD_pooled recomputed per metric over r2-sparse's registered contrasts R1-R6 (incl. R2b)
and A1-A4; the attribution verdict (§2.3); secondary probes, matched-init gains; the §2.4 diagnostics.
C2 and C3 (and the other r2-sparse cells) are read from the r2-sparse eval tree, C3e and C2dc from this phase's.

Part B (SK): reads ``MASK_SCALE_3seed.json`` (``mask_scale_analysis.py --sk-seeds A B C``) for the gains over
matched init and the patch-vs-random contrasts at 10,000 steps; adds the patch-vs-random contrasts at 5,000 steps
(same statistics, SD_pooled over the step-5,000 contrasts); B1-B3 and the kill rule (§2.5).

Reads only evaluator outputs (no pack is opened).

    follow_table.py --diagnostics DIAGNOSTICS_A.json --mask3 MASK_SCALE_3seed.json --sk-eval-root <union> \
        --out-dir results/round2/20261124_follow
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
import tier1_gate as tg  # noqa: E402

SPARSE_EVAL = Path("/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/eval")
FOLLOW_EVAL = Path("/ceph/dwong/jepa/returns/r2_follow_2026-11-24/eval")
st.CELLS["C3e"] = "wcte__mae_repaired__s{s}__dw2__lfixed__psent__efourier"
st.CELLS["C2dc"] = "wcte__mae_repaired__s{s}__dw2__lfixed__psent__disol__cnt"
PART_A = {"A1": ("C2", "C3e"), "A2": ("C2", "C2dc"), "A3": ("C3e", "C3"), "A4": ("C2dc", "C3")}
MEANING = {"A1": "the new embedding alone", "A2": "the query-isolated decoder and count head alone",
           "A3": "decoder and count head, given the new embedding", "A4": "the new embedding, given the new decoder"}
R2B_REGISTERED = 0.323          # SPARSE_REPORT / SPARSE_TABLE (C3 vs C2, sealed vertex, 3 seeds)
HALF_R2B = 0.162                # §2.3: "at least half of the registered R2b (+32.3 %, i.e. T̄ >= +16.2 %)"
A_CELLS = ("C2", "C3e", "C2dc", "C3")
SK_CFG = ("random", "patch43", "patch173", "patch692")
SECONDARY_A = ("vertex_mlp", "vertex_attentive", "dir_mu", "dir_e")
RHO_MIN = 0.8 - 1e-9          # Spearman on 4 points takes multiples of 0.2; 0.8 may be 0.7999... in floating point


class MultiStore(st.Store):
    """sparse_table.Store over two eval trees: a run is read from the first tree that has it."""

    def __init__(self, roots):
        super().__init__(roots[0])
        self.stores = [st.Store(r) for r in roots]

    def get(self, rid, step, cohort):
        for s in self.stores:
            got = s.get(rid, step, cohort)
            if got is not None:
                return got
        return None


def part_a(diag):
    store = MultiStore([SPARSE_EVAL, FOLLOW_EVAL])
    rows = {}
    reg = {**st.REGISTERED, **PART_A}
    for cohort in ("sealed", "validation"):
        for cid, (b, t) in reg.items():
            for metric in ("vertex",) + SECONDARY_A:
                rows[(cid, metric, cohort)] = st.contrast(store, b, t, metric, cohort)
    pooled = st.finish(rows, set(reg))
    out = {"SD_pooled": pooled, "SD_pooled_over": sorted(reg), "contrasts": {}, "cells": {}, "matched_init_gain": {}}
    for (cid, metric, cohort), c in rows.items():
        if cid in PART_A:
            out["contrasts"].setdefault(cid, {"base": PART_A[cid][0], "test": PART_A[cid][1],
                                              "measures": MEANING[cid]}).setdefault(cohort, {})[metric] = c
    out["r2_sparse_recomputed"] = {cid: rows[(cid, "vertex", "sealed")] for cid in st.REGISTERED}
    for cell in A_CELLS:
        d = {}
        for cohort in ("sealed", "validation"):
            d[cohort] = {m: dict(zip(("seed_mean", "n_seeds"), st.seed_mean(store, cell, m, cohort)))
                         for m in ("vertex",) + SECONDARY_A}
            per = {}
            for lab, s in st.SEEDS.items():
                v = store.values(st.CELLS[cell].format(s=s), 10000, cohort, "vertex")
                per[lab] = st.point(v) if v is not None else None
            d[cohort]["vertex_per_seed"] = per
        out["cells"][cell] = d
        g = {}
        for lab, s in st.SEEDS.items():
            rid = st.CELLS[cell].format(s=s)
            v0, v1 = store.values(rid, 0, "sealed", "vertex"), store.values(rid, 10000, "sealed", "vertex")
            if v0 is not None and v1 is not None:
                g[lab] = {"init_m": st.point(v0), "final_m": st.point(v1), "gain": 1.0 - st.point(v1) / st.point(v0)}
        out["matched_init_gain"][cell] = g
    # attribution rule (§2.3)
    S = lambda cid: out["contrasts"][cid]["sealed"]["vertex"]  # noqa: E731
    accounts = lambda c: bool("T_bar" in c and c["clear"] and c["T_bar"] >= HALF_R2B)  # noqa: E731
    comp = {"embedding (A1)": accounts(S("A1")), "decoder + count head (A2)": accounts(S("A2"))}
    inter = {cid: accounts(S(cid)) for cid in ("A3", "A4")}
    clear = {cid: bool(S(cid).get("clear")) for cid in PART_A}
    who = [k for k, v in comp.items() if v]
    if who:
        verdict = " and ".join(who) + (" accounts" if len(who) == 1 else " each account") + " for the C3 loss"
    elif any(inter.values()):
        verdict = "interaction: neither single change accounts for the C3 loss, but " + \
                  ", ".join(k for k, v in inter.items() if v) + " does"
    elif not any(clear.values()):
        verdict = "R2b not reproduced in parts (no contrast A1-A4 is clear)"
    else:
        verdict = ("no component accounts for the C3 loss by the registered rule; clear contrasts: " +
                   ", ".join(f"{k} ({100 * S(k)['T_bar']:+.1f} %)" for k, v in clear.items() if v))
    out["attribution"] = {"rule": "§2.3: a component accounts for the C3 loss if its single-change contrast "
                                  "(A1 embedding, A2 decoder + count head) is clear and T̄ >= +16.2 % (half of the "
                                  "registered R2b +32.3 %); else interaction if A3 or A4 meets that; else, if none "
                                  "is clear, R2b is reported as not reproduced in parts.",
                          "threshold": HALF_R2B, "single_change_accounts": comp, "conditional_accounts": inter,
                          "clear": clear, "verdict": verdict}
    out["diagnostics"] = diag
    return out


def part_b(mask3, sk_root):
    sk = mask3["detectors"]["sk"]
    g = tg
    store = g.Store(sk_root, cohort="sealed", step=10000)
    labels = sk["seeds"]
    rid = lambda c, lab: (f"sk__mae_repaired__s{st.SEEDS[lab]}" + ("__mrandom" if c == "random" else  # noqa: E731
                          "" if c == "patch173" else f"__ps{int(c[5:])}"))
    c05 = {}
    for c in SK_CFG[1:]:
        r = g.contrast(store, [(lab, rid("random", lab), rid(c, lab)) for lab in labels], "vertex", "sealed_test", 5000)
        c05[f"sk:{c}-vs-random"] = r
    sds = [c["seed_sd"] for c in c05.values() if c.get("seed_sd") is not None]
    sd5 = float(np.sqrt(np.mean(np.square(sds)))) if sds else None
    for c in c05.values():
        if "T_bar" in c:
            c["SD_pooled"] = sd5
            c["SE_seed"] = sd5 / np.sqrt(c["n_seeds"]) if sd5 else None
            den = max(c["SE_boot"], c["SE_seed"] or 0.0)
            c["N_sigma"] = float(abs(c["T_bar"]) / den) if den > 0 else None
            c["clear"] = bool(c["N_sigma"] is not None and c["N_sigma"] >= 2 and c["same_sign"] and c["n_seeds"] == 3)
    c10 = sk["contrasts_vertex"]
    b1 = {k: {"T_bar": v.get("T_bar"), "per_seed": v.get("per_seed"), "N_sigma": v.get("N_sigma"),
              "clear": v.get("clear"), "n_seeds": v.get("n_seeds"),
              "holds": bool(v.get("T_bar") is not None and v["T_bar"] < 0 and v.get("clear") and v.get("n_seeds") == 3)}
          for k, v in c10.items()}
    U = sk["U"]
    from mask_scale_analysis import spearman
    gv = sk["gain_vertex"]
    rho_v = spearman([U[c] for c in SK_CFG], [gv[c]["mean"] for c in SK_CFG])
    rho_d = {m: spearman([U[c] for c in SK_CFG], [sk["gain_secondary"][m][c]["mean"] for c in SK_CFG])
             for m in ("dir_mu", "dir_e")}
    b2 = {"spearman_U_vertex_gain": rho_v, "spearman_U_direction_gain": rho_d,
          "holds": bool(rho_v is not None and rho_v >= RHO_MIN and all(r is not None and r >= RHO_MIN for r in rho_d.values()))}
    gc = {c: gv[c]["per_seed"].get("C") for c in SK_CFG}
    rho_c = spearman([U[c] for c in SK_CFG], [gc[c] for c in SK_CFG]) if all(x is not None for x in gc.values()) else None
    rho_c5 = None
    g5 = sk.get("gain_vertex_5000", {})
    if g5:
        gc5 = {c: g5[c]["per_seed"].get("C") for c in SK_CFG}
        if all(x is not None for x in gc5.values()):
            rho_c5 = spearman([U[c] for c in SK_CFG], [gc5[c] for c in SK_CFG])
    b3 = {"seed_C_gain_vertex": gc, "spearman_U_seedC_gain": rho_c, "spearman_U_seedC_gain_5000_exploratory": rho_c5,
          "holds": bool(rho_c is not None and rho_c >= RHO_MIN)}
    b1_all = all(v["holds"] for v in b1.values()) and len(b1) == 3
    kill = (not b1_all) or (not b2["holds"])
    return {"seeds": labels, "U": U, "gain_vertex_10000": gv, "gain_vertex_5000": sk.get("gain_vertex_5000"),
            "gain_direction_10000": {m: sk["gain_secondary"][m] for m in ("dir_mu", "dir_e")},
            "contrasts_vertex_10000": c10, "contrasts_vertex_5000": c05, "SD_pooled_10000": mask3["SD_pooled"].get("sk"),
            "SD_pooled_5000": sd5, "B1": {"per_config": b1, "holds": b1_all}, "B2": b2, "B3": b3,
            "kill_fires": bool(kill),
            "verdict": ("SK mask claim not reproduced with a third seed (kill fires)" if kill else
                        "SK mask claim holds with three seeds (B1 and B2 hold)")}


def pct(x, nd=1):
    return "—" if x is None else f"{100 * x:+.{nd}f} %"


def num(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def render(out):
    A, B = out["part_a"], out["part_b"]
    L = ["# FOLLOW_TABLE — r2-follow (PREREG_2026-11-24_R2_FOLLOW.md)", "",
         "Primary: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge (READOUT_SPEC_v1 A.2), "
         "step 10,000. T = e_test / e_base − 1 (negative = test better). SE_boot: paired bootstrap of the seed-mean "
         "contrast (2,000 resamples of sealed events); SE_seed = SD_pooled/√3; N_σ = |T̄|/max(SE_boot, SE_seed); "
         "clear = N_σ ≥ 2 and one sign in all three seeds.", "",
         "## Part A — WCTE attribution (lucid_v1_w2; seeds A/B/C = 9000/9001/9002)", "",
         "| Cell | vertex [m] seed mean | A / B / C | MLP | attentive | μ dir [°] | e dir [°] | gain vs own init |",
         "|---|---|---|---|---|---|---|---|"]
    for cell in A_CELLS:
        s = A["cells"][cell]["sealed"]
        g = [v["gain"] for v in A["matched_init_gain"][cell].values()]
        L.append(f"| {cell} | {num(s['vertex']['seed_mean'])} | " +
                 " / ".join(num(s["vertex_per_seed"].get(k)) for k in st.SEEDS) +
                 f" | {num(s['vertex_mlp']['seed_mean'])} | {num(s['vertex_attentive']['seed_mean'])} | "
                 f"{num(s['dir_mu']['seed_mean'], 2)} | {num(s['dir_e']['seed_mean'], 2)} | "
                 f"{pct(float(np.mean(g))) if g else '—'} |")
    L += ["", "| ID | Contrast | measures | T̄ | A / B / C | SE_boot | SE_seed | N_σ | clear |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cid in PART_A:
        c = A["contrasts"][cid]["sealed"]["vertex"]
        b, t = PART_A[cid]
        L.append(f"| {cid} | {t} vs {b} | {MEANING[cid]} | {pct(c.get('T_bar'))} | " +
                 " / ".join(pct(c["per_seed"].get(k)) for k in st.SEEDS) +
                 f" | {pct(c.get('SE_boot'), 2)} | {pct(c.get('SE_seed'), 2)} | {num(c.get('N_sigma'), 2)} | "
                 f"{'yes' if c.get('clear') else 'no'} |")
    r2b = A["r2_sparse_recomputed"]["R2b"]
    L += ["", f"R2b (C3 vs C2) recomputed with the new SD_pooled: T̄ {pct(r2b.get('T_bar'))}, N_σ "
          f"{num(r2b.get('N_sigma'), 2)}, clear {r2b.get('clear')} (registered +32.3 %). SD_pooled (vertex, sealed) = "
          f"{num(A['SD_pooled'].get('vertex/sealed'))} over {', '.join(A['SD_pooled_over'])}.", "",
          f"**Attribution verdict (§2.3): {A['attribution']['verdict']}.**", "",
          "Secondary contrasts (sealed; T̄, N_σ, clear):", "",
          "| ID | MLP vertex | attentive vertex | μ dir | e dir | vertex (validation) |", "|---|---|---|---|---|---|"]
    for cid in PART_A:
        cs = A["contrasts"][cid]

        def cell(m, cohort="sealed"):
            c = cs[cohort][m]
            return "—" if "T_bar" not in c else f"{pct(c['T_bar'])} ({num(c['N_sigma'], 1)}{', clear' if c['clear'] else ''})"
        L.append(f"| {cid} | {cell('vertex_mlp')} | {cell('vertex_attentive')} | {cell('dir_mu')} | {cell('dir_e')} | "
                 f"{cell('vertex', 'validation')} |")
    D = (A.get("diagnostics") or {}).get("results")
    if D:
        L += ["", "§2.4 diagnostics (exploratory; seed 9000; dev_probe_test; every event fully visible):", "",
              "| Cell | step | input RMS positional / content (hit tokens) | block-4 variance explained: geometry / content / joint (all tokens) | same, hit tokens | PMT-position probe R² | vertex R² (frozen readout, validation) |",
              "|---|---|---|---|---|---|---|"]
        for cell in A_CELLS:
            for step, r in sorted((D.get(cell) or {}).items(), key=lambda x: int(x[0])):
                ve, ir = r["block4_variance_explained"], r["input_rms"]
                vx = (r.get("frozen_readout_vertex") or {}).get("r2")
                L.append(f"| {cell} | {step} | {ir['pos_rms']:.3f} / {ir['content_rms']:.3f} ({ir['pos_rms_hit']:.3f} / "
                         f"{ir['content_rms_hit']:.3f}) | {ve['geometry_all']:.3f} / {ve['content_all']:.3f} / "
                         f"{ve['joint_all']:.3f} | {ve['geometry_hit']:.3f} / {ve['content_hit']:.3f} / {ve['joint_hit']:.3f} | "
                         f"{r['probe_pmt_position_r2']:.4f} | {num(vx)} |")
    L += ["", f"## Part B — SK, three seeds (lucid_v1; seeds {', '.join(B['seeds'])})", "",
          "| Config | U (M1) | G @10k A / B / C | G @10k mean | G @5k A / B / C | G @5k mean | T vs random @10k (N_σ, clear) | T vs random @5k (N_σ, clear) |",
          "|---|---|---|---|---|---|---|---|"]
    for c in SK_CFG:
        g10, g5 = B["gain_vertex_10000"][c], (B["gain_vertex_5000"] or {}).get(c, {})
        k = f"sk:{c}-vs-random"
        t10, t5 = B["contrasts_vertex_10000"].get(k), B["contrasts_vertex_5000"].get(k)
        f = lambda t: "—" if not t or "T_bar" not in t else f"{pct(t['T_bar'])} ({num(t['N_sigma'], 1)}{', clear' if t['clear'] else ''})"  # noqa: E731
        L.append(f"| {c} | {B['U'][c]:.4f} | " + " / ".join(pct(g10['per_seed'].get(s)) for s in B["seeds"]) +
                 f" | {pct(g10['mean'])} | " + " / ".join(pct((g5.get('per_seed') or {}).get(s)) for s in B["seeds"]) +
                 f" | {pct(g5.get('mean'))} | {f(t10)} | {f(t5)} |")
    b2 = B["B2"]
    L += ["", "| ID | Statement | Result | Holds |", "|---|---|---|---|",
          "| B1 | every patch size beats random (T̄ < 0, clear, 3 seeds) | " +
          "; ".join(f"{k.split(':')[1].split('-')[0]} {pct(v['T_bar'])} (N_σ {num(v['N_sigma'], 1)}, clear {v['clear']})"
                    for k, v in B["B1"]["per_config"].items()) + f" | **{B['B1']['holds']}** |",
          f"| B2 | U predicts the gain (ρ ≥ 0.8; μ and e direction ρ ≥ 0.8) | ρ vertex {num(b2['spearman_U_vertex_gain'], 2)}; "
          f"μ {num(b2['spearman_U_direction_gain']['dir_mu'], 2)}; e {num(b2['spearman_U_direction_gain']['dir_e'], 2)} | **{b2['holds']}** |",
          f"| B3 | ordering reproduced by seed C alone (ρ ≥ 0.8) | ρ {num(B['B3']['spearman_U_seedC_gain'], 2)} | **{B['B3']['holds']}** |",
          "", f"**Part B verdict: {B['verdict']}.** Figure: `results/round2/20261123_mask/MASK_SCALE_3seed.png`.", "",
          "Everything outside A1–A4, the attribution rule and B1–B3 is secondary or exploratory."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", type=Path)
    ap.add_argument("--mask3", type=Path, required=True)
    ap.add_argument("--sk-eval-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    diag = json.loads(a.diagnostics.read_text()) if a.diagnostics and a.diagnostics.is_file() else None
    out = {"preregistration": "docs/round2/PREREG_2026-11-24_R2_FOLLOW.md",
           "eval_roots": {"wcte": [str(SPARSE_EVAL), str(FOLLOW_EVAL)], "sk": str(a.sk_eval_root)},
           "part_a": part_a(diag), "part_b": part_b(json.loads(a.mask3.read_text()), a.sk_eval_root)}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "FOLLOW_TABLE.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    (a.out_dir / "FOLLOW_TABLE.md").write_text(render(out))
    print(out["part_a"]["attribution"]["verdict"])
    print(out["part_b"]["verdict"])


if __name__ == "__main__":
    main()
