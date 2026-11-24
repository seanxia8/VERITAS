#!/usr/bin/env python3
"""r2-mask M2.4: MASK_SCALE.md / .json — gain per configuration and seed, Q1-Q4 and KM (PREREG §1.3-1.5).

Reads ``tier1_eval.py`` outputs of the r2-mask eval tree (``<eval>/<run_id>/step_NNNNN/<cohort>``)
and the M1 numbers (``SHORTCUT_BUDGET.json``). Statistics are those of ``scripts/tier1_gate.py``
(PREREG_2026-11-22 §2.2): per-seed contrast T = e2/e1 − 1, SE_boot from 2,000 paired bootstrap
resamples of sealed events, SE_seed = SD_pooled/√n (SD_pooled = RMS over the detector's registered
contrasts — here every configuration vs ``random`` — of the across-seed SD), N_σ, clear.

Gain over the matched init: G = 1 − e(run, step) / e(init), the init being the step-0 snapshot of
the seed (identical weights for every mask of a seed; DECISIONS 2026-11-23).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EVAL = Path("/ceph/dwong/jepa/returns/r2_mask_2026-11-23/eval")
SEEDS = {"A": 9000, "B": 9001, "C": 9002}
CONFIGS = {"wcte": ("random", "patch8", "patch32", "patch128", "patch256"),
           "sk": ("random", "patch43", "patch173", "patch692")}
DEFAULT = {"wcte": "patch32", "sk": "patch173"}
SEED_LABELS = {"wcte": ("A", "B", "C"), "sk": ("A", "B")}
SECONDARY = ("vertex_mlp", "vertex_mlp_latent", "vertex_attentive", "dir_mu", "dir_e", "energy_e",
             "energy_mu")


def gate():
    spec = importlib.util.spec_from_file_location("tier1_gate", REPO / "scripts" / "tier1_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rid(det, config, label):
    base = f"{det}__mae_repaired__s{SEEDS[label]}"
    if config == "random":
        return base + "__mrandom"
    if config == DEFAULT[det]:
        return base
    return base + f"__ps{int(config[5:])}"


def init_rid(det, label):
    return f"{det}__mae_repaired__s{SEEDS[label]}"


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    return float(spearmanr(x[ok], y[ok]).correlation)


def gains(g, store, det, metric, step):
    """{config: {"per_seed": {label: G}, "mean", "sd", "e": {label: e}, "e_init": {...}}}."""
    out = {}
    for c in CONFIGS[det]:
        per, e_run, e_init = {}, {}, {}
        for lab in SEED_LABELS[det]:
            v = g.values(store, rid(det, c, lab), metric, step)
            v0 = g.values(store, init_rid(det, lab), metric, 0)
            if v is None or v0 is None:
                continue
            e1, e0 = g.point(v), g.point(v0)
            e_run[lab], e_init[lab] = e1, e0
            per[lab] = 1.0 - e1 / e0 if e0 else None
        vals = np.array([x for x in per.values() if x is not None], float)
        out[c] = {"per_seed": per, "e": e_run, "e_init": e_init,
                  "mean": float(vals.mean()) if vals.size else None,
                  "sd": float(vals.std(ddof=1)) if vals.size > 1 else None,
                  "e_mean": float(np.mean(list(e_run.values()))) if e_run else None}
    return out


def contrasts(g, store, det, metric, step, cohort_tag, pairs):
    out = {}
    for name, (c1, c2) in pairs.items():
        p = [(lab, rid(det, c1, lab), rid(det, c2, lab)) for lab in SEED_LABELS[det]]
        r = g.contrast(store, p, metric, cohort_tag, step)
        r.update(detector=det, registered=True, cells=[c1, c2], metric=metric, step=step)
        out[name] = r
    return out


def half_gain_avgdist(cfgs, gain, avgdist):
    """AvgDist at which the gain first reaches half-way from ``random`` to the best configuration
    (piecewise-linear in AvgDist over configurations ordered by AvgDist, up to the best one)."""
    order = sorted(cfgs, key=lambda c: avgdist[c])
    best = max(cfgs, key=lambda c: gain[c])
    g_r, g_b = gain["random"], gain[best]
    if best == "random" or g_b <= g_r:
        return None, best, None
    half = 0.5 * (g_r + g_b)
    path = order[:order.index(best) + 1]
    for a, b in zip(path[:-1], path[1:]):
        if gain[a] < half <= gain[b]:
            f = (half - gain[a]) / (gain[b] - gain[a])
            return float(avgdist[a] + f * (avgdist[b] - avgdist[a])), best, half
    return float(avgdist[path[0]]) if gain[path[0]] >= half else None, best, half


def analyse(args):
    g = gate()
    budget = json.loads(Path(args.budget).read_text())
    out = {"phase": "r2-mask", "stage": "M2", "eval_root": str(args.eval_root), "cohort": "sealed",
           "detectors": {}}
    sealed = g.Store(args.eval_root, cohort="sealed", step=10000)
    val = g.Store(args.eval_root, cohort="validation", step=10000)
    all_contrasts = {}
    for det in ("wcte", "sk"):
        m1 = budget["detectors"][det]["domains"]["photon"]
        U = {c: m1["configs"][c]["U"] for c in CONFIGS[det]}
        avg = {c: m1["configs"][c]["avg_dist_m"] for c in CONFIGS[det]}
        lc = m1["l_c_m"]
        d = {"U": U, "avg_dist_m": avg, "l_c_photon_m": lc, "seeds": list(SEED_LABELS[det])}
        d["gain_vertex"] = gains(g, sealed, det, "vertex", 10000)
        d["gain_vertex_5000"] = gains(g, sealed, det, "vertex", 5000)
        d["gain_vertex_validation"] = gains(g, val, det, "vertex", 10000)
        d["gain_secondary"] = {m: gains(g, sealed, det, m, 10000) for m in SECONDARY}
        pairs = {f"{det}:{c}-vs-random": ("random", c) for c in CONFIGS[det] if c != "random"}
        cs = contrasts(g, sealed, det, "vertex", 10000, "sealed_test", pairs)
        for k, v in cs.items():
            all_contrasts[k] = v
        d["contrasts_vertex"] = cs
        d["contrasts_secondary"] = {
            m: contrasts(g, sealed, det, m, 10000, "sealed_test", pairs) for m in SECONDARY}
        out["detectors"][det] = d
    pooled = g.finish(all_contrasts)
    # secondary metrics: SD_pooled per (detector, metric) over that metric's config-vs-random contrasts
    out["SD_pooled_secondary"] = {}
    for det in ("wcte", "sk"):
        for m, cs in out["detectors"][det]["contrasts_secondary"].items():
            sds = [c["seed_sd"] for c in cs.values() if c.get("seed_sd") is not None]
            sd = float(np.sqrt(np.mean(np.square(sds)))) if sds else None
            out["SD_pooled_secondary"][f"{det}:{m}"] = sd
            for c in cs.values():
                c["registered"] = False
                c["SD_pooled"] = sd
                if "T_bar" in c:
                    c["SE_seed"] = sd / np.sqrt(c["n_seeds"]) if sd is not None else None
                    den = max(c["SE_boot"], c["SE_seed"] or 0.0)
                    c["N_sigma"] = float(abs(c["T_bar"]) / den) if den > 0 else None
                    c["clear"] = bool(c["N_sigma"] is not None and c["N_sigma"] >= 2 and c["same_sign"])
    out["SD_pooled"] = pooled
    v = {}
    # ---------------- Q1 (M1 only)
    q1 = {}
    for det in ("wcte", "sk"):
        cf = budget["detectors"][det]["domains"]["photon"]["configs"]
        patches = sorted([c for c in cf if c.startswith("patch")], key=lambda c: int(c[5:]))
        up = [cf[c]["U"] for c in patches]
        a = cf["parity"]["U"] <= cf["random"]["U"]
        b = all(cf["random"]["U"] < u for u in up)
        mono = all(x < y for x, y in zip(up[:-1], up[1:]))
        q1[det] = {"U_parity": cf["parity"]["U"], "U_random": cf["random"]["U"],
                   "U_patch": dict(zip(patches, up)), "parity_le_random": a,
                   "random_lt_every_patch": b, "patch_increasing_in_s": mono, "holds": bool(a and b and mono)}
    v["Q1"] = q1
    # ---------------- Q2
    q2 = {}
    for det in ("wcte", "sk"):
        d = out["detectors"][det]
        cfgs = [c for c in CONFIGS[det] if d["gain_vertex"][c]["mean"] is not None]
        rho = spearman([d["U"][c] for c in cfgs], [d["gain_vertex"][c]["mean"] for c in cfgs])
        dirs = {}
        for m in ("dir_mu", "dir_e"):
            gg = d["gain_secondary"][m]
            cc = [c for c in cfgs if gg[c]["mean"] is not None]
            dirs[m] = spearman([d["U"][c] for c in cc], [gg[c]["mean"] for c in cc])
        same = all(r is not None and rho is not None and np.sign(r) == np.sign(rho) for r in dirs.values())
        q2[det] = {"configs": cfgs, "spearman_U_vertex_gain": rho, "spearman_U_direction_gain": dirs,
                   "direction_same_sign": bool(same),
                   "holds": bool(rho is not None and rho >= 0.8 and same)}
    v["Q2"] = q2
    # ---------------- Q3
    q3 = {}
    for det in ("wcte", "sk"):
        d = out["detectors"][det]
        cfgs = [c for c in CONFIGS[det] if d["gain_vertex"][c]["mean"] is not None]
        gain = {c: d["gain_vertex"][c]["mean"] for c in cfgs}
        ad, best, half = half_gain_avgdist(cfgs, gain, d["avg_dist_m"]) if "random" in cfgs else (None, None, None)
        ratio = ad / d["l_c_photon_m"] if ad else None
        q3[det] = {"avg_dist_half_gain_m": ad, "best_config": best, "half_gain": half,
                   "l_c_photon_m": d["l_c_photon_m"], "ratio": ratio,
                   "holds": bool(ratio is not None and 1 / 3 <= ratio <= 3)}
    v["Q3"] = q3
    # ---------------- Q4 (SK, patch default vs random, vertex, steps 5,000 and 10,000)
    p = [(lab, rid("sk", "random", lab), rid("sk", DEFAULT["sk"], lab)) for lab in SEED_LABELS["sk"]]
    c10 = g.contrast(sealed, p, "vertex", "sealed_test", 10000)
    c05 = g.contrast(sealed, p, "vertex", "sealed_test", 5000)
    for c in (c10, c05):
        if "T_bar" in c:
            sd = pooled.get("sk")
            c["SE_seed"] = sd / np.sqrt(c["n_seeds"]) if sd else None
            c["N_sigma"] = float(abs(c["T_bar"]) / max(c["SE_boot"], c["SE_seed"] or 0.0))
    # paired difference of the contrast between the steps (bootstrap over the same events)
    q4 = {"T_10000": c10.get("T_bar"), "T_5000": c05.get("T_bar"), "per_seed_10000": c10.get("per_seed"),
          "per_seed_5000": c05.get("per_seed"), "contrast_10000": c10, "contrast_5000": c05}
    if c10.get("T_bar") is not None and c05.get("T_bar") is not None:
        q4["holds"] = bool(-c10["T_bar"] >= -c05["T_bar"])
        # exploratory: paired step difference D = T(10k) - T(5k) per seed; SE_boot over the same
        # resampled sealed events for both steps; SE_seed from the across-seed SD of D
        dd, bt = {}, []
        for lab, r1, r2 in p:
            vals = [g.values(sealed, r, "vertex", st) for r, st in ((r1, 10000), (r2, 10000), (r1, 5000), (r2, 5000))]
            if any(x is None for x in vals):
                continue
            idx = g.boot_idx(len(vals[0]["x"]), f"sealed_test/{vals[0]['tag']}")
            b = [np.asarray(g.boot(x, idx), float) for x in vals]
            dd[lab] = (g.point(vals[1]) / g.point(vals[0])) - (g.point(vals[3]) / g.point(vals[2]))
            bt.append(b[1] / b[0] - b[3] / b[2])
        if dd:
            arr = np.array(list(dd.values()))
            se_b = float(np.std(np.mean(bt, axis=0), ddof=1))
            se_s = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else None
            q4["exploratory_step_difference"] = {
                "D_per_seed": dd, "D_bar": float(arr.mean()), "SE_boot": se_b, "SE_seed": se_s,
                "N_sigma": float(abs(arr.mean()) / max(se_b, se_s or 0.0)),
                "note": "D = T(10k) - T(5k); D > 0 means the patch-vs-random contrast shrank (exploratory)"}
    v["Q4"] = q4
    # ---------------- KM
    sk = out["detectors"]["sk"]
    cfgs = [c for c in CONFIGS["sk"] if sk["gain_vertex"][c]["mean"] is not None]
    if cfgs:
        e = {c: sk["gain_vertex"][c]["e_mean"] for c in cfgs}
        best, worst = min(cfgs, key=e.get), max(cfgs, key=e.get)
        pb = [(lab, rid("sk", worst, lab), rid("sk", best, lab)) for lab in SEED_LABELS["sk"]]
        cbw = g.contrast(sealed, pb, "vertex", "sealed_test", 10000)
        if "T_bar" in cbw:
            sd = pooled.get("sk")
            cbw["SE_seed"] = sd / np.sqrt(cbw["n_seeds"]) if sd else None
            cbw["N_sigma"] = float(abs(cbw["T_bar"]) / max(cbw["SE_boot"], cbw["SE_seed"] or 0.0))
            cbw["clear"] = bool(cbw["N_sigma"] >= 2 and cbw["same_sign"])
        small = (cbw.get("T_bar") is None or abs(cbw["T_bar"]) < 0.05 or not cbw.get("clear"))
        rhos = [q2[dd]["spearman_U_vertex_gain"] for dd in ("wcte", "sk")]
        weak = all(r is not None and r < 0.5 for r in rhos)
        v["KM"] = {"best": best, "worst": worst, "contrast_best_vs_worst": cbw,
                   "sk_difference_small_or_unclear": bool(small), "spearman_below_0p5_both": bool(weak),
                   "fires": bool(small and weak)}
    out["verdicts"] = v
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:+.1f} %"


def render(out):
    v = out["verdicts"]
    md = ["# MASK_SCALE — r2-mask M2 (patch-size sweep)", "",
          "Runbook `RUNBOOK_2026-11-23_R2_MASK_SCALE.md` §M2, PREREG `PREREG_2026-11-23_R2_MASK.md` §1.3–1.5 "
          "(pushed 2c0204b before any M2 evaluation). Primary: mean 3-D vertex error on sealed photon bombs, "
          "frozen spatial-moment Ridge readout (READOUT_SPEC_v1), gain G = 1 − e/e_init over the matched init "
          "(step-0 snapshot of the seed). Contrasts T = e(config)/e(random) − 1 with the statistics of "
          "PREREG_2026-11-22 §2.2. Every cell is `mae_repaired`, default recipe, 50 ns, 10,000 steps; only the "
          "mask differs.", ""]
    for det in ("wcte", "sk"):
        d = out["detectors"][det]
        md += [f"## {det.upper()} (seeds {', '.join(d['seeds'])}; ℓ_c photon {d['l_c_photon_m']:.2f} m)", "",
               "| Config | U (M1) | AvgDist (m) | e_init (m) | e @10k per seed (m) | e @10k mean | G @10k per seed | "
               "G mean | G @5k mean | T vs random | N_σ | clear |",
               "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|"]
        for c in CONFIGS[det]:
            gv, g5 = d["gain_vertex"][c], d["gain_vertex_5000"][c]
            ct = d["contrasts_vertex"].get(f"{det}:{c}-vs-random", {})
            e_init = np.mean(list(gv["e_init"].values())) if gv["e_init"] else None
            md.append(
                f"| {c}{' (default)' if c == DEFAULT[det] else ''} | {d['U'][c]:.4f} | {d['avg_dist_m'][c]:.3f} | "
                f"{'–' if e_init is None else f'{e_init:.4f}'} | "
                f"{', '.join(f'{k} {x:.4f}' for k, x in gv['e'].items()) or '–'} | "
                f"{'–' if gv['e_mean'] is None else f'{gv['e_mean']:.4f}'} | "
                f"{', '.join(f'{k} {pct(x)}' for k, x in gv['per_seed'].items()) or '–'} | {pct(gv['mean'])} | "
                f"{pct(g5['mean'])} | {pct(ct.get('T_bar')) if c != 'random' else '–'} | "
                f"{'–' if ct.get('N_sigma') is None else f'{ct['N_sigma']:.2f}'} | "
                f"{'–' if c == 'random' else ('yes' if ct.get('clear') else 'no')} |")
        md += ["", "Secondary (T vs random at 10k, sealed; N_σ):", "",
               "| Config | " + " | ".join(SECONDARY) + " |", "|---|" + "---:|" * len(SECONDARY)]
        for c in CONFIGS[det]:
            if c == "random":
                continue
            cells = []
            for m in SECONDARY:
                ct = d["contrasts_secondary"][m].get(f"{det}:{c}-vs-random", {})
                cells.append("–" if ct.get("T_bar") is None else
                             f"{pct(ct['T_bar'])} ({ct['N_sigma']:.1f}{'*' if ct.get('clear') else ''})")
            md.append(f"| {c} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Verdicts", ""]
    for det in ("wcte", "sk"):
        q = v["Q1"][det]
        md.append(f"- **Q1 {det.upper()}**: {'holds' if q['holds'] else 'does not hold'} — U parity {q['U_parity']:.4f}, "
                  f"random {q['U_random']:.4f}, patches " + ", ".join(f"{k} {x:.4f}" for k, x in q["U_patch"].items()) +
                  f" (parity ≤ random: {q['parity_le_random']}; random < every patch: {q['random_lt_every_patch']}; "
                  f"increasing in s: {q['patch_increasing_in_s']}).")
    for det in ("wcte", "sk"):
        q = v["Q2"][det]
        rho = q["spearman_U_vertex_gain"]
        md.append(f"- **Q2 {det.upper()}**: {'holds' if q['holds'] else 'does not hold'} — Spearman(U, vertex gain) = "
                  f"{'–' if rho is None else f'{rho:.2f}'} over {len(q['configs'])} configurations; direction gain "
                  + ", ".join(f"{k} {'–' if r is None else f'{r:.2f}'}" for k, r in q["spearman_U_direction_gain"].items())
                  + f" (same sign: {q['direction_same_sign']}).")
    for det in ("wcte", "sk"):
        q = v["Q3"][det]
        md.append(f"- **Q3 {det.upper()}**: {'holds' if q['holds'] else 'does not hold'} — best {q['best_config']}; "
                  f"half-gain AvgDist {'–' if q['avg_dist_half_gain_m'] is None else f'{q['avg_dist_half_gain_m']:.3f} m'}"
                  f" vs ℓ_c {q['l_c_photon_m']:.2f} m (ratio {'–' if q['ratio'] is None else f'{q['ratio']:.2f}'}).")
    q = v["Q4"]
    ps10 = ", ".join(f"{k} {pct(x)}" for k, x in (q["per_seed_10000"] or {}).items())
    ps05 = ", ".join(f"{k} {pct(x)}" for k, x in (q["per_seed_5000"] or {}).items())
    ex = q.get("exploratory_step_difference")
    exs = (f" Exploratory: D = T(10k) − T(5k) = {100 * ex['D_bar']:+.2f} points (SE_boot {100 * ex['SE_boot']:.2f}, "
           f"SE_seed {100 * (ex['SE_seed'] or 0):.2f}, N_σ {ex['N_sigma']:.1f}).") if ex else ""
    md.append(f"- **Q4 SK**: {'holds' if q.get('holds') else 'does not hold'} — T(patch default vs random) "
              f"{pct(q['T_10000'])} at 10k ({ps10}) vs {pct(q['T_5000'])} at 5k ({ps05}).{exs}")
    k = v.get("KM", {})
    if k:
        c = k["contrast_best_vs_worst"]
        md.append(f"- **KM**: {'FIRES' if k['fires'] else 'does not fire'} — SK best {k['best']} vs worst {k['worst']}: "
                  f"T {pct(c.get('T_bar'))}, N_σ {c.get('N_sigma', float('nan')):.2f}, clear {c.get('clear')}; "
                  f"Spearman < 0.5 on both detectors: {k['spearman_below_0p5_both']}.")
    md += ["", f"SD_pooled (RMS over each detector's config-vs-random contrasts of the across-seed SD): "
           + ", ".join(f"{d} {x:.4f}" for d, x in out["SD_pooled"].items()) + ".", ""]
    return "\n".join(md)


def figures(out, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, det in enumerate(("wcte", "sk")):
        d = out["detectors"][det]
        cf = [c for c in CONFIGS[det] if d["gain_vertex"][c]["mean"] is not None]
        ax = axes[0, col]
        for c in cf:
            gv = d["gain_vertex"][c]
            ax.errorbar(d["U"][c], 100 * gv["mean"], yerr=100 * (gv["sd"] or 0), fmt="o", capsize=3)
            ax.annotate(c, (d["U"][c], 100 * gv["mean"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("U = 1 − E_L (M1, photon bombs)")
        ax.set_ylabel("vertex gain over matched init (%)")
        ax.set_title(f"{det.upper()}: gain vs U")
        ax = axes[1, col]
        sizes = {c: (int(c[5:]) if c.startswith("patch") else None) for c in cf}
        for lab in d["seeds"]:
            xs = [sizes[c] for c in cf if sizes[c] and lab in d["gain_vertex"][c]["e"]]
            ys = [d["gain_vertex"][c]["e"][lab] for c in cf if sizes[c] and lab in d["gain_vertex"][c]["e"]]
            ax.plot(xs, ys, "o-", label=f"seed {lab}")
            r = d["gain_vertex"]["random"]["e"].get(lab)
            if r is not None:
                ax.axhline(r, ls=":", lw=1, color=ax.lines[-1].get_color())
        ax.set_xscale("log")
        ax.set_xlabel("patch size (PMTs); dotted: random mask")
        ax.set_ylabel("sealed vertex error @10k (m)")
        ax.set_title(f"{det.upper()}: error vs patch size")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "MASK_SCALE.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=EVAL)
    ap.add_argument("--budget", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    out = analyse(a)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "MASK_SCALE.json").write_text(json.dumps(out, indent=1, default=float))
    (a.out_dir / "MASK_SCALE.md").write_text(render(out) + "\nFigure: `MASK_SCALE.png` (top: vertex gain vs U; "
                                             "bottom: sealed vertex error vs patch size, per seed).\n")
    figures(out, a.out_dir)
    print(render(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
