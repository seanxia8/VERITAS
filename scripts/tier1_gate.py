#!/usr/bin/env python3
"""r2-tier1 T7.2/T7.4: registered contrasts, statistics and verdicts (PREREG §2.2-§2.5).

Reads ``tier1_eval.py`` outputs (``<eval>/<run_id>/step_<NNNNN>/<cohort>.{json,npz}``) and writes
``GATE_TABLE.json`` / ``.md`` (or ``INTERIM_E1.*`` with ``--interim``). Statistics exactly as
registered: per-seed contrast T = e2/e1 - 1; SE_boot = SD over 2,000 paired bootstrap resamples (the
same resample of sealed events for every run) of the seed-mean contrast; SE_seed = SD_pooled /
sqrt(n_seeds), SD_pooled = RMS over the detector's registered contrasts of the across-seed SD;
N_sigma = |T̄| / max(SE_boot, SE_seed); clear = N_sigma >= 2 and the same sign in every seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EVAL = Path("/ceph/dwong/jepa/returns/r2_tier1_2026-11-22/eval")
N_BOOT, BOOT_SEED = 2000, 20261122
SEEDS = {"A": 9000, "B": 9001, "C": 9002}
U50 = {"wcte": 6.50, "sk": 62.64, "juno": 59.65, "hk": 149.13}


def run_id(det, mask, unit, seed, cw=None):
    rid = f"{det}__mae_repaired__s{seed}"
    if mask == "random":
        rid += "__mrandom"
    elif mask == "parity":
        rid += "__mwithin"
    if float(unit) != 50.0:
        rid += f"__ts{float(unit):g}".replace(".", "p")
    if cw:
        rid += f"__cw{float(cw):g}".replace(".", "p")
    return rid


class Store:
    def __init__(self, root, cohort="sealed", step=10000):
        self.root, self.cohort, self.step = Path(root), cohort, step
        self._cache = {}

    def get(self, rid, step=None):
        step = self.step if step is None else step
        key = (rid, step)
        if key not in self._cache:
            base = self.root / rid / f"step_{step:05d}" / self.cohort
            if not base.with_suffix(".json").is_file():
                self._cache[key] = None
            else:
                js = json.loads(base.with_suffix(".json").read_text())
                with np.load(base.with_suffix(".npz"), allow_pickle=False) as z:
                    arr = {k: z[k] for k in z.files if not k.startswith("boot_")}
                self._cache[key] = (js, arr)
        return self._cache[key]


def boot_idx(n, tag):
    seed = BOOT_SEED + int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16) % 10_000
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


# metric -> (array key, statistic, bootstrap tag suffix, event-id key)
METRICS = {
    "vertex": ("err_ridge_spatial_moment_latent", "mean", "photon"),
    "vertex_mlp": ("err_mlp_spatial_moment_full_mean3", "mean", "photon"),
    "vertex_mlp_latent": ("err_mlp_spatial_moment_latent_mean3", "mean", "photon"),
    "vertex_attentive": ("err_attentive_mean3", "mean", "photon"),
    "dir_mu": ("ang_muon_ridge_spatial", "median", "muon"),
    "dir_e": ("ang_electron_ridge_spatial", "median", "electron"),
    "energy_e": ("rel_energy_electron", "hw68", "electron_energy"),
    "energy_mu": ("rel_energy_muon", "hw68", "muon_energy"),
    "misid_mu": ("pid", "misid", "pid"),
}


def stat(x, kind):
    if kind == "mean":
        return x.mean(axis=-1)
    if kind == "median":
        return np.median(x, axis=-1)
    if kind == "hw68":
        q = np.quantile(x, [0.16, 0.84], axis=-1)
        return 0.5 * (q[1] - q[0])
    raise ValueError(kind)


def misid_stat(p, is_mu):
    """mu mis-ID at 99 % e efficiency, vectorised over resample rows."""
    p = np.atleast_2d(p)
    is_mu = np.atleast_2d(is_mu)
    out = []
    for pr, mr in zip(p, is_mu):
        e = pr[~mr]
        thr = np.quantile(e, 0.99)
        out.append(np.mean(pr[mr] <= thr))
    return np.asarray(out)


def values(store, rid, metric, step=None):
    """-> (point value, per-event array or None, aligned ids) for a run."""
    got = store.get(rid, step)
    if got is None:
        return None
    js, arr = got
    key, kind, tag = METRICS[metric]
    if metric == "misid_mu":
        if "pid_p_mu" not in arr:
            return None
        return {"kind": kind, "p": arr["pid_p_mu"], "is_mu": arr["pid_is_mu"].astype(bool), "tag": tag}
    if key not in arr:
        return None
    return {"kind": kind, "x": arr[key], "tag": tag}


def point(v):
    if v["kind"] == "misid":
        return float(misid_stat(v["p"], v["is_mu"])[0])
    return float(stat(v["x"], v["kind"]))


def boot(v, idx):
    if v["kind"] == "misid":
        return misid_stat(v["p"][idx], v["is_mu"][idx])
    return stat(v["x"][idx], v["kind"])


def contrast(store, pairs, metric, cohort_tag, step=None):
    """pairs = [(rid_1, rid_2)] per seed; T = e2/e1 - 1. Returns per-seed values and bootstrap rows."""
    per_seed, boots, missing, undefined = {}, [], [], {}
    for label, r1, r2 in pairs:
        v1, v2 = values(store, r1, metric, step), values(store, r2, metric, step)
        if v1 is None or v2 is None:
            missing.append(r1 if v1 is None else r2)
            continue
        n = len(v1["p"] if v1["kind"] == "misid" else v1["x"])
        n2 = len(v2["p"] if v2["kind"] == "misid" else v2["x"])
        if n != n2:
            raise ValueError(f"unaligned events {r1} vs {r2}")
        idx = boot_idx(n, f"{cohort_tag}/{v1['tag']}")
        p1, p2 = point(v1), point(v2)
        if p1 == 0.0:  # ratio undefined (e.g. no mis-identified muon in the baseline cell)
            undefined[label] = {"value_1": p1, "value_2": p2}
            continue
        per_seed[label] = p2 / p1 - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = np.asarray(boot(v1, idx), float)
            b = np.asarray(boot(v2, idx), float) / np.where(b1 == 0, np.nan, b1) - 1.0
        boots.append(b)
    out = {"per_seed": per_seed, "missing": sorted(set(missing))}
    if undefined:
        out["undefined_zero_baseline"] = undefined
    if per_seed:
        vals = np.array(list(per_seed.values()))
        out["T_bar"] = float(vals.mean())
        out["seed_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else None
        out["SE_boot"] = float(np.nanstd(np.nanmean(boots, axis=0), ddof=1))
        out["same_sign"] = bool(np.all(vals < 0) or np.all(vals > 0))
        out["n_seeds"] = int(vals.size)
    return out


def finish(contrasts):
    """SE_seed (pooled per detector over registered contrasts), N_sigma, clear."""
    by_det = {}
    for c in contrasts.values():
        if c.get("registered") and c.get("seed_sd") is not None:
            by_det.setdefault(c["detector"], []).append(c["seed_sd"])
    pooled = {d: float(np.sqrt(np.mean(np.square(v)))) for d, v in by_det.items()}
    for c in contrasts.values():
        if "T_bar" not in c:
            c.update(N_sigma=None, clear=False)
            continue
        sd = pooled.get(c["detector"])
        c["SD_pooled"] = sd
        c["SE_seed"] = sd / np.sqrt(c["n_seeds"]) if sd is not None else None
        den = max(c["SE_boot"], c["SE_seed"] or 0.0)
        c["N_sigma"] = float(abs(c["T_bar"]) / den) if den > 0 else None
        c["clear"] = bool(c["N_sigma"] is not None and c["N_sigma"] >= 2 and c["same_sign"])
    return pooled


def registered(seeds_sk=("A", "B"), seeds_wc=("A", "B", "C")):
    """ID -> (detector, list of (seed label, run1, run2)) for every registered contrast."""
    R = {}
    sk = lambda m, u, s: run_id("sk", m, u, SEEDS[s])
    wc = lambda m, u, s: run_id("wcte", m, u, SEEDS[s])
    R["CT-p50"] = ("sk", [(s, sk("patch", 450, s), sk("patch", 50, s)) for s in seeds_sk])
    R["CT-p5"] = ("sk", [(s, sk("patch", 450, s), sk("patch", 5, s)) for s in seeds_sk])
    R["CT-r50"] = ("sk", [(s, sk("random", 450, s), sk("random", 50, s)) for s in seeds_sk])
    R["CM-450"] = ("sk", [(s, sk("random", 450, s), sk("patch", 450, s)) for s in seeds_sk])
    R["CM-50"] = ("sk", [(s, sk("random", 50, s), sk("patch", 50, s)) for s in seeds_sk])
    for m, tag in (("patch", "WT-p"), ("random", "WT-r")):
        for u in (50, 5, 1.5):
            R[f"{tag}:450->{u:g}"] = ("wcte", [(s, wc(m, 450, s), wc(m, u, s)) for s in seeds_wc])
    for u in (450, 5):
        R[f"WM:{u:g}"] = ("wcte", [(s, wc("random", u, s), wc("patch", u, s)) for s in seeds_wc])
    return R


def seeds_available(store, det, cells, labels):
    return [s for s in labels if all(store.get(run_id(det, m, u, SEEDS[s])) is not None for m, u in cells)]


def build(args):
    store = Store(args.eval_root, "sealed")
    out = {"cohort": "sealed_test", "step": 10000, "registered": {}, "secondary": {}, "verdicts": {},
           "notes": [], "u50": U50}
    if getattr(args, "k4", None):
        out["K4"] = json.loads(Path(args.k4).read_text())
    # seed C for SK only when the borderline rule triggered it
    sk_seeds = ["A", "B"] + (["C"] if store.get(run_id("sk", "patch", 450, 9002)) else [])
    reg = registered(seeds_sk=tuple(sk_seeds))
    if args.interim:
        reg = {k: v for k, v in reg.items() if k in ("CT-p50", "CT-r50", "CM-450", "CM-50")}
    tags = {"sk": "sealed_test", "wcte": "sealed_test"}
    for cid, (det, pairs) in reg.items():
        c = contrast(store, pairs, "vertex", tags[det])
        c.update(detector=det, registered=True, runs=[[s, a, b] for s, a, b in pairs])
        out["registered"][cid] = c
    out["SD_pooled"] = finish(out["registered"])
    # secondary outcomes (same contrasts), probe variants
    for metric in ("dir_mu", "dir_e", "energy_e", "energy_mu", "misid_mu", "vertex_mlp", "vertex_mlp_latent",
                   "vertex_attentive"):
        sec = {}
        for cid, (det, pairs) in reg.items():
            if det == "wcte" and metric.startswith("energy"):
                continue  # WCTE energy is not evaluable under LUCiD's containment flag (D13)
            c = contrast(store, pairs, metric, tags[det])
            c.update(detector=det, registered=True)
            sec[cid] = c
        finish(sec)
        out["secondary"][metric] = sec
    if not args.interim:
        verdicts(store, out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    Path(args.out).with_suffix(".md").write_text(render(out, args.interim))
    print("wrote", args.out)


def seed_mean_e(store, det, mask, unit, labels, step=None):
    vals = [values(store, run_id(det, mask, unit, SEEDS[s]), "vertex", step) for s in labels]
    vals = [v for v in vals if v is not None]
    return (float(np.mean([point(v) for v in vals])), len(vals)) if vals else (None, 0)


def adjacent(store, det, mask, u1, u2, labels):
    pairs = [(s, run_id(det, mask, u1, SEEDS[s]), run_id(det, mask, u2, SEEDS[s])) for s in labels]
    c = contrast(store, pairs, "vertex", "sealed_test")
    if "T_bar" in c:
        c["N_sigma"] = abs(c["T_bar"]) / c["SE_boot"] if c["SE_boot"] > 0 else None
    return c


def verdicts(store, out):
    R, S, V = out["registered"], out["secondary"], out["verdicts"]
    g = lambda cid: R.get(cid, {})
    # P1 sign
    ids = ["CT-p50", "CT-p5", "CT-r50"] + [k for k in R if k.startswith("WT-")]
    have = [k for k in ids if "T_bar" in R.get(k, {})]
    V["P1"] = {"holds": all(R[k]["T_bar"] < 0 for k in have), "contrasts": {k: R[k]["T_bar"] for k in have},
               "missing": [k for k in ids if k not in have]}
    # P2 monotone on SK patch 450 -> 150 -> 50 -> 15 -> 5
    units = [450, 150, 50, 15, 5]
    curve = {u: seed_mean_e(store, "sk", "patch", u, ["A", "B", "C"]) for u in units}
    reversals = []
    for u1, u2 in zip(units, units[1:]):
        labels = [s for s in ("A", "B", "C") if store.get(run_id("sk", "patch", u1, SEEDS[s])) and
                  store.get(run_id("sk", "patch", u2, SEEDS[s]))]
        if not labels:
            continue
        c = adjacent(store, "sk", "patch", u1, u2, labels)
        if c.get("T_bar") is not None and c["T_bar"] >= 0.03 and (c.get("N_sigma") or 0) >= 2 and c["same_sign"]:
            reversals.append({"from": u1, "to": u2, **{k: c[k] for k in ("T_bar", "N_sigma", "per_seed")}})
    total = (curve[5][0] / curve[450][0] - 1.0) if curve[5][0] and curve[450][0] else None
    V["P2"] = {"holds": not reversals, "clear_reversals": reversals,
               "curve_seed_mean_m": {str(u): curve[u][0] for u in units},
               "n_seeds": {str(u): curve[u][1] for u in units}, "total_reduction_450_to_5": total}
    # P3 trade-off: electron energy resolution worsens 450 -> 5 (CT-p5 on energy_e positive and clear)
    ee, em = S["energy_e"].get("CT-p5", {}), S["energy_mu"].get("CT-p5", {})
    V["P3"] = {"holds": bool(ee.get("T_bar") is not None and ee["T_bar"] > 0 and ee.get("clear")
                             and em.get("T_bar") is not None and em["T_bar"] > 0),
               "electron": {k: ee.get(k) for k in ("T_bar", "N_sigma", "clear", "per_seed")},
               "muon": {k: em.get(k) for k in ("T_bar", "N_sigma", "clear", "per_seed")}}
    # P4 knee
    knee = {}
    for det, mask, tested in (("sk", "patch", [450, 150, 50, 15, 5, 1.5]), ("wcte", "patch", [450, 50, 15, 5, 1.5])):
        e = {u: seed_mean_e(store, det, mask, u, ["A", "B", "C"])[0] for u in tested}
        e = {u: v for u, v in e.items() if v is not None}
        if 450 not in e or len(e) < 3:
            knee[det] = {"u_half": None, "curve": e}
            continue
        finest = min(e)
        tot = e[450] - e[finest]
        u_half = max((u for u in e if e[450] - e[u] >= 0.5 * tot), default=None) if tot > 0 else None
        lo, hi = U50[det] / 3, 3 * U50[det]
        knee[det] = {"u_half": u_half, "finest": finest, "total_reduction_m": tot, "u50": U50[det],
                     "window": [lo, hi], "in_window": (u_half is not None and lo <= u_half <= hi),
                     "curve": {str(k): v for k, v in sorted(e.items(), reverse=True)}}
    both = knee.get("sk", {}).get("u_half"), knee.get("wcte", {}).get("u_half")
    V["P4"] = {"holds": bool(all(knee.get(d, {}).get("in_window") for d in ("sk", "wcte"))
                             and None not in both and both[0] > both[1]), "per_detector": knee}
    # P5 specificity: charge weight x100 at (patch, 50), seed A
    base, cw = run_id("sk", "patch", 50, 9000), run_id("sk", "patch", 50, 9000, cw=100)
    pv = contrast(store, [("A", base, cw)], "vertex", "sealed_test")
    pe_ = contrast(store, [("A", base, cw)], "energy_e", "sealed_test")
    if "T_bar" in pv:
        pv["N_sigma"] = abs(pv["T_bar"]) / pv["SE_boot"]
    if "T_bar" in pe_:
        pe_["N_sigma"] = abs(pe_["T_bar"]) / pe_["SE_boot"]
    V["P5"] = {"holds": (bool(pv["T_bar"] >= -0.03 and not (pe_["T_bar"] > 0 and pe_["N_sigma"] >= 2))
                         if "T_bar" in pv and "T_bar" in pe_ else None),
               "vertex": pv, "electron_energy": pe_,
               "rule": "vertex not more than 3 % lower; electron energy resolution not clearly worse (N_sigma >= 2)"}
    # P6 direction sign = vertex sign
    p6 = {}
    for cid, c in R.items():
        if "T_bar" not in c:
            continue
        for m in ("dir_mu", "dir_e"):
            d = S[m].get(cid, {})
            if "T_bar" in d:
                p6[f"{cid}/{m}"] = bool(np.sign(d["T_bar"]) == np.sign(c["T_bar"]))
    V["P6"] = {"holds": all(p6.values()) if p6 else None, "agree": p6}
    # F1, A' weak, K1, K2, K3/F4, F5
    p5 = g("CT-p5")
    V["F1"] = bool(p5.get("T_bar") is not None and p5["T_bar"] <= -0.10 and (p5.get("N_sigma") or 0) >= 3
                   and V["P2"]["holds"])
    V["A_prime_weak_condition"] = bool(p5.get("T_bar") is not None and -0.10 < p5["T_bar"] <= -0.05
                                       and p5.get("clear") and V["P3"]["holds"])
    small = lambda c: (c.get("T_bar") is None) or abs(c["T_bar"]) < 0.05 or not c.get("clear")
    V["K1"] = bool(small(g("CT-p50")) and small(g("CT-p5")))
    V["K2"] = bool((not V["P2"]["holds"]) or (V["P2"]["total_reduction_450_to_5"] is not None
                                              and -V["P2"]["total_reduction_450_to_5"] < 0.05))
    mlp, att = S["vertex_mlp"].get("CT-p5", {}), S["vertex_attentive"].get("CT-p5", {})
    sign = np.sign(p5.get("T_bar", 0.0))
    V["F4_K3"] = {"ridge": p5.get("T_bar"), "mlp_mean3": mlp.get("T_bar"), "attentive_mean3": att.get("T_bar"),
                  "mlp_same_sign": bool(np.sign(mlp.get("T_bar", 0.0)) == sign),
                  "attentive_same_sign": bool(np.sign(att.get("T_bar", 0.0)) == sign),
                  "attentive_clear_opposite": bool(np.sign(att.get("T_bar", 0.0)) == -sign and att.get("clear")),
                  "attentive_N_sigma": att.get("N_sigma"), "mlp_N_sigma": mlp.get("N_sigma")}
    V["K3"] = V["F4_K3"]["attentive_clear_opposite"]
    f5 = {}
    pairs = [(s, run_id("sk", "patch", 450, SEEDS[s]), run_id("sk", "patch", 5, SEEDS[s])) for s in ("A", "B")]
    for step in (5000, 10000):
        f5[step] = contrast(store, pairs, "vertex", "sealed_test", step=step).get("T_bar")
    if None in f5.values():
        wording = "undetermined (snapshots missing)"
    elif abs(f5[10000]) <= 0.5 * abs(f5[5000]):
        wording = "the loss scale sets the order and rate of learning"
    elif abs(f5[10000]) >= abs(f5[5000]):
        wording = "at this budget the loss scale selects the solution"
    else:
        wording = "undetermined at 10,000 steps"
    V["F5"] = {"CT_p5_5000": f5[5000], "CT_p5_10000": f5[10000], "wording": wording}
    V["K4"] = out.get("K4", {"a": None, "b": None, "fired": False,
                             "note": "filled from the blind recomputation and YARDSTICKS.json"})
    # outcome
    k4 = bool(V["K4"].get("fired"))
    cm = [g("CM-450"), g("CM-50")]
    mask_clear = any(c.get("T_bar") is not None and c["T_bar"] <= -0.10 and c.get("clear") for c in cm)
    if V["F1"] and not (V["K2"] or V["K3"] or k4):
        outcome = "A′ strong"
    elif V["A_prime_weak_condition"] and not (V["K2"] or V["K3"] or k4):
        outcome = "A′ weak"
    elif (V["K1"] or V["K2"]) and mask_clear:  # PREREG §2.5 table: A-alt has no K4 condition
        outcome = "A-alt"
    else:
        outcome = "P2"
    V["outcome"] = outcome
    V["outcome_inputs"] = {"K4_fired": k4, "mask_contrast_clear_le_-10pct": mask_clear}


def fmt(x, nd=4, pct=False):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{100 * x:+.1f} %" if pct else f"{x:.{nd}f}"


def render(out, interim):
    L = [f"# {'INTERIM_E1' if interim else 'GATE_TABLE'} — r2-tier1 registered contrasts (sealed_test, step 10,000)", "",
         "Primary metric: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge readout "
         "(READOUT_SPEC_v1). T = e(cell 2)/e(cell 1) − 1 per seed; T̄ its seed mean; SE_boot paired bootstrap "
         "(2,000 resamples of sealed events, shared by all runs); SE_seed = SD_pooled/√n_seeds; "
         "N_σ = |T̄|/max(SE_boot, SE_seed); clear = N_σ ≥ 2 and one sign in every seed.", "",
         "| Contrast | T̄ | per seed | SE_boot | SE_seed | N_σ | clear | missing runs |",
         "|---|---:|---|---:|---:|---:|---|---|"]
    for cid, c in out["registered"].items():
        ps = ", ".join(f"{k} {100 * v:+.1f} %" for k, v in c.get("per_seed", {}).items())
        L.append(f"| {cid} | {fmt(c.get('T_bar'), pct=True)} | {ps or '–'} | {fmt(c.get('SE_boot'))} | "
                 f"{fmt(c.get('SE_seed'))} | {fmt(c.get('N_sigma'), 2)} | {fmt(c.get('clear'))} | "
                 f"{', '.join(c.get('missing', [])) or '–'} |")
    L += ["", f"SD_pooled per detector: {json.dumps({k: round(v, 4) for k, v in out['SD_pooled'].items()})}", ""]
    names = {"dir_mu": "μ direction (median angle)", "dir_e": "e direction (median angle)",
             "energy_e": "e energy 68 % half-width", "energy_mu": "μ energy 68 % half-width",
             "misid_mu": "μ mis-ID at 99 % e eff.", "vertex_mlp": "vertex, MLP (spatial_moment_full, 3 seeds)",
             "vertex_mlp_latent": "vertex, MLP (latent, 3 seeds)", "vertex_attentive": "vertex, attentive (3 seeds)"}
    L += ["## Secondary outcomes and probe variants (same contrasts)", "",
          "| Contrast | " + " | ".join(names[m] for m in out["secondary"]) + " |",
          "|---|" + "---|" * len(out["secondary"])]
    for cid in out["registered"]:
        cells = []
        for m, sec in out["secondary"].items():
            c = sec.get(cid, {})
            cells.append(f"{fmt(c.get('T_bar'), pct=True)} ({fmt(c.get('N_sigma'), 1)}{'*' if c.get('clear') else ''})"
                         if c.get("T_bar") is not None else "–")
        L.append(f"| {cid} | " + " | ".join(cells) + " |")
    L += ["", "Cells: T̄ (N_σ); * = clear.", ""]
    if not interim:
        V = out["verdicts"]
        L += ["## Verdicts", ""]
        for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
            L.append(f"- **{k}**: {fmt(V[k].get('holds'))}")
        L += [f"- **F1** (A′ strong condition): {fmt(V['F1'])}",
              f"- **A′ weak condition**: {fmt(V['A_prime_weak_condition'])}",
              f"- **K1**: {fmt(V['K1'])}; **K2**: {fmt(V['K2'])}; **K3**: {fmt(V['K3'])}; **K4**: {fmt(V['K4'].get('fired'))}",
              f"- **F4**: MLP same sign {fmt(V['F4_K3']['mlp_same_sign'])}, attentive same sign {fmt(V['F4_K3']['attentive_same_sign'])}",
              f"- **F5**: CT-p5 at 5,000 = {fmt(V['F5']['CT_p5_5000'], pct=True)}, at 10,000 = {fmt(V['F5']['CT_p5_10000'], pct=True)} → \"{V['F5']['wording']}\"",
              "", f"**Outcome (§2.5): {V['outcome']}**", ""]
        L += ["P2 curve (SK, patch, seed mean, m): " + json.dumps({k: (round(v, 4) if v else None)
                                                                  for k, v in V["P2"]["curve_seed_mean_m"].items()}), ""]
        for det, k in V["P4"]["per_detector"].items():
            L.append(f"P4 {det}: u_half = {k.get('u_half')} ns, u₅₀ = {k.get('u50')} ns, window {k.get('window')}, "
                     f"in window {fmt(k.get('in_window'))}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=EVAL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interim", action="store_true")
    ap.add_argument("--k4", type=Path, help="JSON with K4 a/b inputs")
    a = ap.parse_args()
    build(a)


if __name__ == "__main__":
    main()
