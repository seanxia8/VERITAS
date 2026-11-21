"""V6: blind comparison of the independent results (sealed at b755b23) with the committed
values, using the pre-registered tolerances of runbook §V6.2.

Usage: python3 v6_compare.py RESULTS_DIR   (writes RESULTS_DIR/v6_comparison.json)
"""
import json
import math
import os
import sys

R = sys.argv[1]
B = os.path.join(R, "..", "20261117_repair")
J = lambda *p: json.load(open(os.path.join(*p)))
k1, k2, k3, k4, k5 = (J(R, f"{n}.json") for n in ("k1_readout", "k2_collapse", "k3_sigreg", "k4_tof", "k5_regime"))
probes = J(B, "probes", "probe_repair_results.json")["runs"]
retrain = J(B, "retrain", "retrain_readings.json")["runs"]
lat = J(B, "diagnostics", "latent_variance_cka.json")["detectors"]
a6 = J(B, "diagnostics", "a6_training_checks.json")["heldout"]
ceil = J(B, "diagnostics", "physics_ceiling.json")["detectors"]

FORMS = {"a_allpmt_mean": "ridge_all_mean", "b_hit_mean": "ridge_hit_mean",
         "c_hit_moments": "ridge_spatial_moment_latent"}


def tol_k1(mine, ref):
    d = mine - ref
    return {"mine_m": mine, "committed_m": ref, "delta_m": d, "rel": d / ref,
            "pass": abs(d) <= max(0.10 * abs(ref), 0.003)}


def committed_vertex(case):
    det, arm, s = case.split("__")
    if arm in ("jepa_repaired", "mae_repaired"):
        return retrain[case]["repaired"]["vertex"]
    for r in probes:
        if r["detector"] == det and r["arm"] == arm and r["seed"] == int(s[1:]):
            return r["vertex"]


out = {"sealed_commit": "b755b23", "k1": {}, "k1_raw": {}, "k2": {}, "k3": {}, "k4": {}}
for case, v in k1["cases"].items():
    cv = committed_vertex(case)
    out["k1"][case] = {f: tol_k1(v[f]["mean_err_m"], cv[c]["mean_m"]) for f, c in FORMS.items()}
for det, v in k1["raw"].items():
    ref = next(r["vertex"]["ridge_raw_spatial_moment"]["mean_m"] for r in probes if r["detector"] == det)
    out["k1_raw"][det] = tol_k1(v["mean_err_m"], ref)

for case, v in k2["cases"].items():
    det, arm = v["det"], v["arm"]
    if v["seed"] != 9000 or arm not in lat[det]["variance"]:
        continue
    c = lat[det]["variance"][arm]
    ref_static = c["component_fractions"]["static_layout"]
    ref_across = c["same_pmt_across_event_variance"] / c["total_variance"]
    mine_static, mine_across = v["all_pmt"]["static_frac"], v["all_pmt"]["across_event_frac"]
    import math
    om = abs(math.log10(mine_across / ref_across))
    out["k2"][case] = {"static_mine": mine_static, "static_committed": ref_static,
                       "across_mine": mine_across, "across_committed": ref_across,
                       "pass": abs(mine_static - ref_static) <= 0.03 and om < 1.0}
# I7 recomputed on the committed numbers, and the absolute-variance reading
i7c = {}
for d in ("wcte", "sk", "hk"):
    s, m = lat[d]["variance"]["sigreg_jepa"], lat[d]["variance"]["mae_mse"]
    i7c[d] = {"fraction_ratio": (m["same_pmt_across_event_variance"] / m["total_variance"]) /
                                (s["same_pmt_across_event_variance"] / s["total_variance"]),
              "absolute_variance_ratio": m["same_pmt_across_event_variance"] / s["same_pmt_across_event_variance"],
              "sigreg_static": s["component_fractions"]["static_layout"]}
out["k2_committed_I7"] = i7c
j = lat["juno"]["variance"]["sigreg_jepa"]
out["k2_committed_I8"] = {"juno_static": j["component_fractions"]["static_layout"],
                          "sk_static": lat["sk"]["variance"]["sigreg_jepa"]["component_fractions"]["static_layout"],
                          "hk_static": lat["hk"]["variance"]["sigreg_jepa"]["component_fractions"]["static_layout"]}

# Supplementary (C4 headline quantity): event-mean spread f_event and layout share, vs the
# committed per-seed readings (results/round2/20261112_eval/readings.json, seed 9000).
rd = J(R, "..", "20261112_eval", "readings.json")["detectors"]
out["k2_c4_fevent"] = {}
for case, v in k2["cases"].items():
    det, arm = v["det"], v["arm"]
    m = rd.get(det, {}).get("readings", {}).get("two_seed", {}).get("metrics", {}).get(f"{arm}:{v['seed']}")
    if m and "f_event" in m:
        out["k2_c4_fevent"][case] = {"f_event_mine": v["event_mean_spread"], "f_event_committed": m["f_event"],
                                     "rel": v["event_mean_spread"] / m["f_event"] - 1,
                                     "same_order": abs(math.log10(v["event_mean_spread"] / m["f_event"])) < 1}

ref = a6["sk__sigreg_jepa__s9000"]["sigreg_context_by_batch_size"]
mine = k3["real_sk_sigreg_s9000"]["scaling"]
out["k3"] = {"values": {n: {"mine": mine[n], "committed": ref[n], "rel": mine[n] / ref[n] - 1} for n in mine},
             "ratios": {n: {"mine": mine[n] / mine["4"], "committed": ref[n] / ref["4"],
                            "rel": (mine[n] / mine["4"]) / (ref[n] / ref["4"]) - 1} for n in mine}}
out["k3"]["pass"] = all(abs(x["rel"]) <= 0.10 for x in out["k3"]["ratios"].values())

for det, v in k4["detectors"].items():
    c = ceil[det]
    tof, cr = v["tof_ng1.33"]["mean_err_m"], v["crlb_timing_sigma1p5_median_m"]
    out["k4"][det] = {"tof_mine": tof, "tof_committed": c["fits"]["tof"]["mean_m"],
                      "tof_rel": tof / c["fits"]["tof"]["mean_m"] - 1,
                      "crlb_mine": cr, "crlb_committed": c["crlb"]["median_m"], "crlb_rel": cr / c["crlb"]["median_m"] - 1,
                      "convergence_mine": v["tof_ng1.33"]["convergence_rate"],
                      "convergence_committed": c["fits"]["tof"]["n_converged"] / c["n_events"]}
    out["k4"][det]["pass"] = abs(out["k4"][det]["tof_rel"]) <= 0.35 and abs(out["k4"][det]["crlb_rel"]) <= 0.35
# I13: ToF vs best (c) readout over all K1 cases of the detector
out["I13"] = {}
for det in ("sk", "juno", "hk"):
    best = min(v["c_hit_moments"]["mean_err_m"] for v in k1["cases"].values() if v["det"] == det)
    tof = k4["detectors"][det]["tof_ng1.33"]["mean_err_m"]
    out["I13"][det] = {"best_c_m": best, "tof_m": tof, "ratio": best / tof, "pass": best / tof >= 5}
out["I12"] = {d: {"mean_err_m": k4["detectors"][d]["tof_ng1.33"]["mean_err_m"],
                  "convergence": k4["detectors"][d]["tof_ng1.33"]["convergence_rate"],
                  "pass": k4["detectors"][d]["tof_ng1.33"]["mean_err_m"] <= 0.05 and
                          k4["detectors"][d]["tof_ng1.33"]["convergence_rate"] >= 0.95} for d in ("sk", "juno", "hk")}

# V6 investigation: own re-implementation of the committed construction (v6_confirm.py)
cdir = os.path.join(R, "v6_confirm")
if os.path.isdir(cdir):
    conf = {"c_prime": {}, "d_prime": {}, "c_centred": {}}
    for fn in sorted(os.listdir(cdir)):
        x = J(cdir, fn)
        if fn.endswith("__confirm.json"):
            ref = committed_vertex(x["case"])["ridge_spatial_moment_latent"]["mean_m"]
            conf["c_prime"][x["case"]] = tol_k1(x["c_prime_committed_style_latent_m"], ref)
        elif fn == "confirm_raw.json":
            for det, v in x.items():
                conf["d_prime"][det] = tol_k1(v["d_prime_committed_style_raw_m"], out["k1_raw"][det]["committed_m"])
        elif fn == "confirm_centred.json":
            for case, v in x.items():
                conf["c_centred"][case] = tol_k1(v["c_centred_m"], out["k1"][case]["c_hit_moments"]["committed_m"])
    conf["summary"] = {k: f"{sum(v['pass'] for v in conf[k].values())}/{len(conf[k])}" for k in ("c_prime", "d_prime", "c_centred")}
    conf["max_abs_rel_c_prime"] = max(abs(v["rel"]) for v in conf["c_prime"].values())
    conf["max_abs_rel_d_prime"] = max(abs(v["rel"]) for v in conf["d_prime"].values())
    # I1 on the committed-style form where available
    conf["I1_on_c_prime"] = {c: out["k1"][c]["a_allpmt_mean"]["mine_m"] / v["mine_m"] for c, v in conf["c_prime"].items() if c.startswith(("sk", "juno", "hk"))}
    out["v6_confirm"] = conf

summ = {"k1_pass": sum(f["pass"] for c in out["k1"].values() for f in c.values()),
        "k1_total": sum(len(c) for c in out["k1"].values())}
for f in FORMS:
    summ[f"k1_{f}_pass"] = sum(c[f]["pass"] for c in out["k1"].values())
summ["k1_raw_pass"] = sum(v["pass"] for v in out["k1_raw"].values())
summ["k2_pass"] = sum(v["pass"] for v in out["k2"].values()); summ["k2_total"] = len(out["k2"])
summ["k3_pass"] = out["k3"]["pass"]; summ["k4_pass"] = sum(v["pass"] for v in out["k4"].values())
out["summary"] = summ
print(json.dumps(summ, indent=1))
json.dump(out, open(os.path.join(R, "v6_comparison.json"), "w"), indent=1)
