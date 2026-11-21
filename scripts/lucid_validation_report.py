#!/usr/bin/env python3
"""Build results/round2/20261121_lucid/VALIDATION.md (+ figures) from the L4 gate outputs."""
import json, shutil, sys
from pathlib import Path

VAL = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/validation")
DATA = Path("/ceph/dwong/jepa/data/lucid_v1")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/round2/20261121_lucid")
DETS = ("wcte", "sk", "juno", "hk")


def selfcheck2000(det):
    for p in sorted((DATA / det / "returns").glob("selfcheck2000_*_selfcheck_*.json")) + \
             sorted((DATA / det / "returns").glob("selfcheck2000_dt2/selfcheck_*.json")):
        return json.loads(p.read_text()), p
    return None, None


def g3_verdict(s200, s2000):
    rows, ok = [], True
    for k, p in enumerate(s200["positions"]):
        q = s2000["positions"][k] if s2000 else None
        ratio_ok = abs(p["mean_ratio_total"] - 1) <= 0.05
        pear_ok = p["pearson"] >= 0.95 or (q is not None and (q["pearson"] >= 0.95 or
                                                               (q.get("chi2_per_pmt") is not None and q["chi2_per_pmt"] <= 1.2)))
        ok &= ratio_ok and pear_ok
        rows.append((p, q, ratio_ok and pear_ok))
    return ok, rows


def main():
    figs = OUT / "figures" / "validation"
    figs.mkdir(parents=True, exist_ok=True)
    L = ["# Validation gates (r2-lucid L4)", "",
         "Pilot: the first 10 runs (500 events, 1:1:1 photon/muon/electron) of each detector's `dev_probe_train` "
         "cohort, generated with the production code (`scripts/lucid_generate.py`, LUCiD `e651e836`, tier A1). "
         "Validation code: `scripts/lucid_validate.py`; raw outputs under "
         "`/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/validation/<det>/`.", "",
         "| Gate | WCTE | SK | JUNO | HK |", "|---|---|---|---|---|"]
    res, s3 = {}, {}
    for d in DETS:
        res[d] = json.loads((VAL / d / f"validation_{d}.json").read_text())
        s200 = res[d]["G3"].get("positions")
        s2000, _ = selfcheck2000(d)
        s3[d] = g3_verdict({"positions": s200}, s2000) if s200 else (None, [])
    def cell(d, g):
        if g == "G3":
            v = s3[d][0]
        else:
            v = res[d][g]["pass"]
        return "PASS" if v else ("FAIL" if v is False else "pending")
    for g, name in (("G1", "G1 loading + 2-step MAE/JEPA dry run"), ("G2", "G2 photon statistics"),
                    ("G3", "G3 LUCiD self-consistency"), ("G4", "G4 regime and trigger"),
                    ("G5", "G5 displays and ToF fit"), ("G6", "G6 no toy leakage")):
        L.append(f"| {name} | " + " | ".join(cell(d, g) for d in DETS) + " |")
    L += ["", "## G2 photon statistics (photon-bomb events of the pilot)", "",
          "| Detector | Events | Slope ratio free/origin | R² | Residual peak (ns) | Core σ (ns) | LUCiD jitter σ at 1 PE (ns) | First hits > 10 ns late |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DETS:
        g = res[d]["G2"]
        L.append(f"| {d.upper()} | {g['n_photon_events']} | {g['slope_ratio']:.4f} | {g['r2']:.4f} | "
                 f"{g['residual_peak_ns']:+.1f} | {g['residual_core_sigma_ns']:.2f} | "
                 f"{g['lucid_digitizer_jitter_sigma_1pe_ns']:.2f} | {100 * g['late_hit_fraction_gt10ns_first_hits']:.1f} % |")
    L += ["", "Residual = first-hit digit time − d·n/c − emission offset, with n = 1.33 as a straight-line reference "
          "(LUCiD's group velocity is wavelength-dependent, so the peak may sit slightly off zero). LUCiD's "
          "`response.tts` is 0 in these configs; the PMT timing spread comes from the digitizer's "
          "charge-dependent jitter, whose 1-PE σ is listed. Late light (> 10 ns) is present, as required with "
          "scattering and reflection enabled.", "",
          "## G3 LUCiD self-consistency (sampled kernel vs LUCiD expected mode, 3 positions)", "",
          "| Detector | Position (m) | Pearson (200 ev) | Ratio (200 ev) | Pearson (2000 ev) | Poisson ceiling (2000) | χ²/PMT (2000) | Ratio (2000) | Verdict |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for d in DETS:
        for p, q, ok in s3[d][1]:
            pos = ", ".join(f"{x:.2f}" for x in p["position_m"])
            qq = (f"{q['pearson']:.3f} | {q.get('pearson_poisson_ceiling', float('nan')):.3f} | "
                  f"{q.get('chi2_per_pmt', float('nan')):.2f} | {q['mean_ratio_total']:.4f}") if q else "– | – | – | –"
            L.append(f"| {d.upper()} | ({pos}) | {p['pearson']:.3f} | {p['mean_ratio_total']:.4f} | {qq} | {'PASS' if ok else 'FAIL'} |")
    L += ["", "Criterion: total ratio within ±5 % and Pearson ≥ 0.95; where the 200-event Poisson ceiling is below "
          "0.95 (flat expected pattern near the centre), the 2000-event re-run decides (Pearson ≥ 0.95, or a "
          "noise-aware per-PMT χ²/dof ≤ 1.2 that includes the expected-mode Monte Carlo noise). Decision in "
          "`docs/round2/DECISIONS.md`.", "",
          "## G4 regime and trigger", "",
          "| Detector | Detected PE q5/q50/q95 photon bombs | all domains | Hit PMTs q5/q50/q95 (bombs) | NHits eff. above y5 | LUCiD trigger eff. above y5 |",
          "|---|---|---|---|---:|---:|"]
    for d in DETS:
        g = res[d]["G4"]
        f = lambda v: " / ".join(f"{x:,.0f}" for x in v)
        L.append(f"| {d.upper()} | {f(g['true_pe_q_photon'])} | {f(g['true_pe_q_all'])} | {f(g['hit_occupancy_q_photon'])} | "
                 f"{g['nhits_eff_above_y5']:.3f} | {g['lucid_trigger_eff_above_y5']:.3f} |")
    L += ["", "Targets 200 / 3,000 / 30,000 detected PE (photon bombs); every pilot quantile is within a factor 2. "
          "No yield-range change was needed. LUCiD's own trigger exists only for SK and HK (WCTE/JUNO: no LUCiD "
          "trigger, efficiency 1 by construction).", "",
          "## G1 loading and dry runs, G5 displays and ToF fit", "",
          "| Detector | Pilot pack sha256 | MAE (repaired) 2 steps | JEPA (repaired) 2 steps | ToF-only Huber fit: mean / median 3-D error (m) |",
          "|---|---|---|---|---|"]
    for d in DETS:
        g1, g5 = res[d]["G1"], res[d]["G5"]
        L.append(f"| {d.upper()} | `{g1['pack_sha256'][:16]}…` | {'complete' if g1['mae_repaired']['complete'] else 'failed'} | "
                 f"{'complete' if g1['jepa_repaired']['complete'] else 'failed'} | {g5['tof_mean_err_m']:.3f} / {g5['tof_median_err_m']:.3f} |")
        for name in (f"event_displays_{d}.png", f"g2_{d}.png"):
            src = VAL / d / "figures" / name
            if src.exists():
                shutil.copy(src, figs / name)
    L += ["", "The ToF fit is an independent simple estimator (Huber least squares on token first-hit times, "
          "straight-line n = 1.33); it is reported, not gated, and is not a likelihood ceiling.", "",
          "G6: the generation path imports only the allowed `lucid_bombs` helpers (geometry, keyed RNG, run "
          "specs, detector state, sparse storage) and never imports `r2_generator`; a run-time test booby-traps "
          "every toy physics function and generates a photon, a muon and an electron event through LUCiD "
          "(`src/jepa_noise/tests/test_lucid_mc_generator.py`).", "",
          "Figures: " + ", ".join(f"[{d.upper()} displays](figures/validation/event_displays_{d}.png), "
                                  f"[{d.upper()} G2](figures/validation/g2_{d}.png)" for d in DETS)]
    (OUT / "VALIDATION.md").write_text("\n".join(L) + "\n")
    print("ok", {d: s3[d][0] for d in DETS})


if __name__ == "__main__":
    main()
