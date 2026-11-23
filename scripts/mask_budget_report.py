#!/usr/bin/env python3
"""r2-mask M1.4: SHORTCUT_BUDGET.md / .json from the per-detector outputs of mask_shortcut_budget.py.

Usage: mask_budget_report.py --inputs wcte.json sk.json juno.json hk.json --out-dir results/round2/20261123_mask
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DOMAINS = ("photon", "muon", "electron")
PRIMARY = ("wcte", "sk")


def table(det, dom, configs):
    rows = ["| Config | hidden frac. | never hidden | E_S (static) | E_L − E_S (local) | U = 1 − E_L | "
            "AvgDist (m) | AvgDist (spacings) | E_L charge / hit / time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, r in configs.items():
        ch = r["channels"]
        rows.append(f"| {name} | {r['hidden_fraction']:.3f} | {r['never_hidden_fraction']:.3f} | "
                    f"{r['E_S']:.4f} | {r['local']:.4f} | {r['U']:.4f} | {r['avg_dist_m']:.3f} | "
                    f"{r['avg_dist_spacing']:.2f} | {ch['charge']['E_L']:.3f} / {ch['hit']['E_L']:.3f} / "
                    f"{ch['time']['E_L']:.3f} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    data = {}
    for p in a.inputs:
        r = json.loads(p.read_text())
        data[r["detector"]] = r
    summary = {"phase": "r2-mask", "stage": "M1", "definition": "runbook 2026-11-23 §1.2; "
               "src/jepa_noise/shortcut_budget.py", "detectors": {}}
    md = ["# SHORTCUT_BUDGET — r2-mask M1 (network-free shortcut budget per mask)", "",
          "Runbook `RUNBOOK_2026-11-23_R2_MASK_SCALE.md` §1.2. Targets of the default `mae_repaired` loss "
          "(hit flag and ln q on every hidden PMT, time / 50 ns on hidden hit PMTs; unit weights). "
          "S = per-PMT mean over fitting events; L = one Ridge per channel, shared across PMTs, from the "
          "k = 8 nearest visible PMTs (h, ln q, t, distance) plus the static means. "
          "E = 1 − MSE/Var on scoring events (combined: 1 − ΣMSE/ΣVar). U contains irreducible noise "
          "that is the same for every mask of a detector: only differences in U between masks are "
          "interpreted. Events: `dev_probe_train`, fitting and scoring runs disjoint; photon bombs "
          "1,700 + 1,700 (the whole cohort), μ and e 1,000 + 1,000. Masks: the training masker "
          "(`ViewMasker`, training seed, n_min = 0). WCTE and SK are registered; JUNO and HK are "
          "**exploratory**.", ""]
    for det in ("wcte", "sk", "juno", "hk"):
        if det not in data:
            continue
        r = data[det]
        tag = "" if det in PRIMARY else " (exploratory)"
        md += [f"## {det.upper()}{tag}", "",
               f"N_PMT = {r['n_pmt']}, median nearest-neighbour spacing {r['nn_spacing_m']:.3f} m, "
               f"default patch {r['default_patch_size']}, code {r['code_sha'][:8]}.", ""]
        lc_line = ", ".join(f"{dom} {r['domains'][dom]['correlation']['l_c_m']:.2f} m "
                            f"({r['domains'][dom]['correlation']['l_c_spacing']:.1f} spacings)"
                            for dom in DOMAINS if dom in r["domains"])
        md += [f"Correlation length ℓ_c (charge channel): {lc_line}.", ""]
        summary["detectors"][det] = {"n_pmt": r["n_pmt"], "nn_spacing_m": r["nn_spacing_m"],
                                     "default_patch_size": r["default_patch_size"],
                                     "code_sha": r["code_sha"], "counts": r["counts"],
                                     "domains": {}}
        for dom in DOMAINS:
            if dom not in r["domains"]:
                continue
            dd = r["domains"][dom]
            md += [f"### {det.upper()} — {dom}{' (primary)' if dom == 'photon' else ''}", "",
                   table(det, dom, dd["configs"]), ""]
            summary["detectors"][det]["domains"][dom] = {
                "l_c_m": dd["correlation"]["l_c_m"], "l_c_spacing": dd["correlation"]["l_c_spacing"],
                "E1_nn": dd["correlation"]["E1_nn"], "E1_curve": dd["correlation"]["curve"],
                "configs": dd["configs"]}
    md += ["## Notes", "",
           "- `parity` = `within` (fixed odd 1-based `pmt_id_in_mpmt` hidden; 10 of 19 per WCTE/SK module, "
           "hence hidden fraction 0.526); `random` = fresh uniform half per event; `patchS` = fresh "
           "half of the contiguous Hilbert patches of S PMTs (⌈count/2⌉ for non-default S; the tail "
           "of the Hilbert order beyond count × S is never hidden).",
           "- E_S is ≈ 0 for every configuration: the PMT identity alone carries ≤ 2 % of the target "
           "variance on these cohorts (slightly negative values are fit/score sampling noise).",
           "- Ridge α chosen by generalised cross-validation over 10⁻³…10³; with 10⁵–10⁶ fitting rows "
           "the upper end of the grid is still weak regularisation.", ""]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "SHORTCUT_BUDGET.md").write_text("\n".join(md))
    (a.out_dir / "SHORTCUT_BUDGET.json").write_text(json.dumps(summary, indent=1, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
