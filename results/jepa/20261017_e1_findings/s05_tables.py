"""s05: emit the markdown tables of E1_BRIDGE_FINDINGS.md from the JSON files.

Reads every diagnostic JSON in this directory and prints the markdown tables of
the findings document, so the numbers are generated from the stored evidence
rather than transcribed (stdout only; no file is written).  Diagnostic only; no new computation.
"""
from __future__ import annotations

import json

import e1f_common as C


def load(name):
    return json.loads((C.OUT / name).read_text(encoding="utf-8"))


def floor_table(fs):
    lines = ["| fixture | n | replicate angles (deg) | floor (deg) | floor / angle |",
             "|---|---|---|---|---|"]
    e1b = fs["E1b_clean_channel"]["n"]
    for n in ("10000", "100000", "1000000"):
        v = e1b[n]
        lines.append(f"| E1b clean cell (channel) | {int(n):,} | "
                     f"{v['angles_deg'][0]:.2f}, {v['angles_deg'][1]:.2f} | "
                     f"{v['floor_deg']:.2f} | {v['floor_deg']/v['mean_angle_deg']:.2f} |")
    for label, rec in fs["E1a_theta0.1"].items():
        if rec["population_cca_deg"] > 1e-6:
            continue
        for n in ("10000", "100000", "1000000"):
            v = rec["n"][n]
            lines.append(f"| E1a {label} | {int(n):,} | "
                         f"{v['angles_deg'][0]:.2f}, {v['angles_deg'][1]:.2f} | "
                         f"{v['floor_deg']:.2f} | {v['floor_deg']/v['mean_angle_deg']:.2f} |")
    return "\n".join(lines)


def scaling_table(fs):
    s = fs["scaling"]
    lines = ["| fixture | log-log slope of angle | log-log slope of floor |",
             "|---|---|---|"]
    e = s["E1b_clean_channel"]
    lines.append(f"| E1b clean cell | {e['angle_slope_loglog']:+.3f} | "
                 f"{e['floor_slope_loglog']:+.3f} |")
    for label, v in s["E1a_gamma0"].items():
        lines.append(f"| E1a {label} | {v['angle_slope_loglog']:+.3f} | "
                     f"{v['floor_slope_loglog']:+.3f} |")
    return "\n".join(lines)


def saturated_table(fs):
    lines = ["| E1a cell | population CCA angle vs R_c (deg) | canonical rho (top 3) | "
             "seed-mean angle at n=1e5 (deg) | n=1e5 floor (deg) |",
             "|---|---|---|---|---|"]
    for label, rec in fs["E1a_theta0.1"].items():
        v = rec["n"]["100000"]
        rho = ", ".join(f"{x:.3f}" for x in rec["population_canonical_correlations"])
        lines.append(f"| {label} | {rec['population_cca_deg']:.1f} | {rho} | "
                     f"{v['mean_angle_deg']:.2f} | {v['floor_deg']:.2f} |")
    return "\n".join(lines)


def r2_table(r2):
    lines = ["| cell | registered Spearman | pass | saturated points (>=80 deg) | "
             "diagnostic Spearman without saturated points | points kept |",
             "|---|---|---|---|---|---|"]
    for label, reg in r2["registered_R2"]["cells"].items():
        diag = r2["diagnostic_only"]["cells"][label]
        rho = ("—" if diag["spearman_unsaturated"] is None
               else f"{diag['spearman_unsaturated']:.3f}")
        lines.append(f"| {label} | {reg['spearman_registered']:+.4f} | "
                     f"{'pass' if reg['spearman_pass'] else 'fail'} | "
                     f"{reg['saturated_points']}/7 | {rho} | {diag['n_points']}/7 |")
    return "\n".join(lines)


def r3_table(r3, rows):
    reg = r3["registered"]
    lines = ["| quantity | value |", "|---|---|",
             f"| n=1e5 rows with an R3 diagnostic | {reg['rows_with_r3']} |",
             f"| evaluated rows | {reg['evaluated_rows']} |",
             f"| max observed spectrum difference (208 rows) | "
             f"{reg['max_observed_spectrum_difference_208']:.3e} |",
             f"| median / p99 observed difference | "
             f"{reg['median_observed_spectrum_difference']:.2e} / "
             f"{reg['p99_observed_spectrum_difference']:.2e} |",
             f"| rows outside the 1e-8 tolerance | {reg['failed_rows']} |"]
    lines += ["", "| failing row | cond(S_cc) | cond(S_tt) | cond(T_c S_cc T_c^T) | "
                  "cond(T_t S_tt T_t^T) | cond(T_c) | cond(T_t) |",
              "|---|---|---|---|---|---|---|"]
    for name, row in rows["rows"].items():
        lines.append(f"| {name} | {row['base']['cond_scc']:.1f} | "
                     f"{row['base']['cond_stt']:.1f} | {row['transformed']['cond_scc']:.3e} | "
                     f"{row['transformed']['cond_stt']:.3e} | {row['cond_tc']:.3e} | "
                     f"{row['cond_tt']:.3e} |")
    lines += ["", "| failing row | base isotropy | transformed isotropy | TOL | "
                  "transformed objective gap |", "|---|---|---|---|---|"]
    for name, row in rows["rows"].items():
        lines.append(f"| {name} | {row['base']['isotropy_residual']:.2e} | "
                     f"{row['transformed']['isotropy_residual']:.2e} | 1e-8 | "
                     f"{row['transformed']['objective_gap']:.2e} |")
    return "\n".join(lines)


def amplitude_table(amp):
    pc = amp["per_channel"]
    lines = ["| channel | Var(A_c) | CRLB joint (prior mean) | CRLB joint at theta0 | "
             "CRLB channel-local | R2_max joint | R2_max channel-local |",
             "|---|---|---|---|---|---|---|"]
    for c in pc["channels"]:
        lines.append(f"| {c} | {pc['var_amplitude'][c]:.4f} | "
                     f"{pc['crlb_joint_prior_mean'][c]:.4f} | "
                     f"{pc['crlb_joint_theta0'][c]:.4f} | "
                     f"{pc['crlb_channel_local'][c]:.5f} | "
                     f"{pc['r2_max_joint_prior'][c]:.4f} | "
                     f"{pc['r2_max_channel_local'][c]:.4f} |")
    return "\n".join(lines)


def measured_table(amp):
    lines = ["| arm | seed | tokens R2 | tokens / R2_max | post-time R2 | "
             "post-time / R2_max | floor (tokens) |",
             "|---|---|---|---|---|---|---|"]
    for arm, seeds in amp["measured"].items():
        for seed, row in seeds.items():
            lines.append(f"| {arm} | {seed} | {row['tokens']['mean_r2']:.4f} | "
                         f"{row['tokens']['ratio_to_joint_ceiling']:.3f} | "
                         f"{row['post_time']['mean_r2']:.4f} | "
                         f"{row['post_time']['ratio_to_joint_ceiling']:.3f} | "
                         f"{row['random_init_floor']['tokens']:.4f} |")
    return "\n".join(lines)


def main():
    fs, r2 = load("finite_sample.json"), load("r2_saturation.json")
    r4a = fs["R4a_population_recovery"]
    r3, rows = load("r3_conditioning.json"), load("r3_residuals.json")
    amp, val = load("amplitude_ceiling.json"), load("validation.json")
    parts = [
        "## Replay validation", "",
        f"E1a sample-covariance SHA-256 match: {val['e1a_replay']['scc_digest_matches']}; "
        f"angle difference {val['e1a_replay']['cca_abs_diff_deg']:.1e} deg.  "
        f"E1b sample-covariance SHA-256 match: {val['e1b_replay']['scc_digest_matches']}; "
        f"angle difference {val['e1b_replay']['cca_abs_diff_deg']:.1e} deg.", "",
        "## Finite-sample floor", "", floor_table(fs), "",
        "## 1/sqrt(n) slopes", "", scaling_table(fs), "",
        "## E1a theta=0.1 cells at n=1e5", "", saturated_table(fs), "",
        "## R2 registered vs unsaturated Spearman", "", r2_table(r2), "",
        "## R3", "", r3_table(r3, rows), "",
        "## Amplitude ceiling", "", amplitude_table(amp), "",
        "## Measured amplitude R2 and ratio to the ceiling", "", measured_table(amp),
        "",
        f"R4a population recovery error (GEVP): recorded "
        f"{r4a['aggregate_reported_rad']:.3e} rad; recomputed max over 42 cells "
        f"{r4a['max_gevp_rad']:.3e} rad. Whitened-PCA contrast "
        f"{r4a['max_whitened_pca_rad']:.3e} rad.",
        f"E1b population CCA angle: {fs['E1b_clean_channel']['population_cca_rad']:.3e} rad.",
        ""]
    print("\n".join(parts))


if __name__ == "__main__":
    main()
